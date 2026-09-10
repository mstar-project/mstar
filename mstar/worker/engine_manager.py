import logging
from dataclasses import dataclass, field

import torch

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import (
    ON_DEMAND,
    RELOAD,
    RESIDENCY_POLICIES,
    RESIDENT,
    Engine,
)
from mstar.engine.resources import ResourceReqConfig, apply_yaml_overrides
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.base import Model

logger = logging.getLogger(__name__)


def parse_residency(model_config: dict, node_names: set[str]) -> dict[str, str]:
    """Per-node weight residency for the nodes this worker owns, policy by name.

    Declared on the node_group that places the node. Only non-default entries
    are returned; anything absent is ``resident``.

    ``resident``   weights load to the device once and stay (the default, and
                   the only policy that keeps CUDA-graph capture, torch.compile
                   and cross-request batching for that node).
    ``on_demand``  weights live on the host and move to the device around each
                   execution. Frees device memory; on a unified-memory part
                   (GB10) host and device are the same pool, so it bounds the
                   *device allocator* but not system memory.
    ``reload``     weights are dropped after each execution and rebuilt from the
                   checkpoint on the next one. The only policy that frees memory
                   outright, and so the only one that helps when a model's total
                   exceeds physical memory.

    An unrecognized policy raises rather than silently defaulting, matching how
    the resource overrides treat a misspelled key.
    """
    residency: dict[str, str] = {}
    for group in model_config.get("node_groups", []) or []:
        policy = str(group.get("residency", RESIDENT)).lower()
        if policy not in RESIDENCY_POLICIES:
            raise ValueError(
                f"node_group residency must be one of {list(RESIDENCY_POLICIES)}, "
                f"got {policy!r} for nodes {group.get('node_names')}"
            )
        for node in group.get("node_names", []):
            residency[node] = policy
    return {
        n: p for n, p in residency.items() if n in node_names and p != RESIDENT
    }


def parse_on_demand_nodes(model_config: dict, node_names: set[str]) -> set[str]:
    """The ``on_demand`` subset of :func:`parse_residency`."""
    return {
        n for n, p in parse_residency(model_config, node_names).items()
        if p == ON_DEMAND
    }



def _make_submodule_factory(model, device, parallel_groups, autocast_dtype):
    """Rebuild a node's submodule from the checkpoint, onto the device.

    Used by the `reload` policy, whose evictions drop the weights entirely.
    Mirrors the construction in ``build`` so a rebuilt module is identical to
    the one loaded at startup.
    """
    def factory(name: str):
        # The caller released the cache entry on eviction, so this really does
        # rebuild from the checkpoint rather than returning the evicted object.
        extra = {}
        node_sp_group = parallel_groups.get_sp_config_for_node(name)
        if node_sp_group.world_size > 1:
            extra["sp_group"] = node_sp_group
        submodule = model.get_submodule(
            name, device,
            tp_group=parallel_groups.get_tp_config_for_node(name),
            autocast_dtype=autocast_dtype, **extra,
        )
        if submodule is None:
            raise ValueError(
                f"reload-residency node {name!r} has no submodule to rebuild "
                "(get_submodule returned None)"
            )
        if submodule.disable_autocast:
            return submodule.to(device=device)
        node_dtype = submodule.get_autocast_dtype() or autocast_dtype
        return submodule.to(device=device, dtype=node_dtype)

    return factory


@dataclass
class EngineManager:
    """Owns the worker's engine.

    One engine serves every node: it holds all the submodules and the
    resources they share, so there is nothing to group or dedup.
    """
    engine: Engine
    node_names: set[str] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        node_names: set[str],
        device: torch.device,
        model_config: dict,
        parallel_groups: WorkerParallelGroups,
        transfer_engine_info: TransferEngineInfo,
        model: Model,
        enable_nvtx: bool = False,
        enable_prof: bool=False,
    ) -> "EngineManager":
        """Build the engine and load this worker's nodes into it.

        The model supplies each node's submodule via ``get_submodule`` and its
        resources via ``get_node_resources``; the KV cache's shape rides on
        the latter, so there is no separate cache config.
        """
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, model_config)

        residency = parse_residency(model_config, node_names)
        on_demand = {n for n, p in residency.items() if p == ON_DEMAND}
        reload_nodes = {n for n, p in residency.items() if p == RELOAD}
        paged = on_demand | reload_nodes

        # Resolve autocast dtype: explicit YAML config wins; otherwise we
        # fall back to the Model's own preference (so models that need to
        # match a reference numerically can override get_autocast_dtype
        # without forcing every config file to set the same value).
        autocast_dtype = model.get_autocast_dtype()
        if "autocast_dtype" in model_config:
            autocast_dtype = model_config["autocast_dtype"]

        # Allocation dtype hint for get_submodule: a node's params should be
        # allocated directly in ``autocast_dtype`` (cast meta -> dtype before
        # ``to_empty``) instead of allocating fp32 then down-casting after
        # return — the latter doubles the load-time VRAM peak. The cast below
        # applies the same resolved dtype, so allocation and runtime agree.
        submodules: dict[str, torch.nn.Module] = {}
        for name in node_names:
            # Sequence parallelism is opt-in per node; only forward the SP
            # group when this node actually participates in one, so models
            # that don't support SP keep their existing get_submodule call.
            extra = {}
            node_sp_group = parallel_groups.get_sp_config_for_node(name)
            if node_sp_group.world_size > 1:
                extra["sp_group"] = node_sp_group
            # An on_demand node materialises on the host; the engine moves it
            # to the device around each execution.
            build_device = torch.device("cpu") if name in paged else device
            submodule = model.get_submodule(
                name, build_device,
                tp_group=parallel_groups.get_tp_config_for_node(name),
                autocast_dtype=autocast_dtype, **extra,
            )
            if submodule is None:
                continue
            if submodule.disable_autocast:
                # keeps the dtype it was built in; the engine won't autocast it
                submodules[name] = submodule.to(device=build_device)
                continue
            # A submodule that must run in its own dtype (an audio codec in
            # fp32, say) says so; otherwise it takes the engine's.
            node_dtype = submodule.get_autocast_dtype() or autocast_dtype
            submodules[name] = submodule.to(device=build_device, dtype=node_dtype)

        # A reload node keeps no weights between executions, so release the
        # construction copy now: startup then holds one component at a time
        # instead of all of them, which is the whole point on a box where the
        # model's total exceeds physical memory. `meta` frees the storage while
        # leaving the module object valid for bookkeeping (remove_request calls
        # cleanup_request on every node, resident or not).
        for name in reload_nodes:
            if name in submodules:
                submodules[name] = submodules[name].to("meta")
                # Release the model's cached reference too. Without this the
                # FIRST load has nothing to evict, so the factory returns this
                # same object straight from get_submodule's memo — now on meta —
                # and dies on .to(device). The eviction path releases as well;
                # startup is the case that never reaches it.
                model.release_submodule(name)

        engine = Engine(
            autocast_dtype=autocast_dtype,
            enable_nvtx=enable_nvtx,
            enable_profile=enable_prof,
        )
        engine.load_model(
            submodules,
            specs=specs,
            parallel_groups=parallel_groups,
            device=device,
            transfer_engine_info=transfer_engine_info,
            kv_cache_type=autocast_dtype,
            on_demand_nodes=on_demand,
            reload_nodes=reload_nodes,
            submodule_factory=_make_submodule_factory(
                model, device, parallel_groups, autocast_dtype,
            ) if reload_nodes else None,
            submodule_release=model.release_submodule if reload_nodes else None,
        )
        logger.info("Engine loaded on device %s for nodes %s", device, sorted(node_names))
        for policy, names in (("on_demand", on_demand), ("reload", reload_nodes)):
            if names:
                logger.info("Engine: %s nodes: %s", policy, sorted(names))

        return cls(engine=engine, node_names=set(node_names))

    def warmup_all(self) -> None:
        """CUDA graph capture, for the whole forward and any piecewise region."""
        with torch.no_grad():
            self.engine.warmup()

    def get_engine(self, node_name: str) -> Engine:
        del node_name  # one engine serves every node
        return self.engine

    def add_request(
        self, request_id: str,
        resource_configs: dict[str, ResourceReqConfig] | None = None,
    ) -> None:
        """Open resource state for a request, on the per-resource configs the
        conductor resolved for it (``Model.get_request_resource_configs``)."""
        self.engine.add_request(request_id, resource_configs)

    def remove_request(self, request_id: str) -> None:
        self.engine.remove_request(request_id)

    def evictable_nodes(self) -> list[str]:
        """Nodes whose state can be reclaimed. The worker keeps the per-request
        LRU timestamps it needs to pick victims among them."""
        return [name for name in sorted(self.node_names) if self.engine.evictable(name)]

    def shutdown(self) -> None:
        self.engine.shutdown()
