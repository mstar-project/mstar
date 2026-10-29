import logging
from dataclasses import dataclass, field

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.chunking import DEFAULT_MIN_CHUNK_TOKENS
from mstar.engine.engine import Engine
from mstar.engine.resources import ResourceReqConfig, apply_yaml_overrides
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.graph.runtime.base import GraphRuntime
from mstar.model.base import Model, node_moe_parallel_configs
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.device import DeviceCapability
from mstar.utils.streams import reset_device_scheduling

logger = logging.getLogger(__name__)


def _unchunkable_nodes(
    submodules: dict, model: Model, parallel_groups: WorkerParallelGroups,
) -> frozenset[str]:
    """Check what a chunked walk must declare, and name the nodes this worker
    turns chunking off for.

    Reporting row lengths is the model's own declaration, so a walk that
    chunks without ``get_input_sequence_len`` is refused anywhere. Under TP the
    leader cuts the chunks and the followers run them off its head. Chunking
    under SP is not supported yet; refusing there would take SP away from
    every model that chunks, so those nodes run their prefills whole instead.
    """
    off = set()
    for (node, walk), policy in model.chunking_policy_of().items():
        submodule = submodules.get(node)
        if submodule is None or not policy.chunkable:
            continue
        if type(submodule).get_input_sequence_len is NodeSubmodule.get_input_sequence_len:
            raise ValueError(
                f"{type(model).__name__}: {node!r} chunks {walk!r} but does not "
                "report get_input_sequence_len"
            )
        if parallel_groups.get_sp_config_for_node(node).world_size > 1:
            off.add(node)
    if off:
        logger.warning(
            "chunked prefill is off for %s: not supported under SP yet, so "
            "their prefills run whole", sorted(off),
        )
    return frozenset(off)

def _refuse_split_combined_walks(submodules: dict, model: Model) -> None:
    """A combined walk batches its walks together, so it must not split them by capture key."""
    for (node, _), combined in model.combined_walk_of().items():
        submodule = submodules.get(node)
        if submodule is not None and submodule.split_batches_by_capture_key(combined):
            raise ValueError(
                f"{type(model).__name__}: {node!r} splits its combined walk "
                f"{combined!r} by capture key"
            )


def _refuse_speculated_spans(submodules: dict, model: Model) -> None:
    """A spanned output is cut after its step ends, but speculation hands a
    step's outputs to the next step before then: no spanned output may feed a
    node this worker speculates into (see ``NodeSubmodule.spanned_outputs``)."""
    for walk, section in model.get_graph_walk_graphs().items():
        nodes = section.get_nodes()
        for name, node in nodes.items():
            submodule = submodules.get(name)
            if submodule is None or not node.enable_async_scheduling:
                continue
            spanned = submodule.spanned_outputs(walk)
            for edge in node.outputs:
                target = nodes.get(edge.next_node)
                if (
                    edge.name in spanned and edge.next_node in submodules
                    and target is not None and target.enable_async_scheduling
                ):
                    raise ValueError(
                        f"{type(model).__name__}: {name!r} spans {edge.name!r}, which "
                        f"feeds {edge.next_node!r} on the same worker in walk {walk!r}; "
                        "set enable_async_scheduling=False on one of them"
                    )


def _tp_group_of(sharding: ShardingConfig, node: str, walk: str) -> ShardingGroup | None:
    """The TP group ``node`` runs ``walk`` in, from the static config; None
    when it has none or one of size 1."""
    for group in sharding.groups:
        if node in group.nodes and (group.graph_walks is None or walk in group.graph_walks):
            return group if group.tp_size > 1 else None
    return None


def refuse_combined_walks_across_tp_groups(
    model: Model, sharding: ShardingConfig, my_walks: set[tuple[str, str]],
) -> None:
    """A combined walk runs every walk's rows in one forward, so on this
    worker its walks must share one TP group, or all have world size 1.

    Keyed on the (node, walk) pairs this worker runs, not its nodes: under
    P/D each half of a combined walk sits on a different worker, so neither
    can build a step that mixes them and neither needs one group.
    """
    groups: dict[tuple[str, str], dict[str, ShardingGroup | None]] = {}
    for (node, walk), combined in model.combined_walk_of().items():
        if (node, walk) in my_walks:
            groups.setdefault((node, combined), {})[walk] = _tp_group_of(sharding, node, walk)
    for (node, combined), by_walk in groups.items():
        if len({id(group) for group in by_walk.values()}) > 1:
            sizes = {walk: 1 if g is None else g.tp_size for walk, g in sorted(by_walk.items())}
            raise ValueError(
                f"{type(model).__name__}: combined walk {combined!r} of {node!r} "
                f"spans TP groups (walk -> TP size: {sizes}); its walks must "
                "share one group or all have world size 1"
            )


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
        graph_runtime: GraphRuntime,
        enable_nvtx: bool = False,
        enable_prof: bool=False,
        device_capability: DeviceCapability | None = None,
    ) -> "EngineManager":
        """Build the engine and load this worker's nodes into it.

        The model supplies each node's submodule via ``get_submodule`` and its
        resources via ``get_node_resources``; the KV cache's shape rides on
        the latter, so there is no separate cache config. The conductor's
        ``validate_deployment`` has already checked the specs and overrides.
        """
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, model_config)

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
        # Checked by the conductor's get_sharding_config before spawn.
        moe_parallel = node_moe_parallel_configs(model_config["node_groups"])
        for name in sorted(node_names):
            # Sequence parallelism is opt-in per node; only forward the SP
            # group when this node actually participates in one, so models
            # that don't support SP keep their existing get_submodule call.
            # Likewise ``moe_parallel``, only for a node group's ``ep_config``.
            extra = {}
            node_sp_group = parallel_groups.get_sp_config_for_node(name)
            if node_sp_group.world_size > 1:
                extra["sp_group"] = node_sp_group
            if name in moe_parallel:
                extra["moe_parallel"] = moe_parallel[name]
            submodule = model.get_submodule(
                name, device,
                tp_group=parallel_groups.get_tp_config_for_node(name),
                autocast_dtype=autocast_dtype, **extra,
            )
            if submodule is None:
                continue
            if submodule.disable_autocast:
                # keeps the dtype it was built in; the engine won't autocast it
                submodules[name] = submodule.to(device=device)
                continue
            # A submodule that must run in its own dtype (an audio codec in
            # fp32, say) says so; otherwise it takes the engine's.
            node_dtype = submodule.get_autocast_dtype() or autocast_dtype
            submodules[name] = submodule.to(device=device, dtype=node_dtype)
        _refuse_split_combined_walks(submodules, model)
        _refuse_speculated_spans(submodules, model)
        no_chunking = _unchunkable_nodes(submodules, model, parallel_groups)

        engine = Engine(
            graph_runtime=graph_runtime,
            autocast_dtype=autocast_dtype,
            enable_nvtx=enable_nvtx,
            enable_profile=enable_prof,
            gpu_memory_fraction=model_config.get("gpu_memory_fraction"),
            device_capability=device_capability,
        )
        engine.load_model(
            submodules,
            specs=specs,
            parallel_groups=parallel_groups,
            device=device,
            transfer_engine_info=transfer_engine_info,
            kv_cache_type=autocast_dtype,
            model=model,
        )
        budgets = model_config.get("max_batch_tokens") or {}
        override_mixed = model_config.get("can_override_mixed_max_tokens") or {}
        known = {n for group in model_config.get("node_groups", []) for n in group.get("node_names", [])}
        for key, setting in (("max_batch_tokens", budgets), ("can_override_mixed_max_tokens", override_mixed)):
            unknown = sorted(set(setting) - known)
            if unknown:
                raise ValueError(f"{key} names nodes {unknown} that no node group serves")
        if not all(isinstance(flag, bool) for flag in override_mixed.values()):
            raise ValueError(f"can_override_mixed_max_tokens takes node -> true/false, not {override_mixed!r}")
        engine.set_chunking_policies(model.chunking_policy_of())
        engine.set_token_budgets(
            budgets, set(model.get_graph_walk_graphs()),
            combined_walk_of=model.combined_walk_of(),
            override_mixed={node for node, flag in override_mixed.items() if flag},
            no_chunking=no_chunking,
            min_chunk_tokens=model_config.get("min_chunk_tokens", DEFAULT_MIN_CHUNK_TOKENS),
        )
        logger.info("Engine loaded on device %s for nodes %s", device, sorted(node_names))

        return cls(engine=engine, node_names=set(node_names))

    def warmup_all(self) -> None:
        """CUDA graph capture, for the whole forward and any piecewise region."""
        with torch.no_grad():
            self.engine.warmup()
        # warmup ran side-stream work beside graph replays; start serving from
        # a clean scheduling state (see reset_device_scheduling)
        reset_device_scheduling(self.engine._device)

    def get_engine(self, node_name: str) -> Engine:
        del node_name  # one engine serves every node
        return self.engine

    def add_request(
        self, request_id: str,
        resource_configs: dict[str, ResourceReqConfig] | None = None,
        request_info: CurrentForwardPassInfo | None = None,
    ) -> None:
        """Open resource state for a request, on the per-resource configs the
        conductor resolved for it (``Model.get_request_resource_configs``),
        and hand the submodules its ``request_info``."""
        self.engine.add_request(request_id, resource_configs, request_info)

    def remove_request(self, request_id: str) -> set[str]:
        """Tear the request down; returns the resources it handed pages back to."""
        return self.engine.remove_request(request_id)

    def evictable_nodes(self) -> list[str]:
        """Nodes whose state can be reclaimed. The worker keeps the per-request
        LRU timestamps it needs to pick victims among them."""
        return [name for name in sorted(self.node_names) if self.engine.evictable(name)]

    def shutdown(self) -> None:
        self.engine.shutdown()
