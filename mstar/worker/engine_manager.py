import logging
from dataclasses import dataclass, field

import torch

from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources import (
    NodeResourceSpec,
    ResourceReqConfig,
    apply_yaml_overrides,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.position.config import PositionSpec, PosScheme
from mstar.graph.runtime.base import GraphRuntime
from mstar.graph.special_destinations import SPECIAL_DESTINATIONS
from mstar.model.base import Model, keyed_labels
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.device import DeviceCapability
from mstar.utils.streams import reset_device_scheduling

logger = logging.getLogger(__name__)


def _refuse_uncacheable_positions(
    specs: list[NodeResourceSpec], model: Model,
) -> None:
    """Refuse a keyed cache whose positions are not sequential.

    A cached page is reusable only at the positions it was written at, since its
    K is stored already rotated.
    """
    declared = set(keyed_labels(model))
    if not declared:
        return
    for spec in specs:
        if not isinstance(spec, PositionSpec):
            continue
        cached = declared & spec.depends_on()
        if cached and spec.config.scheme is not PosScheme.SEQUENTIAL:
            raise ValueError(
                f"resource {spec.resource_key!r} places positions by "
                f"{spec.config.scheme.value}, but it sits over "
                f"{sorted(cached)}, which the model keyed for prefix reuse; "
                "either drop the stream or give the resource a sequential "
                "scheme"
            )


def _refuse_unskippable_resources(
    specs: list[NodeResourceSpec], model: Model,
) -> None:
    """Refuse a declared node carrying a resource that cannot skip a prefix.

    A hit takes the front of a request away from every resource on the node,
    not just the ones that answered the probe, so one that cannot account for
    tokens it never saw would plan its step against a sequence that never ran.
    """
    declared = set(keyed_labels(model))
    if not declared:
        return
    cached_nodes = {
        node for spec in specs if spec.resource_key in declared
        for node in spec.nodes
    }
    for spec in specs:
        shared = spec.nodes & cached_nodes
        if not shared or spec.resource_class.prefix_skip_safe:
            continue
        raise ValueError(
            f"resource {spec.resource_key!r} cannot serve a request whose "
            f"prefix came from the cache, but it sits on {sorted(shared)}, "
            "which the model keyed for prefix reuse; either drop the stream "
            "or give the resource the prefix hooks"
        )


def _refuse_unsupported_chunking(
    submodules: dict, model: Model, parallel_groups: WorkerParallelGroups,
) -> None:
    """A chunked walk must report row lengths, and chunking under SP is not supported yet."""
    for walk, section in model.get_graph_walk_graphs().items():
        for node in section.get_nodes():
            submodule = submodules.get(node)
            if submodule is None or not submodule.get_chunking_policy(walk).chunkable:
                continue
            if type(submodule).get_input_sequence_len is NodeSubmodule.get_input_sequence_len:
                raise ValueError(
                    f"{type(model).__name__}: {node!r} chunks {walk!r} but does not "
                    "report get_input_sequence_len"
                )
            if parallel_groups.get_sp_config_for_node(node).world_size > 1:
                raise NotImplementedError(
                    f"{type(model).__name__}: chunked prefill of {node!r} under SP"
                )


def _refuse_split_combined_walks(submodules: dict, model: Model) -> None:
    """A combined walk batches its walks together, so it must not split them by capture key."""
    for (node, _), combined in model.combined_walk_of().items():
        submodule = submodules.get(node)
        if submodule is not None and submodule.split_batches_by_capture_key(combined):
            raise ValueError(
                f"{type(model).__name__}: {node!r} splits its combined walk "
                f"{combined!r} by capture key"
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


def _refuse_unknown_walks(model: Model) -> None:
    """Refuse a declared label that names a walk the model never runs, or a
    growth loop its walk doesn't run.

    Only the named walks may file pages or be probed, so a name that matches
    nothing leaves that cache empty for good, and nothing says why.
    """
    graphs = model.get_graph_walk_graphs()
    walks = set(graphs)
    for key, by_label in model.kv_labels().items():
        for label, kv_label in by_label.items():
            named = {walk for walk, _ in kv_label.growth}
            if kv_label.keys is not None:
                named.update(kv_label.keys.walks)
            unknown = sorted(named - walks)
            if unknown:
                raise ValueError(
                    f"{type(model).__name__} declares {key!r}/{label!r} over "
                    f"{unknown}, which it never runs; its walks are "
                    f"{sorted(walks)}"
                )
            for walk, loop in kv_label.growth:
                loops = graphs[walk].get_loops()
                if loop not in loops:
                    raise ValueError(
                        f"{type(model).__name__} declares {key!r}/{label!r} growing on "
                        f"{loop!r}, which {walk!r} doesn't run; its loops are {sorted(loops)}"
                    )


def _refuse_undeclared_pools(specs: list[NodeResourceSpec], model: Model) -> None:
    """A ring pool needs no labels, since its size is fixed per request."""
    declared = model.kv_labels()
    for spec in specs:
        if isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig) and not declared.get(spec.resource_key):
            raise ValueError(
                f"{type(model).__name__} declares no labels for its paged pool "
                f"{spec.resource_key!r}; add them to `kv_labels`, and return each "
                "request's layout for them from `process_prompt`"
            )


def _refuse_unservable_walks(specs: list[NodeResourceSpec], model: Model) -> None:
    """Refuse a keyed label over several walks if the cache could not serve one
    of them whole: a walk where a keyed node feeds another node, or where no
    keyed node runs.

    A walk the cache serves whole completes with no outputs, so a node that
    reads them would get nothing. A label keyed over one walk is not checked,
    because its one span always runs its last token.
    """
    graphs = model.get_graph_walk_graphs()
    by_key = {spec.resource_key: spec for spec in specs}
    for key, by_label in keyed_labels(model).items():
        for label, kv_label in by_label.items():
            if len(kv_label.keys.walks) < 2:
                continue
            keyed_nodes = by_key[key].nodes
            for walk in kv_label.keys.walks:
                nodes = graphs[walk].get_nodes()
                ran = sorted(keyed_nodes & set(nodes))
                if not ran:
                    raise ValueError(
                        f"{type(model).__name__} lays {key!r}/{label!r} out over "
                        f"{walk!r}, where none of {sorted(keyed_nodes)} runs"
                    )
                for name in ran:
                    fed = sorted(
                        {edge.next_node for edge in nodes[name].outputs}
                        - SPECIAL_DESTINATIONS
                    )
                    if fed:
                        raise ValueError(
                            f"{type(model).__name__} lays {key!r}/{label!r} out over "
                            f"{walk!r}, where {name} feeds {fed}, which would get "
                            "nothing from a walk the cache serves whole"
                        )


@dataclass
class LoadedSubmodules:
    """A worker's submodules before its engine exists."""
    submodules: dict[str, NodeSubmodule]
    specs: list
    autocast_dtype: torch.dtype

    def chunkable_node_walks(self, model: Model) -> list[tuple[str, str]]:
        """(node, walk) pairs whose prefill may be split, which speculation
        must never target (``disable_spec_node_walks``)."""
        return sorted(
            (node, walk)
            for walk, section in model.get_graph_walk_graphs().items()
            for node in section.get_nodes()
            if node in self.submodules
            and self.submodules[node].get_chunking_policy(walk).chunkable
        )


@dataclass
class EngineManager:
    """Owns the worker's engine.

    One engine serves every node: it holds all the submodules and the
    resources they share, so there is nothing to group or dedup.
    """
    engine: Engine
    node_names: set[str] = field(default_factory=set)

    @staticmethod
    def load_submodules(
        node_names: set[str],
        device: torch.device,
        model_config: dict,
        parallel_groups: WorkerParallelGroups,
        model: Model,
    ) -> LoadedSubmodules:
        """This worker's nodes' submodules and resource specs, checked; the
        engine is built from them by ``build``.

        The model supplies each node's submodule via ``get_submodule`` and its
        resources via ``get_node_resources``; the KV cache's shape rides on
        the latter, so there is no separate cache config.
        """
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, model_config)
        _refuse_uncacheable_positions(specs, model)
        _refuse_unskippable_resources(specs, model)
        _refuse_unknown_walks(model)
        _refuse_undeclared_pools(specs, model)
        _refuse_unservable_walks(specs, model)

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
        for name in sorted(node_names):
            # Sequence parallelism is opt-in per node; only forward the SP
            # group when this node actually participates in one, so models
            # that don't support SP keep their existing get_submodule call.
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
        _refuse_unsupported_chunking(submodules, model, parallel_groups)
        return LoadedSubmodules(submodules, specs, autocast_dtype)

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
        loaded: LoadedSubmodules | None = None,
    ) -> "EngineManager":
        """Build the engine and load this worker's nodes into it, from
        ``loaded`` when the caller needed the submodules first."""
        if loaded is None:
            loaded = cls.load_submodules(node_names, device, model_config, parallel_groups, model)
        submodules, specs, autocast_dtype = loaded.submodules, loaded.specs, loaded.autocast_dtype

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
        engine.set_token_budgets(
            budgets, set(model.get_graph_walk_graphs()),
            combined_walk_of=model.combined_walk_of(),
            override_mixed={node for node, flag in override_mixed.items() if flag},
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
