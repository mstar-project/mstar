import logging
from collections.abc import Collection
from dataclasses import dataclass, field

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.chunking import DEFAULT_MIN_CHUNK_TOKENS
from mstar.engine.engine import Engine
from mstar.engine.node_dtype import resolve_model_dtype, resolve_node_dtype
from mstar.engine.resources import ResourceReqConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.graph.base import GraphNode
from mstar.graph.runtime.base import GraphRuntime
from mstar.model.base import Model, node_moe_parallel_configs, parse_autocast_dtype, serving_resources
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


def _speculates_into(node, edge, nodes: dict, submodules: dict) -> bool:
    """Whether this worker may run ``edge``'s consumer speculatively, on the
    step's outputs before the step's postprocess has seen them."""
    target = nodes.get(edge.next_node)
    return (
        node.enable_async_scheduling and edge.next_node in submodules
        and target is not None and target.enable_async_scheduling
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
                if edge.name in spanned and _speculates_into(node, edge, nodes, submodules):
                    raise ValueError(
                        f"{type(model).__name__}: {name!r} spans {edge.name!r}, which "
                        f"feeds {edge.next_node!r} on the same worker in walk {walk!r}; "
                        "set enable_async_scheduling=False on one of them"
                    )


def _refuse_unroutable_withholds(submodules: dict, model: Model) -> None:
    """A withheld output (``TokenStop.withhold``) is left out of the route
    after the stop check, so it must be one the worker routes as a tensor on
    a streaming edge (which then gets no item; any other consumer would wait
    for one), and that no speculated step reads first. The loop-back signal
    (``ARNodeSubmodule.device_loopback``) is sent inline or kept on the
    device, so withholding it does nothing."""
    for walk, section in model.get_graph_walk_graphs().items():
        nodes = section.get_nodes()
        for name, node in nodes.items():
            submodule = submodules.get(name)
            stop = getattr(submodule, "token_stop", None)
            if stop is None:
                continue
            loopback = getattr(submodule, "device_loopback", None)
            loopback_signal = loopback.signal if loopback is not None and loopback.walk == walk else None
            for edge in node.outputs:
                if edge.name in stop.withhold and (
                    not edge.is_streaming or edge.name == loopback_signal
                    or _speculates_into(node, edge, nodes, submodules)
                ):
                    raise ValueError(
                        f"{type(model).__name__}: {name!r} withholds {edge.name!r} in walk {walk!r}; "
                        "only a streaming edge's output that no speculated node on this worker reads, "
                        "and not the loop-back signal, can be withheld"
                    )


def side_stream_nodes(node_groups: list[dict]) -> frozenset[str]:
    """The nodes whose node groups set ``side_stream: true``. A node listed by
    several groups, one per graph walk, must get the same answer from each."""
    asked: dict[str, bool] = {}
    for group in node_groups:
        flag = group.get("side_stream", False)
        if not isinstance(flag, bool):
            raise ValueError(f"side_stream takes true or false, not {flag!r}")
        for node in group["node_names"]:
            if asked.setdefault(node, flag) != flag:
                raise ValueError(f"node {node!r} gets different side_stream settings from two node_groups")
    return frozenset(node for node, flag in asked.items() if flag)


def _side_lane_keys(
    submodules: dict, specs: list, model: Model, node_names: Collection[str], asked: Collection[str],
    parallel_groups: WorkerParallelGroups, accelerator: bool,
) -> tuple[tuple[str, str], ...]:
    """The (node, walk) keys this worker runs beside its chain
    (``mstar.worker.side_steps``): the ``side_stream_walks`` of the node whose
    group sets ``side_stream: true``, or a ``ValueError`` naming why the lane
    can't take it. Only nodes that ask are checked.

    A side step is admitted and planned inline on the GPU thread, never
    speculated or planned ahead, and lands between two chain steps, so the
    lane takes a stream-fed walk of one node with no resources and no stop
    check, beside another node's chain.
    """
    walks = model.get_graph_walk_graphs()
    combined = model.combined_walk_of()
    keys = []
    for name in sorted(set(asked) & set(submodules)):
        submodule = submodules[name]
        side = submodule.side_stream_walks

        def refuse(reason: str, name=name) -> ValueError:
            return ValueError(f"{type(model).__name__}: {name!r} sets side_stream, but {reason}")

        if not side:
            raise refuse("its submodule declares no side_stream_walks")
        if parallel_groups.get_instance_world_size_for_node(name) > 1:
            raise refuse("it runs tensor or sequence parallel: when a side step lands is per rank, "
                         "and every rank of a lockstep instance has to agree on it")
        if set(node_names) <= {name}:
            raise refuse("it is the only node on its worker, so there is no chain to run beside")
        if not accelerator:
            raise refuse("its device has no streams")
        on_chain = sorted(w for w, graph in walks.items() if name in graph.get_nodes() and w not in side)
        if on_chain:
            raise refuse(f"it also runs {on_chain} on the chain, whose graphs would share its capture "
                         "stream and pool with the side walks' and could run beside them")
        held = sorted(spec.resource_key for spec in specs if name in spec.nodes)
        if held:
            raise refuse(f"it holds resources {held}, and the lane neither admits nor plans a step ahead")
        for walk in sorted(side):
            graph = walks.get(walk)
            if not isinstance(graph, GraphNode) or graph.name != name:
                raise refuse(f"side walk {walk!r} is not a walk of that one node")
            if (name, walk) in combined:
                raise refuse(f"side walk {walk!r} is part of combined walk {combined[name, walk]!r}, "
                             "which the chain schedules")
            if submodule.spanned_outputs(walk) or _stops_on_a_token(submodule, walk):
                raise refuse(f"side walk {walk!r} has output spans or a stop check, whose host rows "
                             "stage through a ring the chain's next steps reuse before a side step lands")
            keys.append((name, walk))
    if len({name for name, _ in keys}) > 1:
        raise ValueError(f"{type(model).__name__}: side_stream is set on {sorted({n for n, _ in keys})}; "
                         "a worker runs one side lane, for one node")
    return tuple(keys)


def _stops_on_a_token(submodule: NodeSubmodule, walk: str) -> bool:
    """Whether ``walk``'s steps hand the worker row buffers for a stop check."""
    stop = getattr(submodule, "token_stop", None)
    loopback = getattr(submodule, "device_loopback", None)
    return (
        stop is not None and (stop.walk in (None, walk) or walk in stop.prefill_walks)
    ) or (loopback is not None and loopback.walk == walk)


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
    # (node, walk) keys that run beside the chain, as ``build`` decided from
    # the config's ``side_stream`` (see ``_side_lane_keys``)
    side_keys: tuple[tuple[str, str], ...] = ()

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
        specs = serving_resources(model, model_config)

        # YAML autocast_dtype over the model's own, but never re-enabling
        # autocast for a model that pinned its numerics (see node_dtype)
        autocast_dtype = parse_autocast_dtype(resolve_model_dtype(model, model_config))

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
            # the one dtype the load cast and every forward and capture use;
            # None keeps the dtype the node was built in
            node_dtype = resolve_node_dtype(submodule, autocast_dtype)
            submodules[name] = submodule.to(device=device, dtype=node_dtype)
        _refuse_split_combined_walks(submodules, model)
        _refuse_speculated_spans(submodules, model)
        _refuse_unroutable_withholds(submodules, model)
        side_keys = _side_lane_keys(
            submodules, specs, model, node_names, side_stream_nodes(model_config["node_groups"]),
            parallel_groups, (device_capability or DeviceCapability.of(device)).is_accelerator,
        )
        no_chunking = _unchunkable_nodes(submodules, model, parallel_groups)

        engine = Engine(
            graph_runtime=graph_runtime,
            autocast_dtype=autocast_dtype,
            enable_nvtx=enable_nvtx,
            enable_profile=enable_prof,
            gpu_memory_fraction=model_config.get("gpu_memory_fraction"),
            device_capability=device_capability,
            side_stream_nodes={name for name, _ in side_keys},
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

        return cls(engine=engine, node_names=set(node_names), side_keys=side_keys)

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

    def shutdown(self) -> None:
        self.engine.shutdown()
