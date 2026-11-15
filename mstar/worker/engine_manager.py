import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.accelerator_graph_config import check_token_ladder
from mstar.engine.chunking import DEFAULT_MIN_CHUNK_TOKENS
from mstar.engine.engine import Engine
from mstar.engine.node_dtype import resolve_model_dtype, resolve_node_dtype
from mstar.engine.resources import ResourceReqConfig
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.state.analysis import declaration_digests, digest_mismatch
from mstar.graph.base import GraphNode
from mstar.graph.runtime.base import GraphRuntime, inline_column
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    Model,
    deployment_setup,
    node_moe_parallel_configs,
    parse_autocast_dtype,
    serving_resources,
)
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.device import DeviceCapability
from mstar.utils.streams import reset_device_scheduling

logger = logging.getLogger(__name__)


def _prefill_token_buckets(model_config: dict) -> Mapping:
    """The serving config's ``prefill_token_buckets``: node -> capture walk or
    piecewise region -> ladder (`_split_token_ladders`)."""
    # the model_kwargs ladders this key replaced; a model now drops them in **kwargs unread
    moved = sorted(
        {"prefill_token_buckets", "prefill_batched_token_buckets", "backbone_token_buckets", "lm_prefill_token_buckets"}
        & set(model_config.get("model_kwargs") or {})
    )
    if moved:
        raise ValueError(
            f"model_kwargs {moved} moved to the top level of the serving config, per node and "
            "capture walk or region: prefill_token_buckets: {<node>: {<walk or region>: [...]}}"
        )
    ladders = model_config.get("prefill_token_buckets")
    if ladders is None:
        return {}
    if not isinstance(ladders, Mapping) or not all(isinstance(v, Mapping) for v in ladders.values()):
        raise ValueError(
            f"prefill_token_buckets takes node -> capture walk -> ladder, got {ladders!r}"
        )
    return ladders


def _split_token_ladders(
    ladders: Mapping[str, Mapping], model: Model,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Each node's ladders by walk (a walk the node is in, or a combined walk
    of it), which its packed captures take (`apply_token_ladders`), and by
    any other name, which can only be a piecewise region's label
    (`Engine._build_piecewise_runners` checks that, once the node declares
    its regions). Refuses, from the config and the model's graph alone, a
    ladder that is not one, a region's that is not a list, and a walk of the
    model that does not run the node."""
    graphs = model.get_graph_walk_graphs()
    combined = model.get_combined_graph_walks()
    every_walk = set(graphs) | {name for walks in combined.values() for name in walks}
    by_walk: dict[str, dict] = {}
    by_region: dict[str, dict] = {}
    for node, named in ladders.items():
        walks = {walk for walk, graph in graphs.items() if node in graph.get_nodes()} | set(combined.get(node, {}))
        for name, ladder in named.items():
            check_token_ladder(f"{node}/{name}", ladder)
            if name in walks:
                by_walk.setdefault(node, {})[name] = ladder
            elif name in every_walk:
                raise ValueError(
                    f"prefill_token_buckets names walk {name!r} for {node}, which it does not run; "
                    f"its walks are {sorted(walks)}"
                )
            elif isinstance(ladder, Mapping):
                raise ValueError(
                    f"prefill_token_buckets names {name!r} for {node}, not one of its walks {sorted(walks)}, "
                    f"so a piecewise region, which takes one ladder for every batch size, got {ladder!r}"
                )
            else:
                by_region.setdefault(node, {})[name] = ladder
    return by_walk, by_region


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
        if parallel_groups.get_joint_group_for_node(node).sp_group.world_size > 1:
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


def _refuse_unlisted_grammar_stops(submodules: dict, specs: Collection, model: Model) -> None:
    """The grammar sampler ends a constrained reply on the ids ``text_output()``
    declares, and the loop ends it on its ``TokenStop`` ids. A stop id of a loop
    that draws from the grammar sampler and is not declared would be masked out
    of a finished grammar's row, so the reply would never end there."""
    keys = {spec.resource_key: spec for spec in specs if isinstance(spec, SamplerSpec) and spec.enable_grammar}
    for name, submodule in submodules.items():
        stop = getattr(submodule, "token_stop", None)
        if stop is None or stop.sampler not in keys:
            continue
        text = model.text_output()
        if text is None:  # the sampler's own refusal says so
            return
        tokenizer = getattr(text.detokenizer, "tokenizer", None)
        vocab_size = keys[stop.sampler].vocab_size
        if tokenizer is not None and vocab_size is not None and vocab_size < len(tokenizer):
            continue  # not the text sampler: the sampler's own refusal names that
        if missing := stop.stop_ids - frozenset(text.stop_token_ids):
            raise ValueError(
                f"{type(model).__name__}: {name!r} draws from {stop.sampler!r} with enable_grammar and stops on "
                f"token ids {sorted(missing)} that its text_output() does not list: declare them there, "
                "or put enable_grammar on the text sampler"
            )


def _refuse_unroutable_withholds(submodules: dict, model: Model) -> None:
    """A withheld output (``TokenStop.withhold``) is left out of the route
    after the stop check, so it must be a streaming edge's (which then gets no
    item, as a tensor or by value; any other consumer would wait for one), and
    one that no speculated step reads first. The loop-back signal
    (``ARNodeSubmodule.device_loopback``) is kept on the device, so
    withholding it does nothing."""
    for walk, section in model.get_graph_walk_graphs().items():
        nodes = section.get_nodes()
        for name, node in nodes.items():
            submodule = submodules.get(name)
            stop = getattr(submodule, "token_stop", None)
            if stop is None:
                continue
            loopback = getattr(submodule, "device_loopback", None)
            loopback_signal = loopback.signal if loopback is not None and loopback.covers(walk) else None
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


def _refuse_combined_loopbacks(submodules: dict, model: Model) -> None:
    """A ``DeviceLoopback`` names real walks: a combined step's rows loop back
    per their own walk (``GpuStepBody._loopback_rids``), and a combined walk
    named here would take the whole step rows-only past that check."""
    for node, combined in model.get_combined_graph_walks().items():
        loopback = getattr(submodules.get(node), "device_loopback", None)
        named = sorted(walk for walk in combined if loopback is not None and loopback.covers(walk))
        if named:
            raise ValueError(
                f"{type(model).__name__}: {node!r}'s DeviceLoopback names combined walk(s) {named}; "
                "name their real walks instead"
            )


class InlineStreams(NamedTuple):
    """Which streamed signals go as values (``_inline_streams``)."""
    # (node, signal) pairs whose streamed items the route sends as values
    keys: frozenset[tuple[str, str]]
    # (node, walk) -> its streamed signals that go as tensors there: a step keeps
    # such a signal's rows on the device only if nothing streams it
    tensor_streams: dict[tuple[str, str], frozenset[str]]
    # (node, walk) -> the signals its inline column carries on every edge, so
    # a step needs no tensor for them (``GpuStepBody.all_on_device``)
    carried: dict[tuple[str, str], frozenset[str]]


def _inline_streams(
    submodules: dict, model: Model, parallel_groups: WorkerParallelGroups | None = None,
) -> InlineStreams:
    """The streams whose items the route sends as values: a ``by_value``
    connection's, for the graph runtime. On a walk, such a stream goes by value
    only when its signal rides the step's inline column (``inline_column``);
    every other stream, and a lockstep node's (whose replicated stream's
    consumer counts items from every rank, which a value cannot say), keeps
    its tensors.

    A signal of the column is carried when every edge of it rides too: a
    client emit or a by-value stream, neither persisted, or an empty
    destination; none of them counted toward the conductor's output tokens,
    which it counts off tensors; and no enclosing loop collects it as an
    output.

    These are ``node.outputs``, and the step's ``output_signals`` are the
    same edges' names: the worker graphs keep the model's ``GraphNode``
    objects, and the runtime lists a node's outputs from them."""
    topology = model.get_partition_topology()
    by_value = {c.edge_name for c in (topology.connections if topology else []) if c.by_value}
    keys = set()
    tensor_streams: dict[tuple[str, str], set[str]] = {}
    carried: dict[tuple[str, str], frozenset[str]] = {}
    for walk, section in model.get_graph_walk_graphs().items():
        loops = section.get_loops().values()
        for name, node in section.get_nodes().items():
            submodule = submodules.get(name)
            lockstep = parallel_groups is not None and parallel_groups.get_instance_world_size_for_node(name) > 1
            outputs = {edge.name for edge in node.outputs}
            inline = submodule.inline_signals(walk) if submodule is not None else {}
            riding = inline_column(inline, outputs).signals
            for edge in node.outputs:
                if not edge.is_streaming:
                    continue
                if edge.name in by_value and not lockstep:
                    keys.add((name, edge.name))
                    if edge.name in riding:
                        continue
                tensor_streams.setdefault((name, walk), set()).add(edge.name)
            if riding:
                collected = {
                    edge.name for loop in loops if name in loop.get_nodes()
                    for edge in (*loop.outputs, *loop.accumulated_outputs)
                }

                def rides(edge, lockstep=lockstep):
                    # the route counts a step's new tokens off its tensors
                    if edge.persist or (edge.conductor_new_token and not edge.is_streaming):
                        return False
                    if edge.is_streaming:
                        return edge.name in by_value and not lockstep
                    return edge.next_node in (EMIT_TO_CLIENT, EMPTY_DESTINATION)

                kept = frozenset(
                    signal for signal in riding if signal not in collected
                    and all(rides(edge) for edge in node.outputs if edge.name == signal)
                )
                if kept:
                    carried[(name, walk)] = kept
    return InlineStreams(frozenset(keys), {k: frozenset(v) for k, v in tensor_streams.items()}, carried)


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
    (``StepPipeline._launch_side``): the ``side_stream_walks`` of the node whose
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
            raise refuse("it runs in a lockstep instance (TP, SP or CFG): when a side step lands "
                         "is per rank, and every rank of the instance has to agree on it")
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
        stop is not None and (stop.stops_on(walk) or walk in stop.prefill_walks)
    ) or (loopback is not None and loopback.covers(walk))


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
    # which streamed signals go as values (``_inline_streams``)
    inline_streams: InlineStreams = InlineStreams(frozenset(), {}, {})

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
        conductor_digests: Mapping[str, str] | None = None,
    ) -> "EngineManager":
        """Build the engine and load this worker's nodes into it.

        The model supplies each node's submodule via ``get_submodule`` and its
        resources via ``declare_state`` and ``get_node_resources``; the KV cache's
        shape rides on those, so there is no separate cache config. The conductor's
        ``validate_deployment`` has already checked the specs and overrides.
        ``conductor_digests`` are the conductor's `declaration_digests`, which this
        worker's must equal.
        """
        if conductor_digests is not None:
            mismatch = digest_mismatch(
                declaration_digests(model.declare_state(), model.kernel_settings()), conductor_digests,
                here=f"worker {transfer_engine_info.my_entity_id}", there="conductor",
            )
            if mismatch is not None:
                raise RuntimeError(mismatch)
        # the per-node settings read only the config: refuse a bad one before any weights load
        budgets = model_config.get("max_batch_tokens") or {}
        override_mixed = model_config.get("can_override_mixed_max_tokens") or {}
        ladders = _prefill_token_buckets(model_config)
        known = {n for group in model_config.get("node_groups", []) for n in group.get("node_names", [])}
        for key, setting in (
            ("max_batch_tokens", budgets), ("can_override_mixed_max_tokens", override_mixed),
            ("prefill_token_buckets", ladders),
        ):
            unknown = sorted(set(setting) - known)
            if unknown:
                raise ValueError(f"{key} names nodes {unknown} that no node group serves")
        walk_ladders, region_ladders = _split_token_ladders(ladders, model)
        setup = deployment_setup(model, model_config)
        specs = serving_resources(model, model_config, setup)

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
            # CFG parallelism is opt-in per node (``cfg_group``), and ``moe_parallel`` only for a
            # node group's ``ep_config``, so every other node keeps its existing get_submodule call
            extra = {}
            joint = parallel_groups.get_joint_group_for_node(name)
            if joint.cfg_group.world_size > 1:
                extra["cfg_group"] = joint.cfg_group
            if name in moe_parallel:
                extra["moe_parallel"] = moe_parallel[name]
            submodule = model.get_submodule(
                name, device, tp_group=joint.tp_group,
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
        _refuse_combined_loopbacks(submodules, model)
        _refuse_unlisted_grammar_stops(submodules, specs, model)
        inline_streams = _inline_streams(submodules, model, parallel_groups)
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
            state_setup=setup,
        )
        engine.load_model(
            submodules,
            specs=specs,
            parallel_groups=parallel_groups,
            device=device,
            transfer_engine_info=transfer_engine_info,
            kv_cache_type=autocast_dtype,
            model=model,
            moe_parallel=moe_parallel,
            token_ladders=walk_ladders,
            region_ladders=region_ladders,
        )
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

        return cls(
            engine=engine, node_names=set(node_names), side_keys=side_keys,
            inline_streams=inline_streams,
        )

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
