import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import NamedTuple, Type

import torch
import yaml

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    DEFAULT_PARTITION,
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.engine.resources import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.kv.config import RetentionPolicy
from mstar.engine.resources.sampler.utils import SamplingConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
    top_level_entities,
)
from mstar.model.media_limits import VideoBudget, read_image
from mstar.model.multimodal import TEXT, PromptPart

DECODE = "decode"
MAX_OUTPUT_TOKENS = 2048

@dataclass
class TensorAndMetadata:
    data: torch.Tensor
    metadata: dict = field(default_factory=dict)


class ProcessPromptOutput(NamedTuple):
    """Tensors from `process_prompt`, and metadata to carry beside them.

    The data worker folds ``metadata`` into the request's ``model_kwargs``, so
    what a model settles while tokenizing reaches
    `get_request_resource_configs` without a field of its own on the request.
    """
    new_input_tensors: NameToTensorList
    metadata: dict


class Span(NamedTuple):
    """One write to a KV stream. A layout lists them in the order the walks write.

    ``length`` is how many KV slots the write fills, and ``advance`` is how far
    it moves the position counter, which is the length for text. ``source`` is
    the tensor whose next entry an ids span keys, or the file of a digest span
    as ``(modality, index)``. ``params`` are the preprocessing arguments that
    change the tensors made from the file. A ``rows`` span writes KV made from
    anything else, such as another node's outputs, and is never keyed.
    ``scratch`` is the most a non-committing step writes before the span
    commits into those slots, as a denoised window does.
    """
    kind: str
    walk: str
    length: int
    advance: int
    source: str | tuple[str, int]
    params: tuple = ()
    scratch: int = 0


class LabelLayout(NamedTuple):
    """Everything a request writes into one paged KV label, from `process_prompt`'s ``kv_layout``.

    ``spans`` are the committed writes known at intake, in walk order. Each
    ``(loop, slots)`` in ``growth`` writes ``slots`` per iteration of the named
    loop. ``scratch`` is the most one non-committing step writes past the
    committed length, and ``retention`` the policy the label's committing steps declare.
    """
    spans: tuple[Span, ...] = ()
    growth: tuple[tuple[str, int], ...] = ()
    scratch: int = 0
    retention: RetentionPolicy | None = None
    # how many leading spans the label's keys cover, by default every span before the first rows span
    keyed_spans: int | None = None

    def keyed_count(self) -> int:
        if self.keyed_spans is not None:
            return self.keyed_spans
        return next((i for i, span in enumerate(self.spans) if span.kind == "rows"), len(self.spans))

    def row(self) -> list:
        """The layout as ``[spans, growth, scratch, retention, keyed spans]``, a span as
        ``[length, advance, walk, scratch]``: plain values, which keep a request's kwargs on the
        wire's msgpack path."""
        for span in self.spans:
            if span.kind not in ("ids", "digest", "rows"):
                raise ValueError(f"a span of kind {span.kind!r}")
            if not (isinstance(span.length, int) and isinstance(span.advance, int) and isinstance(span.scratch, int)):
                raise ValueError(
                    f"a {span.kind} span's length, advance and scratch are {type(span.length).__name__}, "
                    f"{type(span.advance).__name__} and {type(span.scratch).__name__}"
                )
        retention = self.retention
        return [
            [[span.length, span.advance, span.walk, span.scratch] for span in self.spans],
            [[loop, slots] for loop, slots in self.growth],
            self.scratch,
            None if retention is None else [retention.context_budget, retention.protected_prefix],
            self.keyed_count(),
        ]


def prompt_spans(
    plan: Iterable[PromptPart], texts: Sequence[torch.Tensor], text_walk: str,
    item_spans: Callable[[PromptPart], Iterable[Span]],
) -> list[Span]:
    """A prompt's prefill writes in ``plan`` order: a text step's ids from ``texts``, and an
    attachment's from ``item_spans``."""
    spans = []
    for step in plan:
        if step.modality != TEXT:
            spans.extend(item_spans(step))
            continue
        length = texts[step.index].numel()
        spans.append(Span("ids", text_walk, length, length, "text_inputs"))
    return spans


class PrefixKeys(NamedTuple):
    """How a label's pages are keyed for cross-request prefix reuse.

    ``tensor`` holds the ids an ids span is keyed by. Only a write from one of
    ``walks`` keeps the keys going. With ``growth``, the ids the label's growth
    walk samples extend the keys.
    """
    tensor: str
    walks: tuple[str, ...]
    growth: bool = False


class KVLabel(NamedTuple):
    """A paged KV label, as the model's walks write it.

    Each ``(walk, loop)`` in ``growth`` writes after admission, bounded by the
    named loop. A label with ``keys`` is keyed for prefix reuse.
    """
    growth: tuple[tuple[str, str], ...] = ()
    keys: PrefixKeys | None = None


def keyed_labels(model: "Model") -> dict[str, dict[str, KVLabel]]:
    keyed = {}
    for key, labels in model.kv_labels().items():
        marked = {label: kv_label for label, kv_label in labels.items() if kv_label.keys is not None}
        if marked:
            keyed[key] = marked
    return keyed


def video_metadata_dict(metadata_obj) -> dict:
    # torchcodec>=0.9 reports pixel_aspect_ratio as a Fraction, which the wire codec rejects.
    return {k: float(v) if isinstance(v, Fraction) else v for k, v in asdict(metadata_obj).items()}


_wg_id_counter = itertools.count(1 << 20)


@dataclass
class WorkerGraph:
    section: GraphSection
    graph_walks: set[str]  # e.g., prefill, decode, image_gen
    consumes_stream: bool = field(default=False)

    ranks: list[int] = field(default_factory=list)
    tp_size: int = 1
    sp_size: int = 1

    # ``ranks`` form a 2D [sp_size, tp_size] mesh per instance (instances tile
    # for data parallelism). ``_tp_ranks`` are the mesh rows (tensor-parallel
    # comm groups: sharded heads/weights, all-reduce); ``_sp_ranks`` are the
    # columns (sequence-parallel comm groups: Ulysses all-to-all);
    # ``_instance_ranks`` are the whole instances (the conductor's lockstep
    # routing unit — every rank of an instance runs a request together). With
    # sp_size==1 this reduces to the old ``ranks`` chunked by ``tp_size``
    # (trivial SP groups; instance == TP row). ``tp_size`` is rewritten in
    # __post_init__ to the instance size (tp*sp) so the conductor — which models
    # a parallel group as its lockstep unit — routes to the whole instance; the
    # original per-axis degrees live in ``_tp_comm_size`` and ``sp_size``.
    _tp_ranks: list[list[int]] = field(default_factory=list)
    _sp_ranks: list[list[int]] = field(default_factory=list)
    _instance_ranks: list[list[int]] = field(default_factory=list)
    _tp_comm_size: int = 1
    _group_id: int = field(default=-1)  # original index into config's node_groups

    # Dense index over the deployment's worker graphs, reassigned by the
    # conductor so every process agrees on it. The default only keeps a
    # standalone WorkerGraph unique; it is not stable across processes, which
    # is why the conductor -- the one place that sees every graph -- owns the
    # real numbering.
    worker_graph_id: int = field(default_factory=lambda: next(_wg_id_counter))

    def __post_init__(self):
        if (self.tp_size > 1 or self.sp_size > 1) and not self._tp_ranks:
            tp = self.tp_size  # tensor-parallel (row / all-reduce) degree
            sp = self.sp_size  # sequence-parallel (all-to-all) degree
            inst = tp * sp
            assert len(self.ranks) % inst == 0, (
                f"WorkerGraph {self.worker_graph_id} has {len(self.ranks)} ranks; "
                f"not divisible by tp_size*sp_size ({tp}*{sp}={inst})."
            )
            tp_ranks: list[list[int]] = []
            sp_ranks: list[list[int]] = []
            instance_ranks: list[list[int]] = []
            for base in range(0, len(self.ranks), inst):
                block = self.ranks[base:base + inst]
                instance_ranks.append(block)
                # row-major [sp][tp] grid over this instance's ranks
                grid = [block[s * tp:(s + 1) * tp] for s in range(sp)]
                tp_ranks += grid  # rows: vary tp within fixed sp
                sp_ranks += [
                    [grid[s][t] for s in range(sp)] for t in range(tp)
                ]  # columns: vary sp within fixed tp
            self._tp_ranks = tp_ranks
            self._sp_ranks = sp_ranks
            self._instance_ranks = instance_ranks
            self._tp_comm_size = tp
            # Conductor routes to the lockstep unit (the whole instance).
            self.tp_size = inst


def _combine_sections_sequential_or_parallel(
    section: GraphSection, other: GraphSection, comb_type: Type[Sequential] | Type[Parallel]
):
    if isinstance(section, comb_type) and isinstance(other, comb_type):
        section.sections.extend(other.sections)
        return section
    if isinstance(section, comb_type):
        section.sections.append(other)
        return section
    if isinstance(other, comb_type):
        other.sections.insert(0, section)
        return other
    return comb_type([section, other])


def _divide_into_worker_graphs(
    graph: GraphSection,
    graph_walk: str,
    node_to_group_idx: dict[str, int],
    node_groups: list[dict],
    input_streams: set[str],
) -> list[WorkerGraph]:
    """
    Given a graph, break it into worker graphs
    """
    if isinstance(graph, GraphNode):
        # Mutate via the GraphNode helper so the ReadySignals instances built in
        # __post_init__ keep referencing the live set (they captured by reference).
        graph._register_streaming(input_streams.intersection(graph.input_names))

        group_idx = node_to_group_idx[graph.name]
        ng = node_groups[group_idx]
        return [
            WorkerGraph(
                section=graph,
                graph_walks=set([graph_walk]),
                consumes_stream=graph.consumes_stream,
                _group_id=group_idx,
                ranks=ng["ranks"],
                tp_size=ng.get("tp_size", 1),
                sp_size=ng.get("sp_size", 1),
            )
        ]

    if isinstance(graph, Sequential):
        worker_graphs = _divide_into_worker_graphs(
            graph.sections[0],
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams=input_streams,
        )

        for i in range(1, len(graph.sections)):
            # Go through it sequentially and merge adjacent sections
            # that are on the same device
            new_worker_graphs = _divide_into_worker_graphs(
                graph.sections[i],
                graph_walk=graph_walk,
                node_to_group_idx=node_to_group_idx,
                node_groups=node_groups,
                input_streams=input_streams,
            )
            if (
                new_worker_graphs[0]._group_id == worker_graphs[-1]._group_id
                and not new_worker_graphs[0].consumes_stream
            ):
                worker_graphs[-1].section = _combine_sections_sequential_or_parallel(
                    worker_graphs[-1].section, new_worker_graphs.pop(0).section, comb_type=Sequential
                )
            worker_graphs.extend(new_worker_graphs)
        return worker_graphs

    if isinstance(graph, Parallel):
        all_worker_graphs = [
            _divide_into_worker_graphs(
                s,
                graph_walk=graph_walk,
                node_to_group_idx=node_to_group_idx,
                node_groups=node_groups,
                input_streams=input_streams,
            )
            for s in graph.sections
        ]
        # parallel sections that are all on the same worker can be merged
        singleton_worker_graphs = [s[0] for s in all_worker_graphs if len(s) == 1 and not s[0].consumes_stream]
        group_id_to_worker_graph = {}
        for s in singleton_worker_graphs:
            if s._group_id in group_id_to_worker_graph:
                existing = group_id_to_worker_graph[s._group_id]
                existing.section = _combine_sections_sequential_or_parallel(
                    existing.section, s.section, comb_type=Parallel
                )
            else:
                group_id_to_worker_graph[s._group_id] = s

        return list(group_id_to_worker_graph.values()) + sum(
            [s for s in all_worker_graphs if len(s) > 1 or s[0].consumes_stream], start=[]
        )  # remaining worker graphs

    if isinstance(graph, Loop):
        loop_section_worker_graphs = _divide_into_worker_graphs(
            graph.section,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams=input_streams,
        )
        # A streamed external input is dropped, so the loop takes the NEXT
        # chunk every iteration. TODO: that makes "N steps on ONE chunk" (flow
        # matching over a streamed latent) inexpressible. Honouring it instead
        # means changing this, Loop._is_streamed_input, and the Rust held_mask
        # together, or the two runtimes disagree.
        ext_inps = set([
            (name, dest) for name, dest in graph._external_inputs \
                if name not in input_streams
        ])
        for s in loop_section_worker_graphs:
            # ``accumulated_outputs`` must be propagated to the per-worker
            # reconstruction — otherwise ``Loop.cache_outputs`` sees an
            # empty ``_accumulated_output_names`` on the worker side, every
            # per-iter tensor silently falls on the floor, and nothing ever
            # reaches ``EMIT_TO_CLIENT`` (symptom: 4 "Deferring cleanup"
            # warnings + empty client response in the V-JEPA 2 rollout run).
            s.section = Loop(
                section=s.section,
                max_iters=graph.max_iters,
                curr_iter=graph.curr_iter,
                outputs=graph.outputs,
                name=graph.name,
                accumulated_outputs=graph.accumulated_outputs,
                _external_inputs=ext_inps,
                _loop_back_inputs=graph._loop_back_inputs,
            )
        return loop_section_worker_graphs


def _self_triggered(entity: GraphNode | Loop) -> bool:
    """Whether the stream alone can re-run a top-level entity once it completes:
    it is, or (a Loop) holds, a consumer with every input streamed. Completion
    reseeds streaming readiness only for such nodes; a consumer with a
    non-streamed input has to receive it again first."""
    return any(
        node.consumes_stream and node.input_names <= node._streaming_inputs
        for node in entity.get_nodes().values()
    )


def _on_same_workers(wgs: list[WorkerGraph]) -> bool:
    """Whether every request runs all of these worker graphs on the same
    workers: the conductor picks one replica per node_group, so they share a
    node_group, or each is a single instance on the same ranks."""
    if len({wg._group_id for wg in wgs}) == 1:
        return True
    return all(len(wg.ranks) == wg.tp_size and wg.ranks == wgs[0].ranks for wg in wgs)


@dataclass
class ForwardPassArgs:
    # full_metadata is at the conductor level
    full_metadata: CurrentForwardConductorMetadata
    inputs: list[GraphEdge]

    # de_persist_tensors are tensors that will be used for the final time and
    # not go into future graph nodes
    unpersist_tensors: list[TensorPointerInfo]

    # e.g., saw EOS or max tokens. Is used to end the request
    request_done: bool = False

    # step_metadata is at the engine / worker level; and
    # is passed into the fwd pass
    step_metadata: dict = field(default_factory=dict)

    # initial args only: CurrentForwardPassInfo.stream_lead_items for the
    # partition's consumer streams
    stream_lead_items: dict[str, int] = field(default_factory=dict)


class Model(ABC):
    # Inputs consumed by process_prompt (tokenized or placed on tensor edges).
    # Retain them locally for postprocessing, but do not send the original
    # values to the conductor alongside their tensor descriptors.
    PREPROCESS_ONLY_KWARGS: tuple[str, ...] = ()

    # Keys only this model's process_prompt sets (prompt layouts, stream
    # leads) for the conductor to read back from model_kwargs. The data
    # worker drops a client's values for them before process_prompt.
    INTERNAL_KWARGS: tuple[str, ...] = ()

    # Intra-op threads for the API server's preprocessing of this model's
    # inputs (``load_image``, ``process_prompt``, ...). None leaves torch's
    # default of one per core; ``preprocess_torch_threads`` in the model's
    # config.yaml overrides either. See ``entrypoint._set_preprocess_threads``.
    PREPROCESS_TORCH_THREADS: int | None = None

    def _get_worker_graphs_for_graph_walk(
        self,
        graph_walk: str,
        graph: GraphSection,
        node_groups: list[dict],
    ):
        # Use the *original* (pre-filter) index as the group id so it's stable
        # across graph walks. A node_group that only fires for some walks
        # still keeps the same id everywhere it appears.
        node_to_group_idx: dict[str, int] = {}
        for i, group in enumerate(node_groups):
            if "graph_walks" in group and graph_walk not in group["graph_walks"]:
                continue
            node_to_group_idx.update({name: i for name in group["node_names"]})

        partition = DEFAULT_PARTITION
        for part in self.get_partitions():
            if graph_walk in part.graph_walks:
                partition = part.name
                break
        input_streams = set()
        for conn in self.get_partition_topology().connections:
            if conn.to_partition == partition:
                input_streams.add(conn.edge_name)

        return _divide_into_worker_graphs(
            graph,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams=input_streams,
        )

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Reject a deployment YAML this model cannot serve.

        Called by the Conductor at startup with the parsed YAML. The model
        sees only ``model_kwargs`` in ``__init__``, so checks on other keys
        (``max_concurrent_requests``, ``resources``, ...) belong here. Raise
        ``ValueError`` naming the key; ``config_path`` is for the message.
        """
        return

    def get_worker_graphs(self, config_path: str) -> list[WorkerGraph]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        node_groups = config.get("node_groups")
        if node_groups is None:
            raise KeyError("Config must define `node_groups`.")

        walk_graphs = self.get_graph_walk_graphs()
        self._check_walk_placement(walk_graphs, node_groups, config_path)
        # TODO: merge identical worker graphs from different graph walks
        walk_to_wgs = {
            graph_walk: self._get_worker_graphs_for_graph_walk(graph_walk, graph, node_groups)
            for graph_walk, graph in walk_graphs.items()
        }
        self._check_stream_driven_walks(walk_to_wgs, config_path)
        self._check_loop_successors(walk_to_wgs, config_path)
        return sum(walk_to_wgs.values(), start=[])

    def _check_walk_placement(
        self, walk_graphs: dict[str, GraphSection], node_groups: list[dict], config_path: str,
    ) -> None:
        """Reject declarations the conductor would accept and then serve wrongly:
        a walk in no partition or in two (its worker graphs are registered on no
        worker, or in both partitions), and a node of a walk in no node_group or
        in two (the later group silently wins)."""
        partitions = self.get_partitions()
        for part in partitions:
            unknown = sorted(part.graph_walks - walk_graphs.keys())
            if unknown:
                raise ValueError(
                    f"Partition {part.name!r} lists graph walks {unknown} that "
                    f"get_graph_walk_graphs() does not define (walks: {sorted(walk_graphs)})."
                )
        for walk in walk_graphs:
            owners = [part.name for part in partitions if walk in part.graph_walks]
            if len(owners) != 1:
                raise ValueError(
                    f"Graph walk {walk!r} is in {len(owners)} partitions {owners}; "
                    "every walk must be listed in the graph_walks of exactly one "
                    "PartitionDefinition from get_partitions()."
                )

        for i, group in enumerate(node_groups):
            unknown = sorted(set(group.get("graph_walks", [])) - walk_graphs.keys())
            if unknown:
                raise ValueError(
                    f"{config_path}: node_groups[{i}] (nodes {group['node_names']}) lists "
                    f"graph_walks {unknown}, which this model does not define "
                    f"(walks: {sorted(walk_graphs)})."
                )
        for walk, graph in walk_graphs.items():
            for node in graph.get_nodes():
                groups = [
                    i for i, group in enumerate(node_groups)
                    if node in group["node_names"]
                    and ("graph_walks" not in group or walk in group["graph_walks"])
                ]
                if len(groups) != 1:
                    where = ", ".join(f"node_groups[{i}]" for i in groups) or "no node_groups entry"
                    raise ValueError(
                        f"{config_path}: node {node!r} of graph walk {walk!r} is placed by "
                        f"{where}; place it in exactly one node_groups entry for that walk "
                        "(use graph_walks to give one node different ranks per walk)."
                    )

    def _check_stream_driven_walks(
        self, walk_to_wgs: dict[str, list[WorkerGraph]], config_path: str,
    ) -> None:
        """Reject stream-driven walk shapes the runtime and conductor can't
        account for.

        A worker graph counts entity completions per pass, and a self-triggered
        consumer (every input streamed) is re-armed for the next chunk as soon as
        it completes, so a second top-level entity can be counted done before it
        runs, and the pass-end reset drops input it has buffered. A consumer
        gated on a non-streamed input waits for that input instead, which is
        safe when it arrives once per pass (from the conductor, say).
        Across worker graphs, self-triggered passes overlap with no pass
        identity, and the conductor reads the final-stream flag only off the
        report that closes a pass: the flag is per worker, so it is there only
        if every worker graph of the walk runs on the consumer's workers. A walk
        with no stream-consuming node is conductor driven and unaffected, and a
        TP worker graph is one worker graph on several ranks.
        """
        topology = self.get_partition_topology()
        consumer_partitions = {conn.to_partition for conn in topology.connections}
        walk_partition = {
            walk: part.name for part in self.get_partitions() for walk in part.graph_walks
        }
        for walk, wgs in walk_to_wgs.items():
            if walk_partition[walk] not in consumer_partitions:
                continue
            if not any(wg.consumes_stream for wg in wgs):
                continue
            where = f"{config_path}: graph walk {walk!r} of consumer partition {walk_partition[walk]!r}"
            entities = [top_level_entities(wg.section) for wg in wgs]
            triggered = sorted(
                name for ents in entities for name, e in ents.items() if _self_triggered(e)
            )
            if len(wgs) > 1 and (triggered or not _on_same_workers(wgs)):
                raise ValueError(
                    f"{where} consumes a stream and spans {len(wgs)} worker graphs "
                    f"{[sorted(wg.section.get_nodes()) for wg in wgs]} on ranks "
                    f"{[wg.ranks for wg in wgs]}. Across worker graphs a stream-driven walk "
                    "is served only with every consumer gated on a non-streamed input that "
                    "arrives once per pass and every worker graph on the same workers (one "
                    "node_group, or one instance on the same ranks); otherwise run it as "
                    "one worker graph, or move the work after the consumer into a "
                    "downstream partition fed by a StreamingGraphEdge."
                )
            if len(wgs) == 1 and len(entities[0]) > 1 and triggered:
                raise ValueError(
                    f"{where} has self-triggered consumers {triggered} (every input "
                    f"streamed) among {len(entities[0])} top-level entities {sorted(entities[0])} "
                    "in one worker graph. Gate the consumer on a non-streamed input that "
                    "arrives once per pass (from the conductor, say), or move the work "
                    "after it into a downstream partition fed by a StreamingGraphEdge."
                )

    def _check_loop_successors(
        self, walk_to_wgs: dict[str, list[WorkerGraph]], config_path: str,
    ) -> None:
        """Reject a Loop member with an edge to a node outside the Loop in the
        same worker graph.

        A worker graph counts top-level completions, and that successor
        completes once per iteration, so the pass ends before the Loop and the
        successor's later inputs are done, and the pass-end reset drops them.
        ``Loop.accumulated_outputs`` hands the successor every iteration's
        output once, when the Loop finishes.
        """
        for walk, wgs in walk_to_wgs.items():
            for wg in wgs:
                wg_nodes = wg.section.get_nodes()
                for loop in wg.section.get_loops().values():
                    members = loop.get_nodes()
                    for name, node in members.items():
                        for edge in node.outputs:
                            if edge.next_node in wg_nodes and edge.next_node not in members:
                                raise ValueError(
                                    f"{config_path}: graph walk {walk!r}: node {name!r} of Loop "
                                    f"{loop.name!r} sends {edge.name!r} to {edge.next_node!r}, "
                                    "outside the Loop in the same worker graph. The worker graph "
                                    "would report done before the Loop and its successor finish. "
                                    "Send it through Loop.accumulated_outputs (every iteration's "
                                    "output, once the Loop finishes) or Loop.outputs instead."
                                )

    def get_sharding_config(self, config_path: str) -> ShardingConfig:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        sharding_config = self.get_default_sharding_config()

        # Derive sharding groups from node_groups with tp_size > 1. The
        # node_groups section drives both worker assignment and TP grouping;
        # the separate "sharding_config" section just holds shard_dim.
        groups: list[ShardingGroup] = []
        for ng in config.get("node_groups", []):
            tp = ng.get("tp_size", 1)
            sp = ng.get("sp_size", 1)
            instance_size = tp * sp
            if instance_size <= 1:
                continue
            if tp > 1 and any(
                node not in sharding_config.tp_enabled_nodes
                for node in ng["node_names"]
            ):
                raise ValueError(
                    f"Node group with tp_size {tp} contains "
                    f"nodes not in tp_enabled_nodes {sharding_config.tp_enabled_nodes}: "
                    f"{ng['node_names']}"
                )
            if sp > 1 and any(
                node not in sharding_config.sp_enabled_nodes
                for node in ng["node_names"]
            ):
                raise ValueError(
                    f"Node group with sp_size {sp} contains "
                    f"nodes not in sp_enabled_nodes {sharding_config.sp_enabled_nodes}: "
                    f"{ng['node_names']}"
                )
            # The sharding group is the lockstep unit — tp*sp workers routed
            # together. TP shards weights via shard_dim-driven fanout; SP is
            # in-module (replicated signals, no shard_dim) but its ranks must
            # still run each request together, so the group spans the whole
            # instance.
            groups.append(ShardingGroup(
                nodes=set(ng["node_names"]),
                tp_size=instance_size,
                graph_walks=set(ng["graph_walks"]) if "graph_walks" in ng else None,
            ))

        sharding_config.groups = groups

        sharding_config_yaml = config.get("sharding_config")
        if sharding_config_yaml is not None and "shard_dim" in sharding_config_yaml:
            sharding_config.shard_dim.update(sharding_config_yaml["shard_dim"])
        return sharding_config

    def get_default_sharding_config(self) -> ShardingConfig:
        return ShardingConfig(
            groups=[], tp_enabled_nodes=[], shard_dim={}
        ) # default: no sharding

    @property
    def nodes(self) -> list[str]:
        return sorted({
            node
            for graph in self.get_graph_walk_graphs().values()
            for node in graph.get_nodes()
        })

    @abstractmethod
    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Declare the resources the engine builds, and which nodes share each.

        The whole declaration: a KV cache's shape, the attention backend over
        it, positions, samplers. The engine builds one resource per spec and
        hands each node the subset naming it.
        """
        pass

    def checkpoint_path(self) -> str | None:
        """Where this model's weights and config sit on disk.

        The prefix cache hashes the checkpoint's manifest into its root. None leaves
        the weights out, so the cache stays shut unless the deployment sets a salt.
        """
        return None

    def preprocess_fingerprint(self) -> str:
        """What this model's preprocessing contributes to the cache's root.

        Two deployments that tokenize or template a prompt differently must not
        match each other's pages; the class name separates them, and a model
        whose preprocessing changes shape between versions overrides this.
        """
        return type(self).__name__

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        """``{node: {combined_walk: {constituent walks}}}``: walks the scheduler may batch together."""
        return {}

    def combined_walk_of(self) -> dict[tuple[str, str], str]:
        """``(node, real walk) -> combined walk``, validated."""
        walks = self.get_graph_walk_graphs()
        out: dict[tuple[str, str], str] = {}
        for node, combined in self.get_combined_graph_walks().items():
            for name, members in combined.items():
                if name in walks:
                    raise ValueError(
                        f"{type(self).__name__}: combined walk {name!r} of {node!r} "
                        "reuses the name of a real walk"
                    )
                for walk in members:
                    if walk not in walks or node not in walks[walk].get_nodes():
                        raise ValueError(
                            f"{type(self).__name__}: combined walk {name!r} names "
                            f"{walk!r}, which is not a walk of node {node!r}"
                        )
                    if (node, walk) in out:
                        raise ValueError(
                            f"{type(self).__name__}: walk {walk!r} of {node!r} is in "
                            f"both {out[(node, walk)]!r} and {name!r}"
                        )
                    out[(node, walk)] = name
        return out

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        """The paged KV labels the model's walks write, by resource and label.

        `process_prompt` returns a `LabelLayout` for each under ``kv_layout``, as
        ``{resource: {label: LabelLayout}}``. A label with keys opts its node into
        cross-request prefix reuse: the preprocess worker keys its layout's keyed
        spans and the KV resource matches what it already holds. With no keyed
        label the model is uncached whatever the deployment asks for.
        """
        return {}

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        """Per-resource config a new request is opened with, by resource label.

        Where a request's knobs live now: sampling params, whether it needs
        CFG, retention. The conductor resolves this once per request and the
        engine hands each config to its resource at ingest.
        """
        del model_kwargs, partition_fwd_args
        return {}

    def get_sampling_config(
        self, node_name: str,
        model_kwargs: dict | None = None,
    )  -> SamplingConfig | None:
        """This node's sampling params. Feed it to the sampler resource's
        config in ``get_request_resource_configs``.

        A node that samples under several configs (a talker and its code
        predictor, say) declares one sampler resource per label rather than
        nesting them here.
        """
        return SamplingConfig(
            ignore_eos=(model_kwargs or {}).get("ignore_eos", False)
        )

    @abstractmethod
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        pass

    @abstractmethod
    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        pass

    @abstractmethod
    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> "NameToTensorList | ProcessPromptOutput":
        """Tokenize prompt and produce initial tensors for the request.

        Called by the API server data worker AFTER it has loaded raw
        multimodal tensors from file_paths (images, audio, video).
        The model may inspect the raw tensors dict and compute additional
        derived tensors (e.g., Qwen3-Omni computes ``pixel_values``,
        ``image_grid_thw``, ``audio_features``, ``audio_seqlens`` from the
        raw ``image_inputs`` / ``audio_inputs`` / ``video_inputs``).

        Args:
            prompt: Raw text input from the user, or None if no text.
            input_modalities: List of input modality types for this request.
            output_modalities: List of desired output modality types.
            tensors: Raw modality tensors already loaded by the data worker
                (``image_inputs``, ``audio_inputs``, ``video_inputs``).
                The model may read these to compute derived tensors.
                Models that don't need the raw tensors may ignore this.
            prompt_parts: Ordered text/attachment sequence as written.
                ``None`` from entrypoints with no ordering to preserve; see
                :func:`mstar.model.multimodal.parts_from_modalities`.
            **kwargs: Model-specific parameters (e.g., from model_kwargs).

        Returns:
            NameToTensorList with tensors to MERGE into the request's
            tensor dict.  Typically includes ``text_inputs`` plus any
            model-specific derived tensors.  The returned dict is merged
            into the existing ``tensors`` dict via ``dict.update``.
        """
        pass

    def max_input_files(self) -> dict[str, int]:
        """Most files of each input modality one request may attach, e.g.
        ``{"audio": 1}``. The data worker refuses a request over it before
        decoding any file; a modality left out is held only to
        ``MSTAR_MAX_MEDIA_FILES``."""
        return {}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        # RGB whatever the file holds: image processors cannot place the
        # channel axis of a 4-channel (RGBA or CMYK) image.
        try:
            img = read_image(filepath, rgb=True)
        except (RuntimeError, OSError) as exc:  # torchvision's decoders raise RuntimeError, Pillow's OSError
            raise ValueError("could not decode image: the file is corrupt or truncated") from exc
        img = img.to(device).float().div_(255.0)  # CxHxW

        return TensorAndMetadata(img)

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        # libsndfile first, so WAV, FLAC, OGG and MP3 do not depend on the
        # FFmpeg build torchcodec happens to load (some base images ship a
        # video-only one); torchcodec for the other containers.
        from mstar.model.components.audio_features import load_audio_file

        audio = load_audio_file(filepath, sample_rate=16000)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=16000, num_channels=1))

    def load_video(self, filepath: str, device: str):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(filepath, device=device)
        budget = VideoBudget()
        frames = []
        for frame in decoder:
            budget.add(*frame.shape[-2:])
            frames.append(frame)
        video = torch.stack(frames)
        del frames
        video = video.float().div_(255.0)
        return TensorAndMetadata(data=video, metadata=video_metadata_dict(decoder.metadata))

    @abstractmethod
    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,  # text | image | video | video_frame | audio
        request_kwargs: dict | None = None,
    ) -> bytes:
        """
        Given an output of a certain modality, encode and return as bytes.
        This will likely need to overridden with model-specific behavior.
        ``request_kwargs`` is the request's model_kwargs, so encoders can honor
        per-request parameters (e.g. writing a video container at the request's
        frame rate).

        Modality to expected encoding type:
        - text: utf-8
        - image: png
        """
        return output.cpu().numpy().tobytes()

    @abstractmethod
    def get_submodule(
        self, node_name: str, device="cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        """Return the nn.Module for this node, or None for dummy mode.

        ``tp_group`` is the :class:`CommGroup` for this node on the
        calling worker (``None`` when TP is not active for this node).
        Models that support tensor parallelism should forward it to
        parallel-linear constructors so weight shapes are partitioned at
        init time and the weight loaders shard correctly.

        ``sp_group`` is the node's sequence-parallel comm group, orthogonal
        to ``tp_group``. The engine manager only passes it when the node
        participates in a non-trivial SP group, so models without SP support
        never receive it. Models that support SP forward it to their
        attention so it can all-to-all the sequence around the kernel.

        ``autocast_dtype`` is the dtype this node's params should be
        allocated in. Models that build on the meta device must cast the
        meta module to this dtype *before* ``to_empty(device)`` so storage
        is allocated directly in the target dtype — casting after
        ``to_empty`` first materialises every param in the meta default
        (float32), doubling the load-time VRAM peak. ``None`` means leave
        the default (float32); the engine manager passes ``None`` for
        nodes whose engine does not autocast.
        """
        pass

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", MAX_OUTPUT_TOKENS)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        """The iterations each named loop may run for this request, each at least 1.

        A loop runs the smaller of its graph ``max_iters`` and its cap. A model
        whose stop check ends a loop at a count sets the cap to that count, so the
        loop ends on the same iteration and never speculates past it.
        """
        return {}

    def request_kwargs(self) -> frozenset[str] | None:
        """Every ``model_kwargs`` key this model reads, including the ones an
        OpenAI adapter maps onto it. A request carrying any other key is served,
        and the key named in the ``X-MStar-Ignored-Params`` response header.
        None (the default) declares nothing, so nothing is reported.
        """
        return None

    def get_autocast_dtype(self):
        return torch.bfloat16

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        """Sample rate (Hz) of audio output produced by ``postprocess``.

        Audio is emitted as headerless 16-bit PCM, so the client surfaces
        (SDK / OpenAI-compatible endpoints) need this rate to wrap it into a
        WAV/other container. Override per model; the 24 kHz default matches the
        speech codecs used by the audio models here.
        """
        return 24000

    def get_output_audio_channels(self, modality: str = "audio") -> int:
        """Channel count of the interleaved 16-bit PCM ``postprocess`` emits for
        audio. Mono default (the speech models); stereo models override."""
        return 1

    def warmup_preprocess(self) -> None:
        """Called once in the preprocess worker before it takes requests.

        Build what ``process_prompt`` needs lazily (a G2P front end, a
        tokenizer, a feature extractor) so the first request does not pay for
        it. The default does nothing.
        """
        return None

    def get_voices(self) -> list[str] | None:
        """The speaker ids a speech model accepts as ``voice``, for
        ``GET /v1/audio/voices``. ``None`` (the default) means the model has no
        fixed voice list, and the route answers 404."""
        return None

    def get_default_voice(self) -> str | None:
        """The ``voice`` used when a speech request names none."""
        return None

    def get_output_frame_rate(
        self,
        modality: str = "video_frame",
        request_kwargs: dict | None = None,
    ) -> float:
        """Frame rate for raw RGB ``video_frame`` output.

        Raw frames have no container header from which a client could recover
        timing, so models that expose this modality must override this hook.
        ``request_kwargs`` permits a future model with a per-request frame rate;
        Waypoint's checkpoint uses a fixed rate.
        """
        del request_kwargs
        raise ValueError(
            f"{type(self).__name__} does not define a frame rate for {modality!r} output"
        )

    # ------------------------------------------------------------------
    # Partition API (optional, backward-compatible defaults)
    # ------------------------------------------------------------------

    def get_partition_topology(self):
        """Return a PartitionTopology describing async partitions and streaming connections.

        Default: single "default" partition with no connections.
        Override for models with async partitions (e.g., Orpheus LLM + SNAC).
        """
        from mstar.streaming.topology import PartitionTopology

        return PartitionTopology(partitions=[DEFAULT_PARTITION], connections=[])

    def get_partitions(self) -> list[PartitionDefinition]:
        """Return partition definitions.

        Default: single "default" partition containing all graph walks.
        Override for models with async partitions (e.g., Orpheus LLM + SNAC).
        """
        walks = set(self.get_graph_walk_graphs().keys())
        return [
            PartitionDefinition(
                name=DEFAULT_PARTITION,
                graph_walks=walks,
                initial_walk=None,
                producer_partitions=[],
            )
        ]

    @abstractmethod
    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> "ForwardPassArgs":
        """Return the next forward pass arguments for a specific partition.

        Called by the conductor after each completed forward pass to determine
        the next graph walk, inputs, and whether the request is done.

        ``incoming_connections`` contains streaming-specific state (token counts,
        producer_done) for consumer partitions. For single-partition models,
        this will be ``None`` or an empty list.
        """
        pass
