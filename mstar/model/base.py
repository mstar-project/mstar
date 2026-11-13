import enum
import functools
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, NamedTuple, Type

import torch
import yaml

from mstar.api_server import media_io
from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    DEFAULT_PARTITION,
    PartitionDefinition,
    WalkStep,
)
from mstar.distributed.base import (
    PARALLEL_AXES,
    ShardingConfig,
    ShardingGroup,
    mesh_groups,
    node_group_axis_sizes,
)
from mstar.engine.resources import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.kv.config import RetentionPolicy, Span, SpanKind
from mstar.engine.resources.lora.config import LORA_ADAPTER_KWARG, LORA_DIGEST_KWARG
from mstar.engine.resources.sampler.utils import SamplingConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.model.media_limits import VideoBudget, decoding_upload, read_image, video_decode_threads
from mstar.model.multimodal import TEXT, PromptPart
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import (
    CHUNK_LOOP_END,
    ChunkingPolicy,
    chunk_loop_name,
    chunk_tick_name,
)
from mstar.model.utils import TextChunkDetokenizer, TextOutput, drop_stop_tokens


def _wrap_node_in_loop(
    section: GraphSection, node_name: str, graph_walk: str,
    policy: "ChunkingPolicy",
) -> GraphSection:
    """``section`` with ``node_name`` wrapped in the Loop its chunks run as.

    The node's real edges move onto the Loop, which is what makes a chunk's
    output the loop's business rather than the worker's: an edge the policy
    gathers gets every chunk's tensors at the end, and any other gets the
    final chunk's alone, earlier ones dropped as the loop advances. A
    ``per_chunk_outputs`` edge is the exception: it stays on the body, so
    every chunk routes it as it completes.

    The body keeps a bare same-named edge to ``EMPTY_DESTINATION``. The Loop
    caches by edge NAME, so the body has to emit the name for the Loop to have
    anything to carry; the copy routes nowhere and drops every flag, since
    persisting or counting a per-chunk tick would stage it and bill the
    conductor for a token per chunk.

    The Loop counts tokens rather than iterations: each chunk moves it by its
    ``ChunkRange.stride``, and it ends at the prompt's length, which the
    scheduler sets as its cap on the first chunk.

    The body also routes ``chunk_tick_name`` back to itself, which is what
    makes the next chunk a speculation target; see that function for why the
    conductor seeds the first one. The submodule produces no tensor for it, and
    a declared signal with no tensor still satisfies readiness, so it is free.
    """
    if isinstance(section, GraphNode):
        if section.name != node_name:
            return section
        ticks, final, gathered = [], [], []
        for edge in section.outputs:
            if edge.name in policy.per_chunk_outputs:
                ticks.append(edge)
                continue
            ticks.append(GraphEdge(next_node=EMPTY_DESTINATION, name=edge.name))
            dest = gathered if edge.name in policy.accumulated_outputs else final
            dest.append(edge)
        # a misspelt name would quietly carry only the final chunk
        unknown = policy.accumulated_outputs - {edge.name for edge in section.outputs}
        if unknown:
            raise ValueError(
                f"node {node_name!r} accumulates {sorted(unknown)} in walk "
                f"{graph_walk!r}, which are not among its outputs there"
            )
        tick = chunk_tick_name(node_name, graph_walk)
        ticks.append(GraphEdge(next_node=node_name, name=tick))
        return Loop(
            name=chunk_loop_name(node_name, graph_walk),
            section=replace(
                section, outputs=ticks,
                input_names=set(section.input_names) | {tick},
            ),
            max_iters=CHUNK_LOOP_END,
            outputs=final,
            accumulated_outputs=gathered,
        )
    if isinstance(section, Sequential):
        return Sequential([
            _wrap_node_in_loop(s, node_name, graph_walk, policy) for s in section.sections
        ])
    if isinstance(section, Parallel):
        return Parallel([
            _wrap_node_in_loop(s, node_name, graph_walk, policy) for s in section.sections
        ])
    if isinstance(section, Loop):
        # A chunkable node inside an existing Loop nests one loop in another,
        # whose completion cascades into the parent's (rust/src/graph/state.rs).
        # Chunking does not handle that, so refuse it.
        if node_name in section.section.get_nodes():
            raise ValueError(
                f"node {node_name!r} is chunkable and already inside Loop "
                f"{section.name!r}; nested chunk loops are not supported"
            )
        return section
    raise TypeError(f"cannot wrap a chunkable node in {type(section).__name__}")


def _applying_chunking(build):
    """A model's own ``get_graph_walk_graphs``, its result passed through
    ``Model.apply_chunking``."""
    @functools.wraps(build)
    def get_graph_walk_graphs(self):
        return self.apply_chunking(build(self))
    get_graph_walk_graphs._applies_chunking = True
    return get_graph_walk_graphs

if TYPE_CHECKING:
    from mstar.model.components.moe_parallel import MoeParallelConfig

if TYPE_CHECKING:
    from mstar.engine.state import Array, Sampler, SampleSettings, SeqShard
    from mstar.engine.state.adapters import StateSetup
    from mstar.worker.scheduling_policy import WalkPriority

DECODE = "decode"
MAX_OUTPUT_TOKENS = 2048

@dataclass
class TensorAndMetadata:
    data: torch.Tensor
    metadata: dict = field(default_factory=dict)


class RobotIO(NamedTuple):
    """What a robot policy's requests and outputs look like, for the API
    server's robot protocols (the openpi WebSocket)."""
    # the model_kwargs key a request's robot state goes under
    state_kwarg: str
    # the shape of the float32 actions ``postprocess`` returns for "action"
    action_horizon: int
    action_dim: int


class KeepAdapter(enum.Enum):
    """``ProcessPromptOutput.lora_adapter`` left unset."""
    KEEP = "keep"


# the request runs on the adapter its client named, or on the base model
KEEP_ADAPTER = KeepAdapter.KEEP


class ProcessPromptOutput(NamedTuple):
    """Tensors from `process_prompt`, and metadata to carry beside them.

    The data worker folds ``metadata`` into the request's ``model_kwargs``, so
    what a model settles while tokenizing reaches
    `get_request_resource_configs` without a field of its own on the request.

    ``lora_adapter`` is the LoRA adapter the model runs this prompt on, one of
    ``Model.prompt_lora_adapters``, or None for the base model; the data worker
    gives the request what the API server gives a client that names it
    (``ServedModels.model_chose``). Unset keeps the client's choice.
    """
    new_input_tensors: NameToTensorList
    metadata: dict
    lora_adapter: str | None | KeepAdapter = KEEP_ADAPTER


class ClientChunk(NamedTuple):
    """One append the client sends to a running request, as
    ``Model.process_stream_chunk`` sees it."""
    text: str | None
    # the append's files, loaded as a request's are (``audio_inputs`` and so on)
    tensors: NameToTensorList
    input_metadata: dict[str, list[dict]]
    # per ``tensors`` list, the client stream each file names, None where it names none
    streams: dict[str, list[str | None]]
    # this append's model_kwargs, and the opening message's
    model_kwargs: dict
    request_kwargs: dict


class LabelLayout(NamedTuple):
    """Everything a request writes into one paged KV label, from `process_prompt`'s ``kv_layout``.

    ``spans`` are the committed writes known at intake, in walk order. Each
    ``(loop, slots)`` in ``growth`` writes ``slots`` per iteration of the named
    loop, and ``max_growth``, when set, is the most slots ``growth`` writes in all,
    for loops that end before their caps. ``scratch`` is the most one
    non-committing step writes past the committed length, and ``retention`` the
    policy the label's committing steps declare.
    """
    spans: tuple[Span, ...] = ()
    growth: tuple[tuple[str, int], ...] = ()
    scratch: int = 0
    retention: RetentionPolicy | None = None
    # how many leading spans the label's keys cover, by default every span before the first rows span
    keyed_spans: int | None = None
    max_growth: int | None = None

    def keyed_count(self) -> int:
        if self.keyed_spans is not None:
            return self.keyed_spans
        return next((i for i, span in enumerate(self.spans) if span.kind == SpanKind.ROWS), len(self.spans))

    def row(self) -> list:
        """The layout as ``[spans, growth, scratch, retention, keyed spans, max growth]``, a span as
        ``[length, advance, walk, scratch, digest, end key, opens step, kind]``: plain values, which
        keep a request's kwargs on the wire's msgpack path. Digest and end key are None until the
        preprocess worker keys the span."""
        for i, span in enumerate(self.spans):
            if span.kind not in tuple(SpanKind):
                raise ValueError(f"a span of kind {span.kind!r}")
            if not span.opens_step and (i == 0 or self.spans[i - 1].walk != span.walk):
                raise ValueError(f"span {i} of walk {span.walk!r} opens no step, but " + (
                    f"span {i - 1} before it is of walk {self.spans[i - 1].walk!r}: a step runs one walk"
                    if i else "no span comes before it"
                ))
            if not (isinstance(span.length, int) and isinstance(span.advance, int) and isinstance(span.scratch, int)):
                raise ValueError(
                    f"a {span.kind} span's length, advance and scratch are {type(span.length).__name__}, "
                    f"{type(span.advance).__name__} and {type(span.scratch).__name__}"
                )
        retention = self.retention
        return [
            [
                [span.length, span.advance, span.walk, span.scratch, None, None, span.opens_step,
                 SpanKind(span.kind).value]
                for span in self.spans
            ],
            [[loop, slots] for loop, slots in self.growth],
            self.scratch,
            None if retention is None else [retention.context_budget, retention.protected_prefix, retention.exact],
            self.keyed_count(),
            self.max_growth,
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
        spans.append(Span(SpanKind.IDS, text_walk, length, length, "text_inputs"))
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


def names_checkpoint(model: "Model") -> bool:
    """Whether ``model`` overrides ``checkpoint_path``, read without calling it,
    which may download the checkpoint (BAGEL's does)."""
    return type(model).checkpoint_path is not Model.checkpoint_path


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
    cfg_size: int = 1

    # ``ranks`` tile into lockstep instances (instances tile for data
    # parallelism), each a row-major mesh over ``PARALLEL_AXES``.
    # ``_axis_ranks`` holds each axis's comm groups (tp: sharded heads and
    # weights, all-reduce; sp: Ulysses all-to-all; cfg: one share of the
    # guidance branches each); ``_instance_ranks`` are the whole instances
    # (every rank of an instance runs a request together; empty without a
    # parallel axis, so the data-parallel replicas the conductor routes to are
    # ``replica_ranks``).
    _axis_ranks: dict[str, list[list[int]]] = field(default_factory=dict)
    _instance_ranks: list[list[int]] = field(default_factory=list)
    _group_id: int = field(default=-1)  # original index into config's node_groups

    # Dense index over the deployment's worker graphs, reassigned by the
    # conductor so every process agrees on it. The default only keeps a
    # standalone WorkerGraph unique; it is not stable across processes, which
    # is why the conductor -- the one place that sees every graph -- owns the
    # real numbering.
    worker_graph_id: int = field(default_factory=lambda: next(_wg_id_counter))

    def __post_init__(self):
        inst = self.instance_size
        if inst > 1:
            assert len(self.ranks) % inst == 0, (
                f"WorkerGraph {self.worker_graph_id} has {len(self.ranks)} ranks; "
                f"not divisible by its instance size ({self.axis_sizes}: {inst})."
            )
            self._instance_ranks = [self.ranks[base:base + inst] for base in range(0, len(self.ranks), inst)]
            self._axis_ranks = {axis: [] for axis in PARALLEL_AXES}
            for block in self._instance_ranks:
                for axis, groups in mesh_groups(block, self.axis_sizes).items():
                    self._axis_ranks[axis] += groups

    @property
    def axis_sizes(self) -> dict[str, int]:
        return {axis: getattr(self, f"{axis}_size") for axis in PARALLEL_AXES}

    @property
    def instance_size(self) -> int:
        """Ranks in one lockstep instance: the conductor's routing unit."""
        return math.prod(self.axis_sizes.values())

    @property
    def replica_ranks(self) -> list[list[int]]:
        """The ranks of each data-parallel replica: the lockstep instances, or
        one rank apiece without a parallel axis. The conductor routes a request to one."""
        return self._instance_ranks or [[r] for r in self.ranks]


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
                **{f"{axis}_size": size for axis, size in node_group_axis_sizes(ng).items()},
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
        # together.
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


def node_moe_parallel_configs(node_groups: list[dict]) -> dict[str, "MoeParallelConfig"]:
    """Each node's parsed ``ep_config``, from the node groups that list it.

    Nodes whose config is the default (TP) are left out. A node listed by
    several groups, one per graph walk, must get the same config from each.
    """
    from mstar.model.components.moe_parallel import MoeParallelConfig

    configs: dict[str, MoeParallelConfig] = {}
    for ng in node_groups:
        cfg = MoeParallelConfig.from_yaml(ng.get("ep_config"))
        for node in ng["node_names"]:
            if configs.setdefault(node, cfg) != cfg:
                raise ValueError(f"node {node!r} gets different ep_config from two node_groups")
    return {node: cfg for node, cfg in configs.items() if cfg != MoeParallelConfig()}


def streamed_inputs(model: "Model", graph_walk: str, graph: GraphSection) -> set[tuple[str, str]]:
    """The ``(name, node)`` inputs of ``graph_walk`` that a stream feeds: an
    edge of a connection into the walk's partition (the client's included),
    at each node of the walk that takes it. The worker graphs and the
    conductor's walk plans both read this, so they agree on what is streamed."""
    partition = next(
        (part.name for part in model.get_partitions() if graph_walk in part.graph_walks), DEFAULT_PARTITION,
    )
    names = {conn.edge_name for conn in model.get_partition_topology().connections if conn.to_partition == partition}
    return {
        (name, node.name) for node in graph.get_nodes().values() for name in names & set(node.input_names)
    }


def enabled_resources(model: "Model", model_config: dict, setup: "StateSetup") -> list[NodeResourceSpec]:
    """The resource specs a deployment's YAML turns on, before its ``resources:``
    overrides: the model's declaration, less the opt-in specs (``NodeResourceSpec.opt_in``)
    its ``resources:`` block does not name. Raises the first problem with the declaration,
    for callers that skip ``validate_deployment``, which reports them all."""
    from mstar.engine.state.adapters import node_resources
    from mstar.engine.state.analysis import state_declaration_problems

    problems = state_declaration_problems(model)
    if problems:
        raise ValueError(problems[0])
    named = (model_config.get("resources") or {}).keys()
    return [
        spec for spec in node_resources(model, setup=setup)
        if not spec.opt_in or spec.resource_key in named
    ]


def declared_resources(
    model: "Model", model_config: dict, setup: "StateSetup | None" = None,
) -> list[NodeResourceSpec]:
    """The resource specs a deployment declares: ``enabled_resources`` with the YAML's
    overrides applied. A ``per_node`` spec is whole here, as the API server checks
    requests and adapters against it; ``serving_resources`` splits it."""
    from mstar.engine.resources.spec import apply_yaml_overrides
    from mstar.engine.state.adapters import spec_overrides

    specs = enabled_resources(model, model_config, setup or deployment_setup(model, model_config))
    apply_yaml_overrides(specs, spec_overrides(model_config))
    return specs


def built_resources(specs: list[NodeResourceSpec]) -> list[NodeResourceSpec]:
    """The specs a deployment builds of its declared ones: each ``per_node`` spec as its
    instances (``NodeResourceSpec.for_node``)."""
    built: list[NodeResourceSpec] = []
    for spec in specs:
        if not spec.per_node:
            built.append(spec)
            continue
        instances = (spec.for_node(node) for node in sorted(spec.nodes))
        built.extend(instance for instance in instances if instance is not None)
    return built


def parse_autocast_dtype(value) -> torch.dtype | None:
    """The serving YAML's ``autocast_dtype``, spelled as a string (``float16``). ``null``
    stays None, as the serving config passed it before dtypes were parsed."""
    if value is None or isinstance(value, torch.dtype):
        return value
    dtype = getattr(torch, str(value).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"autocast_dtype {value!r} is not a torch dtype (e.g. bfloat16, float16)")
    return dtype


def deployment_setup(model: "Model", model_config: dict) -> "StateSetup":
    """The `StateSetup` ``model_config`` asks for, raising its first problem, for callers that skip
    ``validate_deployment``, which reports them all."""
    from mstar.engine.state.adapters import state_setup

    setup, problems = state_setup(model_config, model.declare_state())
    if problems:
        raise ValueError(problems[0])
    return setup


def serving_resources(
    model: "Model", model_config: dict, setup: "StateSetup | None" = None,
) -> list[NodeResourceSpec]:
    """The resource specs a deployment builds: ``declared_resources``, as
    ``built_resources``.

    Every process that reads specs resolves them here, so the conductor validates
    requests against what the workers built, under the same keys.
    ``validate_deployment`` takes the same steps, reporting every problem rather than
    the first.
    """
    return built_resources(declared_resources(model, model_config, setup))


class Model(ABC):
    def __init_subclass__(cls, **kwargs):
        # A chunkable node runs inside its chunk Loop wherever the walk graphs
        # are read, so every model's declaration is wrapped here rather than
        # each model having to call `apply_chunking` itself.
        super().__init_subclass__(**kwargs)
        build = cls.__dict__.get("get_graph_walk_graphs")
        if build is not None and not getattr(build, "_applies_chunking", False):
            cls.get_graph_walk_graphs = _applying_chunking(build)

    # Inputs consumed by process_prompt (tokenized or placed on tensor edges).
    # Retain them locally for postprocessing, but do not send the original
    # values to the conductor alongside their tensor descriptors.
    PREPROCESS_ONLY_KWARGS: tuple[str, ...] = ()

    # Keys only this model's process_prompt sets (prompt layouts, stream
    # leads) for the conductor to read back from model_kwargs. The data
    # worker drops a client's values for them before process_prompt.
    INTERNAL_KWARGS: tuple[str, ...] = ()

    # Request keys that set an output length. A request that sets one is refused when it
    # doesn't fit the context, instead of having its length cut to fit.
    LENGTH_KWARGS: tuple[str, ...] = ("max_output_tokens",)
    # Request keys the conductor or data worker reads for every model, never logged as ignored.
    ENGINE_KWARGS: tuple[str, ...] = (
        "seed", "prefix_cache", "return_token_ids", "return_logprobs", LORA_ADAPTER_KWARG, LORA_DIGEST_KWARG,
    )
    # Output modalities a client plays as they stream, where a pause is a gap. A streamed request with one of
    # them reserves its whole KV footprint and is never preempted. ``unpausable_outputs`` in the serving config
    # overrides it.
    UNPAUSABLE_OUTPUTS: frozenset[str] = frozenset({"audio"})

    # Intra-op threads for the API server's preprocessing of this model's
    # inputs (``load_image``, ``process_prompt``, ...). None takes a default
    # sized from the CPUs the process may use; ``preprocess_torch_threads`` in
    # the model's config.yaml overrides either. See
    # ``entrypoint._set_preprocess_threads``.
    PREPROCESS_TORCH_THREADS: int | None = None
    # ``load_image`` / ``load_audio`` / ``load_video`` and ``process_prompt``
    # may run for several requests at once. The data worker then prepares
    # requests on a pool (``CpuPlan.prepare_workers`` at this process's intra-op
    # thread count; ``preprocess_workers`` in the model's config.yaml overrides
    # it) and still hands them to the
    # conductor in arrival order. Declare it only once those calls are checked
    # for shared state, and for the GIL: work that holds it gains nothing here.
    PREPROCESS_PARALLEL: bool = False
    # Also decode one request's images at once, for a ``load_image`` that is mostly
    # a GIL-free decode: intra-op work in it would run on each of these threads.
    # It needs PREPROCESS_PARALLEL, since the decode pool is sized like the prepare pool.
    PREPROCESS_PARALLEL_IMAGES: bool = False

    # The rate the default ``load_audio`` decodes an upload to, and the longest
    # clip it decodes (default 30 min). The request's audio budget
    # (``media_limits``) holds on top of that.
    AUDIO_INPUT_RATE: int = 16_000
    MAX_AUDIO_SECONDS: float = media_io.MAX_AUDIO_SECONDS

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

        return _divide_into_worker_graphs(
            graph,
            graph_walk=graph_walk,
            node_to_group_idx=node_to_group_idx,
            node_groups=node_groups,
            input_streams={name for name, _ in streamed_inputs(self, graph_walk, graph)},
        )

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Reject a deployment YAML this model cannot serve.

        Called by ``validate_deployment`` in the conductor at startup, with the
        parsed YAML. The model sees only ``model_kwargs`` in ``__init__``, so
        checks on other keys (``max_concurrent_requests``, ``resources``, ...)
        belong here. Raise ``ValueError`` naming the key; ``config_path`` is
        for the message.
        """
        return

    def get_worker_graphs(self, config_path: str) -> list[WorkerGraph]:
        """Divide every walk into worker graphs by ``node_groups``, first
        refusing a placement that cannot be divided, with every problem in one
        ``DeploymentError``. ``validate_deployment`` checks the result."""
        from mstar.validation import DeploymentError
        from mstar.validation.placement import check_placement

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        problems = check_placement(self, config)
        if problems:
            raise DeploymentError(config_path, problems)

        node_groups = config["node_groups"]
        # TODO: merge identical worker graphs from different graph walks
        return [
            wg
            for graph_walk, graph in self.get_graph_walk_graphs().items()
            for wg in self._get_worker_graphs_for_graph_walk(graph_walk, graph, node_groups)
        ]

    def get_sharding_config(self, config_path: str) -> ShardingConfig:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        sharding_config = self.get_default_sharding_config()

        # Derive sharding groups from node_groups with tp_size > 1. The
        # node_groups section drives both worker assignment and TP grouping;
        # the separate "sharding_config" section just holds shard_dim.
        groups: list[ShardingGroup] = []
        node_moe_parallel_configs(config.get("node_groups", []))
        for ng in config.get("node_groups", []):
            if "ep_config" in ng and any(
                node not in sharding_config.ep_enabled_nodes for node in ng["node_names"]
            ):
                raise ValueError(
                    f"Node group with ep_config contains nodes not in "
                    f"ep_enabled_nodes {sharding_config.ep_enabled_nodes}: {ng['node_names']}"
                )
            instance_size = math.prod(node_group_axis_sizes(ng).values())
            if instance_size <= 1:
                continue
            # The sharding group is the lockstep unit — the whole instance's
            # workers, routed together. TP shards weights via shard_dim-driven
            # fanout; SP and CFG are in-module (replicated signals, no
            # shard_dim) but their ranks must still run each request together.
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

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """Declare the resources the engine builds, and which nodes share each.

        The whole declaration, apart from what `declare_state` declares: a KV
        cache's shape, the attention backend over it, positions, samplers. The
        engine builds one resource per spec and hands each node the subset naming it.
        An opt-in spec (per-request LoRA adapters) is declared here too, and built
        only for a deployment whose ``resources:`` block names it.
        """
        return []

    def declare_state(self) -> "list[Array | Sampler | SeqShard]":
        """The state the model keeps between steps, from which the engine builds the specs for it.

        A pure function of the model's config and the deployment's settings, since each process calls it.
        """
        return []

    def kernel_settings(self) -> dict[str, dict[str, Any]]:
        """By read, update or array name, what the engine doesn't pick itself yet: ``splits``, ``decode_fmha`` and
        ``decode_tensor_cores`` on a read of an array that keeps entries, ``splits`` on a GatedDeltaNet, the capture
        sizes of a ``Block`` read, and a pool's default size, ``max_num_pages`` on a paged array and ``max_slots`` on a
        Value with tensor fields. A deployment's ``resources:`` block overrides each pool size.

        A pure function of the model's config and the deployment's settings, as `declare_state` is, since each process
        calls it.
        """
        return {}

    def checkpoint_path(self) -> str | None:
        """Where this model's weights and config sit on disk.

        The prefix cache hashes the checkpoint's manifest into its root. None leaves
        the weights out, so the cache stays shut unless the deployment sets a salt.
        An override returns a path: processes that must not download it read the
        override alone (``names_checkpoint``).
        """
        return None

    def preprocess_fingerprint(self) -> str:
        """What this model's preprocessing contributes to the cache's root.

        Two deployments that tokenize or template a prompt differently must not
        match each other's pages; the class name separates them, and a model
        whose preprocessing changes shape between versions overrides this.
        """
        return type(self).__name__

    def get_walk_priorities(self) -> "list[WalkPriority]":
        """Entries naming a node, a walk, or one (node, walk), each with a priority and
        optionally a ``max_wait``: keys a worker serves before others while they have
        ready rows, higher first, unlisted 0, and the most steps a key may go without
        running while ready before it goes first. Where entries overlap, the most
        specific decides. A speculation chain gives way to a ready key above its own,
        or one past its bound, at once. Empty, the default, keeps round robin; a
        deployment's ``scheduling`` block can replace the list or turn it off. Declare
        a priority for a walk whose steps are short and latency-sensitive, such as a
        TTS prompt's prefill: a higher key that stays ready holds the rest back up to
        their ``max_wait``, or indefinitely without one."""
        return []

    def get_combined_graph_walks(self) -> dict[str, dict[str, set[str]]]:
        """``{node: {combined_walk: {constituent walks}}}``: walks the scheduler may batch together."""
        return {}

    def rebuild_policies(self) -> dict[str, RebuildPolicy]:
        """``{node: policy}``: how a request dropped from each node's KV is rebuilt. A node without one never
        drops a request."""
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
        ``{resource: {label: LabelLayout}}``, or a `Layout` under ``state_layout``. A label with keys opts its node into
        cross-request prefix reuse: the preprocess worker keys its layout's keyed
        spans and the KV resource matches what it already holds. With no keyed
        label the model is uncached whatever the deployment asks for.
        """
        return {}

    def get_request_resource_configs(
        self,
        input_modalities: list[str],
        output_modalities: list[str],
        model_kwargs: dict,
    ) -> "dict[str, ResourceReqConfig | SampleSettings]":
        """Per-resource config a new request is opened with, by resource label.

        Where a request's knobs live now: sampling params, whether it needs
        CFG, retention. The conductor resolves this once per request, from the
        request's modalities and kwargs as `walk_plan` gets them, and the
        engine hands each config to its resource at ingest.
        """
        del input_modalities, output_modalities, model_kwargs
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

    def get_chunking_policies(self) -> dict[str, tuple[ChunkingPolicy, ...]]:
        """``{node: (policy, ...)}``, each policy naming the walks it covers.

        A node's prefill for a chunkable walk runs as a Loop, one iteration per
        chunk, so the declaration has to be here rather than on the submodule:
        the graph build reads it to wrap the node. Pass the walk dict through
        ``get_graph_walk_graphs`` returns them wrapped (``apply_chunking``).
        """
        return {}

    def chunking_policy_of(self) -> dict[tuple[str, str], ChunkingPolicy]:
        """``(node, walk) -> policy``, validated."""
        walks = self.get_graph_walk_graphs()
        combined = self.get_combined_graph_walks()
        out: dict[tuple[str, str], ChunkingPolicy] = {}
        for node, policies in self.get_chunking_policies().items():
            for policy in policies:
                if not policy.walks:
                    raise ValueError(
                        f"{type(self).__name__}: a chunking policy for node "
                        f"{node!r} names no walks"
                    )
                for walk in policy.walks:
                    # a combined walk carries a budget for the steps it batches,
                    # and is not a walk of its own
                    real = walk in walks and node in walks[walk].get_nodes()
                    if not real and walk not in combined.get(node, {}):
                        raise ValueError(
                            f"{type(self).__name__}: chunking policy names walk "
                            f"{walk!r}, which is neither a walk of node {node!r} "
                            "nor a combined walk of it"
                        )
                    if policy.chunkable and not real:
                        raise ValueError(
                            f"{type(self).__name__}: node {node!r} cannot chunk "
                            f"combined walk {walk!r}; chunk its constituents"
                        )
                    if (node, walk) in out:
                        raise ValueError(
                            f"{type(self).__name__}: node {node!r} gets two "
                            f"chunking policies for walk {walk!r}"
                        )
                    out[(node, walk)] = policy
        return out

    def apply_chunking(
        self, walk_graphs: dict[str, GraphSection],
    ) -> dict[str, GraphSection]:
        """``get_graph_walk_graphs``'s result with every chunkable node wrapped
        in a Loop, one iteration per chunk.

        Applied to every model's ``get_graph_walk_graphs`` by ``Model``
        itself, so a model returns its walks unwrapped. A node already in its
        chunk Loop (a subclass building on its parent's walks) is left alone.
        """
        for node, policies in self.get_chunking_policies().items():
            for policy in policies:
                if not policy.chunkable:
                    continue
                for walk in policy.walks:
                    section = walk_graphs.get(walk)
                    if section is None or chunk_loop_name(node, walk) in section.get_loops():
                        continue
                    walk_graphs[walk] = _wrap_node_in_loop(section, node, walk, policy)
        return walk_graphs

    @abstractmethod
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        pass

    @abstractmethod
    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """The walks ``partition_name`` runs for this request, in order, decided
        at ingest; ``[]`` when the request doesn't use the partition.

        The conductor runs the plan: it sends each step's inputs from the walk's
        graph (see `WalkStep`), releases a persisted tensor once no later step
        reads it, and ends the partition after the last step reports. The step
        that samples the reply's first token sets `FINAL_PREFILL` in its
        ``step_metadata``. A step whose
        walk a stream feeds, and that runs no Loop but chunk loops, runs
        once per chunk until the worker reports the stream's final chunk, its
        inputs' tensors sent with the first chunk only. A
        ``ValueError`` refuses the request with a 400, and so does a plan of
        ``[]`` for every partition. A plan the conductor can't run is a model
        bug, refused with a 500: a step whose walk is not the partition's, a
        ``bind`` key that is not an input of the step's walk, a step after one
        that runs once per chunk, or a later step whose walk takes no input
        the conductor sends. In a partition a stream
        moves, the conductor moves past a step on its first report when it is
        the last step or the next step takes inputs it sends, so such a step
        must run once.
        """

    @abstractmethod
    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        open_streams: frozenset[str] = frozenset(),
        **kwargs,
    ) -> "NameToTensorList | ProcessPromptOutput":
        """Tokenize prompt and produce initial tensors for the request.

        Called by the API server data worker AFTER it has loaded raw
        multimodal tensors from file_paths (images, audio, video).
        The model may inspect the raw tensors dict and compute additional
        derived tensors (e.g., Qwen3-Omni computes ``pixel_values``,
        ``image_grid_thw`` and ``audio_features`` from the raw
        ``image_inputs`` / ``audio_inputs`` / ``video_inputs``).

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
            open_streams: The client streams the opening leaves open for
                appends; empty for a one-shot request. Raise
                ``ValueError`` here for a live opening the request's settings
                can't serve.
            **kwargs: Model-specific parameters (e.g., from model_kwargs).

        Returns:
            NameToTensorList with tensors to MERGE into the request's
            tensor dict.  Typically includes ``text_inputs`` plus any
            model-specific derived tensors.  The returned dict is merged
            into the existing ``tensors`` dict via ``dict.update``.
        """
        pass

    def prompt_tokens(self, tensors: NameToTensorList, kv_layout: dict[str, dict[str, LabelLayout]]) -> int:
        """A response's ``usage.prompt_tokens``: the positions the prompt fills,
        attachments included, as OpenAI and vLLM count them. ``tensors`` and
        ``kv_layout`` are what `process_prompt` returned, with a ``state_layout``
        converted to a ``kv_layout``. By default the ids in
        ``text_inputs``, which hold each attachment's pads in place; a model
        that splits ``text_inputs`` around its attachments adds theirs."""
        return sum(ids.numel() for ids in tensors.get("text_inputs", ()))

    def prompt_lora_adapters(self) -> frozenset[str]:
        """The LoRA adapters ``process_prompt`` may run a prompt on
        (``ProcessPromptOutput.lora_adapter``). The API server refuses, at startup, a
        deployment whose ``resources: lora: adapters:`` block does not serve them all."""
        return frozenset()

    def max_input_files(self) -> dict[str, int]:
        """Most files of each input modality one request may attach, e.g.
        ``{"audio": 1}``. The data worker refuses a request over it before
        decoding any file; a modality left out is held only to
        ``MSTAR_MAX_MEDIA_FILES``."""
        return {}

    def process_stream_chunk(self, chunk: ClientChunk) -> NameToTensorList:
        """Turn one append the client sends to a running request into items of
        the model's client streams (``Connection(from_partition=CLIENT, ...)``).

        Called by the API server data worker for each append that carries text,
        files or ``model_kwargs``, after it has loaded the files the way it loads a request's
        (``chunk.tensors``: ``audio_inputs``, ``image_inputs``,
        ``video_inputs``). ``chunk.streams`` names the stream the client meant
        each file for, where it named one; ``chunk.model_kwargs`` are the
        append's own, beside the opening's ``chunk.request_kwargs``. Returns
        items by client edge name; each list entry is one item of that stream.
        Raise ``ValueError`` for an append the model can't take (the client
        gets a 400); an append of text or ``model_kwargs`` that yields no item
        is refused.

        Keep it stateless: the consuming node folds items into its own state.
        Tensors ``process_prompt`` returns under a client edge's name are the
        stream's first chunk, so a one-shot request needs no append. A model
        whose client streams all declare ``ClientAudio`` needs none: the data
        worker turns its appends into items.
        """
        raise NotImplementedError(f"{type(self).__name__} takes no streamed input")

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        # RGB whatever the file holds: image processors cannot place the
        # channel axis of a 4-channel (RGBA or CMYK) image.
        img = read_image(filepath, rgb=True).to(device).float().div_(255.0)  # CxHxW

        return TensorAndMetadata(img)

    def load_decoded_image(self, image: torch.Tensor, device: str) -> TensorAndMetadata:
        """What :meth:`load_image` returns, for an image that reached the API
        server already decoded: RGB uint8 ``[3, H, W]``, as ``read_image(path,
        rgb=True)`` decodes a file. A model that takes such images (e.g. from
        the openpi endpoint) overrides this so the two paths agree; the rest
        take images only as files."""
        raise NotImplementedError(f"{type(self).__name__} takes images only as files")

    def robot_io(self) -> RobotIO | None:
        """A robot policy's request and output shapes, which the openpi
        WebSocket serves it by; None for a model that is not one."""
        return None

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        # libsndfile first, so WAV, FLAC, OGG and MP3 do not depend on the
        # FFmpeg build torchcodec happens to load (some base images ship a
        # video-only one); torchcodec for the other containers. When FFmpeg
        # loads, an upload neither decodes is the client's error (400).
        from mstar.model.components.audio_features import load_speech_file

        rate = self.AUDIO_INPUT_RATE
        audio = load_speech_file(filepath, rate, self.MAX_AUDIO_SECONDS)
        return TensorAndMetadata(data=audio, metadata=dict(sample_rate=rate, num_channels=1))

    def load_video(self, filepath: str, device: str):
        from torchcodec.decoders import VideoDecoder

        budget = VideoBudget()
        frames = []
        with decoding_upload(filepath, "the video", ffmpeg=True):
            decoder = VideoDecoder(filepath, device=device, num_ffmpeg_threads=video_decode_threads())
            for frame in decoder:
                budget.add(*frame.shape[-2:])
                frames.append(frame)
        video = torch.stack(frames)
        del frames
        video = video.float().div_(255.0)
        return TensorAndMetadata(data=video, metadata=video_metadata_dict(decoder.metadata))

    def text_output(self) -> TextOutput | None:
        """How this model's text output, token ids, becomes bytes, or None for a
        model that encodes text in its own ``postprocess``. ``postprocess`` and
        ``postprocess_inline`` derive their text paths from it."""
        return None

    _text_detokenizer: TextChunkDetokenizer | None = None

    def _chunk_detokenizer(self) -> TextChunkDetokenizer:
        """A ``TextChunkDetokenizer`` over ``self.tokenizer``, for ``text_output``."""
        if self._text_detokenizer is None or self._text_detokenizer.tokenizer is not self.tokenizer:
            self._text_detokenizer = TextChunkDetokenizer(self.tokenizer)
        return self._text_detokenizer

    def postprocess(
        self,
        output: torch.Tensor,
        modality: str,  # text | image | video | video_frame | audio
        request_kwargs: dict | None = None,
    ) -> bytes:
        """
        Given an output of a certain modality, encode and return as bytes.
        Text follows ``text_output`` when the model declares it, and audio is
        encoded below; the default refuses any other modality, which needs
        model-specific behavior.
        ``request_kwargs`` is the request's model_kwargs, so encoders can honor
        per-request parameters (e.g. writing a video container at the request's
        frame rate).

        An image or video artifact encodes on the API server's pool, beside the
        data worker's inline calls for the request's other outputs, so an
        override must not keep state across calls.

        Modality to expected encoding type:
        - text: utf-8; a chunk may end inside a character, since the API
          server joins a request's text chunks before decoding them
        - image: png
        - audio: headerless 16-bit PCM at ``get_output_sample_rate``. The
          default encodes mono: a float waveform in [-1, 1] is scaled and
          truncated, integer samples are cast. A model with more channels
          overrides it.
        """
        text = self.text_output() if modality == TEXT else None
        if text is not None:
            ids = drop_stop_tokens(output, text.stop_token_ids, request_kwargs)
            return text.detokenizer.to_bytes(ids)
        if modality == "audio":
            pcm = output.detach().cpu()
            if pcm.is_floating_point():
                # in fp32: bf16 and fp16 round 32767 up to 32768, which wraps
                pcm = pcm.float().clamp(-1, 1) * 32767
            return pcm.to(torch.int16).numpy().tobytes()
        raise ValueError(f"{type(self).__name__} has no {modality!r} output")

    def postprocess_inline(
        self,
        values: list[int],
        modality: str,
        request_kwargs: list[dict | None],
    ) -> list[bytes]:
        """``postprocess`` for one step's inline values (one per request) in
        one call: a declared ``text_output`` maps token ids straight to bytes,
        and anything else gets ``postprocess`` of a one-element tensor."""
        text = self.text_output() if modality == TEXT else None
        if text is None:
            return [
                self.postprocess(torch.tensor([v], dtype=torch.int64), modality, request_kwargs=kw)
                for v, kw in zip(values, request_kwargs, strict=True)
            ]
        stop, token_bytes = text.stop_token_ids, text.detokenizer.token_bytes
        return [
            b"" if v in stop and not (kw or {}).get("ignore_eos") else token_bytes(v)
            for v, kw in zip(values, request_kwargs, strict=True)
        ]

    @abstractmethod
    def get_submodule(
        self, node_name: str, device="cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        moe_parallel: "MoeParallelConfig | None" = None, cfg_group=None,
    ) -> torch.nn.Module | None:
        """Return the nn.Module for this node, or None for dummy mode.

        ``tp_group`` is the :class:`CommGroup` for this node on the
        calling worker (``None`` when TP is not active for this node).
        Models that support tensor parallelism should forward it to
        parallel-linear constructors so weight shapes are partitioned at
        init time and the weight loaders shard correctly.

        ``cfg_group`` is the node's classifier-free-guidance comm group, for
        a node in ``cfg_enabled_nodes`` whose group sets ``cfg_size > 1``
        (passed only then): each rank runs its ``my_rows(cfg_group,
        branches)`` share of the branches and ``cfg_group.all_gather``
        reassembles them. Nothing else about the node changes.

        ``moe_parallel`` is the node group's ``ep_config``: TP or EP for the
        node's routed experts. Passed only for a node in ``ep_enabled_nodes``
        whose group sets one; build MoE layers with ``instantiate_moe_block``.

        ``autocast_dtype`` is the dtype this node's params should be
        allocated in. Models that build on the meta device must cast the
        meta module to this dtype *before* ``to_empty(device)`` so storage
        is allocated directly in the target dtype — casting after
        ``to_empty`` first materialises every param in the meta default
        (float32), doubling the load-time VRAM peak. It is always the
        model-wide dtype (``get_autocast_dtype``, or YAML ``autocast_dtype``):
        the submodule does not exist yet, so a node with ``disable_autocast``
        gets it too. The engine manager's cast afterwards keeps such a node in
        whatever dtype it was built in, and moves a node with its own
        ``get_autocast_dtype`` to that dtype.
        """
        pass

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", MAX_OUTPUT_TOKENS)

    # The AR decode loop the token budget caps. Its prefill samples the reply's
    # first token, so the default ``get_loop_caps`` runs it one fewer than
    # ``get_max_output_tokens``.
    DECODE_LOOP: str | None = None

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        """The iterations each named loop may run for this request, each at least 1.

        A loop runs the smaller of its graph ``max_iters`` and its cap. A model
        whose request knows a loop's count sets the cap to that count, so the
        loop ends on that iteration and never speculates past it. Default:
        ``DECODE_LOOP``'s token budget, if the model declares one.
        """
        if self.DECODE_LOOP is None:
            return {}
        return {self.DECODE_LOOP: max(1, self.get_max_output_tokens(**model_kwargs) - 1)}

    def request_kwargs(self) -> frozenset[str] | None:
        """Every ``model_kwargs`` key this model reads beyond ``ENGINE_KWARGS``,
        ``LENGTH_KWARGS`` and ``INTERNAL_KWARGS``, including the ones an OpenAI
        adapter maps onto it. A request carrying any other key is served, and
        the key logged as ignored. None (the default) declares nothing, so
        nothing is logged.
        """
        return None

    def get_autocast_dtype(self):
        """The model-wide dtype nodes are cast to and autocast in. ``None``
        runs every node without autocast, in the dtype it was built in; YAML
        ``autocast_dtype`` may then only be ``null``."""
        return torch.bfloat16

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        """Sample rate (Hz) of audio output produced by ``postprocess``.

        Audio is emitted as headerless 16-bit PCM, so the client surfaces
        (SDK / OpenAI-compatible endpoints) need this rate to wrap it into a
        WAV/other container. Override per model; the 24 kHz default matches the
        speech codecs used by the audio models here.
        """
        return 24000

    # ``(start_id, end_id)``: the text tokens that open and close a full-duplex
    # model's turns (see "Streams from the client" in docs/adding_models.rst); None: none.
    turn_tokens: tuple[int, int] | None = None
    # A model whose replies may never get their end mark: a reply ends after this many
    # steps with no text, no mark and no audio above ``turn_silence_rms`` (the RMS of
    # samples in [-1, 1]), and text outside a reply opens one; None: only the marks end
    # and open them.
    turn_idle_steps: int | None = None
    turn_silence_rms: float = 0.005

    def get_output_audio_channels(self, modality: str = "audio") -> int:
        """Channel count of the interleaved 16-bit PCM ``postprocess`` emits for
        audio. Mono default (the speech models); stereo models override."""
        return 1

    def warmup_preprocess(self) -> None:
        """Called once in the preprocess worker before it takes requests.

        Build what ``process_prompt`` and ``postprocess`` would otherwise build
        lazily (a G2P front end, a tokenizer, a feature extractor, a compiled
        kernel) so the first request does not pay for it. The default does
        nothing.
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
            )
        ]
