"""The glue between declarations and today's resources, as one layer: a storage per kind of array, a binding per
storage and op type, and a handle per storage. It builds the specs, steps and layouts everything downstream reads."""

import dataclasses
import functools
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, replace
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.kv.bounded.config import (
    BoundedKVConfig,
    BoundedKVSpec,
    BoundedKVStep,
    SinkWindow,
    StreamPosition,
)
from mstar.engine.resources.kv.bounded.layout import tiles_head_dim
from mstar.engine.resources.kv.config import KVLayout, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
    bf16_gdn_state_fits,
)
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    Mamba2Geometry,
    RecurrentBlockConfig,
    RecurrentGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.sequence_shard import SeqShardSpec
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state.analysis import (
    combined_walks,
    position_key,
    query_array,
    reads_differ_by_walk,
    runs_on_walk,
    shard_axis,
    written_by_step,
)
from mstar.engine.state.calls import ArrayHandle, SamplerHandle, ShardHandle, pool_layers, segments_of
from mstar.engine.state.contract import (
    EMPTY,
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Block,
    Causal,
    CausalConv,
    Entry,
    GatedDeltaNet,
    KVEntry,
    Layout,
    Mamba2,
    Op,
    PagedNHDTarget,
    Positions,
    Reached,
    Read,
    Sampler,
    SampleSettings,
    SampleStep,
    SeqShard,
    TensorType,
    Window,
    is_value,
    read_layers,
)
from mstar.engine.state.host_fields import HostFields, is_host_value
from mstar.model.components.diffusion.attention import FrameSDPA, joint_attention, sdpa_attention

if TYPE_CHECKING:
    from mstar.engine.state.state_op import StateOp
    from mstar.model.base import LabelLayout, Model


def node_resources(model: "Model", *, setup: "StateSetup | None" = None) -> list[NodeResourceSpec]:
    """The specs ``model`` declares through `declare_state`, then the ones it writes by hand."""
    declarations = model.declare_state()
    setup = _with_walks(model, declarations, setup or StateSetup())
    specs = build_specs(
        declarations, kernel_settings=model.kernel_settings(), setup=setup,
        combined_walks=model.get_combined_graph_walks(),
    )
    return specs + list(model.get_node_resources() or [])


def _with_walks(model: "Model", declarations: Sequence[Array | Sampler], setup: "StateSetup") -> "StateSetup":
    """``setup`` with ``model``'s `Walks`. Building the walk graphs can read checkpoints, so each field is built only
    when an array's storage takes it: the nodes each walk runs, for a pool on several nodes, and the combined walks, for
    an array whose reads differ by walk."""
    held = [(storage_of(declared), declared) for declared in declarations if isinstance(declared, Array)]
    nodes = EMPTY
    if any(storage.takes_leader(array) for storage, array in held):
        nodes = {walk: frozenset(graph.get_nodes()) for walk, graph in model.get_graph_walk_graphs().items()}
    combined = model.get_combined_graph_walks() if any(storage.takes_combined_walks for storage, _ in held) else EMPTY
    return replace(setup, walks=Walks(nodes, combined))


def build_specs(
    declarations: Sequence[Array | Sampler | SeqShard], *, kernel_settings: Mapping[str, Mapping[str, Any]] = EMPTY,
    setup: "StateSetup | None" = None, combined_walks: Mapping[str, Mapping[str, Collection[str]]] = EMPTY,
) -> list[NodeResourceSpec]:
    """Each array's specs, from its storage and its ops' bindings, under the names the declarations give, then a
    `SamplerSpec` per sampler and a `SeqShardSpec` per sequence shard, which also lists the combined walks
    (``combined_walks``, the model's) that combine a listed walk. A read that ``setup`` binds takes that binding, and a
    pool on several nodes takes its leader from the walks ``setup`` gives.

    Raises ``ValueError`` on an array today's specs can't hold, naming what they lack. That includes a second update of
    one type on an array, a temporary limit: an update call finds its op by type, which today's resources need and the
    contract doesn't.
    """
    arrays = [declared for declared in declarations if isinstance(declared, Array)]
    resolved = _resolve(arrays, setup or StateSetup())
    specs: list[NodeResourceSpec] = []
    for array in arrays:
        specs += resolved.specs[array.name]
    specs += [_sampler_spec(declared) for declared in declarations if isinstance(declared, Sampler)]
    specs += [
        _shard_spec(declared, combined_walks, arrays) for declared in declarations if isinstance(declared, SeqShard)
    ]
    _apply_kernel_settings(specs, arrays, resolved, kernel_settings)
    return specs


def _resolve(arrays: Sequence[Array], setup: "StateSetup") -> "Resolved":
    """``arrays`` resolved by their storages, all of one storage's arrays in one call."""
    by_storage: dict[Storage, list[Array]] = {}
    for array in arrays:
        storage = storage_of(array)
        by_storage.setdefault(storage, []).append(array)
    specs: dict[str, list[NodeResourceSpec]] = {}
    bound: dict[str, tuple[BoundOp, ...]] = {}
    resource_of: dict[str, str] = {}
    queries: dict[str, QuerySide] = {}
    walk_reads: dict[str, WalkReads] = {}
    for storage, held in by_storage.items():
        each = storage.resolve(held, arrays, setup)
        specs |= each.specs
        bound |= each.bound
        resource_of |= each.resource_of
        queries |= each.queries
        walk_reads |= each.walk_reads
    return Resolved(specs, bound, resource_of, queries, walk_reads)


def state_setup(config: Mapping[str, Any], declarations: Sequence[Array | Sampler]) -> tuple["StateSetup", list[str]]:
    """The setup ``config`` asks for ``declarations``, and the problems with what it asks, each left out of the setup.
    A ``backend: sdpa`` block binds every read on its resource to SDPA."""
    # a config that still asks for SDPA on a node group would otherwise run the ragged kernel
    problems = [
        f"node_groups[{i}].attention_backend: the backend is a resource's setting, as "
        "`resources: {<the read's resource key>: {backend: sdpa}}`"
        for i, group in enumerate(config.get("node_groups") or ()) if "attention_backend" in group
    ]
    arrays = [declared for declared in declarations if isinstance(declared, Array)]
    resolved = _resolve(arrays, StateSetup())
    resource_of = {read: resolved.resource_of[read] for array in arrays for read in array.reads}
    by_read: dict[str, OpBinding] = {}
    for key, block in _sdpa_blocks(config).items():
        reads = [(array, read) for array in arrays for read in array.reads if resource_of[read] == key]
        if block.keys() != {"backend"}:
            problems.append(f"resources.{key}: backend sdpa builds no resource, so it takes no other setting, and it "
                            f"sets {sorted(block.keys() - {'backend'})}")
            continue
        if not reads:
            runs_on = f"the read {key!r} runs on {resource_of[key]!r}" if key in resource_of else "no read runs on it"
            problems.append(f"resources.{key}: backend sdpa, and {runs_on}, of the resources "
                            f"{sorted(set(resource_of.values()))}")
            continue
        refused = []
        for array, read in reads:
            binding = SDPA.get((storage_of(array), type(array.reads[read].op)))
            if binding is None:
                refused.append(f"resources.{key}: backend sdpa runs only an Attend read within a step, and Array "
                               f"{array.name!r}.reads[{read!r}] isn't one")
                continue
            try:
                binding.check(array, read, array.reads[read])
                by_read[read] = binding
            except ValueError as e:
                refused.append(f"resources.{key}: backend sdpa: {e}")
        if refused:
            problems += refused
            for _, read in reads:
                by_read.pop(read, None)
    return StateSetup(by_read), problems


def _sdpa_blocks(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The ``resources:`` blocks of ``config`` that set ``backend: sdpa``."""
    return {
        key: block for key, block in (config.get("resources") or {}).items()
        if isinstance(block, Mapping) and block.get("backend") == "sdpa"
    }


def spec_overrides(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """``config`` without its ``backend: sdpa`` blocks, which no spec takes: SDPA builds none, and `state_setup`
    reports a block it refuses."""
    sdpa = _sdpa_blocks(config)
    if not sdpa:
        return config
    return {**config, "resources": {key: block for key, block in config["resources"].items() if key not in sdpa}}


def _no_layout(array: Array, missing: str) -> ValueError:
    return ValueError(f"Array {array.name!r}: no layout yet: {missing}")


def _kv_shape(array: Array) -> tuple[int, ...]:
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or array.entry != KVEntry(heads=k.shape[0], head_dim=k.shape[1]):
        raise _no_layout(array, "entry must be a KVEntry, all PagedKVConfig holds: k and v of one shape, sharded on "
                                "their heads, with no dtype")
    return k.shape


def _kv_entry(array: Array, config: str) -> tuple[int, int, torch.dtype | None]:
    """``array``'s heads, head dim and dtype, None for the compute dtype. Its entry must be a `KVEntry` with one dtype,
    all that the resource config named ``config`` holds."""
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or k.shard != 0 or array.entry != Entry({"k": k, "v": k}):
        raise _no_layout(array, f"entry must be a KVEntry, which may declare one dtype, all {config} holds")
    return k.shape[0], k.shape[1], k.dtype


def _within_step_read(array: Array, setup: "StateSetup") -> tuple["BoundOp", int, int, torch.dtype | None]:
    """``array``'s one read, bound as ``setup`` binds it, its heads and head dim, and its dtype, None for the compute
    dtype."""
    heads, head_dim, dtype = _kv_entry(array, "RaggedAttentionConfig")
    if len(array.reads) != 1:
        raise _no_layout(array, f"max_len 0 needs one read, the one RaggedAttentionConfig runs, and it has "
                                f"{sorted(array.reads)}")
    (bound,) = _bindings(WITHIN_STEP, array, setup.by_read)
    return bound, heads, head_dim, dtype


def _block(tensor: TensorType) -> RecurrentBlockConfig:
    return RecurrentBlockConfig(
        shape=tensor.shape, dtype=tensor.dtype,
        shard_dims=() if tensor.shard is None else (tensor.shard,),
    )


def _position_spec(array: Array) -> PositionSpec:
    config = PositionConfig(kv_cache=array.name)
    rope = array.positions.rope
    if rope is not None:
        config = replace(
            config, rotary_dim=rope.rotary_dim, interleave=rope.interleave, rope_scale=rope.scale,
            rope_theta=rope.theta, rope_dtype=rope.dtype, rope_table_dtype=rope.table_dtype,
            low_freq_factor=rope.low_freq_factor, high_freq_factor=rope.high_freq_factor,
            old_context_len=rope.old_context_len,
        )
    return PositionSpec(resource_key=position_key(array), nodes=array.nodes, config=config)


def _sampler_spec(sampler: Sampler) -> SamplerSpec:
    return SamplerSpec(
        resource_key=sampler.name, nodes=sampler.nodes, vocab_size=sampler.vocab_size,
        enable_repetition_penalty=sampler.penalty, enable_min_p=sampler.min_p, min_tokens_stop_ids=sampler.eos_ids,
    )


def _shard_spec(
    shard: SeqShard, combined_walks: Mapping[str, Mapping[str, Collection[str]]], arrays: Sequence[Array],
) -> SeqShardSpec:
    walks = shard.walks
    if walks is not None:
        walks = walks | {
            combined for node in shard.nodes for combined, rows in combined_walks.get(node, {}).items()
            if walks & set(rows)
        }
    return SeqShardSpec(
        shard.name, shard.nodes, labels=tuple(sorted(shard.labels)), axis=shard_axis(shard, arrays), walks=walks,
        exchange=shard.exchange,
    )


def _apply_kernel_settings(
    specs: Sequence[NodeResourceSpec], arrays: Sequence[Array], resolved: "Resolved",
    kernel_settings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set the kernel choices, capture sizes and default pool sizes the contract leaves to the engine, and refuse any
    other, so this hook can't carry other physical choices until the engine picks kernels and sizes its pools itself."""
    # each key's owner and the resource its settings go on: an array's pool is under its name
    owners = {array.name: (array, storage_of(array), array.name) for array in arrays} | {
        bound.key: (array, bound.binding, resolved.resource_of[bound.key])
        for array in arrays for bound in resolved.bound[array.name] if bound.binding.kernel_settings
    }
    by_key = {spec.resource_key: spec for spec in specs}
    # a resource that several reads share holds one value per field
    set_by: dict[tuple[str, str], tuple[str, Any]] = {}
    for key, settings in kernel_settings.items():
        if key not in owners:
            raise ValueError(f"kernel_settings[{key!r}] names no read or update of the declarations that takes kernel "
                             f"settings, and no array, of {sorted(owners)}")
        array, owner, resource = owners[key]
        refused = settings.keys() - owner.kernel_settings
        if refused:
            raise ValueError(f"Array {array.name!r}: kernel_settings[{key!r}] sets {sorted(refused)}; "
                             f"it may set only {sorted(owner.kernel_settings)}")
        for field, value in settings.items():
            other, was = set_by.setdefault((resource, field), (key, value))
            if was != value:
                raise ValueError(f"kernel_settings[{other!r}] and kernel_settings[{key!r}] set {field} to {was!r} and "
                                 f"{value!r}, and both reads run on the resource {resource!r}, which holds one value")
        try:
            owner.apply(by_key.get(resource), settings)
        except ValueError as e:
            raise ValueError(f"Array {array.name!r}: kernel_settings[{key!r}]: {e}") from e


def node_handles(
    model: "Model", node: str, resources: Mapping[str, Any], *, setup: "StateSetup | None" = None,
    built: dict[str, ArrayHandle | SamplerHandle | ShardHandle] | None = None,
) -> dict[str, ArrayHandle | SamplerHandle | ShardHandle]:
    """``node``'s handles, by declared name, over its built ``resources``, each read bound as ``setup`` binds it.

    ``built`` maps declared names to the handles this process's other nodes built, and this call adds the ones it
    builds. A node shares the handle another node built for a declaration: nodes can share layers, as BAGEL's LLM
    nodes share one language model, and a handle per node would leave the shared layers on the handle of the node
    bound last, which the other nodes' steps never open.
    """
    setup = setup or StateSetup()
    built = {} if built is None else built
    declarations = model.declare_state()
    setup = _with_walks(model, declarations, setup)
    resolved = _resolve([declared for declared in declarations if isinstance(declared, Array)], setup)
    for declared in declarations:
        if node in declared.nodes and declared.name not in built:
            built[declared.name] = build_handle(declared, resources, resolved, setup)
    return {declared.name: built[declared.name] for declared in declarations if node in declared.nodes}


def build_handle(
    declared: Array | Sampler | SeqShard, resources: Mapping[str, Any], resolved: "Resolved",
    setup: "StateSetup | None" = None,
) -> ArrayHandle | SamplerHandle | ShardHandle:
    """``declared``'s handle over the built ``resources``, with each of its ops bound to the resource ``resolved``
    gives it."""
    if isinstance(declared, Sampler):
        return SamplerHandle(declared, resources[declared.name])
    if isinstance(declared, SeqShard):
        return ShardHandle(declared, resources[declared.name])
    return storage_of(declared).handle(declared, resources, setup or StateSetup(), resolved)


def state_steps(
    step: SubmoduleStep | None, handles: Mapping[str, ArrayHandle | SamplerHandle | ShardHandle], *,
    graph_walk: str | None = None,
) -> SubmoduleStep | None:
    """``step`` on ``graph_walk`` as a new step, with what its ``state`` declares expanded into today's step objects by
    each storage's builder, afresh for each step, since each carries the step's segments. The handles it opens are
    those of what ``state`` lists, then the node's sequence shard's, whose step is the node's segments. None when no
    array or sampler it lists needs it, nor a shard that can split, and it holds no other resource's step, so the step
    skips its resources' admit, plan and commit."""
    shards = tuple(handle for handle in handles.values() if isinstance(handle, ShardHandle))
    if step is None or not (step.state or shards):
        return step
    if not step.steps and not any(
        getattr(handles.get(name), "needs_step", True) for name in step.state
    ) and not any(shard.needs_step for shard in shards):
        return None
    steps = dict(step.steps)
    builders: dict[Storage, StepBuilder] = {}
    for name, declared in step.state.items():
        handle = handles.get(name)
        # the step's type first: a check against a handle's class goes through `ABCMeta` and costs more
        if isinstance(declared, ArrayStep) and isinstance(handle, ArrayHandle):
            if segments_of(declared, step) is None:
                raise ValueError(f"Array {name!r}: a step lists it with no segments, its own or the step's")
            builder = builders.get(handle.storage)
            if builder is None:
                builder = builders[handle.storage] = handle.storage.step_builder()
            builder.add(handle, declared, step, graph_walk)
        elif isinstance(declared, SampleStep) and isinstance(handle, SamplerHandle):
            steps[name] = SamplerStep(
                apply_penalty=declared.apply_penalty, apply_filters=declared.apply_filters,
                prefill_tracked_tokens=declared.prompt_tokens, kept_rids=declared.kept_rows,
            )
        else:
            raise ValueError(
                f"a step's state[{name!r}] is a {type(declared).__name__}, and on this node {name!r} is "
                f"{'not declared' if handle is None else type(handle).__name__}, which take an ArrayStep for an array "
                "and a SampleStep for a sampler, and a sequence shard takes none"
            )
    for builder in builders.values():
        steps.update(builder.steps())
    # A step's forward packs one set of labels together, which `SubmoduleStep` holds for all of its resources.
    combined = dict(step.combined_labels)
    for built in steps.values():
        combined.update(built.combined_labels)
    expanded = replace(step, steps=steps, combined_labels=combined)
    expanded.set_state_handles((*(handles[name] for name in step.state), *shards))
    return expanded


def check_capture(handles: Mapping[str, ArrayHandle | SamplerHandle], walks: Collection[str]) -> None:
    """Raise if a capture replayed on ``walks`` would plan an array's reads differently on some of them."""
    for handle in handles.values():
        if isinstance(handle, ArrayHandle):
            handle._check_capture(walks)


def step_labels(segments: Sequence[Segment]) -> tuple[str, ...]:
    """The labels ``segments`` name, in the order they first appear."""
    return tuple(dict.fromkeys(segment.label for segment in segments))


def _forks(array: Array, declared: ArrayStep) -> tuple[tuple, tuple, frozenset | None]:
    if not declared.forks:
        return (), (), None
    rows = {fork.rows for fork in declared.forks}
    if len(rows) > 1:
        raise _no_layout(array, "a step's forks name different rows, and a step's forks have one set of rows")
    return (
        tuple((fork.src, fork.dst) for fork in declared.forks if fork.at == "start"),
        tuple((fork.src, fork.dst) for fork in declared.forks if fork.at == "end"),
        rows.pop(),
    )


class StepBuilder(ABC):
    """Today's steps of one storage's arrays in one step."""

    @abstractmethod
    def add(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, walk: str | None = None) -> None:
        """``declared`` is what ``step``, on ``walk``, lists for ``handle``'s array."""

    @abstractmethod
    def steps(self) -> dict[str, Any]:
        """The steps of the arrays added, by resource key."""


class ArraySteps(StepBuilder):
    """Each array's steps on its own: its storage's, its ops' and its positions'."""

    def __init__(self, storage: "Storage"):
        self._storage = storage
        self._steps: dict[str, Any] = {}

    def add(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, walk: str | None = None) -> None:
        segments = segments_of(declared, step)
        self._storage.steps(handle, declared, step, segments, self._steps)
        for key, bound in handle._step_ops(walk):
            bound.binding.steps(handle.array, key, bound.declared, declared, segments, self._steps)

    def steps(self) -> dict[str, Any]:
        return self._steps


class WithinStepSteps(StepBuilder):
    """The step of each ragged resource, from the reads that share it: each label's segments, and the cross reads'
    pairs. A label is one layout on the resource, so two arrays with one label have to mean the same tokens."""

    def __init__(self):
        self._labels: dict[str, dict[str, Sequence[Segment]]] = {}
        self._pairs: dict[str, list[tuple[str, str]]] = {}
        # how the reads on each resource lay out their queries, which is one way per resource
        self._queries: dict[str, Queries] = {}
        # a resource's layout is fixed only if every read on it declares it so
        self._fixed: dict[str, bool] = {}

    def add(
        self, handle: "WithinStepHandle", declared: ArrayStep, step: SubmoduleStep, walk: str | None = None,
    ) -> None:
        """Plan ``handle``'s read over the step's whole segments: on a step the node's sequence shard splits, the
        resource narrows a split side to this rank's share itself."""
        if not handle._plans:
            return
        key = handle.resource_key
        self._fixed[key] = self._fixed.get(key, True) and declared.fixed_layout
        self._queries[key] = handle.queries
        handle.queries.plan(self, handle, segments_of(declared, step), step)

    def _label(self, key: str, array: Array, segments: Sequence[Segment]) -> str:
        """Hold ``segments``, ``array``'s in this step, under its label on the resource ``key``, and give the label."""
        if not segments:
            raise ValueError(f"Array {array.name!r}: a step lays it out over an empty list of segments")
        label = _plan_label(array, array)
        # on a replay too, whose step `_check_step` doesn't see
        for segment in segments:
            if segment.label != label:
                raise ValueError(f"Array {array.name!r}: a step lays it out under labels "
                                 f"{sorted({segment.label for segment in segments} - {label})}, and its plans run "
                                 f"under its label {label!r}")
        if tuple(self._labels.setdefault(key, {}).setdefault(label, segments)) != tuple(segments):
            raise ValueError(f"Array {array.name!r}: label {label!r} on its resource already holds other segments")
        return label

    def steps(self) -> dict[str, Any]:
        steps: dict[str, Any] = {}
        for key, by_label in self._labels.items():
            segments = tuple(s for each in by_label.values() for s in each) if len(by_label) > 1 else (
                next(iter(by_label.values()))
            )
            steps[key] = self._queries[key].step(segments, self._pairs.get(key, ()), self._fixed[key])
        return steps


def kv_layouts(
    declarations: Sequence[Array | Sampler | SeqShard],
    state_layout: Mapping[str, Mapping[str, Layout]],
    keyed_spans: Mapping[str, Mapping[str, int]],
) -> "dict[str, dict[str, LabelLayout]]":
    """Today's `LabelLayout` per paged array and label, from what a model's `process_prompt` returned. Any other
    array's layout is dropped: a Value's pool reserves a slot per label, an unbounded array's bounded KV a slot per
    request, and an array with ``max_len`` 0 keeps nothing.

    The model is at fault for a name it doesn't declare, so this raises ``RuntimeError``, which fails the request with a
    500, as `refuse_undeclared_labels` does.
    """
    arrays = {declared.name: declared for declared in declarations if isinstance(declared, Array)}
    for where, named in (("state_layout", state_layout), ("keyed_spans", keyed_spans)):
        for name, by_label in named.items():
            if name not in arrays:
                raise RuntimeError(f"{where}[{name!r}]: it names no declared array, of {sorted(arrays)}")
            undeclared = by_label.keys() - arrays[name].labels
            if undeclared:
                raise RuntimeError(f"{where}[{name!r}]: labels {sorted(undeclared)} aren't in Array {name!r}.labels")
    converted = {}
    for name, by_label in state_layout.items():
        layouts = storage_of(arrays[name]).layouts(by_label, keyed_spans.get(name, EMPTY))
        if layouts is not None:
            converted[name] = layouts
    for name, by_label in keyed_spans.items():
        unlaid = by_label.keys() - converted.get(name, {}).keys()
        if unlaid:
            raise RuntimeError(f"keyed_spans[{name!r}]: labels {sorted(unlaid)} have no paged layout in state_layout")
    return converted


def sampling_configs(
    configs: Mapping[str, ResourceReqConfig | SampleSettings], declarations: Sequence[Array | Sampler | SeqShard],
) -> dict[str, ResourceReqConfig]:
    """``configs`` with each `SampleSettings` as today's `SamplingReqConfig`. A name no sampler declares fails the
    request, and so do log-probs on a sampler with ``draws`` above 1, which take one token per step."""
    samplers = {declared.name: declared for declared in declarations if isinstance(declared, Sampler)}
    converted = dict(configs)
    for name, settings in configs.items():
        if not isinstance(settings, SampleSettings):
            continue
        sampler = samplers.get(name)
        if sampler is None:
            raise ValueError(f"SampleSettings for {name!r}, which names no declared sampler, of {sorted(samplers)}")
        if settings.return_logprobs and sampler.draws > 1:
            raise ValueError(f"Sampler {name!r}: return_logprobs, and draws is {sampler.draws}; log-probs take one "
                             "token per step")
        converted[name] = SamplingReqConfig(**asdict(settings))
    return converted


# Bindings


class BoundOp(NamedTuple):
    """One of an array's reads or updates, or its positions, under its key, with the binding that runs it."""

    key: str
    declared: Read | Op | Positions
    binding: "OpBinding"


class OpBinding:
    """How today's resources run one type of op on one storage."""

    # the fields `kernel_settings` may set on the op's spec, until the engine picks kernels itself
    kernel_settings: ClassVar[frozenset[str]] = frozenset()
    # those of them whose value in effect the bound runner reports, which `kernel_setting` reads back
    read_back: ClassVar[frozenset[str]] = frozenset()

    def check(self, array: Array, key: str, declared: Read | Op | Positions) -> None:
        """Raise ``ValueError`` on what today's resources can't run, naming ``array``."""
        return

    def specs(self, array: Array, key: str, declared: Read | Op | Positions) -> list[NodeResourceSpec]:
        return []

    def steps(
        self, array: Array, key: str, op: Read | Op | Positions, declared: ArrayStep, segments, steps: dict[str, Any],
    ) -> None:
        """Add the step of ``op``'s resource under ``key``, for ``array``'s ``declared`` step over ``segments``."""
        return

    def block(self, tensor: TensorType) -> RecurrentBlockConfig:
        """The pool's block for a field the op writes, declared as ``tensor``."""
        return _block(tensor)

    def stored(self, value: torch.Tensor) -> torch.Tensor:
        """``value`` in the layout `block` gives its field in the pool."""
        return value

    def from_stored(self, stored: torch.Tensor) -> torch.Tensor:
        """``stored``, read from the pool, in the layout the op declares its field in."""
        return stored

    def resource(self, array: Array, key: str) -> str:
        """The key of the resource the op under ``key`` runs on."""
        return key

    def apply(self, spec: NodeResourceSpec | None, settings: Mapping[str, Any]) -> None:
        """Set ``settings``, the op's `kernel_settings`, on ``spec``, its resource's."""
        for field, value in settings.items():
            setattr(spec.config, field, value)

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        """What a handle calls to run the op, on the resource under ``key``."""
        return resources[key]

    def attach(self, handle: ArrayHandle, key: str, run: Any) -> None:
        """Give ``handle`` ``run``, what it calls for the op under ``key``."""
        raise _no_layout(handle.array, f"{key!r} binds through {type(self).__name__}, which no paged array holds")


class PagedAttend(OpBinding):
    """`Attend` on a paged array: an `AttentionSpec` over the array's pool."""

    kernel_settings = frozenset({"splits", "decode_fmha", "decode_tensor_cores"})

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Causal | Window | Bidirectional):
            raise _no_layout(array, f"reads[{key!r}].mask must be Causal, Window or Bidirectional, the masks "
                                    "AttentionConfig has")
        if not read.packed:
            raise _no_layout(array, f"reads[{key!r}].packed is False, and the paged kernels take packed queries")

    def specs(self, array: Array, key: str, read: Read) -> list[NodeResourceSpec]:
        windowed = isinstance(read.mask, Window)
        # AttentionConfig.layers counts among the pool's layers
        kv_layer = pool_layers(array)
        return [AttentionSpec(
            resource_key=key, nodes=array.nodes,
            config=AttentionConfig(
                kv_cache=array.name,
                sliding_window=read.mask.size if windowed else None,
                layers=tuple(kv_layer[layer] for layer in sorted(read_layers(array, read))) if windowed else None,
            ),
        )]

    def steps(self, array: Array, key: str, read: Read, declared: ArrayStep, segments, steps: dict[str, Any]) -> None:
        steps[key] = AttentionStep(segments=segments, causal=read.mask.uses_query_position)

    def apply(self, spec: NodeResourceSpec | None, settings: Mapping[str, Any]) -> None:
        """The decode keys go through the spec's own YAML path and its checks, as they do under `resources:`."""
        decode = {field: settings[field] for field in settings.keys() & {"decode_tensor_cores", "decode_fmha"}}
        super().apply(spec, {field: settings[field] for field in settings.keys() - decode.keys()})
        if decode:
            spec.apply_yaml_overrides(**decode)

    def attach(self, handle: ArrayHandle, key: str, run: Any) -> None:
        handle.reads[key] = run


class PagedCrossAttend(PagedAttend):
    """`Attend` with ``queries`` on a paged array: a `CrossAttentionSpec` over the array's pool, planned by the queried
    array's segments."""

    kernel_settings = frozenset({"decode_tensor_cores"})

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Bidirectional):
            raise _no_layout(array, f"reads[{key!r}].mask must be Bidirectional, the mask CrossAttentionConfig has")
        super().check(array, key, read)
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since CrossAttentionConfig has no walk")

    def specs(self, array: Array, key: str, read: Read) -> list[NodeResourceSpec]:
        return [CrossAttentionSpec(
            resource_key=key, nodes=array.nodes,
            config=CrossAttentionConfig(
                kv_cache=array.name, query_kv_cache=read.queries, context_label=_plan_label(array, array),
            ),
        )]

    def steps(self, array: Array, key: str, read: Read, declared: ArrayStep, segments, steps: dict[str, Any]) -> None:
        steps[key] = AttentionStep(segments=segments, causal=False)


class PositionsBinding(OpBinding):
    """An array's `Positions`, which a position resource keeps per request under `position_key`, with the `Rope` its
    calls rotate by."""

    def specs(self, array: Array, key: str, declared: Positions) -> list[NodeResourceSpec]:
        return [_position_spec(array)]

    def steps(
        self, array: Array, key: str, positions: Positions, declared: ArrayStep, segments, steps: dict[str, Any],
    ) -> None:
        # a step that keeps nothing doesn't advance the counter
        advance = declared.advance if declared.keep else (0,) * len(segments)
        steps[key] = PositionStep(segments=segments, advance=advance, pos_ids=declared.position_ids)

    def attach(self, handle: ArrayHandle, key: str, run: Any) -> None:
        """Set the handle's position resource, and the dtype the declared `Rope` rotates in, None for the query's."""
        rope = handle.array.positions.rope
        handle.positions, handle.rope_dtype = run, None if rope is None else rope.dtype


class CounterPositions(OpBinding):
    """`Positions` with no `Rope`, on an array whose own resource counts each row's entries and answers `position`."""

    def check(self, array: Array, key: str, positions: Positions) -> None:
        if positions.rope is not None:
            raise _no_layout(array, "positions.rope, and its storage keeps no position ids to rotate at")

    def resource(self, array: Array, key: str) -> str:
        return array.name

    def steps(
        self, array: Array, key: str, positions: Positions, declared: ArrayStep, segments, steps: dict[str, Any],
    ) -> None:
        if declared.advance is not None or declared.position_ids is not None:
            raise _no_layout(array, "a step sets advance or position_ids, and its storage counts each row's entries")

    def attach(self, handle: ArrayHandle, key: str, run: Any) -> None:
        handle.positions = _Counter(handle.array, run)


class _Counter:
    """The positions of a resource that counts each row's entries: `position` and nothing else."""

    def __init__(self, array: Array, run: Any):
        self._array = array
        self.position = run.position

    def pos_ids(self, label: str) -> torch.Tensor:
        raise _no_layout(self._array, "positions(), and its positions are a count with no ids")

    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, *, label: str, rope_dtype: torch.dtype) -> tuple:
        raise _no_layout(self._array, "rope(), and its positions declare no Rope")


class BoundedAttend(OpBinding):
    """`Attend` on an unbounded array: the bounded KV's attention, which stores the step's keys as it attends. No spec,
    no step and no kernel setting."""

    def check(self, array: Array, key: str, read: Read) -> None:
        heads, head_dim, _ = _kv_entry(array, "BoundedKVConfig")
        if read.packed:
            raise _no_layout(array, f"reads[{key!r}] takes packed queries, and the bounded kernels take rows padded to "
                                    "the step's longest span; declare packed=False")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since BoundedKVConfig has no walk")
        if read.layers is not None and read.layers != array.layers:
            raise _no_layout(array, f"reads[{key!r}].layers must be the array's, since the bounded kernel stores a "
                                    "layer's add at its read")
        if read.op.q_heads != heads:
            raise _no_layout(array, f"reads[{key!r}].op.q_heads is {read.op.q_heads}, and the bounded kernels run "
                                    f"one query head per head of the entry's {heads}")
        if not tiles_head_dim(head_dim):
            raise _no_layout(array, f"entry head_dim {head_dim}, and the bounded kernels tile a power of two of at "
                                    "least 16")

    def resource(self, array: Array, key: str) -> str:
        return array.name

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        return functools.partial(_bounded_attend, resources[key])

    def attach(self, handle: ArrayHandle, key: str, run: Any) -> None:
        handle.reads[key] = run


def _bounded_attend(
    kv: Any, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, distance_bias: torch.Tensor | None,
) -> torch.Tensor:
    """The bounded KV's attention over ``q``, ``k`` and ``v``, (rows, T, heads, head_dim), in that layout too. The
    kernel takes the head axis second and one dtype, so q takes k's, and returns rows by time by heads."""
    return kv.attend(
        layer_idx, q.to(k.dtype).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), rel_bias=distance_bias,
        ieee=k.dtype == torch.float32,
    )


class WithinStepAttend(OpBinding, ABC):
    """`Attend` within a step, over the step's own adds."""

    # the model's settings are for the ragged kernel, whichever binding the deployment picks
    kernel_settings = frozenset({"max_segments_per_request", "max_segments_per_step", "compile_op", "workspace_mb"})
    # whether the read runs a plan its step builds, so its resource has a spec and its calls need the step open
    plans: ClassVar[bool]

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Block):
            raise _no_layout(array, f"reads[{key!r}].mask must be Block, the only mask a read within a step takes")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since a read within a step has no walk")

    @abstractmethod
    def attend(self, run: Any, label: str, rows: "RowLayout") -> Callable[..., torch.Tensor]:
        """The attention over rows of one length that the read's exchange runs, where ``run`` is what `bind` gave."""


class RaggedAttend(WithinStepAttend):
    """`Attend` within a step, on a ragged resource that the reads of one kind, nodes and config share."""

    read_back = frozenset({"max_segments_per_request", "max_segments_per_step"})
    plans = True

    def attend(self, run: Any, label: str, rows: "RowLayout") -> Callable[..., torch.Tensor]:
        return functools.partial(joint_attention, ragged=functools.partial(run.run, label=label))


class SdpaAttend(WithinStepAttend):
    """`Attend` within a step through SDPA, the reference pipelines' kernel: no spec, no step and no kernel setting."""

    plans = False

    def check(self, array: Array, key: str, read: Read) -> None:
        super().check(array, key, read)
        if read.packed:
            raise _no_layout(array, f"reads[{key!r}] runs SDPA, which can't tell one segment of packed queries from "
                                    "the next, and a read of rows of one length declares packed=False")

    def apply(self, spec: NodeResourceSpec | None, settings: Mapping[str, Any]) -> None:
        """Nothing: SDPA builds no resource to set."""

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        return sdpa_attention

    def attend(self, run: Any, label: str, rows: "RowLayout") -> Callable[..., torch.Tensor]:
        return rows.sdpa(run)


class LinearAttnBinding(OpBinding):
    """A recurrence a `LinearAttnSpec` runs under the op's key. Today's resources run the array's `CausalConv` inside
    it too."""

    # what the resource reads the pool's blocks as
    geometry: ClassVar[type[RecurrentGeometry]]

    def check(self, array: Array, key: str, op: Op) -> None:
        fields = [*op.writes, *CausalConv.writes]
        if set(array.entry.tensors) != set(fields):
            raise _no_layout(array, f"entry must be exactly {' and '.join(fields)}, the blocks "
                                    f"{self.geometry.__name__} reads")
        for field in op.writes:
            if array.entry.tensors[field].dtype is None:
                raise ValueError(f"Array {array.name!r}.entry.tensors[{field!r}].dtype must be declared, since the "
                                 f"rule that picks its stored dtype reads it")

    def steps(self, array: Array, key: str, op: Op, declared: ArrayStep, segments, steps: dict[str, Any]) -> None:
        steps[key] = LinearAttnStep(segments=segments)

    @staticmethod
    def kinds() -> str:
        """The op types bound as a linear attention update, for a refusal to name."""
        return ", ".join(sorted(kind.__name__ for (_, kind), binding in BINDINGS.items()
                                if isinstance(binding, LinearAttnBinding)))


class GDN(LinearAttnBinding):
    """`GatedDeltaNet` on a Value. It stores ``state`` K-last, (h_v, d_v, d_k) where the reference declares
    (h_v, d_k, d_v), since FlashInfer's kernels read it that way."""

    kernel_settings = frozenset({"splits"})
    geometry = DeltaNetGeometry

    def specs(self, array: Array, key: str, op: GatedDeltaNet) -> list[NodeResourceSpec]:
        return [LinearAttnSpec(
            resource_key=key, nodes=array.nodes,
            config=LinearAttnConfig(
                recurrent_state=array.name, variant=LinearAttnVariant.GDN, qk_l2norm=op.qk_l2norm, sm_scale=op.scale,
            ),
        )]

    def block(self, tensor: TensorType) -> RecurrentBlockConfig:
        """Stored in fp32 unless the decode kernel takes the declared dtype (`bf16_gdn_state_fits`)."""
        h_v, d_k, d_v = tensor.shape
        dtype = torch.bfloat16 if tensor.dtype is torch.bfloat16 and bf16_gdn_state_fits(d_k, d_v) else torch.float32
        # the head axis stays first, and the last two swap
        shard = None if tensor.shard is None else (0, 2, 1)[tensor.shard]
        return _block(TensorType((h_v, d_v, d_k), dtype=dtype, shard=shard))

    def stored(self, value: torch.Tensor) -> torch.Tensor:
        """The swap of the last two axes, which `from_stored` swaps back."""
        return value.transpose(-1, -2)

    def from_stored(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.transpose(-1, -2)


class SSD(LinearAttnBinding):
    """`Mamba2` on a Value: a `LinearAttnSpec` of the Mamba-2 variant over the Value's pool, which stores ``ssm`` as
    declared."""

    geometry = Mamba2Geometry

    def check(self, array: Array, key: str, op: Mamba2) -> None:
        super().check(array, key, op)
        try:
            self.geometry.from_blocks({field: _block(tensor) for field, tensor in array.entry.tensors.items()})
        except ValueError as e:
            raise _no_layout(array, str(e)) from e

    def specs(self, array: Array, key: str, op: Mamba2) -> list[NodeResourceSpec]:
        return [LinearAttnSpec(
            resource_key=key, nodes=array.nodes,
            config=LinearAttnConfig(
                recurrent_state=array.name, variant=LinearAttnVariant.MAMBA2, time_step_limit=op.time_step_limit,
            ),
        )]


class ConvOnLinearAttn(OpBinding):
    """`CausalConv` on a Value. Today's resources run it inside the array's one linear attention update, so its key
    maps to that update's resource and has no spec or step of its own."""

    def check(self, array: Array, key: str, op: CausalConv) -> None:
        self._linear_key(array, key)
        if op.activation != "silu":
            raise _no_layout(array, f"updates[{key!r}].activation is {op.activation!r}, and LinearAttnConfig has no "
                                    "activation but silu")

    def resource(self, array: Array, key: str) -> str:
        return self._linear_key(array, key)

    def _linear_key(self, array: Array, key: str) -> str:
        storage = storage_of(array)
        linear = [
            other for other, op in array.updates.items()
            if isinstance(BINDINGS.get((storage, type(op))), LinearAttnBinding)
        ]
        if len(linear) != 1:
            raise _no_layout(array, f"updates[{key!r}] is a CausalConv, which today's resources run inside one linear "
                                    f"attention update ({LinearAttnBinding.kinds()}), and the array has {linear}")
        return linear[0]


# Handles


def _exchanges(attn: Any, label: str) -> bool:
    """Whether a call on ``attn`` for ``label`` goes through its exchange (`AttentionManager.step_exchanges`); a
    stand-in attention, which is no `AttentionManager`, keeps the direct call."""
    return isinstance(attn, AttentionManager) and attn.step_exchanges(label)


class PagedArrayHandle(ArrayHandle):
    """Adds and reads through the KV pages of an array that keeps entries. Each read runs on the resource its
    `Resolved` gives it."""

    _write_refused = "write on an append-only array"

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
    ):
        super().__init__(array, resources)
        self._bindings = bindings
        self.reads = {}
        for bound in bindings:
            run = bound.binding.bind(array, resolved.resource_of[bound.key], resources)
            bound.binding.attach(self, bound.key, run)
        self.first_read = next(iter(self.reads.values()))
        self._attn = {
            layer: self.reads[key]
            for key, read in array.reads.items() for layer in read_layers(array, read)
        }
        self._kv = resources[array.name]
        self._views = {layer: self._kv.layer_view(i) for layer, i in self._pool_layer.items()}
        # each op with its resource, built once since `state_steps` runs on every step, replays too
        self._ops = [(resolved.resource_of[bound.key], bound) for bound in bindings]

    def _calls_cover(self, declared: ArrayStep, ctx) -> bool:
        # a capture opens the handles once around `preprocess`, where `eager_add` adds run, and again around the forward
        return bool(self._kept) or (declared.eager_add and ctx.capture)

    def _uncovered(self, rows: list[int]) -> ValueError:
        if self._region_rows:
            return super()._uncovered(rows)
        return self._refusal("a step that keeps lists rows at span above 0 and adds nothing, so the pages would keep "
                             "entries no add wrote; list those rows at span 0")

    def _check_labels(self, labels: set[str], segments) -> None:
        runs = [label for label, _ in groupby(segment.label for segment in segments)]
        if len(runs) > len(labels):
            raise self._no_layout(f"segments name {', '.join(map(repr, runs))}; a combined plan packs each label's "
                                  "rows together, in the step's order")
        twice = [stream for stream, n in Counter((s.request_id, s.label) for s in segments).items() if n > 1]
        if twice:
            raise self._refusal(f"row {twice[0][0]} has two segments on label {twice[0][1]!r} in one step, and a "
                                "combined plan holds one segment per row and label")

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        # A capture's dummy rows open only the default label.
        if ctx.capture:
            return
        opens_label = self._kv.opens_label
        for segment in segments:
            if (segment.request_id, segment.label) not in declared.placeholders and not opens_label(
                segment.request_id, segment.label,
            ):
                raise self._refusal(f"a step touches label {segment.label!r} of request {segment.request_id}, "
                                    "which it didn't open")

    def begin_step(self, step) -> None:
        super().begin_step(step)
        # adds that run before the reads, outside the captured graph, which the reads find in the pages
        self._eager = step.state[self.name].eager_add
        # the one plan `PagedStorage.steps` names for a step over several labels
        combined = step.combined_labels
        self._combined = bool(combined)
        if combined:
            self.label = next(iter(combined.values()))

    def check_step(self, step) -> None:
        # in the step's forward, before its commit: the last add, if its layer went unread
        self._write_unread()
        super().check_step(step)
        if not self._keep or step.ctx.replays:
            return
        if self._unwritten():
            raise self._refusal(f"the step ends with layer {self._held_layer}'s add unread, on a kernel that takes k "
                                "and v at the read, so no page holds it")
        if self._grows and not self._kept:
            raise self._refusal("a step that keeps lists rows at span above 0 and adds nothing, so the pages "
                                "would keep entries no add wrote; list those rows at span 0")

    @torch.compiler.disable
    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        layer = self._layer_id(layer)
        self._record_add(layer, keep)
        if self._held_layer != layer and self._unwritten():
            raise self._refusal(f"an add on layer {layer} while layer {self._held_layer}'s add waits for its read, on "
                                "a kernel that takes k and v at the read")
        self._write_unread()
        k, v = fields["k"], fields["v"]
        attn = self._attn.get(layer)
        if self._eager and attn is not None and self._needs_step_kv(attn):
            raise self._refusal(
                f"an eager_add step adds layer {layer} outside the graph and keeps nothing for its read, and that read "
                "needs the step's own k and v (an exchange, a context-parallel pool on a multi-token step, or the "
                "dense backend)")
        if attn is None or self._eager:
            self._write(layer, k, v)
            return
        # held for the read, which passes them on as a layer calls attention (`requires_kv_write`): a kernel may
        # attend the step's own k and v rather than the pages (`reads_step_kv`), and an exchange writes them
        self._held, self._held_layer = (k, v), layer
        if not _exchanges(attn, self.label):
            self._write(layer, k, v)

    def _write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """``layer``'s write alone, where its kernel reads the pages: through the exchange where the attention
        exchanges, as `AttentionCallable.write` makes it."""
        attn = self._attn.get(layer)
        if attn is not None and _exchanges(attn, self.label):
            attn.exchange(partial(self._attend, layer), None, k, v, label=self.label)
        elif attn is None or attn.requires_kv_write:
            self._kv.write_kv(k, v, layer_idx=self._pool_layer[layer], label=self.label)

    def _write_unread(self) -> None:
        """Write the held add of a layer the step didn't read where its write waits for the read's exchange, so
        each add is written once: when the next add or the step's check finds it still held."""
        layer = self._held_layer
        if layer is not None and _exchanges(self._attn[layer], self.label):
            self._write(layer, *self._take_hold(layer))

    def _unwritten(self) -> bool:
        """Whether no page holds the held add: its kernel takes k and v at the read, and no exchange writes them."""
        attn = self._attn.get(self._held_layer)
        return attn is not None and not attn.requires_kv_write and not _exchanges(attn, self.label)

    def _needs_step_kv(self, attn: Any) -> bool:
        """Whether a read through ``attn`` attends the step's own k and v rather than only the pages, or goes through
        its exchange."""
        return _exchanges(attn, self.label) or attn.reads_step_kv(self.label)

    @torch.compiler.disable
    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        if q is None:
            raise self._no_layout("a read with no op")
        if "distance_bias" in mask_args:
            raise self._no_layout("distance_bias on a paged read")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        layer = self._layer_id(layer)
        self._record_read(layer)
        return self._run(layer, q)

    def _check_add_read(self, call: str) -> None:
        """Nothing: a paged add and read run outside the graph already."""

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        attn = self._attn[layer]
        if _exchanges(attn, self.label):
            # the add's write and the attention as one call, on the tokens and heads this rank stores
            return attn.exchange(partial(self._attend, layer), q, *self._take_hold(layer), label=self.label)
        # a read with no add before it attends the pages alone, which a kernel that takes no writes can't
        k, v = self._take_hold(layer) if self._held is not None or not attn.requires_kv_write else (None, None)
        return attn.run(
            q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=self._pool_layer[layer],
        )

    def _attend(
        self, layer: int, q: torch.Tensor | None, k: torch.Tensor, v: torch.Tensor, tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """An exchange's ``run``: this layer's write where the kernel reads the pages, then its attention (``q``
        None: the write alone), as `AttentionCallable` makes the call. ``tokens``: which of the step's tokens k and v
        are, named by the ring's exchange alone (`sequence_shard.ring`)."""
        attn, i = self._attn[layer], self._pool_layer[layer]
        if attn.requires_kv_write:
            self._kv.write_kv(k, v, layer_idx=i, label=self.label, tokens=tokens)
        if q is None:
            return None
        return attn.run(q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=i)

    def at_last_token(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
        self._check_open("at_last_token")
        segments = self._segments
        first, end = 0, len(segments)
        if label is not None:
            at = [i for i, segment in enumerate(segments) if segment.label == label]
            if not at:
                raise self._refusal(f"at_last_token on label {label!r}, and the step's segments name "
                                    f"{list(step_labels(segments))}")
            first, end = at[0], at[-1] + 1
        # A span-0 segment adds no token to the plan, so its row gets the previous segment's last token. A call over
        # every row of a one-label step takes that for the rows that pad it.
        if label is not None or self._combined:
            empty = next((segment for segment in segments[first:end] if not segment.span), None)
            if empty is not None:
                raise self._refusal(f"at_last_token on row {empty.request_id}, label {empty.label!r}, whose segment "
                                    "has span 0 and so no token in this step")
        last = self._last_token_read().select_last_hidden(x, label=self.label)
        return last if label is None else last[first:end]

    def _last_token_read(self) -> Any:
        """The resource whose plan selects each segment's last token."""
        return self.first_read

    def _step_ops(self, walk: str | None) -> Sequence[tuple[str, BoundOp]]:
        return self._ops

    def takes_fused(self, op: "StateOp") -> bool:
        """An op that adds to the array, on a pool in one of its formats, while no layer's read takes k and v at its
        kernel, which would never see the pages the op writes. A step whose reads attend its own k and v also runs
        the reference (``fused_ops``)."""
        cache = self._kv.kv_cache
        dtype, device = cache.dtype, cache.device.type
        dense = any(not attn.requires_kv_write for attn in self._attn.values())
        return bool(op.adds) and cache.layout is KVLayout.NHD and not dense and any(
            (fmt.dtypes is None or dtype in fmt.dtypes) and (fmt.devices is None or device in fmt.devices)
            for fmt in op.formats
        )

    @property
    def fused_ops(self) -> frozenset[str]:
        # A fused op writes the pages itself and never passes k and v through `add`, so on a step whose reads attend
        # the step's own k and v (a context-parallel pool's multi-token step: `reads_step_kv`) every op runs its
        # reference. Fixed once the step is planned, so a captured bucket replays the same choice.
        if any(map(self._needs_step_kv, self.reads.values())):
            return frozenset()
        return self._fused_ops

    @fused_ops.setter
    def fused_ops(self, ops: frozenset[str]) -> None:
        self._fused_ops = ops

    def fused_target(self, layer: int) -> PagedNHDTarget:
        """Record ``layer``'s add, which a fused op makes, and give where it writes the step's entries."""
        self._record_add(layer, None)
        return PagedNHDTarget(*self._kv.write_target(layer_idx=self._pool_layer[layer], label=self.label))


class PagedByWalkHandle(PagedArrayHandle):
    """A paged array whose reads differ by walk: a step plans, reads through and selects by the reads of its walk,
    which ``walk_reads`` gives."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
    ):
        super().__init__(array, resources, bindings, resolved)
        self.walk_reads = walk_reads = resolved.walk_reads[array.name]
        by_key = {bound.key: bound for bound in bindings}
        every = [(resolved.resource_of[bound.key], bound) for bound in bindings if bound.key not in array.reads]
        # each walk's ops with their resources, under None for a walk no read names: its reads, then the other ops
        self._walk_ops = {
            walk: [*((resource, by_key[key]) for resource, key in reads.items()), *every]
            for walk, reads in walk_reads.reads.items()
        }
        # the layers some read covers on the step's walk `_walk`
        self._walk: str | None = None
        self._covered = walk_reads.covers[None]

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if not step.ctx.replays:
            self._walk = step.ctx.graph_walk
            self._covered = self.walk_reads.covers[self.walk_reads.key(self._walk)]

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        if layer not in self._covered:
            raise self._refusal(f"a read on layer {layer}, which no read covers on walk {self._walk!r}")
        return super()._run(layer, q)

    def _last_token_read(self) -> Any:
        # a walk plans only the resources of its own reads
        reads = self.walk_reads.reads[self.walk_reads.key(self._walk)]
        if not reads:
            raise self._refusal(f"at_last_token on walk {self._walk!r}, which runs no read of this array")
        return self.reads[next(iter(reads.values()))]

    def _check_capture(self, walks: Collection[str]) -> None:
        """FlashInfer fixes the mask when it plans, so every replay would run the capture walk's."""
        reads = self.walk_reads.reads
        for resource in {resource for by_resource in reads.values() for resource in by_resource}:
            keys = {reads[self.walk_reads.key(walk)].get(resource) for walk in walks} - {None}
            if len({self.array.reads[key].mask for key in keys}) > 1:
                raise self._no_layout(f"a capture replayed on walks {sorted(walks)} runs reads {sorted(keys)} on "
                                      f"{resource!r} with different masks, and a capture plans one")

    def _step_ops(self, walk: str | None) -> Sequence[tuple[str, BoundOp]]:
        if walk is None:
            raise self._no_layout("a step with no walk, on an array whose reads differ by walk")
        return self._walk_ops[self.walk_reads.key(walk)]


class PagedCrossHandle(PagedArrayHandle):
    """A paged array read with the queries of another array, whose segments lay them out. Its adds go to the pages, and
    its read runs under the query side's label, which the cross resource keys its plans by."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
    ):
        super().__init__(array, resources, bindings, resolved)
        ((key, _),) = array.reads.items()
        self.queries = resolved.queries[key]
        self._query_label = "main"
        # The cross read attends the pages and takes no k and v, so no layer has an attention its add is held for:
        # `add` writes the pages. The cross resource has no ``requires_kv_write`` to ask either.
        self._cross, self._attn = self._attn, {}

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if step.ctx.replays:
            return
        queries = self.queries.segments(self, step)
        labels = step_labels(queries)
        if len(labels) > 1:
            raise self._no_layout(f"a step over labels {sorted(labels)} on {self.queries.array.name!r}, whose segments "
                                  "lay out this read's queries")
        self.queries.check(self, self._segments, step)
        if queries:
            self._query_label = queries[0].label

    def _needs_step_kv(self, attn: Any) -> bool:
        """Never: a cross read attends the pages alone, and its resource has no ``reads_step_kv``."""
        return False

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        return self._cross[layer].run(q, label=self._query_label, kv_cache_layer=self._views[layer])

    def at_last_token(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
        self._check_open("at_last_token")
        raise self._refusal(f"at_last_token lays x out by this array's segments, and its read's queries are laid out "
                            f"by {self.queries.array.name!r}; call it on that array")


class ValueHandle(ArrayHandle):
    """The calls on a Value: its updates, which run in place on one layer's block, and reads and writes of whole
    fields."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
    ):
        super().__init__(array, resources)
        self._bindings = bindings
        self._pool = resources[array.name]
        self._blocks = {
            field: {layer: self._pool.block(field, i) for layer, i in self._pool_layer.items()}
            for field in array.entry.tensors
        }
        self._runners = {
            type(bound.declared): bound.binding.bind(array, resolved.resource_of[bound.key], resources)
            for bound in bindings
        }
        self._layouts = _field_layouts(array, bindings)
        # `_step_written` holds the fields a kept step writes itself, and `_written` those each layer's write set.
        # `_start` holds each layer's start state from `set_start`, in the pool's layout, fixed from the first step on.
        self._step_written = written_by_step(array)
        self._written: dict[int, set[str]] = {}
        self._start: dict[int, dict[str, torch.Tensor]] = {}
        self._stepped = False

    def begin_step(self, step) -> None:
        self._stepped = True
        self._written.clear()
        super().begin_step(step)

    def _calls_cover(self, declared: ArrayStep, ctx) -> bool:
        return all(self._step_written <= self._written.get(layer, set()) for layer in self.array.layers)

    def _uncovered(self, rows: list[int]) -> ValueError:
        layer = min(layer for layer in self.array.layers if not self._step_written <= self._written.get(layer, set()))
        unwritten = sorted(self._step_written - self._written.get(layer, set()))
        return self._refusal(f"layer {layer}'s fields {unwritten} are unwritten in a kept step for rows {rows}, and no "
                             "update op writes them")

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        if not declared.keep:
            raise self._refusal("keep=False on a Value with tensor fields, whose entry has no saved copy to return to, "
                                "until saved copies exist")
        super()._check_storage(declared, segments, ctx)

    def block(self, field: str, layer: int) -> torch.Tensor:
        blocks = self._blocks.get(field)
        if blocks is None or layer not in blocks:
            raise self._refusal(f"no field {field!r} on layer {layer}, of layers {sorted(self._pool_layer)}")
        return blocks[layer]

    @torch.compiler.disable
    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> Reached:
        """The step's rows' entries, as `Reached`. A Value takes no ``q``. A row with nothing written yet, with span 0
        or that pads the step reads the start `set_start` describes."""
        layer = self._layer_id(layer)
        if q is not None or mask_args:
            given = sorted({*mask_args, *(("q",) if q is not None else ())})
            raise self._refusal(f"a Value's read takes no q and no keyword, and got {given}")
        self._record_read(layer)
        if layer in self._written or any(updated == layer for _, updated in self._updated):
            raise self._refusal(f"a read of layer {layer} after its write or update in this step")
        start = self._start.get(layer, EMPTY)
        fields = {}
        for field in self.array.entry.tensors:
            read = self._pool.gather(field, self._pool_layer[layer], label=self.label, start=start.get(field))
            fields[field] = self._declared(field, read)
        return _OneEntryPerRow(fields)

    def set_start(self, start: Mapping[str, torch.Tensor]) -> None:
        if self._stepped:
            raise self._refusal("set_start after the handle's first step, whose captured reads keep the start "
                                "they read")
        undeclared = start.keys() - self.array.entry.tensors.keys()
        if undeclared:
            raise self._refusal(f"start names fields {sorted(undeclared)}, of {sorted(self.array.entry.tensors)}")
        self._start = {
            layer: {field: self._stored(field, layer, value, value.shape, "start") for field, value in start.items()}
            for layer in self.array.layers
        }

    def _stored(self, field: str, layer: int, value: torch.Tensor, shape: torch.Size, what: str) -> torch.Tensor:
        """``value``, whose entries are of ``shape``, in the layout and dtype the pool stores ``field`` in."""
        block = self.block(field, layer)
        layout = self._layouts[field]
        expected = layout.from_stored(block).shape[1:]
        dtype = self.array.entry.tensors[field].dtype or block.dtype
        if tuple(shape) != tuple(expected) or value.dtype != dtype:
            raise self._refusal(f"{what} of field {field!r} on layer {layer} is {tuple(shape)} of {value.dtype}, and "
                                f"the field is {tuple(expected)} of {dtype} on this rank")
        return layout.stored(value).to(block.dtype)

    def _declared(self, field: str, stored: torch.Tensor) -> torch.Tensor:
        """``stored``, read from the pool, in ``field``'s declared layout and dtype."""
        value = self._layouts[field].from_stored(stored)
        return value.to(self.array.entry.tensors[field].dtype or value.dtype)

    @torch.compiler.disable
    def write(self, layer: int | torch.Tensor, *, at: torch.Tensor | None = None, **fields: torch.Tensor) -> None:
        """Write each row's ``fields`` into its slot in place. The scatter changes the slots where the forward runs
        it, in an eager step or a replay, so a step refused after a field's write keeps it in the slots its rows
        already held."""
        if at is not None:
            raise self._no_layout("write at positions, on anything but a Value")
        layer = self._layer_id(layer)
        self._check_open("write")
        written = self._written.setdefault(layer, set())
        for field, value in fields.items():
            if field in written:
                raise self._refusal(f"a second write of field {field!r} on layer {layer} in one step")
            if value.ndim == 0:
                raise self._refusal(f"field {field!r} written with no row axis")
            written.add(field)
            stored = self._stored(field, layer, value, value.shape[1:], "write")
            self._pool.scatter_(field, stored, self._pool_layer[layer], label=self.label)

    def _update(self, kind: type[Op], layer: int) -> tuple[Any, torch.Tensor]:
        runner = self._runners.get(kind)
        if runner is None:
            raise self._refusal(f"an update of type {kind.__name__}, which the array doesn't declare")
        (field,) = kind.writes
        view = self.block(field, layer)
        self._record_update(field, layer)
        return runner, view


class _OneEntryPerRow(Reached):
    """A Value read's `Reached`. Its ``row_offsets`` is built only when asked for, so a read that doesn't ask launches
    no kernel for it, and a captured read records none."""

    def __init__(self, fields: Mapping[str, torch.Tensor]):
        object.__setattr__(self, "fields", fields)

    @functools.cached_property
    def row_offsets(self) -> torch.Tensor:
        rows = next(iter(self.fields.values()))
        return torch.arange(rows.shape[0] + 1, dtype=torch.int32, device=rows.device)


class _HostCalls:
    """The calls on a Value's host fields, which ``host_fields`` holds rather than a resource."""

    host_fields: HostFields

    def _check_host_writes(self, declared: ArrayStep, ctx) -> None:
        if not declared.keep:
            raise self._refusal("a host write on a step that keeps nothing, which has no commit to apply it at")
        for (row, label, layer), written in declared.host.items():
            if not ctx.is_real_row(row):
                raise self._write_refusal(row, label, layer, ", and that row isn't one of the step's requests")
            if label not in self.array.labels or layer not in self.array.layers:
                raise self._write_refusal(row, label, layer, f", and the array declares labels "
                                          f"{sorted(self.array.labels)} and layers {sorted(self.array.layers)}")
            if not self.host_fields.is_open(row):
                raise self._write_refusal(row, label, layer, ", and that request has no open fields")
            for field, value in written.items():
                kind = self.array.entry.host.get(field)
                if kind is None:
                    raise self._write_refusal(row, label, layer, f": field {field!r} isn't declared, of "
                                              f"{sorted(self.array.entry.host)}")
                if not is_host_value(value, kind):
                    raise self._write_refusal(row, label, layer, f": field {field!r} takes {kind}, and got {value!r}")

    def _write_refusal(self, row: int, label: str, layer: int, cause: str) -> ValueError:
        # built only on a refusal: every row of every step passes the checks above
        return self._refusal(f"a host write on row {row}, label {label!r}, layer {layer}{cause}")

    def commit_step(self, step) -> None:
        super().commit_step(step)
        self.host_fields.commit(self.array, step.state[self.name], step.ctx.request_ids)

    def _host(self, request: int, label: str, layer: int) -> Mapping[str, object]:
        fields = self.host_fields.fields(request, self.name, label, layer)
        if fields is None:
            raise self._refusal(f"no host fields for request {request}, label {label!r}, layer {layer}")
        return fields


class HostValueHandle(_HostCalls, ArrayHandle):
    """The calls on a Value of host fields only."""

    def __init__(self, array: Array, host_fields: HostFields):
        super().__init__(array, EMPTY)
        self.host_fields = host_fields
        host_fields.declare(array)

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        super()._check_storage(declared, segments, ctx)

    def _calls_cover(self, declared: ArrayStep, ctx) -> bool:
        """True: a Value of host fields only takes its rows' values from the step's host writes, not from calls."""
        return True


class ValueWithHostHandle(_HostCalls, ValueHandle):
    """The calls on a Value with host fields beside its tensor fields."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
        host_fields: HostFields,
    ):
        super().__init__(array, resources, bindings, resolved)
        self.host_fields = host_fields
        host_fields.declare(array)


class RowLayout:
    """How a within-step read's rows hold a step's segments: here a row a segment, all of one span, which keeps
    nothing of its step. Its handle picks one at bind, from the read."""

    # whether a step that lists the array opens, whatever the read's binding
    must_open: ClassVar[bool] = False

    def __init__(self, handle: "WithinStepHandle"):
        self._handle = handle

    def begin(self, segments: Sequence[Segment]) -> None:
        """Take a step's ``segments``."""

    def end(self) -> None:
        """Drop what the step's segments gave."""

    def sdpa(self, run: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        """The SDPA a read on these rows runs, given ``run`` over whole rows."""
        return run


class SegmentedRows(RowLayout):
    """A request's segments end to end in one row (`Read.segmented`), which SDPA attends within each segment of."""

    must_open = True

    def __init__(self, handle: "WithinStepHandle"):
        super().__init__(handle)
        self._frames = FrameSDPA()
        self._frames.lengths = None
        # One callable per array, as the handle's attend is. The disable keeps the spans out of a compiled caller's
        # guards, so its blocks don't retrace per layout.
        self._split = functools.partial(joint_attention, ragged=torch.compiler.disable(self._run))

    def begin(self, segments: Sequence[Segment]) -> None:
        if len({s.request_id for s in segments}) > 1:
            raise self._handle._no_layout("a segmented read in a step of several requests, whose rows need a split "
                                          "each")
        self._frames.lengths = [s.span for s in segments]

    def end(self) -> None:
        self._frames.lengths = None

    def sdpa(self, run: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        return self._split

    def _run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._frames.lengths is None:
            raise self._handle._refusal("read in a step that doesn't list the array")
        return self._frames.run(q, k, v)


def _plan_label(array: Array, reader: Array) -> str:
    """The label a plan of ``reader``'s read runs under on ``array``'s side: ``array``'s one declared label. A plan
    holds one label per call, so another count is refused, naming ``reader``."""
    try:
        (label,) = array.labels
    except ValueError:
        raise _no_layout(reader, f"a plan runs under one label, and Array {array.name!r} declares "
                                 f"{sorted(array.labels)}") from None
    return label


class QuerySide:
    """Whose segments lay out a read's queries, picked once per read at resolve: its own array's, or those of the array
    its ``queries`` names."""

    def __init__(self, array: Array):
        # the array whose segments lay out the queries
        self.array = array

    def check(self, handle: ArrayHandle, segments, step: SubmoduleStep) -> Sequence[Segment]:
        """Refuse a step whose queries don't line up with ``segments``, the read's own array's, and give the queries'
        segments."""
        return segments


class OtherArrayQueries(QuerySide):
    """Queries another array's segments lay out, which a step lists over the same requests in the same order."""

    def segments(self, handle: ArrayHandle, step: SubmoduleStep) -> Sequence[Segment]:
        """The segments that lay out ``handle``'s queries in ``step``, refused where it lists none."""
        name = self.array.name
        queries = step.state.get(name)
        if queries is None:
            raise ValueError(f"Array {handle.name!r}: a step lists it without {name!r}, whose segments lay out its "
                             "queries")
        query_segments = segments_of(queries, step)
        if query_segments is None:
            raise ValueError(f"Array {name!r}: a step lists it with no segments, its own or the step's")
        return query_segments

    def check(self, handle: ArrayHandle, segments, step: SubmoduleStep) -> Sequence[Segment]:
        queries = self.segments(handle, step)
        if [s.request_id for s in queries] != [s.request_id for s in segments]:
            raise handle._refusal(f"its segments and those of {self.array.name!r}, which lay out its queries, must "
                                  "name the same requests in the same order, with one segment to one")
        return queries


class Queries(QuerySide, ABC):
    """How a read within a step plans its queries on its ragged resource."""

    # the spec of the resource the read plans on
    spec_class: ClassVar[type[NodeResourceSpec]]

    @abstractmethod
    def plan_label(self, keys: Array) -> str:
        """The label the read's plan runs under, where ``keys`` is the read's own array."""

    @abstractmethod
    def plan(self, steps: WithinStepSteps, handle: "WithinStepHandle", segments, step: SubmoduleStep) -> None:
        """Lay out ``handle``'s read in ``steps``, over ``segments``, its own array's in ``step``."""

    @abstractmethod
    def step(self, segments, pairs: Sequence[tuple[str, str]], fixed: bool) -> Any:
        """The step of a resource whose reads lay out their queries this way."""


class OwnQueries(Queries):
    """A read whose queries are its own array's."""

    spec_class = RaggedAttentionSpec

    def plan_label(self, keys: Array) -> str:
        return _plan_label(keys, keys)

    def plan(self, steps: WithinStepSteps, handle: "WithinStepHandle", segments, step: SubmoduleStep) -> None:
        steps._label(handle.resource_key, handle.array, segments)

    def step(self, segments, pairs: Sequence[tuple[str, str]], fixed: bool) -> Any:
        return AttentionStep(segments=segments, causal=False, fixed_layout=fixed)


class CrossQueries(OtherArrayQueries, Queries):
    """A read within a step whose queries are another array's, on a ragged cross resource."""

    spec_class = RaggedCrossAttentionSpec

    def plan_label(self, keys: Array) -> str:
        # Both sides fix the plan's label, so it is one string for the life of the handle. See `cross_label`.
        return cross_label(_plan_label(self.array, keys), _plan_label(keys, keys))

    def plan(self, steps: WithinStepSteps, handle: "WithinStepHandle", segments, step: SubmoduleStep) -> None:
        key = handle.resource_key
        q_label = steps._label(key, self.array, self.segments(handle, step))
        steps._pairs.setdefault(key, []).append((q_label, steps._label(key, handle.array, segments)))

    def step(self, segments, pairs: Sequence[tuple[str, str]], fixed: bool) -> Any:
        return RaggedCrossAttentionStep(segments=tuple(segments), pairs=tuple(pairs))


class BoundedHandle(ArrayHandle):
    """Adds and reads through the bounded KV of an unbounded array, which keeps each row's newest entries in a slot and
    stores a step's keys as it attends, so an add holds k and v for its layer's read."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
    ):
        super().__init__(array, resources)
        self._bindings = bindings
        self.reads = {}
        for bound in bindings:
            run = bound.binding.bind(array, resolved.resource_of[bound.key], resources)
            bound.binding.attach(self, bound.key, run)
        self._attend = self.reads[next(iter(array.reads))]
        self._kv = resources[array.name]

    def written(self, rid: int) -> int:
        """The entries the bounded KV committed for ``rid``."""
        return self._kv.written(rid)

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        if not declared.keep:
            raise self._no_layout("keep=False on an unbounded array, whose kernel stores the step's keys as it attends")
        if declared.forks:
            raise self._no_layout("a fork of an unbounded array, which the bounded KV doesn't fork")
        super()._check_storage(declared, segments, ctx)

    @torch.compiler.disable
    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        layer = self._layer_id(layer)
        if keep is False:
            raise self._no_layout(f"keep=False on layer {layer} of an unbounded array, whose kernel stores the add it "
                                  "reads")
        self._record_add(layer, keep)
        self._check_nothing_held(layer)
        self._held, self._held_layer = (fields["k"], fields["v"]), layer

    @torch.compiler.disable
    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        """Attend ``q`` (rows, T, heads, head_dim) over each row's window and this layer's add, which the kernel then
        stores. ``distance_bias`` is the `Window`'s keyword."""
        if q is None:
            raise self._no_layout("a read with no op")
        distance_bias = mask_args.pop("distance_bias", None)
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        layer = self._layer_id(layer)
        self._record_read(layer)
        k, v = self._take_hold(layer)
        return self._attend(self._pool_layer[layer], q, k, v, distance_bias)


class WithinStepHandle(ArrayHandle):
    """The calls on an array with ``max_len`` 0, which a block compiled alone traces. With ``compile_op`` set, the read
    stays in that graph through ``mstar::ragged_attention``. A check here runs only while not compiling: it would guard
    on the layer, and the block would retrace per layer.

    ``queries`` says whose segments lay out the read's queries. Under sequence parallelism the read's resource picks
    the exchange per step from the node's sequence shard, which says whether the step splits the array's label: the
    handle hands it whole segments and calls its ``run``, and exchanges nothing itself.
    """

    _write_refused = "write on an append-only array"

    def __init__(
        self, array: Array, resources: Mapping[str, Any], bindings: tuple[BoundOp, ...], resolved: "Resolved",
    ):
        super().__init__(array, resources)
        self._bindings = bindings
        ((key, read, self._binding),) = bindings
        self.resource_key = resolved.resource_of[key]
        self.reads = {key: self._binding.bind(array, self.resource_key, resources)}
        self.first_read = self.reads[key]
        self.queries = queries = resolved.queries[key]
        self._packed = read.packed
        # a plain attribute, which a compiled caller's trace reads as a constant
        self._plans = self._binding.plans
        self._rows = (SegmentedRows if read.segmented else RowLayout)(self)
        self.needs_step = self._plans or self._rows.must_open
        self._plan_label = queries.plan_label(array)
        # One callable per array, not per layer: a callable per layer would retrace a compiled caller per layer.
        self._attend = self._binding.attend(self.first_read, self._plan_label, self._rows)

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if step.ctx.replays:
            return
        self.queries.check(self, self._segments, step)
        self._rows.begin(self._segments)

    def end_step(self, step) -> None:
        super().end_step(step)
        self._rows.end()

    def check_step(self, step) -> None:
        """Every add is scratch, so a step keeps none and has nothing to refuse as a whole."""

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        super()._check_storage(declared, segments, ctx)
        if declared.forks:
            raise self._refusal("a step forks an array with max_len 0, which keeps nothing to fork")

    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        """Hold k and v for this layer's read. Every add here is scratch."""
        if keep:
            raise self._refusal("keep=True, and max_len 0 keeps nothing between steps")
        if self._plans and not torch.compiler.is_compiling():
            self._held_layer = self._layer_id(layer)
            self._record_add(self._held_layer, False)
        self._held = fields["k"], fields["v"]

    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        """Attend ``q`` over this layer's latest add, within each row's segment. ``q`` is (tokens, heads, head_dim), or
        (rows, len, heads, head_dim) for rows of one length, which is the view an exchange and SDPA take."""
        if q is None:
            raise self._no_layout("a read with no op")
        if "distance_bias" in mask_args:
            raise self._no_layout("distance_bias within a step")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        if self._plans and not torch.compiler.is_compiling():
            self._record_read(self._layer_id(layer))
        # Dropped here, not at the step's end: a compiled frame would otherwise return k and v from its graph, which
        # keeps the whole projection they are views of alive through the next layer.
        k, v = self._take_hold(layer)
        if q.ndim == 3:
            if not self._packed:
                raise self._refusal("packed queries on a read declared packed=False")
            return self.first_read.run(q, k, v, label=self._plan_label)
        return self._attend(q, k, v)

    def _check_add_read(self, call: str) -> None:
        raise self._refusal(f"{call} with max_len 0, whose read a compiled block traces: a break here takes the read "
                            "out of that graph")


# Storages


class WalkReads(NamedTuple):
    """The reads of an array whose reads differ by walk: for each walk, the read it runs on each resource and the
    layers those reads cover, under None for a walk no read names. A combined walk's rows read by their own walks,
    which agree on each mask, and its layers are those every one of them covers."""

    reads: dict[str | None, dict[str, str]]
    covers: dict[str | None, frozenset[int]]

    def key(self, walk: str | None) -> str | None:
        """Where ``walk``'s reads sit: under its own name, or under None for a walk no read names."""
        return walk if walk in self.reads else None


class Resolved(NamedTuple):
    """Arrays resolved into today's resources: each array's specs, its ops' included, and its ops with the bindings that
    run them, by its name, the key of the resource each op runs on, how each read within a step or with ``queries``
    lays out its queries, and the `WalkReads` of each array whose reads differ by walk."""

    specs: dict[str, list[NodeResourceSpec]]
    bound: dict[str, tuple[BoundOp, ...]]
    resource_of: dict[str, str]
    queries: dict[str, QuerySide]
    walk_reads: Mapping[str, WalkReads] = EMPTY


class Storage(ABC):
    """How today's resources hold one kind of array: its specs, its own step, its layouts and its handle."""

    handle_class: ClassVar[type[ArrayHandle]]
    # the pool sizes `kernel_settings` may set under the array's name, until the engine sizes its pools itself
    kernel_settings: ClassVar[frozenset[str]] = frozenset()
    # whether an array here reads by walk, so a combined walk takes its reads from the walks it batches
    takes_combined_walks: ClassVar[bool] = False
    # whether an array here holds each row's own entries, whose pages a Value's update snapshots are keyed by
    holds_row_entries: ClassVar[bool] = False

    def __init__(self):
        # `state_steps` reaches a handle's storage, for its step builder, through the handle's class
        self.handle_class.storage = self

    @abstractmethod
    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        """The specs and read keys of ``arrays``, which are every declared array this storage holds, where ``declared``
        are all the declared arrays."""

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
    ) -> ArrayHandle:
        """``array``'s handle over a node's built ``resources``."""
        return self.handle_class(array, resources, resolved.bound[array.name], resolved)

    def steps(
        self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, segments, steps: dict[str, Any],
    ) -> None:
        """Add the step of the array's own resource, under its name, for ``declared`` in ``step`` over ``segments``."""
        return

    def takes_leader(self, array: Array) -> bool:
        """Whether ``array``'s pool takes a leader, the node that admits each request, from the nodes each walk runs."""
        return False

    def step_builder(self) -> StepBuilder:
        """What gathers the steps of this storage's arrays in one step."""
        return ArraySteps(self)

    def layouts(
        self, by_label: Mapping[str, Layout], keyed_spans: Mapping[str, int],
    ) -> "dict[str, LabelLayout] | None":
        """Today's `LabelLayout` per label, or None where the array's pool takes no layout."""
        return None

    def apply(self, spec: NodeResourceSpec, settings: Mapping[str, Any]) -> None:
        """Size the array's pool ``spec`` through its own YAML path, so a value here means what the same key under
        `resources:` means."""
        if settings:
            spec.apply_yaml_overrides(**settings)


class PagedStorage(Storage):
    """An append-only array that keeps entries, in a pool of KV pages. Its reads of one config share the spec of the
    first of them, in declaration order, unless a walk runs two of them with different masks. A step over several labels
    plans them as one combined plan."""

    handle_class = PagedArrayHandle
    kernel_settings = frozenset({"max_num_pages"})
    holds_row_entries = True

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        resolved = Resolved({}, {}, {}, {})
        for array in arrays:
            bound = resolved.bound[array.name] = _bindings(self, array, setup.by_read)
            reads = [each for each in bound if each.key in array.reads]
            others = [each for each in bound if each.key not in array.reads]
            specs, resource_of = self._read_specs(array, reads)
            resolved.specs[array.name] = [self._pool(array, setup.walks), *specs, *_op_specs(array, others)]
            resolved.resource_of.update(resource_of | _resources_of(array, others))
        return resolved

    def takes_leader(self, array: Array) -> bool:
        return len(array.nodes) > 1

    def _pool(self, array: Array, walks: "Walks") -> KVSpec:
        heads, head_dim = _kv_shape(array)
        if not array.reads:
            raise _no_layout(array, "reads is empty, and PagedKVConfig needs num_qo_heads from an Attend read")
        joined = sorted(label for label in array.labels if "+" in label)
        if joined:
            raise _no_layout(array, f"labels {joined} hold a '+', which joins the labels of a step over several labels "
                                    "into the name of its plan")
        q_heads = {read.op.q_heads for read in array.reads.values()}
        if len(q_heads) > 1:
            raise _no_layout(array, f"reads have q_heads {sorted(q_heads)}, and PagedKVConfig has one num_qo_heads")
        return KVSpec(
            resource_key=array.name, nodes=array.nodes,
            leader=_leader(array, walks) if self.takes_leader(array) else None,
            config=PagedKVConfig(
                num_layers=len(array.layers), num_qo_heads=q_heads.pop(),
                num_kv_heads=heads, head_dim=head_dim, max_seq_len=array.max_len,
            ),
        )

    def _read_specs(self, array: Array, bound: Sequence[BoundOp]) -> tuple[list[NodeResourceSpec], dict[str, str]]:
        """The specs ``array``'s reads build, and the resource key each read runs on.

        Raises ``ValueError`` on two specs over one layer, since a layer reads through one resource on every walk.
        """
        specs: dict[str, NodeResourceSpec] = {}
        resource_of: dict[str, str] = {}
        for key, read, binding in bound:
            (spec,) = binding.specs(array, key, read)
            resource_of[key] = next((
                first for first, shared in specs.items() if shared.config == spec.config and not any(
                    array.reads[member].mask != read.mask and _share_a_walk(array.reads[member], read)
                    for member, resource in resource_of.items() if resource == first
                )
            ), key)
            specs.setdefault(resource_of[key], spec)
        reader: dict[int, str] = {}
        for key, read in array.reads.items():
            for layer in sorted(read_layers(array, read)):
                other = reader.setdefault(layer, key)
                if resource_of[other] != resource_of[key]:
                    raise _no_layout(array, f"reads[{other!r}] and reads[{key!r}] build different specs and both cover "
                                            f"layer {layer}, which reads through one resource on every walk")
        return list(specs.values()), resource_of

    def steps(
        self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, segments, steps: dict[str, Any],
    ) -> None:
        pre, post, rows = _forks(handle.array, declared)
        ordered = segments_of(declared, step)
        # label-major, so a step over several labels starts and ends on different ones
        labels = step_labels(ordered) if ordered and ordered[0].label != ordered[-1].label else ()
        steps[handle.name] = KVStep(
            segments=segments, commit=declared.keep, pre_forks=pre, post_forks=post, fork_rids=rows,
            sink=declared.placeholders, eager_write=declared.eager_add,
            combined_labels={labels: "+".join(labels)} if labels else {},
        )

    def layouts(self, by_label: Mapping[str, Layout], keyed_spans: Mapping[str, int]) -> "dict[str, LabelLayout]":
        from mstar.model.base import LabelLayout

        return {
            label: LabelLayout(
                spans=layout.spans, growth=layout.growth, scratch=layout.scratch, keyed_spans=keyed_spans.get(label),
                max_growth=layout.max_growth,
            )
            for label, layout in by_label.items()
        }


def _leader(array: Array, walks: "Walks") -> str | None:
    """The node of ``array``'s pool that admits each request: the one node that runs every walk any of its nodes runs.
    None where no node or several do, so each node admits for itself."""
    if not walks.nodes:
        return None
    running = [nodes for nodes in walks.nodes.values() if not array.nodes.isdisjoint(nodes)]
    leaders = [node for node in sorted(array.nodes) if all(node in nodes for nodes in running)]
    return leaders[0] if len(leaders) == 1 else None


def _share_a_walk(a: Read, b: Read) -> bool:
    return any(runs_on_walk(a, walk) and runs_on_walk(b, walk) for walk in (None, *(a.walks or ()), *(b.walks or ())))


def _walk_reads(array: Array, resource_of: Mapping[str, str], combined: Mapping[str, Collection[str]]) -> WalkReads:
    reads: dict[str | None, dict[str, str]] = {}
    covers: dict[str | None, frozenset[int]] = {}
    for walk in (None, *{walk for read in array.reads.values() for walk in read.walks or ()}):
        covering = [key for key, read in array.reads.items() if runs_on_walk(read, walk)]
        reads[walk] = {resource_of[key]: key for key in covering}
        covers[walk] = frozenset().union(*(read_layers(array, array.reads[key]) for key in covering))
    walk_reads = WalkReads(reads, covers)
    for walk, members in combined.items():
        reads[walk] = {
            resource: key for member in sorted(members) for resource, key in reads[walk_reads.key(member)].items()
        }
        covers[walk] = frozenset.intersection(*(covers[walk_reads.key(member)] for member in members))
    return walk_reads


class PagedByWalkStorage(PagedStorage):
    """An append-only array that keeps entries, whose reads differ by walk. Each walk runs its own reads, a combined
    walk those of the walks it batches."""

    handle_class = PagedByWalkHandle
    takes_combined_walks = True

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        resolved = super().resolve(arrays, declared, setup)
        return resolved._replace(walk_reads={
            array.name: _walk_reads(array, resolved.resource_of, combined_walks(array, setup.walks.combined))
            for array in arrays
        })


class PagedCrossStorage(PagedStorage):
    """An append-only array that keeps entries, read with another array's queries: a cross resource over its pages,
    which plans the queries by the queried array's segments in each step. Its handle holds one queried array, so for
    now the array takes no other read."""

    handle_class = PagedCrossHandle
    # its pages hold a context, not a row's own entries
    holds_row_entries = False

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        for array in arrays:
            if len(array.reads) != 1:
                raise _no_layout(array, f"reads {sorted(array.reads)}, and a cross read's array takes no other read "
                                        "yet; declare each other read on an array of its own")
        by_name = {array.name: array for array in declared}
        resolved = super().resolve(arrays, declared, setup)
        for array in arrays:
            ((key, read),) = array.reads.items()
            queried = query_array(array, read, by_name)
            if not isinstance(storage_of(queried), PagedStorage):
                raise _no_layout(array, f"reads[{key!r}] keeps entries and reads the queries of {read.queries!r}, "
                                        "which isn't an append-only array that keeps entries in pages")
            resolved.queries[key] = OtherArrayQueries(queried)
        return resolved

    def steps(
        self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, segments, steps: dict[str, Any],
    ) -> None:
        handle.queries.segments(handle, step)
        super().steps(handle, declared, step, segments, steps)


class ValueStorage(Storage):
    """A rewritable array of one entry, in a recurrent state pool."""

    handle_class = ValueHandle
    kernel_settings = frozenset({"max_slots"})

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        resolved = Resolved({}, {}, {}, {})
        for array in arrays:
            resolved.specs[array.name], resolved.bound[array.name] = self._specs(array, declared)
            resolved.resource_of.update(_resources_of(array, resolved.bound[array.name]))
        return resolved

    def _specs(self, array: Array, arrays: Sequence[Array]) -> tuple[list[NodeResourceSpec], tuple[BoundOp, ...]]:
        bound = _bindings(self, array)
        ops = _op_specs(array, bound)
        kv_cache = None
        if any(spec.writes_prefix_snapshots for spec in ops):
            # the snapshots its update kernels take are keyed by the pages of the one paged array on its nodes
            holders = [other for other in arrays if storage_of(other).holds_row_entries and other.nodes & array.nodes]
            if len(holders) != 1 or not array.nodes <= holders[0].nodes:
                raise _no_layout(array, f"its nodes must hold one append-only array, the one "
                                        f"RecurrentStateConfig.kv_cache links to, and they hold "
                                        f"{[other.name for other in holders]}")
            kv_cache = holders[0].name
        layouts = _field_layouts(array, bound)
        pool = RecurrentStateSpec(
            resource_key=array.name, nodes=array.nodes,
            config=RecurrentStateConfig(
                num_layers=len(array.layers),
                blocks={field: layouts[field].block(tensor) for field, tensor in array.entry.tensors.items()},
                kv_cache=kv_cache,
                labels=tuple(sorted(array.labels)),
            ),
        )
        return [pool, *ops], bound

    def steps(
        self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, segments, steps: dict[str, Any],
    ) -> None:
        pre, post, rows = _forks(handle.array, declared)
        steps[handle.name] = RecurrentStep(segments=segments, pre_forks=pre, post_forks=post, fork_rids=rows)


class ValueWithHostStorage(ValueStorage):
    """A Value with host fields beside its tensor fields: the tensors in a recurrent state pool, which has no block
    for a host field, and the host fields where the engine holds them."""

    handle_class = ValueWithHostHandle

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
    ) -> ArrayHandle:
        return ValueWithHostHandle(array, resources, resolved.bound[array.name], resolved, setup.host_fields)


class HostValueStorage(Storage):
    """A Value of host fields only, which the engine holds, so it has no spec, no step and no layout."""

    handle_class = HostValueHandle

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        return Resolved(
            {array.name: [] for array in arrays}, {array.name: _bindings(self, array) for array in arrays}, {}, {},
        )

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
    ) -> ArrayHandle:
        return HostValueHandle(array, setup.host_fields)


class WithinStepStorage(Storage):
    """An array with ``max_len`` 0, whose adds last only the step, so it has no pool."""

    handle_class = WithinStepHandle

    def step_builder(self) -> StepBuilder:
        return WithinStepSteps()

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        """A ragged spec per read, a `RaggedCrossAttentionSpec` for a read with ``queries``. Reads of one kind, on one
        node set and with one config share the spec of the first of them, under its key, so each read is a plan on that
        resource rather than a resource of its own."""
        by_name = {array.name: array for array in declared}
        resolved = Resolved({}, {}, {}, {})
        shared: dict[tuple, NodeResourceSpec] = {}
        for array in arrays:
            bound, heads, head_dim, dtype = _within_step_read(array, setup)
            resolved.bound[array.name] = (bound,)
            key, read = bound.key, bound.declared
            queries = resolved.queries[key] = (OwnQueries if read.queries is None else CrossQueries)(
                query_array(array, read, by_name),
            )
            # refuses either side's second label at resolve, which validation and `build_specs` run before any handle
            queries.plan_label(array)
            config = RaggedAttentionConfig(
                num_qo_heads=read.op.q_heads, num_kv_heads=heads, head_dim=head_dim, dtype=dtype,
            )
            spec = queries.spec_class(resource_key=key, nodes=array.nodes, config=config)
            spec = shared.setdefault((queries.spec_class, array.nodes, astuple(config)), spec)
            resolved.resource_of[key] = spec.resource_key
            if bound.binding.plans and isinstance(spec, RaggedAttentionSpec):
                # the labels the node's sequence shard may split (`RaggedAttentionSpec.derive`)
                spec.labels = spec.labels | {_plan_label(array, array)}
            resolved.specs[array.name] = [spec] if bound.binding.plans and spec.resource_key == key else []
        return resolved

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
    ) -> ArrayHandle:
        return WithinStepHandle(array, resources, resolved.bound[array.name], resolved)


class BoundedStorage(Storage):
    """An unbounded array with one `Window` read, in a bounded KV that keeps each request's newest entries in a slot.
    Today's bounded pool holds one row per request and keeps the window as ``SinkWindow(0, size - 1)``."""

    handle_class = BoundedHandle
    kernel_settings = frozenset({"max_slots"})

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        resolved = Resolved({}, {}, {}, {})
        for array in arrays:
            heads, head_dim, dtype = _kv_entry(array, "BoundedKVConfig")
            if len(array.labels) != 1:
                raise _no_layout(array, f"an unbounded array takes one label, the bounded KV's one row per request, "
                                        f"and it has {sorted(array.labels)}")
            bound = resolved.bound[array.name] = _bindings(self, array, setup.by_read)
            reach = next(iter(array.reads.values())).mask.size - 1
            resolved.specs[array.name] = [BoundedKVSpec(
                resource_key=array.name, nodes=array.nodes,
                config=BoundedKVConfig(
                    num_layers=len(array.layers), num_heads=heads, head_dim=head_dim, rows_per_request=1,
                    max_source_len=0, retention=SinkWindow(0, reach), query_window=reach, dtype=dtype,
                ),
            )]
            resolved.resource_of.update(_resources_of(array, bound))
        return resolved

    def steps(self, handle: BoundedHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        steps[handle.name] = BoundedKVStep(segments=declared.segments, positions={
            segment.request_id: StreamPosition(source_len=0, written=handle.written(segment.request_id))
            for segment in segments_of(declared, step)
        })


PAGED = PagedStorage()
PAGED_BY_WALK = PagedByWalkStorage()
PAGED_CROSS = PagedCrossStorage()
VALUE = ValueStorage()
VALUE_WITH_HOST = ValueWithHostStorage()
HOST_VALUE = HostValueStorage()
WITHIN_STEP = WithinStepStorage()
BOUNDED = BoundedStorage()


def storage_of(array: Array) -> Storage:
    """The storage today's resources keep ``array`` in."""
    if is_value(array):
        if not array.entry.tensors:
            return HOST_VALUE
        return VALUE_WITH_HOST if array.entry.host else VALUE
    if array.rewritable:
        raise _no_layout(array, "a rewritable array has no spec unless it is a Value, of max_len 1")
    if array.max_len is None:
        if len(array.reads) != 1 or not isinstance(next(iter(array.reads.values())).mask, Window):
            masks = [type(read.mask).__name__ for read in array.reads.values()]
            raise _no_layout(array, f"an unbounded array reads through one Window read, whose window the bounded KV "
                                    f"keeps, and its reads have masks {masks}")
        return BOUNDED
    if array.max_len == 0:
        return WITHIN_STEP
    if any(read.queries is not None for read in array.reads.values()):
        return PAGED_CROSS
    return PAGED_BY_WALK if reads_differ_by_walk(array) else PAGED


# `Attend` is an `AttentionSpec` on a paged array, a `CrossAttentionSpec` on one read with another array's queries, the
# bounded KV's own attention on an unbounded array and a `RaggedAttentionSpec` within a step, so a binding is per
# storage and op type.
BINDINGS: dict[tuple[Storage, type[Op | Positions]], OpBinding] = {
    (PAGED, Attend): PagedAttend(),
    (PAGED, Positions): PositionsBinding(),
    (PAGED_BY_WALK, Attend): PagedAttend(),
    (PAGED_BY_WALK, Positions): PositionsBinding(),
    (PAGED_CROSS, Attend): PagedCrossAttend(),
    (PAGED_CROSS, Positions): PositionsBinding(),
    (WITHIN_STEP, Attend): RaggedAttend(),
    (BOUNDED, Attend): BoundedAttend(),
    (BOUNDED, Positions): CounterPositions(),
    (VALUE, GatedDeltaNet): GDN(),
    (VALUE, Mamba2): SSD(),
    (VALUE, CausalConv): ConvOnLinearAttn(),
}
# What `backend: sdpa` binds in place of the default, for each storage and op type SDPA can run.
SDPA: dict[tuple[Storage, type[Op]], OpBinding] = {(WITHIN_STEP, Attend): SdpaAttend()}


class Walks(NamedTuple):
    """The model's walks, where a storage needs them: the nodes each walk runs, from which a pool on several nodes takes
    its leader, and each node's combined walks, with the walks each batches, which give a combined walk its reads."""

    nodes: Mapping[str, frozenset[str]] = EMPTY
    combined: Mapping[str, Mapping[str, Collection[str]]] = EMPTY


@dataclass(frozen=True)
class StateSetup:
    """What a deployment sets up for the declared state: the binding it picks for each read whose resource doesn't take
    the default, the store of the Values' host fields, and the walks its nodes run."""

    by_read: Mapping[str, OpBinding] = EMPTY
    host_fields: HostFields = dataclasses.field(default_factory=HostFields)
    walks: Walks = Walks()


def _bindings(
    storage: Storage, array: Array, chosen: Mapping[str, OpBinding] = EMPTY,
) -> tuple[BoundOp, ...]:
    """``array``'s reads, then its updates, then its positions, each with the binding that runs it on ``storage``:
    ``chosen``'s for a read it names, and `BINDINGS`' for the rest. Each binding checks its op. Refuses a pair
    `BINDINGS` has no entry for, and a second update of one type."""
    bound = []
    for key, read in array.reads.items():
        binding = BINDINGS.get((storage, type(read.op)))
        if binding is None:
            raise _no_binding(storage, array, f"reads[{key!r}].op", update=False)
        bound.append(BoundOp(key, read, chosen.get(key, binding)))
    unbound = sorted(key for key, op in array.updates.items() if (storage, type(op)) not in BINDINGS)
    if unbound:
        raise _no_binding(storage, array, f"updates {unbound}", update=True)
    by_type: dict[type[Op], str] = {}
    for key, op in array.updates.items():
        other = by_type.setdefault(type(op), key)
        if other != key:
            raise _no_layout(array, f"updates[{other!r}] and updates[{key!r}] are both {type(op).__name__}, and an "
                                    "update call finds its op by type")
        bound.append(BoundOp(key, op, BINDINGS[storage, type(op)]))
    if array.positions is not None:
        binding = BINDINGS.get((storage, Positions))
        if binding is None:
            raise _no_layout(array, "positions, which only an append-only array that keeps entries holds")
        bound.append(BoundOp(position_key(array), array.positions, binding))
    for each in bound:
        each.binding.check(array, each.key, each.declared)
    return tuple(bound)


def _no_binding(storage: Storage, array: Array, where: str, *, update: bool) -> ValueError:
    """The refusal of ``where``, a read's op or an update ``storage`` has no binding for, naming the ops of that kind it
    binds: a read's op changes no field, and an update writes one."""
    kinds = sorted(
        kind.__name__ for held, kind in BINDINGS
        if held is storage and kind is not Positions and bool(kind.writes) == update
    )
    side = "update" if update else "read"
    if not kinds:
        return _no_layout(array, f"{where}: today's resources run no {side} on this array")
    return _no_layout(array, f"{where} must be {' or '.join(kinds)}, the {side}s today's resources run on this array")


def _resources_of(array: Array, bound: Sequence[BoundOp]) -> dict[str, str]:
    return {each.key: each.binding.resource(array, each.key) for each in bound}


def _op_specs(array: Array, bound: Sequence[BoundOp]) -> list[NodeResourceSpec]:
    return [spec for each in bound for spec in each.binding.specs(array, each.key, each.declared)]


def _field_layouts(array: Array, bound: Sequence[BoundOp]) -> dict[str, OpBinding]:
    """Each of ``array``'s fields with the binding that lays it out in the pool: the update's that writes it, or the
    base binding's layout as declared."""
    written = {field: each.binding for each in bound for field in each.declared.writes}
    return {field: written.get(field, OpBinding()) for field in array.entry.tensors}
