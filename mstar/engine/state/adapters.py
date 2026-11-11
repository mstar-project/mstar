"""The glue between declarations and today's resources, as one layer: a storage per kind of array, a binding per
storage and op type, and a handle per storage. It builds the specs, steps and layouts everything downstream reads."""

import dataclasses
import functools
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, replace
from itertools import groupby
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch

from mstar.distributed.communication import CommGroup
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
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state.analysis import combined_walks, position_key, reads_differ_by_walk, written_by_step
from mstar.engine.state.calls import ArrayHandle, SamplerHandle
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
    Op,
    Reached,
    Read,
    Sampler,
    SampleSettings,
    SampleStep,
    TensorType,
    Window,
    is_value,
)
from mstar.engine.state.host_fields import HostFields, is_host_value
from mstar.model.components.diffusion.attention import joint_attention, sdpa_attention
from mstar.model.components.distributed.sequence_parallel import (
    attention_over_sharded_keys,
    sp_seq_split,
    ulysses_attention,
)

if TYPE_CHECKING:
    from mstar.model.base import LabelLayout, Model


def node_resources(model: "Model", *, setup: "StateSetup | None" = None) -> list[NodeResourceSpec]:
    """The specs ``model`` declares through `declare_state`, then the ones it writes by hand."""
    declarations = model.declare_state()
    setup = _with_walks(model, declarations, setup or StateSetup())
    specs = build_specs(declarations, kernel_settings=model.kernel_settings(), setup=setup)
    return specs + list(model.get_node_resources() or [])


def _with_walks(model: "Model", declarations: Sequence[Array | Sampler], setup: "StateSetup") -> "StateSetup":
    """``setup`` with ``model``'s `Walks`, each built only where an array needs it, since building the walk graphs can
    read checkpoints: the nodes each walk runs for an array on several nodes, and the combined walks for an array whose
    reads differ by walk."""
    arrays = [declared for declared in declarations if isinstance(declared, Array)]
    nodes = EMPTY
    if any(len(array.nodes) > 1 for array in arrays):
        nodes = {walk: frozenset(graph.get_nodes()) for walk, graph in model.get_graph_walk_graphs().items()}
    combined = model.get_combined_graph_walks() if any(reads_differ_by_walk(array) for array in arrays) else EMPTY
    return replace(setup, walks=Walks(nodes, combined))


def build_specs(
    declarations: Sequence[Array | Sampler], *, kernel_settings: Mapping[str, Mapping[str, Any]] = EMPTY,
    setup: "StateSetup | None" = None,
) -> list[NodeResourceSpec]:
    """Each array's specs, from its storage and its ops' bindings, under the names the declarations give, then a
    `PositionSpec` for each array that declares positions and a `SamplerSpec` per sampler. A read that ``setup`` binds
    takes that binding, and a pool on several nodes takes its leader from the walks ``setup`` gives.

    Raises ``ValueError`` on an array today's specs can't hold, naming what they lack. That includes a second update of
    one type on an array, a temporary limit: an update call finds its op by type, which today's resources need and the
    contract doesn't.
    """
    arrays = [declared for declared in declarations if isinstance(declared, Array)]
    resolved = _resolve(arrays, setup or StateSetup())
    specs: list[NodeResourceSpec] = []
    for array in arrays:
        specs += resolved.specs[array.name]
        if array.positions is not None:
            specs.append(_position_spec(array))
    specs += [_sampler_spec(declared) for declared in declarations if isinstance(declared, Sampler)]
    _apply_kernel_settings(specs, arrays, resolved.resource_of, kernel_settings)
    return specs


def _resolve(arrays: Sequence[Array], setup: "StateSetup") -> "Resolved":
    """``arrays`` resolved by their storages, all of one storage's arrays in one call."""
    by_storage: dict[Storage, list[Array]] = {}
    for array in arrays:
        by_storage.setdefault(storage_of(array), []).append(array)
    specs: dict[str, list[NodeResourceSpec]] = {}
    resource_of: dict[str, str] = {}
    queried: dict[str, Array] = {}
    walk_reads: dict[str, WalkReads] = {}
    for storage, held in by_storage.items():
        each = storage.resolve(held, arrays, setup)
        specs |= each.specs
        resource_of |= each.resource_of
        queried |= each.queried
        walk_reads |= each.walk_reads
    return Resolved(specs, resource_of, queried, walk_reads)


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
    resource_of = _resolve(arrays, StateSetup()).resource_of
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


def _positions_key(array: Array) -> str | None:
    return None if array.positions is None else position_key(array)


def _no_layout(array: Array, missing: str) -> ValueError:
    return ValueError(f"Array {array.name!r}: no layout yet: {missing}")


def _kv_shape(array: Array) -> tuple[int, ...]:
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or array.entry != KVEntry(heads=k.shape[0], head_dim=k.shape[1]):
        raise _no_layout(array, "entry must be a KVEntry, all PagedKVConfig holds: k and v of one shape, sharded on "
                                "their heads, with no dtype")
    return k.shape


def _within_step_read(array: Array, setup: "StateSetup") -> tuple["BoundOp", int, int, torch.dtype | None]:
    """``array``'s one read, bound as ``setup`` binds it, its heads and head dim, and its dtype, None for the compute
    dtype."""
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or k.shard != 0 or array.entry != Entry({"k": k, "v": k}):
        raise _no_layout(array, "entry must be a KVEntry, which may declare one dtype, all RaggedAttentionConfig holds")
    if len(array.reads) != 1:
        raise _no_layout(array, f"max_len 0 needs one read, the one RaggedAttentionConfig runs, and it has "
                                f"{sorted(array.reads)}")
    if len(array.labels) != 1:
        raise _no_layout(array, f"max_len 0 takes one label, since a plan holds one per call, and it has "
                                f"{sorted(array.labels)}")
    (bound,) = _bindings(WITHIN_STEP, array, setup.by_read)
    return bound, k.shape[0], k.shape[1], k.dtype


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


def _apply_kernel_settings(
    specs: Sequence[NodeResourceSpec], arrays: Sequence[Array], resource_of: Mapping[str, str],
    kernel_settings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set the kernel choices, capture sizes and default pool sizes the contract leaves to the engine, and refuse any
    other, so this hook can't carry other physical choices until the engine picks kernels and sizes its pools itself."""
    owners = {array.name: (array, storage_of(array)) for array in arrays} | {
        bound.key: (array, bound.binding)
        for array in arrays for bound in _bindings(storage_of(array), array) if bound.binding.kernel_settings
    }
    by_key = {spec.resource_key: spec for spec in specs}
    # (spec, field) -> the key of `kernel_settings` that set it and its value
    set_by: dict[tuple[str, str], tuple[str, Any]] = {}
    for key, settings in kernel_settings.items():
        if key not in owners:
            raise ValueError(f"kernel_settings[{key!r}] names no read or update of the declarations that takes kernel "
                             f"settings, and no array, of {sorted(owners)}")
        array, owner = owners[key]
        refused = settings.keys() - owner.kernel_settings
        if refused:
            raise ValueError(f"Array {array.name!r}: kernel_settings[{key!r}] sets {sorted(refused)}; "
                             f"it may set only {sorted(owner.kernel_settings)}")
        # A read's settings go on the spec of the resource it runs on, which SDPA has none of. An update's or an
        # array's go on its own spec.
        spec = by_key.get(resource_of.get(key, key))
        for field, value in settings.items() if spec is not None else ():
            other, earlier = set_by.setdefault((spec.resource_key, field), (key, value))
            if earlier != value:
                raise ValueError(f"Array {array.name!r}: kernel_settings[{other!r}] and kernel_settings[{key!r}] set "
                                 f"{field} to {earlier!r} and {value!r}, and both reads run on {spec.resource_key!r}")
        try:
            owner.set_kernel_settings(array, spec, settings)
        except ValueError as e:
            raise ValueError(f"Array {array.name!r}: kernel_settings[{key!r}]: {e}") from e


def node_handles(
    model: "Model", node: str, resources: Mapping[str, Any], *, sp_group: CommGroup | None = None,
    setup: "StateSetup | None" = None, built: dict[str, ArrayHandle | SamplerHandle] | None = None,
) -> dict[str, ArrayHandle | SamplerHandle]:
    """``node``'s handles, by declared name, over its built ``resources``, each read bound as ``setup`` binds it.

    ``built`` maps declared names to the handles this process's other nodes built, and this call adds the ones it
    builds. A node shares the handle of an array or sampler another node built: nodes can share layers, as BAGEL's LLM
    nodes share one language model, and a handle per node would leave the shared layers on the handle of the node
    bound last, which the other nodes' steps never open. A shared within-step handle keeps the sequence-parallel group
    of the node that built it.
    """
    setup = setup or StateSetup()
    built = {} if built is None else built
    declarations = model.declare_state()
    setup = _with_walks(model, declarations, setup)
    resolved = _resolve([declared for declared in declarations if isinstance(declared, Array)], setup)
    for declared in declarations:
        if node in declared.nodes and declared.name not in built:
            built[declared.name] = build_handle(declared, resources, setup, resolved, sp_group)
    return {declared.name: built[declared.name] for declared in declarations if node in declared.nodes}


def build_handle(
    declared: Array | Sampler, resources: Mapping[str, Any], setup: "StateSetup | None" = None,
    resolved: "Resolved | None" = None, sp_group: CommGroup | None = None,
) -> ArrayHandle | SamplerHandle:
    """``declared``'s handle over the built ``resources``, with each of its ops bound. Each read runs on the resource
    ``resolved`` gives it, or on its own key when ``resolved`` is None."""
    if isinstance(declared, Sampler):
        return SamplerHandle(declared, resources[declared.name])
    resolved = resolved or Resolved({}, {}, {})
    return storage_of(declared).handle(declared, resources, setup or StateSetup(), resolved, sp_group)


def state_steps(
    step: SubmoduleStep | None, handles: Mapping[str, ArrayHandle | SamplerHandle], *, graph_walk: str | None = None,
) -> SubmoduleStep | None:
    """``step`` on ``graph_walk`` as a new step, with what its ``state`` declares expanded into today's step objects by
    each storage's builder, built afresh for each step because `SubmoduleStep` writes the segments onto them. None when
    no array or sampler it lists needs it, so the step skips its resources' admit, plan and commit."""
    if step is None or not step.state:
        return step
    if not any(getattr(handles.get(name), "needs_step", True) for name in step.state):
        return None
    steps = dict(step.steps)
    builders: dict[Storage, StepBuilder] = {}
    for name, declared in step.state.items():
        handle = handles.get(name)
        # the step's type first: a check against a handle's class goes through `ABCMeta` and costs more
        if isinstance(declared, ArrayStep) and isinstance(handle, ArrayHandle):
            if declared.segments is None and step.segments is None:
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
                "and a SampleStep for a sampler"
            )
    for builder in builders.values():
        steps.update(builder.steps())
    expanded = replace(step, steps=steps)
    expanded.set_state_handles(tuple(handles[name] for name in step.state))
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


def _queries_step(handle: "PagedCrossHandle | WithinStepHandle", step: SubmoduleStep) -> ArrayStep:
    """What ``step`` lists for the array whose segments lay out the queries of ``handle``'s cross read."""
    queries = step.state.get(handle.queried.name)
    if queries is None:
        raise ValueError(f"Array {handle.name!r}: a step lists it without {handle.queried.name!r}, whose segments lay "
                         "out its queries")
    return queries


def _query_segments(
    handle: "PagedCrossHandle | WithinStepHandle", step: SubmoduleStep, segments: Sequence[Segment],
) -> Sequence[Segment]:
    """The segments that lay out the queries of ``handle``'s cross read in ``step``, refused unless they name the
    requests of ``segments``, the array's own, in their order."""
    queries = step.state[handle.queried.name].segments or step.segments or ()
    if [s.request_id for s in queries] != [s.request_id for s in segments]:
        raise handle._refusal(f"its segments and those of {handle.queried.name!r}, which lay out its queries, must "
                              "name the same requests in the same order, with one segment to one")
    return queries


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
        segments = declared.segments
        self._storage.steps(handle, declared, step, self._steps)
        for key, bound in handle._step_ops(walk):
            bound.binding.steps(key, bound.declared, segments, self._steps)
        if handle.array.positions is not None:
            # a step that keeps nothing doesn't advance the counter
            advance = declared.advance if declared.keep else (0,) * len(segments or step.segments)
            self._steps[handle.positions_key] = PositionStep(
                segments=segments, advance=advance, pos_ids=declared.position_ids,
            )

    def steps(self) -> dict[str, Any]:
        return self._steps


class WithinStepSteps(StepBuilder):
    """The step of each ragged resource, from the reads that share it: each label's segments, and the cross reads'
    pairs. A label is one layout on the resource, so two arrays with one label have to mean the same tokens."""

    def __init__(self):
        self._labels: dict[str, dict[str, Sequence[Segment]]] = {}
        self._pairs: dict[str, list[tuple[str, str]]] = {}
        # a resource's layout is fixed only if every read on it declares it so
        self._fixed: dict[str, bool] = {}

    def add(
        self, handle: "WithinStepHandle", declared: ArrayStep, step: SubmoduleStep, walk: str | None = None,
    ) -> None:
        """Plan ``handle``'s read over the segments its kernel sees. A Ulysses read runs over whole segments after the
        exchange, and any other over this rank's share of a sharded side."""
        if not handle._plans:
            return
        key, name, segments = handle.resource_key, handle.name, declared.segments or step.segments
        self._fixed[key] = self._fixed.get(key, True) and declared.fixed_layout
        if handle.queried is None:
            self._label(key, name, segments)
            return
        q_label = self._label(key, name, handle.query_share(_queries_step(handle, step).segments or step.segments))
        self._pairs.setdefault(key, []).append((q_label, self._label(key, name, handle.key_share(segments))))

    def _label(self, key: str, name: str, segments: Sequence[Segment]) -> str:
        label = segments[0].label
        if tuple(self._labels.setdefault(key, {}).setdefault(label, segments)) != tuple(segments):
            raise ValueError(f"Array {name!r}: label {label!r} on its resource already holds other segments")
        return label

    def steps(self) -> dict[str, Any]:
        steps: dict[str, Any] = {}
        for key, by_label in self._labels.items():
            segments = tuple(s for each in by_label.values() for s in each) if len(by_label) > 1 else (
                next(iter(by_label.values()))
            )
            if key in self._pairs:
                steps[key] = RaggedCrossAttentionStep(segments=tuple(segments), pairs=tuple(self._pairs[key]))
            else:
                steps[key] = AttentionStep(segments=segments, causal=False, fixed_layout=self._fixed[key])
        return steps


def kv_layouts(
    declarations: Sequence[Array | Sampler],
    state_layout: Mapping[str, Mapping[str, Layout]],
    keyed_spans: Mapping[str, Mapping[str, int]],
) -> "dict[str, dict[str, LabelLayout]]":
    """Today's `LabelLayout` per paged array and label, from what a model's `process_prompt` returned. Any other
    array's layout is dropped: a Value's pool reserves a slot per label, and an array with ``max_len`` 0 keeps nothing.

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
    configs: Mapping[str, ResourceReqConfig | SampleSettings], declarations: Sequence[Array | Sampler],
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
    """One of an array's reads or updates, under its key, with the binding that runs it."""

    key: str
    declared: Read | Op
    binding: "OpBinding"


class OpBinding:
    """How today's resources run one type of op on one storage."""

    # the fields `kernel_settings` may set on the op's spec, until the engine picks kernels itself
    kernel_settings: ClassVar[frozenset[str]] = frozenset()

    def check(self, array: Array, key: str, declared: Read | Op) -> None:
        """Raise ``ValueError`` on what today's resources can't run, naming ``array``."""
        return

    def specs(self, array: Array, key: str, declared: Read | Op) -> list[NodeResourceSpec]:
        return []

    def steps(self, key: str, declared: Read | Op, segments, steps: dict[str, Any]) -> None:
        """Add the step of the op's resource under ``key``, over ``segments``."""
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

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        """What a handle calls to run the op."""
        return resources[key]

    def set_kernel_settings(self, array: Array, spec: NodeResourceSpec | None, settings: Mapping[str, Any]) -> None:
        """Set ``settings`` on the op's ``spec``, which is None for a read on SDPA."""
        if spec is not None:
            for field, value in settings.items():
                setattr(spec.config, field, value)


class PagedAttend(OpBinding):
    """`Attend` on a paged array: an `AttentionSpec` over the array's pool."""

    kernel_settings = frozenset({"splits", "decode_fmha", "decode_tensor_cores"})

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Causal | Window | Bidirectional):
            raise _no_layout(array, f"reads[{key!r}].mask must be Causal, Window or Bidirectional, the masks "
                                    "AttentionConfig has")

    def specs(self, array: Array, key: str, read: Read) -> list[NodeResourceSpec]:
        windowed = isinstance(read.mask, Window)
        # the pool holds only the array's layers, so AttentionConfig.layers counts among them
        kv_layer = {layer: i for i, layer in enumerate(sorted(array.layers))}
        return [AttentionSpec(
            resource_key=key, nodes=array.nodes,
            config=AttentionConfig(
                kv_cache=array.name,
                sliding_window=read.mask.size if windowed else None,
                layers=tuple(kv_layer[layer] for layer in sorted(read.layers or array.layers)) if windowed else None,
            ),
        )]

    def steps(self, key: str, read: Read, segments, steps: dict[str, Any]) -> None:
        steps[key] = AttentionStep(segments=segments, causal=not isinstance(read.mask, Bidirectional))

    def set_kernel_settings(self, array: Array, spec: NodeResourceSpec | None, settings: Mapping[str, Any]) -> None:
        """The decode keys go through the spec's own YAML path and its checks, as they do under `resources:`."""
        decode = {field: settings[field] for field in settings.keys() & {"decode_tensor_cores", "decode_fmha"}}
        super().set_kernel_settings(array, spec, {field: settings[field] for field in settings.keys() - decode.keys()})
        if decode:
            spec.apply_yaml_overrides(**decode)


class PagedCrossAttend(PagedAttend):
    """`Attend` with ``queries`` on a paged array: a `CrossAttentionSpec` over the array's pool, planned by the queried
    array's segments."""

    kernel_settings = frozenset({"decode_tensor_cores"})

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Bidirectional):
            raise _no_layout(array, f"reads[{key!r}].mask must be Bidirectional, the mask CrossAttentionConfig has")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since CrossAttentionConfig has no walk")
        if len(array.labels) != 1:
            raise _no_layout(array, f"a cross read's array declares one label, CrossAttentionConfig's one "
                                    f"context_label, and it has {sorted(array.labels)}")

    def specs(self, array: Array, key: str, read: Read) -> list[NodeResourceSpec]:
        return [CrossAttentionSpec(
            resource_key=key, nodes=array.nodes,
            config=CrossAttentionConfig(
                kv_cache=array.name, query_kv_cache=read.queries, context_label=min(array.labels),
            ),
        )]

    def steps(self, key: str, read: Read, segments, steps: dict[str, Any]) -> None:
        steps[key] = AttentionStep(segments=segments, causal=False)


class WithinStepAttend(OpBinding, ABC):
    """`Attend` within a step, over the step's own adds."""

    # whether the read runs a plan its step builds, so its resource has a spec and its calls need the step open
    plans: ClassVar[bool]

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Block):
            raise _no_layout(array, f"reads[{key!r}].mask must be Block, the only mask a read within a step takes")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since a read within a step has no walk")

    @abstractmethod
    def attend(self, run: Any, label: str) -> Callable[..., torch.Tensor]:
        """The attention over rows of one length that the read's exchange runs, where ``run`` is what `bind` gave."""


class RaggedAttend(WithinStepAttend):
    """`Attend` within a step, on a ragged resource that the reads of one kind, nodes and config share."""

    kernel_settings = frozenset({"max_segments_per_request", "max_segments_per_step", "compile_op", "workspace_mb"})
    plans = True

    def attend(self, run: Any, label: str) -> Callable[..., torch.Tensor]:
        return functools.partial(joint_attention, ragged=functools.partial(run.run, label=label))


class SdpaAttend(WithinStepAttend):
    """`Attend` within a step through SDPA, the reference pipelines' kernel: no spec, no step and no kernel setting."""

    plans = False

    def check(self, array: Array, key: str, read: Read) -> None:
        super().check(array, key, read)
        if read.packed:
            raise _no_layout(array, f"reads[{key!r}] runs SDPA, which can't tell one segment of packed queries from "
                                    "the next, and a read of rows of one length declares packed=False")

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        return sdpa_attention

    def attend(self, run: Any, label: str) -> Callable[..., torch.Tensor]:
        return run


class LinearAttnBinding(OpBinding):
    """A recurrence a `LinearAttnSpec` runs under the op's key. Today's resources run the array's `CausalConv` inside
    it too."""

    kernel_settings = frozenset({"splits"})

    def steps(self, key: str, op: Op, segments, steps: dict[str, Any]) -> None:
        steps[key] = LinearAttnStep(segments=segments)

    @staticmethod
    def kinds() -> str:
        """The op types bound as a linear attention update, for a refusal to name."""
        return ", ".join(sorted(kind.__name__ for (_, kind), binding in BINDINGS.items()
                                if isinstance(binding, LinearAttnBinding)))


class GDN(LinearAttnBinding):
    """`GatedDeltaNet` on a Value. It stores ``state`` K-last, (h_v, d_v, d_k) where the reference declares
    (h_v, d_k, d_v), since FlashInfer's kernels read it that way."""

    def check(self, array: Array, key: str, op: GatedDeltaNet) -> None:
        if set(array.entry.tensors) != {"state", "conv"}:
            raise _no_layout(array, "entry must be exactly state and conv, the blocks DeltaNetGeometry reads")
        if array.entry.tensors["state"].dtype is None:
            raise ValueError(f"Array {array.name!r}.entry.tensors['state'].dtype must be declared, since the rule "
                             f"that picks its stored dtype reads it")

    def specs(self, array: Array, key: str, op: GatedDeltaNet) -> list[NodeResourceSpec]:
        return [LinearAttnSpec(
            resource_key=key, nodes=array.nodes,
            config=LinearAttnConfig(
                recurrent_state=array.name, variant=LinearAttnVariant.GDN, qk_l2norm=op.qk_l2norm, sm_scale=op.scale,
            ),
        )]

    def block(self, tensor: TensorType) -> RecurrentBlockConfig:
        """Stored in fp32 unless the decode kernel takes the declared dtype, as it does for bf16 only at K = V = 128."""
        h_v, d_k, d_v = tensor.shape
        dtype = torch.bfloat16 if tensor.dtype is torch.bfloat16 and d_k == d_v == 128 else torch.float32
        # the head axis stays first, and the last two swap
        shard = None if tensor.shard is None else (0, 2, 1)[tensor.shard]
        return _block(TensorType((h_v, d_v, d_k), dtype=dtype, shard=shard))

    def stored(self, value: torch.Tensor) -> torch.Tensor:
        """The swap of the last two axes, which `from_stored` swaps back."""
        return value.transpose(-1, -2)

    def from_stored(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.transpose(-1, -2)


class ConvOnLinearAttn(OpBinding):
    """`CausalConv` on a Value. Today's resources run it inside the array's one linear attention update, so its key
    maps to that update's resource and has no spec or step of its own."""

    def check(self, array: Array, key: str, op: CausalConv) -> None:
        self._linear_key(array, key)
        if op.activation != "silu":
            raise _no_layout(array, f"updates[{key!r}].activation is {op.activation!r}, and LinearAttnConfig has no "
                                    "activation but silu")

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        return resources[self._linear_key(array, key)]

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


class PagedArrayHandle(ArrayHandle):
    """Adds and reads through the KV pages of an array that keeps entries. Each read runs on the resource
    ``resource_of`` gives it."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...], *,
        resource_of: Mapping[str, str] = EMPTY,
    ):
        super().__init__(array, resources, positions_key)
        self._bindings = bindings
        resource = {bound.key: resource_of.get(bound.key, bound.key) for bound in bindings}
        self.reads = {bound.key: bound.binding.bind(array, resource[bound.key], resources) for bound in bindings}
        self.first_read = next(iter(self.reads.values()))
        self._attn = {
            layer: self.reads[key]
            for key, read in array.reads.items() for layer in read.layers or array.layers
        }
        self._kv = resources[array.name]
        self._views = {layer: self._kv.layer_view(i) for layer, i in self._pool_layer.items()}
        # each op with its resource, built once since `state_steps` runs on every step, replays too
        self._ops = [(resource[bound.key], bound) for bound in bindings]
        # whether `check_step` needs an add, for a step that lists a row on the pages at span above 0
        self._grows = False

    def begin_step(self, step) -> None:
        super().begin_step(step)
        # the one plan `PagedStorage.steps` names for a step over several labels
        combined = getattr(step.steps.get(self.name), "combined_labels", None)
        if combined:
            self.label = next(iter(combined.values()))

    def check_step(self, step) -> None:
        super().check_step(step)
        if not self._keep or step.ctx.replays:
            return
        if self._held is not None:
            raise self._refusal(f"the step ends with layer {self._held_layer}'s add unread, on a kernel that takes k "
                                "and v at the read, so no page holds it")
        if self._grows and not self._kept:
            raise self._refusal("a step that keeps lists rows at span above 0 and adds nothing, so the pages "
                                "would keep entries no add wrote; list those rows at span 0")

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
        # a capture opens the handles once around `preprocess`, where `eager_add` adds run, and again around the forward
        self._grows = not (declared.eager_add and ctx.capture) and any(
            segment.span and (segment.request_id, segment.label) not in declared.placeholders for segment in segments
        )
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

    def _holds(self, layer: int) -> bool:
        """Whether this layer's add waits for its read: the kernel takes k and v there rather than from the pages."""
        attn = self._attn.get(layer)
        return attn is not None and not attn.requires_kv_write

    @torch.compiler.disable
    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        layer = self._layer_id(layer)
        self._record_add(layer, keep)
        if self._held is not None and self._held_layer != layer:
            raise self._refusal(f"an add on layer {layer} while layer {self._held_layer}'s add waits for its read, on "
                                "a kernel that takes k and v at the read")
        if self._holds(layer):
            self._held, self._held_layer = (fields["k"], fields["v"]), layer
            return
        self._kv.write_kv(fields["k"], fields["v"], layer_idx=self._pool_layer[layer], label=self.label)

    @torch.compiler.disable
    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        if q is None:
            raise self._no_layout("a read with no op")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        layer = self._layer_id(layer)
        self._record_read(layer)
        return self._run(layer, q)

    def _check_add_read(self, call: str) -> None:
        """Nothing: a paged add and read run outside the graph already."""

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        k, v = self._take_hold(layer) if self._holds(layer) else (None, None)
        return self._attn[layer].run(
            q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=self._pool_layer[layer],
        )

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
        if label is not None or len(step_labels(segments)) > 1:
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

    def _kernel_setting(self, read: str, field: str) -> Any:
        return getattr(self.reads[read], field)


class PagedByWalkHandle(PagedArrayHandle):
    """A paged array whose reads differ by walk: a step plans, reads through and selects by the reads of its walk,
    which ``walk_reads`` gives."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...], *,
        resource_of: Mapping[str, str], walk_reads: "WalkReads",
    ):
        super().__init__(array, resources, positions_key, bindings, resource_of=resource_of)
        self.walk_reads = walk_reads
        by_key = {bound.key: bound for bound in bindings}
        # each walk's ops with their resources, under None for a walk no read names
        self._walk_ops = {
            walk: [(resource, by_key[key]) for resource, key in reads.items()]
            for walk, reads in walk_reads.reads.items()
        }
        # the layers some read covers on the step's walk `_walk`
        self._walk: str | None = None
        self._covered = walk_reads.covers[None]

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if not step.ctx.replays:
            self._walk = step.ctx.graph_walk
            self._covered = self.walk_reads.covers.get(self._walk, self.walk_reads.covers[None])

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        if layer not in self._covered:
            raise self._refusal(f"a read on layer {layer}, which no read covers on walk {self._walk!r}")
        return super()._run(layer, q)

    def _last_token_read(self) -> Any:
        # a walk plans only the resources of its own reads
        reads = self.walk_reads.reads.get(self._walk, self.walk_reads.reads[None])
        if not reads:
            raise self._refusal(f"at_last_token on walk {self._walk!r}, which runs no read of this array")
        return self.reads[next(iter(reads.values()))]

    def _check_capture(self, walks: Collection[str]) -> None:
        """FlashInfer fixes the mask when it plans, so every replay would run the capture walk's."""
        reads = self.walk_reads.reads
        for resource in {resource for by_resource in reads.values() for resource in by_resource}:
            keys = {reads.get(walk, reads[None]).get(resource) for walk in walks} - {None}
            if len({self.array.reads[key].mask for key in keys}) > 1:
                raise self._no_layout(f"a capture replayed on walks {sorted(walks)} runs reads {sorted(keys)} on "
                                      f"{resource!r} with different masks, and a capture plans one")

    def _step_ops(self, walk: str | None) -> Sequence[tuple[str, BoundOp]]:
        if walk is None:
            raise self._no_layout("a step with no walk, on an array whose reads differ by walk")
        return self._walk_ops.get(walk, self._walk_ops[None])


class PagedCrossHandle(PagedArrayHandle):
    """A paged array read with the queries of ``queried``, whose segments lay them out. Its adds go to the pages, and
    its read runs under the query side's label, which the cross resource keys its plans by."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...], *,
        queried: Array,
    ):
        super().__init__(array, resources, positions_key, bindings)
        self.queried = queried
        self._query_label = "main"

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if step.ctx.replays:
            return
        labels = step_labels(_queries_step(self, step).segments or step.segments or ())
        if len(labels) > 1:
            raise self._no_layout(f"a step over labels {sorted(labels)} on {self.queried.name!r}, whose segments lay "
                                  "out this read's queries")
        queries = _query_segments(self, step, self._segments)
        if queries:
            self._query_label = queries[0].label

    def _holds(self, layer: int) -> bool:
        """Never: a cross read attends the pages, and its resource has no ``requires_kv_write``."""
        return False

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        return self._attn[layer].run(q, label=self._query_label, kv_cache_layer=self._views[layer])

    def at_last_token(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
        self._check_open("at_last_token")
        raise self._refusal(f"at_last_token lays x out by this array's segments, and its read's queries are laid out "
                            f"by {self.queried.name!r}; call it on that array")


class ValueHandle(ArrayHandle):
    """The calls on a Value: its updates, which run in place on one layer's block, and reads and writes of whole
    fields."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...],
    ):
        super().__init__(array, resources, positions_key)
        self._bindings = bindings
        self._pool = resources[array.name]
        self._blocks = {
            field: {layer: self._pool.block(field, i) for layer, i in self._pool_layer.items()}
            for field in array.entry.tensors
        }
        self._runners = {type(bound.declared): bound.binding.bind(array, bound.key, resources) for bound in bindings}
        self._layouts = _field_layouts(bindings)
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

    def check_step(self, step) -> None:
        super().check_step(step)
        if self._step_written and not step.ctx.replays:
            self._check_written(self._segments)

    def _check_written(self, segments) -> None:
        rows = [segment.request_id for segment in segments if segment.span >= 1]
        if not rows:
            return
        for layer in sorted(self.array.layers):
            unwritten = sorted(self._step_written - self._written.get(layer, set()))
            if unwritten:
                raise self._refusal(f"layer {layer}'s fields {unwritten} are unwritten in a kept step for rows {rows}, "
                                    "and no update op writes them")

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
        or that pads the step reads the start `set_start` gave, zeros for a field it doesn't name."""
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
        layout = self._layouts.get(field)
        expected = block.shape[1:] if layout is None else layout.from_stored(block).shape[1:]
        dtype = self.array.entry.tensors[field].dtype or block.dtype
        if tuple(shape) != tuple(expected) or value.dtype != dtype:
            raise self._refusal(f"{what} of field {field!r} on layer {layer} is {tuple(shape)} of {value.dtype}, and "
                                f"the field is {tuple(expected)} of {dtype} on this rank")
        return (value if layout is None else layout.stored(value)).to(block.dtype)

    def _declared(self, field: str, stored: torch.Tensor) -> torch.Tensor:
        """``stored``, read from the pool, in ``field``'s declared layout and dtype."""
        layout = self._layouts.get(field)
        value = stored if layout is None else layout.from_stored(stored)
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

    def _update(self, kind: type[Op], field: str, layer: int) -> tuple[Any, torch.Tensor]:
        runner = self._runners.get(kind)
        if runner is None:
            raise self._refusal(f"an update of type {kind.__name__}, which the array doesn't declare")
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
            where = f"a host write on row {row}, label {label!r}, layer {layer}"
            if not ctx.is_real_row(row):
                raise self._refusal(f"{where}, and that row isn't one of the step's requests")
            if label not in self.array.labels or layer not in self.array.layers:
                raise self._refusal(f"{where}, and the array declares labels {sorted(self.array.labels)} and layers "
                                    f"{sorted(self.array.layers)}")
            if not self.host_fields.is_open(row):
                raise self._refusal(f"{where}, and that request has no open fields")
            for field, value in written.items():
                kind = self.array.entry.host.get(field)
                if kind is None:
                    raise self._refusal(f"{where}: field {field!r} isn't declared, of {sorted(self.array.entry.host)}")
                if not is_host_value(value, kind):
                    raise self._refusal(f"{where}: field {field!r} takes {kind}, and got {value!r}")

    def commit_step(self, step) -> None:
        self.host_fields.commit(self.array, step.state[self.name], step.ctx.request_ids)

    def _host(self, request: int, label: str, layer: int) -> Mapping[str, object]:
        fields = self.host_fields.fields(request, self.name, label, layer)
        if fields is None:
            raise self._refusal(f"no host fields for request {request}, label {label!r}, layer {layer}")
        return fields


class HostValueHandle(_HostCalls, ArrayHandle):
    """The calls on a Value of host fields only."""

    def __init__(self, array: Array, host_fields: HostFields):
        super().__init__(array, EMPTY, None)
        self._bindings: tuple[BoundOp, ...] = ()
        self.host_fields = host_fields
        host_fields.declare(array)

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        super()._check_storage(declared, segments, ctx)


class ValueWithHostHandle(_HostCalls, ValueHandle):
    """The calls on a Value with host fields beside its tensor fields."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...],
        host_fields: HostFields,
    ):
        super().__init__(array, resources, positions_key, bindings)
        self.host_fields = host_fields
        host_fields.declare(array)


class WithinStepHandle(ArrayHandle):
    """The calls on an array with ``max_len`` 0, which a block compiled alone traces. With ``compile_op`` set, the read
    stays in that graph through ``mstar::ragged_attention``. A check here runs only while not compiling: it would guard
    on the layer, and the block would retrace per layer.

    ``queried`` is the array whose segments lay out the queries of a cross read. Under sequence parallelism the two
    flags pick the exchange: Ulysses when both sides are sharded, none when only the queries are, and a merge of the
    ranks' partial attentions when only the keys are. The read's binding supplies the attention each exchange runs.
    """

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...], *,
        resource_key: str | None = None, queried: Array | None = None, sp_group: CommGroup | None = None,
    ):
        super().__init__(array, resources, positions_key)
        self._bindings = bindings
        ((key, read, self._binding),) = bindings
        self.resource_key = resource_key or key
        self.reads = {key: self._binding.bind(array, self.resource_key, resources)}
        self.first_read = self.reads[key]
        self.queried = queried
        self._packed = read.packed
        # a plain attribute, which a compiled caller's trace reads as a constant
        self._plans = self._binding.plans
        self._sp = sp_group
        sharded = sp_group is not None and sp_group.world_size > 1
        self._keys_sharded = sharded and array.seq_sharded
        self._queries_sharded = sharded and (array if queried is None else queried).seq_sharded
        # A read on SDPA plans nothing, so a step needs to run for it only to learn a Ulysses exchange's rank sizes.
        self.needs_step = self._plans or (self._keys_sharded and self._queries_sharded)
        self._seq_sizes: list[int] | None = None
        # Both sides fix the plan's label, so it is one string for the life of the handle. See `cross_label`.
        self._plan_label = min(array.labels) if queried is None else cross_label(min(queried.labels), min(array.labels))
        # One callable per array, not per layer: Dynamo specializes `ulysses_attention` on the identity of the callable
        # it runs, so a callable per layer would retrace that frame per layer.
        self._attend = self._binding.attend(self.first_read, self._plan_label)

    def begin_step(self, step) -> None:
        super().begin_step(step)
        if step.ctx.replays:
            return
        if self.queried is not None:
            _query_segments(self, step, self._segments)
        if self._keys_sharded and self._queries_sharded:
            spans = {s.span for s in self._segments}
            if len(spans) > 1:
                raise self._no_layout(f"a Ulysses exchange over spans {sorted(spans)}, and mixed lengths need a "
                                      "permutation in the unpack")
            self._seq_sizes = sp_seq_split(spans.pop(), self._sp.world_size) if spans else None

    def check_step(self, step) -> None:
        """Every add is scratch, so a step keeps none and has nothing to refuse as a whole."""

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        super()._check_storage(declared, segments, ctx)
        if declared.forks:
            raise self._refusal("a step forks an array with max_len 0, which keeps nothing to fork")

    def query_share(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        """The query layout this read's plan sees of ``segments``, the query array's in this step."""
        return self._share(segments) if self._queries_sharded and not self._keys_sharded else tuple(segments)

    def key_share(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        """The key layout this read's plan sees of ``segments``, this array's in this step."""
        return self._share(segments) if self._keys_sharded and not self._queries_sharded else tuple(segments)

    def _share(self, segments: Sequence[Segment]) -> tuple[Segment, ...]:
        world_size, rank = self._sp.world_size, self._sp.rank
        return tuple(s._replace(span=sp_seq_split(s.span, world_size)[rank]) for s in segments)

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
        if self._keys_sharded and self._queries_sharded:
            return ulysses_attention(self._sp, q, k, v, self._attend, self._seq_sizes, seq_dim=1, head_dim=2)
        if self._keys_sharded:
            return attention_over_sharded_keys(self._sp, q, k, v, self._attend)
        return self._attend(q, k, v)

    def _check_add_read(self, call: str) -> None:
        raise self._refusal(f"{call} with max_len 0, whose read a compiled block traces: a break here takes the read "
                            "out of that graph")

    def _kernel_setting(self, read: str, field: str) -> Any:
        if not self._binding.kernel_settings:
            raise self._refusal(f"no {field} for read {read!r}, whose binding {type(self._binding).__name__} takes no "
                                "kernel settings")
        return getattr(self.reads[read], field)


# Storages


class WalkReads(NamedTuple):
    """The reads of an array whose reads differ by walk: for each walk, the read it runs on each resource and the
    layers those reads cover, under None for a walk no read names. A combined walk's rows read by their own walks,
    which agree on each mask, and its layers are those every one of them covers."""

    reads: dict[str | None, dict[str, str]]
    covers: dict[str | None, frozenset[int]]


class Resolved(NamedTuple):
    """Arrays resolved into today's resources: each array's specs, its ops' included, by its name, each read's resource
    key, the array that lays out the queries of each read with ``queries``, and the `WalkReads` of each array whose
    reads differ by walk."""

    specs: dict[str, list[NodeResourceSpec]]
    resource_of: dict[str, str]
    queried: dict[str, Array]
    walk_reads: Mapping[str, WalkReads] = EMPTY


class Storage(ABC):
    """How today's resources hold one kind of array: its specs, its own step, its layouts and its handle."""

    handle_class: ClassVar[type[ArrayHandle]]
    # the pool sizes `kernel_settings` may set under the array's name, until the engine sizes its pools itself
    kernel_settings: ClassVar[frozenset[str]] = frozenset()

    def __init__(self):
        # `state_steps` reaches a handle's storage, for its step builder, through the handle's class
        self.handle_class.storage = self

    @abstractmethod
    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        """The specs and read keys of ``arrays``, which are every declared array this storage holds, where ``declared``
        are all the declared arrays."""

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
    ) -> ArrayHandle:
        """``array``'s handle over a node's built ``resources``."""
        return self.handle_class(array, resources, _positions_key(array), _bindings(self, array, setup.by_read))

    def steps(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        """Add the step of the array's own resource, under its name, for ``declared`` in ``step``."""
        return

    def step_builder(self) -> StepBuilder:
        """What gathers the steps of this storage's arrays in one step."""
        return ArraySteps(self)

    def layouts(
        self, by_label: Mapping[str, Layout], keyed_spans: Mapping[str, int],
    ) -> "dict[str, LabelLayout] | None":
        """Today's `LabelLayout` per label, or None where the array's pool takes no layout."""
        return None

    def set_kernel_settings(self, array: Array, spec: NodeResourceSpec | None, settings: Mapping[str, Any]) -> None:
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

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        resolved = Resolved({}, {}, {})
        for array in arrays:
            reads, resource_of = self._read_specs(array, _bindings(self, array, setup.by_read))
            resolved.specs[array.name] = [self._pool(array, setup.walks), *reads]
            resolved.resource_of.update(resource_of)
        return resolved

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
    ) -> ArrayHandle:
        return self.handle_class(
            array, resources, _positions_key(array), _bindings(self, array, setup.by_read),
            resource_of=resolved.resource_of,
        )

    def _pool(self, array: Array, walks: "Walks") -> KVSpec:
        heads, head_dim = _kv_shape(array)
        if array.seq_sharded:
            raise _no_layout(array, "seq_sharded takes max_len 0, and a paged pool plans whole segments")
        if not array.reads:
            raise _no_layout(array, "reads is empty, and PagedKVConfig needs num_qo_heads from an Attend read")
        joined = sorted(label for label in array.labels if "+" in label)
        if joined:
            raise ValueError(f"Array {array.name!r}.labels: {joined} hold a '+', which joins the labels of a step over "
                             "several labels into the name of its plan")
        q_heads = {read.op.q_heads for read in array.reads.values()}
        if len(q_heads) > 1:
            raise _no_layout(array, f"reads have q_heads {sorted(q_heads)}, and PagedKVConfig has one num_qo_heads")
        return KVSpec(
            resource_key=array.name, nodes=array.nodes, leader=_leader(array, walks),
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
            for layer in sorted(read.layers or array.layers):
                other = reader.setdefault(layer, key)
                if resource_of[other] != resource_of[key]:
                    raise _no_layout(array, f"reads[{other!r}] and reads[{key!r}] build different specs and both cover "
                                            f"layer {layer}, which reads through one resource on every walk")
        return list(specs.values()), resource_of

    def steps(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        pre, post, rows = _forks(handle.array, declared)
        ordered = declared.segments or step.segments
        # label-major, so a step over several labels starts and ends on different ones
        labels = step_labels(ordered) if ordered[0].label != ordered[-1].label else ()
        steps[handle.name] = KVStep(
            segments=declared.segments, commit=declared.keep, pre_forks=pre, post_forks=post, fork_rids=rows,
            sink=declared.placeholders, eager_write=declared.eager_add,
            combined_labels={labels: "+".join(labels)} if labels else {},
        )

    def layouts(self, by_label: Mapping[str, Layout], keyed_spans: Mapping[str, int]) -> "dict[str, LabelLayout]":
        from mstar.model.base import LabelLayout

        return {
            label: LabelLayout(
                spans=layout.spans, growth=layout.growth, scratch=layout.scratch, keyed_spans=keyed_spans.get(label),
            )
            for label, layout in by_label.items()
        }


def _leader(array: Array, walks: "Walks") -> str | None:
    """The node of ``array``'s pool that admits each request: the one node that runs every walk any of its nodes runs.
    None on one node, and where no node or several do, so each node admits for itself."""
    if not walks.nodes or len(array.nodes) == 1:
        return None
    running = [nodes for nodes in walks.nodes.values() if not array.nodes.isdisjoint(nodes)]
    leaders = [node for node in sorted(array.nodes) if all(node in nodes for nodes in running)]
    return leaders[0] if len(leaders) == 1 else None


def _share_a_walk(a: Read, b: Read) -> bool:
    return a.walks is None or b.walks is None or not a.walks.isdisjoint(b.walks)


def _walk_reads(array: Array, resource_of: Mapping[str, str], combined: Mapping[str, Collection[str]]) -> WalkReads:
    reads: dict[str | None, dict[str, str]] = {}
    covers: dict[str | None, frozenset[int]] = {}
    for walk in (None, *{walk for read in array.reads.values() for walk in read.walks or ()}):
        covering = [key for key, read in array.reads.items() if read.walks is None or walk in read.walks]
        reads[walk] = {resource_of[key]: key for key in covering}
        covers[walk] = frozenset().union(*(array.reads[key].layers or array.layers for key in covering))
    for walk, members in combined.items():
        reads[walk] = {
            resource: key for member in sorted(members) for resource, key in reads.get(member, reads[None]).items()
        }
        covers[walk] = frozenset.intersection(*(covers.get(member, covers[None]) for member in members))
    return WalkReads(reads, covers)


class PagedByWalkStorage(PagedStorage):
    """An append-only array that keeps entries, whose reads differ by walk. Each walk runs its own reads, a combined
    walk those of the walks it batches."""

    handle_class = PagedByWalkHandle

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        resolved = super().resolve(arrays, declared, setup)
        return resolved._replace(walk_reads={
            array.name: _walk_reads(array, resolved.resource_of, combined_walks(array, setup.walks.combined))
            for array in arrays
        })

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
    ) -> ArrayHandle:
        return PagedByWalkHandle(
            array, resources, _positions_key(array), _bindings(self, array, setup.by_read),
            resource_of=resolved.resource_of, walk_reads=resolved.walk_reads[array.name],
        )


class PagedCrossStorage(PagedStorage):
    """An append-only array that keeps entries, read with another array's queries: a cross resource over its pages,
    which plans the queries by the queried array's segments in each step. Its handle holds one queried array, so for
    now the array takes no other read."""

    handle_class = PagedCrossHandle

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        for array in arrays:
            if len(array.reads) != 1:
                raise _no_layout(array, f"reads {sorted(array.reads)}, and a cross read's array takes no other read "
                                        "yet; declare each other read on an array of its own")
        by_name = {array.name: array for array in declared}
        resolved = super().resolve(arrays, declared, setup)
        for array in arrays:
            ((key, read),) = array.reads.items()
            queried = by_name[read.queries]
            if not isinstance(storage_of(queried), PagedStorage):
                raise _no_layout(array, f"reads[{key!r}] keeps entries and reads the queries of {read.queries!r}, "
                                        "which isn't an append-only array that keeps entries")
            resolved.queried[key] = queried
        return resolved

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
    ) -> ArrayHandle:
        (key,) = array.reads
        return PagedCrossHandle(
            array, resources, _positions_key(array), _bindings(self, array, setup.by_read),
            queried=resolved.queried[key],
        )

    def steps(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        _queries_step(handle, step)
        super().steps(handle, declared, step, steps)


class ValueStorage(Storage):
    """A rewritable array of one entry, in a recurrent state pool."""

    handle_class = ValueHandle
    kernel_settings = frozenset({"max_slots"})

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        return Resolved({array.name: self._specs(array, declared) for array in arrays}, {}, {})

    def _specs(self, array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
        if array.reads:
            raise _no_layout(array, "reads must be empty, since RecurrentStateConfig has no read")
        if array.seq_sharded:
            raise _no_layout(array, "seq_sharded takes max_len 0, and a Value's entries stay whole on each rank")
        bound = _bindings(self, array)
        kv_cache = None
        if bound:
            # the snapshots its update kernels take are keyed by the pages of the one paged array on its nodes
            holders = [
                other for other in arrays if isinstance(storage_of(other), PagedStorage) and other.nodes & array.nodes
            ]
            if len(holders) != 1 or not array.nodes <= holders[0].nodes:
                raise _no_layout(array, f"its nodes must hold one append-only array, the one "
                                        f"RecurrentStateConfig.kv_cache links to, and they hold "
                                        f"{[other.name for other in holders]}")
            kv_cache = holders[0].name
        layouts = _field_layouts(bound)
        pool = RecurrentStateSpec(
            resource_key=array.name, nodes=array.nodes,
            config=RecurrentStateConfig(
                num_layers=len(array.layers),
                blocks={
                    field: layouts[field].block(tensor) if field in layouts else _block(tensor)
                    for field, tensor in array.entry.tensors.items()
                },
                kv_cache=kv_cache,
                labels=tuple(sorted(array.labels)),
            ),
        )
        return [pool, *_op_specs(array, bound)]

    def steps(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        pre, post, rows = _forks(handle.array, declared)
        steps[handle.name] = RecurrentStep(segments=declared.segments, pre_forks=pre, post_forks=post, fork_rids=rows)


class ValueWithHostStorage(ValueStorage):
    """A Value with host fields beside its tensor fields: the tensors in a recurrent state pool, which has no block
    for a host field, and the host fields where the engine holds them."""

    handle_class = ValueWithHostHandle

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
    ) -> ArrayHandle:
        return ValueWithHostHandle(
            array, resources, _positions_key(array), _bindings(self, array, setup.by_read), setup.host_fields,
        )


class HostValueStorage(Storage):
    """A Value of host fields only, which the engine holds, so it has no spec, no step and no layout."""

    handle_class = HostValueHandle

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        for array in arrays:
            if array.reads or array.updates:
                raise _no_layout(array, "a Value of host fields only takes no read or update")
        return Resolved({array.name: [] for array in arrays}, {}, {})

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
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
        sharded_nodes = {node for array in declared if array.seq_sharded for node in array.nodes}
        by_name = {array.name: array for array in declared}
        resolved = Resolved({}, {}, {})
        shared: dict[tuple, NodeResourceSpec] = {}
        for array in arrays:
            bound, heads, head_dim, dtype = _within_step_read(array, setup)
            key, read = bound.key, bound.declared
            queried = by_name.get(read.queries, array)
            if read.queries is not None:
                if len(queried.labels) != 1:
                    raise _no_layout(array, f"reads[{key!r}] reads the queries of {read.queries!r}, which has labels "
                                            f"{sorted(queried.labels)}, and a max_len 0 read plans the queries of one "
                                            "label")
                resolved.queried[key] = queried
            config = RaggedAttentionConfig(
                num_qo_heads=read.op.q_heads, num_kv_heads=heads, head_dim=head_dim, dtype=dtype,
                # the heads narrow by SP only where the exchange runs over them
                shard_over_sp=not array.nodes & sharded_nodes or (array.seq_sharded and queried.seq_sharded),
            )
            spec_class = RaggedAttentionSpec if read.queries is None else RaggedCrossAttentionSpec
            spec = spec_class(resource_key=key, nodes=array.nodes, config=config)
            spec = shared.setdefault((spec_class, array.nodes, astuple(config)), spec)
            resolved.resource_of[key] = spec.resource_key
            resolved.specs[array.name] = [spec] if bound.binding.plans and spec.resource_key == key else []
        return resolved

    def handle(
        self, array: Array, resources: Mapping[str, Any], setup: "StateSetup", resolved: Resolved,
        sp_group: CommGroup | None,
    ) -> ArrayHandle:
        (key,) = array.reads
        return WithinStepHandle(
            array, resources, None, _bindings(self, array, setup.by_read),
            resource_key=resolved.resource_of.get(key, key), queried=resolved.queried.get(key), sp_group=sp_group,
        )


PAGED = PagedStorage()
PAGED_BY_WALK = PagedByWalkStorage()
PAGED_CROSS = PagedCrossStorage()
VALUE = ValueStorage()
VALUE_WITH_HOST = ValueWithHostStorage()
HOST_VALUE = HostValueStorage()
WITHIN_STEP = WithinStepStorage()


def storage_of(array: Array) -> Storage:
    """The storage today's resources keep ``array`` in."""
    if is_value(array):
        if not array.entry.tensors:
            return HOST_VALUE
        return VALUE_WITH_HOST if array.entry.host else VALUE
    if array.rewritable:
        raise _no_layout(array, "a rewritable array has no spec unless it is a Value, of max_len 1")
    if not array.max_len:
        return WITHIN_STEP
    if any(read.queries is not None for read in array.reads.values()):
        return PAGED_CROSS
    return PAGED_BY_WALK if reads_differ_by_walk(array) else PAGED


# `Attend` is an `AttentionSpec` on a paged array, a `CrossAttentionSpec` on one read with another array's queries and a
# `RaggedAttentionSpec` within a step, so a binding is per storage and op type.
BINDINGS: dict[tuple[Storage, type[Op]], OpBinding] = {
    (PAGED, Attend): PagedAttend(),
    (PAGED_BY_WALK, Attend): PagedAttend(),
    (PAGED_CROSS, Attend): PagedCrossAttend(),
    (WITHIN_STEP, Attend): RaggedAttend(),
    (VALUE, GatedDeltaNet): GDN(),
    (VALUE, CausalConv): ConvOnLinearAttn(),
}
# What `backend: sdpa` binds in place of the default, for each storage and op type SDPA can run.
SDPA: dict[tuple[Storage, type[Op]], OpBinding] = {(WITHIN_STEP, Attend): SdpaAttend()}


class Walks(NamedTuple):
    """The model's walks, where a storage needs them: the nodes each walk runs, from which a pool on several nodes takes
    its leader, and each node's combined walks with the walks each batches, through which a read that differs by walk
    finds the read of a combined walk."""

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
    """``array``'s reads, then its updates, each with the binding that runs it on ``storage``: ``chosen``'s for a read
    it names, and `BINDINGS`' for the rest. Each binding checks its op. Refuses a pair `BINDINGS` has no entry for, and
    a second update of one type."""
    bound = []
    for key, read in array.reads.items():
        binding = BINDINGS.get((storage, type(read.op)))
        if binding is None:
            raise _no_binding(storage, array, f"reads[{key!r}].op")
        bound.append(BoundOp(key, read, chosen.get(key, binding)))
    unbound = sorted(key for key, op in array.updates.items() if (storage, type(op)) not in BINDINGS)
    if unbound:
        raise _no_binding(storage, array, f"updates {unbound}")
    by_type: dict[type[Op], str] = {}
    for key, op in array.updates.items():
        other = by_type.setdefault(type(op), key)
        if other != key:
            raise _no_layout(array, f"updates[{other!r}] and updates[{key!r}] are both {type(op).__name__}, and an "
                                    "update call finds its op by type")
        bound.append(BoundOp(key, op, BINDINGS[storage, type(op)]))
    for each in bound:
        each.binding.check(array, each.key, each.declared)
    return tuple(bound)


def _no_binding(storage: Storage, array: Array, where: str) -> ValueError:
    kinds = " or ".join(sorted(kind.__name__ for held, kind in BINDINGS if held is storage))
    if not kinds:
        return _no_layout(array, f"{where}, and today's resources run no op on this array")
    return _no_layout(array, f"{where} must be {kinds}, the ops today's resources run on this array")


def _op_specs(array: Array, bound: Sequence[BoundOp]) -> list[NodeResourceSpec]:
    return [spec for each in bound for spec in each.binding.specs(array, each.key, each.declared)]


def _field_layouts(bound: Sequence[BoundOp]) -> dict[str, OpBinding]:
    """Each field an update writes, with that update's binding, which lays the field out in the pool."""
    return {field: each.binding for each in bound for field in each.declared.writes}
