"""The glue between declarations and today's resources, as one layer: a storage per kind of array, a binding per
storage and op type, and a handle per storage. It builds the specs, steps and layouts everything downstream reads."""

import dataclasses
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, replace
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
from mstar.engine.state.analysis import position_key
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
    specs = build_specs(model.declare_state(), kernel_settings=model.kernel_settings(), setup=setup)
    return specs + list(model.get_node_resources() or [])


def build_specs(
    declarations: Sequence[Array | Sampler], *, kernel_settings: Mapping[str, Mapping[str, Any]] = EMPTY,
    setup: "StateSetup | None" = None,
) -> list[NodeResourceSpec]:
    """Each array's specs, from its storage and its ops' bindings, under the names the declarations give, then a
    `PositionSpec` for each array that declares positions and a `SamplerSpec` per sampler. A read that ``setup`` binds
    takes that binding.

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
    for storage, held in by_storage.items():
        each = storage.resolve(held, arrays, setup)
        specs |= each.specs
        resource_of |= each.resource_of
        queried |= each.queried
    return Resolved(specs, resource_of, queried)


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
        enable_repetion_penalty=sampler.penalty, enable_min_p=sampler.min_p, min_tokens_stop_ids=sampler.eos_ids,
    )


def _apply_kernel_settings(
    specs: Sequence[NodeResourceSpec], arrays: Sequence[Array], resource_of: Mapping[str, str],
    kernel_settings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set the kernel choices, capture sizes and default pool sizes the contract leaves to the engine, and refuse any
    other, so this hook can't carry other physical choices until the engine picks kernels and sizes its pools itself."""
    owners = {array.name: (array, None) for array in arrays} | {
        bound.key: (array, bound.binding)
        for array in arrays for bound in _bindings(storage_of(array), array) if bound.binding.kernel_settings
    }
    by_key = {spec.resource_key: spec for spec in specs}
    for key, settings in kernel_settings.items():
        if key not in owners:
            raise ValueError(f"kernel_settings[{key!r}] names no read or update of the declarations that takes kernel "
                             f"settings, and no array, of {sorted(owners)}")
        array, binding = owners[key]
        # an update's settings go on its own spec, and a read on SDPA has no spec to set
        spec = by_key.get(resource_of.get(key, key))
        if binding is not None:
            allowed = binding.kernel_settings
        elif isinstance(spec, KVSpec):
            allowed = {"max_num_pages"}
        elif isinstance(spec, RecurrentStateSpec):
            allowed = {"max_slots"}
        else:
            allowed = set()    # a host-only Value or an array with max_len 0 has no pool
        refused = settings.keys() - allowed
        if refused:
            raise ValueError(f"Array {array.name!r}: kernel_settings[{key!r}] sets {sorted(refused)}; "
                             f"it may set only {sorted(allowed)}")
        if binding is not None:
            for field, value in settings.items():
                if spec is not None:
                    setattr(spec.config, field, value)
        elif settings:
            try:
                # the spec's own YAML path, so a value here means what the same key under `resources:` means
                spec.apply_yaml_overrides(**settings)
            except ValueError as e:
                raise ValueError(f"Array {key!r}: kernel_settings[{key!r}]: {e}") from e


def node_handles(
    model: "Model", node: str, resources: Mapping[str, Any], *, sp_group: CommGroup | None = None,
    setup: "StateSetup | None" = None,
) -> dict[str, ArrayHandle | SamplerHandle]:
    """``node``'s handles, by declared name, over its built ``resources``, each read bound as ``setup`` binds it."""
    setup = setup or StateSetup()
    declarations = model.declare_state()
    resolved = _resolve([declared for declared in declarations if isinstance(declared, Array)], setup)
    return {
        declared.name: build_handle(declared, resources, setup, resolved, sp_group)
        for declared in declarations if node in declared.nodes
    }


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


def state_steps(step: SubmoduleStep | None, handles: Mapping[str, ArrayHandle | SamplerHandle]) -> SubmoduleStep | None:
    """``step`` as a new step, with what its ``state`` declares expanded into today's step objects by each storage's
    builder, built afresh for each step because `SubmoduleStep` writes the segments onto them. None when no array or
    sampler it lists needs it, so the step skips its resources' admit, plan and commit."""
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
            builder.add(handle, declared, step)
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
    def add(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep) -> None:
        """``declared`` is what ``step`` lists for ``handle``'s array."""

    @abstractmethod
    def steps(self) -> dict[str, Any]:
        """The steps of the arrays added, by resource key."""


class ArraySteps(StepBuilder):
    """Each array's steps on its own: its storage's, its ops' and its positions'."""

    def __init__(self, storage: "Storage"):
        self._storage = storage
        self._steps: dict[str, Any] = {}

    def add(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep) -> None:
        segments = declared.segments
        self._storage.steps(handle, declared, step, self._steps)
        for bound in handle._bindings:
            bound.binding.steps(bound.key, bound.declared, segments, self._steps)
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

    def add(self, handle: "WithinStepHandle", declared: ArrayStep, step: SubmoduleStep) -> None:
        """Plan ``handle``'s read over the segments its kernel sees. A Ulysses read runs over whole segments after the
        exchange, and any other over this rank's share of a sharded side."""
        if not handle._plans:
            return
        key, name, segments = handle.resource_key, handle.name, declared.segments or step.segments
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
                steps[key] = AttentionStep(segments=segments, causal=False)
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

    def bind(self, array: Array, key: str, resources: Mapping[str, Any]) -> Any:
        """What a handle calls to run the op."""
        return resources[key]


class PagedAttend(OpBinding):
    """`Attend` on a paged array: an `AttentionSpec` over the array's pool."""

    kernel_settings = frozenset({"splits", "decode_fmha"})

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Causal | Window | Bidirectional):
            raise _no_layout(array, f"reads[{key!r}].mask must be Causal, Window or Bidirectional, the masks "
                                    "AttentionConfig has")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since AttentionConfig has no walk")

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


class PagedCrossAttend(OpBinding):
    """`Attend` with ``queries`` on a paged array: a `CrossAttentionSpec` over the array's pool, planned by the queried
    array's segments. It takes no kernel setting yet."""

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
        """The swap of the last two axes, which is its own inverse."""
        return value.transpose(-1, -2)


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
    """Adds and reads through the KV pages of an array that keeps entries."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...],
    ):
        super().__init__(array, resources, positions_key)
        self._bindings = bindings
        self.reads = {bound.key: bound.binding.bind(array, bound.key, resources) for bound in bindings}
        self.first_read = next(iter(self.reads.values()))
        self._attn = {
            layer: self.reads[key]
            for key, read in array.reads.items() for layer in read.layers or array.layers
        }
        self._kv = resources[array.name]
        self._views = {layer: self._kv.layer_view(i) for layer, i in self._pool_layer.items()}
        # whether `check_step` needs an add, for a step that lists a row on the pages at span above 0
        self._grows = False

    def check_step(self, step) -> None:
        super().check_step(step)
        if self._grows and not self._kept and self._keep and not step.ctx.replays:
            raise self._refusal("a step that keeps lists rows at span above 0 and adds nothing, so the pages "
                                "would keep entries no add wrote; list those rows at span 0")

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

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        k, v = self._take_hold(layer) if self._holds(layer) else (None, None)
        return self._attn[layer].run(
            q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=self._pool_layer[layer],
        )

    def at_last_token(self, x: torch.Tensor) -> torch.Tensor:
        self._check_open("at_last_token")
        return self.first_read.select_last_hidden(x, label=self.label)

    def _kernel_setting(self, read: str, field: str) -> Any:
        return getattr(self.reads[read], field)


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
        queries = _query_segments(self, step, step.state[self.name].segments or step.segments or ())
        if queries:
            self._query_label = queries[0].label

    def _holds(self, layer: int) -> bool:
        """Never: a cross read attends the pages, and its resource has no ``requires_kv_write``."""
        return False

    def _run(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        return self._attn[layer].run(q, label=self._query_label, kv_cache_layer=self._views[layer])

    def at_last_token(self, x: torch.Tensor) -> torch.Tensor:
        self._check_open("at_last_token")
        raise self._refusal(f"at_last_token lays x out by this array's segments, and its read's queries are laid out "
                            f"by {self.queried.name!r}; call it on that array")


class ValueHandle(ArrayHandle):
    """The calls on a Value: its updates, which run in place on one layer's block, and writes of whole fields."""

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

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        if not declared.keep:
            raise self._refusal("keep=False on an array with updates, whose kernels write in place, until saved "
                                "copies exist")
        super()._check_storage(declared, segments, ctx)

    def block(self, field: str, layer: int) -> torch.Tensor:
        blocks = self._blocks.get(field)
        if blocks is None or layer not in blocks:
            raise self._refusal(f"no field {field!r} on layer {layer}, of layers {sorted(self._pool_layer)}")
        return blocks[layer]

    @torch.compiler.disable
    def write(self, layer: int | torch.Tensor, *, at: torch.Tensor | None = None, **fields: torch.Tensor) -> None:
        """Overwrite each row's ``fields`` in place."""
        if at is not None:
            raise self._no_layout("write at positions, on anything but a Value")
        layer = self._layer_id(layer)
        self._check_open("write")
        for field, value in fields.items():
            self.block(field, layer)
            layout = self._layouts.get(field)
            stored = value if layout is None else layout.stored(value)
            self._pool.scatter_(field, stored, self._pool_layer[layer], label=self.label)

    def _update(self, kind: type[Op], field: str, layer: int) -> tuple[Any, torch.Tensor]:
        runner = self._runners.get(kind)
        if runner is None:
            raise self._refusal(f"an update of type {kind.__name__}, which the array doesn't declare")
        view = self.block(field, layer)
        self._record_update(field, layer)
        return runner, view


class HostValueHandle(ArrayHandle):
    """The calls on a Value of host fields only, which ``host_fields`` holds rather than a resource."""

    def __init__(self, array: Array, host_fields: HostFields):
        super().__init__(array, EMPTY, None)
        self._bindings: tuple[BoundOp, ...] = ()
        self.host_fields = host_fields
        host_fields.declare(array)

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        super()._check_storage(declared, segments, ctx)

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
        segments = step.state[self.name].segments or step.segments or ()
        if self.queried is not None:
            _query_segments(self, step, segments)
        if self._keys_sharded and self._queries_sharded:
            spans = {s.span for s in segments}
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

    def _kernel_setting(self, read: str, field: str) -> Any:
        if not self._binding.kernel_settings:
            raise self._refusal(f"no {field} for read {read!r}, whose binding {type(self._binding).__name__} takes no "
                                "kernel settings")
        return getattr(self.reads[read], field)


# Storages


class Resolved(NamedTuple):
    """Arrays resolved into today's resources: each array's specs, its ops' included, by its name, each read's resource
    key, and the array that lays out the queries of each read with ``queries``."""

    specs: dict[str, list[NodeResourceSpec]]
    resource_of: dict[str, str]
    queried: dict[str, Array]


class Storage(ABC):
    """How today's resources hold one kind of array: its specs, its own step, its layouts and its handle."""

    handle_class: ClassVar[type[ArrayHandle]]

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


class PagedStorage(Storage):
    """An append-only array that keeps entries, in a pool of KV pages."""

    handle_class = PagedArrayHandle

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        return Resolved(
            {array.name: self._specs(array, setup) for array in arrays},
            {key: key for array in arrays for key in array.reads}, {},
        )

    def _specs(self, array: Array, setup: "StateSetup") -> list[NodeResourceSpec]:
        heads, head_dim = _kv_shape(array)
        if array.seq_sharded:
            raise _no_layout(array, "seq_sharded takes max_len 0, and a paged pool plans whole segments")
        if not array.reads:
            raise _no_layout(array, "reads is empty, and PagedKVConfig needs num_qo_heads from an Attend read")
        bound = _bindings(self, array, setup.by_read)
        q_heads = {read.op.q_heads for read in array.reads.values()}
        if len(q_heads) > 1:
            raise _no_layout(array, f"reads have q_heads {sorted(q_heads)}, and PagedKVConfig has one num_qo_heads")
        pool = KVSpec(
            resource_key=array.name, nodes=array.nodes,
            config=PagedKVConfig(
                num_layers=len(array.layers), num_qo_heads=q_heads.pop(),
                num_kv_heads=heads, head_dim=head_dim, max_seq_len=array.max_len,
            ),
        )
        return [pool, *_op_specs(array, bound)]

    def steps(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        pre, post, rows = _forks(handle.array, declared)
        steps[handle.name] = KVStep(
            segments=declared.segments, commit=declared.keep, pre_forks=pre, post_forks=post, fork_rids=rows,
            sink=declared.placeholders, eager_write=declared.eager_add,
        )

    def layouts(self, by_label: Mapping[str, Layout], keyed_spans: Mapping[str, int]) -> "dict[str, LabelLayout]":
        from mstar.model.base import LabelLayout

        return {
            label: LabelLayout(
                spans=layout.spans, growth=layout.growth, scratch=layout.scratch, keyed_spans=keyed_spans.get(label),
            )
            for label, layout in by_label.items()
        }


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
    """A rewritable array of one entry, in a recurrent state pool that links to the paged array on its nodes."""

    handle_class = ValueHandle

    def resolve(self, arrays: Sequence[Array], declared: Sequence[Array], setup: "StateSetup") -> Resolved:
        return Resolved({array.name: self._specs(array, declared) for array in arrays}, {}, {})

    def _specs(self, array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
        if array.reads:
            raise _no_layout(array, "reads must be empty, since RecurrentStateConfig has no read")
        if array.seq_sharded:
            raise _no_layout(array, "seq_sharded takes max_len 0, and a Value's entries stay whole on each rank")
        if array.entry.host:
            raise _no_layout(array, f"entry has host fields {sorted(array.entry.host)}, and RecurrentStateConfig has "
                                    "no block for one")
        if not array.updates:
            raise _no_layout(array, f"updates is empty, and today's resources hold a Value only for a linear attention "
                                    f"update ({LinearAttnBinding.kinds()})")
        bound = _bindings(self, array)
        holders = [
            other for other in arrays if isinstance(storage_of(other), PagedStorage) and other.nodes & array.nodes
        ]
        if len(holders) != 1 or not array.nodes <= holders[0].nodes:
            raise _no_layout(array, f"its nodes must hold one append-only array, the one RecurrentStateConfig.kv_cache "
                                    f"links to, and they hold {[other.name for other in holders]}")
        layouts = _field_layouts(bound)
        pool = RecurrentStateSpec(
            resource_key=array.name, nodes=array.nodes,
            config=RecurrentStateConfig(
                num_layers=len(array.layers),
                blocks={
                    field: layouts[field].block(tensor) if field in layouts else _block(tensor)
                    for field, tensor in array.entry.tensors.items()
                },
                kv_cache=holders[0].name,
            ),
        )
        return [pool, *_op_specs(array, bound)]

    def steps(self, handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
        pre, post, rows = _forks(handle.array, declared)
        steps[handle.name] = RecurrentStep(segments=declared.segments, pre_forks=pre, post_forks=post, fork_rids=rows)


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
PAGED_CROSS = PagedCrossStorage()
VALUE = ValueStorage()
HOST_VALUE = HostValueStorage()
WITHIN_STEP = WithinStepStorage()


def storage_of(array: Array) -> Storage:
    """The storage today's resources keep ``array`` in."""
    if is_value(array):
        return VALUE if array.entry.tensors else HOST_VALUE
    if array.rewritable:
        raise _no_layout(array, "a rewritable array has no spec unless it is a Value, of max_len 1")
    if not array.max_len:
        return WITHIN_STEP
    return PAGED_CROSS if any(read.queries is not None for read in array.reads.values()) else PAGED


# `Attend` is an `AttentionSpec` on a paged array, a `CrossAttentionSpec` on one read with another array's queries and a
# `RaggedAttentionSpec` within a step, so a binding is per storage and op type.
BINDINGS: dict[tuple[Storage, type[Op]], OpBinding] = {
    (PAGED, Attend): PagedAttend(),
    (PAGED_CROSS, Attend): PagedCrossAttend(),
    (WITHIN_STEP, Attend): RaggedAttend(),
    (VALUE, GatedDeltaNet): GDN(),
    (VALUE, CausalConv): ConvOnLinearAttn(),
}
# What `backend: sdpa` binds in place of the default, for each storage and op type SDPA can run.
SDPA: dict[tuple[Storage, type[Op]], OpBinding] = {(WITHIN_STEP, Attend): SdpaAttend()}


@dataclass(frozen=True)
class StateSetup:
    """What a deployment sets up for the declared state: the binding it picks for each read whose resource doesn't take
    the default, and the store of the Values' host fields."""

    by_read: Mapping[str, OpBinding] = EMPTY
    host_fields: HostFields = dataclasses.field(default_factory=HostFields)


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
    return _no_layout(array, f"{where} must be {kinds}, the ops today's resources run on this array")


def _op_specs(array: Array, bound: Sequence[BoundOp]) -> list[NodeResourceSpec]:
    return [spec for each in bound for spec in each.binding.specs(array, each.key, each.declared)]


def _field_layouts(bound: Sequence[BoundOp]) -> dict[str, OpBinding]:
    """Each field an update writes, with that update's binding, which lays the field out in the pool."""
    return {field: each.binding for each in bound for field in each.declared.writes}
