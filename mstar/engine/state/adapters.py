"""The glue between declarations and today's resources, as one layer: a storage per kind of array, a binding per
storage and op type, and a handle per storage. It builds the specs, steps and layouts everything downstream reads."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
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
from mstar.engine.resources.step import SubmoduleStep
from mstar.engine.state.analysis import position_key
from mstar.engine.state.calls import ArrayHandle, SamplerHandle
from mstar.engine.state.contract import (
    EMPTY,
    Array,
    ArrayStep,
    Attend,
    Block,
    Causal,
    CausalConv,
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

if TYPE_CHECKING:
    from mstar.model.base import LabelLayout, Model


def node_resources(model: "Model") -> list[NodeResourceSpec]:
    """The specs ``model`` declares through `declare_state`, then the ones it writes by hand."""
    specs = build_specs(model.declare_state(), kernel_settings=model.kernel_settings())
    return specs + list(model.get_node_resources() or [])


def build_specs(
    declarations: Sequence[Array | Sampler], *, kernel_settings: Mapping[str, Mapping[str, Any]] = EMPTY,
) -> list[NodeResourceSpec]:
    """Each array's specs, from its storage and its ops' bindings, under the names the declarations give, then a
    `PositionSpec` for each array that declares positions and a `SamplerSpec` per sampler.

    Raises ``ValueError`` on an array today's specs can't hold, naming what they lack. That includes a second update of
    one type on an array, a temporary limit: an update call finds its op by type, which today's resources need and the
    contract doesn't.
    """
    arrays = [declared for declared in declarations if isinstance(declared, Array)]
    specs: list[NodeResourceSpec] = []
    for array in arrays:
        specs += storage_of(array).specs(array, arrays)
        if array.positions is not None:
            specs.append(_position_spec(array))
    specs += [_sampler_spec(declared) for declared in declarations if isinstance(declared, Sampler)]
    _apply_kernel_settings(specs, arrays, kernel_settings)
    return specs


def _no_layout(array: Array, missing: str) -> ValueError:
    return ValueError(f"Array {array.name!r}: no layout yet: {missing}")


def _kv_shape(array: Array) -> tuple[int, ...]:
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or array.entry != KVEntry(heads=k.shape[0], head_dim=k.shape[1]):
        raise _no_layout(array, "entry must be a KVEntry, all PagedKVConfig holds: k and v of one shape, sharded on "
                                "their heads, with no dtype and no host field")
    return k.shape


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
    specs: Sequence[NodeResourceSpec], arrays: Sequence[Array], kernel_settings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set the kernel choices and capture sizes the contract leaves to the engine, and refuse any other, so this hook
    can't carry physical choices until the engine picks kernels itself."""
    owners = {
        bound.key: (array, bound.binding)
        for array in arrays for bound in _bindings(storage_of(array), array) if bound.binding.kernel_settings
    }
    by_key = {spec.resource_key: spec for spec in specs}
    for key, settings in kernel_settings.items():
        if key not in owners:
            raise ValueError(f"kernel_settings[{key!r}] names no read or update of the declarations that takes kernel "
                             f"settings, of {sorted(owners)}")
        array, binding = owners[key]
        refused = settings.keys() - binding.kernel_settings
        if refused:
            raise ValueError(f"Array {array.name!r}: kernel_settings[{key!r}] sets {sorted(refused)}; "
                             f"it may set only {sorted(binding.kernel_settings)}")
        for field, value in settings.items():
            setattr(by_key[key].config, field, value)


def node_handles(model: "Model", node: str, resources: Mapping[str, Any]) -> dict[str, ArrayHandle | SamplerHandle]:
    """``node``'s handles, by declared name, over its built ``resources``."""
    return {
        declared.name: build_handle(declared, resources)
        for declared in model.declare_state() if node in declared.nodes
    }


def build_handle(declared: Array | Sampler, resources: Mapping[str, Any]) -> ArrayHandle | SamplerHandle:
    """``declared``'s handle over the built ``resources``, with each of its ops bound."""
    if isinstance(declared, Sampler):
        return SamplerHandle(declared, resources[declared.name])
    storage = storage_of(declared)
    key = None if declared.positions is None else position_key(declared)
    return storage.handle_class(declared, resources, key, _bindings(storage, declared))


def state_steps(step: SubmoduleStep | None, handles: Mapping[str, ArrayHandle | SamplerHandle]) -> SubmoduleStep | None:
    """``step`` as a new step, with what its ``state`` declares expanded into today's step objects, built afresh for
    each step because `SubmoduleStep` writes the segments onto them."""
    if step is None or not step.state:
        return step
    steps = dict(step.steps)
    for name, declared in step.state.items():
        handle = handles.get(name)
        if isinstance(handle, SamplerHandle) and isinstance(declared, SampleStep):
            steps[name] = SamplerStep(
                apply_penalty=declared.apply_penalty, apply_filters=declared.apply_filters,
                prefill_tracked_tokens=declared.prompt_tokens, kept_rids=declared.kept_rows,
            )
        elif isinstance(handle, ArrayHandle) and isinstance(declared, ArrayStep):
            if declared.segments is None and step.segments is None:
                raise ValueError(f"Array {name!r}: a step lists it with no segments, its own or the step's")
            _array_steps(handle, declared, step, steps)
        else:
            raise ValueError(
                f"a step's state[{name!r}] is a {type(declared).__name__}, and on this node {name!r} is "
                f"{'not declared' if handle is None else type(handle).__name__}, which take an ArrayStep for an array "
                "and a SampleStep for a sampler"
            )
    expanded = replace(step, steps=steps)
    expanded.set_state_handles(tuple(handles[name] for name in step.state))
    return expanded


def _array_steps(handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
    array, segments = handle.array, declared.segments
    storage_of(array).steps(handle, declared, steps)
    for bound in handle._bindings:
        bound.binding.steps(bound.key, segments, steps)
    if array.positions is not None:
        # a step that keeps nothing doesn't advance the counter
        advance = declared.advance if declared.keep else (0,) * len(segments or step.segments)
        steps[handle.positions_key] = PositionStep(segments=segments, advance=advance, pos_ids=declared.position_ids)


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

    def steps(self, key: str, segments, steps: dict[str, Any]) -> None:
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
        if not isinstance(read.mask, Causal | Window):
            raise _no_layout(array, f"reads[{key!r}].mask must be Causal or Window, the masks AttentionConfig has")
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

    def steps(self, key: str, segments, steps: dict[str, Any]) -> None:
        steps[key] = AttentionStep(segments=segments, causal=True)


class RaggedAttend(OpBinding):
    """`Attend` within a step: a `RaggedAttentionSpec` over the step's own adds."""

    kernel_settings = frozenset({"max_segments_per_step", "compile_op"})

    def check(self, array: Array, key: str, read: Read) -> None:
        if not isinstance(read.mask, Block):
            raise _no_layout(array, f"reads[{key!r}].mask must be Block, the mask RaggedAttentionConfig has")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since RaggedAttentionConfig has no walk")

    def specs(self, array: Array, key: str, read: Read) -> list[NodeResourceSpec]:
        heads, head_dim = _kv_shape(array)
        return [RaggedAttentionSpec(
            resource_key=key, nodes=array.nodes,
            config=RaggedAttentionConfig(num_qo_heads=read.op.q_heads, num_kv_heads=heads, head_dim=head_dim),
        )]

    def steps(self, key: str, segments, steps: dict[str, Any]) -> None:
        steps[key] = AttentionStep(segments=segments, causal=False)


class LinearAttnBinding(OpBinding):
    """A recurrence a `LinearAttnSpec` runs under the op's key. Today's resources run the array's `CausalConv` inside
    it too."""

    kernel_settings = frozenset({"splits"})

    def steps(self, key: str, segments, steps: dict[str, Any]) -> None:
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


def _exchanges(attn: Any) -> bool:
    """Whether a call on ``attn`` goes through its exchange (`AttentionManager.splits`); a stand-in attention, which
    is no `AttentionManager`, keeps the direct call."""
    return isinstance(attn, AttentionManager) and attn.splits


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

    def _check_storage(self, declared: ArrayStep, segments, ctx) -> None:
        # A capture's dummy rows open only the default label.
        if ctx.capture:
            return
        for segment in segments:
            if (segment.request_id, segment.label) not in declared.placeholders and not self._kv.opens_label(
                segment.request_id, segment.label,
            ):
                raise self._refusal(f"a step touches label {segment.label!r} of request {segment.request_id}, "
                                    "which it didn't open")

    def begin_step(self, step) -> None:
        super().begin_step(step)
        # adds that run before the reads, outside the captured graph, which the reads find in the pages
        self._eager = step.state[self.name].eager_add

    def check_step(self, step) -> None:
        # in the step's forward, before its commit: the last add, if its layer went unread
        self._write_unread()
        super().check_step(step)

    @torch.compiler.disable
    def add(self, layer: int | torch.Tensor, *, keep: bool | None = None, **fields: torch.Tensor) -> None:
        layer = self._layer_id(layer)
        self._record_add(layer, keep)
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
        if not _exchanges(attn):
            self._write(layer, k, v)

    def _write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """``layer``'s write alone, where its kernel reads the pages: through the exchange where the attention
        exchanges, as `AttentionCallable.write` makes it."""
        attn = self._attn.get(layer)
        if attn is not None and _exchanges(attn):
            attn.exchange(partial(self._attend, layer), None, k, v, label=self.label)
        elif attn is None or attn.requires_kv_write:
            self._kv.write_kv(k, v, layer_idx=self._pool_layer[layer], label=self.label)

    def _write_unread(self) -> None:
        """Write the held add of a layer the step didn't read where its write waits for the read's exchange, so
        each add is written once: when the next add or the step's check finds it still held."""
        layer = self._held_layer
        if layer is not None and _exchanges(self._attn[layer]):
            self._write(layer, *self._take_hold(layer))

    def _needs_step_kv(self, attn: Any) -> bool:
        """Whether a read through ``attn`` attends the step's own k and v rather than only the pages, or goes through
        its exchange."""
        return _exchanges(attn) or attn.reads_step_kv(self.label)

    @torch.compiler.disable
    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        if q is None:
            raise self._no_layout("a read with no op")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        layer = self._layer_id(layer)
        self._record_read(layer)
        attn = self._attn[layer]
        if _exchanges(attn):
            # the add's write and the attention as one call, on the tokens and heads this rank stores
            return attn.exchange(partial(self._attend, layer), q, *self._take_hold(layer), label=self.label)
        # a read with no add before it attends the pages alone, which a kernel that takes no writes can't
        k, v = self._take_hold(layer) if self._held is not None or not attn.requires_kv_write else (None, None)
        return attn.run(
            q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=self._pool_layer[layer],
        )

    def _attend(self, layer: int, q: torch.Tensor | None, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor | None:
        """An exchange's ``run``: this layer's write where the kernel reads the pages, then its attention (``q``
        None: the write alone), as `AttentionCallable` makes the call."""
        attn, i = self._attn[layer], self._pool_layer[layer]
        if attn.requires_kv_write:
            self._kv.write_kv(k, v, layer_idx=i, label=self.label)
        if q is None:
            return None
        return attn.run(q, label=self.label, kv_cache_layer=self._views[layer], k=k, v=v, layer_idx=i)

    def at_last_token(self, x: torch.Tensor) -> torch.Tensor:
        self._check_open("at_last_token")
        return self.first_read.select_last_hidden(x, label=self.label)

    def _kernel_setting(self, read: str, field: str) -> Any:
        return getattr(self.reads[read], field)


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


class WithinStepHandle(ArrayHandle):
    """The calls on an array with ``max_len`` 0, which a block compiled alone traces. With ``compile_op`` set, the read
    stays in that graph through ``mstar::ragged_attention``. A check here runs only while not compiling: it would guard
    on the layer, and the block would retrace per layer. The graph doesn't take the layer at all, so nothing specializes
    on it."""

    def __init__(
        self, array: Array, resources: Mapping[str, Any], positions_key: str | None, bindings: tuple[BoundOp, ...],
    ):
        super().__init__(array, resources, positions_key)
        self._bindings = bindings
        self.reads = {bound.key: bound.binding.bind(array, bound.key, resources) for bound in bindings}
        self.first_read = next(iter(self.reads.values()))

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
        if not torch.compiler.is_compiling():
            self._held_layer = self._layer_id(layer)
            self._record_add(self._held_layer, False)
        self._held = fields["k"], fields["v"]

    def read(self, layer: int | torch.Tensor, q: torch.Tensor | None = None, **mask_args: Any) -> torch.Tensor:
        """Attend ``q`` over this layer's latest add, within each row's segment."""
        if q is None:
            raise self._no_layout("a read with no op")
        if mask_args:
            raise self._no_layout(f"mask arguments {sorted(mask_args)}, which no mask defines")
        if not torch.compiler.is_compiling():
            self._record_read(self._layer_id(layer))
        # Dropped here, not at the step's end: a compiled frame would otherwise return k and v from its graph, which
        # keeps the whole projection they are views of alive through the next layer.
        k, v = self._take_hold(layer)
        return self.first_read.run(q, k, v, label=self.label)

    def _kernel_setting(self, read: str, field: str) -> Any:
        return getattr(self.reads[read], field)


# Storages


class Storage(ABC):
    """How today's resources hold one kind of array: its specs, its own step, its layouts and its handle."""

    handle_class: ClassVar[type[ArrayHandle]]

    @abstractmethod
    def specs(self, array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
        """``array``'s specs, its ops' included, where ``arrays`` are all the declared ones."""

    def steps(self, handle: ArrayHandle, declared: ArrayStep, steps: dict[str, Any]) -> None:
        """Add the step of the array's own resource, under its name."""
        return

    def layouts(
        self, by_label: Mapping[str, Layout], keyed_spans: Mapping[str, int],
    ) -> "dict[str, LabelLayout] | None":
        """Today's `LabelLayout` per label, or None where the array's pool takes no layout."""
        return None


class PagedStorage(Storage):
    """An append-only array that keeps entries, in a pool of KV pages."""

    handle_class = PagedArrayHandle

    def specs(self, array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
        heads, head_dim = _kv_shape(array)
        if not array.reads:
            raise _no_layout(array, "reads is empty, and PagedKVConfig needs num_qo_heads from an Attend read")
        bound = _bindings(self, array)
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

    def steps(self, handle: ArrayHandle, declared: ArrayStep, steps: dict[str, Any]) -> None:
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


class ValueStorage(Storage):
    """A rewritable array of one entry, in a recurrent state pool that links to the paged array on its nodes."""

    handle_class = ValueHandle

    def specs(self, array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
        if array.reads:
            raise _no_layout(array, "reads must be empty, since RecurrentStateConfig has no read")
        if array.entry.host:
            raise _no_layout(array, f"entry has host fields {sorted(array.entry.host)}, and RecurrentStateConfig has "
                                    "no block for one")
        if not array.updates:
            raise _no_layout(array, f"updates is empty, and today's resources hold a Value only for a linear attention "
                                    f"update ({LinearAttnBinding.kinds()})")
        bound = _bindings(self, array)
        holders = [other for other in arrays if storage_of(other) is PAGED and other.nodes & array.nodes]
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

    def steps(self, handle: ArrayHandle, declared: ArrayStep, steps: dict[str, Any]) -> None:
        pre, post, rows = _forks(handle.array, declared)
        steps[handle.name] = RecurrentStep(segments=declared.segments, pre_forks=pre, post_forks=post, fork_rids=rows)


class WithinStepStorage(Storage):
    """An array with ``max_len`` 0, whose adds last only the step, so it has no pool."""

    handle_class = WithinStepHandle

    def specs(self, array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
        # an entry RaggedAttentionConfig can't hold is refused before its read is looked at
        _kv_shape(array)
        if len(array.reads) != 1:
            raise _no_layout(array, f"max_len 0 needs one read, the one RaggedAttentionConfig runs, and it has "
                                    f"{sorted(array.reads)}")
        return _op_specs(array, _bindings(self, array))


PAGED = PagedStorage()
VALUE = ValueStorage()
WITHIN_STEP = WithinStepStorage()


def storage_of(array: Array) -> Storage:
    """The storage today's resources keep ``array`` in."""
    if is_value(array):
        return VALUE
    if array.rewritable:
        raise _no_layout(array, "a rewritable array has no spec unless it is a Value, of max_len 1")
    return PAGED if array.max_len else WITHIN_STEP


# `Attend` is an `AttentionSpec` on a paged array and a `RaggedAttentionSpec` within a step, so a binding is per storage
# and op type.
BINDINGS: dict[tuple[Storage, type[Op]], OpBinding] = {
    (PAGED, Attend): PagedAttend(),
    (WITHIN_STEP, Attend): RaggedAttend(),
    (VALUE, GatedDeltaNet): GDN(),
    (VALUE, CausalConv): ConvOnLinearAttn(),
}


def _bindings(storage: Storage, array: Array) -> tuple[BoundOp, ...]:
    """``array``'s reads, then its updates, each with the binding that runs it on ``storage``, once the binding has
    checked it. Refuses a pair `BINDINGS` has no entry for, and a second update of one type."""
    bound = []
    for key, read in array.reads.items():
        binding = BINDINGS.get((storage, type(read.op)))
        if binding is None:
            raise _no_binding(storage, array, f"reads[{key!r}].op")
        bound.append(BoundOp(key, read, binding))
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
