"""Today's specs, steps and layouts, built from declarations, so everything downstream of them sees the same objects."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, astuple, replace
from typing import TYPE_CHECKING, Any

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
    Causal,
    CausalConv,
    Entry,
    GatedDeltaNet,
    KVEntry,
    Layout,
    Read,
    Sampler,
    SampleSettings,
    SampleStep,
    TensorType,
    Window,
    is_value,
)
from mstar.engine.state.host import HostFields
from mstar.engine.state.within_step import WithinStepHandle

if TYPE_CHECKING:
    from mstar.model.base import LabelLayout, Model


def node_resources(model: "Model", *, sdpa_nodes: Collection[str] = frozenset()) -> list[NodeResourceSpec]:
    """The specs ``model`` declares through `declare_state`, then the ones it writes by hand."""
    specs = build_specs(model.declare_state(), kernel_settings=model.kernel_settings(), sdpa_nodes=sdpa_nodes)
    return specs + list(model.get_node_resources() or [])


def build_specs(
    declarations: Sequence[Array | Sampler], *, kernel_settings: Mapping[str, Mapping[str, Any]] = EMPTY,
    sdpa_nodes: Collection[str] = frozenset(),
) -> list[NodeResourceSpec]:
    """One pool per array that keeps entries, under its name, one `AttentionSpec` per read and one `LinearAttnSpec`
    per `GatedDeltaNet`, each under its key, one ragged spec per kind of read of the arrays with ``max_len`` 0 (see
    `_within_step_specs`), and one `PositionSpec` and `SamplerSpec` for each that declares them. A node in
    ``sdpa_nodes`` builds no ragged spec, since its reads run SDPA.

    Raises ``ValueError`` on an array today's specs can't hold, naming what they lack.
    """
    arrays = [declared for declared in declarations if isinstance(declared, Array)]
    by_name = {array.name: array for array in arrays}
    by_read = _within_step_specs(arrays)
    specs: list[NodeResourceSpec] = []
    for array in arrays:
        if array.rewritable:
            specs += _value_specs(array, arrays)
        elif array.max_len:
            specs += _kv_specs(array, by_name)
        elif not array.nodes <= sdpa_nodes:
            specs += [by_read[key] for key in array.reads if by_read[key].resource_key == key]
        if array.positions is not None:
            specs.append(_position_spec(array))
    specs += [_sampler_spec(declared) for declared in declarations if isinstance(declared, Sampler)]
    _apply_kernel_settings(specs, arrays, by_read, kernel_settings)
    return specs


def _no_layout(array: Array, missing: str) -> ValueError:
    return ValueError(f"Array {array.name!r}: no layout yet: {missing}")


def _kv_shape(array: Array) -> tuple[int, ...]:
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or array.entry != KVEntry(heads=k.shape[0], head_dim=k.shape[1]):
        raise _no_layout(array, "entry must be a KVEntry, all PagedKVConfig holds: k and v of one shape, sharded on "
                                "their heads, with no dtype")
    return k.shape


def _kv_specs(array: Array, by_name: Mapping[str, Array]) -> list[NodeResourceSpec]:
    heads, head_dim = _kv_shape(array)
    if array.seq_sharded:
        raise _no_layout(array, "seq_sharded takes max_len 0, and a paged pool plans whole segments")
    if not array.reads:
        raise _no_layout(array, "reads is empty, and PagedKVConfig needs num_qo_heads from an Attend read")
    for key, read in array.reads.items():
        if not isinstance(read.op, Attend):
            raise _no_layout(array, f"reads[{key!r}].op must be Attend, the only read AttentionConfig runs")
        if not isinstance(read.mask, Causal | Window | Bidirectional):
            raise _no_layout(array, f"reads[{key!r}].mask must be Causal, Window or Bidirectional, the masks "
                                    "AttentionConfig has")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since AttentionConfig has no walk")
        if read.queries is None:
            continue
        if len(array.labels) != 1:
            raise _no_layout(array, f"a cross read's array declares one label, CrossAttentionConfig's one "
                                    f"context_label, and it has {sorted(array.labels)}")
        if len(array.reads) != 1:
            raise _no_layout(array, f"reads[{key!r}] reads another array's queries, and a cross read's array takes no "
                                    "other read yet; declare that read on an array of its own")
        queried = by_name[read.queries]
        if queried.rewritable or not queried.max_len:
            raise _no_layout(array, f"reads[{key!r}] keeps entries and reads the queries of {read.queries!r}, which "
                                    "isn't an append-only array that keeps entries")
    q_heads = {read.op.q_heads for read in array.reads.values()}
    if len(q_heads) > 1:
        raise _no_layout(array, f"reads have q_heads {sorted(q_heads)}, and PagedKVConfig has one num_qo_heads")
    specs: list[NodeResourceSpec] = [KVSpec(
        resource_key=array.name, nodes=array.nodes,
        config=PagedKVConfig(
            num_layers=len(array.layers), num_qo_heads=q_heads.pop(),
            num_kv_heads=heads, head_dim=head_dim, max_seq_len=array.max_len,
        ),
    )]
    # the pool holds only the array's layers, so AttentionConfig.layers counts among them
    kv_layer = {layer: i for i, layer in enumerate(sorted(array.layers))}
    for key, read in array.reads.items():
        if read.queries is not None:
            specs.append(CrossAttentionSpec(
                resource_key=key, nodes=array.nodes,
                config=CrossAttentionConfig(
                    kv_cache=array.name, query_kv_cache=read.queries, context_label=min(array.labels),
                ),
            ))
            continue
        windowed = isinstance(read.mask, Window)
        specs.append(AttentionSpec(
            resource_key=key, nodes=array.nodes,
            config=AttentionConfig(
                kv_cache=array.name,
                sliding_window=read.mask.size if windowed else None,
                layers=tuple(kv_layer[layer] for layer in sorted(read.layers or array.layers)) if windowed else None,
            ),
        ))
    return specs


def _within_step_specs(arrays: Sequence[Array]) -> dict[str, NodeResourceSpec]:
    """The ragged spec of each read of an array with ``max_len`` 0, a `RaggedCrossAttentionSpec` for a read with
    ``queries``. Reads of one kind, on one node set and with one config share the spec of the first of them, under its
    key, so each read is a plan on that resource rather than a resource of its own."""
    sharded_nodes = {node for array in arrays if array.seq_sharded for node in array.nodes}
    by_name = {array.name: array for array in arrays}
    by_read: dict[str, NodeResourceSpec] = {}
    shared: dict[tuple, NodeResourceSpec] = {}
    for array in arrays:
        if array.max_len or array.rewritable:
            continue
        (key, read), heads, head_dim, dtype = _within_step_read(array)
        queried = by_name.get(read.queries, array)
        if read.queries is not None and len(queried.labels) != 1:
            raise _no_layout(array, f"reads[{key!r}] reads the queries of {read.queries!r}, which has labels "
                                    f"{sorted(queried.labels)}, and a max_len 0 read plans the queries of one label")
        config = RaggedAttentionConfig(
            num_qo_heads=read.op.q_heads, num_kv_heads=heads, head_dim=head_dim, dtype=dtype,
            # the heads narrow by SP only where the exchange runs over them
            shard_over_sp=not array.nodes & sharded_nodes or (array.seq_sharded and queried.seq_sharded),
        )
        spec_class = RaggedAttentionSpec if read.queries is None else RaggedCrossAttentionSpec
        spec = spec_class(resource_key=key, nodes=array.nodes, config=config)
        by_read[key] = shared.setdefault((spec_class, array.nodes, astuple(config)), spec)
    return by_read


def _within_step_read(array: Array) -> tuple[tuple[str, Read], int, int, torch.dtype | None]:
    """``array``'s one read, its heads and head dim, and its dtype, None for the compute dtype."""
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or k.shard != 0 or array.entry != Entry({"k": k, "v": k}):
        raise _no_layout(array, "entry must be a KVEntry, which may declare one dtype, all RaggedAttentionConfig holds")
    if len(array.reads) != 1:
        raise _no_layout(array, f"max_len 0 needs one read, the one RaggedAttentionConfig runs, and it has "
                                f"{sorted(array.reads)}")
    if len(array.labels) != 1:
        raise _no_layout(array, f"max_len 0 takes one label, since a plan holds one per call, and it has "
                                f"{sorted(array.labels)}")
    ((key, read),) = array.reads.items()
    if not isinstance(read.op, Attend):
        raise _no_layout(array, f"reads[{key!r}].op must be Attend, the only read RaggedAttentionConfig runs")
    if read.walks is not None:
        raise _no_layout(array, f"reads[{key!r}].walks must be None, since RaggedAttentionConfig has no walk")
    return (key, read), k.shape[0], k.shape[1], k.dtype


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


def _value_specs(array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
    if not is_value(array):
        raise _no_layout(array, "a rewritable array longer than one entry has no spec")
    if array.entry.host and not array.entry.tensors:
        # the engine holds a host field, so no resource does
        return []
    if array.reads:
        raise _no_layout(array, "reads must be empty, since RecurrentStateConfig has no read")
    if array.seq_sharded:
        raise _no_layout(array, "seq_sharded takes max_len 0, and a Value's entries stay whole on each rank")
    labels = tuple(sorted(array.labels))
    if array.entry.tensors and not array.updates:
        # one block per field, read and written whole by the model, linked to no KV
        return [RecurrentStateSpec(
            resource_key=array.name, nodes=array.nodes,
            config=RecurrentStateConfig(
                num_layers=len(array.layers),
                blocks={name: _block(tensor) for name, tensor in array.entry.tensors.items()},
                labels=labels,
            ),
        )]
    if set(array.entry.tensors) != {"state", "conv"} or array.entry.host:
        raise _no_layout(array, "entry must be exactly state and conv, the blocks DeltaNetGeometry reads")
    unknown = sorted(key for key, op in array.updates.items() if not isinstance(op, GatedDeltaNet | CausalConv))
    if unknown:
        raise _no_layout(array, f"updates {unknown} are neither GatedDeltaNet nor CausalConv, the kernels "
                                "LinearAttnConfig runs")
    deltas = [(key, op) for key, op in array.updates.items() if isinstance(op, GatedDeltaNet)]
    convs = [op for op in array.updates.values() if isinstance(op, CausalConv)]
    if len(deltas) != 1:
        raise _no_layout(array, f"updates must hold one GatedDeltaNet, got {len(deltas)}")
    if len(convs) > 1 or any(conv.activation != CausalConv().activation for conv in convs):
        raise _no_layout(array, "updates may hold one CausalConv, of activation silu, since LinearAttnConfig has no "
                                "activation")
    holders = [other for other in arrays if not other.rewritable and other.max_len and other.nodes & array.nodes]
    if len(holders) != 1 or not array.nodes <= holders[0].nodes:
        raise _no_layout(array, f"its nodes must hold one append-only array, the one RecurrentStateConfig.kv_cache "
                                f"links to, and they hold {[other.name for other in holders]}")
    if array.entry.tensors["state"].dtype is None:
        raise ValueError(f"Array {array.name!r}.entry.tensors['state'].dtype must be declared, since the rule that "
                         f"picks its stored dtype reads it")
    key, delta = deltas[0]
    return [
        RecurrentStateSpec(
            resource_key=array.name, nodes=array.nodes,
            config=RecurrentStateConfig(
                num_layers=len(array.layers),
                blocks={
                    "state": _state_block(array.entry.tensors["state"]),
                    "conv": _block(array.entry.tensors["conv"]),
                },
                kv_cache=holders[0].name,
                labels=labels,
            ),
        ),
        LinearAttnSpec(
            resource_key=key, nodes=array.nodes,
            config=LinearAttnConfig(
                recurrent_state=array.name, variant=LinearAttnVariant.GDN,
                qk_l2norm=delta.qk_l2norm, sm_scale=delta.scale,
            ),
        ),
    ]


def _block(tensor: TensorType) -> RecurrentBlockConfig:
    return RecurrentBlockConfig(
        shape=tensor.shape, dtype=tensor.dtype,
        shard_dims=() if tensor.shard is None else (tensor.shard,),
    )


def _state_block(state: TensorType) -> RecurrentBlockConfig:
    """The K-last layout FlashInfer's kernels read, (h_v, d_v, d_k), from the reference's (h_v, d_k, d_v).

    Stored in fp32 unless the decode kernel takes the declared dtype, as it does for bf16 only at K = V = 128.
    """
    h_v, d_k, d_v = state.shape
    dtype = torch.bfloat16 if state.dtype is torch.bfloat16 and d_k == d_v == 128 else torch.float32
    # the head axis stays first, and the last two swap
    shard = None if state.shard is None else (0, 2, 1)[state.shard]
    return _block(TensorType((h_v, d_v, d_k), dtype=dtype, shard=shard))


def _apply_kernel_settings(
    specs: Sequence[NodeResourceSpec], arrays: Sequence[Array], by_read: Mapping[str, NodeResourceSpec],
    kernel_settings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set the kernel choices, capture sizes and default pool sizes the contract leaves to the engine, and refuse any
    other, so this hook can't carry other physical choices until the engine picks kernels and sizes its pools itself."""
    by_name = {array.name: array for array in arrays}
    reads = {key: array for array in arrays for key in array.reads}
    deltas = {key: array for array in arrays for key, op in array.updates.items() if isinstance(op, GatedDeltaNet)}
    owners = by_name | reads | deltas
    by_key = {spec.resource_key: spec for spec in specs}
    for key, settings in kernel_settings.items():
        if key not in owners:
            raise ValueError(f"kernel_settings[{key!r}] names no read, GatedDeltaNet update or array of the "
                             f"declarations, which have {sorted(owners)}")
        # a read on a node in ``sdpa_nodes`` has no spec to set
        spec = by_key.get(by_read[key].resource_key if key in by_read else key)
        if isinstance(spec, CrossAttentionSpec) and settings:
            raise ValueError(f"Array {owners[key].name!r}: kernel_settings[{key!r}] sets {sorted(settings)}; a cross "
                             "read on a kept array takes no setting yet")
        if isinstance(spec, KVSpec):
            allowed = {"max_num_pages"}
        elif isinstance(spec, RecurrentStateSpec):
            allowed = {"max_slots"}
        elif key in by_name:
            allowed = set()    # a host-only Value or an array with max_len 0 has no pool
        elif not owners[key].max_len:
            allowed = {"max_segments_per_request", "max_segments_per_step", "compile_op", "workspace_mb"}
        else:
            allowed = {"splits", "decode_fmha", "decode_tensor_cores"} if key in reads else {"splits"}
        refused = settings.keys() - allowed
        if refused:
            raise ValueError(f"Array {owners[key].name!r}: kernel_settings[{key!r}] sets {sorted(refused)}; "
                             f"it may set only {sorted(allowed)}")
        if key not in by_name:
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
    model: "Model", node: str, resources: Mapping[str, Any], *, sp_group: CommGroup | None = None, sdpa: bool = False,
    host_fields: HostFields | None = None,
) -> dict[str, ArrayHandle | SamplerHandle]:
    """``node``'s handles, by declared name, over its built ``resources``. With ``sdpa``, its reads of arrays with
    ``max_len`` 0 have no resource and run SDPA. ``host_fields`` holds the fields of the Values that declare them."""
    declarations = model.declare_state()
    arrays = {declared.name: declared for declared in declarations if isinstance(declared, Array)}
    by_read = _within_step_specs(list(arrays.values()))
    handles: dict[str, ArrayHandle | SamplerHandle] = {}
    for declared in declarations:
        if node not in declared.nodes:
            continue
        if isinstance(declared, Sampler):
            handles[declared.name] = SamplerHandle(declared, resources[declared.name])
        elif declared.entry.host:
            host_fields.declare(declared)
            handles[declared.name] = ArrayHandle(declared, resources, None, host_fields)
        elif declared.max_len or declared.rewritable:
            key = None if declared.positions is None else position_key(declared)
            # `_kv_specs` refuses a read beside a cross read, so the first read says
            queries = next((read.queries for read in declared.reads.values()), None)
            handles[declared.name] = ArrayHandle(declared, resources, key, queried=arrays.get(queries))
        else:
            ((read_key, read),) = declared.reads.items()
            resource_key = by_read[read_key].resource_key
            handles[declared.name] = WithinStepHandle(
                declared, {read_key: None if sdpa else resources[resource_key]}, None, resource_key=resource_key,
                queried=arrays.get(read.queries), sp_group=sp_group,
            )
    return handles


def state_steps(step: SubmoduleStep | None, handles: Mapping[str, ArrayHandle | SamplerHandle]) -> SubmoduleStep | None:
    """``step`` as a new step, with what its ``state`` declares expanded into today's step objects, built afresh for
    each step because `SubmoduleStep` writes the segments onto them. None when every read it lists runs SDPA with no
    exchange, which leaves the step nothing to do."""
    if step is None or not step.state:
        return step
    if all(getattr(handles.get(name), "plans_nothing", False) for name in step.state):
        return None
    steps = dict(step.steps)
    plans: dict[str, _WithinStepPlan] = {}
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
            if handle.queried is not None and handle.queried.name not in step.state:
                raise ValueError(f"Array {name!r}: a step lists it without {handle.queried.name!r}, whose segments "
                                 "lay out its queries")
            if isinstance(handle, WithinStepHandle):
                if not handle.runs_sdpa:
                    _plan_within_step(handle, declared, step, plans.setdefault(handle.resource_key, _WithinStepPlan()))
            else:
                _array_steps(handle, declared, step, steps)
        else:
            raise ValueError(
                f"a step's state[{name!r}] is a {type(declared).__name__}, and on this node {name!r} is "
                f"{'not declared' if handle is None else type(handle).__name__}, which take an ArrayStep for an array "
                "and a SampleStep for a sampler"
            )
    steps.update((key, plan.step()) for key, plan in plans.items())
    expanded = replace(step, steps=steps)
    expanded.set_state_handles(tuple(handles[name] for name in step.state))
    return expanded


class _WithinStepPlan:
    """What the arrays sharing one ragged resource plan in one step: each label's segments, and the cross reads' pairs.

    A label is one layout on the resource, so two arrays with one label have to mean the same tokens.
    """

    __slots__ = ("by_label", "pairs")

    def __init__(self):
        self.by_label: dict[str, Sequence[Segment]] = {}
        self.pairs: list[tuple[str, str]] = []

    def add(self, name: str, segments: Sequence[Segment]) -> str:
        label = segments[0].label
        if tuple(self.by_label.setdefault(label, segments)) != tuple(segments):
            raise ValueError(f"Array {name!r}: label {label!r} on its resource already holds other segments")
        return label

    def step(self) -> AttentionStep | RaggedCrossAttentionStep:
        if self.pairs:
            return RaggedCrossAttentionStep(
                segments=tuple(s for segments in self.by_label.values() for s in segments), pairs=tuple(self.pairs),
            )
        if len(self.by_label) == 1:
            return AttentionStep(segments=next(iter(self.by_label.values())), causal=False)
        return AttentionStep(segments=tuple(s for segments in self.by_label.values() for s in segments), causal=False)


def _plan_within_step(
    handle: WithinStepHandle, declared: ArrayStep, step: SubmoduleStep, plan: _WithinStepPlan,
) -> None:
    """Add ``handle``'s read to ``plan``, over the segments its kernel sees. A Ulysses read runs over whole segments
    after the exchange, and any other over this rank's share of a sharded side."""
    name = handle.name
    segments = declared.segments or step.segments
    if handle.queried is None:
        plan.add(name, segments)
        return
    q_label = plan.add(name, handle.query_share(step.state[handle.queried.name].segments or step.segments))
    plan.pairs.append((q_label, plan.add(name, handle.key_share(segments))))


def _array_steps(handle: ArrayHandle, declared: ArrayStep, step: SubmoduleStep, steps: dict[str, Any]) -> None:
    array, segments = handle.array, declared.segments
    pre, post, rows = _forks(array, declared)
    if is_value(array):
        if not array.entry.tensors:
            return
        steps[array.name] = RecurrentStep(segments=segments, pre_forks=pre, post_forks=post, fork_rids=rows)
        for key, update in array.updates.items():
            if isinstance(update, GatedDeltaNet):
                steps[key] = LinearAttnStep(segments=segments)
        return
    if array.max_len:
        steps[array.name] = KVStep(
            segments=segments, commit=declared.keep, pre_forks=pre, post_forks=post, fork_rids=rows,
            sink=declared.placeholders, eager_write=declared.eager_add,
        )
    for key, causal in handle.causal_reads.items():
        steps[key] = AttentionStep(segments=segments, causal=causal)
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
    """Today's `LabelLayout` per paged array and label, from what a model's `process_prompt` returned.

    A Value's layout is dropped, since its pool reserves a slot per label. The model is at fault for a name it doesn't
    declare, so this raises ``RuntimeError``, which fails the request with a 500, as `refuse_undeclared_labels` does.
    """
    from mstar.model.base import LabelLayout

    arrays = {declared.name: declared for declared in declarations if isinstance(declared, Array)}
    for where, named in (("state_layout", state_layout), ("keyed_spans", keyed_spans)):
        for name, by_label in named.items():
            if name not in arrays:
                raise RuntimeError(f"{where}[{name!r}]: it names no declared array, of {sorted(arrays)}")
            undeclared = by_label.keys() - arrays[name].labels
            if undeclared:
                raise RuntimeError(f"{where}[{name!r}]: labels {sorted(undeclared)} aren't in Array {name!r}.labels")
    converted = {
        name: {
            label: LabelLayout(
                spans=layout.spans, growth=layout.growth, scratch=layout.scratch,
                keyed_spans=keyed_spans.get(name, {}).get(label),
            )
            for label, layout in by_label.items()
        }
        for name, by_label in state_layout.items() if not arrays[name].rewritable and arrays[name].max_len
    }
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
