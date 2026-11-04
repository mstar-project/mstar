"""Today's resource specs, built from declarations, so everything downstream of the specs sees the same objects."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.state.contract import (
    EMPTY,
    Array,
    Attend,
    Causal,
    CausalConv,
    GatedDeltaNet,
    KVEntry,
    TensorType,
    Window,
    is_value,
)

if TYPE_CHECKING:
    from mstar.model.base import Model


def node_resources(model: "Model") -> list[NodeResourceSpec]:
    """The specs ``model`` declares through `declare_state`, then the ones it writes by hand."""
    specs = build_specs(model.declare_state(), kernel_settings=model.kernel_settings())
    return specs + list(model.get_node_resources() or [])


def build_specs(
    arrays: Sequence[Array], *, kernel_settings: Mapping[str, Mapping[str, Any]] = EMPTY,
) -> list[NodeResourceSpec]:
    """One pool per array under its name, one `AttentionSpec` per read and one `LinearAttnSpec` per `GatedDeltaNet`,
    each under its key.

    Raises ``ValueError`` on an array today's specs can't hold, naming what they lack.
    """
    specs: list[NodeResourceSpec] = []
    for array in arrays:
        specs += _value_specs(array, arrays) if array.rewritable else _kv_specs(array)
    _apply_kernel_settings(specs, arrays, kernel_settings)
    return specs


def _no_layout(array: Array, missing: str) -> ValueError:
    return ValueError(f"Array {array.name!r}: no layout yet: {missing}")


def _kv_specs(array: Array) -> list[NodeResourceSpec]:
    k = array.entry.tensors.get("k")
    if k is None or len(k.shape) != 2 or array.entry != KVEntry(heads=k.shape[0], head_dim=k.shape[1]):
        raise _no_layout(array, "entry must be a KVEntry, all PagedKVConfig holds: k and v of one shape, sharded on "
                                "their heads, with no dtype and no host field")
    if not array.reads:
        raise _no_layout(array, "reads is empty, and PagedKVConfig needs num_qo_heads from an Attend read")
    for key, read in array.reads.items():
        if not isinstance(read.op, Attend):
            raise _no_layout(array, f"reads[{key!r}].op must be Attend, the only read AttentionConfig runs")
        if not isinstance(read.mask, Causal | Window):
            raise _no_layout(array, f"reads[{key!r}].mask must be Causal or Window, the masks AttentionConfig has")
        if read.walks is not None:
            raise _no_layout(array, f"reads[{key!r}].walks must be None, since AttentionConfig has no walk")
    q_heads = {read.op.q_heads for read in array.reads.values()}
    if len(q_heads) > 1:
        raise _no_layout(array, f"reads have q_heads {sorted(q_heads)}, and PagedKVConfig has one num_qo_heads")
    specs: list[NodeResourceSpec] = [KVSpec(
        resource_key=array.name, nodes=array.nodes,
        config=PagedKVConfig(
            num_layers=len(array.layers), num_qo_heads=q_heads.pop(),
            num_kv_heads=k.shape[0], head_dim=k.shape[1], max_seq_len=array.max_len,
        ),
    )]
    # the pool holds only the array's layers, so AttentionConfig.layers counts among them
    kv_layer = {layer: i for i, layer in enumerate(sorted(array.layers))}
    for key, read in array.reads.items():
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


def _value_specs(array: Array, arrays: Sequence[Array]) -> list[NodeResourceSpec]:
    if not is_value(array):
        raise _no_layout(array, "a rewritable array longer than one entry has no spec")
    if array.reads:
        raise _no_layout(array, "reads must be empty, since RecurrentStateConfig has no read")
    if set(array.entry.tensors) != {"state", "conv"} or array.entry.host:
        raise _no_layout(array, "entry must be exactly state and conv, the blocks DeltaNetGeometry reads")
    deltas = [(key, op) for key, op in array.updates.items() if isinstance(op, GatedDeltaNet)]
    convs = [op for op in array.updates.values() if isinstance(op, CausalConv)]
    if len(deltas) != 1:
        raise _no_layout(array, f"updates must hold one GatedDeltaNet, got {len(deltas)}")
    if len(convs) > 1 or any(conv.activation != CausalConv().activation for conv in convs):
        raise _no_layout(array, "updates may hold one CausalConv, of activation silu, since LinearAttnConfig has no "
                                "activation")
    holders = [other for other in arrays if not other.rewritable and other.nodes & array.nodes]
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
    specs: Sequence[NodeResourceSpec], arrays: Sequence[Array], kernel_settings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set the two kernel choices the contract leaves to the engine, and refuse any other, so this hook can't carry
    physical choices until the engine picks kernels itself."""
    reads = {key: array.name for array in arrays for key in array.reads}
    deltas = {key: array.name for array in arrays for key, op in array.updates.items() if isinstance(op, GatedDeltaNet)}
    owners = reads | deltas
    by_key = {spec.resource_key: spec for spec in specs}
    for key, settings in kernel_settings.items():
        if key not in owners:
            raise ValueError(f"kernel_settings[{key!r}] names no read or GatedDeltaNet update of the declarations, "
                             f"which have {sorted(owners)}")
        allowed = {"splits", "decode_fmha"} if key in reads else {"splits"}
        refused = settings.keys() - allowed
        if refused:
            raise ValueError(f"Array {owners[key]!r}: kernel_settings[{key!r}] sets {sorted(refused)}; "
                             f"it may set only {sorted(allowed)}")
        for field, value in settings.items():
            setattr(by_key[key].config, field, value)
