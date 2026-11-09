"""mstar's GDN and causal-conv kernels as kernels of the simulated GPU.

Same names and signatures as ``flashinfer.gdn_prefill.chunk_gated_delta_rule``,
``flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose`` and
``mstar.utils.causal_conv1d``; each returns its outputs' shapes and is charged
the work the real fused kernel does, with the geometry the measured tables
need. And :data:`KEPT_WRITES`, the Triton kernels whose integer bookkeeping the
emulator has to keep.
"""

from __future__ import annotations

import functools
import inspect
import warnings

import torch

from mstar_sim import clock


def _nbytes(t) -> float:
    return float(t.numel() * t.element_size()) if isinstance(t, torch.Tensor) else 0.0


def _per_row(pool) -> float:
    return _nbytes(pool) / pool.shape[0] if isinstance(pool, torch.Tensor) and pool.shape[0] else 0.0


def _gdn_cost(a: dict) -> tuple:
    q, v = a["q"], a["v"]
    tokens = q.shape[0] * (q.shape[1] if q.dim() == 4 else 1)
    heads, kdim, vdim = v.shape[-2], q.shape[-1], v.shape[-1]
    rows = a.get("cu_seqlens")
    rows = (rows.numel() - 1) if isinstance(rows, torch.Tensor) else q.shape[0]
    io = sum(_nbytes(a.get(n)) for n in ("q", "k", "v")) * 2
    features = {"batch": rows, "tokens": tokens, "k_heads": q.shape[-2], "v_heads": heads,
                "head_k_dim": kdim, "head_v_dim": vdim}
    return 6.0 * tokens * heads * kdim * vdim, io + 2 * rows * _per_row(a.get("initial_state")), features


def _conv_cost(a: dict) -> tuple:
    x = a["x"]
    rows = a.get("cache_indices", a.get("conv_state_indices"))
    rows = rows.numel() if isinstance(rows, torch.Tensor) else 1
    if "conv_states" in a:  # varlen x [dim, tokens], one row per query_start_loc segment
        pool, channels, tokens = a["conv_states"], x.shape[0], x.shape[-1]
        starts = a.get("query_start_loc")
        batch = starts.numel() - 1 if isinstance(starts, torch.Tensor) else rows
    else:  # x [batch, dim] or [batch, dim, seqlen]
        pool, batch, channels = a.get("conv_state"), x.shape[0], x.shape[1]
        tokens = batch * (x.shape[2] if x.dim() == 3 else 1)
    features = {"batch": batch, "tokens": tokens, "channels": channels}
    return (2.0 * x.numel() * a["weight"].shape[-1],
            2 * _nbytes(x) + _nbytes(a["weight"]) + 2 * rows * _per_row(pool), features)


def _kernel(name: str, cost, writes_args: tuple[str, ...] = ()):
    """``kernel_fn`` with ``cost`` given the call's arguments by name."""

    def decorate(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def by_name(*args, **kwargs):
            return cost(sig.bind(*args, **kwargs).arguments)

        return clock.kernel_fn(name, cost=by_name, writes_args=writes_args)(fn)

    return decorate


@_kernel("flashinfer.gdn_prefill", _gdn_cost, writes_args=("output_state", "state_checkpoints"))
def chunk_gated_delta_rule(
    q, k, v, g=None, beta=None, scale=None, initial_state=None, output_final_state=False,
    cu_seqlens=None, use_qk_l2norm_in_kernel=False, output=None, output_state=None,
    state_checkpoints=None, checkpoint_cu_starts=None, checkpoint_every_n_tokens=0, use_cp="auto",
    state_indices=None,
):
    """``q``/``k`` ``[T, H, K]``, ``v`` ``[T, HV, V]``; out ``[T, max(H, HV), V]`` and,
    with ``output_final_state``, the state ``[num_seqs, max(H, HV), V, K]``."""
    heads = max(q.shape[1], v.shape[1])
    out = output if output is not None else torch.empty(
        (q.shape[0], heads, v.shape[2]), dtype=q.dtype, device=q.device)
    if not output_final_state:
        return out
    state = output_state if output_state is not None else torch.empty(
        (cu_seqlens.numel() - 1, heads, v.shape[2], q.shape[2]), dtype=torch.float32, device=q.device)
    return out, state


@_kernel("flashinfer.gdn_decode", _gdn_cost)
def gated_delta_rule_decode_pretranspose(
    q, k, v, state, A_log, a, dt_bias, b, scale=None, output=None, use_qk_l2norm=True,
    initial_state=None, initial_state_indices=None, output_state_indices=None,
):
    """``q``/``k`` ``[B, T, H, K]``, ``v`` ``[B, T, HV, V]``; the state is updated in place."""
    out = output if output is not None else torch.empty(
        (*q.shape[:2], v.shape[2], v.shape[3]), dtype=q.dtype, device=q.device)
    return out, initial_state if state is None else state


@_kernel("triton.causal_conv1d_fn", _conv_cost, writes_args=("conv_states",))
def causal_conv1d_fn(
    x, weight, bias, conv_states, query_start_loc, cache_indices=None, has_initial_state=None,
    activation="silu", pad_slot_id=-1, null_block_id=0, block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None, initial_state_idx=None, num_computed_tokens=None,
    block_size_to_align=0, metadata=None, validate_data=False, seqlens_cpu=None,
):
    """Varlen conv over ``x`` ``[dim, total_tokens]``."""
    return torch.empty_like(x)


@_kernel("triton.causal_conv1d_update", _conv_cost, writes_args=("conv_state",))
def causal_conv1d_update(
    x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
    num_accepted_tokens=None, query_start_loc=None, max_query_len=-1, null_block_id=0,
    block_idx_last_scheduled_token=None, initial_state_idx=None, validate_data=False, out=None,
):
    """Decode conv over ``x`` ``[batch, dim(, seqlen)]``, into ``out`` (default ``x``)."""
    return out if out is not None else x


def _scatter_cells(cells, values, store, **_):
    store.index_copy_(0, cells, values)


def _gather_column(slots, store, out, plus, STRIDE, COL, PLUS, **_):
    value = store.index_select(0, slots * STRIDE + COL)
    out[:slots.numel()].copy_(value + plus if PLUS else value)


def _advance_column(slots, store, STRIDE, COL, TRASH, **_):
    cells = slots * STRIDE + COL
    store.index_copy_(0, cells, store.index_select(0, cells) + (slots != TRASH))


def _gather_block_rows(slots, kv_len, store, table, lens, **_):
    if store._data.device.type == "meta":  # no lengths to read: an unknown length costs the row's longest
        warnings.warn("mstar_sim: the resident table is too big to keep: decode attention is priced at max_seq_len",
                      stacklevel=2)
        lens.fill_(torch.iinfo(lens.dtype).max)
    else:
        lens.copy_(kv_len + 1)  # the kernel's own count, with this step's token: what attention is priced by


# Triton kernels by name, for what they write on values the emulator keeps (a launch computes
# nothing else: :mod:`mstar_sim.triton_launch`). Resident planning's integer bookkeeping, as
# ``mstar.engine.resources.resident`` does it off a GPU: the lengths a step's attention is priced by.
KEPT_WRITES = {
    "_scatter_cells_kernel": _scatter_cells,
    "_gather_column_kernel": _gather_column,
    "_advance_column_kernel": _advance_column,
    "_gather_block_rows_kernel": _gather_block_rows,
}
