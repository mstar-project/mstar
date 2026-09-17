"""The KDA recurrence of a speculative verify step: prefix, checkpoint, block, in one launch.

A copy of fla's ``fused_recurrent_kda_fwd_kernel`` (flash-linear-attention, MIT) trimmed to what
Kimi K3 uses (varlen rows, fp32 V-first states in a slot pool, L2-normalised q/k, the gate and the
beta sigmoid computed in the kernel, one beta per head) with two changes for the checkpoint
recurrence (plan section 8.3 item 6):

* the initial state is read from the row's pool slot (``slot_ids``, 1-D), whatever the row's length;
* the state is stored back to that slot once, after token ``checkpoint_pos[row]`` (the last token of
  the row's accepted prefix), and never otherwise: a negative position stores nothing.

fla's own kernel stores a state after every token of a multi-token row (its speculative mode keeps
one slot per token), which is 15 extra 64 KB stores per row per head per layer here; and its 1-D
slot list is read per token, so it cannot take multi-token rows at all.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.heuristics({"USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None})
@triton.jit(do_not_specialize=["N"])
def _kda_recurrent_checkpoint_kernel(
    q, k, v, g, beta, A_log, dt_bias, o, states, slot_ids, checkpoint_pos, cu_seqlens, lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    stride_state_slot: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(slot_ids + i_n).to(tl.int64)
    ckpt = tl.load(checkpoint_pos + i_n).to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]  # V-first state tile [BV, BK]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_g = g + (bos * H + i_h) * K + o_k
    p_beta = beta + bos * H + i_h
    p_o = o + (bos * H + i_h) * V + o_v
    p_h = states + slot * stride_state_slot + i_h * K * V + o_v[:, None] * K + o_k[None, :]

    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)

    for i_t in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32) + b_bias
        if USE_LOWER_BOUND:
            b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
        else:
            b_gk = -tl.exp(b_A) * _softplus(b_g)
        b_h *= tl.exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta = tl.sigmoid(tl.load(p_beta).to(tl.float32))
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if i_t == ckpt:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)
        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_g += H * K
        p_beta += H
        p_o += H * V


def kda_recurrent_checkpoint(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    A_log: torch.Tensor, dt_bias: torch.Tensor, states: torch.Tensor, slot_ids: torch.Tensor,
    checkpoint_pos: torch.Tensor, cu_seqlens: torch.Tensor, scale: float, lower_bound: float | None,
) -> torch.Tensor:
    """``q, k, v, g [T, H, D]`` (raw g), ``beta [T, H]`` (raw), packed rows per ``cu_seqlens [N+1]``
    int32; ``states [slots, H, D, D]`` fp32 V-first, read at ``slot_ids [N]`` and written there
    after token ``checkpoint_pos [N]`` of the row (negative: never). Returns ``o [T, H, D]`` in
    ``v``'s dtype. Every argument is a device tensor of static shape: capturable."""
    t, h, d = k.shape
    n = cu_seqlens.numel() - 1
    assert q.shape == k.shape == g.shape and v.shape == (t, h, d) and beta.shape == (t, h)
    assert states.dim() == 4 and states.shape[1:] == (h, d, d) and states.dtype == torch.float32
    assert slot_ids.numel() == n and checkpoint_pos.numel() == n
    for x in (q, k, v, g, beta, states):
        assert x.is_contiguous(), "contiguous inputs"
    o = torch.empty_like(v)
    bk = triton.next_power_of_2(d)
    bv = 32
    grid = (triton.cdiv(d, bv) * n * h,)
    _kda_recurrent_checkpoint_kernel[grid](
        q, k, v, g, beta, A_log, dt_bias, o, states, slot_ids, checkpoint_pos, cu_seqlens, lower_bound,
        scale=float(scale), N=n, H=h, K=d, V=d, BK=bk, BV=bv, stride_state_slot=states.stride(0),
        num_warps=4, num_stages=2,
    )
    return o
