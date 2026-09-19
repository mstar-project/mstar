"""The KDA recurrence for one decode token per row, reading its inputs where they already are.

The layer's merged projections leave q | k | v side by side in the conv output ``y [N, 3P]`` and
the raw beta as a column slice of the in_proj output; fla's ``fused_recurrent_kda_fwd`` wants each
of them contiguous, which cost three copies per layer per step for every batch of more than one
row (a one-row slice is contiguous by itself, so bs=1 never paid). This kernel takes row strides
and column offsets instead. The arithmetic is fla's, in the same order (and the checkpoint kernel's
in ``kda_spec_recurrent.py``): L2-normalised q and k, the gate and the beta sigmoid in the kernel,
one beta per head, fp32 V-first state read from the row's pool slot and written back in place.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.heuristics({"USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None})
@triton.jit
def _kda_decode_kernel(
    y, g, beta, A_log, dt_bias, o, states, slot_ids, lower_bound,
    stride_y_row, stride_g_row, stride_beta_row, stride_state_slot,
    scale: tl.constexpr,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H
    slot = tl.load(slot_ids + i_n).to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    P = H * K
    row = y + i_n * stride_y_row
    p_q = row + i_h * K + o_k
    p_k = row + P + i_h * K + o_k
    p_v = row + 2 * P + i_h * V + o_v
    p_g = g + i_n * stride_g_row + i_h * K + o_k
    p_beta = beta + i_n * stride_beta_row + i_h
    p_o = o + (i_n * H + i_h) * V + o_v
    p_h = states + slot * stride_state_slot + i_h * K * V + o_v[:, None] * K + o_k[None, :]

    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)

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
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def kda_decode(
    y: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
    states: torch.Tensor, slot_ids: torch.Tensor, scale: float, lower_bound: float | None,
) -> torch.Tensor:
    """``y [N, 3P]`` the conv output (q | k | v, any row stride), ``g [N, H, D]`` raw gates (any row
    stride, heads and dims dense), ``beta [N, H]`` raw (any row stride); ``states [slots, H, D, D]``
    fp32 V-first, read at ``slot_ids [N]`` and written back there. Returns ``o [N, H, D]`` in ``y``'s
    dtype. Device tensors of static shape throughout: capturable."""
    n, h, d = g.shape
    assert y.shape == (n, 3 * h * d) and beta.shape == (n, h), (y.shape, g.shape, beta.shape)
    assert y.stride(1) == 1 and g.stride(2) == 1 and g.stride(1) == d and beta.stride(1) == 1
    assert states.dim() == 4 and states.shape[1:] == (h, d, d) and states.dtype == torch.float32
    assert states.is_contiguous() and slot_ids.numel() == n
    o = torch.empty(n, h, d, dtype=y.dtype, device=y.device)
    bk = triton.next_power_of_2(d)
    bv = 32
    grid = (triton.cdiv(d, bv) * n * h,)
    _kda_decode_kernel[grid](
        y, g, beta, A_log, dt_bias, o, states, slot_ids, lower_bound,
        y.stride(0), g.stride(0), beta.stride(0), states.stride(0),
        scale=float(scale), H=h, K=d, V=d, BK=bk, BV=bv, num_warps=4, num_stages=2,
    )
    return o
