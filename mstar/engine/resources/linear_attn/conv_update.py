"""Slot-indexed causal depthwise conv update for one-token decode rows.

fla's ``causal_conv1d_update`` takes the rows' conv windows as one dense ``[N, D, W]`` tensor, so
a paged deployment gathers the rows' slots into it before the call and scatters the updated
windows back afterwards: two extra launches per layer per step (index_select and index_put),
plus the intermediate copy. This kernel does the same arithmetic as fla's (fp32 taps, the new
input in the last position, ``rtne`` downcasts) but addresses each row's window through
``slot_ids`` directly in the pool's ``[slots, D, W - 1]`` conv block, reading and updating it in
place. The block holds the ``W - 1`` inputs before the current token (the pool's convention,
``DeltaNetGeometry``); fla's window is one column wider, its oldest column dead weight. In
bf16/fp16 the output and the updated windows match fla's kernel bit for bit (GPU test in
``test/kimi_k3/test_gpu_fused_ops.py``); fp32 stays on fla (see ``conv_update_slots_supported``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_update_slots_kernel(
    x_ptr, state_ptr, slot_ptr, w_ptr, y_ptr,
    stride_x_n, stride_y_n,
    D: tl.constexpr, W: tl.constexpr, BD: tl.constexpr, BW: tl.constexpr, SILU: tl.constexpr,
):
    S: tl.constexpr = W - 1  # columns of the state block
    pid = tl.program_id(0)
    nd = tl.cdiv(D, BD)
    i_d, i_n = pid % nd, (pid // nd).to(tl.int64)
    slot = tl.load(slot_ptr + i_n).to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    b_x = tl.load(x_ptr + i_n * stride_x_n + o_d, mask=m_d, other=0).to(tl.float32)
    base = state_ptr + slot * D * S + o_d[:, None] * S
    # the W taps' window: the S kept inputs, then the new one (tap order = window position)
    b_win = tl.load(base + o_w[None, :], mask=m_d[:, None] & (o_w < S)[None, :], other=0.0).to(tl.float32)
    b_win = tl.where((o_w == S)[None, :], b_x[:, None], b_win)
    b_w = tl.load(w_ptr + o_d[:, None] * W + o_w[None, :], mask=m_d[:, None] & (o_w < W)[None, :], other=0)
    b_y = tl.sum(b_win * b_w, 1)
    if SILU:
        b_y = b_y * tl.sigmoid(b_y)
    b_y = tl.cast(b_y, dtype=y_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(y_ptr + i_n * stride_y_n + o_d, b_y, mask=m_d)
    # the window shifted by one is the next state: positions 1..S-1 move to 0..S-2, the new input last
    b_new = tl.load(base + (o_w + 1)[None, :], mask=m_d[:, None] & ((o_w + 1) < S)[None, :], other=0.0).to(tl.float32)
    b_new = tl.where((o_w == S - 1)[None, :], b_x[:, None], b_new)
    tl.store(base + o_w[None, :], tl.cast(b_new, dtype=state_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
             mask=m_d[:, None] & (o_w < S)[None, :])


def conv_update_slots(
    x: torch.Tensor, state: torch.Tensor, slot_ids: torch.Tensor, weight: torch.Tensor, activation: str | None = "silu",
) -> torch.Tensor:
    """One decode token per row: ``x [N, D]`` (unit stride along D), ``state [slots, D, W - 1]``
    (the pool's layer block, updated in place at ``slot_ids [N]``), ``weight [D, W]`` in ``x``'s
    dtype. Returns ``y [N, D]`` in ``x``'s dtype."""
    n, d = x.shape
    slots, d2, s = state.shape
    w = weight.shape[1]
    assert d == d2 and weight.shape == (d, w) and s == w - 1 and x.stride(1) == 1, (x.shape, state.shape, weight.shape)
    assert state.is_contiguous() and weight.is_contiguous() and slot_ids.numel() == n
    assert activation in (None, "silu", "swish"), activation
    y = torch.empty(n, d, dtype=x.dtype, device=x.device)
    bd = min(triton.next_power_of_2(d), 256)
    grid = (triton.cdiv(d, bd) * n,)
    _conv_update_slots_kernel[grid](
        x, state, slot_ids, weight, y, x.stride(0), y.stride(0),
        D=d, W=w, BD=bd, BW=triton.next_power_of_2(w), SILU=activation is not None,
    )
    return y


def conv_update_slots_supported(x: torch.Tensor, state: torch.Tensor) -> bool:
    """bf16/fp16 rows only: there the output rounds to the same value as fla's kernel; in fp32 the
    two can differ by one ulp for small widths (fla autotunes its block/warp split, which changes
    the four-tap summation order), so fp32 reference models keep fla's kernel."""
    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and state.dtype == x.dtype


__all__ = ["conv_update_slots", "conv_update_slots_supported"]
