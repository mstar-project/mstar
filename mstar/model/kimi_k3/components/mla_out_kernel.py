"""MLA's per-head output: ``o_lat[t, h] @ W_UV[h]`` then the output gate, written in place.

torch spells this as an einsum (a batched GEMM over heads whose result is head-major), a reshape to
``[T, H * V]`` that copies for more than one row, a sigmoid over a strided gate slice and a multiply:
four launches per layer. One program per (row, head) here does the small product in fp32 chunks over
the latent dim and writes the gated row slice where o_proj reads it. Rounding follows torch's: the
product to the activation dtype, the sigmoid to it, then the multiply.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_out_kernel(
    o_lat, w_uv, g, y,
    stride_o_row, stride_o_head, stride_w_head, stride_w_l, stride_g_row,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr, BL: tl.constexpr, BV: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    i_n, i_h = pid // H, pid % H
    o_v = tl.arange(0, BV)
    m_v = o_v < V
    acc = tl.zeros([BV], dtype=tl.float32)
    for l0 in tl.range(0, L, BL):
        o_l = l0 + tl.arange(0, BL)
        m_l = o_l < L
        b_o = tl.load(o_lat + i_n * stride_o_row + i_h * stride_o_head + o_l, mask=m_l, other=0.0).to(tl.float32)
        b_w = tl.load(w_uv + i_h * stride_w_head + o_l[:, None] * stride_w_l + o_v[None, :],
                      mask=m_l[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(b_o[:, None] * b_w, axis=0)
    p_y = y + i_n * (H * V) + i_h * V + o_v
    out = acc.to(p_y.dtype.element_ty)
    if HAS_GATE:
        b_g = tl.load(g + i_n * stride_g_row + i_h * V + o_v, mask=m_v, other=0.0).to(tl.float32)
        sig = tl.sigmoid(b_g).to(p_y.dtype.element_ty)
        out = (out.to(tl.float32) * sig.to(tl.float32)).to(p_y.dtype.element_ty)
    tl.store(p_y, out, mask=m_v)


def mla_out(o_lat: torch.Tensor, w_uv: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """``o_lat [T, H, L]`` (unit stride along L), ``w_uv [H, L, V]`` (unit stride along V),
    ``gate [T, H * V]`` or None (any row stride). Returns ``[T, H * V]`` contiguous in ``o_lat``'s dtype."""
    t, h, l = o_lat.shape
    v = w_uv.shape[-1]
    assert w_uv.shape == (h, l, v) and o_lat.stride(2) == 1 and w_uv.stride(2) == 1
    if gate is not None:
        assert gate.shape == (t, h * v) and gate.stride(1) == 1
    y = torch.empty(t, h * v, dtype=o_lat.dtype, device=o_lat.device)
    _mla_out_kernel[(t * h,)](
        o_lat, w_uv, gate if gate is not None else y, y,
        o_lat.stride(0), o_lat.stride(1), w_uv.stride(0), w_uv.stride(1), gate.stride(0) if gate is not None else 0,
        H=h, L=l, V=v, BL=32, BV=triton.next_power_of_2(v), HAS_GATE=gate is not None, num_warps=4,
    )
    return y
