"""KDA's output norm, ``RMSNorm_headdim(o) * weight * sigmoid(g)``, one launch, strided gate.

fla's ``rms_norm_gated`` does the same in the same fp32 order but reshapes ``g`` to rows of D, and
a gate that is a column slice of the merged in_proj output then gets copied first for every batch
of more than one row. Here the gate is addressed by row, head and dim strides.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_rmsnorm_kernel(
    x, g, w, y, eps,
    stride_x_row, stride_g_row, stride_g_head,
    H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    i_n, i_h = pid // H, pid % H
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + i_n * stride_x_row + i_h * D + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_xbar = tl.where(m_d, b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    b_y = (b_x * b_rstd) * b_w
    b_g = tl.load(g + i_n * stride_g_row + i_h * stride_g_head + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_y = b_y * tl.sigmoid(b_g)
    p_y = y + (i_n * H + i_h) * D + o_d
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_d)


def gated_rmsnorm(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x [N, H, D]`` (rows any stride, heads and dims dense), ``g [N, H, D]`` (any strides with unit
    dim stride), ``weight [D]``. Returns ``[N, H, D]`` contiguous in ``x``'s dtype."""
    n, h, d = x.shape
    assert g.shape == (n, h, d) and weight.shape == (d,), (x.shape, g.shape, weight.shape)
    assert x.stride(2) == 1 and x.stride(1) == d and g.stride(2) == 1
    y = torch.empty(n, h, d, dtype=x.dtype, device=x.device)
    _gated_rmsnorm_kernel[(n * h,)](
        x, g, weight, y, float(eps), x.stride(0), g.stride(0), g.stride(1),
        H=h, D=d, BD=triton.next_power_of_2(d), num_warps=1,
    )
    return y
