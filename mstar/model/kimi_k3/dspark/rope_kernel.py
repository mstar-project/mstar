"""The yarn rotary embedding as one launch: the tables gathered at the positions and the GPT-J
interleaved rotation applied, in fp32 with the reference's operation order (no fused multiply-add,
so the result is bit-identical to ``YarnRotary.apply``'s torch path). Nine launches per call in
torch (gathers, casts, slices, a stack, the products and the sum) made the draft's rope the largest
single source of small kernels in a speculative step: three calls per layer, on 5 layers."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _yarn_rope_kernel(
    x, out, cos, sin, positions, rows_per_pos, stride_t, stride_h,
    DIM: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
):
    """Program (position ``t``, block of ``BR`` heads): the ``DIM`` values of ``x`` at
    ``t * stride_t + h * stride_h`` rotated by ``cos/sin[positions[t]]`` into the contiguous ``out``."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    pos = tl.load(positions + t).to(tl.int64)
    d = tl.arange(0, BD)
    dm = d < DIM
    c = tl.load(cos + pos * DIM + d, mask=dm, other=0.0)
    s = tl.load(sin + pos * DIM + d, mask=dm, other=0.0)
    h = hb * BR + tl.arange(0, BR)
    hm = h < rows_per_pos
    src = (t * stride_t + h * stride_h)[:, None]
    m = hm[:, None] & dm[None, :]
    xf = tl.load(x + src + d[None, :], mask=m, other=0.0).to(tl.float32)
    even = (d % 2) == 0
    # rotated[2i] = -x[2i + 1], rotated[2i + 1] = x[2i]
    x_next = tl.load(x + src + (d + 1)[None, :], mask=m & even[None, :], other=0.0).to(tl.float32)
    x_prev = tl.load(x + src + (d - 1)[None, :], mask=m & (~even)[None, :], other=0.0).to(tl.float32)
    rotated = tl.where(even[None, :], -x_next, x_prev)
    y = xf * c[None, :] + rotated * s[None, :]
    dst = (t * rows_per_pos + h)[:, None] * DIM
    tl.store(out + dst + d[None, :], y.to(out.dtype.element_ty), mask=m)


def yarn_rope_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``x [T, DIM]`` or ``[T, H, DIM]`` (any float dtype; the last dimension contiguous, the others
    strided at will: the rope part of a query or key is usually a split view), fp32 tables
    ``[max_positions, DIM]``, ``positions [T]`` int -> ``x`` rotated, same shape and dtype,
    contiguous. Static shapes, device tensors only."""
    dim = x.shape[-1]
    t = x.shape[0]
    rows_per_pos = x.shape[1] if x.dim() == 3 else 1
    assert x.dim() in (2, 3) and x.stride(-1) == 1, "the rope dimension must be contiguous"
    assert cos.is_contiguous() and sin.is_contiguous() and positions.is_contiguous()
    assert cos.shape[-1] == dim and sin.shape == cos.shape and cos.dtype == torch.float32 and dim % 2 == 0
    assert positions.shape == (t,)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if t == 0:
        return out
    br = min(8, triton.next_power_of_2(rows_per_pos))
    grid = (t, triton.cdiv(rows_per_pos, br))
    _yarn_rope_kernel[grid](
        x, out, cos, sin, positions, rows_per_pos, x.stride(0), x.stride(1) if x.dim() == 3 else 0,
        DIM=dim, BD=triton.next_power_of_2(dim), BR=br, num_warps=1, enable_fp_fusion=False,
    )
    return out
