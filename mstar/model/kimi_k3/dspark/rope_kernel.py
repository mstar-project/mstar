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
    x, out, cos, sin, positions, rows_per_pos,
    DIM: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
):
    """Program (position ``t``, block of ``BR`` heads): ``x [T * rows_per_pos, DIM]`` rows
    ``t * rows_per_pos + h`` rotated by ``cos/sin[positions[t]]``."""
    t = tl.program_id(0)
    hb = tl.program_id(1)
    pos = tl.load(positions + t).to(tl.int64)
    d = tl.arange(0, BD)
    dm = d < DIM
    c = tl.load(cos + pos * DIM + d, mask=dm, other=0.0)
    s = tl.load(sin + pos * DIM + d, mask=dm, other=0.0)
    h = hb * BR + tl.arange(0, BR)
    hm = h < rows_per_pos
    base = (t * rows_per_pos + h)[:, None] * DIM
    m = hm[:, None] & dm[None, :]
    xf = tl.load(x + base + d[None, :], mask=m, other=0.0).to(tl.float32)
    even = (d % 2) == 0
    # rotated[2i] = -x[2i + 1], rotated[2i + 1] = x[2i]
    x_next = tl.load(x + base + (d + 1)[None, :], mask=m & even[None, :], other=0.0).to(tl.float32)
    x_prev = tl.load(x + base + (d - 1)[None, :], mask=m & (~even)[None, :], other=0.0).to(tl.float32)
    rotated = tl.where(even[None, :], -x_next, x_prev)
    y = xf * c[None, :] + rotated * s[None, :]
    tl.store(out + base + d[None, :], y.to(out.dtype.element_ty), mask=m)


def yarn_rope_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``x [T, DIM]`` or ``[T, H, DIM]`` (any float dtype), fp32 tables ``[max_positions, DIM]``,
    ``positions [T]`` int -> ``x`` rotated, same shape and dtype. Static shapes, device tensors only."""
    dim = x.shape[-1]
    t = x.shape[0]
    rows_per_pos = x.numel() // (t * dim) if t else 1
    assert x.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and positions.is_contiguous()
    assert cos.shape[-1] == dim and sin.shape == cos.shape and cos.dtype == torch.float32 and dim % 2 == 0
    assert positions.shape == (t,)
    out = torch.empty_like(x)
    if t == 0:
        return out
    br = min(8, triton.next_power_of_2(rows_per_pos))
    grid = (t, triton.cdiv(rows_per_pos, br))
    _yarn_rope_kernel[grid](
        x, out, cos, sin, positions, rows_per_pos,
        DIM=dim, BD=triton.next_power_of_2(dim), BR=br, num_warps=1, enable_fp_fusion=False,
    )
    return out
