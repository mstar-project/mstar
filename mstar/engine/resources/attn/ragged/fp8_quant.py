"""FP8 (e4m3) quantization of packed ``[n * S, H, D]`` q / k / v for an FP8 attention
kernel: one scale per (segment, head), the layout FA4's ``TensorwiseScales`` take.

Two Triton passes over the tensor: the absolute max of each (segment, head) block, then
the scaled cast. Inside a captured graph the max buffer is zeroed and refilled on every
replay.
"""

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0
_BLOCK_S = 64


@triton.jit
def _amax_kernel(x_ptr, amax_ptr, S, H, stride_t, stride_h, D: tl.constexpr, BLOCK_S: tl.constexpr):
    sb, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, D)
    offs = (b * S + rows[:, None]).to(tl.int64) * stride_t + h * stride_h + cols[None, :]
    x = tl.load(x_ptr + offs, mask=rows[:, None] < S, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr + b * H + h, tl.max(tl.abs(x)))


@triton.jit
def _cast_kernel(x_ptr, y_ptr, amax_ptr, S, H, stride_t, stride_h, D: tl.constexpr, BLOCK_S: tl.constexpr,
                 FMAX: tl.constexpr):
    sb, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, D)
    mask = rows[:, None] < S
    src = (b * S + rows[:, None]).to(tl.int64) * stride_t + h * stride_h + cols[None, :]
    dst = ((b * S + rows[:, None]).to(tl.int64) * H + h) * D + cols[None, :]
    inv = FMAX / tl.maximum(tl.load(amax_ptr + b * H + h), 1e-6)
    x = tl.load(x_ptr + src, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + dst, (x * inv).to(y_ptr.dtype.element_ty), mask=mask)


def quantize_per_segment_head(x: torch.Tensor, segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` ``[segments * S, H, D]`` (rows by stride, last dim contiguous) to
    ``(e4m3 [segments * S, H, D] contiguous, descale [segments, H] fp32)``, so that
    ``x ~= y * descale`` per (segment, head)."""
    rows, heads, dim = x.shape
    assert x.stride(-1) == 1 and rows % segments == 0
    seq = rows // segments
    amax = torch.zeros(segments, heads, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(seq, _BLOCK_S), heads, segments)
    _amax_kernel[grid](x, amax, seq, heads, x.stride(0), x.stride(1), D=dim, BLOCK_S=_BLOCK_S)
    y = torch.empty((rows, heads, dim), device=x.device, dtype=torch.float8_e4m3fn)
    _cast_kernel[grid](x, y, amax, seq, heads, x.stride(0), x.stride(1), D=dim, BLOCK_S=_BLOCK_S, FMAX=E4M3_MAX)
    return y, amax.clamp_min(1e-6) / E4M3_MAX
