"""The Snake activation of audio codec decoders, ``x + inv_beta * sin(alpha * x)^2``
per channel, in one Triton pass.

The expression runs as five elementwise kernels over a decoder's audio-rate
activations, often half of a conv stack's time. The kernel computes in fp32
whatever the input dtype, with libdevice's ``sin`` (CUDA's ``sinf``, which
``torch.sin`` uses too) rather than the approximate hardware sine, so an fp32
decode keeps the eager numerics.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _snake_kernel(x_ptr, out_ptr, alpha_ptr, inv_beta_ptr, channels, length, BLOCK: tl.constexpr):
        row = tl.program_id(0)  # one (batch, channel) row of a contiguous [B, C, T]
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < length
        channel = row % channels
        alpha = tl.load(alpha_ptr + channel)
        inv_beta = tl.load(inv_beta_ptr + channel)
        offsets = row.to(tl.int64) * length + cols
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        s = libdevice.sin(x * alpha)
        tl.store(out_ptr + offsets, (x + inv_beta * (s * s)).to(out_ptr.dtype.element_ty), mask=mask)


_BLOCK = 1024


def snake(x: torch.Tensor, alpha: torch.Tensor, inv_beta: torch.Tensor) -> torch.Tensor:
    """``x + inv_beta * sin(alpha * x)^2`` for ``x`` ``[B, C, T]``, with ``alpha``
    and ``inv_beta`` fp32 ``[C]``. Eager off CUDA."""
    if triton is None or not x.is_cuda:
        a = alpha.to(x.dtype)[None, :, None]
        b = inv_beta.to(x.dtype)[None, :, None]
        return x + b * torch.sin(x * a).pow(2)
    x = x.contiguous()
    out = torch.empty_like(x)
    batch, channels, length = x.shape
    grid = (batch * channels, triton.cdiv(length, _BLOCK))
    _snake_kernel[grid](x, out, alpha.contiguous(), inv_beta.contiguous(), channels, length, BLOCK=_BLOCK)
    return out
