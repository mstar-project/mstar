"""Fused residual add + bias-free LayerNorm Triton kernel, for ``BiasFreeLayerNorm.add_forward``."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_kernel(
    residual_ptr, delta_ptr, weight_ptr, residual_out_ptr, normed_ptr,
    stride_r, stride_d, stride_ro, stride_n,
    H: tl.constexpr, BLOCK: tl.constexpr, HAS_DELTA: tl.constexpr, eps,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(residual_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_DELTA:
        d = tl.load(delta_ptr + row * stride_d + offs, mask=mask, other=0.0).to(tl.float32)
        # The eager layer stores the residual sum in bf16 before normalizing it.
        h = (x + d).to(residual_out_ptr.dtype.element_ty)
        tl.store(residual_out_ptr + row * stride_ro + offs, h, mask=mask)
        x = h.to(tl.float32)
    mean = tl.sum(x, axis=0) / H
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / H
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = w * (centered * tl.rsqrt(var + eps))
    tl.store(normed_ptr + row * stride_n + offs, y.to(normed_ptr.dtype.element_ty), mask=mask)


def add_layernorm(
    residual: torch.Tensor, delta: torch.Tensor | None,
    weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = residual + delta`` (bf16), then bias-free LayerNorm of ``h``.

    Returns ``(h, layernorm(h))``. With ``delta=None`` returns
    ``(residual, layernorm(residual))``.
    """
    tokens, hidden = residual.shape
    normed = torch.empty_like(residual)
    residual_out = torch.empty_like(residual) if delta is not None else residual
    if tokens == 0:
        return residual_out, normed
    _add_layernorm_kernel[(tokens,)](
        residual, delta if delta is not None else residual, weight, residual_out, normed,
        residual.stride(0), (delta if delta is not None else residual).stride(0),
        residual_out.stride(0), normed.stride(0),
        H=hidden, BLOCK=triton.next_power_of_2(hidden), HAS_DELTA=delta is not None,
        eps=eps, num_warps=8,
    )
    return residual_out, normed
