"""Stand-in for FlashInfer's fused gated activation."""

from __future__ import annotations

import torch

from mstar_sim import clock


@clock.kernel_fn("flashinfer.silu_and_mul", static=True,
                 cost=lambda input, *a, **kw: (2.5 * input.numel(), 1.5 * input.numel() * input.element_size()))
def silu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None,
                 enable_pdl: bool | None = None) -> torch.Tensor:
    """``silu(input[..., :d]) * input[..., d:]``."""
    if input.shape[-1] * input.element_size() % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        return out
    return torch.empty((*input.shape[:-1], input.shape[-1] // 2), dtype=input.dtype, device=input.device)
