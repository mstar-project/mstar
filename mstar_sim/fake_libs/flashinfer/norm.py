"""Stand-ins for FlashInfer's RMSNorm kernels."""

from __future__ import annotations

import torch

from mstar_sim import clock


def _nbytes(*tensors: torch.Tensor) -> float:
    return float(sum(t.numel() * t.element_size() for t in tensors))


def _norm_cost(input, *args, **kwargs):
    return 4.0 * input.numel(), 2 * _nbytes(input)


def _fused_cost(input, residual, *args, **kwargs):
    return 5.0 * input.numel(), 2 * _nbytes(input, residual)


@clock.kernel_fn("flashinfer.rmsnorm", cost=_norm_cost)
def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
            out: torch.Tensor | None = None, enable_pdl: bool | None = None) -> torch.Tensor:
    return torch.empty_like(input) if out is None else out


@clock.kernel_fn("flashinfer.gemma_rmsnorm", cost=_norm_cost)
def gemma_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
                  out: torch.Tensor | None = None, enable_pdl: bool | None = None) -> torch.Tensor:
    return torch.empty_like(input) if out is None else out


@clock.kernel_fn("flashinfer.fused_add_rmsnorm", cost=_fused_cost)
def fused_add_rmsnorm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float = 1e-6, enable_pdl: bool | None = None) -> None:
    return None  # in place


@clock.kernel_fn("flashinfer.gemma_fused_add_rmsnorm", cost=_fused_cost)
def gemma_fused_add_rmsnorm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                            eps: float = 1e-6, enable_pdl: bool | None = None) -> None:
    return None  # in place
