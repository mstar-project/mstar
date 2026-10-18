"""RMSNorm kernels on AMD GPUs: FlashInfer's ROCm fork.

Its native HIP kernels keep FlashInfer's numerics (fp32 math, one
round-to-nearest-even store), so AMD GPUs compute what NVIDIA ones do. AITER's
are a little faster per call but store bf16 by truncation on gfx942, which biases
a residual stream that passes through every layer. Imported only on ROCm.
"""

import torch

from mstar.utils.flashinfer_rocm import import_flashinfer_rocm

_norm = import_flashinfer_rocm("RMSNorm").norm


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # the fork's kernel takes 2-D input only; per-head norms arrive 3-D
    if x.dim() == 2:
        return _norm.rmsnorm(x, weight, eps=eps)
    return _norm.rmsnorm(x.reshape(-1, x.shape[-1]), weight, eps=eps).view(x.shape)


def fused_add_rmsnorm(
    hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    eps: float, gemma: bool,
) -> None:
    fused = _norm.gemma_fused_add_rmsnorm if gemma else _norm.fused_add_rmsnorm
    fused(hidden_states, residual, weight, eps)
