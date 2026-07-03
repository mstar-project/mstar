"""RMSNorm and AdaRMSNorm.

``RMSNorm`` supports both the standard Llama-style normalization
(``normed * weight``) and Gemma's variant (``normed * (1 + weight)``)
through the ``gemma_mode`` flag. In standard mode the FlashInfer fused
kernel is used; Gemma mode falls back to a fp32 manual computation that
matches HF Gemma exactly.

``AdaRMSNorm`` adds adaRMS conditioning (scale / shift / gate from a
condition vector). Used by pi05's action expert flow-matching path; the
output is consumed by ``GatedDecoderLayer`` rather than a plain residual
add.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.utils.flashinfer_utils import run_rms_norm


class RMSNorm(nn.Module):
    """RMSNorm with optional Gemma-style ``(1 + weight)`` scaling.

    Args:
        hidden_size: feature dimension to normalize over.
        eps: variance epsilon.
        gemma_mode: if True, use ``(1 + weight)`` and a fp32 manual
            implementation (matches HF Gemma exactly; the loaded
            checkpoint weight is centered around zero, not one). If
            False, use ``weight`` and dispatch to FlashInfer's fused
            kernel.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, gemma_mode: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.gemma_mode = gemma_mode
        # Llama-style starts at 1; Gemma-style starts at 0 because the
        # checkpoint stores (weight - 1).
        init = torch.zeros(hidden_size) if gemma_mode else torch.ones(hidden_size)
        self.weight = nn.Parameter(init)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gemma_mode:
            orig_dtype = hidden_states.dtype
            x = hidden_states.to(torch.float32)
            var = x.square().mean(dim=-1, keepdim=True)
            normed = x * torch.rsqrt(var + self.variance_epsilon)
            normed = normed * (1.0 + self.weight.to(torch.float32))
            return normed.to(orig_dtype)
        # FlashInfer's rmsnorm wants 2D input — reshape/restore around the
        # call so callers can pass arbitrary leading dims (e.g. the talker
        # code_predictor's [bs, seq_len, hidden_size] batched path).
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])
        out = run_rms_norm(flat, self.weight, eps=self.variance_epsilon)
        return out.reshape(orig_shape)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}, gemma_mode={self.gemma_mode}"


class AdaRMSNorm(nn.Module):
    """RMSNorm with adaRMS conditioning.

    A per-norm ``nn.Linear(cond_dim, hidden_size*3)`` maps a shared
    condition vector to ``(scale, shift, gate)``. The normalization is
    ``rmsnorm(x) * (1 + scale) + shift`` and the gate is returned for the
    enclosing decoder layer to apply at the residual.

    The ``dense.weight`` and ``dense.bias`` are zero-initialized so the
    norm starts as the identity (matches HF Gemma / lerobot openpi).
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        self.dense = nn.Linear(cond_dim, hidden_size * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy import keeps the Triton dependency narrow (only AdaRMSNorm
        # imports it).
        from mstar.utils.adarms_norm import adarms_norm_fused

        BS = cond.shape[0]
        x_flat = x.view(BS * (x.shape[0] // BS), -1).contiguous()

        modulation = self.dense(cond)  # [BS, 3*H]
        H = self.hidden_size
        scale = modulation[:, :H]
        shift = modulation[:, H:2 * H]
        gate = modulation[:, 2 * H:]

        return adarms_norm_fused(x_flat, scale, shift, gate, self.variance_epsilon)
