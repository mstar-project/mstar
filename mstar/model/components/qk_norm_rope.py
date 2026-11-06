"""Per-head q/k RMSNorm and rotate-half RoPE in torch, which a fused kernel's reference computes."""

from __future__ import annotations

import math

import torch

from mstar.model.components.norm import rms_norm


def norm_rotate(
    x: torch.Tensor, weight: torch.Tensor, eps: float, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    half = x.shape[-1] // 2
    normed = rms_norm(x.reshape(-1, x.shape[-1]), weight, eps).view(x.shape)
    return normed * cos + torch.cat((-normed[..., half:], normed[..., :half]), dim=-1) * sin


def rope_cos_sin(pos_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain rotate-half ``cos`` and ``sin`` rows at ``pos_ids``, ``(tokens, head_dim)`` in fp32, as
    `mstar.utils.attention.rms_norm_rope` computes them."""
    half = torch.arange(0, head_dim // 2, dtype=torch.float32, device=pos_ids.device)
    angle = pos_ids.to(torch.float32)[:, None] * torch.exp(-math.log(theta) * (half * 2.0 / head_dim))
    return angle.cos().repeat(1, 2), angle.sin().repeat(1, 2)
