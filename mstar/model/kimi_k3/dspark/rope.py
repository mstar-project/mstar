"""DeepSeek-style yarn rotary embedding for the draft's 64 rope dims, in the convention vLLM uses
for K3DSparkModel: GPT-J interleaved rotation (``is_neox_style=False``), fp32 tables, the yarn
frequency correction of Peng et al. (beta_fast / beta_slow over the original window), and the
attention scale multiplied by ``mscale_all_dim``'s factor squared (``0.1 * m * ln(factor) + 1``).
With ``mscale == mscale_all_dim`` the table itself carries no scale."""
from __future__ import annotations

import math

import torch
from torch import nn

from mstar.model.kimi_k3.dspark.config import YarnParams


def _correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_inv_freq(dim: int, p: YarnParams) -> torch.Tensor:
    pos_freqs = p.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extrapolation, interpolation = 1.0 / pos_freqs, 1.0 / (p.factor * pos_freqs)
    low = math.floor(_correction_dim(p.beta_fast, dim, p.rope_theta, p.original_max_position_embeddings))
    high = math.ceil(_correction_dim(p.beta_slow, dim, p.rope_theta, p.original_max_position_embeddings))
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp  # extrapolation_factor 1
    return interpolation * (1 - mask) + extrapolation * mask


def yarn_mscale(scale: float, mscale: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


class YarnRotary(nn.Module):
    """``cos_sin(positions)`` tables and ``apply(x, positions)`` on the last ``dim`` values of ``x``
    (``[T, dim]`` or ``[T, H, dim]``), positions ``[T]`` int; ``attn_scale_factor`` multiplies the
    attention's ``1/sqrt(d)``."""

    def __init__(self, dim: int, params: YarnParams, max_positions: int):
        super().__init__()
        self.dim = dim
        self.params = params
        table_scale = yarn_mscale(params.factor, params.mscale) / yarn_mscale(params.factor, params.mscale_all_dim)
        m = yarn_mscale(params.factor, params.mscale_all_dim)
        self.attn_scale_factor = m * m
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(t, yarn_inv_freq(dim, params))
        self.register_buffer("cos", (freqs.cos() * table_scale).repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin", (freqs.sin() * table_scale).repeat_interleave(2, dim=-1), persistent=False)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.is_contiguous() and positions.is_contiguous():
            from mstar.model.kimi_k3.dspark.rope_kernel import yarn_rope_fused

            return yarn_rope_fused(x, self.cos, self.sin, positions)  # one launch, bit-identical to below
        cos, sin = self.cos[positions], self.sin[positions]  # [T, dim]
        if x.dim() == 3:
            cos, sin = cos[:, None, :], sin[:, None, :]
        xf = x.float()
        x1, x2 = xf[..., ::2], xf[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (xf * cos + rotated * sin).to(x.dtype)
