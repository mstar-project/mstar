"""DeepSeek-style yarn rotary embedding for the drafter's rope dims, after PR #257's: GPT-J
interleaved rotation (``rope_interleave: true``, vLLM's ``is_neox_style=False``), fp32 tables, the
yarn frequency correction (beta_fast / beta_slow over the original window). As vLLM's
``DeepseekScalingRotaryEmbedding``: the tables scale by ``mscale(factor, mscale) /
mscale(factor, mscale_all_dim) * attn_factor`` and the attention's ``1/sqrt(d)`` by
``mscale(factor, mscale_all_dim)`` squared; both are 1 for the K3.1 preview release."""
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
    """``apply(x, positions)`` on the last ``dim`` values of ``x`` (``[T, dim]`` or ``[T, H, dim]``),
    positions ``[T]``; ``attn_scale_factor`` multiplies the attention's ``1/sqrt(d)``. The tables
    are computed per call from the positions in fp32, whatever the module's dtype, so no position
    bound is baked in."""

    def __init__(self, dim: int, params: YarnParams):
        super().__init__()
        self.dim = dim
        self.params = params
        self.table_scale = (yarn_mscale(params.factor, params.mscale)
                            / yarn_mscale(params.factor, params.mscale_all_dim) * params.attn_factor)
        m = yarn_mscale(params.factor, params.mscale_all_dim)
        self.attn_scale_factor = m * m
        self.register_buffer("inv_freq", yarn_inv_freq(dim, params), persistent=False)

    def _apply(self, fn, recurse=True):
        # A dtype cast of the drafter (``.to(torch.bfloat16)``) must not reach the frequencies: at
        # position 8K a bf16 frequency is off by up to ~10 rad (GPU job 1465). Moves still apply.
        super()._apply(fn, recurse)
        self.inv_freq = yarn_inv_freq(self.dim, self.params).to(self.inv_freq.device)
        return self

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[:, None] * self.inv_freq.float()[None, :]
        cos = (freqs.cos() * self.table_scale).repeat_interleave(2, dim=-1)
        sin = (freqs.sin() * self.table_scale).repeat_interleave(2, dim=-1)
        return cos, sin

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos_sin(positions)  # [T, dim]
        if x.dim() == 3:
            cos, sin = cos[:, None, :], sin[:, None, :]
        xf = x.float()
        x1, x2 = xf[..., ::2], xf[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (xf * cos + rotated * sin).to(x.dtype)
