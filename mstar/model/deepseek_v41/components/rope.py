"""DeepSeek-V4.1's rotary embedding: adjacent pairs of a latent's last ``qk_rope_head_dim`` dims.

Window-only layers rotate at ``rope_theta``. Layers that attend a compressed cache rotate at
``compress_rope_theta`` with YaRN's frequency blend, since one latent stands for several tokens
and the cache reaches past the training context. The position resource has neither a second
theta nor YaRN, so the angles are computed here from the step's positions, in fp32.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from mstar.model.deepseek_v41.config import DeepseekV41Config


def inv_freq(dim: int, theta: float, original_seq_len: int = 0, factor: float = 1.0,
             beta_fast: float = 32.0, beta_slow: float = 1.0) -> torch.Tensor:
    """``dim / 2`` rotary frequencies. With ``original_seq_len`` (YaRN), a frequency whose
    wavelength fits the training context stays, one far past it is divided by ``factor``, and
    the band between ``beta_fast`` and ``beta_slow`` turns over that context ramps between."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def turns_at(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(turns_at(beta_fast)), 0)
        high = min(math.ceil(turns_at(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        freqs = freqs / factor * ramp + freqs * (1 - ramp)
    return freqs


class Angles(NamedTuple):
    cos: torch.Tensor  # (rows, dim / 2) fp32
    sin: torch.Tensor


def rotate(x: torch.Tensor, angles: Angles, inverse: bool = False) -> torch.Tensor:
    """``x (rows, [heads,] dim)`` rotated by its row's angles (``inverse``: back), in fp32,
    returned in ``x``'s dtype."""
    cos, sin = angles
    if x.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if inverse:
        sin = -sin
    pairs = x.float().unflatten(-1, (-1, 2))
    re, im = pairs[..., 0], pairs[..., 1]
    return torch.stack([re * cos - im * sin, re * sin + im * cos], dim=-1).flatten(-2).to(x.dtype)


def rotate_tail(x: torch.Tensor, angles: Angles, inverse: bool = False) -> torch.Tensor:
    """``x`` with its last ``2 * angles.cos.shape[-1]`` dims rotated."""
    dim = 2 * angles.cos.shape[-1]
    return torch.cat([x[..., :-dim], rotate(x[..., -dim:], angles, inverse)], dim=-1)


class Rope(nn.Module):
    """Both frequency sets, built on the model's device once it is known."""

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.config = config
        self._plain: torch.Tensor | None = None
        self._compress: torch.Tensor | None = None

    def process_weights_after_loading(self, device) -> None:
        cfg = self.config
        self._plain = inv_freq(cfg.qk_rope_head_dim, cfg.rope_theta).to(device)
        self._compress = inv_freq(
            cfg.qk_rope_head_dim, cfg.compress_rope_theta, cfg.rope_original_max_position_embeddings,
            cfg.rope_factor, cfg.rope_beta_fast, cfg.rope_beta_slow).to(device)

    def angles(self, positions: torch.Tensor, compressed: bool) -> Angles:
        freqs = self._compress if compressed else self._plain
        theta = torch.outer(positions.to(torch.float32), freqs)
        return Angles(theta.cos(), theta.sin())
