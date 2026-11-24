"""DeepSeek-V4.1 plain and YaRN interleaved rotary embeddings with inverse support."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

if TYPE_CHECKING:
    from mstar.model.components.deepseek_v41.config import DeepSeekV41Config


class Angles(NamedTuple):
    cos: torch.Tensor
    sin: torch.Tensor


def inv_freq(
    dim: int,
    theta: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """Compute ``dim // 2`` rotary frequencies (plain, or YaRN when ``original_seq_len > 0``)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
    if original_seq_len > 0:

        def turns_at(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(turns_at(beta_fast)), 0)
        high = min(math.ceil(turns_at(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / max(high - low, 1e-3)).clamp(0, 1)
        freqs = freqs / factor * ramp + freqs * (1 - ramp)
    return freqs


def precompute_freqs_cis(
    positions: torch.Tensor,
    dim: int,
    theta: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Angles:
    """Compute fp32 ``(cos, sin)`` angles for ``positions``."""
    freqs = inv_freq(dim, theta, original_seq_len, factor, beta_fast, beta_slow).to(positions.device)
    angles = torch.outer(positions.reshape(-1).to(torch.float32), freqs)
    return Angles(angles.cos(), angles.sin())


def apply_rotary_emb(x: torch.Tensor, angles: Angles, inverse: bool = False) -> torch.Tensor:
    """Apply interleaved complex RoPE (or its inverse) to ``x`` along its last dimension."""
    cos, sin = angles
    if x.ndim == 3 and cos.ndim == 2:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif x.ndim == 4 and cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    if inverse:
        sin = -sin
    pairs = x.float().unflatten(-1, (-1, 2))
    re, im = pairs[..., 0], pairs[..., 1]
    return torch.stack([re * cos - im * sin, re * sin + im * cos], dim=-1).flatten(-2).to(x.dtype)


def rotate_tail(x: torch.Tensor, angles: Angles, inverse: bool = False) -> torch.Tensor:
    """Rotate the last ``2 * angles.cos.shape[-1]`` dims of ``x``."""
    dim = 2 * angles.cos.shape[-1]
    return torch.cat([x[..., :-dim], apply_rotary_emb(x[..., -dim:], angles, inverse=inverse)], dim=-1)


class DeepSeekV41Rope(nn.Module):
    """Plain (window-only) and YaRN-compressed rotary frequency tables."""

    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.config = config
        self._plain_cpu = inv_freq(config.qk_rope_head_dim, config.rope_theta)
        self._compress_cpu = inv_freq(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_original_max_position_embeddings,
            config.rope_factor,
            config.rope_beta_fast,
            config.rope_beta_slow,
        )
        self._dev_tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def angles(self, positions: torch.Tensor, compressed: bool) -> Angles:
        dev = positions.device
        tables = self._dev_tables.get(dev)
        if tables is None:
            tables = (self._plain_cpu.to(dev), self._compress_cpu.to(dev))
            self._dev_tables[dev] = tables
        freqs = tables[1] if compressed else tables[0]
        theta = torch.outer(positions.reshape(-1).to(torch.float32), freqs)
        return Angles(theta.cos(), theta.sin())
