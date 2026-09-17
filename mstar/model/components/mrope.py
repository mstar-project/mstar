"""Shared interleaved three-axis multimodal RoPE helpers."""

from __future__ import annotations

import torch


def compute_rope_freqs(
    head_dim: int,
    rope_theta: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return standard inverse RoPE frequencies for ``head_dim``."""
    return 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))


def apply_interleaved_mrope_freqs(
    freqs: torch.Tensor,
    mrope_section: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Interleave temporal, height, and width frequencies as THWTHW...TT."""
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        freqs_t[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
    return freqs_t


def compute_3d_cos_sin(
    position_ids_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: list[int] | tuple[int, ...] = (24, 20, 20),
    attention_scaling: float = 1.0,
    target_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interleaved three-axis cos/sin tables for rotate-half RoPE."""
    pos = position_ids_3d[:, None, None, :].float()
    ifreq = inv_freq[None, None, :, None].float()
    freqs = apply_interleaved_mrope_freqs((ifreq * pos).transpose(2, 3), mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).squeeze(0)
    sin = (emb.sin() * attention_scaling).squeeze(0)
    if target_dtype is not None:
        cos = cos.to(target_dtype)
        sin = sin.to(target_dtype)
    return cos, sin
