"""Qwen3.5's interleaved 3D MRoPE, over a partial rotary dim.

Model-side because the position resource serves only 1D RoPE. The three grids
are interleaved ``[T,H,W,T,H,W,...]`` over the frequency dims, and only
``head_dim * partial_rotary_factor`` of each head is rotated (64 of 256 for
every released size).

The position and table helpers live in ``model/components/mrope.py``, shared with
the scaffold's Qwen3-VL prompt encoder; the partial rotation stays here.

TODO: ``qwen3_omni/components/rope.py`` still duplicates them.
"""
from __future__ import annotations

import torch

from mstar.model.components.mrope import (  # noqa: F401
    compute_3d_cos_sin,
    compute_inv_freq,
    text_position_ids,
)
from mstar.model.components.mrope import image_position_advance as vision_position_advance  # noqa: F401
from mstar.model.components.mrope import image_position_ids as vision_position_ids  # noqa: F401


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading ``cos.shape[-1]`` dims of each head, pass the rest.

    ``q``/``k`` are ``[tokens, heads, head_dim]``; cos/sin are
    ``[tokens, rotary_dim]`` and broadcast over the head axis.
    """
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return (
        torch.cat((q_rot, q_pass), dim=-1),
        torch.cat((k_rot, k_pass), dim=-1),
    )
