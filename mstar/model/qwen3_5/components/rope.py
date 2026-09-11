"""Qwen3.5's interleaved 3D MRoPE, over a partial rotary dim.

Two differences from the 1D RoPE the position resource serves, which is why
this is model-side (``position/manager.py`` still has 3D positions as a TODO):

* the three position grids are woven into the frequency dims in a
  ``[T,H,W,T,H,W,...]`` pattern rather than chunked, and
* only ``head_dim * partial_rotary_factor`` of each head is rotated — 64 of
  256 for every released size — with the remainder passed through untouched.

TODO: this duplicates ``qwen3_omni/components/rope.py`` apart from the partial
rotation. Both belong in ``model/components/rope.py``.
"""
from __future__ import annotations

import torch


def compute_inv_freq(
    rotary_dim: int, rope_theta: float, device: torch.device | None = None,
) -> torch.Tensor:
    """``1 / theta^(2i/d)`` over the rotated dims only; ``[rotary_dim // 2]``."""
    return 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float()
            / rotary_dim
        )
    )


def _recompose_frequencies(
    freqs: torch.Tensor, mrope_section: list[int],
) -> torch.Tensor:
    """Weave the three grids together, T as the base and H/W overwriting.

    ``freqs`` is ``[3, seq_len, rotary_dim // 2]``. With section ``[11,11,10]``
    T keeps dims 0,3,6..., H takes 1,4,7... and W takes 2,5,8..., which is the
    ``mrope_interleaved`` layout.
    """
    out = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        idx = slice(offset, mrope_section[dim] * 3, 3)
        out[..., idx] = freqs[dim, ..., idx]
    return out


def compute_3d_cos_sin(
    position_ids_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: list[int],
    target_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape ``[seq_len, rotary_dim]`` from ``[3, seq_len]`` positions."""
    pos = position_ids_3d[:, None, :].float()        # [3, 1, seq_len]
    ifreq = inv_freq[None, :, None].float()          # [1, rotary_dim//2, 1]
    freqs = (ifreq * pos).transpose(1, 2)            # [3, seq_len, rotary_dim//2]
    freqs = _recompose_frequencies(freqs, mrope_section)
    # doubled for the rotate-half convention
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if target_dtype is not None:
        cos, sin = cos.to(target_dtype), sin.to(target_dtype)
    return cos, sin


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


def text_position_ids(
    seq_len: int, start_pos: int = 0, device: torch.device | None = None,
) -> torch.Tensor:
    """``[3, seq_len]`` for a pure-text span: all three grids advance together."""
    pos = torch.arange(seq_len, dtype=torch.float, device=device) + float(start_pos)
    return pos.unsqueeze(0).expand(3, -1).contiguous()
