"""3D rotary position embedding helpers for V-JEPA 2.

Ported verbatim from HuggingFace ``transformers/models/vjepa2/modeling_vjepa2.py``
and upstream ``vjepa2/src/models/utils/modules.py``.  Both copies share the
same frequency schedule and the same "duplicate then rotate adjacent pair"
construction noted in the upstream comment as a pretraining-compatibility
quirk.  Do not "fix" the `.repeat(..., 2)` to `.repeat_interleave(2)` — it
would break the pretrained weights.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings along the last ``D`` dims of ``x``.

    Args:
        x: ``[B, num_heads, N, D]`` (or broadcastable) — queries or keys.
        pos: positions of shape ``[N]`` or ``[B, num_heads, N]``.

    Returns:
        Rotated tensor, same shape as ``x``.
    """
    _, _, _, D = x.size()

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat(1, 1, 1, 2)
    emb_cos = emb_cos.repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


def rope_tables_BNHD(
    pos: torch.Tensor, dim: int, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sin, cos)`` for ``apply_rope_BNHD`` over the last ``dim`` dims."""
    omega = torch.arange(dim // 2, dtype=dtype, device=device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega

    if pos.dim() == 1:
        freq = pos.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * omega
    else:
        freq = pos.unsqueeze(-1) * omega  # [B, N, num_heads, D//2]

    emb_sin = freq.sin().repeat(1, 1, 1, 2)  # [B, N, num_heads, D]
    emb_cos = freq.cos().repeat(1, 1, 1, 2)
    return emb_sin, emb_cos


def apply_rope_BNHD(x: torch.Tensor, emb_sin: torch.Tensor, emb_cos: torch.Tensor) -> torch.Tensor:
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class RopeTables(NamedTuple):
    """One cached step's ``(sin, cos)`` per axis (``rope_tables_BNHD``): built once
    from the step's positions and shared by every block, q and k."""

    d: tuple[torch.Tensor, torch.Tensor]
    h: tuple[torch.Tensor, torch.Tensor]
    w: tuple[torch.Tensor, torch.Tensor]
    time: tuple[torch.Tensor, torch.Tensor] | None  # action tokens, over the d dims
