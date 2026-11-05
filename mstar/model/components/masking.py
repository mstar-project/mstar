"""Length masks for padded batches."""

from __future__ import annotations

import torch


def length_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    """``[B, size]`` boolean mask, True where ``t < lengths[b]``."""
    positions = torch.arange(size, device=lengths.device)
    return positions[None, :] < lengths[:, None]
