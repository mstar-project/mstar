"""Length masks for padded batches.

Every Kokoro block runs over a padded ``[B, C, T]`` (or ``[B, T, C]``) batch in
which row ``b`` is valid up to ``lengths[b]``. The reference implementation
processes one request at a time, so its convolutions see zero padding only at
the true end of the sequence. Re-zeroing the padded tail after every op that
can write into it keeps a padded batch equal to a stack of single requests.
"""

from __future__ import annotations

import torch


def length_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    """``[B, size]`` boolean mask, True where ``t < lengths[b]``."""
    positions = torch.arange(size, device=lengths.device)
    return positions[None, :] < lengths[:, None]


def mask_channels(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the padded tail of a channel-major ``[B, C, T]`` tensor."""
    return x.masked_fill(~mask[:, None, :], 0.0)


def mask_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the padded tail of a time-major ``[B, T, C]`` tensor."""
    return x.masked_fill(~mask[:, :, None], 0.0)


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse the valid prefix of every row of ``[B, T, C]``, leaving the tail
    in place. Applying it twice is the identity."""
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    index = torch.where(positions < lengths[:, None], lengths[:, None] - 1 - positions, positions)
    return x.gather(1, index[:, :, None].expand(-1, -1, x.shape[-1]))
