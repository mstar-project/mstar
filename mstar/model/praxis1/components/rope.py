"""Rotary positions for the Praxis-1 trunk.

Two things here are unusual enough that a stock RoPE will not do.

**Positions are a 3-D grid flattened by strides**, not a running token
index. A video token at ``(frame, row, col)`` sits at
``start + 1 + frame * 129600 + row * 360 + col``, so positions run to the
hundreds of millions. That is what forces the second unusual part.

**The table is built in two halves and recombined.** A position is split as
``high * 1_000_000 + low``; each part's angle is a float64 product whose cos
and sin are taken in float64 and then *rounded through float32 and back*;
the two are combined by the angle-sum identity and the result rounded to
float32. The reference states this is bit-equal to a lookup in precomputed
million-row tables, and the rounding is what makes that true. Computing the
angle directly in float64 would be more accurate and would not match.

Only the first ``effective_dim // 2`` frequency pairs rotate. The rest have
zero frequency, carried as an explicit zero pad rather than as a slice, so
the table keeps the full head width and the rotation touches everything
uniformly. With the shipped config that is 32 of 64 pairs.

The frequency tensor is a plain attribute rather than a buffer, matching the
reference: ``module.to(device)`` deliberately does not move it, because it
is built once on the CPU in float32 and the device copy is cached per
device.
"""

from __future__ import annotations

import torch
from torch import nn

# The 3-D grid's strides. A row is 360 columns and a frame is 360 rows.
ROW_STRIDE = 360
FRAME_STRIDE = 129_600


def video_positions(frames: int, height: int, width: int, start: int) -> torch.Tensor:
    """Rotary positions of one video slot, frame-major.

    ``start + 1`` rather than ``start``: position 0 is reserved for the
    registers, which all share it.
    """
    frame = torch.arange(frames).view(-1, 1, 1) * FRAME_STRIDE
    row = torch.arange(height).view(1, -1, 1) * ROW_STRIDE
    col = torch.arange(width).view(1, 1, -1)
    return (start + 1 + frame + row + col).reshape(-1)


class RotaryEmbedding(nn.Module):
    """The ``(cos, sin)`` table for integer positions.

    Returns ``(B, N, dim // 2, 2)`` float32. See the module docstring for why
    the split and the rounding are not optional.
    """

    def __init__(
        self,
        dim: int = 128,
        effective_dim: int = 64,
        base: int = 26_000_000,
        table_size: int = 1_000_000,
    ) -> None:
        super().__init__()
        if effective_dim > dim:
            raise ValueError(f"effective_dim {effective_dim} exceeds dim {dim}")
        self.dim = dim
        self.effective_dim = effective_dim
        self.base = base
        self.table_size = table_size
        exponent = (
            torch.arange(0, effective_dim, 2, device="cpu")[: effective_dim // 2].float()
            / effective_dim
        )
        theta = 1.0 / (base**exponent)
        pad = torch.zeros((dim - effective_dim) // 2, device="cpu")
        # A plain attribute, not a buffer: module casts and moves must not
        # touch it, or the float32 frequencies silently become bfloat16.
        self.theta = torch.cat([theta, pad], dim=0)

    @staticmethod
    def _cos_sin(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """float64 cos and sin, each rounded through float32 and back."""
        return angle.cos().float().double(), angle.sin().float().double()

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        theta = self.theta.to(positions.device).double()
        high = positions // self.table_size
        low = positions % self.table_size
        # Both casts are exact: the rows hold float32 positions.
        high_pos = (high.to(torch.float32) * self.table_size).double()
        low_pos = low.to(torch.float32).double()
        cos_high, sin_high = self._cos_sin(high_pos[..., None] * theta)
        cos_low, sin_low = self._cos_sin(low_pos[..., None] * theta)
        cos = cos_high * cos_low - sin_high * sin_low
        sin = sin_high * cos_low + cos_high * sin_low
        table = torch.stack([cos, sin], dim=-1).float()
        if table.dim() == 3:
            table = table.unsqueeze(0)
        return table


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` and ``k`` ``(B, H, N, D)`` in float32.

    Channel ``j`` pairs with channel ``j + D // 2``: halves, not adjacent
    pairs. The rotation runs in float32 whatever the inputs are, and the
    outputs carry the input dtype back.

    The reference applies this **before** the q/k norm. That order is the
    checkpoint's (``rope_after_norm: false``) and swapping it changes the
    result while leaving every shape intact.
    """
    orig_dtype = q.dtype
    q0, q1 = q.float().chunk(2, dim=-1)
    k0, k1 = k.float().chunk(2, dim=-1)
    seq_len, half = q0.shape[2], q0.shape[3]
    cos, sin = table.float().chunk(2, dim=-1)
    cos = cos.reshape(-1, 1, seq_len, half)
    sin = sin.reshape(-1, 1, seq_len, half)
    q_out = torch.stack([q0 * cos - q1 * sin, q1 * cos + q0 * sin], dim=-2)
    k_out = torch.stack([k0 * cos - k1 * sin, k1 * cos + k0 * sin], dim=-2)
    return q_out.flatten(-2).to(orig_dtype), k_out.flatten(-2).to(orig_dtype)
