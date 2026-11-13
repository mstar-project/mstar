"""Praxis-1 decoder tower for a chunk-causal noised window.

The decoder expands channel patches into a full spatial grid, conditions each
trunk block on the encoder output, then patches the predicted velocity back.
The attention callable and mask are supplied by the caller until the engine
has a chunk-causal attention resource.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from mstar.model.praxis1.components.attention import sdpa_attend
from mstar.model.praxis1.components.chunk_mask import encoder_codes, visible
from mstar.model.praxis1.components.encoder import (
    VIDEO,
    TrunkStack,
    VideoInputProjection,
    chunk_rows,
    modulate,
)
from mstar.model.praxis1.components.norm_time import TimestepEmbedding, ZeroCenteredRMSNorm
from mstar.model.praxis1.components.rope import video_positions


def unpatch(x: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """Move a channel patch into each spatial grid cell."""
    b, c, t, h, w = x.shape
    ph, pw = patch_size
    if c % (ph * pw):
        raise ValueError(f"{c} channels do not divide into {ph}x{pw} patches")
    x = x.view(b, c // (ph * pw), ph, pw, t, h, w)
    return x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, -1, t, h * ph, w * pw)


def patch(x: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """Move each spatial patch back into channels."""
    b, c, t, hh, ww = x.shape
    ph, pw = patch_size
    if hh % ph or ww % pw:
        raise ValueError(f"grid {(hh, ww)} does not divide into {ph}x{pw} patches")
    x = x.view(b, c, t, hh // ph, ph, ww // pw, pw)
    x = x.permute(0, 1, 4, 6, 2, 3, 5)
    return x.reshape(b, c * ph * pw, t, hh // ph, ww // pw)


class DecoderOutputProjection(nn.Module):
    """Normalize decoder tokens and project them to latent channels."""

    def __init__(self, dim: int, c_out: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(ZeroCenteredRMSNorm(dim), nn.Linear(dim, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(torch.cat([x], dim=1))
        return torch.split(y, [y.shape[1]], dim=1)[0]


class DecoderTower(nn.Module):
    """Decode a noised window with per-block encoder conditioning."""

    def __init__(
        self,
        core: TrunkStack,
        dim: int = 3072,
        video_dim: int = 48,
        time_dim: int = 128,
        num_registers: int = 4,
        patch_size: tuple[int, int] = (2, 2),
        positive_zero_residual_add: bool = True,
    ) -> None:
        super().__init__()
        if core.modulators is None or not core.modulators:
            raise ValueError("the decoder core needs per-block modulators")
        self.registers = nn.Parameter(torch.zeros(1, num_registers, dim))
        self.t_embedding = TimestepEmbedding(time_dim)
        self.t_proj = nn.ModuleDict({VIDEO: nn.Linear(time_dim, 2 * dim)})
        self.input_projections = nn.ModuleDict({VIDEO: VideoInputProjection(video_dim, dim)})
        self.output_projections = nn.ModuleDict({VIDEO: DecoderOutputProjection(dim, video_dim)})
        self.core = core
        self.patch_size = patch_size
        self.positive_zero_residual_add = positive_zero_residual_add

    def forward_chunk_causal(
        self,
        noised: torch.Tensor,
        cond: torch.Tensor,
        timesteps: torch.Tensor,
        rope_shift: int = 0,
        mask: torch.Tensor | None = None,
        attend: Callable = sdpa_attend,
    ) -> torch.Tensor:
        """Predict velocity for ``K`` chunks; ``cond`` is the encoder grid.

        ``noised`` is ``(B, video_dim, K*F, h, w)`` at patch resolution.
        ``cond`` is ``(B, K*F, h, w, dim)``. The sequence is
        ``[registers | noised]`` at full spatial resolution.
        """
        ph, pw = self.patch_size
        tok = self.input_projections[VIDEO](unpatch(noised, self.patch_size))
        shape = tok.shape
        batch, frames, height, width, _ = shape
        kk = int(timesteps.shape[1])
        if not kk or frames % kk:
            raise ValueError(f"{frames} noised frames do not split into {kk} chunks")
        mod_dim = self.core.modulators[0].in_features
        if cond.shape != (batch, frames, height // ph, width // pw, mod_dim):
            raise ValueError(f"conditioning shape {tuple(cond.shape)} does not match noised grid")
        chunk_tokens = (frames // kk) * height * width
        shift, scale = self.t_proj[VIDEO](self.t_embedding(timesteps)).chunk(2, dim=-1)
        flat = modulate(
            tok.flatten(1, -2), chunk_rows(shift, [chunk_tokens] * kk), chunk_rows(1 + scale, [chunk_tokens] * kk)
        )
        x = torch.cat([self.registers, flat], dim=1)

        b, cf, h, w, c = cond.shape
        up = cond[:, :, :, None, :, None, :].expand(b, cf, h, ph, w, pw, c)
        up = up.reshape(b, cf * h * ph * w * pw, c)
        n_reg = self.registers.size(1)
        pad = torch.zeros(b, n_reg, c, device=up.device, dtype=up.dtype)
        x_mod = torch.cat([pad, up], dim=1)
        positions = (
            torch.cat(
                [
                    torch.zeros(n_reg, dtype=torch.int64),
                    video_positions(frames, height, width, start=rope_shift),
                ]
            )
            .unsqueeze(0)
            .to(x.device)
        )
        if mask is None:
            codes = encoder_codes(n_reg, 0, 0, chunk_tokens, kk, 0)
            mask = visible(codes).to(x.device)
        out, _ = self.core(x, positions, mask=mask, attend=attend, x_mod=x_mod)
        if self.positive_zero_residual_add:
            out = out + 0.0
        v = self.output_projections[VIDEO](out.narrow(1, n_reg, flat.size(1)))
        v = v.view(*shape[:-1], -1).movedim(-1, 1)
        return patch(v, self.patch_size)
