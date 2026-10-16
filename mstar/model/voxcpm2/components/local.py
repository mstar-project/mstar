# Copyright (c) 2026 OpenBMB.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the per-patch modules batched
# over requests, with the CFM solve kept capturable in a CUDA graph.
#
# Original files (OpenBMB/VoxCPM, src/voxcpm/modules/locenc/local_encoder.py,
# src/voxcpm/modules/locdit/local_dit_v2.py,
# src/voxcpm/modules/locdit/unified_cfm.py and
# src/voxcpm/modules/layers/scalar_quantization_layer.py) were released under
# Apache-2.0, with the full license text available at
# https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""VoxCPM2's per-patch modules: local encoder, local DiT, CFM solver, FSQ.

Each LM step predicts one patch of ``patch_size`` latent frames. The local
DiT denoises that patch by flow matching, conditioned on the two LMs' hidden
states and on the previous patch; the local encoder folds a patch back into
one LM input embedding. Both are bidirectional MiniCPM-4 stacks over a few
tokens per row, so they run densely and batch over requests (the DiT over
``2 * requests`` rows, the conditional and unconditional CFG halves).

The arithmetic mirrors the reference ``VoxCPMLocEnc``, ``VoxCPMLocDiTV2`` and
``UnifiedCFM`` op for op, dtypes included: the timestep schedule and its
sinusoidal embedding are computed in the activation dtype (bf16), as there.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mstar.model.voxcpm2.components.minicpm import MiniCPMLocalTransformer
from mstar.model.voxcpm2.config import MiniCPM4Config


class ScalarQuantization(nn.Module):
    """FSQ bottleneck on the base LM's audio-position hidden states (inference)."""

    def __init__(self, dim: int, latent_dim: int, scale: int):
        super().__init__()
        self.scale = scale
        self.in_proj = nn.Linear(dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.in_proj(hidden))
        return self.out_proj(torch.round(hidden * self.scale) / self.scale)


class LocalEncoder(nn.Module):
    """``[rows, patch, feat]`` latent patches -> ``[rows, hidden]`` (CLS token)."""

    def __init__(self, config: MiniCPM4Config, feat_dim: int, patch_size: int):
        super().__init__()
        self.special_token = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(feat_dim, config.hidden_size)
        self.encoder = MiniCPMLocalTransformer(config, max_len=patch_size + 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(patches)
        special = self.special_token.view(1, 1, -1).expand(x.shape[0], 1, -1).to(x.dtype)
        return self.encoder(torch.cat([special, x], dim=1))[:, 0]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class LocalDiT(nn.Module):
    """Velocity estimator over ``[mu (2 tokens), t, cond patch, x patch]``.

    ``mu`` is the concatenated LM-to-DiT and residual-to-DiT projections, read
    as two tokens of the DiT width.
    """

    def __init__(self, config: MiniCPM4Config, feat_dim: int, patch_size: int):
        super().__init__()
        hidden = config.hidden_size
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.cond_proj = nn.Linear(feat_dim, hidden)
        self.out_proj = nn.Linear(hidden, feat_dim)
        self.time_embeddings = SinusoidalPosEmb(hidden)
        self.time_mlp = TimestepEmbedding(hidden)
        self.delta_time_mlp = TimestepEmbedding(hidden)
        self.decoder = MiniCPMLocalTransformer(config, max_len=2 + 1 + 2 * patch_size)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """``x``/``cond``: ``[rows, feat, patch]``; ``mu``: ``[rows, 2*hidden]``;
        ``t``/``dt``: ``[rows]``. Returns the velocity ``[rows, feat, patch]``."""
        x = self.in_proj(x.transpose(1, 2))
        cond = self.cond_proj(cond.transpose(1, 2))
        prefix = cond.shape[1]
        t = self.time_mlp(self.time_embeddings(t).to(x.dtype))
        dt = self.delta_time_mlp(self.time_embeddings(dt).to(x.dtype))
        t = t + dt
        mu = mu.view(x.shape[0], -1, x.shape[-1])
        hidden = self.decoder(torch.cat([mu, t.unsqueeze(1), cond, x], dim=1))
        hidden = hidden[:, prefix + mu.shape[1] + 1 :]
        return self.out_proj(hidden).transpose(1, 2)


def cfm_t_span(n_timesteps: int, device, dtype, sway_sampling_coef: float = 1.0) -> torch.Tensor:
    """Sway-sampled ``1 -> 0`` schedule of ``UnifiedCFM.forward`` (in ``dtype``)."""
    t_span = torch.linspace(1, 0, n_timesteps + 1, device=device, dtype=dtype)
    return t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)


def solve_cfm(
    estimator: LocalDiT,
    noise: torch.Tensor,
    mu: torch.Tensor,
    cond: torch.Tensor,
    cfg_value: torch.Tensor,
    t_span: torch.Tensor,
    mean_mode: bool = False,
) -> torch.Tensor:
    """Euler solve with CFG-Zero* (``UnifiedCFM.solve_euler``), batched.

    ``noise``/``cond``: ``[rows, feat, patch]``; ``mu``: ``[rows, 2*hidden]``;
    ``cfg_value``: ``[rows]``. The unconditional half zeroes ``mu`` and keeps
    the patch condition. Graph-safe: fixed shapes, no host reads.
    """
    rows = noise.shape[0]
    x = noise
    t, dt = t_span[0], t_span[0] - t_span[1]
    zero_init_steps = max(1, int(len(t_span) * 0.04))
    mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
    cond_in = torch.cat([cond, cond], dim=0)
    scale = cfg_value.to(x.dtype).view(rows, 1, 1)
    for step in range(1, len(t_span)):
        if step <= zero_init_steps:
            dphi_dt = torch.zeros_like(x)
        else:
            x_in = torch.cat([x, x], dim=0)
            t_in = t.expand(2 * rows)
            dt_in = dt.expand(2 * rows) if mean_mode else torch.zeros_like(t_in)
            velocity = estimator(x_in, mu_in, t_in, cond_in, dt_in)
            pos, neg = velocity[:rows], velocity[rows:]
            dot = torch.sum(pos.reshape(rows, -1) * neg.reshape(rows, -1), dim=1, keepdim=True)
            norm = torch.sum(neg.reshape(rows, -1) ** 2, dim=1, keepdim=True) + 1e-8
            st_star = (dot / norm).view(rows, 1, 1)
            dphi_dt = neg * st_star + scale * (pos - neg * st_star)
        x = x - dt * dphi_dt
        t = t - dt
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return x
