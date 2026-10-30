# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Adapted for M* on 2026-10-01 from NVIDIA/personaplex models/lm.py,
# modules/transformer.py, modules/gating.py and modules/rope.py at
# 3428dfd95309a7f3c84fd93259ded0f810d1ff91: native temporal transformer.
"""Native PersonaPlex temporal transformer ([PP] models/lm.py:191-215, 425-455).

Parameter names follow NVIDIA's ``LMModel`` so the checkpoint maps by prefix
(``weights.py``). Attention and RoPE go through a backend so the same layer
code runs on CPU against a dense reference cache (``DenseTemporalCache``) and,
in the submodule, on M*'s paged KV and position resources.

Numerics follow the reference: fp32 RMSNorm with eps 1e-8 and ``(1, 1, D)``
alpha; adjacent-pair RoPE in fp32; SiLU gating whose first half is activated;
bias-free projections; summed embeddings where token -1 is an exact zero.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CONTEXT_FRAMES, MAX_TEMPORAL_FORWARDS, NUM_AUDIO_CODEBOOKS, ZERO_TOKEN_ID, LMArchConfig


class RMSNormF32(nn.Module):
    """``rms_norm_f32`` ([PP] modules/transformer.py:55-87, 141-143)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        var = self.eps + torch.mean(x**2, dim=-1, keepdim=True)
        return (x * (self.alpha.view(-1).to(var) * torch.rsqrt(var))).to(x_dtype)


class ScaledEmbedding(nn.Embedding):
    """Embedding whose ``-1`` input is an exact zero vector ([PP] models/lm.py:191-215)."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        y = super().forward(ids.clamp(min=0))
        return torch.where((ids == ZERO_TOKEN_ID)[..., None], torch.zeros((), dtype=y.dtype, device=y.device), y)


class ActivationGating(nn.Module):
    """SiLU gating FFN; the first half of ``linear_in`` is activated ([PP] modules/gating.py:33-70)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x).unflatten(-1, (2, -1))
        return self.linear_out(F.silu(x[..., 0, :]) * x[..., 1, :])


def apply_rope_adjacent(
    q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, max_period: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate adjacent pairs ``(2i, 2i+1)`` in fp32 ([PP] modules/rope.py:31-88).

    ``q``/``k`` are ``[..., T, H, D]`` with ``positions`` broadcastable to
    ``[..., T]``. This is FlashInfer's ``interleave=True`` layout.
    """
    D = q.shape[-1]
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    angles = positions.to(torch.float32)[..., None, None] * freqs  # [..., T, 1, D/2]
    cos, sin = torch.cos(angles), torch.sin(angles)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x.unflatten(-1, (D // 2, 2))
        xr, xi = pairs[..., 0].float(), pairs[..., 1].float()
        out = torch.stack(((xr * cos - xi * sin).to(x.dtype), (xr * sin + xi * cos).to(x.dtype)), dim=-1)
        return out.flatten(-2)

    return rotate(q), rotate(k)


class TemporalAttentionBackend(Protocol):
    """Where a temporal layer's RoPE positions and KV live."""

    def rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


class DenseTemporalCache:
    """Pure-torch reference backend for ``[B, T, H, D]`` (or token-major ``[B, H, D]``) inputs.

    Keeps occupied keys and pads SDPA inputs to NVIDIA's physical 3000-slot
    ring ([PP] modules/transformer.py:263-299, 426-434). A shorter key tensor
    has the same visibility but changes floating-point kernel rounding.
    Valid for at most ``MAX_TEMPORAL_FORWARDS`` positions, before the ring
    expires its first key. For reference comparisons, not serving.
    """

    def __init__(self, num_layers: int, rope_theta: float):
        self.rope_theta = rope_theta
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self.offset = 0
        self._pending = 0

    def rope(self, q, k):
        if q.dim() == 3:  # token-major, one token per request
            q, k = self.rope(q[:, None], k[:, None])
            return q[:, 0], k[:, 0]
        T = q.shape[-3]
        self._pending = T
        positions = torch.arange(self.offset, self.offset + T, device=q.device)
        return apply_rope_adjacent(q, k, positions, self.rope_theta)

    def attend(self, layer_idx, q, k, v):
        if q.dim() == 3:
            return self.attend(layer_idx, q[:, None], k[:, None], v[:, None])[:, 0]
        if self.keys[layer_idx] is not None:
            k = torch.cat((self.keys[layer_idx], k), dim=-3)
            v = torch.cat((self.values[layer_idx], v), dim=-3)
        T, S = q.shape[-3], k.shape[-3]
        if S > MAX_TEMPORAL_FORWARDS:
            raise ValueError(f"dense reference cache supports at most {MAX_TEMPORAL_FORWARDS} positions")
        self.keys[layer_idx], self.values[layer_idx] = k, v
        pos_q = torch.arange(S - T, S, device=q.device).view(-1, 1)
        mask = torch.arange(CONTEXT_FRAMES, device=q.device).view(1, -1) <= pos_q
        keys = F.pad(k.transpose(-3, -2), (0, 0, 0, CONTEXT_FRAMES - S))
        values = F.pad(v.transpose(-3, -2), (0, 0, 0, CONTEXT_FRAMES - S))
        out = F.scaled_dot_product_attention(q.transpose(-3, -2), keys, values, mask)
        return out.transpose(-3, -2)

    def advance(self) -> None:
        """Commit the forward's positions, once per forward after every layer."""
        self.offset += self._pending
        self._pending = 0


class TemporalAttention(nn.Module):
    """Bias-free MHA with a fused ``[q; k; v]`` in-projection ([PP] modules/transformer.py:317-455)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor, layer_idx: int, backend: TemporalAttentionBackend) -> torch.Tensor:
        # "b t (p h d) -> p b h t d": the projection is [q; k; v], each head-major.
        q, k, v = F.linear(x, self.in_proj_weight).unflatten(-1, (3, self.num_heads, -1)).unbind(-3)
        q, k = backend.rope(q, k)
        out = backend.attend(layer_idx, q, k, v)
        return self.out_proj(out.flatten(-2))


class TemporalLayer(nn.Module):
    """Pre-norm attention and gating blocks ([PP] modules/transformer.py:575-616)."""

    def __init__(self, config: LMArchConfig):
        super().__init__()
        self.self_attn = TemporalAttention(config.dim, config.num_heads)
        self.norm1 = RMSNormF32(config.dim, config.norm_eps)
        self.norm2 = RMSNormF32(config.dim, config.norm_eps)
        self.gating = ActivationGating(config.dim, config.ffn_hidden)

    def forward(self, x: torch.Tensor, layer_idx: int, backend: TemporalAttentionBackend) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), layer_idx, backend)
        return x + self.gating(self.norm2(x))


class TemporalLayers(nn.Module):
    """Holds ``layers`` so checkpoint keys keep NVIDIA's ``transformer.layers.*``."""

    def __init__(self, config: LMArchConfig):
        super().__init__()
        self.layers = nn.ModuleList([TemporalLayer(config) for _ in range(config.num_layers)])


class TemporalTransformer(nn.Module):
    """Embeddings, the 32-layer stack, ``out_norm`` and the text head."""

    def __init__(self, config: LMArchConfig):
        super().__init__()
        self.config = config
        self.emb = nn.ModuleList([ScaledEmbedding(config.card + 1, config.dim) for _ in range(NUM_AUDIO_CODEBOOKS)])
        self.text_emb = ScaledEmbedding(config.text_card + 1, config.dim)
        self.transformer = TemporalLayers(config)
        self.out_norm = RMSNormF32(config.dim, config.norm_eps)
        # existing_text_padding_id is set, so there is no extra output row ([PP] models/lm.py:293-296).
        self.text_linear = nn.Linear(config.dim, config.text_card, bias=False)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, 17, T]`` tokens -> ``[B, T, D]``, summed in NVIDIA's order ([PP] models/lm.py:425-439)."""
        if codes.dim() != 3 or codes.shape[1] != NUM_AUDIO_CODEBOOKS + 1:
            raise ValueError(f"expected [B, {NUM_AUDIO_CODEBOOKS + 1}, T] codes, got {tuple(codes.shape)}")
        x = None
        for index, emb in enumerate(self.emb):
            y = emb(codes[:, index + 1])
            x = y if x is None else x + y
        return x + self.text_emb(codes[:, 0])

    def forward_embeddings(
        self, x: torch.Tensor, backend: TemporalAttentionBackend, *, text_logits: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the normed hidden state fed to the depformer and the text logits ([PP] models/lm.py:447-455).

        ``text_logits=False`` skips the 32000-way head for fully forced
        (conditioning) frames, whose samples the cache discards.
        """
        for layer_idx, layer in enumerate(self.transformer.layers):
            x = layer(x, layer_idx, backend)
        x = self.out_norm(x)
        return x, self.text_linear(x) if text_logits else None
