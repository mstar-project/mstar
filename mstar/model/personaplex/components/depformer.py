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
# Adapted for M* on 2026-10-01 from NVIDIA/personaplex models/lm.py and
# modules/transformer.py at 3428dfd95309a7f3c84fd93259ded0f810d1ff91:
# native per-step-weight depformer.
"""Native PersonaPlex depformer ([PP] models/lm.py:319-352, 457-498, 1129-1178).

Within one 80 ms frame the depformer walks 16 audio codebooks. Step ``s``
uses slot ``s`` of every per-step weight: ``depformer_in[s]``, the attention
``in_proj``/``out_proj`` blocks and ``gating[s]`` ([PP] modules/transformer.py:
176-201, 412-447, 575-597). Its input token is the text token at step 0 and
the previous codebook's token afterwards (forced when provided, else
sampled). There is no positional embedding, and NVIDIA creates a fresh
16-slot ring cache for every frame ([PP] models/lm.py:1142), so state resets
per frame. The ring's full-buffer quirk masks step 0's key at step 15
(``depth_attention_start``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DEP_Q, LMArchConfig
from .temporal import ActivationGating, RMSNormF32, ScaledEmbedding


def depth_attention_start(step: int, capacity: int = DEP_Q) -> int:
    """First depth step visible to ``step`` in NVIDIA's ring KV.

    With one key per step, ``RingKVCache.complete`` labels the slot at
    ``end_offset % capacity`` as position ``end_offset`` once the ring is full
    ([PP] modules/transformer.py:263-299), so at ``step = capacity - 1`` the
    key of step 0 is masked. Executed check: worklog §11 3.1-a.
    """
    return max(0, step + 2 - capacity)


class DepformerAttention(nn.Module):
    """Per-step-weight MHA ([PP] modules/transformer.py:356-368, 412-447)."""

    def __init__(self, dim: int, num_heads: int, steps: int):
        super().__init__()
        self.num_heads = num_heads
        self.steps = steps
        self.in_proj_weight = nn.Parameter(torch.empty(steps * 3 * dim, dim))
        self.out_proj = nn.Linear(dim, steps * dim, bias=False)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor, step: int, kv: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, D]``; ``kv`` is this layer's ``[B, 2, H, steps, Dh]`` scratch."""
        dim = x.shape[-1]
        weight = self.in_proj_weight.view(self.steps, 3 * dim, dim)[step]
        q, k, v = F.linear(x, weight).unflatten(-1, (3, self.num_heads, -1)).unbind(-3)
        kv[:, 0, :, step] = k
        kv[:, 1, :, step] = v
        # NVIDIA passes the entire 16-slot ring plus a validity mask to SDPA
        # ([PP] modules/transformer.py:263-299, 426-434). Slicing valid keys is
        # mathematically equivalent but changes BF16 kernel rounding: trained
        # depth logits differed by 0.125 and could change argmax. Keep the
        # source's physical cache shape as well as its visibility semantics.
        positions = torch.arange(self.steps, device=x.device)
        visible = (positions >= depth_attention_start(step, self.steps)) & (positions <= step)
        out = F.scaled_dot_product_attention(q[:, :, None], kv[:, 0], kv[:, 1], visible[None])[:, :, 0]
        return F.linear(out.flatten(-2), self.out_proj.weight.view(self.steps, dim, dim)[step])


class DepformerLayer(nn.Module):
    def __init__(self, config: LMArchConfig):
        super().__init__()
        dim = config.depformer_dim
        self.self_attn = DepformerAttention(dim, config.depformer_heads, DEP_Q)
        self.norm1 = RMSNormF32(dim, config.norm_eps)
        self.norm2 = RMSNormF32(dim, config.norm_eps)
        self.gating = nn.ModuleList([ActivationGating(dim, config.depformer_ffn_hidden) for _ in range(DEP_Q)])

    def forward(self, x: torch.Tensor, step: int, kv: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), step, kv)
        return x + self.gating[step](self.norm2(x))


class DepformerLayers(nn.Module):
    """Holds ``layers`` so checkpoint keys keep NVIDIA's ``depformer.layers.*``."""

    def __init__(self, config: LMArchConfig):
        super().__init__()
        self.layers = nn.ModuleList([DepformerLayer(config) for _ in range(config.depformer_layers)])


class Depformer(nn.Module):
    """Input projections, token embeddings, the 6-layer stack and 16 heads."""

    def __init__(self, config: LMArchConfig):
        super().__init__()
        self.config = config
        ddim = config.depformer_dim
        self.depformer_in = nn.ModuleList([nn.Linear(config.dim, ddim, bias=False) for _ in range(DEP_Q)])
        # The last codebook is never an input ([PP] models/lm.py:333-336).
        self.depformer_emb = nn.ModuleList([ScaledEmbedding(config.card + 1, ddim) for _ in range(DEP_Q - 1)])
        self.depformer_text_emb = ScaledEmbedding(config.text_card + 1, ddim)
        self.depformer = DepformerLayers(config)
        self.linears = nn.ModuleList([nn.Linear(ddim, config.card, bias=False) for _ in range(DEP_Q)])

    def new_scratch(self, batch: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Frame-local KV ``[layers, B, 2, H, 16, Dh]``; overwritten from step 0 every frame."""
        c = self.config
        return torch.zeros(
            c.depformer_layers, batch, 2, c.depformer_heads, DEP_Q, c.depformer_head_dim, dtype=dtype, device=device
        )

    def step(self, index: int, token: torch.Tensor, hidden: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
        """Logits ``[B, card]`` for codebook ``index`` ([PP] models/lm.py:457-498)."""
        x = self.depformer_in[index](hidden)
        emb = self.depformer_text_emb if index == 0 else self.depformer_emb[index - 1]
        x = x + emb(token)
        for layer_idx, layer in enumerate(self.depformer.layers):
            x = layer(x, index, scratch[layer_idx])
        return self.linears[index](x)

    def run_frame(
        self,
        text_token: torch.Tensor,
        hidden: torch.Tensor,
        target_audio: torch.Tensor,
        provided_audio: torch.Tensor,
        sample: Callable[[torch.Tensor, int], torch.Tensor],
        scratch: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Mirror ``LMGen.depformer_step`` ([PP] models/lm.py:1129-1178).

        ``text_token`` is the forced-or-sampled text ``[B]``; ``hidden`` the
        normed temporal output ``[B, D]``; ``target_audio``/``provided_audio``
        the cache target and mask for codebooks 1-16, ``[B, 16]``. ``sample``
        maps ``([B, card] logits, step)`` to ``[B]`` tokens. Returns the raw
        sampled tokens ``[B, 16]`` (forcing is applied to the next input only;
        the cache commit preserves forced values).
        """
        batch = text_token.shape[0]
        if scratch is None:
            scratch = self.new_scratch(batch, dtype=hidden.dtype, device=hidden.device)
        prev = text_token
        tokens, logits = [], []
        for index in range(DEP_Q):
            step_logits = self.step(index, prev, hidden, scratch)
            token = sample(step_logits, index)
            prev = torch.where(provided_audio[:, index], target_audio[:, index], token)
            tokens.append(token)
            if return_logits:
                logits.append(step_logits)
        tokens = torch.stack(tokens, dim=1)
        return (tokens, torch.stack(logits, dim=1)) if return_logits else tokens
