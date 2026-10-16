# Copyright (c) 2024 Alibaba Inc.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: reimplemented as an M* module
# with the same numerics, with block-causal chunk masks for streaming.
#
# Original files (FunAudioLLM/CosyVoice, cosyvoice/flow/DiT/dit.py and
# cosyvoice/flow/DiT/modules.py) were released under Apache-2.0, with the full
# license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE. Those files
# adapt the DiT of SWivid/F5-TTS, and the rotary helpers follow
# lucidrains/x-transformers (both MIT).
#
# This modified file is released under the same license.

"""CosyVoice3's flow-matching velocity estimator: an F5-TTS style DiT.

``x`` (noisy mel), ``cond`` (prompt mel, zero elsewhere), ``mu`` (token
condition) and the projected x-vector are concatenated per frame, projected to
the model width and given a causal convolutional position embedding; 22
AdaLN-zero blocks follow, then a modulated output norm and a projection back
to 80 mel bins.

Two quirks of the reference are kept on purpose:

* rotary positions are applied to the whole ``to_q``/``to_k`` projection
  before it is split into heads, rotating only its first ``dim_head`` (64)
  channels, i.e. only head 0 sees positions (x-transformers' partial rotary,
  interleaved pairs, computed in float32);
* attention takes a mask, not a length: a block-causal chunk mask when
  streaming (``chunk_attention_mask``), the key padding otherwise.

Ported from ``cosyvoice/flow/DiT/{dit,modules}.py`` and x-transformers'
``RotaryEmbedding`` / ``apply_rotary_pos_emb``. Parameter
paths match ``flow.pt`` under ``decoder.estimator.``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.cosyvoice3.config import DiTConfig


def chunk_attention_mask(valid: torch.Tensor, total: int, chunk: int, streaming: torch.Tensor) -> torch.Tensor:
    """``[B, 1, T, T]`` bool attention mask (True = attend).

    Keys past a row's ``valid`` frames are masked. A ``streaming`` row also
    masks every key past the end of the query's chunk, chunks counted from
    frame 0 (the reference's ``add_optional_chunk_mask`` with a static chunk
    size and all left chunks).
    """
    pos = torch.arange(total, device=valid.device)
    keys = pos[None, :] < valid[:, None]  # [B, T]
    block_end = (torch.div(pos, chunk, rounding_mode="floor") + 1) * chunk
    causal = pos[None, :] < block_end[:, None]  # [T, T]
    causal = causal[None] | ~streaming[:, None, None]
    return (causal & keys[:, None, :]).unsqueeze(1)


class _SinusEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device).float() * -(math.log(10000) / (half - 1)))
        emb = self.scale * t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_dim: int):
        super().__init__()
        self.time_embed = _SinusEmbedding(freq_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.time_embed(t).to(self.time_mlp[0].weight.dtype))


class CausalConvPositionEmbedding(nn.Module):
    """Two left-padded grouped convolutions with Mish."""

    def __init__(self, dim: int, kernel_size: int, groups: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(F.pad(x, (self.kernel_size - 1, 0)))
        x = self.conv2(F.pad(x, (self.kernel_size - 1, 0)))
        return x.transpose(1, 2)


class InputEmbedding(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.proj = nn.Linear(config.mel_dim * 2 + config.mu_dim + config.spk_dim, config.dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(config.dim, config.conv_pos_kernel, config.conv_pos_groups)

    def forward(self, x, cond, mu, spks) -> torch.Tensor:
        spks = spks[:, None, :].expand(-1, x.shape[1], -1)
        x = self.proj(torch.cat([x, cond, mu, spks], dim=-1))
        return self.conv_pos_embed(x) + x


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_partial_rotary(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``freqs.shape[-1]`` channels of ``t`` ``[B, T, C]`` in
    float32 (interleaved pairs); the rest pass through."""
    rot = freqs.shape[-1]
    head, tail = t[..., :rot].float(), t[..., rot:]
    head = head * freqs.cos() + _rotate_pairs(head) * freqs.sin()
    return torch.cat((head.to(t.dtype), tail), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, row_keep: torch.Tensor, freqs: torch.Tensor):
        b, t, _ = x.shape
        q = apply_partial_rotary(self.to_q(x), freqs)
        k = apply_partial_rotary(self.to_k(x), freqs)
        v = self.to_v(x)
        q, k, v = (y.view(b, t, self.heads, -1).transpose(1, 2) for y in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, t, -1).to(q.dtype)
        return self.to_out[0](out) * row_keep


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(F.silu(emb)).chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(emb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        inner = dim * mult
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner), nn.GELU(approximate="tanh")),
            nn.Dropout(0.0),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class DiTBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(config.dim)
        self.attn = Attention(config.dim, config.heads, config.dim_head)
        self.ff_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=1e-6)
        self.ff = _FeedForward(config.dim, config.ff_mult)

    def forward(self, x, t, attn_mask, row_keep, freqs):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm, attn_mask, row_keep, freqs)
        ff_in = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp.unsqueeze(1) * self.ff(ff_in)


class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.time_embed = TimestepEmbedding(config.dim, config.time_freq_dim)
        self.input_embed = InputEmbedding(config)
        self.transformer_blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.depth)])
        self.norm_out = AdaLayerNormZeroFinal(config.dim)
        self.proj_out = nn.Linear(config.dim, config.mel_dim)
        self.register_buffer("inv_freq", self._inv_freq(torch.device("cpu")), persistent=False)

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        dim_head = self.config.dim_head
        return 1.0 / (10000 ** (torch.arange(0, dim_head, 2, device=device).float() / dim_head))

    def _apply(self, fn, recurse=True):
        # a half-precision estimator keeps float32 rotary frequencies: rounded ones
        # shift late positions' phases by up to a radian
        super()._apply(fn, recurse)
        if self.inv_freq.dtype in (torch.bfloat16, torch.float16):
            self.inv_freq = self._inv_freq(self.inv_freq.device)
        return self

    def rotary_freqs(self, seq_len: int) -> torch.Tensor:
        """``[1, T, dim_head]``: each frequency repeated for its pair."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.stack((freqs, freqs), dim=-1).flatten(-2)[None]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """``x, mu, cond``: ``[B, 80, T]``; ``mask``: ``[B, 1, T, T]`` attention
        mask (``chunk_attention_mask``; bool, or 0/1 in any float dtype);
        ``t``: ``[B]``; ``spks``: ``[B, 80]``.

        Runs in the weights' dtype and returns the inputs' dtype; ``t`` keeps
        its own precision through the sinusoid, so a bfloat16 estimator is fed
        the solve's float32 time grid, not a rounded one."""
        out_dtype, dtype = x.dtype, self.proj_out.weight.dtype
        x, mu, cond, spks = (v.to(dtype) for v in (x, mu, cond, spks))
        keep = mask if mask.dtype == torch.bool else mask > 0.5
        # a padded query row: its last-row key mask is the row's valid frames
        row_keep = keep[:, 0, -1, :].unsqueeze(-1).to(dtype)
        x, mu, cond = x.transpose(1, 2), mu.transpose(1, 2), cond.transpose(1, 2)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        temb = self.time_embed(t)
        h = self.input_embed(x, cond, mu, spks)
        freqs = self.rotary_freqs(h.shape[1])
        for block in self.transformer_blocks:
            h = block(h, temb, keep, row_keep, freqs)
        return self.proj_out(self.norm_out(h, temb)).transpose(1, 2).to(out_dtype)
