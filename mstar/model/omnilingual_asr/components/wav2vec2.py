# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified for M* in 2026: the fairseq2 wav2vec 2.0 encoder
# and CTC head that Omnilingual ASR runs, reimplemented in plain PyTorch with
# the same operations in the same order. Inference only: no masking,
# quantizer, dropout or LayerDrop.
#
# Original files (facebookresearch/fairseq2 v0.6.0, src/fairseq2/models/wav2vec2/
# {feature_extractor,frontend,position_encoder}.py, asr/model.py and
# src/fairseq2/models/transformer/) were released under the MIT license, with
# the full license text available at
# https://github.com/facebookresearch/fairseq2/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""The Omnilingual ASR CTC network: a wav2vec 2.0 encoder plus a linear head.

    waveform (N, S), normalized per utterance, 16 kHz
      -> 7 strided convs, each LayerNorm (in fp32) + GELU     (x320: 50 frames/s)
      -> LayerNorm, Linear 512 -> model_dim
      -> + GELU(grouped conv positional encoding)            (kernel 128, 16 groups)
      -> pre-norm Transformer encoder, final LayerNorm
      -> Linear model_dim -> vocab                            (CTC logits)

Module and parameter names follow the fairseq2 checkpoint (``encoder_frontend``,
``encoder``, ``final_proj``), so the state dict loads with only the positional
conv's weight norm folded. Every module here is plain torch with no M* imports,
so the parity harness can run it inside the reference's own environment.

Batches are right-padded. ``seq_lens`` gives each row's sample count; the
padding never reaches a valid frame: the convolutions only look backwards
from a valid output frame over valid samples, padded frames are zeroed before
the positional convolution, and attention keys past a row's length are masked.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

_ATTENTION_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class FeatureExtractionLayer(nn.Module):
    """Conv1d -> LayerNorm over channels (computed in fp32) -> GELU."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, stride: int, eps: float):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, bias=True)
        self.layer_norm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        dtype = x.dtype
        x = x.transpose(1, 2)
        x = F.layer_norm(
            x.float(),
            self.layer_norm.normalized_shape,
            self.layer_norm.weight.float(),
            self.layer_norm.bias.float(),
            self.layer_norm.eps,
        ).type(dtype)
        x = x.transpose(1, 2)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, layer_descs: list[tuple[int, int, int]], eps: float):
        super().__init__()
        layers = []
        in_dim = 1
        for out_dim, kernel_size, stride in layer_descs:
            layers.append(FeatureExtractionLayer(in_dim, out_dim, kernel_size, stride, eps))
            in_dim = out_dim
        self.layers = nn.Sequential(*layers)
        self.layer_descs = list(layer_descs)

    def output_length(self, num_samples: int) -> int:
        """Frames a row yields: ``floor((len - kernel) / stride) + 1`` per conv."""
        n = num_samples
        for _, kernel_size, stride in self.layer_descs:
            n = (n - kernel_size) // stride + 1
        return n

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # (N, S) -> (N, 1, S) -> (N, C, T) -> (N, T, C)
        return self.layers(waveforms.unsqueeze(1)).transpose(1, 2)


class PositionEncoder(nn.Module):
    """Adds ``GELU(conv(x))`` with a grouped, even-kernel conv (its last output
    column dropped, so the output keeps the input length). The checkpoint's
    weight norm (``weight_g``/``weight_v``) is folded into ``conv.weight`` at load."""

    def __init__(self, model_dim: int, kernel_size: int, num_groups: int):
        super().__init__()
        self.conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=True,
        )
        self.remove_pad = kernel_size % 2 == 0

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None) -> torch.Tensor:
        if frame_mask is not None:
            # padded frames must be zero, or they leak into the valid ones' encodings
            x = x * frame_mask.unsqueeze(-1).to(x.dtype)
        enc = self.conv(x.transpose(1, 2))
        if self.remove_pad:
            enc = enc[:, :, :-1]
        return x + F.gelu(enc).transpose(1, 2)


class Frontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_extractor = FeatureExtractor(cfg.feature_extractor_layer_descs, cfg.layer_norm_eps)
        self.post_extract_layer_norm = nn.LayerNorm(cfg.feature_dim, eps=cfg.layer_norm_eps)
        self.model_dim_proj = nn.Linear(cfg.feature_dim, cfg.model_dim)
        self.pos_encoder = PositionEncoder(cfg.model_dim, cfg.pos_conv_kernel_size, cfg.num_pos_conv_groups)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        # (N, S, M) -> (N, S, H, K) -> (N, H, S, K)
        q = self.q_proj(x).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        k = self.k_proj(x).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v = self.v_proj(x).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.output_proj(attn.transpose(1, 2).flatten(-2))


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, inner_dim: int):
        super().__init__()
        self.inner_proj = nn.Linear(model_dim, inner_dim)
        self.output_proj = nn.Linear(inner_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(F.gelu(self.inner_proj(x)))


class EncoderLayer(nn.Module):
    """Pre-norm Transformer layer (fairseq2 ``TransformerNormOrder.PRE``)."""

    def __init__(self, cfg):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(cfg.model_dim, eps=cfg.layer_norm_eps)
        self.self_attn = SelfAttention(cfg.model_dim, cfg.num_heads)
        self.ffn_layer_norm = nn.LayerNorm(cfg.model_dim, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(cfg.model_dim, cfg.ffn_inner_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.self_attn(self.self_attn_layer_norm(x), attn_mask) + x
        return self.ffn(self.ffn_layer_norm(x)) + x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))
        self.layer_norm = nn.LayerNorm(cfg.model_dim, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.layer_norm(x)


def attention_mask(frame_mask: torch.Tensor) -> torch.Tensor:
    """``(N, 1, T, T)`` boolean mask: a valid frame attends to its row's valid
    frames only. A padded frame attends to every frame, which keeps its
    (discarded) output finite instead of a fully masked NaN row."""
    return (frame_mask[:, None, :] | ~frame_mask[:, :, None]).unsqueeze(1)


class Wav2Vec2CTC(nn.Module):
    """Omnilingual ASR CTC model (fairseq2 ``Wav2Vec2AsrModel`` at inference)."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.encoder_frontend = Frontend(cfg)
        self.encoder = Encoder(cfg)
        self.final_proj = nn.Linear(cfg.model_dim, cfg.vocab_size)

    def frame_lengths(self, seq_lens: list[int]) -> list[int]:
        return [self.encoder_frontend.feature_extractor.output_length(n) for n in seq_lens]

    def forward(
        self,
        waveforms: torch.Tensor,
        seq_lens: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """``(N, S)`` waveforms (and each row's sample count, for a padded batch) ->
        ``(N, T, vocab)`` logits and each row's valid frame count.

        The lengths are host ints, so a padded batch is masked without reading
        anything back from the device."""
        front = self.encoder_frontend
        x = front.feature_extractor(waveforms)
        n, t = x.shape[:2]
        frame_lens = [t] * n if seq_lens is None else self.frame_lengths(seq_lens)
        frame_mask = None
        if any(f < t for f in frame_lens):
            lens = torch.tensor(frame_lens)
            if x.is_cuda:
                lens = lens.pin_memory()
            lens = lens.to(x.device, non_blocking=True)
            frame_mask = torch.arange(t, device=x.device)[None, :] < lens[:, None]
        x = front.model_dim_proj(front.post_extract_layer_norm(x))
        x = front.pos_encoder(x, frame_mask)
        # cuDNN attention builds a plan per new sequence length, and nearly
        # every upload has its own length
        with sdpa_kernel(_ATTENTION_BACKENDS):
            x = self.encoder(x, None if frame_mask is None else attention_mask(frame_mask))
        return self.final_proj(x), frame_lens
