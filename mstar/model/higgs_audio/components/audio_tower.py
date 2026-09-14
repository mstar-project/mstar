"""Higgs-Audio audio tower + feature projector.

The checkpoint ships its own modeling code, but the layer call
convention differs between transformers 4 and 5 (4 takes a positional
``layer_head_mask`` and returns a tuple; 5 drops it and returns the
tensor). These are small modules, so they're reimplemented here and the
layer call is adapted to whichever signature is installed.

Tower = Whisper encoder (conv1 -> conv2(stride 2) -> sinusoidal
positions -> ``WhisperEncoderLayer`` stack) followed by an
``AvgPool1d(2)`` over time and a final LayerNorm — 25 embeddings/s.
Projector = depthwise ``Conv1d(stride 2)`` temporal downsample + 2-layer
ReLU MLP into LLM space — 12.5 embeddings/s. Parameter paths mirror the
checkpoint's ``audio_tower.*`` / ``audio_encoder_proj.*``.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from mstar.model.higgs_audio.config import HiggsAudioModelConfig

# transformers 4 requires a positional layer_head_mask; transformers 5 dropped it.
_LAYER_TAKES_HEAD_MASK = (
    "layer_head_mask" in inspect.signature(WhisperEncoderLayer.forward).parameters
)


class HiggsAudioTower(nn.Module):
    def __init__(self, config: HiggsAudioModelConfig):
        super().__init__()
        embed_dim = config.audio_d_model
        whisper_cfg = WhisperConfig(
            d_model=embed_dim,
            encoder_layers=config.audio_encoder_layers,
            encoder_attention_heads=config.audio_encoder_attention_heads,
            encoder_ffn_dim=config.audio_encoder_ffn_dim,
            num_mel_bins=config.audio_num_mel_bins,
            max_source_positions=config.audio_max_source_positions,
            activation_function="gelu",
            dropout=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
        )
        whisper_cfg._attn_implementation = "sdpa"

        self.conv1 = nn.Conv1d(config.audio_num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.audio_max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(whisper_cfg) for _ in range(config.audio_encoder_layers)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """(batch, num_mel_bins, T_mel) -> (batch, ~T_mel/4, d_model)."""
        hidden = F.gelu(self.conv1(input_features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)  # (B, T, D)
        hidden = hidden + self.embed_positions.weight[: hidden.shape[1]]
        for layer in self.layers:
            out = layer(hidden, None, None) if _LAYER_TAKES_HEAD_MASK else layer(hidden, None)
            hidden = out[0] if isinstance(out, tuple) else out  # tf4 returns (hidden, attn)
        hidden = self.avg_pooler(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        return self.layer_norm(hidden)


class HiggsAudioFeatureProjector(nn.Module):
    """"mlp" projector with stride-2 temporal downsample (the v3-stt
    config; the "linear" variant is not supported here)."""

    def __init__(self, config: HiggsAudioModelConfig):
        super().__init__()
        audio_dim = config.audio_d_model
        assert config.projector_temporal_downsample == 2, (
            f"Only stride-2 temporal downsample is supported; "
            f"got {config.projector_temporal_downsample}."
        )
        self.temporal = nn.Conv1d(
            audio_dim, audio_dim, 3, 2, padding=1, groups=audio_dim, bias=True,
        )
        self.linear1 = nn.Linear(audio_dim, 2048, bias=True)
        self.linear2 = nn.Linear(2048, config.hidden_size, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """(B, T, audio_dim) -> (B, ceil(T/2), hidden_size)."""
        x = self.temporal(audio_features.permute(0, 2, 1)).permute(0, 2, 1)
        return self.linear2(F.relu(self.linear1(x)))
