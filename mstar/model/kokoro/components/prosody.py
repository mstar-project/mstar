"""Text encoder and prosody predictor (durations, F0 and energy) of Kokoro.

All blocks take per-row lengths and keep padded frames at zero, so a batch of
sentences of different lengths produces the same durations and prosody curves
as running each sentence alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.kokoro.components.adain import LEAKY_SLOPE, AdainResBlk1d
from mstar.model.kokoro.components.lstm import MaskedBiLSTM
from mstar.model.kokoro.components.masking import length_mask, mask_channels, mask_features
from mstar.model.kokoro.config import KokoroModelConfig

LAYER_NORM_EPS = 1e-5


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a ``[B, C, T]`` tensor (``gamma``/``beta``
    parameter names follow the checkpoint)."""

    def __init__(self, channels: int, eps: float = LAYER_NORM_EPS):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x.transpose(1, 2), (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)


class TextEncoder(nn.Module):
    """Phoneme embedding -> 3 x (conv, LayerNorm, LeakyReLU) -> BiLSTM, as ``[B, C, T]``."""

    def __init__(self, config: KokoroModelConfig):
        super().__init__()
        channels = config.hidden_dim
        padding = (config.text_encoder_kernel_size - 1) // 2
        self.embedding = nn.Embedding(config.n_token, channels)
        self.cnn = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, channels, config.text_encoder_kernel_size, padding=padding),
                ChannelLayerNorm(channels),
            )
            for _ in range(config.n_layer)
        )
        self.lstm = MaskedBiLSTM(channels, channels // 2)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = length_mask(lengths, input_ids.shape[1])
        x = mask_channels(self.embedding(input_ids).transpose(1, 2), mask)
        for conv, norm in self.cnn:
            x = mask_channels(F.leaky_relu(norm(conv(x)), LEAKY_SLOPE), mask)
        x = self.lstm(x.transpose(1, 2), lengths)
        return x.transpose(1, 2)


class AdaLayerNorm(nn.Module):
    """LayerNorm over features of ``[B, T, C]`` with style-derived scale and shift."""

    def __init__(self, style_dim: int, channels: int, eps: float = LAYER_NORM_EPS):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(style).chunk(2, dim=-1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma[:, None, :]) * x + beta[:, None, :]


class DurationEncoder(nn.Module):
    """3 x (BiLSTM, AdaLayerNorm) over [phoneme states ; style], ``[B, T, d_model + style]``."""

    def __init__(self, style_dim: int, d_model: int, nlayers: int):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(MaskedBiLSTM(d_model + style_dim, d_model // 2))
            self.lstms.append(AdaLayerNorm(style_dim, d_model))

    def forward(self, x: torch.Tensor, style: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = length_mask(lengths, x.shape[1])
        style_seq = style[:, None, :].expand(-1, x.shape[1], -1)
        x = mask_features(torch.cat([x, style_seq], dim=-1), mask)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = mask_features(torch.cat([block(x, style), style_seq], dim=-1), mask)
            else:
                x = block(x, lengths)
        return x


class ProsodyPredictor(nn.Module):
    """Duration, F0 and energy prediction from PL-BERT states and a style vector."""

    def __init__(self, config: KokoroModelConfig):
        super().__init__()
        d_hid, style_dim = config.hidden_dim, config.style_dim
        self.text_encoder = DurationEncoder(style_dim, d_hid, config.n_layer)
        self.lstm = MaskedBiLSTM(d_hid + style_dim, d_hid // 2)
        self.duration_proj = nn.Linear(d_hid, config.max_dur)
        self.shared = MaskedBiLSTM(d_hid + style_dim, d_hid // 2)
        self.F0 = self._curve_blocks(d_hid, style_dim)
        self.N = self._curve_blocks(d_hid, style_dim)
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1)

    @staticmethod
    def _curve_blocks(d_hid: int, style_dim: int) -> nn.ModuleList:
        return nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
        ])

    def encode(self, d_en: torch.Tensor, style: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``[B, T, hidden]`` phoneme states -> ``[B, T, hidden + style]`` prosody states."""
        return self.text_encoder(d_en, style, lengths)

    def durations(self, d: torch.Tensor, lengths: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Integer frames per phoneme, ``[B, T]``; zero on the padded tail."""
        x = self.lstm(d, lengths)
        duration = torch.sigmoid(self.duration_proj(x)).sum(dim=-1) / speed[:, None]
        pred_dur = torch.round(duration).clamp(min=1).long()
        return pred_dur.masked_fill(~length_mask(lengths, d.shape[1]), 0)

    def f0n(
        self, en: torch.Tensor, style: torch.Tensor, frame_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame-aligned prosody states ``[B, F, hidden + style]`` -> F0 and energy
        curves at twice the frame rate, ``[B, 2F]`` each."""
        x = self.shared(en, frame_lengths).transpose(1, 2)
        f0, lengths = x, frame_lengths
        for block in self.F0:
            f0, lengths = block(f0, style, lengths)
        energy, lengths = x, frame_lengths
        for block in self.N:
            energy, lengths = block(energy, style, lengths)
        mask = length_mask(lengths, f0.shape[-1])
        f0 = mask_channels(self.F0_proj(f0), mask).squeeze(1)
        energy = mask_channels(self.N_proj(energy), mask).squeeze(1)
        return f0, energy
