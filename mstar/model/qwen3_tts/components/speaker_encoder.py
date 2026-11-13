"""ECAPA-TDNN speaker encoder and its mel front end (Qwen3-TTS Base).

The Base checkpoint clones a voice from reference audio: a 128-bin log-mel
spectrogram of the 24 kHz reference goes through an ECAPA-TDNN encoder whose
2048-wide output (the "x-vector") takes the place of the built-in speaker tag
in the Talker prefill. Module names follow the ``speaker_encoder.*`` keys of
the checkpoint so weights stream in without remapping.

Both pieces are plain, batch-friendly PyTorch: the encoder is a stack of 1-D
convolutions with "same" reflect padding, and the front end is the shared
Matcha/CosyVoice log-mel (``audio_features.MatchaLogMel``).

Given each row's frame count, the encoder takes mels padded to one width (a
capture bucket) and gives each row its unpadded x-vector, up to kernel
choice: every conv reflects about the row's own last frame, and the
squeeze-excitation means and the attentive pooling read the row's frames only.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.audio_features import MatchaLogMel, reflect_pad_index
from mstar.model.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig

# ---------------------------------------------------------------------------
# Mel front end
# ---------------------------------------------------------------------------


def speaker_mel(config: Qwen3TTSSpeakerEncoderConfig) -> MatchaLogMel:
    """The log-mel ``extract_speaker_embedding`` computes of 24 kHz reference
    audio: ``[batch, samples]`` -> ``[batch, mel_dim, frames]``."""
    return MatchaLogMel(
        sample_rate=config.sample_rate, n_fft=config.n_fft, num_mels=config.mel_dim, hop_size=config.hop_size,
        win_size=config.win_size, fmin=config.fmin, fmax=config.fmax,
    )


# ---------------------------------------------------------------------------
# ECAPA-TDNN
# ---------------------------------------------------------------------------


class PaddedFrames(NamedTuple):
    """Which frames of a padded batch ``[B, C, T]`` are each row's own."""

    valid: torch.Tensor                   # [B, 1, T]: 1 on a row's frames, 0 past them
    lengths: torch.Tensor                 # [B, 1, 1] float32: frames per row
    reflect: dict[int, torch.Tensor]      # pad -> [B, T + 2 pad], reflect_pad_index


def padded_frames(lengths: torch.Tensor, width: int, dtype: torch.dtype, pads: set[int]) -> PaddedFrames:
    valid = torch.arange(width, device=lengths.device) < lengths.unsqueeze(1)
    return PaddedFrames(
        valid=valid.unsqueeze(1).to(dtype),
        lengths=lengths.view(-1, 1, 1).float(),
        reflect={pad: reflect_pad_index(lengths, width, pad) for pad in pads},
    )


class _SameReflectConv1d(nn.Module):
    """``nn.Conv1d(padding="same", padding_mode="reflect")`` with the checkpoint's ``conv`` name."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if (kernel_size - 1) * dilation % 2:
            raise ValueError("a same-padded conv needs an even receptive field to pad symmetrically")
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation,
            padding="same", padding_mode="reflect",
        )
        self.pad = (kernel_size - 1) * dilation // 2

    def forward(self, hidden_states: torch.Tensor, frames: PaddedFrames | None = None) -> torch.Tensor:
        if frames is None:
            return self.conv(hidden_states)
        if self.pad:
            index = frames.reflect[self.pad].unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            hidden_states = hidden_states.gather(2, index)
        return F.conv1d(hidden_states, self.conv.weight, self.conv.bias, dilation=self.conv.dilation)


class TimeDelayNetBlock(_SameReflectConv1d):
    """Conv1d + ReLU."""

    def forward(self, hidden_states: torch.Tensor, frames: PaddedFrames | None = None) -> torch.Tensor:
        return F.relu(super().forward(hidden_states, frames))


class Res2NetBlock(nn.Module):
    """Hierarchical residual convolutions over ``scale`` channel groups."""

    def __init__(self, in_channels: int, out_channels: int, scale: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.scale = scale
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(in_channels // scale, out_channels // scale, kernel_size, dilation)
            for _ in range(scale - 1)
        ])

    def forward(self, hidden_states: torch.Tensor, frames: PaddedFrames | None = None) -> torch.Tensor:
        outputs = []
        previous = None
        for index, part in enumerate(hidden_states.chunk(self.scale, dim=1)):
            if index == 0:
                previous = part
            elif index == 1:
                previous = self.blocks[0](part, frames)
            else:
                previous = self.blocks[index - 1](part + previous, frames)
            outputs.append(previous)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, 1)
        self.conv2 = nn.Conv1d(se_channels, out_channels, 1)

    def forward(self, hidden_states: torch.Tensor, frames: PaddedFrames | None = None) -> torch.Tensor:
        if frames is None:
            pooled = hidden_states.mean(dim=2, keepdim=True)
        else:
            # summed in float32 and rounded once, as ``mean`` does
            total = (hidden_states * frames.valid).sum(dim=2, keepdim=True, dtype=torch.float32)
            pooled = (total / frames.lengths).to(hidden_states.dtype)
        gate = torch.sigmoid(self.conv2(F.relu(self.conv1(pooled))))
        return hidden_states * gate


class SqueezeExcitationRes2NetBlock(nn.Module):
    """TDNN -> Res2Net -> TDNN -> squeeze-excitation, with a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, res2net_scale: int, se_channels: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, 1, 1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, 1, 1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_states: torch.Tensor, frames: PaddedFrames | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.tdnn1(hidden_states, frames)
        hidden_states = self.res2net_block(hidden_states, frames)
        hidden_states = self.tdnn2(hidden_states, frames)
        return self.se_block(hidden_states, frames) + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attention-weighted mean and standard deviation over time: ``[B, C, T] -> [B, 2C, 1]``."""

    def __init__(self, channels: int, attention_channels: int) -> None:
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.conv = nn.Conv1d(attention_channels, channels, 1)

    def _statistics(self, hidden_states: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (weights * hidden_states).sum(dim=2)
        variance = (weights * (hidden_states - mean.unsqueeze(2)).pow(2)).sum(dim=2)
        return mean, torch.sqrt(variance.clamp(min=self.eps))

    def forward(self, hidden_states: torch.Tensor, frames: PaddedFrames | None = None) -> torch.Tensor:
        width = hidden_states.shape[-1]
        if frames is None:
            uniform = torch.full_like(hidden_states[:, :1, :], 1.0 / width)
        else:
            uniform = (frames.valid / frames.lengths).to(hidden_states.dtype)
        mean, std = self._statistics(hidden_states, uniform)
        context = torch.cat([
            hidden_states,
            mean.unsqueeze(2).expand(-1, -1, width),
            std.unsqueeze(2).expand(-1, -1, width),
        ], dim=1)
        scores = self.conv(torch.tanh(self.tdnn(context, frames)))
        if frames is not None:
            scores = scores.masked_fill(frames.valid == 0, float("-inf"))
        attention = torch.softmax(scores, dim=2)
        mean, std = self._statistics(hidden_states, attention)
        return torch.cat([mean, std], dim=1).unsqueeze(2)


class Qwen3TTSSpeakerEncoder(nn.Module):
    """ECAPA-TDNN x-vector extractor: ``[batch, frames, mel_dim] -> [batch, enc_dim]``."""

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig) -> None:
        super().__init__()
        channels = config.enc_channels
        kernels = config.enc_kernel_sizes
        dilations = config.enc_dilations
        if not len(channels) == len(kernels) == len(dilations):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations must have the same length")
        self.blocks = nn.ModuleList([TimeDelayNetBlock(config.mel_dim, channels[0], kernels[0], dilations[0])])
        for index in range(1, len(channels) - 1):
            self.blocks.append(SqueezeExcitationRes2NetBlock(
                channels[index - 1], channels[index], config.enc_res2net_scale, config.enc_se_channels,
                kernels[index], dilations[index],
            ))
        self.mfa = TimeDelayNetBlock(channels[-1], channels[-1], kernels[-1], dilations[-1])
        self.asp = AttentiveStatisticsPooling(channels[-1], config.enc_attention_channels)
        self.fc = nn.Conv1d(channels[-1] * 2, config.enc_dim, 1)

        self._pads = {
            module.pad for module in self.modules() if isinstance(module, _SameReflectConv1d) and module.pad
        }

    def forward(self, mels: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``lengths`` ([B] frames): rows padded to one width, each encoded as its own first frames."""
        hidden_states = mels.transpose(1, 2)
        frames = None if lengths is None else padded_frames(
            lengths, hidden_states.shape[-1], hidden_states.dtype, self._pads,
        )
        features = []
        for block in self.blocks:
            hidden_states = block(hidden_states, frames)
            features.append(hidden_states)
        # Multi-layer feature aggregation over the SE-Res2Net outputs only.
        hidden_states = self.mfa(torch.cat(features[1:], dim=1), frames)
        return self.fc(self.asp(hidden_states, frames)).squeeze(-1)
