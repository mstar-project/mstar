"""ECAPA-TDNN speaker encoder and its mel front end (Qwen3-TTS Base).

The Base checkpoint clones a voice from reference audio: a 128-bin log-mel
spectrogram of the 24 kHz reference goes through an ECAPA-TDNN encoder whose
2048-wide output (the "x-vector") takes the place of the built-in speaker tag
in the Talker prefill. Module names follow the ``speaker_encoder.*`` keys of
the checkpoint so weights stream in without remapping.

Both pieces are plain, batch-friendly PyTorch: the encoder is a stack of 1-D
convolutions with "same" reflect padding, and the front end is an STFT plus a
Slaney-normalized mel filterbank (the filterbank ``librosa.filters.mel``
produces, computed here without the librosa dependency).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig

# ---------------------------------------------------------------------------
# Mel front end
# ---------------------------------------------------------------------------


def _hz_to_mel_slaney(freq: torch.Tensor) -> torch.Tensor:
    """Slaney mel scale: linear below 1 kHz, logarithmic above (librosa default)."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mel = freq / f_sp
    log_region = freq >= min_log_hz
    mel = torch.where(
        log_region,
        min_log_mel + torch.log(freq.clamp(min=min_log_hz) / min_log_hz) / logstep,
        mel,
    )
    return mel


def _mel_to_hz_slaney(mel: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    freq = f_sp * mel
    log_region = mel >= min_log_mel
    return torch.where(
        log_region, min_log_hz * torch.exp(logstep * (mel - min_log_mel)), freq
    )


def slaney_mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> torch.Tensor:
    """``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)``: ``[n_mels, n_fft // 2 + 1]``."""
    fft_freqs = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    mel_edges = torch.linspace(
        _hz_to_mel_slaney(torch.tensor(float(fmin), dtype=torch.float64)).item(),
        _hz_to_mel_slaney(torch.tensor(float(fmax), dtype=torch.float64)).item(),
        n_mels + 2,
        dtype=torch.float64,
    )
    hz_edges = _mel_to_hz_slaney(mel_edges)
    fdiff = hz_edges[1:] - hz_edges[:-1]
    ramps = hz_edges.unsqueeze(1) - fft_freqs.unsqueeze(0)  # [n_mels + 2, bins]
    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
    # Slaney normalization: each filter integrates to about the same area.
    enorm = 2.0 / (hz_edges[2:] - hz_edges[:-2])
    return (weights * enorm.unsqueeze(1)).to(torch.float32)


class Qwen3TTSMelFrontEnd(nn.Module):
    """Log-mel features of 24 kHz reference audio, as ``extract_speaker_embedding`` computes them.

    ``waveform`` is ``[batch, samples]`` in ``[-1, 1]``; the output is
    ``[batch, frames, mel_dim]`` in float32, ready for the encoder. Windows are
    Hann, frames are not centered but the signal is reflect-padded by
    ``(n_fft - hop) // 2`` on both sides, magnitudes get a ``1e-9`` floor and
    the mel energies are log-compressed with a ``1e-5`` clamp.
    """

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig) -> None:
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_size = config.hop_size
        self.win_size = config.win_size
        self.register_buffer(
            "mel_basis",
            slaney_mel_filterbank(config.sample_rate, config.n_fft, config.mel_dim, config.fmin, config.fmax),
            persistent=False,
        )
        self.register_buffer("window", torch.hann_window(config.win_size), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        padding = (self.n_fft - self.hop_size) // 2
        waveform = F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)


# ---------------------------------------------------------------------------
# ECAPA-TDNN
# ---------------------------------------------------------------------------


class _SameReflectConv1d(nn.Module):
    """``nn.Conv1d(padding="same", padding_mode="reflect")`` with the checkpoint's ``conv`` name."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation,
            padding="same", padding_mode="reflect",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states)


class TimeDelayNetBlock(_SameReflectConv1d):
    """Conv1d + ReLU."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(hidden_states))


class Res2NetBlock(nn.Module):
    """Hierarchical residual convolutions over ``scale`` channel groups."""

    def __init__(self, in_channels: int, out_channels: int, scale: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.scale = scale
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(in_channels // scale, out_channels // scale, kernel_size, dilation)
            for _ in range(scale - 1)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = []
        previous = None
        for index, part in enumerate(hidden_states.chunk(self.scale, dim=1)):
            if index == 0:
                previous = part
            elif index == 1:
                previous = self.blocks[0](part)
            else:
                previous = self.blocks[index - 1](part + previous)
            outputs.append(previous)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, 1)
        self.conv2 = nn.Conv1d(se_channels, out_channels, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=2, keepdim=True)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.tdnn1(hidden_states)
        hidden_states = self.res2net_block(hidden_states)
        hidden_states = self.tdnn2(hidden_states)
        return self.se_block(hidden_states) + residual


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        frames = hidden_states.shape[-1]
        uniform = torch.full_like(hidden_states[:, :1, :], 1.0 / frames)
        mean, std = self._statistics(hidden_states, uniform)
        context = torch.cat([
            hidden_states,
            mean.unsqueeze(2).expand(-1, -1, frames),
            std.unsqueeze(2).expand(-1, -1, frames),
        ], dim=1)
        attention = torch.softmax(self.conv(torch.tanh(self.tdnn(context))), dim=2)
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

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        hidden_states = mels.transpose(1, 2)
        features = []
        for block in self.blocks:
            hidden_states = block(hidden_states)
            features.append(hidden_states)
        # Multi-layer feature aggregation over the SE-Res2Net outputs only.
        hidden_states = self.mfa(torch.cat(features[1:], dim=1))
        return self.fc(self.asp(hidden_states)).squeeze(-1)
