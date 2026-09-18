"""Whisper-family log-mel front end, in torch.

Whisper, Qwen3-ASR's AuT and the Higgs audio tower all consume the same
features: a 16 kHz waveform, 25 ms Hann-windowed STFT at a 10 ms hop, a
Slaney-scaled mel filterbank (80 or 128 bins), ``log10``, an 8-unit dynamic
range clamp under the clip's maximum, then ``(x + 4) / 4``. This module
reproduces HF's ``WhisperFeatureExtractor`` torch path to ~1e-5 so a
natively served encoder sees the features its HF oracle was tested with,
without ``transformers`` in the serving path. It runs wherever the waveform
lives: on the CPU of the API-server data worker, or on the GPU inside an
encoder's forward (every op is fixed-shape for a fixed clip length, so it
captures into a CUDA graph).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _hertz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Slaney mel scale (linear below 1 kHz, logarithmic above), as in
    librosa and HF ``audio_utils.hertz_to_mel(mel_scale="slaney")``."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * np.asarray(freq, dtype=np.float64) / 200.0
    high = np.asarray(freq, dtype=np.float64) >= min_log_hertz
    mels = np.where(
        high,
        min_log_mel + np.log(np.maximum(np.asarray(freq, dtype=np.float64), 1e-12) / min_log_hertz) * logstep,
        mels,
    )
    return mels


def _mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    return np.where(mels >= min_log_mel, min_log_hertz * np.exp(logstep * (mels - min_log_mel)), freq)


def slaney_mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    sampling_rate: int,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
) -> np.ndarray:
    """Area-normalized triangular mel filters, ``(num_mel_filters,
    num_frequency_bins)`` in float64 — the transpose of HF's layout, so the
    forward is one ``filters @ power_spectrogram`` matmul."""
    fft_freqs = np.linspace(0.0, sampling_rate / 2.0, num_frequency_bins)
    mel_min = float(_hertz_to_mel(min_frequency))
    mel_max = float(_hertz_to_mel(max_frequency))
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    # Slaney normalization: each filter's area is 2 / bandwidth.
    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    filters *= np.expand_dims(enorm, 0)
    return filters.T


class LogMelSpectrogram(nn.Module):
    """``(B, samples)`` or ``(samples,)`` float waveform -> ``(B, n_mels,
    frames)`` log-mel features, ``frames = samples // hop_length``.

    Matches HF's ``_torch_extract_fbank_features``: ``torch.stft`` with
    ``center=True`` yields ``samples // hop + 1`` frames and the last is
    dropped, and the dynamic-range clamp is per clip. Whisper pads or
    truncates to ``chunk_length`` seconds *before* this transform; use
    :meth:`pad_or_trim` for that.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        sampling_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        filters = slaney_mel_filter_bank(1 + n_fft // 2, num_mel_bins, sampling_rate)
        # float32, as HF casts its float64 bank before the matmul
        self.register_buffer("mel_filters", torch.from_numpy(filters).to(torch.float32), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True), persistent=False)

    def pad_or_trim(self, waveform: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """Zero-pad or truncate the last dim to ``length`` samples (the 30 s
        window by default), as Whisper's extractor does before the STFT."""
        length = self.n_samples if length is None else length
        n = waveform.shape[-1]
        if n > length:
            return waveform[..., :length]
        if n < length:
            return torch.nn.functional.pad(waveform, (0, length - n))
        return waveform

    def num_frames(self, num_samples: int) -> int:
        return num_samples // self.hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32)
        stft = torch.stft(
            waveform, self.n_fft, self.hop_length, window=self.window, return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.squeeze(0) if squeeze else log_spec

    def extra_repr(self) -> str:
        return (
            f"num_mel_bins={self.num_mel_bins}, sampling_rate={self.sampling_rate}, "
            f"n_fft={self.n_fft}, hop_length={self.hop_length}, chunk_length={self.chunk_length}"
        )


def load_audio_file(path: str, sample_rate: int = 16_000) -> torch.Tensor:
    """Decode an audio file to a float32 mono waveform at ``sample_rate``.

    libsndfile (``soundfile``) first — it needs no FFmpeg and covers WAV,
    FLAC, OGG and MP3 — then torchcodec for anything else (M4A, WebM, ...).
    Resampling goes through torchaudio.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).mean(dim=1)
    except Exception:  # noqa: BLE001 — not a libsndfile container; try FFmpeg
        from torchcodec.decoders import AudioDecoder

        frames = AudioDecoder(path, sample_rate=sample_rate, num_channels=1).get_all_samples()
        return frames.data[0].to(torch.float32)
    if sr != sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.contiguous()


def sinusoid_positions(length: int, channels: int, max_timescale: float = 10_000.0) -> torch.Tensor:
    """Whisper's fixed sinusoidal position table, ``(length, channels)``:
    ``[sin | cos]`` halves over geometrically spaced timescales. The
    checkpoint ships it as ``embed_positions.weight``; this is for models
    (Qwen3-ASR's AuT) that compute it instead."""
    if channels % 2:
        raise ValueError(f"channels must be even for sinusoid positions; got {channels}")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32))
    scaled_time = torch.arange(length, dtype=torch.float32)[:, None] * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
