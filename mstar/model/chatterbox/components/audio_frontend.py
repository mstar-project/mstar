"""Waveform front-end shared by the Chatterbox conditioning encoders.

Everything here runs in torch on whatever device the waveform lives on, so
the reference-audio path stays on the GPU node instead of round-tripping
through numpy/librosa as the reference package does. The numerics mirror the
librosa calls the reference makes (``librosa.filters.mel`` with the slaney
scale and norm, ``librosa.stft`` centred with reflect padding,
``librosa.effects.trim``), and the tests pin them against librosa.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale (librosa ``hz_to_mel`` with ``htk=False``)."""
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    mels = np.where(
        log_region,
        min_log_mel + np.log(np.maximum(freq, 1e-12) / min_log_hz) / logstep,
        mels,
    )
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def mel_filter_bank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None,
) -> torch.Tensor:
    """``[n_mels, n_fft // 2 + 1]`` triangular filters, slaney-normalised.

    Reproduces ``librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)`` with its
    defaults (``htk=False``, ``norm="slaney"``, float32 output), including the
    order of the float32 casts, so the result matches the filter banks the
    reference package computed and the one stored in the checkpoint.
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=np.float64)
    mel_pts = np.linspace(_hz_to_mel(np.array(fmin, dtype=np.float64)),
                          _hz_to_mel(np.array(fmax, dtype=np.float64)), n_mels + 2)
    mel_f = _mel_to_hz(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return torch.from_numpy(weights)


def stft_power(
    wav: torch.Tensor, n_fft: int, hop: int, win: int, window: torch.Tensor, *, center: bool = True,
) -> torch.Tensor:
    """``|STFT|^2`` of ``wav [B, T]`` -> ``[B, n_fft // 2 + 1, frames]`` the way
    librosa computes it (centred, reflect-padded, periodic Hann window)."""
    spec = torch.stft(
        wav, n_fft, hop_length=hop, win_length=win, window=window,
        center=center, pad_mode="reflect", return_complex=True,
    )
    return spec.real.square() + spec.imag.square()


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    """Sinc resampling with torchaudio's defaults, which is what the reference
    S3Gen uses (``torchaudio.transforms.Resample``) for its 24 kHz -> 16 kHz
    step. (The reference T3 side resamples with librosa/soxr instead; the two
    differ at the 1e-3 level, see the tests.)"""
    if orig_sr == new_sr:
        return wav
    return AF.resample(wav, orig_sr, new_sr)


def trim_reference(wav: torch.Tensor, sample_rate: int, seconds: float) -> torch.Tensor:
    """Keep the first ``seconds`` of a reference clip (last dim is time)."""
    n = int(round(seconds * sample_rate))
    return wav[..., :n]


def trim_silence(
    wav: torch.Tensor, top_db: float = 20.0, frame_length: int = 2048, hop_length: int = 512,
) -> torch.Tensor:
    """Drop leading and trailing frames quieter than ``top_db`` below the peak.

    Mirrors ``librosa.effects.trim``: zero-padded, centred RMS frames, decibels
    relative to the loudest frame (``amplitude_to_db`` with ``amin=1e-5``), the
    kept span running from the first non-silent frame to one past the last.
    ``wav`` is 1-D.
    """
    T = wav.shape[-1]
    padded = F.pad(wav.float().unsqueeze(0), (frame_length // 2, frame_length // 2)).squeeze(0)
    if padded.shape[-1] < frame_length:
        return wav
    frames = padded.unfold(0, frame_length, hop_length)  # [n_frames, frame_length]
    rms = frames.square().mean(dim=-1).sqrt()
    amin = 1e-5
    power = torch.clamp(rms, min=amin).square()
    ref = torch.clamp(rms.max(), min=amin).square()
    db = 10.0 * torch.log10(power) - 10.0 * torch.log10(ref)
    non_silent = torch.nonzero(db > -top_db).flatten()
    if non_silent.numel() == 0:
        return wav[..., :0]
    start = int(non_silent[0]) * hop_length
    end = min(T, (int(non_silent[-1]) + 1) * hop_length)
    return wav[..., start:end]
