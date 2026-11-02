"""Waveform front-end shared by the Chatterbox conditioning encoders.

Everything here runs in torch on whatever device the waveform lives on, so
the reference-audio path stays on the GPU node instead of round-tripping
through numpy/librosa as the reference package does. The numerics mirror the
librosa calls the reference makes (``librosa.stft`` centred with reflect
padding, ``librosa.effects.trim``; the slaney mel filters are
``audio_features.librosa_mel_filters``), and the tests pin them against librosa.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torchaudio.compliance import kaldi


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


def kaldi_fbank_80(wav16: torch.Tensor, num_mel_bins: int = 80) -> torch.Tensor:
    """Mean-normalised Kaldi filter bank of one 16 kHz utterance, ``[T', bins]``:
    the CAMPPlus speaker-encoder feature (reference
    ``chatterbox/models/s3gen/xvector.py:extract_feature``).

    ``wav16`` is ``[T]`` or ``[1, T]``; the per-utterance mean over time is
    subtracted, as the speaker encoder's reference feature extractor does.
    """
    if wav16.dim() == 1:
        wav16 = wav16[None]
    feature = kaldi.fbank(wav16, num_mel_bins=num_mel_bins)
    return feature - feature.mean(dim=0, keepdim=True)
