"""Audio front ends for S3Gen's reference conditioning.

``MelSpectrogram24k`` is the Matcha/CosyVoice mel used to condition the flow
decoder (reference ``chatterbox/models/s3gen/utils/mel.py``); ``kaldi_fbank_80``
is the CAMPPlus speaker-encoder feature (reference
``chatterbox/models/s3gen/xvector.py:extract_feature``). Both take a waveform
tensor and run wherever it lives; the mel keeps its filter bank and window as
buffers so it can be captured in a CUDA graph.
"""

from __future__ import annotations

import torch
from torch import nn
from torchaudio.compliance import kaldi

from mstar.model.chatterbox.components.audio_frontend import mel_filter_bank
from mstar.model.chatterbox.config import S3GenMelConfig


class MelSpectrogram24k(nn.Module):
    """Log-mel spectrogram of a 24 kHz waveform, ``[B, T] -> [B, n_mels, F]``.

    Reflect-pads ``(n_fft - hop) / 2`` samples on both sides, takes an
    uncentered STFT, ``sqrt(|X|^2 + 1e-9)``, the librosa (Slaney) mel filter
    bank and ``log(max(x, 1e-5))``. A waveform whose length is a multiple of
    the hop yields exactly ``T / hop`` frames.
    """

    def __init__(self, config: S3GenMelConfig):
        super().__init__()
        self.config = config
        mel = mel_filter_bank(
            config.sample_rate, config.n_fft, config.num_mels, config.fmin, config.fmax,
        )
        self.register_buffer("mel_basis", mel, persistent=False)
        self.register_buffer("window", torch.hann_window(config.win_size), persistent=False)
        self.pad = (config.n_fft - config.hop_size) // 2

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav[None]
        cfg = self.config
        wav = torch.nn.functional.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            wav,
            cfg.n_fft,
            hop_length=cfg.hop_size,
            win_length=cfg.win_size,
            window=self.window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        return torch.log(torch.clamp(spec, min=1e-5))

    def num_frames(self, num_samples: int) -> int:
        cfg = self.config
        return (num_samples + 2 * self.pad - cfg.n_fft) // cfg.hop_size + 1


def kaldi_fbank_80(wav16: torch.Tensor, num_mel_bins: int = 80) -> torch.Tensor:
    """Mean-normalised Kaldi filter bank of one 16 kHz utterance, ``[T', bins]``.

    ``wav16`` is ``[T]`` or ``[1, T]``; the per-utterance mean over time is
    subtracted, as the speaker encoder's reference feature extractor does.
    """
    if wav16.dim() == 1:
        wav16 = wav16[None]
    feature = kaldi.fbank(wav16, num_mel_bins=num_mel_bins)
    return feature - feature.mean(dim=0, keepdim=True)
