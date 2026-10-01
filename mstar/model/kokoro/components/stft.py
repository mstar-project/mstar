"""Short-time Fourier analysis and synthesis over padded batches.

The reference calls ``torch.stft``/``torch.istft`` on one signal at a time, so
their reflect padding and overlap-add normalization assume every row ends
where the tensor does. Here the reflect padding is a per-row gather, and the
synthesis is a ``conv_transpose1d`` with a cosine/sine bank divided by a
per-row window envelope, which reproduces ``torch.istft`` for any mix of
lengths without complex tensors.

The analysis transform keeps ``torch.stft`` (on the pre-padded signal): its
phase is consumed downstream as a raw feature, and in near-silent bins the
phase of an FFT and of a DFT-by-convolution differ by a full turn, which is
audible. Sharing the kernel with the reference keeps those bins identical.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.kokoro.components.masking import length_mask, mask_channels

ENVELOPE_FLOOR = 1e-11  # torch.istft's NOLA threshold


class RealSTFT(nn.Module):
    """Hann-windowed, centered STFT with ``win_length == n_fft`` and its inverse."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = n_fft // 2
        self.bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(self.bins, dtype=torch.float64)
        angle = 2 * math.pi * k[:, None] * n[None, :] / n_fft  # [bins, n_fft]

        # irfft of a one-sided spectrum: DC and Nyquist count once and drop
        # their imaginary part; every other bin is doubled by its conjugate.
        weight = torch.full((self.bins,), 2.0, dtype=torch.float64)
        weight[0] = 1.0
        weight[-1] = 1.0
        imag_weight = weight.clone()
        imag_weight[0] = 0.0
        imag_weight[-1] = 0.0
        scale = window / n_fft
        inverse_real = (weight[:, None] * torch.cos(angle) * scale)[:, None, :]
        inverse_imag = (-imag_weight[:, None] * torch.sin(angle) * scale)[:, None, :]

        self.register_buffer("window", window.float(), persistent=False)
        self.register_buffer("inverse_real", inverse_real.float(), persistent=False)
        self.register_buffer("inverse_imag", inverse_imag.float(), persistent=False)
        self.register_buffer("window_sq", window.square().float()[None, None, :], persistent=False)

    def num_frames(self, length: int) -> int:
        return (length + 2 * self.pad - self.n_fft) // self.hop_length + 1

    def frame_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        return (sample_lengths + 2 * self.pad - self.n_fft) // self.hop_length + 1

    def _reflect_pad(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Centered reflect padding of every row about its own ends."""
        positions = torch.arange(x.shape[1] + 2 * self.pad, device=x.device)[None, :] - self.pad
        last = (lengths - 1)[:, None]
        index = torch.where(positions < 0, -positions, positions)
        index = torch.where(index > last, 2 * last - index, index)
        return x.gather(1, index.clamp(min=0))

    def transform(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, L]`` waveform -> (magnitude, phase) ``[B, bins, frames]`` and the
        per-row frame counts. Frames past a row's end are zero."""
        spectrum = torch.stft(
            self._reflect_pad(x, lengths),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        frame_lengths = self.frame_lengths(lengths)
        mask = length_mask(frame_lengths, spectrum.shape[-1])
        return mask_channels(spectrum.abs(), mask), mask_channels(spectrum.angle(), mask), frame_lengths

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, frame_lengths: torch.Tensor) -> torch.Tensor:
        """(magnitude, phase) ``[B, bins, frames]`` -> ``[B, (frames - 1) * hop]``
        waveform, overlap-added and normalized over each row's valid frames."""
        mask = length_mask(frame_lengths, magnitude.shape[-1])
        magnitude = mask_channels(magnitude, mask)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        signal = F.conv_transpose1d(real, self.inverse_real, stride=self.hop_length) + F.conv_transpose1d(
            imag, self.inverse_imag, stride=self.hop_length
        )
        envelope = F.conv_transpose1d(mask[:, None, :].to(signal.dtype), self.window_sq, stride=self.hop_length)
        signal = torch.where(envelope > ENVELOPE_FLOOR, signal / envelope, torch.zeros_like(signal))
        signal = signal[:, 0, self.pad : signal.shape[-1] - self.pad]
        # Past a row's last frame the envelope is a partial window: zero that
        # tail, which the single-row transform would not have emitted at all.
        return signal.masked_fill(~length_mask((frame_lengths - 1) * self.hop_length, signal.shape[-1]), 0.0)
