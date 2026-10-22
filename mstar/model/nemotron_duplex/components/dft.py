"""Fixed-size real DFTs as matmuls, for the transforms inside captured CUDA graphs.

``torch.stft`` / ``torch.fft`` on CUDA execute cuFFT plans from torch's
per-device plan cache, an LRU whose eviction destroys the plan. A graph replay
doesn't touch the LRU, so once enough newer transform geometries have gone
through the cache eagerly, a plan a graph captured is destroyed and the replay
computes garbage. A matmul against a precomputed basis has no plan to lose.

The matmuls run in float64: TF32 (which ``mstar.engine`` enables) doesn't apply
to it and autocast doesn't cast it, so the result rounded to float32 is at least
as accurate as the float32 FFT, and at these sizes it costs microseconds.
"""
from __future__ import annotations

import functools
import math

import torch


def _phase(n: torch.Tensor, k: torch.Tensor, size: int) -> torch.Tensor:
    """``2 pi n k / size`` in float64, with ``n k`` reduced mod ``size`` first."""
    return (n.view(-1, 1) * k.view(1, -1) % size).to(torch.float64) * (2 * math.pi / size)


def stft_basis(window: torch.Tensor, n_fft: int) -> torch.Tensor:
    """``(win_length, 2 * (n_fft // 2 + 1))`` float64: a frame's ``win_length``
    samples -> its one-sided spectrum under ``window``, real parts then
    imaginary, as ``torch.stft`` computes it with the window centred in ``n_fft``."""
    win = window.numel()
    left = (n_fft - win) // 2
    n = torch.arange(left, left + win, device=window.device)
    phase = _phase(n, torch.arange(n_fft // 2 + 1, device=window.device), n_fft)
    w = window.to(torch.float64).view(-1, 1)
    return torch.cat([w * torch.cos(phase), -w * torch.sin(phase)], dim=1)


def stft_as_real(x: torch.Tensor, basis: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    """``torch.view_as_real(torch.stft(x, n_fft, hop_length, window=window,
    center=False, return_complex=True))`` for ``x (B, samples)``, with ``basis``
    from ``stft_basis(window, n_fft)``: ``(B, n_fft // 2 + 1, T, 2)`` in x's dtype."""
    win = basis.shape[0]
    left = (n_fft - win) // 2
    frames = (x.shape[-1] - n_fft) // hop_length + 1
    framed = x[:, left: left + (frames - 1) * hop_length + win].unfold(-1, win, hop_length)   # (B, T, win)
    with torch.autocast(device_type=x.device.type, enabled=False):
        spec = torch.matmul(framed.to(torch.float64), basis).to(x.dtype)                      # (B, T, 2F)
    return spec.unflatten(-1, (2, -1)).permute(0, 3, 1, 2)


@functools.cache
def irfft_basis(n: int, device: torch.device) -> torch.Tensor:
    """``(n, 2 * (n // 2 + 1))`` float64: a one-sided spectrum, real parts then
    imaginary -> ``torch.fft.irfft(spectrum, n, norm="backward")``. Like irfft it
    ignores the imaginary parts of the DC bin and (n even) the Nyquist bin.

    Built on first use; a captured region's eager warmup makes that use before
    the capture."""
    bins = n // 2 + 1
    k = torch.arange(bins, device=device)
    phase = _phase(torch.arange(n, device=device), k, n)
    scale = torch.full((bins,), 2.0 / n, dtype=torch.float64, device=device)
    real_only = (k == 0) | ((k == n // 2) & (n % 2 == 0))
    scale = torch.where(real_only, scale / 2, scale)
    return torch.cat([scale * torch.cos(phase), torch.where(real_only, 0.0, -scale * torch.sin(phase))], dim=1)


def irfft_dim1(spec: torch.Tensor, n: int) -> torch.Tensor:
    """``torch.fft.irfft(spec, n, dim=1, norm="backward")`` for a complex
    ``spec (B, n // 2 + 1, T)``: ``(B, n, T)`` in spec's real dtype."""
    assert spec.shape[1] == n // 2 + 1, f"{spec.shape[1]} bins for an inverse DFT of {n}"
    parts = torch.view_as_real(spec)                                          # (B, F, T, 2)
    stacked = torch.cat([parts[..., 0], parts[..., 1]], dim=1)                # (B, 2F, T)
    with torch.autocast(device_type=spec.device.type, enabled=False):
        return torch.matmul(irfft_basis(n, spec.device), stacked.to(torch.float64)).to(parts.dtype)
