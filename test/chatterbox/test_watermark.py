"""The device-side Perth path embeds the same watermark as the package's numpy path.

Needs ``resemble-perth`` (optional dependency); skipped without it. Runs on
the CPU: the adapter's code path is the same on every device.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

perth = pytest.importorskip("perth")

from mstar.model.chatterbox.components.watermark import PerthWatermarker  # noqa: E402

SR = 24_000


def _speechlike(seconds: float, seed: int = 0) -> torch.Tensor:
    """A few harmonics with vibrato and an amplitude envelope, plus a little noise."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * SR)) / SR
    f0 = 140 + 8 * torch.sin(2 * math.pi * 5 * t)
    phase = 2 * math.pi * torch.cumsum(f0, 0) / SR
    wav = sum(torch.sin(k * phase) / k for k in range(1, 8))
    wav = wav * (0.5 + 0.5 * torch.sin(2 * math.pi * 2.5 * t).clamp_min(0))
    wav = wav + 0.01 * torch.randn(t.numel(), generator=gen)
    return (0.5 * wav / wav.abs().max()).float()


def _snr_db(reference: torch.Tensor, other: torch.Tensor) -> float:
    n = min(reference.numel(), other.numel())
    noise = (reference[:n] - other[:n]).pow(2).mean()
    return float(10 * torch.log10(reference[:n].pow(2).mean() / noise))


@pytest.fixture(scope="module")
def marker() -> PerthWatermarker:
    marker = PerthWatermarker.build("cpu")
    assert marker is not None
    return marker


def test_marked_audio_keeps_shape_and_dtype(marker):
    wav = _speechlike(1.5)
    marked = marker.apply(wav, SR)
    assert marked.shape == wav.shape and marked.dtype == wav.dtype
    assert marker.apply(wav[:0], SR).numel() == 0


def test_mark_matches_the_package_path_and_is_detected(marker):
    wav = _speechlike(2.0, seed=3)
    ours = marker.apply(wav, SR)
    theirs = torch.from_numpy(
        np.asarray(marker._impl.apply_watermark(wav.numpy(), sample_rate=SR), dtype=np.float32)
    )
    # the watermark is a small perturbation of the input, the same one either way
    assert 5 < _snr_db(wav, ours) < 40
    assert _snr_db(theirs, ours) > 30
    detector = perth.PerthImplicitWatermarker(device="cpu")
    assert float(np.mean(detector.get_watermark(ours.numpy(), sample_rate=SR))) >= 0.5
    assert float(np.mean(detector.get_watermark(wav.numpy(), sample_rate=SR))) < 0.5
