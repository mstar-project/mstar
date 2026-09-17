"""Perth watermarking of generated audio.

Every Chatterbox output carries Resemble's PerTh (perceptual threshold)
watermark. The ``resemble-perth`` package (MIT) holds the network and its
weights; it is an optional dependency, imported lazily, and the
``watermark`` request knob (default on) or ``watermark: false`` in the
deployment's ``model_kwargs`` turns it off. When the package is missing the
server starts and logs that outputs are unwatermarked.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PerthWatermarker:
    """Thin adapter over ``perth.PerthImplicitWatermarker`` working on
    ``(samples,)`` float tensors at a given sample rate."""

    def __init__(self, impl, device: str):
        self._impl = impl
        self.device = device

    @classmethod
    def build(cls, device: str = "cpu") -> "PerthWatermarker | None":
        try:
            import perth
        except ImportError:
            logger.warning(
                "resemble-perth is not installed: Chatterbox audio will not be "
                "watermarked (pip install resemble-perth)"
            )
            return None
        # The network runs on the worker's device; its STFT front end wants a
        # CPU numpy round trip today, which is one small copy per utterance.
        impl = perth.PerthImplicitWatermarker(device=device)
        return cls(impl, device)

    @torch.no_grad()
    def apply(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Watermark a mono waveform in [-1, 1]; returns the same shape/dtype."""
        if wav.numel() == 0:
            return wav
        signal = wav.detach().to("cpu", torch.float32).numpy()
        marked = self._impl.apply_watermark(signal, sample_rate=sample_rate)
        marked = np.asarray(marked, dtype=np.float32)[: signal.shape[-1]]
        if marked.shape[-1] < signal.shape[-1]:
            marked = np.pad(marked, (0, signal.shape[-1] - marked.shape[-1]))
        return torch.from_numpy(marked).to(wav.device, wav.dtype)
