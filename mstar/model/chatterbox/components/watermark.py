"""Perth watermarking of generated audio.

Every Chatterbox output carries Resemble's PerTh (perceptual threshold)
watermark. The ``resemble-perth`` package (MIT) holds the network and its
weights; it is an optional dependency, imported lazily, and the
``watermark`` request knob (default on) or ``watermark: false`` in the
deployment's ``model_kwargs`` turns it off. When the package is missing the
server starts and logs that outputs are unwatermarked.

The package's own ``apply_watermark`` works on numpy arrays: it resamples to
the network's 32 kHz with librosa on the CPU, runs the STFT, encoder and
inverse STFT on the network's device, copies back and resamples again on the
CPU. This adapter runs the same network and the same magnitude/phase
front end on the worker's device end to end, with the sinc resampler the
rest of S3Gen uses, so a watermark costs a few small kernels instead of two
host round trips per utterance. The mark it embeds decodes with the
package's detector exactly like the package's own (the two outputs differ
by the resampler only, about 50 dB below the signal).
"""

from __future__ import annotations

import logging

import torch

from mstar.model.chatterbox.components.audio_frontend import resample

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
        impl = perth.PerthImplicitWatermarker(device=device)
        return cls(impl, device)

    @property
    def network_sample_rate(self) -> int:
        return int(self._impl.perth_net.hp.sample_rate)

    @torch.no_grad()
    def apply(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Watermark a mono waveform in [-1, 1]; returns the same shape/dtype."""
        if wav.numel() == 0:
            return wav
        net = self._impl.perth_net
        native = self.network_sample_rate
        signal = wav.detach().reshape(-1).to(net.device, torch.float32)
        marked = self._mark(net, resample(signal, sample_rate, native))
        marked = resample(marked, native, sample_rate)
        n = signal.shape[-1]
        marked = marked[:n]
        if marked.shape[-1] < n:
            marked = torch.nn.functional.pad(marked, (0, n - marked.shape[-1]))
        return marked.to(wav.device, wav.dtype).reshape(wav.shape)

    @staticmethod
    def _mark(net, signal: torch.Tensor) -> torch.Tensor:
        """The package's ``apply_watermark`` body at the network's sample rate:
        magnitude/phase split, encoder on the magnitude, inverse STFT."""
        magspec, phase = net.ap.signal_to_magphase(signal)
        marked_magspec, _mask = net.encoder(magspec[None])
        return net.ap.magphase_to_signal(marked_magspec[0], phase)
