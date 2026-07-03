"""DAC audio-codec vocoder for Zonos2 (44.1 kHz).

Ported / adapted from ``../ZONOS2/python/zonos2/tokenizer/vocoder.py``.
Converts multi-codebook audio tokens to PCM via Descript's DAC. The model
emits codes with the inter-codebook shear (codebook ``j`` delayed by ``j``
frames); :func:`shear_up` removes that delay before decoding.

``StreamingDacDecoder`` keeps a per-request frame buffer and decodes new
frames incrementally as they arrive (a simplified version of the reference
manager — no overlap-add crossfade). ``dac`` is imported lazily so the
package imports without the optional dependency; install
``descript-audio-codec`` to run the vocoder.
"""
from __future__ import annotations

import torch

# Cached DAC model (loaded lazily on first decode).
_dac_model = None


def _get_dac(model_type: str = "44khz", device: str = "cuda"):
    global _dac_model
    if _dac_model is None:
        try:
            import dac as dac_module
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The Zonos2 vocoder needs the 'descript-audio-codec' package. "
                "Install it with `pip install descript-audio-codec`."
            ) from exc
        _dac_model = (
            dac_module.DAC.load(dac_module.utils.download(model_type=model_type))
            .eval()
            .to(device)
        )
    return _dac_model


def shear_up(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Remove the inter-codebook delay: column ``j`` shifted up by ``j``.

    Inverse of ``prompt.shear``. ``x`` is ``(..., H, W)`` (H frames, W
    codebooks); empty tail positions are filled with ``pad_id``.
    """
    H, W = x.shape[-2:]
    out = x.new_full(x.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out


def decode_dac(
    codes: torch.Tensor, model_type: str = "44khz", codebook_size: int = 1024,
) -> torch.Tensor:
    """Decode ``(batch, seq_len, n_codebooks)`` audio codes to a waveform.

    Returns ``(batch, num_samples)`` float audio at 44.1 kHz.
    """
    dac = _get_dac(model_type=model_type, device=str(codes.device))
    codes = torch.clamp(codes, min=0, max=codebook_size - 1)
    codes = codes.permute(0, 2, 1)  # DAC wants (batch, codebooks, seq)
    with torch.no_grad(), torch.inference_mode():
        z = dac.quantizer.from_codes(codes)[0]
        audio = dac.decode(z).float().squeeze(1)
    return audio


def to_int16_pcm(audio: torch.Tensor) -> torch.Tensor:
    """Convert float audio in [-1, 1] to int16 PCM."""
    return (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)


class StreamingDacDecoder:
    """Per-request incremental DAC decoder.

    Accumulates frames per request; each call decodes the frames that now
    have enough future context for ``shear_up`` (output frame ``i`` needs
    input frames ``i .. i + n_codebooks - 1``). On the final call it flushes
    the remaining frames with padding fill.
    """

    def __init__(
        self,
        n_codebooks: int = 9,
        audio_pad_id: int = 1025,
        codebook_size: int = 1024,
        sample_rate: int = 44100,
        model_type: str = "44khz",
    ):
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.model_type = model_type
        self._buffers: dict[str, list[list[int]]] = {}
        self._decoded: dict[str, int] = {}

    def reset(self, request_id: str | None = None) -> None:
        if request_id is None:
            self._buffers.clear()
            self._decoded.clear()
        else:
            self._buffers.pop(request_id, None)
            self._decoded.pop(request_id, None)

    def add_frames(self, request_id: str, frames: torch.Tensor, is_final: bool) -> torch.Tensor:
        """Append ``frames`` ``(num, n_codebooks)`` and decode what's ready.

        Returns an int16 PCM tensor ``(num_samples,)`` (possibly empty).
        """
        buf = self._buffers.setdefault(request_id, [])
        self._decoded.setdefault(request_id, 0)
        buf.extend(frames.tolist())

        total = len(buf)
        decoded = self._decoded[request_id]
        # Frames fully covered by shear context (all but the last C-1),
        # or everything on the final flush.
        target = total if is_final else total - (self.n_codebooks - 1)
        if target <= decoded:
            if is_final:
                self.reset(request_id)
            return torch.empty(0, dtype=torch.int16)

        # Slice new frames plus the future context shear_up needs.
        raw_end = min(target + self.n_codebooks - 1, total)
        raw = buf[decoded:raw_end]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = torch.tensor(raw, dtype=torch.int64, device=device)
        codes = shear_up(codes, self.audio_pad_id)
        out_count = target - decoded
        codes = codes[:out_count].unsqueeze(0)  # (1, out_count, n_codebooks)

        audio = decode_dac(codes, self.model_type, self.codebook_size)
        pcm = to_int16_pcm(audio[0]).cpu()

        self._decoded[request_id] = target
        if is_final:
            self.reset(request_id)
        return pcm
