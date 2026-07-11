"""DAC audio-codec vocoder for Zonos2 (44.1 kHz).

Ported / adapted from ``../ZONOS2/python/zonos2/tokenizer/vocoder.py``.
Converts multi-codebook audio tokens to PCM via Descript's DAC. The model
emits codes with the inter-codebook shear (codebook ``j`` delayed by ``j``
frames); :func:`shear_up` removes that delay before decoding.

``StreamingDacDecoder`` keeps a per-request frame buffer and decodes new
frames incrementally as they arrive. ``dac`` is imported
lazily so the package imports without the optional dependency; install
``descript-audio-codec`` to run the vocoder.
# TODO: We should implement the StreamingDacDecoder a la Qwen Omni encoders. 
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
    """Per-request incremental DAC decoder with overlap-add crossfading.

    Accumulates frames per request; each call decodes the frames that now
    have enough future context for ``shear_up`` (output frame ``i`` needs
    input frames ``i .. i + n_codebooks - 1``).

    Each non-final chunk withholds its last ``overlap_frames * hop_length`` 
    samples; the final flush emits them.

    Crossfading is done in float; the emitted samples are converted to
    int16 PCM on the way out.
    """

    def __init__(
        self,
        n_codebooks: int = 9,
        audio_pad_id: int = 1025,
        codebook_size: int = 1024,
        sample_rate: int = 44100,
        model_type: str = "44khz",
        overlap_frames: int = 4,
        hop_length: int = 512,
        min_decode_chunk: int = 1,
    ):
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.overlap_frames = overlap_frames
        self.hop_length = hop_length
        self.min_decode_chunk = max(1, min_decode_chunk)
        self._buffers: dict[str, list[list[int]]] = {}
        self._decoded: dict[str, int] = {}
        # Withheld float tails per request (last overlap region of the
        # previously decoded chunk, not yet emitted — crossfaded into the
        # next chunk's head).
        self._overlap_tails: dict[str, torch.Tensor] = {}
        # Raised-cosine fade-in windows, cached by length.
        self._window_cache: dict[int, torch.Tensor] = {}

    def reset(self, request_id: str | None = None) -> None:
        if request_id is None:
            self._buffers.clear()
            self._decoded.clear()
            self._overlap_tails.clear()
        else:
            self._buffers.pop(request_id, None)
            self._decoded.pop(request_id, None)
            self._overlap_tails.pop(request_id, None)

    def _fade_in(self, length: int) -> torch.Tensor:
        """Cached raised-cosine fade-in of ``length`` samples (0 -> 1)."""
        win = self._window_cache.get(length)
        if win is None:
            if length <= 1:
                win = torch.ones(length, dtype=torch.float32)
            else:
                t = torch.linspace(0.0, torch.pi, length, dtype=torch.float32)
                win = 0.5 * (1.0 - torch.cos(t))
            self._window_cache[length] = win
        return win

    def add_frames(self, request_id: str, frames: torch.Tensor, is_final: bool) -> torch.Tensor:
        """Append ``frames`` ``(num, n_codebooks)`` and decode what's ready.

        Returns an int16 PCM tensor ``(num_samples,)`` (possibly empty).
        """
        buf = self._buffers.setdefault(request_id, [])
        self._decoded.setdefault(request_id, 0)
        if frames.numel():
            buf.extend(frames.tolist())

        total = len(buf)
        decoded = self._decoded[request_id]
        # Output frames fully covered by shear context: all but the trailing
        # (n_codebooks - 1) frames, which only exist to de-shear earlier
        # frames and are not audio of their own.
        target = max(total - (self.n_codebooks - 1), 0)
        new_decodable = target - decoded

        if is_final:
            should_decode = new_decodable > 0 or request_id in self._overlap_tails
        else:
            should_decode = new_decodable >= self.min_decode_chunk
        if not should_decode:
            if is_final:
                self.reset(request_id)
            return torch.empty(0, dtype=torch.int16)

        # Nothing new to decode but a withheld tail remains (final flush).
        if new_decodable <= 0:
            tail = self._overlap_tails.pop(request_id, None)
            out = to_int16_pcm(tail) if tail is not None else torch.empty(0, dtype=torch.int16)
            if is_final:
                self.reset(request_id)
            return out

        # Re-decode ``overlap`` already-emitted frames as left context so the
        # convolutions warm up on real signal at the chunk boundary.
        overlap = min(self.overlap_frames, decoded)
        decode_start = decoded - overlap
        raw_end = min(target + self.n_codebooks - 1, total)  # future frames for shear
        raw = buf[decode_start:raw_end]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = torch.tensor(raw, dtype=torch.int64, device=device)
        codes = shear_up(codes, self.audio_pad_id)
        out_count = target - decode_start  # overlap + new frames
        codes = codes[:out_count].unsqueeze(0)  # (1, out_count, n_codebooks)

        audio = decode_dac(codes, self.model_type, self.codebook_size)
        audio = audio[0].detach().float().cpu()

        # Crossfade the overlap region with the previous chunk's withheld tail.
        prev_tail = self._overlap_tails.get(request_id)
        if overlap > 0 and prev_tail is not None:
            k = min(overlap * self.hop_length, prev_tail.numel(), audio.numel())
            if k > 0:
                fade = self._fade_in(k)
                audio[:k] = (1.0 - fade) * prev_tail[-k:] + fade * audio[:k]

        if is_final:
            output = audio
            self._overlap_tails.pop(request_id, None)
        else:
            # Withhold the tail so the next chunk (which re-decodes it with
            # real right-context) can crossfade over this boundary.
            tail_samples = min(self.overlap_frames * self.hop_length, audio.numel())
            if tail_samples > 0:
                self._overlap_tails[request_id] = audio[-tail_samples:].clone()
                output = audio[:-tail_samples]
            else:
                self._overlap_tails.pop(request_id, None)
                output = audio

        self._decoded[request_id] = target
        pcm = to_int16_pcm(output)
        if is_final:
            self.reset(request_id)
        return pcm
