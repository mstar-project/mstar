"""DAC audio-codec vocoder for Zonos2 (44.1 kHz).

This is adapted from ``../ZONOS2/python/zonos2/tokenizer/vocoder.py``. It
converts multi-codebook audio tokens into PCM with the DAC of Descript. The
model emits codes with the inter-codebook shear, where codebook ``j`` lags by
``j`` frames. :func:`shear_up` removes that delay before the decode.

:class:`StreamingDacDecoder` keeps the state of each request in device tensors.
The state holds the frame history, the withheld overlap tail, and the fade
windows. The decoder decodes new frames as they arrive, with no host
round-trip. The DAC call is in ``_decode_codes``, so a batched caller can stack
the windows of several requests into one call.

The ``dac`` import is lazy, so this package imports without the optional
dependency. Install ``descript-audio-codec`` to run the vocoder.
"""
from __future__ import annotations

from typing import Any

import torch

# The cached DAC model. It loads on the first decode.
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
    """Remove the inter-codebook delay. Shift column ``j`` up by ``j`` rows.

    This is the inverse of ``prompt.shear``. ``x`` is ``(..., H, W)``, with H
    frames and W codebooks. ``pad_id`` fills the empty tail positions.
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
    """Decode the audio codes ``(batch, seq_len, n_codebooks)`` to a waveform.

    The result is float audio ``(batch, num_samples)`` at 44.1 kHz.
    """
    dac = _get_dac(model_type=model_type, device=str(codes.device))
    codes = torch.clamp(codes, min=0, max=codebook_size - 1)
    codes = codes.permute(0, 2, 1)  # DAC needs (batch, codebooks, seq)
    with torch.no_grad(), torch.inference_mode():
        z = dac.quantizer.from_codes(codes)[0]
        audio = dac.decode(z).float().squeeze(1)
    return audio


def to_int16_pcm(audio: torch.Tensor) -> torch.Tensor:
    """Convert float audio in [-1, 1] to int16 PCM."""
    return (audio.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)


class StreamingDacDecoder:
    """Incremental DAC decoder for each request, with overlap-add crossfading.

    The decoder collects the frames of a request. Each call decodes the frames
    that now have enough future context for ``shear_up``. Output frame ``i``
    needs the input frames ``i`` to ``i + n_codebooks - 1``.

    Each chunk that is not the last withholds its final ``overlap_frames *
    hop_length`` samples, and the final flush emits them. The crossfade runs in
    float, and the decoder then converts the emitted samples to int16 PCM.

    The state lives in device tensors: the frame history (``_frames``), the
    withheld overlap tails (``_overlap_tails``), and the cached fade windows.
    The DAC call is in :meth:`_decode_codes`, so a batched caller can stack the
    windows of several requests into one call.
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
        # The frame history of each request. Each entry is one
        # ``(T, n_codebooks)`` int64 tensor on the compute device.
        # ``torch.cat`` grows it.
        self._frames: dict[str, torch.Tensor] = {}
        self._decoded: dict[str, int] = {}
        # The withheld float tail of each request. Each entry is the last
        # overlap region of the previous chunk, which the decoder did not emit.
        # The code crossfades it into the head of the next chunk.
        self._overlap_tails: dict[str, torch.Tensor] = {}
        # Raised-cosine fade-in windows, keyed by (length, device).
        self._window_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def reset(self, request_id: str | None = None) -> None:
        if request_id is None:
            self._frames.clear()
            self._decoded.clear()
            self._overlap_tails.clear()
        else:
            self._frames.pop(request_id, None)
            self._decoded.pop(request_id, None)
            self._overlap_tails.pop(request_id, None)

    def _fade_in(self, length: int, device: torch.device) -> torch.Tensor:
        """Return a cached raised-cosine fade-in of ``length`` samples: 0 to 1."""
        key = (length, device)
        win = self._window_cache.get(key)
        if win is None:
            if length <= 1:
                win = torch.ones(length, dtype=torch.float32, device=device)
            else:
                t = torch.linspace(0.0, torch.pi, length, dtype=torch.float32, device=device)
                win = 0.5 * (1.0 - torch.cos(t))
            self._window_cache[key] = win
        return win

    def _decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode de-sheared code windows ``(B, W, n_codebooks)`` to ``(B, W * hop)``.

        The batched path assembles one padded ``(B, W, C)`` tensor and calls
        this method once.
        """
        audio = decode_dac(codes, self.model_type, self.codebook_size)
        return audio.detach().float()

    def _prep_window(
        self, request_id: str, frames: torch.Tensor, is_final: bool
    ) -> tuple[str, Any]:
        """Add ``frames`` and find the decode window of this request.

        The method returns ``("done", pcm)`` when this step has nothing to
        decode. ``pcm`` is an int16 tensor, and it can be empty, for example on
        a final tail flush. If not, the method returns ``("decode", plan)``.
        ``plan`` holds the de-sheared code window ``(out_count, n_codebooks)``
        and the state to finalize it. The DAC call is deferred, so a batched
        caller can stack the windows.
        """
        buf = self._frames.get(request_id)
        if frames.numel():
            add = frames.to(dtype=torch.int64)
            buf = add if buf is None else torch.cat([buf, add], dim=0)
            self._frames[request_id] = buf
        if buf is None:
            # No frames have arrived yet. For example, an empty final flush
            # before any audio.
            buf = frames.reshape(0, self.n_codebooks).to(dtype=torch.int64)
            self._frames[request_id] = buf
        self._decoded.setdefault(request_id, 0)

        total = buf.shape[0]
        decoded = self._decoded[request_id]
        # Output only the frames that have full shear context: all frames
        # except the last (n_codebooks - 1). Those last frames de-shear earlier
        # frames. They are not audio of their own.
        target = max(total - (self.n_codebooks - 1), 0)
        new_decodable = target - decoded

        if is_final:
            should_decode = new_decodable > 0 or request_id in self._overlap_tails
        else:
            should_decode = new_decodable >= self.min_decode_chunk
        if not should_decode:
            if is_final:
                self.reset(request_id)
            return "done", torch.empty(0, dtype=torch.int16)

        # There is nothing new to decode, but a withheld tail remains. This is
        # a final flush.
        if new_decodable <= 0:
            tail = self._overlap_tails.pop(request_id, None)
            out = to_int16_pcm(tail) if tail is not None else torch.empty(0, dtype=torch.int16)
            if is_final:
                self.reset(request_id)
            return "done", out

        # Decode ``overlap`` frames that the decoder already emitted, as left
        # context. The convolutions then start on real signal at the boundary.
        overlap = min(self.overlap_frames, decoded)
        decode_start = decoded - overlap
        raw_end = min(target + self.n_codebooks - 1, total)  # future frames for shear
        raw = buf[decode_start:raw_end]                      # (w, C) int64, on device
        codes = shear_up(raw, self.audio_pad_id)
        out_count = target - decode_start  # overlap + new frames
        codes = codes[:out_count]  # (out_count, n_codebooks), no batch dim
        return "decode", {
            "codes": codes,
            "overlap": overlap,
            "target": target,
            "is_final": is_final,
        }

    def _finalize_chunk(
        self, request_id: str, audio: torch.Tensor, overlap: int,
        is_final: bool, target: int,
    ) -> torch.Tensor:
        """Crossfade, withhold or emit the overlap tail, and encode to int16.

        ``audio`` is the raw ``(out_count * hop,)`` decode of this request. The
        single path and the batched path both call this method, so their output
        agrees exactly.
        """
        # Crossfade the overlap region with the withheld tail of the previous
        # chunk. The code stays functional (cat, not in-place), so it is safe
        # on the inference-mode output of the decoder.
        prev_tail = self._overlap_tails.get(request_id)
        if overlap > 0 and prev_tail is not None:
            k = min(overlap * self.hop_length, prev_tail.numel(), audio.numel())
            if k > 0:
                fade = self._fade_in(k, audio.device)
                head = (1.0 - fade) * prev_tail[-k:] + fade * audio[:k]
                audio = torch.cat([head, audio[k:]], dim=0)

        if is_final:
            output = audio
            self._overlap_tails.pop(request_id, None)
        else:
            # Withhold the tail. The next chunk decodes it again with real
            # right context, then crossfades over this boundary.
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

    def add_frames(self, request_id: str, frames: torch.Tensor, is_final: bool) -> torch.Tensor:
        """Add the frames ``(num, n_codebooks)`` and decode what is ready.

        The result is an int16 PCM tensor ``(num_samples,)``. It can be empty.
        """
        kind, plan = self._prep_window(request_id, frames, is_final)
        if kind == "done":
            return plan
        audio = self._decode_codes(plan["codes"].unsqueeze(0))[0]  # (out_count * hop,)
        return self._finalize_chunk(
            request_id, audio, plan["overlap"], is_final, plan["target"]
        )

    def add_frames_batched(
        self,
        request_ids: list[str],
        frames_list: list[torch.Tensor],
        finals: list[bool],
    ) -> dict[str, torch.Tensor]:
        """Run :meth:`add_frames` for several requests in one call.

        The method prepares the window of each request separately. It then
        stacks the windows of the same length into one DAC call.
        """
        results: dict[str, torch.Tensor] = {}
        groups: dict[int, list[tuple[str, dict]]] = {}
        for rid, frames, is_final in zip(request_ids, frames_list, finals, strict=True):
            kind, plan = self._prep_window(rid, frames, is_final)
            if kind == "done":
                results[rid] = plan
            else:
                groups.setdefault(plan["codes"].shape[0], []).append((rid, plan))

        for items in groups.values():
            batch = torch.stack([plan["codes"] for _, plan in items], dim=0)
            audios = self._decode_codes(batch)  # (g, out_count * hop)
            for i, (rid, plan) in enumerate(items):
                results[rid] = self._finalize_chunk(
                    rid, audios[i], plan["overlap"], plan["is_final"], plan["target"]
                )
        return results
