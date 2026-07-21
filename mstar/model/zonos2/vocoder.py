"""DAC audio-codec vocoder for Zonos2 (44.1 kHz).

Adapted from ``../ZONOS2/python/zonos2/tokenizer/vocoder.py``. It converts
multi-codebook audio tokens to PCM with Descript's DAC. The model emits codes
with the inter-codebook shear. Codebook ``j`` lags by ``j`` frames.
:func:`shear_up` removes that delay before the decode.

``StreamingDacDecoder`` keeps per-request state as device tensors. The state
holds frame history, the withheld overlap tail, and fade windows. It decodes
new frames as they arrive, with no host round-trip. The per-window decode
lives in ``_decode_codes``. So a batched caller stacks several requests'
windows into one DAC call.

``dac`` imports lazily. So the package imports without the optional
dependency. Install ``descript-audio-codec`` to run the vocoder.
"""
from __future__ import annotations

from typing import Any

import torch

# Cached DAC model. It loads lazily on the first decode.
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
    """Remove the inter-codebook delay. It shifts column ``j`` up by ``j``.

    This inverts ``prompt.shear``. ``x`` is ``(..., H, W)`` (H frames, W
    codebooks). The code fills empty tail positions with ``pad_id``.
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

    It accumulates frames per request. Each call decodes the frames that now
    have enough future context for ``shear_up``. Output frame ``i`` needs input
    frames ``i .. i + n_codebooks - 1``.

    Each non-final chunk withholds its last ``overlap_frames * hop_length``
    samples. The final flush emits them. The crossfade runs in float. The code
    then converts the emitted samples to int16 PCM.

    The state lives in device tensors. It holds frame history (_frames),
    withheld overlap tails (_overlap_tails), and cached fade windows. The
    per-window DAC call lives in :meth:`_decode_codes`. So a batched caller
    stacks several requests' windows into one decode.
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
        # Per-request frame history. Each is one ``(T, n_codebooks)`` int64
        # tensor on the compute device. ``torch.cat`` grows it.
        self._frames: dict[str, torch.Tensor] = {}
        self._decoded: dict[str, int] = {}
        # Withheld float tails per request. Each is the last overlap region of
        # the previous chunk, not yet emitted. The code crossfades it into the
        # next chunk's head.
        self._overlap_tails: dict[str, torch.Tensor] = {}
        # Raised-cosine fade-in windows. The cache keys them by (length, device).
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
        """Cached raised-cosine fade-in of ``length`` samples (0 -> 1)."""
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
        """De-sheared code windows ``(B, W, n_codebooks)`` -> ``(B, W * hop)``.

        The batched DAC path assembles a padded ``(B, W, C)`` tensor. It calls
        this once.
        """
        audio = decode_dac(codes, self.model_type, self.codebook_size)
        return audio.detach().float()

    def _prep_window(
        self, request_id: str, frames: torch.Tensor, is_final: bool
    ) -> tuple[str, Any]:
        """Append ``frames`` and find this request's decode window.

        Returns ``("done", pcm)`` when this step has nothing to decode. ``pcm``
        is an int16 PCM tensor, possibly empty (for example, a final tail
        flush). Otherwise it returns ``("decode", plan)``. ``plan`` carries the
        de-sheared code window ``(out_count, n_codebooks)`` plus the state to
        finalize it. The code defers the DAC call. So a batched caller stacks
        windows.
        """
        buf = self._frames.get(request_id)
        if frames.numel():
            add = frames.to(dtype=torch.int64)
            buf = add if buf is None else torch.cat([buf, add], dim=0)
            self._frames[request_id] = buf
        if buf is None:
            # The code saw no frames yet. For example, an empty final flush
            # before any audio.
            buf = frames.reshape(0, self.n_codebooks).to(dtype=torch.int64)
            self._frames[request_id] = buf
        self._decoded.setdefault(request_id, 0)

        total = buf.shape[0]
        decoded = self._decoded[request_id]
        # Output frames with full shear context: all but the trailing
        # (n_codebooks - 1) frames. Those frames only de-shear earlier frames.
        # They are not audio of their own.
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

        # Nothing new to decode, but a withheld tail remains (final flush).
        if new_decodable <= 0:
            tail = self._overlap_tails.pop(request_id, None)
            out = to_int16_pcm(tail) if tail is not None else torch.empty(0, dtype=torch.int16)
            if is_final:
                self.reset(request_id)
            return "done", out

        # Re-decode ``overlap`` emitted frames as left context. The
        # convolutions then warm up on real signal at the chunk boundary.
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
        """Crossfade, withhold or emit the overlap tail, and int16-encode.

        ``audio`` is this request's raw ``(out_count * hop,)`` decode. This runs
        the same for the single and batched paths. So both stay bit-for-bit in
        step."""
        # Crossfade the overlap region with the previous chunk's withheld tail.
        # This stays functional (cat, not in-place). So it is safe on the
        # decoder's inference-mode output and reads cleanly when batched.
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
            # Withhold the tail. The next chunk re-decodes it with real
            # right-context. Then it crossfades over this boundary.
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
        """Append ``frames`` ``(num, n_codebooks)`` and decode what's ready.

        Returns an int16 PCM tensor ``(num_samples,)`` (possibly empty).
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
        """Batched :meth:`add_frames` over several requests.

        The code prepares each request's window on its own. Then it stacks
        windows of the same length into one DAC call.
        """
        results: dict[str, torch.Tensor] = {}
        groups: dict[int, list[tuple[str, dict]]] = {}
        for rid, frames, is_final in zip(request_ids, frames_list, finals):
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
