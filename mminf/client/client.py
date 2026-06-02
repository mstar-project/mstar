"""mminf Python SDK — a thin HTTP client over a running mminf server.

The client wraps the native ``POST /generate`` endpoint: it builds the
multipart form, parses the NDJSON stream (or grouped JSON), base64-decodes
payloads, and returns typed results. It works for every model the server can
host (text, image, audio, video) and pulls in only ``requests`` (+ ``numpy``
for audio helpers) — no torch / CUDA.

    from mminf import MMInfClient
    client = MMInfClient("http://localhost:8000")
    print(client.chat("Hello!").text)
    client.tts("Hi there", voice="tara").to_wav("out.wav")
    open("cat.png", "wb").write(client.generate_image("a cat in a hat"))
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterator

import requests

from mminf.client.media import parse_ndjson_line
from mminf.client.types import (
    AudioBuffer,
    AudioChunk,
    GenerateResult,
    ImageChunk,
    StreamEvent,
    TextChunk,
)

# When attaching raw bytes we need a filename whose extension lets the server
# infer the modality (it keys off the extension).
_DEFAULT_EXT = {"images": "png", "audio": "wav", "video": "mp4"}
_MODALITY_OF = {"images": "image", "audio": "audio", "video": "video"}

MediaItem = "str | bytes | Path | tuple[str, bytes]"


class MMInfClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 600.0,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        text: str | None = None,
        images=None,
        audio=None,
        video=None,
        output_modalities=("text",),
        input_modalities=None,
        stream: bool = False,
        request_id: str | None = None,
        **model_kwargs,
    ):
        """Submit a multimodal generation request.

        ``images`` / ``audio`` / ``video`` accept a single item or a list, where
        each item is a local path, raw ``bytes``, or a ``(filename, bytes)``
        tuple. Extra keyword args are forwarded verbatim as the model's
        ``model_kwargs`` (e.g. ``voice="tara"``, ``think_mode=True``,
        ``temperature=0.7``, ``max_output_tokens=256``); ``None`` values are
        dropped so server-side defaults apply.

        Returns a :class:`GenerateResult` when ``stream=False``, or an iterator
        of :class:`StreamEvent` (``TextChunk`` / ``ImageChunk`` / ``AudioChunk``)
        when ``stream=True``.
        """
        files = self._build_files(images, audio, video)
        data: dict[str, str] = {
            "output_modalities": ",".join(output_modalities),
            "streaming": "true" if stream else "false",
        }
        if text is not None:
            data["text"] = text
        if input_modalities is not None:
            data["input_modalities"] = ",".join(input_modalities)
        if request_id is not None:
            data["request_id"] = request_id
        mk = {k: v for k, v in model_kwargs.items() if v is not None}
        if mk:
            data["model_kwargs"] = json.dumps(mk)

        url = f"{self.base_url}/generate"
        if stream:
            return self._stream(url, data, files)
        resp = self._session.post(url, data=data, files=files or None, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def stream(self, **kwargs) -> Iterator[StreamEvent]:
        """Sugar for ``generate(stream=True, ...)``."""
        return self.generate(stream=True, **kwargs)

    # ------------------------------------------------------------------
    # Convenience sugar
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        images=None,
        audio=None,
        output_modalities=("text",),
        stream: bool = False,
        **model_kwargs,
    ):
        """Text (optionally + audio) generation. For omni speech output pass
        ``output_modalities=("text", "audio")``."""
        return self.generate(
            text=prompt,
            images=images,
            audio=audio,
            output_modalities=output_modalities,
            stream=stream,
            **model_kwargs,
        )

    def generate_image(self, prompt: str, **model_kwargs) -> bytes:
        """Return PNG bytes for a text-to-image request (e.g. BAGEL)."""
        res = self.generate(text=prompt, output_modalities=("image",), **model_kwargs)
        if not res.images:
            raise RuntimeError("Server returned no image output")
        return res.images[0]

    def tts(self, text: str, *, voice: str | None = None, **model_kwargs) -> AudioBuffer:
        """Text-to-speech. Returns an :class:`AudioBuffer` (``.to_wav(path)``)."""
        res = self.generate(text=text, output_modalities=("audio",), voice=voice, **model_kwargs)
        if res.audio is None:
            raise RuntimeError("Server returned no audio output")
        return res.audio

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=10)
            return r.ok and r.json().get("status") == "healthy"
        except Exception:  # noqa: BLE001 — health is best-effort
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_files(self, images, audio, video) -> list[tuple[str, tuple[str, bytes]]]:
        files: list[tuple[str, tuple[str, bytes]]] = []
        for kind, items in (("images", images), ("audio", audio), ("video", video)):
            if not items:
                continue
            if isinstance(items, (str, bytes, bytearray, Path)):
                items = [items]
            for i, item in enumerate(items):
                fname, blob = self._coerce_file(kind, i, item)
                files.append(("files", (fname, blob)))
        return files

    @staticmethod
    def _coerce_file(kind: str, idx: int, item) -> tuple[str, bytes]:
        if isinstance(item, (str, Path)):
            p = Path(item)
            return p.name, p.read_bytes()
        if isinstance(item, (bytes, bytearray)):
            return f"{_MODALITY_OF[kind]}_{idx}.{_DEFAULT_EXT[kind]}", bytes(item)
        if isinstance(item, tuple) and len(item) == 2:
            return item[0], bytes(item[1])
        raise TypeError(f"Unsupported {kind} item type: {type(item)!r}")

    def _stream(self, url, data, files) -> Iterator[StreamEvent]:
        with self._session.post(
            url, data=data, files=files or None, stream=True, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not isinstance(line, str):
                    continue
                parsed = parse_ndjson_line(line)
                if parsed is None:
                    continue
                yield self._to_event(parsed)

    @staticmethod
    def _to_event(parsed: dict) -> StreamEvent:
        modality = parsed["modality"]
        raw = parsed["bytes"]
        meta = parsed["metadata"]
        if modality == "image":
            return ImageChunk(raw, meta)
        if modality == "audio":
            return AudioChunk(raw, int(meta.get("sample_rate", 24000)), meta)
        # text and any unrecognized modality decode as utf-8 text
        return TextChunk(raw.decode("utf-8", "replace"), meta)

    @staticmethod
    def _parse_result(payload: dict) -> GenerateResult:
        outputs = payload.get("outputs", {})
        text_parts: list[str] = []
        images: list[bytes] = []
        audio_pcm: list[bytes] = []
        sample_rate = 24000
        raw: list[dict] = []
        for modality, entries in outputs.items():
            for e in entries:
                b = base64.b64decode(e["data"]) if e.get("data") else b""
                meta = e.get("metadata") or {}
                raw.append({"modality": modality, "bytes": b, "metadata": meta})
                if modality == "text":
                    text_parts.append(b.decode("utf-8", "replace"))
                elif modality == "image":
                    images.append(b)
                elif modality == "audio":
                    audio_pcm.append(b)
                    sample_rate = int(meta.get("sample_rate", sample_rate))
        audio = AudioBuffer(b"".join(audio_pcm), sample_rate) if audio_pcm else None
        return GenerateResult(
            request_id=payload.get("request_id"),
            text="".join(text_parts) if text_parts else None,
            images=images,
            audio=audio,
            raw=raw,
        )
