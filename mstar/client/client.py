"""mstar Python SDK — a thin HTTP client over a running mstar server.

The client wraps the native ``POST /generate`` endpoint: it builds the
multipart form, parses the NDJSON stream (or grouped JSON), base64-decodes
payloads, and returns typed results. It works for every model the server can
host (text, image, audio, video) and pulls in only ``requests`` (+ ``numpy``
for audio helpers) — no torch / CUDA.

    from mstar import MStarClient
    client = MStarClient("http://localhost:8000")
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

from mstar.client.media import (
    BINARY_STREAM_MEDIA_TYPE,
    NDJSON_STREAM_MEDIA_TYPE,
    iter_binary_frames,
    parse_ndjson_line,
)
from mstar.client.types import (
    AudioBuffer,
    AudioChunk,
    GenerateResult,
    ImageChunk,
    StreamEvent,
    TextChunk,
    VideoFrameChunk,
)

# When attaching raw bytes we need a filename whose extension lets the server
# infer the modality (it keys off the extension).
_DEFAULT_EXT = {"images": "png", "audio": "wav", "video": "mp4"}
_STREAM_READ_CHUNK_SIZE = 1024 * 1024
_MODALITY_OF = {"images": "image", "audio": "audio", "video": "video"}

MediaItem = "str | bytes | Path | tuple[str, bytes]"


def _load_nvtx():
    """Return ``(range_push, range_pop)``, importing torch only on demand.

    The SDK's dependency contract is stdlib + ``requests`` (+ ``numpy``), so
    the profiler import cannot happen at module scope. Only a caller that
    explicitly asks for NVTX pays for it, and such a caller is by definition
    running under a CUDA profiler already.
    """
    from mstar.utils.profiler import range_pop, range_push

    return range_push, range_pop


class MStarClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 600.0,
        session: requests.Session | None = None,
        enable_nvtx: bool = False,
        prefer_binary: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        # Opt in to ask for length-framed binary payloads (e.g. waypoint's
        # large video frames), but decide how to parse from the response's
        # Content-Type. A server that does not implement the framing — an
        # older build, or the Rust frontend, neither of which reads
        # ``Accept`` — answers NDJSON and the historical path handles it.
        self._prefer_binary = prefer_binary
        # Splits the client's share of the streaming gap into the blocking
        # socket read and the decode of the line it returns. Without this the
        # whole of both lands in the caller's "waiting for a chunk" range and
        # is indistinguishable from server time.
        self._nvtx = _load_nvtx() if enable_nvtx else None

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
        of :class:`StreamEvent` when ``stream=True``. Raw ``video_frame`` output
        is streaming-only and yields :class:`VideoFrameChunk` objects.
        """
        if "video_frame" in output_modalities and not stream:
            raise ValueError(
                "output modality 'video_frame' requires stream=True; raw frame "
                "chunks cannot be returned as an aggregated response"
            )
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
            named_bytes = (
                isinstance(items, tuple)
                and len(items) == 2
                and isinstance(items[0], str)
                and isinstance(items[1], (bytes, bytearray))
            )
            if isinstance(items, (str, bytes, bytearray, Path)) or named_bytes:
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

    def _stream_headers(self) -> dict[str, str]:
        if not self._prefer_binary:
            return {}
        return {
            "Accept": f"{BINARY_STREAM_MEDIA_TYPE}, {NDJSON_STREAM_MEDIA_TYPE};q=0.9",
            # Compressing ~11 MB of near-incompressible RGB per chunk would put
            # back the byte-proportional CPU pass this framing exists to remove.
            "Accept-Encoding": "identity",
        }

    def _stream_binary(self, resp) -> Iterator[StreamEvent]:
        """Consume a length-framed response body off the raw socket."""
        encoding = resp.headers.get("content-encoding", "").lower()
        if encoding and encoding != "identity":
            # ``resp.raw.read`` hands back undecoded bytes, so a compressed body
            # would surface as an unreadable header. Fail with the cause named.
            raise RuntimeError(
                f"Binary frame stream arrived with Content-Encoding {encoding!r}; "
                "only 'identity' can be read from the raw socket"
            )
        frames = iter_binary_frames(resp.raw)
        if self._nvtx is None:
            for parsed in frames:
                yield self._to_event(parsed)
            return

        range_push, range_pop = self._nvtx
        while True:
            # The blocking socket read plus frame reassembly. Counterpart of
            # ``client.iter_lines`` on the NDJSON path; there is no decode step
            # to measure after it, which is the point.
            range_push("client.read_frame")
            try:
                parsed = next(frames)
            except StopIteration:
                range_pop()
                break
            range_pop()
            range_push("client.to_event")
            event = self._to_event(parsed)
            range_pop()
            yield event

    def _stream(self, url, data, files) -> Iterator[StreamEvent]:
        with self._session.post(
            url,
            data=data,
            files=files or None,
            stream=True,
            timeout=self.timeout,
            headers=self._stream_headers(),
        ) as resp:
            resp.raise_for_status()
            # Branch on what the server actually sent, not on what was asked
            # for: that is what makes the negotiation safe against servers that
            # ignore ``Accept`` entirely.
            if resp.headers.get("content-type", "").startswith(BINARY_STREAM_MEDIA_TYPE):
                yield from self._stream_binary(resp)
                return
            # ``decode_unicode=True`` only yields ``str`` when ``resp.encoding``
            # is set, and that is derived from the Content-Type charset. The
            # server streams ``application/x-ndjson`` without one, so default to
            # UTF-8 (the NDJSON encoding) instead of dropping every line.
            if resp.encoding is None:
                resp.encoding = "utf-8"
            # Raw RGB frame events are multi-megabyte NDJSON lines. Requests'
            # 512-byte default repeatedly concatenates the growing partial
            # line and becomes quadratic at 720p, so read them in large slabs.
            lines = resp.iter_lines(
                chunk_size=_STREAM_READ_CHUNK_SIZE,
                decode_unicode=True,
            )
            if self._nvtx is None:
                for line in lines:
                    if not isinstance(line, str):
                        continue
                    parsed = parse_ndjson_line(line)
                    if parsed is None:
                        continue
                    yield self._to_event(parsed)
                return

            range_push, range_pop = self._nvtx
            while True:
                # Socket read, UTF-8 decode and line reassembly. At 720p one
                # line is ~14.7 MB of ASCII, so this is where transport time
                # and the client's own copying both land.
                range_push("client.iter_lines")
                try:
                    line = next(lines)
                except StopIteration:
                    range_pop()
                    break
                range_pop()
                if not isinstance(line, str):
                    continue
                # json.loads over that line plus the base64 decode back to
                # the original 11 MiB of RGB.
                range_push(f"client.parse_ndjson.chars[{len(line)}]")
                parsed = parse_ndjson_line(line)
                range_pop()
                if parsed is None:
                    continue
                range_push("client.to_event")
                event = self._to_event(parsed)
                range_pop()
                yield event

    @staticmethod
    def _to_event(parsed: dict) -> StreamEvent:
        modality = parsed["modality"]
        raw = parsed["bytes"]
        meta = parsed["metadata"]
        if modality == "error":
            status = meta.get("status")
            status_suffix = f" (status {status})" if status is not None else ""
            raise RuntimeError(
                f"Server stream failed{status_suffix}: "
                f"{raw.decode('utf-8', 'replace')}"
            )
        if modality == "text":
            return TextChunk(raw.decode("utf-8", "replace"), meta)
        if modality == "image":
            return ImageChunk(raw, meta)
        if modality == "audio":
            return AudioChunk(raw, int(meta.get("sample_rate", 24000)), meta)
        if modality == "video_frame":
            return VideoFrameChunk(raw, meta)
        # Existing action/scalar/tensor streams are text-compatible. Keep the
        # historical fallback while giving raw video frames their strict type.
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
