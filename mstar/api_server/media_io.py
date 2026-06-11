"""Media decode/encode helpers shared by the native and OpenAI-compatible APIs.

Two directions:

* **Inbound** — turn media referenced by an OpenAI-style request (``data:`` URLs,
  ``http(s)`` URLs, or base64 blobs) into files under the API server's
  ``upload_dir``, so ``model.load_image`` / ``load_audio`` / ``load_video`` can
  read them by path. This is the same contract ``/generate`` already uses for
  multipart uploads.
* **Outbound** — wrap raw model audio output (16-bit PCM, no container header)
  into a real audio container (WAV by default) and encode image bytes (PNG) as a
  ``data:`` URL for OpenAI chat image output.

Only stdlib + numpy are required. ``mp3`` / ``flac`` / ``ogg`` encoding is opt-in
and degrades to WAV when the optional ``soundfile`` backend is unavailable, so
the base install stays slim.
"""

from __future__ import annotations

import base64
import io
import logging
import wave
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

# MIME (top-level type or full type) -> file extension used when persisting.
_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}

# Output audio container -> (extension, mime type) for OpenAI audio responses.
AUDIO_FORMAT_MIME: dict[str, str] = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}


def modality_from_mime(mime: str) -> str:
    """Map a MIME type to one of our modality strings (image/audio/video)."""
    top = (mime or "").split("/", 1)[0].lower()
    if top in ("image", "audio", "video"):
        return top
    return "unknown"


def _ext_for(mime: str, fallback: str = ".bin") -> str:
    mime = (mime or "").lower()
    if mime in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime]
    # Fall back to the top-level type's most common extension.
    top = mime.split("/", 1)[0]
    return {"image": ".png", "audio": ".wav", "video": ".mp4"}.get(top, fallback)


# ---------------------------------------------------------------------------
# Inbound: persist request media into upload_dir, return (modality, path)
# ---------------------------------------------------------------------------

def _save_bytes(raw: bytes, mime: str, upload_dir: Path) -> tuple[str, str]:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid4()}{_ext_for(mime)}"
    path.write_bytes(raw)
    return modality_from_mime(mime), str(path)


def save_data_url(data_url: str, upload_dir: Path) -> tuple[str, str]:
    """Persist a ``data:<mime>;base64,<payload>`` URL. Returns (modality, path)."""
    header, _, payload = data_url.partition(",")
    if not payload:
        raise ValueError("Malformed data URL: missing payload")
    mime = header[len("data:"):].split(";", 1)[0] or "application/octet-stream"
    raw = base64.b64decode(payload)
    return _save_bytes(raw, mime, upload_dir)


def save_base64(b64: str, fmt: str, modality_hint: str, upload_dir: Path) -> tuple[str, str]:
    """Persist a bare base64 blob with a known ``fmt`` (e.g. ``"wav"``)."""
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(b64)
    ext = "." + fmt.lstrip(".") if fmt else ".bin"
    path = upload_dir / f"{uuid4()}{ext}"
    path.write_bytes(raw)
    return modality_hint, str(path)


def save_remote_url(url: str, upload_dir: Path, timeout: float = 30.0) -> tuple[str, str]:
    """Download an ``http(s)`` URL into ``upload_dir``. Returns (modality, path).

    Note: fetching arbitrary URLs has SSRF surface. Callers exposing this
    publicly should allowlist hosts or disable remote fetch (data-URL only).
    """
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (caller-gated)
        raw = resp.read()
        mime = resp.headers.get_content_type() if resp.headers else ""
    if not mime:
        # Infer from the URL path extension.
        suffix = Path(urlparse(url).path).suffix.lower()
        rev = {v: k for k, v in _MIME_TO_EXT.items()}
        mime = rev.get(suffix, "application/octet-stream")
    return _save_bytes(raw, mime, upload_dir)


def resolve_media_ref(ref: str, upload_dir: Path, *, allow_remote: bool = True) -> tuple[str, str]:
    """Resolve a media reference (data URL, http(s) URL, or local path).

    Returns ``(modality, path)``. Local paths are passed through unchanged
    (modality inferred from extension).
    """
    if ref.startswith("data:"):
        return save_data_url(ref, upload_dir)
    scheme = urlparse(ref).scheme.lower()
    if scheme in ("http", "https"):
        if not allow_remote:
            raise ValueError("Remote media fetch is disabled on this server")
        return save_remote_url(ref, upload_dir)
    # Treat as a local filesystem path.
    suffix = Path(ref).suffix.lower()
    rev = {v: k for k, v in _MIME_TO_EXT.items()}
    return modality_from_mime(rev.get(suffix, "")), ref


# ---------------------------------------------------------------------------
# Outbound: wrap raw model output for client surfaces
# ---------------------------------------------------------------------------

def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw little-endian 16-bit PCM (the model's audio output) into a WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def pcm16_to_container(pcm: bytes, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    """Encode raw 16-bit PCM into ``fmt``. Returns ``(bytes, mime_type)``.

    ``wav`` and ``pcm`` use the stdlib (the bytes are already PCM_16). Compressed
    formats need the optional ``soundfile`` backend; if it is missing we fall back
    to WAV and log once.
    """
    fmt = (fmt or "wav").lower()
    if fmt == "wav":
        return pcm16_to_wav_bytes(pcm, sample_rate), AUDIO_FORMAT_MIME["wav"]
    if fmt == "pcm":
        return pcm, AUDIO_FORMAT_MIME["pcm"]

    try:
        import soundfile as sf  # type: ignore

        audio = np.frombuffer(pcm, dtype="<i2")
        buf = io.BytesIO()
        sf.write(buf, audio, int(sample_rate), format=fmt.upper())
        return buf.getvalue(), AUDIO_FORMAT_MIME.get(fmt, "application/octet-stream")
    except Exception:  # noqa: BLE001 — any backend failure degrades to WAV
        logger.warning("Audio format %r unavailable (need soundfile); returning WAV", fmt)
        return pcm16_to_wav_bytes(pcm, sample_rate), AUDIO_FORMAT_MIME["wav"]


def wav_stream_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    """A 44-byte WAV header with streaming (unknown-length) size fields.

    Used to stream TTS audio over a single HTTP response: emit this header, then
    16-bit PCM frames as they arrive. The 0xFFFFFFFF placeholders signal an
    open-ended stream, which players and the OpenAI client's
    ``stream_to_file`` handle.
    """
    import struct

    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def png_to_data_url(png_bytes: bytes) -> str:
    """Encode PNG image bytes (the model's image output) as a data URL."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
