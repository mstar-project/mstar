"""Low-level decode/encode helpers for the SDK.

Kept self-contained (stdlib + numpy only, no torch / server imports) so the
client stays light and independently importable.
"""

from __future__ import annotations

import base64
import io
import json
import wave
from collections.abc import Iterator

NDJSON_STREAM_MEDIA_TYPE = "application/x-ndjson"
# Opt-in framing for raw binary payloads, requested via ``Accept``. Duplicated
# rather than imported from the server so this module keeps its stdlib-only
# contract; ``test_binary_framing.py`` asserts the two stay equal.
BINARY_STREAM_MEDIA_TYPE = "application/vnd.mstar.frames"

_FRAME_HEADER_READ_SIZE = 64 * 1024


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw little-endian 16-bit PCM into a WAV blob (stdlib only)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return buf.getvalue()


def parse_ndjson_line(line: str) -> dict | None:
    """Parse one NDJSON line from ``/generate`` streaming into a decoded dict.

    Returns ``{"modality", "bytes", "metadata"}`` or ``None`` for blank /
    unparseable lines. A top-level ``error`` is the Rust frontend's in-band
    failure envelope and raises rather than being mistaken for an empty chunk.
    """
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    if "error" in msg:
        raise RuntimeError(f"Server stream failed: {msg['error']}")
    data = msg.get("data")
    return {
        "modality": msg.get("modality"),
        "bytes": base64.b64decode(data) if data else b"",
        "metadata": msg.get("metadata") or {},
    }


def iter_binary_frames(raw, read_size: int = _FRAME_HEADER_READ_SIZE) -> Iterator[dict]:
    """Decode a length-framed binary stream into the same dicts as NDJSON.

    Each frame is a one-line JSON header (``modality``, ``nbytes``,
    ``metadata``) followed by exactly ``nbytes`` of untouched payload. Framing
    by length rather than by delimiter is what lets the payload travel raw: at
    720p the NDJSON form spends ~85 ms per chunk base64-ing 11 MB and scanning
    the result for characters that need escaping, against a 66.67 ms budget.

    Yields ``{"modality", "bytes", "metadata"}``, matching
    :func:`parse_ndjson_line`, so both protocols share one event path.

    ``raw`` is anything with ``read(n) -> bytes``: ``requests``' ``resp.raw`` in
    production, a ``BytesIO`` in tests. ``read_size`` only bounds the header
    scan; payload reads ask for exactly what is outstanding.
    """
    # ``read(n)`` blocks until ``n`` bytes or EOF, which stalls the header scan
    # behind a full 64 KiB on small frames. ``read1`` returns as soon as any
    # bytes are available, like a single ``recv()``; fall back to ``read`` for
    # objects that lack it.
    read_header = raw.read1 if hasattr(raw, "read1") else raw.read
    buf = bytearray()
    while True:
        newline = buf.find(b"\n")
        while newline < 0:
            block = read_header(read_size)
            if not block:
                if buf:
                    raise RuntimeError(
                        f"Binary stream ended mid-header after {len(buf)} bytes"
                    )
                return  # Clean end: the server closed between frames.
            buf += block
            newline = buf.find(b"\n")

        # Unlike NDJSON, a malformed header is not recoverable by skipping the
        # line: without ``nbytes`` there is no way to find the next frame, so
        # this raises rather than returning None.
        header = json.loads(bytes(buf[:newline]))
        del buf[: newline + 1]
        if "modality" not in header and "error" in header:
            raise RuntimeError(f"Server stream failed: {header['error']}")

        nbytes = header.get("nbytes") or 0
        parts: list[bytes] = []
        buffered = min(len(buf), nbytes)
        if buffered:
            parts.append(bytes(buf[:buffered]))
            del buf[:buffered]
        have = buffered
        while have < nbytes:
            block = raw.read(nbytes - have)
            if not block:
                raise RuntimeError(
                    f"Binary stream ended {nbytes - have} bytes short of the "
                    f"{nbytes}-byte {header.get('modality')!r} payload"
                )
            parts.append(block)
            have += len(block)

        yield {
            "modality": header.get("modality"),
            # Must be ``bytes``: VideoFrameChunk rejects ``bytearray``. The join
            # is the only copy of the payload on this path, and a socket read
            # that satisfied the frame in one go skips even that.
            "bytes": parts[0] if len(parts) == 1 else b"".join(parts),
            "metadata": header.get("metadata") or {},
        }
