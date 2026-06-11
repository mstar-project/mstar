"""Low-level decode/encode helpers for the SDK.

Kept self-contained (stdlib + numpy only, no torch / server imports) so the
client stays light and independently importable.
"""

from __future__ import annotations

import base64
import io
import json
import wave


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
    unparseable lines.
    """
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    data = msg.get("data")
    return {
        "modality": msg.get("modality"),
        "bytes": base64.b64decode(data) if data else b"",
        "metadata": msg.get("metadata") or {},
    }
