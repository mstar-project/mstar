"""Result-stream frames as the server writes them, for client-side tests.

mstar-server (``binary_frame`` and ``ndjson_line`` in rust/server/src/main.rs)
writes each result chunk either as an NDJSON line with a base64 payload, or,
when the client asks for ``application/vnd.mstar.frames``, as a one-line JSON
header naming ``nbytes`` followed by the untouched payload. ``json.dumps``
escapes control characters, so a header never holds a raw newline.

Import it by module name (``from stream_frames import ...``): on CI the
stdlib ``test`` package shadows the repo's ``test/`` directory.
"""
import base64
import json

from mstar.api_server.request_types import ResultChunk


def binary_frame(chunk: ResultChunk) -> tuple[bytes, bytes]:
    """``chunk`` as a binary-framing header line and its payload."""
    header = json.dumps(
        {"modality": chunk.modality, "nbytes": len(chunk.data), "metadata": chunk.metadata},
        separators=(",", ":"),
    )
    return header.encode("utf-8") + b"\n", chunk.data


def ndjson_line(chunk: ResultChunk) -> str:
    """``chunk`` as one NDJSON line."""
    return json.dumps({
        "modality": chunk.modality,
        "data": base64.b64encode(chunk.data).decode("ascii"),
        "metadata": chunk.metadata,
    }) + "\n"
