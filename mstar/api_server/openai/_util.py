"""Small shared helpers for the OpenAI-compatible serving handlers."""

from __future__ import annotations

import json
import time
import uuid

SSE_DONE = "data: [DONE]\n\n"


def now() -> int:
    return int(time.time())


def rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def sse(obj: dict) -> str:
    return f"data: {_dumps(obj)}\n\n"


try:
    import orjson

    def _dumps(obj: dict) -> str:
        # ~5x json.dumps; one event per token at tens of thousands a second
        return orjson.dumps(obj).decode()
except ImportError:  # pragma: no cover
    def _dumps(obj: dict) -> str:
        return json.dumps(obj)


def error_type(status: int) -> str:
    """The OpenAI error ``type`` for an HTTP status: a 4xx is the client's
    (``invalid_request_error``), anything else is ours (``server_error``)."""
    return "invalid_request_error" if 400 <= status < 500 else "server_error"
