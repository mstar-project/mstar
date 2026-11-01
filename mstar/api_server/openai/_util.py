"""Small shared helpers for the OpenAI-compatible serving handlers."""

from __future__ import annotations

import json
import logging
import time
import uuid

import pydantic_core
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

SSE_DONE = "data: [DONE]\n\n"


def now() -> int:
    return int(time.time())


def rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def release_unread(api, request_ids: list[str], read: int) -> None:
    """Release every submitted request from ``request_ids[read]`` on (not yet read).

    Runs while the caller's own exception is propagating, so a release that
    raises is logged and skipped: it must neither stop the others nor replace
    that exception.
    """
    for request_id in request_ids[read:]:
        try:
            api.release_request(request_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to release request %s: %r", request_id, exc)


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


class MediaJSONResponse(JSONResponse):
    """JSON for a body that is mostly a base64 string. ``json.dumps`` escapes it one character at a
    time on the event loop, 18 ms for a 1024x1024 PNG; this writes the same bytes in under 1 ms."""

    def render(self, content) -> bytes:
        return pydantic_core.to_json(content)


def error_type(status: int) -> str:
    """The OpenAI error ``type`` for an HTTP status: a 4xx is the client's
    (``invalid_request_error``), anything else is ours (``server_error``)."""
    return "invalid_request_error" if 400 <= status < 500 else "server_error"
