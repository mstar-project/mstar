"""Small shared helpers for the OpenAI-compatible serving handlers."""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.responses import Response

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


class MediaJSONResponse(Response):
    """The JSON body of an images or videos response, whose ``b64_json`` values the
    handlers leave as the ``bytes`` ``base64.b64encode`` returned. A bytes value is
    written as a JSON string as it is (base64 needs no escaping); everything else is
    rendered as ``JSONResponse`` renders it, so the body is the one ``JSONResponse``
    gives for the same values as str. That render escapes an ``ensure_ascii=False``
    string a character at a time, on the event loop: about 40 ms for a 1024x1024
    PNG and 160 ms for an 8 MB clip, against 6 and 22 ms for this body."""

    media_type = "application/json"

    def render(self, content) -> bytes:
        parts: list[bytes] = []
        _render_json(content, parts)
        return b"".join(parts)


def _render_json(value, parts: list[bytes]) -> None:
    if isinstance(value, bytes):
        parts += (b'"', value, b'"')
    elif isinstance(value, dict):
        parts.append(b"{")
        for i, (key, item) in enumerate(value.items()):
            if i:
                parts.append(b",")
            parts += (_json_bytes(key), b":")
            _render_json(item, parts)
        parts.append(b"}")
    elif isinstance(value, list):
        parts.append(b"[")
        for i, item in enumerate(value):
            if i:
                parts.append(b",")
            _render_json(item, parts)
        parts.append(b"]")
    else:
        parts.append(_json_bytes(value))


def _json_bytes(value) -> bytes:
    # JSONResponse.render's settings
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


def error_type(status: int) -> str:
    """The OpenAI error ``type`` for an HTTP status: a 4xx is the client's
    (``invalid_request_error``), anything else is ours (``server_error``)."""
    return "invalid_request_error" if 400 <= status < 500 else "server_error"
