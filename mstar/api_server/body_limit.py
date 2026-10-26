"""Request body limit for the Python frontend (``MSTAR_MAX_BODY_MB``).

uvicorn has no body limit of its own, and every route reads its whole body
(JSON with base64 media, multipart uploads spooled to disk), so without this a
single request can fill the API server's memory or disk. The Rust frontend
enforces the same variable with the same default (``DefaultBodyLimit``).
"""
from __future__ import annotations

import json
import os
import re

from mstar.api_server.openai.serving_transcriptions import MAX_UPLOAD_BYTES as TRANSCRIPTIONS_MAX_UPLOAD_BYTES

DEFAULT_MAX_BODY_MB = 128
# transcriptions accept a 30 min clip, which as 16-bit 48 kHz stereo WAV is
# about 330 MiB, so that route keeps its own larger file cap
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
_FORM_FIELDS_SLACK = 2**20


def max_body_bytes() -> int:
    """``MSTAR_MAX_BODY_MB`` in bytes, parsed as the Rust frontend parses it:
    a value that is not a non-negative integer means the default."""
    raw = os.environ.get("MSTAR_MAX_BODY_MB", "").strip()
    mb = int(raw) if re.fullmatch(r"\+?[0-9]+", raw) else DEFAULT_MAX_BODY_MB
    return mb << 20


def _mib(n: int) -> str:
    return f"{n / 2**20:g} MiB"


class BodyTooLargeError(Exception):
    status_code = 413

    def __init__(self, limit: int):
        self.limit = limit
        self.detail = f"request body is larger than the {_mib(limit)} limit (MSTAR_MAX_BODY_MB)"
        super().__init__(self.detail)


class BodyLimitMiddleware:
    """Refuse a body over the limit with a 413: up front on ``Content-Length``,
    else by counting the bytes as the app reads them (chunked uploads).

    Once the count passes the limit the app's own read fails, and whatever
    response the app makes of that failure (route handlers turn exceptions
    into 400s or 500s) is replaced by the 413.
    """

    def __init__(self, app, max_bytes: int | None = None):
        self.app = app
        self.max_bytes = max_body_bytes() if max_bytes is None else max_bytes

    def _limit_for(self, path: str) -> int:
        if path == TRANSCRIPTIONS_PATH:
            return max(self.max_bytes, TRANSCRIPTIONS_MAX_UPLOAD_BYTES + _FORM_FIELDS_SLACK)
        return self.max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        # as Starlette routes: the root path is a prefix only up to a "/"
        if root and (path == root or path.startswith(root.rstrip("/") + "/")):
            path = path[len(root.rstrip("/")):]
        limit = self._limit_for(path)

        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > limit:
                    await _reject(send, path, limit)
                    return
                break

        received = 0
        exceeded = False
        started = False

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                raise BodyTooLargeError(limit)
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise BodyTooLargeError(limit)
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await _reject(send, path, limit)


async def _reject(send, path: str, limit: int) -> None:
    message = BodyTooLargeError(limit).detail
    if path.startswith("/v1/"):
        payload = {"error": {"message": message, "type": "invalid_request_error", "code": 413}}
    else:
        payload = {"detail": message}
    body = json.dumps(payload).encode()
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ],
    })
    await send({"type": "http.response.body", "body": body})
