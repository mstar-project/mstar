"""Optional API key for the HTTP and WebSocket surface.

Off unless ``--api-key`` or ``MSTAR_API_KEY`` sets a key. When set, every
request except the probe and scrape paths in ``UNAUTHENTICATED_PATHS`` must carry
``Authorization: Bearer <key>`` (what the OpenAI SDKs send for ``api_key``);
WebSocket routes are checked at the handshake. The check is one ASGI
middleware over the whole app rather than a per-route dependency, so a route
added later is covered without anyone remembering to opt it in.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

API_KEY_ENV = "MSTAR_API_KEY"

#: Served without a key: liveness probes from load balancers and orchestrators,
#: and the Prometheus scrape endpoint (request counts, latencies and the model
#: name; never prompts or outputs).
UNAUTHENTICATED_PATHS = frozenset({"/health", "/metrics"})

_UNAUTHORIZED_BODY = json.dumps({"error": {
    "message": "missing or invalid API key; send 'Authorization: Bearer <key>'",
    "type": "invalid_request_error",
    "code": 401,
}}).encode()


def resolve_api_key(flag: str | None = None) -> str | None:
    """The deployment's key: ``--api-key`` if given, else ``MSTAR_API_KEY``.
    An empty value means no key."""
    key = flag if flag is not None else os.environ.get(API_KEY_ENV)
    return key or None


def keep_api_key_from_children(api_key: str | None) -> None:
    """Only the HTTP frontend needs the key. Drop it from the environment and
    ``sys.argv`` (the spawn start method copies both into each child) before
    this process spawns the conductor, workers or media subprocesses."""
    os.environ.pop(API_KEY_ENV, None)
    if api_key is None:
        return
    redacted = []
    for arg in sys.argv:
        if arg == api_key:
            arg = "<redacted>"
        elif arg.partition("=")[2] == api_key:
            arg = arg.partition("=")[0] + "=<redacted>"
        redacted.append(arg)
    sys.argv[:] = redacted


def warn_if_no_api_key(api_key: str | None, host: str, port: int) -> None:
    if api_key is None:
        logger.warning(
            "No API key set (--api-key or %s): any client that can reach %s:%s can "
            "run requests. Set a key, or bind --host 127.0.0.1, unless a gateway in "
            "front of this server authenticates clients.", API_KEY_ENV, host, port)


def _digest(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


class APIKeyGuard:
    """Holds the key (if any) and checks a request's headers against it.

    Mutable so the module-level app can install the middleware at import and
    the CLI can set the key once its flags are parsed.
    """

    def __init__(self, key: str | None = None):
        self.set(key)

    def set(self, key: str | None) -> None:
        self._digest = _digest(key.encode()) if key else None

    @property
    def enabled(self) -> bool:
        return self._digest is not None

    def accepts(self, headers: list[tuple[bytes, bytes]]) -> bool:
        if self._digest is None:
            return True
        token = _bearer_token(headers)
        # Comparing digests keeps the time independent of the key's length too.
        return token is not None and hmac.compare_digest(_digest(token), self._digest)


def _bearer_token(headers: list[tuple[bytes, bytes]]) -> bytes | None:
    for name, value in headers:
        if name == b"authorization":
            scheme, _, token = value.strip().partition(b" ")
            return token.strip() if scheme.lower() == b"bearer" else None
    return None


def _route_path(scope: dict) -> str:
    """The path the router matches on: with a ``root_path`` (``MSTAR_ROOT_PATH``
    behind a prefix-keeping ingress) the prefix is stripped, as Starlette's
    router does, so ``/<prefix>/health`` is the probe and nothing else is."""
    path = scope.get("path", "")
    root = scope.get("root_path", "")
    if not root or not path.startswith(root):
        return path
    if path == root:
        return ""
    return path[len(root):] if path[len(root)] == "/" else path


class APIKeyMiddleware:
    """Pure ASGI (not ``BaseHTTPMiddleware``), so it sees WebSocket scopes and
    never buffers a streamed response."""

    def __init__(self, app, guard: APIKeyGuard):
        self.app = app
        self.guard = guard

    async def __call__(self, scope, receive, send):
        if (scope["type"] not in ("http", "websocket")
                or not self.guard.enabled
                or _route_path(scope) in UNAUTHENTICATED_PATHS
                or self.guard.accepts(scope.get("headers", []))):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # Closing before accept refuses the handshake (HTTP 403).
            await send({"type": "websocket.close", "code": 1008, "reason": "invalid API key"})
            return
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
