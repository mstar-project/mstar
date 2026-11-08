"""MSTAR_MAX_BODY_MB in the Python frontend: a body over the limit gets a 413,
declared (Content-Length) or not (chunked), on every route (no GPU)."""

import sys
import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

MIB = 2**20


@pytest.fixture
def server(monkeypatch):
    """The real app's middleware stack, built with a 1 MiB limit."""
    from mstar.api_server import entrypoint

    monkeypatch.setenv("MSTAR_MAX_BODY_MB", "1")
    monkeypatch.setattr(entrypoint, "api_server", types.SimpleNamespace(model_name="bagel"))
    return TestClient(entrypoint.app.build_middleware_stack())


def _chunks(total: int, size: int = 64 * 1024):
    sent = 0
    while sent < total:
        n = min(size, total - sent)
        sent += n
        yield b"x" * n


def test_generate_upload_over_limit_is_413(server):
    r = server.post("/generate", files={"files": ("a.wav", b"x" * (2 * MIB))})
    assert r.status_code == 413
    assert "1 MiB" in r.json()["detail"] and "MSTAR_MAX_BODY_MB" in r.json()["detail"]


def test_openai_json_over_limit_is_413_envelope(server):
    big = "a" * (2 * MIB)
    r = server.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": big}]})
    assert r.status_code == 413
    err = r.json()["error"]
    assert err["code"] == 413 and err["type"] == "invalid_request_error"
    assert "MSTAR_MAX_BODY_MB" in err["message"]


def test_chunked_body_over_limit_is_413(server):
    # no Content-Length: refused while the app reads it, and the 413 replaces
    # whatever the route made of its failed read
    r = server.post(
        "/generate",
        content=_chunks(2 * MIB),
        headers={"content-type": "multipart/form-data; boundary=b"},
    )
    assert "content-length" not in r.request.headers
    assert r.status_code == 413
    r = server.post(
        "/v1/images/edits",
        content=_chunks(2 * MIB),
        headers={"content-type": "multipart/form-data; boundary=b"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == 413


def test_body_under_limit_reaches_the_route(server):
    r = server.post("/generate", data={"model_kwargs": "{"})
    assert r.status_code == 400
    assert r.json() == {"detail": "model_kwargs must be valid JSON"}
    r = server.post("/generate", content=_chunks(MIB // 2), headers={"content-type": "application/octet-stream"})
    assert r.status_code != 413


def test_entrypoint_installs_limit_inside_cors():
    from mstar.api_server import entrypoint
    from mstar.api_server.body_limit import BodyLimitMiddleware

    names = [m.cls.__name__ for m in entrypoint.app.user_middleware]
    assert BodyLimitMiddleware.__name__ in names
    # user_middleware[0] is outermost: CORS must wrap the 413
    assert names.index("CORSMiddleware") < names.index(BodyLimitMiddleware.__name__)


@pytest.mark.parametrize(
    "raw, mb",
    [(None, 128), ("", 128), ("16", 16), (" 16 ", 16), ("0", 0), ("abc", 128), ("1.5", 128), ("-1", 128)],
)
def test_max_body_bytes_parses_like_the_rust_frontend(monkeypatch, raw, mb):
    from mstar.api_server.body_limit import max_body_bytes

    if raw is None:
        monkeypatch.delenv("MSTAR_MAX_BODY_MB", raising=False)
    else:
        monkeypatch.setenv("MSTAR_MAX_BODY_MB", raw)
    assert max_body_bytes() == mb * MIB


def test_transcriptions_keep_their_file_cap_and_root_path_is_stripped():
    from mstar.api_server.body_limit import TRANSCRIPTIONS_MAX_UPLOAD_BYTES, BodyLimitMiddleware

    mw = BodyLimitMiddleware(app=None, max_bytes=MIB)
    assert mw._limit_for("/v1/audio/transcriptions") > TRANSCRIPTIONS_MAX_UPLOAD_BYTES
    assert mw._limit_for("/v1/chat/completions") == MIB
    big = BodyLimitMiddleware(app=None, max_bytes=2 * TRANSCRIPTIONS_MAX_UPLOAD_BYTES)
    assert big._limit_for("/v1/audio/transcriptions") == 2 * TRANSCRIPTIONS_MAX_UPLOAD_BYTES


def test_root_path_prefix_is_matched(monkeypatch):
    from mstar.api_server.body_limit import BodyLimitMiddleware

    seen = {}

    async def app(scope, receive, send):
        seen["called"] = True

    mw = BodyLimitMiddleware(app, max_bytes=MIB)
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    import asyncio

    big = str(2 * MIB).encode()
    scope = {"type": "http", "path": "/proj/job/v1/audio/transcriptions", "root_path": "/proj/job",
             "headers": [(b"content-length", big)]}
    asyncio.run(mw(scope, receive, send))
    assert seen.get("called") and not sent
    scope = {"type": "http", "path": "/proj/job/v1/chat/completions", "root_path": "/proj/job",
             "headers": [(b"content-length", big)]}
    asyncio.run(mw(scope, receive, send))
    assert sent[0]["status"] == 413
    # "/v" is not a whole-segment prefix of "/v1/...": the route keeps its own cap
    seen.clear()
    rejected = len(sent)
    for root in ("/v", "/proj/job/"):
        path = "/v1/audio/transcriptions" if root == "/v" else "/proj/job/v1/audio/transcriptions"
        scope = {"type": "http", "path": path, "root_path": root, "headers": [(b"content-length", big)]}
        asyncio.run(mw(scope, receive, send))
        assert seen.pop("called", False) and len(sent) == rejected, root


def test_streamed_chunks_are_forwarded_as_the_app_sends_them():
    """A streamed response (SSE, /generate's chunk stream) is not held back:
    each message reaches the server before the app's send returns, and a
    disconnect still reaches the app once the body is read."""
    import asyncio

    from mstar.api_server.body_limit import BodyLimitMiddleware

    sent = []
    seen_by_app = []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for i in range(3):
            message = {"type": "http.response.body", "body": b"c%d" % i, "more_body": True}
            await send(message)
            seen_by_app.append(sent[-1] is message)
        seen_by_app.append((await receive())["type"])

    inbox = [{"type": "http.request", "body": b"{}", "more_body": False}, {"type": "http.disconnect"}]

    async def receive():
        return inbox.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/v1/chat/completions", "headers": [(b"content-length", b"2")]}
    asyncio.run(BodyLimitMiddleware(app, max_bytes=MIB)(scope, receive, send))
    assert seen_by_app == [True, True, True, "http.disconnect"]
    assert [m.get("body") for m in sent[1:]] == [b"c0", b"c1", b"c2"]


def test_images_edits_names_its_upload_limit(monkeypatch):
    # the route checks the file itself too, so the limit holds when the router
    # is mounted without the entrypoint's middleware
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    fake_ep.api_server = types.SimpleNamespace(model_name="bagel")
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    monkeypatch.setenv("MSTAR_MAX_BODY_MB", "1")
    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    r = TestClient(app).post(
        "/v1/images/edits", files={"image": ("a.png", b"x" * (MIB + 1))}, data={"prompt": "p"}
    )
    assert r.status_code == 413
    assert r.json()["error"]["message"] == "images/edits accepts files up to 1 MiB (MSTAR_MAX_BODY_MB)"
