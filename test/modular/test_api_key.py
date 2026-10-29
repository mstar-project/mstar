"""The optional API key (``--api-key`` / ``MSTAR_API_KEY``) on the served app.

One middleware covers every HTTP route and the WebSocket handshakes
(``/generate/ws``, ``/v1/realtime``); ``/health`` and CORS preflights pass
without a key, and nothing changes when no key is set.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from mstar.api_server import auth, entrypoint  # noqa: E402

KEY = "s3cret-key"
GOOD = {"Authorization": f"Bearer {KEY}"}
BAD_HEADERS = [
    {},
    {"Authorization": "Bearer wrong"},
    {"Authorization": f"Bearer {KEY}x"},
    {"Authorization": f"Basic {KEY}"},
    {"Authorization": KEY},
]


@pytest.fixture
def keyed():
    entrypoint.api_key_guard.set(KEY)
    yield TestClient(entrypoint.app)
    entrypoint.api_key_guard.set(auth.resolve_api_key())


class _FakeServer:
    def __init__(self, upload_dir):
        self.upload_dir = upload_dir


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/models"),
    ("GET", "/v1/audio/voices"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/audio/speech"),
    ("POST", "/generate"),
    ("GET", "/docs"),
    ("GET", "/no-such-route"),
])
@pytest.mark.parametrize("headers", BAD_HEADERS)
def test_every_route_refuses_a_missing_or_wrong_key(keyed, method, path, headers):
    r = keyed.request(method, path, headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == 401


def test_the_right_key_reaches_the_route(keyed):
    r = keyed.get("/v1/models", headers=GOOD)
    assert r.status_code == 200 and r.json()["object"] == "list"
    assert keyed.get("/v1/models", headers={"Authorization": f"bearer  {KEY} "}).status_code == 200


def test_health_needs_no_key(keyed):
    assert keyed.get("/health").json() == {"status": "healthy"}


def test_no_key_configured_leaves_every_route_open():
    entrypoint.api_key_guard.set(None)
    try:
        assert TestClient(entrypoint.app).get("/v1/models").status_code == 200
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())


@pytest.mark.parametrize("path", ["/generate/ws", "/v1/realtime"])
def test_websocket_handshake_needs_the_key(keyed, monkeypatch, tmp_path, path):
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(tmp_path))
    with pytest.raises(WebSocketDisconnect) as info:
        with keyed.websocket_connect(path, headers={"Authorization": "Bearer wrong"}):
            pass
    assert info.value.code == 1008


def test_websocket_with_the_key_is_accepted(keyed, monkeypatch, tmp_path):
    monkeypatch.setattr(entrypoint, "api_server", _FakeServer(tmp_path))
    with keyed.websocket_connect("/generate/ws", headers=GOOD) as ws:
        ws.send_text("not json")
        assert "undecodable" in json.loads(ws.receive_text())["error"]


def test_cors_preflight_needs_no_key_and_a_401_keeps_cors_headers(keyed):
    origin = {"Origin": "https://app.example"}
    pre = keyed.options("/v1/chat/completions", headers={
        **origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    })
    assert pre.status_code == 200 and "access-control-allow-origin" in pre.headers
    denied = keyed.get("/v1/models", headers=origin)
    assert denied.status_code == 401 and "access-control-allow-origin" in denied.headers


def test_an_oversized_body_without_the_key_is_a_401_and_with_it_a_413(keyed, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_BODY_MB", "1")
    client = TestClient(entrypoint.app.build_middleware_stack())
    big = b"x" * (2 * 2**20)
    assert client.post("/v1/chat/completions", content=big).status_code == 401
    assert client.post("/v1/chat/completions", content=big, headers=GOOD).status_code == 413


def _sub_path_app():
    app = FastAPI(root_path="/proj/job")
    app.add_middleware(auth.APIKeyMiddleware, guard=auth.APIKeyGuard(KEY))
    app.get("/health")(lambda: {"status": "healthy"})
    app.get("/v1/models")(lambda: {"object": "list"})
    return TestClient(app)


@pytest.mark.parametrize("path", ["/proj/job/health", "/health"])
def test_health_under_a_root_path_needs_no_key(path):
    assert _sub_path_app().get(path).status_code == 200


@pytest.mark.parametrize("path", ["/proj/job/v1/models", "/proj/jobhealth", "/proj/job/health/"])
def test_other_paths_under_a_root_path_need_the_key(path):
    assert _sub_path_app().get(path).status_code == 401


def test_flag_wins_over_env_and_empty_is_no_key(monkeypatch):
    monkeypatch.setenv(auth.API_KEY_ENV, "from-env")
    assert auth.resolve_api_key() == "from-env"
    assert auth.resolve_api_key("from-flag") == "from-flag"
    assert auth.resolve_api_key("") is None
    monkeypatch.setenv(auth.API_KEY_ENV, "")
    assert auth.resolve_api_key() is None


def test_startup_warns_only_without_a_key(caplog):
    with caplog.at_level(logging.WARNING, logger=auth.__name__):
        auth.warn_if_no_api_key(KEY, "0.0.0.0", 8000)
        assert not caplog.records
        auth.warn_if_no_api_key(None, "0.0.0.0", 8000)
    (record,) = caplog.records
    assert "MSTAR_API_KEY" in record.getMessage() and "0.0.0.0:8000" in record.getMessage()


@pytest.mark.parametrize("via", ["flag", "env"])
def test_serve_main_applies_the_key_and_keeps_it_from_child_processes(monkeypatch, tmp_path, via):
    argv = ["--config", str(tmp_path / "missing.yaml")]
    if via == "flag":
        monkeypatch.delenv(auth.API_KEY_ENV, raising=False)
        argv += ["--api-key", KEY]
    else:
        monkeypatch.setenv(auth.API_KEY_ENV, KEY)
    monkeypatch.setattr("sys.argv", ["mstar", "serve", "kokoro", f"--api-key={KEY}", "--api-key", KEY])
    try:
        # The config is read right after the key is applied; a missing one stops
        # main() before it spawns anything.
        with pytest.raises(FileNotFoundError):
            entrypoint.main(argv)
        assert entrypoint.api_key_guard.accepts([(b"authorization", f"Bearer {KEY}".encode())])
        assert not entrypoint.api_key_guard.accepts([])
        # The conductor and workers are spawned later, from this environment
        # and (spawn start method) this sys.argv.
        assert auth.API_KEY_ENV not in os.environ
        assert not any(KEY in a for a in sys.argv)
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())


def test_dynamo_worker_keeps_an_exported_key_from_child_processes(monkeypatch, tmp_path):
    """The Dynamo worker starts its conductor without ``entrypoint.main()``."""
    from types import SimpleNamespace

    from mstar.integrations.dynamo import worker as dynamo_worker

    monkeypatch.setenv(auth.API_KEY_ENV, KEY)
    # The config is read before the model is built or anything is spawned.
    with pytest.raises(FileNotFoundError):
        dynamo_worker.build_server(SimpleNamespace(config=str(tmp_path / "missing.yaml")))
    assert auth.API_KEY_ENV not in os.environ


def test_redaction_replaces_only_the_key_argument(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mstar-serve", "--config", "a.yaml", "--api-key", "a",
                                     "--api-k=a", "--model-path=/models/a"])
    auth.keep_api_key_from_children("a")
    assert sys.argv == ["mstar-serve", "--config", "a.yaml", "--api-key", "<redacted>",
                        "--api-k=<redacted>", "--model-path=/models/a"]


def test_mstar_serve_forwards_the_key_and_never_prints_it(monkeypatch, capsys):
    from mstar.cli import main as cli

    seen = {}
    monkeypatch.setattr(entrypoint, "main", lambda argv: seen.setdefault("argv", argv))
    cli.main(["serve", "kokoro", "--api-key", KEY])
    argv = seen["argv"]
    assert argv[argv.index("--api-key") + 1] == KEY
    printed = capsys.readouterr().err
    assert KEY not in printed and 'api_key="<your key>"' in printed
