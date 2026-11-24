"""Python-vs-Rust HTTP parity for generation control admin routes."""

import contextlib
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _free_port, _StubAPIServer, _wait_healthy

from mstar.api_server import generation_routes
from mstar.api_server.auth import APIKeyGuard, APIKeyMiddleware
from mstar.api_server.generation_admin import RELOAD_ROOTS_ENV, GenerationAdmin, reload_roots
from mstar.api_server.request_types import GenerationControlResult
from mstar.api_server.rust_frontend import RustFrontendBridge, launch_rust_server
from mstar.utils.ipc_format import ConductorMessageType

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")


class _Conductor:
    def __init__(self):
        self.admin = None
        self.sent = []
        self.pause_conflict = False
        self.weight_version = None

    def reset(self):
        self.sent.clear()
        self.pause_conflict = False
        self.weight_version = None

    def send(self, message):
        self.sent.append(message)
        body = message.body

        def answer():
            if message.message_type == ConductorMessageType.PAUSE_GENERATION:
                result = GenerationControlResult(
                    op_id=body.op_id,
                    status=409 if self.pause_conflict else 200,
                    state="paused",
                    paused=True,
                    weight_version=self.weight_version,
                    error_message="rank still dirty" if self.pause_conflict else None,
                )
            elif message.message_type == ConductorMessageType.UPDATE_WEIGHTS_FROM_DISK:
                self.weight_version = body.weight_version
                result = GenerationControlResult(
                    op_id=body.op_id, status=200, state="paused", paused=True, weight_version=self.weight_version,
                )
            elif message.message_type == ConductorMessageType.CONTINUE_GENERATION:
                result = GenerationControlResult(
                    op_id=body.op_id, status=200, state="running", paused=False, weight_version=self.weight_version,
                )
            elif message.message_type == ConductorMessageType.GET_GENERATION_STATE:
                result = GenerationControlResult(
                    op_id=body.op_id, status=200, state="paused", paused=True, weight_version=self.weight_version,
                )
            elif message.message_type == ConductorMessageType.CHECK_WEIGHTS:
                resident = {"worker_0": "a" * 64}
                mismatch = body.mode == "compare" and body.checksums != resident
                result = GenerationControlResult(
                    op_id=body.op_id,
                    status=409 if mismatch else 200,
                    state="paused",
                    paused=True,
                    weight_version=self.weight_version,
                    error_message="weight checksum mismatch" if mismatch else None,
                    rank_checksums=resident,
                )
            else:
                result = GenerationControlResult(op_id=body.op_id, status=400, error_message="unexpected op")
            self.admin.on_result(result)

        threading.Thread(target=answer, daemon=True).start()


class _Server(_StubAPIServer):
    def __init__(self, *, key_enabled: bool, roots: tuple):
        super().__init__()
        self.conductor = _Conductor()
        self.generation_admin = GenerationAdmin(self.conductor.send, key_enabled=key_enabled, roots=roots)
        self.conductor.admin = self.generation_admin


@contextlib.contextmanager
def _launch(server: _Server, *, api_key: str | None, roots_env: str | None):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_ga_bridge_")
    upload_dir = tempfile.mkdtemp(prefix="mstar_ga_up_")
    old_roots = os.environ.get(RELOAD_ROOTS_ENV)
    if roots_env is None:
        os.environ.pop(RELOAD_ROOTS_ENV, None)
    else:
        os.environ[RELOAD_ROOTS_ENV] = roots_env
    try:
        proc = launch_rust_server(BINARY, "qwen3_omni", port, bridge_dir, upload_dir, api_key=api_key)
    finally:
        if old_roots is None:
            os.environ.pop(RELOAD_ROOTS_ENV, None)
        else:
            os.environ[RELOAD_ROOTS_ENV] = old_roots
    bridge = RustFrontendBridge(server, bridge_dir)
    thread = threading.Thread(target=bridge.run, daemon=True)
    thread.start()
    try:
        if not _wait_healthy(port):
            raise RuntimeError("mstar-server never became healthy")
        yield port
    finally:
        bridge.stop()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


def _python_app(monkeypatch, server: _Server, *, api_key: str | None, enable_routes: bool) -> FastAPI:
    monkeypatch.setattr(generation_routes, "_admin", lambda: server.generation_admin)
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, guard=APIKeyGuard(api_key))
    if enable_routes:
        app.include_router(generation_routes.router)
    return app


def _answer(status, ctype, text):
    body = json.loads(text) if ctype == "application/json" and text else text
    if isinstance(body, dict) and "op_id" in body:
        assert isinstance(body["op_id"], str) and body["op_id"].startswith("generation-")
        body = {**body, "op_id": "<op_id>"}
    return status, ctype, body


def _rust_http(port, method, path, body=None, headers=None):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    req_headers = dict(headers or {})
    if data is not None and "Content-Type" not in req_headers:
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return _answer(r.status, r.headers.get("content-type"), r.read().decode())
    except urllib.error.HTTPError as e:
        return _answer(e.code, e.headers.get("content-type"), e.read().decode())


def _py_http(client, method, path, body=None, headers=None):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    req_headers = dict(headers or {})
    if data is not None and "Content-Type" not in req_headers:
        req_headers["Content-Type"] = "application/json"
    r = client.request(method, path, content=data, headers=req_headers)
    return _answer(r.status_code, r.headers.get("content-type"), r.text)


def test_generation_admin_routes_parity(tmp_path, monkeypatch):
    root = tmp_path / "weights"
    ckpt = root / "step_10"
    outside = tmp_path / "outside"
    ckpt.mkdir(parents=True)
    outside.mkdir()

    api_key = "secret-key"
    auth = {"Authorization": f"Bearer {api_key}"}
    server = _Server(key_enabled=True, roots=reload_roots(str(root)))
    app = _python_app(monkeypatch, server, api_key=api_key, enable_routes=True)

    calls = [
        ("POST", "/pause_generation", {"mode": "wait"}, {}),
        ("GET", "/is_paused", None, {}),
        ("POST", "/weight_checker", {"mode": "checksum"}, {}),
        ("POST", "/pause_generation", b"", auth),
        ("POST", "/pause_generation", {}, auth),
        ("POST", "/pause_generation", {"mode": ""}, auth),
        ("POST", "/pause_generation", {"mode": None}, auth),
        ("POST", "/pause_generation", {"mode": 123}, auth),
        ("POST", "/pause_generation", {"mode": "invalid"}, auth),
        ("POST", "/pause_generation", {"mode": "abort"}, auth),
        ("POST", "/pause_generation", {"mode": "wait"}, auth),
        ("POST", "/update_weights_from_disk", {"model_path": str(outside), "weight_version": "v2"}, auth),
        ("POST", "/update_weights_from_disk", {"model_path": str(ckpt), "weight_version": ""}, auth),
        ("POST", "/update_weights_from_disk", {"model_path": str(ckpt), "weight_version": "v2"}, auth),
        ("GET", "/is_paused", None, auth),
        ("POST", "/weight_checker", {"mode": "bad"}, auth),
        ("POST", "/weight_checker", {"mode": "compare", "checksums": {"worker_0": 1}}, auth),
        ("POST", "/weight_checker", {"mode": "checksum"}, auth),
        ("POST", "/weight_checker", {"mode": "compare", "checksums": {"worker_0": "a" * 64}}, auth),
        ("POST", "/weight_checker", {"mode": "compare", "checksums": {"worker_0": "b" * 64}}, auth),
        ("POST", "/continue_generation", {}, auth),
    ]

    with _launch(server, api_key=api_key, roots_env=str(root)) as port, TestClient(app) as client:
        server.conductor.reset()
        py_answers = [_py_http(client, *call) for call in calls]
        py_sent = [(m.message_type, getattr(m.body, "mode", None)) for m in server.conductor.sent]

        server.conductor.reset()
        rust_answers = [_rust_http(port, *call) for call in calls]
        rust_sent = [(m.message_type, getattr(m.body, "mode", None)) for m in server.conductor.sent]

        server.conductor.pause_conflict = True
        assert _rust_http(port, "POST", "/pause_generation", {"mode": "abort"}, auth) == _py_http(
            client, "POST", "/pause_generation", {"mode": "abort"}, auth
        )

    assert rust_answers == py_answers
    assert rust_sent == py_sent
    assert py_answers[0][0] == 401 and py_answers[1][0] == 401 and py_answers[2][0] == 401
    for idx in range(3, 9):
        assert py_answers[idx] == (400, "application/json", {"detail": "mode must be 'abort' or 'wait'"})
    assert py_answers[9][0] == 200 and py_answers[10][0] == 200
    assert py_answers[11][0] == 400 and py_answers[12][0] == 400
    assert py_answers[13][0] == 200 and py_answers[13][2]["weight_version"] == "v2"
    assert py_answers[14][2]["weight_version"] == "v2"
    assert py_answers[15][0] == 400 and py_answers[16][0] == 400
    assert py_answers[17][0] == 200 and py_answers[17][2]["rank_checksums"] == {"worker_0": "a" * 64}
    assert py_answers[18][0] == 200
    assert py_answers[19][0] == 409 and py_answers[19][2]["error_message"] == "weight checksum mismatch"
    assert py_answers[20][0] == 200 and py_answers[20][2]["paused"] is False


@pytest.mark.parametrize(
    ("api_key", "roots_env"),
    [("secret-key", None), (None, "roots")],
    ids=["missing-roots", "missing-key"],
)
def test_generation_admin_routes_disabled_404_parity(tmp_path, monkeypatch, api_key, roots_env):
    root = tmp_path / "weights"
    root.mkdir()
    roots_str = str(root) if roots_env else None
    server = _Server(key_enabled=bool(api_key), roots=reload_roots(roots_str or ""))
    app = _python_app(monkeypatch, server, api_key=api_key, enable_routes=False)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    with _launch(server, api_key=api_key, roots_env=roots_str) as port, TestClient(app) as client:
        for method, path, body in (
            ("POST", "/pause_generation", {"mode": "wait"}),
            ("POST", "/update_weights_from_disk", {"model_path": str(root), "weight_version": "v2"}),
            ("POST", "/continue_generation", {}),
            ("GET", "/is_paused", None),
            ("POST", "/weight_checker", {"mode": "checksum"}),
        ):
            assert _py_http(client, method, path, body, headers)[0] == 404
            assert _rust_http(port, method, path, body, headers)[0] == 404
    assert server.conductor.sent == []
