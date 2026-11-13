"""LoRA's runtime updates on both frontends: ``/v1/load_lora_adapter`` and
``/v1/unload_lora_adapter`` answer alike, and mstar-server's own selection
follows the API server's served set, which the bridge pushes after every change,
a conductor's answer that came after its request timed out included.

Both frontends call the same ``LoraAdmin``, over a scripted conductor that
answers from another thread, as the message thread does; the adapter file check
is scripted too (``bad`` in a path refuses it)."""
import json
import threading
import time
import types
import urllib.error
import urllib.request

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _model_stack, _StubAPIServer

from mstar.api_server import entrypoint, lora_admin
from mstar.api_server.lora_admin import LoraAdmin
from mstar.api_server.request_types import LoraAdapterResult
from mstar.api_server.rust_frontend import SERVED_MODELS_ENV, served_models_json
from mstar.api_server.served_models import ServedModels

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

MODEL = "qwen3_5_0.8b"


class _Conductor:
    """Answers each load or unload after ``delay`` s, from its own thread; a
    name in ``refuse`` is refused (as a worker that could not read it)."""

    def __init__(self):
        self.delay, self.refuse, self.admin = 0.0, set(), None

    def send(self, message):
        body = message.body

        def answer():
            time.sleep(self.delay)
            error = f"cannot read adapter {body.lora_name!r}" if body.lora_name in self.refuse else None
            self.admin.on_result(LoraAdapterResult(op_id=body.op_id, error_message=error,
                                                   status=400 if error else 200))
        threading.Thread(target=answer, daemon=True).start()


class _Server(_StubAPIServer):
    """The API server both frontends reach: its served set, and the admin over it."""

    def __init__(self, runtime_updates=True, timeout_s=10.0):
        super().__init__()
        self.model_name, self.model, self.upload_dir = MODEL, None, "/tmp"
        self.served_models = ServedModels(base=[MODEL, "Qwen/Qwen3.5-0.8B"], digests={"sql": b"\x01" * 32}, lora=True)
        self.conductor = _Conductor()
        spec = types.SimpleNamespace(runtime_updates=True) if runtime_updates else None
        self.lora_admin = LoraAdmin(spec, self.served_models, self.conductor.send, timeout_s=timeout_s)
        self.conductor.admin = self.lora_admin

    def reset(self):
        """Back to the launch set, the served list mstar-server started with."""
        for name in self.served_models.adapters:
            self.served_models.remove(name)
        self.served_models.add("sql", b"\x01" * 32)
        self.conductor.delay, self.conductor.refuse = 0.0, set()


@pytest.fixture(autouse=True)
def _scripted_files(monkeypatch):
    def check(spec, name, path):
        if "bad" in path:
            raise ValueError(f"{path}: not a LoRA adapter")
        return name.encode().ljust(32, b"\0")
    monkeypatch.setattr(lora_admin, "check_adapter_file", check)
    monkeypatch.setattr(lora_admin, "resolve_adapter_path", lambda path: path)


def _launch(server):
    return _model_stack(MODEL, server, env={SERVED_MODELS_ENV: served_models_json(server.served_models)})


def _answer(status, ctype, text):
    """A JSON body parsed: key order is not part of the answer (serde_json sorts keys here)."""
    return status, ctype, json.loads(text) if ctype == "application/json" else text


def _http(port, method, path, body=None, content_type="application/json"):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": content_type} if content_type else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return _answer(r.status, r.headers.get("content-type"), r.read().decode())
    except urllib.error.HTTPError as e:
        return _answer(e.code, e.headers.get("content-type"), e.read().decode())


def _python_http(client, method, path, body=None, content_type="application/json"):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    r = client.request(method, path, content=data, headers={"Content-Type": content_type} if content_type else {})
    return _answer(r.status_code, r.headers.get("content-type"), r.text)


def _models(answer):
    status, _, body = answer
    return status, [(m["id"], m.get("parent")) for m in body["data"]]


def _both(monkeypatch, server, port, calls):
    """Each frontend's answers to ``calls`` (method, path, body[, content type]),
    each from the launch set, with ``/v1/models`` after every call."""
    monkeypatch.setattr(entrypoint, "api_server", server)
    client = TestClient(entrypoint.app)
    out = []
    for send in (lambda *c: _python_http(client, *c), lambda *c: _http(port, *c)):
        server.reset()
        answers = []
        for call in calls:
            answers.append(send(*call))
            answers.append(_models(send("GET", "/v1/models")))
        out.append(answers)
    return out


@pytest.fixture(scope="module")
def stack():
    server = _Server()
    with _launch(server) as (port, _):
        yield server, port


LOAD, UNLOAD = "/v1/load_lora_adapter", "/v1/unload_lora_adapter"


def test_loads_and_unloads_answer_alike_and_change_what_is_served(stack, monkeypatch):
    server, port = stack
    calls = [
        ("POST", LOAD, {"lora_name": "chat", "lora_path": "/ckpt/chat"}),
        ("POST", LOAD, {"lora_name": "chat", "lora_path": "/ckpt/chat"}),   # already loaded
        ("POST", LOAD, {"lora_name": MODEL, "lora_path": "/ckpt/x"}),       # the base model's name
        ("POST", LOAD, {"lora_name": "x", "lora_path": "/srv/bad/x.safetensors"}),  # a refused file
        ("POST", LOAD, {"lora_name": "", "lora_path": "/ckpt/x"}),
        ("POST", UNLOAD, {"lora_name": "nope"}),                              # not served: 404
        ("POST", UNLOAD, {"lora_name": "sql"}),
        ("POST", UNLOAD, {"lora_name": "chat", "extra": 1}),
    ]
    py, rust = _both(monkeypatch, server, port, calls)
    assert rust == py
    assert py[0][0] == 200 and py[-1][1] == [(MODEL, None)]


@pytest.mark.parametrize("path,body,content_type", [
    (LOAD, {"lora_name": "a"}, "application/json"),
    (LOAD, {"lora_name": 5, "lora_path": None}, "application/json"),
    (LOAD, [1], "application/json"),
    (LOAD, "nope", "application/json"),
    (LOAD, b"", "application/json"),
    (LOAD, {"lora_name": "a", "lora_path": "/p"}, None),           # no JSON content type: read as bytes
    (LOAD, b"{'it': 's'}", "text/plain"),
    (UNLOAD, {}, "application/json"),
    (UNLOAD, {"lora_name": ["a"]}, "application/vnd.api+json"),
], ids=["missing", "types", "not-an-object", "a-string", "empty", "no-type", "text", "unload-missing", "plus-json"])
def test_a_malformed_body_is_refused_alike(stack, monkeypatch, path, body, content_type):
    server, port = stack
    py, rust = _both(monkeypatch, server, port, [("POST", path, body, content_type)])
    assert rust == py
    assert py[0][0] == 400


@pytest.mark.parametrize("body", [b"not json", b'{"lora_name": "a",}', b"[1, 2"])
def test_a_body_that_is_not_json_is_refused_alike_but_for_the_decoders_words(stack, monkeypatch, body):
    """Python's text quotes its JSON decoder's error and where it stopped, which
    serde words and places otherwise; the rest of the answer is the same."""
    server, port = stack
    py, rust = _both(monkeypatch, server, port, [("POST", LOAD, body)])
    (status, ctype, answer), (rstatus, rctype, ranswer) = py[0], rust[0]
    assert (rstatus, rctype) == (status, ctype) == (400, "application/json")
    for message in (answer["error"]["message"], ranswer["error"]["message"]):
        assert message.startswith("[{'type': 'json_invalid', 'loc': ('body', ")
        assert "'msg': 'JSON decode error', 'input': {}, 'ctx': {'error': " in message
    assert {**ranswer["error"], "message": None} == {**answer["error"], "message": None}
    assert py[1] == rust[1]   # and nothing changed


def test_a_load_selects_its_adapter_on_mstar_server_with_no_round_trip(stack, monkeypatch):
    server, port = stack
    server.reset()
    chat = {"model": "fresh", "messages": [{"role": "user", "content": "hi"}]}
    assert _http(port, "POST", "/v1/chat/completions", chat)[0] == 400   # not served yet
    assert _http(port, "POST", LOAD, {"lora_name": "fresh", "lora_path": "/ckpt/fresh"})[0] == 200
    server.submitted.clear()
    assert _http(port, "POST", "/v1/chat/completions", chat)[0] == 200
    assert server.submitted[-1]["mk"]["lora_adapter"] == "fresh"


def test_a_late_answer_reaches_mstar_server():
    """A load whose workers answer after its request timed out is a 504, and
    then served on both: the conductor's answer is applied when it comes."""
    server = _Server(timeout_s=0.3)
    with _launch(server) as (port, _):
        server.conductor.delay = 1.0
        status, _, body = _http(port, "POST", LOAD, {"lora_name": "late", "lora_path": "/ckpt/late"})
        assert status == 504 and "may still be loaded later" in body["error"]["message"]
        deadline = time.monotonic() + 10
        while "late" not in [m for m, _ in _models(_http(port, "GET", "/v1/models"))[1]]:
            assert time.monotonic() < deadline, "mstar-server never heard of the late load"
            time.sleep(0.05)
        # and an unload the conductor refuses after its request timed out comes back
        server.conductor.refuse = {"late"}
        assert _http(port, "POST", UNLOAD, {"lora_name": "late"})[0] == 504
        assert "late" not in [m for m, _ in _models(_http(port, "GET", "/v1/models"))[1]]
        deadline = time.monotonic() + 10
        while "late" not in [m for m, _ in _models(_http(port, "GET", "/v1/models"))[1]]:
            assert time.monotonic() < deadline, "mstar-server never heard the unload was refused"
            time.sleep(0.05)


def test_without_runtime_updates_the_routes_are_404_alike(monkeypatch):
    server = _Server(runtime_updates=False)
    with _launch(server) as (port, _):
        py, rust = _both(monkeypatch, server, port, [
            ("POST", LOAD, {"lora_name": "a", "lora_path": "/p"}), ("POST", UNLOAD, {"lora_name": "sql"}),
        ])
    assert rust == py
    assert py[0][0] == 404
