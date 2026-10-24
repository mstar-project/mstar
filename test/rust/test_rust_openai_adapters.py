"""``mstar-server`` serves the OpenAI routes from the Python adapter registry:
the launcher names the model's adapter class (``MSTAR_OPENAI_ADAPTER``) and
the Rust frontend translates requests with its port of that class.

Checks, against ``ADAPTER_REGISTRY`` itself: every class and surface the Rust
frontend serves or lists as not ported yet; for each chat class, that a chat
reaches the backend as the Python adapter builds it; and, where fastapi is
installed, that both frontends answer a chat with the same status and body.
Skipped unless the ``mstar_rust`` extension is installed and the server binary
is built."""
import base64
import contextlib
import json
import os
import subprocess
import sys
import types
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")

from test_rust_frontend import (
    BINARY,
    _Chunk,
    _free_port,
    _model_stack,
    _StubAPIServer,
    _wait_healthy,
)

from mstar.api_server.openai.adapters import ADAPTER_REGISTRY, OpenAIAdapter
from mstar.api_server.rust_frontend import OPENAI_ADAPTER_ENV, openai_adapter_class

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

# The routes mstar-server has, by the adapter flag that gates each.
ROUTES = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    "speech": ("/v1/audio/speech", {"input": "hi"}),
    "images": ("/v1/images/generations", {"prompt": "a cat"}),
    "videos": ("/v1/videos/generations", {"prompt": "a cat"}),
}
# Surfaces the Python adapter serves and mstar-server does not yet (#742).
# /v1/audio/transcriptions and /v1/realtime have no route there at all.
NOT_PORTED = {
    ("ChatterboxAdapter", "speech"),
    ("FishS2ProAdapter", "speech"),
    ("KokoroAdapter", "speech"),
    ("OmniVoiceAdapter", "speech"),
    ("Qwen3TTSAdapter", "speech"),
    ("VoxCPM2Adapter", "speech"),
    ("DiffusionImageAdapter", "images"),
    ("LTX25Adapter", "videos"),
    ("Wan22Adapter", "videos"),
}


def _classes() -> dict[str, object]:
    """One adapter of each class in the registry, by class name."""
    return {type(a).__name__: a for a in ADAPTER_REGISTRY.values()}


def _model_of(cls: str) -> str:
    return next(m for m, a in ADAPTER_REGISTRY.items() if type(a).__name__ == cls)


@contextlib.contextmanager
def _mock_server(adapter_class: str):
    """mstar-server with no backend (its mock generator), for ``adapter_class``."""
    port = _free_port()
    env = dict(os.environ, **{OPENAI_ADAPTER_ENV: adapter_class})
    proc = subprocess.Popen([BINARY, "some_model", str(port)], env=env,
                            stderr=subprocess.PIPE, text=True)
    try:
        assert _wait_healthy(port)
        yield port, proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _post(port, path, body, timeout=15):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_launcher_names_the_registry_s_adapter_class():
    assert openai_adapter_class("qwen3_5_0.8b") == "Qwen3_5Adapter"
    assert openai_adapter_class("qwen3_8_27b") == "Qwen3_5Adapter"
    assert openai_adapter_class("cosmos3_edge_droid") == "Cosmos3EdgeAdapter"
    assert openai_adapter_class("pi05") == ""


@pytest.mark.parametrize("cls", sorted(_classes()))
def test_each_registry_class_serves_its_python_surfaces_or_lists_them_as_not_ported(cls):
    """Every route answers non-404 exactly when the Python adapter serves it
    and the surface is ported. A new adapter class, or a new surface on one,
    fails here until it is ported or listed in NOT_PORTED."""
    _check_routes(cls)


def _check_routes(cls: str) -> None:
    adapter = _classes()[cls]
    with _mock_server(cls) as (port, _proc):
        for surface, (path, body) in ROUTES.items():
            python = getattr(adapter, f"supports_{surface}")
            assert not ((cls, surface) in NOT_PORTED and not python), \
                f"NOT_PORTED lists {cls} {surface}, which Python does not serve"
            status, data = _post(port, path, body)
            want = python and (cls, surface) not in NOT_PORTED
            assert (status != 404) == want, (cls, surface, status, data)


def test_a_model_without_an_adapter_gets_the_python_frontend_s_404():
    with _mock_server("") as (port, _proc):
        status, data = _post(port, *ROUTES["chat"])
    assert (status, json.loads(data)) == (404, {"error": {
        "message": "Model 'some_model' has no OpenAI-compatible adapter; use POST /generate",
        "type": "model_not_found", "code": 404}})


def test_a_class_with_no_port_is_a_404_naming_it_and_a_startup_warning():
    with _mock_server("WhisperAdapter") as (port, proc):
        status, data = _post(port, *ROUTES["chat"])
        proc.terminate()
        _, stderr = proc.communicate(timeout=10)
    assert status == 404
    assert "WhisperAdapter" in json.loads(data)["error"]["message"]
    assert "some_model's OpenAI adapter WhisperAdapter has no port in this frontend" in stderr


# ── a chat reaches the backend as the Python adapter builds it ──────────────

# A 1x1 PNG, so the media path is the real one.
PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360f8cfc0f01f0005000201e7a5b6"
    "2a0000000049454e44ae426082")).decode()

CHAT = {
    "model": "ignored",
    "messages": [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [
            {"type": "text", "text": "이 그림은 무엇인가요?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
            {"type": "text", "text": "Опишите."},
        ]},
        {"role": "assistant", "content": "海です。"},
        {"role": "user", "content": "And now?"},
    ],
    "temperature": 0.5, "top_p": 0.8, "max_completion_tokens": 16, "max_tokens": 64, "seed": 3,
    "modalities": ["text", "audio"], "audio": {"voice": "Ethan"},
    "repetition_penalty": 1.05, "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
    "tools": [], "tool_choice": "auto",
}

# ChatCompletionRequest's declared fields; the rest is model_extra.
DECLARED = {"messages": None, "model": None, "temperature": None, "top_p": None,
            "max_tokens": None, "max_completion_tokens": None, "n": 1, "seed": None,
            "stop": None, "stream": False, "modalities": None, "audio": None}


def _python_request(body: dict):
    """The request the Python route hands its adapter. The CI Rust job runs
    without pydantic, so there it is the same fields on a namespace."""
    try:
        from mstar.api_server.openai.protocol import ChatCompletionRequest
    except ImportError:
        extra = {k: v for k, v in body.items() if k not in DECLARED}
        return types.SimpleNamespace(**{**DECLARED, **body}, model_extra=extra)
    assert set(ChatCompletionRequest.model_fields) == set(DECLARED)
    return ChatCompletionRequest(**body)


def _files(paths: dict[str, list[str]] | None) -> dict[str, list[bytes]]:
    return {m: [Path(p).read_bytes() for p in ps] for m, ps in (paths or {}).items()}


def _chat_classes() -> list[str]:
    """The classes whose chat mstar-server ports: those NOT_PORTED lists for
    chat answer it with the 404 that names them, so they reach no backend."""
    return sorted(c for c, a in _classes().items()
                  if a.supports_chat and (c, "chat") not in NOT_PORTED)


CHAT_CLASSES = _chat_classes()


def test_the_chat_classes_are_the_ones_this_pr_ports():
    assert CHAT_CLASSES == ["BagelAdapter", "Cosmos3EdgeAdapter", "Qwen3OmniAdapter", "Qwen3_5Adapter"]


class _UnportedChatAdapter(OpenAIAdapter):
    supports_chat = True


def test_a_chat_class_listed_as_not_ported_passes_the_route_and_chat_checks(monkeypatch):
    """A model PR can register a chat adapter before mstar-server ports it, as
    long as NOT_PORTED lists it."""
    cls = _UnportedChatAdapter.__name__
    monkeypatch.setitem(ADAPTER_REGISTRY, "unported_chat_model", _UnportedChatAdapter())
    monkeypatch.setattr(sys.modules[__name__], "NOT_PORTED", NOT_PORTED | {(cls, "chat")})
    assert cls in _classes()
    assert cls not in _chat_classes()
    _check_routes(cls)


@pytest.mark.parametrize("fields", [{}, {"max_completion_tokens": 0}], ids=["fields", "zero-max"])
@pytest.mark.parametrize("cls", CHAT_CLASSES)
def test_a_chat_reaches_the_backend_as_the_python_adapter_builds_it(cls, fields, tmp_path):
    """``max_completion_tokens: 0`` falls through to ``max_tokens`` in Python."""
    model = _model_of(cls)
    stub = _StubAPIServer()
    body = {**CHAT, **fields}
    with _model_stack(model, stub) as (port, _upload):
        status, data = _post(port, ROUTES["chat"][0], body, timeout=30)
        assert status == 200, data
        (sub,) = stub.submitted
        rust_files = _files(sub["files"])
    want = ADAPTER_REGISTRY[model].chat_to_request(_python_request(body), tmp_path)

    assert sub["text"] == want.text
    assert sub["in"] == want.input_modalities
    assert sub["out"] == want.output_modalities
    assert sub["mk"] == want.model_kwargs
    assert [asdict(p) for p in sub["parts"]] == [asdict(p) for p in want.prompt_parts]
    assert rust_files == _files(want.file_paths) == {"image": [base64.b64decode(PNG)]}


# ── both frontends answer a chat the same way ────────────────────────────────

class _PythonServer:
    """What the Python routes call on APIServer, replying as the stub does."""

    def __init__(self, model_name, upload_dir):
        self.model_name, self.upload_dir, self.model = model_name, upload_dir, None

    def submit_request(self, **kwargs):
        return kwargs["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return [_Chunk("text", b"Hello "), _Chunk("text", b"world")]

    async def iter_result_chunk_batches(self, request_id):
        yield [_Chunk("text", b"Hello ")]
        yield [_Chunk("text", b"world")]


def _normalized(status, data: bytes, stream: bool):
    """The body with ids and timestamps blanked; SSE as its list of events."""
    def blank(obj):
        if isinstance(obj, dict):
            return {k: ("*" if k in ("id", "created") else blank(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [blank(v) for v in obj]
        return obj

    if not stream or status != 200:
        return status, blank(json.loads(data))
    events = [line[len("data:"):].strip() for line in data.decode().splitlines()
              if line.startswith("data:")]
    return status, [e if e == "[DONE]" else blank(json.loads(e)) for e in events]


PARITY_CASES = {
    "chat": ({"messages": [{"role": "user", "content": "Привет, 世界"}], "max_tokens": 8}, False),
    "chat-stream": ({"messages": [{"role": "user", "content": "안녕"}], "stream": True}, True),
    "tool-call": ({"messages": [{"role": "user", "content": "hi"}], "tool_choice": "required"}, False),
    "bad-role": ({"messages": [{"role": "tool", "content": "42"}]}, False),
}


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
@pytest.mark.parametrize("model", ["qwen3_5_0.8b", "cosmos3_edge", "pi05"])
def test_both_frontends_answer_a_chat_with_the_same_status_and_body(case, model, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    body, stream = PARITY_CASES[case]
    monkeypatch.setattr(entrypoint, "api_server", _PythonServer(model, tmp_path))
    py = TestClient(entrypoint.app).post("/v1/chat/completions", json=body)
    with _model_stack(model, _StubAPIServer()) as (port, _upload):
        rust = _post(port, "/v1/chat/completions", body, timeout=30)
    assert _normalized(*rust, stream) == _normalized(py.status_code, py.content, stream)
