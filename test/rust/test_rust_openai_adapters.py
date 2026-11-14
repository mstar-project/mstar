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
    _USAGE,
    BINARY,
    _Chunk,
    _free_port,
    _model_stack,
    _StubAPIServer,
    _wait_healthy,
)

from mstar.api_server.entrypoint import APIServer
from mstar.api_server.openai.adapters import ADAPTER_REGISTRY, OpenAIAdapter
from mstar.api_server.rust_frontend import OPENAI_ADAPTER_ENV, openai_adapter_class
from mstar.api_server.served_models import ServedModels

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

# The routes mstar-server has, by the adapter flag that gates each.
ROUTES = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    "speech": ("/v1/audio/speech", {"input": "hi"}),
    "images": ("/v1/images/generations", {"prompt": "a cat"}),
    "videos": ("/v1/videos/generations", {"prompt": "a cat"}),
    # a JSON body carries no upload: a served route refuses it with a 400, not a 404
    "transcriptions": ("/v1/audio/transcriptions", {"model": "m"}),
}
# Surfaces the Python adapter serves and mstar-server does not yet (#742).
NOT_PORTED: set[tuple[str, str]] = set()


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
    # every registry class is ported; a class from a newer Python side has none
    with _mock_server("FutureAdapter") as (port, proc):
        status, data = _post(port, *ROUTES["chat"])
        proc.terminate()
        _, stderr = proc.communicate(timeout=10)
    assert status == 404
    assert "FutureAdapter" in json.loads(data)["error"]["message"]
    assert "some_model's OpenAI adapter FutureAdapter has no port in this frontend" in stderr


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
    # a template key named like a refused field reaches the model, as the flat off one is dropped first
    "chat_template_kwargs": {"enable_thinking": False, "current_date": "2026-11-08", "logprobs": True},
    "tools": [], "tool_choice": "auto", "logprobs": False,
}

# ChatCompletionRequest's declared fields; the rest is model_extra.
DECLARED = {"messages": None, "model": None, "temperature": None, "top_p": None,
            "max_tokens": None, "max_completion_tokens": None, "n": 1, "seed": None,
            "stop": None, "stream": False, "stream_options": None, "modalities": None, "audio": None}


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


# Text-only chat refuses the multimodal CHAT body; its own table is below.
TEXT_ONLY_CHAT = {"TextChatAdapter"}
CHAT_CLASSES = [c for c in _chat_classes() if c not in TEXT_ONLY_CHAT]


def test_the_chat_classes_are_the_ones_this_pr_ports():
    assert _chat_classes() == [
        "BagelAdapter", "Cosmos3EdgeAdapter", "Qwen3OmniAdapter", "Qwen3_5Adapter", "TextChatAdapter",
    ]


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
    """``max_completion_tokens: 0`` is kept, not replaced by ``max_tokens``, as in Python."""
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


def _wrapped(text: str, width: int, sep: str) -> str:
    return sep.join(text[i:i + width] for i in range(0, len(text), width))


# PNG as clients send it, by whether the Python adapter takes it. Its base64 is
# padded and holds "+" and "/".
IMAGE_URLS = {
    "padded": (f"data:image/png;base64,{PNG}", True),
    "unpadded": (f"data:image/png;base64,{PNG.rstrip('=')}", True),
    "wrapped": ("data:image/png;base64," + _wrapped(PNG, 76, "\n") + "\n", True),
    "wrapped-unpadded": ("data:image/png;base64," + _wrapped(PNG.rstrip("="), 64, "\r\n"), True),
    "unicode-spaces": (f"data:image/png;base64,{PNG[:8]}\u00a0{PNG[8:]}\u3000", True),
    "unused-bits-set": (f"data:image/png;base64,{PNG[:-2]}J=", True),
    "excess-padding": (f"data:image/png;base64,{PNG}==", False),
    "url-safe": ("data:image/png;base64," + PNG.replace("+", "-").replace("/", "_"), False),
    "cut-mid-quad": (f"data:image/png;base64,{PNG[:-3]}", False),
}


@pytest.mark.parametrize("cls", CHAT_CLASSES)
def test_an_image_chat_takes_the_data_urls_the_python_adapter_takes(cls, tmp_path):
    model = _model_of(cls)
    stub = _StubAPIServer()
    with _model_stack(model, stub) as (port, _upload):
        for name, (url, taken) in IMAGE_URLS.items():
            body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Read the page."},
            ]}]}
            try:
                want = ADAPTER_REGISTRY[model].chat_to_request(_python_request(body), tmp_path)
            except ValueError:
                want = None
            assert (want is not None) == taken, name
            before = len(stub.submitted)
            status, data = _post(port, ROUTES["chat"][0], body, timeout=30)
            if want is None:
                assert status == 400 and len(stub.submitted) == before, (name, status, data)
                continue
            assert status == 200, (name, data)
            sub = stub.submitted[-1]
            assert (sub["text"], sub["in"], sub["mk"]) == (want.text, want.input_modalities, want.model_kwargs), name
            assert [asdict(p) for p in sub["parts"]] == [asdict(p) for p in want.prompt_parts], name
            assert _files(sub["files"]) == _files(want.file_paths) == {"image": [base64.b64decode(PNG)]}, name


# ── both frontends answer a chat the same way ────────────────────────────────

class _PythonServer:
    """What the Python routes call on APIServer, replying as the stub does."""

    open_result_stream = APIServer.open_result_stream
    lora_kwargs = APIServer.lora_kwargs

    def __init__(self, model_name, upload_dir):
        self.model_name, self.upload_dir, self.model = model_name, upload_dir, None
        self.served_models = ServedModels.of(model_name, None)
        self.tail = []

    def submit_request(self, **kwargs):
        if kwargs.get("report_usage"):
            self.tail = [_Chunk("usage", b"", _USAGE)]
        return kwargs["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return [_Chunk("text", b"Hello "), _Chunk("text", b"world"), *self.tail]

    async def iter_result_chunk_batches(self, request_id):
        yield [_Chunk("text", b"Hello ")]
        yield [_Chunk("text", b"world"), *self.tail]


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


def _image_chat(url: str) -> dict:
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}


PARITY_CASES = {
    "chat": ({"messages": [{"role": "user", "content": "Привет, 世界"}], "max_tokens": 8}, False),
    "chat-stream": ({"messages": [{"role": "user", "content": "안녕"}], "stream": True}, True),
    "chat-stream-usage": ({"messages": [{"role": "user", "content": "hi"}], "stream": True,
                           "stream_options": {"include_usage": True}}, True),
    "tool-call": ({"messages": [{"role": "user", "content": "hi"}], "tool_choice": "required"}, False),
    "json-mode": ({"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}},
                  False),
    "bad-role": ({"messages": [{"role": "tool", "content": "42"}]}, False),
    # remote fetch is off unless MSTAR_ALLOW_REMOTE=1, and a server path never reads
    "remote-image": (_image_chat("https://example.com/page.png"), False),
    "server-path-image": (_image_chat("/etc/hosts"), False),
}


@pytest.mark.parametrize("case", sorted(PARITY_CASES))
@pytest.mark.parametrize("model", ["qwen3_5_0.8b", "cosmos3_edge", "pi05"])
def test_both_frontends_answer_a_chat_with_the_same_status_and_body(case, model, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    body, stream = PARITY_CASES[case]
    monkeypatch.setattr(entrypoint, "api_server", _PythonServer(model, tmp_path))
    py = TestClient(entrypoint.app).post("/v1/chat/completions", json=body)
    with _model_stack(model, _StubAPIServer()) as (port, _upload):
        rust = _post(port, "/v1/chat/completions", body, timeout=30)
    assert _normalized(*rust, stream) == _normalized(py.status_code, py.content, stream)


# ── GET /v1/audio/voices ─────────────────────────────────────────────────────

class _VoicedModel:
    def __init__(self, voices=("tara", "leo")):
        self.voices = list(voices)

    def get_voices(self):
        return list(self.voices)

    def get_default_voice(self):
        return "tara"


def _get(port, path, timeout=15):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.mark.parametrize("model,voiced", [("orpheus", True), ("orpheus", False), ("qwen3_5_0.8b", True)],
                         ids=["voices", "no-voice-list", "not-a-speech-model"])
def test_both_frontends_list_the_voices_alike(model, voiced, tmp_path, monkeypatch):
    """Both ask the model (``serving_speech.list_voices``) per request, so a voice
    added while serving shows on both; a model with none, or no speech surface,
    gets the same 404."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mstar.api_server import entrypoint

    server = _PythonServer(model, tmp_path)
    stub = _StubAPIServer()
    server.model = stub.model = _VoicedModel() if voiced else None
    monkeypatch.setattr(entrypoint, "api_server", server)
    py = TestClient(entrypoint.app)
    with _model_stack(model, stub) as (port, _upload):
        for added in ([], ["nova"]):
            if voiced:
                server.model.voices += added
            expected = py.get("/v1/audio/voices")
            status, body = _get(port, "/v1/audio/voices")
            assert (status, json.loads(body)) == (expected.status_code, expected.json())
    if voiced and model == "orpheus":
        assert expected.json()["voices"][-1] == {"id": "nova", "name": "nova"}


# ── text-only chat (TextChatAdapter) ─────────────────────────────────────────

def _text(**fields):
    return {"model": "glm52", "messages": [{"role": "user", "content": "Bonjour, 世界"}], **fields}


TEXT_CHAT_CASES = {
    "accepted": _text(temperature=0.3, top_p=0.9, max_tokens=16, seed=2, top_k=5),
    # one input modality per part, or a model that rebuilds its turns from them keeps only the first
    "multi-turn": {"model": "glm52", "messages": [
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"},
    ]},
    "text-response-format": _text(response_format={"type": "text"}, best_of=1, n=1, logprobs=False),
    "image": {"model": "glm52", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is it"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
    ]}]},
    "no-text": {"model": "glm52", "messages": [{"role": "user", "content": ""}]},
    "zero-max-tokens": _text(max_tokens=0),
    "zero-max-completion-tokens": _text(max_completion_tokens=0),
    "n": _text(n=2),
    "stop": _text(stop=["\n"]),
    "logprobs": _text(logprobs=True),
    "logit-bias": _text(logit_bias={"50256": -100}),
    "presence-penalty": _text(presence_penalty=0.5),
    "tiny-penalty": _text(frequency_penalty=1e-05),
    "stop-token-ids": _text(stop_token_ids=[2, 3]),
    "best-of": _text(best_of=2),
    "json-format": _text(response_format={"type": "json_object"}),
    "tool-call": _text(tool_choice="required"),
}


@pytest.mark.parametrize("case", sorted(TEXT_CHAT_CASES))
def test_a_text_only_chat_is_refused_or_submitted_as_python_does(case, tmp_path):
    """The Python adapter's refusal is the Rust 400's message, word for word
    (values quoted with Python's ``repr``); an accepted chat reaches the backend
    with the same arguments, once ``submit_request`` normalizes them."""
    from mstar.utils.generation_kwargs import normalize_generation_kwargs

    body = TEXT_CHAT_CASES[case]
    try:
        want, refusal = ADAPTER_REGISTRY["glm52"].chat_to_request(_python_request(body), tmp_path), None
    except ValueError as e:
        want, refusal = None, str(e)
    stub = _StubAPIServer()
    with _model_stack("glm52", stub) as (port, _upload):
        status, data = _post(port, ROUTES["chat"][0], body, timeout=30)
    if refusal is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, refusal)
        assert not stub.submitted
        return
    assert status == 200, data
    (sub,) = stub.submitted
    assert (sub["text"], sub["in"], sub["out"]) == (want.text, want.input_modalities, want.output_modalities)
    assert normalize_generation_kwargs(sub["mk"]) == want.model_kwargs
    assert [asdict(p) for p in sub["parts"]] == [asdict(p) for p in want.prompt_parts]
    assert not sub["files"]


# ── response_format and structured_outputs ───────────────────────────────────

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
STRUCTURED_CASES = {
    "json-object": {"response_format": {"type": "json_object"}},
    "json-schema": {"response_format": {"type": "json_schema",
                                        "json_schema": {"name": "n", "strict": True, "schema": SCHEMA}}},
    "text": {"response_format": {"type": "text"}},
    "null": {"response_format": None},
    "unknown-type": {"response_format": {"type": "yaml"}},
    "no-type": {"response_format": {}},
    "numeric-type": {"response_format": {"type": 5}},
    "schema-missing": {"response_format": {"type": "json_schema", "json_schema": {"name": "n"}}},
    "not-an-object": {"response_format": "json_object"},
    "both": {"response_format": {"type": "json_object"}, "structured_outputs": {"regex": "a+"}},
    "structured-outputs": {"structured_outputs": {"choice": ["yes", "no"]}},
}


@pytest.mark.parametrize("case", sorted(STRUCTURED_CASES))
@pytest.mark.parametrize("model", ["qwen3_5_0.8b", "glm52"])
def test_both_frontends_turn_a_response_format_into_the_same_structured_outputs(model, case, tmp_path):
    """A JSON ``response_format`` reaches the backend as the same ``structured_outputs``
    from either frontend, and a bad one is the same 400, word for word."""
    body = {"model": model, "messages": [{"role": "user", "content": "Reply in JSON."}], **STRUCTURED_CASES[case]}
    try:
        want, refusal = ADAPTER_REGISTRY[model].chat_to_request(_python_request(body), tmp_path), None
    except ValueError as e:
        want, refusal = None, str(e)
    stub = _StubAPIServer()
    with _model_stack(model, stub) as (port, _upload):
        status, data = _post(port, ROUTES["chat"][0], body, timeout=30)
    if refusal is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, refusal)
        assert not stub.submitted
        return
    assert status == 200, data
    (sub,) = stub.submitted
    assert sub["mk"].get("structured_outputs") == want.model_kwargs.get("structured_outputs")
    assert "response_format" not in sub["mk"] and "response_format" not in want.model_kwargs


# ── an image request reaches the backend as the Python adapter builds it ────

IMAGE_CASES = {
    "size": {"prompt": "a cat", "size": "512x256"},
    "size-spaced": {"prompt": "a cat", "size": " 512 X 256 "},
    "size-underscored": {"prompt": "a cat", "size": "1_024x512"},
    "size-empty": {"prompt": "a cat", "size": ""},
    "bad-size": {"prompt": "a cat", "size": "big"},
    "three-sides": {"prompt": "a cat", "size": "1x2x3"},
    "seeded": {"prompt": "a cat", "seed": 3, "guidance_scale": 4.0},
}


@pytest.mark.parametrize("case", sorted(IMAGE_CASES))
@pytest.mark.parametrize("cls", sorted(c for c, a in _classes().items() if a.supports_images))
def test_an_image_request_is_submitted_or_refused_as_the_python_adapter_does(cls, case, tmp_path):
    from mstar.api_server.openai.protocol import ImageGenerationRequest

    model = _model_of(cls)
    body = {"model": model, **IMAGE_CASES[case]}
    try:
        want = ADAPTER_REGISTRY[model].image_to_request(ImageGenerationRequest(**body), tmp_path)
        refusal = None
    except ValueError as e:
        want, refusal = None, str(e)
    stub = _StubAPIServer()
    with _model_stack(model, stub) as (port, _upload):
        status, data = _post(port, "/v1/images/generations", body)
    if refusal is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, refusal)
        return
    assert status == 200, data
    sub = stub.submitted[0]
    assert (sub["text"], sub["in"], sub["out"], sub["mk"] or {}) == (
        want.text, want.input_modalities, want.output_modalities, want.model_kwargs)
