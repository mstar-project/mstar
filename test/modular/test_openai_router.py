"""End-to-end test of the OpenAI router with a stubbed APIServer (no GPU/torch).

Mounts the real router on a FastAPI app and drives it via TestClient; the
APIServer + model are stubbed so adapters, serving handlers, SSE, and error
paths are all exercised without the engine.
"""

import base64
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _StubModel:
    def get_output_sample_rate(self, modality="audio"):
        return 24000


class _StubAPI:
    def __init__(self, model_name="bagel"):
        self.model_name = model_name
        self.model = _StubModel()
        self.upload_dir = Path(tempfile.mkdtemp())
        self.last_submit = None
        self._chunks: dict = {}
        self.next_chunks: list = []

    def submit_request(self, **kw):
        self.last_submit = kw
        self._chunks[kw["request_id"]] = list(self.next_chunks)
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks.get(request_id, [])

    async def iter_result_chunks(self, request_id):
        for c in self._chunks.get(request_id, []):
            yield c


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _StubAPI()
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


def _pcm(vals):
    return np.array(vals, dtype="<i2").tobytes()  # int16 PCM, as the models emit


def test_models(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    body = client.get("/v1/models").json()
    assert body["object"] == "list" and body["data"][0]["id"] == "bagel"


def test_chat_text(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b"Hello "), _Chunk("text", b"world")]
    body = client.post(
        "/v1/chat/completions",
        json={"model": "bagel", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    ).json()
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert stub.last_submit["model_kwargs"]["max_output_tokens"] == 16


def test_chat_audio_output(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "qwen3_omni"
    stub.next_chunks = [_Chunk("text", b"hi"), _Chunk("audio", _pcm([0, 16000, -16000]), {"sample_rate": 24000})]
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3_omni",
            "messages": [{"role": "user", "content": "x"}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "Ethan"},
        },
    ).json()
    assert stub.last_submit["output_modalities"] == ["text", "audio"]
    wav = base64.b64decode(body["choices"][0]["message"]["audio"]["data"])
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_audio_speech(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.next_chunks = [_Chunk("audio", _pcm([100, -100]), {"sample_rate": 24000})]
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": "hi", "voice": "tara"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert stub.last_submit["model_kwargs"]["voice"] == "tara"


def test_images(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    png = b"\x89PNGfake"
    stub.next_chunks = [_Chunk("image", png)]
    body = client.post("/v1/images/generations", json={"model": "bagel", "prompt": "a cat"}).json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == png


def test_images_edits(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    edited = b"\x89PNGedited"
    stub.next_chunks = [_Chunk("image", edited)]
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"\x89PNGinput", "image/png")},
        data={"prompt": "make it neon", "cfg_img_scale": "2.0", "cfg_interval": "[0.0, 1.0]"},
    )
    body = resp.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == edited
    # input image + prompt submitted as I2I; passthrough kwargs JSON-parsed
    assert stub.last_submit["output_modalities"] == ["image"]
    assert "image" in stub.last_submit["input_modalities"] and "text" in stub.last_submit["input_modalities"]
    assert stub.last_submit["text"] == "make it neon"
    mk = stub.last_submit["model_kwargs"]
    assert mk.get("cfg_img_scale") == 2.0 and mk.get("cfg_interval") == [0.0, 1.0]
    assert "image" in (stub.last_submit["file_paths"] or {})


def test_chat_stream(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b"strea"), _Chunk("text", b"ming")]
    text = client.post(
        "/v1/chat/completions",
        json={"model": "bagel", "messages": [{"role": "user", "content": "go"}], "stream": True},
    ).text
    assert "[DONE]" in text
    lines = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    assert lines[0]["choices"][0]["delta"].get("role") == "assistant"
    assert "".join(l["choices"][0]["delta"].get("content", "") for l in lines) == "streaming"
    assert lines[-1]["choices"][0]["finish_reason"] == "stop"


def test_unsupported_model_404(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "pi05"
    r = client.post(
        "/v1/chat/completions", json={"model": "pi05", "messages": [{"role": "user", "content": "x"}]}
    )
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


def test_api_resolves_from_main_module(monkeypatch):
    # Simulate `python mstar/api_server/entrypoint.py` (runs as __main__):
    # main() sets api_server on __main__, not on the package module. _api()
    # must still find it — this is the launch_server_*.sh launch path.
    from mstar.api_server.openai import router as router_mod

    ep = types.ModuleType("mstar.api_server.entrypoint")  # present but no api_server
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", ep)
    stub = _StubAPI("bagel")
    monkeypatch.setattr(sys.modules["__main__"], "api_server", stub, raising=False)
    assert router_mod._api() is stub
