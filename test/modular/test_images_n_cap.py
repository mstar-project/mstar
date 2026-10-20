"""``/v1/images/generations`` refuses ``n`` outside ``1..MSTAR_MAX_IMAGES_PER_REQUEST``
with a 400 before submitting anything (the real router, a stubbed APIServer).
The Rust frontend's twin is in test/rust/test_rust_frontend.py."""

import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _Chunk:
    modality = "image"
    data = b"\x89PNGfake"


class _StubAPI:
    model_name = "cosmos3"

    def __init__(self):
        self.upload_dir = Path(tempfile.mkdtemp())
        self.submitted = []

    def submit_request(self, **kw):
        self.submitted.append(kw)
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return [_Chunk()]


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _StubAPI()
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    monkeypatch.delenv("MSTAR_MAX_IMAGES_PER_REQUEST", raising=False)

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


def _images(client, **fields):
    return client.post("/v1/images/generations", json={"model": "cosmos3", "prompt": "x", **fields})


@pytest.mark.parametrize("n", [0, -1, 11])
def test_n_outside_one_to_the_cap_is_400_before_submit(client_and_stub, n):
    client, stub = client_and_stub
    r = _images(client, n=n)
    assert r.status_code == 400
    assert r.json() == {"error": {
        "message": f"n must be between 1 and 10, got {n}", "type": "invalid_request_error", "code": 400}}
    assert not stub.submitted


def test_n_at_the_cap_returns_that_many_images(client_and_stub):
    client, stub = client_and_stub
    r = _images(client, n=10)
    assert r.status_code == 200
    assert len(r.json()["data"]) == 10 and len(stub.submitted) == 10


@pytest.mark.parametrize("fields", [{}, {"n": None}], ids=["omitted", "null"])
def test_n_omitted_is_one_image(client_and_stub, fields):
    client, stub = client_and_stub
    assert len(_images(client, **fields).json()["data"]) == 1
    assert len(stub.submitted) == 1


def test_the_cap_follows_the_env(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setenv("MSTAR_MAX_IMAGES_PER_REQUEST", "2")
    r = _images(client, n=3)
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "n must be between 1 and 2, got 3"
    assert not stub.submitted
    assert len(_images(client, n=2).json()["data"]) == 2


@pytest.mark.parametrize("raw, cap", [
    ("2", 2), (" 2 ", 2), ("2\n", 2), ("+3", 3), ("0", 1), ("-1", 1),
    ("", 10), ("abc", 10), ("2_0", 10), ("1.5", 10), ("\uff12", 10),
])
def test_the_cap_parses_as_the_rust_frontend_does(monkeypatch, raw, cap):
    # rust/server/src/main.rs::max_images_parses_as_python_does checks the same table
    from mstar.api_server.openai.serving_images import max_images_per_request

    monkeypatch.setenv("MSTAR_MAX_IMAGES_PER_REQUEST", raw)
    assert max_images_per_request() == cap


def test_a_non_positive_cap_still_serves_one_image(client_and_stub, monkeypatch):
    client, _ = client_and_stub
    monkeypatch.setenv("MSTAR_MAX_IMAGES_PER_REQUEST", "0")
    for body in ({"prompt": "x"}, {"prompt": "x", "n": None}, {"prompt": "x", "n": 1}):
        assert client.post("/v1/images/generations", json=body).status_code == 200, body
    assert client.post("/v1/images/generations", json={"prompt": "x", "n": 2}).status_code == 400
