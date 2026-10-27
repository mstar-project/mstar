"""A client-pinned ``request_id`` is at most 128 characters from
``[A-Za-z0-9._:-]``: ``/generate`` answers anything else with a 400,
``/generate/ws`` with its in-band error, and ``APIServer.submit_request``
refuses it too, while every id the server mints still fits."""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import REQUEST_ID_ERROR, APIServer, check_request_id
from mstar.api_server.openai._util import rid
from mstar.api_server.request_types import ResultChunk

LONGEST = "a" * 128
BAD_IDS = ["a" * 129, "caf\u00e9", "a b", "a/b", "a#1", "a\n"]


class _Server:
    takes_stream_input = False

    def __init__(self, upload_dir):
        self.upload_dir = upload_dir
        self.submitted = []

    async def wait_for_request_id(self, request_id):
        pass

    def submission_key(self, request_id):
        return request_id

    def abort_submission(self, key):
        pass

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs["request_id"])
        return kwargs["request_id"]

    async def _chunks(self, request_id):
        yield ResultChunk(request_id=request_id, modality="text", data=b"ok")

    def iter_result_chunks(self, request_id):
        return self._chunks(request_id)

    def async_stream_results(self, request_id, binary=False):
        async def lines():
            async for chunk in self._chunks(request_id):
                yield json.dumps({"modality": chunk.modality}) + "\n"
        return lines()

    async def collect_results(self, request_id, raw_request=None):
        return [chunk async for chunk in self._chunks(request_id)]


@pytest.fixture
def server(monkeypatch, tmp_path):
    fake = _Server(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", fake)
    return fake


@pytest.mark.parametrize("request_id", [
    "", "warmup-0", "obs.1:a_b-C", LONGEST, str(uuid.uuid4()), rid("img"),
    f"{rid('speech')}-3", f"tensor-{uuid.uuid4()}", rid(f"rt-{uuid.uuid4().hex[:16]}"),
], ids=[
    "empty", "warmup", "punctuation", "longest", "uuid", "minted", "minted-suffix",
    "tensor-uuid", "minted-realtime",
])
def test_check_accepts_client_and_minted_ids(request_id):
    check_request_id(request_id)


@pytest.mark.parametrize("request_id", BAD_IDS + [7])
def test_check_refuses(request_id):
    with pytest.raises(ValueError, match=r"^request_id must be at most 128 characters"):
        check_request_id(request_id)


def test_submit_request_refuses_directly():
    with pytest.raises(ValueError, match=r"^request_id must be at most 128 characters"):
        APIServer.__new__(APIServer).submit_request(
            text="hi", input_modalities=["text"], output_modalities=["text"], request_id="a b",
        )


def test_generate_refuses_with_400_before_saving_uploads(server, tmp_path):
    response = TestClient(entrypoint.app).post(
        "/generate", data={"request_id": "a b", "streaming": "true"}, files={"files": ("obs.png", b"PNG")},
    )
    assert (response.status_code, response.json()) == (400, {"detail": REQUEST_ID_ERROR})
    assert server.submitted == []
    assert list(tmp_path.iterdir()) == []


def test_error_text_matches_the_rust_frontend_test():
    assert REQUEST_ID_ERROR == "request_id must be at most 128 characters from [A-Za-z0-9._:-]"


def test_a_client_sent_empty_request_id_is_none_on_both_routes(server):
    """A client's empty id means none, as the Rust /generate reads it too.
    ``submit_request`` itself still pins ``""`` (test_request_id_reuse.py)."""
    TestClient(entrypoint.app).post("/generate", data={"text": "hi", "streaming": "false", "request_id": ""})
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "hi", "request_id": ""}))
        ws.receive_text()
    assert server.submitted == [None, None]


def test_generate_ws_refuses_in_band_and_serves_the_longest_id(server):
    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        for request_id in BAD_IDS:
            ws.send_text(json.dumps({"text": "hi", "request_id": request_id}))
            assert json.loads(ws.receive_text()) == {"request_id": request_id, "error": REQUEST_ID_ERROR}
        ws.send_text(json.dumps({"text": "hi", "request_id": LONGEST}))
        replies = [json.loads(ws.receive_text()) for _ in range(2)]
    assert replies[-1] == {"request_id": LONGEST, "finish": True}
    assert server.submitted == [LONGEST]
