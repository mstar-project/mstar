import json

import pytest
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import NDJSON_STREAM_MEDIA_TYPE


class _RecordingServer:
    """Stand-in for ``APIServer`` that records the ``streaming`` it was given."""

    def __init__(self):
        self.streaming = None

    async def wait_for_request_id(self, request_id):
        pass

    def submit_request(self, **kwargs):
        self.streaming = kwargs["streaming"]
        return "r"

    def async_stream_results(self, request_id, binary=False):
        async def body():
            yield b'{"modality": "text"}\n'
        return body()

    async def collect_results(self, request_id, request):
        return []


def _post_text(monkeypatch, multipart, **fields):
    server = _RecordingServer()
    monkeypatch.setattr(entrypoint, "api_server", server)
    fields = {"text": "hi", **fields}
    client = TestClient(entrypoint.app)
    if multipart:
        # ``curl -F`` sends multipart/form-data; ``data=`` alone is urlencoded.
        response = client.post("/generate", files={k: (None, v) for k, v in fields.items()})
    else:
        response = client.post("/generate", data=fields)
    return server, response


@pytest.mark.parametrize("multipart", [False, True], ids=["urlencoded", "multipart"])
def test_generate_streams_when_streaming_field_is_omitted(monkeypatch, multipart):
    server, response = _post_text(monkeypatch, multipart)

    assert response.status_code == 200
    assert server.streaming is True
    assert response.headers["content-type"].startswith(NDJSON_STREAM_MEDIA_TYPE)


@pytest.mark.parametrize("multipart", [False, True], ids=["urlencoded", "multipart"])
def test_generate_aggregates_when_streaming_is_false(monkeypatch, multipart):
    server, response = _post_text(monkeypatch, multipart, streaming="false")

    assert response.status_code == 200
    assert server.streaming is False
    assert response.json() == {"request_id": "r", "outputs": {}}


def test_generate_accepts_streaming_only_output_without_streaming_field(monkeypatch):
    server = _RecordingServer()
    monkeypatch.setattr(entrypoint, "api_server", server)

    response = TestClient(entrypoint.app).post(
        "/generate", data={"text": "hi", "output_modalities": "video_frame"},
    )

    assert response.status_code == 200
    assert server.streaming is True


def test_generate_rejects_malformed_model_kwargs(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", object())

    response = TestClient(entrypoint.app).post(
        "/generate",
        data={"model_kwargs": "{"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "model_kwargs must be valid JSON"}


def test_generate_rejects_non_object_model_kwargs(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", object())

    client = TestClient(entrypoint.app)
    for model_kwargs in ("[1, 2]", '"hello"', "42", "true"):
        response = client.post(
            "/generate",
            data={"model_kwargs": model_kwargs},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "model_kwargs must be a JSON object"}


def test_generate_does_not_mislabel_downstream_json_errors(monkeypatch):
    class FailingServer:
        async def wait_for_request_id(self, request_id):
            pass

        def submit_request(self, **kwargs):
            raise json.JSONDecodeError("downstream failure", "{}", 0)

    monkeypatch.setattr(entrypoint, "api_server", FailingServer())

    response = TestClient(entrypoint.app).post("/generate")

    assert response.status_code == 500
    assert response.json()["detail"].startswith("downstream failure")
