import json

import pytest
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import NDJSON_STREAM_MEDIA_TYPE


class _RecordingServer:
    """Stand-in for ``APIServer`` that records the ``streaming`` and outputs it was given."""

    def __init__(self):
        self.streaming = None
        self.output_modalities = None

    async def wait_for_request_id(self, request_id):
        pass

    def submit_request(self, **kwargs):
        self.streaming = kwargs["streaming"]
        self.output_modalities = kwargs["output_modalities"]
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


@pytest.mark.parametrize("multipart", [False, True], ids=["urlencoded", "multipart"])
def test_generate_leaves_an_unnamed_output_to_submit_request(monkeypatch, multipart):
    # submit_request fills in the model's default (not a hard-coded "text")
    server, response = _post_text(monkeypatch, multipart)
    assert response.status_code == 200 and server.output_modalities == []
    server, _ = _post_text(monkeypatch, multipart, output_modalities="audio, text")
    assert server.output_modalities == ["audio", "text"]


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


class _UploadServer(_RecordingServer):
    def __init__(self, upload_dir):
        super().__init__()
        self.upload_dir = upload_dir


@pytest.mark.parametrize("files,fields,detail", [
    ([("clip.wav", b"RIFF"), ("notes.xyz", b"?")], {}, "Cannot determine modality for file: notes.xyz"),
    ([("clip.wav", b"RIFF")], {"input_modalities": "video_frame"}, "'video_frame' is an output-only modality"),
    ([("clip.wav", b"RIFF")], {"model_kwargs": "{"}, "model_kwargs must be valid JSON"),
    ([("clip.wav", b"RIFF")], {"model_kwargs": "[1]"}, "model_kwargs must be a JSON object"),
], ids=["unknown-file", "video-frame-input", "bad-json", "not-an-object"])
def test_a_refused_generate_leaves_no_upload_behind(monkeypatch, tmp_path, files, fields, detail):
    """A request refused after its files are saved is never submitted, so the
    deferred cleanup never sees them; the refusal deletes them."""
    server = _UploadServer(tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", server)

    response = TestClient(entrypoint.app).post(
        "/generate",
        data={"text": "hi", **fields},
        files=[("files", (name, data, "application/octet-stream")) for name, data in files],
    )

    assert (response.status_code, response.json()) == (400, {"detail": detail})
    assert server.streaming is None
    assert list(tmp_path.iterdir()) == []
