"""``/v1/videos/generations`` on both frontends: each video adapter's submit or
refusal, the generated audio track muxed into the mp4, and ``stream_video``'s
NDJSON lines. The real ``mstar-server`` and bridge run against a scripted data
plane; the Python route runs in-process (FastAPI's TestClient)."""
import base64
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _model_stack, _StubAPIServer
from test_rust_openai_adapters import _model_of

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import APIServer
from mstar.api_server.openai._util import check_inline_result
from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.served_models import ServedModels

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _clip(seconds: float = 1.0, fps: int = 24) -> bytes:
    """A small H.264 mp4, as the video models emit."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=64x64:rate={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-"],
        capture_output=True, check=True,
    )
    return out.stdout


def _pcm(seconds: float = 1.0, rate: int = 48000, channels: int = 2) -> bytes:
    return bytes((i * 7) % 251 for i in range(int(seconds * rate) * channels * 2))


def _chunks(prompt: str, video: bytes) -> list:
    """What the scripted data plane returns for ``prompt``."""
    if "sound" in prompt:
        return [_Chunk("video", video), _Chunk("audio", _pcm(), {"sample_rate": 48000, "num_channels": 2})]
    if "windows" in prompt:
        return [_Chunk("video", video, {"window": 0}), _Chunk("video", video, {"window": 1})]
    if "fails" in prompt:
        return [_Chunk("video", video), _Chunk("error", b"Error in worker: boom", {"status": 500})]
    if "refused" in prompt:
        return [_Chunk("error", b"could not decode the conditioning image", {"status": 400})]
    return [_Chunk("video", video)]


class _RustVideoStub(_StubAPIServer):
    def __init__(self, video: bytes):
        super().__init__()
        self.video = video

    async def iter_result_chunks(self, request_id):
        for c in _chunks(self.texts.get(request_id) or "", self.video):
            yield c


class _PythonVideoServer:
    """What ``create_videos`` calls on APIServer, answering as the Rust stub does."""

    def __init__(self, model_name, upload_dir, video: bytes):
        self.model_name, self.upload_dir, self.model, self.video = model_name, upload_dir, None, video
        self.served_models = ServedModels.of(model_name, None)
        self.submitted, self.texts = [], {}

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        self.texts[kwargs["request_id"]] = kwargs["text"]
        return kwargs["request_id"]

    async def iter_result_chunks(self, request_id):
        for c in _chunks(self.texts[request_id], self.video):
            yield c

    async def collect_results(self, request_id, raw_request=None):
        return _chunks(self.texts[request_id], self.video)

    def release_request(self, request_id):
        pass

    lora_kwargs = APIServer.lora_kwargs
    open_result_stream = APIServer.open_result_stream


def _post(port, body, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/videos/generations", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("content-type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type"), e.read()


VIDEO_CLASSES = ["Cosmos3Adapter", "LTX25Adapter", "MiniMaxH3Adapter", "Wan22Adapter"]

VIDEO_CASES = {
    "text": {"prompt": "a cat"},
    "fields": {"prompt": "a cat", "size": "832x480", "seed": 4, "num_frames": 33, "fps": 16.0,
               "guidance_scale": 5.0},
    "size-spaced": {"prompt": "a cat", "size": " 960 X 544 "},
    "bad-size": {"prompt": "a cat", "size": "wide"},
    "three-sides": {"prompt": "a cat", "size": "1x2x3"},
    "image": {"prompt": "a cat", "image": f"data:image/png;base64,{PNG}"},
    "video": {"prompt": "a cat", "video": f"data:video/mp4;base64,{PNG}"},
    "both": {"prompt": "a cat", "image": f"data:image/png;base64,{PNG}", "video": f"data:video/mp4;base64,{PNG}"},
    "empty-strings": {"prompt": "a cat", "image": "", "video": "", "size": ""},
    "no-audio": {"prompt": "a cat", "audio": False},
    "server-path": {"prompt": "a cat", "image": "/etc/passwd"},
    # MiniMax-H3's keyframes, sound switch, frame rate and count
    "keyframes": {"prompt": "a cat", "image": f"data:image/png;base64,{PNG}",
                  "last_image": f"data:image/png;base64,{PNG}"},
    "last-image-only": {"prompt": "a cat", "last_image": f"data:image/png;base64,{PNG}"},
    "last-image-not-a-string": {"prompt": "a cat", "last_image": 5},
    "last-image-refused": {"prompt": "a cat", "image": f"data:image/png;base64,{PNG}", "last_image": "/etc/passwd"},
    "audio-not-a-bool": {"prompt": "a cat", "audio": "yes"},
    "audio-null": {"prompt": "a cat", "audio": None},
    "fps-24": {"prompt": "a cat", "fps": 24},
    "fps-30": {"prompt": "a cat", "fps": 30},
    "two": {"prompt": "a cat", "n": 2},
    "url-format": {"prompt": "a cat", "response_format": "url"},
}


@pytest.mark.parametrize("case", sorted(VIDEO_CASES))
@pytest.mark.parametrize("cls", VIDEO_CLASSES)
def test_a_video_request_is_submitted_or_refused_as_the_python_adapter_does(cls, case, tmp_path):
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    model = _model_of(cls)
    body = {"model": model, **VIDEO_CASES[case]}
    try:
        req = VideoGenerationRequest(**body)
        check_inline_result(req.response_format, req.n)  # create_videos', before the adapter
        want = ADAPTER_REGISTRY[model].video_to_request(req, tmp_path)
        refusal = None
    except ValueError as e:
        want, refusal = None, str(e)
    stub = _RustVideoStub(b"mp4")
    with _model_stack(model, stub) as (port, upload):
        status, _, data = _post(port, body)
        left = list(Path(upload).iterdir())
    if refusal is not None:
        assert (status, json.loads(data)["error"]["message"]) == (400, refusal)
        assert not stub.submitted
        # a keyframe saved before the refusal goes with it
        assert left == []
        return
    assert status == 200, data
    (sub,) = stub.submitted
    assert (sub["text"], sub["in"], sub["out"], sub["mk"]) == (
        want.text, want.input_modalities, want.output_modalities, want.model_kwargs)
    counts = lambda files: {m: len(p) for m, p in (files or {}).items()}  # noqa: E731
    assert counts(sub["files"]) == counts(want.file_paths)



@pytest.mark.parametrize("field", ["image", "last_image"])
@pytest.mark.parametrize("cls", VIDEO_CLASSES)
def test_a_remote_keyframe_follows_the_servers_remote_fetch_setting(cls, field, tmp_path, monkeypatch):
    from mstar.api_server.openai.protocol import VideoGenerationRequest

    # remote fetch on, but no host allowed: the refusal says which check ran
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    monkeypatch.setenv("MSTAR_REMOTE_ALLOWED_HOSTS", "")
    model = _model_of(cls)
    body = {"model": model, "prompt": "a cat", field: "https://example.com/frame.png"}
    try:
        want = ADAPTER_REGISTRY[model].video_to_request(VideoGenerationRequest(**body), tmp_path)
        refusal = None
    except ValueError as e:
        want, refusal = None, str(e)
    stub = _RustVideoStub(b"mp4")
    with _model_stack(model, stub) as (port, _upload):
        status, _, data = _post(port, body)
    if refusal is None:
        assert status == 200, data
        assert stub.submitted[0]["mk"] == want.model_kwargs
    else:
        assert (status, json.loads(data)["error"]["message"]) == (400, refusal)

def _python_videos(model, body, tmp_path, monkeypatch, video):
    monkeypatch.setattr(entrypoint, "api_server", _PythonVideoServer(model, tmp_path, video))
    return TestClient(entrypoint.app).post("/v1/videos/generations", json=body)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg and ffprobe mux the audio track")


@needs_ffmpeg
@pytest.mark.parametrize("fps", [None, 24.0, 16.0, "12"], ids=["no-fps", "container-fps", "rescaled", "string-fps"])
def test_a_videos_generated_sound_is_muxed_into_the_python_routes_mp4(fps, tmp_path, monkeypatch):
    """The audio chunk becomes the mp4's AAC track by the same ffprobe and
    ffmpeg commands, its timestamps rescaled to a request ``fps`` that differs
    from the container's: the file is the Python route's, byte for byte."""
    video = _clip()
    model = _model_of("LTX25Adapter")
    body = {"model": model, "prompt": "with sound"}
    if fps is not None:
        body["fps"] = fps
    py = _python_videos(model, body, tmp_path, monkeypatch, video)
    with _model_stack(model, _RustVideoStub(video)) as (port, _upload):
        status, _, data = _post(port, body)
    assert (status, py.status_code) == (200, 200), data
    rust_mp4 = base64.b64decode(json.loads(data)["data"][0]["b64_json"])
    python_mp4 = base64.b64decode(py.json()["data"][0]["b64_json"])
    assert rust_mp4 == python_mp4
    assert rust_mp4 != video, "no audio track was muxed in"
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", "-"],
                             input=rust_mp4, capture_output=True, check=True).stdout.decode().split()
    assert sorted(streams) == ["audio", "video"]


@needs_ffmpeg
def test_a_failed_mux_returns_the_video_alone_as_python_does(tmp_path, monkeypatch):
    video = b"not an mp4"
    model = _model_of("LTX25Adapter")
    body = {"model": model, "prompt": "with sound"}
    py = _python_videos(model, body, tmp_path, monkeypatch, video)
    with _model_stack(model, _RustVideoStub(video)) as (port, _upload):
        status, _, data = _post(port, body)
    assert status == py.status_code == 200
    assert base64.b64decode(json.loads(data)["data"][0]["b64_json"]) == video == base64.b64decode(
        py.json()["data"][0]["b64_json"])


@pytest.mark.parametrize("prompt", ["two windows", "it fails late"], ids=["windows", "late-failure"])
def test_stream_video_writes_the_python_routes_ndjson(prompt, tmp_path, monkeypatch):
    """Each window as a line with a running ``index``, then ``done`` with the
    count; a failure after the 200 ends it with its ``error`` line instead."""
    model = _model_of("Wan22Adapter")
    body = {"model": model, "prompt": prompt, "stream_video": True}
    py = _python_videos(model, body, tmp_path, monkeypatch, b"mp4-bytes")
    with _model_stack(model, _RustVideoStub(b"mp4-bytes")) as (port, _upload):
        status, ctype, data = _post(port, body)
    assert (status, ctype) == (py.status_code, py.headers["content-type"]) == (200, "application/x-ndjson")
    rust_lines = [json.loads(line) for line in data.splitlines()]
    python_lines = [json.loads(line) for line in py.content.splitlines()]
    assert rust_lines == python_lines
    assert rust_lines[-1]["modality"] == ("done" if "windows" in prompt else "error")


def test_a_stream_video_the_engine_refuses_at_once_gets_its_status_on_both(tmp_path, monkeypatch):
    """A refusal before the first window (a conditioning image that doesn't
    decode) is the response's status, not a 200 with an ``error`` line."""
    model = _model_of("Wan22Adapter")
    body = {"model": model, "prompt": "refused", "stream_video": True}
    py = _python_videos(model, body, tmp_path, monkeypatch, b"mp4-bytes")
    with _model_stack(model, _RustVideoStub(b"mp4-bytes")) as (port, _upload):
        status, _, data = _post(port, body)
    assert status == py.status_code == 400
    assert json.loads(data) == py.json()
