"""Error text that reaches a client names no server path, user name or process
id (#592). A ValueError or TypeError raised on purpose keeps its message with
paths cut to base names; anything else becomes an error id, and the server log
carries the full error and traceback under that id."""

import asyncio
import getpass
import json
import os
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_output_failure_teardown import _Model, _TensorManager, _thread

from mstar.api_server import entrypoint
from mstar.api_server.rust_frontend import RustFrontendBridge
from mstar.engine.engine import ExecutingBatch
from mstar.utils.ipc_format import ConductorMessageType
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.worker import Worker

USER = getpass.getuser()
ERROR_ID = re.compile(r"error id ([0-9a-f]{12})\)")


def _assert_clean(text: str) -> None:
    for leak in ("/tmp", "/dev/shm", "/home", "mstar_arena", "mstar_uploads", USER):
        assert leak not in text, f"{leak!r} reaches the client: {text!r}"


def _error_id(text: str) -> str:
    (error_id,) = ERROR_ID.findall(text)
    return error_id


def _logged(caplog, error_id: str):
    """The one log record for ``error_id``, with the traceback attached."""
    (record,) = [r for r in caplog.records if f"[error id {error_id}]" in r.getMessage()]
    assert record.exc_info is not None
    return record


def _upload(tmp_path, name: str) -> str:
    """An upload under a per-user upload dir, as a deployment may set it."""
    upload_dir = tmp_path / f"mstar_uploads_{USER}"
    upload_dir.mkdir()
    path = upload_dir / name
    path.write_bytes(b"not a picture")
    return str(path)


def _data_worker():
    """A data worker with r1 in flight."""
    return _thread(_Model(), _TensorManager())


def test_a_decoder_failure_on_an_upload_reaches_the_client_as_an_error_id(tmp_path, caplog):
    path = _upload(tmp_path, "3f2a9c_photo.png")
    wt = _data_worker()
    with pytest.raises(OSError) as raised:
        Image.open(path)
    assert path in str(raised.value)  # the decoder names the upload
    wt._fail_request("r1", raised.value, "preprocessing", abort=False)

    chunk = wt.out_queue.get_nowait()
    text = chunk.data.decode()
    _assert_clean(text)
    assert text.startswith("preprocessing failed: internal error (error id ")
    assert chunk.metadata["status"] == 500
    assert path in _logged(caplog, _error_id(text)).getMessage()


def test_a_deliberate_decoder_refusal_keeps_its_message_without_the_path(tmp_path, caplog):
    path = _upload(tmp_path, "3f2a9c_clip.mp4")
    wt = _data_worker()
    wt._fail_request(
        "r1", ValueError(f"no video frames decoded from {path}"), "preprocessing", abort=False,
    )

    chunk = wt.out_queue.get_nowait()
    text = chunk.data.decode()
    _assert_clean(text)
    assert text == "preprocessing failed: ValueError: no video frames decoded from 3f2a9c_clip.mp4"
    assert chunk.metadata["status"] == 400
    assert any(path in r.getMessage() for r in caplog.records)


def test_an_arena_io_error_reaches_the_client_as_an_error_id(caplog):
    segment = f"/dev/shm/mstar_arena_worker_0_{os.getpid()}_9c1e.seg3"
    wt = _data_worker()
    wt._fail_request("r1", OSError(f"io on {segment}: Bad address"), "audio output transfer", count=2)

    texts = [wt.out_queue.get_nowait().data.decode() for _ in range(2)]
    for text in texts:
        _assert_clean(text)
    assert texts[0] == texts[1]
    assert segment in _logged(caplog, _error_id(texts[0])).getMessage()
    # the scrubbed error still stops the request's walk, once
    assert wt.communicator.to_conductor() == [
        (ConductorMessageType.ABORT_REQUEST, "r1"),
        (ConductorMessageType.READS_DONE, "r1"),
    ]


def _worker(rids=("r1",)):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.sent = []
    w.communicator = SimpleNamespace(send=lambda entity, msg: w.sent.append(msg))
    w.request_state = SimpleNamespace(per_request_info={rid: object() for rid in rids})
    w._graph_runtime = SimpleNamespace(
        get_rid_string=lambda h: h, set_in_flight=lambda *args: None,
    )
    w.scheduler = MicroScheduler.__new__(MicroScheduler)
    w.scheduler.failed_rids = set()
    w.scheduler.held_until = {}
    w.scheduler.admit_errors = {}
    w.scheduler._core = SimpleNamespace(drop_rids=lambda rids: None)  # nothing backlogged
    w._in_flight_rids = set(rids)
    return w


def test_a_worker_exception_reaches_the_client_as_an_error_id(tmp_path, caplog):
    tensor_file = f"{tmp_path}/mstar_tensors_{USER}/r1_audio.pt"
    w = _worker()
    w._handle_main_loop_error(FileNotFoundError(f"cannot open {tensor_file}"), (None, None), None)

    (message,) = w.sent
    text = message.body.errors["r1"]
    _assert_clean(text)
    assert text.startswith("Error in worker: internal error (error id ")
    assert tensor_file in _logged(caplog, _error_id(text)).getMessage()


def test_a_per_request_stage_failure_reaches_the_client_as_an_error_id(tmp_path, caplog):
    batch = SimpleNamespace(failed_requests={}, node_name="talker", graph_walk="decode")
    ExecutingBatch.register_failure(batch, 7, RuntimeError(f"bad weights at {tmp_path}"))
    ExecutingBatch.register_failure(batch, 8, ValueError(f"no frames in {tmp_path}/a_b.wav"))

    _assert_clean(batch.failed_requests[7])
    assert str(tmp_path) in _logged(caplog, _error_id(batch.failed_requests[7])).getMessage()
    assert batch.failed_requests[8] == "ValueError: no frames in a_b.wav"


class _FailingServer:
    takes_stream_input = False

    def __init__(self, exc, upload_dir=None):
        self.exc = exc
        self.upload_dir = upload_dir

    async def wait_for_request_id(self, request_id):
        return None

    def submission_key(self, request_id):
        return request_id

    def abort_submission(self, key):
        pass

    def submit_request(self, **kwargs):
        raise self.exc

    async def iter_result_chunks(self, rid):
        raise self.exc
        yield


def test_the_generate_catch_all_answers_with_an_error_id(monkeypatch, caplog):
    segment = f"/dev/shm/mstar_arena_api_{os.getpid()}_77.seg0"
    monkeypatch.setattr(entrypoint, "api_server", _FailingServer(RuntimeError(f"io on {segment}")))

    response = TestClient(entrypoint.app).post("/generate", data={"text": "hi"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    _assert_clean(detail)
    assert detail.startswith("internal error (error id ")
    assert segment in _logged(caplog, _error_id(detail)).getMessage()


def test_the_generate_catch_all_keeps_a_deliberate_refusal(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", _FailingServer(ValueError("Unsupported modality: 'smell'")))
    response = TestClient(entrypoint.app).post("/generate", data={"text": "hi"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported modality: 'smell'"}


def test_generate_ws_answers_with_an_error_id(monkeypatch, tmp_path, caplog):
    segment = f"/dev/shm/mstar_arena_api_{os.getpid()}_77.seg0"
    server = _FailingServer(RuntimeError(f"io on {segment}"), upload_dir=tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", server)

    with TestClient(entrypoint.app).websocket_connect("/generate/ws") as ws:
        ws.send_text(json.dumps({"text": "hi", "request_id": "r1"}))
        reply = json.loads(ws.receive_text())

    assert reply["request_id"] == "r1"
    _assert_clean(reply["error"])
    assert segment in _logged(caplog, _error_id(reply["error"])).getMessage()


def test_the_rust_frontend_bridge_answers_with_an_error_id(tmp_path, caplog):
    tensor_file = f"{tmp_path}/mstar_tensors_{USER}/r1.pt"
    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.sent = []
    bridge._send = bridge.sent.append
    bridge._keys = {}
    bridge.server = _FailingServer(FileNotFoundError(f"cannot open {tensor_file}"))

    bridge._submit({"rid": "r1", "text": "hi"})
    asyncio.run(bridge._relay("r2", None, bridge.server.iter_result_chunks("r2")))

    assert [m["rid"] for m in bridge.sent] == ["r1", "r2"]
    for message in bridge.sent:
        assert message["t"] == "err"
        _assert_clean(message["msg"])
        assert tensor_file in _logged(caplog, _error_id(message["msg"])).getMessage()


@pytest.mark.parametrize("raw,scrubbed", [
    ("Error opening '/tmp/up_alice/9f_a.wav': Format not recognised.",
     "Error opening '9f_a.wav': Format not recognised."),
    ("io on /dev/shm/mstar_arena_w_41_t.seg2: Bad address", "io on mstar_arena_w_41_t.seg2: Bad address"),
    ("cannot read file:///home/alice/x.png", "cannot read x.png"),
    ("not a file: /home/alice/", "not a file: <path>"),
    ("Error opening '/tmp/uploads (alice)/7_bad.wav': Format not recognised.",
     "Error opening '7_bad.wav': Format not recognised."),
    ('[Errno 2] No such file or directory: "/srv/my uploads/[alice]/x.pt"',
     '[Errno 2] No such file or directory: "x.pt"'),
    ("fetch https://example.com/a/b.png failed", "fetch https://example.com/a/b.png failed"),
    ("expected 3/4 of shape (2, 3)/2 and/or ./rel/path", "expected 3/4 of shape (2, 3)/2 and/or ./rel/path"),
])
def test_scrub_paths_cuts_absolute_paths_to_base_names(raw, scrubbed):
    from mstar.utils.client_errors import scrub_paths

    assert scrub_paths(raw) == scrubbed


def test_a_failed_kv_retrieve_reaches_the_client_without_its_path(caplog):
    from concurrent.futures import Future

    from mstar.engine.resources.kv.manager import CacheStream, KVManager

    snapshot = "/srv/private/alice/kv-worker-12345.pt"
    future = Future()
    future.set_exception(FileNotFoundError(2, "No such file or directory", snapshot))
    manager = SimpleNamespace(_streams={"r1": {"main": CacheStream(read_pending=True, read_future=future)}})
    with caplog.at_level("ERROR"):
        reasons = [KVManager._check_ready(manager, "r1", "main") for _ in range(2)]
    assert [ready for ready, _ in reasons] == [False, False]
    messages = {reason.message for _, reason in reasons}
    (message,) = messages
    _assert_clean(message)
    assert "/srv" not in message and "alice" not in message
    assert snapshot in _logged(caplog, _error_id(message)).getMessage()


SECRET = "token=sk-not-a-real-key-4242"

# Each OpenAI route, the model it is served for, the serving function its
# handler awaits, and a request that reaches that function.
_ROUTES = [
    ("/v1/chat/completions", "bagel", "serving_chat", "create_chat_completion",
     {"json": {"model": "bagel", "messages": [{"role": "user", "content": "hi"}]}}),
    ("/v1/audio/speech", "orpheus", "serving_speech", "create_speech",
     {"json": {"model": "orpheus", "input": "hi", "voice": "tara"}}),
    ("/v1/audio/transcriptions", "whisper_large", "serving_transcriptions", "create_transcription",
     {"files": {"file": ("a.wav", b"RIFF")}}),
    ("/v1/images/generations", "bagel", "serving_images", "create_images",
     {"json": {"model": "bagel", "prompt": "a cat"}}),
    ("/v1/videos/generations", "wan22", "serving_videos", "create_videos",
     {"json": {"model": "wan22", "prompt": "a cat"}}),
    ("/v1/images/edits", "bagel", "serving_images", "create_image_edit",
     {"files": {"image": ("in.png", b"\x89PNG")}, "data": {"prompt": "x"}}),
]


def _openai_client(monkeypatch, model_name, upload_dir):
    from fastapi import FastAPI

    from mstar.api_server.openai.router import router

    monkeypatch.setattr(entrypoint, "api_server", SimpleNamespace(
        model_name=model_name, upload_dir=str(upload_dir), model=None,
    ))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.parametrize("route,model,module,func,request_kw", _ROUTES, ids=[r[0] for r in _ROUTES])
def test_an_openai_route_answers_an_internal_error_with_an_error_id(
    monkeypatch, tmp_path, caplog, route, model, module, func, request_kw,
):
    import importlib

    upload = _upload(tmp_path, "7_clip.wav")

    async def raising(*args, **kwargs):
        raise RuntimeError(f"Could not open input file: {upload} ({SECRET})")

    monkeypatch.setattr(importlib.import_module(f"mstar.api_server.openai.{module}"), func, raising)
    response = _openai_client(monkeypatch, model, tmp_path).post(route, **request_kw)

    assert response.status_code == 500
    message = response.json()["error"]["message"]
    _assert_clean(response.text)
    assert SECRET not in response.text and "7_clip.wav" not in response.text
    assert message.startswith("internal error (error id ")
    logged = _logged(caplog, _error_id(message)).getMessage()
    assert upload in logged and SECRET in logged and route in logged


@pytest.mark.parametrize("route,model,module,func,request_kw", _ROUTES, ids=[r[0] for r in _ROUTES])
def test_an_openai_route_keeps_a_refusal_without_its_path(
    monkeypatch, tmp_path, caplog, route, model, module, func, request_kw,
):
    import importlib

    upload = _upload(tmp_path, "7_clip.wav")

    async def raising(*args, **kwargs):
        raise ValueError(f"unsupported sample rate in {upload}")

    monkeypatch.setattr(importlib.import_module(f"mstar.api_server.openai.{module}"), func, raising)
    response = _openai_client(monkeypatch, model, tmp_path).post(route, **request_kw)

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "unsupported sample rate in 7_clip.wav"


def _wav_header(seconds_claimed: float, body: bytes, rate: int = 16000) -> bytes:
    """A 16-bit mono WAV whose header claims ``seconds_claimed`` of audio."""
    import struct

    claimed = int(seconds_claimed * rate * 2)
    return (
        b"RIFF" + struct.pack("<I", 36 + claimed) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", claimed) + body
    )


def _corrupt_flac(seconds: float) -> bytes:
    """A FLAC whose header says ``seconds`` and whose middle is noise, so
    libsndfile opens it and loses sync partway through the decode."""
    import io

    import numpy as np
    sf = pytest.importorskip("soundfile")

    buf = io.BytesIO()
    noise = np.random.default_rng(0).standard_normal(int(seconds * 16000)).astype("float32") * 0.1
    sf.write(buf, noise, 16000, format="FLAC")
    data = bytearray(buf.getvalue())
    third = len(data) // 3
    data[third:third + 20000] = np.random.default_rng(1).bytes(20000)
    return bytes(data)


def _rss_mib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def test_a_transcription_whose_decode_fails_in_the_frontend_answers_with_an_error_id(
    monkeypatch, tmp_path, caplog,
):
    """A header longer than the model's window makes the frontend decode the
    upload (``WindowPlanner.prepare``). When that decode fails with an
    unexpected error, here the FFmpeg fallback naming the saved upload as
    torchcodec does, the client gets an error id and the upload is deleted."""
    from mstar.api_server import media_io

    seen = []

    def ffmpeg_fails(path, *args, **kwargs):
        seen.append(path)
        raise RuntimeError(f"Could not open input file: {path} ({SECRET})")

    monkeypatch.setattr(media_io, "decode_torchcodec_capped", ffmpeg_fails)
    response = _openai_client(monkeypatch, "whisper_large", tmp_path).post(
        "/v1/audio/transcriptions", files={"file": ("lecture.flac", _corrupt_flac(40))},
    )

    assert len(seen) == 1, "the frontend decode did not run"
    assert response.status_code == 500
    _assert_clean(response.text)
    assert SECRET not in response.text and "lecture" not in response.text
    message = response.json()["error"]["message"]
    assert seen[0] in _logged(caplog, _error_id(message)).getMessage()
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("name,payload", [
    ("garbage.wav", bytes(range(256)) * 200),
    ("truncated.wav", _wav_header(1, b"")[:30]),
    ("huge_header.wav", _wav_header(4 * 3600, b"\0" * 32000)),
    ("long_header_short_body.wav", _wav_header(600, b"\0" * 32000)),
    ("corrupt_long.flac", None),
], ids=lambda v: v if isinstance(v, str) else "")
def test_hostile_transcription_uploads_get_a_clean_answer(monkeypatch, tmp_path, caplog, name, payload):
    """Malformed files and headers that lie about their length: the answer
    is the engine's (a stub here), a 4xx, or a 500 with an error id; never a
    server path, never unbounded memory, and no upload left behind."""
    if payload is None:
        payload = _corrupt_flac(40)
    client = _openai_client(monkeypatch, "whisper_large", tmp_path)

    async def transcribe(**kwargs):
        return [SimpleNamespace(modality="text", data=b"<|en|> hi", metadata={})]

    entrypoint.api_server.submit_request = lambda **kw: kw["request_id"]
    entrypoint.api_server.collect_results = lambda request_id, raw_request=None: transcribe()
    before = _rss_mib()
    response = client.post("/v1/audio/transcriptions", files={"file": (name, payload)})
    grew = _rss_mib() - before

    _assert_clean(response.text)
    if response.status_code >= 500:
        assert response.json()["error"]["message"].startswith("internal error (error id ")
    else:
        assert response.status_code in (200, 400, 413), response.text
    assert grew < 256, f"{name}: RSS grew {grew:.0f} MiB"
    assert os.listdir(tmp_path) == []


def test_an_openai_route_keeps_the_status_an_exception_names(monkeypatch, tmp_path):
    from mstar.api_server import media_io
    from mstar.api_server.openai import serving_chat

    async def too_long(*args, **kwargs):
        raise media_io.AudioTooLongError(1800)

    monkeypatch.setattr(serving_chat, "create_chat_completion", too_long)
    response = _openai_client(monkeypatch, "bagel", tmp_path).post(
        "/v1/chat/completions", json={"model": "bagel", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 413
    assert response.json()["error"]["message"] == str(media_io.AudioTooLongError(1800))
