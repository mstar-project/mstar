"""A streaming response whose body Starlette never started still releases
what the body would have released.

Starlette sends the headers before it iterates the body. When that send raises
(uvicorn 0.28.0 does once the client has left), or a server cancels the response
before its first step, the body is never started, so its ``finally`` never runs:
the engine request ran to the end with nobody reading it, its ``request_id`` was
refused with a 409 until then, and its pending entry and read-ahead pieces or
uploads stayed for the life of the process. Current uvicorn always starts the
body, so these tests drive the response with the ASGI calls directly."""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_server_helpers import make_test_api_server  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import ClientDisconnect  # noqa: E402
from test_serving_metrics import _count, _labels, _wait, served  # noqa: E402, F401

from mstar.api_server.request_types import ResultChunk  # noqa: E402

_ASGI_2_4 = {"type": "http", "asgi": {"spec_version": "2.4"}}
_ASGI_2_0 = {"type": "http", "asgi": {"spec_version": "2.0"}}


async def _disconnected():
    return {"type": "http.disconnect"}


async def _client_gone(message):
    raise OSError("client gone")


class _PreprocessWorker:
    def __init__(self):
        self.submitted, self.aborted = [], []

    def new_request(self, input):
        self.submitted.append(input.request_id)

    def abort_request(self, key):
        self.aborted.append(key)


def _generate_stream(monkeypatch, request_id="r1"):
    """The streaming ``/generate`` response, built by the real route."""
    from mstar.api_server import entrypoint

    async def form():
        return {"text": "hi", "streaming": "true"}

    server = make_test_api_server(preprocess_worker=_PreprocessWorker())
    monkeypatch.setattr(entrypoint, "api_server", server)
    response = asyncio.run(entrypoint.generate(
        request=SimpleNamespace(headers={}, form=form), text="hi", files=None,
        input_modalities=None, output_modalities="text", streaming=True,
        model_kwargs=None, request_id=request_id,
    ))
    return server, response, server._live[request_id]


def test_generate_aborts_its_request_when_the_headers_cannot_be_sent(monkeypatch):
    server, response, key = _generate_stream(monkeypatch)
    with pytest.raises(ClientDisconnect):
        asyncio.run(response(_ASGI_2_4, _disconnected, _client_gone))
    assert key not in server.pending_requests and "r1" not in server._live
    assert server.preprocess_worker.aborted == [key]


def test_generate_aborts_its_request_when_the_disconnect_wins_the_race(monkeypatch):
    """A server whose ``send`` blocks while the client leaves, so Starlette
    (below ASGI 2.4) cancels the response before the body starts. uvicorn's
    ``send`` never blocks like this; the case guards other ASGI servers."""
    server, response, key = _generate_stream(monkeypatch)

    async def stuck(message):
        await asyncio.Event().wait()

    asyncio.run(response(_ASGI_2_0, _disconnected, stuck))
    assert key not in server.pending_requests
    assert server.preprocess_worker.aborted == [key]


def test_a_generate_stream_that_was_read_is_not_aborted_on_close(monkeypatch):
    server, response, key = _generate_stream(monkeypatch)
    req = server.pending_requests[key]
    req.chunks.append(ResultChunk(request_id="r1", modality="text", data=b"out", metadata={}))
    req.event.set()
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(response(_ASGI_2_4, _disconnected, send))
    assert b"".join(m.get("body", b"") for m in sent)
    assert server.preprocess_worker.aborted == []


def test_a_finished_stream_released_before_its_body_counts_once_as_an_abort(served):  # noqa: F811
    from mstar.api_server import entrypoint
    from mstar.observability.metrics import OUTCOMES

    async def form():
        return {"text": "hi", "streaming": "true"}

    response = asyncio.run(entrypoint.generate(
        request=SimpleNamespace(headers={}, form=form), text="hi", files=None,
        input_modalities=None, output_modalities="text", streaming=True,
        model_kwargs=None, request_id="r1",
    ))
    key = served.key()
    served.answer(key)
    _wait(served.server.pending_requests[key].event.is_set)
    with pytest.raises(ClientDisconnect):
        asyncio.run(response(_ASGI_2_4, _disconnected, _client_gone))

    samples = served.scrape()
    assert {o: _count(samples, o) for o in OUTCOMES if _count(samples, o)} == {"abort": 1}
    assert samples[("mstar_requests_in_flight", _labels())] == 0


def test_a_release_targets_its_own_submission_not_a_later_one(monkeypatch):
    """The first stream's request finished and the client resubmitted its id
    before that stream closed: closing it leaves the new submission alone."""
    server, response, old = _generate_stream(monkeypatch)
    server.pending_requests[old].event.set()
    server.submit_request(input_modalities=["text"], output_modalities=["text"], request_id="r1")
    new = server._live["r1"]
    assert new != old
    with pytest.raises(ClientDisconnect):
        asyncio.run(response(_ASGI_2_4, _disconnected, _client_gone))
    assert old not in server.pending_requests and new in server.pending_requests
    assert server.preprocess_worker.aborted == []


# ── OpenAI routes ──────────────────────────────────────────────────────────


class _HeadersFail:
    """ASGI 2.4 with a ``send`` that raises once the client is gone, as
    uvicorn 0.28.0 does, before the headers are sent."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def gone(message):
            if message["type"] == "http.response.start":
                raise OSError("client gone")
            await send(message)

        if scope["type"] == "http":
            scope = {**scope, "asgi": {**scope.get("asgi", {}), "spec_version": "2.4"}}
        await self.app(scope, receive, gone)


def _client(monkeypatch, stub):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(_HeadersFail(app), raise_server_exceptions=False)


def _router_stub(model_name):
    from test_openai_router import _StubAPI

    stub = _StubAPI(model_name)
    stub.released = []
    stub.release_request = stub.released.append
    return stub


def test_chat_completions_stream_releases_its_request(monkeypatch):
    stub = _router_stub("bagel")
    client = _client(monkeypatch, stub)
    client.post("/v1/chat/completions", json={
        "model": "bagel", "stream": True, "messages": [{"role": "user", "content": "hi"}],
    })
    assert stub.released == [stub.last_submit["request_id"]]


def test_video_stream_releases_its_request(monkeypatch):
    stub = _router_stub("cosmos3")
    client = _client(monkeypatch, stub)
    client.post("/v1/videos/generations", json={
        "model": "cosmos3", "prompt": "a road", "num_frames": 57,
        "window_mode": "chained", "stream_video": True,
    })
    assert stub.released == [stub.last_submit["request_id"]]


def test_speech_stream_releases_every_piece_it_submitted(monkeypatch):
    from test_speech_chunking import LONG_TEXT, _RecordingAPI

    from mstar.api_server.openai import adapters

    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_min_chars", 100)
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_chunk_max_chars", 200)
    stub = _RecordingAPI()
    client = _client(monkeypatch, stub)
    client.post("/v1/audio/speech", json={"model": "orpheus", "input": LONG_TEXT, "stream": True})
    submitted = [kw["request_id"] for kw in stub.submits]
    assert len(submitted) >= 2, "the first piece and one read ahead"
    assert stub.released == submitted


def test_speech_stream_schedules_its_reference_clip_cleanup(monkeypatch):
    """The voice-clone clip is deleted after the cleanup delay, as it is when
    the body runs."""
    pytest.importorskip("soundfile")
    from test_openai_router import _Chunk, _left_after_cleanup, _pcm, _wav_b64

    from mstar.api_server import media_io

    monkeypatch.setattr(media_io, "UPLOAD_CLEANUP_DELAY_S", 0.0)
    stub = _router_stub("qwen3_tts_base")
    stub.next_chunks = [_Chunk("audio", _pcm([1, 2, 3]))]
    client = _client(monkeypatch, stub)
    client.post("/v1/audio/speech", json={
        "input": "hi", "ref_text": "hi", "ref_audio": _wav_b64(), "stream": True,
    })
    saved = [p for ps in stub.last_submit["file_paths"].values() for p in ps]
    assert saved and all(Path(p).parent == stub.upload_dir for p in saved)
    assert stub.released == [stub.last_submit["request_id"]]
    assert _left_after_cleanup(stub.upload_dir) == []


def test_transcription_stream_deletes_its_upload(monkeypatch):
    from test_openai_transcriptions import _StubAPI, _wav_bytes

    stub = _StubAPI()
    client = _client(monkeypatch, stub)
    client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true"},
        files={"file": ("long.wav", _wav_bytes(45), "audio/wav")},
    )
    assert stub.submits == []
    assert not list(Path(stub.upload_dir).iterdir())


# ── release failures and a real client ─────────────────────────────────────


def test_a_failing_release_is_logged_and_the_send_error_stays(caplog):
    """``release`` raising in the ``finally`` must not replace the error that
    left the body unstarted: the server sees the client leaving, and the
    release failure is in the log."""
    from mstar.api_server.streaming_response import ReleasingStreamingResponse

    async def body():
        yield b"never"

    def release():
        raise RuntimeError("release broke")

    response = ReleasingStreamingResponse(body(), release=release)
    with caplog.at_level("ERROR"), pytest.raises(ClientDisconnect):
        asyncio.run(response(_ASGI_2_4, _disconnected, _client_gone))
    assert "body never started failed" in caplog.text and "release broke" in caplog.text


@pytest.mark.parametrize("close", ["right_after_sending", "once_submitted"])
def test_a_real_client_that_leaves_before_the_first_byte_is_released(served, close):  # noqa: F811
    """A real uvicorn server and a raw socket client: the client sends a
    streaming /generate and closes before any byte of the reply arrives,
    either at once or once the request is submitted. The request is aborted,
    and neither its pending entry, its request_id nor its in-flight slot is
    left behind. Closing at once, while the route is still waiting (here on
    a slow request_id check), is the case uvicorn 0.28.0 leaves unstarted:
    its send raises on the headers. Current uvicorn starts the body."""
    import socket
    import threading

    import uvicorn

    from mstar.api_server import entrypoint

    if close == "right_after_sending":
        wait_for_request_id = served.server.wait_for_request_id

        async def slow(request_id):
            await asyncio.sleep(0.5)   # the client is gone by now
            await wait_for_request_id(request_id)

        served.server.wait_for_request_id = slow
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(entrypoint.app, port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait(lambda: server.started, timeout=15)
        form = b"text=hi&streaming=true&request_id=r1"
        with socket.create_connection(("127.0.0.1", port)) as client:
            client.sendall(b"POST /generate HTTP/1.1\r\nHost: x\r\n"
                           b"Content-Type: application/x-www-form-urlencoded\r\n"
                           b"Content-Length: %d\r\n\r\n%s" % (len(form), form))
            if close == "once_submitted":
                _wait(lambda: served.data_worker.inputs)
                assert "r1" in served.server._live
        # closed without reading; nothing was answered, so no body byte was sent
        _wait(lambda: served.data_worker.inputs)
        key = served.key()
        _wait(lambda: served.data_worker.aborted == [key], timeout=10)
        _wait(lambda: key not in served.server.pending_requests and "r1" not in served.server._live)
        samples = served.scrape()
        assert samples[("mstar_requests_in_flight", _labels())] == 0
    finally:
        server.should_exit = True
        thread.join(timeout=10)
