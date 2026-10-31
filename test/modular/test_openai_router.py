"""End-to-end test of the OpenAI router with a stubbed APIServer (no GPU/torch).

Mounts the real router on a FastAPI app and drives it via TestClient; the
APIServer + model are stubbed so adapters, serving handlers, SSE, and error
paths are all exercised without the engine.
"""

import asyncio
import base64
import io
import json
import sys
import tempfile
import time
import types
import wave
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server.entrypoint import APIServer  # noqa: E402


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _StubModel:
    def get_output_sample_rate(self, modality="audio"):
        return 24000


class _StubAPI:
    open_result_stream = APIServer.open_result_stream

    def __init__(self, model_name="bagel"):
        self.model_name = model_name
        self.model = _StubModel()
        self.upload_dir = Path(tempfile.mkdtemp())
        self.last_submit = None
        self._chunks: dict = {}
        self.next_chunks: list = []
        self.last_raw_request = None
        # raised out of the stream after its chunks, like the delivery timeout
        self.raise_after: Exception | None = None
        self.aborted: list = []
        self.ended: dict = {}  # how each submission stopped early ended

    def submit_request(self, **kw):
        self.last_submit = kw
        self._chunks[kw["request_id"]] = list(self.next_chunks)
        return kw["request_id"]

    def submission_key(self, request_id):
        return request_id

    def abort_submission(self, key, outcome="abort"):
        # once per submission, as the real one: after a handler ends a whole
        # reply (a success), the stream's close aborts nothing more
        if key not in self.ended:
            self.ended[key] = outcome
            self.aborted.append(key)

    async def collect_results(self, request_id, raw_request=None):
        self.last_raw_request = raw_request
        return self._chunks.get(request_id, [])

    async def iter_result_chunk_batches(self, request_id):
        # Same contract as the real one: a consumer that stops early, or an
        # exception, aborts the request; a fully drained stream does not. The
        # chunks come the way the server hands them out, in batches (here all
        # at once), so the streaming handlers see the coalesced form.
        finished = False
        try:
            chunks = self._chunks.get(request_id, [])
            if chunks:
                yield list(chunks)
            if self.raise_after is not None:
                raise self.raise_after
            finished = True
        finally:
            if not finished:
                self.abort_submission(request_id)

    async def iter_result_chunks(self, request_id):
        batches = self.iter_result_chunk_batches(request_id)
        try:
            async for batch in batches:
                for c in batch:
                    yield c
        finally:
            await batches.aclose()


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


@pytest.mark.parametrize("top_k", ["5", 5.0])
def test_chat_extra_numeric_knob_is_coerced(client_and_stub, top_k):
    from mstar.utils.generation_kwargs import normalize_generation_kwargs

    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b"ok")]
    submit = stub.submit_request
    # the real submit_request normalizes; the stub only records
    stub.submit_request = lambda **kw: submit(**{**kw, "model_kwargs": normalize_generation_kwargs(kw["model_kwargs"])})
    r = client.post(
        "/v1/chat/completions",
        json={"model": "bagel", "messages": [{"role": "user", "content": "hi"}], "top_k": top_k},
    )
    assert r.status_code == 200
    got = stub.last_submit["model_kwargs"]["top_k"]
    assert got == 5 and type(got) is int


def test_chat_rejects_malformed_data_url_as_bad_request(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "bagel",
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}}],
            }],
        },
    )
    assert r.status_code == 400
    # a refused input is the client's error, on every route
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_chat_refuses_server_paths_uniformly(client_and_stub, tmp_path):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    loop = tmp_path / "loop.png"
    loop.symlink_to(loop)
    for ref in (str(loop), str(tmp_path / "missing.png"), "/etc/hostname"):
        r = client.post("/v1/chat/completions", json={"model": "bagel", "messages": [{
            "role": "user", "content": [{"type": "image_url", "image_url": {"url": ref}}]}]})
        assert r.status_code == 400, (ref, r.text)
        message = r.json()["error"]["message"]
        assert message.startswith("server file paths are refused") and str(tmp_path) not in message
    assert stub.last_submit is None


@pytest.mark.parametrize("model", ["qwen3_tts_base", "omnivoice"])
def test_voice_clone_refuses_server_paths_uniformly(client_and_stub, model):
    client, stub = client_and_stub
    stub.model_name = model
    # the suffix-less ones are valid base64; "//..." even decodes to an MPEG frame sync
    for ref in ("/etc/passwd", "//etc/passwd", "//etc/shadow", "secret", "///x", "/etc/hostname.wav"):
        r = client.post("/v1/audio/speech", json={"model": model, "input": "hi", "ref_audio": ref, "ref_text": "hi"})
        assert r.status_code == 400, (ref, r.text)
        assert r.json()["error"]["message"].startswith("server file paths are refused")
    assert stub.last_submit is None and not any(stub.upload_dir.iterdir())


def _require_torchcodec_decode():
    """Skip unless torchcodec can decode audio here. Newer torchcodec imports
    without FFmpeg and raises only on the first decode, which the server reports
    as an undecodable clip."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(bytes(3200))
    try:
        from torchcodec.decoders import AudioDecoder

        AudioDecoder(buf.getvalue()).get_all_samples()
    except Exception as exc:  # noqa: BLE001 — RuntimeError at import or decode without FFmpeg
        pytest.skip(f"torchcodec cannot decode here: {exc}")


@pytest.mark.parametrize("model", ["qwen3_tts_base", "omnivoice"])
def test_voice_clone_refuses_a_1_hz_data_url_with_400(client_and_stub, model):
    if model == "omnivoice":
        _require_torchcodec_decode()
    else:
        pytest.importorskip("soundfile")
    client, stub = client_and_stub
    stub.model_name = model
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(1)
        wf.writeframes(bytes(64000))
    ref = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    r = client.post("/v1/audio/speech", json={"model": model, "input": "hi", "ref_audio": ref, "ref_text": "hi"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["message"].startswith("ref_audio must be audio the model can decode")
    assert stub.last_submit is None and not any(stub.upload_dir.iterdir())


@pytest.mark.parametrize("form", ["base64", "data URL"])
@pytest.mark.parametrize("model", ["qwen3_tts_base", "omnivoice"])
def test_voice_clone_refuses_a_61_s_clip_with_400(client_and_stub, model, form):
    if model == "omnivoice":
        _require_torchcodec_decode()
    else:
        pytest.importorskip("soundfile")
    client, stub = client_and_stub
    stub.model_name = model
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(bytes(2 * 8000 * 61))
    ref = base64.b64encode(buf.getvalue()).decode()
    if form == "data URL":
        ref = "data:audio/wav;base64," + ref
    r = client.post("/v1/audio/speech", json={"model": model, "input": "hi", "ref_audio": ref, "ref_text": "hi"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["message"].startswith("ref_audio must be 60 s or shorter")
    assert stub.last_submit is None and not any(stub.upload_dir.iterdir())


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


@pytest.mark.parametrize("exc,status", [
    (ValueError("ref_audio must be a data URL or an http(s) URL"), 400),  # the request's fault
    (OSError("No space left on device"), 500),  # the server's
])
def test_speech_maps_adapter_errors(client_and_stub, monkeypatch, exc, status):
    from mstar.api_server.openai import adapters

    def raising(self, req, upload_dir):
        raise exc

    client, stub = client_and_stub
    stub.model_name = "orpheus"
    monkeypatch.setattr(adapters.OrpheusAdapter, "speech_to_request", raising)
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": "hi", "voice": "tara"})
    assert r.status_code == status
    # a refusal keeps its message; the server's own error shows only an error id
    message = r.json()["error"]["message"]
    assert (str(exc) in message) == (status == 400)
    assert stub.last_submit is None


@pytest.mark.parametrize("seed,status", [(2**63 - 1, 200), (-(2**63), 200), (2**63, 422), (-(2**63) - 1, 422)])
def test_speech_seed_is_an_int64(client_and_stub, seed, status):
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.next_chunks = [_Chunk("audio", _pcm([100, -100]), {"sample_rate": 24000})]
    r = client.post("/v1/audio/speech", json={"model": "orpheus", "input": "hi", "voice": "tara", "seed": seed})
    assert r.status_code == status


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
    assert stub.last_raw_request is not None


def test_videos_generations_wan22(client_and_stub):
    # Route-level smoke for the wan22 adapter: the mp4 chunk round-trips as
    # b64_json and the first-class video fields (size -> width/height,
    # num_frames, fps) land in model_kwargs alongside extra_body passthrough.
    client, stub = client_and_stub
    stub.model_name = "wan22"
    mp4 = b"\x00\x00\x00 ftypisommp4fake"
    stub.next_chunks = [_Chunk("video", mp4)]
    body = client.post(
        "/v1/videos/generations",
        json={
            "model": "wan22", "prompt": "a cat surfing", "size": "832x480",
            "num_frames": 33, "fps": 16, "seed": 7, "num_inference_steps": 20,
        },
    ).json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == mp4
    assert stub.last_submit["output_modalities"] == ["video"]
    assert stub.last_submit["input_modalities"] == ["text"]
    mk = stub.last_submit["model_kwargs"]
    assert (mk["width"], mk["height"]) == (832, 480)
    assert mk["num_frames"] == 33 and mk["fps"] == 16 and mk["seed"] == 7
    assert mk["num_inference_steps"] == 20  # extra_body passthrough


def test_videos_generations_wan22_rejects_video_conditioning(client_and_stub):
    # wan22 has no video-conditioned mode; the adapter's ValueError must
    # surface as a 400, not a silent text-to-video fallback.
    client, stub = client_and_stub
    stub.model_name = "wan22"
    r = client.post(
        "/v1/videos/generations",
        json={"model": "wan22", "prompt": "x", "video": "data:video/mp4;base64,AAAA"},
    )
    assert r.status_code == 400
    assert "video" in r.json()["error"]["message"]


def test_videos_generations_ltx2_5_requests_audio_and_muxes(client_and_stub, monkeypatch):
    # LTX-2.5 generates audio with the video: the route asks the model for both and
    # muxes the PCM chunk into the mp4 with its sample rate and channel count.
    from mstar.api_server import media_io

    client, stub = client_and_stub
    stub.model_name = "ltx2_5"
    mp4, pcm = b"\x00\x00\x00 ftypisommp4fake", b"\x01\x00\x02\x00"
    stub.next_chunks = [_Chunk("video", mp4), _Chunk("audio", pcm, {"sample_rate": 48000, "num_channels": 2})]
    muxed = {}

    def fake_mux(video, audio, sample_rate, num_channels, video_fps=None):
        muxed.update(video=video, audio=audio, sample_rate=sample_rate, num_channels=num_channels)
        return b"muxed"

    monkeypatch.setattr(media_io, "mux_mp4_with_pcm16", fake_mux)
    body = client.post(
        "/v1/videos/generations",
        json={"model": "ltx2_5", "prompt": "a dog barking", "size": "960x544", "num_frames": 121,
              "seed": 3, "recipe": "two_stage"},
    ).json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == b"muxed"
    assert muxed == {"video": mp4, "audio": pcm, "sample_rate": 48000, "num_channels": 2}
    assert stub.last_submit["output_modalities"] == ["video", "audio"]
    mk = stub.last_submit["model_kwargs"]
    assert (mk["width"], mk["height"], mk["num_frames"], mk["seed"]) == (960, 544, 121, 3)
    assert mk["recipe"] == "two_stage" and "audio" not in mk


def test_videos_generations_ltx2_5_video_only_and_rejects_conditioning(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "ltx2_5"
    stub.next_chunks = [_Chunk("video", b"mp4")]
    client.post("/v1/videos/generations", json={"model": "ltx2_5", "prompt": "x", "audio": False})
    assert stub.last_submit["output_modalities"] == ["video"]
    r = client.post(
        "/v1/videos/generations",
        json={"model": "ltx2_5", "prompt": "x", "image": "data:image/png;base64,AAAA"},
    )
    assert r.status_code == 400 and "text-to-video" in r.json()["error"]["message"]


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


def test_videos_stream_ndjson(client_and_stub):
    import base64 as _b64
    import json as _json

    client, stub = client_and_stub
    stub.model_name = "cosmos3"
    stub.next_chunks = [_Chunk("video", b"mp4-w0"), _Chunk("video", b"mp4-w1")]
    r = client.post(
        "/v1/videos/generations",
        json={
            "model": "cosmos3", "prompt": "a road", "num_frames": 57,
            "window_mode": "chained", "stream_video": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert stub.last_submit["streaming"] is True
    assert stub.last_submit["model_kwargs"]["stream_video"] is True
    lines = [_json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert [ln["modality"] for ln in lines] == ["video", "video", "done"]
    assert [_b64.b64decode(ln["data"]) for ln in lines[:2]] == [b"mp4-w0", b"mp4-w1"]
    assert lines[2]["metadata"]["chunks"] == 2

    # Without the flag the endpoint returns the grouped JSON body as before.
    stub.next_chunks = [_Chunk("video", b"mp4-full")]
    body = client.post(
        "/v1/videos/generations",
        json={"model": "cosmos3", "prompt": "a road", "num_frames": 57},
    ).json()
    assert stub.last_submit["streaming"] is False
    assert _b64.b64decode(body["data"][0]["b64_json"]) == b"mp4-full"


def test_chat_stream_sends_a_batch_of_tokens_as_one_frame():
    """Tokens that were waiting together leave as one body frame (one write
    on the socket), still one SSE event per token for the client; a chunk
    of another modality gets a frame of its own."""
    from mstar.api_server.openai import reply
    from mstar.api_server.openai.serving_chat import _stream_events

    class _Batches:
        def submission_key(self, request_id):
            return request_id

        async def iter_result_chunk_batches(self, request_id):
            yield [_Chunk("text", b"a"), _Chunk("text", b"b"), _Chunk("text", b"c")]
            yield [_Chunk("text", b"d"), _Chunk("audio", b"\x00\x01"), _Chunk("text", b"e")]

    async def _collect():
        events = reply.reply_events(_Batches(), "rid")
        return [frame async for frame in _stream_events(events, "bagel", "rid")]

    frames = asyncio.run(_collect())
    # abc (the role rides on a), d, audio, e, stop, DONE
    assert [f.count("data: ") for f in frames] == [3, 1, 1, 1, 1, 1]
    events = [json.loads(l[6:]) for f in frames[:-1] for l in f.splitlines() if l.startswith("data: ")]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": "a"}
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == "abcde"
    assert events[4]["choices"][0]["delta"]["audio"]["data"] == base64.b64encode(b"\x00\x01").decode()
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "data: [DONE]\n\n"


def test_sse_event_is_one_json_line():
    from mstar.api_server.openai._util import sse

    obj = {"choices": [{"delta": {"content": "h\u00e9llo \"q\""}}], "n": 1}
    line = sse(obj)
    assert line.startswith("data: ") and line.endswith("\n\n")
    assert "\n" not in line[:-2]
    assert json.loads(line[6:-2]) == obj


def test_chat_stream_reports_a_failed_request_in_band(client_and_stub):
    """A request that fails after the stream opened ends with an error event,
    not a ``finish_reason: stop`` that reads as a complete answer."""
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [
        _Chunk("text", b"Paris"),
        _Chunk("error", b"Error in worker: ValueError: q implies q_len_per_req=5", {"status": 500}),
    ]
    text = client.post(
        "/v1/chat/completions",
        json={"model": "bagel", "messages": [{"role": "user", "content": "go"}], "stream": True},
    ).text
    events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Paris"}
    assert events[-1]["error"] == {
        "message": "Error in worker: ValueError: q implies q_len_per_req=5", "type": "server_error", "code": 500,
    }
    assert not any(e.get("choices", [{}])[0].get("finish_reason") for e in events)
    assert text.rstrip().endswith("data: [DONE]")
    # The request is already gone by the time its error chunk arrives; the
    # stream must not abort it on the way out.
    assert stub.aborted == []


def _stream_events(client, model="bagel"):
    text = client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "go"}], "stream": True},
    ).text
    events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    return text, events


def test_a_chat_stream_refused_up_front_gets_its_status(client_and_stub):
    """A 4xx before the first token is the response's status, as on ``/generate``."""
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("error", b"ValueError: unknown domain 'x'", {"status": 400})]
    r = client.post(
        "/v1/chat/completions",
        json={"model": "bagel", "messages": [{"role": "user", "content": "go"}], "stream": True},
    )
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "ValueError: unknown domain 'x'"
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert stub.aborted == []


def test_chat_stream_reports_a_timeout_in_band(client_and_stub):
    """The delivery timeout raises an HTTPException out of the chunk iterator
    mid-stream; the client gets an error event and [DONE], not a cut
    connection, and the iterator's own abort still runs."""
    from fastapi import HTTPException

    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b"Par")]
    stub.raise_after = HTTPException(status_code=500, detail="Request timed out")
    text, events = _stream_events(client)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Par"}
    assert events[-1]["error"] == {"message": "Request timed out", "type": "server_error", "code": 500}
    assert not any(e.get("choices", [{}])[0].get("finish_reason") for e in events)
    assert text.rstrip().endswith("data: [DONE]")
    assert stub.aborted == [stub.last_submit["request_id"]]


def test_chat_stream_sends_role_with_first_token(client_and_stub):
    """Clients time TTFT at the first chunk, so none may precede the first token."""
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b"strea"), _Chunk("text", b"ming")]
    _, lines = _stream_events(client)
    assert lines[0]["choices"][0]["delta"] == {"role": "assistant", "content": "strea"}
    assert len(lines) == 3


def test_chat_joins_a_character_split_across_tokens(client_and_stub):
    """Byte-level BPE models emit each token's raw bytes; "é√" arrives in 4 chunks."""
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b) for b in (b"\xc3", b"\xa9\xe2", b"\x88", b"\x9a!")]
    body = client.post(
        "/v1/chat/completions",
        json={"model": "bagel", "messages": [{"role": "user", "content": "hi"}]},
    ).json()
    assert body["choices"][0]["message"]["content"] == "é√!"
    _, lines = _stream_events(client)
    assert lines[0]["choices"][0]["delta"] == {"role": "assistant", "content": "é"}
    assert "".join(l["choices"][0]["delta"].get("content", "") for l in lines) == "é√!"


def test_chat_stream_flushes_a_truncated_character(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("text", b"a"), _Chunk("text", b"\xe2\x88")]
    _, lines = _stream_events(client)
    assert lines[-1]["choices"][0] == {"index": 0, "delta": {"content": "�"}, "finish_reason": "stop"}


def test_chat_stream_without_output_still_names_the_role(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = []
    _, lines = _stream_events(client)
    assert lines == [lines[0]] and lines[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert lines[0]["choices"][0]["finish_reason"] == "stop"


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


def test_images_edits_forwards_size(client_and_stub):
    """``size`` is a known field of the multipart route but still has to reach the adapter: an edit at
    768x1024 used to come back at the reference's size because the field was dropped."""
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.next_chunks = [_Chunk("image", b"\x89PNGedited")]
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"\x89PNGinput", "image/png")},
        data={"prompt": "make it neon", "size": "768x1024"},
    )
    assert resp.status_code == 200
    # BAGEL reads width / height, so the adapter maps "WxH" onto them
    mk = stub.last_submit["model_kwargs"]
    assert (mk.get("width"), mk.get("height"), "size" in mk) == (768, 1024, False)


def test_image_routes_report_validation_errors_as_400(client_and_stub):
    """A ValueError raised while building or validating the request is the client's fault (400, not 500)."""
    client, stub = client_and_stub
    stub.model_name = "bagel"

    def reject(**kw):
        raise ValueError("size must be 'WxH'")

    stub.submit_request = reject
    resp = client.post("/v1/images/generations", json={"model": "bagel", "prompt": "a cat", "size": "big"})
    assert resp.status_code == 400 and "WxH" in resp.json()["error"]["message"]
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"\x89PNGinput", "image/png")},
        data={"prompt": "make it neon", "size": "big"},
    )
    assert resp.status_code == 400

def test_images_n_below_one_is_400(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    stub.submit_request = lambda **kw: pytest.fail("a refused n must not be submitted")
    resp = client.post("/v1/images/generations", json={"model": "bagel", "prompt": "a cat", "n": 0})
    message = resp.json()["error"]["message"]
    assert resp.status_code == 400 and message.startswith("n must be between 1 and") and "got 0" in message

@pytest.mark.parametrize("model", ["cosmos3", "cosmos3_edge"])
def test_an_edit_the_model_cannot_do_is_a_400_and_leaves_no_upload(client_and_stub, model):
    """A model with image generation but no editing refuses the edit as the
    Rust frontend does, and the saved image goes with the refusal."""
    client, stub = client_and_stub
    stub.model_name = model
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"\x89PNGinput", "image/png")},
        data={"prompt": "make it neon"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "image editing is not supported by this model"
    assert stub.last_submit is None and not any(stub.upload_dir.iterdir())


def _refuse(**kw):
    raise ValueError("size must be 'WxH'")


@pytest.mark.parametrize("model,path,body", [
    ("bagel", "/v1/chat/completions", {"json": {"messages": [{"role": "tool", "content": "sunny"}]}}),
    ("orpheus", "/v1/audio/speech", {"json": {"input": "hi", "response_format": "xyz"}}),
    ("whisper_large", "/v1/audio/transcriptions",
     {"data": {"long_form": "xyz"}, "files": {"file": ("a.wav", b"RIFF", "audio/wav")}}),
    ("bagel", "/v1/images/generations", {"json": {"prompt": "a cat"}}),
    ("bagel", "/v1/images/edits",
     {"data": {"prompt": "make it neon"}, "files": {"image": ("in.png", b"\x89PNGinput", "image/png")}}),
    ("wan22", "/v1/videos/generations", {"json": {"prompt": "x", "video": "data:video/mp4;base64,AAAA"}}),
], ids=["chat", "speech", "transcriptions", "images", "edits", "videos"])
def test_every_route_calls_a_400_an_invalid_request_error(client_and_stub, model, path, body):
    """OpenAI types a 400 ``invalid_request_error``: the request is at fault, and sending it again gets the same
    400. ``server_error`` says the opposite."""
    client, stub = client_and_stub
    stub.model_name = model
    stub.submit_request = _refuse  # bagel's image adapters refuse no field, so its routes refuse at submit
    r = client.post(path, **body)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error", "a 400 reads as the server's fault"


def test_audio_voices_lists_model_voices(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.model.get_voices = lambda: ["tara", "zoe"]
    stub.model.get_default_voice = lambda: "tara"
    body = client.get("/v1/audio/voices").json()
    assert body["object"] == "list" and body["default_voice"] == "tara"
    assert body["voices"] == [{"id": "tara", "name": "tara"}, {"id": "zoe", "name": "zoe"}]


def test_audio_voices_404_without_a_voice_list(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.model.get_voices = lambda: None
    stub.model.get_default_voice = lambda: None
    r = client.get("/v1/audio/voices")
    assert r.status_code == 404 and "voice list" in r.json()["error"]["message"]


def test_audio_voices_404_for_non_speech_model(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    assert client.get("/v1/audio/voices").status_code == 404


@pytest.mark.parametrize("body", [
    {"response_format": "xyz"},
    {"response_format": "aac"},
    {"response_format": "mp3", "stream": True},
    {"response_format": "opus", "stream": True},
])
def test_speech_refuses_a_format_it_cannot_produce(client_and_stub, body):
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.next_chunks = [_Chunk("audio", _pcm([1, 2, 3]))]
    r = client.post("/v1/audio/speech", json={"input": "hi", **body})
    assert r.status_code == 400 and "response_format" in r.json()["error"]["message"]
    assert stub.last_submit is None  # refused before any work was submitted


@pytest.mark.parametrize("ref,message", [
    ("/etc/passwd", "server file paths are refused"),
    ("file:///etc/passwd", "server file paths are refused"),
    (123, "got int"),
    ({"url": "x"}, "got dict"),
])
def test_chatterbox_adapter_refuses_a_bad_ref_audio(ref, message, tmp_path):
    # refused as a client error (ValueError) before anything is opened
    from mstar.api_server.openai import adapters
    from mstar.api_server.openai.protocol import SpeechRequest

    req = SpeechRequest.model_validate({"model": "chatterbox", "input": "hi", "ref_audio": ref})
    with pytest.raises(ValueError, match=message):
        adapters.ChatterboxAdapter().speech_to_request(req, tmp_path)


@pytest.mark.parametrize("fmt,magic", [("mp3", None), ("flac", b"fLaC"), ("opus", b"OggS")])
def test_speech_encodes_the_requested_container(client_and_stub, fmt, magic):
    pytest.importorskip("soundfile")
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.next_chunks = [_Chunk("audio", _pcm(list(range(0, 24000, 7))))]
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": fmt})
    assert r.status_code == 200 and r.content[:4] != b"RIFF"  # not WAV under another name
    if magic:
        assert r.content[:4] == magic


def test_speech_stream_pcm_has_no_wav_header(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "orpheus"
    stub.next_chunks = [_Chunk("audio", _pcm([1, 2, 3]))]
    r = client.post("/v1/audio/speech", json={"input": "hi", "stream": True, "response_format": "pcm"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/pcm")
    assert r.content == _pcm([1, 2, 3])
    r = client.post("/v1/audio/speech", json={"input": "hi", "stream": True})
    assert r.headers["content-type"].startswith("audio/wav") and r.content[:4] == b"RIFF"
    assert r.content[44:] == _pcm([1, 2, 3])


_GLM_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}}}}]
_GLM_REPLY = [b"Need the ", b"weather.</th", b"ink>\n<tool_call>get_weather<arg_key>city",
              b"</arg_key><arg_value>Paris</arg_value></tool_call>"]


def test_glm_chat_returns_reasoning_and_tool_calls(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.next_chunks = [_Chunk("text", b) for b in _GLM_REPLY]
    choice = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "weather?"}], "tools": _GLM_TOOLS,
    }).json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None and message["reasoning_content"] == "Need the weather."
    assert message["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert stub.last_submit["model_kwargs"]["tools"] == _GLM_TOOLS


def test_glm_chat_streams_reasoning_then_the_call(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.next_chunks = [_Chunk("text", b) for b in _GLM_REPLY]
    text = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "weather?"}],
        "tools": _GLM_TOOLS, "stream": True,
    }).text
    events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    deltas = [e["choices"][0]["delta"] for e in events]
    assert deltas[0] == {"role": "assistant", "reasoning_content": "Need the "}
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "Need the weather."
    assert not any(d.get("content") for d in deltas)
    first, *rest = [c for d in deltas for c in d.get("tool_calls", [])]
    assert first["index"] == 0 and first["function"]["name"] == "get_weather" and first["id"].startswith("call_")
    # the arguments stream after the name
    assert first["function"]["arguments"] + "".join(c["function"]["arguments"] for c in rest) == '{"city": "Paris"}'
    assert all(c["index"] == 0 and "id" not in c for c in rest)
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_glm_chat_forces_a_required_tool_call(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    # the prompt ended in <tool_call>get_weather: the reply starts at the arguments
    stub.next_chunks = [_Chunk("text", b"<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>")]
    choice = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "hi"}],
        "tools": _GLM_TOOLS, "tool_choice": "required",
    }).json()["choices"][0]
    assert stub.last_submit["model_kwargs"]["reply_prefix"] == "<tool_call>get_weather"
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}


def _usage_chunks(*parts, stop=True):
    """Text chunks as the data worker tags them: per-chunk token counts and the prompt's."""
    return [_Chunk("text", text, {"tokens": n, "prompt_tokens": 11, "stop_token": stop and i == len(parts) - 1})
            for i, (text, n) in enumerate(parts)]


@pytest.mark.parametrize("stop, finish", [(True, "stop"), (False, "length")])
def test_chat_reports_usage_and_why_it_stopped(client_and_stub, stop, finish):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.next_chunks = _usage_chunks((b"Plan</think>", 2), (b"Hello", 1), stop=stop)
    body = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
    assert body["choices"][0]["finish_reason"] == finish


def test_chat_stream_sends_usage_when_asked(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.next_chunks = _usage_chunks((b"Plan</think>", 2), (b"Hello", 1), stop=False)
    text = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        "stream_options": {"include_usage": True},
    }).text
    events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    assert events[-2]["choices"][0]["finish_reason"] == "length"
    assert events[-1]["choices"] == [] and events[-1]["usage"]["completion_tokens"] == 3


def test_chat_refuses_bad_stream_options_before_running(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.last_submit = None
    r = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "hi"}], "stream": True, "stream_options": True})
    assert r.status_code == 400 and stub.last_submit is None


def test_a_parsed_reply_stops_when_its_client_disconnects():
    """Non-streaming, a reply is read chunk by chunk; a client that goes away
    unwinds the chunk iterator, which aborts the request."""
    from fastapi import HTTPException

    from mstar.api_server.openai import reply

    aborted = []

    class _Endless:
        def submission_key(self, request_id):
            return request_id

        async def iter_result_chunk_batches(self, request_id):
            try:
                while True:
                    yield [_Chunk("text", b"x", {"tokens": 1})]
                    await asyncio.sleep(0.01)
            finally:
                aborted.append(request_id)

    class _Gone:
        calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 2

    async def _collect():
        return await reply.collect(reply.reply_events(_Endless(), "r1"), _Gone())

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_collect())
    assert exc.value.status_code == 499 and aborted == ["r1"]


def _glm_events(text):
    return [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]


@pytest.mark.parametrize("stream", [False, True])
def test_glm_chat_ends_at_a_stop_string_and_stops_the_request(client_and_stub, stream):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    # the stop spans two tokens; what follows it is never read
    stub.next_chunks = _usage_chunks((b"Plan. STOP</think>", 3), (b"Done: 4", 2), (b"2. ST", 2), (b"OP more", 2),
                                     (b" text", 1), stop=False)
    r = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "hi"}], "stop": ["STOP"], "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
    })
    if stream:
        events = _glm_events(r.text)
        deltas = [e["choices"][0]["delta"] for e in events if e["choices"]]
        content = "".join(d.get("content", "") for d in deltas)
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        finish = events[-2]["choices"][0]["finish_reason"]
        usage = events[-1]["usage"]
    else:
        choice = r.json()["choices"][0]
        content, reasoning = choice["message"]["content"], choice["message"]["reasoning_content"]
        finish, usage = choice["finish_reason"], r.json()["usage"]
    # stops match the answer only: the reasoning's STOP went out
    assert reasoning == "Plan. STOP" and content == "Done: 42. " and finish == "stop"
    # the tokens read through the one that completed the stop
    assert usage["completion_tokens"] == 9
    assert stub.aborted == [stub.last_submit["request_id"]]
    assert stub.ended == {stub.last_submit["request_id"]: "success"}  # a whole reply, served


def test_a_stop_string_does_not_cut_a_tool_call(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.next_chunks = [_Chunk("text", b) for b in _GLM_REPLY]
    body = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "weather?"}], "tools": _GLM_TOOLS,
        "stop": ["Paris", "weather"],
    }).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["reasoning_content"] == "Need the weather."
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert stub.aborted == []


@pytest.mark.parametrize("stream", [False, True])
def test_glm_chat_without_parallel_calls_ends_at_the_first(client_and_stub, stream):
    client, stub = client_and_stub
    stub.model_name = "glm52"
    call = b"<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
    stub.next_chunks = _usage_chunks((b"Two.</think>", 3), (call, 9), (b"\n" + call, 9), stop=False)
    r = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "weather?"}], "tools": _GLM_TOOLS,
        "parallel_tool_calls": False, "stream": stream,
    })
    if stream:
        events = _glm_events(r.text)
        calls = [c for e in events for c in e["choices"][0]["delta"].get("tool_calls", [])]
        finish = events[-1]["choices"][0]["finish_reason"]
    else:
        calls, finish = r.json()["choices"][0]["message"]["tool_calls"], r.json()["choices"][0]["finish_reason"]
    assert [c["function"]["arguments"] for c in calls] == ['{"city": "Paris"}'] and finish == "tool_calls"
    assert "parallel_tool_calls" not in stub.last_submit["model_kwargs"]
    assert stub.aborted == [stub.last_submit["request_id"]]
    assert stub.ended == {stub.last_submit["request_id"]: "success"}  # a whole reply, served


def test_a_whitespace_stop_is_seen_when_text_follows_it(client_and_stub):
    """The answer's trailing whitespace is held until text follows it, so a
    stop that ends in whitespace ends the reply one chunk later, and that
    chunk's tokens count."""
    client, stub = client_and_stub
    stub.model_name = "glm52"
    stub.next_chunks = _usage_chunks((b"Plan</think>", 2), (b"Line one.", 3), (b"\n\n", 1), (b"Line", 1),
                                     (b" two", 1), stop=False)
    body = client.post("/v1/chat/completions", json={
        "model": "glm52", "messages": [{"role": "user", "content": "hi"}], "stop": "\n\n"}).json()
    assert body["choices"][0]["message"]["content"] == "Line one."
    assert body["choices"][0]["finish_reason"] == "stop" and body["usage"]["completion_tokens"] == 7


_PNG_URL = "data:image/png;base64," + base64.b64encode(
    base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
).decode()


def _wav_b64(seconds=1, rate=16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(2 * rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def no_cleanup_delay(monkeypatch):
    from mstar.api_server import media_io

    monkeypatch.setattr(media_io, "UPLOAD_CLEANUP_DELAY_S", 0.0, raising=False)


def _left_after_cleanup(directory: Path, timeout_s: float = 5.0) -> list[str]:
    """The upload dir's files once the deferred cleanup has had its chance."""
    deadline = time.monotonic() + timeout_s
    while any(directory.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    return sorted(p.name for p in directory.iterdir())


_CHAT = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": _PNG_URL}},
    {"type": "input_audio", "input_audio": {"data": _wav_b64(), "format": "wav"}},
    {"type": "text", "text": "what is this?"},
]}]}


@pytest.mark.parametrize("model,path,body,chunks", [
    ("qwen3_omni", "/v1/chat/completions", {"json": _CHAT}, [_Chunk("text", b"a cat")]),
    ("qwen3_omni", "/v1/chat/completions", {"json": {**_CHAT, "stream": True}}, [_Chunk("text", b"a cat")]),
    ("bagel", "/v1/images/edits",
     {"data": {"prompt": "make it neon"}, "files": {"image": ("in.png", b"\x89PNGinput", "image/png")}},
     [_Chunk("image", b"\x89PNGout")]),
    ("wan22", "/v1/videos/generations", {"json": {"prompt": "a cat", "image": _PNG_URL}},
     [_Chunk("video", b"mp4")]),
    ("qwen3_tts_base", "/v1/audio/speech",
     {"json": {"input": "hi", "ref_text": "hi", "ref_audio": "data:audio/wav;base64," + _wav_b64()}},
     [_Chunk("audio", _pcm([1, 2, 3]))]),
    ("qwen3_tts_base", "/v1/audio/speech",
     {"json": {"input": "hi", "ref_text": "hi", "ref_audio": _wav_b64(), "stream": True}},
     [_Chunk("audio", _pcm([1, 2, 3]))]),
], ids=["chat", "chat-stream", "edits", "videos", "speech-clip", "speech-clip-stream"])
def test_a_served_request_leaves_no_upload_behind(client_and_stub, no_cleanup_delay, model, path, body, chunks):
    """What a request saved is deleted after it is served, as /generate and the
    Rust frontend do, and was on disk when it was submitted."""
    if model == "qwen3_tts_base":
        pytest.importorskip("soundfile")
    client, stub = client_and_stub
    stub.model_name = model
    stub.next_chunks = chunks

    r = client.post(path, **body)

    assert r.status_code == 200, r.text
    saved = [p for ps in stub.last_submit["file_paths"].values() for p in ps]
    assert saved and all(Path(p).parent == stub.upload_dir for p in saved)
    assert _left_after_cleanup(stub.upload_dir) == []


@pytest.mark.parametrize("model,path,body", [
    ("qwen3_omni", "/v1/chat/completions", {"json": _CHAT}),
    ("bagel", "/v1/images/edits",
     {"data": {"prompt": "make it neon"}, "files": {"image": ("in.png", b"\x89PNGinput", "image/png")}}),
], ids=["chat", "edits"])
def test_a_submission_the_server_refuses_leaves_no_upload_behind(
    client_and_stub, no_cleanup_delay, model, path, body,
):
    from fastapi import HTTPException

    client, stub = client_and_stub
    stub.model_name = model

    def unavailable(**kw):
        raise HTTPException(status_code=503, detail="shutting down")

    stub.submit_request = unavailable
    r = client.post(path, **body)

    assert r.status_code == 503
    assert _left_after_cleanup(stub.upload_dir) == []


_ATTACHMENTS = _CHAT["messages"][0]["content"][:2]


@pytest.mark.parametrize("model,body,message", [
    ("bagel", {"messages": [{"role": "user", "content": [
        *_ATTACHMENTS, {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%"}},
    ]}]}, "Only base64 data is allowed"),
    ("qwen3_omni", {"messages": [{"role": "user", "content": _ATTACHMENTS}, {"role": "tool", "content": "42"}]},
     "a message's role must be system, developer, user or assistant, not 'tool'"),
    ("cosmos3_edge", {"messages": [{"role": "user", "content": _ATTACHMENTS}], "chat_template_kwargs": True},
     "argument of type 'bool' is not iterable"),
], ids=["bad-attachment", "bad-role", "bad-template-kwargs"])
def test_a_chat_refused_after_its_attachments_leaves_no_upload_behind(client_and_stub, model, body, message):
    """A chat refused once some attachments are saved is never submitted, so
    nothing else would delete them."""
    client, stub = client_and_stub
    stub.model_name = model

    r = client.post("/v1/chat/completions", json=body)

    assert r.status_code == 400 and r.json()["error"]["message"].startswith(message), r.text
    assert stub.last_submit is None and not any(stub.upload_dir.iterdir())


def test_a_chunked_speech_stream_keeps_its_clip_until_the_last_chunk_is_submitted(client_and_stub, no_cleanup_delay):
    """Every sentence chunk reads the reference clip, and a stream submits the
    later ones while it plays: the clip stays until the stream ends."""
    pytest.importorskip("soundfile")
    client, stub = client_and_stub
    stub.model_name = "qwen3_tts_base"
    stub.next_chunks = [_Chunk("audio", _pcm([1, 2, 3]))]
    on_disk_at_submit = []
    submit, batches = stub.submit_request, stub.iter_result_chunk_batches

    def recording_submit(**kw):
        on_disk_at_submit.append(all(Path(p).exists() for ps in kw["file_paths"].values() for p in ps))
        return submit(**kw)

    async def slow_batches(request_id):
        await asyncio.sleep(0.05)  # room for a cleanup scheduled too early to run
        async for batch in batches(request_id):
            yield batch

    stub.submit_request, stub.iter_result_chunk_batches = recording_submit, slow_batches
    text = " ".join(f"Sentence number {i} is long enough to count toward the chunking threshold." for i in range(30))

    r = client.post("/v1/audio/speech", json={
        "input": text, "ref_text": "hi", "ref_audio": _wav_b64(), "stream": True, "response_format": "pcm",
    })

    assert r.status_code == 200 and r.content == _pcm([1, 2, 3]) * len(on_disk_at_submit)
    assert len(on_disk_at_submit) > 2 and all(on_disk_at_submit)
    assert _left_after_cleanup(stub.upload_dir) == []
