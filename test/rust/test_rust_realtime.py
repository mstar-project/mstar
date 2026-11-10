"""``/v1/realtime`` on mstar-server against the Python route.

Each session sends the same client events to both frontends, served by one
scripted ASR (it continues from the assistant prefix as Qwen3-ASR does), and
compares every server event (ids and timestamps aside) and every step the
engine got: its kwargs and the audio it heard, sample for sample. Turn
detection runs the real Silero VAD on both, through the same ONNX Runtime.
"""

import base64
import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")
pytest.importorskip("websockets")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

import websockets.exceptions
import websockets.sync.client
from fastapi.testclient import TestClient
from test_rust_frontend import BINARY, _Chunk, _free_port, _wait_healthy

from mstar.api_server import entrypoint, media_io
from mstar.api_server.openai import realtime_vad
from mstar.api_server.openai.adapters import ADAPTER_REGISTRY
from mstar.api_server.rust_frontend import (
    OPENAI_ADAPTER_ENV,
    REALTIME_ENV,
    TRANSCRIPTION_ENV,
    RustFrontendBridge,
    openai_adapter_class,
    realtime_facts,
    transcription_facts,
)

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")
needs_vad = pytest.mark.skipif(realtime_vad.silero_factory() is None, reason="Silero VAD is not installed")
HAS_SMART_TURN = realtime_vad.silero_factory() is not None and realtime_vad.smart_turn() is not None

CLIP = Path(__file__).parents[1] / "qwen3-omni" / "audio.wav"


def _speech_24k() -> np.ndarray:
    audio, rate = sf.read(CLIP, dtype="float32")
    assert rate == 24000
    return audio


def _speech_16k() -> np.ndarray:
    r = media_io.StreamResampler(24000, 16000)
    return np.concatenate([r.push(_speech_24k()), r.flush()])


def _pcm(audio: np.ndarray) -> str:
    pcm = np.clip(np.rint(audio.astype(np.float64) * 32768), -32768, 32767).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode()


def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-1.7B", local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached: partials hold nothing back on either side
        return None


class _Model:
    live_transcription_stream = None   # re-decoded, not heard as it comes

    def __init__(self):
        self.tokenizer = _tokenizer()

    def request_kwargs(self):
        return None


class _ASR:
    """The engine for both frontends: a step's text grows with the audio it heard
    and continues from its assistant prefix; language "xx" is a 400, as is a
    prompt "paths" (its error naming the upload and other paths), and a prompt
    "boom" a 500."""

    def __init__(self, upload_dir: Path):
        self.model_name = "qwen3_asr"
        # no ClientAudio stream: this model is transcribed, not conversed with
        self.client_audio = []
        self.model = _Model()
        self.upload_dir = upload_dir
        self.metrics = None
        self.fatal_error = None
        self.steps: list[dict] = []
        self._chunks: dict = {}

    def cleanup(self):
        pass

    def submit_request(self, *, text=None, file_paths=None, input_modalities, output_modalities, model_kwargs=None,
                       prompt_parts=None, streaming=True, request_id=None, model=None, **_):
        path = file_paths["audio"][0]
        data, rate = sf.read(path, dtype="int16")
        mk = dict(model_kwargs or {})
        self.steps.append({"mk": mk, "seconds": len(data) / rate, "audio": hashlib.sha1(data.tobytes()).hexdigest()})
        if mk.get("language") == "xx":
            chunks = [_Chunk("error", b"unsupported language 'xx' (see /srv/models/qwen/config.json)", {"status": 400})]
        elif mk.get("initial_prompt") == "paths":
            error = (f"cannot read {self.upload_dir}/rt-abc-1.wav "
                     "(not under /models/, nor /tmp, nor '/srv/my models/a b.json')")
            chunks = [_Chunk("error", error.encode(), {"status": 400})]
        elif mk.get("initial_prompt") == "boom":
            chunks = [_Chunk("error", b"CUDA error: device-side assert", {"status": 500})]
        else:
            head = "" if mk.get("language") else "language English<asr_text>"
            full = head + " ".join(f"wörd{i}" for i in range(int(len(data) / rate * 2)))
            prefix = mk.get("assistant_prefix") or ""
            out = full[len(prefix):] if full.startswith(prefix) else full
            chunks = [_Chunk("text", out.encode()[i:i + 7]) for i in range(0, len(out.encode()), 7)]
        self._chunks[request_id] = chunks
        return request_id

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks.pop(request_id, [])

    async def iter_result_chunks(self, request_id):
        for c in self._chunks.pop(request_id, []):
            yield c

    def release_request(self, request_id):
        pass

    def submission_key(self, request_id):
        return request_id

    def relay_chunks_to(self, key, sink, tag):
        pass

    def abort_submission(self, key):
        pass


@contextlib.contextmanager
def _rust(server: _ASR, **extra_env):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_rt_")
    adapter = ADAPTER_REGISTRY["qwen3_asr"]
    env = dict(os.environ, **{
        OPENAI_ADAPTER_ENV: openai_adapter_class("qwen3_asr"),
        TRANSCRIPTION_ENV: json.dumps(transcription_facts(server, adapter)),
        REALTIME_ENV: json.dumps(realtime_facts(server, adapter, bridge_dir)),
    }, **extra_env)
    proc = subprocess.Popen([BINARY, "qwen3_asr", str(port), bridge_dir, str(server.upload_dir)], env=env)
    bridge = RustFrontendBridge(server, bridge_dir)
    threading.Thread(target=bridge.run, daemon=True).start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def stack():
    server = _ASR(Path(tempfile.mkdtemp(prefix="mstar_rt_up_")))
    with _rust(server) as port:
        yield server, port


END = {"type": "mstar.end"}


def _normalized(events: list[dict]) -> list[dict]:
    """Ids and timestamps aside: items numbered in order of appearance, error ids as X."""
    import re

    items: dict = {}
    out = []
    for e in events:
        e = {k: v for k, v in e.items() if k not in ("event_id", "created")}
        if e.get("type") == "error":
            message = re.sub(r"error id [0-9a-f]{12}", "error id X", e["error"]["message"])
            e = {**e, "error": {**e["error"], "message": message}}
        if "item_id" in e:
            e["item_id"] = items.setdefault(e["item_id"], len(items))
        if "session" in e:
            e["session"] = {**e["session"], "id": "S"}
        out.append(e)
    return out


def _is_end(e: dict) -> bool:
    return e.get("type") == "error" and e["error"]["message"] == "unknown event 'mstar.end'"


def _python(server, frames):
    entrypoint.api_server = server
    events = []
    with TestClient(entrypoint.app).websocket_connect("/v1/realtime?intent=transcription") as ws:
        events.append(json.loads(ws.receive_text()))
        for f in [*frames, json.dumps(END)]:
            (ws.send_bytes if isinstance(f, bytes) else ws.send_text)(f)
        while not events or not _is_end(events[-1]):
            events.append(json.loads(ws.receive_text()))
    return events


def _rust_session(port, frames):
    events = []
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/v1/realtime?intent=transcription",
                                        compression=None, max_size=None) as ws:
        events.append(json.loads(ws.recv(timeout=30)))
        for f in [*frames, json.dumps(END)]:
            ws.send(f)
        while not _is_end(events[-1]):
            events.append(json.loads(ws.recv(timeout=60)))
    return events


def _both(stack, monkeypatch, events):
    server, port = stack
    monkeypatch.setattr(entrypoint, "api_server", server)
    frames = [e if isinstance(e, (str, bytes)) else json.dumps(e) for e in events]
    server.steps = []
    py = _normalized(_python(server, frames))
    py_steps = server.steps
    server.steps = []
    rust = _normalized(_rust_session(port, frames))
    return (py, py_steps), (rust, server.steps)


def _appends(audio: np.ndarray, rate: int, seconds: float = 0.3) -> list[dict]:
    n = int(rate * seconds)
    return [{"type": "input_audio_buffer.append", "audio": _pcm(audio[i:i + n])} for i in range(0, len(audio), n)]


def _session(**session) -> dict:
    return {"type": "transcription_session.update", "session": session}


def _assert_same(py, rust):
    (py_events, py_steps), (rust_events, rust_steps) = py, rust
    assert rust_steps == py_steps
    assert rust_events == py_events


def test_client_committed_items_stream_as_the_python_route(stack, monkeypatch):
    audio = _speech_16k()[: 16000 * 7]
    events = [*_appends(audio[: 16000 * 4], 16000), {"type": "input_audio_buffer.commit"},
              _session(mstar={"chunk_seconds": 1.0, "unfixed_tokens": 3, "unfixed_chunks": 0},
                       input_audio_transcription={"language": "en"}),
              *_appends(audio[16000 * 4:], 16000, 0.7), {"type": "input_audio_buffer.commit"}]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)
    assert sum(e["type"] == "conversation.item.input_audio_transcription.completed" for e in py[0]) == 2
    # the second item's steps continue from the stable prefix of the hypothesis so far
    assert any(step["mk"].get("assistant_prefix") for step in py[1])


def test_24khz_input_is_resampled_alike(stack, monkeypatch):
    audio = _speech_24k()[: 24000 * 5]
    events = [_session(input_audio_format={"type": "audio/pcm", "rate": 24000}), *_appends(audio, 24000, 0.11),
              {"type": "input_audio_buffer.commit"}, *_appends(audio[:24000], 24000, 0.4),
              {"type": "input_audio_buffer.clear"}]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)


@needs_vad
@pytest.mark.parametrize("rate", [16000, 24000])
def test_server_vad_cuts_the_same_turns(stack, monkeypatch, rate):
    speech = _speech_16k() if rate == 16000 else _speech_24k()
    gap = np.zeros(int(rate * 1.2), dtype=np.float32)
    audio = np.concatenate([speech[: rate * 3], gap, speech[rate * 5: rate * 8], gap])
    fmt = "pcm16" if rate == 16000 else {"type": "audio/pcm", "rate": 24000}
    events = [_session(input_audio_format=fmt, turn_detection={"type": "server_vad", "silence_duration_ms": 600}),
              *_appends(audio, rate, 0.25), {"type": "input_audio_buffer.commit"}]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)
    assert any(e["type"] == "input_audio_buffer.speech_started" for e in py[0])


@pytest.mark.skipif(not HAS_SMART_TURN, reason="no Smart Turn weights (hf download pipecat-ai/smart-turn-v3)")
@pytest.mark.parametrize("eagerness", ["high", "auto"])
def test_semantic_vad_asks_the_end_of_turn_model_alike(stack, monkeypatch, eagerness):
    speech = _speech_16k()
    pause = np.zeros(int(16000 * 0.9), dtype=np.float32)
    audio = np.concatenate([speech[: 16000 * 4], pause, speech[16000 * 6: 16000 * 9], np.zeros(16000 * 3, np.float32)])
    events = [_session(turn_detection={"type": "semantic_vad", "eagerness": eagerness}),
              *_appends(audio, 16000, 0.25), {"type": "input_audio_buffer.commit"}]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)
    assert any(e["type"] == "input_audio_buffer.speech_stopped" for e in py[0])


REFUSED_UPDATES = {
    "rate": {"input_audio_format": {"type": "audio/pcm", "rate": 44000}},
    "rate-float": {"input_audio_format": {"type": "audio/pcm", "rate": 16000.0}},
    "rate-bool": {"input_audio_format": {"type": "audio/pcm", "rate": True}},
    "format": {"input_audio_format": "mp3"},
    "format-type": {"input_audio_format": {"type": "audio/opus"}},
    "format-no-type": {"input_audio_format": {"rate": 16000}},
    "format-extra": {"input_audio_format": {"type": "audio/pcm", "rate": 24000, "channels": 2}},
    "g711-rate": {"input_audio_format": {"type": "audio/pcmu", "rate": 16000}},
    "g711-rate-bool": {"input_audio_format": {"type": "audio/pcma", "rate": True}},
    "turn-type": {"turn_detection": {"type": "magic"}},
    "turn-key": {"turn_detection": {"type": "server_vad", "speed": 1}},
    "idle": {"turn_detection": {"type": "server_vad", "idle_timeout_ms": 5}},
    "threshold": {"turn_detection": {"type": "server_vad", "threshold": 2}},
    "silence": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 1.5}},
    "flag": {"turn_detection": {"type": "server_vad", "create_response": "yes"}},
    "eagerness": {"turn_detection": {"type": "semantic_vad", "eagerness": "very"}},
    # refused where the end-of-turn model is missing, on both
    **({} if HAS_SMART_TURN else {"semantic": {"turn_detection": {"type": "semantic_vad"}}}),
    "chunk": {"mstar": {"chunk_seconds": "fast"}},
    "chunk-cap": {"mstar": {"chunk_seconds": 1000}},
    "unfixed": {"mstar": {"unfixed_tokens": -1}},
    "unfixed-float": {"mstar": {"unfixed_chunks": 1.5}},
    "language": {"input_audio_transcription": {"language": 5}},
    "language-long": {"input_audio_transcription": {"language": "x" * 40}},
    "not-objects": {"mstar": [1, 2, 3, 4, 5, 6]},
}


@pytest.mark.parametrize("case", sorted(REFUSED_UPDATES))
def test_a_refused_update_reads_as_pythons(stack, monkeypatch, case):
    py, rust = _both(stack, monkeypatch, [_session(**REFUSED_UPDATES[case]), _session()])
    _assert_same(py, rust)
    assert py[0][1]["type"] == "error"


def test_protocol_errors_read_as_pythons(stack, monkeypatch):
    audio = _speech_16k()[:16000]
    events = [
        b"\x00binary", "not json", "[1, 2]", {"type": "nope", "x": [1, 2, 3, 4, 5]},
        {"type": "input_audio_buffer.append", "audio": "!!!"},
        {"type": "input_audio_buffer.append", "audio": None},
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01\x02\x03").decode()},
        *_appends(audio, 16000),
        {"type": "input_audio_buffer.commit"},
        _session(input_audio_transcription={"language": "xx"}),
        *_appends(audio[:16000], 16000, 1.0), {"type": "input_audio_buffer.commit"},
        _session(input_audio_transcription={"language": None, "prompt": "boom"}),
        *_appends(audio[:16000], 16000, 1.0), {"type": "input_audio_buffer.commit"},
        # scrubbed as serving_realtime._scrub_paths scrubs it
        _session(input_audio_transcription={"prompt": "paths"}),
        *_appends(audio[:16000], 16000, 1.0), {"type": "input_audio_buffer.commit"},
        _session(mstar={"chunk_seconds": 0.1}),
    ]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)


def test_an_append_or_item_past_the_cap_is_refused_alike(stack, monkeypatch):
    # one append past 300 s, at 8 kHz G.711 (one byte a sample), then an item
    # that two appends take past it
    long = base64.b64encode(np.full(8000 * 301, 0xFF, np.uint8).tobytes()).decode()
    half = _pcm(np.zeros(16000 * 160, dtype=np.float32))
    events = [_session(input_audio_format="g711_ulaw"), {"type": "input_audio_buffer.append", "audio": long},
              _session(input_audio_format="pcm16"), {"type": "input_audio_buffer.append", "audio": half},
              {"type": "input_audio_buffer.append", "audio": half}]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)
    errors = [e["error"]["message"] for e in py[0] if e["type"] == "error"]
    assert errors[0].startswith("an append holds at most 300 s of audio")
    assert errors[1].startswith("the item would pass 300 s of audio")


@pytest.mark.parametrize("fmt", [
    "g711_ulaw", "g711_alaw", {"type": "audio/pcmu"}, {"type": "audio/pcma", "rate": 8000},
    {"type": "audio/pcm", "rate": 8000}, {"type": "audio/pcm", "rate": 11025}, {"type": "audio/pcm", "rate": 22050},
    {"type": "audio/pcm", "rate": 32000}, {"type": "audio/pcm", "rate": 44100}, {"type": "audio/pcm", "rate": 48000},
], ids=lambda f: f if isinstance(f, str) else f"{f['type'].split('/')[1]}-{f.get('rate', 'default')}")
def test_every_input_format_reaches_the_engine_alike(stack, monkeypatch, fmt):
    law = {"g711_ulaw": "audio/pcmu", "g711_alaw": "audio/pcma"}.get(fmt) if isinstance(fmt, str) else (
        fmt["type"] if fmt["type"] != "audio/pcm" else None)
    rate = 8000 if law else fmt["rate"]
    r = media_io.StreamResampler(24000, rate)
    audio = np.concatenate([r.push(_speech_24k()[: 24000 * 5]), r.flush()])
    if law:
        coded = media_io.g711_encode(media_io.to_pcm16(audio), law)
        n = int(rate * 0.3)
        appends = [{"type": "input_audio_buffer.append", "audio": base64.b64encode(coded[i:i + n]).decode()}
                   for i in range(0, len(coded), n)]
    else:
        appends = _appends(audio, rate, 0.13)
    events = [_session(input_audio_format=fmt), *appends, {"type": "input_audio_buffer.commit"}]
    py, rust = _both(stack, monkeypatch, events)
    _assert_same(py, rust)
    assert py[1], "the engine heard nothing"


def test_a_model_without_realtime_is_closed_with_pythons_reason(stack, monkeypatch):
    server, port = stack
    monkeypatch.setattr(entrypoint, "api_server", server)
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/v1/realtime?intent=conversation") as ws:
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            ws.recv(timeout=10)
    reason = "unsupported intent 'conversation'; use intent=transcription"
    assert (closed.value.rcvd.code, closed.value.rcvd.reason) == (1008, reason)



# what uvicorn (websockets) closes a socket with when a frame passes ws_max_message_mb
TOO_BIG = (1009, "frame with 2000 bytes exceeds limit of 1000 bytes")


def test_a_message_past_the_limit_closes_the_socket_as_uvicorn_does():
    with _rust(_ASR(Path(tempfile.mkdtemp(prefix="mstar_rt_up_"))), MSTAR_WS_MAX_BYTES="1000") as port:
        url = f"ws://127.0.0.1:{port}/v1/realtime?intent=transcription"
        with websockets.sync.client.connect(url, compression=None, max_size=None) as ws:
            ws.recv(timeout=10)   # the greeting frame
            ws.send("x" * 2000)
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                ws.recv(timeout=10)
    assert (closed.value.rcvd.code, closed.value.rcvd.reason) == TOO_BIG
