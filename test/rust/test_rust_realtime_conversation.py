"""``/v1/realtime`` conversations on mstar-server against the Python route.

One scripted full-duplex model serves both frontends: it records the call's
opening and every append's samples, and plays a fixed script of text and
audio chunks (each stamped with its loop step and turn marks, as the data
worker stamps them). Each case sends the same client events to both and
compares every server event (ids numbered in order of appearance), how the
socket closed, and the audio the stream got, sample for sample.
"""

import asyncio
import base64
import contextlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mstar_rust")
pytest.importorskip("fastapi")
pytest.importorskip("websockets")
np = pytest.importorskip("numpy")

import websockets.exceptions
import websockets.sync.client
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_rust_frontend import BINARY, _Chunk, _free_port, _wait_healthy

from mstar.api_server import entrypoint, media_io
from mstar.api_server.rust_frontend import REALTIME_ENV, RustFrontendBridge, realtime_facts
from mstar.streaming.topology import CLIENT, ClientAudio, Connection

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")

FRAME = 1280   # the stream's frame: 80 ms at 16 kHz
BUDGET = 4 * 16000 * 4   # float32 bytes of input a call takes: 4 s


class _Model:
    turn_tokens = (1, 2)
    turn_idle_steps = None
    turn_silence_rms = 0.005
    channels = 1

    def get_output_audio_channels(self, modality="audio"):
        return self.channels

    def get_output_sample_rate(self, modality="audio"):
        return 24000

    def get_default_voice(self):
        return "Aria"


class _Duplex:
    """One live call at a time, on either frontend: plays ``script`` as the
    call's output once the call has heard audio, then ends it (``end``) or waits until the call is aborted
    or rejected. ``refuse`` opens are refused at capacity first; past
    ``BUDGET`` an append is refused as the call's input cap refuses it, and
    an append of one sample of 0.5 fails inside the server."""

    def __init__(self, upload_dir: Path, **model):
        self.model_name = "duplex_test"
        self.model = _Model()
        for k, v in model.items():
            setattr(self.model, k, v)
        self.client_audio = [Connection(CLIENT, "Encoder", "audio_in", chunk_policy_factory=lambda: None,
                                        audio=ClientAudio(16000, FRAME))]
        self.client_streams = frozenset({"audio_in"})
        self.takes_stream_input = True
        self.upload_dir = upload_dir
        self.metrics = None
        self.fatal_error = None
        self.reset()

    def reset(self, script=(), end=True, refuse=0):
        self.script, self.end, self.refuse = list(script), end, refuse
        self.submits, self.appends, self.aborted, self.rejected = [], [], [], []
        self.call = None

    def cleanup(self):
        pass

    async def wait_for_request_id(self, request_id, timeout=None):
        return None

    def submit_request(self, *, request_id=None, **kw):
        if self.refuse:
            self.refuse -= 1
            raise HTTPException(503, "at capacity")
        self.submits.append({k: kw.get(k) for k in ("text", "input_modalities", "output_modalities", "end_input")})
        self.call = {"rid": request_id, "bytes": 0, "error": None, "over": False}
        return request_id

    def submission_key(self, request_id):
        return request_id

    def relay_chunks_to(self, key, sink, tag):
        pass

    def taking_input(self, key):
        return self.call is not None and self.call["rid"] == key and not self.call["over"]

    def stream_input(self, key, file_paths=None, end_input=False, *, text=None, file_streams=None,
                     model_kwargs=None, on_late=None, samples=None):
        assert file_paths is None and text is None and set(samples) == {"audio_in"}
        audio = samples["audio_in"]
        assert audio.dtype == np.float32
        if not self.taking_input(key):
            return False
        if audio.size == 1 and audio[0] == 0.5:
            raise RuntimeError("the stream's buffer is gone")
        if self.call["bytes"] + audio.nbytes > BUDGET:
            raise ValueError(f"call input is past {BUDGET} bytes")
        self.call["bytes"] += audio.nbytes
        self.appends.append(audio.copy())
        return True

    def reject_input(self, key, error, status):
        if not self.taking_input(key):
            return False
        self.rejected.append(error)
        self.call["error"], self.call["over"] = (error, status), True
        return True

    async def iter_result_chunks(self, request_id):
        call = self.call
        # a duplex model speaks once it hears: the script plays after the first append
        while self.script and not self.appends and not call["over"]:
            await asyncio.sleep(0.002)
        for chunk in self.script:
            yield chunk
            await asyncio.sleep(0)
        while not self.end and not call["over"]:
            await asyncio.sleep(0.005)
        if call["error"] is not None:
            error, status = call["error"]
            yield _Chunk("error", error.encode(), {"status": status})
        call["over"] = True

    def abort_submission(self, key, outcome="abort"):
        self.aborted.append(key)
        if self.call is not None:
            self.call["over"] = True


@contextlib.contextmanager
def _rust(server: _Duplex):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="mstar_conv_")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MSTAR_")}
    env[REALTIME_ENV] = json.dumps(realtime_facts(server, None, bridge_dir))
    proc = subprocess.Popen([BINARY, server.model_name, str(port), bridge_dir, str(server.upload_dir)], env=env)
    bridge = RustFrontendBridge(server, bridge_dir)
    threading.Thread(target=bridge.run, daemon=True).start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


MODELS = {
    "marks": {},
    "idle": {"turn_idle_steps": 2},
    "no-turns": {"turn_tokens": None},
    "stereo": {"channels": 2},
}


@pytest.fixture(scope="module")
def stacks():
    with contextlib.ExitStack() as exits:
        made = {}

        def get(kind):
            if kind not in made:
                server = _Duplex(Path(tempfile.mkdtemp(prefix="mstar_conv_up_")), **MODELS[kind])
                made[kind] = (server, exits.enter_context(_rust(server)))
            return made[kind]
        yield get


_IDS = re.compile(r"\b(resp|item|sess|event)_[0-9a-f]{12,16}\b")


def _normalized(events: list[dict]) -> list[dict]:
    """Ids numbered in order of appearance (event ids dropped), error ids as X."""
    seen: dict = {}

    def number(m):
        return f"{m.group(1)}_{seen.setdefault(m.group(0), len(seen))}"
    out = []
    for e in events:
        e = {k: v for k, v in e.items() if k != "event_id"}
        text = re.sub(r"error id [0-9a-f]{12}", "error id X", json.dumps(e, sort_keys=True))
        out.append(json.loads(_IDS.sub(number, text)))
    return out


def _python(server, frames, until, linger):
    entrypoint.api_server = server
    events, code = [], None
    try:
        with TestClient(entrypoint.app).websocket_connect("/v1/realtime") as ws:
            events.append(json.loads(ws.receive_text()))
            for f in frames:
                if isinstance(f, float):
                    time.sleep(f)
                    continue
                (ws.send_bytes if isinstance(f, bytes) else ws.send_text)(f)
            while until is None or not until(events):
                events.append(json.loads(ws.receive_text()))
            time.sleep(linger)
    except WebSocketDisconnect as closed:
        code = closed.code
    return events, code


def _rust_session(port, frames, until, linger):
    events, code = [], None
    try:
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/v1/realtime", compression=None,
                                            max_size=None) as ws:
            events.append(json.loads(ws.recv(timeout=30)))
            for f in frames:
                if isinstance(f, float):
                    time.sleep(f)
                    continue
                ws.send(f)
            while until is None or not until(events):
                events.append(json.loads(ws.recv(timeout=30)))
            time.sleep(linger)
    except websockets.exceptions.ConnectionClosed as closed:
        code = closed.rcvd.code if closed.rcvd is not None else None
    return events, code


def _both(stacks, monkeypatch, kind, events, *, script=(), end=True, refuse=0, until=None, linger=0.0):
    """Each frontend's (events, close code) and what its call got. ``events``
    may hold floats, seconds to wait; ``until`` stops reading (the socket
    left open), else the session reads until the server closes it."""
    server, port = stacks(kind)
    monkeypatch.setattr(entrypoint, "api_server", server)
    frames = [e if isinstance(e, (str, bytes, float)) else json.dumps(e) for e in events]
    out = []
    for run in (lambda: _python(server, frames, until, linger), lambda: _rust_session(port, frames, until, linger)):
        server.reset(script, end, refuse)
        evs, code = run()
        out.append((_normalized(evs), code, {"submits": server.submits, "appends": list(server.appends),
                                              "rejected": server.rejected}))
    return out


def _text(step, data=b"", mark=None):
    return _Chunk("text", data, {"step": step, **({"turn": [mark]} if mark else {})})


def _audio(step, level=None):
    level = step * 100 if level is None else level
    return _Chunk("audio", np.full(1920, level, dtype="<i2").tobytes(), {"sample_rate": 24000, "step": step})


def _append(pcm: bytes) -> dict:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}


def _tone(rate, seconds, amp=8000):
    return (np.sin(2 * np.pi * 440 * np.arange(int(rate * seconds)) / rate) * amp).astype("<i2")


QUIET = _append(b"\0\0" * 480)
# answered with an error after every event before it: reading up to it reads everything they caused
END = {"type": "mstar.end"}


def _is_end(e: dict) -> bool:
    return e.get("type") == "error" and e["error"]["message"] == "unknown event 'mstar.end'"


def _assert_same(py, rust, *, audio=True):
    (py_events, py_code, py_call), (rust_events, rust_code, rust_call) = py, rust
    assert rust_events == py_events
    assert rust_code == py_code
    assert rust_call["submits"] == py_call["submits"]
    assert rust_call["rejected"] == py_call["rejected"]
    if audio:
        assert len(rust_call["appends"]) == len(py_call["appends"])
        for r, p in zip(rust_call["appends"], py_call["appends"], strict=True):
            _assert_same_audio(r, p)


def _assert_same_audio(rust, py):
    """Sample for sample, but where a resampled sum cancels to about zero:
    numpy's exp, on an AVX512F CPU, is an ulp off the libm exp mstar-server's
    Kaiser window uses on some taps (see realtime/resample.rs)."""
    differ = rust.view(np.uint32) != py.view(np.uint32)
    assert np.all(np.abs(py[differ]) < 1e-15) and np.all(np.abs(rust[differ] - py[differ]) < 1e-15)


def _kinds(events):
    return [e["type"] for e in events]


SCRIPTS = {
    # a reply over steps 1-4 of six, turn tokens decoding to nothing as nemotron's BOS / EOS do
    "one-reply": [*(_text(i, d, m) for i, (d, m) in enumerate(
        [(b"", None), (b"", "start"), (b"Hi", None), (b" there", None), (b"", "end"), (b"", None)])),
        *(_audio(i) for i in range(6))],
    # audio ahead of its text, and a reply that starts while the one before waits for its audio
    "paired-by-step": [_text(0), _audio(0), _audio(1), _text(1, mark="start"), _text(2, b"Hi"), _text(3),
                       _text(4, mark="end"), _text(5), _text(6, mark="start"), _text(7, b"Yo"),
                       _text(8, mark="end"), _text(9), *(_audio(i) for i in range(2, 10))],
    "no-end-mark": [_text(0), _text(1, mark="start"), _text(2, b"Hi"), _text(3, mark="start"), _text(4, b"Yo"),
                    _text(5, mark="end"), _text(6), *(_audio(i) for i in range(7))],
    # the call ends inside a reply: it finishes incomplete
    "ends-mid-reply": [_text(0, mark="start"), _text(1, b"Half"), _audio(0), _audio(1)],
    # a character split across two chunks, its tail held until the end mark
    "split-utf8": [_text(0, mark="start"), _text(1, "wö".encode()[:2]), _text(2, "wö".encode()[2:] + b"!"),
                   _text(3, mark="end"), *(_audio(i) for i in range(4))],
}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_replies_read_as_pythons(stacks, monkeypatch, name):
    py, rust = _both(stacks, monkeypatch, "marks", [QUIET], script=SCRIPTS[name])
    _assert_same(py, rust)
    assert "response.done" in _kinds(py[0])


def test_a_reply_that_ends_on_quiet_reads_as_pythons(stacks, monkeypatch):
    # turn_idle_steps=2: no end mark; the reply ends two steps after its last text and its last audible audio
    texts = [b"Hi", b" you", b"", b"", b"", b"Again", b"", b"", b"", b""]
    levels = [3000, 3000, 3000, 0, 0, 3000, 0, 0, 0, 0]
    script = [c for i in range(10) for c in (_text(i, texts[i]), _audio(i, levels[i]))]
    py, rust = _both(stacks, monkeypatch, "idle", [QUIET], script=script)
    _assert_same(py, rust)
    assert [e["response"]["status"] for e in py[0] if e["type"] == "response.done"] == ["completed", "completed"]


@pytest.mark.parametrize("status", [400, 503])
def test_a_failed_call_finishes_its_reply_and_closes_as_pythons(stacks, monkeypatch, status):
    script = [_text(0, mark="start"), _text(1, b"Half"), _audio(0), _audio(1),
              _Chunk("error", b"the engine failed reading /srv/ckpt/model.safetensors", {"status": status})]
    py, rust = _both(stacks, monkeypatch, "marks", [QUIET], script=script)
    _assert_same(py, rust)
    assert py[1] == (1011 if status >= 500 else 1000)


@pytest.mark.parametrize("fmt", [
    {"type": "audio/pcm", "rate": 24000}, {"type": "audio/pcm", "rate": 16000}, {"type": "audio/pcm", "rate": 48000},
    {"type": "audio/pcm", "rate": 11025}, {"type": "audio/pcmu"}, {"type": "audio/pcma"},
], ids=lambda f: f"{f['type'].split('/')[1]}-{f.get('rate', 8000)}")
def test_audio_reaches_the_stream_and_the_client_alike(stacks, monkeypatch, fmt):
    """In: the client's codec and rate, resampled to the stream's 16 kHz. Out:
    the model's 24 kHz PCM16 in the client's format."""
    rate = fmt.get("rate", 8000)
    tone = _tone(rate, 0.5)
    sent = tone.tobytes() if fmt["type"] == "audio/pcm" else media_io.g711_encode(tone, fmt["type"])
    width = len(sent) // len(tone)
    appends, start = [], 0
    for n in (7, 333, 1000, 2, len(tone)):   # samples, ending mid-frame
        appends.append(_append(sent[start * width:(start + n) * width]))
        start += n
    update = {"type": "session.update", "session": {"audio": {"input": {"format": fmt}, "output": {"format": fmt}}}}
    # the call stays open, so every append is taken; END and the reply race on either frontend
    py, rust = _both(stacks, monkeypatch, "marks", [update, *appends, END], script=SCRIPTS["one-reply"], end=False,
                     until=lambda evs: "response.done" in _kinds(evs) and any(_is_end(e) for e in evs))
    for side in (py, rust):
        side[0][:] = [e for e in side[0] if not _is_end(e)]
    _assert_same(py, rust, audio=False)
    # the clock pads the pause after the last append by each frontend's own wall time
    heard_py, heard_rust = np.concatenate(py[2]["appends"]), np.concatenate(rust[2]["appends"])
    n = min(heard_py.size, heard_rust.size)
    assert n >= 0.45 * 16000
    _assert_same_audio(heard_rust[:n], heard_py[:n])
    assert py[2]["submits"][0]["end_input"] is False


SETTINGS = {
    "instructions": {"instructions": "Be brief."},
    "ignored-options": {"output_modalities": ["text"], "tools": [{"type": "function"}], "max_output_tokens": 5,
                        "audio": {"input": {"transcription": {"model": "x"}, "noise_reduction": {},
                                            "turn_detection": {"type": "semantic_vad", "create_response": False}},
                                  "output": {"voice": "alloy", "speed": 1.5}}},
    "accepted-defaults": {"output_modalities": ["audio"], "max_output_tokens": "inf", "tools": [],
                          "audio": {"input": {"turn_detection": {"type": "semantic_vad", "interrupt_response": True}},
                                    "output": {"voice": "Aria", "speed": 1}}},
    "turn-null": {"audio": {"input": {"turn_detection": None}}},
    "turn-type": {"audio": {"input": {"turn_detection": {"type": "magic"}}}},
    "refused-type": {"type": "transcription"},
    "refused-instructions": {"instructions": 5},
    "refused-format": {"audio": {"input": {"format": {"type": "audio/pcmu", "rate": 16000}}}},
    "refused-audio": {"audio": [1]},
    "refused-turn": {"audio": {"input": {"turn_detection": "server_vad"}}},
}


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_a_session_update_reads_as_pythons(stacks, monkeypatch, name):
    events = [{"type": "session.update", "session": SETTINGS[name]}, END]
    py, rust = _both(stacks, monkeypatch, "marks", events, until=lambda evs: _is_end(evs[-1]))
    _assert_same(py, rust)


def test_settings_are_fixed_once_audio_flows(stacks, monkeypatch):
    events = [{"type": "session.update", "session": {"instructions": "Hi."}}, QUIET,
              {"type": "session.update", "session": {"instructions": "Other."}},
              {"type": "session.update", "session": {"audio": {"output": {"format": {"type": "audio/pcmu"}}}}},
              {"type": "session.update", "session": {"instructions": "Hi.", "tools": [1]}}, END]
    py, rust = _both(stacks, monkeypatch, "marks", events, end=False,
                     until=lambda evs: _is_end(evs[-1]))
    _assert_same(py, rust)
    assert py[2]["submits"] == [{"text": "Hi.", "input_modalities": ["text"], "output_modalities": ["text", "audio"],
                                 "end_input": False}]


def test_client_events_read_as_pythons(stacks, monkeypatch):
    item = {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": QUIET["audio"]}]}
    events = ["{not json", "[1]", b"\x00", {"type": "response.create"}, {"type": "conversation.item.truncate"},
              {"type": "nope"}, {"type": 5}, {"type": "conversation.item.create", "item": item},
              {"type": "conversation.item.create", "item": {**item, "id": "mine"}},
              {"type": "conversation.item.create", "item": {"type": "message", "role": "assistant", "content": []}},
              {"type": "conversation.item.create", "item": {**item, "content": [{"type": "input_text"}]}},
              {"type": "conversation.item.create"}, {"type": "input_audio_buffer.commit"},
              {"type": "input_audio_buffer.append", "audio": None},
              {"type": "input_audio_buffer.append", "audio": "A!"}, {"type": "input_audio_buffer.append"},
              {"type": "input_audio_buffer.clear"}, END]
    py, rust = _both(stacks, monkeypatch, "marks", events, end=False,
                     until=lambda evs: _is_end(evs[-1]))
    _assert_same(py, rust)


def test_a_refused_open_keeps_none_of_its_append(stacks, monkeypatch):
    """An append whose call could not open is not taken, its half sample
    included, and the socket goes on: the next append opens the call."""
    fmt = {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}
    events = [fmt, _append(b"\x01\x02\x03"), _append(np.array([1000, -2000, 7], "<i2").tobytes() + b"\x09"),
              {"type": "input_audio_buffer.commit"}]
    py, rust = _both(stacks, monkeypatch, "marks", events, end=False, refuse=1,
                     until=lambda evs: evs[-1]["type"] == "input_audio_buffer.committed")
    _assert_same(py, rust)
    assert "at capacity" in py[0][2]["error"]["message"]
    np.testing.assert_array_equal(np.concatenate(py[2]["appends"]), np.array([1000, -2000, 7], np.float32) / 32768)


def test_an_append_past_the_calls_budget_ends_the_call_alike(stacks, monkeypatch):
    fmt = {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}
    events = [fmt, _append(_tone(16000, 3).tobytes()), _append(_tone(16000, 3).tobytes())]
    py, rust = _both(stacks, monkeypatch, "marks", events, end=False)
    _assert_same(py, rust)
    assert py[2]["rejected"] == [f"append rejected: call input is past {BUDGET} bytes"]


def test_an_append_that_fails_inside_the_server_is_an_error_event_alike(stacks, monkeypatch):
    fmt = {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}
    events = [fmt, _append(np.array([16384], "<i2").tobytes()), END]
    py, rust = _both(stacks, monkeypatch, "marks", events, end=False,
                     until=lambda evs: _is_end(evs[-1]))
    _assert_same(py, rust)
    assert any(e.get("error", {}).get("message") == "the append failed (error id X)" for e in py[0])


def test_a_paused_client_is_padded_with_whole_frames_of_silence(stacks, monkeypatch):
    """The model steps only on audio it hears: a client that pauses is padded
    with silence once it is a frame past the jitter allowance behind. Timed by
    each frontend's own clock, so what is compared is the shape."""
    fmt = {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}}
    events = [fmt, _append(_tone(16000, 0.25).tobytes()), 1.2, {"type": "input_audio_buffer.commit"}]
    for _, _, call in _both(stacks, monkeypatch, "marks", events, end=False,
                            until=lambda evs: evs[-1]["type"] == "input_audio_buffer.committed"):
        tone, *pads = call["appends"]
        assert tone.size == 4000 and pads and all(not p.any() and p.size % FRAME == 0 for p in pads)
        assert 6 * FRAME <= sum(p.size for p in pads) <= 13 * FRAME


@pytest.mark.parametrize("kind, query", [
    ("marks", "?intent=transcription"), ("marks", "?intent=conversation"), ("marks", "?intent="),
    ("no-turns", ""), ("no-turns", "?intent=conversation"), ("stereo", ""),
])
def test_the_route_is_refused_as_pythons(stacks, monkeypatch, kind, query):
    server, port = stacks(kind)
    monkeypatch.setattr(entrypoint, "api_server", server)
    with pytest.raises(WebSocketDisconnect) as py:
        with TestClient(entrypoint.app).websocket_connect(f"/v1/realtime{query}") as ws:
            ws.receive_text()
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/v1/realtime{query}") as ws:
        with pytest.raises(websockets.exceptions.ConnectionClosed) as rust:
            ws.recv(timeout=10)
    assert (rust.value.rcvd.code, rust.value.rcvd.reason) == (py.value.code, py.value.reason)
    assert py.value.code == 1008
