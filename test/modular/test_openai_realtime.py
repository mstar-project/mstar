"""``/v1/realtime`` transcription sessions over a stubbed APIServer.

A test adapter continues hypotheses through an ``assistant_prefix`` kwarg;
the stub engine answers each step from a queue, so the chunking, the
rollback prefix, append-only deltas with revisions, commit/clear and the
model-gating are all exercised without a GPU.
"""

import asyncio
import base64
import json
import sys
import tempfile
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI, HTTPException, WebSocketDisconnect  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server.openai import adapters, serving_realtime  # noqa: E402
from mstar.api_server.openai.adapters import SubmitArgs, Transcript  # noqa: E402


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split(" ") if text else []

    def decode(self, ids):
        return " ".join(ids)


class _StubModel:
    tokenizer = _WordTokenizer()


class _StubAPI:
    client_audio = []   # not a full-duplex model

    def __init__(self, model_name="rt_test"):
        self.model_name = model_name
        self.model = _StubModel()
        self.upload_dir = Path(tempfile.mkdtemp())
        self.submits: list = []
        self.answers: list[str] = []  # raw generations, one per step, in order
        self.audio_sizes: list[int] = []
        self._chunks: dict = {}

    def submit_request(self, **kw):
        self.submits.append(kw)
        # the step's WAV is deleted once the step is done
        self.audio_sizes.append(Path(kw["file_paths"]["audio"][0]).stat().st_size)
        answer = self.answers.pop(0) if self.answers else ""
        self._chunks[kw["request_id"]] = [_Chunk("text", answer.encode())]
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        # as APIServer.collect_results: a request the data worker failed
        # raises with the error chunk's status instead of returning chunks
        chunks = self._chunks.get(request_id, [])
        for c in chunks:
            if c.modality == "error":
                raise HTTPException(int((c.metadata or {}).get("status", 500)), c.data.decode("utf-8", "replace"))
        return chunks


class _ContinuingAdapter(adapters.OpenAIAdapter):
    """Continues a hypothesis by prefilling it as ``assistant_prefix``, and
    reports ``language X<asr_text>`` output like Qwen3-ASR."""

    supports_transcriptions = True
    supports_realtime_transcription = True

    def transcription_to_request(self, req, audio_path):
        return SubmitArgs(text="", file_paths={"audio": [audio_path]}, input_modalities=["audio", "text"],
                          output_modalities=["text"], model_kwargs=adapters._transcription_kwargs(req))

    def realtime_step_request(self, req, audio_path, prefix):
        args = self.transcription_to_request(req, audio_path)
        if prefix:
            args.model_kwargs["assistant_prefix"] = prefix
        return args

    def parse_transcript(self, text, req):
        language = None
        if "<asr_text>" in text:
            meta, text = text.split("<asr_text>", 1)
            language = meta.replace("language", "").strip() or None
        return Transcript(text=text.strip(), language=language or req.language)


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _StubAPI()
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    monkeypatch.setitem(adapters.ADAPTER_REGISTRY, "rt_test", _ContinuingAdapter())

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


def _pcm(seconds: float) -> str:
    samples = (np.sin(np.arange(int(seconds * 16000)) * 0.01) * 8000).astype("<i2")
    return base64.b64encode(samples.tobytes()).decode("ascii")


def _recv_until(ws, kind: str, limit: int = 20) -> list[dict]:
    events = []
    for _ in range(limit):
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if ev["type"] == kind:
            return events
    raise AssertionError(f"no {kind} event in {[e['type'] for e in events]}")


def test_stable_prefix_rolls_back_tokens():
    tok = _WordTokenizer()
    assert serving_realtime.stable_prefix("a b c d e f g", tok, 5) == "a b"
    assert serving_realtime.stable_prefix("a b", tok, 5) == ""
    assert serving_realtime.stable_prefix("", tok, 5) == ""
    assert serving_realtime.stable_prefix("a b c", None, 1) == ""


def test_session_chunking():
    from mstar.api_server.openai.protocol import TranscriptionRequest

    s = serving_realtime.RealtimeSession("sid", TranscriptionRequest(), chunk_seconds=1.0)
    s.append(np.zeros(12_000, dtype=np.float32))
    assert s.take_chunk() is None
    s.append(np.zeros(8_000, dtype=np.float32))
    chunk = s.take_chunk()
    assert chunk.shape == (16_000,) and s.pending_samples == 4_000 and s.take_chunk() is None
    assert s.flush().shape == (4_000,) and s.flush() is None


def test_session_streams_partials_deltas_and_completion(client_and_stub):
    client, stub = client_and_stub
    # step answers: the model's raw output for the audio heard so far
    stub.answers = [
        "language English<asr_text> hello there my",
        " friend how are",          # continued from the stable prefix
        " you today",
    ]
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        created = json.loads(ws.receive_text())
        assert created["type"] == "transcription_session.created"
        assert created["session"]["mstar"]["chunk_seconds"] == 2.0

        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_transcription": {"language": "en", "prompt": "names: Ada"},
            "mstar": {"chunk_seconds": 1.0, "unfixed_chunks": 1, "unfixed_tokens": 2},
        }}))
        updated = json.loads(ws.receive_text())
        assert updated["type"] == "transcription_session.updated"
        assert updated["session"]["input_audio_transcription"]["language"] == "en"

        # 1.5 s: one chunk decodes, half a second stays pending
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(1.5)}))
        events = _recv_until(ws, "mstar.transcription.partial")
        assert events[0]["type"] == "conversation.item.input_audio_transcription.delta"
        assert events[0]["delta"] == "hello"                # the last 2 tokens ("there my") are held back
        assert events[-1]["text"] == "hello there my" and events[-1]["chunk_id"] == 1
        assert stub.submits[0]["model_kwargs"] == {"language": "en", "initial_prompt": "names: Ada", "temperature": 0.0}
        assert "assistant_prefix" not in stub.submits[0]["model_kwargs"]

        # another 1.0 s: second chunk, now continuing from the stable prefix
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(1.0)}))
        events = _recv_until(ws, "mstar.transcription.partial")
        assert stub.submits[1]["model_kwargs"]["assistant_prefix"] == "language English<asr_text> hello"
        assert events[0]["delta"] == " friend"
        assert events[-1]["text"] == "hello friend how are"

        # commit: the 0.5 s tail is a final step, and nothing is held back any more
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _recv_until(ws, "conversation.item.input_audio_transcription.completed")
        kinds = [e["type"] for e in events]
        assert "input_audio_buffer.committed" in kinds
        assert stub.submits[2]["model_kwargs"]["assistant_prefix"] == "language English<asr_text> hello friend"
        deltas = [e["delta"] for e in events if e["type"].endswith(".delta")]
        assert "".join(deltas) == " you today"
        assert events[-1]["transcript"] == "hello friend you today"
        # the tail flush (0.5 s) was a third step, and every step hears all audio so far
        assert len(stub.submits) == 3
        durations = stub.audio_sizes
        assert not list(stub.upload_dir.iterdir())
        assert durations == sorted(durations) and durations[0] < durations[-1]
        # a committed utterance starts a fresh item
        ws.send_text(json.dumps({"type": "input_audio_buffer.clear"}))
        assert json.loads(ws.receive_text())["type"] == "input_audio_buffer.cleared"


def test_revision_of_sent_text_is_not_resent():
    """Deltas are append-only: when a new hypothesis disagrees with text
    already sent, only the part past the common prefix goes out, and the
    partial event carries the corrected whole."""
    from mstar.api_server.openai.protocol import TranscriptionRequest

    sent: list[dict] = []

    class _WS:
        async def send_text(self, payload):
            sent.append(json.loads(payload))

    rt = serving_realtime.RealtimeTranscription(_StubAPI(), "rt_test", _ContinuingAdapter(), _WS())
    rt.session.request = TranscriptionRequest(language="en")
    rt.session.emitted = "one two three"

    import asyncio

    asyncio.run(rt.emit("one two tree four five", hold=0))
    assert [e["delta"] for e in sent] == ["ree four five"]
    assert rt.session.emitted == "one two tree four five"
    sent.clear()
    asyncio.run(rt.emit("one two tree four five six seven", hold=len(" seven")))
    assert [e["delta"] for e in sent] == [" six"]


def test_bad_events_report_errors_but_keep_the_session(client_and_stub):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text("not json")
        assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(json.dumps({"type": "nope"}))
        assert "unknown event" in json.loads(ws.receive_text())["error"]["message"]
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": "%%%"}))
        assert json.loads(ws.receive_text())["error"]["message"] == "audio is not base64"


@pytest.mark.parametrize("audio", [None, 0, False, []])
def test_null_or_falsy_audio_is_an_invalid_audio_error(client_and_stub, audio):
    """A null append is refused as on the bridge and the duplex server, not
    read as an empty append, and the session keeps its audio."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    tone = _tone(1600)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        _append_bytes(ws, tone.tobytes())
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _recv_until(ws, "conversation.item.input_audio_transcription.completed")
    errors = [e["error"] for e in events if e["type"] == "error"]
    assert errors == [{"type": "invalid_request_error", "message": "audio is not base64", "code": "invalid_audio"}]
    assert _close_to(heard[-1], tone)


def _step_samples(stub) -> list[np.ndarray]:
    """Make the stub keep the int16 samples each engine step heard."""
    heard: list[np.ndarray] = []
    submit = stub.submit_request

    def recording_submit(**kw):
        with wave.open(kw["file_paths"]["audio"][0], "rb") as wf:
            heard.append(np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.int32))
        return submit(**kw)

    stub.submit_request = recording_submit
    return heard


def _tone(n: int, freq: float = 220.0) -> np.ndarray:
    return (np.sin(2 * np.pi * freq * np.arange(n) / 16000) * 8000).astype("<i2")


def _append_bytes(ws, raw: bytes) -> None:
    ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(raw).decode("ascii")}))


def _commit(ws) -> None:
    ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
    _recv_until(ws, "conversation.item.input_audio_transcription.completed")


def _close_to(heard: np.ndarray, samples: np.ndarray) -> bool:
    # the step WAV holds the client's int16 samples exactly
    return heard.shape == samples.shape and np.array_equal(heard, samples.astype(np.int32))


def test_sample_split_across_appends_is_joined(client_and_stub):
    """A client forwarding a byte stream (say, a pipe's ``read()``) can split
    a sample across two appends. The odd byte waits for the next append;
    dropping it would shift every later sample by one byte into noise."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    tone = _tone(3200)
    raw = tone.tobytes()
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        for i in range(0, len(raw), 1001):
            _append_bytes(ws, raw[i:i + 1001])
        _commit(ws)
    assert _close_to(heard[-1], tone)


def test_line_wrapped_base64_is_accepted_and_kept_sample_for_sample(client_and_stub):
    """``base64.encodebytes`` wraps lines every 76 characters; the whitespace
    is skipped, and full-scale samples reach the engine unchanged."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    samples = np.concatenate([_tone(1600), np.array([-32768, 32767, -1, 1, 0], dtype="<i2")])
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append",
                                 "audio": base64.encodebytes(samples.tobytes()).decode("ascii")}))
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": "AAA*"}))
        assert json.loads(ws.receive_text())["error"]["message"] == "audio is not base64"
        _commit(ws)
    assert _close_to(heard[-1], samples)


def test_whitespace_in_base64_costs_memory_linear_in_the_payload():
    """A frame of short base64 pieces between spaces is decoded without a
    Python object per piece, so a client cannot turn one large frame into
    hundreds of megabytes; non-ASCII is refused like any non-base64."""
    import tracemalloc

    payload = "AAAA " * 200_000  # 1 MB
    tracemalloc.start()
    audio, odd = serving_realtime.decode_append(payload)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert audio.size == 300_000 and not odd and not audio.any()
    assert peak < 5_000_000, peak
    with pytest.raises(ValueError):
        serving_realtime.decode_append("AAAA\u00a0AAAA")


def test_odd_byte_survives_commit_and_clear_drops_it(client_and_stub):
    """A commit takes whole samples; the byte of a sample split across the
    commit starts the next item. A clear drops it with the rest of the buffer."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    first, second = _tone(1600), _tone(1600, freq=440.0)
    raw = first.tobytes() + second.tobytes()
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        _append_bytes(ws, raw[:3201])
        _commit(ws)
        _append_bytes(ws, raw[3201:])
        _commit(ws)
        _append_bytes(ws, b"\x7f")
        ws.send_text(json.dumps({"type": "input_audio_buffer.clear"}))
        assert json.loads(ws.receive_text())["type"] == "input_audio_buffer.cleared"
        _append_bytes(ws, first.tobytes())
        _commit(ws)
    assert [len(h) for h in heard] == [1600, 1600, 1600]
    assert _close_to(heard[0], first) and _close_to(heard[1], second) and _close_to(heard[2], first)


def test_bad_session_update_is_an_error_and_changes_nothing(client_and_stub):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_transcription": {"language": "fr"}, "mstar": {"chunk_seconds": "abc"},
        }}))
        assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {}}))
        session = json.loads(ws.receive_text())["session"]
        assert session["input_audio_transcription"]["language"] is None
        assert session["mstar"]["chunk_seconds"] == serving_realtime.DEFAULT_CHUNK_SECONDS


def _step_seconds(stub) -> list[float]:
    # 44-byte WAV header, 16-bit mono
    return [(size - 44) / 2 / serving_realtime.SAMPLE_RATE for size in stub.audio_sizes]


def _update(mstar=None, **session) -> str:
    # json.dumps writes NaN and Infinity, which the server's json.loads accepts
    return json.dumps({"type": "transcription_session.update",
                       "session": {**session, **({"mstar": mstar} if mstar is not None else {})}})


@pytest.mark.parametrize("session", [
    {"mstar": {"chunk_seconds": float("nan")}},
    {"mstar": {"chunk_seconds": float("inf")}},
    {"mstar": {"chunk_seconds": float("-inf")}},
    {"mstar": {"chunk_seconds": "NaN"}},
    {"mstar": {"chunk_seconds": "Infinity"}},
    {"mstar": {"chunk_seconds": "1" + "0" * 5000}},
    {"mstar": {"chunk_seconds": 1e9}},
    {"mstar": {"chunk_seconds": 10**400}},
    {"mstar": {"chunk_seconds": True}},
    {"mstar": {"unfixed_chunks": float("inf")}},
    {"mstar": {"unfixed_chunks": float("nan")}},
    {"mstar": {"unfixed_tokens": float("inf")}},
    {"mstar": {"unfixed_tokens": -1}},
    {"mstar": {"unfixed_tokens": 2.5}},
    {"mstar": {"unfixed_tokens": "2.5"}},
    {"mstar": {"unfixed_tokens": "three"}},
    {"mstar": {"unfixed_chunks": 10**400}},
    {"mstar": [1]},
    {"input_audio_transcription": "en"},
])
def test_invalid_session_update_is_an_error_and_the_session_survives(client_and_stub, session):
    client, stub = client_and_stub
    stub.answers = ["language English<asr_text> hi"]
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": session}))
        assert json.loads(ws.receive_text())["type"] == "error"
        # nothing changed, and the next append decodes a default 2 s chunk
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        assert _recv_until(ws, "mstar.transcription.partial")[-1]["text"] == "hi"
    assert len(stub.submits) == 1 and _step_seconds(stub) == [2.0]


@pytest.mark.parametrize("mstar, expected", [
    ({"chunk_seconds": 0.1}, {"chunk_seconds": 0.5}),
    ({"chunk_seconds": 0}, {"chunk_seconds": 0.5}),
    ({"chunk_seconds": -3}, {"chunk_seconds": 0.5}),
    ({"chunk_seconds": "0.1"}, {"chunk_seconds": 0.5}),
    ({"chunk_seconds": "1.5"}, {"chunk_seconds": 1.5}),
    ({"unfixed_chunks": "3", "unfixed_tokens": " 4 "}, {"unfixed_chunks": 3, "unfixed_tokens": 4}),
    ({"unfixed_tokens": "4.0"}, {"unfixed_tokens": 4}),
])
def test_lenient_update_reads_numeric_strings_and_raises_a_short_chunk(client_and_stub, mstar, expected):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(_update(mstar))
        updated = json.loads(ws.receive_text())
        assert updated["type"] == "transcription_session.updated"
        knobs = updated["session"]["mstar"]
        assert {k: knobs[k] for k in expected} == expected
        assert all(type(knobs[k]) is type(v) for k, v in expected.items())


def _nested(value="x" * 64, depth=4, width=6):
    for _ in range(depth):
        value = [value] * width
    return value


@pytest.mark.parametrize("message", [
    _update({"chunk_seconds": "x" * 1_000_000}),
    _update({"chunk_seconds": "9" * 4300}),
    _update({"chunk_seconds": 10**4000}),
    _update({"chunk_seconds": _nested()}),
    _update({"unfixed_tokens": {"k" * 1000: _nested()}}),
    json.dumps({"type": "x" * 1_000_000}),
    json.dumps({"type": _nested()}),
    _update(input_audio_format="x" * 1_000_000),
    _update(input_audio_format=_nested()),
], ids=["long_string", "digit_string", "big_int", "nested_list", "nested_dict",
        "long_event_type", "nested_event_type", "long_audio_format", "nested_audio_format"])
def test_error_quotes_only_a_short_excerpt_of_client_input(message):
    import asyncio

    ws = _ScriptedWS([message, _update({"chunk_seconds": 1.0})])
    asyncio.run(serving_realtime.RealtimeTranscription(_StubAPI(), "rt_test", _ContinuingAdapter(), ws).run())
    assert [e["type"] for e in ws.sent] == ["transcription_session.created", "error", "transcription_session.updated"]
    # an excerpt of at most 100 characters plus the fixed text around it
    assert len(ws.sent[1]["error"]["message"]) <= 180


def test_excerpt_of_a_wide_container_does_not_sort_it():
    """A dict's keys are not compared at all: the excerpt reads its first
    few items instead of sorting every key."""
    comparisons = 0

    class _Key(str):
        def __lt__(self, other):
            nonlocal comparisons
            comparisons += 1
            return str.__lt__(self, other)

        __gt__ = __le__ = __ge__ = __lt__

    keys = [_Key(i) for i in range(100_000)]
    for value in (dict.fromkeys(keys, 0), set(keys), frozenset(keys), {"type": dict.fromkeys(keys, 0)}):
        assert len(serving_realtime._excerpt(value)) <= serving_realtime.EXCERPT_CHARS
    assert comparisons == 0


def test_wide_dict_event_type_is_an_error(client_and_stub):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": {str(i): 0 for i in range(100_000)}}))
        error = json.loads(ws.receive_text())
        assert error["type"] == "error" and len(error["error"]["message"]) <= 180
        ws.send_text(_update())
        assert json.loads(ws.receive_text())["type"] == "transcription_session.updated"


def test_non_object_events_are_errors(client_and_stub):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        for message in ("[1]", "3", json.dumps({"type": "transcription_session.update", "session": "x"})):
            ws.send_text(message)
            assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(_update())
        assert json.loads(ws.receive_text())["type"] == "transcription_session.updated"


def test_binary_frame_is_an_error_and_the_session_survives(client_and_stub):
    client, stub = client_and_stub
    stub.answers = ["language English<asr_text> hi"]
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_bytes(base64.b64decode(_pcm(1.0)))   # raw PCM instead of an append event
        assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        assert _recv_until(ws, "mstar.transcription.partial")[-1]["text"] == "hi"


class _ScriptedWS:
    """Feeds ``run()`` a fixed list of messages, then disconnects."""

    def __init__(self, messages):
        self.inbox = list(messages)
        self.sent: list[dict] = []

    async def accept(self):
        pass

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))

    async def receive(self):
        if not self.inbox:
            return {"type": "websocket.disconnect", "code": 1000}
        return {"type": "websocket.receive", "text": self.inbox.pop(0)}

    async def receive_text(self):
        message = await self.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message["code"])
        return message["text"]


@pytest.mark.parametrize("message", [
    # past Python's int-string conversion limit (4300 digits): a plain ValueError
    '{"type": "transcription_session.update", "session": {"mstar": {"unfixed_tokens": 1' + "0" * 4300 + "}}}",
    "[" * 10_000 + "0" + "]" * 10_000,   # RecursionError
], ids=["int_digits_limit", "deep_nesting"])
def test_unparseable_message_is_an_error_and_the_session_survives(message):
    import asyncio

    ws = _ScriptedWS([message, _update({"chunk_seconds": 1.0})])
    asyncio.run(serving_realtime.RealtimeTranscription(_StubAPI(), "rt_test", _ContinuingAdapter(), ws).run())
    assert [e["type"] for e in ws.sent] == ["transcription_session.created", "error", "transcription_session.updated"]
    assert ws.sent[-1]["session"]["mstar"]["chunk_seconds"] == 1.0


def test_engine_requests_per_audio_second_are_bounded(client_and_stub):
    """The chunk floor caps a session at 2 engine requests per audio second,
    and an append that completes several chunks is one step."""
    client, stub = client_and_stub
    floor = serving_realtime.MIN_CHUNK_SECONDS
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(_update({"chunk_seconds": floor}))
        assert json.loads(ws.receive_text())["session"]["mstar"]["chunk_seconds"] == floor
        for _ in range(4):   # 4 s in 1 s appends
            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(1.0)}))
            _recv_until(ws, "mstar.transcription.partial")
        assert len(stub.submits) == 4 and _step_seconds(stub) == [1.0, 2.0, 3.0, 4.0]
        # a 10 s burst is one step over the whole item, not 20
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(10.0)}))
        _recv_until(ws, "mstar.transcription.partial")
    assert len(stub.submits) == 5 and _step_seconds(stub)[-1] == 14.0


def test_coalesced_step_counts_its_chunks_for_the_prefix(client_and_stub):
    """A 6 s burst is 3 chunks: the next step continues from the stable
    prefix, as it would after 3 separate steps, so text already sent as
    deltas is not decoded afresh."""
    client, stub = client_and_stub
    stub.answers = ["language English<asr_text> one two three four five six seven eight", " nine ten"]
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(6.0)}))
        assert _recv_until(ws, "mstar.transcription.partial")[-1]["chunk_id"] == 3
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        _recv_until(ws, "mstar.transcription.partial")
    assert _step_seconds(stub) == [6.0, 8.0]
    assert stub.submits[1]["model_kwargs"]["assistant_prefix"] == "language English<asr_text> one two three"


def test_item_length_is_capped(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setattr(serving_realtime, "MAX_ITEM_SECONDS", 5.0)
    stub.answers = ["language English<asr_text> one", "language English<asr_text> one two", ""]
    with client.websocket_connect("/v1/realtime") as ws:
        created = json.loads(ws.receive_text())
        assert created["session"]["mstar"]["max_item_seconds"] == 5.0
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(4.0)}))
        _recv_until(ws, "mstar.transcription.partial")
        # 4 s heard + 1.5 s would pass the cap: refused, and nothing is buffered
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(1.5)}))
        error = json.loads(ws.receive_text())
        assert error["type"] == "error" and "commit" in error["error"]["message"]
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(1.0)}))
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        done = _recv_until(ws, "conversation.item.input_audio_transcription.completed")[-1]
        assert done["transcript"] == "one two"
        # a committed item starts the budget over
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(4.0)}))
        _recv_until(ws, "mstar.transcription.partial")
    assert _step_seconds(stub) == [4.0, 5.0, 4.0]


def test_item_cap_follows_a_shorter_model_limit(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setattr(adapters.ADAPTER_REGISTRY["rt_test"], "max_audio_seconds", 3.0, raising=False)
    with client.websocket_connect("/v1/realtime") as ws:
        assert json.loads(ws.receive_text())["session"]["mstar"]["max_item_seconds"] == 3.0
        ws.send_text(_update({"chunk_seconds": 4.0}))
        assert "between 0.5 and 3" in json.loads(ws.receive_text())["error"]["message"]


def test_failed_step_does_not_reuse_its_request_id(client_and_stub):
    client, stub = client_and_stub
    submit = stub.submit_request

    def failing_first(**kw):
        rid = submit(**kw)
        if len(stub.submits) == 1:
            stub._chunks[rid] = [_Chunk("error", b"bad language")]
        return rid

    stub.submit_request = failing_first
    stub.answers = ["", "language English<asr_text> hi"]
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        assert json.loads(ws.receive_text())["type"] == "error"
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        _recv_until(ws, "mstar.transcription.partial")
    ids = [s["request_id"] for s in stub.submits]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_realtime_is_gated_on_the_adapter(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "whisper_large"
    with pytest.raises(Exception):  # noqa: B017 — the server closes with 1008 before any event
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()


@pytest.mark.parametrize("transcription", [
    {"language": ["en"]},
    {"language": True},
    {"language": 7},
    {"prompt": {"text": "names"}},
    {"prompt": 5},
    {"language": "x" * 1_000_000},
    {"language": "x" * 33},  # one past MAX_LANGUAGE_CHARS
], ids=["language_list", "language_bool", "language_int", "prompt_object", "prompt_int", "language_1mb",
        "language_one_past"])
def test_language_and_prompt_are_validated_on_update(client_and_stub, transcription):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(_update(input_audio_transcription=transcription, mstar={"chunk_seconds": 1.0}))
        error = json.loads(ws.receive_text())
        assert error["type"] == "error" and len(error["error"]["message"]) <= 180
        assert "input_audio_transcription." in error["error"]["message"]
        # all or nothing: the knobs in the refused update did not apply either
        ws.send_text(_update())
        view = json.loads(ws.receive_text())["session"]
        assert view["input_audio_transcription"]["language"] is None
        assert view["input_audio_transcription"]["prompt"] == ""
        assert view["mstar"]["chunk_seconds"] == serving_realtime.DEFAULT_CHUNK_SECONDS


def test_language_and_prompt_strings_and_null_are_accepted(client_and_stub):
    client, stub = client_and_stub
    language = "x" * 32  # MAX_LANGUAGE_CHARS
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(_update(input_audio_transcription={"language": language, "prompt": "p" * 10_000}))
        view = json.loads(ws.receive_text())["session"]["input_audio_transcription"]
        assert view["language"] == language and view["prompt"] == "p" * 10_000
        ws.send_text(_update(input_audio_transcription={"language": None, "prompt": None}))
        view = json.loads(ws.receive_text())["session"]["input_audio_transcription"]
        assert view["language"] is None and view["prompt"] == ""


def _failing_steps(stub, message: bytes, status=None):
    submit = stub.submit_request

    def failing(**kw):
        rid = submit(**kw)
        stub._chunks[rid] = [_Chunk("error", message, {"status": status} if status else {})]
        return rid

    stub.submit_request = failing


def _step_failure_records(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("realtime step failed")]


def test_failed_step_quotes_a_bounded_request_error_and_logs_it_once_per_item(client_and_stub, caplog):
    # a model refusing a value it quotes whole (#318's Qwen3-ASR quotes the
    # language): a 400 from the engine
    client, stub = client_and_stub
    detail = ("preprocessing failed: ValueError: Unsupported language " + repr("x" * 1_000_000)).encode()
    _failing_steps(stub, detail, status=400)
    caplog.set_level("INFO", logger=serving_realtime.__name__)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        for _ in range(3):
            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
            error = json.loads(ws.receive_text())
            assert error["type"] == "error" and error["error"]["type"] == "invalid_request_error"
            message = error["error"]["message"]
            assert len(message) <= serving_realtime.MAX_ERROR_CHARS
            assert message.startswith("preprocessing failed: ValueError: Unsupported language 'xxx")
        assert len(_step_failure_records(caplog)) == 1
        ws.send_text(json.dumps({"type": "input_audio_buffer.clear"}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        assert json.loads(ws.receive_text())["type"] == "error"
    # a new item logs its failure again
    assert len(_step_failure_records(caplog)) == 2


def test_failed_step_server_error_is_generic_with_an_error_id(client_and_stub, caplog):
    client, stub = client_and_stub
    detail = b"Error in worker: io on /dev/shm/mstar_arena_w0_4242_ab12.seg3: No space left on device"
    _failing_steps(stub, detail, status=500)
    caplog.set_level("INFO", logger=serving_realtime.__name__)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        error = json.loads(ws.receive_text())
    assert error["error"]["type"] == "server_error"
    message = error["error"]["message"]
    assert "/dev/shm" not in message and message.startswith("realtime step failed (error id ")
    error_id = message.removeprefix("realtime step failed (error id ").rstrip(")")
    (record,) = _step_failure_records(caplog)
    assert error_id in record.getMessage() and record.levelname == "ERROR"
    assert "/dev/shm/mstar_arena_w0_4242_ab12.seg3" in caplog.text


def test_request_error_raised_locally_does_not_name_the_upload_dir(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    adapter = adapters.ADAPTER_REGISTRY["rt_test"]

    def refuse(req, audio_path, prefix):
        raise ValueError(f"cannot read {audio_path}")

    monkeypatch.setattr(adapter, "realtime_step_request", refuse)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        message = json.loads(ws.receive_text())["error"]["message"]
    assert message.startswith("cannot read <upload>/") and str(stub.upload_dir) not in message


@pytest.mark.parametrize("raw, scrubbed", [
    ("cannot load '/home/alice/My Models/private/config.json'", "cannot load 'config.json'"),
    ('cannot load "/srv/a b/[c]/x.pt" now', 'cannot load "x.pt" now'),
    ("cannot load path:/home/alice/private/config.json", "cannot load path:config.json"),
    ("see https://example.com/a/b and file:/etc/x/y", "see https://example.com/a/b and file:y"),
])
def test_scrub_paths_takes_quoted_and_colon_prefixed_paths_whole(raw, scrubbed, tmp_path):
    assert serving_realtime._scrub_paths(raw, tmp_path) == scrubbed


def test_alternating_step_failures_are_each_logged_once_per_item(client_and_stub, caplog):
    client, stub = client_and_stub
    details = iter([b"refused: A", b"refused: B"] * 3)
    submit = stub.submit_request

    def failing(**kw):
        rid = submit(**kw)
        stub._chunks[rid] = [_Chunk("error", next(details), {"status": 400})]
        return rid

    stub.submit_request = failing
    caplog.set_level("INFO", logger=serving_realtime.__name__)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        for _ in range(6):
            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
            assert json.loads(ws.receive_text())["type"] == "error"
    assert len(_step_failure_records(caplog)) == 2


def test_request_error_from_the_engine_keeps_only_base_names_of_server_paths(client_and_stub):
    client, stub = client_and_stub
    detail = (b"preprocessing failed: ValueError: cannot decode /tmp/mstar_uploads_alice/3f2a.wav "
              b"(see /home/alice/.cache/mstar/codec.json); docs at https://example.com/a/b, lang en/US")
    _failing_steps(stub, detail, status=400)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": _pcm(2.0)}))
        message = json.loads(ws.receive_text())["error"]["message"]
    assert message == ("preprocessing failed: ValueError: cannot decode 3f2a.wav "
                       "(see codec.json); docs at https://example.com/a/b, lang en/US")


def test_unsupported_turn_detection_is_refused_and_echoed_as_null(client_and_stub):
    """A turn detection the server does not have is refused, instead of the
    client waiting for commits that never come; the session says null."""
    client, _ = client_and_stub
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        assert json.loads(ws.receive_text())["session"]["turn_detection"] is None
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "turn_detection": {"type": "push_to_talk"},
            "input_audio_transcription": {"language": "en"},
        }}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert "turn_detection type 'push_to_talk' is not supported" in err["error"]["message"]
        for td, shown in (({}, "turn_detection {} is"), ({"type": None}, "turn_detection {'type': None} is"),
                          (False, "turn_detection must be an object")):
            ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"turn_detection": td}}))
            assert json.loads(ws.receive_text())["error"]["message"].startswith(shown)
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"turn_detection": None}}))
        updated = json.loads(ws.receive_text())
    assert updated["type"] == "transcription_session.updated"
    assert updated["session"]["turn_detection"] is None
    assert updated["session"]["input_audio_transcription"]["language"] is None  # the refused update changed nothing


@pytest.mark.parametrize("fmt", [None, "pcm16"])
def test_pcm16_is_the_default_16khz(client_and_stub, fmt):
    """"pcm16" names what an omitted format means, 16 kHz: one meaning for both."""
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"input_audio_format": fmt}}))
        session = json.loads(ws.receive_text())["session"]
    assert (session["input_audio_format"], session["mstar"]["sample_rate"]) == ("pcm16", 16000)


@pytest.mark.parametrize("fmt", [{"type": "audio/pcm", "rate": 24000}])
def test_24khz_input_is_resampled_onto_the_true_timeline(client_and_stub, fmt):
    """OpenAI's pcm rate, 24 kHz, declared, is resampled as it arrives, so one
    second of audio is one second to the model and in audio_seconds."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    tone_24k = (np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 8000).astype("<i2")
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": fmt, "mstar": {"chunk_seconds": 0.5},
        }}))
        updated = json.loads(ws.receive_text())
        assert updated["session"]["input_audio_format"] == fmt
        assert updated["session"]["mstar"]["sample_rate"] == 24000
        for i in range(0, tone_24k.size, 240):  # 10 ms appends
            _append_bytes(ws, tone_24k[i:i + 240].tobytes())
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _recv_until(ws, "conversation.item.input_audio_transcription.completed")
    partials = [e["audio_seconds"] for e in events if e["type"] == "mstar.transcription.partial"]
    assert partials[-1] == pytest.approx(1.0)
    assert len(heard[-1]) == 16000
    tone_16k = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 8000
    assert np.abs(heard[-1][50:-50] - tone_16k[50:-50]).max() < 4


def test_24khz_chunk_is_decoded_on_the_append_that_completes_it(client_and_stub):
    """The resampler's 3 ms lookahead does not hold a full chunk back until
    the client's next append: the partial comes before the next event's reply."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    tone = (np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 8000).astype("<i2")
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}, "mstar": {"chunk_seconds": 0.5}}}))
        ws.receive_text()
        for half in (tone[:12000], tone[12000:]):
            _append_bytes(ws, half.tobytes())
            ws.send_text(json.dumps({"type": "transcription_session.update", "session": {}}))
            events = _recv_until(ws, "transcription_session.updated")
            assert [e["type"] for e in events][0] == "mstar.transcription.partial"
        _commit(ws)
    assert [len(h) for h in heard] == [7952, 15952, 16000]


def test_24khz_chunk_boundaries_stay_on_the_item_timeline(client_and_stub):
    """The lookahead samples count toward a boundary once: appends that end
    just short of each 0.5 s boundary decode when the input passes it, so the
    steps stay on 8000-sample boundaries and do not drift earlier."""
    client, stub = client_and_stub
    heard = _step_samples(stub)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}, "mstar": {"chunk_seconds": 0.5}}}))
        ws.receive_text()
        for n in (12000, 11964, 11964, 11964):
            _append_bytes(ws, _tone(n).tobytes())
        _commit(ws)
    assert [len(h) for h in heard] == [7952, 16000, 24000, 31928]


def test_24khz_coalesced_step_counts_every_chunk_it_heard(client_and_stub):
    """One append of two chunks is one step that counts both, so the stable
    prefix starts where it would with one step per chunk."""
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}}}))
        ws.receive_text()
        _append_bytes(ws, _tone(96000).tobytes())
        events = _recv_until(ws, "mstar.transcription.partial")
    assert events[-1]["chunk_id"] == 2 and events[-1]["audio_seconds"] == pytest.approx(4.0, abs=0.004)


def test_24khz_item_cap_does_not_depend_on_how_the_audio_was_split(client_and_stub, monkeypatch):
    """At 24 kHz the cap rounds the whole item once: one second fits however
    the appends split it, and one sample more is refused."""
    client, stub = client_and_stub
    monkeypatch.setattr(serving_realtime, "MAX_ITEM_SECONDS", 1.0)
    heard = _step_samples(stub)
    tone = (np.sin(2 * np.pi * 440 * np.arange(24001) / 24000) * 8000).astype("<i2")
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}, "mstar": {"chunk_seconds": 0.5}}}))
        assert json.loads(ws.receive_text())["session"]["mstar"]["sample_rate"] == 24000
        for piece in (tone[:1], tone[1:23999], tone[23999:24000], tone[24000:]):
            _append_bytes(ws, piece.tobytes())
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        events = _recv_until(ws, "conversation.item.input_audio_transcription.completed")
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and "would pass 1 s" in errors[0]["error"]["message"]
    assert len(heard[-1]) == 16000


def test_24khz_resampling_runs_off_the_event_loop(client_and_stub, monkeypatch):
    """A long 24 kHz append is resampled in a worker thread, so it does not
    stall the event loop every session shares; a short one, or one with no
    whole sample, costs less than the hop and stays on the loop."""
    import threading

    from mstar.api_server import media_io

    client, _ = client_and_stub
    threads = {}
    push, send = media_io.StreamResampler.push, serving_realtime.RealtimeTranscription.send

    def recording_push(self, x):
        threads.setdefault("resample", []).append((len(x), threading.get_ident()))
        return push(self, x)

    async def recording_send(self, event):
        threads["loop"] = threading.get_ident()
        await send(self, event)

    monkeypatch.setattr(media_io.StreamResampler, "push", recording_push)
    monkeypatch.setattr(serving_realtime.RealtimeTranscription, "send", recording_send)
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}}}))
        ws.receive_text()
        _append_bytes(ws, b"\x01")
        _append_bytes(ws, b"\x02" + np.zeros(95, dtype="<i2").tobytes() + b"\x03")
        _append_bytes(ws, np.zeros(24000, dtype="<i2").tobytes())
        _commit(ws)
    loop = threads["loop"]
    assert [(n, t == loop) for n, t in threads["resample"]] == [(96, True), (24000, False)]


def test_unsupported_rate_and_a_rate_change_mid_item_are_refused(client_and_stub):
    client, stub = client_and_stub
    with client.websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_text()
        for fmt, needle in (({"type": "audio/pcm", "rate": 44101}, "44101 Hz is not supported"),
                            ({"type": "audio/pcmu", "rate": 16000}, "audio/pcmu is 8000 Hz"),
                            ("pcm24", "input_audio_format must be"),
                            ({"type": "audio/pcm", "rate": 24000, "channels": 2}, "takes only type and rate")):
            ws.send_text(json.dumps({"type": "transcription_session.update", "session": {"input_audio_format": fmt}}))
            assert needle in json.loads(ws.receive_text())["error"]["message"]
        _append_bytes(ws, _tone(800).tobytes())
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}}}))
        assert "before changing the input audio format" in json.loads(ws.receive_text())["error"]["message"]
        _commit(ws)
        # half a sample survives the commit, and only clear drops it
        _append_bytes(ws, b"\x01")
        _commit(ws)
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}}}))
        assert "clear the input audio buffer" in json.loads(ws.receive_text())["error"]["message"]
        ws.send_text(json.dumps({"type": "input_audio_buffer.clear"}))
        assert json.loads(ws.receive_text())["type"] == "input_audio_buffer.cleared"
        ws.send_text(json.dumps({"type": "transcription_session.update", "session": {
            "input_audio_format": {"type": "audio/pcm", "rate": 24000}}}))
        assert json.loads(ws.receive_text())["session"]["mstar"]["sample_rate"] == 24000


class _QuietVAD:
    def __call__(self, frame):
        return 0.0

    def reset(self):
        pass


@pytest.mark.parametrize("intent, session, refused", [
    ("transcription", {"input_audio_format": {"type": "audio/pcm", "rate": 16}}, True),
    ("transcription", {"input_audio_format": {"type": "audio/pcm", "rate": 16_384_000}}, True),
    ("conversation", {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16}}}}, True),
    ("conversation", {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 24_576_000}}}}, True),
    ("conversation", None, False),
    # the segmenter ends a turn at the item cap, so only the append cap bounds this one
    ("transcription", {"turn_detection": {"type": "server_vad"}, "mstar": {"chunk_seconds": 1}}, False),
])
def test_resampling_work_per_append_is_bounded(request, monkeypatch, intent, session, refused):
    """A rate whose resampling costs up to 1000 outputs per sample, or a
    kernel thousands of taps wide, is refused; so is one append longer than
    an item, which is resampled before the call's own budget sees it."""
    monkeypatch.setattr(serving_realtime, "MAX_ITEM_SECONDS", 1.0)
    monkeypatch.setattr(serving_realtime.realtime_vad, "silero_factory", lambda: _QuietVAD)
    if intent == "transcription":
        client, _ = request.getfixturevalue("client_and_stub")
        url, update = "/v1/realtime?intent=transcription", "transcription_session.update"
    else:
        client, _ = request.getfixturevalue("duplex")()
        url, update = "/v1/realtime", "session.update"
    with client.websocket_connect(url) as ws:
        ws.receive_text()
        if session is not None:
            _send(ws, update, session=session)
            reply = json.loads(ws.receive_text())
            if refused:
                assert "is not supported; use one of" in reply["error"]["message"]
                return
            assert reply["type"] == f"{update}d", reply
        _send(ws, "input_audio_buffer.append", audio=_pcm(2.0))   # 1.33 s at 24 kHz, 2 s at 16 kHz
        _send(ws, "input_audio_buffer.commit")
        assert "at most 1 s of audio" in json.loads(ws.receive_text())["error"]["message"]


def test_stream_resampler_is_split_invariant_and_keeps_time():
    from mstar.api_server.media_io import StreamResampler

    rng = np.random.default_rng(0)
    n = 24000 * 2 + 7
    x = (np.sin(2 * np.pi * 1000 * np.arange(n) / 24000) * 0.5).astype(np.float32)
    r = StreamResampler(24000, 16000)
    whole = np.concatenate([r.push(x), r.flush()])
    pieces, i = [], 0
    while i < n:
        step = int(rng.integers(1, 700))
        pieces.append(r.push(x[i:i + step]))
        i += step
    pieces.append(r.flush())
    assert whole.size == -(-n * 2 // 3)
    assert np.array_equal(np.concatenate(pieces), whole)
    exact = np.sin(2 * np.pi * 1000 * np.arange(whole.size) / 16000) * 0.5
    assert np.abs(whole[50:-50] - exact[50:-50]).max() < 1e-4
    # above the 8 kHz Nyquist of the output: filtered, not folded back
    for hz in (8250, 10000):  # just past the Nyquist folds to just under it
        alias = np.sin(2 * np.pi * hz * np.arange(n) / 24000).astype(np.float32)
        out = np.concatenate([r.push(alias), r.flush()])
        assert np.sqrt((out[100:-100] ** 2).mean()) < 1e-3, hz


def test_a_character_split_across_token_chunks_survives_a_step():
    """Qwen3-ASR emits one byte-level BPE token per chunk, so a CJK character
    or an emoji can arrive split across chunks: the step decodes the joined
    bytes, not each chunk."""
    import asyncio

    text = "真正的存在😀"

    class _ByteChunks(_StubAPI):
        def submit_request(self, **kw):
            super().submit_request(**kw)
            data = self._chunks[kw["request_id"]][0].data
            self._chunks[kw["request_id"]] = [_Chunk("text", data[i:i + 1]) for i in range(len(data))]
            return kw["request_id"]

    api = _ByteChunks()
    api.answers = [text]
    session = types.SimpleNamespace(
        session_id="s", request=serving_realtime.TranscriptionRequest(model="rt_test"),
        audio_so_far=lambda: np.zeros(1600, dtype=np.float32),
    )
    out = asyncio.run(serving_realtime._run_step(api, _ContinuingAdapter(), session, "pre "))
    assert out == "pre " + text


# -- conversation intent (full duplex) ---------------------------------------


class _DuplexModel:
    turn_tokens = (1, 2)
    channels = 1

    def get_output_audio_channels(self, modality="audio"):
        return self.channels

    def get_output_sample_rate(self, modality="audio"):
        return 24000

    def get_default_voice(self):
        return "Aria"


class _DuplexAPI:
    """One live call: records its opening and each append's samples, and
    plays ``outputs`` as the call's result chunks; with ``end`` the call
    then ends, else its output goes on."""

    def __init__(self, outputs=(), end=False):
        from mstar.streaming.topology import CLIENT, ClientAudio, Connection

        self.model_name = "duplex_test"
        self.model = _DuplexModel()
        self.client_audio = [Connection(CLIENT, "Encoder", "audio_in", chunk_policy_factory=lambda: None,
                                        audio=ClientAudio(16000, 1280))]
        self.end = end
        self.upload_dir = Path(tempfile.mkdtemp())
        self.submits: list = []
        self.appends: list[np.ndarray] = []
        self.aborted: list = []
        self.outputs = list(outputs)

    refuse = 0   # opens refused (at capacity) before one is taken

    def submit_request(self, **kw):
        if self.refuse:
            self.refuse -= 1
            raise HTTPException(503, "at capacity")
        self.submits.append(kw)
        return kw["request_id"]

    def submission_key(self, request_id):
        return request_id

    def stream_input(self, key, file_paths=None, *, samples, **kw):
        assert file_paths is None and set(samples) == {"audio_in"}
        self.appends.append(samples["audio_in"])
        return True

    async def _chunks(self):
        for chunk in self.outputs:
            yield chunk
        if not self.end:
            await asyncio.Event().wait()   # a live call's output goes on until the call ends

    def iter_result_chunks(self, request_id):
        return self._chunks()

    def abort_submission(self, key, outcome="abort"):
        self.aborted.append(key)


@pytest.fixture
def duplex(monkeypatch):
    import mstar.api_server

    def connect(outputs=(), end=False):
        fake_ep = types.ModuleType("mstar.api_server.entrypoint")
        fake_ep.api_server = api = _DuplexAPI(outputs, end)
        monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
        monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
        from mstar.api_server.openai.router import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app), api
    return connect


def _send(ws, kind, **fields):
    ws.send_text(json.dumps({"type": kind, **fields}))


def _session_update(ws, **session):
    _send(ws, "session.update", session=session)
    return _recv_until(ws, "session.updated")[-1]["session"]


@pytest.mark.parametrize("attr, value, reason", [("turn_tokens", None, "no turn tokens"),
                                                  ("channels", 2, "speaks in 2 channels")])
def test_a_conversation_needs_turn_tokens_and_mono_output(duplex, attr, value, reason):
    """A duplex model without turn tokens serves /generate/ws, but a
    conversation has no replies to cut, and its audio out is mono: either
    is refused before any event."""
    client, api = duplex()
    setattr(api.model, attr, value)
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()
    assert refused.value.code == 1008 and reason in refused.value.reason and "/generate/ws" in refused.value.reason
    assert api.submits == []


def test_a_model_that_cannot_converse_still_serves_transcription_by_default(client_and_stub, monkeypatch):
    """An omitted intent is the one the model serves: transcription, when an
    audio-streaming model has no turn tokens but its adapter transcribes."""
    from mstar.streaming.topology import CLIENT, ClientAudio, Connection

    client, stub = client_and_stub
    monkeypatch.setattr(stub, "client_audio", [Connection(CLIENT, "Ear", "audio_in", chunk_policy_factory=lambda: None,
                                                          audio=ClientAudio(16000, 1280))])
    stub.model.turn_tokens, stub.model.get_output_audio_channels = None, lambda modality="audio": 1
    with client.websocket_connect("/v1/realtime") as ws:
        assert json.loads(ws.receive_text())["type"] == "transcription_session.created"


@pytest.mark.parametrize("fmt", [{"type": "audio/pcm", "rate": 24000}, {"type": "audio/pcmu"}])
def test_conversation_audio_reaches_the_stream_resampled(duplex, fmt):
    """Audio in the client's codec and rate reaches the stream resampled to
    16 kHz, the same however the appends split it (the data worker cuts it
    into the stream's frames)."""
    from mstar.api_server import media_io

    client, api = duplex()
    rate = fmt.get("rate", 8000)
    tone = (np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate) * 8000).astype("<i2")   # half a second
    if fmt["type"] == "audio/pcm":
        sent, heard = tone.tobytes(), tone
    else:
        sent = media_io.g711_encode(tone, fmt["type"])
        heard = media_io.g711_decode(sent, fmt["type"])
    expected = media_io.StreamResampler(rate, 16000).push(heard.astype(np.float32) / 32768.0)
    step = len(sent) // len(tone)
    with client.websocket_connect("/v1/realtime") as ws:
        assert json.loads(ws.receive_text())["type"] == "session.created"
        assert _session_update(ws, audio={"input": {"format": fmt}})["audio"]["input"]["format"]["type"] == fmt["type"]
        start = 0
        for n in (7, 333, 1000, 2, len(tone)):   # samples, ending mid-frame
            piece = sent[start * step:(start + n) * step]
            _send(ws, "input_audio_buffer.append", audio=base64.b64encode(piece).decode())
            start += n
        _send(ws, "input_audio_buffer.commit")
        _recv_until(ws, "input_audio_buffer.committed")
    assert len(api.submits) == 1 and api.submits[0]["end_input"] is False
    assert all(a.dtype == np.float32 for a in api.appends)
    np.testing.assert_allclose(np.concatenate(api.appends), expected, atol=1 / 32768)


def test_a_refused_open_keeps_none_of_its_append(duplex):
    """An append whose call could not open is not taken, its half sample
    included, so the retry is read from its own first byte."""
    client, api = duplex()
    api.refuse = 1
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _session_update(ws, audio={"input": {"format": {"type": "audio/pcm", "rate": 16000}}})
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\x01\x02\x03").decode())
        assert "at capacity" in json.loads(ws.receive_text())["error"]["message"]
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(np.array([1000, -2000, 7], "<i2").tobytes()
                                                                       + b"\x09").decode())
        _send(ws, "input_audio_buffer.commit")
        _recv_until(ws, "input_audio_buffer.committed")
    (heard,) = api.appends
    np.testing.assert_array_equal(heard, np.array([1000, -2000, 7], np.float32) / 32768)


@pytest.mark.parametrize("half_sample", [False, True])
def test_a_paused_client_is_padded_with_silence(duplex, monkeypatch, half_sample):
    """The model steps only on the audio it hears, so the server keeps time:
    a client that stops sending is padded with silence once it is a frame
    behind, at most MAX_PAD_SECONDS a pause, and never while half a sample
    is pending."""
    monkeypatch.setattr(serving_realtime, "MAX_PAD_SECONDS", 0.24)   # three frames
    client, api = duplex()
    frame = np.full(1280, 100, "<i2").tobytes() + (b"\x01" if half_sample else b"")
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _session_update(ws, audio={"input": {"format": {"type": "audio/pcm", "rate": 16000}}})
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(frame).decode())
        time.sleep(0.6)
        _send(ws, "input_audio_buffer.commit")
        _recv_until(ws, "input_audio_buffer.committed")
    first, *pads = api.appends
    padded = sum(p.size for p in pads)
    assert first.size == 1280 and not any(p.any() for p in pads)
    assert padded == 0 if half_sample else 0 < padded <= 0.24 * 16000 + len(pads)


def test_an_append_that_fails_inside_the_server_is_an_error_event(duplex, monkeypatch):
    client, api = duplex()

    def broken(*a, **k):
        raise RuntimeError("/srv/secret/path is gone")

    monkeypatch.setattr(api, "stream_input", broken)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\0\0" * 480).decode())
        error = json.loads(ws.receive_text())["error"]
        _send(ws, "input_audio_buffer.commit")   # the socket is still up
        _recv_until(ws, "input_audio_buffer.committed")
    assert error["type"] == "server_error" and "error id" in error["message"] and "secret" not in error["message"]


def _text(step, data=b"", mark=None):
    return _Chunk("text", data, {"step": step, **({"turn": [mark]} if mark else {})})


def _audio(step):
    return _Chunk("audio", _step_pcm(step), {"sample_rate": 24000, "step": step})


def _step_pcm(step):
    return np.full(1920, step * 100, dtype="<i2").tobytes()


def _turn_outputs():
    """A reply over steps 1-4 of six: turn tokens decode to nothing, as
    nemotron's BOS / EOS do, and each step has its text and its audio."""
    texts = [(b"", None), (b"", "start"), (b"Hi", None), (b" there", None), (b"", "end"), (b"", None)]
    return [_text(i, data, mark) for i, (data, mark) in enumerate(texts)] + [_audio(i) for i in range(6)]


def _until_closed(ws) -> list[dict]:
    events = []
    with pytest.raises(WebSocketDisconnect):
        while True:
            events.append(json.loads(ws.receive_text()))
    return events


def test_conversation_reply_is_one_response_of_ga_events(duplex):
    client, api = duplex(_turn_outputs())
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\0\0" * 480).decode())
        events = _recv_until(ws, "response.done", limit=40)
    kinds = [e["type"] for e in events]
    assert kinds == [
        "response.created", "conversation.item.added", "response.output_item.added", "response.content_part.added",
        "response.output_audio_transcript.delta", "response.output_audio_transcript.delta",
        *["response.output_audio.delta"] * 4,
        "response.output_audio.done", "response.output_audio_transcript.done", "response.content_part.done",
        "response.output_item.done", "conversation.item.done", "response.done",
    ]
    response_id = events[0]["response"]["id"]
    assert {e.get("response_id", response_id) for e in events if e["type"].startswith("response.")} == {response_id}
    audio = [base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta"]
    assert audio == [_step_pcm(i) for i in range(1, 5)]
    done = events[-1]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"] == [{"type": "output_audio", "transcript": "Hi there"}]


def test_replies_are_paired_by_step_however_the_chunks_arrive(duplex):
    """A reply that starts while the one before still waits for its last
    audio (text runs ahead of the talker) is a response of its own, and
    audio that arrives before its step's text goes to the reply that text
    opens. One response is in progress at a time, as OpenAI's clients expect."""
    texts = [_text(0), _text(1, mark="start"), _text(2, b"Hi"), _text(3), _text(4, mark="end"),
             _text(5), _text(6, mark="start"), _text(7, b"Yo"), _text(8, mark="end"), _text(9)]
    outputs = [texts[0], _audio(0), _audio(1), *texts[1:], *(_audio(i) for i in range(2, 10))]
    client, api = duplex(outputs, end=True)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\0\0" * 480).decode())
        events = _until_closed(ws)
    active = None
    for e in events:
        if e["type"] == "response.created":
            assert active is None, "a response opened before the one in progress was done"
            active = e["response"]["id"]
        elif e["type"].startswith("response."):
            assert e.get("response_id", e.get("response", {}).get("id")) == active, e["type"]
            if e["type"] == "response.done":
                active = None
    done = [e["response"] for e in events if e["type"] == "response.done"]
    assert [(r["status"], r["output"][0]["content"][0]["transcript"]) for r in done] == [
        ("completed", "Hi"), ("completed", "Yo")]
    for response, steps in zip(done, ([1, 2, 3, 4], [6, 7, 8]), strict=True):
        assert [base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta"
                and e["response_id"] == response["id"]] == [_step_pcm(i) for i in steps]
        # a client assembles the text from the deltas, held ones included
        assert "".join(e["delta"] for e in events if e["type"] == "response.output_audio_transcript.delta"
                       and e["response_id"] == response["id"]) == response["output"][0]["content"][0]["transcript"]


def test_a_reply_without_an_end_mark_ends_where_the_next_starts(duplex):
    texts = [_text(0), _text(1, mark="start"), _text(2, b"Hi"), _text(3, mark="start"), _text(4, b"Yo"),
             _text(5, mark="end"), _text(6)]
    client, api = duplex([*texts, *(_audio(i) for i in range(7))], end=True)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\0\0" * 480).decode())
        events = _until_closed(ws)
    done = [e["response"] for e in events if e["type"] == "response.done"]
    assert [(r["status"], r["output"][0]["content"][0]["transcript"]) for r in done] == [
        ("completed", "Hi"), ("completed", "Yo")]
    for response, steps in zip(done, ([1, 2], [3, 4, 5]), strict=True):
        assert [base64.b64decode(e["delta"]) for e in events if e["type"] == "response.output_audio.delta"
                and e["response_id"] == response["id"]] == [_step_pcm(i) for i in steps]


def test_a_call_that_ends_mid_reply_finishes_its_response(duplex):
    outputs = [_text(0, mark="start"), _text(1, b"Hel"), _audio(0),
               _Chunk("error", b"the call passed its length", {"status": 400})]
    client, api = duplex(outputs, end=True)
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\0\0" * 480).decode())
        events = _until_closed(ws)
    kinds = [e["type"] for e in events]
    assert kinds.index("error") < kinds.index("response.done")
    done = events[kinds.index("response.done")]["response"]
    assert done["status"] == "failed" and done["output"][0]["status"] == "incomplete"
    assert done["output"][0]["content"][0]["transcript"] == "Hel"


def test_events_after_the_call_ends_get_no_answer():
    """The call's end closes the socket; a client event already on its way
    gets no answer, since sending on a closed socket raises in Starlette."""
    class _ClosingWS:
        closed, tried = False, []

        async def send_text(self, text):
            _ClosingWS.tried.append(json.loads(text)["type"])
            if _ClosingWS.closed:
                raise RuntimeError('Cannot call "send" once a close message has been sent.')

        async def close(self, code):
            _ClosingWS.closed = True

    api = _DuplexAPI()
    call = serving_realtime.RealtimeConversation(api, api.model_name, _ClosingWS(), api.client_audio[0])

    async def no_output():
        return
        yield

    async def go():
        await call.relay(no_output())
        await call.handle({"type": "input_audio_buffer.commit"})

    asyncio.run(go())
    assert _ClosingWS.closed and _ClosingWS.tried == []


def test_a_client_gone_mid_reply_ends_the_output_quietly(caplog):
    """A send that fails because the client left is the end of the socket:
    nothing more is sent, no error is logged, and the relay finishes."""
    class _GoneWS:
        sent = 0

        async def send_text(self, text):
            _GoneWS.sent += 1
            raise WebSocketDisconnect(1006)

    api = _DuplexAPI()
    call = serving_realtime.RealtimeConversation(api, api.model_name, _GoneWS(), api.client_audio[0])

    async def chunks():
        for chunk in (_text(0, mark="start"), _text(1, b"Hi"), _audio(0), _audio(1)):
            yield chunk

    asyncio.run(call.relay(chunks()))
    assert call.closed and _GoneWS.sent == 1
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_an_item_is_reported_only_when_its_audio_was_taken(duplex, monkeypatch):
    """An item's audio is taken whole or not at all: a bad part keeps the
    good parts before it off the call too, and the append cap bounds the
    whole item, not each part."""
    monkeypatch.setattr(serving_realtime, "MAX_ITEM_SECONDS", 0.2)
    client, api = duplex()
    good = base64.b64encode(b"\1\0" * 3840).decode()   # 160 ms at 24 kHz: one whole frame at 16 kHz
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        for parts, error in ((["!!"], "not base64"), ([good, "!!"], "not base64"), ([good, good], "at most 0.2 s")):
            _send(ws, "conversation.item.create", item={"type": "message", "role": "user", "content": [
                {"type": "input_audio", "audio": audio} for audio in parts]})
            assert error in json.loads(ws.receive_text())["error"]["message"]
        _send(ws, "conversation.item.create", item={"type": "message", "role": "user",
                                                     "content": [{"type": "input_audio", "audio": good}]})
        added = json.loads(ws.receive_text())
    assert added["type"] == "conversation.item.added"
    assert added["item"]["content"] == [{"type": "input_audio"}]
    assert len(api.appends) == 1


def test_an_option_the_model_cannot_honour_costs_the_update_nothing_else(duplex):
    """A stock client's defaults (another voice, input transcription) are
    answered with an error each, and its instructions still open the call."""
    client, api = duplex()
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _send(ws, "session.update", session={"instructions": "be brief", "audio": {
            "input": {"transcription": {"model": "gpt-4o-transcribe"}}, "output": {"voice": "ash"}}})
        replies = _recv_until(ws, "session.updated")
        _send(ws, "input_audio_buffer.append", audio=base64.b64encode(b"\0\0" * 480).decode())
        _send(ws, "input_audio_buffer.commit")
        _recv_until(ws, "input_audio_buffer.committed")
    errors = [r["error"]["message"] for r in replies if r["type"] == "error"]
    assert len(errors) == 2 and "transcription" in errors[0] and "voice" in errors[1]
    served = replies[-1]["session"]
    assert served["instructions"] == "be brief" and served["audio"]["output"]["voice"] == "Aria"
    assert api.submits[0]["text"] == "be brief"


def test_unsupported_events_are_errors_and_the_call_goes_on(duplex):
    client, api = duplex()
    frame = base64.b64encode(b"\1\0" * 1280).decode()   # 80 ms at 16 kHz
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_text()
        _session_update(ws, audio={"input": {"format": {"type": "audio/pcm", "rate": 16000}}})
        _send(ws, "input_audio_buffer.append", audio=frame)
        for kind in ("response.create", "response.cancel", "conversation.item.truncate", "output_audio_buffer.clear"):
            _send(ws, kind)
            error = json.loads(ws.receive_text())
            assert error["type"] == "error" and error["error"]["code"] == "unsupported", kind
        _send(ws, "session.update", session={"audio": {"input": {"turn_detection": None}}})
        assert "turn_detection null is not supported" in json.loads(ws.receive_text())["error"]["message"]
        assert json.loads(ws.receive_text())["type"] == "session.updated"
        # what the model cannot do is answered with an error, and the served value stays
        for session in ({"max_output_tokens": 100}, {"audio": {"output": {"speed": 1.5}}},
                        {"audio": {"output": {"voice": "Ash"}}}):
            _send(ws, "session.update", session=session)
            assert json.loads(ws.receive_text())["error"]["code"] == "unsupported", session
            assert json.loads(ws.receive_text())["type"] == "session.updated"
        _send(ws, "session.update", session={"instructions": "be brief"})
        assert "fixed once audio flows" in json.loads(ws.receive_text())["error"]["message"]
        _send(ws, "input_audio_buffer.append", audio=frame)
        _send(ws, "input_audio_buffer.commit")
        _recv_until(ws, "input_audio_buffer.committed")
        assert not api.aborted
    assert len(api.submits) == 1 and len(api.appends) == 2
