"""``/v1/realtime`` transcription sessions over a stubbed APIServer.

A test adapter continues hypotheses through an ``assistant_prefix`` kwarg;
the stub engine answers each step from a queue, so the chunking, the
rollback prefix, append-only deltas with revisions, commit/clear and the
model-gating are all exercised without a GPU.
"""

import base64
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
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
        return self._chunks.get(request_id, [])


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
