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
        self._chunks: dict = {}

    def submit_request(self, **kw):
        self.submits.append(kw)
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
        # the tail flush (0.5 s) was a third step; every step hears all audio so far
        assert len(stub.submits) == 3
        durations = [Path(s["file_paths"]["audio"][0]).stat().st_size for s in stub.submits]
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


def test_realtime_is_gated_on_the_adapter(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "whisper_large"
    with pytest.raises(Exception):  # noqa: B017 — the server closes with 1008 before any event
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_text()
