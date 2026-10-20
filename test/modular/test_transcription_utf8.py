"""``/v1/audio/transcriptions`` keeps characters whose UTF-8 bytes a byte-level
BPE split across tokens (Whisper's CJK and emoji), in every response path.

Self-contained so it runs without ``soundfile``/``torchcodec``: WAV duration
and decoding for the long-form window planner come from the stdlib here.
"""

import io
import json
import sys
import tempfile
import types
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server import media_io  # noqa: E402
from mstar.api_server.openai import serving_transcriptions  # noqa: E402

# a leading space, five 3-byte CJK characters and a 4-byte emoji
TEXT = " 真正的存在😀"


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _StubAPI:
    def __init__(self):
        self.model_name = "whisper_large"
        self.model = None
        self.upload_dir = Path(tempfile.mkdtemp())
        self.submits: list = []
        self.queued_chunks: list = []
        self._chunks: dict = {}

    def submit_request(self, **kw):
        self.submits.append(kw)
        self._chunks[kw["request_id"]] = self.queued_chunks.pop(0)
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks[request_id]

    async def iter_result_chunks(self, request_id):
        for c in self._chunks[request_id]:
            yield c

    def release_request(self, request_id):
        pass


def _wav_seconds(path: str) -> float | None:
    try:
        with wave.open(path) as wf:
            return wf.getnframes() / wf.getframerate()
    except (wave.Error, EOFError):
        return None


def _read_wav(path: str, sample_rate: int = 16_000, max_seconds: float | None = None):
    with wave.open(path) as wf:
        assert wf.getframerate() == sample_rate
        return np.frombuffer(wf.readframes(wf.getnframes()), "<i2").astype(np.float32) / 32768


@pytest.fixture
def client_and_stub(monkeypatch):
    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    stub = _StubAPI()
    fake_ep.api_server = stub
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    monkeypatch.setattr(serving_transcriptions, "audio_duration_seconds", _wav_seconds)
    monkeypatch.setattr(media_io, "decode_audio", _read_wav)

    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), stub


def _text(*parts):
    return [_Chunk("text", p.encode("utf-8")) for p in parts]


def _split(text: str, size: int = 2):
    """``text`` as chunks of ``size`` bytes, which cuts every multi-byte
    character of :data:`TEXT` across two chunks."""
    data = text.encode("utf-8")
    return [_Chunk("text", data[i:i + size]) for i in range(0, len(data), size)]


def _wav_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x01" * int(seconds * sample_rate))
    return buf.getvalue()


def _post(client, data, seconds: float = 5.0):
    files = {"file": ("speech.wav", _wav_bytes(seconds), "audio/wav")}
    return client.post("/v1/audio/transcriptions", data={"model": "whisper_large", **data}, files=files)


def _stream_events(client, data, seconds: float = 5.0) -> list[dict]:
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true", **data},
        files={"file": ("speech.wav", _wav_bytes(seconds), "audio/wav")},
    ) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    return [json.loads(ln[len("data: "):]) for ln in lines[:-1]]


def _deltas(events: list[dict]) -> list[str]:
    return [e["delta"] for e in events if e["type"] == "transcript.text.delta"]


def test_json_keeps_characters_split_across_chunks(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|zh|>") + _split(TEXT)]
    r = _post(client, {})
    assert r.status_code == 200, r.text
    assert r.json() == {"text": TEXT.strip()}


def test_verbose_json_segments_keep_characters_split_across_chunks(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|zh|>", "<|0.00|>") + _split(TEXT) + _text("<|1.50|>")]
    body = _post(client, {"response_format": "verbose_json"}).json()
    assert body["language"] == "zh" and body["text"] == TEXT.strip()
    assert [(s["start"], s["end"], s["text"]) for s in body["segments"]] == [(0.0, 1.5, TEXT.strip())]


def test_stream_deltas_carry_only_whole_characters(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|zh|>") + _split(TEXT)]
    events = _stream_events(client, {})
    deltas = _deltas(events)
    assert all("\ufffd" not in d for d in deltas), deltas
    assert "".join(deltas) == TEXT
    assert events[-1]["type"] == "transcript.text.done" and events[-1]["text"] == TEXT.strip()


def test_sequential_long_form_keeps_split_characters_per_window(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [
        _text("<|zh|>", "<|0.00|>") + _split(" 真正的") + _text("<|29.50|>"),
        _text("<|0.00|>") + _split(" 存在😀") + _text("<|15.00|>"),
    ]
    body = _post(client, {"response_format": "verbose_json"}, seconds=45).json()
    assert body["text"] == "真正的 存在😀"
    assert [s["text"] for s in body["segments"]] == ["真正的", "存在😀"]
    assert len(stub.submits) == 2
    assert stub.submits[1]["model_kwargs"]["initial_prompt"] == "真正的"


def test_streaming_parallel_windows_keep_split_characters(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|zh|>") + _split(" 真正的"), _split(" 存在😀")]
    events = _stream_events(client, {"long_form": "parallel"}, seconds=45)
    assert "".join(_deltas(events)) == " 真正的" + " " + " 存在😀"
    assert events[-1]["text"] == "真正的 存在😀"


def test_a_truncated_character_does_not_continue_into_the_next_window(client_and_stub):
    # window 1 stops mid-character; window 2 opening on a stray continuation
    # byte is its own request and must not complete it
    cut = " 真".encode("utf-8")
    windows = [
        [_Chunk("text", cut[:2]), _Chunk("text", cut[2:3])],
        [_Chunk("text", cut[3:]), _Chunk("text", " 正".encode("utf-8"))],
    ]
    client, stub = client_and_stub
    stub.queued_chunks = list(windows)
    assert _post(client, {"long_form": "parallel"}, seconds=45).json() == {"text": "\ufffd \ufffd 正"}
    stub.queued_chunks = list(windows)
    events = _stream_events(client, {"long_form": "parallel"}, seconds=45)
    assert "".join(_deltas(events)) == " \ufffd \ufffd 正"
    assert events[-1]["text"] == "\ufffd \ufffd 正"
