"""``/v1/audio/transcriptions``: adapters, response formats, streaming events.

The router is mounted on a FastAPI app with a stubbed APIServer, like
``test_openai_router.py``; the adapters are exercised directly as well.
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mstar.api_server.openai import adapters, serving_transcriptions  # noqa: E402
from mstar.api_server.openai.protocol import TranscriptionRequest  # noqa: E402


class _Chunk:
    def __init__(self, modality, data, metadata=None):
        self.modality = modality
        self.data = data
        self.metadata = metadata or {}


class _StubAPI:
    def __init__(self, model_name="whisper_large"):
        self.model_name = model_name
        self.model = None
        self.upload_dir = Path(tempfile.mkdtemp())
        self.last_submit = None
        self.submits: list = []
        self._chunks: dict = {}
        self.next_chunks: list = []
        # per-submission chunk lists, consumed in order (long-form windows)
        self.queued_chunks: list = []

    def submit_request(self, **kw):
        self.last_submit = kw
        self.submits.append(kw)
        chunks = self.queued_chunks.pop(0) if self.queued_chunks else list(self.next_chunks)
        self._chunks[kw["request_id"]] = chunks
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks.get(request_id, [])

    async def iter_result_chunks(self, request_id):
        for c in self._chunks.get(request_id, []):
            yield c


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


def _text(*parts):
    return [_Chunk("text", p.encode("utf-8")) for p in parts]


def _wav_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x01" * int(seconds * sample_rate))
    return buf.getvalue()


def _post(client, data=None, filename="speech.wav", content=b"RIFFfake"):
    files = {"file": (filename, content, "audio/wav")} if filename else None
    return client.post("/v1/audio/transcriptions", data=data or {"model": "whisper_large"}, files=files)


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


def test_registry_has_asr_models():
    assert {"whisper_large", "higgs_audio"} <= set(adapters.ADAPTER_REGISTRY)
    assert adapters.get_adapter("whisper_large").supports_transcriptions
    assert not adapters.get_adapter("bagel").supports_transcriptions


def test_whisper_request_maps_openai_fields():
    req = TranscriptionRequest(
        model="whisper_large", language="de", prompt="Guten Tag", temperature=0.2, seed=3,
        response_format="json",
    )
    sa = adapters.WhisperAdapter().transcription_to_request(req, "/tmp/a.wav")
    assert sa.file_paths == {"audio": ["/tmp/a.wav"]}
    assert sa.input_modalities == ["audio", "text"] and sa.output_modalities == ["text"]
    assert sa.text == ""
    assert sa.model_kwargs == {"language": "de", "initial_prompt": "Guten Tag", "temperature": 0.2, "seed": 3}


def test_timed_formats_ask_for_timestamps():
    ad = adapters.WhisperAdapter()
    plain = TranscriptionRequest(response_format="json")
    assert "timestamps" not in ad.transcription_to_request(plain, "a").model_kwargs
    for fmt in ("verbose_json", "srt", "vtt"):
        req = TranscriptionRequest(response_format=fmt)
        assert ad.transcription_to_request(req, "a").model_kwargs["timestamps"] == "segment"
    words = TranscriptionRequest(response_format="verbose_json", timestamp_granularities=["word", "segment"])
    assert ad.transcription_to_request(words, "a").model_kwargs["timestamps"] == "word"


def test_extra_fields_pass_through_as_model_kwargs():
    req = TranscriptionRequest(model="whisper_large", beam_size=4, task="translate")
    mk = adapters.WhisperAdapter().transcription_to_request(req, "a").model_kwargs
    assert mk["beam_size"] == 4 and mk["task"] == "translate"


def test_whisper_parse_lifts_language_and_segments():
    req = TranscriptionRequest(response_format="verbose_json")
    raw = "<|en|><|0.00|> Hello there.<|1.50|><|1.50|> How are you?<|3.20|>"
    t = adapters.WhisperAdapter().parse_transcript(raw, req)
    assert t.language == "en"
    assert t.text == "Hello there. How are you?"
    assert t.segments == [
        {"start": 0.0, "end": 1.5, "text": "Hello there.", "id": 0},
        {"start": 1.5, "end": 3.2, "text": "How are you?", "id": 1},
    ]


def test_whisper_parse_plain_text_and_forced_language():
    req = TranscriptionRequest(language="fr")
    t = adapters.WhisperAdapter().parse_transcript(" Bonjour tout le monde ", req)
    assert t.text == "Bonjour tout le monde" and t.language == "fr" and t.segments == []


def test_whisper_parse_keeps_text_of_an_unterminated_segment():
    req = TranscriptionRequest()
    t = adapters.WhisperAdapter().parse_transcript("<|en|><|0.00|> cut off", req)
    assert t.text == "cut off" and t.segments == []


def test_whisper_stream_delta_hides_control_tokens():
    ad = adapters.WhisperAdapter()
    assert ad.stream_delta("<|en|>") == ""
    assert ad.stream_delta("<|0.00|> Hello") == " Hello"
    assert ad.stream_delta(" world") == " world"


def test_higgs_request_uses_prompt_as_instruction():
    req = TranscriptionRequest(prompt="Transcribe verbatim.", language="en")
    sa = adapters.HiggsAudioAdapter().transcription_to_request(req, "/tmp/a.wav")
    assert sa.text == "Transcribe verbatim."
    assert "initial_prompt" not in sa.model_kwargs and sa.model_kwargs["language"] == "en"


# --------------------------------------------------------------------------
# serving helpers
# --------------------------------------------------------------------------


def test_srt_and_vtt_rendering():
    segs = [{"start": 0.0, "end": 1.5, "text": " Hi "}, {"start": 61.25, "end": 3661.0, "text": "Bye"}]
    srt = serving_transcriptions.render_srt(segs)
    assert srt.splitlines()[:3] == ["1", "00:00:00,000 --> 00:00:01,500", "Hi"]
    assert "00:01:01,250 --> 01:01:01,000" in srt
    vtt = serving_transcriptions.render_vtt(segs)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHi")


def test_save_upload_keeps_extension_and_drops_directories(tmp_path):
    path = serving_transcriptions.save_upload(b"x", "../../evil.flac", tmp_path)
    assert Path(path).parent == tmp_path and path.endswith("_evil.flac")
    assert Path(path).read_bytes() == b"x"


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------


def test_transcription_json(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text("<|en|>", " Hello", " world")
    r = _post(client, {"model": "whisper_large", "language": "en", "temperature": "0"})
    assert r.status_code == 200
    assert r.json() == {"text": "Hello world"}
    submit = stub.last_submit
    assert submit["input_modalities"] == ["audio", "text"]
    assert submit["model_kwargs"]["language"] == "en"
    assert submit["model_kwargs"]["temperature"] == 0
    assert submit["streaming"] is False
    (saved,) = submit["file_paths"]["audio"]
    assert Path(saved).read_bytes() == b"RIFFfake" and saved.endswith("_speech.wav")


def test_transcription_text_format(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text(" Plain", " words")
    r = _post(client, {"model": "whisper_large", "response_format": "text"})
    assert r.status_code == 200
    assert r.text == "Plain words"
    assert r.headers["content-type"].startswith("text/plain")


def test_transcription_verbose_json_segments(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text("<|de|>", "<|0.00|>", " Hallo", "<|0.80|>", "<|0.80|>", " Welt", "<|1.40|>")
    r = _post(client, {
        "model": "whisper_large", "response_format": "verbose_json",
        "timestamp_granularities[]": ["segment", "word"],
    })
    body = r.json()
    assert body["task"] == "transcribe" and body["language"] == "de"
    assert body["text"] == "Hallo Welt"
    assert [(s["start"], s["end"], s["text"]) for s in body["segments"]] == [
        (0.0, 0.8, "Hallo"), (0.8, 1.4, "Welt"),
    ]
    assert body["words"] == []
    assert body["usage"]["type"] == "duration"
    assert stub.last_submit["model_kwargs"]["timestamps"] == "word"


def test_transcription_srt(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text("<|0.00|> One<|1.00|>")
    r = _post(client, {"model": "whisper_large", "response_format": "srt"})
    assert r.text == "1\n00:00:00,000 --> 00:00:01,000\nOne\n"


def test_transcription_stream_events(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text("<|en|>", " Hello", " world", "<|eot|>")
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true"},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    assert [e["type"] for e in events] == [
        "transcript.text.delta", "transcript.text.delta", "transcript.text.done",
    ]
    assert [e["delta"] for e in events[:2]] == [" Hello", " world"]
    assert events[-1]["text"] == "Hello world"
    assert stub.last_submit["streaming"] is True


def test_transcription_stream_reports_in_band_error(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = [_Chunk("error", b"boom", {"status": 500})]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true"},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert json.loads(lines[0][len("data: "):])["type"] == "error"
    assert lines[-1] == "data: [DONE]"


def test_transcription_requires_a_file(client_and_stub):
    client, _ = client_and_stub
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper_large"})
    assert r.status_code == 400
    assert "file" in r.json()["error"]["message"]


def test_transcription_rejects_unknown_format(client_and_stub):
    client, stub = client_and_stub
    r = _post(client, {"model": "whisper_large", "response_format": "xml"})
    assert r.status_code == 400
    assert stub.last_submit is None


def test_transcription_404_for_models_without_the_surface(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    r = _post(client)
    assert r.status_code == 404


# --------------------------------------------------------------------------
# long form
# --------------------------------------------------------------------------


def test_plan_windows_cuts_only_bounded_models(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(16_000 * 75, dtype="float32"), 16_000)
    whole = serving_transcriptions.plan_windows(str(path), None, tmp_path)
    assert len(whole) == 1 and whole[0].path == str(path) and whole[0].duration == pytest.approx(75.0)
    windows = serving_transcriptions.plan_windows(str(path), 30.0, tmp_path)
    assert [w.offset for w in windows] == [0.0, 30.0, 60.0]
    assert [round(w.duration, 3) for w in windows] == [30.0, 30.0, 15.0]
    assert all(Path(w.path).exists() for w in windows)
    short = serving_transcriptions.plan_windows(str(path), 80.0, tmp_path)
    assert len(short) == 1


def test_sequential_long_form_carries_language_and_text(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [
        _text("<|en|>", "<|0.00|>", " First window.", "<|29.00|>"),
        _text("<|0.00|>", " Second window.", "<|10.00|>"),
        _text("<|0.00|>", " Tail.", "<|5.00|>"),
    ]
    r = _post(client, {"model": "whisper_large", "response_format": "verbose_json"},
              filename="long.wav", content=_wav_bytes(65))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "First window. Second window. Tail."
    assert body["language"] == "en"
    assert [(s["start"], s["end"]) for s in body["segments"]] == [(0.0, 29.0), (30.0, 40.0), (60.0, 65.0)]
    assert body["duration"] == pytest.approx(65.0)
    assert len(stub.submits) == 3
    # window 2 and 3 are conditioned on the transcript so far and the detected language
    assert "initial_prompt" not in stub.submits[0]["model_kwargs"]
    assert stub.submits[1]["model_kwargs"]["initial_prompt"] == "First window."
    assert stub.submits[1]["model_kwargs"]["language"] == "en"
    assert stub.submits[2]["model_kwargs"]["initial_prompt"] == "First window. Second window."
    # each window is its own upload
    assert len({s["file_paths"]["audio"][0] for s in stub.submits}) == 3


def test_parallel_long_form_submits_everything_up_front(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text(" one"), _text(" two"), _text(" three")]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel", "language": "en"},
              filename="long.wav", content=_wav_bytes(61))
    assert r.json() == {"text": "one two three"}
    assert len(stub.submits) == 3
    assert all("initial_prompt" not in s["model_kwargs"] for s in stub.submits)
    assert all("long_form" not in s["model_kwargs"] for s in stub.submits)


def test_long_form_rejects_unknown_mode(client_and_stub):
    client, stub = client_and_stub
    r = _post(client, {"model": "whisper_large", "long_form": "zigzag"}, filename="a.wav", content=_wav_bytes(1))
    assert r.status_code == 400 and not stub.submits


def test_short_upload_is_a_single_request(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text(" short")
    r = _post(client, {"model": "whisper_large"}, filename="short.wav", content=_wav_bytes(12))
    assert r.json() == {"text": "short"} and len(stub.submits) == 1


def test_streaming_long_form_emits_every_window_in_order(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|en|>", " one"), _text(" two")]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true"},
        files={"file": ("long.wav", _wav_bytes(45), "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    events = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    deltas = [e["delta"] for e in events if e["type"] == "transcript.text.delta"]
    assert deltas == [" one", " ", " two"]
    assert events[-1]["type"] == "transcript.text.done" and events[-1]["text"] == "one two"
    assert stub.submits[1]["model_kwargs"]["initial_prompt"] == "one"
    assert stub.submits[1]["model_kwargs"]["language"] == "en"
