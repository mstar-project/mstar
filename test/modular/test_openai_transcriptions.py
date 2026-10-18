"""``/v1/audio/transcriptions``: adapters, response formats, streaming events.

The router is mounted on a FastAPI app with a stubbed APIServer, like
``test_openai_router.py``; the adapters are exercised directly as well.
"""

import json
import re
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
        self.released: list = []
        self.wav_seconds: list = []

    def submit_request(self, **kw):
        self.last_submit = kw
        self.submits.append(kw)
        # window files are deleted once the request is done
        path = Path(kw["file_paths"]["audio"][0])
        if re.search(r"_w\d{4}$", path.stem):
            self.wav_seconds.append(_wav_seconds(str(path)))
        chunks = self.queued_chunks.pop(0) if self.queued_chunks else list(self.next_chunks)
        self._chunks[kw["request_id"]] = chunks
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return self._chunks.get(request_id, [])

    async def iter_result_chunks(self, request_id):
        for c in self._chunks.get(request_id, []):
            yield c

    def release_request(self, request_id):
        self.released.append(request_id)


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


def _can_read_audio_duration() -> bool:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(_wav_bytes(1))
        f.flush()
        return serving_transcriptions.audio_duration_seconds(f.name) is not None


# Long-form windowing reads the upload's duration with soundfile or torchcodec;
# without a working one, every upload is one window.
requires_audio_duration = pytest.mark.skipif(
    not _can_read_audio_duration(), reason="long-form windowing needs soundfile or torchcodec"
)


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


def test_whisper_stream_text_hides_control_tokens():
    ad = adapters.WhisperAdapter()
    assert ad.stream_text("<|en|>") == ""
    assert ad.stream_text("<|en|><|0.00|> Hello") == " Hello"
    assert ad.stream_text("<|en|><|0.00|> Hello world") == " Hello world"


def test_whisper_stream_text_holds_back_a_split_control_token():
    ad = adapters.WhisperAdapter()
    assert ad.stream_text("<|e") == ""
    assert ad.stream_text("<|en|><|0.00|> Hi<") == " Hi"
    assert ad.stream_text("<|en|><|0.00|> Hi<|1.0") == " Hi"
    assert ad.stream_text("<|en|><|0.00|> Hi<|1.00|>") == " Hi"


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
    # deleted once the request is done
    assert saved.endswith("_speech.wav") and not Path(saved).exists()


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


@pytest.mark.parametrize("status,error_type", [(500, "server_error"), (400, "invalid_request_error")])
def test_transcription_stream_reports_in_band_error(client_and_stub, status, error_type):
    client, stub = client_and_stub
    stub.next_chunks = [_Chunk("error", b"boom", {"status": status})]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true"},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    event = json.loads(lines[0][len("data: "):])
    assert event["type"] == "error"
    assert event["error"] == {"message": "boom", "type": error_type, "code": status}, "the type ignores the status"
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


def test_transcription_404_for_models_without_the_endpoint(client_and_stub):
    client, stub = client_and_stub
    stub.model_name = "bagel"
    r = _post(client)
    assert r.status_code == 404


def test_whisper_words_follow_the_transcript_after_the_marker():
    ad = adapters.WhisperAdapter()
    req = TranscriptionRequest(model="whisper_large", response_format="verbose_json",
                               timestamp_granularities=["word"])
    raw = ("<|en|><|0.00|> Hello there.<|1.20|>"
           "<|startoflm|><|0.02|>Hello<|0.60|><|0.60|>there.<|1.18|>")
    t = ad.parse_transcript(raw, req)
    assert t.text == "Hello there." and t.language == "en"
    assert [(s["start"], s["end"], s["text"]) for s in t.segments] == [(0.0, 1.2, "Hello there.")]
    assert t.words == [
        {"word": "Hello", "start": 0.02, "end": 0.6},
        {"word": "there.", "start": 0.6, "end": 1.18},
    ]
    assert not t.unfinished
    # streaming clients never see the timings
    assert ad.stream_text(raw) == " Hello there."


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


def test_plan_windows_drops_a_tail_too_short_to_hold_speech(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    path = tmp_path / "long.wav"
    # silence cuts at the boundary, leaving 10 ms that a model would refuse
    sf.write(path, np.zeros(16_000 * 60 + 160, dtype="float32"), 16_000)
    windows = serving_transcriptions.plan_windows(str(path), 30.0, tmp_path)
    assert [round(w.duration, 3) for w in windows] == [30.0, 30.0]
    planner = serving_transcriptions.WindowPlanner(str(path), 30.0, tmp_path)
    assert planner.done(60.0) and not planner.done(59.9)


@pytest.mark.parametrize("adapter", [adapters.HiggsAudioAdapter(), adapters.WhisperAdapter()], ids=["higgs", "whisper"])
def test_windows_without_timestamps_are_one_segment_each(adapter):
    """A window that decoded no timestamps (Higgs never emits them, and M*
    Whisper decodes with ``<|notimestamps|>``) is one segment spanning it, so
    the captions are timed to the windows."""
    req = TranscriptionRequest(model="higgs_audio", response_format="srt")
    windows = [
        serving_transcriptions.Window(path="", offset=0.0, duration=30.0, raw_text="one"),
        serving_transcriptions.Window(path="", offset=30.0, duration=15.0, raw_text="two"),
    ]
    t = serving_transcriptions.merge_windows(windows, adapter, req)
    assert t.text == "one two"
    assert [(s["start"], s["end"], s["text"]) for s in t.segments] == [(0.0, 30.0, "one"), (30.0, 45.0, "two")]


def test_cleanup_while_a_thread_cuts_leaves_no_files(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(16_000 * 75, dtype="float32"), 16_000)
    planner = serving_transcriptions.WindowPlanner(str(path), 30.0, tmp_path)
    write = serving_transcriptions.media_io.write_wav

    def disconnect_after_a_window(*args, **kwargs):
        # a client disconnect runs the request's cleanup while the thread cuts on
        out = write(*args, **kwargs)
        planner.cleanup()
        return out

    monkeypatch.setattr(serving_transcriptions.media_io, "write_wav", disconnect_after_a_window)
    with pytest.raises(RuntimeError):
        planner.fixed()
    assert not list(tmp_path.iterdir())


@requires_audio_duration
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


@requires_audio_duration
def test_parallel_long_form_submits_everything_up_front(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text(" one"), _text(" two"), _text(" three")]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel", "language": "en"},
              filename="long.wav", content=_wav_bytes(61))
    assert r.json() == {"text": "one two three"}
    assert len(stub.submits) == 3
    assert all("initial_prompt" not in s["model_kwargs"] for s in stub.submits)
    assert all("long_form" not in s["model_kwargs"] for s in stub.submits)



@requires_audio_duration
def test_parallel_timed_window_that_stops_inside_a_segment_decodes_its_tail(client_and_stub):
    """Whisper writes the next segment's start near a window's end and stops;
    in parallel mode the next window starts at the cut, so the audio from the
    last closed segment to the cut is decoded as a window of its own."""
    client, stub = client_and_stub
    stub.queued_chunks = [
        _text("<|en|>", "<|0.00|>", " One.", "<|20.00|>", "<|20.40|>", " cut off"),
        _text("<|0.00|>", " Two.", "<|29.00|>"),
        _text("<|0.00|>", " Three.", "<|1.00|>"),
        _text("<|0.00|>", " The tail.", "<|4.00|>"),
    ]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel", "language": "en",
                       "response_format": "verbose_json"}, filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 200, r.text
    body = r.json()
    # the open segment's partial text gives way to the tail's decode
    assert body["text"] == "One. The tail. Two. Three."
    segments = [(s["start"], s["end"], s["text"]) for s in body["segments"]]
    assert segments[:2] == [(0.0, 20.0, "One."), (20.0, 24.0, "The tail.")]
    assert len(stub.submits) == 4
    first, tail = stub.wav_seconds[0], stub.wav_seconds[3]
    assert tail == pytest.approx(first - 20.0, abs=1e-3)


@requires_audio_duration
@pytest.mark.parametrize("last_end", ["0.50", "29.80"])
def test_parallel_unfinished_window_keeps_its_text_when_there_is_no_tail_to_decode(client_and_stub, last_end):
    """A seek point under a second in, or a tail under half a second, is left
    as it is, as sequential mode does with a seek under a second."""
    client, stub = client_and_stub
    stub.queued_chunks = [
        _text("<|en|>", "<|0.00|>", " One.", f"<|{last_end}|>", f"<|{last_end}|>", " open"),
        _text("<|0.00|>", " Two.", "<|29.00|>"),
        _text("<|0.00|>", " Three.", "<|1.00|>"),
    ]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel", "language": "en",
                       "response_format": "verbose_json"}, filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "One. open Two. Three."
    assert len(stub.submits) == 3


@requires_audio_duration
def test_client_gone_after_a_parallel_tail_releases_it_and_submits_nothing_more(client_and_stub, monkeypatch):
    """A window's tail is submitted as soon as that window is in. Once the
    client is found gone, nothing more is submitted and every request still
    on the engine, the tail included, is released."""
    from starlette.requests import Request

    client, stub = client_and_stub
    gone = False
    collect = stub.collect_results

    async def collect_until_gone(request_id, raw_request=None):
        nonlocal gone
        if len(stub.submits) > 1 and stub.submits[1]["request_id"] == request_id:
            gone = True
            return []
        return await collect(request_id, raw_request)

    async def is_disconnected(self):
        return gone

    stub.collect_results = collect_until_gone
    monkeypatch.setattr(Request, "is_disconnected", is_disconnected)
    # window 0 stopped inside a segment, so it has a tail to decode
    stub.queued_chunks = [_text("<|en|>", "<|0.00|>", " One.", "<|20.00|>", "<|20.40|>", " cut off")]
    stub.queued_chunks += [_text("<|0.00|>", f" w{i}.", "<|29.00|>") for i in range(1, 30)]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel", "language": "en",
                       "response_format": "verbose_json"}, filename="long.wav", content=_wav_bytes(10 * 60))
    assert r.status_code == 200, r.text
    # the first window's tail went out when that window came back, before the disconnect
    assert len(stub.submits) == 2 + serving_transcriptions.PARALLEL_INFLIGHT
    # the tail's file is cut after every fixed window's, so it has the highest _wNNNN
    tail = max(stub.submits, key=lambda sub: sub["file_paths"]["audio"][0])["request_id"]
    assert tail in stub.released
    assert not list(stub.upload_dir.iterdir())


def test_long_form_rejects_unknown_mode(client_and_stub):
    client, stub = client_and_stub
    r = _post(client, {"model": "whisper_large", "long_form": "zigzag"}, filename="a.wav", content=_wav_bytes(1))
    assert r.status_code == 400 and not stub.submits


def test_short_upload_is_a_single_request(client_and_stub):
    client, stub = client_and_stub
    stub.next_chunks = _text(" short")
    r = _post(client, {"model": "whisper_large"}, filename="short.wav", content=_wav_bytes(12))
    assert r.json() == {"text": "short"} and len(stub.submits) == 1


@requires_audio_duration
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
    # sequential windows stream once accepted, as their clean text
    assert deltas == ["one", " ", "two"]
    assert events[-1]["type"] == "transcript.text.done" and events[-1]["text"] == "one two"
    assert stub.submits[1]["model_kwargs"]["initial_prompt"] == "one"
    assert stub.submits[1]["model_kwargs"]["language"] == "en"


@requires_audio_duration
def test_streaming_parallel_windows_stream_token_by_token(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|en|>", " one", " two"), _text(" three")]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true", "long_form": "parallel"},
        files={"file": ("long.wav", _wav_bytes(45), "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    events = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    deltas = [e["delta"] for e in events if e["type"] == "transcript.text.delta"]
    assert deltas == [" one", " two", " ", " three"]
    assert events[-1]["text"] == "one two three"


@requires_audio_duration
def test_streamed_parallel_windows_decode_without_timestamps(client_and_stub):
    """A stream carries text only; a window decoded with timestamps could stop
    early and leave its tail to the next window, which in parallel mode starts
    at the cut."""
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|en|>", " one"), _text(" two")]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true", "long_form": "parallel",
              "response_format": "verbose_json"},
        files={"file": ("long.wav", _wav_bytes(45), "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert json.loads(lines[-2][len("data: "):])["text"] == "one two"
    assert len(stub.submits) == 2
    assert all("timestamps" not in s["model_kwargs"] for s in stub.submits)


@pytest.mark.parametrize("long_form, expected", [
    ("parallel", ["one", " two", " ", "three"]),
    ("sequential", ["one two", " ", "three"]),
])
def test_streamed_deltas_add_up_to_the_final_text(client_and_stub, monkeypatch, long_form, expected):
    """A SentencePiece model's text starts with a space, and a silent window
    has no text: the deltas still join to exactly the done event's text."""
    client, stub = client_and_stub
    stub.model_name = "nemotron_asr"
    monkeypatch.setattr(adapters.TransducerAdapter, "max_audio_seconds", 20.0)
    stub.queued_chunks = [_text(" one", " two"), [], _text(" three")]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "nemotron_asr", "stream": "true", "long_form": long_form},
        files={"file": ("long.wav", _wav_bytes(45), "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    events = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
    deltas = [e["delta"] for e in events if e["type"] == "transcript.text.delta"]
    assert len(stub.submits) == 3
    assert deltas == expected
    assert "".join(deltas) == events[-1]["text"] == "one two three"


def _wav_seconds(path: str) -> float:
    import wave

    with wave.open(path) as wf:
        return wf.getnframes() / wf.getframerate()


@requires_audio_duration
def test_sequential_seeks_to_the_last_closed_segment(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [
        # stopped inside a segment: "Sec" is provisional and 20 s is where to resume
        _text("<|en|>", "<|0.00|>", " First.", "<|20.00|>", "<|20.00|>", " Sec"),
        _text("<|0.00|>", " Second.", "<|29.00|>"),
        _text("<|0.00|>", " Third.", "<|15.00|>"),
    ]
    r = _post(client, {"model": "whisper_large", "response_format": "verbose_json"},
              filename="long.wav", content=_wav_bytes(65))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "First. Second. Third."
    assert [(s["start"], s["end"]) for s in body["segments"]] == [(0.0, 20.0), (20.0, 49.0), (50.0, 65.0)]
    assert len(stub.submits) == 3
    # the unfinished segment neither reaches the transcript nor conditions the next window
    assert stub.submits[1]["model_kwargs"]["initial_prompt"] == "First."
    assert stub.submits[2]["model_kwargs"]["initial_prompt"] == "First. Second."
    # window 2 starts at 20 s and is a full window, window 3 starts where it ended (50 s)
    assert [round(t, 2) for t in stub.wav_seconds] == [30.0, 30.0, 15.0]


@requires_audio_duration
def test_sequential_json_decodes_windows_with_timestamps(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text("<|en|>", "<|0.00|>", " one", "<|29.50|>"), _text("<|0.00|>", " two", "<|15.00|>")]
    r = _post(client, {"model": "whisper_large"}, filename="long.wav", content=_wav_bytes(45))
    assert r.json() == {"text": "one two"}
    assert all(s["model_kwargs"]["timestamps"] == "segment" for s in stub.submits)


@requires_audio_duration
def test_short_seek_or_no_closed_segment_takes_the_whole_window(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [
        _text("<|en|>", "<|0.00|>", " open ended"),                  # no closed segment
        _text("<|0.00|>", " a", "<|0.50|>", "<|0.50|>", " tail"),    # closed too early to seek to
        _text("<|0.00|>", " last", "<|5.00|>"),
    ]
    r = _post(client, {"model": "whisper_large"}, filename="long.wav", content=_wav_bytes(65))
    assert r.json() == {"text": "open ended a tail last"}
    assert [round(t, 2) for t in stub.wav_seconds] == [30.0, 30.0, 5.0]


@requires_audio_duration
def test_repetitive_window_first_drops_the_prompt_then_warms_up(client_and_stub):
    client, stub = client_and_stub
    loop = " the apostle and" * 60
    assert serving_transcriptions.compression_ratio(loop) > 2.4
    assert serving_transcriptions.compression_ratio("He hoped there would be stew for dinner.") < 2.4
    stub.queued_chunks = [
        _text("<|en|>", "<|0.00|>", " Hello there.", "<|20.00|>"),
        _text("<|0.00|>", loop, "<|29.00|>"),           # window 2: greedy with the prompt loops
        _text("<|0.00|>", loop, "<|29.00|>"),           # greedy without it still loops
        _text("<|0.00|>", " Fine now.", "<|10.00|>"),   # 0.2 without the prompt: accepted
        _text("<|0.00|>", " The end.", "<|15.00|>"),
    ]
    r = _post(client, {"model": "whisper_large", "response_format": "verbose_json"},
              filename="long.wav", content=_wav_bytes(75))
    body = r.json()
    assert body["text"] == "Hello there. Fine now. The end."
    assert [s["model_kwargs"]["temperature"] for s in stub.submits] == [0.0, 0.0, 0.0, 0.2, 0.0]
    assert [s["model_kwargs"].get("initial_prompt") for s in stub.submits] == [
        None, "Hello there.", None, None, "Hello there. Fine now.",
    ]
    assert body["segments"][1]["temperature"] == 0.2 and body["segments"][1]["compression_ratio"] < 2.4


@requires_audio_duration
def test_window_accepted_hot_stops_conditioning_the_next(client_and_stub):
    client, stub = client_and_stub
    loop = " the apostle and" * 60
    stub.queued_chunks = [_text("<|en|>", "<|0.00|>", " Hello there.", "<|20.00|>")]
    stub.queued_chunks += [_text("<|0.00|>", loop, "<|29.00|>")] * 5   # prompt, no prompt, 0.2, 0.4, 0.6
    stub.queued_chunks += [_text("<|0.00|>", " Warm.", "<|10.00|>"), _text("<|0.00|>", " The end.", "<|15.00|>")]
    r = _post(client, {"model": "whisper_large"}, filename="long.wav", content=_wav_bytes(75))
    assert r.json() == {"text": "Hello there. Warm. The end."}
    assert [s["model_kwargs"]["temperature"] for s in stub.submits] == [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0]
    # the window accepted at 0.8 does not condition the one after it
    assert "initial_prompt" not in stub.submits[7]["model_kwargs"]


@requires_audio_duration
def test_pinned_temperature_is_not_escalated(client_and_stub):
    client, stub = client_and_stub
    loop = " the apostle and" * 60
    stub.queued_chunks = [_text("<|en|>", "<|0.00|>", loop, "<|29.00|>"), _text("<|0.00|>", " two", "<|15.00|>")]
    r = _post(client, {"model": "whisper_large", "temperature": "0.3"}, filename="long.wav", content=_wav_bytes(45))
    assert r.status_code == 200
    # window 1 has no prompt to drop, so its loop is kept as is and nothing escalates
    assert [s["model_kwargs"]["temperature"] for s in stub.submits] == [0.3, 0.3]


def test_split_windows_cuts_at_the_quietest_point():
    import numpy as np

    from mstar.api_server import media_io

    audio = np.ones(16_000 * 75, dtype="float32")
    audio[16_000 * 28: 16_000 * 28 + 8_000] = 0.0  # a pause at 28.0-28.5 s
    pieces = media_io.split_windows(audio, 30.0, 16_000, search_seconds=2.0)
    assert [len(p) / 16_000 for p in pieces] == [28.5, 30.0, 16.5]
    assert sum(len(p) for p in pieces) == len(audio)
    fixed = media_io.split_windows(audio, 30.0, 16_000)
    assert [len(p) / 16_000 for p in fixed] == [30.0, 30.0, 15.0]


# --------------------------------------------------------------------------
# hardening
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("top_p", "abc"), ("max_output_tokens", "abc"), ("top_k", "1.5"), ("top_k", "true"),
])
def test_non_numeric_sampling_kwargs_are_rejected_before_submission(client_and_stub, field, value):
    client, stub = client_and_stub
    r = _post(client, {"model": "whisper_large", field: value}, filename="a.wav", content=_wav_bytes(1))
    assert r.status_code == 400 and field in r.json()["error"]["message"]
    assert not stub.submits


def test_non_numeric_sampling_kwargs_are_rejected_before_a_stream_opens(client_and_stub):
    client, stub = client_and_stub
    r = _post(client, {"model": "whisper_large", "stream": "true", "top_p": "abc"},
              filename="a.wav", content=_wav_bytes(1))
    assert r.status_code == 400 and not stub.submits


def test_numeric_strings_are_coerced():
    req = TranscriptionRequest(top_p="0.9", max_output_tokens="100")
    mk = adapters.WhisperAdapter().transcription_to_request(req, "/tmp/a.wav").model_kwargs
    assert mk["top_p"] == 0.9 and mk["max_output_tokens"] == 100


def test_integral_float_form_field_arrives_as_an_int(client_and_stub):
    client, stub = client_and_stub
    r = _post(client, {"model": "whisper_large", "top_k": "5.0"}, filename="a.wav", content=_wav_bytes(1))
    assert r.status_code == 200
    top_k = stub.last_submit["model_kwargs"]["top_k"]
    assert top_k == 5 and type(top_k) is int


def test_upload_and_window_files_are_deleted(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text(" one"), _text(" two"), _text(" three")]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel"},
              filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 200
    assert not list(stub.upload_dir.iterdir())


def test_streamed_upload_files_are_deleted(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [_text(" one"), _text(" two")]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true"},
        files={"file": ("long.wav", _wav_bytes(45), "audio/wav")},
    ) as r:
        list(r.iter_lines())
    assert not list(stub.upload_dir.iterdir())


@requires_audio_duration
def test_parallel_bounds_windows_in_flight(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setattr(serving_transcriptions, "PARALLEL_INFLIGHT", 2)
    seen = []
    collect = stub.collect_results

    async def watching(request_id, raw_request=None):
        seen.append(len(stub.submits))
        return await collect(request_id, raw_request)

    stub.collect_results = watching
    stub.queued_chunks = [_text(f" w{i}") for i in range(5)]
    r = _post(client, {"model": "whisper_large", "long_form": "parallel"},
              filename="long.wav", content=_wav_bytes(140))
    assert r.json() == {"text": "w0 w1 w2 w3 w4"}
    assert seen == [2, 3, 4, 5, 5]


def test_parallel_failure_releases_the_other_windows(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [[_Chunk("error", b"bad language", {"status": 400})], _text(" two"), _text(" three")]

    async def collect(request_id, raw_request=None):
        return serving_transcriptions._text_of(stub._chunks[request_id]) and []

    stub.collect_results = collect
    r = _post(client, {"model": "whisper_large", "long_form": "parallel"},
              filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 400
    assert sorted(stub.released) == sorted(s["request_id"] for s in stub.submits[1:])
    assert not list(stub.upload_dir.iterdir())


def test_streaming_parallel_error_releases_the_other_windows(client_and_stub):
    client, stub = client_and_stub
    stub.queued_chunks = [[_Chunk("error", b"bad language", {"status": 400})], _text(" two"), _text(" three")]
    with client.stream(
        "POST", "/v1/audio/transcriptions",
        data={"model": "whisper_large", "stream": "true", "long_form": "parallel"},
        files={"file": ("long.wav", _wav_bytes(61), "audio/wav")},
    ) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert json.loads(lines[0][len("data: "):])["type"] == "error"
    # the failed window is released too: its iterator was left mid-stream
    assert sorted(stub.released) == sorted(s["request_id"] for s in stub.submits)


@requires_audio_duration
def test_undecodable_upload_is_a_400(client_and_stub, monkeypatch):
    client, stub = client_and_stub

    from mstar.model.components import audio_features

    def refuse(*args, **kwargs):
        raise RuntimeError("Error opening 'long.wav': Format not recognised.")

    def ffmpeg_refuses(*args, **kwargs):
        raise RuntimeError("decode_av_frame: /src/torchcodec/_core/SingleStreamDecoder.cpp")

    # the header reads fine, the payload does not: libsndfile and FFmpeg (torchcodec where it works,
    # else PyAV) both refuse it, and an FFmpeg decoder is there to refuse, so the file is the client's error
    monkeypatch.setattr(serving_transcriptions.media_io, "read_soundfile_capped", refuse)
    monkeypatch.setattr(serving_transcriptions.media_io, "decode_media_capped", ffmpeg_refuses)
    monkeypatch.setattr(audio_features, "_ffmpeg_decoder_loads", lambda: True)
    data = {"model": "whisper_large", "long_form": "parallel"}
    r = _post(client, data, filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 400 and "could not decode" in r.text and "torchcodec" not in r.text
    # a stream is refused the same way, before it commits to a 200
    r = _post(client, {**data, "stream": "true"}, filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 400 and "torchcodec" not in r.text
    assert not stub.submits and not list(stub.upload_dir.iterdir())


@requires_audio_duration
def test_upload_longer_than_the_cap_is_rejected(client_and_stub, monkeypatch):
    client, stub = client_and_stub
    monkeypatch.setattr(serving_transcriptions, "MAX_DURATION_SECONDS", 60)
    r = _post(client, {"model": "whisper_large"}, filename="long.wav", content=_wav_bytes(61))
    assert r.status_code == 413 and not stub.submits
    assert not list(stub.upload_dir.iterdir())


@pytest.mark.parametrize(
    "mode, submitted", [("parallel", 1 + serving_transcriptions.PARALLEL_INFLIGHT), ("sequential", 2)],
)
def test_client_gone_mid_long_form_stops_submitting(client_and_stub, monkeypatch, mode, submitted):
    # collect_results aborts and returns [] once the client is gone; the
    # remaining windows must not be submitted after that
    from starlette.requests import Request

    client, stub = client_and_stub
    gone = False
    collect = stub.collect_results

    async def collect_until_gone(request_id, raw_request=None):
        nonlocal gone
        if len(stub.submits) > 1 and stub.submits[1]["request_id"] == request_id:
            gone = True
            return []
        return await collect(request_id, raw_request)

    async def is_disconnected(self):
        return gone

    stub.collect_results = collect_until_gone
    monkeypatch.setattr(Request, "is_disconnected", is_disconnected)
    stub.queued_chunks = [_text(f" w{i}") for i in range(20)]
    r = _post(client, {"model": "whisper_large", "long_form": mode},
              filename="long.wav", content=_wav_bytes(10 * 60))
    assert r.status_code == 200, r.text
    assert len(stub.submits) == submitted
    assert set(stub.released) == {s["request_id"] for s in stub.submits[2:]}
    assert not list(stub.upload_dir.iterdir())
