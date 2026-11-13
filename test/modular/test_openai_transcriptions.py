"""``/v1/audio/transcriptions``' Python pieces: the ASR adapters (request
mapping, transcript parsing) and the ``WindowPlanner`` that cuts long uploads,
which mstar-server drives through the bridge. The route is mstar-server's,
tested in test/rust/test_rust_transcriptions.py.
"""

from pathlib import Path

import pytest

from mstar.api_server.openai import adapters, serving_transcriptions
from mstar.api_server.openai.protocol import TranscriptionRequest

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


def test_qwen3_asr_is_not_sent_the_route_s_timestamps():
    """Qwen3-ASR writes no timestamps; a timed format's key would only be logged as ignored."""
    req = TranscriptionRequest(response_format="verbose_json")
    assert "timestamps" not in adapters.Qwen3ASRAdapter().transcription_to_request(req, "a").model_kwargs


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
    whole = serving_transcriptions.WindowPlanner(str(path), None, tmp_path).fixed()
    assert len(whole) == 1 and whole[0].path == str(path) and whole[0].duration == pytest.approx(75.0)
    windows = serving_transcriptions.WindowPlanner(str(path), 30.0, tmp_path).fixed()
    assert [w.offset for w in windows] == [0.0, 30.0, 60.0]
    assert [round(w.duration, 3) for w in windows] == [30.0, 30.0, 15.0]
    assert all(Path(w.path).exists() for w in windows)
    short = serving_transcriptions.WindowPlanner(str(path), 80.0, tmp_path).fixed()
    assert len(short) == 1


def test_plan_windows_drops_a_tail_too_short_to_hold_speech(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    path = tmp_path / "long.wav"
    # silence cuts at the boundary, leaving 10 ms that a model would refuse
    sf.write(path, np.zeros(16_000 * 60 + 160, dtype="float32"), 16_000)
    windows = serving_transcriptions.WindowPlanner(str(path), 30.0, tmp_path).fixed()
    assert [round(w.duration, 3) for w in windows] == [30.0, 30.0]
    planner = serving_transcriptions.WindowPlanner(str(path), 30.0, tmp_path)
    assert planner.done(60.0) and not planner.done(59.9)


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


def test_numeric_strings_are_coerced():
    req = TranscriptionRequest(top_p="0.9", max_output_tokens="100")
    mk = adapters.WhisperAdapter().transcription_to_request(req, "/tmp/a.wav").model_kwargs
    assert mk["top_p"] == 0.9 and mk["max_output_tokens"] == 100


@pytest.mark.parametrize("cls", [adapters.WhisperAdapter, adapters.Qwen3ASRAdapter])
def test_asr_models_cut_a_long_upload_unless_asked_not_to(cls):
    assert cls.long_form == "parallel"
    assert (cls.window_seconds or cls.max_audio_seconds) == 30.0
