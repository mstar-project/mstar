"""benchmark/tts_speech_wer.py and chatterbox/wer_eval.py: long clips are transcribed whole, and a missing
WAV counts against WER."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark import tts_speech_wer

transformers = pytest.importorskip("transformers")

SR = tts_speech_wer.ASR_SAMPLE_RATE
WINDOW = 30 * SR
HOP = 160


class FakeProcessor:
    """Records each feature call; decodes every row to its clip's length in samples."""

    feature_extractor = SimpleNamespace(n_samples=WINDOW)

    def __init__(self):
        self.calls = []

    def __call__(self, clips, **kwargs):
        self.calls.append(([len(c) for c in clips], kwargs))
        frames = max(len(c) for c in clips) // HOP if kwargs.get("truncation") is False else WINDOW // HOP
        out = SimpleNamespace(input_features=torch.zeros(len(clips), 80, frames))
        if kwargs.get("return_attention_mask"):
            out.attention_mask = torch.ones(len(clips), frames, dtype=torch.long)
        return out

    def batch_decode(self, generated, skip_special_tokens):
        lengths, _ = self.calls[-1]
        return [f" {n} " for n in lengths]


class FakeModel:
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, features, **kwargs):
        self.calls.append((tuple(features.shape), kwargs))
        return torch.zeros(features.shape[0], 1, dtype=torch.long)


@pytest.fixture
def whisper(monkeypatch):
    processor, model = FakeProcessor(), FakeModel()
    monkeypatch.setattr(transformers.WhisperProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(transformers.WhisperForConditionalGeneration, "from_pretrained",
                        lambda *a, **k: model)
    monkeypatch.setattr(tts_speech_wer, "load_for_asr",
                        lambda path: {"raw": np.zeros(int(Path(path).stem), dtype=np.float32)})
    return processor, model


def run(lengths, batch_size=16):
    paths = [Path(f"{n}.wav") for n in lengths]
    return tts_speech_wer.transcribe(paths, "openai/whisper-tiny", "cpu", batch_size)


def test_a_clip_over_30_s_is_decoded_long_form(whisper):
    processor, model = whisper
    run([10 * SR, 45 * SR])

    long_calls = [kwargs for lengths, kwargs in processor.calls if 45 * SR in lengths]
    assert long_calls == [{"sampling_rate": SR, "return_tensors": "pt", "truncation": False,
                           "padding": "longest", "return_attention_mask": True}]
    (shape, kwargs), = [call for call in model.calls if call[0][2] > WINDOW // HOP]
    assert shape == (1, 80, 45 * SR // HOP)
    assert kwargs["return_timestamps"] is True
    assert kwargs["attention_mask"].shape == (1, 45 * SR // HOP)


def test_clips_within_the_window_keep_the_single_window_call(whisper):
    processor, model = whisper
    run([5 * SR, WINDOW, 20 * SR])

    assert processor.calls == [([5 * SR, WINDOW, 20 * SR], {"sampling_rate": SR, "return_tensors": "pt"})]
    assert model.calls == [((3, 80, WINDOW // HOP),
                            {"language": "en", "task": "transcribe", "max_new_tokens": 440})]


def test_transcripts_keep_the_file_order(whisper):
    lengths = [45 * SR, 10 * SR, 60 * SR, 20 * SR, 31 * SR]
    assert run(lengths, batch_size=3) == [str(n) for n in lengths]


def test_the_long_form_features_cover_the_whole_clip():
    """The real feature extractor: the default call keeps only the first 30 s."""
    extractor = transformers.WhisperFeatureExtractor()
    clip = np.random.default_rng(0).standard_normal(45 * SR).astype(np.float32) * 0.1

    default = extractor([clip], sampling_rate=SR, return_tensors="pt")
    assert default.input_features.shape[-1] == WINDOW // HOP

    whole = extractor([clip], sampling_rate=SR, return_tensors="pt", truncation=False,
                      padding="longest", return_attention_mask=True)
    assert whole.input_features.shape[-1] == 45 * SR // HOP
    assert int(whole.attention_mask.sum()) == 45 * SR // HOP


def test_a_sentence_without_audio_scores_as_an_empty_transcript(tmp_path, monkeypatch):
    """A failed request leaves no WAV; skipping it would make WER look better (#1047).
    The run sent the first three sentences, as its report beside the WAVs says."""
    (tmp_path / "sentences.txt").write_text("red fox\nblue cat\ngreen dog\nbrown cow\n")
    (tmp_path / "report.json").write_text(json.dumps({"num_sentences": 3}))
    for sid in (1, 3):
        (tmp_path / f"{sid:04d}.wav").touch()
    monkeypatch.setattr(tts_speech_wer, "transcribe", lambda paths, *a: ["red fox", "green dog"])
    out = tmp_path / "wer.json"
    tts_speech_wer.main(["--audio-dir", str(tmp_path), "--sentences", str(tmp_path / "sentences.txt"),
                         "--out", str(out)])
    report = json.loads(out.read_text())
    assert (report["total_errors"], report["total_words"]) == (2, 6)
    assert report["missing"] == [2]


@pytest.mark.parametrize(("record", "content", "wavs", "sentences", "references", "hypotheses"), [
    # benchmark.runner: 4 requests over 3 sentences (they wrap); request 2 failed
    ("results.json", {"completed": 3, "failed": 1}, ["req_0", "req_1", "req_3"], ["red fox", "blue cat", "green dog"],
     ["red fox", "blue cat", "green dog", "red fox"], ["req_0", "req_1", "", "req_3"]),
    # bench_chatterbox_vllm.py: 3 of the file's 4 sentences sent; request 1 failed
    ("summary.json", {"num_sentences": 3}, ["000", "002"], ["red fox", "blue cat", "green dog", "brown cow"],
     ["red fox", "blue cat", "green dog"], ["000", "", "002"]),
], ids=["runner", "vllm"])
def test_wer_eval_scores_every_request_its_producer_recorded(tmp_path, monkeypatch, record, content, wavs,
                                                             sentences, references, hypotheses):
    """chatterbox/wer_eval.py scores the requests the run's own record says it sent, not one per sentence."""
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("torchaudio")
    from benchmark.chatterbox import wer_eval

    (tmp_path / "sentences.txt").write_text("\n".join(sentences) + "\n")
    (tmp_path / record).write_text(json.dumps(content))
    for stem in wavs:
        sf.write(str(tmp_path / f"{stem}.wav"), np.zeros(1, dtype=np.float32), 16000)
    scored = []
    monkeypatch.setattr(wer_eval, "transcribe", lambda paths, *a: ([p.stem for p in paths], str.lower))
    monkeypatch.setattr(wer_eval, "_compute_wer",
                        lambda refs, hyps: scored.append((refs, hyps)) or {"wer": 0.0, "per_sample": []})
    out = tmp_path / "wer.json"
    monkeypatch.setattr(sys, "argv", ["wer_eval", "--wavs", str(tmp_path), "--sentences",
                                      str(tmp_path / "sentences.txt"), "--out", str(out)])
    wer_eval.main()
    assert scored[-1] == (references, hypotheses)
    assert json.loads(out.read_text())["missing"] == [hypotheses.index("")]
