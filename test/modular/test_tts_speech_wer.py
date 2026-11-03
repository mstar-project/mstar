"""benchmark/tts_speech_wer.py: clips over Whisper's 30 s window are transcribed whole."""

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
