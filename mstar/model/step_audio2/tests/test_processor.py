"""Step-Audio2's chat rendering, tokenization and log-mel against the reference (CPU).

The rendering and placeholder arithmetic run anywhere. The comparisons with the reference's
own prompt ids and mels for the recorded oracle cases need the checkpoint's tokenizer, the
oracle and the reference's audio assets; they skip without them (``STEPAUDIO2_CKPT``,
``STEPAUDIO2_ORACLE``, ``STEPAUDIO2_ASSETS``, defaulting to this project's pod layout).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mstar.model.step_audio2 import processor
from mstar.model.step_audio2.config import AUDIO_PATCH, AUDIO_START, EOT, audio_tokens

CKPT = Path(os.environ.get("STEPAUDIO2_CKPT", "/workspace/stepaudio2-ref/hf/Step-Audio-2-mini"))
ORACLE = Path(os.environ.get("STEPAUDIO2_ORACLE", "/workspace/stepaudio2-ref/oracle/rec"))
ASSETS = Path(os.environ.get("STEPAUDIO2_ASSETS", "/workspace/stepaudio2-ref/github/Step-Audio2/assets"))
CLIP = "give_me_a_brief_introduction_to_the_great_wall.wav"
LONG_CLIP = "paralinguistic_information_understanding.wav"


def no_audio(_):
    raise AssertionError("no audio expected")


def test_turns_render_as_the_reference_does():
    pieces, mels = processor.render([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": [{"type": "text", "text": "<tts_start>"},
                                          {"type": "token", "token": [5, 6]}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ], no_audio)
    assert mels == []
    assert pieces == [
        "<|BOT|>system\nYou are a helpful assistant.<|EOT|>",
        "<|BOT|>human\nHi.<|EOT|>",
        "<|BOT|>assistant\n", "<tts_start>", [5, 6], "<|EOT|>",
        "<|BOT|>assistant\n<tts_start>",
    ]
    assert processor.render([{"role": "assistant", "content": None}], no_audio)[0] == ["<|BOT|>assistant\n"]


def test_pieces_tokenize_on_their_own():
    seen = []

    def encode(text):
        seen.append(text)
        return [len(text)]

    assert processor.tokenize(["ab", [7, 8], "c"], encode) == [2, 7, 8, 1]
    assert seen == ["ab", "c"]


def test_audio_cuts_into_25_second_windows_with_a_placeholder_run_each():
    audio = torch.randn(16000 * 27)
    pieces, mels = processor.render([{"role": "user", "content": [{"type": "audio", "audio": "x"}]}], lambda _: audio)
    assert [m.shape for m in mels] == [(128, 2502), (128, 2 * 100 + 2)]
    runs = [p for p in pieces if isinstance(p, str) and p.startswith("<audio_start>")]
    assert [r.count("<audio_patch>") for r in runs] == [audio_tokens(2502), audio_tokens(202)]


def test_audio_positions_follow_each_audio_start():
    ids = [1, AUDIO_START, AUDIO_PATCH, AUDIO_PATCH, 2, AUDIO_START, AUDIO_PATCH, 3, EOT]
    assert processor.audio_positions(ids, [2, 1]) == [2, 3, 6]
    with pytest.raises(ValueError):
        processor.audio_positions(ids, [3, 1])
    with pytest.raises(ValueError):
        processor.audio_positions(ids, [2])


# --- against the reference's recorded prompts -------------------------------------------

def _needs(*paths: Path):
    missing = [p for p in paths if not p.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"needs {missing}")


def load_16k(path) -> torch.Tensor:
    """``utils.load_audio``: ``transforms.Resample`` over every channel, then the first.
    (``torchaudio.functional.resample`` of the first channel, as ``audio_features`` does,
    moves the log-mel by up to 3e-5.)"""
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())
    if sr != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
    return audio[0]


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
CASES = {
    "t2t": [SYSTEM, {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
            {"role": "assistant", "content": None}],
    "s2t": [SYSTEM, {"role": "human", "content": [{"type": "audio", "audio": CLIP}]},
            {"role": "assistant", "content": None}],
    "s2s": [SYSTEM, {"role": "human", "content": [{"type": "audio", "audio": CLIP}]},
            {"role": "assistant", "content": "<tts_start>", "eot": False}],
    "s2t_long": [SYSTEM, {"role": "human", "content": [{"type": "text", "text": "Describe this audio."},
                                                        {"type": "audio", "audio": LONG_CLIP}]},
                 {"role": "assistant", "content": None}],
}


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    if not (CKPT / "tokenizer.json").exists():
        pytest.skip(f"needs the Step-Audio-2-mini tokenizer under {CKPT}")
    return transformers.AutoTokenizer.from_pretrained(str(CKPT))


@_needs(ORACLE / "llm_s2t.pt", ASSETS / CLIP)
@pytest.mark.parametrize("case", list(CASES))
def test_prompt_ids_and_mels_match_the_references(case, tokenizer):
    rec = torch.load(ORACLE / f"llm_{case}.pt", weights_only=False)
    pieces, mels = processor.render(CASES[case], lambda name: load_16k(ASSETS / name))
    ids = processor.tokenize(pieces, lambda text: tokenizer(text, add_special_tokens=False)["input_ids"])
    assert ids == rec["input_ids"][0].tolist()
    assert len(mels) == len(rec["mels"])
    for got, want in zip(mels, rec["mels"], strict=True):
        assert got.shape == want.shape
        assert torch.equal(got, want)
    if mels:
        positions = processor.audio_positions(ids, [audio_tokens(m.shape[-1]) for m in mels])
        assert len(positions) == sum(rec["feat_lens"].tolist())
