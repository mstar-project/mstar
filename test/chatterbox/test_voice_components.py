"""CPU parity of the reference-audio front end against the reference package.

Real weights from the HF cache (``ResembleAI/chatterbox``), fp32 on CPU, short
synthetic 16 kHz signals. Skips when the ``chatterbox`` oracle or the
snapshot is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

chatterbox = pytest.importorskip("chatterbox")
librosa = pytest.importorskip("librosa")
from safetensors.torch import load_file  # noqa: E402

from mstar.model.chatterbox.components.audio_frontend import (  # noqa: E402
    mel_filter_bank,
    resample,
    trim_silence,
)
from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer  # noqa: E402
from mstar.model.chatterbox.components.voice_encoder import VoiceEncoder  # noqa: E402
from mstar.model.chatterbox.loader import iter_weights, resolve_snapshot  # noqa: E402

SR = 16000


@pytest.fixture(scope="module")
def snapshot() -> str:
    try:
        return resolve_snapshot("ResembleAI/chatterbox")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"chatterbox snapshot unavailable: {exc}")


def _signal(seconds: float, sr: int = SR, seed: int = 0) -> torch.Tensor:
    """A voiced-ish test tone: harmonics with vibrato under a raised-sine
    envelope that starts and ends quietly (so the silence trim has work to
    do), plus a little noise."""
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float64) / sr
    f0 = 180.0 + 12.0 * torch.sin(2 * math.pi * 3.0 * t)
    phase = 2 * math.pi * torch.cumsum(f0, 0) / sr
    wav = sum(torch.sin(k * phase) / k for k in range(1, 6))
    env = torch.sin(math.pi * t / seconds) ** 2
    wav = 0.4 * wav * env + 0.002 * torch.randn(n, generator=g, dtype=torch.float64)
    return wav.float()


@pytest.fixture(scope="module")
def signals() -> dict[str, torch.Tensor]:
    return {"long": _signal(2.5), "short": _signal(0.7, seed=1), "prompt": _signal(6.5, seed=2)}


# --------------------------------------------------------------------------- voice encoder


@pytest.fixture(scope="module")
def voice_encoders(snapshot):
    from chatterbox.models.voice_encoder import VoiceEncoder as RefVoiceEncoder

    state = load_file(f"{snapshot}/ve.safetensors")
    ref = RefVoiceEncoder()
    ref.load_state_dict(state)
    ref.eval()
    mine = VoiceEncoder()
    mine.load_weights(iter_weights(f"{snapshot}/ve.safetensors"))
    mine.eval()
    return ref, mine


def test_mel_filter_banks_match_librosa():
    for n_mels, fmax in ((40, 8000.0), (128, None)):
        mine = mel_filter_bank(SR, 400, n_mels, 0.0, fmax).numpy()
        ref = librosa.filters.mel(sr=SR, n_fft=400, n_mels=n_mels, fmin=0.0, fmax=fmax)
        assert mine.shape == ref.shape
        assert np.abs(mine - ref).max() <= 1e-6


def test_silence_trim_matches_librosa(signals):
    for wav in signals.values():
        ref, (start, end) = librosa.effects.trim(wav.numpy(), top_db=20)
        mine = trim_silence(wav, 20.0)
        assert mine.shape[0] == end - start
        assert torch.equal(mine, torch.from_numpy(ref))


def test_voice_encoder_mel_matches_reference(voice_encoders, signals):
    from chatterbox.models.voice_encoder.config import VoiceEncConfig
    from chatterbox.models.voice_encoder.melspec import melspectrogram

    _, mine = voice_encoders
    for wav in signals.values():
        ref = melspectrogram(wav.numpy(), VoiceEncConfig()).T  # (T, 40)
        got = mine.mel(wav).numpy()
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() <= 1e-4 * ref.max()


def test_voice_encoder_partials_and_embedding_match_reference(voice_encoders, signals):
    from chatterbox.models.voice_encoder.config import VoiceEncConfig
    from chatterbox.models.voice_encoder.voice_encoder import get_frame_step, get_num_wins

    ref, mine = voice_encoders
    hp = VoiceEncConfig()
    assert mine.frame_step == get_frame_step(0.5, 1.3, hp) == 77
    for name, wav in signals.items():
        trimmed = trim_silence(wav, 20.0)
        n_frames = mine.mel(trimmed).shape[0]
        n_partials, target = get_num_wins(n_frames, mine.frame_step, 0.8, hp)
        assert mine.num_partials(n_frames) == (n_partials, target)
        assert mine.mel_partials(trimmed).shape == (n_partials, 160, 40)

        with torch.no_grad():
            expected = ref.embeds_from_wavs([wav.numpy()], sample_rate=SR)  # (1, 256)
        expected = torch.from_numpy(expected).mean(0)
        got = mine.embed_utterance(wav)
        assert got.shape == (256,)
        assert torch.allclose(got, expected, atol=1e-4), (name, (got - expected).abs().max())
    # the short clip really exercised the padding branch: one partial, zero-padded mel
    short = trim_silence(signals["short"], 20.0)
    assert mine.num_partials(mine.mel(short).shape[0])[0] == 1


# --------------------------------------------------------------------------- S3 tokenizer


@pytest.fixture(scope="module")
def tokenizers(snapshot):
    from chatterbox.models.s3tokenizer import S3Tokenizer as RefS3Tokenizer

    path = f"{snapshot}/s3gen.safetensors"
    state = {
        name: tensor for name, tensor in iter_weights(path, prefix="tokenizer.")
    }
    ref = RefS3Tokenizer("speech_tokenizer_v2_25hz")
    result = ref.load_state_dict(state, strict=False)
    assert set(result.missing_keys) == {"window"}, result
    assert not result.unexpected_keys, result
    ref.eval()
    mine = S3Tokenizer()
    mine.load_weights(iter_weights(path, prefix="tokenizer."))
    mine.eval()
    return ref, mine, state["_mel_filters"]


def test_tokenizer_mel_filters_match_checkpoint(tokenizers):
    _, mine, stored = tokenizers
    assert mine.mel_filters.shape == stored.shape == (128, 201)
    assert (mine.mel_filters - stored).abs().max() <= 1e-6


def test_tokenizer_log_mel_matches_reference(tokenizers, signals):
    ref, mine, _ = tokenizers
    for wav in signals.values():
        expected = ref.log_mel_spectrogram(wav.unsqueeze(0))
        got = mine.log_mel(wav)
        assert got.shape == expected.shape
        assert (got - expected).abs().max() <= 1e-4


def test_tokenizer_tokens_match_reference_exactly(tokenizers, signals):
    ref, mine, _ = tokenizers
    cases = [
        ("long", None, 63),  # 2.5 s -> 250 frames -> 63 tokens
        ("long", 150, 63),
        ("prompt", 150, 150),  # 6.5 s -> 650 frames, capped to 600 -> 150 tokens
        ("prompt", None, 163),
        ("short", None, 18),
    ]
    for name, max_len, n_tokens in cases:
        wav = signals[name]
        with torch.no_grad():
            exp_tokens, exp_lens = ref.forward([wav], max_len=max_len)
            got_tokens, got_lens = mine([wav], max_len=max_len)
        assert exp_lens.tolist() == [n_tokens] == got_lens.tolist()
        assert got_tokens.shape == exp_tokens.shape == (1, n_tokens)
        assert torch.equal(got_tokens, exp_tokens), (name, max_len, (got_tokens != exp_tokens).sum())
        assert int(got_tokens.max()) < 6561 and int(got_tokens.min()) >= 0


def test_tokenizer_batches_pad_without_changing_tokens(tokenizers, signals):
    """Padded rows tokenize like their solo runs up to their own length."""
    _, mine, _ = tokenizers
    wavs = [signals["long"], signals["short"]]
    with torch.no_grad():
        batched, lens = mine(wavs)
        solo = [mine([w])[0][0] for w in wavs]
    for row, n, single in zip(batched, lens.tolist(), solo, strict=True):
        assert torch.equal(row[:n], single)


def test_tokenizer_rejects_over_long_prompts(tokenizers):
    _, mine, _ = tokenizers
    with pytest.raises(ValueError, match="limited to 3000"):
        mine.encode_mel(torch.zeros(1, 128, 3001), torch.tensor([3001]))


# --------------------------------------------------------------------------- resampling


def test_resample_24k_to_16k_is_close_to_librosa():
    wav24 = _signal(2.5, sr=24000, seed=3)
    mine = resample(wav24, 24000, SR)
    ref = torch.from_numpy(librosa.resample(wav24.numpy(), orig_sr=24000, target_sr=SR))
    n = min(mine.shape[0], ref.shape[0])
    assert abs(mine.shape[0] - ref.shape[0]) <= 1
    diff = (mine[:n] - ref[:n]).abs().max().item()
    # Different resamplers (torchaudio sinc vs soxr); they agree to ~1e-3.
    assert diff < 1e-2, diff
