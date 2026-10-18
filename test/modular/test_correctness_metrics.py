"""Tests for benchmark/correctness/metrics.py on synthetic inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmark.correctness import metrics as M


def test_first_divergence():
    assert M.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert M.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert M.first_divergence([1, 2, 3], [1, 2]) == 2
    assert M.first_divergence([], []) is None
    assert M.first_divergence([], [5]) == 0


def test_compare_greedy_exact_within_horizon():
    c = M.compare_greedy([1, 2, 3, 4], [1, 2, 3, 9], horizon=3)
    assert c.exact and c.first_divergence is None


def test_compare_greedy_near_tie_and_real_divergence():
    topk = [{1: -0.1}, {2: -0.6, 7: -0.9, 8: -4.0}, {3: -0.2}]
    near = M.compare_greedy([1, 2, 3], [1, 7, 3], horizon=3, ref_topk=topk, near_tie_nats=1.0)
    assert near.first_divergence == 1 and near.out_in_ref_topk and near.near_tie
    assert near.logprob_gap == pytest.approx(0.3)
    far = M.compare_greedy([1, 2, 3], [1, 8, 3], horizon=3, ref_topk=topk, near_tie_nats=1.0)
    assert far.out_in_ref_topk and not far.near_tie and far.logprob_gap == pytest.approx(3.4)
    absent = M.compare_greedy([1, 2, 3], [1, 5, 3], horizon=3, ref_topk=topk)
    assert absent.out_in_ref_topk is False and absent.near_tie is False


def test_compare_greedy_eos_decisions():
    # The output stopped where the reference kept going, so its token there is EOS.
    topk = [{1: -0.1}, {2: -0.3, 0: -0.5}]
    c = M.compare_greedy([1, 2], [1], horizon=8, ref_topk=topk, eos_id=0, out_stopped=True)
    assert c.first_divergence == 1 and c.out_token == 0 and c.near_tie
    # The reference stopped (EOS at step 1) and the output went on with token 4.
    topk = [{1: -0.1}, {0: -0.2, 4: -2.5}]
    c = M.compare_greedy([1], [1, 4], horizon=8, ref_topk=topk, eos_id=0, ref_stopped=True)
    assert c.ref_token == 0 and c.out_token == 4 and c.near_tie is False


def test_summarize_greedy():
    topk = [{1: 0.0}, {2: -0.1, 3: -0.2}]
    comps = [M.compare_greedy([1, 2], [1, 2], horizon=2),
             M.compare_greedy([1, 2], [1, 3], horizon=2, ref_topk=topk),
             M.compare_greedy([1, 2], [4, 2], horizon=2)]
    s = M.summarize_greedy(comps, short_horizon=1)
    assert s["n"] == 3 and s["exact"] == 1 and s["near_tie_divergences"] == 1
    assert s["unexplained"] == [2]
    assert s["exact_first_1"] == 2  # divergence at 1 is beyond the 1-token horizon


def test_wer_and_normalizers():
    r = M.wer(["the cat sat on the mat"], ["the cat sat on mat"])
    assert r["errors"] == 1 and r["words"] == 6 and r["wer"] == pytest.approx(1 / 6)
    assert M.normalize_basic("Hello, World!  It's  fine.") == "hello world it's fine"
    assert M.wer(["A B"], ["a, b."])["errors"] == 0
    assert M.word_edit_distance(list("abc"), list("xbcd")) == 2
    # The whisper normalizer comes from transformers.
    norm = M.make_normalizer("whisper")
    assert norm("Hello, world!") == norm("hello world")
    with pytest.raises(ValueError):
        M.wer(["a"], [])


def test_audio_stats_and_problems(tmp_path):
    sr = 24000
    t = np.arange(sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    clip = np.concatenate([np.zeros(sr // 2, np.float32), tone, np.zeros(sr // 4, np.float32)])
    s = M.audio_stats(clip, sr)
    assert s["duration_s"] == pytest.approx(1.75)
    assert s["lead_silence_s"] == pytest.approx(0.5, abs=0.03)
    assert s["trail_silence_s"] == pytest.approx(0.25, abs=0.03)
    assert 0.3 < s["silence_frac"] < 0.5 and s["nonfinite"] == 0 and not M.audio_problems(s)
    bad = clip.copy()
    bad[10] = np.nan
    assert any("non-finite" in p for p in M.audio_problems(M.audio_stats(bad, sr)))
    assert any("silence" in p for p in M.audio_problems(M.audio_stats(np.zeros(sr, np.float32), sr)))
    # WAV round trip at 16-bit precision.
    M.write_wav(tmp_path / "a.wav", clip, sr)
    back, sr2 = M.read_wav(tmp_path / "a.wav")
    assert sr2 == sr and np.abs(back - clip).max() < 1e-4


def test_resample_and_log_mel_distance():
    sr = 24000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = M.resample(x, sr, 16000)
    assert len(y) == 16000
    assert M.log_mel_distance(x, x, sr) == 0.0
    quieter = 0.9 * x  # -0.9 dB in every band that carries energy
    other = (0.3 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    assert 0 < M.log_mel_distance(x, quieter, sr) < 1.0
    assert M.log_mel_distance(x, other, sr) > 5.0


def test_psnr_ssim():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert math.isinf(M.psnr(a, a)) and M.ssim(a, a) == pytest.approx(1.0)
    b = np.clip(a.astype(int) + rng.integers(-2, 3, a.shape), 0, 255).astype(np.uint8)
    assert 40 < M.psnr(a, b) < 50 and 0.9 < M.ssim(a, b) < 1.0
    c = rng.integers(0, 256, a.shape, dtype=np.uint8)
    assert M.psnr(a, c) < 12 and M.ssim(a, c) < 0.1
    with pytest.raises(ValueError):
        M.psnr(a, a[:32])
    frames = np.stack([a, b])
    assert len(M.frame_psnrs(frames, frames[:1])) == 1


def test_contact_sheet(tmp_path):
    img = np.full((32, 48, 3), 128, np.uint8)
    M.contact_sheet([[img, img], [img, None]], ["ref", "out", "ref", "missing"], tmp_path / "s.png", thumb=32)
    assert (tmp_path / "s.png").stat().st_size > 0


def test_array_errors_and_cosine():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    e = M.array_errors(ref + 0.5, ref)
    assert e["mse"] == pytest.approx(0.25) and e["mae"] == pytest.approx(0.5) and e["max_abs"] == pytest.approx(0.5)
    assert M.array_errors(ref, ref)["rel_l2"] == 0.0
    assert M.cosine(ref, 2 * ref) == pytest.approx(1.0) and M.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    nan = ref.copy()
    nan[0, 0] = np.nan
    assert M.array_errors(nan, ref)["nonfinite"] == 1


def test_distribution():
    d = M.distribution([1.0, 2.0, math.inf, 3.0])
    assert d["n"] == 4 and d["n_inf"] == 1 and d["median"] == 2.0 and d["max"] == 3.0
    assert M.distribution([]) == {"n": 0}
