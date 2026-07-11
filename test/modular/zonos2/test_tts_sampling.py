"""Unit tests for the Zonos2 multi-codebook sampler (``tts_sampling``).

All pure-CPU and deterministic: they exercise the per-codebook logit
filters (top-k / top-p / min-p), the per-codebook repetition penalty, and
:func:`sample_frame`'s framing (shape, text-placeholder column, greedy vs.
seeded-stochastic paths, and the all-zero-row greedy fallback).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from mstar.model.zonos2.tts_sampling import (
    TTSSamplingParams,
    apply_min_p,
    apply_repetition_penalty,
    apply_top_p,
    sample_frame,
)

C, V = 3, 8  # codebooks, per-codebook vocab


def _logits(peaks: list[int]) -> torch.Tensor:
    """(1, C, V) logits with a clear argmax at ``peaks[c]`` for each codebook."""
    x = torch.zeros(1, C, V)
    for c, p in enumerate(peaks):
        x[0, c, p] = 10.0
    return x


# -- sample_frame framing ---------------------------------------------------
def test_frame_shape_dtype_and_text_placeholder():
    params = TTSSamplingParams(temperature=0.0)
    frame = sample_frame(_logits([1, 2, 3]), params, text_placeholder=7)
    assert frame.shape == (1, C + 1)
    assert frame.dtype == torch.long
    assert frame[0, -1].item() == 7  # appended text column


def test_greedy_when_temperature_zero():
    params = TTSSamplingParams(temperature=0.0)
    frame = sample_frame(_logits([1, 2, 3]), params, text_placeholder=0)
    assert frame[0, :C].tolist() == [1, 2, 3]  # argmax per codebook


def test_sampled_ids_in_range():
    params = TTSSamplingParams(temperature=1.0, topk=V, min_p=0.0)
    frame = sample_frame(torch.randn(1, C, V), params)
    assert (frame[0, :C] >= 0).all() and (frame[0, :C] < V).all()


# -- determinism / RNG ------------------------------------------------------
def test_seed_reproducible_and_seed_dependent():
    logits = torch.randn(1, C, V)
    params = TTSSamplingParams(temperature=1.0, topk=V, min_p=0.0)

    def draw(seed: int) -> list[int]:
        return sample_frame(logits, params, seed=seed)[0, :C].tolist()

    assert draw(1234) == draw(1234)          # same seed -> identical
    # Some seed produces a different frame (not degenerate/constant).
    assert any(draw(1234) != draw(s) for s in (1, 2, 3, 4, 5))


def test_step_offset_changes_draw():
    logits = torch.randn(1, C, V)
    params = TTSSamplingParams(temperature=1.0, topk=V, min_p=0.0)

    def draw(step: int) -> list[int]:
        return sample_frame(
            logits, params, seed=7, steps=torch.tensor([step])
        )[0, :C].tolist()

    assert draw(0) == draw(0)                 # same (seed, step) -> identical
    assert any(draw(0) != draw(k) for k in (1, 2, 3, 4, 5))


def test_batched_equals_per_request_and_position_independent():
    B = 4
    logits = torch.randn(B, C, V)
    params = TTSSamplingParams(temperature=1.0, topk=V, min_p=0.0)
    steps = torch.tensor([0, 5, 5, 9])

    batched = sample_frame(logits, params, seed=99, steps=steps)
    assert batched.shape == (B, C + 1)
    # Each row matches sampling that request on its own at the same step.
    for i in range(B):
        one = sample_frame(logits[i:i + 1], params, seed=99, steps=steps[i:i + 1])
        assert torch.equal(one, batched[i:i + 1])

    # Rows 1 and 2 share a step and (here) identical logits -> identical frame,
    # regardless of batch slot: no per-position RNG dependence.
    logits2 = logits.clone()
    logits2[2] = logits2[1]
    out = sample_frame(logits2, params, seed=99, steps=steps)
    assert torch.equal(out[1], out[2])


# -- filters ----------------------------------------------------------------
def test_apply_top_p_keeps_nucleus_and_renormalizes():
    probs = torch.tensor([[0.6, 0.3, 0.08, 0.02]])
    out = apply_top_p(probs.clone(), p=0.8)
    assert out[0, 0] > 0 and out[0, 1] > 0     # top nucleus kept
    assert out[0, 2] == 0 and out[0, 3] == 0   # tail dropped
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_apply_min_p_drops_below_fraction_of_max():
    probs = torch.tensor([[0.7, 0.2, 0.09, 0.01]])
    out = apply_min_p(probs.clone(), min_p=0.5)  # threshold = 0.5 * 0.7 = 0.35
    assert out[0, 0] > 0
    assert (out[0, 1:] == 0).all()
    assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)


# -- repetition penalty -----------------------------------------------------
def test_repetition_penalty_decreases_repeated_positive_logit():
    logits = torch.full((1, 1, V), 2.0)          # all positive
    rep_ids = torch.tensor([[[3, 3]]])            # token 3 seen twice, cb 0
    out = apply_repetition_penalty(logits.clone(), rep_ids, repetition_penalty=1.5)
    assert out[0, 0, 3].item() == pytest.approx(2.0 / 1.5)
    others = [i for i in range(V) if i != 3]
    assert torch.allclose(out[0, 0, others], torch.full((len(others),), 2.0))


def test_repetition_penalty_negative_logit_pushed_down():
    logits = torch.full((1, 1, V), -2.0)
    rep_ids = torch.tensor([[[3]]])
    out = apply_repetition_penalty(logits.clone(), rep_ids, repetition_penalty=1.5)
    assert out[0, 0, 3].item() == pytest.approx(-2.0 * 1.5)


def test_repetition_penalty_ignores_negative_one_ids():
    logits = torch.full((1, 1, V), 2.0)
    rep_ids = torch.full((1, 1, 4), -1)          # excluded codebook / empty window
    out = apply_repetition_penalty(logits.clone(), rep_ids, repetition_penalty=1.5)
    assert torch.equal(out, logits)              # no token penalized


def test_repetition_penalty_noop_when_penalty_one():
    logits = torch.randn(1, C, V)
    rep_ids = torch.zeros(1, C, 4, dtype=torch.long)
    out = apply_repetition_penalty(logits.clone(), rep_ids, repetition_penalty=1.0)
    assert torch.equal(out, logits)


# -- robustness -------------------------------------------------------------
def test_aggressive_filters_do_not_crash_and_stay_in_range():
    # min_p just under 1 keeps ~only the argmax; combined with a tiny top_p
    # this stresses the all-zero-row greedy fallback path in sample_frame.
    params = TTSSamplingParams(temperature=1.0, topk=V, top_p=1e-6, min_p=0.999)
    frame = sample_frame(_logits([5, 0, 2]), params)
    assert (frame[0, :C] >= 0).all() and (frame[0, :C] < V).all()
