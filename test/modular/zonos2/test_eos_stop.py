"""Regression tests for Zonos2 end-of-generation (EOS) handling.

``check_stop`` mirrors the reference (zonos2 ``tts/sequence.py`` /
``core.py``): the first frame in which *any* codebook emits eoa starts a
delayed stop countdown of ``n_codebooks + 1`` frames, and the aligned end
frame is shifted back by the highest eoa codebook index (that codebook is
delayed by its index under the inter-codebook shear).

Pure-CPU: the submodule's ``check_stop`` never touches the model, so we
build it with a dummy module.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mstar.model.zonos2.submodules import Zonos2LLMSubmodule
from mstar.model.zonos2.tts_sampling import TTSSamplingParams

C, EOA = 9, 1024


def _sub() -> Zonos2LLMSubmodule:
    return Zonos2LLMSubmodule(
        model=nn.Identity(), n_codebooks=C, text_vocab=512, eoa_id=EOA,
        params=TTSSamplingParams(max_tokens=100_000, ignore_eos=False),
    )


def _frame(eoa_cols):
    f = torch.zeros(1, C + 1, dtype=torch.long)   # all-real-audio frame
    for c in eoa_cols:
        f[0, c] = EOA
    return f


def _first_stop_step(sub, script, n=80):
    """script: {step: [codebooks emitting eoa]}. Returns the step check_stop
    first signals the decode loop to stop, or None."""
    ri = SimpleNamespace(dynamic_loop_iter_counts={}, max_tokens=None)
    for step in range(n):
        if sub.check_stop("r", ri, {"new_token": [_frame(script.get(step, []))]}):
            return step
    return None


def test_cb0_eoa_starts_countdown():
    # cb0 emits eoa at step 25; countdown = n_codebooks + 1 = 10 -> the decode
    # loop is told to stop 9 steps later (the trigger frame consumes one).
    assert _first_stop_step(_sub(), {25: [0]}) == 34


def test_any_codebook_triggers_like_the_reference():
    # The reference stops on the *first* eoa from any codebook, so a delayed
    # codebook (cb5 at step 10) starts the countdown immediately (stop 9 steps
    # later); the later cb0 eoa at step 30 is never reached.
    assert _first_stop_step(_sub(), {10: [5], 30: [0]}) == 19


def test_multiple_eoa_align_to_highest_codebook():
    # When several codebooks emit eoa in the same frame, the end frame aligns
    # to the highest index (max_eos_cb), but the countdown length is fixed, so
    # the stop step depends only on the trigger step.
    assert _first_stop_step(_sub(), {12: [0, 3, 7]}) == 21


def test_countdown_generates_delay_flush():
    # After the eoa trigger, generation must continue >= n_codebooks-1 more
    # frames so the delayed codebooks of the last real frame are emitted.
    stop = _first_stop_step(_sub(), {12: [0]})
    assert stop is not None and stop - 12 >= C - 1
