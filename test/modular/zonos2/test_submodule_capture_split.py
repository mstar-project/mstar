"""Parity tests for the Phase-3 capture-boundary split of Zonos2 sampling.

Phase 3 moves the sampler's *host-side* lifecycle (lazy register, per-step
gather, and the ``buf -> master`` write-back) out of ``_sample`` and into
``Zonos2LLMSubmodule.preprocess`` (``_prepare_sampler_step``), leaving only the
fixed-shape in-graph portion (``repetition_ids -> sample_frame -> write_frame``)
in ``forward``/``forward_batched``. The write-back is *deferred* to the next
step's preprocess so it stays outside the captured graph.

The Phase-2 ring-buffer tests (``test_sampler_buffers.py``) already pin the ring
against the frozen dict oracle. These tests pin the *split itself*:

* the deferred-sync order reproduces the pre-Phase-3 **inline** draws exactly,
  across join / evict / slot-reuse and capture-style padding;
* the real request ids are recovered under CUDA-graph replay (where
  ``request_ids`` is the dummy capture slots) via ``real_request_ids``, and the
  ``__cg_`` capture placeholders are filtered out;
* **sync-before-register** holds: a new request reusing a just-freed slot is not
  clobbered by the departing request's deferred write-back.

Pure-CPU: no model forward is exercised (the split touches only the sampler
buffers), so a tiny parameter-only stand-in model supplies ``get_device()``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mstar.model.zonos2.sampler_buffers import Zonos2SamplerBuffers
from mstar.model.zonos2.submodules import Zonos2LLMSubmodule
from mstar.model.zonos2.tts_sampling import TTSSamplingParams, sample_frame

C = 4              # n_codebooks (small for the test)
V = 32             # audio vocab
TEXT_VOCAB = 10
DEVICE = "cpu"     # pure tensor ops; runs anywhere


class _FakeModel(nn.Module):
    """Parameter-only stand-in so ``ARNodeSubmodule.get_device`` resolves."""

    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1, device=DEVICE))


def _params(window=4, penalty=1.3, rc=-1, seed=0) -> TTSSamplingParams:
    return TTSSamplingParams(
        repetition_window=window, repetition_penalty=penalty,
        repetition_codebooks=rc, seed=seed, temperature=1.0, topk=0, min_p=0.0,
    )


def _sub(params: TTSSamplingParams) -> Zonos2LLMSubmodule:
    return Zonos2LLMSubmodule(
        model=_FakeModel(), n_codebooks=C, text_vocab=TEXT_VOCAB, eoa_id=8,
        params=params,
    )


def _engine_inputs(request_ids, real_request_ids=None):
    return SimpleNamespace(
        request_ids=request_ids,
        real_request_ids=real_request_ids,
        cache_manager=None,
    )


def _inline_step(buf, rids, logits, params):
    """Pre-Phase-3 order: register -> gather -> sample -> write -> **sync now**.

    The frozen reference for what the deferred split must reproduce.
    """
    for rid in rids:
        buf.register_request(rid)
    pb = len(rids)
    buf.gather_for_request_ids(rids, padded_bs=pb)
    frames = sample_frame(
        logits, params,
        repetition_token_ids=buf.repetition_ids(pb),
        text_placeholder=TEXT_VOCAB,
        seed=params.seed, steps=buf.steps(pb),
    )
    buf.write_frame(frames, padded_bs=pb)
    buf.sync_after_step(rids)
    return frames


def _split_step(sub, rids, logits, *, padded_bs=None, real_request_ids=None,
                request_ids=None):
    """Phase-3 order: host-side preprocess (deferred sync) then in-graph sample."""
    pb = padded_bs if padded_bs is not None else len(rids)
    ei = _engine_inputs(
        request_ids=request_ids if request_ids is not None else list(rids),
        real_request_ids=real_request_ids,
    )
    sub._prepare_sampler_step(ei, padded_bs=pb)
    return sub._sample_in_graph(logits)  # (pb, C + 1)


def _bare_buf(params, capacity=None):
    return Zonos2SamplerBuffers.allocate(
        max_batch_size=256, n_codebooks=C, window=params.repetition_window,
        repetition_codebooks=params.repetition_codebooks, device=DEVICE,
        capacity=capacity,
    )


# --------------------------------------------------------------------------
def test_deferred_split_matches_inline_single_request():
    """Deferred split == inline order, frame-for-frame, across the window wrap."""
    params = _params(window=3)
    sub = _sub(params)
    ref = _bare_buf(params)
    gen = torch.Generator(device=DEVICE).manual_seed(1)

    for _ in range(params.repetition_window * 4 + 2):
        logits = torch.randn(1, C, V, generator=gen, device=DEVICE)
        inline = _inline_step(ref, ["r0"], logits.clone(), params)
        split = _split_step(sub, ["r0"], logits.clone(), real_request_ids=["r0"])
        assert torch.equal(inline, split)


def test_deferred_split_matches_inline_multibatch_join_evict_reuse():
    """Deferred split == inline through join, evict, and slot reuse."""
    params = _params(window=4, penalty=1.25)
    sub = _sub(params)
    ref = _bare_buf(params)
    gen = torch.Generator(device=DEVICE).manual_seed(7)

    def both(rids):
        logits = torch.randn(len(rids), C, V, generator=gen, device=DEVICE)
        inline = _inline_step(ref, rids, logits.clone(), params)
        split = _split_step(sub, rids, logits.clone(), real_request_ids=list(rids))
        assert torch.equal(inline, split), f"mismatch on {rids}"

    for _ in range(6):
        both(["r0"])                      # r0 past the window
    for _ in range(3):
        both(["r0", "r1"])                # r1 joins
    # r0 evicted from both driver and reference.
    ref.unregister_request("r0")
    sub.cleanup_request("r0")
    for _ in range(6):
        both(["r1", "r2"])                # r2 joins, reuses r0's freed slot


def test_capture_style_padding_matches_unpadded_reals():
    """Padded batch (real_request_ids shorter than padded_bs): the real rows'
    frames match the unpadded inline run, and padding rows never corrupt master.
    """
    params = _params(window=4)
    sub = _sub(params)
    ref = _bare_buf(params)
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    padded_bs = 4
    reals = ["r0", "r1"]

    for _ in range(10):
        # Reference runs the reals unpadded (bs=2).
        logits_real = torch.randn(len(reals), C, V, generator=gen, device=DEVICE)
        inline = _inline_step(ref, reals, logits_real.clone(), params)

        # Split runs capture-style: request_ids is padded_bs dummy slots,
        # real_request_ids carries the two reals; logits are padded_bs wide with
        # the real rows first (padding rows get arbitrary logits).
        pad_logits = torch.randn(padded_bs, C, V, generator=gen, device=DEVICE)
        pad_logits[:len(reals)] = logits_real
        dummy_rids = [f"__cg_decode_False_slot0_{i}__" for i in range(padded_bs)]
        split = _split_step(
            sub, reals, pad_logits, padded_bs=padded_bs,
            real_request_ids=reals, request_ids=dummy_rids,
        )
        assert torch.equal(inline, split[:len(reals)])

    # Master offset for the two reals must equal the step count (10), proving
    # padding rows (slot 0) were never synced over them.
    for rid in reals:
        slot = sub._sampler_buffers._rid_to_slot[rid]
        # 9 syncs have landed (the 10th is still deferred), plus the pending one.
        assert sub._sampler_buffers.offset_master[slot].item() in (9, 10)


def test_sync_before_register_no_clobber_on_slot_reuse():
    """The load-bearing ordering case: a new request reusing a just-freed slot
    starts fresh, i.e. the departing request's deferred write-back (step 1 of
    preprocess) lands *before* the new request's reset (step 2), never after.

    Oracle: the reused-slot request's frame stream must equal the same request
    run from scratch on a pristine submodule.
    """
    params = _params(window=4)
    gen = torch.Generator(device=DEVICE).manual_seed(11)

    # Pre-generate a fixed logits stream for the "D" phase so both runs match.
    n_d = 8
    d_logits = [torch.randn(1, C, V, generator=gen, device=DEVICE) for _ in range(n_d)]

    # --- Scenario A: A runs and is evicted; D reuses A's (LIFO) freed slot. ---
    sub = _sub(params)
    warm = torch.Generator(device=DEVICE).manual_seed(99)
    for _ in range(7):  # A accumulates a long history / large offset
        _split_step(sub, ["A"], torch.randn(1, C, V, generator=warm, device=DEVICE),
                    real_request_ids=["A"])
    a_slot = sub._sampler_buffers._rid_to_slot["A"]
    sub.cleanup_request("A")  # frees A's slot (LIFO -> D will reuse it)

    d_frames_reuse = []
    for lg in d_logits:
        d_frames_reuse.append(_split_step(sub, ["D"], lg.clone(), real_request_ids=["D"]))
    assert sub._sampler_buffers._rid_to_slot["D"] == a_slot, "test needs slot reuse"

    # --- Scenario B: D alone on a pristine submodule. ---
    sub_fresh = _sub(params)
    d_frames_fresh = []
    for lg in d_logits:
        d_frames_fresh.append(_split_step(sub_fresh, ["D"], lg.clone(), real_request_ids=["D"]))

    for a, b in zip(d_frames_reuse, d_frames_fresh, strict=True):
        assert torch.equal(a, b), "reused slot bled prior request's state into D"

    # And D's first step must have started from an empty window / zero step.
    assert torch.equal(d_frames_reuse[0], d_frames_fresh[0])


def test_real_rid_recovery_and_cg_filter():
    """Under replay, real ids come from ``real_request_ids``; ``__cg_`` dummies
    (capture-time placeholders) register nothing and gather to slot 0.
    """
    params = _params()
    sub = _sub(params)

    # Capture-time call: request_ids are dummy slots, real_request_ids is None.
    # Nothing should register; gather must not raise.
    dummy = [f"__cg_decode_False_slot0_{i}__" for i in range(3)]
    logits = torch.randn(3, C, V, device=DEVICE)
    _split_step(sub, [], logits, padded_bs=3, real_request_ids=None, request_ids=dummy)
    assert sub._sampler_buffers._rid_to_slot == {}, "dummy rids must not register"
    assert sub._pending_sync_rids == [], "no reals retained for deferred sync"

    # Replay-time call: real_request_ids carries the live ids (request_ids stays
    # dummy). Exactly those register.
    logits2 = torch.randn(4, C, V, device=DEVICE)
    _split_step(
        sub, ["ra", "rb"], logits2, padded_bs=4,
        real_request_ids=["ra", "rb"],
        request_ids=[f"__cg_decode_False_slot0_{i}__" for i in range(4)],
    )
    assert set(sub._sampler_buffers._rid_to_slot) == {"ra", "rb"}
    assert sub._pending_sync_rids == ["ra", "rb"]


# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph capture needs a GPU"
)
@pytest.mark.parametrize("reals,padded_bs", [(1, 1), (2, 2), (3, 4)])
def test_captured_sampler_matches_eager_token_for_token(reals, padded_bs):
    """The Phase-3 acceptance check, scoped to the newly in-graph portion.

    A CUDA graph of ``_sample_in_graph`` (repetition_ids -> sample_frame ->
    write_frame) driven by the same out-of-graph gather must emit the exact same
    frames, step for step, as a fully eager run — including capture-style padding
    (``reals < padded_bs``). The transformer forward is already captured
    pre-Phase-3; this pins that folding the sampler in preserves behavior.
    """
    dev = "cuda"
    params = _params(window=4, penalty=1.3, seed=0)
    params.temperature, params.topk, params.min_p = 1.0, 8, 0.1  # exercise filters
    rids = [f"r{i}" for i in range(reals)]
    dummy = [f"__cg_decode_False_slot0_{i}__" for i in range(padded_bs)]
    n_steps = 12

    gen = torch.Generator(device=dev).manual_seed(5)
    logit_stream = [
        torch.randn(padded_bs, C, V, generator=gen, device=dev) for _ in range(n_steps)
    ]

    def _cuda_sub():
        sub = Zonos2LLMSubmodule(
            model=_FakeModel().to(dev), n_codebooks=C, text_vocab=TEXT_VOCAB,
            eoa_id=8, params=params,
        )
        return sub

    # --- Eager reference run ---
    eager = _cuda_sub()
    eager_frames = []
    for lg in logit_stream:
        f = _split_step(
            eager, rids, lg[:reals].clone() if reals == padded_bs else lg.clone(),
            padded_bs=padded_bs, real_request_ids=rids, request_ids=dummy,
        )
        eager_frames.append(f[:reals].clone())

    # --- Captured run ---
    cap = _cuda_sub()
    cap._ensure_buffers(torch.device(dev), padded_bs)
    static_logits = torch.empty(padded_bs, C, V, device=dev)

    # Prime buf with a valid gather, then warm up on a side stream before capture.
    ei0 = _engine_inputs(request_ids=dummy, real_request_ids=rids)
    cap._prepare_sampler_step(ei0, padded_bs=padded_bs)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            cap._sample_in_graph(static_logits)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_frames = cap._sample_in_graph(static_logits)

    # Reset the requests so replay starts from step 0 (capture/warmup dirtied buf
    # + master; addresses are unchanged, so the graph stays valid).
    for rid in rids:
        cap.cleanup_request(rid)
    cap._pending_sync_rids = None

    cap_frames = []
    for lg in logit_stream:
        ei = _engine_inputs(request_ids=dummy, real_request_ids=rids)
        cap._prepare_sampler_step(ei, padded_bs=padded_bs)  # gather outside graph
        static_logits.copy_(lg)
        g.replay()
        cap_frames.append(static_frames[:reals].clone())

    for step, (e, c) in enumerate(zip(eager_frames, cap_frames, strict=True)):
        assert torch.equal(e, c), f"captured != eager at step {step} (bs={padded_bs})"
