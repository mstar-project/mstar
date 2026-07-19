"""Parity tests for the Zonos2 graph-safe sampler ring buffer (Phase 2).

The ring buffer (:class:`Zonos2SamplerBuffers`) replaces the dict-of-growing-
tensors repetition history in :class:`Zonos2LLMSubmodule`. These tests pin that
the ring produces a *bit-identical* repetition penalty to the pre-Phase-2
dict/window implementation — frozen here as ``_RefWindow`` — including across the
window boundary and through mid-batch join/evict with slot reuse (where
cursor/offset reset bugs hide).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mstar.model.zonos2.sampler_buffers import Zonos2SamplerBuffers
from mstar.model.zonos2.tts_sampling import TTSSamplingParams, apply_repetition_penalty

C = 4              # n_codebooks (small for the test)
V = 32             # audio vocab
TEXT_VOCAB = 10
EOA = 8
DEVICE = "cpu"     # pure tensor ops; runs anywhere


class _RefWindow:
    """Frozen reference: the pre-Phase-2 dict/window repetition history.

    Verbatim of the old ``Zonos2LLMSubmodule._rep_ids`` / ``_rep_ids_batched`` /
    ``_append_history`` / ``_step_for``, kept here as the parity oracle now that
    the submodule uses the ring buffer instead.
    """

    def __init__(self, params: TTSSamplingParams):
        self.p = params
        self.h: dict[str, torch.Tensor] = {}

    def step_for(self, rid):
        hist = self.h.get(rid)
        return 0 if hist is None else hist.shape[0]

    def _rep_ids(self, rid):
        hist = self.h.get(rid)
        if hist is None or self.p.repetition_window <= 0 or self.p.repetition_penalty == 1.0:
            return None
        window = hist[-self.p.repetition_window:]
        ids = window.t().unsqueeze(0).contiguous()          # (1, C, w)
        rc = self.p.repetition_codebooks
        if 0 <= rc < C:
            ids = ids.clone()
            ids[:, rc:, :] = -1
        return ids

    def rep_ids_batched(self, rids, device):
        per = [self._rep_ids(r) for r in rids]
        widths = [p.shape[-1] for p in per if p is not None]
        if not widths:
            return None
        w = max(widths)
        rows = []
        for p in per:
            if p is None:
                rows.append(torch.full((1, C, w), -1, dtype=torch.long, device=device))
                continue
            p = p.to(device)
            if p.shape[-1] < w:
                pad = torch.full((1, C, w - p.shape[-1]), -1, dtype=p.dtype, device=device)
                p = torch.cat([p, pad], dim=-1)
            rows.append(p)
        return torch.cat(rows, dim=0)

    def append(self, rid, frame):
        codes = frame[:, :C]
        prev = self.h.get(rid)
        self.h[rid] = codes if prev is None else torch.cat([prev, codes], dim=0)

    def cleanup(self, rid):
        self.h.pop(rid, None)


def _frame(gen) -> torch.Tensor:
    audio = torch.randint(0, V, (1, C), generator=gen, device=DEVICE)
    text = torch.full((1, 1), TEXT_VOCAB, device=DEVICE, dtype=torch.long)
    return torch.cat([audio, text], dim=1)  # (1, C+1)


def _penalized(logits, rep_ids, penalty):
    return apply_repetition_penalty(logits.clone(), rep_ids, penalty)


@pytest.mark.parametrize("window", [3, 50])
@pytest.mark.parametrize("rc", [-1, 2, C])
def test_single_request_parity_across_window(window, rc):
    params = TTSSamplingParams(
        repetition_window=window, repetition_penalty=1.2, repetition_codebooks=rc, seed=0,
    )
    ref = _RefWindow(params)
    ring = Zonos2SamplerBuffers.allocate(
        max_batch_size=1, n_codebooks=C, window=window, repetition_codebooks=rc, device=DEVICE,
    )
    ring.register_request("r0")
    gen = torch.Generator(device=DEVICE).manual_seed(1)

    n_steps = window * 3 + 2  # cross the wrap boundary several times
    for _ in range(n_steps):
        logits = torch.randn(1, C, V, generator=gen, device=DEVICE)

        # Reference (dict window) vs ring — both reflect history BEFORE this frame.
        ref_ids = ref.rep_ids_batched(["r0"], DEVICE)
        ring.gather_for_request_ids(["r0"], padded_bs=1)
        assert ring.steps(1).item() == ref.step_for("r0")
        ring_ids = ring.repetition_ids(1)

        ref_out = _penalized(logits, ref_ids, params.repetition_penalty)
        ring_out = _penalized(logits, ring_ids, params.repetition_penalty)
        assert torch.equal(ref_out, ring_out)

        # Advance both with the same sampled frame.
        frame = _frame(gen)
        ref.append("r0", frame)
        ring.write_frame(frame, padded_bs=1)
        ring.sync_after_step(["r0"])


def test_midbatch_join_evict_slot_reuse():
    """r0 runs; r1 joins; r0 evicted; r2 joins and reuses r0's slot fresh."""
    window = 4
    params = TTSSamplingParams(
        repetition_window=window, repetition_penalty=1.3, repetition_codebooks=-1, seed=0,
    )
    ref = _RefWindow(params)
    ring = Zonos2SamplerBuffers.allocate(
        max_batch_size=4, n_codebooks=C, window=window, repetition_codebooks=-1,
        device=DEVICE, capacity=2,   # small capacity to force slot reuse (no grow)
    )
    gen = torch.Generator(device=DEVICE).manual_seed(7)

    def run_step(rids):
        pb = len(rids)
        logits = torch.randn(pb, C, V, generator=gen, device=DEVICE)
        ref_ids = ref.rep_ids_batched(rids, DEVICE)
        ring.gather_for_request_ids(rids, padded_bs=pb)
        ring_ids = ring.repetition_ids(pb)
        assert torch.equal(
            _penalized(logits, ref_ids, params.repetition_penalty),
            _penalized(logits, ring_ids, params.repetition_penalty),
        )
        for i, rid in enumerate(rids):
            assert ring.steps(pb)[i].item() == ref.step_for(rid)
        frames = torch.cat([_frame(gen) for _ in rids], dim=0)  # (pb, C+1)
        for i, rid in enumerate(rids):
            ref.append(rid, frames[i:i + 1])
        ring.write_frame(frames, padded_bs=pb)
        ring.sync_after_step(rids)

    ring.register_request("r0")
    for _ in range(6):
        run_step(["r0"])                       # r0 past the window

    ring.register_request("r1")
    for _ in range(3):
        run_step(["r0", "r1"])                 # both live, capacity full

    ring.unregister_request("r0")
    ref.cleanup("r0")                  # drop r0 from the dict reference too
    ring.register_request("r2")                # must reuse r0's freed slot, reset clean
    assert ring._rid_to_slot["r2"] == 0 or ring._rid_to_slot["r2"] == 1
    for _ in range(6):
        run_step(["r1", "r2"])                 # r2 fresh; no bleed from r0's ring


def test_padding_rows_do_not_corrupt_master():
    """padded_bs > len(rids): padding rows reuse slot 0 but must not be synced."""
    window = 4
    params = TTSSamplingParams(
        repetition_window=window, repetition_penalty=1.25, repetition_codebooks=-1, seed=0,
    )
    ref = _RefWindow(params)
    ring = Zonos2SamplerBuffers.allocate(
        max_batch_size=4, n_codebooks=C, window=window, repetition_codebooks=-1, device=DEVICE,
    )
    ring.register_request("r0")
    gen = torch.Generator(device=DEVICE).manual_seed(3)

    for _ in range(10):
        logits = torch.randn(1, C, V, generator=gen, device=DEVICE)
        ref_ids = ref.rep_ids_batched(["r0"], DEVICE)
        ring.gather_for_request_ids(["r0"], padded_bs=4)   # 3 padding rows
        ring_ids = ring.repetition_ids(4)[:1]              # only r0's row is real
        assert torch.equal(
            _penalized(logits, ref_ids, params.repetition_penalty),
            _penalized(logits, ring_ids, params.repetition_penalty),
        )
        frame = _frame(gen)
        ref.append("r0", frame)
        # write all 4 rows (padding included) but sync only the real request
        pad = torch.cat([frame] + [_frame(gen) for _ in range(3)], dim=0)
        ring.write_frame(pad, padded_bs=4)
        ring.sync_after_step(["r0"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
def test_in_graph_ops_are_capture_safe():
    """repetition_ids + write_frame must run inside a CUDA graph (the whole point).

    The per-step gather (pinned H2D) stays outside the graph; the read+write are
    captured. A host sync in either op would make capture fail here.
    """
    window, pb = 5, 2
    ring = Zonos2SamplerBuffers.allocate(
        max_batch_size=pb, n_codebooks=C, window=window, repetition_codebooks=2, device="cuda",
    )
    for r in ("r0", "r1"):
        ring.register_request(r)
    codes = torch.zeros(pb, C + 1, dtype=torch.long, device="cuda")  # static graph input

    # Warm up (allocations, autotune) on a side stream before capture.
    ring.gather_for_request_ids(["r0", "r1"], padded_bs=pb)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ring.repetition_ids(pb)
            ring.write_frame(codes, pb)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ring.repetition_ids(pb)
        ring.write_frame(codes, pb)
    torch.cuda.synchronize()

    # Snapshot AFTER capture so the count is unambiguous w.r.t. whether capture
    # itself executed the body; each replay must advance offset by exactly 1.
    off0 = ring.offset_buf[:pb].clone()
    for _ in range(4):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(ring.offset_buf[:pb], off0 + 4)
    # cursor stays in [0, window) after wrapping.
    assert (ring.cursor_buf[:pb] >= 0).all() and (ring.cursor_buf[:pb] < window).all()
