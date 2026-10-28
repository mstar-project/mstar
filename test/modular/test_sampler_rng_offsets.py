"""A request's draws never share an RNG offset, whichever path each step takes.

A step samples eagerly (no slot lease: a prefill past the capture buckets, a
batch over the largest bucket) or under a captured graph. Each path used to keep
its own offset counter, both starting at 0, so a request prefilled eagerly and
decoded under a graph drew its first two tokens from the same uniform: at c=1
both sit at row 0 under the same seed, and with equal logits the two tokens were
always equal.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

V = 64
RID = 7
LEASE = SlotLease(slot=0, bucket=BucketKey("decode", 1, 1))


def _resource(device: torch.device) -> SamplerResource:
    res = SamplerResource(vocab_size=V, enable_repetion_penalty=True, device=device)
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=1, max_seq_len=1)
    return res


def _ingest(res: SamplerResource, rid: int, seed: int) -> None:
    cfg = SamplingReqConfig(temperature=1.0)
    cfg.apply_conductor_config(seed=seed, resource_key="sampler")
    res.ingest_request(rid, cfg)


def _ctx(rid: int, captured: bool) -> StepContext:
    return StepContext(
        request_ids=(rid,), graph_walk="decode" if captured else "prefill", slot=0,
        capture=False, slot_lease=LEASE if captured else None,
    )


@pytest.fixture
def drawn(monkeypatch):
    """The (seed, offset) each draw used, in order, via the two kernel entry points."""
    calls: list[tuple[int, int]] = []

    def eager(logits, *, seed, rand_offset, **kwargs):
        calls.append((int(seed[0]), int(rand_offset[0])))
        return torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    def captured(logits, temperature, top_k, top_p, seed, offset, **kwargs):
        calls.append((int(seed[0]), int(offset[0])))
        return torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    monkeypatch.setattr(sampler_utils, "sample_tokens", eager)
    monkeypatch.setattr(sampler_utils, "sample_accelerator_graphable_gpu", captured)
    return calls


def _run(res: SamplerResource, rid: int, modes: str, step: SamplerStep = SamplerStep()) -> None:
    """One step per mode, "E" eager or "C" captured (the replay's gather,
    in-graph advance and commit, run eagerly)."""
    logits = torch.zeros(1, V, device=res._device)
    for mode in modes:
        ctx = _ctx(rid, captured=mode == "C")
        res.plan(step, ctx)
        res.sample([rid], logits)
        res.commit(step, ctx)


@pytest.mark.parametrize("modes", ["EC", "CE", "ECEC", "EECCE", "CCEEC"])
def test_offsets_are_unique_across_eager_and_captured_steps(drawn, modes):
    res = _resource(torch.device("cpu"))
    _ingest(res, RID, seed=1)
    _run(res, RID, modes)
    assert [offset for _, offset in drawn] == list(range(len(modes)))
    assert len({seed for seed, _ in drawn}) == 1


def test_a_dropped_eager_draw_does_not_advance(drawn):
    res = _resource(torch.device("cpu"))
    _ingest(res, RID, seed=1)
    _run(res, RID, "E", SamplerStep(kept_rids=frozenset()))
    _run(res, RID, "EC")
    assert [offset for _, offset in drawn] == [0, 0, 1]


def test_a_recycled_slot_starts_at_zero(drawn):
    """The reset is queued at registration and drained by whichever path reads
    first, so a slot reused after eager and captured draws starts clean."""
    res = _resource(torch.device("cpu"))
    _ingest(res, RID, seed=1)
    _run(res, RID, "ECE")
    slot = res._cg_buffers._rid_to_slot[RID]
    res.remove_request(RID)
    for first in "EC":
        drawn.clear()
        _ingest(res, RID + 1, seed=2)
        assert res._cg_buffers._rid_to_slot[RID + 1] == slot
        _run(res, RID + 1, first + "EC")
        assert [offset for _, offset in drawn] == [0, 1, 2]
        res.remove_request(RID + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer sampling needs CUDA")
def test_eager_prefill_and_captured_decode_draw_independently():
    """Real FlashInfer draws and a real graph replay: with the same near-flat
    logits on an eager step and the captured step after it, two independent
    draws agree about 1/V of the time. Two draws at one offset always agree."""
    pytest.importorskip("flashinfer")
    device = torch.device("cuda")
    res = _resource(device)
    pad = -1
    res.ingest_capture_row(pad)
    step = SamplerStep()
    capture_ctx = StepContext(
        request_ids=(pad,), graph_walk="decode", slot=0, capture=True, slot_lease=LEASE,
    )
    res.plan(step, capture_ctx)
    static_logits = torch.zeros(1, V, device=device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # Triton autotune and FlashInfer JIT outside capture
            res.sample([pad], static_logits)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = res.sample([pad], static_logits)
    res.commit(step, capture_ctx)

    logits = 0.1 * torch.randn(1, V, generator=torch.Generator().manual_seed(0)).to(device)
    n, same = 64, 0
    for i in range(n):
        _ingest(res, i, seed=1000 + i)
        ctx = _ctx(i, captured=False)
        res.plan(step, ctx)
        first = res.sample([i], logits)
        res.commit(step, ctx)
        ctx = _ctx(i, captured=True)
        res.plan(step, ctx)
        static_logits.copy_(logits)
        graph.replay()
        res.commit(step, ctx)
        same += int(first.item() == static_out.item())
        res.remove_request(i)
    # independent: ~n/V = 1 expected; the shared offset gave n
    assert same <= 8, f"{same}/{n} eager/captured token pairs equal"
