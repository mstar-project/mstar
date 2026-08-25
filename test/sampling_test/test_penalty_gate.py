"""The repetition-penalty gate on ``SamplerResource``.

Does the seen-token bookkeeping actually get skipped when no resident request
asks for a penalty, and does it come back the moment one does?

What this covers is the *decision*, not the arithmetic: the penalty itself lives
in a Triton kernel whose ``APPLY_PENALTY`` is a constexpr baked at capture
(``sampling.fused_temperature_softmax``), so the resource cannot turn it off per
step and does not try to. It suppresses only the off-graph mask traffic — bs x
[vocab] staging copies, a [bs, vocab] gather, and bs copies back — which is
inert whenever every resident ``repetition_penalty`` is 1.0. The numerics of the
kernel itself are covered by ``cudagraph_sampler_test.py`` (GPU).

Runs on CPU; ``mstar.engine.resources.sampler.utils`` imports triton at module scope, so the
whole module skips where triton isn't installed.

    pytest test/sampling_test/test_penalty_gate.py
"""

import pytest

pytest.importorskip("triton")

import torch  # noqa: E402

from mstar.engine.resources.step import (  # noqa: E402
    BucketKey,
    SamplerStep,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.sampler.resource import SamplerResource, SamplingReqConfig  # noqa: E402

VOCAB = 64


class _RecordingSampler:
    """Stands in for the CudaGraphableSampler a plan hands back."""

    def __init__(self):
        self.synced = 0
        self.sampled_with: list[bool] = []

    def sync_seen_token_masks(self, seen_masks):
        self.synced += 1

    def sample(self, request_ids, logits, apply_penalty=False):
        self.sampled_with.append(apply_penalty)
        return torch.zeros(len(request_ids), dtype=torch.long)


class _RecordingBuffers:
    """Stands in for SamplerBuffers, recording just what the gate controls."""

    def __init__(self):
        self.sampler = _RecordingSampler()
        self.staged = 0
        self.static_gathers = 0
        self.offset_scatters = 0
        # one entry per gather_dynamic call: was the mask gathered?
        self.gathered_seen: list[bool] = []

    def gather_static(self, request_ids, padded_bs, cg_slot):
        self.static_gathers += 1

    def stage_seen_token_masks(self, request_ids, seen_masks):
        self.staged += 1

    def gather_dynamic(self, request_ids, padded_bs, cg_slot, gather_seen_tokens=True):
        self.gathered_seen.append(gather_seen_tokens)

    def sampler_for(self, padded_bs, cg_slot):
        return self.sampler

    def scatter_offset(self, cg_slot=0):
        self.offset_scatters += 1

    def register_request(self, rid, sampling_config):
        pass

    def unregister_request(self, rid):
        pass


def _resource(*, capable: bool = True) -> SamplerResource:
    res = SamplerResource(
        vocab_size=VOCAB,
        enable_repetion_penalty=capable,
        device=torch.device("cpu"),
    )
    # Skip build_cuda_graph_buffers: the real buffers would allocate a
    # [capacity, VOCAB] mask, and the gate is what's under test, not the copies.
    res._cg_buffers = _RecordingBuffers()
    return res


def _ctx(rids, *, is_preplan: bool = False) -> StepContext:
    bucket = BucketKey(graph_walk="decode", bs=len(rids), num_tokens=len(rids))
    return StepContext(
        request_ids=tuple(rids),
        graph_walk="decode",
        slot=0,
        capture=False,
        is_preplan=is_preplan,
        slot_lease=SlotLease(slot=0, bucket=bucket),
    )


def _drive(res, rids, *, apply_penalty=True, prefill_tokens=None):
    """One step in `_drive_step`'s order: admit -> plan -> forward -> commit."""
    step = SamplerStep(
        apply_penalty=apply_penalty,
        prefill_tracked_tokens=prefill_tokens or {},
    )
    ctx = _ctx(rids)
    res.admit(step, ctx)
    res.plan(step, ctx)
    res.sample(list(rids), torch.zeros(len(rids), VOCAB))
    res.commit(step, ctx)


def _ingest(res, **rid_to_penalty):
    for rid, penalty in rid_to_penalty.items():
        res.ingest_request(rid, SamplingReqConfig(repetition_penalty=penalty))


# ── the gate ────────────────────────────────────────────────────────────


def test_mask_traffic_skipped_when_nobody_asks_for_a_penalty():
    res = _resource()
    _ingest(res, a=1.0, b=1.0)
    assert res._penalty_live is False

    _drive(res, ["a", "b"])
    buffers = res._cg_buffers

    assert buffers.staged == 0, "staged a mask nothing would read"
    assert buffers.gathered_seen == [False], "gathered a mask nothing would read"
    assert buffers.sampler.synced == 0, "synced a mask nothing wrote to"
    # the offset is per-step state regardless of the penalty
    assert buffers.static_gathers == 1
    assert buffers.offset_scatters == 1


def test_capability_survives_the_skip():
    """The kernel variant is baked at capture, so `sample` must keep saying
    True even on steps where the bookkeeping was skipped."""
    res = _resource()
    _ingest(res, a=1.0)
    _drive(res, ["a"])

    assert res._apply_penalty_this_step is True
    assert res._penalty_needed_this_step is False
    assert res._cg_buffers.sampler.sampled_with == [True]


def test_mask_traffic_engaged_when_a_request_asks():
    res = _resource()
    _ingest(res, a=1.0, b=1.05)
    assert res._penalty_live is True

    _drive(res, ["a", "b"])
    buffers = res._cg_buffers

    assert buffers.staged == 1
    assert buffers.gathered_seen == [True]
    assert buffers.sampler.synced == 1


def test_gate_follows_residency_not_the_batch():
    """A penalised request anywhere on the resource keeps every mask live, and
    its departure lets the gate close again."""
    res = _resource()
    _ingest(res, a=1.0, b=1.05)

    _drive(res, ["a"])  # batch excludes the penalised request
    assert res._cg_buffers.gathered_seen == [True]

    res.remove_request("b")
    assert res._penalty_live is False
    _drive(res, ["a"])
    assert res._cg_buffers.gathered_seen == [True, False]


def test_step_may_decline_the_penalty():
    """The Talker's code-predictor declares apply_penalty=False; that must win
    over residency."""
    res = _resource()
    _ingest(res, a=1.05)

    _drive(res, ["a"], apply_penalty=False)

    assert res._apply_penalty_this_step is False
    assert res._penalty_needed_this_step is False
    assert res._cg_buffers.staged == 0
    assert res._cg_buffers.sampler.sampled_with == [False]


def test_incapable_spec_never_engages():
    res = _resource(capable=False)
    _ingest(res, a=1.05)

    assert res._penalty_live is False
    _drive(res, ["a"])

    assert res._apply_penalty_this_step is False
    assert res._cg_buffers.gathered_seen == [False]


# ── prompt-token recording ──────────────────────────────────────────────


def test_prefill_tokens_recorded_only_while_live():
    res = _resource()
    _ingest(res, a=1.0)
    tokens = torch.tensor([3, 5, 7])

    _drive(res, ["a"], prefill_tokens={"a": tokens})
    assert not res._sampler.get_token_mask("a")._seen_token_mask.any(), \
        "recorded prompt tokens into a mask nothing reads"

    _ingest(res, b=1.05)  # now someone reads masks
    _drive(res, ["a"], prefill_tokens={"a": tokens})
    assert res._sampler.get_token_mask("a")._seen_token_mask.sum().item() == 3


# ── flag lifetime ───────────────────────────────────────────────────────


def test_preplan_leaves_the_flags_alone():
    """The flags are latched on the non-preplan path only, which is why they
    need no preplan twin the way `_cg_sampler` does."""
    res = _resource()
    _ingest(res, a=1.05)
    _drive(res, ["a"])
    assert res._penalty_needed_this_step is True

    # a step planned a slot ahead, declaring the opposite
    res.admit(SamplerStep(apply_penalty=False), _ctx(["a"], is_preplan=True))
    assert res._apply_penalty_this_step is True
    assert res._penalty_needed_this_step is True


def test_desynced_bookkeeping_trips_the_assert():
    """`_penalty_rids` going stale (e.g. a mid-request config update that
    forgets to refresh it) must fail loudly, not silently drop the penalty."""
    res = _resource()
    _ingest(res, a=1.05)
    res._penalty_rids.clear()  # simulate the desync

    with pytest.raises(AssertionError, match="out of sync"):
        _drive(res, ["a"])
