"""A step that runs verify rows beside prompt rows (``with_mixed_verify``): its declaration, the
verify rows' last verdicts taken back on the device over a ragged layout while the prompt rows
take nothing back, a commit that records only the verify rows, the drafts written into the
verify rows' positions alone, and what keeps a combined walk over a verifying walk from loading."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources import (
    DraftPolicy,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    StepContext,
    SubmoduleStep,
    VerifyStep,
    with_mixed_verify,
)
from mstar.engine.resources.base import RewindKind
from mstar.engine.resources.draft_verify.resource import DraftVerifyResource
from mstar.engine.resources.draft_verify.startup import mixed_verify_walk, verify_walk
from mstar.engine.resources.sampler.resource import SamplerResource

K = 3
CPU = torch.device("cpu")
POLICY = DraftPolicy(max_drafts=K, device_rewind=True)


def _mixed(rows):
    """rows: (rid, span, verifies)."""
    segments = [Segment(rid, "main", span) for rid, span, _ in rows]
    steps = {
        "kv": KVStep(), "pos": PositionStep(),
        "sampler": SamplerStep(apply_penalty=False, kept_rids=frozenset(r for r, _, v in rows if not v)),
    }
    return with_mixed_verify(
        SubmoduleStep(steps=steps, segments=segments), POLICY, [r for r, _, v in rows if v], "dv", "sampler",
    )


def test_a_mixed_step_runs_its_verify_rows_first_a_block_each():
    step = _mixed([(1, K + 1, True), (2, K + 1, True), (3, 5, False), (-1, 0, False)])
    assert step.get("dv").rows == frozenset({1, 2})
    # paged KV stages a rewindable plan over rows of any span; the sampler keeps the prompts' draws
    assert step.get("kv").verify and step.get("kv").mixed
    assert not step.get("sampler").verify and step.get("sampler").kept_rids == frozenset({3, -1})
    with pytest.raises(ValueError, match="verify rows first"):
        _mixed([(3, 5, False), (1, K + 1, True)])
    with pytest.raises(ValueError, match="spans the block"):
        _mixed([(1, K, True), (3, 5, False)])


class _Sampler:
    def __init__(self):
        self.next = None

    def verify(self, request_ids, block, logits, draft_probs=None):
        return self.next


class _Corrector:
    """Records what a resource whose plan reads lengths is asked to take back."""

    device_correct = False

    def __init__(self):
        self.seen = []

    def correct_on_device(self, rejected):
        self.seen.append(rejected)


def _step(res, sampler, rows, accepted):
    """rows: (rid, span, verifies); a step through the resource as the forward runs it."""
    rids = [rid for rid, _, _ in rows]
    ctx = StepContext(request_ids=tuple(rids), graph_walk="mixed", slot=0, capture=False)
    verify = [rid for rid, _, v in rows if v]
    step = VerifyStep(
        segments=tuple(Segment(rid, "main", span) for rid, span, _ in rows), drafts=K,
        rows=None if len(verify) == len(rows) else frozenset(verify),
    )
    res.plan(step, ctx)
    res.apply_previous()
    n = len(rows)
    tokens = torch.arange(n * (K + 1)).view(n, K + 1) + 10
    sampler.next = (tokens, torch.tensor(list(accepted) + [0] * (n - len(accepted))))
    res.verify(rids, torch.zeros(n, K + 1, dtype=torch.long), torch.zeros(n * (K + 1), 4))
    res.commit(step, ctx)
    return step


def test_a_mixed_step_takes_back_its_verify_rows_and_leaves_its_prompts():
    res = DraftVerifyResource(CPU)
    sampler = _Sampler()
    res.bind_sampler(sampler)
    corrector = _Corrector()
    res.enable_device_rewind([corrector])
    for rid in (1, 2, 3):
        res.ingest_request(rid)
    _step(res, sampler, [(1, K + 1, True), (2, K + 1, True)], accepted=[1, K])
    # 3 joins beside them: its prompt's 5 tokens take nothing back, 1 its 2 rejected rows
    _step(res, sampler, [(2, K + 1, True), (1, K + 1, True), (3, 5, False)], accepted=[0, 2, 0])
    rejected = corrector.seen[-1]
    assert rejected.span is None
    assert rejected.rows.tolist() == [0, 2, 0]
    assert rejected.tokens.tolist() == [0] * (K + 1) + [2] * (K + 1) + [0] * 5
    assert rejected.token_rows.tolist() == [0] * (K + 1) + [1] * (K + 1) + [2] * 5
    assert rejected.offsets.tolist() == [*range(K + 1), *range(K + 1), *range(5)]
    # only the verify rows' verdicts are recorded, at their rows of the step's mirror
    record = res.take_record()[-1]
    assert record.rows == (2, 1) and record.positions == (0, 1)
    assert set(res._pending) == {1, 2} and res._last_rows == {2: 0, 1: 1}


def test_a_mixed_step_drafts_into_its_verify_rows_alone():
    res = DraftVerifyResource(CPU)
    res.bind_sampler(_Sampler())
    for rid in (1, 3):
        res.ingest_request(rid)
    rows = [(1, K + 1, True), (3, 2, False)]
    step = VerifyStep(segments=tuple(Segment(r, "main", s) for r, s, _ in rows), drafts=K, rows=frozenset({1}))
    res.plan(step, StepContext(request_ids=(1, 3), graph_walk="mixed", slot=0, capture=False))
    ids = torch.tensor([7, -1, -1, -1, 40, 41])
    # the block covers the step's two rows, eight tokens, past the six it has
    block = torch.tensor([[7, 1, 2, 3], [9, 9, 9, 9]])
    assert res.place_block(ids, block).tolist() == [7, 1, 2, 3, 40, 41]


class _Sub:
    def __init__(self, policy):
        self.policy = policy

    def get_draft_policy(self, walk):
        return self.policy if walk == "decode" else None

    def device_loopback_signals(self, walk):
        return frozenset()


class _Lengths:
    """A resource whose plan reads lengths and corrects it on the device."""

    rewind_kind = RewindKind.HOST
    plan_reads_lengths = True
    corrects_on_device = True
    device_correct = False

    def __init__(self):
        self.mixed = []

    def correct_mixed_walk(self, walk):
        self.mixed.append(walk)


def _resources(sampled=False):
    lengths = _Lengths()
    sampler = SamplerResource(
        vocab_size=8, enable_repetion_penalty=False, device=CPU, draft_verify=True, enable_sampled_verify=sampled,
    )
    return {"draft_verify": DraftVerifyResource(CPU), "sampler": sampler, "attn": lengths}


def test_a_combined_walk_carries_verify_rows_only_when_they_come_back_on_the_device():
    resources = _resources()
    members = {"prefill", "decode"}
    walks = {walk: verify_walk("llm", _Sub(POLICY), resources, walk, members) for walk in members}
    found = mixed_verify_walk("llm", "mixed", members, walks, resources)
    assert found.rows_walk == "decode" and found.policy is POLICY
    # every step of the combined walk is corrected on the device, whatever its tokens a row
    assert resources["attn"].mixed == ["mixed"]
    with pytest.raises(ValueError, match="come back on the host"):
        verify_walk("llm", _Sub(DraftPolicy(max_drafts=K)), _resources(), "decode", members)
    with pytest.raises(ValueError, match="rejection sampling"):
        verify_walk("llm", _Sub(POLICY), _resources(sampled=True), "decode", members)
