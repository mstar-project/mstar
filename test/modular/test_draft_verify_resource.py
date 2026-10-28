"""The draft-verify resource on its own: a verify step's verdict from the forward's staging to
the rewinds and the emitted tokens, the slots its mirrors live in, the greedy verify, the
runner's rewind hook, and the declarations a walk makes."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources import (
    DraftPolicy,
    KVStep,
    PositionConfig,
    PositionStep,
    Segment,
    StepContext,
    SubmoduleStep,
    VerifyStep,
    with_draft_verify,
)
from mstar.engine.resources.base import RewindKind
from mstar.engine.resources.draft_verify.config import VerdictBatch
from mstar.engine.resources.draft_verify.resource import DraftVerifyResource
from mstar.engine.resources.draft_verify.startup import verify_walk
from mstar.engine.resources.kv import manager as kv_manager
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.position.config import PosScheme
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import BucketKey, SlotLease

K = 3
CPU = torch.device("cpu")
POLICY = DraftPolicy(max_drafts=K, stop_ids=frozenset({9}))


class _Sampler:
    """Hands ``verify`` the verdict a test wants, in place of an argmax."""

    def __init__(self):
        self.next = None

    def verify(self, request_ids, block, logits):
        return self.next


def _resource():
    res = DraftVerifyResource(CPU)
    sampler = _Sampler()
    res.bind_sampler(sampler)
    return res, sampler


def _ctx(rids, padded=None, slot=None, capture=False):
    ctx = StepContext(request_ids=tuple(rids), graph_walk="decode", slot=0, capture=capture)
    if padded is not None:
        ctx.set_padded_rids(tuple(padded))
    if slot is not None:
        rows = len(padded or rids)
        ctx.slot_lease = SlotLease(slot=slot, bucket=BucketKey("decode", rows, rows * (K + 1)))
    return ctx


def _verify_step(res, sampler, rids, accepted, tokens=None, padded=None, slot=None, capture=False, drafts=K):
    """One verify step through the resource: plan, the forward's verify, commit."""
    rows = list(padded or rids)
    ctx = _ctx(rids, padded, slot, capture)
    step = VerifyStep(segments=tuple(Segment(r, "main", drafts + 1) for r in rows), drafts=drafts)
    res.plan(step, ctx)
    n = len(rows)
    if tokens is None:
        tokens = [[10 * (i + 1) + j for j in range(drafts + 1)] for i in range(n)]
    accepted = list(accepted) + [0] * (n - len(accepted))
    sampler.next = (torch.tensor(tokens), torch.tensor(accepted))
    block = torch.full((n, drafts + 1), 5)
    out = res.verify(rows, block, torch.zeros(n * (drafts + 1), 4))
    res.commit(step, ctx)
    return out


def test_a_verdict_reaches_the_rewind_once():
    res, sampler = _resource()
    for rid in (1, 2, 3):
        res.ingest_request(rid)
    out = _verify_step(res, sampler, [1, 2, 3], accepted=[1, 3, 0])
    assert out.next_inputs.view(-1).tolist() == [11, 23, 30]
    # 2 kept every draft: nothing to take back, so it is left out
    assert res.settle([1, 2, 3]) == VerdictBatch([1, 3], ["main", "main"], [2, 3])
    assert res.settle([1, 2, 3]) is None, "a verdict is handed out once"


def test_only_the_rows_asked_for_are_settled():
    res, sampler = _resource()
    for rid in (1, 2):
        res.ingest_request(rid)
    _verify_step(res, sampler, [1, 2], accepted=[0, 0])
    assert res.settle([2]).request_ids == [2]
    # 1 is held out of a step; its verdict waits for the step it is next in
    assert res.settle([1]).request_ids == [1]


def test_padding_and_capture_rows_register_nothing():
    res, sampler = _resource()
    res.ingest_request(1)
    _verify_step(res, sampler, [1], accepted=[0, 0], padded=[1, -5], slot=0)
    assert res.settle([1, -5]) == VerdictBatch([1], ["main"], [3])
    _verify_step(res, sampler, [-1, -2], accepted=[0, 0], padded=[-1, -2], slot=1, capture=True)
    assert not res._pending
    assert [record.rows for record in res._steps] == [(1,)]


def test_removal_drops_a_pending_verdict():
    res, sampler = _resource()
    res.ingest_request(1)
    _verify_step(res, sampler, [1], accepted=[0])
    res.remove_request(1)
    assert res.settle([1]) is None
    # a recycled handle starts clean
    res.ingest_request(1)
    assert res.settle([1]) is None


def test_a_slot_is_read_out_before_its_next_step_writes_it():
    """Two capture slots alternate. A request held out of a step leaves its verdict in its
    slot's mirror, and the next step on that slot must not hand it the newer rows."""
    res, sampler = _resource()
    res.build_accelerator_graph_buffers([], max_bs=4, max_seq_len=64)
    for rid in (1, 2):
        res.ingest_request(rid)
    _verify_step(res, sampler, [1, 2], accepted=[1, 2], slot=0)
    assert res.settle([1]).rejected == [2]
    _verify_step(res, sampler, [1], accepted=[0], slot=1)
    assert res.settle([1]).rejected == [3]
    # 2 is still pending in slot 0; this step overwrites slot 0's row 1 with accepted 3
    _verify_step(res, sampler, [1, 3], accepted=[3, 3], slot=0)
    assert res.settle([2]) == VerdictBatch([2], ["main"], [1])


def test_an_overwritten_verdict_is_an_error():
    res, sampler = _resource()
    res.ingest_request(1)
    _verify_step(res, sampler, [1], accepted=[0], slot=0)
    res._mirrors[0].seq_np[0] += 1
    with pytest.raises(RuntimeError, match="overwritten"):
        res.settle([1])


def test_an_eager_verify_step_must_call_verify():
    res, _ = _resource()
    res.ingest_request(1)
    step = VerifyStep(segments=(Segment(1, "main", K + 1),), drafts=K)
    ctx = _ctx([1])
    res.plan(step, ctx)
    with pytest.raises(RuntimeError, match="did not call"):
        res.commit(step, ctx)


def test_verifying_twice_without_a_rewind_is_an_error():
    res, sampler = _resource()
    res.ingest_request(1)
    _verify_step(res, sampler, [1], accepted=[0])
    with pytest.raises(RuntimeError, match="before its last verdict was rewound"):
        _verify_step(res, sampler, [1], accepted=[0])


def _emit(res, sampler, rids, accepted, tokens, max_tokens, ignore_eos=None, inputs=None, policy=POLICY):
    n = len(rids)
    ctx = _ctx(rids)
    step = VerifyStep(segments=tuple(Segment(r, "main", K + 1) for r in rids), drafts=K)
    res.plan(step, ctx)
    sampler.next = (torch.tensor(tokens), torch.tensor(accepted))
    block = torch.tensor(inputs if inputs is not None else [5] * n).view(n, 1).expand(n, K + 1)
    res.verify(rids, block, torch.zeros(n * (K + 1), 4))
    res.commit(step, ctx)
    rows = res.emitted_rows(rids, max_tokens, ignore_eos or [False] * n, policy)
    res.settle(rids)
    return rows


def test_the_emitted_tokens_stop_at_a_stop_id_and_at_max_tokens():
    res, sampler = _resource()
    for rid in range(5):
        res.ingest_request(rid)
    rows = _emit(
        res, sampler, [0, 1, 2, 3, 4],
        accepted=[3, 3, 1, 3, 2],
        tokens=[[1, 2, 3, 4], [1, 9, 3, 4], [9, 2, 3, 4], [1, 2, 3, 4], [1, 2, 9, 4]],
        # the prefill emitted one: 4 + 1 of budget for row 0, 2 for row 3
        max_tokens=[10, 10, 10, 3, 10], ignore_eos=[False, False, False, False, True],
    )
    assert rows.counts.tolist() == [4, 2, 1, 2, 3]
    assert rows.tokens.tolist() == [1, 2, 3, 4, 1, 9, 9, 1, 2, 1, 2, 9]
    # a stop id ends rows 1 and 2, the cap row 3; row 4 ignores its stop id
    assert rows.done.tolist() == [False, True, True, True, False]
    # the counts carry over: row 0 has 5 of its 10
    rows = _emit(res, sampler, [0], accepted=[3], tokens=[[1, 2, 3, 4]], max_tokens=[7])
    assert rows.counts.tolist() == [2] and rows.done.tolist() == [True]


@pytest.mark.parametrize(("max_tokens", "counts", "done"), [
    (100, [2, 1, 3], [True, True, False]),
    (3, [2, 1, 2], [True, True, True]),
], ids=["stop_inside_the_block", "the_cap_counts_tokens"])
def test_multi_token_rows_stop_on_tokens_not_steps(max_tokens, counts, done):
    """Rows of three tokens, as a k = 2 verify step emits them: a stop token past a row's
    first value ends it there, and the cap counts tokens, so one step of three tokens can
    reach a cap of 3 (the prefill emitted one)."""
    policy = DraftPolicy(max_drafts=2, stop_ids=frozenset({7}))
    res, sampler = _resource()
    rids = [0, 1, 2]
    for rid in rids:
        res.ingest_request(rid)
    ctx = _ctx(rids)
    step = VerifyStep(segments=tuple(Segment(r, "main", 3) for r in rids), drafts=2)
    res.plan(step, ctx)
    sampler.next = (torch.tensor([[1, 7, 2], [7, 1, 1], [1, 1, 1]]), torch.tensor([2, 2, 2]))
    res.verify(rids, torch.full((3, 3), 5), torch.zeros(9, 4))
    res.commit(step, ctx)
    rows = res.emitted_rows(rids, [max_tokens] * 3, [False] * 3, policy)
    assert rows.counts.tolist() == counts
    assert rows.done.tolist() == done


def test_the_stats_carry_the_acceptance_by_position():
    """Rows halving at each position: every draft is accepted half the time given the ones
    before it were."""
    res, sampler = _resource()
    rids = list(range(8))
    for rid in rids:
        res.ingest_request(rid)
    _emit(res, sampler, rids, accepted=[0, 0, 0, 0, 1, 1, 2, 3], tokens=[[1, 2, 3, 4]] * 8, max_tokens=[100] * 8)
    assert res.accepted_hist.tolist() == [4, 2, 1, 1]
    assert res.stats() == (
        "8 verify rows, 0.88 drafts accepted per row (1.88 tokens per step); "
        "accepted by position 0.50 0.50 0.50"
    )


def test_a_row_whose_input_is_a_stop_id_emits_nothing():
    """The prefill sampled the stop token, or the step before emitted it: the reply had
    already ended when this step ran."""
    res, sampler = _resource()
    for rid in (0, 1):
        res.ingest_request(rid)
    rows = _emit(res, sampler, [0, 1], accepted=[2, 2], tokens=[[1, 2, 3, 4]] * 2,
                 max_tokens=[10, 10], ignore_eos=[False, True], inputs=[9, 9])
    assert rows.counts.tolist() == [0, 3]
    assert rows.done.tolist() == [True, False]


def test_a_step_past_the_cap_emits_nothing():
    res, sampler = _resource()
    res.ingest_request(0)
    rows = _emit(res, sampler, [0], accepted=[3], tokens=[[1, 2, 3, 4]], max_tokens=[2])
    assert rows.counts.tolist() == [1]
    # the async worker's step after the stop
    rows = _emit(res, sampler, [0], accepted=[3], tokens=[[1, 2, 3, 4]], max_tokens=[2])
    assert rows.counts.tolist() == [0] and rows.done.tolist() == [True]


def test_one_step_per_request_emits_in_the_batch_order():
    """An unbatched exec runs a verify step per request; the emit reads each from its own."""
    res, sampler = _resource()
    for rid in (0, 1):
        res.ingest_request(rid)
    for rid, tokens in ((1, [[7, 7, 7, 7]]), (0, [[3, 3, 3, 3]])):
        ctx = _ctx([rid])
        step = VerifyStep(segments=(Segment(rid, "main", K + 1),), drafts=K)
        res.plan(step, ctx)
        sampler.next = (torch.tensor(tokens), torch.tensor([1]))
        res.verify([rid], torch.full((1, K + 1), 5), torch.zeros(K + 1, 4))
        res.commit(step, ctx)
    rows = res.emitted_rows([0, 1], [10, 10], [False, False], POLICY)
    assert rows.tokens.tolist() == [3, 3, 7, 7]


def test_greedy_verify_accepts_the_leading_drafts_the_target_agrees_with():
    sampler = SamplerResource(vocab_size=8, enable_repetion_penalty=False, device=CPU)
    # rows: [input, d1, d2, d3]; the target's argmax at each row predicts the next token
    block = torch.tensor([[0, 1, 2, 3], [0, 1, 5, 3], [0, 4, 2, 3], [0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 3, 6], [1, 2, 7, 6]])
    logits = torch.nn.functional.one_hot(targets.view(-1), 8).float()
    tokens, accepted = sampler.verify([0, 1, 2, 3], block, logits)
    assert tokens.tolist() == targets.tolist()
    # all three; d2 is wrong; d1 is wrong; d3 would match but d2 didn't
    assert accepted.tolist() == [3, 1, 0, 2]
    one = sampler.verify([0], torch.tensor([[4]]), torch.nn.functional.one_hot(torch.tensor([2]), 8).float())
    assert one[0].tolist() == [[2]] and one[1].tolist() == [0]


# ── the runner's rewind ──


def _kv():
    return kv_manager.KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=256,
                          max_num_pages=64, page_size=4),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )


@pytest.fixture
def _stub_transfer(monkeypatch):
    class _Transfer:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(kv_manager, "KVTransferManager", _Transfer)


def _runner():
    dv, sampler = _resource()
    kv = _kv()
    pos = RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    runner = StepRunner({"kv": kv, "pos": pos, "dv": dv}, node_resources={"llm": ["kv", "pos", "dv"]})
    for res in (kv, pos, dv):
        res.ingest_request(1)
    return runner, kv, pos, dv, sampler


def _cycle(runner, sampler, span, accepted=None, preplan=False):
    segments = [Segment(1, "main", span)]
    steps = {"kv": KVStep(), "pos": PositionStep()}
    step = SubmoduleStep(steps=steps, segments=segments)
    if accepted is not None:
        step = with_draft_verify(step, DraftPolicy(max_drafts=span - 1), "dv")
    ctx = StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False)
    step.set_ctx(ctx)
    if preplan:
        runner.pre_admit(step)
        runner.clear_preplan()
    assert runner.admit(step).ok
    runner.plan(step)
    if accepted is not None:
        sampler.next = (torch.arange(span).view(1, span), torch.tensor([accepted]))
        runner.resources["dv"].verify([1], torch.zeros(1, span, dtype=torch.long), torch.zeros(span, 4))
    runner.commit(step)


@pytest.mark.usefixtures("_stub_transfer")
@pytest.mark.parametrize("preplan", [False, True], ids=["admit", "pre_admit_then_admit"])
def test_the_runner_rewinds_once_before_the_next_step_admits(preplan):
    runner, kv, pos, dv, sampler = _runner()
    _cycle(runner, sampler, 5)  # a prompt
    _cycle(runner, sampler, K + 1, accepted=1)
    assert kv._streams[1]["main"].stored_len == pos.position(1, "main") == 5 + K + 1
    # the next step takes back the 2 rejected rows before it reserves anything, once, even
    # when a pre-admit ran it first and its stage was dropped
    _cycle(runner, sampler, K + 1, accepted=K, preplan=preplan)
    assert kv._streams[1]["main"].stored_len == pos.position(1, "main") == 5 + 2 + K + 1
    _cycle(runner, sampler, K + 1, accepted=0)
    assert kv._streams[1]["main"].stored_len == 5 + 2 + 4 + K + 1


@pytest.mark.usefixtures("_stub_transfer")
def test_a_runner_with_no_draft_verify_resource_has_no_hook():
    kv = _kv()
    runner = StepRunner({"kv": kv}, node_resources={"llm": ["kv"]})
    assert runner._rewinders == {}


def test_the_resources_declare_what_a_rewind_takes():
    pos = RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    assert pos.rewind_kind is RewindKind.HOST
    block = RopeManager(config=PositionConfig(kv_cache="kv", scheme=PosScheme.BLOCK), device=CPU)
    assert block.rewind_kind is None
    assert kv_manager.KVManager.rewind_kind is RewindKind.HOST
    assert DraftVerifyResource.rewind_kind is RewindKind.NONE
    assert SamplerResource.rewind_kind is RewindKind.NONE


def test_the_position_rewinds_beside_the_kv_length():
    pos = RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    pos.ingest_request(1)
    pos._counters[1]["main"] = 7
    pos.rewind(VerdictBatch([1], ["main"], [3]))
    assert pos.position(1, "main") == 4
    pos.rewind(VerdictBatch([1], ["main"], [10]))
    assert pos.position(1, "main") == 0


# ── declarations ──


def test_a_policy_names_its_block_and_its_lookahead():
    assert (POLICY.block, POLICY.lookahead) == (K + 1, 2 * K + 1)
    with pytest.raises(ValueError, match="at least 1"):
        DraftPolicy(max_drafts=0)
    with pytest.raises(ValueError, match="integer"):
        DraftPolicy(max_drafts=2, emitted_before=-1)


def test_with_draft_verify_adds_the_verify_step_to_rows_of_one_block():
    step = SubmoduleStep(steps={"kv": KVStep()}, segments=[Segment(1, "main", K + 1), Segment(-2, "main", K + 1)])
    out = with_draft_verify(step, POLICY, "dv")
    assert isinstance(out.get("dv"), VerifyStep) and out.get("dv").drafts == K
    assert out.get("dv").segments == out.get("kv").segments
    short = SubmoduleStep(steps={"kv": KVStep()}, segments=[Segment(1, "main", 1)])
    with pytest.raises(ValueError, match="set input_seq_len to policy.block"):
        with_draft_verify(short, POLICY, "dv")


def test_a_walk_the_prefix_cache_keys_cannot_verify():
    class Sub:
        def get_draft_policy(self, walk):
            return POLICY

        def device_loopback_signals(self, walk):
            return frozenset()

    dv = DraftVerifyResource(CPU)
    sampler = SamplerResource(vocab_size=8, enable_repetion_penalty=False, device=CPU, draft_verify=True)
    resources = {"draft_verify": dv, "sampler": sampler}
    assert verify_walk("llm", Sub(), resources, "decode").policy is POLICY
    with pytest.raises(ValueError, match="keyed for the prefix cache"):
        verify_walk("llm", Sub(), resources, "decode", keyed=True)

    class Bad(Sub):
        def get_draft_policy(self, walk):
            return {"max_drafts": 3}

    with pytest.raises(TypeError, match="returned a dict"):
        verify_walk("llm", Bad(), resources, "decode")
