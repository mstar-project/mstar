"""The draft-verify resource on its own: a verify step's verdict from the forward's staging to
the rewinds and the emitted tokens, the slots its mirrors live in, the greedy verify, the
runner's rewind hook, and the declarations a walk makes; an adaptive walk's steps of no drafts
and the choice between the two."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources import (
    DraftPolicy,
    KVStep,
    PositionConfig,
    PositionStep,
    SamplerStep,
    Segment,
    StepContext,
    SubmoduleStep,
    VerifyStep,
    with_draft_verify,
)
from mstar.engine.resources.base import RewindKind
from mstar.engine.resources.draft_verify.config import VerdictBatch, VerifyKey, verify_captures
from mstar.engine.resources.draft_verify.resource import DraftVerifyResource
from mstar.engine.resources.draft_verify.startup import verify_walk
from mstar.engine.resources.kv import manager as kv_manager
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager
from mstar.engine.resources.position.config import PosScheme
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.model.submodule_base import ARNodeInputs

K = 3
CPU = torch.device("cpu")
POLICY = DraftPolicy(max_drafts=K, stop_ids=frozenset({9}))
ADAPTIVE = DraftPolicy(max_drafts=K, stop_ids=frozenset({9}), proposer="ngram", adaptive=True)


class _Sampler:
    """Hands ``verify`` the verdict a test wants, in place of an argmax."""

    def __init__(self):
        self.next = None

    def verify(self, request_ids, block, logits, draft_probs=None):
        return self.next


def _resource():
    res = DraftVerifyResource(CPU)
    sampler = _Sampler()
    res.bind_sampler(sampler)
    return res, sampler


class _Proposer:
    """Records what the draft-verify resource hands an engine-side proposer; predicts what a
    test sets."""

    predicts = True
    max_drafts = K

    def __init__(self):
        self.observed: list[tuple[int, list[int]]] = []
        self.predicted: list[tuple[list[int], int]] = []
        self.gain: dict[int, float] = {}

    def observe(self, rid, row):
        self.observed.append((rid, row.tolist()))

    def predict(self, request_ids, drafts):
        self.predicted.append((list(request_ids), drafts))

    def expected_gain(self, request_ids):
        return sum(self.gain.get(rid, 0.0) for rid in request_ids)


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
    res._mirrors[(0, K + 3)].seq_np[0] += 1
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


# ── an adaptive walk's steps of no drafts ──


def test_a_step_of_no_drafts_is_counted_at_commit_and_read_at_the_stop_check():
    res, sampler = _resource()
    for rid in (0, 1, 2):
        res.ingest_request(rid)
    _verify_step(res, sampler, [0, 1, 2], accepted=[0, 0, 0], tokens=[[4], [9], [6]], drafts=0)
    assert res.ran_plain()
    # counted without its tokens: one a row; row 2 reaches its max_tokens of 2 (the prefill
    # emitted one)
    plain = res.count_plain([0, 1, 2], [10, 10, 2], ADAPTIVE)
    assert plain.counts.tolist() == [1, 1, 1] and plain.capped.tolist() == [False, False, True]
    # nothing for a rewind to wait for, so nothing read yet
    assert res.settle([0, 1, 2]) is None
    assert set(res._pending) == {0, 1, 2}
    # the tokens the worker inlines to the client, off the mirror the forward filled
    assert res.plain_tokens(plain).tolist() == [4, 9, 6]
    rows = res.plain_rows(plain, [False, False, False], ADAPTIVE)
    assert rows.tokens.tolist() == [4, 9, 6] and rows.counts.tolist() == [1, 1, 1]
    # 9 is a stop id
    assert rows.done.tolist() == [False, True, True]
    assert res.stats().endswith("0 verify steps, 1 steps of no drafts (3 rows)")
    # the counts carry over to the rows' next step: row 1 has 2 of 2 now
    _verify_step(res, sampler, [1], accepted=[0], tokens=[[7]], drafts=0)
    plain = res.count_plain([1], [2], ADAPTIVE)
    assert plain.counts.tolist() == [0] and plain.capped.tolist() == [True]
    assert res.plain_rows(plain, [False], ADAPTIVE).tokens.tolist() == []


def test_a_step_of_no_drafts_after_a_stop_emits_nothing():
    res, sampler = _resource()
    for rid in (0, 1):
        res.ingest_request(rid)
    ctx = _ctx([0, 1])
    step = VerifyStep(segments=(Segment(0, "main", 1), Segment(1, "main", 1)), drafts=0)
    res.plan(step, ctx)
    sampler.next = (torch.tensor([[4], [5]]), torch.tensor([0, 0]))
    # both rows' input is the stop id: the reply ended a step earlier, unless it ignores eos
    res.verify([0, 1], torch.tensor([[9], [9]]), torch.zeros(2, 4))
    res.commit(step, ctx)
    rows = res.plain_rows(res.count_plain([0, 1], [10, 10], ADAPTIVE), [False, True], ADAPTIVE)
    assert rows.counts.tolist() == [0, 1] and rows.done.tolist() == [True, False]


def test_a_plain_steps_verdict_is_read_at_the_rows_next_commit():
    """Nothing reads a step of no drafts' verdict for a rewind, so the rows' next commit does,
    behind the forward it just launched, before its own takes its place: the proposer has every
    verdict before a step when it predicts from it."""
    res, sampler = _resource()
    prop = _Proposer()
    res.bind_proposer(prop)
    for rid in (0, 1):
        res.ingest_request(rid)
    _verify_step(res, sampler, [0, 1], accepted=[0, 0], tokens=[[4], [6]], drafts=0)
    assert prop.observed == [] and prop.predicted == [([0, 1], 0)]
    _verify_step(res, sampler, [0, 1], accepted=[2, 0], drafts=K)
    # [accepted, input, t_0]: the block's input was 5
    assert prop.observed == [(0, [0, 5, 4]), (1, [0, 5, 6])]
    assert prop.predicted[-1] == ([0, 1], K)
    # a verify step's verdict waits for the rewind; a request verified again before it is a bug
    assert res.settle([0, 1]) == VerdictBatch([0, 1], ["main", "main"], [1, 3])
    _verify_step(res, sampler, [0], accepted=[0], drafts=K)
    with pytest.raises(RuntimeError, match="before its last verdict was rewound"):
        _verify_step(res, sampler, [0], accepted=[0], drafts=0)


def test_the_two_blocks_keep_their_own_mirrors():
    """Both blocks of an adaptive walk replay on the same capture slots, each graph reading its
    own verdict rows in place."""
    res, sampler = _resource()
    res.build_accelerator_graph_buffers([], max_bs=2, max_seq_len=64)
    for rid in (1, 2):
        res.ingest_request(rid)
    _verify_step(res, sampler, [1, 2], accepted=[1, 2], slot=0)
    res.settle([1, 2])
    _verify_step(res, sampler, [1, 2], accepted=[0, 0], slot=0, drafts=0, tokens=[[3], [4]])
    assert set(res._mirrors) == {(0, K + 3), (0, 3)}
    assert res._mirrors[(0, 3)].width == 3 and res._mirrors[(0, K + 3)].width == K + 3


def test_an_adaptive_step_drafts_when_its_rows_should_keep_enough():
    res, _ = _resource()
    prop = _Proposer()
    res.bind_proposer(prop)
    prop.gain = {0: 2.0}
    assert res.choose_drafts([0], ADAPTIVE) == K
    # 2.0 kept across four rows pays min_gain 0.5 a row; across five it doesn't
    assert res.choose_drafts([0, 1, 2, 3], ADAPTIVE) == K
    assert res.choose_drafts([0, 1, 2, 3, 4], ADAPTIVE) == 0
    assert res.choose_drafts([1, 2], ADAPTIVE) == 0
    # a walk that always drafts doesn't ask
    assert res.choose_drafts([1], POLICY) == K


def test_an_adaptive_walk_verifies_a_block_or_runs_one_token():
    def step(*spans):
        return SubmoduleStep(
            steps={"kv": KVStep(), "lin": LinearAttnStep()},
            segments=[Segment(rid, "main", span) for rid, span in enumerate(spans)],
        )

    one = with_draft_verify(step(1, 1), ADAPTIVE, "dv")
    assert one.get("dv").drafts == 0 and one.cg_key_info == VerifyKey(0)
    # the plain paths: no host write slots, no state left for a settle
    assert not one.get("kv").verify and not one.get("lin").verify
    assert one.get("ngram").drafts == 0 and one.get("ngram").verify and one.get("sampler").verify
    full = with_draft_verify(step(K + 1, K + 1), ADAPTIVE, "dv")
    assert full.get("dv").drafts == K and full.cg_key_info == VerifyKey(K) and full.get("lin").verify
    for spans in ((2, 2), (1, K + 1)):
        with pytest.raises(ValueError, match="span one block"):
            with_draft_verify(step(*spans), ADAPTIVE, "dv")
    # a walk that always drafts runs no step of one token, and keeps its one capture
    with pytest.raises(ValueError, match="span one block"):
        with_draft_verify(step(1), POLICY, "dv")
    assert with_draft_verify(step(K + 1), POLICY, "dv").cg_key_info is None
    keyed = SubmoduleStep(steps={"kv": KVStep()}, segments=[Segment(0, "main", 1)], cg_key_info="cfg")
    with pytest.raises(ValueError, match="capture key is the engine's"):
        with_draft_verify(keyed, ADAPTIVE, "dv")

    def row(block):
        return ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=block)

    configs = verify_captures(ADAPTIVE, "decode", row, [1, 2])
    assert [(c.single_request_inputs.input_seq_len, c.additional_key_info) for c in configs] == [
        (1, VerifyKey(0)), (K + 1, VerifyKey(K)),
    ]
    assert [c.additional_key_info for c in verify_captures(POLICY, "decode", row, [1])] == [None]
    with pytest.raises(ValueError, match="needs an engine-side proposer"):
        DraftPolicy(max_drafts=K, adaptive=True)


def test_an_adaptive_walk_needs_a_proposer_that_predicts():
    from mstar.engine.resources.draft_verify.ngram import NgramProposer

    class Sub:
        def get_draft_policy(self, walk):
            return ADAPTIVE

        def device_loopback_signals(self, walk):
            return frozenset()

    dv = DraftVerifyResource(CPU)
    sampler = SamplerResource(vocab_size=8, enable_repetion_penalty=False, device=CPU, draft_verify=True)
    prop = NgramProposer(CPU, max_drafts=K, min_ngram=2, max_ngram=3)
    resources = {"draft_verify": dv, "sampler": sampler, "ngram": prop}
    assert verify_walk("llm", Sub(), resources, "decode").policy is ADAPTIVE
    prop.predicts = False
    with pytest.raises(ValueError, match="predicts nothing"):
        verify_walk("llm", Sub(), resources, "decode")


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
    # the node's sampler, whose step with_draft_verify adds; a greedy verify's skips it
    greedy = SamplerResource(vocab_size=8, enable_repetion_penalty=False, device=CPU, draft_verify=True)
    resources = {"kv": kv, "pos": pos, "dv": dv, "sampler": greedy}
    runner = StepRunner(resources, node_resources={"llm": list(resources)})
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
        # as the engine pre-plans: a staged step, then dropped
        ctx.is_preplan = True
        runner.pre_admit(step)
        ctx.is_preplan = False
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
    # recurrent state: GDN settles its own layers in the forward; the pool is storage only, and
    # Mamba-2 declares nothing, so a node with it refuses to verify
    assert GDNManager.rewind_kind is RewindKind.DEVICE
    assert RecurrentStatePool.rewind_kind is RewindKind.NONE
    assert Mamba2Manager.rewind_kind is None


def test_the_position_rewinds_beside_the_kv_length():
    pos = RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    pos.ingest_request(1)
    pos._counters[1]["main"] = 7
    pos.rewind(VerdictBatch([1], ["main"], [3]))
    assert pos.position(1, "main") == 4
    pos.rewind(VerdictBatch([1], ["main"], [10]))
    assert pos.position(1, "main") == 0


# ── state the forward updates in place (RewindKind.DEVICE) ──


class _Settler:
    """Records what `verify` hands a resource the forward updates in place."""

    rewind_kind = RewindKind.DEVICE

    def __init__(self):
        self.calls = []

    def settle_verify(self, accepted, label):
        self.calls.append((accepted.tolist(), label))


def test_verify_settles_each_device_resource_once_by_rank_0s_counts():
    res, sampler = _resource()
    settler = _Settler()
    # bound once per verifying walk: a resource two walks bind settles once a step
    res.bind_settlers([settler])
    res.bind_settlers([settler])
    for rid in (1, 2):
        res.ingest_request(rid)
    _verify_step(res, sampler, [1, 2], accepted=[2, 0], padded=[1, 2, -3], slot=0)
    # every row the forward ran, the padding row included, under the step's label
    assert settler.calls == [([2, 0, 0], "main")]


def test_a_verify_step_over_two_labels_has_nothing_to_settle_by():
    res, _ = _resource()
    res.bind_settlers([_Settler()])
    step = VerifyStep(segments=(Segment(1, "a", K + 1), Segment(2, "b", K + 1)), drafts=K)
    with pytest.raises(ValueError, match="under one label"):
        res.plan(step, _ctx([1, 2]))


def test_the_verifying_walk_binds_the_nodes_device_resources():
    class Sub:
        def get_draft_policy(self, walk):
            return POLICY

        def device_loopback_signals(self, walk):
            return frozenset()

    dv = DraftVerifyResource(CPU)
    sampler = SamplerResource(vocab_size=8, enable_repetion_penalty=False, device=CPU, draft_verify=True)
    settler = _Settler()
    verify_walk("llm", Sub(), {"draft_verify": dv, "sampler": sampler, "state": settler}, "decode")
    assert dv._settlers == [settler]


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
    # and the sampler's step, which a sampled verify draws with
    sampler_step = out.get("sampler")
    assert isinstance(sampler_step, SamplerStep) and sampler_step.verify and not sampler_step.apply_penalty
    assert out.get("dv").sampler == "sampler"
    assert with_draft_verify(step, POLICY, "dv", sampler="s2").get("s2").verify
    short = SubmoduleStep(steps={"kv": KVStep()}, segments=[Segment(1, "main", 1)])
    with pytest.raises(ValueError, match="set input_seq_len to policy.block"):
        with_draft_verify(short, POLICY, "dv")
    own = SubmoduleStep(steps={"kv": KVStep(), "sampler": SamplerStep()}, segments=step.segments)
    with pytest.raises(ValueError, match="declare none"):
        with_draft_verify(own, POLICY, "dv")
    # a step type with a verify field is told the step verifies
    linear = SubmoduleStep(steps={"kv": KVStep(), "lin": LinearAttnStep()}, segments=step.segments)
    assert with_draft_verify(linear, POLICY, "dv").get("lin").verify


def test_a_verify_step_names_the_sampler_its_resource_verifies_with():
    res, _ = _resource()
    step = VerifyStep(segments=(Segment(1, "main", K + 1),), drafts=K, sampler="other")
    with pytest.raises(ValueError, match="pass it to with_draft_verify"):
        res.plan(step, _ctx([1]))


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
