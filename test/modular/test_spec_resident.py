"""Speculative decoding on resident tables: a verify step writes ``k + 1``
tokens a row, the graph advances each row's KV length by ``accepted + 1``, and
the host's reserved length catches up at a later admit (the take-back).

The real KV and position resources on the CPU, driven as a replay drives them
(`test_resident_planning.py`): the device length is the truth, every write
slot and position follows from it, and a rewrite of a row with unapplied
results leaves its length cell alone.
"""
from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from test_resident_planning import _stub_transfer  # noqa: E402,F401

from mstar.engine.resources.base import CGSlotKey  # noqa: E402
from mstar.engine.resources.draft.ledger import VerifyLedger  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs  # noqa: E402
from mstar.engine.resources.position.config import (  # noqa: E402
    LongRoPE,
    PositionConfig,
    PositionReqConfig,
    PositionStep,
)
from mstar.engine.resources.position.manager import LongRopeManager, RopeManager  # noqa: E402
from mstar.engine.resources.resident import ResidentPlanner  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep  # noqa: E402

PAGE = 4
K = 3
SPAN = K + 1
WALK = "decode"
CPU = torch.device("cpu")
# a request whose ceiling passes 16 rotates on the long half, 512 rows on
LONGROPE = LongRoPE(short_factor=(1.0,) * 4, long_factor=(2.0,) * 4,
                    original_max_position_embeddings=16, max_position_embeddings=512)
LONG = PositionReqConfig(max_position=100)


def _resources(max_num_pages: int = 128, max_seq_len: int = 512, longrope: bool = False):
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
        max_num_pages=max_num_pages, page_size=PAGE,
    )
    kv = KVManager(
        cfg=cfg,
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    if longrope:
        rope = LongRopeManager(
            PositionConfig(kv_cache="kv", rotary_dim=8, longrope=LONGROPE), CPU, max_seq_len=512, head_dim=8,
        )
    else:
        rope = RopeManager(PositionConfig(kv_cache="kv"), CPU, max_seq_len=512, head_dim=8)
    rope._pinned_pos_ids_buffer = lambda key, n: torch.empty(n, dtype=torch.long)
    return {"kv": kv, "rope": rope}


def _prefill(res, rid: int, n: int, rope_config: PositionReqConfig | None = None) -> None:
    res["kv"].ingest_request(rid, KVReqConfig())
    res["rope"].ingest_request(rid, rope_config)
    segments = (Segment(rid, "main", n),)
    ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
    kv, rope = KVStep(segments=segments), PositionStep(segments=segments)
    assert res["kv"].admit(kv, ctx).ok
    ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
    res["rope"].plan(rope, ctx)
    res["kv"].commit(kv, ctx)
    res["rope"].commit(rope, ctx)


class _Node:
    """One verifying node: resources, planner, ledger, and each row's true length."""

    def __init__(self, prompts: dict[int, int], synchronous: bool = False, max_bs: int = 4, max_seq_len: int = 512,
                 long_rids: frozenset[int] | None = None):
        self.res = _resources(max_seq_len=max_seq_len, longrope=long_rids is not None)
        self.planner = ResidentPlanner("n", {WALK}, self.res, CPU, max_bs=max_bs, max_span=SPAN)
        self.ledger = VerifyLedger("n", synchronous=synchronous)
        self.planner.pending = self.ledger.pending
        self.planner.unrecorded = self.ledger.unrecorded
        self.true_len = dict(prompts)
        self.captured: set = set()
        for rid, n in prompts.items():
            _prefill(self.res, rid, n, LONG if long_rids and rid in long_rids else None)

    def stream(self, rid):
        return self.res["kv"]._streams[rid]["main"]

    def device_len(self, rid) -> int:
        tables = self.res["kv"]._resident
        return int(tables.kv_len.view[self.planner.slot_of(rid), 0])

    def take_back(self, rids) -> None:
        """The runner's pass at the head of admitting the next step."""
        takes = self.ledger.take_due(self.ledger.next_step, frozenset(rids))
        for key in ("kv", "rope"):
            take = takes.get(key)
            if take is not None and take.rids:
                self.res[key].take_back(take.rids, take.counts, take.label)

    def step(self, rids, accepted: dict[int, int], slot: int, bs: int, record_lag: int = 2, pending=None):
        """One verify step: take-back, admit, plan, the replay's gather and
        in-graph advance, commit, then record this step's result (or hold it
        back ``record_lag - 1`` steps, as asynchronous postprocess does)."""
        res, planner = self.res, self.planner
        self.take_back(rids)
        padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
        bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs * SPAN)
        lease = SlotLease(slot=slot, bucket=bucket)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease)
        ctx.set_padded_rids(padded)
        segments = tuple(Segment(r, "main", SPAN) for r in padded)
        kv = KVStep(segments=segments)
        rope = PositionStep(segments=segments)
        step = SubmoduleStep(steps={"kv": kv}, segments=list(segments))
        assert res["kv"].admit(kv, ctx).ok
        if (bucket, slot) not in self.captured:
            self.captured.add((bucket, slot))
            first = StepContext(request_ids=padded, graph_walk=WALK, slot=slot, capture=True, slot_lease=lease)
            first.plan_results["kv"] = res["kv"].plan(kv, first)
            res["rope"].plan(rope, first)
        captured = planner.capture_step(bucket, slot, step)
        ctx.resident = planner
        planner.plan(step, ctx)
        ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
        res["rope"].plan(rope, ctx)
        planner.flush(ctx.resident_plan)
        planner.gather(captured)
        state = res["kv"]._static_plan_state(slot, ("main", 0), bucket.num_tokens)
        table_ids = None
        if isinstance(res["rope"], LongRopeManager):
            # LongRoPE stages positions and table ids as the two rows of one buffer
            both = res["rope"]._static_pos_ids[CGSlotKey(bucket=bucket, slot=slot, label="main#longrope")]
            positions, table_ids = both.view(2, -1)
            table_ids = table_ids[: bs * SPAN].tolist()
        else:
            positions = res["rope"]._static_pos_ids[CGSlotKey(bucket=bucket, slot=slot, label="main")]
        out = SimpleNamespace(
            pages=state.token_to_page[: bs * SPAN].tolist(),
            offsets=state.token_to_cache[: bs * SPAN].tolist(),
            positions=positions[: bs * SPAN].tolist(),
            table_ids=table_ids,
        )
        # what the verify writes in the graph
        captured.buffers.advance.copy_(torch.tensor([accepted[r] + 1 for r in rids] + [1] * (bs - len(rids))))
        planner.commit_in_graph(captured)
        res["kv"].commit(kv, ctx)
        res["rope"].commit(rope, ctx)
        planner.commit(step, ctx)
        counted = [SPAN] * len(rids)
        out.seq = self.ledger.committed(rids, {"kv": "main", "rope": "main"}, counted)
        out.rids, out.accepted = tuple(rids), [accepted[r] for r in rids]
        return out

    def record(self, out) -> None:
        self.ledger.record(out.seq, out.rids, torch.tensor(out.accepted).numpy())


def _check_step(node: _Node, out, rids) -> None:
    """Every row writes and reads at its true length, whatever the host holds."""
    for i, rid in enumerate(rids):
        n = node.true_len[rid]
        pages = node.stream(rid).page_indices
        for j in range(SPAN):
            t = i * SPAN + j
            assert out.pages[t] == pages[(n + j) // PAGE], (rid, j)
            assert out.offsets[t] == (n + j) % PAGE, (rid, j)
            assert out.positions[t] == n + j, (rid, j)


@pytest.mark.parametrize("synchronous", [False, True])
def test_random_acceptance_keeps_device_lengths_exact_and_the_host_catches_up(synchronous):
    rng = random.Random(3)
    node = _Node({1: 5, 2: 3, 3: 10}, synchronous=synchronous)
    rids = [1, 2, 3]
    held = None
    for t in range(14):
        accepted = {rid: rng.randint(0, K) for rid in rids}
        out = node.step(rids, accepted, slot=t % 2, bs=4)
        _check_step(node, out, rids)
        for rid in rids:
            node.true_len[rid] += accepted[rid] + 1
            assert node.device_len(rid) == node.true_len[rid]
            # verified <= device <= reserved
            assert node.stream(rid).stored_len >= node.true_len[rid]
        if synchronous:
            node.record(out)
        else:
            # postprocess(N) lands after admit(N + 1)
            if held is not None:
                node.record(held)
            held = out
    if held is not None:
        node.record(held)
    # two more admits apply everything; then the host holds the true lengths
    node.take_back(rids)
    node.ledger.committed(None)
    node.take_back(rids)
    for rid in rids:
        assert node.stream(rid).stored_len == node.true_len[rid]
        assert node.res["rope"].position(rid, "main") == node.true_len[rid]
        tables = node.res["kv"]._resident
        assert tables.host_len[node.planner.slot_of(rid)] == node.true_len[rid]
        assert not node.ledger.pending(rid)


@pytest.mark.parametrize("advance", ["verified", "by_one", "by_span"])
def test_the_length_cell_advances_by_what_each_row_kept(monkeypatch, advance):
    """Negative control: with the advance by 1 or by the span in
    place of the verify's device count, the device lengths go wrong."""
    from mstar.engine.resources.kv.resident import KVResidentTables
    from mstar.engine.resources.resident import advance_column

    if advance != "verified":
        def commit(self, step):
            by = None if advance == "by_one" else torch.full_like(step.buffers.advance, SPAN)
            advance_column(self.planner.store, step.slots, self.kv_len.stride, self.kv_len.col, by)

        monkeypatch.setattr(KVResidentTables, "commit", commit)
    node = _Node({1: 5})
    out = node.step([1], {1: 2}, slot=0, bs=1)
    node.record(out)
    node.true_len[1] += 3
    assert (node.device_len(1) == node.true_len[1]) is (advance == "verified")


def test_a_rewrite_of_a_pending_row_keeps_its_device_length():
    node = _Node({1: 6, 2: 2})
    node.step([1, 2], {1: 1, 2: 3}, slot=0, bs=2)
    node.true_len = {1: 8, 2: 6}
    # something marks row 1 stale while its result is unapplied (a dropped pre-plan, say)
    node.planner.mark_stale(1)
    out = node.step([1, 2], {1: 0, 2: 0}, slot=1, bs=2)
    _check_step(node, out, [1, 2])
    assert node.device_len(1) == 9
    stream = node.stream(1)
    slot = node.planner.slot_of(1)
    tables = node.res["kv"]._resident
    cols = int(tables.host_cols[slot])
    # the columns were rewritten from the host, the length cell was not
    assert tables.block_table.view[slot, :cols].tolist() == stream.page_indices[:cols]


def test_an_unrecorded_row_is_never_displaced():
    node = _Node({rid: 4 for rid in range(1, 7)}, max_bs=2)
    for step, rids in enumerate(([1, 2], [3, 4])):
        node.record(node.step(rids, dict.fromkeys(rids, 0), slot=step, bs=2))
    # four request slots, all held; only the device knows rows 1 and 2's lengths
    node.planner.unrecorded = lambda rid: rid in (1, 2)
    slots = {rid: node.planner.slot_of(rid) for rid in (1, 2)}
    node.ledger.committed(None)
    node.record(node.step([5], {5: 0}, slot=0, bs=1))
    assert {rid: node.planner.slot_of(rid) for rid in (1, 2)} == slots
    assert node.planner.slot_of(3) is None or node.planner.slot_of(4) is None
    # nothing displaceable left: a loud failure, never an unrecorded row's slot
    node.planner.unrecorded = lambda rid: rid != 6
    with pytest.raises(RuntimeError, match="found no slot"):
        node.step([6], {6: 0}, slot=1, bs=1)


def test_the_take_back_moves_lengths_never_pages_and_refuses_too_much():
    node = _Node({1: 5})
    out = node.step([1], {1: 0}, slot=0, bs=1)
    node.record(out)
    stream = node.stream(1)
    pages = list(stream.page_indices)
    kv = node.res["kv"]
    kv.take_back([1], [SPAN - 1], "main")
    assert stream.stored_len == 6
    assert list(stream.page_indices) == pages
    with pytest.raises(RuntimeError, match="take-back of"):
        kv.take_back([1], [100], "main")
    with pytest.raises(RuntimeError, match="no stream"):
        kv.take_back([9], [1], "main")


def _offload(node: _Node, rid: int, marked: bool) -> None:
    """What rank 0's `Engine.offload_request` does to a verifying row once the offload
    frees pages, through the engine's own calls: the row gives up its slot
    (`Engine._release_resident_slot`), and (``marked``) its results come due at its first
    admit back (`Engine._mark_offloaded`). Nothing is applied at the offload."""
    from mstar.engine.engine import Engine

    eng = Engine.__new__(Engine)
    eng._submodules = {"n": SimpleNamespace(resident=node.planner, resources={})}
    eng._resource_planners = {}
    eng._drafts = {"n": SimpleNamespace(ledger=node.ledger)} if marked else {}
    eng._release_resident_slot("n", rid)
    eng._mark_offloaded("n", rid)


@pytest.mark.parametrize("between", [False, True], ids=["next_commit", "after_another"])
@pytest.mark.parametrize("marked", [True, False], ids=["marked", "unmarked"])
def test_a_victim_comes_back_at_its_true_length(marked, between):
    """Asynchronous mode under KV pressure: admit(N+1) is refused, so nothing commits,
    and postprocess(N) lands before the eviction is decided. The victim's results are all
    recorded, and it gives its slot up with the offload. Back for the node's very next
    commit, the two-step rule leaves step N's result pending on a row with no slot, which
    wedges the node, unless the offload marked it. After another commit of the node in
    between (a prefill, say), both of its results are due either way."""
    node = _Node({1: 5, 2: 3})
    rng = random.Random(7)
    held = None
    for t in range(4):
        accepted = {rid: rng.randint(0, K) for rid in (1, 2)}
        out = node.step([1, 2], accepted, slot=t % 2, bs=2)
        for rid in (1, 2):
            node.true_len[rid] += accepted[rid] + 1
        if held is not None:
            node.record(held)
        held = out
    node.record(held)
    assert node.ledger.pending(2) and not node.ledger.unrecorded(2)
    _offload(node, 2, marked)
    # the host still counts the rejected drafts: nothing was applied at the offload
    assert node.stream(2).stored_len > node.true_len[2]
    if between:
        node.ledger.committed(None)
    if not marked and not between:
        with pytest.raises(RuntimeError, match="no slot"):
            node.step([1, 2], {1: 0, 2: 0}, slot=0, bs=2)
        return
    out = node.step([1, 2], {1: 1, 2: 2}, slot=0, bs=2)
    _check_step(node, out, [1, 2])
    assert not node.ledger._returning
    node.true_len[1] += 2
    node.true_len[2] += 3
    assert node.device_len(2) == node.true_len[2]
    assert node.device_len(1) == node.true_len[1]


@pytest.mark.parametrize("synchronous", [False, True])
def test_a_row_runs_to_the_client_limit_inside_the_slack(synchronous):
    """The pool holds the serving limit plus 2(k + 1). A row decoding to
    exactly the limit, every draft accepted, its last block past it and a step's
    result still unapplied, never outgrows the block table."""
    limit, slack = 40, 2 * SPAN
    node = _Node({1: 5}, synchronous=synchronous, max_seq_len=limit + slack)
    held = None
    t = 0
    while node.true_len[1] < limit:
        out = node.step([1], {1: K}, slot=t % 2, bs=1)
        _check_step(node, out, [1])
        node.true_len[1] += SPAN
        if synchronous:
            node.record(out)
        else:
            if held is not None:
                node.record(held)
            held = out
        t += 1
        assert node.stream(1).stored_len <= limit + slack
    assert node.device_len(1) == node.true_len[1]


def test_a_take_back_lowers_every_copy_that_holds_the_stream():
    """Every copy that holds the stream is lowered: both while an offload claim is open, and the host's alone,
    which no admit reaches today."""
    from mstar.engine.resources.kv.cache import GroupPages

    node = _Node({1: 5})
    out = node.step([1], {1: 0}, slot=0, bs=1)
    node.record(out)
    kv, stream = node.res["kv"], node.stream(1)
    host = SimpleNamespace(groups=(GroupPages(stored_len=stream.stored_len),))
    kv._cpu_pool = SimpleNamespace(offloaded={1: {"main": host}})
    stream.offloaded = True
    # pages still on the device (mid-offload or mid-reload): both copies drop
    before = stream.stored_len
    kv.take_back([1], [1], "main")
    assert stream.stored_len == before - 1 and host.groups[0].stored_len == before - 1
    # the device copy gone: only the host's drops
    stream.page_indices.assign([])
    stream.stored_len = 0
    kv.take_back([1], [1], "main")
    assert stream.stored_len == 0 and host.groups[0].stored_len == before - 2
def test_a_span_a_commit_drops_counts_as_nothing_committed_for_that_row():
    """A stream offloaded mid-step: its host copy predates the span, so commit drops it, and the row's take-back
    has nothing to take."""
    res = _resources()
    _prefill(res, 1, 5)
    _prefill(res, 2, 5)
    kv = res["kv"]
    step = KVStep(segments=(Segment(1, "main", SPAN), Segment(2, "main", SPAN)))
    ctx = StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    # what a claim leaves on a stream while its copy runs with the lock down
    kv._streams[1]["main"].offloaded = True
    kv.commit(step, ctx)
    assert step.dropped == {(1, "main")}
    assert kv.committed_spans(step, [1, 2], "main", SPAN) == [0, SPAN]
    assert kv._streams[1]["main"].stored_len == 5 and kv._streams[2]["main"].stored_len == 5 + SPAN
    # the drop is that step's: the next step's commit, on a fresh step, counts every row
    kv._streams[1]["main"].offloaded = False
    fresh = KVStep(segments=(Segment(1, "main", SPAN), Segment(2, "main", SPAN)))
    assert kv.admit(fresh, ctx).ok
    kv.plan(fresh, ctx)
    kv.commit(fresh, ctx)
    assert fresh.dropped == frozenset()
    assert kv.committed_spans(fresh, [1, 2], "main", SPAN) == [SPAN, SPAN]
    assert kv.committed_spans(step, [1, 2], "main", SPAN) == [0, SPAN]




def _step_ctx(rows):
    ctx = StepContext(request_ids=tuple(dict.fromkeys(r for r, _, _ in rows)), graph_walk=WALK, slot=0, capture=False)
    ctx.plan_results = {"kv": KVPlanOutputs({"main": KVPlanOutput(None, rows=rows)})}
    return ctx


def _plan_rows(rope, rows):
    rope.plan(PositionStep(segments=()), _step_ctx(rows))
    return rope.pos_ids("main").tolist(), rope._current_table_ids["main"].tolist()


def test_a_longrope_verify_step_stages_each_row_on_its_requests_half():
    """A verify row's table ids run over its span on the half its request was opened
    on, and after a take-back the next decode row resumes there, as plain decode's."""
    rope = _resources(longrope=True)["rope"]
    rope.ingest_request(1, None)
    rope.ingest_request(2, LONG)
    rope._counters[1]["main"], rope._counters[2]["main"] = 5, 7

    positions, table_ids = _plan_rows(rope, [(1, "main", SPAN), (2, "main", SPAN)])
    assert positions == [5, 6, 7, 8, 7, 8, 9, 10]
    assert table_ids == [5, 6, 7, 8, 519, 520, 521, 522]

    # the step commits its span; row 1 kept 2 of its 4 tokens, row 2 all of them
    rows = [(1, "main", SPAN), (2, "main", SPAN)]
    rope.commit(PositionStep(segments=tuple(Segment(*row) for row in rows)), _step_ctx(rows))
    rope.take_back([1, 2], [2, 0], "main")
    assert _plan_rows(rope, [(1, "main", 1), (2, "main", 1)]) == ([7, 11], [7, 523])


@pytest.mark.parametrize("synchronous", [False, True])
def test_a_longrope_row_on_resident_tables_gathers_its_requests_half_over_the_span(synchronous):
    """In the graph, a verify row's table ids are gathered over its span beside its
    positions, each on its request's half, at its true length, step after step."""
    rng = random.Random(5)
    node = _Node({1: 5, 2: 3, 3: 10}, synchronous=synchronous, long_rids=frozenset({2}))
    rids = [1, 2, 3]
    held = None
    for t in range(10):
        accepted = {rid: rng.randint(0, K) for rid in rids}
        out = node.step(rids, accepted, slot=t % 2, bs=4)
        _check_step(node, out, rids)
        for i, rid in enumerate(rids):
            half = 512 if rid == 2 else 0
            row = slice(i * SPAN, (i + 1) * SPAN)
            assert out.table_ids[row] == [p + half for p in out.positions[row]], (t, rid)
            node.true_len[rid] += accepted[rid] + 1
        if synchronous:
            node.record(out)
        else:
            if held is not None:
                node.record(held)
            held = out


def test_flashinfers_resident_gather_fills_a_real_fa3_paged_wrapper_off_the_tables():
    """A verify bucket whose wrapper is FA3's (`FA3PagedWrapper`): its page table and lengths
    come off the tables in the graph, through main's in-graph branch, so the step reads the
    tables and keeps no host plan. The table is sized to the pool's block-table width, so the
    gather's slice of a row is exactly its block table."""
    from mstar.engine.resources.attn.fa3_paged import FA3PagedWrapper
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    bs, hide_last = 2, 1
    node = _Node({1: 5, 2: 9}, max_bs=bs)
    res, planner = node.res, node.planner
    wrapper = FA3PagedWrapper(
        page_size=PAGE, table_width=res["kv"].config.block_table_width(PAGE), device=CPU, span=SPAN,
        kv_extent=SPAN - hide_last, causal=False, batch_size=bs, accelerator_graph=True,
    )
    assert wrapper.plans_in_graph and not wrapper.supports_gather_resident
    bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs * SPAN)
    manager = FlashInferManager.__new__(FlashInferManager)
    manager._kv_cache_name = "kv"
    manager._cg_plan_states = {CGSlotKey(bucket=bucket, slot=0, label="main"): wrapper}
    # later in the stack the manager also looks up a lagged binding first; none here
    manager._lagged_cg = {}

    # the verify step `_Node.step` runs, up to the replay's gather
    rids = (1, 2)
    lease = SlotLease(slot=0, bucket=bucket)
    segments = tuple(Segment(r, "main", SPAN) for r in rids)
    kv, rope = KVStep(segments=segments), PositionStep(segments=segments)
    step = SubmoduleStep(steps={"kv": kv}, segments=list(segments))
    ctx = StepContext(request_ids=rids, graph_walk=WALK, slot=0, capture=False, slot_lease=lease)
    ctx.set_padded_rids(rids)
    assert res["kv"].admit(kv, ctx).ok
    first = StepContext(request_ids=rids, graph_walk=WALK, slot=0, capture=True, slot_lease=lease)
    first.plan_results["kv"] = res["kv"].plan(kv, first)
    res["rope"].plan(rope, first)
    captured = planner.capture_step(bucket, 0, step)
    ctx.resident = planner
    planner.plan(step, ctx)
    ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
    res["rope"].plan(rope, ctx)
    planner.flush(ctx.resident_plan)
    planner.gather(captured)

    manager.resident_gather(captured)

    assert captured.reads_tables and not captured.host_planned
    assert "kv" not in captured.lengths_only
    table = captured.block_tables["kv"]
    assert wrapper.width == table.width
    columns = captured.store.view(-1, table.stride)[:, table.col: table.col + table.width]
    assert wrapper._page_table.tolist() == columns.index_select(0, captured.slots).tolist()
    kv_len = captured.buffers.kv_len["kv"]
    assert wrapper._seqused.tolist() == (kv_len + SPAN - hide_last).clamp(min=1).tolist()
