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
from mstar.engine.resources.position.config import PositionConfig, PositionStep  # noqa: E402
from mstar.engine.resources.position.manager import RopeManager  # noqa: E402
from mstar.engine.resources.resident import ResidentPlanner  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep  # noqa: E402

PAGE = 4
K = 3
SPAN = K + 1
WALK = "decode"
CPU = torch.device("cpu")


def _resources(max_num_pages: int = 128, max_seq_len: int = 512):
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
        max_num_pages=max_num_pages, page_size=PAGE,
    )
    kv = KVManager(
        cfg=cfg,
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    rope = RopeManager(PositionConfig(kv_cache="kv"), CPU, max_seq_len=512, head_dim=8)
    rope._pinned_pos_ids_buffer = lambda key, n: torch.empty(n, dtype=torch.long)
    return {"kv": kv, "rope": rope}


def _prefill(res, rid: int, n: int) -> None:
    res["kv"].ingest_request(rid, KVReqConfig())
    res["rope"].ingest_request(rid, None)
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

    def __init__(self, prompts: dict[int, int], synchronous: bool = False, max_bs: int = 4, max_seq_len: int = 512):
        self.res = _resources(max_seq_len=max_seq_len)
        self.planner = ResidentPlanner("n", {WALK}, self.res, CPU, max_bs=max_bs, max_span=SPAN)
        self.ledger = VerifyLedger("n", synchronous=synchronous)
        self.planner.pending = self.ledger.pending
        self.planner.unrecorded = self.ledger.unrecorded
        self.true_len = dict(prompts)
        self.captured: set = set()
        for rid, n in prompts.items():
            _prefill(self.res, rid, n)

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
        positions = res["rope"]._static_pos_ids[CGSlotKey(bucket=bucket, slot=slot, label="main")]
        out = SimpleNamespace(
            pages=state.token_to_page[: bs * SPAN].tolist(),
            offsets=state.token_to_cache[: bs * SPAN].tolist(),
            positions=positions[: bs * SPAN].tolist(),
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


def _settle(node: _Node, rid: int) -> None:
    """What `Engine._settle_verify` does before an offload: the row's recorded
    results, applied through the same fan-out as the admit-time pass."""
    takes = node.ledger.settle(rid)
    for key in ("kv", "rope"):
        take = takes.get(key)
        if take is not None and take.rids:
            node.res[key].take_back(take.rids, take.counts, take.label)


@pytest.mark.parametrize("settle", [True, False])
def test_a_victim_settled_at_offload_comes_back_at_its_true_length(settle):
    """Asynchronous mode under KV pressure: admit(N+1) is refused, so nothing
    commits and step N's result never comes due. Rank 0 settles a victim whose
    results are all recorded, gives its slot up with the offload, and its next
    step's whole-row rewrite comes from the exact host length. Without the
    settle the row has unapplied results and no slot, which wedges the node."""
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
    # admit(N+1) refused; postprocess(N) lands before the eviction is decided
    node.record(held)
    assert node.ledger.pending(2) and not node.ledger.unrecorded(2)
    if settle:
        _settle(node, 2)
        assert node.stream(2).stored_len == node.true_len[2]
    node.planner.release(2)
    if not settle:
        with pytest.raises(RuntimeError, match="no slot"):
            node.step([1, 2], {1: 0, 2: 0}, slot=0, bs=2)
        return
    out = node.step([1, 2], {1: 1, 2: 2}, slot=0, bs=2)
    _check_step(node, out, [1, 2])
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
    """A follower replaying rank 0's offload may be mid-offload or mid-reload."""
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
    assert kv.committed_spans([1, 2], "main", SPAN) == [0, SPAN]
    assert kv._streams[1]["main"].stored_len == 5 and kv._streams[2]["main"].stored_len == 5 + SPAN


