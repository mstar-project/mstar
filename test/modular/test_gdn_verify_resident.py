"""Asynchronous verify steps on a GDN node: resident tables, results recorded
a step late, on the CPU stand-ins (`gdn_cpu`).

A resident verify step plans nothing on the host. Its graph, captured once
per bucket and slot from a host plan of dummy rows, gathers each row's pool
slot off the planner's table, runs the conv with its window's checkpoint copy,
the delta rule that leaves the state as it was, and the closed-gate rerun it recorded, and settles each row at what it
accepted. Here a replay is emulated eagerly, as `test_spec_resident.py` does
for the KV cache: the resources' host plans run once at capture, and each
step after runs only what the graph would (the planner's gather, the layers
against the capture's wrapper, the settle, the in-graph commit). Every row's
state and conv window must be a dense recomputation over the tokens it kept,
and the pool's host count catches up through the ledger's take-backs.
"""
from __future__ import annotations

import collections
import random
from types import SimpleNamespace

import pytest
import torch
from test_gdn_verify import (
    CPU,
    LIN,
    POOL,
    PREFILL,
    VERIFY,
    WEIGHTS,
    K,
    _add,
    _assert_rows,
    _build,
    _cat,
    _head,
    _inputs,
    _split,
    _step,
)

from mstar.engine.resources.draft.ledger import VerifyLedger
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.resident import ResidentPlanner
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep

SPAN = K + 1


@pytest.fixture(autouse=True)
def _kernels(monkeypatch):
    import gdn_cpu

    gdn_cpu.install(monkeypatch)


class _Node:
    """A verifying GDN node on resident tables, with its ledger."""

    def __init__(self, max_bs: int = 4):
        self.pool, self.gdn = _build()
        for resource in (self.pool, self.gdn):
            resource.build_accelerator_graph_buffers([], max_bs, max_bs * SPAN)
        self.planner = ResidentPlanner("n", {VERIFY}, {POOL: self.pool, LIN: self.gdn}, CPU, max_bs, max_span=SPAN)
        self.ledger = VerifyLedger("n", synchronous=False)
        self.planner.pending, self.planner.unrecorded = self.ledger.pending, self.ledger.unrecorded
        # (bucket, slot) -> what its graph holds: the captured step, and the wrapper plan it recorded
        self.graphs: dict = {}
        # rid -> the inputs of the tokens its state holds
        self.kept: dict[int, list] = {}
        self.unrecorded: collections.deque = collections.deque()
        self.seed = 100

    def prompt(self, rid: int, n: int) -> None:
        _add(self.pool, rid)
        inputs = _inputs(n, seed=self._seed())
        _step(self.pool, self.gdn, {rid: n}, PREFILL, inputs)
        self.kept[rid] = inputs

    def _seed(self) -> int:
        self.seed += 1
        return self.seed

    def step(self, rids: list[int], accepted: dict[int, int], slot: int, bs: int, lag: int = 2,
             stale_slots: bool = False) -> None:
        """One verify step, its result recorded ``lag - 1`` steps later.
        ``stale_slots``: undo the gather's slot write, every row left on the sink (a control)."""
        pool, gdn, planner = self.pool, self.gdn, self.planner
        takes = self.ledger.take_due(self.ledger.next_step, frozenset(rids))
        if (take := takes.get(POOL)) is not None and take.rids:
            pool.take_back(take.rids, take.counts, take.label)
        padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
        bucket = BucketKey(graph_walk=VERIFY, bs=bs, num_tokens=bs * SPAN)
        lease = SlotLease(slot=slot, bucket=bucket)
        segments = tuple(Segment(r, "main", SPAN) for r in padded)
        rstep, lstep = RecurrentStep(segments=segments), LinearAttnStep(segments=segments)
        step = SubmoduleStep(steps={POOL: rstep, LIN: lstep}, segments=list(segments))
        ctx = StepContext(request_ids=tuple(rids), graph_walk=VERIFY, slot=slot, capture=False, slot_lease=lease)
        ctx.set_padded_rids(padded)
        assert pool.admit(rstep, ctx).ok
        if (bucket, slot) not in self.graphs:
            # the capture: a host plan of rows holding no state, as the runner's dummy rows are
            first = StepContext(request_ids=padded, graph_walk=VERIFY, slot=slot, capture=True, slot_lease=lease)
            first.plan_results[POOL] = pool.plan(rstep, first)
            gdn.plan(lstep, first)
            captured = planner.capture_step(bucket, slot, step)
            self.graphs[(bucket, slot)] = (captured, gdn._current, gdn._verify_block)
        captured, current, block = self.graphs[(bucket, slot)]
        ctx.resident = planner
        planner.plan(step, ctx)
        pool.plan(rstep, ctx)
        gdn.plan(lstep, ctx)
        planner.flush(ctx.resident_plan)

        # the replay: the capture's kernels, against the capture's wrapper
        gdn._current, gdn._verify_block = current, block
        planner.gather(captured)
        if stale_slots:
            for addressing in pool._cg_addressing.values():
                addressing.slot_indices.zero_()
        inputs = _inputs(bs * SPAN, seed=self._seed())
        for layer, (x, a, b) in enumerate(inputs):
            gdn.set_default_layer_idx(layer)
            w = WEIGHTS[layer]
            q, k, v = _split(gdn.run_conv(x, pool.block("conv", layer), w.conv_w, w.conv_b))
            gdn.run(q, k, v, a, b, pool.block("state", layer), w.a_log, w.dt_bias)
        counts = [accepted[r] for r in rids] + [0] * (bs - len(rids))
        gdn.settle_verify_step(SimpleNamespace(accepted=torch.tensor(counts)))
        planner.commit_in_graph(captured)

        pool.commit(rstep, ctx)
        planner.commit(step, ctx)
        seq = self.ledger.committed(rids, {POOL: "main"}, [SPAN] * len(rids))
        for i, rid in enumerate(rids):
            row = [(x[i * SPAN: (i + 1) * SPAN], a[i * SPAN: (i + 1) * SPAN], b[i * SPAN: (i + 1) * SPAN])
                   for x, a, b in inputs]
            self.kept[rid] = _cat(self.kept[rid], _head(row, accepted[rid] + 1))
        self.unrecorded.append((seq, tuple(rids), [accepted[r] for r in rids]))
        while len(self.unrecorded) >= lag:
            self.record_one()

    def record_one(self) -> None:
        seq, rids, accepted = self.unrecorded.popleft()
        self.ledger.record(seq, rids, torch.tensor(accepted).numpy())

    def settle_all(self, rids: list[int]) -> None:
        """Record what's left and take it back, as the next admit would."""
        while self.unrecorded:
            self.record_one()
        takes = self.ledger.take_due(self.ledger.next_step + 1, frozenset(rids))
        if (take := takes.get(POOL)) is not None and take.rids:
            self.pool.take_back(take.rids, take.counts, take.label)

    def check(self, rids) -> None:
        _assert_rows(self.pool, {rid: self.kept[rid] for rid in rids})


@pytest.mark.parametrize("lag", [1, 2])
def test_resident_verify_steps_settle_every_row_from_the_gathered_slots(lag):
    """Random acceptance over two capture slots, a padding row, a request
    leaving with a result unrecorded and one joining in its slots (a new slot
    row written whole): each row's state is its kept tokens', and the host
    count stays between kept and reserved, catching up once every result is in."""
    rng = random.Random(lag)
    node = _Node()
    rows = [1, 2, 3]
    for rid, n in zip(rows, (3, 5, 2), strict=True):
        node.prompt(rid, n)
    for i in range(8):
        if i == 4:
            # 2 leaves holding a committed, unrecorded result; 4 takes its freed slots straight after
            node.ledger.remove_request(2)
            node.pool.remove_request(2)
            node.planner.release(2)
            rows = [1, 3, 4]
            node.prompt(4, 6)
        node.step(rows, {r: rng.randint(0, K) for r in rows}, slot=i % 2, bs=4, lag=lag)
        node.check(rows)
        for rid in rows:
            # between what the state holds and what was reserved: an asynchronous ledger's take-back
            # is due two admits after its step, so at most two steps' rejected drafts are counted
            kept = node.kept[rid][0][0].shape[0]
            assert kept <= node.pool._slots[rid]["main"].tokens <= kept + 2 * K, rid
    node.settle_all(rows)
    for rid in rows:
        assert node.pool._slots[rid]["main"].tokens == node.kept[rid][0][0].shape[0]


def test_without_the_slot_gather_a_resident_step_settles_nothing():
    """Negative control: a replay that kept the capture's slots (every row on
    the sink) would skip every row, and no request's state would move."""
    node = _Node()
    node.prompt(1, 4)
    node.step([1], {1: 1}, slot=0, bs=2, stale_slots=True)
    with pytest.raises(AssertionError):
        node.check([1])
