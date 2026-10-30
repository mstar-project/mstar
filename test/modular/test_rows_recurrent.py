"""Recurrent state on persistent rows: each row addresses its request's slot,
so a request's state follows it across joins, leaves, rows resident between
its steps, steps of another walk, removals and handles that come back as new
requests. A capture addresses the sink alone.

Driven through ``NodeRows``, with ``gather`` / ``scatter_`` standing in for a
backend, so it runs on CPU rows as well as CUDA ones.
"""
import random

import pytest
import torch

from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.rows import NodeRows, RowWalk

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NODE, ROWS_WALK, OTHER_WALK, POOL, MAX_ROWS, MAX_LIVE, HANDLES = "toy", "decode", "prefill", "gdn_state", 4, 8, 16


def _build(device):
    pool = RecurrentStatePool(
        torch.device(device),
        RecurrentStateConfig(num_layers=1, blocks={"s": RecurrentBlockConfig((2,), torch.float32)}, max_slots=24,
                             labels=("main", "aux")),
        name=POOL,
    )
    node_rows = NodeRows.build(
        NODE, {ROWS_WALK: RowWalk(labels={POOL: ("main",)})}, lambda walk: MAX_ROWS,
        {POOL: pool}, torch.device(device), max_live=MAX_LIVE,
    )
    return pool, node_rows


def _step(pool, node_rows, walk, rids, spans=None):
    """One step of ``walk`` over ``rids`` that reads each row's state and
    writes it back plus one. Returns what each request read, and the slots
    its rows addressed."""
    ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
    if walk == ROWS_WALK:
        ctx.rows = node_rows.begin(walk, rids)
        rids = list(ctx.rows.order)
        ctx.request_ids = tuple(rids)
        spans = dict.fromkeys(rids, 1)
    else:
        node_rows.check_in(rids)
    step = RecurrentStep(segments=tuple(Segment(rid, "main", spans[rid]) for rid in rids))
    assert pool.admit(step, ctx).ok
    addressing = pool.plan(step, ctx)["main"]
    if ctx.rows is not None:
        # the rows themselves, not a copy staged from the slot table
        assert addressing.slot_indices.data_ptr() == pool._rows[walk].slots["main"].device.data_ptr()
        ctx.rows.stage()
        node_rows.commit(walk, ctx.rows)
        ctx.rows.apply()
        node_rows.gather(walk, len(rids))  # what the step's graph runs before its forward
    slots = addressing.slot_indices[: len(rids)].tolist()
    state = pool.gather("s")
    pool.scatter_("s", state + 1)
    if ctx.rows is not None:
        node_rows.after_forward(walk)
    pool.commit(step, ctx)
    return {rid: int(state[i, 0]) for i, rid in enumerate(rids)}, dict(zip(rids, slots, strict=True))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(3))
def test_each_request_resumes_its_own_state(device, seed):
    rng = random.Random(seed)
    pool, node_rows = _build(device)
    space = node_rows.space(ROWS_WALK)
    truth: dict[int, int] = {}
    free = list(range(HANDLES))
    for _ in range(200):
        if truth and rng.random() < 0.1:
            rid = rng.choice(list(truth))
            node_rows.forget(rid)
            pool.remove_request(rid)
            del truth[rid]
            free.insert(0 if rng.random() < 0.5 else len(free), rid)
        while free and len(truth) < 10 and rng.random() < 0.3:
            rid = free.pop(0)
            pool.ingest_request(rid)
            truth[rid] = 0
        if not truth:
            continue
        rids = rng.sample(list(truth), rng.randint(1, min(MAX_ROWS, len(truth))))
        if rng.random() < 0.7:
            read, slots = _step(pool, node_rows, ROWS_WALK, rids)
        else:
            read, slots = _step(pool, node_rows, OTHER_WALK, rids, {rid: rng.randrange(1, 4) for rid in rids})
            assert not any(space.holds(rid) for rid in rids)
        for rid in rids:
            assert slots[rid] == pool._slots[rid]["main"].index, rid
            assert read[rid] == truth[rid], rid
            truth[rid] += 1
        space.check_mirrors()


@pytest.mark.parametrize("device", DEVICES)
def test_a_capture_addresses_only_the_sink(device):
    """Captures run before any step, while the rows they record hold padding."""
    pool, _ = _build(device)
    ctx = StepContext(request_ids=("a", "b"), graph_walk=ROWS_WALK, slot=0, capture=True)
    step = RecurrentStep(segments=(Segment("a", "main", 1), Segment("b", "main", 1)))
    assert pool.admit(step, ctx).ok
    addressing = pool.plan(step, ctx)["main"]
    assert addressing.slot_indices[:2].tolist() == [SINK_SLOT, SINK_SLOT]
    assert addressing.num_rows == 2


def test_a_step_of_a_persistent_walk_takes_no_forks():
    pool, node_rows = _build("cpu")
    pool.ingest_request(0)
    ctx = StepContext(request_ids=(0,), graph_walk=ROWS_WALK, slot=0, capture=False)
    ctx.rows = node_rows.begin(ROWS_WALK, [0])
    step = RecurrentStep(segments=(Segment(0, "main", 1),), post_forks=(("main", "aux"),))
    assert pool.admit(step, ctx).ok
    with pytest.raises(ValueError, match="takes no forks"):
        pool.plan(step, ctx)


@pytest.mark.parametrize("device", DEVICES)
def test_a_held_row_addresses_a_label_it_takes_later(device):
    """A label a row's first steps don't carry: the slot its later admit takes
    reaches the row it already holds."""
    pool = RecurrentStatePool(
        torch.device(device),
        RecurrentStateConfig(num_layers=1, blocks={"s": RecurrentBlockConfig((2,), torch.float32)}, max_slots=8,
                             labels=("main", "aux")),
        name=POOL,
    )
    node_rows = NodeRows.build(
        NODE, {ROWS_WALK: RowWalk(labels={POOL: ("main", "aux")})}, lambda walk: MAX_ROWS,
        {POOL: pool}, torch.device(device),
    )
    pool.ingest_request(0)
    for labels in (("main",), ("main",), ("main", "aux"), ("main", "aux")):
        ctx = StepContext(request_ids=(0,), graph_walk=ROWS_WALK, slot=0, capture=False)
        ctx.rows = node_rows.begin(ROWS_WALK, [0])
        step = RecurrentStep(segments=tuple(Segment(0, label, 1) for label in labels))
        assert pool.admit(step, ctx).ok
        addressing = pool.plan(step, ctx)
        ctx.rows.stage()
        node_rows.commit(ROWS_WALK, ctx.rows)
        ctx.rows.apply()
        node_rows.gather(ROWS_WALK, 1)
        for label in labels:
            assert addressing[label].slot_indices[:1].tolist() == [pool._slots[0][label].index], label
        node_rows.after_forward(ROWS_WALK)
        pool.commit(step, ctx)
