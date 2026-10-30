"""Positions on persistent rows: each request's position on every label
follows it across joins, leaves, steps of another walk, removals and handles
that come back as new requests. A step reads its rows as its position ids, and
host queries read the row while the request holds one.

Driven through ``NodeRows`` with the KV plan a step would hand the positions
built by hand, so it runs on CPU rows as well as CUDA ones.
"""
import random

import pytest
import torch

from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.position.config import PositionConfig, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.rows import NodeRows, RowWalk

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NODE, ROWS_WALK, OTHER_WALK, KV, MAX_ROWS, HANDLES = "toy", "decode", "prefill", "kv", 6, 16
LABELS = ("main", "aux")


def _build(device, labels=LABELS):
    pos = RopeManager(PositionConfig(kv_cache=KV), device=torch.device(device))
    node_rows = NodeRows.build(
        NODE, {ROWS_WALK: RowWalk(labels={KV: labels})}, lambda walk: MAX_ROWS,
        {"pos": pos}, torch.device(device),
    )
    return pos, node_rows


def _step(pos, node_rows, walk, rids, spans=None):
    """One step of ``walk`` over ``rids``, each label of a request extended by
    ``spans[rid, label]`` (one a row on the rows walk). Returns each request's
    first position read, by label."""
    ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
    if walk == ROWS_WALK:
        ctx.rows = node_rows.begin(walk, rids)
        rids = list(ctx.rows.order)
        ctx.request_ids = tuple(rids)
        spans = {(rid, label): 1 for rid in rids for label in LABELS}
    else:
        node_rows.check_in(rids)
    ctx.plan_results[KV] = KVPlanOutputs({
        label: KVPlanOutput(
            None, rows=[(rid, label, spans[rid, label]) for rid in rids], decode=walk == ROWS_WALK,
        )
        for label in LABELS
    })
    step = PositionStep(segments=tuple(
        Segment(rid, label, spans[rid, label]) for label in LABELS for rid in rids
    ))
    ids = pos.plan(step, ctx)
    if ctx.rows is not None:
        ctx.rows.stage()
        node_rows.commit(walk, ctx.rows)
        ctx.rows.apply()
        node_rows.gather(walk, len(rids))  # what the step's graph runs before its forward
    read = {rid: {} for rid in rids}
    for label in LABELS:
        flat = ids[label].tolist()
        at = 0
        for rid in rids:
            read[rid][label] = flat[at]
            at += spans[rid, label] or 1
    if ctx.rows is not None:
        node_rows.after_forward(walk)  # and after it
    pos.commit(step, ctx)
    return read


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(3))
def test_each_request_reads_and_reports_its_own_positions(device, seed):
    rng = random.Random(seed)
    pos, node_rows = _build(device)
    space = node_rows.space(ROWS_WALK)
    truth: dict[int, dict[str, int]] = {}
    free = list(range(HANDLES))
    for _ in range(150):
        if truth and rng.random() < 0.1:
            rid = rng.choice(list(truth))
            node_rows.forget(rid)
            pos.remove_request(rid)
            del truth[rid]
            free.insert(0 if rng.random() < 0.5 else len(free), rid)
        while free and len(truth) < 10 and rng.random() < 0.3:
            rid = free.pop(0)
            pos.ingest_request(rid)
            truth[rid] = dict.fromkeys(LABELS, 0)
        if not truth:
            continue
        rids = rng.sample(list(truth), rng.randint(1, min(MAX_ROWS, len(truth))))
        if rng.random() < 0.7:
            read = _step(pos, node_rows, ROWS_WALK, rids)
            spans = {(rid, label): 1 for rid in rids for label in LABELS}
        else:
            spans = {(rid, label): rng.randrange(1, 4) for rid in rids for label in LABELS}
            read = _step(pos, node_rows, OTHER_WALK, rids, spans)
            assert not any(space.holds(rid) for rid in rids)
        for rid in rids:
            assert read[rid] == truth[rid], rid
            for label in LABELS:
                truth[rid][label] += spans[rid, label]
        for rid, labels in truth.items():
            for label, at in labels.items():
                assert pos.position(rid, label) == at, (rid, label)
            snapshot = pos.publish_snapshot_for_step(rid, NODE, OTHER_WALK)
            if snapshot is None:  # nothing to publish before a request's first step
                assert not any(labels.values()), rid
            else:
                assert dict(snapshot) == labels, rid
        space.check_mirrors()


def test_a_check_in_does_not_land_on_a_recycled_handle():
    """A step that checks a row in is proposed, the row's request is removed,
    and its handle comes back as a new request before the step commits: the
    new request starts at zero."""
    pos, node_rows = _build("cpu")
    for rid in (0, 1):
        pos.ingest_request(rid)
    for _ in range(3):
        _step(pos, node_rows, ROWS_WALK, [0, 1])
    ctx = StepContext(request_ids=(0,), graph_walk=ROWS_WALK, slot=0, capture=False)
    ctx.rows = node_rows.begin(ROWS_WALK, [0])  # 1 checks in
    ctx.plan_results[KV] = KVPlanOutputs({
        label: KVPlanOutput(None, rows=[(0, label, 1)], decode=True) for label in LABELS
    })
    step = PositionStep(segments=tuple(Segment(0, label, 1) for label in LABELS))
    pos.plan(step, ctx)
    node_rows.forget(1)
    pos.remove_request(1)
    pos.ingest_request(1)
    assert pos.position(1, "main") == 0  # the old row, still to leave, isn't the new request's
    ctx.rows.stage()
    node_rows.commit(ROWS_WALK, ctx.rows)
    ctx.rows.apply()
    node_rows.gather(ROWS_WALK, 1)
    node_rows.after_forward(ROWS_WALK)
    pos.commit(step, ctx)
    assert pos.position(1, "main") == 0
    assert _step(pos, node_rows, ROWS_WALK, [0, 1]) == {0: dict.fromkeys(LABELS, 4), 1: dict.fromkeys(LABELS, 0)}


def test_a_walk_must_name_its_labels():
    with pytest.raises(ValueError, match="RowWalk.labels"):
        _build("cpu", labels=())


def test_a_rows_step_computes_one_token_a_row():
    pos, node_rows = _build("cpu")
    pos.ingest_request(0)
    ctx = StepContext(request_ids=(0,), graph_walk=ROWS_WALK, slot=0, capture=False)
    ctx.rows = node_rows.begin(ROWS_WALK, [0])
    ctx.plan_results[KV] = KVPlanOutputs({
        label: KVPlanOutput(None, rows=[(0, label, 2)], decode=False) for label in LABELS
    })
    with pytest.raises(ValueError, match="one token a row"):
        pos.plan(PositionStep(segments=tuple(Segment(0, label, 2) for label in LABELS)), ctx)
