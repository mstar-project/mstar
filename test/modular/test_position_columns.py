"""A step's positions and counter moves through the extension's one call equal
the Python loop's, for every shape of step."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position import manager as position_mod
from mstar.engine.resources.position.config import PositionConfig, PositionStep, PosScheme
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import BucketKey, Segments, SlotLease, StepContext


def _rope(scheme=PosScheme.SEQUENTIAL) -> RopeManager:
    rope = RopeManager(
        config=PositionConfig(kv_cache="kv", scheme=scheme, block_step=3),
        device=torch.device("cpu"), max_seq_len=64, head_dim=8,
    )
    rope._pinned_pos_ids_buffer = lambda key, n: torch.empty(n, dtype=torch.long)
    return rope


def _segments(rng) -> Segments:
    rows = rng.randrange(1, 9)
    labels = "main" if rng.random() < 0.5 else [rng.choice(("main", "aux")) for _ in range(rows)]
    spans = 1 if rng.random() < 0.5 else [rng.randrange(0, 6) for _ in range(rows)]
    # a row with counters, a row with none, a row with its label missing
    return Segments([rng.randrange(-3, 6) for _ in range(rows)], labels, spans)


def _plan_results(rows):
    return {"kv": KVPlanOutputs({"main": _Plan(rows)})}


class _Plan:
    def __init__(self, rows):
        self.rows = rows


needs_extension = pytest.mark.skipif(position_mod._rust_position_ids_into is None, reason="needs mstar_rust")


def _counters(rng):
    return {
        rid: {label: rng.randrange(0, 40) for label in ("main", "aux") if rng.random() < 0.7}
        for rid in range(-3, 6) if rng.random() < 0.7
    }


def _python_ids(rope, rows, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(position_mod, "_rust_position_ids_into", None)
        out = np.full(sum(rows.span_column()), -1, np.int64)
        assert rope._fill_pos_ids(rows, out) == len(out)
    return out.tolist()


def _reference_ids(rope, rows, block):
    """The positions by definition."""
    out = []
    for rid, label, span in rows.rows():
        start = rope._counters.get(rid, {}).get(label, 0)
        out.extend([start] * span if block else range(start, start + span))
    return out


@needs_extension
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("scheme", [PosScheme.SEQUENTIAL, PosScheme.BLOCK])
def test_the_positions_of_a_step(seed, scheme, monkeypatch):
    rng = random.Random(seed)
    rope = _rope(scheme)
    rope._counters = _counters(rng)
    rows = _segments(rng)
    want = _reference_ids(rope, rows, scheme == PosScheme.BLOCK)
    out = np.full(len(want) + 2, -1, np.int64)
    assert rope._fill_pos_ids(rows, out) == len(want)
    assert out.tolist() == want + [-1, -1]
    assert _python_ids(rope, rows, monkeypatch) == want


@needs_extension
def test_positions_that_do_not_fit_write_nothing():
    rope = _rope()
    rope._counters = {1: {"main": 5}}
    out = np.full(2, -1, np.int64)
    assert rope._fill_pos_ids(Segments([1], "main", 3), out) == 3
    assert out.tolist() == [-1, -1]


def _leased_plan(rope, rows, tokens=8, slot=0):
    ctx = StepContext(
        request_ids=tuple(rows.request_ids), graph_walk="w", slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=BucketKey("w", len(rows), tokens)),
    )
    ctx.plan_results = _plan_results(rows)
    return rope.plan(PositionStep(segments=rows), ctx)["main"]


@pytest.mark.parametrize("rust", [True, False])
def test_a_leased_step_fills_the_slots_buffer_and_keeps_its_tail(rust, monkeypatch):
    if rust and position_mod._rust_position_ids_into is None:
        pytest.skip("needs mstar_rust")
    if not rust:
        monkeypatch.setattr(position_mod, "_rust_position_ids_into", None)
    rope = _rope()
    rope._counters = {1: {"main": 5}, 2: {"main": 9}}
    assert _leased_plan(rope, Segments([1, 2], "main", 4), tokens=8).tolist() == [5, 6, 7, 8, 9, 10, 11, 12]
    # a shorter step on the same bucket's slot: the tail stays
    got = _leased_plan(rope, Segments([1, 2], "main", [1, 1]), tokens=8)
    assert got.tolist() == [5, 9, 7, 8, 9, 10, 11, 12]


def test_a_step_past_its_bucket_is_refused():
    rope = _rope()
    rope._counters = {1: {"main": 0}}
    with pytest.raises(AssertionError, match="captured bucket holds 2"):
        _leased_plan(rope, Segments([1], "main", 3), tokens=2)


@pytest.mark.parametrize("rust", [True, False])
def test_an_eager_step_takes_a_new_tensor(rust, monkeypatch):
    if rust and position_mod._rust_position_ids_into is None:
        pytest.skip("needs mstar_rust")
    if not rust:
        monkeypatch.setattr(position_mod, "_rust_position_ids_into", None)
    rope = _rope()
    rope._counters = {1: {"main": 5}}
    rows = Segments([1, 2], "main", 2)
    ctx = StepContext(request_ids=(1, 2), graph_walk="w", slot=0, capture=False)
    ctx.plan_results = _plan_results(rows)
    got = rope.plan(PositionStep(segments=rows), ctx)["main"]
    assert got.dtype == torch.long and got.tolist() == [5, 6, 0, 1]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("scheme", [PosScheme.SEQUENTIAL, PosScheme.BLOCK])
@pytest.mark.parametrize("explicit", [False, True])
def test_a_commit_moves_each_counter(seed, scheme, explicit, monkeypatch):
    if position_mod._rust_advance_positions is None:
        pytest.skip("needs mstar_rust")
    rng = random.Random(seed)
    rows = _segments(rng)
    counters = _counters(rng)
    advance = tuple(rng.randrange(-2, 4) for _ in rows) if explicit else None
    step = PositionStep(segments=rows, advance=advance)
    ctx = StepContext(request_ids=tuple(rows.request_ids), graph_walk="w", slot=0, capture=False)

    def committed(extension: bool):
        with monkeypatch.context() as m:
            if not extension:
                m.setattr(position_mod, "_rust_advance_positions", None)
            rope = _rope(scheme)
            rope._counters = {rid: dict(streams) for rid, streams in counters.items()}
            rope.commit(step, ctx)
        return rope._counters

    assert committed(True) == committed(False)
