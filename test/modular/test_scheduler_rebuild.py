"""Rebuild rows are the micro-scheduler's work.

A dropped request comes back through a row of its node's rebuild walk, which the graph runtime never sees. The
scheduler takes a node's queued rows ahead of anything new, as it takes a backlog, and fills the step's room with
that walk's ready rows. A speculation chain yields to them like to any other ready work, since it never merges one.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from types import SimpleNamespace

import pytest
from test_scheduler_room import NODE, _Engine, _rids, _Runtime, _setup, _State

from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.rebuild import RebuildRow

PREFILL, DECODE = "prefill", "decode"


def _queue(sched, *rids, walk=PREFILL):
    for rid in rids:
        sched.queue_rebuild(rid, RebuildRow(NODE, walk, {}))


def test_rebuild_rows_go_first_and_the_walks_ready_rows_fill_the_room_left():
    sched, runtime, state = _setup({"p0": PREFILL, "p1": PREFILL, "d0": DECODE}, max_bs=3)
    _queue(sched, "r0")

    batch = sched.get_next_batch(state)

    assert (batch.graph_walk, list(batch.rebuild_rows)) == (PREFILL, ["r0"]), "the rebuild waited behind new work"
    assert _rids(batch) == ["p0", "p1"], "the rebuild row's walk left room the ready prompts didn't get"
    assert "d0" in runtime.ready, "a decode row would run under the rebuild's prefill walk"


class _CombinedRuntime(_Runtime):
    """Prefill and decode share the combined walk ``mixed``, and a scan names either walk or both."""
    COMBINED = {(NODE, PREFILL): "mixed", (NODE, DECODE): "mixed"}

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        def named(walk):
            return target is None or target[1] in (walk, self.COMBINED[(NODE, walk)])
        return [spec for spec in super().get_ready_nodes(exclude_rids) if named(spec.graph_walk)]


def test_a_rebuild_step_under_a_combined_walk_asks_the_engine_about_none_of_the_other_walks_rows():
    asked = []

    class Engine(_Engine):
        def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
            asked.append(rid)
            return super().check_ready(node_name, rid, fwd_info, allow_reload)

    runtime, engine = _CombinedRuntime({"p0": PREFILL, "d0": DECODE}), Engine(4)
    sched = MicroScheduler(engine_manager=SimpleNamespace(get_engine=lambda name: engine),
                           parallel_leader_nodes={NODE}, combined_walk_of=_CombinedRuntime.COMBINED)
    sched.runtime = runtime
    _queue(sched, "r0")

    batch = sched.get_next_batch(_State(runtime))

    assert (list(batch.rebuild_rows), _rids(batch)) == (["r0"], ["p0"])
    assert asked == ["p0"], "a decode row's readiness check could read its request back in, for a step it can't join"


def test_a_walks_cap_bounds_its_rebuild_rows_first():
    sched, runtime, state = _setup({"p0": PREFILL}, max_bs=2)
    _queue(sched, "r0", "r1", "r2")

    batch = sched.get_next_batch(state)

    assert list(batch.rebuild_rows) == ["r0", "r1"] and not _rids(batch), "the step outgrew the walk's cap"
    assert "p0" in runtime.ready, "a prompt row would be popped with no room to run"


@pytest.mark.parametrize("why", ["held", "failed", "removing"])
def test_a_rebuild_row_the_scheduler_wouldnt_run_waits(why):
    sched, _, state = _setup({"d0": DECODE}, max_bs=4)
    _queue(sched, "r0")
    if why == "held":
        sched.hold_requests(["r0"])
    elif why == "failed":
        sched.failed_rids.add("r0")
    else:
        sched.pending_removes["r0"] = None

    batch = sched.get_next_batch(state)

    assert batch.graph_walk == DECODE and not batch.rebuild_rows, "a torn-down or held request would prefill again"
    assert sched.rebuild_queued("r0"), "the row left its queue before anything decided its request"


def test_a_row_stays_queued_until_its_step_goes_out_or_its_request_is_removed():
    sched, _, state = _setup({}, max_bs=4)
    _queue(sched, "r0", "r1")

    assert list(sched.get_next_batch(state).rebuild_rows) == ["r0", "r1"]
    assert list(sched.get_next_batch(state).rebuild_rows) == ["r0", "r1"], "a step built and never launched lost rows"
    sched.take_rebuilds({"r0"})
    sched.clear_rid("r1", "r1")
    assert sched.get_next_batch(state) is None and not sched.rebuilds, "a row would run twice, or after its removal"


@pytest.mark.parametrize("chain_walk", [DECODE, PREFILL])
def test_a_speculation_chain_yields_to_a_rebuild_row_whatever_its_walk(chain_walk):
    sched, runtime, state = _setup({}, max_bs=4)
    runtime.has_ready_excluding = lambda exclude, exclude_target: False  # nothing else is ready
    assert not sched.has_ready_excluding(state, (NODE, chain_walk))
    _queue(sched, "r0")

    assert sched.has_ready_excluding(state, (NODE, chain_walk)), "a chain would speculate past the rebuild row"
    sched.hold_requests(["r0"])
    assert not sched.has_ready_excluding(state, (NODE, chain_walk)), "a chain would yield to a row that can't run"


def test_pending_rows_count_the_rebuild_rows():
    sched, _, _ = _setup({"p0": PREFILL}, max_bs=4)
    _queue(sched, "r0", "r1")

    assert sched.pending_rows()[NODE] == 3, "the worker would report its node idle with rebuild rows waiting"
