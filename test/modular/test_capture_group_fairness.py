"""Two capture groups on one node must not starve the worker's other nodes.

A node that splits batches by capture key (Chatterbox's T3 decode, guidance on
and off) runs one group per step and parks the other in the backlog. The
backlog is served before the round-robin pick, so with both groups decoding
there was always a T3 backlog to serve first: s3gen, the voice encoder and a
new request's prefill on the same worker waited until one group finished.

The fairness yield did fire, but its batch came from the excluded node's own
backlog, so it went straight back to T3; and on the serial path every off-group
row parked again behind the step that drained the last one.

Skipping that backlog on every yield is not right either: the other nodes then
get a turn per group instead of per round of groups, twice the s3gen steps per
decoded token with half the chunks each, which costs audio throughput.
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from test_micro_scheduler import (
    NODE,
    WALK,
    _batch,
    _Engine,
    _Manager,
    _Runtime,
    _scheduler,
)

from mstar.worker.batch_builder import BatchBuildRequest, FIFOBatchBuilder
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Speculation, Worker, _SpecFairness

OTHER = "OTHER"
SAME = {0: False, 1: False}
MIXED = {0: False, 1: True}


def _scene(groups):
    """Two decode rows (0, 1) of NODE and one ready row (2) of OTHER."""
    manager = _Manager([0, 1])
    manager.per_request_info[2] = object()
    manager.queues["wg0"]._ready[2] = {OTHER}
    sched = _scheduler(_Engine(max_bs=32, groups=groups))
    sched.parallel_leader_nodes.add(OTHER)
    return manager, sched


# -- the two lines -------------------------------------------------------------


def test_an_excluded_backlog_runs_once_then_leaves_room_for_another_ready_node():
    manager, sched = _scene(MIXED)
    sched.runtime = manager.runtime
    manager.queues["wg0"]._ready.pop(1)
    manager.queues["wg0"]._ready.pop(0)

    sched.backlog[(NODE, WALK)] = _batch([1])
    first = sched.get_next_batch(manager, exclude_target=(NODE, WALK))
    assert first.node_name == NODE and list(first.request_to_worker_graph) == [1], (
        "the other group's turn in this round was skipped"
    )

    sched.backlog[(NODE, WALK)] = _batch([0])
    second = sched.get_next_batch(manager, exclude_target=(NODE, WALK))
    assert second is not None and second.node_name == OTHER, (
        "the yield-away pass was served the excluded node's own backlog again"
    )

    # OTHER had its step, so the next round starts with the backlog again
    third = sched.get_next_batch(manager, exclude_target=(NODE, WALK))
    assert third.node_name == NODE and list(third.request_to_worker_graph) == [0]


def test_an_excluded_backlog_still_runs_once_nothing_else_is_ready():
    manager, sched = _scene(MIXED)
    sched.runtime = manager.runtime
    manager.queues["wg0"]._ready.clear()
    sched.backlog[(NODE, WALK)] = _batch([1])
    assert sched.get_next_batch(manager, exclude_target=(NODE, WALK)).node_name == NODE

    sched.backlog[(NODE, WALK)] = _batch([0])
    assert sched.get_next_batch(manager, exclude_target=(NODE, WALK)) is None
    # the main loop's unexcluded fallback pass
    batch = sched.get_next_batch(manager)
    assert batch.node_name == NODE and list(batch.request_to_worker_graph) == [0]


def test_a_fresh_off_group_row_goes_back_on_a_backlog_step():
    groups = {0: False, 1: True}
    result = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        node_name=NODE, graph_walk=WALK,
        backlog=_batch([1]), fresh=_batch([0]),
        max_batch_size=32, capture_group=groups.get,
    ))

    assert list(result.scheduled.request_to_worker_graph) == [1]
    assert result.returned == {0: "wg0"}, (
        "the other group's fresh row parked, handing the key a backlog again"
    )
    assert result.backlog is None


def test_a_fresh_off_group_row_parks_when_no_backlogged_row_ran():
    # Row 2 is backlogged but blocked (OOM backoff, pending remove, not
    # ready), so only fresh rows run. Returned, row 1 would lose the group to
    # row 0 on every step for as long as row 2 stays blocked.
    groups = {0: False, 1: True, 2: False}
    result = FIFOBatchBuilder().build_batch(BatchBuildRequest(
        node_name=NODE, graph_walk=WALK,
        backlog=_batch([2]), fresh=_batch([0, 1]), blocked_rids={2},
        max_batch_size=32, capture_group=groups.get,
    ))

    assert list(result.scheduled.request_to_worker_graph) == [0]
    assert not result.returned, "the other group's row went back behind row 0 again"
    assert sorted(result.backlog.request_to_worker_graph) == [1, 2]


def test_serial_path_alternates_groups_beside_a_blocked_backlogged_row():
    manager = _Manager([0, 1])
    manager.per_request_info[2] = object()
    sched = _scheduler(_Engine(max_bs=32, groups=MIXED, not_ready={2}))
    sched.runtime = manager.runtime
    sched.backlog[(NODE, WALK)] = _batch([2])
    ready = manager.queues["wg0"]._ready
    order = []
    for _ in range(8):
        batch = sched.get_next_batch(manager)
        order.append(sorted(batch.request_to_worker_graph))
        for rid in batch.request_to_worker_graph:
            manager.in_flight.discard((rid, NODE))
            manager.ingest(rid)
        # the runtime scans its queues in request order, not by arrival
        rows = sorted(ready.items())
        ready.clear()
        ready.update(rows)
    assert [1] in order[:2] and order.count([0]) == order.count([1]), (
        f"a blocked backlogged row starved one group: {order}"
    )


# -- serial path: the scheduler alone ------------------------------------------


def _serial(groups, steps=40):
    """Node names the scheduler hands out, completing each NODE step at once."""
    manager, sched = _scene(groups)
    sched.runtime = manager.runtime
    order = []
    for _ in range(steps):
        batch = sched.get_next_batch(manager)
        if batch is None:
            break
        order.append((batch.node_name, sorted(batch.request_to_worker_graph)))
        if batch.node_name == NODE:
            for rid in batch.request_to_worker_graph:
                manager.in_flight.discard((rid, NODE))
                manager.ingest(rid)
    return order


def test_serial_path_serves_the_other_node_beside_two_groups():
    order = _serial(MIXED)
    first = next((i for i, (node, _) in enumerate(order) if node == OTHER), None)
    assert first is not None and first <= 3, (
        f"OTHER was first served at step {first} of {len(order)}: {order[:6]}"
    )
    assert {tuple(rids) for node, rids in order if node == NODE} == {(0,), (1,)}, (
        "the two groups shared a step"
    )


def test_serial_path_with_one_group_is_unchanged():
    assert _serial(SAME)[:3] == [(NODE, [0, 1]), (OTHER, [2]), (NODE, [0, 1])]


# -- default async path: the worker's speculation and fairness -----------------


class _PastBacklogGateError(Exception):
    pass


class _AsyncRuntime(_Runtime):
    def is_async_schedulable(self, node, walk):
        return node == NODE

    def speculate_node(self, node, walk, rid):
        return [SimpleNamespace(node_name=node, is_new_loop_iter=True, loop_name="decode")]

    def prep_spec_rids(self, inp):
        raise _PastBacklogGateError(list(inp.rids))

    def get_rid_string(self, rid):
        return str(rid)


class _Loop:
    """The worker's scheduling decisions, real down to the speculation gate.

    ``_build_speculation`` (fairness peek, consecutive cap, yield-away) and
    ``_try_speculate_next`` run as written up to their ``backlog_splits_from``
    gate. Past it, prep and tensor gathering are replaced by what a same-node
    decode chain does with them: continue its rids, and merge fresh ones of
    its own capture group through the same targeted ``get_next_batch``.
    """

    _build_speculation = Worker._build_speculation
    _can_speculate = Worker._can_speculate
    _tp_lead_needs_marker = Worker._tp_lead_needs_marker
    _is_tp_lead_pending = Worker._is_tp_lead_pending
    _tp_async_for = Worker._tp_async_for
    _is_tearing_down = Worker._is_tearing_down
    _streams_possible = Worker._streams_possible
    _rid_str = Worker._rid_str
    _real_try_speculate_next = Worker._try_speculate_next

    def __init__(self, sched, manager):
        self.scheduler = sched
        self.request_state = manager
        self._graph_runtime = sched.runtime
        self.parallel_nodes = set()
        self.parallel_leader_nodes = {NODE, OTHER}
        self.tp_async_sched = False
        self.tp_async_nodes = None
        self.enable_nvtx = False
        self._phase_period = 0
        self._pending_removes = set()
        self._pending_drains = set()
        self._draining_rids = set()
        self._stream_buffer_rids = set()

    @contextlib.contextmanager
    def _span(self, name):
        yield

    def maybe_send_zmq_to_tp_followers(self, *args, **kwargs):
        return -1

    def _build_executing_batch(self, batch):
        return SimpleNamespace(request_ids=list(batch.request_to_worker_graph))

    def _try_speculate_next(self, pending):
        try:
            return self._real_try_speculate_next(pending)
        except _PastBacklogGateError as gate:
            rids = list(gate.args[0])
        fresh = self.scheduler.get_next_batch(
            self.request_state, target=(pending.node_name, pending.graph_walk),
            pre_existing_batch_size=len(rids), capture_group_of=rids[0],
        )
        if fresh is not None:
            rids += [r for r in fresh.request_to_worker_graph if r not in rids]
        return Speculation(
            scheduled_batch=ScheduledBatch(
                node_name=pending.node_name, graph_walk=pending.graph_walk,
                request_to_worker_graph=dict.fromkeys(rids, "wg0"),
            ),
            node_batch=SimpleNamespace(request_ids=rids),
            consumed_edges=set(), continuing_rids=set(rids), partition="default",
            is_new_iter=True, is_same_node=True,
        )


def _async_order(groups, steps=200, peek=True, max_consecutive=1024, other_stays_ready=False):
    """What each worker step ran: OTHER, or the sorted NODE rids. With
    ``other_stays_ready`` OTHER goes ready again after every step it runs, as
    s3gen does while decode keeps producing chunks; otherwise the run stops
    at OTHER's first step."""
    manager, sched = _scene(groups)
    sched.runtime = _AsyncRuntime(manager)
    loop = _Loop(sched, manager)
    fair = _SpecFairness(peek=peek, max_consecutive=max_consecutive)
    pending, consecutive, order = None, 0, []
    for _ in range(steps):
        speculation, yield_away = loop._build_speculation(pending, consecutive, fair)
        if pending is not None:
            # N completes: its rids are re-readied unless N+1 already holds them
            done = pending.batch
            sched._set_in_flight(NODE, done.request_to_worker_graph, False)
            held = set() if speculation is None or speculation.is_yield_away \
                else set(speculation.scheduled_batch.request_to_worker_graph)
            for rid in done.request_to_worker_graph:
                if rid not in held:
                    manager.ingest(rid)
        if speculation is not None:
            batch = speculation.scheduled_batch
            consecutive = 0 if speculation.is_yield_away else consecutive + 1
        else:
            consecutive = 0
            batch = None
            if yield_away is not None:
                batch = sched.get_next_batch(manager, exclude_target=yield_away)
            if batch is None:
                batch = sched.get_next_batch(manager)
        if batch is None:
            break
        if batch.node_name == OTHER:
            order.append(OTHER)
            if not other_stays_ready:
                break
            manager.queues["wg0"]._ready[2] = {OTHER}
            pending = None
            continue
        order.append(tuple(sorted(batch.request_to_worker_graph)))
        sched._set_in_flight(NODE, batch.request_to_worker_graph, True)
        pending = SimpleNamespace(
            batch=batch, node_name=NODE, graph_walk=WALK, tp_seq=-1,
            node_batch=SimpleNamespace(admit_error=None),
        )
    return order


def _first_service_async(groups, **kwargs):
    """The worker step at which OTHER first runs, or None."""
    order = _async_order(groups, **kwargs)
    return len(order) - 1 if order and order[-1] == OTHER else None


def _rids_per_turn(order):
    """For each stretch of NODE steps between two OTHER steps: how many steps
    each rid took part in."""
    turns, current = [], {}
    for step in order:
        if step == OTHER:
            turns.append(current)
            current = {}
        else:
            for rid in step:
                current[rid] = current.get(rid, 0) + 1
    return turns[1:]  # the stretch before the first OTHER step is a warm-up


def test_async_path_serves_the_other_node_beside_two_groups():
    first = _first_service_async(MIXED)
    assert first is not None and first <= 5, (
        f"OTHER first ran at step {first}: every fairness yield went back to "
        "the decode node's own backlog"
    )


def test_async_path_with_one_group_is_unchanged():
    assert _first_service_async(SAME) == 2


def test_two_groups_share_the_other_node_like_one_group_does():
    # Between two s3gen steps each guidance group decodes as many tokens as a
    # single group does: s3gen then gets the same share per decoded token, so
    # its batches are as large. A turn per group would halve them.
    same = _rids_per_turn(_async_order(SAME, steps=60, other_stays_ready=True))
    mixed = _rids_per_turn(_async_order(MIXED, steps=60, other_stays_ready=True))
    assert same and set(map(str, same)) == {str({0: 2, 1: 2})}
    assert len(mixed) >= 5, f"OTHER ran only {len(mixed) + 1} times: {mixed}"
    assert all(turn == {0: 2, 1: 2} for turn in mixed), (
        f"tokens per rid between OTHER steps: two groups {mixed}, one group {same[0]}"
    )


def test_the_consecutive_cap_alone_no_longer_needs_the_peek():
    # MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=1 yields every step; the yield has to
    # reach the other node rather than the decode node's backlog
    first = _first_service_async(MIXED, peek=False, max_consecutive=1)
    assert first is not None and first <= 5, f"OTHER first ran at step {first}"
