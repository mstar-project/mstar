"""What the worker does with a step whose admit was refused, and with a
speculation that was built but never submitted.

* The refused-step branch of ``Worker.run`` protects the rids of a surviving
  (yield-away) speculation from the teardown and eviction it runs; it read
  them off a ``ScheduledBatch`` field that no longer exists.
* A speculation that never reached the GPU thread holds rids that came off
  their ready queues; when the iteration raises they go back, whether or not
  a pre-plan was armed (none is without pre-plan, or after a refusal reset).
* A terminal refusal (``AdmitRuntimeError``) fails the requests it names
  instead of re-queueing the step for ever; a full ring KV pool is retryable
  and holds only the request that waits.

Drives the real ``Worker`` methods on a stub ``self`` and a real
``RingKVManager``.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    AdmitRuntimeError,
    AllocationFailed,
    RequestOffloading,
    StepContext,
)
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.worker import PendingBatch, Speculation, Worker  # noqa: E402

NODE, WALK = "dit", "rollout"


class _Runtime:
    def __init__(self):
        self.pushed: list[int] = []
        self.in_flight: dict[int, bool] = {}

    def push_back_node(self, node, rids, wg_ids):
        self.pushed.extend(rids)

    def set_in_flight(self, node, wg_id, rids, value):
        self.in_flight.update(dict.fromkeys(rids, value))

    def commit_speculation(self, spec_id, success, dropped_rids, **kw):
        pass


def _worker(*, follower=False):
    w = Worker.__new__(Worker)
    w.worker_id = 0
    w._graph_runtime = _Runtime()
    w.reported: dict[int, str] = {}
    w._fail_requests = w.reported.update
    w._in_flight_rids = set()
    w.parallel_nodes = {NODE} if follower else set()
    w.parallel_leader_nodes = set()
    w.engine_manager = SimpleNamespace(
        get_engine=lambda node: SimpleNamespace(release_cached_prefix=lambda batch: None),
    )
    return w


def _scheduled(rids):
    return ScheduledBatch(
        node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict.fromkeys(rids, 0),
    )


def _executing(rids):
    return ExecutingBatch(
        node_name=NODE, per_request_info={r: SimpleNamespace() for r in rids},
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
        ),
    )


def _pending(rids):
    future = Future()
    future.set_result({})
    return PendingBatch(
        batch=_scheduled(rids), node_batch=_executing(rids),
        node_name=NODE, partition="p", graph_walk=WALK, future=future,
    )


def _yield_away(rids):
    return Speculation(
        scheduled_batch=_scheduled(rids), node_batch=_executing(rids),
        consumed_edges=set(), continuing_rids=set(), partition="p",
        is_new_iter=False, is_same_node=False, is_yield_away=True,
    )


# ── the refused-step branch ─────────────────────────────────────────────


def test_the_speculation_rids_come_off_a_real_scheduled_batch():
    assert Worker._speculated_rids(_yield_away([3, 4])) == {3, 4}
    assert Worker._speculated_rids(None) == frozenset()


# ── a speculation that was never submitted ──────────────────────────────


def test_an_unsubmitted_yield_away_gives_its_rids_back_when_the_iteration_raises():
    """No pre-plan armed (the refusal branch resets it; without pre-plan there
    is none): its rids still have to go back to their queues."""
    w = _worker()
    pending = _pending([1, 2])
    spec = _yield_away([3, 4])
    assert spec.plan_future is None

    w._handle_main_loop_error(RuntimeError("boom"), (pending, None), None, spec)

    assert sorted(w._graph_runtime.pushed) == [3, 4], "yield-away rids stranded"
    assert set(w.reported) == {1, 2}


def test_a_submitted_speculation_fails_with_its_step():
    w = _worker()
    pending = _pending([1, 2])
    spec = _yield_away([3, 4])
    spec.submitted = True
    submitted = _pending([3, 4])
    submitted.batch = spec.scheduled_batch

    w._handle_main_loop_error(RuntimeError("boom"), (pending, submitted), None, spec)

    assert w._graph_runtime.pushed == []
    assert set(w.reported) == {1, 2, 3, 4}


# ── terminal refusals ───────────────────────────────────────────────────


def _refused(rids, reason):
    node_batch = _executing(rids)
    node_batch.register_admit_error(reason, "kv")
    return _scheduled(rids), node_batch


def test_a_terminal_refusal_fails_the_request_it_names_and_requeues_the_rest():
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("clock skipped", request_ids=frozenset({7})))

    w._handle_admit_failure(batch, node_batch)

    assert set(w.reported) == {7}
    assert w._graph_runtime.pushed == [8]


def test_only_a_runtime_error_is_terminal():
    assert AdmitRuntimeError("x").terminal
    assert not AllocationFailed("x", pages_short=1, label="main", request_id=7).terminal
    assert not RequestOffloading("x", label="main", request_id=7).terminal


def test_a_terminal_refusal_fails_every_request_it_names():
    w = _worker()
    batch, node_batch = _refused([6, 7, 8], AdmitRuntimeError("bad rows", request_ids=frozenset({6, 8})))

    w._handle_admit_failure(batch, node_batch)

    assert set(w.reported) == {6, 8}
    assert w._graph_runtime.pushed == [7]


def test_a_terminal_refusal_naming_no_row_of_the_batch_fails_every_row():
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("elsewhere", request_ids=frozenset({9})))

    w._handle_admit_failure(batch, node_batch)

    assert set(w.reported) == {7, 8}
    assert w._graph_runtime.pushed == []


def test_a_terminal_refusal_about_the_whole_step_fails_every_row():
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("a row named twice"))

    w._handle_admit_failure(batch, node_batch)

    assert set(w.reported) == {7, 8}
    assert w._graph_runtime.pushed == []


def test_a_tp_follower_leaves_the_failure_to_rank_0():
    w = _worker(follower=True)
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("clock skipped", request_ids=frozenset({7})))

    w._handle_admit_failure(batch, node_batch)

    assert not w.reported
    assert sorted(w._graph_runtime.pushed) == [7, 8]


# ── the ring KV pool, end to end ────────────────────────────────────────


def _ring_refusal(num_sessions, held, wanted):
    """A real RingKVManager with ``held`` rids holding every session, asked to
    admit ``wanted``: returns the refused (batch, node_batch)."""
    from test_ring_kv_resource import _ctx, _manager, _open, _ring_config, _step

    kv = _manager(_ring_config(num_sessions=num_sessions))
    _open(kv, *held)
    for rid in wanted:
        kv.ingest_request(rid)
    outcome = kv.admit(_step(*wanted), _ctx(*wanted))
    assert not outcome.ok
    return _refused(list(wanted), outcome.reason)


def test_a_full_ring_pool_holds_the_request_that_waits_and_runs_the_rest():
    w = _worker()
    w.scheduler = SimpleNamespace(held=[])
    w.scheduler.hold_requests = w.scheduler.held.extend
    w.engine_manager = SimpleNamespace(
        get_engine=lambda node: SimpleNamespace(evictable=lambda n: False),
    )
    w._hold_logged = {}
    w._apply_removes_whose_step_landed = lambda: None
    w._apply_pending_removes_safe_to_drop = lambda rids: None
    batch, node_batch = _ring_refusal(2, ["r0"], ["a", "b"])
    assert isinstance(node_batch.admit_error, AllocationFailed)

    w._handle_admit_failure(batch, node_batch)

    assert not w.reported, "a full pool is not a failure"
    assert w.scheduler.held == ["b"]
    assert sorted(w._graph_runtime.pushed) == ["a", "b"]


def test_a_ring_clock_skip_fails_only_that_request():
    from test_ring_kv_resource import _ctx, _drive, _manager, _step

    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)
    outcome = kv.admit(_step("a", frame=2), _ctx("a"))
    batch, node_batch = _refused(["a"], outcome.reason)
    w = _worker()

    w._handle_admit_failure(batch, node_batch)

    assert set(w.reported) == {"a"}
    assert w._graph_runtime.pushed == []
