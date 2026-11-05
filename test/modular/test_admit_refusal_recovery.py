"""What the worker does with a step whose admit was refused, and with a
speculation that was built but never submitted.

* The refused-step branch of ``StepPipeline._finish_chain_step`` protects the rids of a surviving
  (yield-away) speculation from the teardown and eviction it runs; it read
  them off a ``ScheduledBatch`` field that no longer exists.
* A speculation that never reached the GPU thread holds rids that came off
  their ready queues; when the iteration raises they go back, whether or not
  a pre-plan was armed (none is without pre-plan, or after a refusal reset).
* A terminal refusal (``AdmitRuntimeError``) fails the requests it names
  instead of re-queueing the step for ever; a full ring KV pool is retryable
  and holds only the request that waits.

Drives the real ``StepPipeline`` over a stub engine and a real
``RingKVManager``.
"""

from __future__ import annotations

import math
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
from mstar.testing_utils.worker import make_worker  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.speculation import Speculation, Speculator  # noqa: E402
from mstar.worker.step_executor import PendingBatch  # noqa: E402

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


def _worker(*, follower=False, engine_manager=None, **components):
    w = make_worker(
        worker_id=0, graph_runtime=_Runtime(), engine_manager=engine_manager or SimpleNamespace(
            get_engine=lambda node: SimpleNamespace(release_cached_prefix=lambda batch: None),
        ),
        parallel_nodes={NODE} if follower else set(), is_tp_follower=follower, **components,
    )
    w.reported: dict[int, str] = {}
    w.pipeline._fail_requests = w.reported.update
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
    assert Speculator.speculated_rids(_yield_away([3, 4])) == {3, 4}
    assert Speculator.speculated_rids(None) == frozenset()


# ── a speculation that was never submitted ──────────────────────────────


def test_an_unsubmitted_yield_away_gives_its_rids_back_when_the_iteration_raises():
    """No pre-plan armed (the refusal branch resets it; without pre-plan there
    is none): its rids still have to go back to their queues."""
    w = _worker()
    pending = _pending([1, 2])
    spec = _yield_away([3, 4])
    assert spec.plan_future is None

    w.pipeline._handle_error(RuntimeError("boom"), (pending, None), None, spec)

    assert sorted(w._graph_runtime.pushed) == [3, 4], "yield-away rids stranded"
    assert set(w.reported) == {1, 2}


def test_a_submitted_speculation_fails_with_its_step():
    w = _worker()
    pending = _pending([1, 2])
    spec = _yield_away([3, 4])
    spec.submitted = True
    submitted = _pending([3, 4])
    submitted.batch = spec.scheduled_batch

    w.pipeline._handle_error(RuntimeError("boom"), (pending, submitted), None, spec)

    assert w._graph_runtime.pushed == []
    assert set(w.reported) == {1, 2, 3, 4}


# ── terminal refusals ───────────────────────────────────────────────────


def _refused(rids, reason):
    node_batch = _executing(rids)
    node_batch.register_admit_error(reason, "kv")
    return _scheduled(rids), node_batch


def _next_pass(w) -> None:
    """A refused step's admit errors, as the top of the worker's next pass fails them."""
    w.pipeline._fail_requests(w.admission.take_admit_errors())


def test_a_terminal_refusal_only_records_until_the_next_pass():
    """The refusal holds its rows so that no scan takes them meanwhile; they
    are failed once, at the top of the next pass, and the hold goes with it."""
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("clock skipped", request_ids=frozenset({7})))

    w.pipeline._handle_admit_failure(batch, node_batch)

    assert not w.reported
    assert set(w.admission.admit_errors) == {7}
    assert w.admission.held_until == {7: math.inf}

    _next_pass(w)

    assert set(w.reported) == {7}
    assert not w.admission.admit_errors and not w.admission.held_until


def test_a_terminal_refusal_fails_the_request_it_names_and_requeues_the_rest():
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("clock skipped", request_ids=frozenset({7})))

    w.pipeline._handle_admit_failure(batch, node_batch)
    _next_pass(w)

    assert set(w.reported) == {7}
    assert w._graph_runtime.pushed == [8]


def test_only_a_runtime_error_is_terminal():
    assert AdmitRuntimeError("x").terminal
    assert not AllocationFailed("x", pages_short=1, label="main", request_id=7).terminal
    assert not RequestOffloading("x", label="main", request_id=7).terminal


def test_a_terminal_refusal_fails_every_request_it_names():
    w = _worker()
    batch, node_batch = _refused([6, 7, 8], AdmitRuntimeError("bad rows", request_ids=frozenset({6, 8})))

    w.pipeline._handle_admit_failure(batch, node_batch)
    _next_pass(w)

    assert set(w.reported) == {6, 8}
    assert w._graph_runtime.pushed == [7]


def test_a_terminal_refusal_naming_no_row_of_the_batch_fails_every_row():
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("elsewhere", request_ids=frozenset({9})))

    w.pipeline._handle_admit_failure(batch, node_batch)
    _next_pass(w)

    assert set(w.reported) == {7, 8}
    assert w._graph_runtime.pushed == []


def test_a_terminal_refusal_about_the_whole_step_fails_every_row():
    w = _worker()
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("a row named twice"))

    w.pipeline._handle_admit_failure(batch, node_batch)
    _next_pass(w)

    assert set(w.reported) == {7, 8}
    assert w._graph_runtime.pushed == []


def test_a_tp_follower_leaves_the_failure_to_rank_0():
    w = _worker(follower=True)
    batch, node_batch = _refused([7, 8], AdmitRuntimeError("clock skipped", request_ids=frozenset({7})))

    w.pipeline._handle_admit_failure(batch, node_batch)
    _next_pass(w)

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
    engine = SimpleNamespace(evictable=lambda n, resources=None: False)
    w = _worker(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        scheduler=SimpleNamespace(last_in_line=lambda *args: None, could_refill=lambda *args: False),
    )
    batch, node_batch = _ring_refusal(2, ["r0"], ["a", "b"])
    assert isinstance(node_batch.admit_error, AllocationFailed)

    w.pipeline._handle_admit_failure(batch, node_batch)

    assert not w.reported and not w.admission.admit_errors, "a full pool is not a failure"
    assert list(w.admission.held_until) == ["b"]
    assert sorted(w._graph_runtime.pushed) == ["a", "b"]


def test_a_ring_clock_skip_fails_only_that_request():
    from test_ring_kv_resource import _ctx, _drive, _manager, _step

    kv = _manager()
    kv.ingest_request("a")
    _drive(kv, "a", 0)
    outcome = kv.admit(_step("a", frame=2), _ctx("a"))
    batch, node_batch = _refused(["a"], outcome.reason)
    w = _worker()

    w.pipeline._handle_admit_failure(batch, node_batch)
    _next_pass(w)

    assert set(w.reported) == {"a"}
    assert w._graph_runtime.pushed == []
