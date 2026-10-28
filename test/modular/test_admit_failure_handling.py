"""What happens to a batch whose admit refused it.

Admit failure means no forward ran, so the step's per-rid outputs are empty.
Two things follow, and neither used to be keyed off ``admit_error`` — they
read ``allocation_failed``, which was true only because ``AllocationFailed``
was the only reason a resource could give:

1. the engine must not run ``postprocess`` over those empty outputs, and
2. the worker must re-queue the batch, or the requests are never retried.

Only an ``AllocationFailed`` additionally calls for an eviction;
``RequestOffloading`` says the rid is already on its way to the host. When
nothing can be evicted the batch is held and retried every backoff, so the line
saying so is kept to one per node and walk every few seconds.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_micro_scheduler import MIXED, _combined_scheduler, _Engine, _Manager, _next_batch, _scheduler
from test_micro_scheduler import NODE as SCHED_NODE
from test_micro_scheduler import WALK as SCHED_WALK

from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    FullAdmitOutcome,
    RequestOffloading,
    StepContext,
    SubmoduleStep,
)
from mstar.utils.device import DeviceCapability
from mstar.worker import worker as worker_mod
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker


def _alloc_failed(rid: str = "r0") -> AllocationFailed:
    return AllocationFailed(
        message="out of pages", pages_short=4, label="main", request_id=rid,
    )


def _offloading(rid: str = "r0") -> RequestOffloading:
    return RequestOffloading(
        message="being offloaded", label="main", request_id=rid,
    )


# --- engine: the tail must not run over a step that never ran --------------


class _FakeEngine:
    """Just enough engine to exercise the ``exec``/``postprocess`` branch."""

    exec_and_postprocess = Engine.exec_and_postprocess

    def __init__(self, admit_error):
        self._admit_error = admit_error
        self.postprocessed = False

    def exec(self, batch):
        batch.admit_error = self._admit_error
        # what `_exec_single` returns when admit refused the step
        return {rid: {} for rid in batch.request_ids}

    def postprocess_batch(self, batch, outputs):
        del batch, outputs
        self.postprocessed = True


def _batch():
    return SimpleNamespace(request_ids=["r0", "r1"], admit_error=None, cached_rids=set(), rows_only=False)


@pytest.mark.parametrize(
    "reason", [_alloc_failed(), _offloading()], ids=["allocation", "offloading"]
)
def test_postprocess_skipped_on_admit_error(reason):
    engine = _FakeEngine(reason)
    engine.exec_and_postprocess(_batch())
    assert engine.postprocessed is False


def test_postprocess_runs_on_a_step_that_admitted():
    engine = _FakeEngine(None)
    engine.exec_and_postprocess(_batch())
    assert engine.postprocessed is True


# --- worker: every admit failure is re-queued ------------------------------


class _Runtime:
    """The runtime owns the ready queues, so the push-back lands here."""

    def __init__(self):
        self.pushed_back: list[str] = []

    def push_back_node(self, node_name, rids, wg_ids):
        del node_name, wg_ids
        self.pushed_back.extend(rids)


class _FakeWorker:
    """Binds the two handlers onto stubs for their collaborators."""

    _handle_admit_failure = Worker._handle_admit_failure
    _push_back_batch = Worker._push_back_batch

    def __init__(self):
        self._graph_runtime = _Runtime()
        self.held: list[str] = []
        self.scheduler = SimpleNamespace(hold_requests=self.held.extend)
        self.offload_calls: list[str] = []

    @property
    def queue(self):
        # The assertions read .pushed_back; keep that name pointing at
        # whoever owns the ready queues now.
        return self._graph_runtime

    def _handle_allocation_failure(self, batch, node_batch, referenced_rids=frozenset()):
        self.offload_calls.append((node_batch.node_name, referenced_rids))
        # the real one push-backs and holds; stand in for both
        self._push_back_batch(batch)
        self.scheduler.hold_requests(list(batch.request_to_worker_graph))


def _batches(reason, prepared=("r0", "r1"), cached=()):
    batch = ScheduledBatch(
        node_name="node", graph_walk="walk",
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    node_batch = SimpleNamespace(
        node_name="node", admit_error=reason, request_ids=list(prepared), cached_rids=set(cached),
        rebuild_rids=frozenset(),
    )
    return batch, node_batch


def test_offloading_requeues_without_evicting():
    worker = _FakeWorker()
    worker._handle_admit_failure(*_batches(_offloading()))

    assert sorted(worker.queue.pushed_back) == ["r0", "r1"]
    # nothing to evict: the rid is already on its way to the host, and
    # `check_ready` gates the retry on reloading it
    assert worker.offload_calls == []
    assert worker.held == []


def test_allocation_failure_still_evicts():
    worker = _FakeWorker()
    worker._handle_admit_failure(
        *_batches(_alloc_failed()), referenced_rids=frozenset({"spec_rid"}),
    )

    # the set has to reach the handler: it is what the teardown flush spares
    assert worker.offload_calls == [("node", frozenset({"spec_rid"}))]
    # exactly one push-back per request: the delegation must not double up
    assert sorted(worker.queue.pushed_back) == ["r0", "r1"]
    assert sorted(worker.held) == ["r0", "r1"]


class _StoppedLoopRuntime(_Runtime):
    def __init__(self, stopped):
        super().__init__()
        self.stopped = stopped

    def pending_loop_stop_rids(self, graph_walks, loop_name):
        assert (graph_walks, loop_name) == (["walk"], "loop")
        return set(self.stopped)


@pytest.mark.parametrize("new_iter", [True, False])
def test_a_refused_new_iteration_does_not_push_back_a_row_whose_loop_stopped(new_iter):
    """A row on its last token rides speculative N+1; N's postprocess stops its
    loop and empties its node. Pushed back, it popped with no inputs."""
    worker = _FakeWorker()
    worker._graph_runtime = _StoppedLoopRuntime({"r1"})
    batch, node_batch = _batches(_alloc_failed())
    pending = SimpleNamespace(batch=batch, speculative_new_iter=new_iter, loop_name="loop")

    Worker._drop_rows_whose_loop_stopped(worker, pending)
    worker._handle_admit_failure(batch, node_batch)

    assert sorted(worker.queue.pushed_back) == (["r0"] if new_iter else ["r0", "r1"])


# the scheduler's batch-size calls, for stubs whose tests are about the holds
_NO_LIMITS = dict(
    could_refill=lambda *args: False, cap_batch=lambda *args: None, limit_batch=lambda *args: None,
)


class _HoldingWorker:
    """Binds the real allocation handler onto a worker with nothing to evict."""

    _handle_allocation_failure = Worker._handle_allocation_failure
    _push_back_batch = Worker._push_back_batch
    _is_tp_follower_node = Worker._is_tp_follower_node
    _preempt = Worker._preempt
    _log_throttled = Worker._log_throttled

    def __init__(self):
        self._graph_runtime = _Runtime()
        self.scheduler = SimpleNamespace(
            hold_requests=lambda rids: None, **_NO_LIMITS,
        )
        # no paged KV cache, so nothing preempts
        self._kv_caches = {}
        self._hold_logged = {}
        # not a sharded node, so this rank runs its own OOM recovery
        self.parallel_nodes = set()
        self.parallel_leader_nodes = set()
        self._in_flight_rids: set[str] = set()

    def _apply_removes_whose_step_landed(self):
        pass

    def _apply_pending_removes_safe_to_drop(self, in_flight_rids):
        del in_flight_rids

    def _try_offload_cold_request(self, node_name, batch_ids, affected_resources=None, inside_only=False):
        # no victim
        del node_name, batch_ids, affected_resources, inside_only


def _hold(worker, monkeypatch, at: float, walk: str = "walk") -> None:
    monkeypatch.setattr(worker_mod, "_time", SimpleNamespace(monotonic=lambda: at))
    batch, node_batch = _batches(_alloc_failed())
    batch.graph_walk = walk
    node_batch.failed_resource = None
    worker._handle_allocation_failure(batch, node_batch)


def _hold_lines(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if "no offload possible" in record.getMessage()
    ]


def test_a_hundred_holds_on_one_node_log_two_lines_and_count_the_rest(monkeypatch, caplog):
    worker = _HoldingWorker()
    step = worker_mod._HOLD_LOG_INTERVAL / 50

    with caplog.at_level(logging.WARNING, logger=worker_mod.__name__):
        # a retry every fiftieth of the interval, across two intervals
        for tick in range(100):
            _hold(worker, monkeypatch, tick * step)

    lines = _hold_lines(caplog)
    assert len(lines) == 2, f"a hundred holds logged {len(lines)} lines, not one per interval"
    assert "(0 earlier holds not logged)" in lines[0], lines[0]
    assert "(49 earlier holds not logged)" in lines[1], (
        f"the second line lost count of the holds it stands for: {lines[1]}"
    )


class _RefusedWorker(_HoldingWorker):
    """Both real handlers, nothing to evict, and the holds recorded."""

    _handle_admit_failure = Worker._handle_admit_failure

    def __init__(self):
        super().__init__()
        self.held: list[str] = []
        self.scheduler = SimpleNamespace(hold_requests=self.held.extend, **_NO_LIMITS)


@pytest.mark.parametrize(
    "reason", [_alloc_failed("r0"), _offloading("r0")], ids=["allocation", "offloading"]
)
def test_a_rid_prepare_dropped_is_not_requeued(reason):
    """r1 finished its loop; the speculated extra step reached prepare_inputs,
    which vetoed it, then r0's admit refused the step. Re-queued, r1 came back
    with its loop-back inputs spent and the Cosmos3 DiT read that as a fresh
    loop start over its spent scheduler (UniPC ``this_order > 0``, every
    request in the next batch failed). A step that runs drops r1 the same way."""
    worker = _RefusedWorker()
    batch, node_batch = _batches(reason, prepared=["r0"])
    node_batch.failed_resource = None

    worker._handle_admit_failure(batch, node_batch)

    assert worker._graph_runtime.pushed_back == ["r0"]
    assert "r1" not in worker.held


@pytest.mark.parametrize(
    "reason", [_alloc_failed("r0"), _offloading("r0")], ids=["allocation", "offloading"]
)
def test_a_rid_the_cache_served_is_requeued(reason):
    worker = _RefusedWorker()
    batch, node_batch = _batches(reason, prepared=["r0"], cached=["r1"])
    node_batch.failed_resource = None

    worker._handle_admit_failure(batch, node_batch)

    assert sorted(worker._graph_runtime.pushed_back) == ["r0", "r1"], (
        "a row the cache served left the refused step, so its walk never completes"
    )


def test_a_hold_on_another_walk_is_logged_inside_the_interval(monkeypatch, caplog):
    worker = _HoldingWorker()

    with caplog.at_level(logging.WARNING, logger=worker_mod.__name__):
        _hold(worker, monkeypatch, 0.0, walk="prefill")
        _hold(worker, monkeypatch, 0.1, walk="decode")

    assert len(_hold_lines(caplog)) == 2, (
        "one walk's hold silenced another's, so a stall on it would go unseen"
    )


# --- a refused batch with nothing to evict is retried smaller ---------------


class _SchedEngine(_Engine):
    """The scheduler's engine fake, with the offload state the refill check reads."""

    def __init__(self, *args, offloaded=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.offloaded = set(offloaded)

    def is_offloaded(self, node_name, rid):
        del node_name
        return rid in self.offloaded


def _sized_refusal(rid, rows_fit=None, rows_fit_alone=None) -> AllocationFailed:
    error = _alloc_failed(rid)
    error.rows_fit, error.rows_fit_alone = rows_fit, rows_fit_alone
    return error


def _refusing_loop(fits: int, ready: int, cap: int, passes: int, sized: bool = True):
    """Schedule and admit like the worker loop, against a real scheduler: a
    batch of more than ``fits`` rows is refused on its first row past them,
    with nothing to evict. ``sized`` refusals say how many rows fit, as the
    paged KV cache's do. Returns the scheduler, the batch sizes refused, and
    the first batch admitted (None if none was)."""
    sched = _scheduler(_SchedEngine(max_bs=cap))
    manager = _Manager([f"r{i}" for i in range(ready)])
    worker = _HoldingWorker()
    worker.scheduler = sched
    worker._graph_runtime = manager.runtime
    refused = []
    for _ in range(passes):
        batch = _next_batch(sched, manager)
        rids = list(batch.request_to_worker_graph)
        if len(rids) <= fits:
            return sched, refused, rids
        refused.append(len(rids))
        node_batch = SimpleNamespace(
            node_name=SCHED_NODE, failed_resource="kv",
            admit_error=_sized_refusal(rids[fits], fits if sized else None),
        )
        worker._handle_allocation_failure(batch, node_batch)
    return sched, refused, None


def test_a_refused_batch_with_nothing_to_evict_is_retried_smaller():
    """Fifteen rows fit and the cap is sixteen. Holding the blamed row alone
    lets the ready queue refill the batch to sixteen, which is refused again
    on every pass while nothing runs."""
    _, refused, admitted = _refusing_loop(fits=15, ready=48, cap=16, passes=32)

    assert admitted is not None, (
        f"{len(refused)} retries were all refused at sizes {sorted(set(refused))}: "
        "the batch never got smaller than the cap"
    )
    assert refused == [16] and len(admitted) == 15


def test_one_refusal_sizes_the_retry_from_the_rows_that_fit():
    """Three of eight fit: the retry is three rows, not one row fewer per
    refusal (8, 7, 6, 5, 4 refused before)."""
    _, refused, admitted = _refusing_loop(fits=3, ready=48, cap=8, passes=32)

    assert refused == [8]
    assert len(admitted) == 3


def test_a_refusal_that_does_not_say_what_fits_halves_the_batch():
    _, refused, admitted = _refusing_loop(fits=3, ready=48, cap=8, passes=32, sized=False)

    assert refused == [8, 4]
    assert len(admitted) == 2


def test_an_admitted_step_lifts_the_limit():
    sched, _, admitted = _refusing_loop(fits=15, ready=48, cap=16, passes=4)
    assert len(admitted) == 15

    sched.clear_batch_limit(SCHED_NODE, SCHED_WALK, len(admitted))

    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 16


def test_the_limit_never_drops_below_one_row():
    sched = _scheduler(_SchedEngine(max_bs=4))
    sched.limit_batch(SCHED_NODE, SCHED_WALK, 0)
    sched.cap_batch(SCHED_NODE, SCHED_WALK, 0)

    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 1


def test_a_refused_batch_below_its_cap_is_not_limited():
    """Nine running requests decode in a step of nine, the cap is 32, and the
    admit refuses with nothing to evict. Every ready row is already in the
    batch, so holding the blamed one is what makes the retry smaller. A limit
    of eight on top kept a running row out of the steps after the hold, and
    a row outside the batch is the one the next eviction takes first."""
    sched = _scheduler(_SchedEngine(max_bs=32))
    manager = _Manager([f"d{i}" for i in range(9)])
    worker = _HoldingWorker()
    worker.scheduler = sched
    worker._graph_runtime = manager.runtime
    batch = _next_batch(sched, manager)
    rids = list(batch.request_to_worker_graph)
    assert len(rids) == 9

    worker._handle_allocation_failure(batch, SimpleNamespace(
        node_name=SCHED_NODE, failed_resource="kv",
        admit_error=_sized_refusal(rids[6], rows_fit=6, rows_fit_alone=9),
    ))
    sched.held_until.clear()  # the backoff ran out

    retry = _next_batch(sched, manager)
    assert len(retry.request_to_worker_graph) == 9, "a running row was kept out of the step"


def test_a_mixed_step_held_under_its_cap_by_row_caps_is_limited():
    """The combined cap is eight, but prefill's cap and each walk's row cap
    are one, so with three ready rows of each walk every step is a mixed
    step of two. Its refusal left no limit while the batch was under the
    combined cap, and the waiting rows refilled every retry to two."""
    sched = _combined_scheduler(_SchedEngine(
        max_bs={"prefill": 1, SCHED_WALK: 8, MIXED: 8}, row_caps={"prefill": 1, SCHED_WALK: 1},
    ))
    manager = _Manager(
        ["p0", "p1", "p2", "d0", "d1", "d2"], walks={f"p{i}": "prefill" for i in range(3)},
    )
    worker = _HoldingWorker()
    worker.scheduler = sched
    worker._graph_runtime = manager.runtime
    batch = _next_batch(sched, manager)
    rids = list(batch.request_to_worker_graph)
    assert batch.graph_walk == MIXED and len(rids) == 2

    worker._handle_allocation_failure(batch, SimpleNamespace(
        node_name=SCHED_NODE, failed_resource="kv",
        admit_error=_sized_refusal(rids[1], rows_fit=1, rows_fit_alone=2),
    ))

    assert len(_next_batch(sched, manager).request_to_worker_graph) == 1


class _EvictingWorker(_HoldingWorker):
    """The real victim search, over requests ``holders`` that hold pages."""

    _try_offload_cold_request = Worker._try_offload_cold_request
    _select_eviction_victim = Worker._select_eviction_victim

    def __init__(self, holders):
        super().__init__()
        self.offloaded: list[str] = []
        self.eviction_policy = SimpleNamespace(value="lru")
        self._last_active = {(rid, SCHED_NODE): float(i) for i, rid in enumerate(holders)}
        engine = SimpleNamespace(
            evictable=lambda node: True,
            is_offloaded=lambda node, rid: rid in self.offloaded,
            reclaimable=lambda node, rid, resources: rid in holders,
            offload_request=lambda node, rid, affected_resources=None: self.offloaded.append(rid) or 12,
        )
        # #461's worker keeps one; harmless here
        self._no_victim_logged = {}
        self.engine_manager = SimpleNamespace(get_engine=lambda node: engine)


def test_a_batch_the_pool_cannot_hold_alone_is_capped_not_evicted_for():
    """Sixteen prefills of 12 pages need 192 and the pool has 191. Four
    decoding requests hold pages, and on each refusal one was offloaded and
    the batch retried at sixteen: it can never fit, and the victim reloads
    into the pages it freed. Nothing outside the batch is evicted now, and
    the walk stays capped at fifteen after a step of fifteen admits."""
    usable, row = 191, 12
    sched = _scheduler(_SchedEngine(max_bs=16))
    manager = _Manager([f"p{i}" for i in range(48)])
    worker = _EvictingWorker(holders=[f"d{i}" for i in range(4)])
    worker.scheduler = sched
    worker._graph_runtime = manager.runtime
    sizes = []
    for _ in range(8):
        batch = _next_batch(sched, manager)
        rids = list(batch.request_to_worker_graph)
        sizes.append(len(rids))
        # an offloaded decoder reloads by the next pass, so its pages are back
        free = usable - row * 4
        if row * len(rids) <= free:
            break
        worker._handle_allocation_failure(batch, SimpleNamespace(
            node_name=SCHED_NODE, failed_resource="kv",
            admit_error=_sized_refusal(
                rids[free // row], rows_fit=free // row, rows_fit_alone=usable // row,
            ),
        ))
        worker.offloaded.clear()
    assert worker.offloaded == [] and sizes == [16, 11], (
        f"batch sizes {sizes}: the full batch was retried after evicting a decoder"
    )

    sched.clear_batch_limit(SCHED_NODE, SCHED_WALK, 11)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 15
    sched.clear_batch_limit(SCHED_NODE, SCHED_WALK, 15)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 15, (
        "a step at the cap lifted it, so the next full batch is refused again"
    )
    sched.clear_batch_limit(SCHED_NODE, SCHED_WALK, 3)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 16


def test_a_batch_too_big_alone_still_evicts_its_own_rows():
    """Running requests that outgrow the pool together: evicting one of
    them is what lets the rest step, as before."""
    # the request outside the batch is the least recently run
    worker = _EvictingWorker(holders=["outside", "r0", "r1"])
    worker.scheduler = SimpleNamespace(hold_requests=lambda rids: None, **_NO_LIMITS)
    batch, node_batch = _batches(_sized_refusal("r1", rows_fit=1, rows_fit_alone=1))
    node_batch.failed_resource = "kv"
    batch.node_name = node_batch.node_name = SCHED_NODE

    worker._handle_allocation_failure(batch, node_batch)

    assert worker.offloaded == ["r0"], "an outside request was evicted for a batch it cannot help"


def test_a_step_of_one_walk_lifts_its_combined_walks_limit():
    """A mixed batch limited to one row: every step under it holds one walk
    and carries that walk's label. The first of them to admit has to lift
    the limit, or the walks never batch together again."""
    sched = _combined_scheduler(_SchedEngine(max_bs=8))
    sched.limit_batch(SCHED_NODE, MIXED, 1)

    limited = _next_batch(sched, _Manager(["p0", "d0"], walks={"p0": "prefill"}))
    assert limited.graph_walk != MIXED and len(limited.request_to_worker_graph) == 1

    sched.clear_batch_limit(SCHED_NODE, limited.graph_walk, 1)

    mixed = _next_batch(sched, _Manager(["p1", "d1"], walks={"p1": "prefill"}))
    assert mixed.graph_walk == MIXED, "the combined walk stayed limited to one row"


def test_a_mixed_step_lifts_the_limits_of_the_walks_it_ran():
    """A nine-row decode refused to two while prefills wait, then a mixed step
    of twelve, sized by the combined walk's own cap, admits. It ran decode
    rows, so the next pure-decode step is not held to two rows."""
    sched = _combined_scheduler(_SchedEngine(max_bs=16))
    sched.limit_batch(SCHED_NODE, SCHED_WALK, 2)

    mixed = _next_batch(sched, _Manager(
        [f"p{i}" for i in range(6)] + [f"d{i}" for i in range(6)],
        walks={f"p{i}": "prefill" for i in range(6)},
    ))
    assert mixed.graph_walk == MIXED and len(mixed.request_to_worker_graph) == 12

    sched.clear_batch_limit(
        SCHED_NODE, mixed.graph_walk, len(mixed.request_to_worker_graph),
        walks=mixed.rids_by_walk(),
    )

    decode = _next_batch(sched, _Manager([f"e{i}" for i in range(12)]))
    assert decode.graph_walk == SCHED_WALK
    assert len(decode.request_to_worker_graph) == 12, (
        "the decode limit outlived a mixed step that admitted decode rows"
    )


def test_could_refill_counts_the_rows_that_could_run_in_the_retry():
    """The real `could_refill`, which the handler tests above stub. Rows of
    the key outside the batch refill it, unless they are failed, held,
    being removed, offloaded or of another walk. ``pending_removes`` is the
    worker's dict: a set union over it raised TypeError."""
    sched = _scheduler(_SchedEngine(max_bs=8, offloaded={"off"}))
    manager = _Manager(
        ["b0", "b1", "w0", "failed", "held", "gone", "off", "p0"], walks={"p0": "prefill"},
    )
    sched.runtime = manager.runtime
    manager.runtime.combined_walk_of = sched.combined_walk_of
    sched.failed_rids.add("failed")
    sched.hold_requests(["held"])
    sched.pending_removes = {"gone": None}

    assert not sched.could_refill(SCHED_NODE, SCHED_WALK, {"b0", "b1", "w0"})
    assert sched.could_refill(SCHED_NODE, SCHED_WALK, {"b0", "b1"}), "w0 waits to run"

    manager.queues["wg0"]._ready.pop("w0")
    sched.backlog[(SCHED_NODE, SCHED_WALK)] = ScheduledBatch(
        node_name=SCHED_NODE, graph_walk=SCHED_WALK,
        request_to_worker_graph={"w0": "wg0", "gone": "wg0"},
    )
    assert sched.could_refill(SCHED_NODE, SCHED_WALK, {"b0", "b1"}), "backlogged w0 waits to run"
    assert not sched.could_refill(SCHED_NODE, SCHED_WALK, {"b0", "b1", "w0"})


# --- the unbatchable path admits everything before it runs anything --------


class _FakeMgmt:
    """``SubmoduleManagement``'s slot rotation, as `_exec_per_request` uses it."""

    def __init__(self, submodule, num_slots: int, next_slot: int, seen: list):
        self.submodule = submodule
        self.num_slots = num_slots
        self.needs_slot_fence = False
        self._next_slot = next_slot
        self._seen = seen

    @property
    def next_slot(self) -> int:
        return self._next_slot

    def lease_slot(self) -> int:
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % self.num_slots
        return slot

    def set_piecewise_slot(self, slot: int) -> None:
        self._seen.append(slot)


class _FakeExecEngine:
    """Engine internals bound onto stubs, to drive `_exec_per_request` alone."""

    _exec_per_request = Engine._exec_per_request
    _declare_and_admit = Engine._declare_and_admit

    def __init__(
        self, fail_on: str | None = None, num_slots: int = 1, next_slot: int = 0,
    ):
        self._enable_nvtx = False
        self._fail_on = fail_on
        # ordered log, so the test can assert admit-all-then-run
        self.events: list[tuple[str, str]] = []
        # which rids were driven with `set_launch` — the real path signals
        # `launch_started_event` there, and the worker fences on it
        self.launch_signals: list[str] = []
        self._runner = self
        self._device = torch.device("cpu")
        self.device_capability = DeviceCapability.of(self._device)
        # the per-request path's slots; no fence, so no CUDA event is recorded
        self.piecewise_slots: list[int] = []
        self.run_slots: list[int] = []
        self.mgmt = _FakeMgmt(self, num_slots, next_slot, self.piecewise_slots)
        self._submodules = {"node": self.mgmt}

    # --- submodule surface
    def _maybe_lease_piecewise_regions(self, node_name, ctx, inputs):
        pass

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        del graph_walk, inputs
        self.events.append(("declare", request_ids[0]))
        return SubmoduleStep(steps={}, segments=[])

    # --- step runner surface
    def admit(self, step):
        rid = step.ctx.request_ids[0]
        self.events.append(("admit", rid))
        if rid == self._fail_on:
            return FullAdmitOutcome(
                AdmitOutcome(ok=False, reason=_alloc_failed(rid)), "kv",
            )
        return FullAdmitOutcome(AdmitOutcome(ok=True))

    def rollback_admit(self, step):
        self.events.append(("rollback", step.ctx.request_ids[0]))

    # --- engine internals the path calls into
    def _drive_step(
        self, batch, submodule_mgmt, request_ids, inputs, req_info, ctx,
        lease, running_batched, step, set_launch,
    ):
        del batch, submodule_mgmt, inputs, req_info, lease, running_batched
        rid = request_ids[0]
        assert step is not None, "the step admitted for this rid must reach it"
        assert tuple(ctx.request_ids) == (rid,), "each rid drives its own ctx"
        if set_launch:
            # stands in for `_forward` releasing `launch_started_event`
            self.launch_signals.append(rid)
            self.events.append(("launch", rid))
        self.events.append(("run", rid))
        self.run_slots.append(ctx.slot)
        return {rid: {"token": 1}}, step

    def _collect_outputs(self, *a, **kw):
        del a
        return dict(kw["request_ids"] and {kw["request_ids"][0]: {"token": 1}})


def _exec_batch(rids, slot=0):
    return ExecutingBatch(
        node_name="node",
        per_request_info={rid: object() for rid in rids},
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk="walk", slot=slot, capture=False,
        ),
        inputs=[object() for _ in rids],
        slot=slot,
    )


def test_cpu_step_sync_does_not_access_gpu_streams(monkeypatch):
    from mstar.engine import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_ENGINE_STEP_SYNC", True)
    engine = _FakeExecEngine()
    out = engine._exec_per_request(_exec_batch(["a"]))
    assert out.per_rid_outputs == {"a": {"token": 1}}


def test_per_request_admits_every_rid_before_running_any():
    engine = _FakeExecEngine()

    out = engine._exec_per_request(_exec_batch(["a", "b"]))

    assert [e for e in engine.events if e[0] != "declare"] == [
        # the launch signal lands after BOTH admits, not after its own rid's.
        # ``Worker._await_admit_settled`` fences on it and then reads
        # ``admit_error``, so anything that could still refuse has to have run.
        ("admit", "a"), ("admit", "b"), ("launch", "a"), ("run", "a"),
        ("run", "b"),
    ]
    assert set(out.per_rid_outputs) == {"a", "b"}


def test_per_request_runs_nothing_when_a_later_rid_fails_admit():
    """The reviewer's bug: `a` used to run and commit, then `b` failed admit
    and the whole batch was re-queued — replaying a step whose resource state
    had already moved."""
    engine = _FakeExecEngine(fail_on="b")
    batch = _exec_batch(["a", "b"])

    out = engine._exec_per_request(batch)

    assert ("run", "a") not in engine.events, "a ran despite the batch failing"
    assert not any(kind == "run" for kind, _ in engine.events)
    assert out.per_rid_outputs == {"a": {}, "b": {}}
    assert isinstance(batch.admit_error, AllocationFailed)


def test_per_request_unwinds_the_rids_admitted_before_a_refusal():
    """All or nothing, like a batched admit: ``c``'s own refusal is unwound by
    the runner, and ``a``'s and ``b``'s admits have to be given back too,
    latest first."""
    engine = _FakeExecEngine(fail_on="c")

    engine._exec_per_request(_exec_batch(["a", "b", "c"]))

    assert [e for e in engine.events if e[0] != "declare"] == [
        ("admit", "a"), ("admit", "b"), ("admit", "c"),
        ("rollback", "b"), ("rollback", "a"),
    ]


def test_a_refused_batch_never_signals_the_launch():
    """``launch_started_event`` is the worker's admit fence: it waits on that,
    then reads ``admit_error`` to decide whether to speculate on N+1. So the
    error has to be written before anything signals.

    Here nothing signals at all — the refusal returns from the admit loop before
    the drive loop is reached, and ``_execute_on_gpu_thread``'s ``finally`` does
    it instead, well after ``register_admit_error``. If a future change signals
    from inside the admit loop, the worker reads ``admit_error=None``, builds and
    broadcasts a yield-away batch for N+1, and then evicts underneath it — the
    asymmetry that guard exists to prevent.
    """
    engine = _FakeExecEngine(fail_on="b")
    batch = _exec_batch(["a", "b"])

    engine._exec_per_request(batch)

    assert isinstance(batch.admit_error, AllocationFailed)
    assert engine.launch_signals == []


def test_per_request_rotates_slots_within_the_batch():
    """Each rid gets its own slot so one's staging can't land on top of the
    previous one's queued H2D. The slots come off the shared counter — these
    sub-steps are steps as far as the rotation is concerned."""
    # the batch leased slot 1, so the counter sits at 0
    engine = _FakeExecEngine(num_slots=2, next_slot=0)

    engine._exec_per_request(_exec_batch(["a", "b", "c"], slot=1))

    assert engine.run_slots == [1, 0, 1]
    # declare and drive are separate loops, so the region runners are pointed
    # at the slot in both
    assert engine.piecewise_slots == [1, 0, 1, 1, 0, 1]


def test_per_request_leaves_the_counter_one_past_its_last_slot():
    """What lets the fence drain a single slot: the batch hands the counter on
    exactly as a single-forward step would, so only `next_slot` can collide."""
    engine = _FakeExecEngine(num_slots=4, next_slot=1)

    engine._exec_per_request(_exec_batch(["a", "b", "c"], slot=0))

    assert engine.run_slots == [0, 1, 2]
    assert engine.mgmt.next_slot == 3


def test_per_request_keeps_one_slot_when_the_node_is_single_buffered():
    engine = _FakeExecEngine(num_slots=1)

    engine._exec_per_request(_exec_batch(["a", "b", "c"]))

    assert engine.run_slots == [0, 0, 0]
    assert engine.mgmt.next_slot == 0


def test_a_batch_too_big_alone_that_evicts_its_own_row_is_capped():
    """Sixteen running rows the pool holds ten of, with more waiting. Each
    refusal offloaded one row of the batch and the queue refilled the retry
    to sixteen, so every pass took another running row off the GPU."""
    alone = 10
    sched = _scheduler(_SchedEngine(max_bs=16))
    rids_all = [f"r{i}" for i in range(48)]
    manager = _Manager(rids_all)
    worker = _EvictingWorker(holders=rids_all)
    worker.scheduler = sched
    worker._graph_runtime = manager.runtime
    sizes = []
    for _ in range(8):
        batch = _next_batch(sched, manager)
        rids = list(batch.request_to_worker_graph)
        sizes.append(len(rids))
        if len(rids) <= alone:
            break
        worker._handle_allocation_failure(batch, SimpleNamespace(
            node_name=SCHED_NODE, failed_resource="kv",
            admit_error=_sized_refusal(rids[alone], rows_fit=alone, rows_fit_alone=alone),
        ))

    assert sizes == [16, alone], f"batch sizes {sizes}, {len(worker.offloaded)} rows offloaded"
    assert len(worker.offloaded) == 1
