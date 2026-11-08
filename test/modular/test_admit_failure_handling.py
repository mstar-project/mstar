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
import threading
from functools import partial
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_micro_scheduler import MIXED, _combined_scheduler, _Engine, _Manager, _next_batch, _scheduler
from test_micro_scheduler import NODE as SCHED_NODE
from test_micro_scheduler import WALK as SCHED_WALK
from test_micro_scheduler import _batch as _sched_batch
from test_micro_scheduler import _Runtime as _SchedRuntime

from mstar.engine.engine import Engine, ExecutingBatch, SubmoduleManagement
from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    FullAdmitOutcome,
    RequestOffloading,
    Residency,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import NodeSubmodule, StepOrder
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.worker import bound, make_worker
from mstar.utils.device import DeviceCapability
from mstar.worker import admission as admission_mod
from mstar.worker.admission import _HOLD_LOG_INTERVAL
from mstar.worker.micro_scheduler import ScheduledBatch


def _alloc_failed(rid: str = "r0") -> AllocationFailed:
    return AllocationFailed(
        message="out of pages", pages_short=4, label="main", request_id=rid,
    )


def _offloading(rid: str = "r0") -> RequestOffloading:
    return RequestOffloading(
        message="being offloaded", label="main", request_id=rid,
    )


# --- engine: the tail must not run over a step that never ran --------------


class _FakeEngine(Engine):
    """Just enough engine to exercise the ``exec``/``postprocess`` branch."""

    def __init__(self, admit_error):
        super().__init__(graph_runtime=None)
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


class _Runtime(_SchedRuntime):
    """A graph with nothing ready, whose push-backs are recorded."""

    def __init__(self):
        super().__init__(_Manager([]))
        self.pushed_back: list[str] = []

    def push_back_node(self, node_name, rids, wg_ids):
        del node_name, wg_ids
        self.pushed_back.extend(rids)


class _SchedEngine(_Engine):
    """The scheduler's engine fake, with the offload state the refill check and
    the eviction read. ``holders`` are the requests with pages to reclaim;
    ``offloaded`` lists what was evicted, in order."""

    def __init__(self, *args, offloaded=(), holders=(), dropped=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.offloaded = list(offloaded)
        self.holders = set(holders)
        self.dropped = set(dropped)

    def evictable(self, node_name, resources=None):
        del node_name, resources
        return True

    def residency(self, node_name, rid):
        del node_name
        if rid in self.dropped:
            return Residency.DROPPED
        return Residency.ON_HOST if rid in self.offloaded else Residency.RESIDENT

    def reclaimable(self, node_name, rid, resources):
        del node_name, resources
        return rid in self.holders

    def offload_request(self, node_name, rid, affected_resources=None):
        del node_name, affected_resources
        self.offloaded.append(rid)
        return 12


def _refusal_worker(sched=None, runtime=None, **overrides):
    """A real worker over a real scheduler stack, for what follows a refused
    admit. With no ``holders`` on its engine there is nothing to evict."""
    sched = sched or _scheduler(_SchedEngine())
    runtime = runtime or _Runtime()
    bound(sched, runtime=runtime)
    return make_worker(
        scheduler=sched, admission=sched.admission, lockstep=sched.lockstep,
        engine_manager=sched.engine_manager, graph_runtime=runtime, **overrides,
    )


def _evicting_worker(holders, sched=None, runtime=None):
    """``_refusal_worker`` over requests ``holders`` that hold pages and ran a
    step each in that order."""
    worker = _refusal_worker(sched, runtime)
    _engine(worker).holders.update(holders)
    for rid in holders:
        worker.scheduler.lifecycle.add(rid, str(rid))
        worker.scheduler.note_ran(_sched_batch([rid]))
    return worker


def _engine(worker) -> _SchedEngine:
    return worker.engine_manager.get_engine(SCHED_NODE)


def _batches(reason, prepared=("r0", "r1"), cached=()):
    batch = ScheduledBatch(
        node_name="node", graph_walk="walk",
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    node_batch = SimpleNamespace(
        node_name="node", admit_error=reason, request_ids=list(prepared), cached_rids=set(cached),
        failed_resource=None,
    )
    return batch, node_batch


def _refused(batch, reason, failed_resource="kv"):
    """What a step of all of ``batch``'s rows comes back as when admit refuses it."""
    return SimpleNamespace(
        node_name=batch.node_name, admit_error=reason, failed_resource=failed_resource,
        request_ids=list(batch.request_to_worker_graph), cached_rids=set(),
    )


def test_offloading_requeues_without_evicting():
    worker = _evicting_worker(["holder"])
    worker.pipeline._handle_admit_failure(*_batches(_offloading()))

    assert sorted(worker._graph_runtime.pushed_back) == ["r0", "r1"]
    # nothing to evict: the rid is already on its way to the host, and
    # `check_ready` gates the retry on reloading it
    assert _engine(worker).offloaded == []
    assert worker.admission.held_until == {}


def test_allocation_failure_still_evicts():
    worker = _evicting_worker(["holder"])
    spared = []
    worker.pipeline._teardown = spared.append
    worker.pipeline._handle_admit_failure(
        *_batches(_alloc_failed()), referenced_rids=frozenset({"spec_rid"}),
    )

    # the set has to reach the teardown: it is what the teardown spares
    assert spared == [frozenset({"spec_rid"})]
    assert _engine(worker).offloaded == ["holder"]
    # exactly one push-back per request: the delegation must not double up
    assert sorted(worker._graph_runtime.pushed_back) == ["r0", "r1"]
    assert set(worker.admission.held_until) == {"holder"}


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
    worker = _refusal_worker(runtime=_StoppedLoopRuntime({"r1"}))
    batch, node_batch = _batches(_alloc_failed())
    pending = SimpleNamespace(batch=batch, speculative_new_iter=new_iter, loop_name="loop")

    worker.pipeline._drop_rows_whose_loop_stopped(pending)
    worker.pipeline._handle_admit_failure(batch, node_batch)

    assert sorted(worker._graph_runtime.pushed_back) == (["r0"] if new_iter else ["r0", "r1"])


def _hold(worker, monkeypatch, at: float, walk: str = "walk") -> None:
    monkeypatch.setattr(admission_mod, "time", SimpleNamespace(monotonic=lambda: at))
    batch, node_batch = _batches(_alloc_failed())
    batch.graph_walk = walk
    worker.pipeline._handle_admit_failure(batch, node_batch)


def _hold_lines(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if "no offload possible" in record.getMessage()
    ]


def test_a_hundred_holds_on_one_node_log_two_lines_and_count_the_rest(monkeypatch, caplog):
    worker = _refusal_worker()
    step = _HOLD_LOG_INTERVAL / 50

    with caplog.at_level(logging.WARNING, logger=admission_mod.__name__):
        # a retry every fiftieth of the interval, across two intervals
        for tick in range(100):
            _hold(worker, monkeypatch, tick * step)

    lines = _hold_lines(caplog)
    assert len(lines) == 2, f"a hundred holds logged {len(lines)} lines, not one per interval"
    assert "(0 earlier holds not logged)" in lines[0], lines[0]
    assert "(49 earlier holds not logged)" in lines[1], (
        f"the second line lost count of the holds it stands for: {lines[1]}"
    )


@pytest.mark.parametrize(
    "reason", [_alloc_failed("r0"), _offloading("r0")], ids=["allocation", "offloading"]
)
def test_a_rid_prepare_dropped_is_not_requeued(reason):
    """r1 finished its loop; the speculated extra step reached prepare_inputs,
    which vetoed it, then r0's admit refused the step. Re-queued, r1 came back
    with its loop-back inputs spent and the Cosmos3 DiT read that as a fresh
    loop start over its spent scheduler (UniPC ``this_order > 0``, every
    request in the next batch failed). A step that runs drops r1 the same way."""
    worker = _refusal_worker()
    batch, node_batch = _batches(reason, prepared=["r0"])

    worker.pipeline._handle_admit_failure(batch, node_batch)

    assert worker._graph_runtime.pushed_back == ["r0"]
    assert "r1" not in worker.admission.held_until


@pytest.mark.parametrize(
    "reason", [_alloc_failed("r0"), _offloading("r0")], ids=["allocation", "offloading"]
)
def test_a_rid_the_cache_served_is_requeued(reason):
    worker = _refusal_worker()
    batch, node_batch = _batches(reason, prepared=["r0"], cached=["r1"])

    worker.pipeline._handle_admit_failure(batch, node_batch)

    assert sorted(worker._graph_runtime.pushed_back) == ["r0", "r1"], (
        "a row the cache served left the refused step, so its walk never completes"
    )


def test_a_hold_on_another_walk_is_logged_inside_the_interval(monkeypatch, caplog):
    worker = _refusal_worker()

    with caplog.at_level(logging.WARNING, logger=admission_mod.__name__):
        _hold(worker, monkeypatch, 0.0, walk="prefill")
        _hold(worker, monkeypatch, 0.1, walk="decode")

    assert len(_hold_lines(caplog)) == 2, (
        "one walk's hold silenced another's, so a stall on it would go unseen"
    )


# --- a refused batch with nothing to evict is retried smaller ---------------


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
    worker = _refusal_worker(sched, manager.runtime)
    refused = []
    for _ in range(passes):
        batch = _next_batch(sched, manager)
        rids = list(batch.request_to_worker_graph)
        if len(rids) <= fits:
            return sched, refused, rids
        refused.append(len(rids))
        worker.pipeline._handle_admit_failure(
            batch, _refused(batch, _sized_refusal(rids[fits], fits if sized else None)),
        )
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

    sched.admission.clear_batch_limit(SCHED_NODE, SCHED_WALK, len(admitted))

    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 16


def test_the_limit_never_drops_below_one_row():
    sched = _scheduler(_SchedEngine(max_bs=4))
    sched.admission.limit_batch(SCHED_NODE, SCHED_WALK, 0)
    sched.admission.cap_batch(SCHED_NODE, SCHED_WALK, 0)

    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 1


def test_a_refused_batch_below_its_cap_is_not_limited():
    """Nine running requests decode in a step of nine, the cap is 32, and the
    admit refuses with nothing to evict. Every ready row is already in the
    batch, so holding the blamed one is what makes the retry smaller. A limit
    of eight on top kept a running row out of the steps after the hold, and
    a row outside the batch is the one the next eviction takes first."""
    sched = _scheduler(_SchedEngine(max_bs=32))
    manager = _Manager([f"d{i}" for i in range(9)])
    worker = _refusal_worker(sched, manager.runtime)
    batch = _next_batch(sched, manager)
    rids = list(batch.request_to_worker_graph)
    assert len(rids) == 9

    worker.pipeline._handle_admit_failure(
        batch, _refused(batch, _sized_refusal(rids[6], rows_fit=6, rows_fit_alone=9)),
    )
    sched.admission.held_until.clear()  # the backoff ran out

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
    worker = _refusal_worker(sched, manager.runtime)
    batch = _next_batch(sched, manager)
    rids = list(batch.request_to_worker_graph)
    assert batch.graph_walk == MIXED and len(rids) == 2

    worker.pipeline._handle_admit_failure(
        batch, _refused(batch, _sized_refusal(rids[1], rows_fit=1, rows_fit_alone=2)),
    )

    assert len(_next_batch(sched, manager).request_to_worker_graph) == 1


def _victim(worker, batch_ids, walk=SCHED_WALK):
    """Who a refused step of ``walk`` evicts."""
    scheduler = worker.scheduler
    return worker.admission._evict(
        SCHED_NODE, batch_ids, scheduler.lifecycle.arrivals, partial(scheduler.last_in_line, SCHED_NODE, walk),
    )


def test_a_batch_the_pool_cannot_hold_alone_is_capped_not_evicted_for():
    """Sixteen prefills of 12 pages need 192 and the pool has 191. Four
    decoding requests hold pages, and on each refusal one was offloaded and
    the batch retried at sixteen: it can never fit, and the victim reloads
    into the pages it freed. Nothing outside the batch is evicted now, and
    the walk stays capped at fifteen after a step of fifteen admits."""
    usable, row = 191, 12
    sched = _scheduler(_SchedEngine(max_bs=16))
    manager = _Manager([f"p{i}" for i in range(48)])
    worker = _evicting_worker([f"d{i}" for i in range(4)], sched, manager.runtime)
    sizes = []
    evicted = []
    for _ in range(8):
        batch = _next_batch(sched, manager)
        rids = list(batch.request_to_worker_graph)
        sizes.append(len(rids))
        # an offloaded decoder reloads by the next pass, so its pages are back
        free = usable - row * 4
        if row * len(rids) <= free:
            break
        worker.pipeline._handle_admit_failure(batch, _refused(batch, _sized_refusal(
            rids[free // row], rows_fit=free // row, rows_fit_alone=usable // row,
        )))
        evicted += _engine(worker).offloaded
        _engine(worker).offloaded.clear()
    assert evicted == [] and sizes == [16, 11], (
        f"batch sizes {sizes}: the full batch was retried after evicting a decoder"
    )

    sched.admission.clear_batch_limit(SCHED_NODE, SCHED_WALK, 11)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 15
    sched.admission.clear_batch_limit(SCHED_NODE, SCHED_WALK, 15)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 15, (
        "a step at the cap lifted it, so the next full batch is refused again"
    )
    sched.admission.clear_batch_limit(SCHED_NODE, SCHED_WALK, 3)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 16


def test_a_batch_too_big_alone_still_evicts_its_own_rows():
    """Running requests that outgrow the pool together: evicting one of
    them is what lets the rest step, as before."""
    # the request outside the batch arrived first, and it isn't asked about
    worker = _evicting_worker(["outside", "r0", "r1"])
    batch, node_batch = _batches(_sized_refusal("r1", rows_fit=1, rows_fit_alone=1))
    node_batch.failed_resource = "kv"
    batch.node_name = node_batch.node_name = SCHED_NODE

    worker.pipeline._handle_admit_failure(batch, node_batch)

    assert _engine(worker).offloaded == ["r1"], "an outside request was evicted for a batch it cannot help"


@pytest.mark.parametrize("order, victim", [(StepOrder.TURNS, "r3"), (StepOrder.ARRIVAL, "r2")])
def test_a_refused_step_offloads_the_row_last_in_its_line(order, victim):
    """A step of r4 is refused, and r0 to r3 hold pages, r0 having run first.
    While rows take turns r0 runs right after the retry, so offloading it only
    makes it reload, and r3's next turn is the furthest away. Under ARRIVAL r3
    holds the step, then r0 to r2 by arrival."""
    worker = _evicting_worker([f"r{i}" for i in range(4)], _scheduler(_SchedEngine(order=order)))
    batch = _sched_batch(["r4"])

    worker.pipeline._handle_admit_failure(batch, _refused(batch, _alloc_failed("r4")))

    assert _engine(worker).offloaded == [victim]


@pytest.mark.parametrize("order, victim", [(StepOrder.TURNS, "ran"), (StepOrder.ARRIVAL, "never_ran")])
def test_a_row_that_never_ran_at_the_key_takes_its_place_in_line(order, victim):
    """Its pages came some other way (a KV retrieve from another worker):
    under TURNS it is first in line, and under ARRIVAL behind the row holding
    the step."""
    worker = _evicting_worker(["ran"], _scheduler(_SchedEngine(order=order)))
    worker.scheduler.lifecycle.add("never_ran", "never_ran")
    _engine(worker).holders.add("never_ran")

    assert _victim(worker, {"r9"}) == victim


def test_a_refused_step_never_offloads_a_request_mid_rebuild():
    """``mid`` is last in line and holds the pages its rebuild has written so
    far, but those are the rebuild's to finish."""
    worker = _evicting_worker(["r0", "mid"], _scheduler(_SchedEngine(order=StepOrder.TURNS, dropped={"mid"})))

    assert _victim(worker, {"r9"}) == "r0"


def _decode_waiters(order, ran):
    """The worker's eviction over ``ran``'s steps: (rid, walk), oldest first,
    with every rid holding pages. Prefill and decode are separate keys here."""
    worker = _evicting_worker([], _scheduler(_SchedEngine(order=order)))
    holders = list(dict.fromkeys(rid for rid, _ in ran))
    for rid in holders:
        worker.scheduler.lifecycle.add(rid, str(rid))
    for rid, walk in ran:
        worker.scheduler.note_ran(_sched_batch([rid], walk=walk))
    _engine(worker).holders.update(holders)
    return worker


def test_under_arrival_the_row_that_prefilled_first_is_next_and_not_the_victim():
    """Two prefilled rows wait for decode. p0's stamp is the oldest here, but
    it is next in line for decode; p1 waits behind it."""
    worker = _decode_waiters(StepOrder.ARRIVAL, [("p0", "prefill"), ("p1", "prefill")])

    assert _victim(worker, {"r9"}) == "p1"


def test_under_turns_a_row_new_to_decode_is_first_in_its_line_and_not_the_victim():
    """p prefilled on its own walk after d0 and d1 decoded, so it has the
    newest stamp at the node, but it has yet to decode: it leads the decode
    line, and d1, which decoded last, is furthest from another step."""
    worker = _decode_waiters(
        StepOrder.TURNS, [("d0", SCHED_WALK), ("d1", SCHED_WALK), ("p", "prefill")],
    )

    assert _victim(worker, {"r9"}) == "d1"


def test_a_row_inside_the_batch_goes_only_when_none_is_outside_it_under_turns():
    worker = _evicting_worker(["outside", "r0", "r1"], _scheduler(_SchedEngine(order=StepOrder.TURNS)))

    assert _victim(worker, {"r0", "r1"}) == "outside"
    assert _victim(worker, {"r0", "r1"}) == "r1"


def test_a_step_of_one_walk_lifts_its_combined_walks_limit():
    """A mixed batch limited to one row: every step under it holds one walk
    and carries that walk's label. The first of them to admit has to lift
    the limit, or the walks never batch together again."""
    sched = _combined_scheduler(_SchedEngine(max_bs=8))
    sched.admission.limit_batch(SCHED_NODE, MIXED, 1)

    limited = _next_batch(sched, _Manager(["p0", "d0"], walks={"p0": "prefill"}))
    assert limited.graph_walk != MIXED and len(limited.request_to_worker_graph) == 1

    sched.admission.clear_batch_limit(SCHED_NODE, limited.graph_walk, 1)

    mixed = _next_batch(sched, _Manager(["p1", "d1"], walks={"p1": "prefill"}))
    assert mixed.graph_walk == MIXED, "the combined walk stayed limited to one row"


def test_a_mixed_step_lifts_the_limits_of_the_walks_it_ran():
    """A nine-row decode refused to two while prefills wait, then a mixed step
    of twelve, sized by the combined walk's own cap, admits. It ran decode
    rows, so the next pure-decode step is not held to two rows."""
    sched = _combined_scheduler(_SchedEngine(max_bs=16))
    sched.admission.limit_batch(SCHED_NODE, SCHED_WALK, 2)

    mixed = _next_batch(sched, _Manager(
        [f"p{i}" for i in range(6)] + [f"d{i}" for i in range(6)],
        walks={f"p{i}": "prefill" for i in range(6)},
    ))
    assert mixed.graph_walk == MIXED and len(mixed.request_to_worker_graph) == 12

    sched.admission.clear_batch_limit(
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
    being removed, offloaded, dropped or of another walk. ``pending_removes``
    is the worker's dict: a set union over it raised TypeError."""
    sched = _scheduler(_SchedEngine(max_bs=8, offloaded={"off"}, dropped={"drop"}))
    manager = _Manager(
        ["b0", "b1", "w0", "failed", "held", "gone", "off", "drop", "p0"], walks={"p0": "prefill"},
    )
    bound(sched, runtime=manager.runtime)
    manager.runtime.combined_walk_of = sched.combined_walk_of
    sched.lifecycle.failed_rids.add("failed")
    sched.admission.hold(["held"])
    sched.lifecycle.pending_removes = {"gone": -1}

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


class _FakeMgmt(SubmoduleManagement):
    """``SubmoduleManagement``'s slot rotation from a chosen slot, recording what the forward was pointed at."""

    # no idle-capable resource
    idle_keys = ()
    rows_idle = SubmoduleManagement.rows_idle

    def __init__(
        self, submodule, num_slots: int, next_slot: int, seen: list,
        slot_raise_at: int | None = None,
    ):
        super().__init__(submodule=submodule, forward=None, forward_batched=None, joint_comm_group=None, resources={})
        self._num_slots = num_slots
        self._next_slot = next_slot
        self._seen = seen
        self._slot_raise_at = slot_raise_at

    def set_piecewise_slot(self, slot: int) -> None:
        self._seen.append(slot)
        if len(self._seen) == self._slot_raise_at:
            raise RuntimeError("slot setup failed")


class _FakeExecEngine(Engine):
    """Engine internals over stubs, to drive `_exec_per_request` alone."""

    def __init__(
        self, fail_on: str | None = None, num_slots: int = 1, next_slot: int = 0,
        raise_on: str | tuple[str, ...] = (), collect_raise_on: str | None = None,
        slot_raise_at: int | None = None,
    ):
        super().__init__(graph_runtime=None)
        self._fail_on = fail_on
        self._raise_on = {raise_on} if isinstance(raise_on, str) else set(raise_on)
        self._collect_raise_on = collect_raise_on
        # ordered log, so the test can assert admit-all-then-run
        self.events: list[tuple[str, str]] = []
        # which rids were driven with `set_launch` — the real path signals
        # `launch_started_event` there, and the worker fences on it
        self.launch_signals: list[str] = []
        self._runner = self
        self._device = torch.device("cpu")
        self._device_capability = DeviceCapability.of(self._device)
        # the per-request path's slots; no fence, so no CUDA event is recorded
        self.piecewise_slots: list[int] = []
        self.run_slots: list[int] = []
        self.mgmt = _FakeMgmt(
            self, num_slots, next_slot, self.piecewise_slots, slot_raise_at=slot_raise_at,
        )
        self._submodules["node"] = self.mgmt

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
        self.run_slots.append(ctx.slot)
        if rid in self._raise_on:
            self.events.append(("raise", rid))
            raise RuntimeError(f"forward failed for {rid}")
        self.events.append(("run", rid))
        return {rid: {"token": 1}}, step

    def _collect_outputs(self, *a, **kw):
        del a
        if kw["request_ids"][0] == self._collect_raise_on:
            raise RuntimeError("collect failed")
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
        # ``StepPipeline._await_admit_settled`` fences on it and then reads
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
    the drive loop is reached, and the step's completion does it instead
    (``StepExecutor._submit_gpu_step``), well after ``register_admit_error``. If a
    future change signals from inside the admit loop, the worker reads
    ``admit_error=None``, builds and broadcasts a yield-away batch for N+1, and
    then evicts underneath it — the asymmetry that guard exists to prevent.
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
    worker = _evicting_worker(rids_all, sched, manager.runtime)
    sizes = []
    for _ in range(8):
        batch = _next_batch(sched, manager)
        rids = list(batch.request_to_worker_graph)
        sizes.append(len(rids))
        if len(rids) <= alone:
            break
        worker.pipeline._handle_admit_failure(
            batch, _refused(batch, _sized_refusal(rids[alone], rows_fit=alone, rows_fit_alone=alone)),
        )

    offloaded = _engine(worker).offloaded
    assert sizes == [16, alone], f"batch sizes {sizes}, {len(offloaded)} rows offloaded"
    assert len(offloaded) == 1


# --- a forward that raises fails only its own rid --------------------------


@pytest.mark.parametrize("engine_kwargs, failed", [
    ({"collect_raise_on": "b"}, {"b"}),  # past commit, collecting its outputs
    ({"raise_on": ("a", "b", "c")}, {"a", "b", "c"}),
], ids=["collect", "every_forward"])
def test_per_request_fails_only_the_rids_that_raised(engine_kwargs, failed):
    """The rest of the unbatchable batch still runs, and the plan thread is released."""
    engine = _FakeExecEngine(**engine_kwargs)
    batch = _exec_batch(["a", "b", "c"])

    out = engine._exec_per_request(batch)

    assert out.per_rid_outputs == {rid: {} if rid in failed else {"token": 1} for rid in "abc"}
    assert set(batch.failed_requests) == failed
    assert batch.commit_done.is_set()


class _FakeEvent:
    """A slot fence event that logs which slot it was recorded after and
    when it is waited on."""

    def __init__(self, engine):
        self._engine = engine
        self.slot = None

    def record(self):
        self.slot = self._engine.piecewise_slots[-1]
        self._engine.events.append(("record", self.slot))

    def synchronize(self):
        self._engine.events.append(("sync", self.slot))


class _LoggedEvent(threading.Event):
    def __init__(self, events):
        super().__init__()
        self._events = events

    def set(self):
        self._events.append(("commit_done", ""))
        super().set()


def _fenced(monkeypatch, **kwargs) -> _FakeExecEngine:
    engine = _FakeExecEngine(**kwargs)
    engine.mgmt._forced_double_buffer = True
    engine._device_capability = SimpleNamespace(
        is_accelerator=True, is_capturing=lambda: False,
    )
    monkeypatch.setattr(torch, "Event", lambda device=None: _FakeEvent(engine))
    return engine


def test_a_raising_rids_slot_is_still_fenced(monkeypatch):
    """A raising rid's slot is fenced like a finished rid's: d reuses b's slot
    only after waiting on it, and ``next_slot`` drains before the plan thread
    is released."""
    engine = _fenced(monkeypatch, raise_on="b", num_slots=2, next_slot=1)
    batch = _exec_batch(["a", "b", "c", "d"], slot=0)
    batch.commit_done = _LoggedEvent(engine.events)

    engine._exec_per_request(batch)

    assert engine.run_slots == [0, 1, 0, 1]
    assert [e for e in engine.events if e[0] in ("run", "raise", "record", "sync", "commit_done")] == [
        ("run", "a"), ("record", 0),
        ("raise", "b"), ("record", 1),
        ("sync", 0), ("run", "c"), ("record", 0),
        ("sync", 1), ("run", "d"), ("record", 1),
        # the counter ends on slot 0, c's
        ("sync", 0), ("commit_done", ""),
    ]
    assert set(batch.failed_requests) == {"b"}


def test_per_request_drains_and_releases_when_the_loop_itself_raises(monkeypatch):
    """A raise outside the per-rid try (setting b's slot) fails the batch, and
    still drains ``next_slot`` (a's) before releasing the plan thread."""
    # two declare-loop calls, then a's slot, then b's raises
    engine = _fenced(monkeypatch, num_slots=2, next_slot=1, slot_raise_at=4)
    batch = _exec_batch(["a", "b"], slot=0)
    batch.commit_done = _LoggedEvent(engine.events)

    with pytest.raises(RuntimeError, match="slot setup failed"):
        engine._exec_per_request(batch)

    assert [e for e in engine.events if e[0] in ("run", "record", "sync", "commit_done")] == [
        ("run", "a"), ("record", 0), ("sync", 0), ("commit_done", ""),
    ]
    assert not batch.failed_requests  # the worker fails the whole batch


class _TailSubmodule:
    batch_postprocess = NodeSubmodule.batch_postprocess

    def __init__(self):
        self.postprocessed: list[str] = []

    def postprocess(self, request_id, request_info, outputs, inputs):
        del request_info, inputs
        self.postprocessed.append(request_id)
        outputs["token"]  # an empty output can't be finished


def test_postprocess_skips_a_rid_whose_forward_raised():
    """The tail must not run over a failed rid's empty outputs: it would raise
    again and replace the forward's error with its own."""
    submodule = _TailSubmodule()
    engine = make_engine(_submodules={"node": make_mgmt(submodule=submodule)})
    batch = _exec_batch(["a", "b"])
    batch.register_failure("b", ValueError("forward failed for b"))

    engine._postprocess_batch(batch, {"a": {"token": 1}, "b": {}})

    assert submodule.postprocessed == ["a"]
    assert batch.failed_requests == {"b": "ValueError: forward failed for b"}
