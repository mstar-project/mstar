"""A refused KV admit says how many of the step's requests fit.

The worker sizes the retry from it: ``rows_fit`` is what the free pages held,
``rows_fit_alone`` what the pool would hold with nothing else resident. A
request that alone needs more pages than the pool has is refused for good,
since no eviction makes it fit and retrying it only spins until the client
gives up.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_admit_failure_handling import (
    _NO_LIMITS,
    SCHED_NODE,
    SCHED_WALK,
    _EvictingWorker,
    _SchedEngine,
)
from test_micro_scheduler import _Manager, _next_batch, _scheduler
from test_worker_drain import _worker as _teardown_worker

from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import AdmitRuntimeError, AllocationFailed, Segment, StepContext
from mstar.utils.ipc_format import RemoveRequest
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

PAGE_SIZE = 16


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            max_seq_len=4096,
            max_num_pages=max_num_pages,
            page_size=PAGE_SIZE,
            cpu_offload_pages=0,
        ),
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _admit(mgr: KVManager, spans: dict[str, int], commit: bool = False):
    for rid in spans:
        if rid not in mgr._streams:
            mgr.ingest_request(rid)
    step = KVStep(segments=tuple(Segment(rid, "main", span) for rid, span in spans.items()))
    ctx = _ctx(*spans)
    outcome = mgr.admit(step, ctx)
    if outcome.ok and commit:
        mgr.plan(step, ctx)
        mgr.commit(step, ctx)
    elif not outcome.ok:
        mgr.rollback_admit(step, ctx)
    return outcome


def test_a_refusal_counts_the_rows_that_fit_now_and_alone():
    """15 usable pages. A resident request holds 6; five new ones need 3 each.
    The free 9 hold three of them, the pool alone five."""
    mgr = _manager(16)
    assert _admit(mgr, {"old": 6 * PAGE_SIZE}, commit=True).ok

    outcome = _admit(mgr, {f"n{i}": 3 * PAGE_SIZE for i in range(5)})

    assert not outcome.ok
    reason = outcome.reason
    assert isinstance(reason, AllocationFailed) and reason.request_id == "n3"
    assert (reason.rows_fit, reason.rows_fit_alone) == (3, 5)


def test_a_batch_bigger_than_the_pool_says_so():
    """The whisper case: 16 rows of 12 pages need 192, 191 are usable."""
    mgr = _manager(192)

    outcome = _admit(mgr, {f"p{i}": 12 * PAGE_SIZE for i in range(16)})

    assert not outcome.ok
    assert outcome.reason.rows_fit == 15 and outcome.reason.rows_fit_alone == 15


def test_a_request_bigger_than_the_pool_is_refused_for_good():
    """Its retry was refused about 19 times a second until the client gave up."""
    mgr = _manager(8)

    outcome = _admit(mgr, {"huge": 8 * PAGE_SIZE})

    assert not outcome.ok
    assert isinstance(outcome.reason, AdmitRuntimeError) and outcome.reason.terminal
    assert outcome.reason.request_ids == frozenset({"huge"})
    # the client reads this text, and the engine's handle is not an id it sent
    assert "huge" not in outcome.reason.message
    assert "needs 8 pages" in outcome.reason.message


def test_a_request_that_fits_the_pool_but_not_the_free_pages_is_retried():
    mgr = _manager(8)
    assert _admit(mgr, {"old": 4 * PAGE_SIZE}, commit=True).ok

    outcome = _admit(mgr, {"big": 7 * PAGE_SIZE})

    assert isinstance(outcome.reason, AllocationFailed)
    assert (outcome.reason.rows_fit, outcome.reason.rows_fit_alone) == (0, 1)


def test_a_growing_request_counts_the_pages_it_holds():
    """A decode row past a page boundary needs one page more than it holds;
    its footprint is all of them, so one that outgrew the pool is refused
    for good rather than evicting everything else for nothing."""
    mgr = _manager(8)
    assert _admit(mgr, {"long": 7 * PAGE_SIZE}, commit=True).ok

    outcome = _admit(mgr, {"long": 1})

    assert isinstance(outcome.reason, AdmitRuntimeError)


class _FailingWorker(_EvictingWorker):
    """The real admit-failure handlers and victim search, failures recorded."""

    _handle_admit_failure = Worker._handle_admit_failure
    _handle_unservable = Worker._handle_unservable

    def __init__(self, holders):
        super().__init__(holders)
        self.held: list[str] = []
        self.failed: dict[str, str] = {}
        vars(self.scheduler).update(hold_requests=self.held.extend, **_NO_LIMITS)

    def _fail_requests(self, errors):
        self.failed.update(errors)


_POOL_FILLER_ERROR = (
    "Error in worker: The request needs 64 pages of kv for this step, more than "
    "the 63 the pool holds; raise max_num_pages or shorten the request."
)


def _refuse_after_a_pool_filling_prefill(spans: dict[str, int]) -> _FailingWorker:
    """64 pages, 63 usable: ``long``'s prefill takes all 63, then a step of
    ``spans`` is admitted and its refusal handled as the worker does."""
    mgr = _manager(64)
    assert _admit(mgr, {"long": 63 * PAGE_SIZE}, commit=True).ok
    assert mgr._arena.num_free == 0

    outcome = _admit(mgr, spans)
    assert not outcome.ok

    worker = _FailingWorker(holders=["long"])
    batch = ScheduledBatch(
        node_name=SCHED_NODE, graph_walk=SCHED_WALK,
        request_to_worker_graph=dict.fromkeys(spans, "wg"),
    )
    worker._handle_admit_failure(batch, SimpleNamespace(
        node_name=SCHED_NODE, admit_error=outcome.reason,
        failed_resource="kv", request_ids=list(spans), cached_rids=set(),
    ))
    return worker


def test_a_prefill_that_fills_the_pool_fails_at_its_first_decode():
    """The first decode needs a 64th page. The request was its own only
    victim: offloaded, reloaded and refused again every 54 ms until the
    client gave up at 180 s."""
    worker = _refuse_after_a_pool_filling_prefill({"long": 1})

    assert worker.offloaded == [] and worker.held == [], "the request was offloaded to retry"
    assert worker._graph_runtime.pushed_back == []
    assert worker.failed == {"long": _POOL_FILLER_ERROR}


@pytest.mark.parametrize("prompt_first", [True, False], ids=["prompt-first", "decode-first"])
def test_a_pool_filler_fails_in_a_mixed_step_whichever_row_is_refused(prompt_first):
    """The first decode shares a step with a new prompt. Listed first, the
    prompt is the row refused, and the pool filler, the only row holding
    pages, was offloaded to make room for it."""
    rows = [("short", 2 * PAGE_SIZE), ("long", 1)]
    worker = _refuse_after_a_pool_filling_prefill(dict(rows if prompt_first else rows[::-1]))

    assert worker.offloaded == [], "the pool filler was offloaded to retry"
    assert worker.failed == {"long": _POOL_FILLER_ERROR}
    assert worker._graph_runtime.pushed_back == ["short"]


def test_pages_two_labels_share_are_counted_once():
    """Seven usable pages. Two labels of one request share three pages and
    each needs one more: five pages, not eight, so with another request
    holding three the step is retried, not failed for good."""
    mgr = _manager(8)
    assert _admit(mgr, {"other": 3 * PAGE_SIZE}, commit=True).ok
    assert _admit(mgr, {"r": 3 * PAGE_SIZE}, commit=True).ok
    a = mgr._streams["r"]["main"]
    b = mgr._ensure_label("r", "fork")
    b.page_indices = list(a.page_indices)
    b.stored_len = a.stored_len
    mgr._arena.retain(b.page_indices)
    step = KVStep(segments=(Segment("r", "main", PAGE_SIZE), Segment("r", "fork", PAGE_SIZE)))

    outcome = mgr.admit(step, _ctx("r"))
    mgr.rollback_admit(step, _ctx("r"))

    assert isinstance(outcome.reason, AllocationFailed), outcome.reason
    assert (outcome.reason.rows_fit, outcome.reason.rows_fit_alone) == (0, 1)


def _admit_step(mgr: KVManager, step: KVStep, rids: tuple[str, ...]):
    for rid in rids:
        if rid not in mgr._streams:
            mgr.ingest_request(rid)
    outcome = mgr.admit(step, _ctx(*rids))
    if not outcome.ok:
        mgr.rollback_admit(step, _ctx(*rids))
    return outcome


def _unguided_step(spans: dict[str, int]) -> KVStep:
    """Bagel's CFG step for rows that are not guided: their ``cfg_text`` and
    ``cfg_img`` labels run against the sink and reserve nothing."""
    labels = ("main", "cfg_text", "cfg_img")
    return KVStep(
        segments=tuple(Segment(rid, label, span) for label in labels for rid, span in spans.items()),
        sink=frozenset((rid, label) for label in labels[1:] for rid in spans),
    )


def test_labels_on_the_sink_are_not_counted_in_a_footprint():
    """Fifteen usable pages, ten held. An unguided prompt of seven pages was
    counted at 21, its sink labels at seven each, and failed for good as
    bigger than the pool, though it fits once the other request goes."""
    mgr = _manager(16)
    assert _admit(mgr, {"old": 10 * PAGE_SIZE}, commit=True).ok

    outcome = _admit_step(mgr, _unguided_step({"r": 7 * PAGE_SIZE}), ("r",))

    assert isinstance(outcome.reason, AllocationFailed), outcome.reason
    reason = outcome.reason
    assert (reason.rows_fit, reason.rows_fit_alone, reason.rows_sized) == (0, 1, 1)


def test_labels_on_the_sink_do_not_shrink_the_rows_that_fit_alone():
    """Three unguided rows of four pages: twelve pages, which the 15-page
    pool holds. Counted at twelve each, one fit alone."""
    mgr = _manager(16)
    assert _admit(mgr, {"old": 8 * PAGE_SIZE}, commit=True).ok

    outcome = _admit_step(
        mgr, _unguided_step({f"u{i}": 4 * PAGE_SIZE for i in range(3)}), ("u0", "u1", "u2"),
    )

    reason = outcome.reason
    assert isinstance(reason, AllocationFailed), reason
    assert (reason.rows_fit, reason.rows_fit_alone, reason.rows_sized) == (1, 3, 3)


def test_a_capture_step_sizes_no_rows():
    """A capture's rows run against the sink, as `admit` treats them."""
    mgr = _manager(8)
    step = KVStep(segments=(Segment("r", "main", 2 * PAGE_SIZE),))
    ctx = StepContext(request_ids=("r",), graph_walk="w", slot=0, capture=True)
    error = AllocationFailed(message="", pages_short=1, label="main", request_id="r")

    with mgr._lock:
        reason = mgr._sized_refusal(step, ctx, error)

    assert (reason.rows_fit, reason.rows_fit_alone, reason.rows_sized) == (None, 0, 0)


def test_a_row_the_cache_served_does_not_make_a_batch_too_big_alone():
    """Sixteen prompts, one served whole by the prefix cache. The fifteen
    others need 60 of the 63 usable pages and fit once the decoding request
    holding ten goes. Compared with the batch's sixteen rows, the fifteen
    that fit alone read as too big: eviction was kept to the batch's rows,
    which hold nothing, so nothing was evicted and the walk was capped."""
    mgr = _manager(64)
    assert _admit(mgr, {"decoder": 10 * PAGE_SIZE}, commit=True).ok
    spans = {f"p{i}": 4 * PAGE_SIZE for i in range(15)}

    outcome = _admit(mgr, spans)

    reason = outcome.reason
    assert isinstance(reason, AllocationFailed), reason
    assert (reason.rows_fit_alone, reason.rows_sized) == (15, 15)
    worker = _FailingWorker(holders=["decoder"])
    batch = ScheduledBatch(
        node_name=SCHED_NODE, graph_walk=SCHED_WALK,
        request_to_worker_graph=dict.fromkeys([*spans, "cached"], "wg"),
    )
    worker._handle_admit_failure(batch, SimpleNamespace(
        node_name=SCHED_NODE, admit_error=reason, failed_resource="kv",
        request_ids=list(spans), cached_rids={"cached"},
    ))

    assert worker.offloaded == ["decoder"], "the request holding pages was not evicted"
    assert worker.failed == {}


# --- a teardown that hands pages back lifts a cap ---------------------------


def _engine_manager(**pools: KVManager) -> SimpleNamespace:
    """The worker's engine manager over the real ``Engine.remove_request``,
    with these KV pools as the engine's resources."""
    engine = SimpleNamespace(
        _loopback_steps={}, _resources=pools, _submodules={},
        _runner=SimpleNamespace(
            remove_request=lambda rid: [pool.remove_request(rid) for pool in pools.values()],
        ),
    )
    return SimpleNamespace(
        remove_request=lambda rid: Engine.remove_request(engine, rid),
    )


def _remove(worker, *rids: str) -> None:
    for rid in rids:
        Worker._remove_request(worker, RemoveRequest(request_id=rid))


def test_a_teardown_that_frees_pages_lifts_a_cap_a_busy_walk_never_steps_under():
    """Sixteen decoding rows of three full pages each need a fourth, and the
    63 usable pages hold fifteen of them alone, so the victim branch caps the
    walk at fifteen. With 48 rows decoding every step is at the cap, and only
    a smaller step lifted it, so it outlasted the long rows that set it."""
    pool = _manager(64)
    long_rows = [f"d{i}" for i in range(16)]
    for rid in long_rows:
        assert _admit(pool, {rid: 3 * PAGE_SIZE}, commit=True).ok
    outcome = _admit(pool, dict.fromkeys(long_rows, 1))
    assert isinstance(outcome.reason, AllocationFailed)
    assert outcome.reason.rows_fit_alone == 15

    sched = _scheduler(_SchedEngine(max_bs=16))
    manager = _Manager([*long_rows, *(f"s{i}" for i in range(32))])
    batch = _next_batch(sched, manager)
    assert list(batch.request_to_worker_graph) == long_rows
    refuser = _EvictingWorker(long_rows, sched)
    refuser._graph_runtime = manager.runtime
    refuser._handle_allocation_failure(batch, SimpleNamespace(
        node_name=SCHED_NODE, failed_resource="kv", admit_error=outcome.reason,
    ))
    assert refuser.offloaded == ["d14"]  # d15 ran last, so it holds the step

    for _ in range(4):
        step = list(_next_batch(sched, manager).request_to_worker_graph)
        assert len(step) == 15
        sched.clear_batch_limit(SCHED_NODE, SCHED_WALK, len(step))
        manager.in_flight.clear()
        for rid in step:
            manager.ingest(rid)
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 15

    worker = _teardown_worker(known_rids=long_rows)
    worker.scheduler, worker.engine_manager = sched, _engine_manager(kv=pool)
    free = pool._arena.num_free
    _remove(worker, "d1")
    manager.queues["wg0"]._ready.pop("d1", None)

    assert pool._arena.num_free == free + 3
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 16, (
        "the cap outlived a teardown that handed its pool's pages back"
    )
    assert len(_next_batch(sched, manager).request_to_worker_graph) == 16


def test_a_teardown_that_frees_none_of_the_capping_pool_leaves_the_cap():
    """The cross pool's refusal set the cap. A request holding pages of the
    self pool only, and one admitted that never ran, free none of it, so a
    full batch would be refused again."""
    self_kv, cross_kv = _manager(16), _manager(16)
    free = self_kv._arena.num_free
    assert _admit(self_kv, {"a": 2 * PAGE_SIZE}, commit=True).ok
    cross_kv.ingest_request("a")
    for pool in (self_kv, cross_kv):
        pool.ingest_request("b")
    sched = _scheduler(_SchedEngine(max_bs=16))
    sched.cap_batch(SCHED_NODE, SCHED_WALK, 15, "cross_kv")
    worker = _teardown_worker(known_rids=("a", "b", "c"))
    worker.scheduler = sched
    worker.engine_manager = _engine_manager(self_kv=self_kv, cross_kv=cross_kv)

    _remove(worker, "a", "b")

    assert self_kv._arena.num_free == free, "the teardown did not run"
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 15, (
        "a teardown that freed none of the cross pool lifted its cap"
    )

    assert _admit(cross_kv, {"c": 2 * PAGE_SIZE}, commit=True).ok
    _remove(worker, "c")
    assert sched._max_batch_size(SCHED_NODE, SCHED_WALK) == 16
