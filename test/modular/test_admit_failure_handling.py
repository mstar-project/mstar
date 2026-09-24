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

from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    FullAdmitOutcome,
    RequestOffloading,
    StepContext,
    SubmoduleStep,
)
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
    return SimpleNamespace(request_ids=["r0", "r1"], admit_error=None)


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

    def _handle_allocation_failure(self, batch, node_batch):
        self.offload_calls.append(node_batch.node_name)
        # the real one push-backs and holds; stand in for both
        self._push_back_batch(batch)
        self.scheduler.hold_requests(list(batch.request_to_worker_graph))


def _batches(reason):
    batch = ScheduledBatch(
        node_name="node", graph_walk="walk",
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    node_batch = SimpleNamespace(node_name="node", admit_error=reason)
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
    worker._handle_admit_failure(*_batches(_alloc_failed()))

    assert worker.offload_calls == ["node"]
    # exactly one push-back per request: the delegation must not double up
    assert sorted(worker.queue.pushed_back) == ["r0", "r1"]
    assert sorted(worker.held) == ["r0", "r1"]


class _HoldingWorker:
    """Binds the real allocation handler onto a worker with nothing to evict."""

    _handle_allocation_failure = Worker._handle_allocation_failure
    _push_back_batch = Worker._push_back_batch

    def __init__(self):
        self._graph_runtime = _Runtime()
        self.scheduler = SimpleNamespace(hold_requests=lambda rids: None)
        self._hold_logged = {}

    def _try_offload_cold_request(self, node_name, batch_ids, affected_resources=None):
        # no victim
        del node_name, batch_ids, affected_resources


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


def test_a_hold_on_another_walk_is_logged_inside_the_interval(monkeypatch, caplog):
    worker = _HoldingWorker()

    with caplog.at_level(logging.WARNING, logger=worker_mod.__name__):
        _hold(worker, monkeypatch, 0.0, walk="prefill")
        _hold(worker, monkeypatch, 0.1, walk="decode")

    assert len(_hold_lines(caplog)) == 2, (
        "one walk's hold silenced another's, so a stall on it would go unseen"
    )


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
        self._runner = self
        self._device = torch.device("cpu")
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

    # --- engine internals the path calls into
    def _drive_step(
        self, batch, submodule_mgmt, request_ids, inputs, req_info, ctx,
        lease, running_batched, step, set_launch,
    ):
        del batch, submodule_mgmt, inputs, req_info, lease, running_batched
        del set_launch
        rid = request_ids[0]
        assert step is not None, "the step admitted for this rid must reach it"
        assert tuple(ctx.request_ids) == (rid,), "each rid drives its own ctx"
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


def test_per_request_admits_every_rid_before_running_any():
    engine = _FakeExecEngine()

    out = engine._exec_per_request(_exec_batch(["a", "b"]))

    assert [e for e in engine.events if e[0] != "declare"] == [
        ("admit", "a"), ("admit", "b"), ("run", "a"), ("run", "b"),
    ]
    assert set(out) == {"a", "b"}


def test_per_request_runs_nothing_when_a_later_rid_fails_admit():
    """The reviewer's bug: `a` used to run and commit, then `b` failed admit
    and the whole batch was re-queued — replaying a step whose resource state
    had already moved."""
    engine = _FakeExecEngine(fail_on="b")
    batch = _exec_batch(["a", "b"])

    out = engine._exec_per_request(batch)

    assert ("run", "a") not in engine.events, "a ran despite the batch failing"
    assert not any(kind == "run" for kind, _ in engine.events)
    assert out == {"a": {}, "b": {}}
    assert isinstance(batch.admit_error, AllocationFailed)


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
