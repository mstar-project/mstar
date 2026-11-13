"""The two places a pre-plan could be left staged for a step that never runs:
a main-loop error while a speculation is armed but not yet submitted, and a
step that raises before its plan is promoted (``prepare_inputs`` on the GPU
thread, declare/admit inside ``Engine.exec``). Each drops the stage itself,
so the runner's foreign-step guard stays the backstop it is meant to be.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.thread import BrokenThreadPool
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.worker import make_worker
from mstar.worker.micro_scheduler import CHAIN_LANE, ScheduledBatch
from mstar.worker.speculation import Speculation
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker


def _fake_worker(engine=None) -> Worker:
    failed, resets, pushed_back, despeculated, settled = [], [], [], [], []
    # No stream buffers in the fake, so returning a chunk is a no-op.
    request_state = SimpleNamespace(per_request_info={})
    w = make_worker(
        worker_id=0, device=torch.device("cpu"),
        graph_runtime=SimpleNamespace(
            push_back_node=lambda node_name, rids, wg_ids: pushed_back.extend(rids),
            set_in_flight=lambda node_name, rids, wg_ids, value: despeculated.extend(rids),
            commit_speculation=lambda spec_id, success, dropped, **kw: settled.append((success, dropped)),
        ),
        request_state=request_state,
        streams=StreamBufferManager(request_state, None, []),
        engine_manager=SimpleNamespace(
            get_engine=lambda node: engine or SimpleNamespace(release_cached_prefix=lambda batch: None),
        ),
    )
    w.failed, w.resets, w.pushed_back, w.despeculated, w.settled = failed, resets, pushed_back, despeculated, settled
    w.pipeline._fail_requests = lambda errors: w.failed.append(dict(errors))
    w.executor.reset_skip_plan_flags = w.resets.append
    return w


def _speculation(plan_future, *rids: str, continuing=()) -> Speculation:
    return Speculation(
        scheduled_batch=ScheduledBatch(
            node_name="dit", graph_walk="w",
            request_to_worker_graph={rid: 0 for rid in rids},
        ),
        node_batch=SimpleNamespace(node_name="dit"),
        consumed_edges=set(),
        continuing_rids=set(continuing),
        partition="p",
        is_new_iter=False,
        is_same_node=True,
        plan_future=plan_future,
    )


def _done_future(value=True) -> Future:
    fut: Future = Future()
    fut.set_result(value)
    return fut


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc
    return _raise


def test_main_loop_error_drops_an_armed_unsubmitted_speculation() -> None:
    """The pending step raised while N+1 sat pre-planned on the plan thread:
    the plan future is drained, the stage dropped through the engine, the
    fresh rid goes back to its queue and the continuing rid fails with N."""
    w = _fake_worker()
    w.pipeline._hold_rows(CHAIN_LANE, {"a"})
    spec = _speculation(_done_future(), "a", "b", continuing=("a",))
    w.pipeline._handle_error(RuntimeError("boom"), (None, None), None, spec)
    assert w.resets == [spec.node_batch]
    assert spec.plan_future is None
    assert sorted(w.despeculated) == ["a", "b"]
    assert w.pushed_back == ["b"]
    assert list(w.failed[0]) == ["a"] and len(w.failed) == 1
    assert w.failed[0]["a"].startswith("Error in worker: internal error (error id ")
    # The step never ran, so its staged stream chunks are un-ingested and go
    # back to their buffers -- the fresh rid pushed back above would otherwise
    # hold one in a slot the next ingest can overtake.
    assert w.settled == [(False, [])]


def test_main_loop_error_leaves_a_submitted_speculation_alone() -> None:
    """Once the GPU thread has the speculation (and its plan future), its rids
    are that step's, so the handler has nothing to drain, reset or give back."""
    w = _fake_worker()
    spec = _speculation(None, "a")
    spec.submitted = True
    w.pipeline._handle_error(RuntimeError("boom"), (None, None), None, spec)
    assert w.resets == [] and w.pushed_back == []
    assert w.despeculated == []  # nothing de-speculated: not this handler's batch


def test_gpu_thread_drops_the_stage_when_prepare_inputs_raises() -> None:
    resets = []
    engine = SimpleNamespace(
        prepare_inputs=_raiser(RuntimeError("bad inputs")),
        reset_pre_plan_for_batch=resets.append,
        finalize_batch=lambda nb, **kw: None,
    )
    w = _fake_worker(engine)
    released = []
    node_batch = SimpleNamespace(
        node_name="dit", launch_started_event=None, preplanned_rids=("a",), read_check_rids=None,
        request_ids=("a",), step_context=SimpleNamespace(rebuild_rids=frozenset()),
        release_waiters=lambda: released.append(True),
        per_request_info={"a": SimpleNamespace(request_id="a")},
    )
    batch = SimpleNamespace(node_name="dit", graph_walk="decode", output_signals=(),
                            request_walks={})
    with pytest.raises(RuntimeError, match="bad inputs"):
        w.executor._body(batch, node_batch, _done_future())
    assert resets == [node_batch]
    assert released == [True]

    # A batch that was never pre-planned has no stage to drop.
    resets.clear()
    node_batch.preplanned_rids = None
    with pytest.raises(RuntimeError):
        w.executor._body(batch, node_batch, None)
    assert resets == []

    # A plan that raised fails the step before prepare_inputs. It still releases
    # the plan thread, and its launch event is set when it ends, or the admit
    # fence on it would never lift.
    released.clear()
    failed_plan: Future = Future()
    failed_plan.set_exception(RuntimeError("plan failed"))
    with ThreadPoolExecutor(max_workers=1) as gpu:
        w.executor._gpu_pool = gpu
        future = w.executor._submit_gpu_step(batch, node_batch, failed_plan, CHAIN_LANE)
        with pytest.raises(RuntimeError, match="plan failed"):
            future.result()
    assert node_batch.launch_started_event.is_set()
    assert released == [True]


def test_a_step_the_executor_never_runs_still_lifts_the_fence() -> None:
    """The admit fence waits on the launch event without a limit, so a step whose
    future fails without ever running (a GPU thread whose initializer raised) must
    still set it, or the main loop never reaches the ``future.result()`` that
    raises the real error."""
    w = _fake_worker()
    node_batch = SimpleNamespace(node_name="dit", launch_started_event=None)
    batch = SimpleNamespace(node_name="dit", graph_walk="decode")
    with ThreadPoolExecutor(max_workers=1, initializer=_raiser(RuntimeError("init"))) as gpu:
        w.executor._gpu_pool = gpu
        future = w.executor._submit_gpu_step(batch, node_batch, None, CHAIN_LANE)
        assert isinstance(future.exception(), BrokenThreadPool)
    assert node_batch.launch_started_event.is_set()


def _exec_fake(runner, raise_with):
    return make_engine(
        _submodules={"dit": make_mgmt(submodule=object())},
        _autocast_dtype_for=lambda submodule: None,
        reserve_replay_slot=lambda batch: None,
        _exec_single=_raiser(raise_with),
        _exec_per_request=_raiser(raise_with),
        _runner=runner,
    )


def _batch(preplanned) -> SimpleNamespace:
    return SimpleNamespace(
        request_ids=["a"], node_name="dit", running_batched=True,
        step_context=SimpleNamespace(graph_walk="decode", slot_lease=None),
        preplanned_rids=preplanned, preplan_event=object(),
        outputs_ready=SimpleNamespace(set=lambda: None),
        release_waiters=lambda: None,
        exec_timings=SimpleNamespace(start=None),
    )


def test_exec_drops_the_stage_when_a_step_raises_before_promotion() -> None:
    cleared = []
    runner = SimpleNamespace(staged=True, clear_preplan=lambda: cleared.append(True))
    batch = _batch(preplanned=("a",))
    with pytest.raises(ValueError, match="declare"):
        _exec_fake(runner, ValueError("declare failed")).exec(batch)
    assert cleared == [True]
    assert batch.preplanned_rids is None and batch.preplan_event is None


def test_exec_leaves_a_promoted_or_absent_stage_alone() -> None:
    cleared = []
    # promoted already: the runner has no stage
    runner = SimpleNamespace(staged=False, clear_preplan=lambda: cleared.append(True))
    with pytest.raises(ValueError):
        _exec_fake(runner, ValueError("forward failed")).exec(_batch(preplanned=("a",)))
    # never pre-planned: someone else's stage is not ours to drop
    runner = SimpleNamespace(staged=True, clear_preplan=lambda: cleared.append(True))
    with pytest.raises(ValueError):
        _exec_fake(runner, ValueError("forward failed")).exec(_batch(preplanned=None))
    assert cleared == []


def test_reset_drops_the_kept_step_of_a_partly_dropped_batch() -> None:
    # pre-planned over ["a", "b"] with inputs in hand (the step was kept for
    # exec), then prepare_inputs dropped "b": the stale step must not reach
    # admit, so the reset clears it even though the batch still has inputs
    cleared = []
    fake = make_engine(
        _submodules={"dit": make_mgmt()},
        _runner=SimpleNamespace(clear_preplan=lambda: cleared.append(True)),
    )
    batch = SimpleNamespace(
        request_ids=["a"], node_name="dit", inputs=[object()], step=object(),
        preplanned_rids=("a", "b"), preplan_event=object(),
        step_context=SimpleNamespace(slot_lease=None),
    )
    assert fake.preplan_is_stale(batch)
    fake.reset_pre_plan_for_batch(batch)
    assert batch.step is None and batch.preplanned_rids is None and batch.preplan_event is None
    assert cleared == [True]


class _FakeEvent:
    def __init__(self, device=None):
        self.recorded_on = "unrecorded"

    def record(self, stream=None):
        self.recorded_on = stream


def test_a_pre_plan_records_its_event_even_when_it_raises(monkeypatch) -> None:
    # a later step may skip a copy into a slot buffer this pre-plan issued,
    # so its plan-stream copies must be waited on even if it never finishes
    monkeypatch.setattr(torch, "Event", _FakeEvent)
    fake = make_engine(
        _order_after_pre_plans=lambda: None,
        _runner=SimpleNamespace(
            pre_admit=lambda step: SimpleNamespace(ok=True),
            pre_plan=_raiser(RuntimeError("plan failed")),
        ),
    )
    step = SimpleNamespace(set_ctx=lambda ctx: None, state={})
    submodule = SimpleNamespace(declare_step=lambda **kw: step)
    cg_runner = SimpleNamespace(
        declare_inputs_for=lambda lease: [], step_ids=lambda lease, rids: rids,
        plan_stream=lambda: None,
    )
    batch = SimpleNamespace(
        node_name="dit", inputs=None, request_ids=["a"], per_request_info={},
        step=None, preplan_event=None, preplanned_rids=None,
        step_context=SimpleNamespace(
            graph_walk="decode", padded_request_ids=["a"], is_preplan=False,
            set_padded_rids=lambda rids: None, resident=None,
        ),
    )
    submodule.step_is_input_free = lambda walk: False
    mgmt = make_mgmt(submodule)
    with pytest.raises(RuntimeError):
        fake._pre_plan_for_batch(batch, mgmt, cg_runner, object())
    event = fake._pre_plan_done["dit"]
    assert isinstance(event, _FakeEvent) and event.recorded_on is None
    assert batch.preplan_event is None


def test_a_step_is_ordered_after_every_plan_streams_pre_plans(monkeypatch) -> None:
    # nodes can share a resource's slot buffers, so every node's latest
    # pre-plan is waited on, not just the stepping node's
    waits = []
    stream = SimpleNamespace(wait_event=waits.append)
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda device=None: stream)
    fake = make_engine()
    fake._order_after_pre_plans()
    assert waits == []
    fake._pre_plan_done.update(enc="e1", dec="e2")
    fake._order_after_pre_plans()
    assert waits == ["e1", "e2"]


def test_an_inline_plan_waits_on_pre_plans_before_the_resident_stage_copies() -> None:
    # the resident stage copies into the bucket's cells and slots, which a
    # dropped pre-plan may still be copying into on its plan stream
    from mstar.engine.resources.step import BucketKey, SlotLease, StepContext, SubmoduleStep

    calls = []

    def plan(step, phase):
        calls.append("runner")
        raise NotImplementedError("reached plan")

    engine = make_engine(_runner=SimpleNamespace(plan=plan), _order_after_pre_plans=lambda: calls.append("wait"))
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 1, 1))
    ctx = StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False, slot_lease=lease)
    ctx.resident = SimpleNamespace(plan=lambda step, ctx: calls.append("resident"))
    mgmt = make_mgmt()
    with pytest.raises(NotImplementedError, match="reached plan"):
        engine._drive_step(
            SimpleNamespace(node_name="LLM", preplan_event=None), mgmt, [1], [], {},
            ctx, lease, True, SubmoduleStep(steps={}, cg_key_info=None), False,
        )
    assert calls == ["wait", "resident", "runner"]
