"""The two places a pre-plan could be left staged for a step that never runs:
a main-loop error while a speculation is armed but not yet submitted, and a
step that raises before its plan is promoted (``prepare_inputs`` on the GPU
thread, declare/admit inside ``Engine.exec``). Each drops the stage itself,
so the runner's foreign-step guard stays the backstop it is meant to be.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.worker import worker as worker_mod
from mstar.worker.worker import Worker


def _fake_worker() -> Worker:
    w = Worker.__new__(Worker)
    w.worker_id = 0
    w.enable_nvtx = False
    w._phase_period = 0
    w._phase_buf = defaultdict(list)
    w.device = torch.device("cpu")
    w._in_flight_rids = set()
    w.failed = []
    w.resets = []
    w.pushed_back = []
    w._fail_requests = lambda errors: w.failed.append(dict(errors))
    w._reset_skip_plan_flags = w.resets.append
    queue = SimpleNamespace(push_back_node=lambda rid, node: w.pushed_back.append(rid))
    w.worker_graphs_manager = SimpleNamespace(queues={0: queue})
    return w


def _speculation(plan_future, *rids: str, continuing=()) -> worker_mod.Speculation:
    nodes = {rid: SimpleNamespace(_speculatively_scheduled=True) for rid in rids}
    return worker_mod.Speculation(
        scheduled_batch=SimpleNamespace(
            node_objects=nodes, request_to_worker_graph={rid: 0 for rid in rids},
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
    w._in_flight_rids = {"a"}
    spec = _speculation(_done_future(), "a", "b", continuing=("a",))
    Worker._handle_main_loop_error(w, RuntimeError("boom"), (None, None), None, spec)
    assert w.resets == [spec.node_batch]
    assert spec.plan_future is None
    assert all(not n._speculatively_scheduled for n in spec.scheduled_batch.node_objects.values())
    assert w.pushed_back == ["b"]
    assert w.failed == [{"a": "Error in worker: RuntimeError: boom"}]


def test_main_loop_error_leaves_a_submitted_speculation_alone() -> None:
    """Once the GPU thread has the plan future, the speculation carries no
    future any more and the handler has nothing to drain or reset."""
    w = _fake_worker()
    spec = _speculation(None, "a")
    Worker._handle_main_loop_error(w, RuntimeError("boom"), (None, None), None, spec)
    assert w.resets == [] and w.pushed_back == []
    assert spec.scheduled_batch.node_objects["a"]._speculatively_scheduled


def test_gpu_thread_drops_the_stage_when_prepare_inputs_raises() -> None:
    w = _fake_worker()
    resets = []
    engine = SimpleNamespace(
        prepare_inputs=_raiser(RuntimeError("bad inputs")),
        reset_pre_plan_for_batch=resets.append,
        finalize_batch=lambda nb: None,
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda name: engine)
    released = []
    node_batch = SimpleNamespace(
        node_name="dit", launch_started_event=None, preplanned_rids=("a",),
        release_waiters=lambda: released.append(True),
    )
    batch = SimpleNamespace(node_name="dit", graph_walk="decode")
    with pytest.raises(RuntimeError, match="bad inputs"):
        Worker._execute_on_gpu_thread(w, batch, node_batch, _done_future())
    assert resets == [node_batch]
    assert released == [True]

    # A batch that was never pre-planned has no stage to drop.
    resets.clear()
    node_batch.preplanned_rids = None
    with pytest.raises(RuntimeError):
        Worker._execute_on_gpu_thread(w, batch, node_batch, None)
    assert resets == []


def _exec_fake(runner, raise_with):
    fake = SimpleNamespace(
        _enable_nvtx=False, _enable_profile=False, _device=torch.device("cpu"),
        _submodules={"dit": SimpleNamespace(submodule=object())},
        _autocast_dtype_for=lambda submodule: None,
        reserve_replay_slot=lambda batch: None,
        _exec_single=_raiser(raise_with),
        _exec_per_request=_raiser(raise_with),
        _runner=runner,
    )
    fake.exec = Engine.exec.__get__(fake)
    return fake


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
