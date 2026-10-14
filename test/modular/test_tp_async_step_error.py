"""TP async scheduling: a step that raises must not leave a follower holding a
batch rank 0 will never run.

A follower runs every ``ScheduleTPNode`` it is sent, in order, and its forward
waits in the collective until rank 0 joins. A batch the leader sends and then
drops leaves the follower there for good, and the TP group stops serving.

Drives the real ``Worker.run`` on a leader over stubs: a fresh request whose
forward raises on the GPU thread (a seed the sampler cannot hold, say), and the
next iteration, which builds the other requests' step as a yield-away batch
while that forward is still in flight.
"""

import time
import types

import pytest

from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode, TPNoSpeculation
from mstar.worker import worker as worker_mod
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import PendingBatch, Worker

NODE, WALK = "LLM", "decode"


class _Stop(BaseException):
    """Ends ``run``: not an ``Exception``, so the main loop's handler lets it out."""


def _batch(*rids):
    return ScheduledBatch(
        node_name=NODE, graph_walk=WALK,
        request_to_worker_graph={rid: 0 for rid in rids},
    )


def _leader(batches, bad_rid):
    """Rank 0 of a two-rank NODE. The scheduler hands out ``batches`` in order;
    a forward over ``bad_rid`` raises."""
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.tp_async_sched, w.tp_async_nodes = True, None
    w.is_tp_follower = False
    w.parallel_nodes = w.parallel_leader_nodes = {NODE}
    w._tp_broadcast_seq = 0
    w._in_flight_rids = set()
    w.enable_nvtx = False
    w._phase_period, w._phase_buf = 0, {}
    w.sent, w.built, w.ran, w.failed = [], [], [], set()

    group = types.SimpleNamespace(_workers=["w0", "w1"])
    sharding = types.SimpleNamespace(get_sharding_group=lambda node, walk: group)
    engine = types.SimpleNamespace(take_resident_delta=lambda node: OffloadDelta.new())
    queue = list(batches)
    w.communicator = types.SimpleNamespace(send=lambda to, msg: w.sent.append((to, msg)))
    w._graph_runtime = types.SimpleNamespace(
        is_async_schedulable=lambda node, walk: True,
        get_sharding_config=lambda rid: sharding,
        get_rid_string=str,
        push_back_node=lambda *a: None,
    )
    w.engine_manager = types.SimpleNamespace(
        warmup_all=lambda: None, get_engine=lambda node: engine,
    )
    w.parallel_groups = types.SimpleNamespace(barrier_all=lambda: None)
    w.scheduler = types.SimpleNamespace(
        take_admit_errors=dict,
        has_ready_excluding=lambda *a: False,
        get_next_batch=lambda *a, **k: queue.pop(0) if queue else None,
    )
    w.request_state = types.SimpleNamespace(get_partition_for_node=lambda node: "p")
    w.wakeup_event = types.SimpleNamespace(register_future=lambda f: None)

    def _build(batch):
        w.built.append(list(batch.request_to_worker_graph))
        return types.SimpleNamespace(
            node_name=batch.node_name, graph_walk=batch.graph_walk,
            request_ids=list(batch.request_to_worker_graph),
            admit_error=None, failed_requests={},
        )

    def _forward(batch, node_batch, plan_future):
        w.ran.append(list(node_batch.request_ids))
        node_batch.launch_started_event.set()
        if bad_rid in node_batch.request_ids:
            raise OverflowError("Python int too large to convert to C long")
        return {rid: {} for rid in node_batch.request_ids}

    passes = iter(range(3))

    def _next_pass(pending):
        # two passes: the bad step's, then the one that awaits it
        if next(passes) == 2:
            raise _Stop

    w._await_admit_settled = _next_pass
    w._build_executing_batch = _build
    w._execute_on_gpu_thread = _forward
    w._try_speculate_next = lambda pending: None
    w._preplan_spec = lambda pending, spec: False
    w._fail_requests = w.failed.update
    for name in (
        "_verify_tp_async_sched_agrees", "_init_engine_thread",
        "_apply_pending_removes_safe_to_drop", "_apply_pending_drains",
        "_process_messages", "_apply_removes_whose_step_landed",
        "_check_ready_tensors", "_poll_stream_buffers",
        "_clear_in_flight_flag", "_set_in_flight_flag",
        "_reset_skip_plan_flags", "_settle_speculation",
    ):
        setattr(w, name, lambda *a, **k: None)
    return w


def _run(monkeypatch, batches, bad_rid):
    monkeypatch.setattr(worker_mod, "gc", types.SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0,
    ))
    w = _leader(batches, bad_rid)
    with pytest.raises(_Stop):
        w.run()
    return w


@pytest.fixture
def leader(monkeypatch):
    return _run(monkeypatch, [_batch("bad"), _batch("a", "b")], bad_rid="bad")


def _to_follower(leader):
    return [msg.body for to, msg in leader.sent if to == "w1"]


def test_leader_sends_every_batch_it_runs(monkeypatch):
    w = _run(monkeypatch, [_batch("x"), _batch("a", "b")], bad_rid=None)
    sent = [b.request_ids for b in _to_follower(w) if isinstance(b, ScheduleTPNode)]
    assert sent == w.ran == [["x"], ["a", "b"]]


def test_leader_runs_every_batch_it_sends_when_a_step_raises(leader):
    assert leader.built == [["bad"], ["a", "b"]]
    assert "bad" in leader.failed
    sent = [b.request_ids for b in _to_follower(leader) if isinstance(b, ScheduleTPNode)]
    assert sent == leader.ran, (
        f"sent {sent} to the follower but ran {leader.ran}: the follower runs "
        "the rest alone and waits in its collective for good"
    )


def test_follower_holds_nothing_after_the_raised_step(leader):
    """The follower's side of the same run: it takes the bad step, raises on it
    too, and closes it. Anything left in its FIFO it would run unconditionally."""
    f = Worker.__new__(Worker)
    f.worker_id = "w1"
    f.tp_async_sched, f.tp_async_nodes = True, None
    f._tp_nospec = RecentSet(Worker._TP_NOSPEC_KEEP)
    f.scheduler = MicroScheduler(engine_manager=None, parallel_leader_nodes=set())
    for body in _to_follower(leader):
        if isinstance(body, TPNoSpeculation):
            f._register_tp_nospec(body)
        else:
            f._register_tp_follow(body)

    step = f.scheduler.pop_tp_follow_head()
    assert step.request_ids == ["bad"]
    f._close_tp_follow_step(PendingBatch(
        batch=None, node_batch=None, node_name=NODE, partition="p",
        graph_walk=WALK, future=None, tp_seq=step.spec_seq,
    ))

    left = f.scheduler.peek_tp_follow()
    assert left is None, f"follower still holds {left.request_ids}, which rank 0 dropped"


def test_a_deferred_remove_waits_for_the_next_steps_admit(monkeypatch):
    """A remove held back while its request ran is applied once the next step is
    submitted. Rank 0 stamps it after that step, and a follower frees the pages
    only after admitting it, so rank 0 must not free them before its own admit."""
    monkeypatch.setattr(worker_mod, "gc", types.SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0,
    ))
    w = _leader([_batch("a", "x"), _batch("b")], bad_rid=None)
    order = []
    passes = iter(range(3))

    def _messages():
        n = next(passes)
        if n == 1:
            w._pending_removes.add("x")  # x's remove lands while its step runs
        elif n == 2:
            raise _Stop

    def _forward(batch, node_batch, plan_future):
        if "b" in node_batch.request_ids:
            time.sleep(0.2)  # the GPU thread is still preparing b
        order.append(("admit", *node_batch.request_ids))
        node_batch.launch_started_event.set()
        return {rid: {} for rid in node_batch.request_ids}

    w._pending_removes = set()
    w._process_messages = _messages
    w._execute_on_gpu_thread = _forward
    w._postprocess_batch = lambda *a: None
    w._remove_request = lambda body: order.append(("release", body.request_id))
    for name in ("_await_admit_settled", "_apply_pending_removes_safe_to_drop"):
        setattr(w, name, types.MethodType(getattr(Worker, name), w))
    with pytest.raises(_Stop):
        w.run()

    assert order == [("admit", "a", "x"), ("admit", "b"), ("release", "x")]
