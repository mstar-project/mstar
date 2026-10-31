"""``SideStepLane``: a stream-fed walk's steps beside the worker's main chain.

The stream and event backends are mocked, so these run without a GPU; what they
pin is the order of the fence, the wait and the launch, when a step may be
postprocessed, and what a step off the chain must leave alone.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from test_micro_scheduler import NODE, WALK, _Engine, _has_ready, _Manager, _next_batch, _scheduler
from test_send_registration_failure import _decode_step, _postprocess_worker

from mstar.engine import accelerator_graph_runner as runner_module
from mstar.engine.engine import Engine
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.utils.threads import ThreadPool
from mstar.worker import side_steps
from mstar.worker.engine_manager import EngineManager, _refuse_unsuitable_side_walks
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.side_steps import SideStepLane
from mstar.worker.worker import PendingBatch, Worker

SIDE = ("Codec", "codec_chunk")


class _StreamBackend:
    """Records what reaches the stream API, in order."""

    def __init__(self, monkeypatch):
        self.log: list[str] = []
        self.streams = 0
        backend = self

        class Stream:
            def __init__(self, device):
                backend.streams += 1

            def wait_event(self, event):
                backend.log.append(f"wait {event.on}")

            def __enter__(self):
                backend.log.append("enter side")

            def __exit__(self, *exc):
                backend.log.append("exit side")

        class Event:
            def __init__(self, device):
                self.on = None

            def record(self, stream):
                self.on = stream
                backend.log.append(f"record {stream}")

        monkeypatch.setattr(side_steps.torch, "Stream", Stream)
        monkeypatch.setattr(side_steps.torch, "Event", Event)
        monkeypatch.setattr(side_steps.torch.accelerator, "current_stream", lambda device: "main")


def _batch(rids, node="Codec", walk="codec_chunk"):
    return ScheduledBatch(
        node_name=node, graph_walk=walk,
        request_to_worker_graph=dict.fromkeys(rids, 0), output_signals=["audio_chunk"],
    )


def _step(rids, landed=True, done=True, failed=None):
    future = Future()
    if done:
        future.set_result(f"outputs of {rids}")
    return PendingBatch(
        batch=_batch(rids), node_batch=SimpleNamespace(
            completion_event=SimpleNamespace(query=lambda: landed), failed_requests=dict(failed or {}),
        ),
        node_name=SIDE[0], partition="Codec", graph_walk=SIDE[1], future=future,
    )


def _worker():
    log: list = []
    worker = SimpleNamespace(
        log=log,
        worker_id="w0",
        _metrics=None,
        _chain_rids=frozenset(),
        device=torch.device("cpu"),
        _in_flight_rids=set(),
        _span=lambda *a, **k: threading.Lock(),
        _wait_for_step=lambda step: log.append("wait for step"),
        _clear_in_flight_flag=lambda batch: log.append("clear in flight"),
        _set_in_flight_flag=lambda batch, value: log.append(f"in flight {value}"),
        _drop_failed_rids=lambda step, outputs, failed: log.append(("drop", sorted(failed))),
        _fail_requests=lambda failed: log.append(("fail", sorted(failed))),
        _postprocess_batch=lambda step, outputs: log.append(("postprocess", outputs)),
        _build_executing_batch=lambda batch: SimpleNamespace(completion_event=None),
        _side_pending=lambda batch, node_batch, future, at: _step(list(batch.request_to_worker_graph)),
        _submit_gpu_step=lambda *a, **k: Future(),
        _execute_on_gpu_thread=lambda *a: log.append("execute") or "outputs",
        wakeup_event=SimpleNamespace(register_future=lambda future: None),
        request_state=object(),
    )
    worker.scheduler = SimpleNamespace(
        clear_batch_limit=lambda *a, **k: log.append("clear limit"),
        get_next_batch=lambda state, target: None,
    )
    return worker


# ----- stream choice and event order ----------------------------------------

def test_a_step_runs_on_the_side_stream_behind_the_main_stream(monkeypatch):
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    worker._execute_on_gpu_thread = lambda *a: backend.log.append("execute") or "outputs"
    lane = SideStepLane(worker, [SIDE])

    assert lane.execute(_batch([1]), object(), None) == "outputs"

    assert backend.log == ["record main", "wait main", "enter side", "execute", "exit side"]


def test_every_step_shares_one_stream(monkeypatch):
    backend = _StreamBackend(monkeypatch)
    lane = SideStepLane(_worker(), [SIDE])

    lane.execute(_batch([1]), object(), None)
    lane.execute(_batch([2]), object(), None)

    assert backend.streams == 1


def test_the_submitted_step_takes_the_lane_body_and_lifts_the_launch_fence(monkeypatch):
    _StreamBackend(monkeypatch)
    worker = Worker.__new__(Worker)
    worker.device = torch.device("cpu")
    worker._execute_on_gpu_thread = lambda *a: "outputs"
    lane = SideStepLane(worker, [SIDE])
    node_batch = SimpleNamespace(launch_started_event=None)
    with ThreadPool(max_workers=1) as pool:
        future = Worker._submit_gpu_step(worker, pool, _batch([1]), node_batch, None, execute=lane.execute)
        assert future.result() == "outputs"
    assert node_batch.launch_started_event.wait(timeout=5)


# ----- one step in flight ---------------------------------------------------

def test_one_step_in_flight():
    worker = _worker()
    asked = []
    worker.scheduler.get_next_batch = lambda state, target: asked.append(target) or _batch([len(asked)])
    lane = SideStepLane(worker, [SIDE])

    lane.launch(None)
    lane.launch(None)
    assert asked == [SIDE] and worker._in_flight_rids == {1}

    lane.pending = _step([1])
    lane.finish()
    lane.launch(None)
    assert asked == [SIDE, SIDE] and worker._in_flight_rids == {2}


def test_nothing_ready_launches_nothing():
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])

    lane.launch(None)

    assert lane.pending is None and worker.log == []


# ----- finishing a step -----------------------------------------------------

@pytest.mark.parametrize("step", [_step([1], landed=False), _step([1], done=False)], ids=["running", "launching"])
def test_a_step_still_running_is_left_alone(step):
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])
    lane.pending = step

    lane.finish()

    assert lane.pending is step and worker.log == []


def test_postprocess_runs_after_the_step_has_landed_and_its_flags_are_down():
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])
    lane.pending = _step([1, 2])
    worker._in_flight_rids = {1, 2, 9}

    lane.finish()

    assert worker.log == ["wait for step", "clear in flight", "clear limit", ("postprocess", "outputs of [1, 2]")]
    assert lane.pending is None and worker._in_flight_rids == {9}


def test_a_row_the_chain_still_holds_stays_in_flight():
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])
    lane.pending = _step([1, 2])
    worker._chain_rids = {2}
    worker._in_flight_rids = {1, 2}

    lane.finish()

    assert worker._in_flight_rids == {2}


def test_block_waits_for_a_step_still_running():
    """Postprocess frees the step's inputs, main-stream allocations a
    main-stream step could reuse while this one still reads them."""
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])
    lane.pending = _step([1], landed=False)

    lane.finish(block=True)

    assert worker.log[0] == "wait for step" and ("postprocess", "outputs of [1]") in worker.log


def test_failed_rows_are_reported_before_the_step_routes():
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])
    lane.pending = _step([1, 2], failed={2: "boom"})

    lane.finish()

    assert worker.log.index(("fail", [2])) < worker.log.index(("postprocess", "outputs of [1, 2]"))


# ----- the chain's postprocess order ----------------------------------------

def _chain_worker():
    """A real ``_postprocess_batch`` whose runtime keeps pending stops as the real ones do."""
    w = _postprocess_worker(refuse=set())
    stops = {0}

    def clear():
        stops.clear()

    w._graph_runtime.clear_pending_loop_stops = clear
    w._graph_runtime.pending_loop_stop_rids = lambda walks, loop_name: set(stops)
    return w


def test_a_side_step_between_two_chain_steps_leaves_the_loop_stop_in_place():
    """The chain's step past a stop must still be dropped when a side step's
    postprocess lands between the stop and that step."""
    w = _chain_worker()
    lane = SideStepLane(w, [SIDE])
    w._side_steps = lane
    side, side_outputs = _decode_step([5])
    lane.pending = side
    past_the_stop, outputs = _decode_step([0, 1])
    past_the_stop.speculative_new_iter = True

    Worker._postprocess_batch(w, side, side_outputs)
    Worker._postprocess_batch(w, past_the_stop, outputs)

    assert w.log.routed == [[5], [1]]


def test_the_chain_still_clears_its_stops():
    w = _chain_worker()
    w._side_steps = SideStepLane(w, [SIDE])
    step, outputs = _decode_step([1])

    Worker._postprocess_batch(w, step, outputs)

    assert w._graph_runtime.pending_loop_stop_rids([], "decode") == set()


def test_a_worker_with_no_lane_clears_as_before():
    w = _chain_worker()
    step, outputs = _decode_step([1])

    Worker._postprocess_batch(w, step, outputs)

    assert w._graph_runtime.pending_loop_stop_rids([], "decode") == set()


# ----- rollback -------------------------------------------------------------

def _raising_step(rids):
    step = _step(rids, done=False)
    step.future.set_exception(RuntimeError("boom"))
    return step


def test_a_step_that_raised_fails_its_own_rows_and_clears_out():
    worker = _worker()
    lane = SideStepLane(worker, [SIDE])
    lane.pending = _raising_step([3])
    worker._in_flight_rids = {3, 9}

    lane.finish()

    assert ("fail", [3]) in worker.log and "clear in flight" in worker.log
    assert lane.pending is None and worker._in_flight_rids == {9}


def test_a_step_that_could_not_launch_fails_its_rows():
    worker = _worker()
    worker.scheduler.get_next_batch = lambda state, target: _batch([3])

    def refuse(batch):
        raise RuntimeError("boom")

    worker._build_executing_batch = refuse
    lane = SideStepLane(worker, [SIDE])

    lane.launch(None)

    assert ("fail", [3]) in worker.log and lane.pending is None


def test_a_main_loop_error_leaves_the_side_step_to_finish():
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    lane = w._side_steps = SideStepLane(w, [SIDE])
    side = _step([3])
    lane.pending = side
    chain = _step([1], done=True)
    w._in_flight_rids = {1, 3}
    failed = []
    w._clear_in_flight_flag = lambda batch: None
    w._fail_requests = failed.append

    w._handle_main_loop_error(RuntimeError("boom"), (chain, None), None)

    assert set(failed[0]) == {1} and lane.pending is side


def test_a_row_in_both_steps_fails_with_the_chain():
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    lane = w._side_steps = SideStepLane(w, [SIDE])
    lane.pending = _step([1])
    w._in_flight_rids = {1}
    failed = []
    w._clear_in_flight_flag = lambda batch: None
    w._fail_requests = failed.append

    w._handle_main_loop_error(RuntimeError("boom"), (_step([1]), None), None)

    assert set(failed[0]) == {1}


def test_rids_cover_the_step_in_flight_for_teardown():
    lane = SideStepLane(_worker(), [SIDE])
    assert lane.rids == set()
    step = _step([3, 4])
    lane.pending = step

    assert lane.rids == {3, 4} and lane.owns(step) and not lane.owns(_step([3, 4]))


# ----- the scheduler ---------------------------------------------------------

def test_a_side_walk_does_not_break_the_chain():
    sched = _scheduler(_Engine())
    sched.side_keys = frozenset({(NODE, WALK)})
    manager = _Manager([1, 2])

    assert not _has_ready(sched, manager, exclude_target=("other", "walk"))


def test_a_side_walk_is_not_scheduled_on_the_chain():
    sched = _scheduler(_Engine())
    sched.side_keys = frozenset({(NODE, WALK)})
    manager = _Manager([1, 2])

    assert _next_batch(sched, manager) is None
    assert sorted(_next_batch(sched, manager, target=(NODE, WALK)).request_to_worker_graph) == [1, 2]


def test_a_side_walks_backlog_stays_off_the_chain():
    sched = _scheduler(_Engine(max_bs=1))
    key = (NODE, WALK)
    sched.side_keys = frozenset({key})
    manager = _Manager([1, 2])

    first = _next_batch(sched, manager, target=key)
    assert sched.backlog, "the cap should leave a row behind"
    assert not _has_ready(sched, manager, exclude_target=("other", "walk"))
    assert _next_batch(sched, manager) is None
    second = _next_batch(sched, manager, target=key)
    assert {*first.request_to_worker_graph, *second.request_to_worker_graph} == {1, 2}


def test_no_side_keys_leaves_the_scheduler_as_it_was():
    sched = _scheduler(_Engine())
    manager = _Manager([1])

    assert _has_ready(sched, manager, exclude_target=("other", "walk"))


# ----- capture --------------------------------------------------------------

class _GraphBackend:
    def __init__(self):
        self.made = 0

    def is_available(self):
        return True

    def new_stream(self):
        self.made += 1
        return f"stream {self.made}"


def _engine(backend):
    engine = Engine.__new__(Engine)
    engine._device_capability = SimpleNamespace(graph_backend=backend)
    return engine


def test_a_side_walk_captures_on_a_stream_of_its_own():
    """Graphs captured on one stream must not overlap, and a side step overlaps
    every graph of the chain."""
    backend = _GraphBackend()

    assert _engine(backend)._capture_stream_for(_DECLARES) == "stream 1"


def test_a_node_on_the_chain_keeps_the_shared_capture_stream():
    backend = _GraphBackend()

    assert _engine(backend)._capture_stream_for(SimpleNamespace(side_stream_walks=frozenset())) is None
    assert backend.made == 0


def test_no_graph_backend_means_no_capture_stream():
    assert _engine(None)._capture_stream_for(_DECLARES) is None


def test_capture_runs_on_the_stream_it_is_given(monkeypatch):
    seen = {}

    class Capture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    backend = SimpleNamespace(
        device_type="cpu",
        current_stream=lambda: None,
        create_graph=lambda keep_graph: "graph",
        capture=lambda graph, pool, stream: seen.update(stream=stream) or Capture(),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(runner_module.DeviceCapability, "of", lambda device: SimpleNamespace(graph_backend=backend))
    pool = SimpleNamespace(on_success=lambda: None, on_failure=lambda graph: None)

    runner_module.capture_into_graph(lambda: "output", pool, "cpu", None, stream="own")

    assert seen["stream"] == "own"


# ----- declaration and startup ----------------------------------------------

def _codec_node(name="Codec"):
    return GraphNode(
        name=name, input_names=["codec_tokens"],
        outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
    )


def _model(walks):
    return SimpleNamespace(get_graph_walk_graphs=lambda: walks)


_DECLARES = SimpleNamespace(side_stream_walks=frozenset({"codec_chunk"}))


def test_a_stream_fed_walk_with_no_resources_qualifies():
    _refuse_unsuitable_side_walks({"Codec": _DECLARES}, [], _model({"codec_chunk": _codec_node()}))


def test_a_walk_the_chain_could_speculate_into_is_refused():
    loop = Loop(name="decode_loop", section=_codec_node(), max_iters=4, outputs=[])
    with pytest.raises(ValueError, match="not a walk of that one node"):
        _refuse_unsuitable_side_walks({"Codec": _DECLARES}, [], _model({"codec_chunk": loop}))


def test_a_walk_of_another_node_is_refused():
    with pytest.raises(ValueError, match="not a walk of that one node"):
        _refuse_unsuitable_side_walks({"Codec": _DECLARES}, [], _model({"codec_chunk": _codec_node("Talker")}))


def test_a_node_with_resources_is_refused():
    specs = [SimpleNamespace(nodes={"Codec"})]
    with pytest.raises(ValueError, match="holds resources"):
        _refuse_unsuitable_side_walks({"Codec": _DECLARES}, specs, _model({"codec_chunk": _codec_node()}))


def _manager(submodules):
    return EngineManager(engine=SimpleNamespace(submodules_by_node=lambda: submodules), node_names=set(submodules))


def test_side_keys_need_another_node_on_the_worker():
    alone = _manager({"Codec": _DECLARES})
    assert alone.side_stream_keys(True, set()) == []


def test_side_keys_need_streams():
    both = _manager({"Codec": _DECLARES, "Talker": SimpleNamespace(side_stream_walks=frozenset())})
    assert both.side_stream_keys(False, set()) == []
    assert both.side_stream_keys(True, set()) == [SIDE]


def test_a_parallel_node_stays_on_the_chain():
    both = _manager({"Codec": _DECLARES, "Talker": SimpleNamespace(side_stream_walks=frozenset())})
    assert both.side_stream_keys(True, {"Codec"}) == []
