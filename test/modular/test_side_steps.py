"""Side lanes: a node's steps beside the worker's main chain (``StepPipeline._launch_side``,
``StepPipeline._finish_side``).

The stream and event backends are mocked, so these run without a GPU; what they
pin is where a side step's kernels and its window's collate go, when a step may
be postprocessed, what a step off the chain must leave alone, the scheduler's
lane filter, and which configs the startup decision refuses.
"""
from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from test_micro_scheduler import NODE, WALK, _Engine, _has_ready, _Manager, _next_batch, _scheduler
from test_send_registration_failure import _decode_step, _postprocess_worker

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine import accelerator_graph_runner as runner_module
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import DeviceLoopback, NodeSubmodule, TokenStop
from mstar.testing_utils.engine import make_engine
from mstar.testing_utils.model import StubModel
from mstar.testing_utils.worker import make_worker
from mstar.utils.threads import ThreadPool
from mstar.worker import step_executor, step_pipeline
from mstar.worker.engine_manager import EngineManager, _side_lane_keys, side_stream_nodes
from mstar.worker.micro_scheduler import CHAIN_LANE, SIDE_LANE, ScheduledBatch
from mstar.worker.step_executor import PendingBatch

SIDE = ("Codec", "codec_chunk")
_OUTPUTS = SimpleNamespace(host_rows=None, host_spans=None)


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
                backend.log.append("wait")

            def __enter__(self):
                backend.log.append("enter side")

            def __exit__(self, *exc):
                backend.log.append("exit side")

        monkeypatch.setattr(step_executor.torch, "Stream", Stream)


def _batch(rids, node="Codec", walk="codec_chunk"):
    return ScheduledBatch(
        node_name=node, graph_walk=walk,
        request_to_worker_graph=dict.fromkeys(rids, 0), output_signals=["audio_chunk"],
    )


def _step(rids, landed=True, done=True, failed=None, lane=SIDE_LANE):
    future = Future()
    if done:
        future.set_result(f"outputs of {rids}")
    return PendingBatch(
        batch=_batch(rids), node_batch=SimpleNamespace(
            completion_event=SimpleNamespace(query=lambda: landed), failed_requests=dict(failed or {}),
        ),
        node_name=SIDE[0], partition="Codec", graph_walk=SIDE[1], future=future, lane=lane,
    )


def _stub_build(monkeypatch, build):
    monkeypatch.setattr(step_pipeline, "build_executing_batch", lambda services, batch: build(batch))


def _worker(monkeypatch=None):
    """A real ``Worker`` with what the lane calls around it recorded. Given
    ``monkeypatch``, its batch build is recorded too."""
    log: list = []
    worker = make_worker(
        device=torch.device("cpu"),
        wakeup_event=SimpleNamespace(register_future=lambda future: None),
        request_state=SimpleNamespace(get_partition_for_node=lambda node: node),
    )
    worker.log = log
    executor, pipeline = worker.executor, worker.pipeline
    executor.wait_for_step = lambda step: log.append("wait for step")
    executor.clear_in_flight = lambda batch: log.append("clear in flight")
    executor._set_in_flight = lambda batch, value: log.append(f"in flight {value}")
    executor.drop_failed_rids = lambda step, outputs, failed: log.append(("drop", sorted(failed)))
    executor.postprocess = lambda step, outputs, **kw: log.append(("postprocess", outputs, kw))
    executor._submit_gpu_step = lambda *a, **k: log.append("submit") or Future()
    executor._body = lambda *a: log.append("execute") or _OUTPUTS
    pipeline._fail_requests = lambda failed: log.append(("fail", sorted(failed)))
    pipeline._count_turn = lambda step: None
    worker.admission.clear_batch_limit = lambda *a, **k: log.append("clear limit")
    worker.scheduler.next_batch = lambda lane=CHAIN_LANE: None
    if monkeypatch is not None:
        _stub_build(monkeypatch, lambda batch: log.append("build") or SimpleNamespace(completion_event=None))
    return worker


# ----- streams --------------------------------------------------------------

def test_a_step_runs_on_the_lane_stream_with_no_fence_behind_the_main_one(monkeypatch):
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    executor = worker.executor
    executor._body = lambda *a: backend.log.append("execute") or _OUTPUTS
    executor.lane_stream(SIDE_LANE)

    assert executor._execute_on_lane(SIDE_LANE, _batch([1]), object(), None) is _OUTPUTS

    assert backend.log == ["enter side", "execute", "exit side"]


def test_the_window_collates_on_the_lane_stream_before_the_step_is_submitted(monkeypatch):
    """The step needs no fence because its one main-thread device op, stacking
    the window, runs on its own stream."""
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    _stub_build(monkeypatch, lambda batch: backend.log.append("build") or SimpleNamespace())
    worker.executor._submit_gpu_step = lambda *a, **k: backend.log.append("submit") or Future()
    worker.scheduler.next_batch = lambda lane=CHAIN_LANE: _batch([1])

    worker.pipeline._launch_side()

    assert backend.log == ["enter side", "build", "exit side", "submit"]


def test_every_step_shares_one_stream(monkeypatch):
    backend = _StreamBackend(monkeypatch)
    worker = _worker(monkeypatch)
    pipeline = worker.pipeline
    worker.scheduler.next_batch = lambda lane=CHAIN_LANE: _batch([1])

    pipeline._launch_side()
    pipeline.side_pending = _step([1])
    pipeline._finish_side()
    pipeline._launch_side()

    assert backend.streams == 1


def test_a_step_that_staged_host_rows_fails(monkeypatch):
    """Host rows stage through a ring the chain's next steps reuse before a
    side step lands; the startup refusal can't see an undeclared stop buffer."""
    _StreamBackend(monkeypatch)
    worker = _worker()
    executor = worker.executor
    executor._body = lambda *a: SimpleNamespace(host_rows=object(), host_spans=None)
    executor.lane_stream(SIDE_LANE)

    with pytest.raises(RuntimeError, match="stop buffers or output spans"):
        executor._execute_on_lane(SIDE_LANE, _batch([1]), object(), None)


def test_the_submitted_step_takes_the_lane_body_and_lifts_the_launch_fence(monkeypatch):
    _StreamBackend(monkeypatch)
    executor = make_worker(device=torch.device("cpu")).executor
    executor._body = lambda *a: _OUTPUTS
    executor.lane_stream(SIDE_LANE)
    node_batch = SimpleNamespace(launch_started_event=None)
    with ThreadPool(max_workers=1) as pool:
        executor._gpu_pool = pool
        future = executor._submit_gpu_step(_batch([1]), node_batch, None, SIDE_LANE)
        assert future.result() is _OUTPUTS
    assert node_batch.launch_started_event.wait(timeout=5)


# ----- one submit and finish for every lane ----------------------------------

def test_a_side_step_is_submitted_as_its_lane_and_flagged_in_flight():
    worker = _worker()

    step = worker.executor.submit(_batch([1]), SimpleNamespace(), lane=SIDE_LANE)

    assert step.lane == SIDE_LANE and step.partition == "Codec"
    assert worker.log == ["in flight True", "submit"]


def test_a_speculated_chain_step_is_not_flagged_again():
    """Its speculation flagged it in flight when it settled."""
    worker = _worker()

    step = worker.executor.submit(
        _batch([1]), SimpleNamespace(), speculated=True, partition="p", loop_name="decode",
    )

    assert step.lane == CHAIN_LANE and step.partition == "p" and step.loop_name == "decode"
    assert worker.log == ["submit"]


def test_finishing_a_step_lifts_its_limit_then_postprocesses():
    worker = _worker()

    worker.pipeline._finish_step(_step([1]), "outputs")
    worker.pipeline._finish_step(_step([2], lane=CHAIN_LANE), "outputs")

    assert worker.log == [
        "clear limit", ("postprocess", "outputs", {"clear_loop_stops": False}),
        "clear limit", ("postprocess", "outputs", {"clear_loop_stops": True}),
    ]


def test_rows_in_flight_are_every_lanes_and_an_error_fails_the_chains():
    pipeline = _worker().pipeline
    failed = []
    pipeline._fail_requests = failed.append

    pipeline._hold_rows(CHAIN_LANE, {1, 2})
    pipeline._hold_rows(SIDE_LANE, {2, 3})
    assert pipeline.in_flight_rids == {1, 2, 3} and pipeline._held_beside(CHAIN_LANE) == {2, 3}

    pipeline._handle_error(RuntimeError("boom"), (None, None), None)
    assert set(failed[0]) == {1, 2}

    pipeline._hold_rows(SIDE_LANE, ())
    assert pipeline.in_flight_rids == {1, 2}


# ----- one step in flight ---------------------------------------------------

def test_one_step_in_flight(monkeypatch):
    worker = _worker(monkeypatch)
    pipeline = worker.pipeline
    asked = []
    worker.scheduler.next_batch = lambda lane=CHAIN_LANE: asked.append(lane) or _batch([len(asked)])

    pipeline._launch_side()
    pipeline._launch_side()
    assert asked == [SIDE_LANE] and pipeline.in_flight_rids == {1}

    pipeline.side_pending = _step([1])
    pipeline._finish_side()
    pipeline._launch_side()
    assert asked == [SIDE_LANE, SIDE_LANE] and pipeline.in_flight_rids == {2}


def test_nothing_ready_launches_nothing():
    worker = _worker()

    worker.pipeline._launch_side()

    assert worker.pipeline.side_pending is None and worker.log == []


# ----- finishing a step -----------------------------------------------------

@pytest.mark.parametrize("step", [_step([1], landed=False), _step([1], done=False)], ids=["running", "launching"])
def test_a_step_still_running_is_left_alone(step):
    worker = _worker()
    worker.pipeline.side_pending = step

    worker.pipeline._finish_side()

    assert worker.pipeline.side_pending is step and worker.log == []


def test_postprocess_runs_after_the_step_has_landed_and_its_flags_are_down():
    worker = _worker()
    pipeline = worker.pipeline
    pipeline.side_pending = _step([1, 2])
    pipeline._hold_rows(SIDE_LANE, {1, 2})
    pipeline._hold_rows(CHAIN_LANE, {9})

    pipeline._finish_side()

    assert worker.log == [
        "wait for step", "clear in flight", "clear limit",
        ("postprocess", "outputs of [1, 2]", {"clear_loop_stops": False}),
    ]
    assert pipeline.side_pending is None and pipeline.in_flight_rids == {9}


def test_a_row_the_chain_still_holds_stays_in_flight():
    pipeline = _worker().pipeline
    pipeline.side_pending = _step([1, 2])
    pipeline._hold_rows(SIDE_LANE, {1, 2})
    pipeline._hold_rows(CHAIN_LANE, {2})

    pipeline._finish_side()

    assert pipeline.in_flight_rids == {2}


def test_block_waits_for_a_step_still_running():
    """Postprocess frees the step's inputs, main-stream allocations a
    main-stream step could reuse while this one still reads them."""
    worker = _worker()
    worker.pipeline.side_pending = _step([1], landed=False)

    worker.pipeline._finish_side(block=True)

    assert worker.log[0] == "wait for step"
    assert ("postprocess", "outputs of [1]", {"clear_loop_stops": False}) in worker.log


def test_failed_rows_are_reported_before_the_step_routes():
    worker = _worker()
    worker.pipeline.side_pending = _step([1, 2], failed={2: "boom"})

    worker.pipeline._finish_side()

    assert worker.log.index(("fail", [2])) < worker.log.index(
        ("postprocess", "outputs of [1, 2]", {"clear_loop_stops": False}),
    )


# ----- the chain's postprocess order ----------------------------------------

def _chain_worker():
    """A real ``StepExecutor.postprocess`` whose runtime keeps pending stops as the real ones do."""
    w = _postprocess_worker(refuse=set())
    stops = {0}

    def clear():
        stops.clear()

    w._graph_runtime.clear_pending_loop_stops = clear
    w._graph_runtime.pending_loop_stop_rids = lambda walks, loop_name: set(stops)
    w.admission.clear_batch_limit = lambda *a, **k: None
    return w


def _lane_step(rids, lane):
    step, outputs = _decode_step(rids)
    step.lane, step.submitted_at = lane, 0.0
    return step, outputs


def test_a_side_step_between_two_chain_steps_leaves_the_loop_stop_in_place():
    """The chain's step past a stop must still be dropped when a side step's
    postprocess lands between the stop and that step."""
    w = _chain_worker()
    side, side_outputs = _lane_step([5], SIDE_LANE)
    past_the_stop, outputs = _lane_step([0, 1], CHAIN_LANE)
    past_the_stop.speculative_new_iter = True

    w.pipeline._finish_step(side, side_outputs)
    w.pipeline._finish_step(past_the_stop, outputs)

    assert w.log.routed == [[5], [1]]


def test_the_chain_still_clears_its_stops():
    w = _chain_worker()
    step, outputs = _lane_step([1], CHAIN_LANE)

    w.pipeline._finish_step(step, outputs)

    assert w._graph_runtime.pending_loop_stop_rids([], "decode") == set()


# ----- rollback -------------------------------------------------------------

def _raising_step(rids):
    step = _step(rids, done=False)
    step.future.set_exception(RuntimeError("boom"))
    return step


def test_a_step_that_raised_fails_its_own_rows_and_clears_out():
    worker = _worker()
    pipeline = worker.pipeline
    pipeline.side_pending = _raising_step([3])
    pipeline._hold_rows(SIDE_LANE, {3})
    pipeline._hold_rows(CHAIN_LANE, {9})

    pipeline._finish_side()

    assert ("fail", [3]) in worker.log and "clear in flight" in worker.log
    assert pipeline.side_pending is None and pipeline.in_flight_rids == {9}


def test_a_step_that_could_not_launch_fails_its_rows(monkeypatch):
    worker = _worker()
    worker.scheduler.next_batch = lambda lane=CHAIN_LANE: _batch([3])

    def refuse(batch):
        raise RuntimeError("boom")

    _stub_build(monkeypatch, refuse)
    pipeline = worker.pipeline

    pipeline._launch_side()

    assert ("fail", [3]) in worker.log and pipeline.side_pending is None and pipeline.in_flight_rids == set()


def test_a_main_loop_error_leaves_the_side_step_to_finish():
    pipeline = _worker().pipeline
    failed = []
    pipeline._fail_requests = failed.append
    side = pipeline.side_pending = _step([3])
    pipeline._hold_rows(SIDE_LANE, {3})
    pipeline._hold_rows(CHAIN_LANE, {1})

    pipeline._handle_error(RuntimeError("boom"), (_step([1], lane=CHAIN_LANE), None), None)

    assert set(failed[0]) == {1} and pipeline.side_pending is side and pipeline.in_flight_rids == {1, 3}


def test_a_row_in_both_steps_fails_with_the_chain():
    pipeline = _worker().pipeline
    failed = []
    pipeline._fail_requests = failed.append
    pipeline.side_pending = _step([1])
    pipeline._hold_rows(SIDE_LANE, {1})
    pipeline._hold_rows(CHAIN_LANE, {1})

    pipeline._handle_error(RuntimeError("boom"), (_step([1], lane=CHAIN_LANE), None), None)

    assert set(failed[0]) == {1}


# ----- the scheduler ---------------------------------------------------------

def _lanes(sched, key=(NODE, WALK)):
    sched.lane_of = {key: SIDE_LANE}
    return sched


def test_a_side_walk_does_not_break_the_chain():
    sched = _lanes(_scheduler(_Engine()))
    manager = _Manager([1, 2])

    assert not _has_ready(sched, manager, exclude_target=("other", "walk"))


def test_a_side_walk_is_scheduled_on_its_lane_alone():
    sched = _lanes(_scheduler(_Engine()))
    manager = _Manager([1, 2])

    assert _next_batch(sched, manager) is None
    assert sorted(_next_batch(sched, manager, lane=SIDE_LANE).request_to_worker_graph) == [1, 2]


def test_a_side_walks_backlog_stays_on_its_lane():
    sched = _lanes(_scheduler(_Engine(max_bs=1)))
    manager = _Manager([1, 2])

    first = _next_batch(sched, manager, lane=SIDE_LANE)
    assert sched.backlog, "the cap should leave a row behind"
    assert not _has_ready(sched, manager, exclude_target=("other", "walk"))
    assert _next_batch(sched, manager) is None
    second = _next_batch(sched, manager, lane=SIDE_LANE)
    assert {*first.request_to_worker_graph, *second.request_to_worker_graph} == {1, 2}


def test_a_lane_never_takes_the_chains_keys():
    sched = _scheduler(_Engine())
    manager = _Manager([1])

    assert _next_batch(sched, manager, lane=SIDE_LANE) is None
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


def _engine(backend, side=("Codec",)):
    engine = make_engine(_device_capability=SimpleNamespace(graph_backend=backend), _side_stream_nodes=frozenset(side))
    return engine


def test_a_side_node_captures_on_a_stream_of_its_own():
    """Graphs captured on one stream must not overlap, and a side step overlaps
    every graph of the chain."""
    backend = _GraphBackend()

    assert _engine(backend)._capture_stream_for("Codec") == "stream 1"


def test_a_node_on_the_chain_keeps_the_shared_capture_stream():
    backend = _GraphBackend()

    assert _engine(backend)._capture_stream_for("Talker") is None
    assert _engine(backend, side=())._capture_stream_for("Codec") is None
    assert backend.made == 0


def test_no_graph_backend_means_no_capture_stream():
    assert _engine(None)._capture_stream_for("Codec") is None


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


# ----- the config field and the startup decision ------------------------------

def _codec_node(name="Codec"):
    return GraphNode(
        name=name, input_names=["codec_tokens"],
        outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk", output_modality="audio")],
    )


class _Model(StubModel):
    def __init__(self, walks=None, combined=None):
        if walks is None:
            walks = {"codec_chunk": _codec_node(), "talker_decode": _codec_node("Talker")}
        self.walks = walks
        self.combined = combined or {}

    def get_graph_walk_graphs(self):
        return self.walks

    def combined_walk_of(self):
        return self.combined


class _Codec:
    side_stream_walks = frozenset({"codec_chunk"})
    token_stop = None
    device_loopback = None

    def spanned_outputs(self, walk):
        return frozenset()


class _Groups:
    def __init__(self, world=1):
        self.world = world

    def get_instance_world_size_for_node(self, node):
        return self.world


def _keys(submodules=None, specs=(), model=None, nodes=("Codec", "Talker"), asked=("Codec",),
          groups=None, accelerator=True):
    return _side_lane_keys(
        submodules or {"Codec": _Codec(), "Talker": object()}, list(specs), model or _Model(),
        nodes, asked, groups or _Groups(), accelerator,
    )


def test_a_stream_fed_walk_beside_another_node_takes_the_lane():
    assert _keys() == (SIDE,)


def test_a_node_that_does_not_ask_stays_on_the_chain_whatever_it_holds():
    """So a codec with a recurrent pool boots on a config without the field."""
    held = [SimpleNamespace(resource_key="codec_state", nodes={"Codec"})]
    assert _keys(specs=held, asked=()) == ()


def test_the_field_is_read_per_node_and_must_agree():
    groups = [{"node_names": ["Talker"]}, {"node_names": ["Codec"], "side_stream": True}]
    assert side_stream_nodes(groups) == {"Codec"}
    with pytest.raises(ValueError, match="different side_stream"):
        side_stream_nodes(groups + [{"node_names": ["Codec"], "side_stream": False}])
    with pytest.raises(ValueError, match="true or false"):
        side_stream_nodes([{"node_names": ["Codec"], "side_stream": "yes"}])


class _Plain(_Codec):
    side_stream_walks = frozenset()


class _Spans(_Codec):
    def spanned_outputs(self, walk):
        return frozenset({"audio_chunk"})


class _Stops(_Codec):
    token_stop = TokenStop(loop="decode", buffer="tok", stop_ids=frozenset({0}), sampler="s")


class _Loops(_Codec):
    device_loopback = DeviceLoopback(walk="codec_chunk", signal="tok", sampler="s", buffer="tok")


class _Vocoder(_Codec):
    side_stream_walks = frozenset({"vocode"})


_LOOP = Loop(name="decode_loop", section=_codec_node(), max_iters=4, outputs=[])


@pytest.mark.parametrize("kwargs, reason", [
    (dict(submodules={"Codec": _Plain(), "Talker": object()}), "declares no side_stream_walks"),
    (dict(groups=_Groups(world=2)), "runs in a lockstep instance"),
    (dict(nodes=("Codec",)), "only node on its worker"),
    (dict(accelerator=False), "no streams"),
    (dict(model=_Model({"codec_chunk": _codec_node(), "codec_warm": _codec_node()})), "also runs"),
    (dict(specs=[SimpleNamespace(resource_key="codec_state", nodes={"Codec"})]), "holds resources"),
    (dict(model=_Model({"codec_chunk": _LOOP})), "not a walk of that one node"),
    (dict(model=_Model({"codec_chunk": _codec_node("Talker")})), "not a walk of that one node"),
    (dict(model=_Model(combined={SIDE: "codec_mixed"})), "combined walk"),
    (dict(submodules={"Codec": _Spans(), "Talker": object()}), "output spans or a stop check"),
    (dict(submodules={"Codec": _Stops(), "Talker": object()}), "output spans or a stop check"),
    (dict(submodules={"Codec": _Loops(), "Talker": object()}), "output spans or a stop check"),
    (dict(submodules={"Codec": _Codec(), "Vocoder": _Vocoder(), "Talker": object()},
          nodes=("Codec", "Vocoder", "Talker"), asked=("Codec", "Vocoder"),
          model=_Model({"codec_chunk": _codec_node(), "vocode": _codec_node("Vocoder")})),
     "one side lane"),
])
def test_what_the_lane_cannot_serve_is_refused_naming_the_field(kwargs, reason):
    with pytest.raises(ValueError, match=reason) as raised:
        _keys(**kwargs)
    assert "sets side_stream" in str(raised.value) or "side_stream is set" in str(raised.value)


def test_a_cpu_build_that_asks_for_the_lane_fails_at_startup():
    """Through the real ``EngineManager.build``."""
    class Model(_Model):
        def get_autocast_dtype(self):
            return None

        def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
            return _Submodule() if node_name == "Codec" else _Node()

        def declare_state(self):
            return []

        def kernel_settings(self):
            return {}

        def get_node_resources(self):
            return []

        def chunking_policy_of(self):
            return {}

    class _Node(NodeSubmodule):
        def prepare_inputs(self, *args, **kwargs):
            raise NotImplementedError

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class _Submodule(_Node):
        side_stream_walks = frozenset({"codec_chunk"})

    with pytest.raises(ValueError, match="no streams"):
        EngineManager.build(
            node_names={"Codec", "Talker"},
            device=torch.device("cpu"),
            model_config={"node_groups": [
                {"node_names": ["Talker"], "ranks": [0]},
                {"node_names": ["Codec"], "ranks": [0], "side_stream": True},
            ]},
            parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
            transfer_engine_info=TransferEngineInfo(my_entity_id="t", my_session_id="t", transfer_engine=None),
            model=Model(),
            graph_runtime=SimpleNamespace(),
        )
