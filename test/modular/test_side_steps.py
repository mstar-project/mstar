"""Side lanes: a node's steps beside the worker's main chain (``mstar.worker.side_steps``).

The stream and event backends are mocked, so these run without a GPU; what they
pin is where a side step's kernels and its window's collate go, when a step may
be postprocessed, what a step off the chain must leave alone, the scheduler's
lane filter, and which configs the startup decision refuses.
"""
from __future__ import annotations

import contextlib
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from test_micro_scheduler import NODE, WALK, _Engine, _has_ready, _Manager, _next_batch, _scheduler
from test_send_registration_failure import _decode_step, _postprocess_worker

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine import accelerator_graph_runner as runner_module
from mstar.engine.engine import Engine, ExecutingBatch, SubmoduleManagement
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import AllocationFailed, Segment, StepContext, SubmoduleStep
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import BatchedModelOutput, DeviceLoopback, NodeSubmodule, TokenStop
from mstar.utils.threads import ThreadPool
from mstar.worker import side_steps
from mstar.worker.engine_manager import EngineManager, _side_lane_keys, side_stream_nodes
from mstar.worker.micro_scheduler import CHAIN_LANE, ScheduledBatch
from mstar.worker.side_steps import SIDE_LANE, SideStepLane
from mstar.worker.worker import PendingBatch, Worker

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

        monkeypatch.setattr(side_steps.torch, "Stream", Stream)


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
            admit_error=None,
        ),
        node_name=SIDE[0], partition="Codec", graph_walk=SIDE[1], future=future, lane=lane,
    )


def _worker():
    """A real ``Worker`` with what the lane calls around it recorded."""
    log: list = []
    worker = Worker.__new__(Worker)
    worker.log = log
    worker.worker_id = "w0"
    worker.device = torch.device("cpu")
    worker._in_flight_rids = set()
    worker._phase_period = 0
    worker._span = lambda *a, **k: contextlib.nullcontext()
    worker._wait_for_step = lambda step: log.append("wait for step")
    worker._clear_in_flight_flag = lambda batch: log.append("clear in flight")
    worker._set_in_flight_flag = lambda batch, value: log.append(f"in flight {value}")
    worker._drop_failed_rids = lambda step, outputs, failed: log.append(("drop", sorted(failed)))
    worker._fail_requests = lambda failed: log.append(("fail", sorted(failed)))
    worker._postprocess_batch = lambda step, outputs, **kw: log.append(("postprocess", outputs, kw))
    worker._count_turn = lambda step: None
    worker._build_executing_batch = lambda batch: log.append("build") or SimpleNamespace(completion_event=None)
    worker._submit_gpu_step = lambda *a, **k: log.append("submit") or Future()
    worker._execute_on_gpu_thread = lambda *a: log.append("execute") or _OUTPUTS
    worker.wakeup_event = SimpleNamespace(register_future=lambda future: None)
    worker.request_state = SimpleNamespace(get_partition_for_node=lambda node: node)
    worker.scheduler = SimpleNamespace(
        clear_batch_limit=lambda *a, **k: log.append("clear limit"),
        get_next_batch=lambda state, lane=CHAIN_LANE: None,
    )
    worker.side_streams = []
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        set_side_stream=lambda node, stream: worker.side_streams.append(node),
    ))
    return worker


# ----- streams --------------------------------------------------------------

def test_a_step_runs_on_the_lane_stream_with_no_fence_behind_the_main_one(monkeypatch):
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    worker._execute_on_gpu_thread = lambda *a: backend.log.append("execute") or _OUTPUTS
    lane = SideStepLane(worker)
    lane._stream = side_steps.torch.Stream(worker.device)

    assert lane.execute(_batch([1]), object(), None) is _OUTPUTS

    assert backend.log == ["enter side", "execute", "exit side"]


def test_the_window_collates_on_the_lane_stream_before_the_step_is_submitted(monkeypatch):
    """The step needs no fence because its one main-thread device op, stacking
    the window, runs on its own stream."""
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    worker.log = backend.log
    worker._build_executing_batch = lambda batch: backend.log.append("build") or SimpleNamespace()
    worker._submit_gpu_step = lambda *a, **k: backend.log.append("submit") or Future()
    worker.scheduler.get_next_batch = lambda state, lane=CHAIN_LANE: _batch([1])
    lane = SideStepLane(worker)

    lane.launch(None)

    assert backend.log == ["enter side", "build", "exit side", "submit"]


def test_every_step_shares_one_stream_and_the_nodes_teardowns_run_on_it(monkeypatch):
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    worker.scheduler.get_next_batch = lambda state, lane=CHAIN_LANE: _batch([1])
    lane = SideStepLane(worker)

    lane.launch(None)
    lane.pending = _step([1])
    lane.finish()
    lane.launch(None)

    assert backend.streams == 1 and worker.side_streams == ["Codec"]


def test_a_step_that_staged_host_rows_fails(monkeypatch):
    """Host rows stage through a ring the chain's next steps reuse before a
    side step lands; the startup refusal can't see an undeclared stop buffer."""
    _StreamBackend(monkeypatch)
    worker = _worker()
    worker._execute_on_gpu_thread = lambda *a: SimpleNamespace(host_rows=object(), host_spans=None)
    lane = SideStepLane(worker)
    lane._stream = side_steps.torch.Stream(worker.device)

    with pytest.raises(RuntimeError, match="stop buffers or output spans"):
        lane.execute(_batch([1]), object(), None)


def test_the_submitted_step_takes_the_lane_body_and_lifts_the_launch_fence(monkeypatch):
    _StreamBackend(monkeypatch)
    worker = Worker.__new__(Worker)
    worker.device = torch.device("cpu")
    worker._execute_on_gpu_thread = lambda *a: _OUTPUTS
    lane = SideStepLane(worker)
    lane._stream = side_steps.torch.Stream(worker.device)
    node_batch = SimpleNamespace(launch_started_event=None)
    with ThreadPool(max_workers=1) as pool:
        future = Worker._submit_gpu_step(worker, pool, _batch([1]), node_batch, None, execute=lane.execute)
        assert future.result() is _OUTPUTS
    assert node_batch.launch_started_event.wait(timeout=5)


# ----- one submit and finish for every lane ----------------------------------

def test_a_side_step_is_submitted_as_its_lane_and_flagged_in_flight():
    worker = _worker()

    step = worker._submit_step(None, _batch([1]), SimpleNamespace(), lane=SIDE_LANE, execute=print)

    assert step.lane == SIDE_LANE and step.partition == "Codec"
    assert worker.log == ["in flight True", "submit"]


def test_a_speculated_chain_step_is_not_flagged_again():
    """Its speculation flagged it in flight when it settled."""
    worker = _worker()

    step = worker._submit_step(
        None, _batch([1]), SimpleNamespace(), speculated=True, partition="p", loop_name="decode",
    )

    assert step.lane == CHAIN_LANE and step.partition == "p" and step.loop_name == "decode"
    assert worker.log == ["submit"]


def test_finishing_a_step_lifts_its_limit_then_postprocesses():
    worker = _worker()

    worker._finish_step(_step([1]), "outputs")
    worker._finish_step(_step([2], lane=CHAIN_LANE), "outputs")

    assert worker.log == [
        "clear limit", ("postprocess", "outputs", {"clear_loop_stops": False}),
        "clear limit", ("postprocess", "outputs", {"clear_loop_stops": True}),
    ]


def test_rows_in_flight_are_every_lanes_and_an_error_fails_the_chains():
    worker = _worker()
    failed = []
    worker._fail_requests = failed.append

    worker._hold_rows(CHAIN_LANE, {1, 2})
    worker._hold_rows(SIDE_LANE, {2, 3})
    assert worker._in_flight_rids == {1, 2, 3} and worker._held_beside(CHAIN_LANE) == {2, 3}

    worker._handle_main_loop_error(RuntimeError("boom"), (None, None), None)
    assert set(failed[0]) == {1, 2}

    worker._hold_rows(SIDE_LANE, ())
    assert worker._in_flight_rids == {1, 2}


# ----- one step in flight ---------------------------------------------------

def test_one_step_in_flight():
    worker = _worker()
    asked = []
    worker.scheduler.get_next_batch = lambda state, lane=CHAIN_LANE: asked.append(lane) or _batch([len(asked)])
    lane = SideStepLane(worker)

    lane.launch(None)
    lane.launch(None)
    assert asked == [SIDE_LANE] and worker._in_flight_rids == {1}

    lane.pending = _step([1])
    lane.finish()
    lane.launch(None)
    assert asked == [SIDE_LANE, SIDE_LANE] and worker._in_flight_rids == {2}


def test_nothing_ready_launches_nothing():
    worker = _worker()
    lane = SideStepLane(worker)

    lane.launch(None)

    assert lane.pending is None and worker.log == []


# ----- finishing a step -----------------------------------------------------

@pytest.mark.parametrize("step", [_step([1], landed=False), _step([1], done=False)], ids=["running", "launching"])
def test_a_step_still_running_is_left_alone(step):
    worker = _worker()
    lane = SideStepLane(worker)
    lane.pending = step

    lane.finish()

    assert lane.pending is step and worker.log == []


def test_postprocess_runs_after_the_step_has_landed_and_its_flags_are_down():
    worker = _worker()
    lane = SideStepLane(worker)
    lane.pending = _step([1, 2])
    worker._hold_rows(SIDE_LANE, {1, 2})
    worker._hold_rows(CHAIN_LANE, {9})

    lane.finish()

    assert worker.log == [
        "wait for step", "clear in flight", "clear limit",
        ("postprocess", "outputs of [1, 2]", {"clear_loop_stops": False}),
    ]
    assert lane.pending is None and worker._in_flight_rids == {9}


def test_a_row_the_chain_still_holds_stays_in_flight():
    worker = _worker()
    lane = SideStepLane(worker)
    lane.pending = _step([1, 2])
    worker._hold_rows(SIDE_LANE, {1, 2})
    worker._hold_rows(CHAIN_LANE, {2})

    lane.finish()

    assert worker._in_flight_rids == {2}


def test_block_waits_for_a_step_still_running():
    """Postprocess frees the step's inputs, main-stream allocations a
    main-stream step could reuse while this one still reads them."""
    worker = _worker()
    lane = SideStepLane(worker)
    lane.pending = _step([1], landed=False)

    lane.finish(block=True)

    assert worker.log[0] == "wait for step"
    assert ("postprocess", "outputs of [1]", {"clear_loop_stops": False}) in worker.log


def test_failed_rows_are_reported_before_the_step_routes():
    worker = _worker()
    lane = SideStepLane(worker)
    lane.pending = _step([1, 2], failed={2: "boom"})

    lane.finish()

    assert worker.log.index(("fail", [2])) < worker.log.index(
        ("postprocess", "outputs of [1, 2]", {"clear_loop_stops": False}),
    )


def test_a_step_its_pool_refused_goes_back_to_wait_and_routes_nothing():
    """It ran no forward: its rows return to the scheduler as a refused
    chain step's do (``Worker._handle_admit_failure``), and leave the lane."""
    worker = _worker()
    refused = []
    worker._handle_admit_failure = lambda batch, node_batch: refused.append(sorted(batch.request_to_worker_graph))
    lane = SideStepLane(worker)
    step = lane.pending = _step([1, 2])
    step.node_batch.admit_error = AllocationFailed(message="full", pages_short=1, label="main", request_id=2)
    worker._hold_rows(SIDE_LANE, {1, 2})

    lane.finish()

    assert refused == [[1, 2]]
    assert worker.log == ["wait for step", "clear in flight"]
    assert lane.pending is None and worker._in_flight_rids == set()


# ----- the chain's postprocess order ----------------------------------------

def _chain_worker():
    """A real ``_postprocess_batch`` whose runtime keeps pending stops as the real ones do."""
    w = _postprocess_worker(refuse=set())
    stops = {0}

    def clear():
        stops.clear()

    w._graph_runtime.clear_pending_loop_stops = clear
    w._graph_runtime.pending_loop_stop_rids = lambda walks, loop_name: set(stops)
    w.scheduler.clear_batch_limit = lambda *a, **k: None
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

    w._finish_step(side, side_outputs)
    w._finish_step(past_the_stop, outputs)

    assert w.log.routed == [[5], [1]]


def test_the_chain_still_clears_its_stops():
    w = _chain_worker()
    step, outputs = _lane_step([1], CHAIN_LANE)

    w._finish_step(step, outputs)

    assert w._graph_runtime.pending_loop_stop_rids([], "decode") == set()


# ----- rollback -------------------------------------------------------------

def _raising_step(rids):
    step = _step(rids, done=False)
    step.future.set_exception(RuntimeError("boom"))
    return step


def test_a_step_that_raised_fails_its_own_rows_and_clears_out():
    worker = _worker()
    lane = SideStepLane(worker)
    lane.pending = _raising_step([3])
    worker._hold_rows(SIDE_LANE, {3})
    worker._hold_rows(CHAIN_LANE, {9})

    lane.finish()

    assert ("fail", [3]) in worker.log and "clear in flight" in worker.log
    assert lane.pending is None and worker._in_flight_rids == {9}


def test_a_step_that_could_not_launch_fails_its_rows():
    worker = _worker()
    worker.scheduler.get_next_batch = lambda state, lane=CHAIN_LANE: _batch([3])

    def refuse(batch):
        raise RuntimeError("boom")

    worker._build_executing_batch = refuse
    lane = SideStepLane(worker)

    lane.launch(None)

    assert ("fail", [3]) in worker.log and lane.pending is None and worker._in_flight_rids == set()


def test_a_main_loop_error_leaves_the_side_step_to_finish():
    w = _worker()
    failed = []
    w._fail_requests = failed.append
    lane = w._side_steps = SideStepLane(w)
    side = lane.pending = _step([3])
    w._hold_rows(SIDE_LANE, {3})
    w._hold_rows(CHAIN_LANE, {1})

    w._handle_main_loop_error(RuntimeError("boom"), (_step([1], lane=CHAIN_LANE), None), None)

    assert set(failed[0]) == {1} and lane.pending is side and w._in_flight_rids == {1, 3}


def test_a_row_in_both_steps_fails_with_the_chain():
    w = _worker()
    failed = []
    w._fail_requests = failed.append
    lane = w._side_steps = SideStepLane(w)
    lane.pending = _step([1])
    w._hold_rows(SIDE_LANE, {1})
    w._hold_rows(CHAIN_LANE, {1})

    w._handle_main_loop_error(RuntimeError("boom"), (_step([1], lane=CHAIN_LANE), None), None)

    assert set(failed[0]) == {1}


# ----- a step that holds a recurrent pool -------------------------------------
#
# A real RecurrentStatePool and StepRunner behind a hand-built Engine, driven
# through the engine's own declare, admit, plan, forward and commit.

POOL = "codec_state"


def _pool(key=POOL, node="Codec", max_slots=3):
    """The sink plus ``max_slots - 1`` streams of a two-float state."""
    spec = RecurrentStateSpec(key, {node}, RecurrentStateConfig(
        num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)},
        max_slots=max_slots,
    ))
    return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))


class _StateSub(NodeSubmodule):
    """The codec's declaration: a row per stream, reading and writing its slot."""

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        del graph_walk, inputs, kwargs
        return SubmoduleStep(segments=[Segment(rid, "main", 1) for rid in request_ids], steps={POOL: RecurrentStep()})

    def prepare_inputs(self, *a, **kw):  # pragma: no cover - the test builds inputs
        raise NotImplementedError

    def forward(self, *a, **kw):  # pragma: no cover - forward_batched runs
        raise NotImplementedError

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {}


def _advance(graph_walk, engine_inputs, **kwargs):
    """Each row reads its stream's state and writes it back one higher, as a
    window moves its decoder state on."""
    del graph_walk, kwargs
    pool = engine_inputs.resources[POOL]
    state = pool.gather("state")
    pool.scatter_("state", state + 1)
    return BatchedModelOutput(per_rid_outputs={
        rid: {"audio_chunk": [state[row]]} for row, rid in enumerate(engine_inputs.request_ids)
    })


def _state_engine(pool, chain_pool=None):
    resources = {POOL: pool, **({"talker_state": chain_pool} if chain_pool is not None else {})}
    eng = Engine.__new__(Engine)
    eng._runner = StepRunner(resources, node_resources={
        "Codec": [POOL], "Talker": ["talker_state"] if chain_pool is not None else [],
    }, side_resources={POOL})
    eng._resources = resources
    eng._enable_nvtx = eng._enable_profile = False
    eng._device = torch.device("cpu")
    eng._pre_plan_done = {}
    eng._loopback_steps = {}
    eng._chunking = SimpleNamespace(forget_request=lambda rid: None, release=lambda rid, node: None)
    eng._submodules = {"Codec": SubmoduleManagement(
        submodule=_StateSub(), forward=None, forward_batched=_advance,
        joint_comm_group=None, resources={POOL: pool},
    )}
    return eng


def _side_batch(rids) -> ExecutingBatch:
    batch = ExecutingBatch(
        node_name="Codec", per_request_info={rid: SimpleNamespace() for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk="codec_chunk", slot=0, capture=False),
    )
    batch.inputs = [SimpleNamespace(input_seq_len=1, tensor_inputs={}, kwargs={})] * len(rids)
    batch.running_batched = True
    return batch


def _logged(log, obj, *names):
    """Log each call of ``obj``'s methods ``names``, then make it."""
    for name in names:
        method = getattr(obj, name)
        setattr(obj, name, lambda *a, _m=method, _n=name, **k: log.append(_n) or _m(*a, **k))


def _side_lane(eng, monkeypatch):
    """A lane whose GPU-thread step is the engine's exec, on a recorded stream."""
    backend = _StreamBackend(monkeypatch)
    worker = _worker()
    worker._execute_on_gpu_thread = lambda batch, node_batch, plan_future: eng.exec(node_batch) and _OUTPUTS
    lane = SideStepLane(worker)
    lane._stream = side_steps.torch.Stream(worker.device)
    eng.set_side_stream("Codec", lane._stream)
    return lane, backend


def test_a_side_step_takes_plans_and_commits_its_slot_on_the_lane_stream(monkeypatch):
    pool = _pool()
    eng = _state_engine(pool)
    eng.add_request(1)
    free = pool.num_free_slots
    lane, backend = _side_lane(eng, monkeypatch)
    _logged(backend.log, pool, "admit", "plan", "commit", "_zero_slot")

    first, second = _side_batch([1]), _side_batch([1])
    lane.execute(_batch([1]), first, None)
    slot = pool._slots[1]["main"]
    lane.execute(_batch([1]), second, None)

    assert slot.index != SINK_SLOT and pool.num_free_slots == free - 1
    assert backend.log == ["enter side", "admit", "plan", "commit", "exit side"] * 2
    # a fresh stream reads zeros, and its next window what the first one wrote
    assert first.outputs.per_rid_outputs[1]["audio_chunk"][0].tolist() == [0.0, 0.0]
    assert second.outputs.per_rid_outputs[1]["audio_chunk"][0].tolist() == [1.0, 1.0]
    assert slot.has_state and slot.tokens == 2

    eng.remove_request(1)

    assert pool.num_free_slots == free and 1 not in pool._slots
    assert backend.log[-3:] == ["enter side", "_zero_slot", "exit side"], (
        "the slot was zeroed off the lane's stream, where the next stream's first write could land ahead of it"
    )


def test_a_side_step_leaves_the_chains_staged_pre_plan_alone(monkeypatch):
    """The plan thread stages the chain's next step while a side step admits
    and plans on the GPU thread; the side step neither promotes nor drops it."""
    pool, talker = _pool(), _pool("talker_state", "Talker")
    eng = _state_engine(pool, talker)
    eng.add_request(1)
    eng.add_request(9)
    chain = SubmoduleStep(segments=[Segment(9, "main", 1)], steps={"talker_state": RecurrentStep()})
    chain.set_ctx(StepContext(request_ids=(9,), graph_walk="talker_decode", slot=0, capture=False))
    chain.ctx.is_preplan = True
    assert eng._runner.pre_admit(chain).ok
    eng._runner.pre_plan(chain)
    chain.ctx.is_preplan = False
    lane, _ = _side_lane(eng, monkeypatch)

    lane.execute(_batch([1]), _side_batch([1]), None)

    assert eng._runner.staged and talker._preplanned, "the side step dropped the chain's stage"
    assert eng._runner.admit(chain).ok
    staged = talker._cached_plan_output
    assert eng._runner.plan(chain)["talker_state"] is staged, "the chain's step planned afresh"


@pytest.mark.parametrize("stage", ["admit", "plan"])
def test_a_raise_in_a_side_steps_admit_or_plan_fails_its_own_rows(monkeypatch, stage):
    pool, talker = _pool(), _pool("talker_state", "Talker")
    eng = _state_engine(pool, talker)
    for rid in (1, 9):
        eng.add_request(rid)
    chain = SubmoduleStep(segments=[Segment(9, "main", 1)], steps={"talker_state": RecurrentStep()})
    chain.set_ctx(StepContext(request_ids=(9,), graph_walk="talker_decode", slot=0, capture=False))
    chain.ctx.is_preplan = True
    eng._runner.pre_admit(chain)
    eng._runner.pre_plan(chain)
    chain.ctx.is_preplan = False

    def boom(*a, **k):
        raise RuntimeError(f"{stage} failed")

    monkeypatch.setattr(pool, stage, boom)
    lane, _ = _side_lane(eng, monkeypatch)
    worker = lane._worker
    lane.pending = _step([1], done=False)
    try:
        lane.execute(_batch([1]), _side_batch([1]), None)
    except RuntimeError as exc:
        lane.pending.future.set_exception(exc)
    worker._hold_rows(SIDE_LANE, {1})
    worker._hold_rows(CHAIN_LANE, {9})

    lane.finish()

    assert ("fail", [1]) in worker.log and not any(e[0] == "postprocess" for e in worker.log if isinstance(e, tuple))
    assert worker._in_flight_rids == {9} and lane.pending is None
    assert eng._runner.staged and talker._preplanned, "the chain's stage went with the side step's raise"


class _PoolEngine(_Engine):
    """The scheduler's view of an engine whose ``NODE`` holds a real pool:
    readiness asks its admission, as ``Engine.check_ready`` does."""

    def __init__(self, eng):
        super().__init__()
        self.eng = eng

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        if node_name != NODE:
            return super().check_ready(node_name, rid, fwd_info, allow_reload)
        return self.eng._runner.admit_retrieve(rid, "Codec", WALK)

    def admit_waiting(self, published_of):
        return [(NODE, rid) for _, rid in self.eng.admit_waiting(published_of)]


def test_a_full_pool_keeps_the_row_queued_while_the_chain_runs_on():
    """A stream that finds every slot taken waits at readiness: it stays on its
    queue, the chain's rows keep scheduling, and it runs once a slot frees."""
    pool = _pool(max_slots=2)  # one stream beside the sink
    eng = _state_engine(pool)
    sched = _lanes(_scheduler(_PoolEngine(eng)))
    sched.parallel_leader_nodes = {NODE, "Talker"}
    for rid in (1, 2, 7):
        eng.add_request(rid)
    assert sorted(_next_batch(sched, _Manager([1]), lane=SIDE_LANE).request_to_worker_graph) == [1]
    eng.exec(_side_batch([1]))  # its first window takes the slot
    codec = _Manager([2])
    codec.get_fwd_info = lambda rid, partition: SimpleNamespace(resource_publish_info=None)

    assert _next_batch(sched, codec, lane=SIDE_LANE) is None
    assert sched.parked.get(NODE) == {2} and 2 in codec.queues["wg0"]._ready
    assert sorted(_next_batch(sched, _Manager([7], node="Talker", walk="talker_decode")).request_to_worker_graph) == [7]

    eng.remove_request(1)

    assert sorted(_next_batch(sched, codec, lane=SIDE_LANE).request_to_worker_graph) == [2]


def _teardown_worker():
    from test_removed_request_messages import _new
    from test_removed_request_messages import _worker as _messages_worker

    w = _messages_worker()
    w._process_message_list([_new("A#1")])
    return w


def test_a_teardown_waits_for_the_side_steps_admit():
    """The step admits on the GPU thread from the state a teardown releases."""
    from test_removed_request_messages import _remove

    w = _teardown_worker()
    admitted = threading.Event()
    seen = []
    w.engine_manager.remove_request = lambda rid: seen.append(admitted.is_set())
    lane = w._side_steps = SideStepLane(w)
    lane.pending = _step([5], done=False)
    lane.pending.node_batch.launch_started_event = admitted
    w._phase_period = 0
    w._span = lambda *a, **k: contextlib.nullcontext()
    threading.Timer(0.2, admitted.set).start()

    w._process_message_list([_remove("A#1")])

    assert seen == [True], "the teardown ran while the side step had yet to admit"


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
    engine = Engine.__new__(Engine)
    engine._device_capability = SimpleNamespace(graph_backend=backend)
    engine._side_stream_nodes = frozenset(side)
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


class _Model:
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

    def resident_walks(self):
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


def _pool_spec(nodes=("Codec",), kv_cache=None):
    return RecurrentStateSpec(POOL, set(nodes), RecurrentStateConfig(
        num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)}, kv_cache=kv_cache,
    ))


def test_a_node_with_a_recurrent_pool_of_its_own_takes_the_lane():
    assert _keys(specs=[_pool_spec()]) == (SIDE,)


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


class _Resident(_Codec):
    def resident_walks(self):
        return frozenset({"codec_chunk"})


_LOOP = Loop(name="decode_loop", section=_codec_node(), max_iters=4, outputs=[])


@pytest.mark.parametrize("kwargs, reason", [
    (dict(submodules={"Codec": _Plain(), "Talker": object()}), "declares no side_stream_walks"),
    (dict(groups=_Groups(world=2)), "tensor or sequence parallel"),
    (dict(nodes=("Codec",)), "only node on its worker"),
    (dict(accelerator=False), "no streams"),
    (dict(model=_Model({"codec_chunk": _codec_node(), "codec_warm": _codec_node()})), "also runs"),
    (dict(specs=[SimpleNamespace(resource_key="codec_kv", nodes={"Codec"})]), "holds resources .'codec_kv'"),
    (dict(specs=[_pool_spec(), SimpleNamespace(resource_key="codec_kv", nodes={"Codec"})]),
     "holds resources .'codec_kv'.: the lane admits and plans only a recurrent state pool"),
    (dict(specs=[_pool_spec(nodes=("Codec", "Talker"))]), "a pool another node shares"),
    (dict(specs=[_pool_spec(kv_cache="talker_kv")]), "holds resources .'codec_state'.*no prefix snapshots"),
    (dict(submodules={"Codec": _Resident(), "Talker": object()}), "resident tables"),
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
