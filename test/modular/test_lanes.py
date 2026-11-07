"""Lanes: nodes' steps beside the worker's main chain (``mstar.worker.lanes``).

The stream and event backends are mocked, so these run without a GPU; what they
pin is where a lane's kernels, collates and teardowns go, when a step may be
postprocessed, what a step off the chain must leave alone, the scheduler's lane
filter, and which configs and resources the startup decision refuses.
"""
from __future__ import annotations

import contextlib
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from test_micro_scheduler import NODE, WALK, _Engine, _has_ready, _Manager, _next_batch, _scheduler
from test_send_registration_failure import _decode_step, _postprocess_worker

from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine import accelerator_graph_runner as runner_module
from mstar.engine.engine import Engine, ExecutingBatch, SubmoduleManagement
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import ADMIT_OK, AllocationFailed, Segment, StepContext, SubmoduleStep
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import BatchedModelOutput, NodeSubmodule
from mstar.utils.device import DeviceCapability
from mstar.utils.threads import ThreadPool
from mstar.worker.engine_manager import EngineManager, _refuse_unservable_lanes, walk_lanes
from mstar.worker.lanes import Lane, finish_first_to_land
from mstar.worker.micro_scheduler import CHAIN_LANE, ScheduledBatch
from mstar.worker.worker import PendingBatch, Worker

CODEC = ("Codec", "codec_chunk")
_OUTPUTS = SimpleNamespace(host_rows=None, host_spans=None)


LANE = 1


@contextlib.contextmanager
def _entered(log, lane):
    """What ``Engine.on_lane`` does, logged."""
    log.append(f"enter lane {lane}")
    try:
        yield
    finally:
        log.append(f"exit lane {lane}")


def _batch(rids, node="Codec", walk="codec_chunk"):
    return ScheduledBatch(
        node_name=node, graph_walk=walk,
        request_to_worker_graph=dict.fromkeys(rids, 0), output_signals=["audio_chunk"],
    )


def _step(rids, landed=True, done=True, failed=None, lane=LANE):
    future = Future()
    if done:
        future.set_result(f"outputs of {rids}")
    return PendingBatch(
        batch=_batch(rids), node_batch=SimpleNamespace(
            completion_event=SimpleNamespace(query=lambda: landed), failed_requests=dict(failed or {}),
            token_stream=None,
            admit_error=None,
        ),
        node_name=CODEC[0], partition="Codec", graph_walk=CODEC[1], future=future, lane=lane,
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
    worker.engine_manager = SimpleNamespace(engine=SimpleNamespace(on_lane=lambda lane: _entered(log, lane)))
    return worker


# ----- streams --------------------------------------------------------------

def test_a_step_runs_on_the_lane_stream_with_no_fence_behind_the_main_one():
    worker = _worker()
    worker._execute_on_gpu_thread = lambda *a: worker.log.append("execute") or _OUTPUTS
    lane = Lane(worker, LANE)

    assert lane.execute(_batch([1]), object(), None) is _OUTPUTS

    assert worker.log == ["enter lane 1", "execute", "exit lane 1"]


def test_the_window_collates_on_the_lane_stream_before_the_step_is_submitted():
    """The step needs no fence because its one main-thread device op, stacking
    the window, runs on its own stream."""
    worker = _worker()
    worker._build_executing_batch = lambda batch: worker.log.append("build") or SimpleNamespace()
    worker._submit_gpu_step = lambda *a, **k: worker.log.append("submit") or Future()
    worker.scheduler.get_next_batch = lambda state, lane=CHAIN_LANE: _batch([1])
    lane = Lane(worker, LANE)

    lane.launch(None)

    assert worker.log == ["enter lane 1", "build", "exit lane 1", "in flight True", "submit"]


def test_a_lanes_host_rows_stage_through_a_ring_of_its_own():
    """The chain's next steps reuse the chain's ring before a lane's step lands,
    so a lane's stop buffers and spans stage through its own."""
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    worker._host_rows_slots = {}
    staged = []
    worker._copy_to_pinned = lambda buffers, purpose: staged.append(purpose) or dict(buffers)
    for lane in (CHAIN_LANE, LANE, CHAIN_LANE, CHAIN_LANE, CHAIN_LANE, CHAIN_LANE):
        worker._stage_host_rows(SimpleNamespace(
            check_stop_buffers={"tok": torch.zeros(1)}, row_request_ids=[1], output_spans=None,
        ), lane)
    lane_ring = {p for p in staged if "lane" in p}
    assert lane_ring == {f"host_rows_lane{LANE}_1"} and not lane_ring & (set(staged) - lane_ring)


def test_the_submitted_step_takes_the_lane_body_and_lifts_the_launch_fence():
    worker = _worker()
    worker._execute_on_gpu_thread = lambda *a: _OUTPUTS
    lane = Lane(worker, LANE)
    node_batch = SimpleNamespace(launch_started_event=None)
    with ThreadPool(max_workers=1) as pool:
        future = Worker._submit_gpu_step(worker, pool, _batch([1]), node_batch, None, execute=lane.execute)
        assert future.result() is _OUTPUTS
    assert node_batch.launch_started_event.wait(timeout=5)


# ----- one submit and finish for every lane ----------------------------------

def test_a_lane_step_is_submitted_as_its_lane_and_flagged_in_flight():
    worker = _worker()

    step = worker._submit_step(None, _batch([1]), SimpleNamespace(), lane=LANE, execute=print)

    assert step.lane == LANE and step.partition == "Codec"
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
    worker._hold_rows(LANE, {2, 3})
    assert worker._in_flight_rids == {1, 2, 3} and worker._held_beside(CHAIN_LANE) == {2, 3}

    worker._handle_main_loop_error(RuntimeError("boom"), (None, None), None)
    assert set(failed[0]) == {1, 2}

    worker._hold_rows(LANE, ())
    assert worker._in_flight_rids == {1, 2}


# ----- one step in flight ---------------------------------------------------

def test_one_step_in_flight():
    worker = _worker()
    asked = []
    worker.scheduler.get_next_batch = lambda state, lane=CHAIN_LANE: asked.append(lane) or _batch([len(asked)])
    lane = Lane(worker, LANE)

    lane.launch(None)
    lane.launch(None)
    assert asked == [LANE] and worker._in_flight_rids == {1}

    lane.pending = _step([1])
    lane.finish()
    lane.launch(None)
    assert asked == [LANE, LANE] and worker._in_flight_rids == {2}


def test_nothing_ready_launches_nothing():
    worker = _worker()
    lane = Lane(worker, LANE)

    lane.launch(None)

    assert lane.pending is None and worker.log == []


# ----- finishing a step -----------------------------------------------------

@pytest.mark.parametrize("step", [_step([1], landed=False), _step([1], done=False)], ids=["running", "launching"])
def test_a_step_still_running_is_left_alone(step):
    worker = _worker()
    lane = Lane(worker, LANE)
    lane.pending = step

    lane.finish()

    assert lane.pending is step and worker.log == []


def test_postprocess_runs_after_the_step_has_landed_and_its_flags_are_down():
    worker = _worker()
    lane = Lane(worker, LANE)
    lane.pending = _step([1, 2])
    worker._hold_rows(LANE, {1, 2})
    worker._hold_rows(CHAIN_LANE, {9})

    lane.finish()

    assert worker.log == [
        "wait for step", "clear in flight", "clear limit",
        ("postprocess", "outputs of [1, 2]", {"clear_loop_stops": False}),
    ]
    assert lane.pending is None and worker._in_flight_rids == {9}


def test_a_row_the_chain_still_holds_stays_in_flight():
    worker = _worker()
    lane = Lane(worker, LANE)
    lane.pending = _step([1, 2])
    worker._hold_rows(LANE, {1, 2})
    worker._hold_rows(CHAIN_LANE, {2})

    lane.finish()

    assert worker._in_flight_rids == {2}


def test_with_the_chain_idle_the_lane_that_landed_finishes_first():
    """Lane 2's step landed while lane 1's still runs: lane 2 postprocesses
    without waiting out lane 1's step."""
    worker = _worker()
    first, second = Lane(worker, 1), Lane(worker, 2)
    running = first.pending = _step([1], landed=False, lane=1)
    second.pending = _step([2], lane=2)

    finish_first_to_land([first, second])

    assert first.pending is running and second.pending is None
    assert "wait for step" in worker.log and not any(
        entry[0] == "postprocess" and entry[1] == "outputs of [1]" for entry in worker.log if isinstance(entry, tuple)
    )


def test_block_waits_for_a_step_still_running():
    """Postprocess frees the step's inputs, main-stream allocations a
    main-stream step could reuse while this one still reads them."""
    worker = _worker()
    lane = Lane(worker, LANE)
    lane.pending = _step([1], landed=False)

    lane.finish(block=True)

    assert worker.log[0] == "wait for step"
    assert ("postprocess", "outputs of [1]", {"clear_loop_stops": False}) in worker.log


def test_failed_rows_are_reported_before_the_step_routes():
    worker = _worker()
    lane = Lane(worker, LANE)
    lane.pending = _step([1, 2], failed={2: "boom"})

    lane.finish()

    assert worker.log.index(("fail", [2])) < worker.log.index(
        ("postprocess", "outputs of [1, 2]", {"clear_loop_stops": False}),
    )


def test_a_step_its_pool_refused_goes_back_to_wait_and_routes_nothing():
    """It ran no forward: its rows return to the scheduler as a refused
    chain step's do (``Worker._settle_step_rows``), a row that failed
    preparing fails, and all of them leave the lane."""
    worker = _worker()
    refused = []
    worker._handle_admit_failure = lambda batch, node_batch, referenced_rids: refused.append(
        sorted(batch.request_to_worker_graph),
    )
    lane = Lane(worker, LANE)
    step = lane.pending = _step([1, 2, 3], failed={3: "boom"})
    step.node_batch.admit_error = AllocationFailed(message="full", pages_short=1, label="main", request_id=2)
    worker._hold_rows(LANE, {1, 2, 3})
    worker._hold_rows(CHAIN_LANE, {9})

    lane.finish()

    assert refused == [[1, 2, 3]]
    assert worker.log == ["wait for step", "clear in flight", ("drop", [3]), ("fail", [3])]
    assert lane.pending is None and worker._in_flight_rids == {9}


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


def test_a_lane_step_between_two_chain_steps_leaves_the_loop_stop_in_place():
    """The chain's step past a stop must still be dropped when a lane step's
    postprocess lands between the stop and that step."""
    w = _chain_worker()
    lane_step, lane_outputs = _lane_step([5], LANE)
    past_the_stop, outputs = _lane_step([0, 1], CHAIN_LANE)
    past_the_stop.speculative_new_iter = True

    w._finish_step(lane_step, lane_outputs)
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
    lane = Lane(worker, LANE)
    lane.pending = _raising_step([3])
    worker._hold_rows(LANE, {3})
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
    lane = Lane(worker, LANE)

    lane.launch(None)

    assert ("fail", [3]) in worker.log and lane.pending is None and worker._in_flight_rids == set()


def test_a_main_loop_error_leaves_the_lane_step_to_finish():
    w = _worker()
    failed = []
    w._fail_requests = failed.append
    w._lanes = (lane := Lane(w, LANE),)
    lane_step = lane.pending = _step([3])
    w._hold_rows(LANE, {3})
    w._hold_rows(CHAIN_LANE, {1})

    w._handle_main_loop_error(RuntimeError("boom"), (_step([1], lane=CHAIN_LANE), None), None)

    assert set(failed[0]) == {1} and lane.pending is lane_step and w._in_flight_rids == {1, 3}


def test_a_row_in_both_steps_fails_with_the_chain():
    w = _worker()
    failed = []
    w._fail_requests = failed.append
    w._lanes = (lane := Lane(w, LANE),)
    lane.pending = _step([1])
    w._hold_rows(LANE, {1})
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


class _Stream:
    """A lane's stream, which refuses to be entered as an object: a
    ``torch.Stream`` keeps the state of ``with stream:`` on itself, so a second
    thread's enter while the first is inside raises."""

    def __enter__(self):
        raise AssertionError("entered the lane's stream object")

    def __exit__(self, *exc):
        return False


class _Backend:
    """A graph backend's streams, with its per-call stream context logged."""

    def __init__(self, log):
        self.log = log

    def new_stream(self):
        return _Stream()

    def stream_context(self, stream):
        assert isinstance(stream, _Stream)
        return _entered(self.log, LANE)


class _Device(DeviceCapability):
    """The CPU, with a graph backend whose streams are logged."""

    def __init__(self):
        super().__init__(torch.device("cpu"))
        self.log: list = []
        self.graph_backend = _Backend(self.log)


def _state_engine(pool, chain_pool=None):
    resources = {POOL: pool, **({"talker_state": chain_pool} if chain_pool is not None else {})}
    eng = Engine.__new__(Engine)
    eng._device_capability = _Device()
    eng._walk_lanes = {CODEC: LANE}
    eng._node_lanes = {"Codec": LANE}
    eng._lane_streams = {LANE: _Stream()}
    eng._runner = StepRunner(resources, node_resources={
        "Codec": [POOL], "Talker": ["talker_state"] if chain_pool is not None else [],
    }, on_lane={POOL: partial(eng.on_lane, LANE)})
    eng._resources = resources
    eng._enable_nvtx = eng._enable_profile = False
    eng._device = torch.device("cpu")
    eng._pre_plan_done = {}
    eng._plan_peers = eng._plan_peers_of({"Codec": [POOL], "Talker": list(resources.keys() - {POOL})})
    eng._loopback_steps = {}
    eng._chunking = SimpleNamespace(forget_request=lambda rid: None, release=lambda rid, node: None)
    eng._submodules = {"Codec": SubmoduleManagement(
        submodule=_StateSub(), forward=None, forward_batched=_advance,
        joint_comm_group=None, resources={POOL: pool},
    )}
    return eng


def _lane_batch(rids) -> ExecutingBatch:
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


def _engine_lane(eng):
    """A lane whose GPU-thread step is the engine's exec, on the engine's lane."""
    worker = _worker()
    worker._execute_on_gpu_thread = lambda batch, node_batch, plan_future: eng.exec(node_batch) and _OUTPUTS
    worker.engine_manager = SimpleNamespace(engine=eng)
    return Lane(worker, LANE), eng.device_capability


def test_a_lane_step_takes_plans_and_commits_its_slot_on_the_lane_stream():
    pool = _pool()
    eng = _state_engine(pool)
    eng.add_request(1)
    free = pool.num_free_slots
    lane, backend = _engine_lane(eng)
    _logged(backend.log, pool, "admit", "plan", "commit", "_zero_slot")

    first, second = _lane_batch([1]), _lane_batch([1])
    lane.execute(_batch([1]), first, None)
    slot = pool._slots[1]["main"]
    lane.execute(_batch([1]), second, None)

    assert slot.index != SINK_SLOT and pool.num_free_slots == free - 1
    ingest, step = ["enter lane 1", "exit lane 1"], ["enter lane 1", "admit", "plan", "commit", "exit lane 1"]
    assert backend.log == ingest + step * 2
    # a fresh stream reads zeros, and its next window what the first one wrote
    assert first.outputs.per_rid_outputs[1]["audio_chunk"][0].tolist() == [0.0, 0.0]
    assert second.outputs.per_rid_outputs[1]["audio_chunk"][0].tolist() == [1.0, 1.0]
    assert slot.has_state and slot.tokens == 2

    eng.remove_request(1)

    assert pool.num_free_slots == free and 1 not in pool._slots
    zero = backend.log.index("_zero_slot")
    assert backend.log[zero - 1:zero + 2] == ["enter lane 1", "_zero_slot", "exit lane 1"], (
        "the slot was zeroed off the lane's stream, where the next stream's first write could land ahead of it"
    )


def test_a_lane_step_leaves_the_chains_staged_pre_plan_alone():
    """The plan thread stages the chain's next step while a lane step admits
    and plans on the GPU thread; the lane step neither promotes nor drops it."""
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
    lane, _ = _engine_lane(eng)

    lane.execute(_batch([1]), _lane_batch([1]), None)

    assert eng._runner.staged and talker._preplanned, "the lane step dropped the chain's stage"
    assert eng._runner.admit(chain).ok
    staged = talker._cached_plan_output
    assert eng._runner.plan(chain)["talker_state"] is staged, "the chain's step planned afresh"


def test_a_step_with_no_resources_leaves_the_chains_staged_pre_plan_alone():
    """A step that declares no resource (BAGEL's combine_cfg) touches nothing the
    stage holds, so it neither drops it nor promotes it; and a run of it has no
    replay columns and nothing to take back."""
    pool, talker = _pool(), _pool("talker_state", "Talker")
    eng = _state_engine(pool, talker)
    eng.add_request(9)
    chain = SubmoduleStep(segments=[Segment(9, "main", 1)], steps={"talker_state": RecurrentStep()})
    chain.set_ctx(StepContext(request_ids=(9,), graph_walk="talker_decode", slot=0, capture=False))
    chain.ctx.is_preplan = True
    assert eng._runner.pre_admit(chain).ok
    eng._runner.pre_plan(chain)
    chain.ctx.is_preplan = False
    empty = SubmoduleStep(segments=[], steps={})
    empty.set_ctx(StepContext(request_ids=(5,), graph_walk="combine", slot=0, capture=False, repeats=4))

    assert eng._runner.admit(empty).ok
    eng._runner.plan(empty)

    assert eng._runner.staged and talker._preplanned, "a step with no resources dropped the chain's stage"
    assert eng._runner.replay_columns(empty, 4) == [] and eng._runner.replay_columns(None, 4) == []
    eng._runner.settle_run(None, {5: 2})


@pytest.mark.parametrize("stage", ["admit", "plan"])
def test_a_raise_in_a_lane_steps_admit_or_plan_fails_its_own_rows(monkeypatch, stage):
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
    lane, _ = _engine_lane(eng)
    worker = lane._worker
    lane.pending = _step([1], done=False)
    try:
        lane.execute(_batch([1]), _lane_batch([1]), None)
    except RuntimeError as exc:
        lane.pending.future.set_exception(exc)
    worker._hold_rows(LANE, {1})
    worker._hold_rows(CHAIN_LANE, {9})

    lane.finish()

    assert ("fail", [1]) in worker.log and not any(e[0] == "postprocess" for e in worker.log if isinstance(e, tuple))
    assert worker._in_flight_rids == {9} and lane.pending is None
    assert eng._runner.staged and talker._preplanned, "the chain's stage went with the lane step's raise"


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
    assert sorted(_next_batch(sched, _Manager([1]), lane=LANE).request_to_worker_graph) == [1]
    eng.exec(_lane_batch([1]))  # its first window takes the slot
    codec = _Manager([2])
    codec.get_fwd_info = lambda rid, partition: SimpleNamespace(resource_publish_info=None)

    assert _next_batch(sched, codec, lane=LANE) is None
    assert sched.parked.get(NODE) == {2} and 2 in codec.queues["wg0"]._ready
    assert sorted(_next_batch(sched, _Manager([7], node="Talker", walk="talker_decode")).request_to_worker_graph) == [7]

    eng.remove_request(1)

    assert sorted(_next_batch(sched, codec, lane=LANE).request_to_worker_graph) == [2]


# ----- the scheduler ---------------------------------------------------------

def _lanes(sched, node=NODE):
    sched.lane_of = {(node, WALK): LANE}
    return sched


def test_a_lane_walk_does_not_break_the_chain():
    sched = _lanes(_scheduler(_Engine()))
    manager = _Manager([1, 2])

    assert not _has_ready(sched, manager, exclude_target=("other", "walk"))


def test_a_lane_walk_is_scheduled_on_its_lane_alone():
    sched = _lanes(_scheduler(_Engine()))
    manager = _Manager([1, 2])

    assert _next_batch(sched, manager) is None
    assert _next_batch(sched, manager, lane=LANE + 1) is None
    assert sorted(_next_batch(sched, manager, lane=LANE).request_to_worker_graph) == [1, 2]


def test_a_lane_walks_backlog_stays_on_its_lane():
    sched = _lanes(_scheduler(_Engine(max_bs=1)))
    manager = _Manager([1, 2])

    first = _next_batch(sched, manager, lane=LANE)
    assert sched.backlog, "the cap should leave a row behind"
    assert not _has_ready(sched, manager, exclude_target=("other", "walk"))
    assert _next_batch(sched, manager) is None
    second = _next_batch(sched, manager, lane=LANE)
    assert {*first.request_to_worker_graph, *second.request_to_worker_graph} == {1, 2}


def test_a_nodes_walk_on_a_lane_leaves_its_other_walks_on_the_chain():
    """Lanes are per walk: a prompt walk beside the chain, the same node's
    decode on it."""
    sched = _scheduler(_Engine())
    sched.lane_of = {(NODE, "prefill"): LANE}

    assert _next_batch(sched, _Manager([1], walk="prefill")) is None
    assert sorted(_next_batch(sched, _Manager([1], walk="prefill"), lane=LANE).request_to_worker_graph) == [1]
    assert _next_batch(sched, _Manager([2]), lane=LANE) is None
    assert sorted(_next_batch(sched, _Manager([2])).request_to_worker_graph) == [2]


def test_a_lane_never_takes_the_chains_keys():
    sched = _scheduler(_Engine())
    manager = _Manager([1])

    assert _next_batch(sched, manager, lane=LANE) is None
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


def _engine(backend, lanes=None):
    engine = Engine.__new__(Engine)
    engine._device_capability = SimpleNamespace(graph_backend=backend)
    engine._node_lanes = {"Codec": LANE} if lanes is None else lanes
    engine._lane_streams = {lane: f"lane {lane}" for lane in set(engine._node_lanes.values())}
    return engine


def test_a_lane_captures_on_its_own_stream():
    """Graphs captured on one stream must not overlap: a lane's never do, with
    one step in flight, but a step off the chain overlaps every other lane's."""
    engine = _engine(_GraphBackend(), {"Codec": 1, "Vocoder": 1, "Denoiser": 2})

    assert [engine._capture_stream_for(node) for node in ("Codec", "Vocoder", "Denoiser")] == [
        "lane 1", "lane 1", "lane 2",
    ]


def test_a_node_on_the_chain_keeps_the_shared_capture_stream():
    backend = _GraphBackend()

    assert _engine(backend)._capture_stream_for("Talker") is None
    assert _engine(backend, lanes={})._capture_stream_for("Codec") is None
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
    token_stop = None
    device_loopback = None

    def spanned_outputs(self, walk):
        return frozenset()


class _Groups:
    def __init__(self, world=1):
        self.world = world

    def get_instance_world_size_for_node(self, node):
        return self.world


def _placed(submodules=None, model=None, nodes=("Codec", "Talker"), lanes=None, groups=None, graph_backend=True):
    return _refuse_unservable_lanes(
        submodules or {"Codec": _Codec(), "Talker": object()}, model or _Model(),
        nodes, {CODEC: LANE} if lanes is None else lanes, groups or _Groups(), graph_backend,
    )


def test_a_walk_placed_off_the_chain_takes_its_lane():
    assert _placed() == {CODEC: LANE}


def test_two_nodes_take_two_lanes():
    model = _Model({"codec_chunk": _codec_node(), "vocode": _codec_node("Vocoder"),
                    "talker_decode": _codec_node("Talker")})
    lanes = {CODEC: 1, ("Vocoder", "vocode"): 2}

    assert _placed({"Codec": _Codec(), "Vocoder": _Codec(), "Talker": object()}, model,
                   ("Codec", "Vocoder", "Talker"), lanes) == lanes


def test_a_walk_whose_node_follows_a_chain_node_takes_its_lane_and_the_chain_never_speculates_into_it():
    """Whisper's prefill is Sequential([encoder, decoder]): with the decoder's
    prefill on a lane, the encoder's step on the chain hands it rows, which the
    lane runs; the chain doesn't take them as its next step."""
    from mstar.graph.base import Sequential

    prefill = Sequential([_codec_node("Encoder"), _codec_node()])
    model = _Model({"codec_chunk": prefill, "talker_decode": _codec_node("Talker")})
    assert _placed({"Codec": _Codec(), "Encoder": object(), "Talker": object()}, model,
                   ("Codec", "Encoder", "Talker")) == {CODEC: LANE}

    ready = [SimpleNamespace(node_name=node, graph_walk="codec_chunk") for node in ("Codec", "Encoder")]
    worker = Worker.__new__(Worker)
    worker._graph_runtime = SimpleNamespace(speculate_node=lambda *args: ready)
    worker.chunk_loops = SimpleNamespace(walks=frozenset())
    worker._lanes = (object(),)
    worker.scheduler = SimpleNamespace(lane_of={CODEC: LANE})
    batch = ScheduledBatch(node_name="Encoder", graph_walk="codec_chunk", request_to_worker_graph={1: 0})
    assert worker._choose_spec_target(batch).info.node_name == "Encoder"
    worker.scheduler.lane_of[("Encoder", "codec_chunk")] = LANE
    assert worker._choose_spec_target(batch) is None


def test_the_field_is_read_per_walk_and_must_agree():
    """A group places its graph_walks, or every walk of its nodes; every walk of
    a node with one off the chain is listed."""
    walks = {"Talker": {"talker_decode"}, "Codec": {"codec_chunk", "codec_warm"}, "Vocoder": {"vocode"}}
    groups = [{"node_names": ["Talker"]}, {"node_names": ["Codec"], "lane": 1}, {"node_names": ["Vocoder"], "lane": 0}]
    assert walk_lanes(groups, walks) == {CODEC: 1, ("Codec", "codec_warm"): 1}
    split = [{"node_names": ["Codec"], "graph_walks": ["codec_warm"]},
             {"node_names": ["Codec"], "graph_walks": ["codec_chunk"], "lane": 1}]
    assert walk_lanes(split, walks) == {CODEC: 1, ("Codec", "codec_warm"): 0}
    # another worker's decode is no chain here: this worker's only walk of the node is on the lane
    assert walk_lanes(split, {"Codec": {"codec_chunk"}}) == {CODEC: 1}
    with pytest.raises(ValueError, match="no chain to run beside"):
        _placed(nodes=("Codec",), submodules={"Codec": _Codec()}, lanes=walk_lanes(split, {"Codec": {"codec_chunk"}}))
    with pytest.raises(ValueError, match="different lanes"):
        walk_lanes(groups + [{"node_names": ["Codec"], "graph_walks": ["codec_chunk"], "lane": 2}], walks)
    for bad in ("yes", True, -1):
        with pytest.raises(ValueError, match="takes a lane id"):
            walk_lanes([{"node_names": ["Codec"], "lane": bad}], walks)
    # the field it replaced fails loudly rather than leaving the node on the chain
    with pytest.raises(ValueError, match="side_stream was replaced by lane"):
        walk_lanes([{"node_names": ["Codec"], "side_stream": True}], walks)


_LOOP = Loop(name="decode_loop", section=_codec_node(), max_iters=4, outputs=[])


@pytest.mark.parametrize("kwargs, reason", [
    (dict(groups=_Groups(world=2)), "tensor or sequence parallel"),
    (dict(nodes=("Codec",)), "no chain to run beside"),
    (dict(graph_backend=False), "no accelerator-graph backend"),
    (dict(model=_Model({"codec_chunk": _codec_node(), "codec_warm": _LOOP}),
          lanes={CODEC: LANE, ("Codec", "codec_warm"): LANE}), "runs a loop"),
    (dict(model=_Model({"codec_chunk": _LOOP})), "runs a loop"),
    (dict(model=_Model(combined={CODEC: "codec_mixed"})), "combined walk"),
])
def test_what_a_lane_cannot_serve_is_refused_naming_the_field(kwargs, reason):
    with pytest.raises(ValueError, match=reason) as raised:
        _placed(**kwargs)
    assert "placed on lane 1" in str(raised.value) or "off the chain" in str(raised.value)


class _Owned(Resource):
    """A resource kind with per-request state, not a recurrent pool: a lane takes it."""

    @classmethod
    def build(cls, spec, info):  # pragma: no cover - built by hand
        raise NotImplementedError


class _Publishing(_Owned):
    """Published and copied out from the main thread, which does both on the lane's stream."""

    def publish(self, request_id):  # pragma: no cover - never called
        return None

    def after_output_copy(self, step):  # pragma: no cover - never called
        return None


class _Evicting(_Owned):
    @property
    def supports_eviction(self):
        return True


class _Dependent(_Owned):
    def __init__(self, deps):
        self.deps = set(deps)

    def depends_on(self):
        return self.deps


def _contexts(resources, declared_by, nodes=("Codec", "Talker"), prefix_caches=()):
    engine = Engine.__new__(Engine)
    engine._device = torch.device("cpu")
    engine._device_capability = _Device()
    engine._node_lanes = {"Codec": LANE}
    engine._lane_streams = {LANE: _Stream()}
    engine._resources = resources
    return engine, engine._lane_contexts(set(nodes), {k: frozenset(v) for k, v in declared_by.items()}, prefix_caches)


def test_a_lane_holds_the_resources_of_its_nodes_alone_whatever_their_kind():
    """The chain's resources stay on the current stream."""
    engine, contexts = _contexts(
        {POOL: _pool(), "codec_cache": _Owned(), "codec_pos": _Publishing(), "talker_kv": _Evicting()},
        {POOL: {"Codec"}, "codec_cache": {"Codec"}, "codec_pos": {"Codec"}, "talker_kv": {"Talker"}},
    )

    assert set(contexts) == {POOL, "codec_cache", "codec_pos"}
    with contexts[POOL]():
        pass
    assert engine.device_capability.log == ["enter lane 1", "exit lane 1"]


@pytest.mark.parametrize("resources, declared_by, kwargs, reason", [
    ({"state": _Owned()}, {"state": {"Codec", "Talker"}}, {}, "nodes on two lanes"),
    ({"state": _Owned()}, {"state": {"Codec", "Elsewhere"}}, {}, "another worker"),
    ({"state": _Dependent({"talker_kv"}), "talker_kv": _Owned()}, {"state": {"Codec"}, "talker_kv": {"Talker"}}, {},
     "depends on .'talker_kv'"),
    ({"talker_attn": _Dependent({"state"}), "state": _Owned()}, {"talker_attn": {"Talker"}, "state": {"Codec"}}, {},
     "depends on .'state'"),
    ({"talker_attn": _Dependent({"state"}), "state": _Owned()}, {"talker_attn": {"Talker", "Elsewhere"},
                                                                 "state": {"Codec"}}, {}, "depends on .'state'"),
    ({"state": _Owned()}, {"state": {"Codec"}}, dict(prefix_caches={"state"}), "prefix cache"),
    ({"state": _Evicting()}, {"state": {"Codec"}}, {}, "moves its state outside the step"),
])
def test_a_resource_a_lane_cannot_hold_alone_is_refused(resources, declared_by, kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        _contexts(resources, declared_by, **kwargs)


class _Recording(Resource):
    """Logs every call the main thread makes into it from outside a step."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    @classmethod
    def build(cls, spec, info):  # pragma: no cover - built by hand
        raise NotImplementedError

    def _record(self, call, result=None):
        self.log.append(f"{self.name} {call}")
        return result

    def ingest_request(self, rid, overrides=None):
        self._record("ingest")

    def remove_request(self, rid):
        self._record("remove")

    def admit_retrieve(self, rid, node_name, graph_walk, published):
        return self._record("retrieve", ADMIT_OK)

    def after_output_copy(self, step):
        self._record("output copy")

    def extend_prefix_chain(self, rid, node_name, graph_walk, outputs):
        self._record("prefix chain")

    def publish(self, request_id):  # pragma: no cover - the step's own, on its lane already
        return None

    def publish_after_stop(self, rid, node_name, graph_walk):
        self._record("publish after stop")

    def publish_from_snapshot(self, rid, snap, node_name, graph_walk):
        self._record("publish snapshot")


def test_the_main_threads_calls_into_a_lanes_resources_run_on_its_stream():
    """Ingest, readiness, output copies, publishing and teardown reach a lane's
    resource from the main thread, outside its step: each goes on the lane's
    stream, and the chain's resources stay on the current one."""
    log = []
    runner = StepRunner(
        {"codec_state": _Recording(log, "lane"), "talker_state": _Recording(log, "chain")},
        node_resources={"Codec": ["codec_state"], "Talker": ["talker_state"]},
        on_lane={"codec_state": partial(_entered, log, LANE)},
    )
    step = SubmoduleStep(segments=[Segment(1, "main", 1)], steps={"codec_state": RecurrentStep()})

    runner.ingest_request(1)
    runner.admit_retrieve(1, "Codec", "codec_chunk")
    runner.after_output_copy(step)
    runner.extend_prefix_chains(1, "Codec", "codec_chunk", {}, keys=["codec_state"])
    runner.publish_after_stop([1], node_name="Codec", graph_walk="codec_chunk")
    runner.publish_from_snapshots({1: {"codec_state": "snap"}}, [1], node_name="Codec", graph_walk="codec_chunk")
    runner.remove_request(1)

    on_lane = ["enter lane 1", "lane {}", "exit lane 1"]
    assert log == [
        *(e.format("ingest") for e in on_lane), "chain ingest",
        *(e.format(call) for call in ("retrieve", "output copy", "prefix chain", "publish after stop",
                                      "publish snapshot", "remove") for e in on_lane),
        "chain remove",
    ]


def test_a_cpu_build_that_asks_for_the_lane_fails_at_startup():
    """Through the real ``EngineManager.build``."""
    class Model(_Model):
        def get_autocast_dtype(self):
            return None

        def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
            return _Node()

        def get_node_resources(self):
            return []

        def chunking_policy_of(self):
            return {}

    class _Node(NodeSubmodule):
        def prepare_inputs(self, *args, **kwargs):
            raise NotImplementedError

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    with pytest.raises(ValueError, match="no accelerator-graph backend"):
        EngineManager.build(
            node_names={"Codec", "Talker"},
            device=torch.device("cpu"),
            model_config={"node_groups": [
                {"node_names": ["Talker"], "ranks": [0]},
                {"node_names": ["Codec"], "ranks": [0], "lane": 1},
            ]},
            parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
            transfer_engine_info=TransferEngineInfo(my_entity_id="t", my_session_id="t", transfer_engine=None),
            model=Model(),
            graph_runtime=SimpleNamespace(),
            my_walks={CODEC, ("Talker", "talker_decode")},
        )


@pytest.mark.parametrize("held", [frozenset(), frozenset({"kv"})])
def test_a_step_off_the_chain_on_the_chains_resources_waits_out_the_plan_threads_stage(held):
    """The plan thread stages the chain's next step on the node's resources while
    the GPU thread runs a lane's step; if the lane's node plans those resources
    too, its step must not overlap that stage. A lane that holds its resources
    alone doesn't wait."""
    import threading

    engine = Engine.__new__(Engine)
    engine._walk_lanes, engine._node_lanes = {("n", "prefill"): 1}, {"n": 1}
    engine._stage_lock = threading.Lock()
    engine._submodules = {"n": SimpleNamespace(resources={"kv": object()})}
    engine._runner = SimpleNamespace(lane_held=held)
    ran = threading.Event()
    engine.exec = lambda batch: ran.set()
    batch = SimpleNamespace(
        node_name="n", graph_walk="prefill", admit_error=None, rows_only=True, cached_rids=(), run_started=None,
    )
    engine._submodules["n"].submodule = _StateSub.__new__(_StateSub)
    engine._submodules["n"].device_stops = {}
    with engine._stage_lock:  # the plan thread, mid-stage
        step = threading.Thread(target=engine.exec_and_postprocess, args=(batch,))
        step.start()
        assert ran.wait(0.2) is bool(held)
    step.join(2)
    assert ran.is_set()


def test_a_node_with_a_walk_on_the_chain_is_the_chains_and_its_lane_walk_runs_beside_it():
    """Its captures and resources stay the chain's, so its lane walk's steps run
    beside the chain (eager, on a lane slot, under the stage lock) even with no
    resource of the chain's to plan."""
    from mstar.engine.engine import node_lanes_of

    lanes = {("n", "prefill"): 1, ("n", "decode"): 0, ("m", "chunk"): 2}
    assert node_lanes_of(lanes) == {"m": 2}
    engine = Engine.__new__(Engine)
    engine._walk_lanes, engine._node_lanes = lanes, node_lanes_of(lanes)
    engine._submodules = {"n": SimpleNamespace(resources={}), "m": SimpleNamespace(resources={})}
    engine._runner = SimpleNamespace(lane_held=frozenset())
    assert engine._beside_chain("n", "prefill")
    assert not engine._beside_chain("n", "decode")
    assert not engine._beside_chain("m", "chunk")


def test_a_lane_walk_beside_the_chain_refuses_a_node_with_piecewise_regions():
    """A slot past the rotation wraps onto one of the chain's in a region
    (review-r2 F17). A node wholly on its lane keeps its own rotation."""
    from mstar.engine.engine import node_lanes_of

    engine = Engine.__new__(Engine)
    engine._walk_lanes = {("n", "prefill"): 1, ("n", "decode"): 0, ("m", "chunk"): 2}
    engine._node_lanes = node_lanes_of(engine._walk_lanes)
    with pytest.raises(ValueError, match="piecewise regions"):
        engine._refuse_regions_beside_chain("n", {"encoder": object()})
    engine._refuse_regions_beside_chain("n", {})
    engine._refuse_regions_beside_chain("m", {"encoder": object()})


def test_a_step_beside_the_chain_takes_a_slot_of_its_lanes_and_runs_eager():
    """Every per-step buffer is keyed by slot (attention workspaces, KV's static
    plan buffers, position and sampler rows), and the chain holds its rotation:
    a run in flight on one slot, its next step staged on the other."""
    engine = Engine.__new__(Engine)
    engine._walk_lanes, engine._node_lanes = {("n", "prefill"): 1}, {"n": 1}
    mgmt = SubmoduleManagement.__new__(SubmoduleManagement)
    mgmt.slot_rotation = SimpleNamespace(count=2, next_slot=0)
    mgmt.resources = {"kv": object()}
    engine._submodules = {"n": mgmt}
    engine._runner = SimpleNamespace(lane_held=frozenset())
    ctx = StepContext(request_ids=(1,), graph_walk="prefill", slot=None, capture=False, repeats=4)
    batch = SimpleNamespace(node_name="n", graph_walk="prefill", slot=None, step_context=ctx)
    batch.set_slot = lambda slot: (setattr(batch, "slot", slot), setattr(ctx, "slot", slot))
    assert engine.reserve_replay_slot(batch) is None
    assert (batch.slot, ctx.slot, ctx.repeats) == (2, 2, 1)
    assert mgmt.next_slot == 0, "the chain's rotation is untouched"
