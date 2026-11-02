"""A looping node capped at one row serves its requests in the node's order.

Wan2.2's DiT takes one request per step and loops it for every denoising
step. The worker builds step N+1 while N runs, so the row N serves is not on
the ready queue when N+1 is chosen: the speculation chain either continues it
or gives the step to a row waiting there (``pick_chain_step``), and without a
chain the scheduler picks from the queue and backlog (``get_next_batch``).
With the chain blind to rows on the queue, whichever row a backlog drained
last ran to the end of its loop while the others waited on the queue, and a
request holding a low handle got one step per drain.

Who goes first is the node's ``StepOrder``: under TURNS the requests advance a
step each in rotation; under ARRIVAL the earliest request runs to the end of
its loop, then the next.

The driver below runs the real ``MicroScheduler`` in the worker's order. The
runtime scans in handle order, as the Rust one does, so the order a row comes
back in says nothing about how long it waited.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

import mstar.worker.worker as worker_mod  # noqa: E402
from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources.step import FULL_ADMIT_OK, StepContext  # noqa: E402
from mstar.graph.runtime.base import (  # noqa: E402
    ColumnarEdgeSpecs,
    PopRidsOutput,
    ReadyNodeSpec,
    SpeculationOutput,
)
from mstar.model.submodule_base import StepOrder  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.worker.batch_builder import StepPick  # noqa: E402
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.micro_scheduler import ChainPick, MicroScheduler, ScheduledBatch  # noqa: E402
from mstar.worker.worker import SpeculationTarget, Worker  # noqa: E402

NODE, WALK = "dit", "video_gen"
KEY = (NODE, WALK)
TURNS, ARRIVAL = StepOrder.TURNS, StepOrder.ARRIVAL


class _Runtime:
    """One looping node; ``ready`` is the set of rows on its queue."""

    def __init__(self):
        self.ready: set[int] = set()
        self.requests: dict[int, None] = {}  # the worker's live requests
        self.checked: list[int] = []  # rids the engine was asked about, in order

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        if (target is not None and target != KEY) or exclude_target == KEY:
            return []
        rids = sorted(r for r in self.ready if r not in exclude_rids)  # by handle
        return [ReadyNodeSpec(NODE, WALK, rids)] if rids else []

    def has_ready_excluding(self, exclude_rids, exclude_target=None):
        return bool(self.get_ready_nodes(exclude_rids, exclude_target=exclude_target))

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        return 0

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        rids = [r for r in request_ids if r in self.ready]
        self.ready -= set(rids)
        edges = ColumnarEdgeSpecs.empty()
        for rid in rids:
            edges.add(rid, "latents", [rid], False)
        return PopRidsOutput(wg_ids=ParallelList(rids, [0] * len(rids)), input_edges=edges)

    def push_back_node(self, node_name, rids, wg_ids):
        self.ready |= set(rids)

    def set_in_flight(self, node_name, rids, wg_ids, in_flight):
        pass


def _scheduler(runtime: _Runtime, order: StepOrder) -> tuple[MicroScheduler, SimpleNamespace]:
    engine = SimpleNamespace(
        get_max_batch_size=lambda node, walk: 1,
        get_max_batch_tokens=lambda node, walk: None,
        get_max_rows_in_combined=lambda node, walk: None,
        get_step_order=lambda node, walk: order,
        splits_by_capture_key=lambda node, walk: False,
        check_ready=lambda node, rid, fwd_info, allow_reload=True: (
            runtime.checked.append(rid) or FULL_ADMIT_OK
        ),
    )
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine),
        parallel_leader_nodes={NODE},
    )
    sched.runtime = runtime
    state = SimpleNamespace(
        per_request_info=runtime.requests,
        get_partition_for_node=lambda node: "default",
        get_fwd_info=lambda rid, partition: None,
        ingested_stream_chunks=lambda rid, node: {},
    )
    return sched, state


def _admitted(sched: MicroScheduler, batch: ScheduledBatch, refused: bool = False) -> None:
    """The worker's turn count once the step's admit has settled."""
    pending = SimpleNamespace(batch=batch, node_batch=SimpleNamespace(
        admit_error=object() if refused else None,
    ))
    Worker._count_turn(SimpleNamespace(scheduler=sched), pending)


def _run(
    steps: dict[int, int], order: StepOrder, arrive_at: dict[int, int] | None = None,
    elsewhere: ScheduledBatch | None = None, checks: list[tuple[bool, list[int]]] | None = None,
    refuse: frozenset[int] = frozenset(), ready_at: dict[int, int] | None = None,
) -> list[int]:
    """The rid each step served. ``steps``: rid -> loop length; ``arrive_at``:
    rid -> the step before which it arrives (default 0, in rid order);
    ``elsewhere``: a step of another node on the same worker, run after each
    of this one's; ``checks``: filled with whether a chain built each step
    and the rids its build asked the engine about; ``refuse``: the builds
    whose step the admit refuses; ``ready_at``: rid -> the step before which
    its row reaches the node, when that is after it arrived."""
    runtime = _Runtime()
    sched, state = _scheduler(runtime, order)
    arrive_at = arrive_at or {}
    ready_at = {rid: (ready_at or {}).get(rid, arrive_at.get(rid, 0)) for rid in steps}
    left = dict(steps)
    served: list[int] = []
    running = None  # the row step N serves, in flight while N+1 is chosen
    for n in range(2 * sum(steps.values()) + 2):
        for rid in sorted(rid for rid in steps if arrive_at.get(rid, 0) == n):
            runtime.requests[rid] = None
            sched.note_arrival(rid)
        runtime.ready |= {rid for rid in steps if ready_at[rid] == n}
        runtime.checked.clear()
        step = None
        chained = running is not None and bool(left[running])
        if chained:
            chain = sched.pick_chain_step(state, KEY, [running])
            assert chain.pick.room in (0, 1), "the runtime takes the room unsigned"
            if chain.pick.room == 0:
                step = sched.take_chain_step(state, chain)  # the chain gave its step away
            else:
                # prep keeps the chain's row; at a cap of one no waiting row joins
                assert sched.take_chain_step(state, chain) is None
                step = ScheduledBatch(NODE, WALK, request_to_worker_graph={running: 0})
        if step is None:
            step = sched.get_next_batch(state)
        if checks is not None:
            checks.append((chained, list(runtime.checked)))
        # N lands: a row the chain did not continue goes back on the queue
        if running is not None and left[running] and (step is None or running not in step.request_to_worker_graph):
            runtime.ready.add(running)
        running = None
        if step is None:
            assert not any(left.values()), f"stalled with {left} left"
            break
        (nxt,) = step.request_to_worker_graph
        if n in refuse:
            # no forward ran: `_handle_admit_failure` puts the row back
            _admitted(sched, step, refused=True)
            runtime.ready.add(nxt)
            continue
        _admitted(sched, step)
        if elsewhere is not None:
            sched.note_ran(elsewhere)
        left[nxt] -= 1
        served.append(nxt)
        running = nxt
    return served


def _longest_wait_in_rounds(order: list[int]) -> int:
    """The most turns any other request took between two of one request's
    turns (1 = it waited one round at most)."""
    worst = 0
    for rid in set(order):
        turns = [i for i, r in enumerate(order) if r == rid]
        for a, b in zip(turns, turns[1:], strict=False):
            between = order[a + 1:b]
            worst = max([worst] + [between.count(r) for r in set(between)])
    return worst


def test_four_requests_at_a_cap_of_one_take_turns():
    order = _run({rid: 6 for rid in range(4)}, TURNS)

    assert order == [0, 1, 2, 3] * 6
    assert _longest_wait_in_rounds(order) == 1


def test_four_requests_at_a_cap_of_one_run_in_arrival_order():
    """First come, first served to completion: a loop whose output only
    matters at the end finishes each request as early as it can."""
    order = _run({rid: 6 for rid in range(4)}, ARRIVAL, arrive_at={3: 0, 1: 1, 0: 2, 2: 3})

    assert order == [3] * 6 + [1] * 6 + [0] * 6 + [2] * 6


def test_a_request_with_a_recycled_low_handle_does_not_jump_the_others():
    """Handle 0 freed and handed to a newcomer: it scans first from then on,
    which is what let one request take the node while the rest waited."""
    arrive_at = {1: 0, 2: 0, 3: 0, 0: 6}

    turns = _run({0: 4, 1: 8, 2: 8, 3: 8}, TURNS, arrive_at=arrive_at)
    arrival = _run({0: 4, 1: 8, 2: 8, 3: 8}, ARRIVAL, arrive_at=arrive_at)

    # TURNS: the newcomer joins the rotation; nobody is overtaken twice by it
    assert _longest_wait_in_rounds(turns) == 1
    first = turns.index(0)
    assert sorted(turns[first - 3:first]) == [1, 2, 3]
    # ARRIVAL: it waits for the three that came first
    assert arrival == [1] * 8 + [2] * 8 + [3] * 8 + [0] * 4


def test_under_arrival_the_request_holding_the_step_keeps_it():
    """Request 0 came first but reaches the node after 1 has started there:
    1 runs to the end of its loop before 0 gets the node. Under TURNS 0 goes
    as soon as it is ready, being new to the node."""
    arrival = _run({0: 3, 1: 3}, ARRIVAL, ready_at={0: 2})
    turns = _run({0: 3, 1: 3}, TURNS, ready_at={0: 2})

    assert arrival == [1] * 3 + [0] * 3
    assert turns == [1, 1, 0, 1, 0, 0]


def test_a_request_that_also_runs_another_node_keeps_its_turn_here():
    """Request 0 runs an uncapped node of the same worker after every step of
    this one (a streaming thinker/talker pair, say). Its place in line here
    counts only its steps here, so the other node's steps don't push it to
    the back for as long as this node is oversubscribed."""
    other = ScheduledBatch("talker", "stream", request_to_worker_graph={0: 0})

    order = _run({rid: 6 for rid in range(3)}, TURNS, elsewhere=other)

    assert order == [0, 1, 2] * 6


@pytest.mark.parametrize("order", [TURNS, ARRIVAL])
def test_a_waiting_row_is_asked_about_once_a_step(order):
    """The chain's pick checks the rows it reaches, once each, and its take
    or the step built in its place doesn't ask again."""
    checks: list[tuple[bool, list[int]]] = []

    _run({rid: 6 for rid in range(4)}, order, checks=checks)

    assert any(asked for _, asked in checks)
    for _, asked in checks:
        assert len(asked) == len(set(asked)), asked
    # a chain at a cap of one asks about one waiting row at most; the step
    # after a loop ends scans the queues once, for the round robin
    assert all(len(asked) <= 1 for chained, asked in checks if chained), checks


def test_a_refused_step_keeps_its_rows_place_in_line():
    """A step the admit refuses (no KV for it, say) ran nothing, so it isn't
    a turn: the row it was for is first in line for the retry."""
    order = _run({0: 3, 1: 3}, TURNS, refuse=frozenset({1}))

    assert order == [0, 1, 0, 1, 0, 1]


def test_a_refused_step_is_not_a_turn_for_note_ran():
    runtime = _Runtime()
    sched, _ = _scheduler(runtime, TURNS)
    batch = ScheduledBatch(NODE, WALK, request_to_worker_graph={7: 0})

    _admitted(sched, batch, refused=True)
    assert sched.last_ran == {}
    _admitted(sched, batch)
    assert sched.last_ran == {KEY: {7: 1}}


def test_a_refusal_under_turns_keeps_the_next_row_in_line_on_the_device():
    """Rows 0-2 ran a step each and 3's first step is refused. 0 runs right
    after the retry, so the eviction takes 2, whose next turn is furthest away."""
    runtime = _Runtime()
    sched, state = _scheduler(runtime, TURNS)
    for rid in range(4):
        runtime.requests[rid] = None
        sched.note_arrival(rid)
        if rid < 3:
            _admitted(sched, ScheduledBatch(NODE, WALK, request_to_worker_graph={rid: 0}))
    runtime.ready |= {0, 1, 2, 3}

    refused = sched.get_next_batch(state)
    assert list(refused.request_to_worker_graph) == [3]
    _admitted(sched, refused, refused=True)
    worker = SimpleNamespace(scheduler=sched)
    assert Worker._select_eviction_victim(worker, NODE, WALK, [0, 1, 2]) == 2

    runtime.ready.add(3)
    retry = sched.get_next_batch(state)
    assert list(retry.request_to_worker_graph) == [3]
    _admitted(sched, retry)
    assert list(sched.get_next_batch(state).request_to_worker_graph) == [0]


def test_a_refusal_under_arrival_takes_the_row_last_in_line():
    """2 ran the latest step, so it holds the step; then 0, then 1."""
    runtime = _Runtime()
    sched, _ = _scheduler(runtime, ARRIVAL)
    for rid in range(3):
        sched.note_arrival(rid)
        _admitted(sched, ScheduledBatch(NODE, WALK, request_to_worker_graph={rid: 0}))

    assert Worker._select_eviction_victim(SimpleNamespace(scheduler=sched), NODE, WALK, [0, 1, 2]) == 1


class _StopLoopError(BaseException):
    """Ends ``Worker.run``, whose loop handles only ``Exception``."""


def _looping_worker(monkeypatch, sched, state, refuse=frozenset()) -> Worker:
    """The real ``Worker.run`` over this real scheduler, and everything else
    it calls stubbed: nothing speculates, so each step comes from
    ``get_next_batch``, and the loop ends when the scheduler has nothing left.
    The steps of the rows in ``refuse`` have their admit refused."""
    monkeypatch.setenv("MSTAR_PRE_PLAN_SPEC", "0")
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0))

    def stub(*a, **k):
        return None

    def idle(*a, **k):
        raise _StopLoopError

    def executing(batch):
        return ExecutingBatch(
            node_name=batch.node_name,
            per_request_info={rid: SimpleNamespace(rid=rid) for rid in batch.request_to_worker_graph},
            step_context=StepContext(request_ids=tuple(batch.request_to_worker_graph),
                                     graph_walk=batch.graph_walk, slot=0, capture=False),
        )

    def gpu(batch, node_batch, plan_future):
        if refuse.intersection(batch.request_to_worker_graph):
            node_batch.register_admit_error("out of pages")
        return {}

    w = Worker.__new__(Worker)
    w.worker_id = "worker_0"
    w.enable_nvtx = False
    w._phase_period, w._phase_buf = 0, {}
    w._max_consecutive_spec, w._launch_wait_s = 1024, 0.0
    w.tp_async_sched, w.is_tp_follower = False, False
    w._in_flight_rids = set()
    w._metrics = None
    w.parallel_groups = SimpleNamespace(barrier_all=stub, restore_default_timeout=stub)
    w.engine_manager = SimpleNamespace(warmup_all=stub)
    w.communicator = SimpleNamespace(send=stub, wait_for_work=idle)
    w.wakeup_event = SimpleNamespace(register_future=stub)
    w.scheduler, w.request_state = sched, state
    for name in ("_verify_tp_async_sched_agrees", "_pin_step_thread", "_init_engine_thread",
                 "_apply_pending_removes_safe_to_drop", "_fail_requests", "_apply_pending_drains",
                 "_process_messages", "_apply_removes_whose_step_landed", "_check_ready_tensors",
                 "_poll_stream_buffers", "_set_in_flight_flag", "_clear_in_flight_flag",
                 "_drop_rows_whose_loop_stopped", "_handle_admit_failure", "_postprocess_batch"):
        setattr(w, name, stub)
    w.maybe_send_zmq_to_tp_followers = lambda node_batch: -1
    w._build_executing_batch = executing
    w._build_speculation = lambda pending, consecutive, fair: (None, None)
    w._execute_on_gpu_thread = gpu

    def fail_loudly(error, *args):
        raise error

    w._handle_main_loop_error = fail_loudly
    return w


def test_the_run_loop_counts_a_steps_turns_once_its_admit_takes_it(monkeypatch):
    """Two steps through the real loop at a cap of one: 0's is admitted, so
    its row has had its turn; 1's is refused, so its row keeps its place."""
    runtime = _Runtime()
    sched, state = _scheduler(runtime, TURNS)
    for rid in (0, 1):
        runtime.requests[rid] = None
        sched.note_arrival(rid)
    runtime.ready |= {0, 1}

    with pytest.raises(_StopLoopError):
        _looping_worker(monkeypatch, sched, state, refuse=frozenset({1})).run()

    assert sched.last_ran == {KEY: {0: 1}}


def test_a_lone_request_keeps_its_chain_without_a_scan():
    """Nothing else on the worker: the chain continues, so speculation still
    overlaps, and the queues are not scanned for it."""
    runtime = _Runtime()
    runtime.requests[7] = None
    runtime.get_ready_nodes = None  # a scan would raise
    sched, state = _scheduler(runtime, TURNS)

    assert sched.pick_chain_step(state, KEY, [7]).pick.room == 1


# ── the worker's side ───────────────────────────────────────────────────


def _worker(chain: ChainPick, prep, taken=None) -> SimpleNamespace:
    worker = SimpleNamespace(
        worker_id="w0",
        scheduler=SimpleNamespace(
            backlog_splits_from=lambda *a: False,
            pick_chain_step=lambda state, target, continuing, mid_prefill=(): chain,
            take_chain_step=lambda state, chain, *_: taken,
        ),
        _spec_preparer=SimpleNamespace(prepare=prep),
        tensor_manager=SimpleNamespace(get_tensor=lambda uuid: None),
        request_state=SimpleNamespace(),
        _is_tearing_down=lambda rid: False,
        _teardowns_pending=lambda: False,
        chunk_loops=ChunkLoops(),
        _yield_away_speculation=lambda batch, gave_way=False: SimpleNamespace(
            scheduled_batch=batch, gave_way=gave_way,
        ),
    )
    worker._try_speculate_next = Worker._try_speculate_next.__get__(worker)
    return worker


_TARGET = SpeculationOutput(
    node_name=NODE, graph_walk=WALK, is_new_loop_iter=True, loop_name="denoise",
    output_signals=("latents",),
)


def _pending():
    return SimpleNamespace(batch=ScheduledBatch(NODE, WALK, request_to_worker_graph={1: 0, 2: 0, 3: 0}))


def test_prep_gets_the_chains_rows_in_line_and_the_room():
    """Prep keeps the first ``room`` rows it is handed that continue: the
    pick's line, so a row whose loop ends hands its place to the next one in
    line."""
    seen = {}

    def prep(batch_N, node, walk, candidates, room, laps, loop_strides=None):
        seen["rids"], seen["room"] = list(candidates), room
        return SimpleNamespace(ready_rids=[])

    pick = StepPick(taken=[2, 9], continuing=[2, 3, 1], room=1, held_back=set(), ordered=True)
    worker = _worker(ChainPick(KEY, pick, []), prep)

    assert worker._try_speculate_next(_pending(), SpeculationTarget(_TARGET, [1, 2, 3])) is None
    assert seen == {"rids": [2, 3, 1], "room": 1}


def test_a_chain_that_gives_its_whole_step_away_runs_the_waiting_rows():
    """No room left for the chain: the rows the pick chose run as a step of
    their own, which still counts as the chain's for fairness, and nothing is
    prepped for the chain's rows."""
    pick = StepPick(taken=[9], continuing=[1, 2, 3], room=0, held_back=set(), ordered=True)
    waiting = ScheduledBatch(NODE, WALK, request_to_worker_graph={9: 0})
    worker = _worker(ChainPick(KEY, pick, []), prep=None, taken=waiting)

    spec = worker._try_speculate_next(_pending(), SpeculationTarget(_TARGET, [1, 2, 3]))

    assert spec.scheduled_batch is waiting and spec.gave_way
