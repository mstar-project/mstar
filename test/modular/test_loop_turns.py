"""A looping node capped at one row serves its requests in turn.

Wan2.2's DiT takes one request per step and loops it for every denoising
step. The worker builds step N+1 while N runs, so the row N serves is not on
the ready queue when N+1 is chosen: the speculation chain either continues it
(``room_for_continuing``) or the scheduler picks from the queue and backlog
(``get_next_batch``). With the chain blind to rows on the queue, whichever
row a backlog drained last ran to the end of its loop while the others waited
on the queue, and a request holding a low handle got one step per drain.

The driver below runs the real ``MicroScheduler`` in the worker's order. The
runtime scans in handle order, as the Rust one does, so the order a row comes
back in says nothing about how long it waited.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources.step import FULL_ADMIT_OK  # noqa: E402
from mstar.graph.runtime.base import (  # noqa: E402
    ColumnarEdgeSpecs,
    PopRidsOutput,
    ReadyNodeSpec,
    SpeculationOutput,
    SpeculationPrepOutput,
)
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch  # noqa: E402
from mstar.worker.worker import SpeculationTarget, Worker  # noqa: E402

NODE, WALK = "dit", "video_gen"
KEY = (NODE, WALK)


class _Runtime:
    """One looping node; ``ready`` is the set of rows on its queue."""

    def __init__(self):
        self.ready: set[int] = set()
        self.requests: dict[int, None] = {}  # the worker's live requests

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


def _scheduler(runtime: _Runtime) -> tuple[MicroScheduler, SimpleNamespace]:
    engine = SimpleNamespace(
        get_max_batch_size=lambda node, walk: 1,
        get_max_rows_in_combined=lambda node, walk: None,
        splits_by_capture_key=lambda node, walk: False,
        check_ready=lambda node, rid, fwd_info, allow_reload=True: FULL_ADMIT_OK,
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


def _run(
    steps: dict[int, int], arrive_at: dict[int, int] | None = None,
    elsewhere: ScheduledBatch | None = None,
) -> list[int]:
    """The rid each step served. ``steps``: rid -> loop length; ``arrive_at``:
    rid -> the step before which it turns ready (default 0); ``elsewhere``: a
    step of another node on the same worker, run after each of this one's."""
    runtime = _Runtime()
    sched, state = _scheduler(runtime)
    arrive_at = arrive_at or {}
    left = dict(steps)
    order: list[int] = []
    running = None  # the row step N serves, in flight while N+1 is chosen
    for n in range(sum(steps.values()) + 1):
        arrived = {rid for rid in steps if arrive_at.get(rid, 0) == n}
        runtime.ready |= arrived
        runtime.requests.update(dict.fromkeys(arrived))
        nxt = None
        if running is not None and left[running]:
            room = sched.room_for_continuing(KEY, state, [running])
            assert room is None or room >= 0, "the runtime takes the room unsigned"
            if room is None or room >= 1:
                nxt = running
                # the speculative merge: no room left at a cap of one
                assert sched.get_next_batch(state, target=KEY, pre_existing_batch_size=1) is None
        if nxt is None:
            batch = sched.get_next_batch(state)
            if batch is not None:
                (nxt,) = batch.request_to_worker_graph
        # N lands: a row the chain did not continue goes back on the queue
        if running is not None and left[running] and nxt != running:
            runtime.ready.add(running)
        if nxt is None:
            assert not any(left.values()), f"stalled with {left} left"
            break
        sched.note_ran(ScheduledBatch(NODE, WALK, request_to_worker_graph={nxt: 0}))
        if elsewhere is not None:
            sched.note_ran(elsewhere)
        left[nxt] -= 1
        order.append(nxt)
        running = nxt
    return order


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
    order = _run({rid: 6 for rid in range(4)})

    assert order[:4] == [0, 1, 2, 3]
    assert order == [0, 1, 2, 3] * 6
    assert _longest_wait_in_rounds(order) == 1


def test_a_request_with_a_recycled_low_handle_does_not_jump_the_others():
    """Handle 0 freed and handed to a newcomer: it scans first from then on,
    which is what let one request take the node while the rest waited."""
    order = _run({0: 4, 1: 8, 2: 8, 3: 8}, arrive_at={1: 0, 2: 0, 3: 0, 0: 6})

    assert _longest_wait_in_rounds(order) == 1
    # the newcomer joins the rotation; nobody else is overtaken twice by it
    first = order.index(0)
    assert sorted(order[first - 3:first]) == [1, 2, 3]


def test_a_request_that_also_runs_another_node_keeps_its_turn_here():
    """Request 0 runs an uncapped node of the same worker after every step of
    this one (a streaming thinker/talker pair, say). Its place in line here
    counts only its steps here, so the other node's steps don't push it to
    the back for as long as this node is oversubscribed."""
    other = ScheduledBatch("talker", "stream", request_to_worker_graph={0: 0})

    order = _run({rid: 6 for rid in range(3)}, elsewhere=other)

    assert order == [0, 1, 2] * 6


def test_a_lone_request_keeps_its_chain_without_a_scan():
    """Nothing else on the worker: the chain continues, so speculation still
    overlaps, and the queues are not scanned for it."""
    runtime = _Runtime()
    runtime.requests[7] = None
    runtime.get_ready_nodes = None  # a scan would raise
    sched, state = _scheduler(runtime)

    assert sched.room_for_continuing(KEY, state, [7]) == 1


# ── the worker's side: which continuing rows give way ───────────────────


def test_the_rows_that_ran_longest_give_way_first():
    """``prep_spec_rids`` keeps the first ``room_for_continuing`` rows it is
    handed. A step lists its continuing rows before the ones merged into it,
    so the worker rotates the front, longest-running rows to the end."""
    seen = {}

    def prep(inp):
        seen["rids"], seen["room"] = list(inp.rids), inp.room_for_continuing
        return SpeculationPrepOutput(
            consumed_streaming_edge_idxs=[], ready_rids=[], wg_ids=[],
            input_edges=ColumnarEdgeSpecs.empty(),
        )

    worker = SimpleNamespace(
        worker_id="w0",
        scheduler=SimpleNamespace(
            backlog_splits_from=lambda *a: False,
            room_for_continuing=lambda target, state, continuing: 1,
        ),
        _graph_runtime=SimpleNamespace(prep_spec_rids=prep),
        tensor_manager=SimpleNamespace(get_tensor=lambda uuid: None),
        request_state=SimpleNamespace(),
        _is_tearing_down=lambda rid: False,
        _streams_possible=lambda: False,
    )
    worker._try_speculate_next = Worker._try_speculate_next.__get__(worker)
    target = SpeculationOutput(
        node_name=NODE, graph_walk=WALK, is_new_loop_iter=True, loop_name="denoise",
        output_signals=("latents",),
    )
    pending = SimpleNamespace(batch=ScheduledBatch(NODE, WALK, request_to_worker_graph={1: 0, 2: 0, 3: 0}))

    assert worker._try_speculate_next(pending, SpeculationTarget(target, [1, 2, 3])) is None
    assert seen == {"rids": [3, 1, 2], "room": 1}
