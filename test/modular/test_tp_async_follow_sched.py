"""TP async scheduling — the follower-side scheduler surface.

What a follower needs beyond the serial TP-follow path, pinned here against
the same fakes ``test_tp_follow_targeted_gate.py`` uses:

``MicroScheduler.pop_ready_rids`` — pop a named rid set for a node
*all-or-nothing*. The follower rebuilds the leader's speculative head from the
ids on the wire; if any fresh rid is not ready locally yet, NOTHING may be
popped (the caller retries once readiness has progressed). The FIFO accessors
(peek / pop head) keep order and touch only the head. The serial
``_try_schedule_tp_follow`` path now goes through the same pop and stamps the
head's seq on the batch it returns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources.step import FULL_ADMIT_NOT_READY, FULL_ADMIT_OK  # noqa: E402
from mstar.graph.base import GraphNode  # noqa: E402
from mstar.graph.runtime.base import PopRidsOutput
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ScheduleTPNode  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402

NODE = "LLM"
WALK = "decode"


class _FakeEngine:
    def __init__(self, not_ready=()):
        self.not_ready = set(not_ready)

    def check_ready(self, node_name, rid, fwd_info):
        # Retryable not-ready only; a terminal AdmitRuntimeError is the
        # scheduler's ``_check_ready`` business, pinned in test_micro_scheduler.
        return FULL_ADMIT_NOT_READY if rid in self.not_ready else FULL_ADMIT_OK


class _FakeEngineManager:
    def __init__(self, engine):
        self.engine = engine

    def get_engine(self, node_name):
        return self.engine


class _FakeRequestGraph:
    def __init__(self, ready_node_names):
        self.ready_node_names = set(ready_node_names)


class _FakeQueue:
    def __init__(self, ready_rids, not_ready_rids=()):
        self.per_request_queues = {
            rid: _FakeRequestGraph({NODE}) for rid in ready_rids
        }
        for rid in not_ready_rids:
            self.per_request_queues[rid] = _FakeRequestGraph(set())

    def pop_ready_nodes(self, rid, node_names):
        wg = self.per_request_queues[rid]
        popped = [
            GraphNode(name=name, input_names=set(), outputs=[])
            for name in node_names if name in wg.ready_node_names
        ]
        wg.ready_node_names -= set(node_names)
        return popped

    def get_ready_node_names(self):
        return {}


class _FakeRuntime:
    """The runtime owns the (walk, node) -> worker graph index, the
    graph-level ready scan and the pop now."""

    def __init__(self, queue=None):
        self._queue = queue

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        return "wg0"
    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        del graph_walk
        queue = self._queue
        if check_ready:
            for rid in request_ids:
                wg = queue.per_request_queues.get(rid)
                if wg is None or node_name not in wg.ready_node_names:
                    return None
        rids = [
            rid for rid in request_ids
            if queue.pop_ready_nodes(rid, [node_name])
        ]
        return PopRidsOutput(
            wg_ids=ParallelList(rids, ["wg0"] * len(rids)),
            input_edges=[], input_edges_per_rid=[0] * len(rids),
        )

    def get_nodes(self, node_name, rids, wg_ids):
        del wg_ids
        return [
            GraphNode(name=node_name, input_names=set(), outputs=[])
            for _ in rids
        ]


    def has_ready_excluding(self, exclude_rids, exclude_target=None):
        # Pure TP follower: no locally-initiated work, matching _FakeQueue.
        return False

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        return []


class _FakeRequestStateManager:
    def __init__(self, queue):
        self.queues = {"wg0": queue}
        # The real runtime owns the queues and the manager shares them; bind
        # the fake pair the same way.
        self.runtime = _FakeRuntime(queue)
        self.per_request_info = {}

    def get_partition_for_node(self, node_name):
        return "p0"

    def get_fwd_info(self, rid, partition):
        return None


def _sched(engine=None):
    sched = MicroScheduler(
        engine_manager=_FakeEngineManager(engine or _FakeEngine()),
        parallel_leader_nodes=set(),  # follower role
    )
    return sched


def _head(rids, seq=5, from_seq=4):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(rids),
        speculative=True, spec_seq=seq, spec_from_seq=from_seq,
    )


# ---------------------------------------------------------------- pop_ready_rids

def test_pop_ready_rids_pops_exactly_the_named_set():
    sched = _sched()
    queue = _FakeQueue(["r0", "r1", "r2"])
    manager = _FakeRequestStateManager(queue)
    sched.runtime = manager.runtime

    popped = sched.pop_ready_rids(manager, NODE, WALK, ["r1", "r2"])
    assert popped is not None
    wg, input_edges, _output_signals = popped
    assert list(wg) == ["r1", "r2"]  # wire order preserved
    assert wg == {"r1": "wg0", "r2": "wg0"}
    assert set(input_edges) == {"r1", "r2"}
    # r0 untouched, r1/r2 consumed
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert NODE not in queue.per_request_queues["r1"].ready_node_names
    assert NODE not in queue.per_request_queues["r2"].ready_node_names


def test_pop_ready_rids_is_all_or_nothing_on_graph_readiness():
    sched = _sched()
    queue = _FakeQueue(["r0"], not_ready_rids=["r1"])
    manager = _FakeRequestStateManager(queue)
    sched.runtime = manager.runtime

    assert sched.pop_ready_rids(manager, NODE, WALK, ["r0", "r1"]) is None
    # NOTHING was consumed: r0 is still ready for a later attempt.
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert sched.batch_number == 0


def test_pop_ready_rids_is_all_or_nothing_on_engine_readiness():
    sched = _sched(_FakeEngine(not_ready=["r1"]))
    queue = _FakeQueue(["r0", "r1"])
    manager = _FakeRequestStateManager(queue)
    sched.runtime = manager.runtime

    assert sched.pop_ready_rids(manager, NODE, WALK, ["r0", "r1"]) is None
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert NODE in queue.per_request_queues["r1"].ready_node_names


def test_pop_ready_rids_empty_set_is_a_valid_no_op():
    sched = _sched()
    manager = _FakeRequestStateManager(_FakeQueue(["r0"]))
    sched.runtime = manager.runtime
    assert sched.pop_ready_rids(manager, NODE, WALK, []) == ({}, {}, ())
    assert sched.batch_number == 0


def test_serial_tp_follow_path_still_serves_via_shared_pop():
    """Refactor guard: ``_try_schedule_tp_follow`` now goes through
    ``pop_ready_rids``; the observable serial behaviour is unchanged, and the
    batch carries the head's seq."""
    sched = _sched()
    manager = _FakeRequestStateManager(_FakeQueue(["r0", "r1"]))
    sched.runtime = manager.runtime
    sched.register_tp_follow(_head(["r0", "r1"], seq=9))
    batch = sched.get_next_batch(manager)
    assert batch is not None
    assert set(batch.request_to_worker_graph) == {"r0", "r1"}
    assert batch.tp_seq == 9
    assert sched.peek_tp_follow() is None


def test_serial_tp_follow_waits_when_a_rid_is_not_ready_and_pops_nothing():
    sched = _sched()
    queue = _FakeQueue(["r0"], not_ready_rids=["r1"])
    manager = _FakeRequestStateManager(queue)
    sched.runtime = manager.runtime
    sched.register_tp_follow(_head(["r0", "r1"]))
    assert sched.get_next_batch(manager) is None
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert len(sched.tp_batches_pending_schedule) == 1


# ------------------------------------------------------------ FIFO accessors

def test_fifo_accessors_touch_only_the_head_and_keep_order():
    sched = _sched()
    a, b = _head(["r0"], seq=1, from_seq=0), _head(["r1"], seq=2, from_seq=1)
    sched.register_tp_follow(a)
    sched.register_tp_follow(b)

    assert sched.peek_tp_follow() is a
    assert list(sched.tp_batches_pending_schedule) == [a, b]
    assert sched.pop_tp_follow_head() is a
    assert sched.peek_tp_follow() is b
    assert sched.pop_tp_follow_head() is b
    assert sched.peek_tp_follow() is None




# -------------------------------------------------- drain refcount accounting

# ``Worker._complete_drain_if_ready`` withholds the teardown barrier's
# READS_DONE ACK while ``pending_tp_follow_count[rid] > 0``, so every path that
# retires a queued head must discharge the count — the async follower's direct
# head pops included, or an abort never completes.

def test_drain_refcount_discharged_by_the_serial_path():
    sched = _sched()
    manager = _FakeRequestStateManager(_FakeQueue(["r0"]))
    sched.runtime = manager.runtime
    sched.register_tp_follow(_head(["r0"]))
    assert sched.pending_tp_follow_count["r0"] == 1

    assert sched.get_next_batch(manager) is not None
    assert "r0" not in sched.pending_tp_follow_count


def test_drain_refcount_discharged_by_a_direct_head_pop():
    """The async follower retires heads without ``_try_schedule_tp_follow``:
    building one into a speculation, dropping one whose parent step closed, or
    voiding one on its parent's verdict. All three pop the head directly."""
    sched = _sched()
    sched.register_tp_follow(_head(["r0", "r1"]))
    assert sched.pending_tp_follow_count == {"r0": 1, "r1": 1}

    sched.pop_tp_follow_head()
    assert not sched.pending_tp_follow_count


def test_drain_refcount_tracks_each_queued_head_separately():
    sched = _sched()
    sched.register_tp_follow(_head(["r0"], seq=1, from_seq=0))
    sched.register_tp_follow(_head(["r0"], seq=2, from_seq=1))
    assert sched.pending_tp_follow_count["r0"] == 2

    sched.pop_tp_follow_head()
    assert sched.pending_tp_follow_count["r0"] == 1  # one head still queued
    sched.pop_tp_follow_head()
    assert "r0" not in sched.pending_tp_follow_count


def test_drain_refcount_not_resurrected_by_a_pop_after_clear_rid():
    """REMOVE_REQUEST clears the count; a later pop of a head still naming the
    rid must not leave a fresh entry behind (the dict is a defaultdict)."""
    sched = _sched()
    sched.register_tp_follow(_head(["r0"]))
    sched.clear_rid("r0")
    assert "r0" not in sched.pending_tp_follow_count

    sched.pop_tp_follow_head()
    assert "r0" not in sched.pending_tp_follow_count
