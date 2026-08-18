"""TP async scheduling — the follower-side scheduler surface.

What a follower needs beyond the serial TP-follow path, pinned here against
the same fakes ``test_tp_follow_targeted_gate.py`` uses:

``MicroScheduler.pop_ready_rids`` — pop a named rid set for a node
*all-or-nothing*. The follower rebuilds the leader's speculative head from the
ids on the wire; if any fresh rid is not ready locally yet, NOTHING may be
popped (the caller retries once readiness has progressed). The FIFO accessors
(peek / pop / replace head) keep order and touch only the head. The serial
``_try_schedule_tp_follow`` path now goes through the same pop and stamps the
head's seq on the batch it returns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphNode  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402

NODE = "LLM"
WALK = "decode"


class _FakeEngine:
    def __init__(self, not_ready=()):
        self.not_ready = set(not_ready)

    def check_ready(self, node_name, rid, fwd_info):
        return rid not in self.not_ready


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


class _FakeWorkerGraphsManager:
    def __init__(self, queue):
        self.queues = {"wg0": queue}
        self.per_request_info = {}

    def get_partition_for_node(self, node_name):
        return "p0"

    def get_worker_graph_id_for_node(self, rid, node_name, graph_walk=None):
        return "wg0"

    def get_fwd_info(self, rid, partition):
        return None


def _sched(engine=None):
    return MicroScheduler(
        engine_manager=_FakeEngineManager(engine or _FakeEngine()),
        parallel_leader_nodes=set(),  # follower role
    )


def _head(rids, seq=5, from_seq=4):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(rids),
        speculative=True, spec_seq=seq, spec_from_seq=from_seq,
    )


# ---------------------------------------------------------------- pop_ready_rids

def test_pop_ready_rids_pops_exactly_the_named_set():
    sched = _sched()
    queue = _FakeQueue(["r0", "r1", "r2"])
    manager = _FakeWorkerGraphsManager(queue)

    popped = sched.pop_ready_rids(manager, NODE, WALK, ["r1", "r2"])
    assert popped is not None
    nodes, wg = popped
    assert list(nodes) == ["r1", "r2"]  # wire order preserved
    assert wg == {"r1": "wg0", "r2": "wg0"}
    # r0 untouched, r1/r2 consumed
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert NODE not in queue.per_request_queues["r1"].ready_node_names
    assert NODE not in queue.per_request_queues["r2"].ready_node_names


def test_pop_ready_rids_is_all_or_nothing_on_graph_readiness():
    sched = _sched()
    queue = _FakeQueue(["r0"], not_ready_rids=["r1"])
    manager = _FakeWorkerGraphsManager(queue)

    assert sched.pop_ready_rids(manager, NODE, WALK, ["r0", "r1"]) is None
    # NOTHING was consumed: r0 is still ready for a later attempt.
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert sched.batch_number == 0


def test_pop_ready_rids_is_all_or_nothing_on_engine_readiness():
    sched = _sched(_FakeEngine(not_ready=["r1"]))
    queue = _FakeQueue(["r0", "r1"])
    manager = _FakeWorkerGraphsManager(queue)

    assert sched.pop_ready_rids(manager, NODE, WALK, ["r0", "r1"]) is None
    assert NODE in queue.per_request_queues["r0"].ready_node_names
    assert NODE in queue.per_request_queues["r1"].ready_node_names


def test_pop_ready_rids_empty_set_is_a_valid_no_op():
    sched = _sched()
    manager = _FakeWorkerGraphsManager(_FakeQueue(["r0"]))
    assert sched.pop_ready_rids(manager, NODE, WALK, []) == ({}, {})
    assert sched.batch_number == 0


def test_serial_tp_follow_path_still_serves_via_shared_pop():
    """Refactor guard: ``_try_schedule_tp_follow`` now goes through
    ``pop_ready_rids``; the observable serial behaviour is unchanged, and the
    batch carries the head's seq."""
    sched = _sched()
    manager = _FakeWorkerGraphsManager(_FakeQueue(["r0", "r1"]))
    sched.register_tp_follow(_head(["r0", "r1"], seq=9))
    batch = sched.get_next_batch(manager)
    assert batch is not None
    assert set(batch.node_objects) == {"r0", "r1"}
    assert batch.tp_seq == 9
    assert sched.peek_tp_follow() is None


def test_serial_tp_follow_waits_when_a_rid_is_not_ready_and_pops_nothing():
    sched = _sched()
    queue = _FakeQueue(["r0"], not_ready_rids=["r1"])
    manager = _FakeWorkerGraphsManager(queue)
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
    a2 = ScheduleTPNode(NODE, WALK, ["r0"], speculative=True, spec_seq=1, spec_from_seq=0)
    sched.replace_tp_follow_head(a2)
    assert sched.peek_tp_follow() is a2
    assert list(sched.tp_batches_pending_schedule) == [a2, b]

    assert sched.pop_tp_follow_head() is a2
    assert sched.peek_tp_follow() is b
    assert sched.pop_tp_follow_head() is b
    assert sched.peek_tp_follow() is None


