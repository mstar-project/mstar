"""TP async scheduling — the follower-side scheduler surface and the
symmetric void verdict.

Two things a follower needs beyond the serial TP-follow path, pinned here
against the same fakes ``test_tp_follow_targeted_gate.py`` uses:

1. ``MicroScheduler.pop_ready_rids`` — pop a named rid set for a node
   *all-or-nothing*. The follower rebuilds the leader's speculative head from
   the ids on the wire; if any fresh rid is not ready locally yet, NOTHING may
   be popped (the head stays for the serial path, which waits for readiness
   exactly as today). The FIFO accessors (peek / pop / replace head) keep
   order and touch only the head.
2. ``worker.tp_spec_survivors`` — the pure function behind
   ``Worker._reconcile_tp_follow_head``: which rids of a broadcast head
   survive step N's outputs. It must equal what
   ``Worker._thread_outputs_to_speculative`` does on a rank that DID build the
   head early, or two ranks of one TP group would run different compositions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphNode  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402
from mstar.worker.worker import tp_spec_survivors  # noqa: E402

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


# ------------------------------------------------------- tp_spec_survivors

def test_survivors_keeps_fresh_rids_and_continuing_rids_with_outputs():
    head = ["r0", "f1", "r2"]
    pending = {"r0", "r2"}
    consumed = {"token", "kv_cache"}
    outputs = {
        "r0": {"token": ["t"], "kv_cache": ["k"]},
        "r2": {"token": ["t"], "kv_cache": ["k"]},
    }
    assert tp_spec_survivors(head, pending, consumed, outputs) == ["r0", "f1", "r2"]


def test_survivors_drops_continuing_rid_missing_any_consumed_output():
    head = ["r0", "r1", "f2"]
    pending = {"r0", "r1"}
    consumed = {"token", "kv_cache"}
    outputs = {
        "r0": {"token": ["t"], "kv_cache": ["k"]},
        "r1": {"token": ["t"]},  # no kv_cache loop-back → dropped, like threading
    }
    assert tp_spec_survivors(head, pending, consumed, outputs) == ["r0", "f2"]


def test_survivors_empty_when_no_continuing_output_and_no_fresh():
    assert tp_spec_survivors(["r0"], {"r0"}, {"token"}, {"r0": {}}) == []
    assert tp_spec_survivors(["r0"], {"r0"}, {"token"}, {}) == []


def test_survivors_matches_threading_semantics_on_empty_tensor_list():
    """``_thread_outputs_to_speculative`` treats an empty tensor list as
    'no output' (``if not tensors``); so must the pure verdict."""
    assert tp_spec_survivors(["r0"], {"r0"}, {"token"}, {"r0": {"token": []}}) == []


def test_survivors_preserves_wire_order():
    head = ["c", "a", "b"]
    outputs = {r: {"token": ["t"]} for r in head}
    assert tp_spec_survivors(head, set(head), {"token"}, outputs) == ["c", "a", "b"]
