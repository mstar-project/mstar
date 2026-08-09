"""The TP-follow FIFO must never be popped by a targeted get_next_batch call.

A ScheduleTPNode is a mandate from the TP group leader: rank 0 has already
committed to the batch and will sit on the collective inside the forward
until every follower joins it (see the note in MicroScheduler.get_next_batch).
Once the message is popped from the FIFO, nothing on a follower worker will
ever reschedule that batch — followers cannot initiate scheduling for
parallel nodes — so whoever pops it must submit it unconditionally.

The only targeted caller in-tree is the speculation fresh-rid merge in
``worker._try_speculate_next``, which is a discretionary consumer: it may
reject rids from the batch it is handed (the "in-flight rid wins" branch).
Handing it a TP-follow batch therefore risks stranding a popped
ScheduleTPNode with no re-queue path, hanging the whole TP group at the next
collective. The guard under test: targeted calls leave the FIFO untouched;
untargeted calls still serve it, in order.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphNode  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402


class _FakeEngine:
    def check_ready(self, node_name, rid, fwd_info):
        return True


class _FakeEngineManager:
    def get_engine(self, node_name):
        return _FakeEngine()


class _FakeRequestGraph:
    def __init__(self, ready_node_names):
        self.ready_node_names = set(ready_node_names)


class _FakeQueue:
    """Just enough of RequestQueues for the TP-follow path."""

    def __init__(self, rids, node_name):
        self.per_request_queues = {
            rid: _FakeRequestGraph({node_name}) for rid in rids
        }

    def pop_ready_nodes(self, rid, node_names):
        wg = self.per_request_queues[rid]
        popped = [
            GraphNode(name=name, input_names=set(), outputs=[])
            for name in node_names if name in wg.ready_node_names
        ]
        wg.ready_node_names -= set(node_names)
        return popped

    def get_ready_node_names(self):
        # No locally-initiated work: this worker is a pure TP follower.
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


NODE = "LLM"
WALK = "decode"


def _setup(rids=("r0", "r1")):
    sched = MicroScheduler(
        engine_manager=_FakeEngineManager(),
        # Follower role: NODE is parallel here but this rank does not lead it.
        parallel_leader_nodes=set(),
    )
    queue = _FakeQueue(rids, NODE)
    manager = _FakeWorkerGraphsManager(queue)
    return sched, manager


def _msg(rids=("r0", "r1")):
    return ScheduleTPNode(node_name=NODE, graph_walk=WALK, request_ids=list(rids))


def test_targeted_call_never_pops_tp_follow():
    sched, manager = _setup()
    sched.register_tp_follow(_msg())

    # Even an exactly-matching target must be refused: the targeted caller
    # (the speculation merge) may reject the batch, and a popped message
    # has no re-queue path.
    assert sched.get_next_batch(
        manager, target_node_name=NODE, target_graph_walk=WALK
    ) is None
    assert sched.get_next_batch(manager, target_node_name=NODE) is None
    assert sched.get_next_batch(manager, target_graph_walk=WALK) is None
    assert len(sched.tp_batches_pending_schedule) == 1

    # The refusals consumed nothing: an untargeted call still builds the
    # full batch afterwards.
    batch = sched.get_next_batch(manager)
    assert batch is not None
    assert batch.node_name == NODE
    assert set(batch.node_objects) == {"r0", "r1"}


def test_untargeted_call_serves_and_drains():
    sched, manager = _setup()
    sched.register_tp_follow(_msg())

    batch = sched.get_next_batch(manager)
    assert batch is not None
    assert batch.node_name == NODE
    assert batch.graph_walk == WALK
    assert set(batch.node_objects) == {"r0", "r1"}
    assert len(sched.tp_batches_pending_schedule) == 0


def test_exclude_target_skips_head_but_keeps_it_queued():
    sched, manager = _setup()
    sched.register_tp_follow(_msg())

    assert sched.get_next_batch(manager, exclude_target=(NODE, WALK)) is None
    assert len(sched.tp_batches_pending_schedule) == 1

    batch = sched.get_next_batch(manager)
    assert batch is not None
    assert set(batch.node_objects) == {"r0", "r1"}


def test_fifo_order_survives_refused_targeted_calls():
    sched, manager = _setup(rids=("r0", "r1", "r2", "r3"))
    sched.register_tp_follow(_msg(("r0", "r1")))
    sched.register_tp_follow(_msg(("r2", "r3")))

    assert sched.get_next_batch(manager, target_node_name=NODE) is None

    first = sched.get_next_batch(manager)
    assert set(first.node_objects) == {"r0", "r1"}
    second = sched.get_next_batch(manager)
    assert set(second.node_objects) == {"r2", "r3"}
    assert len(sched.tp_batches_pending_schedule) == 0
