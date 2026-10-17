"""Messages for a request this worker already removed are dropped, not parked.

``Worker._process_message_list`` parks a message whose request it does not know
until that request's NEW_REQUEST arrives. A message can also land after the
request was torn down here: the conductor's REMOVE after the TP leader's
forwarded one, or a peer's stream chunk or STOP_LOOPS that crossed a different
socket. Parking those kept them for the life of the worker and replayed them
into a later request reusing the id.

Drives the real ``_add_new_request`` / ``_remove_request`` /
``_process_message_list`` over the real Python graph runtime.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensor_store import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ExpiringSet
from mstar.utils.ipc_format import (
    InputSignals,
    MessageSource,
    NewRequest,
    RemoveRequest,
    StopLoops,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.worker import _REMOVED_RID_TTL_S, Worker

WG_ID = 0
WALK = "prefill"
NODE = "prefill"
WORKER = "worker1"
PEER = "worker0"


class _TensorManager:
    def __init__(self):
        self.tensor_store = TensorStore()
        self.reads: list[tuple[str, int]] = []  # (edge name, uuid) reads started
        self.acked: list[tuple[str, int]] = []  # (wire rid, uuid) acked unread

    def register_request(self, rid, sharding_config):
        pass

    def start_read_tensors(self, rid, edges, graph_walk=None):
        self.reads += [(e.name, i.uuid) for e in edges for i in e.tensor_info]
        return []

    def force_cleanup_request(self, rid):
        pass

    def ack_unread_tensors(self, request_id, graph_edges):
        self.acked += [(request_id, i.uuid) for e in graph_edges for i in e.tensor_info]


def _worker(is_tp_follower=False) -> Worker:
    section = GraphNode(name=NODE, input_names={"prompt"}, outputs=[])
    worker_graph = WorkerGraph(
        section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    w = Worker.__new__(Worker)
    w.worker_id = WORKER
    w.is_tp_follower = is_tp_follower
    w.tensor_manager = _TensorManager()
    w._graph_runtime = PythonGraphRuntime(
        my_worker_id=WORKER,
        my_worker_graphs=[worker_graph],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set()},
        all_wg_ids_to_nodes={WG_ID: {NODE}},
        node_to_partition={NODE: "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        tensor_manager=w.tensor_manager,
    )
    w.request_state = RequestStateManager(node_to_partition={NODE: "default"})
    w.engine_manager = SimpleNamespace(
        evictable_nodes=lambda: [NODE],
        add_request=lambda rid, cfgs: None,
        remove_request=lambda rid: None,
    )
    w.scheduler = SimpleNamespace(
        clear_rid=lambda rid, rid_str: None,
        clear_wire_rid=lambda rid_str: None,
        last_consumed_tp_seq=-1,
    )
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.communicator = SimpleNamespace(send=lambda *a, **k: None)
    w.wakeup_event = SimpleNamespace(register_futures=lambda f: None)
    w._my_consumer_connections = []
    w._draining_rids = set()
    w._pending_drains = set()
    w._reads_done_sent = set()
    w._in_flight_rids = set()
    w._pending_removes = set()
    w._removes_awaiting_step = {}
    w._last_active = {}
    w.streaming_buffers = {}
    w._unprocessed_messages = {}
    w._removed_rids = ExpiringSet(_REMOVED_RID_TTL_S)
    w.enable_nvtx = False
    return w


def _new(rid: str) -> WorkerMessage:
    return WorkerMessage(WorkerMessageType.NEW_REQUEST, NewRequest(
        request_id=rid,
        request_info=CurrentForwardPassInfo(
            request_id=rid, graph_walk=WALK, fwd_index=0,
            random_seed=0, max_tokens=1, partition_name="default",
        ),
        partition_worker_graph_ids=[WG_ID],
        worker_graph_to_workers={WG_ID: [WORKER]},
        initial_inputs=[],
    ))


def _remove(rid: str, source=MessageSource.CONDUCTOR) -> WorkerMessage:
    return WorkerMessage(
        WorkerMessageType.REMOVE_REQUEST, RemoveRequest(request_id=rid, source=source),
    )


def _signals(rid: str, uuid: int, streaming=False) -> WorkerMessage:
    """A peer's output for ``rid``: one tensor it holds until this worker ACKs."""
    info = TensorPointerInfo(
        dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1],
        uuid=uuid, source_session_id="s", source_entity=PEER,
    )
    return WorkerMessage(WorkerMessageType.INPUT_SIGNALS, InputSignals(
        request_id=rid,
        inputs=[GraphEdge(next_node=NODE, name="prompt", tensor_info=[info],
                          is_streaming=streaming)],
        request_info=CurrentForwardPassInfo(
            request_id=rid, graph_walk=WALK, fwd_index=1,
            random_seed=0, max_tokens=1, partition_name="default",
        ),
    ))


def _stop_loops(rid: str) -> WorkerMessage:
    return WorkerMessage(WorkerMessageType.STOP_LOOPS, StopLoops(
        request_id=rid, loop_names={"ar"}, partition_name="default",
    ))


def _parked(w: Worker) -> int:
    return sum(len(v) for v in w._unprocessed_messages.values())


def test_late_messages_for_removed_requests_are_dropped_and_acked():
    w = _worker()
    for i in range(1000):
        rid = f"req-{i}"
        w._process_message_list([_new(rid), _remove(rid)])
        # landed after the teardown, over the peer's socket
        w._process_message_list([_signals(rid, i, streaming=True), _stop_loops(rid)])
    assert _parked(w) == 0, f"{_parked(w)} messages parked for 1000 removed requests"
    # the peer can free each source buffer instead of waiting on its own teardown
    assert w.tensor_manager.acked == [(f"req-{i}", i) for i in range(1000)]
    assert w.tensor_manager.reads == []


def test_a_reused_id_does_not_replay_the_previous_requests_late_messages():
    w = _worker()
    w._process_message_list([_new("X"), _remove("X")])
    w._process_message_list([_signals("X", 7)])  # the old X's straggler

    w._process_message_list([_new("X")])  # a client reuses the id
    assert w.tensor_manager.reads == [], "the new X read the old X's input"

    # ...and the new X's own messages are not dropped as the old one's
    w._process_message_list([_signals("X", 8)])
    assert w.tensor_manager.reads == [("prompt", 8)]


def test_leader_remove_then_conductor_remove_does_not_park():
    """A TP follower tears down on the leader's forwarded REMOVE, so the
    conductor's own REMOVE, sent to every participant, can land after it."""
    w = _worker(is_tp_follower=True)
    w._process_message_list([_new("tp")])
    w._process_message_list([
        _remove("tp", source=MessageSource.TP_RANK_0),
        _remove("tp", source=MessageSource.CONDUCTOR),
    ])
    assert w._rid("tp") is None
    assert not w._unprocessed_messages


def test_conductor_remove_for_a_request_never_admitted_drops_what_was_parked():
    """A NEW_REQUEST refused because the request was already draining: nothing
    replays its parked messages, and the conductor's REMOVE follows it on the
    same socket, so no later NEW_REQUEST will either."""
    w = _worker()
    w._draining_rids.add("D")
    w._process_message_list([_signals("D", 3), _new("D")])
    assert w._rid("D") is None and _parked(w) == 1

    w._process_message_list([_remove("D")])
    assert not w._unprocessed_messages
    assert w.tensor_manager.acked == [("D", 3)]


def test_a_remove_replayed_from_the_parking_lot_acks_what_trailed_it_once():
    """The leader's REMOVE and a peer's chunk can both outrun the follower's
    NEW_REQUEST; the replay tears the request down and the chunk is acked once."""
    w = _worker(is_tp_follower=True)
    w._process_message_list([
        _remove("R", source=MessageSource.TP_RANK_0),
        _signals("R", 5),
    ])
    assert _parked(w) == 2

    w._process_message_list([_new("R")])
    assert w._rid("R") is None, "the replayed REMOVE tears the request down"
    assert not w._unprocessed_messages
    assert w.tensor_manager.acked == [("R", 5)]
    assert w.tensor_manager.reads == []


def test_messages_that_outrun_their_new_request_are_still_replayed():
    w = _worker(is_tp_follower=True)
    w._process_message_list([
        _signals("E", 4), _remove("E", source=MessageSource.TP_RANK_0),
    ])
    assert _parked(w) == 2, "an id never seen here may still be on its way"

    w._process_message_list([_new("E")])
    assert w.tensor_manager.reads == [("prompt", 4)]
    assert w._rid("E") is None
    assert w.tensor_manager.acked == []


# ---- ExpiringSet ----------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_expiring_set_forgets_items_after_the_ttl():
    clock = _Clock()
    s = ExpiringSet(10.0, clock=clock)
    s.add("a")
    clock.now = 5.0
    s.add("b")
    assert "a" in s and "b" in s
    clock.now = 10.0
    assert "a" not in s and "b" in s
    clock.now = 15.0
    assert len(s) == 0


def test_expiring_set_readd_restarts_the_ttl_and_discard_forgets():
    clock = _Clock()
    s = ExpiringSet(10.0, clock=clock)
    s.add("a")
    s.add("b")
    clock.now = 8.0
    s.add("a")
    clock.now = 12.0
    assert "a" in s and "b" not in s
    s.discard("a")
    s.discard("never-added")
    assert "a" not in s and len(s) == 0


def test_expiring_set_holds_only_the_last_ttl_of_adds():
    clock = _Clock()
    s = ExpiringSet(10.0, clock=clock)
    for i in range(10_000):
        clock.now = float(i)
        s.add(i)
    assert len(s) == 10
    assert list(s._expiry) == list(range(9_990, 10_000))


def test_expiring_set_rejects_a_non_positive_ttl():
    with pytest.raises(ValueError):
        ExpiringSet(0.0)
