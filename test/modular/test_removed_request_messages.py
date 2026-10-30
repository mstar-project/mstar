"""Messages for a request this worker already removed are dropped, not parked.

``Worker._process_message_list`` parks a message whose request it does not know
until that request's NEW_REQUEST arrives. A message can also land after the
request was torn down here: the conductor's REMOVE after the TP leader's
forwarded one, or a peer's stream chunk or STOP_LOOPS that crossed a different
socket. Parking those kept them, unACKed, for the life of the worker.

Drives the real ``_add_new_request`` / ``_remove_request`` /
``_process_message_list`` over the real Python graph runtime.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.communication.tensor_store import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import (
    DrainRequest,
    InputSignals,
    MessageSource,
    NewRequest,
    RemoveRequest,
    StopLoops,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.rebuild import Rebuilds
from mstar.worker.worker import Worker

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
        add_request=lambda rid, cfgs, info=None: None,
        remove_request=lambda rid: None,
    )
    w.wire_rids_cleared = []
    w.scheduler = SimpleNamespace(
        clear_rid=lambda rid, rid_str: None,
        clear_wire_rid=w.wire_rids_cleared.append,
        fail_rids=lambda rids: None,
        pending_tp_follow_count={},
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
    w._pending_removes = {}
    w._removes_awaiting_step = {}
    w._last_active = {}
    w.streaming_buffers = {}
    w._unprocessed_messages = {}
    w._removed_rids = RecentSet(16)
    w._rebuilds = Rebuilds({}, {})
    w._kv_keys, w._kv_caches, w._dropped, w._victims, w._preempted = {}, {}, {}, {}, {}
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
        loop_caps={},
    ))


def _remove(rid: str, source=MessageSource.CONDUCTOR) -> WorkerMessage:
    return WorkerMessage(
        WorkerMessageType.REMOVE_REQUEST, RemoveRequest(request_id=rid, source=source),
    )


def _drain(rid: str) -> WorkerMessage:
    return WorkerMessage(WorkerMessageType.DRAIN_REQUEST, DrainRequest(request_id=rid))


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


def test_late_messages_for_a_removed_request_are_dropped_and_acked():
    w = _worker()
    w._process_message_list([_new("A#1"), _remove("A#1")])
    # landed after the teardown, over the peer's socket
    w._process_message_list([_signals("A#1", 1, streaming=True), _stop_loops("A#1")])
    assert _parked(w) == 0
    # the peer can free its source buffer instead of waiting on its own teardown
    assert w.tensor_manager.acked == [("A#1", 1)]
    assert w.tensor_manager.reads == []


def test_leader_remove_then_conductor_remove_goes_through_remove_request():
    """A TP follower tears down on the leader's forwarded REMOVE, so the
    conductor's own REMOVE, sent to every participant, can land after it. It
    takes ``_remove_request``'s no-handle branch, which clears the
    wire-keyed TP-follow count, instead of being parked."""
    w = _worker(is_tp_follower=True)
    w._process_message_list([_new("tp#1")])
    w._process_message_list([
        _remove("tp#1", source=MessageSource.TP_RANK_0),
        _remove("tp#1", source=MessageSource.CONDUCTOR),
    ])
    assert w._rid("tp#1") is None
    assert not w._unprocessed_messages
    assert w.wire_rids_cleared == ["tp#1"]


def test_a_request_drained_before_its_new_request_acks_what_was_parked():
    """A request drained before its NEW_REQUEST is still taken in, without reads, so a TP follower can apply rank
    0's moves for it. Its parked inputs are never read, so they are ACKed unread, and the conductor's REMOVE tears
    it down and clears its drain state."""
    w = _worker()
    w._process_message_list([_drain("D#1"), _signals("D#1", 3), _new("D#1")])
    assert w._rid("D#1") is not None and _parked(w) == 0
    assert w.tensor_manager.acked == [("D#1", 3)] and w.tensor_manager.reads == []

    w._process_message_list([_remove("D#1")])
    assert not w._unprocessed_messages and w._rid("D#1") is None
    assert "D#1" not in w._draining_rids and "D#1" not in w._reads_done_sent


def test_a_remove_replayed_from_the_parking_lot_acks_what_trailed_it_once():
    """The leader's REMOVE and a peer's chunk can both outrun the follower's
    NEW_REQUEST; the replay tears the request down and the chunk is acked once."""
    w = _worker(is_tp_follower=True)
    w._process_message_list([
        _remove("R#1", source=MessageSource.TP_RANK_0),
        _signals("R#1", 5),
    ])
    assert _parked(w) == 2

    w._process_message_list([_new("R#1")])
    assert w._rid("R#1") is None, "the replayed REMOVE tears the request down"
    assert not w._unprocessed_messages
    assert w.tensor_manager.acked == [("R#1", 5)]
    assert w.tensor_manager.reads == []


def test_messages_that_outrun_their_new_request_are_still_replayed():
    w = _worker(is_tp_follower=True)
    w._process_message_list([
        _signals("E#1", 4), _remove("E#1", source=MessageSource.TP_RANK_0),
    ])
    assert _parked(w) == 2, "an id never seen here may still be on its way"

    w._process_message_list([_new("E#1")])
    assert w.tensor_manager.reads == [("prompt", 4)]
    assert w._rid("E#1") is None
    assert w.tensor_manager.acked == []
