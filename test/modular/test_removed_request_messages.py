"""Messages for a request this worker already removed are dropped, not parked.

``InputIngest``'s dispatch parks a message whose request it does not know
until that request's NEW_REQUEST arrives. A message can also land after the
request was torn down here: the conductor's REMOVE after the TP leader's
forwarded one, or a peer's stream chunk or STOP_LOOPS that crossed a different
socket. Parking those kept them, unACKed, for the life of the worker.

Drives the real ``InputIngest`` dispatch and the worker's teardown over the
real graph runtime.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch
from rust_runtime import build_runtime

from mstar.communication.tensor_store import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.model.base import WorkerGraph
from mstar.testing_utils.worker import deliver, make_worker
from mstar.utils.ipc_format import (
    DrainRequest,
    InputSignals,
    MessageSource,
    NewRequest,
    RemoveRequest,
    ScheduleTPNode,
    StopLoops,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.stream_buffers import StreamBufferManager
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
        self.acked: list[int] = []  # uuids acked unread

    def register_request(self, rid, sharding_config):
        pass

    def start_read_tensors(self, rid, edges, graph_walk=None):
        self.reads += [(e.name, i.uuid) for e in edges for i in e.tensor_info]
        return []

    def force_cleanup_request(self, rid):
        pass

    def has_inflight_reads(self, rid):
        return False

    def ack_unread_tensors(self, graph_edges):
        self.acked += [i.uuid for e in graph_edges for i in e.tensor_info]


def _worker(is_tp_follower=False) -> Worker:
    section = GraphNode(name=NODE, input_names={"prompt"}, outputs=[])
    worker_graph = WorkerGraph(
        section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    tensor_manager = _TensorManager()
    runtime = build_runtime(
        bookkeeping=tensor_manager.tensor_store.bookkeeping,
        my_worker_id=WORKER,
        my_worker_graphs=[worker_graph],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set()},
        all_wg_ids_to_nodes={WG_ID: {NODE}},
        node_to_partition={NODE: "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
    )
    request_state = RequestStateManager(node_to_partition={NODE: "default"})
    w = make_worker(
        worker_id=WORKER, is_tp_follower=is_tp_follower, tensor_manager=tensor_manager,
        graph_runtime=runtime, request_state=request_state,
        engine_manager=SimpleNamespace(
            add_request=lambda rid, cfgs, info: None,
            remove_request=lambda rid: None,
            engine=SimpleNamespace(publisher_nodes={}),
        ),
        scheduler=SimpleNamespace(clear_rid=lambda rid: None, drop_backlogged=lambda rids: None),
        communicator=SimpleNamespace(send=lambda *a, **k: None),
        wakeup_event=SimpleNamespace(register_futures=lambda f: None),
        streams=StreamBufferManager(request_state, tensor_manager, []),
        profile_info=SimpleNamespace(pop_request=lambda rid: None),
    )
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


def _follow(rid: str) -> ScheduleTPNode:
    return ScheduleTPNode(node_name=NODE, graph_walk=WALK, request_ids=[rid])


def _parked(w: Worker) -> int:
    return sum(len(v) for v in w.lifecycle._parked.values())


def test_late_messages_for_a_removed_request_are_dropped_and_acked():
    w = _worker()
    deliver(w, _new("A#1"), _remove("A#1"))
    # landed after the teardown, over the peer's socket
    deliver(w, _signals("A#1", 1, streaming=True), _stop_loops("A#1"))
    assert _parked(w) == 0
    # the peer can free its source buffer instead of waiting on its own teardown
    assert w.tensor_manager.acked == [1]
    assert w.tensor_manager.reads == []


def test_leader_remove_then_conductor_remove_goes_through_remove_request():
    """A TP follower tears down on the leader's forwarded REMOVE, so the
    conductor's own REMOVE, sent to every participant, can land after it. It
    takes ``InputIngest._remove_request``'s no-handle branch, which clears the
    wire-keyed TP-follow count, instead of being parked."""
    w = _worker(is_tp_follower=True)
    deliver(w, _new("tp#1"))
    deliver(w, _remove("tp#1", source=MessageSource.TP_RANK_0))
    w.lockstep.enqueue_follow(_follow("tp#1"))
    assert w.lockstep.pending_follow_count("tp#1") == 1
    deliver(w, _remove("tp#1", source=MessageSource.CONDUCTOR))
    assert w._graph_runtime.get_rid_handle("tp#1") is None
    assert not w.lifecycle._parked
    assert w.lockstep.pending_follow_count("tp#1") == 0


def test_conductor_remove_for_a_request_never_admitted_drops_what_was_parked():
    """A NEW_REQUEST refused because the request was already draining: nothing
    replays its parked messages or clears its drain state, and the conductor's
    REMOVE follows it on the same socket, so no later NEW_REQUEST will."""
    w = _worker()
    deliver(w, _drain("D#1"))
    deliver(w, _signals("D#1", 3), _new("D#1"))
    assert w._graph_runtime.get_rid_handle("D#1") is None and _parked(w) == 1
    w.lockstep.enqueue_follow(_follow("D#1"))
    assert w.lockstep.pending_follow_count("D#1") == 1

    deliver(w, _remove("D#1"))
    assert not w.lifecycle._parked
    assert w.tensor_manager.acked == [3]
    assert "D#1" not in w.lifecycle.draining and "D#1" not in w.lifecycle._reads_done_sent
    assert w.lockstep.pending_follow_count("D#1") == 0


def test_a_remove_replayed_from_the_parking_lot_acks_what_trailed_it_once():
    """The leader's REMOVE and a peer's chunk can both outrun the follower's
    NEW_REQUEST; the replay tears the request down and the chunk is acked once."""
    w = _worker(is_tp_follower=True)
    deliver(w,
        _remove("R#1", source=MessageSource.TP_RANK_0),
        _signals("R#1", 5),
    )
    assert _parked(w) == 2

    deliver(w, _new("R#1"))
    assert w._graph_runtime.get_rid_handle("R#1") is None, "the replayed REMOVE tears the request down"
    assert not w.lifecycle._parked
    assert w.tensor_manager.acked == [5]
    assert w.tensor_manager.reads == []


def test_messages_that_outrun_their_new_request_are_still_replayed():
    w = _worker(is_tp_follower=True)
    deliver(w,
        _signals("E#1", 4), _remove("E#1", source=MessageSource.TP_RANK_0),
    )
    assert _parked(w) == 2, "an id never seen here may still be on its way"

    deliver(w, _new("E#1"))
    assert w.tensor_manager.reads == [("prompt", 4)]
    assert w._graph_runtime.get_rid_handle("E#1") is None
    assert w.tensor_manager.acked == []


def test_conductor_remove_reaches_a_non_tp_request_on_a_follower_worker():
    # This worker also follows a TP node, but this request uses only its
    # non-TP node (the runtime's request sharding has no TP group here).
    w = _worker(is_tp_follower=True)
    deliver(w, _new("side#1"), _remove("side#1", source=MessageSource.CONDUCTOR))
    assert w._graph_runtime.get_rid_handle("side#1") is None
