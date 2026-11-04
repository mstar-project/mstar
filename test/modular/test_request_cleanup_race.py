"""Deterministic CPU reproduction of the extra-read completion race.

The tensor managers, async reader, conductor barrier and worker drain/removal
methods are real. A checked host-memory engine substitutes for network transfers;
model execution and scheduling are stubbed. The extra read models an erroneous
output at completion, without assuming that a production model emits one.
"""

import ctypes
from collections import deque
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from mstar.communication import tensors
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensor_store import PythonTensorBookkeeping, RustTensorBookkeeping, TensorStore
from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import CurrentForwardConductorMetadata, PartitionState
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import (
    ConductorMessageType,
    DrainRequest,
    InputSignals,
    ReadsDone,
    WorkerGraphsDone,
    WorkerMessageType,
)
from mstar.worker.rid_table import RidTable
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker


class _CheckedCopyEngine(tensors.LocalTransferEngine):
    """Copy registered host memory, rejecting a read after either endpoint frees.

    Checking before memmove makes the pre-fix failure safe to reproduce rather
    than dereferencing freed pointers. Events select the ordering without sleeps.
    """

    def __init__(self, *, peers, **kwargs):
        super().__init__(f"cpu:{len(peers)}")
        peers[self.get_session_id()] = self
        self.peers = peers
        self._engine = self
        self.regions = {}
        self.started = Event()
        self.resume = Event()
        self.invalid_endpoints = []
        self.copied = None
        self.reads = {}
        self.next_batch = 0

    def register_memory(self, ptr, nbytes):
        self.regions[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr):
        self.regions.pop(ptr)
        return 0

    def batch_transfer_async_read(self, session, local_ptrs, remote_ptrs, lengths):
        self.next_batch += 1
        self.reads[self.next_batch] = session, local_ptrs, remote_ptrs, lengths
        self.started.set()
        return self.next_batch

    def transfer_check_status(self, batch_id):
        if not self.resume.is_set():
            return 0
        session, local_ptrs, remote_ptrs, lengths = self.reads.pop(batch_id)
        peer = self.peers[session]
        for local, remote, nbytes in zip(local_ptrs, remote_ptrs, lengths, strict=True):
            if peer.regions.get(remote, 0) < nbytes:
                self.invalid_endpoints.append("source")
            if self.regions.get(local, 0) < nbytes:
                self.invalid_endpoints.append("destination")
        if self.invalid_endpoints:
            return -1
        for local, remote, nbytes in zip(local_ptrs, remote_ptrs, lengths, strict=True):
            ctypes.memmove(local, remote, nbytes)
        self.copied = ctypes.string_at(local_ptrs[0], lengths[0])
        return 1


class _Recorder:
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))


def _worker(entity_id, manager, rid_table):
    w = Worker.__new__(Worker)
    w.worker_id = entity_id
    w.is_tp_follower = False
    w._in_flight_rids = set()
    w._lockstep_led = frozenset()
    w._removes_awaiting_step = {}
    w._pending_removes = {}
    w._draining_rids = set()
    w._pending_drains = set()
    w._reads_done_sent = set()
    w._removed_rids = RecentSet(16)
    w._unprocessed_messages = {}
    w.streaming_buffers = {}
    w.communicator = manager.communicator
    w.tensor_manager = manager
    w._graph_runtime = SimpleNamespace(
        get_rid_handle=rid_table.handle, get_rid_string=rid_table.name,
        get_sharding_config=lambda handle: SimpleNamespace(groups=[]),
        remove_request=rid_table.release,
    )
    w.engine_manager = SimpleNamespace(remove_request=lambda handle: None)
    w.request_state = SimpleNamespace(remove_request=lambda handle: None)
    w.streams = StreamBufferManager(w.request_state, manager, [])
    w.profile_info = SimpleNamespace(pop_request=lambda handle: None)
    w.scheduler = SimpleNamespace(
        clear_rid=lambda handle, wire_id: None, fail_rids=lambda handles: None,
        pending_tp_follow_count={},
    )
    return w


def _conductor():
    c = Conductor.__new__(Conductor)
    c.communicator = _Recorder()
    c.requests = {"request": RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: ["w0", "w1"]}, all_worker_graph_ids={0},
        max_output_tokens=1, random_seed=0, resource_configs={},
    )}
    c.draining = {}
    c.enable_prof = False
    c.producer_triggered_partitions = set()
    c.requests["request"].partition_states["default"] = PartitionState(
        partition_name="default", metadata=CurrentForwardConductorMetadata(graph_walk="decode", is_prefill=False),
        current_worker_graph_ids={0},
    )
    c._client_streams = {}
    c._early_reads_done = {}
    c._drain_ttl_s = 120
    c._draining_deadlines = deque()
    c._try_admit_waiting = lambda: None
    return c


@pytest.mark.parametrize("teardown_order", [("w0", "w1"), ("w1", "w0")])
@pytest.mark.parametrize("read_timing", ["normal", "stray", "client-stream"])
@pytest.mark.parametrize("bookkeeping_type", ["python", "rust"])
def test_successful_cleanup_does_not_invalidate_a_read(
    monkeypatch, teardown_order, read_timing, bookkeeping_type,
):
    bookkeeping = PythonTensorBookkeeping
    if bookkeeping_type == "rust":
        pytest.importorskip("mstar_rust")
        bookkeeping = RustTensorBookkeeping
    peers = {}
    monkeypatch.setattr(tensors, "MooncakeTransferEngine", lambda **kw: _CheckedCopyEngine(peers=peers, **kw))
    managers, workers, tables = {}, {}, {}
    for entity_id in ("w0", "w1"):
        table = tables[entity_id] = RidTable()
        if entity_id == "w1":
            table.intern("another-request")
        handle = table.intern("request")
        manager = managers[entity_id] = tensors.MooncakeCommunicationManager(
            my_entity_id=entity_id, hostname="localhost", device="cpu",
            communicator=_Recorder(), protocol=CommProtocol.TCP,
        )
        manager.tensor_store = TensorStore(bookkeeping=bookkeeping())
        manager.rid_to_str = table.name
        config = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
        config.setup({})
        manager.register_request(handle, config)
        workers[entity_id] = _worker(entity_id, manager, table)

    producer, receiver = managers["w0"], managers["w1"]
    producer_rid = tables["w0"].handle("request")
    receiver_rid = tables["w1"].handle("request")
    assert producer_rid != receiver_rid
    payload = torch.arange(16, dtype=torch.float32)
    [info] = producer.store_and_return_tensor_info(producer_rid, {"extra": [payload]})["extra"]
    producer.increment_ref(info.uuid)
    producer.register_for_send(producer_rid, [info])
    engine = receiver.transfer_engine
    edge = GraphEdge(
        name="extra", next_node="already_finished_node", tensor_info=[info],
        _total_fanin=1, _shard_dim=None,
    )
    c = _conductor()
    def report_reads():
        for entity, manager in managers.items():
            handle = tables[entity].handle("request")
            c._process_worker_graphs_done(WorkerGraphsDone(
                request_id="request", worker_graph_ids=[0],
                is_first_tp_rank=entity == "w0", worker_id=entity,
                has_inflight_reads=handle in manager.inflight_read_rids([handle]),
            ))

    if read_timing == "client-stream":
        # Open input can arrive after the snapshot, so it must always drain.
        c._client_streams = {"audio": object()}
        report_reads()
        c._process_request_done("request")
    [future] = receiver.start_read_tensors(receiver_rid, [edge])
    try:
        assert engine.started.wait(5), "background read did not start"
        assert not future.done()
        if read_timing == "normal":
            engine.resume.set()
            future.result(timeout=5)
            assert receiver.get_ready_tensors()
            for _, ack in receiver.communicator.sent:
                workers["w0"]._handle_tensor_received(ack.body)
            receiver.communicator.sent.clear()

        if read_timing != "client-stream":
            report_reads()
            c._process_request_done("request")
        c._handle_reads_done(ReadsDone(request_id="request", entity_id=c.PREPROCESS_WORKER))
        drains = {
            entity: msg.body for entity, msg in c.communicator.sent
            if msg.message_type == WorkerMessageType.DRAIN_REQUEST
        }
        assert set(drains) == (set() if read_timing == "normal" else {"w0", "w1"})
        for entity in teardown_order:
            if entity in drains:
                workers[entity]._drain_request(drains[entity])

        def deliver_reads_done():
            for manager in managers.values():
                for _, msg in manager.communicator.sent:
                    if msg.message_type == ConductorMessageType.READS_DONE:
                        c._handle_reads_done(msg.body)
                manager.communicator.sent = [
                    (e, m) for e, m in manager.communicator.sent
                    if m.message_type != ConductorMessageType.READS_DONE
                ]

        deliver_reads_done()
        if "request" in c.draining:
            # The receiver holds its READS_DONE, and so the removal, until its
            # in-flight read has finished.
            assert read_timing != "normal"
            assert c.draining["request"].expected_acks == {"w1"}
            engine.resume.set()
            future.result(timeout=5)
            assert receiver.get_ready_tensors()
            workers["w1"]._apply_pending_drains(set())
            deliver_reads_done()

        # Until READS_DONE, a committed TP step may still need a late input.
        # After the ACK, the real handler must not start any additional read.
        if "w1" in drains:
            assert "request" in workers["w1"]._reads_done_sent
            pending = list(receiver.pending)
            workers["w1"]._process_new_inputs(InputSignals(
                request_id="request", inputs=[edge], request_info=None,
            ))
            assert receiver.pending == pending

        removals = {
            entity: msg.body for entity, msg in c.communicator.sent
            if msg.message_type == WorkerMessageType.REMOVE_REQUEST
        }
        assert set(removals) == {"w0", "w1", c.PREPROCESS_WORKER}
        assert "request" not in c.requests
        for entity in teardown_order:
            workers[entity]._remove_request(removals[entity])
            assert tables[entity].handle("request") is None
            workers[entity]._drain_request(DrainRequest(request_id="request"))
            assert not workers[entity]._draining_rids
            assert not workers[entity]._reads_done_sent

        engine.resume.set()
        try:
            future.result(timeout=5)
        except RuntimeError as exc:
            pytest.fail(f"Race reproduced: cleanup invalidated {engine.invalid_endpoints} before read finished: {exc}")
        assert engine.copied == payload.numpy().tobytes()
        assert engine.invalid_endpoints == []
        assert receiver.get_ready_tensors() == {}
        assert not producer.tensor_store.check_uuid_presence(info.uuid)
        assert not receiver.tensor_store.check_uuid_presence(info.uuid)
        assert not producer.transfer_engine.regions
        assert not receiver.transfer_engine.regions
        for entity, old_handle in (("w0", producer_rid), ("w1", receiver_rid)):
            reused = tables[entity].intern("next-request")
            assert reused == old_handle
            assert not managers[entity].has_inflight_reads(reused)
            assert workers[entity]._takes_input("next-request")
    finally:
        engine.resume.set()
        for manager in managers.values():
            # A failed future must not prevent the pool from being joined.
            manager._async_reader._executor.shutdown(wait=True)
