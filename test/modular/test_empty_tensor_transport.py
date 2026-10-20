"""Empty (0-byte) tensors over the Mooncake transport.

Mooncake refuses to register a 0-byte region on every release ("Transfer Engine
does not support zero length memory region"). Streaming vocoders emit empty
chunks legitimately: Qwen3-Omni's Code2Wav returns 0 samples for the final
flush window when it holds only left context. Registering that chunk raised in
the worker loop and failed every request in the batch.

The fake engine below enforces the Mooncake rules the transport depends on, so
these tests run on CPU without Mooncake. The last test repeats the round trip
with the real engine over TCP when it is installed.
"""
import ctypes
import itertools
import threading
from copy import deepcopy

import pytest
import torch

import mstar.communication.tensors as tensors_mod
from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    MOONCAKE_IMPORT_ERROR,
    create_tensor_communication_manager,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, NodeAndGraphWalk
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import WorkerMessageType


class _Recorder(BaseCommunicator):
    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send(self, entity_id: str, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


class _FakeMooncake:
    """In-process stand-in for ``mooncake.engine.TransferEngine``.

    Same return codes as the real engine: register -1 for a 0-byte region and
    -7 for an overlap, unregister -3 for an unknown address, and a read -1
    when the remote range is outside the peer's registered memory. Reads copy
    host memory, so both managers must use ``device="cpu"``.
    """

    sessions: dict[str, "_FakeMooncake"] = {}
    _ports = itertools.count(20000)
    instances: list["_FakeMooncake"] = []

    def __init__(self):
        self.port = next(self._ports)
        self.regions: dict[int, int] = {}
        self.calls: list[tuple] = []
        self.gate: threading.Event | None = None
        self.instances.append(self)

    def initialize(self, hostname, metadata_server, protocol, device):
        self.sessions[f"{hostname}:{self.port}"] = self
        return 0

    def get_rpc_port(self):
        return self.port

    def _covers(self, ptr, nbytes):
        return any(base <= ptr and ptr + nbytes <= base + size for base, size in self.regions.items())

    def register_memory(self, ptr, nbytes):
        if nbytes == 0:
            ret = -1
        elif any(ptr < base + size and base < ptr + nbytes for base, size in self.regions.items()):
            ret = -7
        else:
            self.regions[ptr] = nbytes
            ret = 0
        self.calls.append(("register", ptr, nbytes, ret))
        return ret

    def unregister_memory(self, ptr):
        ret = 0 if self.regions.pop(ptr, None) is not None else -3
        self.calls.append(("unregister", ptr, ret))
        return ret

    def batch_transfer_sync_read(self, session, local_ptrs, remote_ptrs, lengths):
        self.calls.append(("read", tuple(lengths)))
        if self.gate is not None:
            self.gate.wait(10)
        peer = self.sessions[session]
        if any(n and not peer._covers(r, n) for r, n in zip(remote_ptrs, lengths, strict=True)):
            return -1
        for local, remote, n in zip(local_ptrs, remote_ptrs, lengths, strict=True):
            ctypes.memmove(local, remote, n)
        return 0


@pytest.fixture
def fake_mooncake(monkeypatch):
    _FakeMooncake.sessions = {}
    _FakeMooncake.instances = []
    monkeypatch.setattr(tensors_mod, "TransferEngine", _FakeMooncake)
    return _FakeMooncake


def _manager(entity_id: str):
    return create_tensor_communication_manager(
        protocol=CommProtocol.TCP, my_entity_id=entity_id, hostname="localhost",
        device="cpu", communicator=_Recorder(), tcp_transfer_device="lo",
    )


def _pair(*request_ids):
    sender, receiver = _manager("w0"), _manager("w1")
    for rid in request_ids:
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
        cfg.setup({NodeAndGraphWalk("A", "decode"): ["w0"], NodeAndGraphWalk("B", "decode"): ["w1"]})
        sender.register_request(rid, cfg)
        receiver.register_request(rid, cfg)
    return sender, receiver


def _store(sender, rid, signal, tensors) -> GraphEdge:
    """Producer side up to registration: store, one reference for the edge."""
    infos = sender.store_and_return_tensor_info(rid, {signal: list(tensors)})[signal]
    for info in infos:
        sender.tensor_store.increment_ref(info.uuid, 1)
    return GraphEdge(next_node="B", name=signal, tensor_info=infos, _total_fanin=1, _shard_dim=None)


def _read_all(receiver, rid, edges):
    for fut in receiver.start_read_tensors(rid, [deepcopy(e) for e in edges], graph_walk="decode") or []:
        fut.result()
    return receiver.get_ready_tensors(graph_walk="decode")


def _apply_acks(sender, receiver):
    """What the producer's worker does with TENSOR_RECEIVED."""
    for _, msg in receiver.communicator.sent:
        assert msg.message_type == WorkerMessageType.TENSOR_RECEIVED
        acks = msg.body.successful_tensors
        sender.dereference_batch(list(acks), list(acks.values()))
    receiver.communicator.sent.clear()


def _engine(manager):
    return manager.transfer_engine._engine


EMPTY = [
    torch.empty(0, dtype=torch.int16),
    torch.empty(0, 16, dtype=torch.bfloat16),
    torch.empty(3, 0, dtype=torch.float32),
]


@pytest.mark.parametrize("empty", EMPTY, ids=lambda t: f"{t.dtype}{tuple(t.shape)}")
def test_empty_tensor_round_trips_without_touching_mooncake(fake_mooncake, empty):
    sender, receiver = _pair("r0")
    edge = _store(sender, "r0", "audio_chunk", [empty])
    sender.register_for_send("r0", edge.tensor_info)

    ready = _read_all(receiver, "r0", [edge])

    out = receiver.get_tensor(ready["r0"][0].tensor_info[0].uuid)
    assert out.shape == empty.shape and out.dtype == empty.dtype and out.device == empty.device
    calls = _engine(sender).calls + _engine(receiver).calls
    assert not [c for c in calls if c[0] == "register" and c[2] == 0], calls
    assert not [c for c in calls if c[0] == "read"], "an empty tensor needs no read"


def test_mixed_batch_of_empty_and_full_chunks_all_succeed(fake_mooncake):
    """The B200 failure: one empty final chunk among a batch of full ones.
    Registered the way the worker does it (uuids, one call for the batch)."""
    rids = [f"r{i}" for i in range(4)]
    sender, receiver = _pair(*rids)
    chunks = {rid: torch.randint(-3000, 3000, (1920 * (i % 2),), dtype=torch.int16) for i, rid in enumerate(rids)}
    edges = {rid: _store(sender, rid, "audio_chunk", [chunks[rid]]) for rid in rids}

    sender.register_for_send_uuids(
        ParallelList(rids, [[e.tensor_info[0].uuid] for e in edges.values()]), skip_cuda_sync=True,
    )
    for rid in rids:
        ready = _read_all(receiver, rid, [edges[rid]])
        out = receiver.get_tensor(ready[rid][0].tensor_info[0].uuid)
        assert torch.equal(out, chunks[rid])

    reads = [c[1] for c in _engine(receiver).calls if c[0] == "read"]
    assert reads == [(3840,), (3840,)], "one read per non-empty chunk, none of length 0"


def test_empty_tensor_accounting_is_balanced(fake_mooncake):
    """Acks reach the producer, both sides free every tensor, and Mooncake ends
    with no registered region and no failed unregister."""
    sender, receiver = _pair("r0")
    edge = _store(sender, "r0", "audio_chunk", [torch.empty(0, dtype=torch.int16), torch.ones(8, dtype=torch.int16)])
    sender.register_for_send("r0", edge.tensor_info)
    uuids = [info.uuid for info in edge.tensor_info]

    ready = _read_all(receiver, "r0", [edge])
    acked = {u for _, msg in receiver.communicator.sent for u in msg.body.successful_tensors}
    assert acked == set(uuids), "the empty tensor is acked like any other"
    _apply_acks(sender, receiver)
    for info in ready["r0"][0].tensor_info:
        receiver.dereference(info.uuid)  # the consumer's graph-node reference

    for manager in (sender, receiver):
        assert not any(manager.tensor_store.check_uuid_presence(u) for u in uuids)
        engine = _engine(manager)
        assert engine.regions == {}
        assert not [c for c in engine.calls if c[0] == "unregister" and c[2] != 0], engine.calls


def test_empty_edge_waits_for_the_requests_earlier_reads(fake_mooncake):
    """An edge of only empty tensors has no read, but it must not overtake a
    read the request started earlier: a stream's last empty chunk arrives
    after the chunk before it."""
    sender, receiver = _pair("r0")
    full = _store(sender, "r0", "audio_chunk", [torch.ones(64, dtype=torch.int16)])
    empty = _store(sender, "r0", "audio_chunk", [torch.empty(0, dtype=torch.int16)])
    sender.register_for_send("r0", full.tensor_info + empty.tensor_info)
    gate = threading.Event()
    _engine(receiver).gate = gate

    futures = receiver.start_read_tensors("r0", [deepcopy(full)], graph_walk="decode")
    receiver.start_read_tensors("r0", [deepcopy(empty)], graph_walk="decode")
    assert receiver.get_ready_tensors(graph_walk="decode") == {}
    gate.set()
    for fut in futures:
        fut.result()
    ready = receiver.get_ready_tensors(graph_walk="decode")

    shapes = [receiver.get_tensor(e.tensor_info[0].uuid).shape for e in ready["r0"]]
    assert shapes == [torch.Size([64]), torch.Size([0])]


@pytest.mark.skipif(MOONCAKE_IMPORT_ERROR is not None, reason="mooncake not installed")
def test_empty_and_full_tensors_round_trip_over_real_mooncake_tcp():
    sender, receiver = _pair("r0")
    full = torch.arange(32, dtype=torch.int16)
    edge = _store(sender, "r0", "audio_chunk", [torch.empty(0, dtype=torch.int16), full])
    sender.register_for_send("r0", edge.tensor_info)

    ready = _read_all(receiver, "r0", [edge])

    outs = [receiver.get_tensor(info.uuid) for info in ready["r0"][0].tensor_info]
    assert outs[0].shape == (0,) and outs[0].dtype == torch.int16
    assert torch.equal(outs[1], full)
    _apply_acks(sender, receiver)
    for info in ready["r0"][0].tensor_info:
        receiver.dereference(info.uuid)
