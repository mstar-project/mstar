"""An empty tensor crosses Mooncake.

Mooncake refuses to register zero bytes, and an empty audio chunk is zero
bytes: Qwen3-Omni lost 4 of 24 audio outputs over RDMA to "Mooncake memory
registration failed". The sender leaves an empty tensor unregistered, the
reader hands on its empty buffer without a read, and neither side then
unregisters what it never registered.
"""

from __future__ import annotations

import ctypes
import sys
from concurrent.futures import Future
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication import tensors
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensors import MooncakeCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge


class _StubTransferEngine:
    """Registers the way Mooncake does: zero bytes, or a pointer it never
    registered, is refused with a nonzero status."""

    def __init__(self, **kwargs):
        self._engine = None
        self.registered: set[int] = set()

    def register_memory(self, ptr, nbytes):
        if nbytes == 0:
            return -1
        self.registered.add(ptr)
        return 0

    def unregister_memory(self, ptr):
        return 0 if ptr in self.registered else -1

    get_session_id = staticmethod(lambda: "localhost:0")


class _StubReader:
    """Reads by copying between this process's own buffers, before it returns."""

    def __init__(self, engine, device, enable_prof=False):
        self.nbytes: list[int] = []

    def submit(self, read_info):
        if not read_info:
            return None
        for info in read_info:
            self.nbytes.append(info.nbytes)
            ctypes.memmove(info.local_ptr, info.remote_ptr, info.nbytes)
        future = Future()
        future.set_result(None)
        return future


def _manager(entity_id: str) -> MooncakeCommunicationManager:
    mgr = MooncakeCommunicationManager(
        my_entity_id=entity_id, hostname="localhost", device="cpu", protocol=CommProtocol.TCP,
        communicator=SimpleNamespace(send=lambda entity_id, msg: None, get_all_new_messages=list),
    )
    sharding = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    sharding.setup({})
    mgr.register_request("req", sharding)
    return mgr


@pytest.mark.parametrize(
    "audio", [[torch.empty(0), torch.arange(6.0)], [torch.empty(0)]], ids=["beside_a_full_chunk", "alone"],
)
def test_an_empty_tensor_crosses_the_wire(monkeypatch, audio):
    monkeypatch.setattr(tensors, "MooncakeTransferEngine", _StubTransferEngine)
    monkeypatch.setattr(tensors, "AsyncMooncakeReader", _StubReader)
    sender, receiver = _manager("Code2Wav"), _manager("api_server")
    edge = GraphEdge(next_node="api_server", name="audio")
    edge.tensor_info = sender.store_and_return_tensor_info("req", {"audio": audio}, skip_cuda_sync=True)["audio"]
    for info in edge.tensor_info:
        sender.increment_ref(info.uuid)
    sender.register_for_send("req", edge.tensor_info, skip_cuda_sync=True)

    receiver.start_read_tensors("req", [edge])
    (ready,) = receiver.get_ready_tensors()["req"]
    received = [receiver.get_tensor(info.uuid) for info in ready.tensor_info]
    # teardown unregisters what each side marked registered
    receiver.force_cleanup_request("req")
    sender.force_cleanup_request("req")

    assert all(torch.equal(got, sent) for got, sent in zip(received, audio, strict=True))
    assert receiver._async_reader.nbytes == [t.nbytes for t in audio if t.nbytes], "an empty tensor has nothing to read"
