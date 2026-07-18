"""Unit tests for SharedMemoryCommunicationManager and tensor serialization."""

import os
import tempfile

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    _deserialize_tensor,
    _serialize_tensor,
    create_tensor_communication_manager,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.utils.ipc_format import WorkerMessageType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockCommunicator(BaseCommunicator):
    """Stub communicator that records sent messages."""

    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send(self, entity_id: str, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self) -> list:
        return []


def _empty_sharding_config() -> ShardingConfig:
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    cfg.setup({})
    return cfg


def _stub_info(t: torch.Tensor) -> TensorPointerInfo:
    """Build a minimal TensorPointerInfo carrying dims/dtype for the
    serialize/deserialize round-trip tests.
    """
    return TensorPointerInfo(
        dims=tuple(t.shape), dtype=t.dtype, stride=t.stride(),
        nbytes=t.nbytes, address=0, uuid="stub",
        source_session_id="local", source_entity="local",
    )


def _make_manager(
    shm_dir: str, entity_id: str = "worker_0",
    request_id: str | None = None,
) -> SharedMemoryCommunicationManager:
    mgr = SharedMemoryCommunicationManager(
        my_entity_id=entity_id,
        hostname="localhost",
        device="cpu",
        communicator=MockCommunicator(),
        shm_dir=shm_dir,
    )
    if request_id is not None:
        mgr.register_request(request_id, _empty_sharding_config())
    return mgr


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [
    torch.float32, torch.float64, torch.float16, torch.bfloat16,
    torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool,
])
def test_serialize_roundtrip_dtypes(dtype):
    if dtype == torch.bool:
        t = torch.tensor([True, False, True, False], dtype=dtype)
    elif dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        t = torch.randn(4, 8).to(dtype)
    else:
        t = torch.randint(0, 100, (4, 8), dtype=dtype)

    data = _serialize_tensor(t)
    t2 = _deserialize_tensor(data, "cpu", tensor_info=_stub_info(t))
    assert t2.shape == t.shape
    assert t2.dtype == t.dtype
    assert torch.equal(t2, t)


def test_serialize_scalar():
    t = torch.tensor(3.14, dtype=torch.float32)
    data = _serialize_tensor(t)
    t2 = _deserialize_tensor(data, "cpu", tensor_info=_stub_info(t))
    assert t2.shape == t.shape
    assert torch.equal(t2, t)


def test_serialize_empty():
    t = torch.empty(0, 3, dtype=torch.float32)
    data = _serialize_tensor(t)
    t2 = _deserialize_tensor(data, "cpu", tensor_info=_stub_info(t))
    assert t2.shape == t.shape


def test_serialize_high_dim():
    t = torch.randn(2, 3, 4, 5)
    data = _serialize_tensor(t)
    t2 = _deserialize_tensor(data, "cpu", tensor_info=_stub_info(t))
    assert torch.equal(t2, t)


# ---------------------------------------------------------------------------
# SharedMemoryCommunicationManager tests
# ---------------------------------------------------------------------------

def test_store_and_register_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        tensor = torch.randn(4, 8)
        info = mgr.store_and_return_tensor_info("req1", {"out": [tensor]})
        tensor_info = info["out"][0]
        uuid = tensor_info.uuid
        mgr.register_for_send("req1", [tensor_info])

        expected_path = os.path.join(tmpdir, f"mstar_worker_0_{uuid}")
        assert os.path.isfile(expected_path)


def test_full_sender_receiver_cycle():
    """Simulate a full sender → receiver cycle via SHM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")

        original = torch.randn(10, 32)

        # Sender: store + register
        edges = [GraphEdge(next_node="LLM", name="image_embs")]
        sender.store_and_populate_graph_edges("req1", {"image_embs": [original]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        sender.register_for_send("req1", edges[0].tensor_info)

        # Receiver: start read
        receiver.start_read_tensors("req1", edges, graph_walk="decode")

        # Receiver: poll ready
        ready = receiver.get_ready_tensors(graph_walk="decode")
        assert "req1" in ready
        assert len(ready["req1"]) == 1

        # Verify tensor equality
        received_tensor = receiver.get_tensor("req1", uuids[0])
        assert torch.equal(received_tensor, original)


def test_full_cycle_bfloat16():
    """Ensure bfloat16 tensors survive the SHM round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")

        original = torch.randn(5, 16, dtype=torch.bfloat16)

        edges = [GraphEdge(next_node="LLM", name="embs")]
        sender.store_and_populate_graph_edges("req1", {"embs": [original]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        sender.register_for_send("req1", edges[0].tensor_info)

        receiver.start_read_tensors("req1", edges, graph_walk="decode")
        receiver.get_ready_tensors(graph_walk="decode")
        received = receiver.get_tensor("req1", uuids[0])
        assert torch.equal(received, original)


def test_cleanup_unlinks_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        tensor = torch.randn(4, 8)
        info = mgr.store_and_return_tensor_info("req1", {"out": [tensor]})
        tensor_info = info["out"][0]
        uuid = tensor_info.uuid
        mgr.register_for_send("req1", [tensor_info])

        path = os.path.join(tmpdir, f"mstar_worker_0_{uuid}")
        assert os.path.isfile(path)

        # Dereference to 0 triggers cleanup
        mgr.dereference("req1", uuid, n=0)  # ref is already 0
        mgr.cleanup_request("req1")
        assert not os.path.isfile(path)


def test_local_tensor_skips_shm():
    """When source_entity == my_entity_id, no SHM file I/O should occur."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        tensor = torch.randn(3, 3)

        edges = [GraphEdge(next_node="LLM", name="data")]
        mgr.store_and_populate_graph_edges("req1", {"data": [tensor]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        mgr.register_for_send("req1", edges[0].tensor_info)

        # Reading from self — should NOT open an SHM file, just increment ref
        mgr.start_read_tensors("req1", edges, graph_walk="decode")
        ready = mgr.get_ready_tensors(graph_walk="decode")
        assert "req1" in ready

        retrieved = mgr.get_tensor("req1", uuids[0])
        assert torch.equal(retrieved, tensor)


def test_ack_sent_on_remote_read():
    """Verify that get_ready_tensors sends a TENSOR_RECEIVED ACK for remote tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")

        original = torch.randn(2, 4)
        edges = [GraphEdge(next_node="node", name="t")]
        sender.store_and_populate_graph_edges("req1", {"t": [original]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        sender.register_for_send("req1", edges[0].tensor_info)

        receiver.start_read_tensors("req1", edges, graph_walk="decode")
        receiver.get_ready_tensors(graph_walk="decode")

        # Check that ACK was sent to "worker_0"
        comm = receiver.communicator
        assert len(comm.sent) == 1
        entity_id, msg = comm.sent[0]
        assert entity_id == "worker_0"


def _produce_registered_output(mgr, request_id, name="audio_output"):
    """Producer stores an output edge bound for the api server and registers it
    for send, leaving it held by the +1 awaiting-ack ref (returns the uuid)."""
    edges = [GraphEdge(next_node="api_server", name=name)]
    mgr.store_and_populate_graph_edges(request_id, {name: [torch.randn(8, 16)]}, edges)
    uuid = edges[0].tensor_info[0].uuid
    mgr.register_for_send(request_id, edges[0].tensor_info)
    return edges, uuid


def test_unacked_result_tensors_leak_producer_buffer():
    """Without an ack, the producer never reclaims the output buffer.

    A result dropped for an already-removed request leaves the producer holding
    the buffer (ref never released), so even its own cleanup defers it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        producer = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        _, uuid = _produce_registered_output(producer, "req1")

        shm_path = os.path.join(tmpdir, f"mstar_worker_0_{uuid}")
        assert os.path.isfile(shm_path)
        assert not producer.tensor_store.can_gc("req1", uuid)  # held by send ref

        # No ack arrives; the producer's own cleanup must defer the buffer.
        producer.cleanup_request("req1")
        assert os.path.isfile(shm_path)  # leaked: still on disk, awaiting an ack


def test_ack_unread_tensors_lets_producer_reclaim_buffer():
    """ack_unread_tensors emits the TENSOR_RECEIVED that frees the producer.

    Acking a result for a removed request (without reading) lets the producing
    worker reclaim the buffer it holds.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        producer = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        edges, uuid = _produce_registered_output(producer, "req1")
        shm_path = os.path.join(tmpdir, f"mstar_worker_0_{uuid}")
        assert os.path.isfile(shm_path)

        # api server never registered "req1" (unknown/removed request) yet still
        # acks the abandoned tensors back to the producer.
        api_server = _make_manager(tmpdir, entity_id="api_server_preprocess_worker")
        api_server.ack_unread_tensors("req1", edges)

        sent = api_server.communicator.sent
        assert len(sent) == 1
        entity_id, msg = sent[0]
        assert entity_id == "worker_0"
        assert msg.message_type == WorkerMessageType.TENSOR_RECEIVED
        assert msg.body.successful_tensors == {uuid: 1}

        # Producer applies the ack (mirrors worker._handle_tensor_received) and
        # reclaims the buffer.
        for u, n in msg.body.successful_tensors.items():
            producer.dereference("req1", u, n=n)
        assert not os.path.isfile(shm_path)  # reclaimed -> no leak


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

def test_factory_returns_shm_manager():
    comm = MockCommunicator()
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = create_tensor_communication_manager(
            protocol=CommProtocol.SHM,
            my_entity_id="w0",
            hostname="localhost",
            device="cpu",
            communicator=comm,
            shm_dir=tmpdir,
        )
        assert isinstance(mgr, SharedMemoryCommunicationManager)


def test_factory_returns_mooncake_for_rdma():
    """Factory should return MooncakeCommunicationManager for non-SHM protocols.

    Note: This test may fail if mooncake is not installed. We just check the
    type is correct when it doesn't raise.
    """
    comm = MockCommunicator()
    try:
        mgr = create_tensor_communication_manager(
            protocol=CommProtocol.TCP,
            my_entity_id="w0",
            hostname="localhost",
            device="cpu",
            communicator=comm,
        )
        assert isinstance(mgr, MooncakeCommunicationManager)
    except RuntimeError:
        # Mooncake not installed — expected in CI/dev environments
        pytest.skip("mooncake not installed")
