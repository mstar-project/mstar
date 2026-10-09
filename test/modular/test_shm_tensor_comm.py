"""Unit tests for SharedMemoryCommunicationManager and tensor serialization."""

import errno
import logging
import os
import shutil
import tempfile
from types import SimpleNamespace

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
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.utils.containers import ParallelList
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


def _store_on_edges(mgr, rid, tensors, edges):
    """Store ``tensors`` and fill in the edges that carry them, holding one
    reference per consuming edge -- the setup a producer's routing does."""
    infos = mgr.store_and_return_tensor_info(rid, tensors)
    for name, name_infos in infos.items():
        consumers = [e for e in edges if e.name == name]
        for info in name_infos:
            mgr.increment_ref(info.uuid, n=len([
                e for e in consumers if e.next_node != EMPTY_DESTINATION
            ]))
        for edge in consumers:
            edge.tensor_info = name_infos
    return infos


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


class _DeploymentCommunicator(MockCommunicator):
    def __init__(self, prefix: str, protocol=CommProtocol.IPC):
        super().__init__()
        self.protocol = protocol
        self.ipc_socket_path_prefix = prefix


def _deployed_manager(shm_dir, prefix, entity_id="worker_0",
                      protocol=CommProtocol.IPC):
    mgr = SharedMemoryCommunicationManager(
        my_entity_id=entity_id, hostname="localhost", device="cpu",
        communicator=_DeploymentCommunicator(prefix, protocol), shm_dir=shm_dir,
    )
    mgr.register_request("req1", _empty_sharding_config())
    return mgr


def test_two_deployments_on_one_host_do_not_share_files():
    """Tensor uuids are per-entity counters, so both servers' ``worker_0``
    mint the same uuid. Named by entity and uuid alone, the second server
    overwrote the first's file and its teardown unlinked a tensor the first
    was still reading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _deployed_manager(tmpdir, "/tmp/mstar_a/")
        b = _deployed_manager(tmpdir, "/tmp/mstar_b/")
        [ia] = a.store_and_return_tensor_info("req1", {"out": [torch.ones(4)]})["out"]
        [ib] = b.store_and_return_tensor_info("req1", {"out": [torch.zeros(4)]})["out"]
        assert ia.uuid == ib.uuid, "precondition: the uuids really do collide"
        a.register_for_send("req1", [ia])
        b.register_for_send("req1", [ib])

        assert a._shm_path("worker_0", ia.uuid) != b._shm_path("worker_0", ib.uuid)
        b.force_cleanup_request("req1")
        assert os.path.isfile(a._shm_path("worker_0", ia.uuid)), (
            "the other deployment's teardown removed this one's tensor"
        )


def test_two_tcp_deployments_do_not_share_files(monkeypatch):
    """TCP ignores the socket prefix, so both deployments carry the default
    one; the base port is what actually separates them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("MSTAR_ZMQ_TCP_BASE_PORT", "19000")
        a = _deployed_manager(tmpdir, "/tmp/mstar/", protocol=CommProtocol.TCP)
        monkeypatch.setenv("MSTAR_ZMQ_TCP_BASE_PORT", "29000")
        b = _deployed_manager(tmpdir, "/tmp/mstar/", protocol=CommProtocol.TCP)
        [ia] = a.store_and_return_tensor_info("req1", {"out": [torch.ones(4)]})["out"]
        [ib] = b.store_and_return_tensor_info("req1", {"out": [torch.zeros(4)]})["out"]
        assert ia.uuid == ib.uuid, "precondition: the uuids really do collide"
        a.register_for_send("req1", [ia])
        b.register_for_send("req1", [ib])

        assert a._shm_path("worker_0", ia.uuid) != b._shm_path("worker_0", ib.uuid)
        b.force_cleanup_request("req1")
        assert os.path.isfile(a._shm_path("worker_0", ia.uuid)), (
            "the other deployment's teardown removed this one's tensor"
        )


def test_one_deployment_agrees_on_the_file_name():
    """Producer and consumer derive the name independently, so the prefix has
    to normalise: a trailing slash is the same deployment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _deployed_manager(tmpdir, "/tmp/mstar_a/", "worker_0")
        receiver = _deployed_manager(tmpdir, "/tmp/mstar_a", "worker_1")
        original = torch.randn(3, 5)
        edges = [GraphEdge(next_node="LLM", name="x")]
        _store_on_edges(sender, "req1", {"x": [original]}, edges)
        sender.register_for_send("req1", edges[0].tensor_info)
        receiver.start_read_tensors("req1", edges, graph_walk="decode")
        receiver.get_ready_tensors(graph_walk="decode")
        uuid = edges[0].tensor_info[0].uuid
        assert torch.equal(receiver.get_tensor(uuid), original)


def test_full_sender_receiver_cycle():
    """Simulate a full sender → receiver cycle via SHM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")

        original = torch.randn(10, 32)

        # Sender: store + register
        edges = [GraphEdge(next_node="LLM", name="image_embs")]
        _store_on_edges(sender, "req1", {"image_embs": [original]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        sender.register_for_send("req1", edges[0].tensor_info)

        # Receiver: start read
        receiver.start_read_tensors("req1", edges, graph_walk="decode")

        # Receiver: poll ready
        ready = receiver.get_ready_tensors(graph_walk="decode")
        assert "req1" in ready
        assert len(ready["req1"]) == 1

        # Verify tensor equality
        received_tensor = receiver.get_tensor(uuids[0])
        assert torch.equal(received_tensor, original)


def test_full_cycle_bfloat16():
    """Ensure bfloat16 tensors survive the SHM round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")

        original = torch.randn(5, 16, dtype=torch.bfloat16)

        edges = [GraphEdge(next_node="LLM", name="embs")]
        _store_on_edges(sender, "req1", {"embs": [original]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        sender.register_for_send("req1", edges[0].tensor_info)

        receiver.start_read_tensors("req1", edges, graph_walk="decode")
        receiver.get_ready_tensors(graph_walk="decode")
        received = receiver.get_tensor(uuids[0])
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
        mgr.dereference(uuid, n=0)  # ref is already 0
        mgr.cleanup_request("req1")
        assert not os.path.isfile(path)


class _StubFullDir:
    """Opens files that take half of a write and then fail it, as a tmpfs out of space does."""

    def __init__(self):
        self.partial: dict[str, int] = {}

    def __call__(self, path: str):
        return _StubFullFile(self, path)


class _StubFullFile:
    """One file of a ``_StubFullDir``; records the bytes it kept before failing."""

    def __init__(self, full_dir: _StubFullDir, path: str):
        self._dir, self._path, self._f = full_dir, path, open(path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._f.close()

    def write(self, data: bytes):
        self._f.write(data[: len(data) // 2])
        self._f.flush()
        self._dir.partial[self._path] = os.path.getsize(self._path)
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def _send_into(monkeypatch, mgr, full_dir: _StubFullDir) -> tuple[TensorPointerInfo, OSError]:
    info = mgr.store_and_return_tensor_info("req1", {"out": [torch.zeros(4096)]})["out"][0]
    monkeypatch.setattr(mgr, "_create_shm_file", full_dir)
    with pytest.raises(OSError) as err:
        mgr.register_for_send("req1", [info])
    return info, err.value


def test_a_full_shm_dir_fails_the_send_with_what_ran_out(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    with tempfile.TemporaryDirectory() as tmpdir:
        info, err = _send_into(monkeypatch, _make_manager(tmpdir, request_id="req1"), _StubFullDir())
    ran_out = f"{tmpdir} has 1024 bytes free, too few for a {info.nbytes}-byte tensor"
    assert ran_out in str(err) and "--shm-size" in str(err), "the error hides what ran out"


def test_a_full_shm_dir_with_room_left_is_not_called_too_small(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _, err = _send_into(monkeypatch, _make_manager(tmpdir, request_id="req1"), _StubFullDir())
    assert "after the failed write" in str(err) and "too few" not in str(err), (
        "the error would blame the dir's size for inodes or another writer"
    )


def test_a_failed_send_leaves_nothing_in_the_shm_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, full_dir = _make_manager(tmpdir, request_id="req1"), _StubFullDir()
        info, _ = _send_into(monkeypatch, mgr, full_dir)
        path = mgr._shm_path(mgr.my_entity_id, info.uuid)
        assert full_dir.partial[path] > 0, "the write must fail part way, as a filling tmpfs does"
        assert not os.path.lexists(path), "a partial file no cleanup tracks would hold the space for good"


def test_a_small_shm_dir_warns_at_boot(monkeypatch, caplog):
    # 64 MiB, Docker's default
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_frsize=4096, f_blocks=16384))
    with tempfile.TemporaryDirectory() as tmpdir, caplog.at_level(logging.WARNING):
        _make_manager(tmpdir)
    assert "--shm-size" in caplog.text, "a small /dev/shm would show only as the first large send failing"


def test_cleanup_collectable_reclaims_what_a_runtime_already_freed():
    """A graph runtime behind the contract dereferences inside the bookkeeper
    it shares, so nothing here ever sees the count hit zero. It hands the
    uuids back instead -- and until that path existed, a consumed input's shm
    file sat until the whole request was torn down."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        info = mgr.store_and_return_tensor_info(
            "req1", {"out": [torch.randn(4, 8)]}
        )["out"][0]
        mgr.register_for_send("req1", [info])
        path = os.path.join(tmpdir, f"mstar_worker_0_{info.uuid}")
        assert os.path.isfile(path)

        # Exactly what cleanup_consumed_inputs does and returns.
        freed = mgr.tensor_store.bookkeeping.dereference_batch_uniform(
            [info.uuid], 1, True,
        )
        assert freed[0] == [info.uuid]
        mgr.cleanup_collectable(*freed)

        assert not os.path.isfile(path), "the shm file outlived the tensor"
        assert not mgr.tensor_store.check_uuid_presence(info.uuid)


def test_local_tensor_skips_shm():
    """When source_entity == my_entity_id, no SHM file I/O should occur."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        tensor = torch.randn(3, 3)

        edges = [GraphEdge(next_node="LLM", name="data")]
        _store_on_edges(mgr, "req1", {"data": [tensor]}, edges)
        uuids = [info.uuid for info in edges[0].tensor_info]
        mgr.register_for_send("req1", edges[0].tensor_info)

        # Reading from self — should NOT open an SHM file, just increment ref
        mgr.start_read_tensors("req1", edges, graph_walk="decode")
        ready = mgr.get_ready_tensors(graph_walk="decode")
        assert "req1" in ready

        retrieved = mgr.get_tensor(uuids[0])
        assert torch.equal(retrieved, tensor)


def test_ack_sent_on_remote_read():
    """Verify that get_ready_tensors sends a TENSOR_RECEIVED ACK for remote tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")

        original = torch.randn(2, 4)
        edges = [GraphEdge(next_node="node", name="t")]
        _store_on_edges(sender, "req1", {"t": [original]}, edges)
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
    _store_on_edges(mgr, request_id, {name: [torch.randn(8, 16)]}, edges)
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
        assert not producer.tensor_store.can_gc(uuid)  # held by send ref

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
            producer.dereference(u, n=n)
        assert not os.path.isfile(shm_path)  # reclaimed -> no leak


# ---------------------------------------------------------------------------
# Teardown: cleanup_request (soft) vs force_cleanup_request (hard) + drain gate
# ---------------------------------------------------------------------------

def _store_persisted_input(mgr, request_id, name="in"):
    """Mirror the data worker's input-signal path: store, register for send, and
    mark persisted (ref_cnt stays 0 — persist is the only thing holding it)."""
    info = mgr.store_and_return_tensor_info(request_id, {name: [torch.randn(4, 8)]})
    tensor_info = info[name][0]
    mgr.register_for_send(request_id, [tensor_info])
    mgr.set_persist(tensor_info.uuid, persist=True)
    return tensor_info.uuid


def test_cleanup_request_defers_persisted_tensor():
    """Regression for the abort SHM bug: a persisted input-signal (ref_cnt==0,
    kept alive only by the persist flag) must NOT be unlinked by cleanup_request.
    Force-dropping it here is exactly what unlinked a segment under a peer's
    still-scheduled read and killed the rank."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(
            tmpdir, entity_id="api_server_preprocess_worker", request_id="req1"
        )
        uuid = _store_persisted_input(mgr, "req1")
        path = os.path.join(tmpdir, f"mstar_api_server_preprocess_worker_{uuid}")
        assert os.path.isfile(path)
        assert not mgr.tensor_store.can_gc(uuid)  # persist holds it

        mgr.cleanup_request("req1")
        assert os.path.isfile(path)  # deferred, not force-unlinked


def test_force_cleanup_request_drops_persisted_tensor():
    """Phase-2 hard cleanup drops the persisted signal unconditionally — safe
    only after every reader has drained (READS_DONE)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(
            tmpdir, entity_id="api_server_preprocess_worker", request_id="req1"
        )
        uuid = _store_persisted_input(mgr, "req1")
        path = os.path.join(tmpdir, f"mstar_api_server_preprocess_worker_{uuid}")
        assert os.path.isfile(path)

        mgr.force_cleanup_request("req1")
        assert not os.path.isfile(path)
        assert not mgr.tensor_store.check_uuid_presence(uuid)


def test_force_cleanup_request_reclaims_unacked_buffer():
    """Hard cleanup also reclaims a non-persisted output buffer still held by its
    awaiting-ack ref — the ~leaked segments cleanup_request would defer forever."""
    with tempfile.TemporaryDirectory() as tmpdir:
        producer = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        _, uuid = _produce_registered_output(producer, "req1")
        shm_path = os.path.join(tmpdir, f"mstar_worker_0_{uuid}")
        assert os.path.isfile(shm_path)
        assert not producer.tensor_store.can_gc(uuid)  # held by send ref

        producer.force_cleanup_request("req1")
        assert not os.path.isfile(shm_path)  # dropped regardless of ref count


def test_has_inflight_reads_tracks_pending_futures():
    """The drain gate: only an outstanding async read future counts. Synchronous
    (SHM) reads land in pending with future=None and are already done."""
    from mstar.communication.tensors import FutureAndPointers

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        assert not mgr.has_inflight_reads("req1")

        # Completed synchronous read (future=None) does not count.
        mgr.pending.append(
            FutureAndPointers(future=None, graph_edges=[], request_id="req1")
        )
        assert not mgr.has_inflight_reads("req1")

        # An unresolved async future does.
        class _Fut:
            def done(self):
                return False

        mgr.pending.append(
            FutureAndPointers(future=_Fut(), graph_edges=[], request_id="req1")
        )
        assert mgr.has_inflight_reads("req1")
        assert not mgr.has_inflight_reads("other-req")


def test_a_read_that_finishes_early_waits_for_the_ones_started_before_it():
    """The async reader runs reads on a thread pool, so a request's second read
    can finish before its first. Its edges still have to come out in order."""
    from mstar.communication.tensors import FutureAndPointers

    class _Fut:
        def __init__(self, done: bool):
            self.finished = done

        def done(self):
            return self.finished

        def result(self):
            return None

    def _read(future, rid: str, name: str) -> FutureAndPointers:
        return FutureAndPointers(
            future=future, graph_edges=[GraphEdge(next_node="LLM", name=name)],
            request_id=rid,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        mgr.register_request("req2", _empty_sharding_config())
        first = _Fut(done=False)
        mgr.pending = [
            _read(first, "req1", "token_0"),
            _read(_Fut(done=True), "req1", "token_1"),
            _read(_Fut(done=True), "req2", "token_0"),
        ]

        ready = mgr.get_ready_tensors()
        assert "req1" not in ready, "a token went out ahead of the one before it"
        assert [e.name for e in ready["req2"]] == ["token_0"], (
            "another request's read waited on this one"
        )

        first.finished = True
        ready = mgr.get_ready_tensors()
        assert [e.name for e in ready["req1"]] == ["token_0", "token_1"], (
            "the request's reads came out out of order"
        )


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


# ---------------------------------------------------------------------------
# descriptor bookkeeping
# ---------------------------------------------------------------------------


def _descriptor_fields(info):
    """A descriptor as VALUES, not as an object.

    ``dims``/``stride`` are declared ``list[int]`` but arrive as whatever the
    producer had -- a ``torch.Size`` from ``tensor.shape``, a list back out of
    the Rust bookkeeper, which copies the descriptor rather than keeping the
    caller's object. Comparing the sequences directly would be testing which
    backend is installed.
    """
    return (
        tuple(info.dims), tuple(info.stride), info.dtype, info.nbytes,
        info.address, info.uuid, info.offset,
        info.shm_segment, info.shm_offset,
    )


def test_store_keeps_a_descriptor_for_every_uuid_it_holds(tmp_path):
    """Ingestion and routing carry uuids, so anything that has to put a
    descriptor back on the wire — a disaggregated loop re-emitting its
    external inputs — has to be able to recover it from the uuid alone.

    Recover, not share: the Rust bookkeeper copies the descriptor in, so the
    two are equal without being the same object, and a later in-place edit
    reaches the store through ``update_info`` instead (see
    ``test/rust/test_arena_transport.py``'s write-back cases)."""
    mgr = _make_manager(str(tmp_path), request_id="r1")
    infos = mgr.store_and_return_tensor_info(
        "r1", {"h": [torch.randn(4, 8)], "e": [torch.empty(0, 3)]},
    )
    for name, info_list in infos.items():
        for info in info_list:
            got = mgr.tensor_store.get_info(info.uuid)
            assert got is not None, f"no descriptor kept for {name}"
            assert _descriptor_fields(got) == _descriptor_fields(info)


def test_descriptor_survives_register_for_send(tmp_path):
    """Registering a tensor for sending must not lose or stale the stored
    descriptor: the file transport writes the bytes out and leaves the
    descriptor alone, so what the store holds afterwards still points at the
    same tensor."""
    mgr = _make_manager(str(tmp_path), request_id="r1")
    infos = mgr.store_and_return_tensor_info("r1", {"h": [torch.randn(4, 8)]})
    info = infos["h"][0]
    mgr.register_for_send("r1", [info])

    stored = mgr.tensor_store.get_info(info.uuid)
    assert _descriptor_fields(stored) == _descriptor_fields(info)


def test_descriptor_is_dropped_with_the_tensor(tmp_path):
    """Descriptors are keyed by uuid, and uuids are never reused, but a leak
    here would grow without bound over a long-lived worker."""
    mgr = _make_manager(str(tmp_path), request_id="r1")
    infos = mgr.store_and_return_tensor_info("r1", {"h": [torch.randn(4, 8)]})
    uuid = infos["h"][0].uuid
    assert mgr.tensor_store.get_info(uuid) is not None

    mgr.tensor_store.remove_tensor(uuid)
    assert mgr.tensor_store.get_info(uuid) is None


def test_an_edge_rebuilt_from_uuids_alone_is_readable_by_a_peer(tmp_path):
    """The premise the graph-runtime port rests on.

    ingest_inputs_batch / complete_and_route_batch carry uuids, not
    descriptors. A disaggregated loop re-emits its ingested external inputs
    as outputs, which can be routed to another worker -- and there the
    descriptor IS the payload the peer RDMA-reads from. So reconstructing an
    edge from uuids has to produce something a peer can actually read; a
    stub carrying only the uuid would send a structurally valid but
    unreadable message.
    """
    producer = _make_manager(str(tmp_path), "worker_0", request_id="r1")
    consumer = _make_manager(str(tmp_path), "worker_1", request_id="r1")

    original = torch.randn(4, 8)
    infos = producer.store_and_return_tensor_info("r1", {"h": [original]})
    uuid = infos["h"][0].uuid
    producer.register_for_send("r1", [infos["h"][0]])

    # Nothing kept but the uuid.
    rebuilt = GraphEdge(
        name="h", next_node="consumer_node",
        tensor_info=[producer.tensor_store.get_info(uuid)],
    )
    consumer.start_read_tensors("r1", [rebuilt])
    assert torch.equal(consumer.tensor_store.get_tensor(uuid), original)


# ---------------------------------------------------------------------------
# Owning the create (O_EXCL)
# ---------------------------------------------------------------------------

def test_write_reclaims_our_own_stale_file():
    """The namespace comes from the conductor endpoint, so a deployment
    restarted on the same ports collides with every file its last run left --
    and its uuid counters restart at 1 too. Our own residue is reclaimed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        [info] = mgr.store_and_return_tensor_info(
            "req1", {"out": [torch.ones(4)]},
        )["out"]
        path = mgr._shm_path("worker_0", info.uuid)
        with open(path, "wb") as f:
            f.write(b"residue from the run that crashed")

        mgr.register_for_send("req1", [info])
        with open(path, "rb") as f:
            assert f.read(4) != b"resi"


def test_write_refuses_a_file_we_do_not_own(monkeypatch):
    """The path is namespace + entity + counter: predictable, in a
    world-writable directory. Writing through into someone else's file is
    what O_EXCL is there to stop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        [info] = mgr.store_and_return_tensor_info(
            "req1", {"out": [torch.ones(4)]},
        )["out"]
        path = mgr._shm_path("worker_0", info.uuid)
        with open(path, "wb") as f:
            f.write(b"squatted")
        # Cheaper than actually running as a second uid.
        monkeypatch.setattr(os, "getuid", lambda: os.stat(path).st_uid + 1)

        with pytest.raises(RuntimeError, match="not ours"):
            mgr.register_for_send("req1", [info])
        with open(path, "rb") as f:
            assert f.read() == b"squatted", "someone else's file was written"


def test_write_refuses_a_symlink_planted_at_the_path():
    """A symlink we own still points wherever its planter chose; O_EXCL fails
    on it and the ownership check must not follow it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir, request_id="req1")
        [info] = mgr.store_and_return_tensor_info(
            "req1", {"out": [torch.ones(4)]},
        )["out"]
        path = mgr._shm_path("worker_0", info.uuid)
        target = os.path.join(tmpdir, "elsewhere")
        with open(target, "wb") as f:
            f.write(b"do not clobber")
        os.symlink(target, path)

        with pytest.raises(RuntimeError, match="not ours"):
            mgr.register_for_send("req1", [info])
        with open(target, "rb") as f:
            assert f.read() == b"do not clobber"


def test_register_for_send_uuids_writes_every_request_not_just_the_first():
    """The uuid-driven form on the FILE transport. Only the arena subclass
    covered this before, so truncating the loop to the first request passed
    the suite -- rb and rc would never reach a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir)
        per_request: dict[str, list] = {}
        for rid in ("ra", "rb", "rc"):
            mgr.register_request(rid, _empty_sharding_config())
            infos = mgr.store_and_return_tensor_info(
                rid, {"out": [torch.randn(4, 8)]},
            )
            per_request[rid] = [i for il in infos.values() for i in il]

        mgr.register_for_send_uuids(ParallelList(
            list(per_request),
            [[i.uuid for i in infos] for infos in per_request.values()],
        ))

        for rid, infos in per_request.items():
            for info in infos:
                assert info.uuid in mgr._shm_files, f"{rid} never written"
                assert os.path.exists(mgr._shm_files[info.uuid])


# ---------------------------------------------------------------------------
# Host copies: a transport that sends from host memory reuses the producer's
# ---------------------------------------------------------------------------

def _send_and_read(sender, receiver, rid, edges):
    sender.register_for_send(rid, edges[0].tensor_info)
    receiver.start_read_tensors(rid, edges, graph_walk="decode")
    receiver.get_ready_tensors(graph_walk="decode")
    return receiver.get_tensor(edges[0].tensor_info[0].uuid)


def test_shm_sends_the_stored_host_copy():
    """The send reads the host copy when there is one. The copy differs from
    the stored tensor here only so the test can tell which one was sent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")
        assert sender.needs_cpu_tensor

        tensor, host = torch.zeros(1, 4), torch.arange(4.0).reshape(1, 4)
        infos = sender.store_and_return_tensor_info(
            "req1", {"tok": [tensor]}, cpu_tensors={"tok": [host]},
        )
        assert sender.tensor_store.get_cpu_tensor(infos["tok"][0].uuid) is host
        edges = [GraphEdge(next_node="LLM", name="tok", tensor_info=infos["tok"])]
        assert torch.equal(_send_and_read(sender, receiver, "req1", edges), host)


def test_a_later_send_does_not_read_a_dropped_host_copy():
    """The stop check's host copies are views of pinned buffers the next step
    reuses, so the worker drops them after the step's sends. A loop's
    accumulated tokens go out when the loop ends, after that reuse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sender = _make_manager(tmpdir, entity_id="worker_0", request_id="req1")
        receiver = _make_manager(tmpdir, entity_id="worker_1", request_id="req1")
        pinned = torch.tensor([[5, 6]])
        stored = sender.store_and_return_tensor_info_batch(
            ["req1"], {"req1": {"tok": [torch.tensor([[5, 6]])]}}, ["tok"],
            cpu_tensors={"req1": {"tok": [pinned[:1]]}},
        )
        sender.drop_host_copies(stored.flat_uuids)  # end of the producing step
        pinned.fill_(9)  # the next step's stop check
        infos = [sender.tensor_store.get_info(u) for u in stored.flat_uuids]
        edges = [GraphEdge(next_node="LLM", name="tok", tensor_info=infos)]
        assert _send_and_read(sender, receiver, "req1", edges).tolist() == [[5, 6]]


def test_host_copy_follows_a_renamed_output():
    """A submodule that rebinds an output under its signal name (Qwen3.5's
    ``new_token`` -> ``text_inputs``) aliases the tensor; the stop check's
    copy, keyed by the original name, still reaches the stored signal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _make_manager(tmpdir)
        token = [torch.zeros(1, dtype=torch.int64)]
        outputs = {7: {"new_token": token, "text_inputs": token}}
        host = torch.ones(1, dtype=torch.int64)
        stored = mgr.store_and_return_tensor_info_batch(
            [7], outputs, ["text_inputs"],
            cpu_tensors={7: {"new_token": [host]}},
        )
        assert mgr.tensor_store.get_cpu_tensor(stored.flat_uuids[0]) is host


def test_host_copies_not_kept_for_device_transports():
    """TCP / RDMA read device memory, so a host copy would only be held."""
    assert not MooncakeCommunicationManager.needs_cpu_tensor.fget(object())

    class _DeviceTransport(SharedMemoryCommunicationManager):
        needs_cpu_tensor = False

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = _DeviceTransport(
            my_entity_id="worker_0", hostname="localhost", device="cpu",
            communicator=MockCommunicator(), shm_dir=tmpdir,
        )
        infos = mgr.store_and_return_tensor_info(
            "req1", {"tok": [torch.zeros(1)]}, cpu_tensors={"tok": [torch.ones(1)]},
        )
        stored = mgr.store_and_return_tensor_info_batch(
            [1], {1: {"tok": [torch.zeros(1)]}}, ["tok"],
            cpu_tensors={1: {"tok": [torch.ones(1)]}},
        )
        for uuid in [infos["tok"][0].uuid, *stored.flat_uuids]:
            assert mgr.tensor_store.get_cpu_tensor(uuid) is None
