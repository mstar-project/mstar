"""Mooncake TCP stages CUDA tensors through pinned host memory on non-blocking
CUDA streams so Mooncake's ``TcpTransport`` (which uses synchronous
``cudaMemcpy`` on legacy stream 0) never serializes behind GPU compute or
deadlocks against an NCCL collective (#739).

RDMA stays device-direct (GPUDirect RDMA) and CPU-only entities stay zero-copy.
"""
from __future__ import annotations

import ctypes
from concurrent.futures import Future

import pytest
import torch

from mstar.communication import tensors
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import (
    _TRANSFER_COMPLETED,
    AsyncReadError,
    LocalTransferEngine,
    MemoryRegistrationError,
    MooncakeCommunicationManager,
    _flat_byte_view,
)
from mstar.graph.base import GraphEdge
from mstar.utils.containers import ParallelList


class _NullCommunicator:
    protocol = CommProtocol.TCP

    def _endpoint(self, entity):
        return f"tcp://127.0.0.1:1/{entity}"

    def send(self, *args, **kwargs):
        pass


class _RecordingEngine(LocalTransferEngine):
    """Records every register/unregister pointer and supports scripted failures."""

    def __init__(self, hostname, **kwargs):
        super().__init__(hostname)
        self._engine = None
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []
        self.fail_register_on: set[int] = set()
        self.fail_unregister_on: set[int] = set()
        self._reg_calls = 0

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self._reg_calls += 1
        if self._reg_calls in self.fail_register_on:
            return -7
        self.registered.append((ptr, nbytes))
        return 0

    def unregister_memory(self, ptr: int) -> int:
        if ptr in self.fail_unregister_on:
            return -1
        self.unregistered.append(ptr)
        return 0


class _FlatCopyReader:
    """Emulates TrackedMooncakeReader: flat-copies into local_ptr, then H2D."""

    def __init__(self, engine, device, device_filter=""):
        self._copy_stream = (
            torch.cuda.Stream(device=device)
            if torch.device(device).type == "cuda"
            else None
        )
        self.device_filter = device_filter
        self.submitted: list[list] = []
        self.fail_with: Exception | None = None
        self.status_by_batch: dict[int, int] = {}

    def submit(self, reads):
        self.submitted.append(list(reads))
        fut: Future = Future()
        if self.fail_with is not None:
            fut.set_exception(self.fail_with)
            return fut
        for r in reads:
            ctypes.memmove(r.local_ptr, r.remote_ptr, r.nbytes)
            if r.copy_dst is not None:
                if self._copy_stream is not None:
                    with torch.cuda.stream(self._copy_stream):
                        r.copy_dst.copy_(r.host_buf, non_blocking=True)
                else:
                    r.copy_dst.copy_(r.host_buf)
        if self._copy_stream is not None and any(r.copy_dst is not None for r in reads):
            self._copy_stream.synchronize()
        fut.set_result(None)
        return fut

    def transfer_status(self, batch_id: int) -> int:
        return self.status_by_batch.get(batch_id, _TRANSFER_COMPLETED)


def _bookkeeping(kind: str):
    if kind == "python":
        return PythonTensorBookkeeping()
    try:
        from mstar.communication.tensor_store import RustTensorBookkeeping
        return RustTensorBookkeeping()
    except ImportError:
        pytest.skip("RustTensorBookkeeping requires mstar_rust")


@pytest.fixture(autouse=True)
def _stub_mooncake(monkeypatch):
    monkeypatch.setattr(tensors, "MooncakeTransferEngine", _RecordingEngine)
    monkeypatch.setattr(tensors, "TrackedMooncakeReader", _FlatCopyReader)


def _manager(
    entity: str, *, protocol: CommProtocol = CommProtocol.TCP,
    device: str = "cpu", bookkeeping: str = "python",
) -> MooncakeCommunicationManager:
    mgr = MooncakeCommunicationManager(
        entity, "host", device, _NullCommunicator(), protocol=protocol,
    )
    mgr.tensor_store = TensorStore(_bookkeeping(bookkeeping))
    return mgr


# ── TensorStore.set_addresses ──────────────────────────────────────────────


@pytest.mark.parametrize("bookkeeping", ["python", "rust"])
def test_tensor_store_set_addresses_updates_descriptors(bookkeeping):
    mgr = _manager("worker_0", bookkeeping=bookkeeping)
    t = torch.arange(8, dtype=torch.float32)
    [info] = mgr.store_and_return_tensor_info(1, {"x": [t]})["x"]
    assert mgr.tensor_store.get_info(info.uuid).address == t.data_ptr()

    new_addr = t.data_ptr() + 4096
    mgr.tensor_store.set_addresses([info.uuid, 999999], [new_addr, 1234])
    assert mgr.tensor_store.get_info(info.uuid).address == new_addr

    with pytest.raises(ValueError):
        mgr.tensor_store.set_addresses([info.uuid], [])


# ── _flat_byte_view ────────────────────────────────────────────────────────


def test_flat_byte_view_preserves_storage_bytes():
    contig = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    trans = contig.t()
    scalar = torch.tensor(42, dtype=torch.int64)
    for t in (contig, trans, scalar):
        v = _flat_byte_view(t)
        assert v.dtype == torch.uint8 and v.ndim == 1
        assert v.numel() == t.nbytes
        assert v.data_ptr() == t.data_ptr()


# ── CPU-only TCP stays zero-copy ───────────────────────────────────────────


def test_tcp_cpu_tensors_stay_zero_copy():
    prod = _manager("api_server", protocol=CommProtocol.TCP, device="cpu")
    t = torch.arange(16, dtype=torch.float32)
    [info] = prod.store_and_return_tensor_info(1, {"x": [t]})["x"]
    prod.register_for_send(1, [info])

    assert prod._tcp_send_buffers == {}
    assert info.address == t.data_ptr()
    assert prod.transfer_engine.registered == [(t.data_ptr(), t.nbytes)]


# ── Simulated CUDA host-staging lifecycle (runs on CPU CI too) ─────────────


@pytest.mark.parametrize("bookkeeping", ["python", "rust"])
def test_tcp_send_staging_registers_host_ptr_and_unregisters_on_cleanup(
    monkeypatch, bookkeeping,
):
    prod = _manager("worker_0", protocol=CommProtocol.TCP, device="cpu", bookkeeping=bookkeeping)
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    [info] = prod.store_and_return_tensor_info(1, {"vision_embeds": [t]})["vision_embeds"]

    host_copy = torch.empty(t.nbytes, dtype=torch.uint8)
    host_copy.copy_(_flat_byte_view(t))
    monkeypatch.setattr(prod, "_stage_cuda_for_tcp_send", lambda uuids: {info.uuid: host_copy})

    # register_for_send_uuids (worker hot path)
    failed = prod.register_for_send_uuids(ParallelList([1], [[info.uuid]]), skip_cuda_sync=True)
    assert failed is None
    assert info.uuid in prod._tcp_send_buffers
    assert prod.transfer_engine.registered == [(host_copy.data_ptr(), host_copy.nbytes)]
    assert prod.tensor_store.get_info(info.uuid).address == host_copy.data_ptr()
    # Local consumer still sees the original tensor in TensorStore.
    assert prod.get_tensor(info.uuid) is t

    prod.cleanup_request(1, force=True)
    assert prod._tcp_send_buffers == {}
    assert prod.transfer_engine.unregistered == [host_copy.data_ptr()]
    assert prod._unreleasable_buffers == []


def test_tcp_send_registration_failure_does_not_leak_staged_buffer(monkeypatch):
    prod = _manager("worker_0", protocol=CommProtocol.TCP, device="cpu")
    t = torch.arange(16, dtype=torch.float32)
    [info] = prod.store_and_return_tensor_info(1, {"x": [t]})["x"]
    host_copy = torch.empty(t.nbytes, dtype=torch.uint8)
    monkeypatch.setattr(prod, "_stage_cuda_for_tcp_send", lambda uuids: {info.uuid: host_copy})
    prod.transfer_engine.fail_register_on.add(1)

    with pytest.raises(MemoryRegistrationError):
        prod.register_for_send(1, [info])

    assert prod._tcp_send_buffers == {}
    assert info.address == t.data_ptr()
    assert prod.tensor_store.get_info(info.uuid).address == t.data_ptr()


def test_tcp_send_failed_unregister_keeps_host_buffer_not_device_tensor(monkeypatch):
    prod = _manager("worker_0", protocol=CommProtocol.TCP, device="cpu")
    t = torch.arange(16, dtype=torch.float32)
    [info] = prod.store_and_return_tensor_info(1, {"x": [t]})["x"]
    host_copy = torch.empty(t.nbytes, dtype=torch.uint8)
    monkeypatch.setattr(prod, "_stage_cuda_for_tcp_send", lambda uuids: {info.uuid: host_copy})
    prod.register_for_send(1, [info])
    prod.transfer_engine.fail_unregister_on.add(host_copy.data_ptr())

    prod.cleanup_request(1, force=True)
    assert prod._tcp_send_buffers == {}
    assert prod._unreleasable_buffers == [host_copy]


def test_tcp_recv_keep_buffers_tracks_host_buffer_on_failed_read():
    """On a failed TCP read, _keep_read_buffers must hold the pinned host
    receive buffer whose pointer matches AsyncReadError.running, and free it
    once Mooncake reports COMPLETED."""
    cons = _manager("worker_1", protocol=CommProtocol.TCP, device="cpu")
    dst = torch.zeros(16, dtype=torch.float32)
    rx_host = torch.empty(dst.nbytes, dtype=torch.uint8)
    [info] = cons.store_and_return_tensor_info(1, {"x": [dst]})["x"]
    cons._tcp_recv_buffers[info.uuid] = rx_host
    cons.tensor_store.set_metadata(info.uuid, mem_registered=True)

    fut: Future = Future()
    fut.set_exception(AsyncReadError("stalled", {rx_host.data_ptr(): 42}, set()))
    edge = GraphEdge(name="x", next_node="LLM", tensor_info=[info])
    cons.pending = [tensors.FutureAndPointers(
        future=fut, graph_edges=[edge], request_id=1, transfer_uuids=[info.uuid],
    )]
    cons._async_reader.status_by_batch[42] = 0  # still running

    assert cons.get_ready_tensors() == {}
    assert len(cons.take_failed_reads()) == 1
    assert cons._tcp_recv_buffers == {}
    assert len(cons._kept_reads) == 1
    assert cons._kept_reads[0].buffers == [rx_host]
    assert cons._kept_reads[0].running == {rx_host.data_ptr(): 42}

    cons._async_reader.status_by_batch[42] = _TRANSFER_COMPLETED
    cons.get_ready_tensors()
    assert cons._kept_reads == []
    assert cons.transfer_engine.unregistered == [rx_host.data_ptr()]


def test_rdma_never_enables_tcp_host_staging():
    mgr = _manager("worker_0", protocol=CommProtocol.RDMA, device="cpu")
    assert mgr._tcp_host_staging is False
    assert mgr._stage_cuda_for_tcp_send([1, 2, 3]) == {}


# ── Real CUDA staging roundtrip (contiguous, transposed, scalar) ───────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bookkeeping", ["python", "rust"])
def test_cuda_tcp_host_staging_roundtrip_and_cleanup(bookkeeping):
    prod = _manager("worker_0", protocol=CommProtocol.TCP, device="cuda:0", bookkeeping=bookkeeping)
    cons = _manager("worker_1", protocol=CommProtocol.TCP, device="cuda:0", bookkeeping=bookkeeping)

    t_contig = torch.arange(256, dtype=torch.bfloat16, device="cuda:0").reshape(16, 16)
    t_trans = torch.arange(48, dtype=torch.float32, device="cuda:0").reshape(8, 6).t()
    t_scalar = torch.tensor(99, dtype=torch.int64, device="cuda:0")
    assert not t_trans.is_contiguous()
    torch.cuda.synchronize()

    infos = prod.store_and_return_tensor_info(
        1, {"vision_embeds": [t_contig, t_trans, t_scalar]},
    )["vision_embeds"]
    gpu_ptrs = [t.data_ptr() for t in (t_contig, t_trans, t_scalar)]
    prod.register_for_send(1, infos)

    assert len(prod._tcp_send_buffers) == 3
    for info, gpu_ptr in zip(infos, gpu_ptrs, strict=True):
        staged = prod._tcp_send_buffers[info.uuid]
        assert staged.device.type == "cpu" and staged.is_pinned()
        assert info.address == staged.data_ptr() and info.address != gpu_ptr
        assert prod.tensor_store.get_info(info.uuid).address == staged.data_ptr()
        assert prod.get_tensor(info.uuid).is_cuda

    edge = GraphEdge(name="vision_embeds", next_node="LLM", tensor_info=infos)
    futs = cons.start_read_tensors(2, [edge])
    assert len(futs) == 1
    futs[0].result(timeout=5.0)
    # Every submitted read wrote into a pinned host buffer, not a GPU pointer.
    submitted = cons._async_reader.submitted[-1]
    assert len(submitted) == 3
    for ri in submitted:
        assert ri.host_buf is not None and ri.host_buf.is_pinned()
        assert ri.local_ptr == ri.host_buf.data_ptr()
        assert ri.copy_dst is not None and ri.copy_dst.is_cuda

    ready = cons.get_ready_tensors()
    assert cons._tcp_recv_buffers == {}
    got = [cons.get_tensor(i.uuid) for i in ready[2][0].tensor_info]
    assert all(g.is_cuda for g in got)
    assert torch.equal(got[0], t_contig)
    assert torch.equal(got[1], t_trans) and got[1].stride() == t_trans.stride()
    assert torch.equal(got[2], t_scalar)

    prod.cleanup_request(1, force=True)
    cons.cleanup_request(2, force=True)
    assert prod._tcp_send_buffers == {}
    assert cons._tcp_recv_buffers == {}
    assert prod._unreleasable_buffers == []
    assert cons._unreleasable_buffers == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_rdma_registers_device_pointer_directly():
    prod = _manager("worker_0", protocol=CommProtocol.RDMA, device="cuda:0")
    t = torch.arange(32, dtype=torch.bfloat16, device="cuda:0")
    torch.cuda.synchronize()
    [info] = prod.store_and_return_tensor_info(1, {"x": [t]})["x"]
    prod.register_for_send(1, [info])

    assert prod._tcp_send_buffers == {}
    assert info.address == t.data_ptr()
    assert prod.transfer_engine.registered == [(t.data_ptr(), t.nbytes)]
    prod.cleanup_request(1, force=True)
    assert prod.transfer_engine.unregistered == [t.data_ptr()]
