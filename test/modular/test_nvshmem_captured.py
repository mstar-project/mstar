"""Captured-graph NVSHMEM transport tests (multi-GPU).

Launch:
    NVSHMEM_REMOTE_TRANSPORT=none CUDA_VISIBLE_DEVICES=4,5,6,7 \\
    uv run --extra cu128 torchrun --nproc_per_node=2 \\
        -m pytest test/modular/test_nvshmem_captured.py -v -s

Reuses the fixtures from test_nvshmem_manager.py (pytest auto-discovers
them within the same test/modular directory via conftest? — no: they're
local to each file. Re-defined here to keep the file self-contained.)
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from mminf.communication.communicator import BaseCommunicator
from mminf.communication.tensors import (
    EdgeSpec,
    NVSHMEMCommunicationManager,
)

# ---------------------------------------------------------------------------
# Fixtures (mirror of test_nvshmem_manager.py)
# ---------------------------------------------------------------------------

class _NullCommunicator(BaseCommunicator):
    def send(self, entity_id, msg): pass
    def get_all_new_messages(self): return []


@pytest.fixture(scope="session", autouse=True)
def _dist_setup():
    if "RANK" not in os.environ:
        pytest.skip("must be launched under torchrun")
    if not symm_mem.is_nvshmem_available():
        pytest.skip("NVSHMEM unavailable")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    symm_mem.set_backend("NVSHMEM")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture
def rank() -> int:
    return dist.get_rank()


@pytest.fixture
def device() -> torch.device:
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def manager(rank, device):
    world = dist.get_world_size()
    mgr = NVSHMEMCommunicationManager(
        my_entity_id=f"worker_{rank}",
        rank=rank,
        world_size=world,
        device=device,
        communicator=_NullCommunicator(),
        group=dist.group.WORLD,
        entity_id_to_rank={f"worker_{i}": i for i in range(world)},
    )
    yield mgr
    mgr.teardown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_captured_replay_byte_exact(manager, rank, device):
    """Capture record_recv / record_send inside torch.cuda.graph; replay 5×
    — every replay byte-exact (validates in-graph pad reset)."""
    n = 512
    spec = EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=n * 4)
    manager.init_edges([spec])
    manager.warmup()

    cap_stream = torch.cuda.Stream(device=device)
    src = torch.zeros(n, dtype=torch.int32, device=device)
    dst = torch.zeros(n, dtype=torch.int32, device=device)

    # Capture under a single stream context.
    g = torch.cuda.CUDAGraph()
    cap_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.graph(g, stream=cap_stream):
        if rank == spec.producer_rank:
            manager.record_send(cap_stream, 0, src)
        elif rank == spec.consumer_rank:
            manager.record_recv(cap_stream, 0, dst)

    for i in range(5):
        if rank == spec.producer_rank:
            src.copy_(torch.arange(n, dtype=torch.int32, device=device) + i * 1000)
        g.replay()
        torch.cuda.current_stream(device).synchronize()
        dist.barrier()
        if rank == spec.consumer_rank:
            expected = torch.arange(n, dtype=torch.int32, device=device) + i * 1000
            torch.testing.assert_close(dst, expected)


def test_captured_n56_x_8mib_round_trip(manager, rank, device):
    """N=56 × 8 MiB captured replay byte-exact (real MoT scale).

    Pins §10.1 of the design doc — symm-heap contention at MoT scale.
    Skipped if device memory is too low to safely allocate the heap.
    """
    n_edges = 56
    payload_bytes = 8 * 1024 * 1024  # 8 MiB
    n_int32 = payload_bytes // 4
    total_heap = n_edges * payload_bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 2 * total_heap:  # heap on every rank + safety margin
        pytest.skip(f"insufficient device memory ({free_bytes/1e9:.1f} GB free)")

    specs = [
        EdgeSpec(
            edge_id=e, producer_rank=0, consumer_rank=1, max_bytes=payload_bytes,
        )
        for e in range(n_edges)
    ]
    manager.init_edges(specs)
    manager.warmup()

    cap_stream = torch.cuda.Stream(device=device)
    bufs = [torch.zeros(n_int32, dtype=torch.int32, device=device) for _ in range(n_edges)]

    g = torch.cuda.CUDAGraph()
    cap_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.graph(g, stream=cap_stream):
        for e, buf in enumerate(bufs):
            if rank == 0:
                manager.record_send(cap_stream, e, buf)
            elif rank == 1:
                manager.record_recv(cap_stream, e, buf)

    if rank == 0:
        for e, buf in enumerate(bufs):
            buf.fill_(e + 7)
    g.replay()
    torch.cuda.current_stream(device).synchronize()
    dist.barrier()
    if rank == 1:
        for e, buf in enumerate(bufs):
            assert (buf == e + 7).all(), f"edge {e} mismatch: first={buf[0].item()}"


def test_record_send_before_warmup_raises(manager, rank, device):
    """record_send called before warmup() raises with a clear error."""
    spec = EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=64)
    manager.init_edges([spec])
    # Intentionally skip warmup().
    if rank == spec.producer_rank:
        x = torch.zeros(16, dtype=torch.int32, device=device)
        stream = torch.cuda.Stream(device=device)
        with pytest.raises(RuntimeError, match="warmup"):
            manager.record_send(stream, 0, x)
    dist.barrier()
