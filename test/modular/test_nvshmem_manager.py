"""NVSHMEMCommunicationManager eager-API tests (multi-GPU).

Launch:
    NVSHMEM_REMOTE_TRANSPORT=none \\
    uv run --extra cu128 torchrun --nproc_per_node=2 \\
        -m pytest test/modular/test_nvshmem_manager.py -v -s

Skips the whole module if NVSHMEM is unavailable. Fixtures own the
process-group lifecycle; each test gets a fresh manager.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from mminf.communication.communicator import BaseCommunicator
from mminf.communication.tensors import (
    _NVSHMEM_ACK_VAL,
    EdgeSpec,
    NVSHMEMCommunicationManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _NullCommunicator(BaseCommunicator):
    def send(self, entity_id, msg): pass
    def get_all_new_messages(self): return []


@pytest.fixture(scope="session", autouse=True)
def _dist_setup():
    """Initialize NCCL + NVSHMEM once per pytest session (per rank)."""
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
    """Fresh NVSHMEMCommunicationManager per test; teardown after."""
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

def test_warmup_sets_pad_states(manager, rank):
    """After warmup: producer's ack_pad=ACK_VAL ('slot free'); consumer's data_pad=0."""
    spec = EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=4096)
    manager.init_edges([spec])
    manager.warmup()

    info = manager._captured
    assert info is not None
    if rank == spec.producer_rank:
        assert info.ack_pad_slot(0).item() == _NVSHMEM_ACK_VAL
    if rank == spec.consumer_rank:
        assert info.data_pad_slot(0).item() == 0


def test_eager_round_trip_byte_exact(manager, rank, device):
    """Producer's tensor reaches consumer byte-identical (no graph capture)."""
    spec = EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=2048)
    manager.init_edges([spec])
    manager.warmup()

    expected = torch.arange(512, dtype=torch.int32, device=device) + 111
    actual = torch.zeros_like(expected)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        if rank == spec.producer_rank:
            manager.record_send(stream, 0, expected)
        elif rank == spec.consumer_rank:
            manager.record_recv(stream, 0, actual)
    stream.synchronize()
    dist.barrier()

    if rank == spec.consumer_rank:
        torch.testing.assert_close(actual, expected)


def test_oversized_send_raises_before_device_op(manager, rank, device):
    """record_send with bytes > slot capacity raises ValueError; no NVSHMEM put."""
    spec = EdgeSpec(edge_id=5, producer_rank=0, consumer_rank=1, max_bytes=256)
    manager.init_edges([spec])
    manager.warmup()
    if rank == spec.producer_rank:
        oversized = torch.zeros(1024, dtype=torch.int32, device=device)  # 4 KiB > 256 B
        stream = torch.cuda.Stream(device=device)
        with pytest.raises(ValueError):
            manager.record_send(stream, 5, oversized)
    dist.barrier()


def test_unknown_edge_id_raises(manager, rank, device):
    """record_send/record_recv with an unregistered edge_id raises KeyError."""
    manager.init_edges([EdgeSpec(edge_id=3, producer_rank=0, consumer_rank=1, max_bytes=128)])
    manager.warmup()
    stream = torch.cuda.Stream(device=device)
    x = torch.zeros(16, dtype=torch.int32, device=device)
    with pytest.raises(KeyError):
        if rank == 0:
            manager.record_send(stream, 999, x)
        else:
            manager.record_recv(stream, 999, x)
    dist.barrier()


def test_warmup_is_idempotent(manager, rank):
    """warmup() twice is no-op; init_edges with same spec is no-op; differing spec raises."""
    spec = EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=64)
    manager.init_edges([spec])
    manager.warmup()
    manager.warmup()  # no-op
    assert manager._captured_warmed_up
    manager.init_edges([spec])  # same spec → no-op
    with pytest.raises(RuntimeError):
        manager.init_edges([
            EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=65)
        ])


def test_bidirectional_swap(manager, rank, device):
    """Both ranks act as producer AND consumer via two opposite edges in one test."""
    n = 512
    specs = [
        EdgeSpec(edge_id=0, producer_rank=0, consumer_rank=1, max_bytes=n * 4),
        EdgeSpec(edge_id=1, producer_rank=1, consumer_rank=0, max_bytes=n * 4),
    ]
    manager.init_edges(specs)
    manager.warmup()

    sent = torch.arange(n, dtype=torch.int32, device=device) + (rank * 10000)
    received = torch.zeros(n, dtype=torch.int32, device=device)
    out_edge = 0 if rank == 0 else 1
    in_edge = 1 if rank == 0 else 0
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        manager.record_send(stream, out_edge, sent)
        manager.record_recv(stream, in_edge, received)
    stream.synchronize()
    dist.barrier()

    expected = torch.arange(n, dtype=torch.int32, device=device) + ((1 - rank) * 10000)
    torch.testing.assert_close(received, expected)
