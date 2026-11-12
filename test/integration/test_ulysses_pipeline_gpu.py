"""The copy-engine pipelined Ulysses exchange on 2 and 4 GPUs (NCCL + gloo, one process a GPU): its output against
the serving exchange's, bit for bit, over repeated calls (the slots and flags are reused) and a second shape (new
buffers), with q/k/v as strided row views of one tensor, as a fused QKV projection hands them over."""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.sequence_parallel import head_grouped, ulysses_attention
from mstar.model.components.distributed.ulysses_pipeline import UlyssesPipeline

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs with peer access")

HEADS, HEAD_DIM = 56, 128


def _attend(q, k, v):
    """[S, H, D] -> [S, H, D]; every head on its own, as the ragged kernel does."""
    return F.scaled_dot_product_attention(*(t.transpose(0, 1)[None] for t in (q, k, v)))[0].transpose(0, 1)


def _entry(rank, world, port, out, group_streams):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    try:
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        group.cpu_group = dist.new_group(backend="gloo")
        groups = 7
        pipe = UlyssesPipeline(group, groups, group_streams=group_streams)
        results = []
        for rows, seed in ((640, 0), (640, 1), (640, 2), (1152, 3)):
            g = torch.Generator(device="cuda").manual_seed(seed * world + rank)
            qkv = torch.randn(rows, 3 * HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
            q, k, v = qkv.split(HEADS, dim=1)  # strided rows, contiguous heads
            sequential = lambda q=q, k=k, v=v, rows=rows: ulysses_attention(  # noqa: E731
                group, q, k, v, head_grouped(_attend, groups), [rows] * world)
            want = sequential()
            got = pipe(q, k, v, _attend, sequential)
            torch.cuda.synchronize()
            results.append((got is not None, got is not None and torch.equal(got, want)))
        results.append(len(pipe._verdict) == 2 and all(pipe._verdict.values()))  # two shapes, both kept
        torch.save(results, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("group_streams", [True, False], ids=["group-streams", "one-stream"])
@pytest.mark.parametrize("world", [2, 4])
def test_the_pipelined_exchange_is_the_serving_exchange_bit_for_bit(tmp_path, world, group_streams):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    out = str(tmp_path / "r")
    mp.spawn(_entry, args=(world, 29500 + 2 * world + group_streams, out, group_streams), nprocs=world)
    for rank in range(world):
        *calls, verified = torch.load(f"{out}.{rank}")
        assert verified, f"rank {rank}: a shape fell back to the serving exchange"
        assert all(served and same for served, same in calls), (rank, calls)
