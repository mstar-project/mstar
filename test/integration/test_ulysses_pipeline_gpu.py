"""The copy-engine pipelined Ulysses exchange on 2 and 4 GPUs (NCCL + gloo, one process a GPU): its output against
the NCCL exchange's (``sequence_shard.ulysses``), bit for bit, over repeated calls (the slots and flags are reused),
a second, larger shape (new buffers), packed tokens and a two-row label (a CFG batch), with q/k/v as strided token
views of one tensor, as a fused QKV projection hands them over; and unequal shares left to the NCCL exchange."""
import functools
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.distributed.ulysses_pipeline import UlyssesPipeline
from mstar.engine.resources.sequence_shard import SeqRange, pipelined_ulysses, ulysses

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs with peer access")

HEADS, HEAD_DIM = 24, 128


def _attend(rows):
    def run(q, k, v):
        """[rows * S, H, D] -> [rows * S, H, D]; every head and row on its own, as the ragged kernel does."""
        q, k, v = (t.unflatten(0, (rows, -1)).transpose(1, 2) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(0, 1)
    return run


def _entry(rank, world, port, out, groups):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    try:
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        group.cpu_group = dist.new_group(backend="gloo")
        pipe = UlyssesPipeline(group, groups)
        results = []
        for rows, tokens, seed in ((2, 320, 0), (2, 320, 1), (None, 320, 2), (2, 576, 3), (2, 576, 4)):
            r = SeqRange(group, (tokens,) * world, rows)
            g = torch.Generator(device="cuda").manual_seed(seed * world + rank)
            qkv = torch.randn((rows or 1) * tokens, 3 * HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
                              generator=g)
            q, k, v = qkv.split(HEADS, dim=1)  # strided tokens, contiguous heads
            run = _attend(rows or 1)
            want = ulysses(run, q, k, v, r=r)
            got = pipe(q, k, v, list(r.counts), rows, run, functools.partial(ulysses, run, q, k, v, r=r))
            torch.cuda.synchronize()
            results.append((got is not None, got is not None and torch.equal(got, want)))
        # unequal shares: the NCCL exchange, through the resource's entry point
        counts = tuple(64 + (i == 0) for i in range(world))
        r = SeqRange(group, counts, 2)
        x = torch.randn(2 * counts[rank], HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        uneven = pipelined_ulysses(_attend(2), x, x, x, r=r, pipeline=pipe)
        results.append(torch.equal(uneven, ulysses(_attend(2), x, x, x, r=r)))
        results.append(len(pipe._verdict) == 3 and all(pipe._verdict.values()))  # three shapes, all kept
        torch.save(results, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("world", [2, 4])
def test_the_pipelined_exchange_is_the_nccl_exchange_bit_for_bit(tmp_path, world, groups):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    if HEADS % (world * groups):
        pytest.skip(f"{HEADS} heads don't split into {world} x {groups}")
    out = str(tmp_path / "r")
    mp.spawn(_entry, args=(world, 29600 + 8 * world + groups, out, groups), nprocs=world)
    for rank in range(world):
        *calls, uneven, verified = torch.load(f"{out}.{rank}")
        assert verified, f"rank {rank}: a shape fell back to the NCCL exchange"
        assert uneven, f"rank {rank}: unequal shares"
        assert all(served and same for served, same in calls), (rank, calls)
