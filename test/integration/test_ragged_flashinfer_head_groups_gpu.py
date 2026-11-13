"""FlashInfer's eager ragged self-attention wrapper planned for a head group as well (``group_heads``), which a
pipelined Ulysses exchange needs to attend one head group at a time: a group's heads through the group plan are
the same heads of the full plan bit for bit, over a layout with a padding segment (MiniMax-H3's), and on 2 GPUs the
pipeline over this wrapper is the NCCL exchange over it, bit for bit."""
import functools
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper
from mstar.engine.resources.attn.workspace import WorkspacePool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEADS, HEAD_DIM = 16, 128


def _wrapper(groups: int, device="cuda"):
    pool = WorkspacePool(torch.device(device), 128)
    return FlashInferRaggedPrefillWrapper(
        pool.get("attn", 0), HEADS, HEADS, HEAD_DIM, device=torch.device(device), q_data_type=torch.bfloat16,
        group_heads=HEADS // groups, group_workspace=pool.get("attn#group", 0),
    )


@pytest.mark.parametrize("groups", [2, 4])
def test_a_head_group_through_its_plan_is_those_heads_of_the_full_plan(groups):
    wrapper = _wrapper(groups)
    assert wrapper.any_heads
    torch.manual_seed(0)
    for lens in ((1931, 1), (700, 700, 3), (2048,)):
        n = sum(lens)
        wrapper.plan(cu_seqlens=torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32))
        q, k, v = (torch.randn(n, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        full = wrapper.run(q, k, v)
        hg = HEADS // groups
        for g in range(groups):
            part = slice(g * hg, (g + 1) * hg)
            got = wrapper.run(*(t[:, part].contiguous() for t in (q, k, v)))
            assert torch.equal(got, full[:, part]), (lens, g)


def test_another_head_count_is_refused():
    wrapper = _wrapper(2)
    wrapper.plan(cu_seqlens=torch.tensor([0, 64], dtype=torch.int32))
    x = torch.randn(64, 4, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="q heads"):
        wrapper.run(x, x, x)


def test_group_heads_must_divide_the_heads():
    pool = WorkspacePool(torch.device("cuda"), 128)
    with pytest.raises(ValueError, match="must divide"):
        FlashInferRaggedPrefillWrapper(pool.get("a", 0), HEADS, HEADS, HEAD_DIM, group_heads=5,
                                       group_workspace=pool.get("b", 0))


def test_without_group_heads_it_takes_its_own_head_count_only():
    pool = WorkspacePool(torch.device("cuda"), 128)
    wrapper = FlashInferRaggedPrefillWrapper(pool.get("a", 0), HEADS, HEADS, HEAD_DIM)
    assert not wrapper.any_heads


def _entry(rank, world, port, out, groups):
    from mstar.distributed.communication import CommGroup
    from mstar.distributed.ulysses_pipeline import UlyssesPipeline
    from mstar.engine.resources.sequence_shard import SeqRange, ulysses

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank))
    try:
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        group.cpu_group = dist.new_group(backend="gloo")
        pipe = UlyssesPipeline(group, groups)
        pool = WorkspacePool(torch.device("cuda", rank), 128)
        local = HEADS * 2 // world  # 2 x HEADS heads in all, this rank's share after the exchange
        wrapper = FlashInferRaggedPrefillWrapper(
            pool.get("attn", 0), local, local, HEAD_DIM, device=torch.device("cuda", rank),
            q_data_type=torch.bfloat16, group_heads=local // groups, group_workspace=pool.get("attn#g", 0))
        results = []
        for tokens, seed in ((600, 0), (600, 1), (1000, 2)):
            total = tokens * world
            # one real span and a padding row of its own, as MiniMax-H3's padded bucket
            wrapper.plan(cu_seqlens=torch.tensor([0, total - 1, total], dtype=torch.int32))
            r = SeqRange(group, (tokens,) * world, None)
            g = torch.Generator(device="cuda").manual_seed(seed * world + rank)
            qkv = torch.randn(tokens, 3 * 2 * HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g)
            q, k, v = qkv.split(2 * HEADS, dim=1)
            want = ulysses(wrapper.run, q, k, v, r=r)
            nccl = functools.partial(ulysses, wrapper.run, q, k, v, r=r)
            got = pipe(q, k, v, list(r.counts), None, wrapper.run, nccl)
            torch.cuda.synchronize()
            results.append(got is not None and torch.equal(got, want))
        results.append(len(pipe._verdict) == 2 and all(pipe._verdict.values()))
        torch.save(results, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs with peer access")
@pytest.mark.parametrize("groups", [2, 4])
def test_the_pipelined_exchange_over_flashinfer_is_the_nccl_exchange_bit_for_bit(tmp_path, groups):
    out = str(tmp_path / "r")
    mp.spawn(_entry, args=(2, 29700 + groups, out, groups), nprocs=2)
    for rank in range(2):
        *calls, verified = torch.load(f"{out}.{rank}")
        assert verified, f"rank {rank}: a shape fell back to the NCCL exchange"
        assert all(calls), (rank, calls)
