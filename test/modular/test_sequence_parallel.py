"""Ulysses sequence parallelism on CPU processes over gloo.

``CommGroup.all_to_all`` against the layout it promises, on even and uneven
splits, and ``ulysses_attention`` over sharded tokens against one rank attending
the whole sequence, for the ``[S, H, D]`` and ``[B, S, H, D]`` layouts.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.sequence_parallel import sp_seq_split, ulysses_attention

HEADS = 6


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _group(rank: int, world: int) -> CommGroup:
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group = dist.group.WORLD
    group.initialized = True
    return group


def _run(fn, world, *args):
    mp.spawn(_entry, args=(fn, world, _free_port(), args), nprocs=world)


def _entry(rank, fn, world, port, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _layout(batched: bool) -> tuple[int, int]:
    """``(seq_dim, head_dim)``."""
    return (1, 2) if batched else (0, 1)


def _full(total: int, batched: bool, seed: int = 0) -> torch.Tensor:
    shape = (2, total, HEADS, 4) if batched else (total, HEADS, 4)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _exchanges(rank, world, total, batched, negative):
    group = _group(rank, world)
    seq_dim, head_dim = _layout(batched)
    sizes = sp_seq_split(total, world)
    full = _full(total, batched)
    shard = full.narrow(seq_dim, sum(sizes[:rank]), sizes[rank])
    per_rank = HEADS // world
    seq, head = (seq_dim - full.dim(), head_dim - full.dim()) if negative else (seq_dim, head_dim)
    heads = group.all_to_all(shard, scatter_dim=head, gather_dim=seq, gather_sizes=sizes)
    assert torch.equal(heads, full.narrow(head_dim, rank * per_rank, per_rank))
    back = group.all_to_all(heads, scatter_dim=seq, gather_dim=head, scatter_sizes=sizes)
    assert torch.equal(back, shard)


@pytest.mark.parametrize("negative", [False, True], ids=["dims", "negative-dims"])
@pytest.mark.parametrize("batched", [False, True], ids=["SHD", "BSHD"])
@pytest.mark.parametrize(
    "world,total", [(2, 6), (2, 7), (3, 9), (3, 11)], ids=["2-even", "2-uneven", "3-even", "3-uneven"],
)
def test_all_to_all_exchanges(world, total, batched, negative):
    _run(_exchanges, world, total, batched, negative)


def _sdpa(batched):
    def attend(q, k, v):
        if not batched:
            q, k, v = (t.unsqueeze(0) for t in (q, k, v))
        out = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
        return out if batched else out.squeeze(0)
    return attend


def _attention(rank, world, total, batched):
    group = _group(rank, world)
    seq_dim, _ = _layout(batched)
    sizes = sp_seq_split(total, world)
    q, k, v = (_full(total, batched, seed) for seed in range(3))
    want = _sdpa(batched)(q, k, v)
    start = sum(sizes[:rank])
    q, k, v = (t.narrow(seq_dim, start, sizes[rank]) for t in (q, k, v))
    out = ulysses_attention(group, q, k, v, _sdpa(batched), sizes, *_layout(batched))
    torch.testing.assert_close(out, want.narrow(seq_dim, start, sizes[rank]))


@pytest.mark.parametrize("batched", [False, True], ids=["SHD", "BSHD"])
@pytest.mark.parametrize("world,total", [(2, 6), (3, 11)], ids=["2-even", "3-uneven"])
def test_ulysses_attention_matches_one_rank(world, total, batched):
    _run(_attention, world, total, batched)
