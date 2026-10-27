"""Ulysses sequence parallelism on CPU processes over gloo.

``CommGroup.all_to_all`` against the layout it promises, on even and uneven
splits, and of several tensors at once against one at a time; and
``ulysses_attention`` over sharded tokens against one rank attending the whole
sequence (with fewer key/value heads than query heads too), for the
``[S, H, D]`` and ``[B, S, H, D]`` layouts; and ``attention_over_sharded_keys``, a
replicated query over sharded keys merged by log-sum-exp, against one rank attending
all the keys.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.sequence_parallel import (
    attention_over_sharded_keys,
    sp_seq_split,
    ulysses_attention,
)

HEADS = 6


def _group(rank: int, world: int) -> CommGroup:
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group = dist.group.WORLD
    group.initialized = True
    return group


def _run(fn, world, store_path, *args):
    mp.spawn(_entry, args=(fn, world, str(store_path), args), nprocs=world)


def _entry(rank, fn, world, store_path, args):
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _layout(batched: bool) -> tuple[int, int]:
    """``(seq_dim, head_dim)``."""
    return (1, 2) if batched else (0, 1)


def _full(total: int, batched: bool, seed: int = 0, heads: int = HEADS) -> torch.Tensor:
    shape = (2, total, heads, 4) if batched else (total, heads, 4)
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
def test_all_to_all_exchanges(tmp_path, world, total, batched, negative):
    _run(_exchanges, world, tmp_path / "store", total, batched, negative)


def _sdpa(batched):
    def attend(q, k, v):
        if not batched:
            q, k, v = (t.unsqueeze(0) for t in (q, k, v))
        out = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)), enable_gqa=True).transpose(1, 2)
        return out if batched else out.squeeze(0)
    return attend


def _attention(rank, world, total, batched, kv_heads):
    group = _group(rank, world)
    seq_dim, _ = _layout(batched)
    sizes = sp_seq_split(total, world)
    q, k, v = (_full(total, batched, seed, heads) for seed, heads in enumerate((HEADS, kv_heads, kv_heads)))
    want = _sdpa(batched)(q, k, v)
    start = sum(sizes[:rank])
    q, k, v = (t.narrow(seq_dim, start, sizes[rank]) for t in (q, k, v))
    out = ulysses_attention(group, q, k, v, _sdpa(batched), sizes, *_layout(batched))
    torch.testing.assert_close(out, want.narrow(seq_dim, start, sizes[rank]))


@pytest.mark.parametrize("batched", [False, True], ids=["SHD", "BSHD"])
@pytest.mark.parametrize("world,total,kv_heads", [(2, 6, HEADS), (3, 11, HEADS), (2, 7, 2), (3, 9, 3)],
                         ids=["2-even", "3-uneven", "2-gqa", "3-gqa"])
def test_ulysses_attention_matches_one_rank(tmp_path, world, total, kv_heads, batched):
    _run(_attention, world, tmp_path / "store", total, batched, kv_heads)


def _several_at_once(rank, world, total, batched):
    group = _group(rank, world)
    seq_dim, head_dim = _layout(batched)
    sizes = sp_seq_split(total, world)
    start = sum(sizes[:rank])
    tensors = [_full(total, batched, seed, heads).narrow(seq_dim, start, sizes[rank])
               for seed, heads in enumerate((HEADS, world, 2 * world))]
    together = group.all_to_all(tensors, scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=sizes)
    for got, x in zip(together, tensors, strict=True):
        assert torch.equal(got, group.all_to_all(x, scatter_dim=head_dim, gather_dim=seq_dim, gather_sizes=sizes))


@pytest.mark.parametrize("batched", [False, True], ids=["SHD", "BSHD"])
@pytest.mark.parametrize("world,total", [(2, 6), (3, 11)], ids=["2-even", "3-uneven"])
def test_all_to_all_of_several_tensors_matches_one_at_a_time(tmp_path, world, total, batched):
    _run(_several_at_once, world, tmp_path / "store", total, batched)


def _sharded_keys(rank, world, total):
    group = _group(rank, world)
    sizes = sp_seq_split(total, world)
    q = _full(5, batched=True, seed=0)               # the same short query on every rank
    k, v = (_full(total, batched=True, seed=seed) for seed in (1, 2))
    want = _sdpa(batched=True)(q, k, v)
    start = sum(sizes[:rank])
    out = attention_over_sharded_keys(group, q, k.narrow(1, start, sizes[rank]), v.narrow(1, start, sizes[rank]))
    torch.testing.assert_close(out, want)


@pytest.mark.parametrize("world,total", [(1, 7), (2, 7), (3, 11)], ids=["1", "2-uneven", "3-uneven"])
def test_attention_over_sharded_keys_matches_one_rank(tmp_path, world, total):
    _run(_sharded_keys, world, tmp_path / "store", total)
