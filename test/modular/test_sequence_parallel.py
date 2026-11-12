"""Ulysses sequence parallelism on CPU processes over gloo.

``CommGroup.all_to_all`` and ``gather_sequence`` against the layout they promise, on
even and uneven splits and over negative dims, and of several tensors at once against
one at a time; and
``ulysses_attention`` over sharded tokens against one rank attending the whole
sequence (with fewer key/value heads than query heads too), for the
``[S, H, D]`` and ``[B, S, H, D]`` layouts; ``attention_over_sharded_keys``, a
replicated query over sharded keys merged by log-sum-exp, against one rank attending
all the keys; a compiled stack over packs of new lengths, without recompiling; and the
Wan2.2 DiT under Ulysses against the same DiT on one rank.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.attention import sdpa_attention
from mstar.model.components.distributed.sequence_parallel import (
    attention_over_sharded_keys,
    gather_sequence,
    scatter_sequence,
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
    assert torch.equal(gather_sequence(group, shard, sizes, dim=seq), full)


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
    out = attention_over_sharded_keys(
        group, q, k.narrow(1, start, sizes[rank]), v.narrow(1, start, sizes[rank]), sdpa_attention,
    )
    torch.testing.assert_close(out, want)


@pytest.mark.parametrize("world,total", [(1, 7), (2, 7), (3, 11)], ids=["1", "2-uneven", "3-uneven"])
def test_attention_over_sharded_keys_matches_one_rank(tmp_path, world, total):
    _run(_sharded_keys, world, tmp_path / "store", total)


def _compiled_packs(rank, world):
    """The compiled stack Cosmos3's batched denoise step runs, over packs whose length varies:
    the per-rank split comes from the dynamic length, so no new length, whatever its remainder
    by the group size, may recompile."""
    group = _group(rank, world)

    def stack(q, k, v):
        sizes = sp_seq_split(q.shape[0], world)
        q, k, v = (scatter_sequence(group, t, sizes) for t in (q, k, v))
        for _ in range(2):
            q = ulysses_attention(group, q, k, v, _sdpa(batched=False), sizes)
        return gather_sequence(group, q, sizes)

    # Serving compiles once, at one even pack (Cosmos3's `warmup_compiles`, two equal requests),
    # with the length dynamic; every later length, uneven split or not, reuses that compile. 30
    # splits evenly at SP2 and SP3 into shards that match no other dimension: a warm-up pack that
    # does match one is `warmup_compiles`'s to handle (test_denoise_compile.py).
    compiled = torch.compile(stack, backend="eager", dynamic=True)
    for i, total in enumerate((30, 7, 9, 22, 13, 8, 31, 10, 12)):
        q, k, v = (_full(total, batched=False, seed=seed) for seed in range(3))
        with torch.compiler.set_stance("fail_on_recompile" if i >= 1 else "default"):
            out = compiled(q, k, v)
        attend = _sdpa(batched=False)
        torch.testing.assert_close(out, attend(attend(q, k, v), k, v))


@pytest.mark.parametrize("world", [2, 3])
def test_compiled_sequence_parallel_stack_takes_new_pack_lengths_without_recompiling(tmp_path, world):
    _run(_compiled_packs, world, tmp_path / "store")


def _opened_on_sdpa(dit, config, group, tokens: int) -> None:
    """``dit`` with its handles bound and open for a step of one row on SDPA, as the engine would on a node set to it:
    the array's read runs the exchange."""
    from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
    from mstar.engine.state import ArrayStep
    from mstar.engine.state.adapters import node_handles, state_setup, state_steps
    from mstar.model.wan22.config import DIT_TOKENS, SELF_ATTN
    from mstar.model.wan22.wan22_model import Wan22Model

    model = object.__new__(Wan22Model)
    model.config = config
    setup, _ = state_setup({"resources": {SELF_ATTN: {"backend": "sdpa"}}}, model.declare_state())
    handles = node_handles(model, "dit", {}, sp_group=group, setup=setup)
    for module in dit.modules():
        if hasattr(module, "bind_state"):
            module.bind_state(handles)
    step = state_steps(SubmoduleStep(segments=[Segment(0, "main", tokens)], state={DIT_TOKENS: ArrayStep()}), handles)
    step.set_ctx(StepContext(request_ids=[0], graph_walk="video_gen", slot=0, capture=False))
    step.begin_calls()


def _wan_dit_forward(grid, sp_group=None) -> torch.Tensor:
    from mstar.model.wan22.components.dit import Wan22DiT
    from mstar.model.wan22.config import Wan22Config

    config = Wan22Config(
        num_attention_heads=6, attention_head_dim=12, num_layers=2, ffn_dim=32,
        in_channels=4, out_channels=4, text_dim=8, freq_dim=16, rope_max_seq_len=16,
    )
    torch.manual_seed(0)
    dit = Wan22DiT(config, sp_group=sp_group)
    for param in dit.parameters():
        torch.nn.init.normal_(param, std=0.2)
    gen = torch.Generator().manual_seed(1)
    frames, height, width = grid
    if sp_group is not None:
        p_t, p_h, p_w = config.patch_size
        _opened_on_sdpa(dit, config, sp_group, frames // p_t * (height // p_h) * (width // p_w))
    latents = torch.randn(1, 4, frames, height, width, generator=gen)
    timestep = torch.rand(1, frames, generator=gen) * 1000
    text = torch.randn(1, 5, 8, generator=gen)
    with torch.no_grad():
        return dit(latents, timestep, text)


def _wan_dit(rank, world, grid, out_path):
    out = _wan_dit_forward(grid, _group(rank, world))
    if rank == 0:
        torch.save(out, out_path)


@pytest.mark.parametrize("grid", [(2, 4, 4), (3, 2, 6), (3, 4, 6)], ids=["even", "uneven", "runs"])
def test_wan_dit_under_ulysses_matches_one_rank(tmp_path, grid):
    """Each SP rank keeps its slice of the tokens, their RoPE and their
    frames' timesteps, and the gathered output is the one-rank DiT's. Over two
    ranks, 8 tokens split on whole frames, 9 cut a frame (modulation per
    token), and 18 split 9/9 cut a frame of 6 (modulation per half
    frame)."""
    _run(_wan_dit, 2, tmp_path / "two", grid, str(tmp_path / "two.pt"))
    torch.testing.assert_close(torch.load(tmp_path / "two.pt"), _wan_dit_forward(grid))


def _wan_dit_reuse_forward(sp_group=None) -> torch.Tensor:
    """Step caching under SP: one step keeps this rank's residual, the next reuses it on new inputs."""
    from mstar.model.wan22.components.dit import Wan22DiT
    from mstar.model.wan22.config import Wan22Config

    config = Wan22Config(
        num_attention_heads=6, attention_head_dim=12, num_layers=2, ffn_dim=32,
        in_channels=4, out_channels=4, text_dim=8, freq_dim=16, rope_max_seq_len=16,
    )
    torch.manual_seed(0)
    dit = Wan22DiT(config, sp_group=sp_group)
    for param in dit.parameters():
        torch.nn.init.normal_(param, std=0.2)
    gen = torch.Generator().manual_seed(1)
    grid = (3, 4, 6)
    if sp_group is not None:
        _opened_on_sdpa(dit, config, sp_group, 3 * 2 * 3)
    first, second = (torch.randn(1, 4, *grid, generator=gen) for _ in range(2))
    timestep = torch.rand(1, 3, generator=gen) * 1000
    text = torch.randn(1, 5, 8, generator=gen)
    with torch.no_grad():
        _, residual = dit(first, timestep, text, return_residual=True)
        if sp_group is not None:
            assert residual.shape[1] == sp_seq_split(18, sp_group.world_size)[sp_group.rank]
        return dit(second, timestep * 0.9, text, cached_residual=residual)


def _wan_dit_reuse(rank, world, out_path):
    out = _wan_dit_reuse_forward(_group(rank, world))
    if rank == 0:
        torch.save(out, out_path)


def test_wan_dit_step_cache_under_ulysses_keeps_each_ranks_shard(tmp_path):
    """Each SP rank keeps the residual of its own token slice and reuses it there; the gathered reusing step is the
    one-rank one."""
    _run(_wan_dit_reuse, 2, tmp_path / "two", str(tmp_path / "two.pt"))
    torch.testing.assert_close(torch.load(tmp_path / "two.pt"), _wan_dit_reuse_forward())
