"""The ``ragged_attention`` table's ``fa3`` (Dao's FlashAttention-3, ``flash_attn_3``): what it refuses, and on a
Hopper GPU with flash_attn_3 installed, its eager wrapper against SDPA, its dense call per segment against its
varlen call, a head group against those heads of the full call, and on 2 GPUs the pipelined Ulysses exchange over
it against the NCCL exchange, bit for bit."""

from __future__ import annotations

import functools
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from mstar.engine.resources.attn.ragged.kernels import RAGGED_ATTENTION

SHAPE = {"num_qo_heads": 8, "num_kv_heads": 8, "head_dim": 128}


def _why(device: torch.device, dtype=torch.bfloat16, **shape) -> str | None:
    return RAGGED_ATTENTION.unavailable(RAGGED_ATTENTION.target(dtype, {**SHAPE, **shape}, device)).get("fa3")


def test_fa3_is_refused_off_the_gpu():
    assert "NVIDIA GPU" in _why(torch.device("cpu"))


_GPU_REASON = "no CUDA" if not torch.cuda.is_available() else _why(torch.device("cuda", 0))
needs_fa3 = pytest.mark.skipif(_GPU_REASON is not None, reason=f"fa3 unavailable: {_GPU_REASON}")


def _wrapper(heads: int, device="cuda", **kw):
    from mstar.engine.resources.attn.ragged.dense_wrappers import Fa3RaggedPrefillWrapper

    return Fa3RaggedPrefillWrapper(num_qo_heads=heads, num_kv_heads=heads, head_dim=128, device=torch.device(device),
                                   q_data_type=torch.bfloat16, **kw)


def _qkv(n: int, heads: int, seed: int, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(n, heads, 128, dtype=torch.bfloat16, device=device, generator=gen) for _ in range(3)]


def _cu(lens) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)


@needs_fa3
def test_fa3_refuses_a_padded_head_dim_and_fp32():
    assert "padded" in _why(torch.device("cuda", 0), head_dim=96)
    assert "bf16" in _why(torch.device("cuda", 0), dtype=torch.float32)


@needs_fa3
@pytest.mark.parametrize("lens", [[384, 200, 40], [7, 1, 1300]], ids=["mixed", "ragged"])
def test_fa3_eager_matches_sdpa(lens):
    w = _wrapper(7)
    w.plan(_cu(lens))
    q, k, v = _qkv(sum(lens), 7, 0)
    got = w.run(q, k, v)
    cu = _cu(lens).tolist()
    for start, end in zip(cu[:-1], cu[1:], strict=True):
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0).float() for t in (q, k, v))
        want = F.scaled_dot_product_attention(qs, ks, vs).squeeze(0).transpose(0, 1)
        assert (got[start:end].float() - want).abs().max().item() < 3e-2


@needs_fa3
@pytest.mark.parametrize("lens", [[3000, 1], [1931, 159], [2048]], ids=["h3-pad-row", "pad-segment", "one"])
def test_fa3_dense_per_segment_is_its_varlen_call_and_a_head_group_is_those_heads(lens):
    """An eager plan of a few segments runs FA3's dense call per segment (faster on a long one), its varlen call's
    bits; and q / k / v of a head group give those heads of the full call (a Ulysses pipeline attends a group)."""
    heads = 8
    w = _wrapper(heads)
    w.plan(_cu(lens))
    assert w.any_heads and w._segments is not None
    q, k, v = _qkv(sum(lens), heads, 1)
    dense = w.run(q, k, v)
    for g in range(2):
        part = slice(g * heads // 2, (g + 1) * heads // 2)
        assert torch.equal(w.run(*(t[:, part].contiguous() for t in (q, k, v))), dense[:, part])
    w._segments = None  # the varlen call
    assert torch.equal(w.run(q, k, v), dense)


@needs_fa3
def test_fa3_graph_wrapper_zeroes_rows_past_its_plan():
    w = _wrapper(4, accelerator_graph=True, max_num_segments=2, max_total_tokens=512)
    w.plan(_cu([300, 100]))
    q, k, v = _qkv(512, 4, 2)
    out = w.run(q, k, v)
    assert out[400:].abs().sum() == 0
    eager = _wrapper(4)
    eager.plan(_cu([300, 100]))
    assert torch.equal(out[:400], eager.run(*(t[:400] for t in (q, k, v))))


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
        heads = 16
        wrapper = _wrapper(heads // world, device=f"cuda:{rank}")
        results = []
        for tokens, seed in ((600, 0), (1000, 1)):
            total = tokens * world
            # one real span and a padding row of its own, as MiniMax-H3's padded bucket
            wrapper.plan(cu_seqlens=torch.tensor([0, total - 1, total], dtype=torch.int32))
            r = SeqRange(group, (tokens,) * world, None)
            q, k, v = _qkv(tokens, 3 * heads, seed * world + rank).pop().split(heads, dim=1)
            want = ulysses(wrapper.run, q, k, v, r=r)
            nccl = functools.partial(ulysses, wrapper.run, q, k, v, r=r)
            got = pipe(q, k, v, list(r.counts), None, wrapper.run, nccl)
            torch.cuda.synchronize()
            results.append(got is not None and torch.equal(got, want))
        results.append(len(pipe._verdict) == 2 and all(pipe._verdict.values()))
        torch.save(results, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@needs_fa3
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs with peer access")
@pytest.mark.parametrize("groups", [2, 4])
def test_the_pipelined_exchange_over_fa3_is_the_nccl_exchange_bit_for_bit(tmp_path, groups):
    out = str(tmp_path / "r")
    mp.spawn(_entry, args=(2, 29740 + groups, out, groups), nprocs=2)
    for rank in range(2):
        *calls, verified = torch.load(f"{out}.{rank}")
        assert verified, f"rank {rank}: a shape fell back to the NCCL exchange"
        assert all(calls), (rank, calls)
