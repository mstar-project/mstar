"""``CommGroup.allreduce_rmsnorm`` on two real ranks: the fused all-reduce +
norm kernel (rows up to the workspace's ``max_tokens``, eager and replayed from a
CUDA graph) and its NCCL fallback (past it) both give
``rms_norm(all_reduce(x))``. Needs two CUDA GPUs and FlashInfer's fused all-reduce."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mstar.distributed.communication import CommGroup

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 CUDA GPUs",
)

HIDDEN = 2048
MAX_TOKENS = 2048


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float, bias: float) -> torch.Tensor:
    """NCCL's sum, then the norm in fp32."""
    r = x.clone()
    dist.all_reduce(r)
    v = r.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * (weight.float() + bias)).to(x.dtype)


def _ranks(rank: int, world: int, store: str, out: dict) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store}", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    try:
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group = dist.group.WORLD
        group.cpu_group = dist.new_group(backend="gloo")
        group.initialized = True
        group.init_all_reduce()
        fused = group.init_allreduce_fusion(HIDDEN, torch.bfloat16, MAX_TOKENS)
        errors = {}
        for bias in (0.0, 1.0):  # unit weights, and Gemma-style zero-centred ones
            weight = torch.randn(HIDDEN, device="cuda", generator=torch.Generator("cuda").manual_seed(1)) * 0.1
            weight = weight + (1.0 - bias)
            for rows in (1, 7, MAX_TOKENS, MAX_TOKENS + 1):
                gen = torch.Generator("cuda").manual_seed(100 * rows + rank)
                x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen)
                want = _reference(x, weight, 1e-6, bias)
                got = group.allreduce_rmsnorm(x, weight, 1e-6, bias)
                if rows > MAX_TOKENS:
                    # NCCL's fallback, eager only: a captured NCCL call has hung
                    # this harness's teardown (see #952)
                    errors[(bias, rows)] = ((got.float() - want.float()).abs().max().item(), 0.0)
                    continue
                # captured: the replay reads the static input written after capture
                static = torch.zeros_like(x)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    group.allreduce_rmsnorm(static, weight, 1e-6, bias)  # warm-up
                    with torch.cuda.graph(graph, stream=stream):
                        captured = group.allreduce_rmsnorm(static, weight, 1e-6, bias)
                torch.cuda.current_stream().wait_stream(stream)
                static.copy_(x)
                graph.replay()
                torch.cuda.synchronize()
                errors[(bias, rows)] = (
                    (got.float() - want.float()).abs().max().item(),
                    (captured.float() - want.float()).abs().max().item(),
                )
        out[rank] = (fused, errors)
    finally:
        dist.destroy_process_group()


def _entry(rank, world, store, out):
    _ranks(rank, world, store, out)


def test_the_fused_norm_matches_nccl_then_norm(tmp_path):
    world = 2
    with mp.Manager() as manager:
        out = manager.dict()
        mp.spawn(_entry, args=(world, str(tmp_path / "store"), out), nprocs=world)
        results = dict(out)
    for rank, (fused, errors) in results.items():
        assert fused, f"rank {rank}: no fused workspace, so only the fallback ran"
        for (bias, rows), (eager, captured) in errors.items():
            # bf16 output: one or two ulps of a unit-RMS row
            assert eager <= 0.0625 and captured <= 0.0625, (rank, bias, rows, eager, captured)
