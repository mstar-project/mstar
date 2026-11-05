"""``CommGroup.allreduce_rmsnorm`` on two real ranks: the fused all-reduce +
norm kernel (rows up to the workspace's ``max_tokens``, eager and replayed
from a CUDA graph) and its NCCL fallback (past it) both give
``rms_norm(all_reduce(x))``. Needs two peer-connected CUDA GPUs and
FlashInfer's comm kernels; on such a box a workspace that fails to build fails
the test, since the server would quietly lose the fusion."""

from __future__ import annotations

import time
from datetime import timedelta

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
EPS = 1e-6
# bf16 output of a unit-RMS row: a couple of ulps at its largest values
TOLERANCE = 0.0625
# a hung collective fails the test instead of the session
DEADLINE_S = 600


def _reference(x: torch.Tensor, weight: torch.Tensor, bias: float) -> torch.Tensor:
    """NCCL's sum, then the norm in fp32."""
    r = x.clone()
    dist.all_reduce(r)
    v = r.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    return (v * (weight.float() + bias)).to(x.dtype)


def _inputs(rows: int, rank: int, seed: int) -> torch.Tensor:
    g = torch.Generator("cuda").manual_seed(1000 * seed + 10 * rows + rank)
    return torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)


def _error(got: torch.Tensor, want: torch.Tensor) -> float:
    return (got.float() - want.float()).abs().max().item()


def _entry(rank: int, world: int, store: str, out: dict) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{store}", rank=rank, world_size=world,
        timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank),
    )
    try:
        # as `init_dist` sets up a TP group
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group = dist.group.WORLD
        group.cpu_group = dist.new_group(backend="gloo")
        group.initialized = True
        group.init_all_reduce()
        fused = group.init_allreduce_fusion(HIDDEN, torch.bfloat16, MAX_TOKENS)
        errors = {}
        for bias in (0.0, 1.0):  # unit weights, and Gemma-style zero-centred ones
            gen = torch.Generator("cuda").manual_seed(1)
            # the checkpoint's dtype: the fused kernel reads the weight as the input's
            weight = (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.1 + (1.0 - bias)).to(torch.bfloat16)
            for rows in (1, 7, MAX_TOKENS, MAX_TOKENS + 1):
                x = _inputs(rows, rank, 0)
                errs = [_error(group.allreduce_rmsnorm(x, weight, EPS, bias), _reference(x, weight, bias))]
                if rows <= MAX_TOKENS:
                    # captured, replayed twice on new inputs: the kernel's
                    # buffers rotate between calls
                    static = torch.zeros_like(x)
                    graph = torch.cuda.CUDAGraph()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        group.allreduce_rmsnorm(static, weight, EPS, bias)  # warm-up
                        with torch.cuda.graph(graph, stream=stream):
                            captured = group.allreduce_rmsnorm(static, weight, EPS, bias)
                    torch.cuda.current_stream().wait_stream(stream)
                    for seed in (1, 2):
                        static.copy_(_inputs(rows, rank, seed))
                        graph.replay()
                        errs.append(_error(captured, _reference(static, weight, bias)))
                # past max_tokens NCCL's fallback runs eagerly only: a captured
                # NCCL call hung an earlier harness's teardown
                errors[(bias, rows)] = errs
        torch.cuda.synchronize()
        out[rank] = (fused, errors)
    finally:
        dist.destroy_process_group()


def test_the_fused_norm_matches_nccl_then_norm(tmp_path):
    pytest.importorskip("flashinfer.comm")
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("GPUs 0 and 1 have no peer access, which the fused all-reduce needs")
    world = 2
    with mp.Manager() as manager:
        out = manager.dict()
        ctx = mp.start_processes(
            _entry, args=(world, str(tmp_path / "store"), out), nprocs=world, join=False,
        )
        deadline = time.monotonic() + DEADLINE_S
        while not ctx.join(timeout=5):
            if time.monotonic() > deadline:
                for p in ctx.processes:
                    p.kill()
                pytest.fail(f"the two ranks did not finish within {DEADLINE_S} s")
        results = dict(out)
    assert sorted(results) == [0, 1]
    for rank, (fused, errors) in results.items():
        assert fused, f"rank {rank}: no fused workspace, so only the NCCL fallback ran"
        for (bias, rows), errs in errors.items():
            assert max(errs) <= TOLERANCE, (rank, bias, rows, errs)
