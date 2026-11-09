"""Kimi K3.1's latent MoE reduction on N GPUs: the routed experts' partial (T, 3584) all-reduced
then RMS-normed (``tp_allreduce``'s all-reduce + the norm kernel, what the port ran) against the
norm reducing it in one kernel (``RMSNorm.forward(x, comm_group)``: FlashInfer's
allreduce_fusion), and the hidden-size (T, 7168) all-reduce for scale. Microseconds per call,
``--calls`` calls captured in one CUDA graph per rank, rank 0's timing; outputs compared.

    python benchmark/kimi_k3/ar_norm.py --world 2 --tokens 1,8,32,64,256 --tp-allreduce flashinfer
"""
from __future__ import annotations

import argparse
import json
import os
import socket

import torch
import torch.multiprocessing as mp


def _time(fn, calls: int, replays: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * calls)


def _kernels(fn) -> set[str]:
    """The communication kernels one call of ``fn`` launches (NCCL's, FlashInfer's all-reduce)."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    return {n[:60] for n in names if any(k in n.lower() for k in ("nccl", "allreduce", "lamport", "fusion"))}


def _bench_rows(group, norm, T: int, rank: int, args) -> dict:
    latent, hidden = norm.weight.numel(), 7168
    g = torch.Generator(device="cuda").manual_seed(100 + rank * 7 + T)
    x = torch.randn(T, latent, device="cuda", generator=g).bfloat16()
    h = torch.randn(T, hidden, device="cuda", generator=g).bfloat16()
    with torch.no_grad():
        apart = norm(group.all_reduce(x.clone()))
        together = norm(x, group)
        diff = (apart.float() - together.float()).abs()
        row = {"world": args.world, "T": T, "tp_allreduce": args.tp_allreduce,
               "max_diff": diff.max().item(), "differ": (diff > 0).float().mean().item()}
        row["ar_latent"] = _time(lambda: group.all_reduce(x.clone()), args.calls, args.replays)
        row["ar_latent+norm"] = _time(lambda: norm(group.all_reduce(x.clone())), args.calls, args.replays)
        row["fused_ar_norm"] = _time(lambda: norm(x, group), args.calls, args.replays)
        row["ar_hidden"] = _time(lambda: group.all_reduce(h.clone()), args.calls, args.replays)
        row["clone_latent"] = _time(x.clone, args.calls, args.replays)
        # the same inside torch.compile (a compiled region's all-reduce, as Kimi's MoE block runs it)
        ar_c = torch.compile(lambda: group.all_reduce(h.clone()), dynamic=True)
        norm_c = torch.compile(lambda: norm(x, group), dynamic=True)
        row["ar_hidden_compiled"] = _time(ar_c, args.calls, args.replays)
        row["fused_ar_norm_compiled"] = _time(norm_c, args.calls, args.replays)
        row["compiled_kernels"] = sorted(_kernels(ar_c) | _kernels(norm_c))
    if rank == 0:
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    return row


def _rank(rank: int, args, port: int) -> None:
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups
    from mstar.model.components.norm import RMSNorm

    world = args.world
    torch.cuda.set_device(rank)
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world)))
    groups = WorkerParallelGroups(num_workers=world, global_rank=rank, any_parallelism=True,
                                  world_parallel_groups=[tuple(range(world))], dist_timeout_s=600,
                                  tp_allreduce=args.tp_allreduce)
    groups.add("LLM", group)
    groups.init_dist(init_method=f"tcp://127.0.0.1:{port}", device=torch.device("cuda", rank))
    latent = 3584
    fused = group.init_allreduce_fusion(latent, torch.bfloat16, max_tokens=256)
    if rank == 0:
        print(f"# {torch.cuda.get_device_name()} x{world} tp_allreduce={args.tp_allreduce} fused workspace {fused}")
    norm = RMSNorm(latent, eps=1e-6).cuda().bfloat16()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(latent, generator=torch.Generator().manual_seed(1)))
    rows = [_bench_rows(group, norm, int(T), rank, args) for T in args.tokens.split(",")]
    if rank == 0 and args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    torch.distributed.barrier()
    os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=2)
    p.add_argument("--tokens", default="1,8,32,64,256")
    p.add_argument("--tp-allreduce", default="flashinfer")
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank, args=(args, port), nprocs=args.world)


if __name__ == "__main__":
    main()
