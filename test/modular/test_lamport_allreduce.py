"""The Triton Lamport one-shot all-reduce (``mstar/distributed/lamport_allreduce.py``) on two GPUs:
exact against the fp32 sum, robust to -0.0 inputs, correct over many rounds (the two buffers
alternate) and under CUDA-graph replay (the device-side round counter)."""
import os
import socket

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int) -> None:
    import torch.distributed as dist

    from mstar.distributed.lamport_allreduce import LamportAllReduce

    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    group = dist.group.WORLD

    def reference(x: torch.Tensor) -> torch.Tensor:
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x, group=group)
        acc = torch.zeros_like(x, dtype=torch.float32)
        for p in parts:  # two bf16 values sum exactly in fp32, so this is the kernel's own result
            acc += p.float()
        return acc.to(x.dtype)

    try:
        for width in (1000, 3584, 7168):
            ws = LamportAllReduce(group.group_name, rank, world, 64, width, torch.bfloat16, dev)
            assert not ws.applies(torch.zeros(1, width, device=dev))  # fp32
            assert not ws.applies(torch.zeros(65, width, device=dev, dtype=torch.bfloat16))  # too many rows
            torch.manual_seed(100 * width + rank)
            for step in range(30):  # odd and even rounds, both buffers
                rows = (1, 5, 64)[step % 3]
                x = torch.randn(rows, width, device=dev, dtype=torch.bfloat16) * 4
                if step == 4:
                    x.fill_(-0.0)  # the sentinel bit pattern as data
                if step == 7:
                    x[:, ::3] = -0.0
                out = ws.all_reduce(x)
                ref = reference(x)
                torch.cuda.synchronize()
                assert torch.equal(out, ref), (width, step, rows, (out.float() - ref.float()).abs().max())
            assert not ws.timed_out()
            # CUDA graph: the round counter advances on the device, so replays stay correct
            xs = [torch.randn(3, width, device=dev, dtype=torch.bfloat16) for _ in range(5)]
            outs = [torch.empty_like(t) for t in xs]
            refs = [reference(t) for t in xs]
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                ws.all_reduce(xs[0], out=outs[0])
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for t, o in zip(xs, outs):
                    ws.all_reduce(t, out=o)
            for _ in range(4):
                for o in outs:
                    o.zero_()
                g.replay()
                torch.cuda.synchronize()
                for o, r in zip(outs, refs):
                    assert torch.equal(o, r)
            assert not ws.timed_out()
            # after the graph work the eager path still agrees (the counter is shared)
            x = torch.randn(2, width, device=dev, dtype=torch.bfloat16)
            assert torch.equal(ws.all_reduce(x), reference(x))
            # the all-gather mode on the same workspace: rank p's shard at columns [p * width, (p + 1) * width)
            for rows in (1, 6, 64):
                shard = torch.randn(rows, width, device=dev, dtype=torch.bfloat16)
                shard[:, 1::7] = -0.0
                parts = [torch.empty_like(shard) for _ in range(world)]
                dist.all_gather(parts, shard, group=group)
                gathered = ws.all_gather(shard)
                torch.cuda.synchronize()
                assert gathered.shape == (rows, world * width)
                assert torch.equal(gathered, torch.cat(parts, dim=1)), (width, rows)
            assert not ws.timed_out()
        # flashinfer's one-shot kernel behind the same channel contract (skipped without flashinfer)
        try:
            from mstar.distributed.lamport_allreduce import FlashInferAllReduce

            fch = FlashInferAllReduce(group, rank, world, 64, 3584, torch.bfloat16, dev)
        except ImportError:
            fch = None
        if fch is not None:
            for rows in (1, 7, 64):
                x = torch.randn(rows, 3584, device=dev, dtype=torch.bfloat16) * 4
                out = fch.all_reduce(x)
                ref = reference(x)
                torch.cuda.synchronize()
                assert torch.equal(out, ref), ("flashinfer", rows, (out.float() - ref.float()).abs().max())
            xs = [torch.randn(3, 3584, device=dev, dtype=torch.bfloat16) for _ in range(4)]
            outs = [torch.empty_like(t) for t in xs]
            refs = [reference(t) for t in xs]
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                fch.all_reduce(xs[0], out=outs[0])
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for t, o in zip(xs, outs):
                    fch.all_reduce(t, out=o)
            for _ in range(3):
                for o in outs:
                    o.zero_()
                g.replay()
                torch.cuda.synchronize()
                for o, r in zip(outs, refs):
                    assert torch.equal(o, r), "flashinfer graph replay"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_lamport_all_reduce_two_gpus():
    import time

    import torch.multiprocessing as mp

    ctx = mp.start_processes(_worker, args=(2, _free_port()), nprocs=2, join=False, start_method="spawn")
    deadline = time.monotonic() + 600
    # join() returns as soon as one process exits; a worker that raised surfaces here as an exception
    while not ctx.join(timeout=max(1.0, deadline - time.monotonic())):
        if time.monotonic() > deadline:
            for p in ctx.processes:
                p.terminate()
            pytest.fail("Lamport all-reduce workers did not finish in 600 s (a wedged poll?)")
