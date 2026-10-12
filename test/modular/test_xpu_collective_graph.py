"""Native TP collectives capture through a compiled forward."""

import sys
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup


def _capture_collectives(rank, rendezvous):
    torch.xpu.set_device(rank)
    dist.init_process_group(
        "xccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        device_id=torch.device("xpu", rank),
        timeout=timedelta(seconds=60),
    )
    try:
        group = CommGroup(rank, rank, [0, 1])
        group.device_group = dist.group.WORLD

        def forward(x):
            reduced = group.all_reduce(x.sin())
            return reduced.square(), group.all_gather(x, dim=-1)

        # Exercise decode and packed-prefill sizes and consecutive captures.
        graphs = []
        for tokens in (4, 2048):
            inputs = torch.full(
                (tokens, 3584), float(rank + 1), dtype=torch.bfloat16, device="xpu",
            )
            compiled = torch.compile(
                forward, mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False,
            )
            for _ in range(3):
                compiled(inputs)
            torch.xpu.synchronize()
            dist.barrier()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                output, gathered = compiled(inputs)
            torch.xpu.synchronize()
            dist.barrier()
            graphs.append((graph, inputs, output, gathered))
        for graph, inputs, output, gathered in graphs:
            for step in range(3):
                inputs.fill_(rank + step + 1)
                graph.replay()
                torch.xpu.synchronize()
                # Replay must reduce and gather the new inputs on both ranks.
                reduced = (
                    torch.full_like(inputs, step + 1).sin()
                    + torch.full_like(inputs, step + 2).sin()
                )
                torch.testing.assert_close(output, reduced.square())
                expected_gather = torch.cat(
                    [torch.full_like(inputs, step + 1), torch.full_like(inputs, step + 2)],
                    dim=-1,
                )
                torch.testing.assert_close(gathered, expected_gather, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.xpu.is_available()
    or torch.xpu.device_count() < 2
    or not dist.is_xccl_available(),
    reason="requires two XPUs and XCCL",
)
def test_compiled_tp_collectives_capture_and_replay(tmp_path):
    workers = mp.spawn(
        _capture_collectives,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 120
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("XPU collective graph workers timed out")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
