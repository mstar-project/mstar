"""Native TP collectives capture through a compiled forward."""

import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup, WorkerParallelGroups


def _capture_collectives(rank, rendezvous):
    parallel = WorkerParallelGroups(
        num_workers=3, global_rank=rank, any_parallelism=True,
        world_parallel_groups=[(0, 1)],
    )
    group = CommGroup(rank, rank, [0, 1]) if rank < 2 else CommGroup.trivial()
    parallel.add("llm", group)
    parallel.init_dist(init_method=rendezvous, device=torch.device("xpu", rank))
    try:
        if rank == 2:
            # A worker with no local TP still joins subgroup creation, then
            # waits while the two TP members exercise their mixed subgroup.
            dist.barrier()
            return
        assert group.device_group is group.cpu_group
        payloads = [None, None]
        dist.all_gather_object(payloads, {"rank": rank}, group=group.cpu_group)
        assert payloads == [{"rank": 0}, {"rank": 1}]
        cpu_values = torch.tensor([rank + 1.0])
        dist.all_reduce(cpu_values, group=group.cpu_group)
        torch.testing.assert_close(cpu_values, torch.tensor([3.0]))

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
            group.barrier()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                output, gathered = compiled(inputs)
            torch.xpu.synchronize()
            group.barrier()
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
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.xpu.is_available()
    or torch.xpu.device_count() < 3
    or not dist.is_xccl_available(),
    reason="requires three XPUs and XCCL",
)
def test_compiled_tp_collectives_capture_and_replay(tmp_path):
    workers = mp.spawn(
        _capture_collectives,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=3,
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
