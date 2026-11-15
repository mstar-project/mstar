
import socket
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, ".")

from mstar.distributed.communication import CommGroup, WorkerParallelGroups


def _exercise_parallel_groups(rank, rendezvous):
    parallel = WorkerParallelGroups(
        num_workers=3, global_rank=rank, any_parallelism=True,
        world_parallel_groups=[(0, 1)],
    )
    group = CommGroup(rank, rank, [0, 1]) if rank < 2 else CommGroup.trivial()
    parallel.add("llm", group, "tp")
    parallel.init_dist(init_method=rendezvous, device=torch.device("xpu", rank))
    try:
        if rank == 2:
            dist.barrier()
            return
        assert group.device_group is group.cpu_group
        payloads = [None, None]
        dist.all_gather_object(payloads, {"rank": rank}, group=group.cpu_group)
        assert payloads == [{"rank": 0}, {"rank": 1}]
        cpu_values = torch.tensor([rank + 1.0])
        dist.all_reduce(cpu_values, group=group.cpu_group)
        torch.testing.assert_close(cpu_values, torch.tensor([3.0]))

        for tokens in (4, 2048):
            inputs = torch.full(
                (tokens, 3584), float(rank + 1), dtype=torch.bfloat16, device="xpu",
            )
            for step in range(3):
                inputs.fill_(rank + step + 1)
                output = group.all_reduce(inputs.clone())
                gathered = group.all_gather(inputs, dim=-1)
                torch.xpu.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(inputs, 2 * step + 3), atol=0, rtol=0,
                )
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
def test_mixed_parallel_groups_with_nonmember():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    workers = mp.spawn(
        _exercise_parallel_groups,
        args=(f"tcp://127.0.0.1:{port}",),
        nprocs=3,
        join=False,
    )
    try:
        deadline = time.monotonic() + 120
        while not workers.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("XPU parallel group workers timed out")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
