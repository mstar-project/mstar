"""Profile Command A+ decode steps with a local checkpoint and real M* resources.

Reports per-step latency at several batch sizes, then a torch.profiler
breakdown (GPU busy time vs wall time, top ops) for batch size 1.

    torchrun --standalone --nproc-per-node=4 --module \
      test.command_a_plus.profile_decode --checkpoint /local/checkpoint --output /scratch/prof
"""

import argparse
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from mstar.distributed.communication import CommGroup
from test.command_a_plus.validate_checkpoint import PROMPTS, Runtime


def decode_timings(runtime, batch_size, steps, warmup):
    rids = [f"prof_{batch_size}_{i}" for i in range(batch_size)]
    prompt = runtime.model.process_prompt(PROMPTS[0], ["text"], ["text"])["text_inputs"][0].cuda()
    for rid in rids:
        runtime.start(rid)
    try:
        _, tokens, _ = runtime.step(rids, [prompt] * batch_size, 0)
        timings = []
        for step in range(1, warmup + steps + 1):
            rows = [tokens[i:i + 1] for i in range(batch_size)]
            _, tokens, info = runtime.step(rids, rows, step)
            if step > warmup:
                timings.append(info)
        return timings
    finally:
        for rid in rids:
            runtime.remove(rid)
        runtime.check_empty()


def profile_bs1(runtime, steps, output, rank):
    from torch.profiler import ProfilerActivity, profile

    rid = "prof_trace"
    prompt = runtime.model.process_prompt(PROMPTS[0], ["text"], ["text"])["text_inputs"][0].cuda()
    runtime.start(rid)
    try:
        _, tokens, _ = runtime.step([rid], [prompt], 0)
        for step in range(1, 6):
            _, tokens, _ = runtime.step([rid], [tokens], step)
        torch.cuda.synchronize()
        dist.barrier()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            started = time.perf_counter()
            for step in range(6, 6 + steps):
                _, tokens, _ = runtime.step([rid], [tokens], step)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - started) * 1000 / steps
    finally:
        runtime.remove(rid)
    if rank != 0:
        return
    events = prof.key_averages()
    gpu_us = sum(e.self_device_time_total for e in events) / steps
    kernels = sum(e.count for e in events if e.self_device_time_total > 0) / steps
    print(f"\n[profile bs=1] wall {wall_ms:.2f} ms/step; GPU busy {gpu_us / 1000:.2f} ms/step "
          f"({100 * gpu_us / 1000 / wall_ms:.0f}%); ~{kernels:.0f} GPU kernels/step", flush=True)
    print(events.table(sort_by="self_device_time_total", row_limit=25), flush=True)
    print(events.table(sort_by="self_cpu_time_total", row_limit=25), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output / "decode_bs1_rank0.json"))


@torch.inference_mode()
def main_worker(args):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    group = CommGroup(dist.get_rank(), dist.get_rank(), list(range(dist.get_world_size())))
    group.device_group = dist.group.WORLD
    runtime = None
    try:
        runtime = Runtime(args, group)
        for batch_size in args.batch_sizes:
            timings = decode_timings(runtime, batch_size, args.steps, args.warmup)
            if rank == 0:
                total = statistics.median(t["total_ms"] for t in timings)
                plan = statistics.median(t["planning_ms"] for t in timings)
                print(f"[decode bs={batch_size}] median step {total:.2f} ms "
                      f"(planning {plan:.2f} ms) -> {1000 / total:.1f} tok/s/request, "
                      f"{1000 * batch_size / total:.1f} tok/s total", flush=True)
        profile_bs1(runtime, args.profile_steps, args.output, rank)
    finally:
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=5)
    main_worker(parser.parse_args())


if __name__ == "__main__":
    main()
