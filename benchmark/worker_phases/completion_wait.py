"""Measure completion-wait overhead, without weights or a serving workload.

Run ``python -m benchmark.worker_phases.completion_wait --backend cuda``.
``--backend host`` isolates the handoff for an event that completes between
query and synchronize; it is not a GPU or throughput measurement. Arms run
interleaved, in alternating order, with warmup for each duration and arm.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from concurrent.futures import wait
from pathlib import Path
from types import SimpleNamespace

import torch

from mstar.testing_utils.worker import make_worker
from mstar.utils.threads import ThreadPool


class _HostEvent:
    def query(self):
        return False

    def synchronize(self):
        pass


def _threaded_wait(executor, step, callback):
    """The always-threaded completion-event path before the short-wait fix."""
    with executor.timer.span("worker.postprocess.event_sync"):
        event = step.node_batch.completion_event
        if event.query():
            return
        if executor._wait_pool is None:
            executor._wait_pool = ThreadPool(
                max_workers=1, thread_name_prefix=f"mstar-wait-{executor.worker_id}",
                initializer=executor.pin_this_thread,
            )
        future = executor._wait_pool.submit(event.synchronize)
        try:
            while not wait((future,), timeout=0.01).done:
                callback()
        finally:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "host"), default="cuda")
    parser.add_argument("--durations-us", type=float, nargs="+", default=[50, 250, 2000, 20000])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pin-cpu", type=int, help="pin main/helper threads together, as step_thread_core does")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.rounds) < 1:
        parser.error("iterations, warmup and rounds must be positive")
    if any(d <= 0 for d in args.durations_us):
        parser.error("durations must be positive")
    cuda = args.backend == "cuda"
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; --backend host only measures host handoffs")
    executor = make_worker(device=torch.device("cuda")).executor
    if args.pin_cpu is not None:
        if not hasattr(os, "sched_getaffinity") or args.pin_cpu not in os.sched_getaffinity(0):
            parser.error("pin-cpu must be in this process's CPU affinity mask")
        executor._process_cpus = os.sched_getaffinity(0)
        executor._step_cpus = SimpleNamespace(cpus=(args.pin_cpu,))
        executor.pin_this_thread()
    event = torch.cuda.Event(enable_timing=True) if cuda else _HostEvent()
    start = torch.cuda.Event(enable_timing=True) if cuda else None
    step = SimpleNamespace(
        node_name="probe", graph_walk="step", node_batch=SimpleNamespace(completion_event=event),
    )
    callbacks = 0

    def service():
        nonlocal callbacks
        callbacks += 1

    arms = {
        "direct": lambda: executor.wait_for_step(step),
        "threaded": lambda: _threaded_wait(executor, step, service),
        "adaptive": lambda: executor.wait_for_step(step, while_waiting=service),
    }
    # Calibrate _sleep against CUDA events, not an advertised GPU clock.
    cycles_per_us = 0.0
    if cuda:
        calibration = []
        for _ in range(10):
            start.record()
            torch.cuda._sleep(1_000_000)
            event.record()
            event.synchronize()
            calibration.append(1_000_000 / (start.elapsed_time(event) * 1000))
        cycles_per_us = statistics.median(calibration[5:])
    metadata = {
        "backend": args.backend, "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if cuda else None,
        "iterations": args.iterations, "warmup": args.warmup, "rounds": args.rounds,
        "pin_cpu": args.pin_cpu,
    }
    print(json.dumps(metadata), flush=True)
    print("round duration_us arm wall_us cpu_us gpu_us wall_minus_gpu_us callbacks", flush=True)
    results = []
    try:
        for duration in args.durations_us if cuda else [0]:
            for round_index in range(args.rounds):
                order = list(arms) if round_index % 2 == 0 else list(reversed(arms))
                for arm in order:
                    samples = []
                    for i in range(args.warmup + args.iterations):
                        previous_callbacks = callbacks
                        cpu_start = time.process_time_ns()
                        wall_start = time.perf_counter_ns()
                        if cuda:
                            start.record()
                            torch.cuda._sleep(max(1, int(duration * cycles_per_us)))
                            event.record()
                        arms[arm]()
                        wall_us = (time.perf_counter_ns() - wall_start) / 1000
                        cpu_us = (time.process_time_ns() - cpu_start) / 1000
                        gpu_us = start.elapsed_time(event) * 1000 if cuda else 0.0
                        if i >= args.warmup:
                            samples.append({
                                "wall_us": wall_us, "cpu_us": cpu_us, "gpu_us": gpu_us,
                                "wall_minus_gpu_us": wall_us - gpu_us,
                                "callbacks": callbacks - previous_callbacks,
                            })
                    row = {"round": round_index, "duration_us": duration, "arm": arm}
                    row.update({key: statistics.mean(s[key] for s in samples) for key in samples[0]})
                    results.append({**row, "samples": samples})
                    print(
                        f"{round_index} {duration:g} {arm} {row['wall_us']:.3f} {row['cpu_us']:.3f} "
                        f"{row['gpu_us']:.3f} {row['wall_minus_gpu_us']:.3f} {row['callbacks']:.3f}",
                        flush=True,
                    )
    finally:
        if executor._wait_pool is not None:
            executor._wait_pool.shutdown()
    if args.out:
        args.out.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
