"""The CPU a worker's step threads share.

A step passes between the worker's main thread, its GPU thread and its plan
thread several times (submit, launch, done, commit, plan ready), and each
handoff blocks one thread and wakes another. Woken on another core, a thread
first waits for that core to leave a deep idle state and then runs on cold
caches; woken on the core the other thread just released, it starts at once.
So the three threads share one CPU, a distinct one per worker on the host.
Everything else in the process (intra-op pools, transport threads) keeps the
process mask.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Collection, Sequence
from typing import NamedTuple

import torch

from mstar.utils.cpu_list import format_cpu_list, parse_cpu_list

PINNABLE = hasattr(os, "sched_setaffinity")

# ATen splits an op into chunks of at least this many elements per thread
_INTRA_OP_GRAIN = 32768


class StepCore(NamedTuple):
    """The CPU for a worker's step threads (None: leave them on the process
    mask) and how it was chosen, for the log."""
    cpu: int | None
    source: str


def smt_order(cpu: int) -> tuple[int, int]:
    """Sort key placing each core's first hardware thread before its siblings."""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
            return sorted(parse_cpu_list(f.read())).index(cpu), cpu
    except (OSError, ValueError):
        return 0, cpu


def choose_step_core(
    setting: str | int,
    slot: int,
    local_cpus: Sequence[Collection[int] | None],
    allowed: Collection[int],
    order: Callable[[int], object] | None = None,
) -> StepCore:
    """The step-thread CPU of local worker ``slot``.

    ``local_cpus`` has one entry per worker the conductor runs on this host, in
    its worker order: the CPUs local to that worker's device, or None.
    ``allowed`` is the process mask, the same in every worker. Each worker
    builds the same table and takes its own row, so no two share a CPU: worker
    by worker, the first allowed CPU local to its device that no earlier worker
    took, else the first allowed CPU none took. ``setting`` is ``auto``,
    ``off`` or a CPU number, from which the workers take allowed CPUs upward.
    """
    n = len(local_cpus)
    if setting == "off":
        return StepCore(None, "off")
    if len(allowed) < n + 1:
        return StepCore(None, f"skipped: {len(allowed)} allowed CPU(s) for {n} worker(s) "
                              "would leave none for the rest of the process")
    if setting != "auto":
        first = int(setting)
        upward = sorted(c for c in allowed if c >= first)
        if first not in allowed or slot >= len(upward):
            return StepCore(None, f"skipped: step_thread_core {first} leaves no allowed CPU "
                                  f"for worker {slot} in {format_cpu_list(allowed)}")
        return StepCore(upward[slot], f"step_thread_core {first}")
    ordered = sorted(allowed, key=order)
    taken: set[int] = set()
    for j, local in enumerate(local_cpus):
        cpu = next((c for c in ordered if local and c in local and c not in taken), None)
        if cpu is not None:
            source = f"local to its device: {format_cpu_list(local)}"
        else:
            cpu = next(c for c in ordered if c not in taken)
            source = ("process affinity, no free allowed CPU local to its device"
                      if local else "process affinity, no local CPU list for its device")
        if j == slot:
            return StepCore(cpu, source)
        taken.add(cpu)
    raise ValueError(f"worker slot {slot} out of range for {n} local worker(s)")


def pin_step_thread(cpu: int, process_cpus: Collection[int]) -> None:
    """Confine the calling thread to ``cpu``.

    A thread starts with its creator's mask, and a thread's intra-op (OpenMP)
    pool is started by its first parallel op with the mask the thread has
    then; a pool confined to one CPU runs a parallel op many times slower. So
    go back to the process mask first (the main thread, already confined,
    starts the GPU and plan threads), start the pool there with one op as wide
    as the pool, and only then move.
    """
    os.sched_setaffinity(0, process_cpus)
    torch.empty(torch.get_num_threads() * _INTRA_OP_GRAIN, dtype=torch.uint8).fill_(0)
    os.sched_setaffinity(0, {cpu})
