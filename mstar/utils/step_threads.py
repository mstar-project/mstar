"""The CPUs that the threads a step passes through share.

A step passes between the worker's main thread, its GPU thread and its plan
thread several times (submit, launch, done, commit, plan ready), and each
handoff blocks one thread and wakes another. Woken on another core, a thread
first waits for that core to leave a deep idle state and then runs on cold
caches; woken on the core the other thread just released, it starts at once.
So the three threads share one CPU, a distinct one per worker on the host. The
API server's result threads (the message thread, and the Rust frontend's
bridge) hand every step's tokens on the same way and share one CPU of their
own. Everything else (intra-op pools, transport threads, the data worker)
keeps the process mask; a pool or thread that a pinned thread starts later
would inherit its CPU, so those start with ``unpin_thread``.

``auto`` pins only under a process mask narrower than the host, a ``taskset``
or a cpuset: separate servers on one bare host would otherwise all pick the
same CPUs. A CPU number pins anywhere.
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

# the mask the pinned threads left, for the threads they start later
_process_cpus: frozenset[int] | None = None


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


def narrowed_mask(allowed: Collection[int], host_cpus: int | None = None) -> bool:
    """Whether the process mask leaves out some of the host's CPUs, i.e. the
    server runs under ``taskset`` or a cpuset."""
    if host_cpus is None:
        host_cpus = os.cpu_count() or len(allowed)
    return len(allowed) < host_cpus


def api_thread_cpu(
    setting: str | int, allowed: Collection[int],
    order: Callable[[int], tuple[int, int]] = smt_order,
    host_cpus: int | None = None,
) -> int | None:
    """The CPU the API server's result threads share: the last allowed CPU
    (first hardware threads before siblings), or None when the setting is
    ``off``, the mask is too small to spare one, or ``auto`` meets a mask that
    is the whole host. The workers leave it out of their table, so the API
    server can pick it without knowing their devices.
    """
    if setting == "off" or len(allowed) < 3:
        return None
    if setting == "auto" and not narrowed_mask(allowed, host_cpus):
        return None
    return max(allowed, key=lambda c: (-order(c)[0], c))


def choose_step_core(
    setting: str | int,
    slot: int,
    local_cpus: Sequence[Collection[int] | None],
    allowed: Collection[int],
    order: Callable[[int], object] | None = None,
    host_cpus: int | None = None,
) -> StepCore:
    """The step-thread CPU of local worker ``slot``.

    ``local_cpus`` has one entry per worker the conductor runs on this host, in
    its worker order: the CPUs local to that worker's device, or None.
    ``allowed`` is the process mask, the same in every worker. Each worker
    builds the same table and takes its own row, so no two share a CPU: worker
    by worker, the first allowed CPU local to its device that no earlier worker
    took, else the first allowed CPU none took. ``setting`` is ``auto``,
    ``off`` or a CPU number, from which the workers take allowed CPUs upward.
    ``auto`` pins only when the mask is narrower than the host (``host_cpus``,
    by default the online CPUs): the table covers one deployment, and servers
    started side by side on a bare host would all take the same CPUs.
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
    if not narrowed_mask(allowed, host_cpus):
        return StepCore(None, "skipped: auto pins only under a process mask narrower than the host "
                              "(taskset or a cpuset); a CPU number pins anywhere")
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
    global _process_cpus
    _process_cpus = frozenset(process_cpus)
    os.sched_setaffinity(0, process_cpus)
    torch.empty(torch.get_num_threads() * _INTRA_OP_GRAIN, dtype=torch.uint8).fill_(0)
    os.sched_setaffinity(0, {cpu})


def unpin_thread() -> None:
    """Put the calling thread back on the process mask: the initializer of a
    pool, or the first call of a thread, that a pinned thread starts, since a
    new thread copies its creator's mask. A no-op where nothing is pinned."""
    if PINNABLE and _process_cpus is not None:
        os.sched_setaffinity(0, _process_cpus)
