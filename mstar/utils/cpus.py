"""How many CPUs this process can keep busy at once.

``os.cpu_count()`` reports the host's CPUs. In a container the cgroup CPU quota
can be far smaller, and a thread pool sized to the host then runs more threads
than the quota lets run, so a fork-join op waits on whichever thread the
scheduler parked.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def cgroup_cpu_quota(root: Path = _CGROUP_ROOT, proc_cgroup: str = "/proc/self/cgroup") -> float | None:
    """The tightest cgroup v2 ``cpu.max`` quota over this process's cgroup and its
    ancestors, in CPUs; None when no level sets one (or cgroup v2 is absent)."""
    try:
        with open(proc_cgroup) as f:
            rel = next((line.split("::", 1)[1].strip() for line in f if line.startswith("0::")), "/")
    except OSError:
        rel = "/"
    quota = None
    level = root / rel.lstrip("/")
    while True:
        try:
            limit, period = (level / "cpu.max").read_text().split()[:2]
            if limit != "max":
                cpus = int(limit) / int(period)
                quota = cpus if quota is None else min(quota, cpus)
        except (OSError, ValueError):
            pass
        if level == root or root not in level.parents:
            return quota
        level = level.parent


def usable_cpus() -> int:
    """CPUs this process may run on (its affinity, else ``os.cpu_count()``), capped
    by the cgroup quota rounded down. At least 1."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, math.floor(quota))
    return max(1, cpus)


# The engine worker's main, GPU and plan threads each want a CPU of their own.
MIN_ENGINE_CPUS = 3


def preprocess_cpu_split(pool: int, cpus: Sequence[int] | None = None) -> tuple[list[int], list[int]] | None:
    """Split the CPUs this process may run on between a media-preprocessing pool
    of ``pool`` CPUs (the last ones) and the engine (the rest). None when the
    affinity cannot spare them and still leave the engine ``MIN_ENGINE_CPUS``."""
    if cpus is None:
        try:
            cpus = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            return None
    cpus = sorted(cpus)
    if pool < 1 or len(cpus) - pool < MIN_ENGINE_CPUS:
        return None
    return cpus[-pool:], cpus[:-pool]
