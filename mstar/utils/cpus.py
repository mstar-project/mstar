"""The CPUs this process may use, and the plan every process of a deployment reads.

``os.cpu_count()`` reports the host's CPUs. In a container the cgroup CPU quota
can be far smaller, and a thread pool sized to the host then runs more threads
than the quota lets run, so a fork-join op waits on whichever thread the
scheduler parked.

``CpuPlan`` holds the process mask, how many of its CPUs the quota lets run,
the CPUs the ``step_thread_core`` knob pins (each worker's step threads and the
API server's result threads, see ``mstar.utils.threads``), and the pool sizes,
taken from the CPUs left after those. The API server and every worker build it
with ``deployment_plan`` from the same inputs (the knob, the mask, the
deployment's devices), so they agree on it without talking.

Placement under ``auto`` is by the device's rank among the host's GPUs on its
NUMA node, as sysfs lists them (every GPU of the host, not only the ones this
process was given), so servers on different GPUs pick different CPUs without
knowing about each other. A worker takes the core at its device's rank from the
bottom of the node's allowed CPUs, the API server the core at its first
device's rank from the top.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PCI_DEVICES = Path("/sys/bus/pci/devices")
_CPU_TOPOLOGY = "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list"

PINNABLE = hasattr(os, "sched_setaffinity")


# ── CPU lists ───────────────────────────────────────────────────────────────

def parse_cpu_list(text: str) -> frozenset[int]:
    """The CPUs in a list such as ``0-3,8,10-11`` (as sysfs and ``taskset``
    write them); empty for an empty string."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return frozenset(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """``cpus`` as ranges, the inverse of ``parse_cpu_list``."""
    ordered = sorted(set(cpus))
    parts = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        parts.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(parts)


# ── how many ────────────────────────────────────────────────────────────────

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


def process_mask() -> frozenset[int]:
    """The CPUs this process may run on; every CPU where affinity is unknown."""
    try:
        return frozenset(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return frozenset(range(os.cpu_count() or 1))


def usable_cpus() -> int:
    """CPUs this process may run on (its affinity, else ``os.cpu_count()``), capped
    by the cgroup quota rounded down. At least 1."""
    cpus = len(process_mask())
    quota = cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, math.floor(quota))
    return max(1, cpus)


# ── the plan ────────────────────────────────────────────────────────────────

class StepCpus(NamedTuple):
    """The CPUs one worker's main, GPU and plan threads share (one core's
    hardware threads in the mask), and how they were chosen, for the log."""
    cpus: frozenset[int]
    why: str


@dataclass(frozen=True)
class CpuPlan:
    """See the module docstring. ``workers`` has one entry per worker of the
    deployment, in the conductor's worker order (empty, or all None, when
    nothing is pinned)."""
    mask: frozenset[int]
    usable: int
    workers: tuple[StepCpus | None, ...] = ()
    result: int | None = None
    why: str = "step_thread_core off"

    @property
    def pinned(self) -> frozenset[int]:
        cpus = set() if self.result is None else {self.result}
        for step in self.workers:
            if step is not None:
                cpus |= step.cpus
        return frozenset(cpus)

    @property
    def left(self) -> int:
        """The CPUs the quota lets run, less the pinned ones: what the pools
        are sized from."""
        return max(1, self.usable - len(self.pinned & self.mask))

    @property
    def preprocess_threads(self) -> int:
        """The API server's intra-op threads when neither the config nor the
        model sets them. A quarter of the CPUs left, at most 4: the process
        shares them with the engine workers, and its per-request ops are a few
        MB."""
        return max(1, min(4, self.left // 4))

    @property
    def png_threads(self) -> int:
        """Threads one PNG deflates on (``image_io.uint8_to_png``)."""
        return min(8, self.left)

    @property
    def encode_workers(self) -> int:
        """Outputs the data worker encodes at once. A PNG deflates on
        ``png_threads`` of its own, so the pool times that stays within the
        CPUs left."""
        return max(1, min(4, self.left // self.png_threads))

    @property
    def copy_workers(self) -> int:
        """Threads a large output read copies over (``host_copy.HostCopier``):
        one encode's share of the CPUs left. Reads run one at a time on the
        data worker's thread, each ahead of its own encode."""
        return max(1, self.left // self.encode_workers)


_plan: CpuPlan | None = None


def plan() -> CpuPlan:
    """This process's plan: the one ``install`` set, else nothing pinned."""
    global _plan
    if _plan is None:
        _plan = CpuPlan(process_mask(), usable_cpus())
    return _plan


def install(new: CpuPlan | None) -> None:
    """Make ``new`` this process's plan (None: back to nothing pinned)."""
    global _plan
    _plan = new


class CpuPlanError(ValueError):
    """A ``step_thread_core`` value the process mask can't honour."""


# ── where a GPU sits ────────────────────────────────────────────────────────

class HostGpu(NamedTuple):
    """One of the host's GPUs as sysfs lists it: its PCI address, NUMA node
    and the node's CPUs (None where sysfs doesn't say)."""
    address: str
    node: str | None
    local: frozenset[int] | None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cpu_list(text: str | None) -> frozenset[int] | None:
    try:
        return parse_cpu_list(text) or None if text else None
    except ValueError:
        return None


def host_gpus(like: str, root: Path = _PCI_DEVICES) -> list[HostGpu]:
    """The host's GPUs in PCI address order: the functions with the vendor
    and class of the device at ``like`` (``0000:40:00.0``). Every process on
    the host sees the same list, whatever GPUs it was given; empty without
    sysfs."""
    vendor, cls = _read(root / like / "vendor"), _read(root / like / "class")
    if vendor is None or cls is None:
        return []
    return [
        HostGpu(dev.name, _read(dev / "numa_node"), _cpu_list(_read(dev / "local_cpulist")))
        for dev in sorted(root.iterdir(), key=lambda d: d.name)  # domain:bus:device.function
        if _read(dev / "vendor") == vendor and _read(dev / "class") == cls
    ]


def smt_siblings(cpu: int) -> frozenset[int]:
    """The hardware threads of ``cpu``'s core, ``cpu`` included."""
    try:
        with open(_CPU_TOPOLOGY.format(cpu)) as f:
            return parse_cpu_list(f.read()) | {cpu}
    except (OSError, ValueError):
        return frozenset({cpu})


def _cores(cpus: Collection[int], siblings: Callable[[int], frozenset[int]]) -> list[tuple[int, ...]]:
    """``cpus`` grouped by physical core (each core's CPUs within ``cpus``, in
    order), cores ordered by their first CPU."""
    cores = {tuple(sorted(siblings(c) & set(cpus))) for c in cpus}
    return sorted(cores)


def _step_cpus(core: tuple[int, ...], why: str) -> StepCpus:
    """A worker's step threads on all of ``core``'s hardware threads in the
    mask. On one CPU the three queue behind each other (a Qwen3-TTS worker's
    main thread alone keeps about 0.6 of a CPU busy at c=1); on the core's two
    hardware threads there is room, and a woken thread still finds the core
    awake."""
    return StepCpus(frozenset(core), why)


# ── building it ─────────────────────────────────────────────────────────────

def deployment_plan(
    setting: str | int,
    addresses: Sequence[str | None],
    *,
    mask: Collection[int] | None = None,
    usable: int | None = None,
    pci_root: Path = _PCI_DEVICES,
    siblings: Callable[[int], frozenset[int]] = smt_siblings,
) -> CpuPlan:
    """The plan for a deployment whose workers' devices sit at PCI
    ``addresses``, in the conductor's worker order (None for a device sysfs
    doesn't place, such as a CPU worker; only ``auto`` reads them).

    ``off`` pins nothing. A CPU number is the first worker's CPU; the next
    workers take the next cores above it, and the API server the last core of
    the mask; a number outside the mask, or too near its top for every worker,
    raises ``CpuPlanError``. ``auto`` places by device rank (module docstring),
    and pins nothing, saying why in ``CpuPlan.why``, when the mask is too small.
    Either way at least one CPU of the mask stays unpinned.
    """
    mask = frozenset(process_mask() if mask is None else mask)
    usable = usable_cpus() if usable is None else usable
    if setting == "off":
        return CpuPlan(mask, usable, why="step_thread_core off")
    if not PINNABLE:
        return CpuPlan(mask, usable, why="not pinned: no sched_setaffinity here")
    cores = _cores(mask, siblings)
    if setting == "auto":
        placed = _auto(addresses, mask, cores, pci_root, siblings)
        if isinstance(placed, str):
            return CpuPlan(mask, usable, why=f"not pinned: {placed}")
        workers, result, why = placed
    else:
        workers, result, why = _explicit(int(setting), len(addresses), mask, cores)
    out = CpuPlan(mask, usable, tuple(workers), result, why)
    if len(out.pinned) >= len(mask):
        reason = (f"{len(addresses)} worker(s) and the API server would pin all {len(mask)} CPU(s) "
                  f"of the process mask {format_cpu_list(mask)}, leaving none for the rest")
        if setting == "auto":
            return CpuPlan(mask, usable, why=f"not pinned: {reason}")
        raise CpuPlanError(f"step_thread_core {setting}: {reason}")
    return out


def _explicit(first: int, n: int, mask: frozenset[int], cores: list[tuple[int, ...]]):
    if first not in mask:
        raise CpuPlanError(f"step_thread_core {first} is outside the process mask {format_cpu_list(mask)}")
    result_core = cores[-1]
    start = next(i for i, core in enumerate(cores) if first in core)
    above = [core for core in cores[start:] if core != result_core]
    if len(above) < n:
        raise CpuPlanError(
            f"step_thread_core {first} leaves {len(above)} core(s) at or above it for {n} worker(s) in the "
            f"process mask {format_cpu_list(mask)} (the last, {result_core[0]}, is the API server's)"
        )
    why = f"step_thread_core {first}"
    workers = [_step_cpus(core, why) for core in above[:n]]
    return workers, result_core[0], f"{why}; the API server on the last core of the mask"


def _auto(addresses, mask, cores, pci_root, siblings):
    """``(workers, result, why)``, or why nothing can be pinned."""
    if not cores or not addresses:
        return "no CPU or no worker to place"
    gpus = next((found for a in addresses if a and (found := host_gpus(a, pci_root))), [])
    # a GPU with allowed CPUs on its node picks among those by its rank on the
    # node; the cores those ranks take are left out for a GPU with none
    nodes: dict[str | None, list[HostGpu]] = {}
    for gpu in gpus:
        nodes.setdefault(gpu.node, []).append(gpu)
    local, claimed = {}, set()
    for members in nodes.values():
        pool = _cores(members[0].local & mask, siblings) if members[0].local else []
        for rank, gpu in enumerate(members if pool else []):
            local[gpu.address] = pool, rank, len(members)
            claimed |= {pool[rank % len(pool)], pool[(len(pool) - 1 - rank) % len(pool)]}
    spare = [core for core in cores if core not in claimed] or cores

    def home(address):
        """The cores a device's threads pick from, its rank among the GPUs
        that pick from them, how many those are, and why."""
        if address in local:
            pool, rank, count = local[address]
            return pool, rank, count, (
                f"GPU {rank + 1} of {count} on its NUMA node, over its allowed CPUs "
                f"{format_cpu_list(c for core in pool for c in core)}")
        rank = next((i for i, gpu in enumerate(gpus) if gpu.address == address), None)
        if rank is None:
            return cores, 0, 1, "no sysfs entry for its device, so over the process mask"
        return spare, rank, len(gpus), (
            f"no allowed CPU on its NUMA node, so GPU {rank + 1} of {len(gpus)} on the host, over the "
            "cores the node-local GPUs leave")

    homes = [home(a) for a in addresses]
    first, rank, _, why = homes[0]
    result_core = first[(len(first) - 1 - rank) % len(first)]
    taken = {result_core}
    workers = []
    for i, (pool, rank, gpus_here, why_i) in enumerate(homes):
        # workers of this deployment on one device (a split config) step by
        # the GPU count, so the next GPU's first worker keeps its core
        nth = sum(a == addresses[i] for a in addresses[:i])
        start = (rank + nth * gpus_here) % len(pool)
        core = next((c for c in pool[start:] + pool[:start] if c not in taken), None)
        if core is None:
            core = next((c for c in cores if c not in taken), None)
            why_i += "; none free there, so the first free core of the mask"
        if core is None:
            return f"{len(addresses)} worker(s) and the API server need more cores than the mask has"
        taken.add(core)
        workers.append(_step_cpus(core, f"auto: {why_i}"))
    return workers, result_core[0], f"auto: {why} (the first worker's device), from the top"


def device_addresses(indices: Sequence[int | None]) -> list[str | None]:
    """The PCI address of each accelerator index (None for None, or where the
    platform doesn't report one)."""
    from mstar.platforms import current_platform

    return [None if i is None else current_platform.pci_address(i) for i in indices]
