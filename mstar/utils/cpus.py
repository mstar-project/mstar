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

Placement under ``always`` and ``siblings`` is by the device's rank among the
host's GPUs on its NUMA node, as sysfs lists them (every GPU of the host, not
only the ones this process was given), so servers on different GPUs pick
different CPUs without knowing about each other. The API server takes the core
at its first device's rank from the top of the node's allowed CPUs, and a worker
the step unit at its device's rank from the bottom of the cores below the API
servers'. Under ``always`` a unit is ``STEP_CORES`` cores, one hardware thread
of each, so the cores a GPU picks from must hold four per GPU, and no two pinned
threads share a core; ``siblings`` (``_siblings``) puts a worker on one core's
hardware threads instead. A mask too small across the host may still do for the
GPUs this container can open, which are the only ones that can share it, so those
rank instead. Failing that, the GPUs this process was given rank, and the mask
is cut into as many core sets as it holds for them. Servers on one GPU can't be
told apart by rank, so the first holds a lock on it (``claim_gpus``) and the next
pins nothing. Under the last rule servers on different GPUs can't either, so
every server that pins also locks its cores (``claim_cores``), and one that
finds a set held takes the next.

Pinning can pay where a thread woken on an idle core waits for that core to
leave a deep idle state, and costs throughput where it doesn't, since the
threads only queue on their CPUs. Which layout pays depends on the host: on a
deep-idle Xeon one core's hardware threads beat no pin and separate cores lost
to it, while one core's hardware threads lost on two H200 hosts. So ``auto``
pins nothing and logs why (``_auto_declines``): the reason names the idle hold,
a CPU vendor other than Intel, idle states under ``DEEP_IDLE_US``
(``deepest_idle``), or what the separate cores measured.
"""
from __future__ import annotations

import fcntl
import logging
import math
import os
import sys
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PCI_DEVICES = Path("/sys/bus/pci/devices")
_CPU_TOPOLOGY = "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list"
_CPU_CLUSTER = "/sys/devices/system/cpu/cpu{}/topology/cluster_cpus_list"
_CPU_ROOT = Path("/sys/devices/system/cpu")
_CPUINFO = Path("/proc/cpuinfo")
_PM_QOS_DEVICE = Path("/dev/cpu_dma_latency")
# not TMPDIR: every server on the host has to find the same file
_GPU_LOCK_DIR = Path("/tmp")
# The core set an API server locked (``claim_cores``), for the workers it starts:
# they build the same plan, but which set was free only that process knows.
CORE_SET_ENV = "MSTAR_STEP_CORE_SET"

logger = logging.getLogger(__name__)

PINNABLE = hasattr(os, "sched_setaffinity")

# Exit latency, in µs, from which a handoff to a sleeping core costs more than
# queueing on a pinned one. VM-exposed and C1-class states sit well below it,
# the C6 class well above.
DEEP_IDLE_US = 100
# settings that place by the GPU's rank
PLACED = ("always", "siblings")
# Physical cores one worker's main, GPU and plan threads share under ``always``,
# one hardware thread of each: a woken step thread finds a CPU of its own, and
# none of them shares a core's execution units with another.
STEP_CORES = 3
# CPU latency the idle hold asks for: admits C1 and C1E, refuses the C6 class
IDLE_HOLD_US = 10
# ``/proc/cpuinfo`` vendors ``auto``'s measurements come from (Xeon 6767P and
# 6740E hosts); elsewhere its reason says so rather than quote them
AUTO_VENDORS = ("GenuineIntel",)


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
    """The CPUs one worker's main, GPU and plan threads share, and how they
    were chosen, for the log."""
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
    # auto chose not to pin here, which is no fault of the deployment's
    declined: bool = False
    # the disjoint core sets the mask holds for this deployment, and the one taken
    sets: int = 1
    slot: int = 0
    # the cores whose locks this process holds (``claim_cores``), for the log:
    # the workers build the same plan without them
    locked: str = field(default="", compare=False)

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

    def prepare_workers(self, threads: int) -> int:
        """Requests the data worker prepares at once, for a model that
        declares ``PREPROCESS_PARALLEL``, when each runs on ``threads``
        intra-op threads: at most 8, and the pool times the threads stays
        within the CPUs left."""
        return max(1, min(8, self.left // max(1, threads)))

    @property
    def preprocess_workers(self) -> int:
        """``prepare_workers`` at the default thread count."""
        return self.prepare_workers(self.preprocess_threads)

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


def l2_cluster(cpu: int) -> frozenset[int]:
    """The CPUs sharing ``cpu``'s L2 cluster, ``cpu`` included."""
    try:
        with open(_CPU_CLUSTER.format(cpu)) as f:
            return parse_cpu_list(f.read()) | {cpu}
    except (OSError, ValueError):
        return frozenset({cpu})


def _cores(cpus: Collection[int], siblings: Callable[[int], frozenset[int]]) -> list[tuple[int, ...]]:
    """``cpus`` grouped by physical core (each core's CPUs within ``cpus``, in
    order), cores ordered by their first CPU."""
    cores = {tuple(sorted(siblings(c) & set(cpus))) for c in cpus}
    return sorted(cores)


class _Unit(NamedTuple):
    """The CPUs one worker's step threads share, and the physical cores they
    come from."""
    cpus: tuple[int, ...]
    cores: tuple[tuple[int, ...], ...]


class _Layout(NamedTuple):
    """How a placing setting cuts a pool's cores into workers' step units, and
    what each GPU needs, for the log."""
    units: Callable[[list[tuple[int, ...]]], list[_Unit]]
    need: str


def _spread(cores: list[tuple[int, ...]]) -> list[_Unit]:
    """``cores`` taken ``STEP_CORES`` at a time, in order, one hardware thread
    of each: a worker's step threads under ``always``."""
    groups = (cores[i:i + STEP_CORES] for i in range(0, len(cores) - STEP_CORES + 1, STEP_CORES))
    return [_Unit(tuple(core[0] for core in group), tuple(group)) for group in groups]


def _paired(cores: list[tuple[int, ...]], cluster: Callable[[int], frozenset[int]]) -> list[tuple[int, ...]]:
    """``cores`` with each run of one-CPU cores of an L2 cluster taken two at a
    time, in the same order."""
    alone: dict[frozenset[int], list[tuple[int, ...]]] = {}
    for core in cores:
        if len(core) == 1:
            alone.setdefault(cluster(core[0]), []).append(core)
    # a core -> the pair it leads, or None if it follows one
    merged: dict[tuple[int, ...], tuple[int, ...] | None] = {}
    for members in alone.values():
        for first, second in zip(members[::2], members[1::2], strict=False):
            merged[first], merged[second] = first + second, None
    placed = (merged.get(core, core) for core in cores)
    return [core for core in placed if core is not None]


def _siblings(cores: list[tuple[int, ...]], cluster: Callable[[int], frozenset[int]]) -> list[_Unit]:
    """``step_thread_core: siblings``: a worker's step threads on all hardware
    threads of one core, or on two cores of an L2 cluster where a core has one
    CPU in the mask. The core (or cluster) stays awake while any of the three
    runs, so a handoff rarely waits for it to leave a deep idle state, but the
    three share its execution units.

    Qwen3.5-0.8B tok/s at c=1 / c=8 on 8 cores of the server's mask: Xeon
    6767P (SMT, an L2 per core, C6 at 170 µs) 808 / 5670 against 548 / 3939
    with no pin and 526 / 3851 on three cores of their own; Xeon 6740E (no SMT,
    four cores to an L2 cluster) 527 / 3635 against 371 / 2688. With a
    ``SCHED_IDLE`` spinner holding the 6767P's cores awake it lost at c=1, 753
    against 906 with no pin. On two H200 hosts with SMT it lost to no pin: c=1
    506 against 634 on a 2-socket AMD EPYC (C2 at 800 µs), 13-26% at c=1-32 on
    the other. So ``auto`` never takes it."""
    return [_Unit(unit, tuple(core for core in cores if core[0] in unit)) for unit in _paired(cores, cluster)]


def _step_cpus(core: tuple[int, ...], why: str) -> StepCpus:
    """A worker's step threads on all of ``core``'s CPUs: what a
    ``step_thread_core`` number gives each worker."""
    return StepCpus(frozenset(core), why)


def cpu_vendor(cpuinfo: Path | None = None) -> str | None:
    """The first ``vendor_id`` ``/proc/cpuinfo`` lists (``GenuineIntel``,
    ``AuthenticAMD``); None where it lists none, as on Arm, or can't be read."""
    try:
        with open(cpuinfo or _CPUINFO) as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "vendor_id":
                    return value.strip() or None
    except OSError:
        pass
    return None


# ── how deep the host sleeps ────────────────────────────────────────────────

class IdleState(NamedTuple):
    """An idle state a CPU may enter, as sysfs names it, with its exit latency."""
    name: str
    latency_us: int


def deepest_idle(cpus: Iterable[int], root: Path = _CPU_ROOT) -> IdleState | None:
    """The idle state of ``cpus`` with the longest exit latency, leaving out
    states disabled in sysfs; None where sysfs lists none (a VM that exposes
    no cpuidle)."""
    deepest = None
    for cpu in cpus:
        try:
            states = list((root / f"cpu{cpu}" / "cpuidle").glob("state*"))
        except OSError:
            continue
        for state in states:
            latency, name = _read(state / "latency"), _read(state / "name")
            if latency is None or not latency.isdigit() or _read(state / "disable") == "1":
                continue
            if deepest is None or int(latency) > deepest.latency_us:
                deepest = IdleState(name or state.name, int(latency))
    return deepest


_idle_hold_fd: int | None = None


def hold_idle(device: Path = _PM_QOS_DEVICE) -> str:
    """Hold a PM QoS CPU latency request of ``IDLE_HOLD_US`` for the life of
    this process, and say what happened, for the log. The kernel keeps every
    CPU of the host out of idle states slower than that while the fd stays
    open, and drops the request however the process exits. Failing to get it
    costs only the hold, so this never raises."""
    global _idle_hold_fd
    if _idle_hold_fd is not None:
        return f"idle_hold qos: already holding {device}"
    try:
        fd = os.open(device, os.O_WRONLY)
    except FileNotFoundError:
        return f"idle_hold qos: nothing held, {device} is absent (a container often hides it)"
    except OSError as e:
        return f"idle_hold qos: nothing held, can't open {device}: {e.strerror}"
    try:
        os.write(fd, IDLE_HOLD_US.to_bytes(4, sys.byteorder))
    except OSError as e:
        os.close(fd)
        return f"idle_hold qos: nothing held, can't write {device}: {e.strerror}"
    _idle_hold_fd = fd
    return f"idle_hold qos: holding {device} at {IDLE_HOLD_US} µs for this server's life"


# ── building it ─────────────────────────────────────────────────────────────

def deployment_plan(
    setting: str | int,
    addresses: Sequence[str | None],
    *,
    mask: Collection[int] | None = None,
    usable: int | None = None,
    pci_root: Path = _PCI_DEVICES,
    siblings: Callable[[int], frozenset[int]] = smt_siblings,
    cluster: Callable[[int], frozenset[int]] = l2_cluster,
    cpu_root: Path = _CPU_ROOT,
    reachable: Collection[str] = (),
    own: Collection[str] = (),
    slot: int = 0,
    cpuinfo: Path | None = None,
    idle_hold: str = "off",
) -> CpuPlan:
    """The plan for a deployment whose workers' devices sit at PCI
    ``addresses``, in the conductor's worker order (None for a device sysfs
    doesn't place, such as a CPU worker; only ``always`` and ``siblings``
    read them), ``reachable`` the GPUs this container can open, ``own`` the
    ones this process was given, and ``slot`` the core set to take when the
    mask holds more than one (``claim_cores``).

    ``off`` pins nothing, and ``auto`` pins nothing and says why. A CPU number
    is the first worker's core; the next workers take the next cores above it,
    and the API server the last core of the mask; a number outside the mask,
    or too near its top for every worker, raises ``CpuPlanError``. ``always``
    and ``siblings`` place by device rank (module docstring), and pin nothing,
    saying why in ``CpuPlan.why``, when the mask is too small. At least one CPU
    of the mask stays unpinned.
    """
    mask = frozenset(process_mask() if mask is None else mask)
    usable = usable_cpus() if usable is None else usable
    if setting == "off":
        return CpuPlan(mask, usable, why="step_thread_core off")
    if not PINNABLE:
        return CpuPlan(mask, usable, why="not pinned: no sched_setaffinity here")
    if setting == "auto":
        declined = _auto_declines(idle_hold, cpu_vendor(cpuinfo), mask, cpu_root)
        return CpuPlan(mask, usable, why=f"not pinned: {declined}", declined=True)
    cores = _cores(mask, siblings)
    if setting in PLACED:
        def cores_of(cpus: Collection[int]) -> list[tuple[int, ...]]:
            return _cores(cpus, siblings)

        if setting == "siblings":
            layout = _Layout(lambda cores: _siblings(cores, cluster),
                             "2, one for its worker's step threads and one for its API server (3 where a core has "
                             "one CPU in the mask)")
        else:
            layout = _Layout(_spread, f"{STEP_CORES + 1} ({STEP_CORES} for its worker's step threads, 1 for its API "
                                      "server)")
        placed = _by_rank(setting, addresses, mask, cores_of, layout, pci_root, reachable, own, slot)
        if isinstance(placed, str):
            return CpuPlan(mask, usable, why=f"not pinned: {placed}")
        workers, result, why, sets = placed
    else:
        workers, result, why = _explicit(int(setting), len(addresses), mask, cores)
        sets, slot = 1, 0
    out = CpuPlan(mask, usable, tuple(workers), result, why, sets=sets, slot=slot)
    if len(out.pinned) >= len(mask):
        reason = (f"{len(addresses)} worker(s) and the API server would pin all {len(mask)} CPU(s) "
                  f"of the process mask {format_cpu_list(mask)}, leaving none for the rest")
        if setting in PLACED:
            return CpuPlan(mask, usable, why=f"not pinned: {reason}")
        raise CpuPlanError(f"step_thread_core {setting}: {reason}")
    return out


def _auto_declines(idle_hold: str, vendor: str | None, mask: frozenset[int], cpu_root: Path) -> str:
    """Why ``auto`` leaves this host's threads alone. No host it can tell
    apart has a measurement in which its layout beats no pin yet, so the
    answer is never None; the reason names what the host has."""
    anyway = "step_thread_core always pins anyway"
    if idle_hold != "off":
        return (f"idle_hold {idle_hold} keeps the cores out of deep idle, where pinning measured no faster than none "
                f"({anyway})")
    if vendor not in AUTO_VENDORS:
        this = f"this host's is {vendor}" if vendor else "/proc/cpuinfo names no vendor here"
        return f"auto is measured on Intel hosts only and {this} ({anyway})"
    shallow = _no_deep_idle(mask, cpu_root)
    if shallow is not None:
        return shallow
    return (f"on an Intel host whose cores sleep deep, the step threads on {STEP_CORES} cores of their own measured "
            f"no faster than no pin; step_thread_core siblings, one core's hardware threads, measured faster there "
            f"but lost on other hosts ({anyway})")


def _no_deep_idle(mask: frozenset[int], cpu_root: Path) -> str | None:
    """Why the mask's CPUs don't pay for pinning; None when they do."""
    deepest = deepest_idle(mask, cpu_root)
    if deepest is None:
        where = "sysfs lists no idle state"
    elif deepest.latency_us < DEEP_IDLE_US:
        where = f"the deepest idle state is {deepest.name} at {deepest.latency_us} µs"
    else:
        return None
    return (f"{where} on CPUs {format_cpu_list(mask)}, under the {DEEP_IDLE_US} µs from which pinning could pay "
            "(step_thread_core always pins anyway)")


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


class _Pool(NamedTuple):
    """The physical cores some GPUs pick from. The top ``gpus`` cores are their
    API servers', by ``rank`` from the top; a GPU's worker takes the step
    units of the cores below at ``rank`` from the bottom."""
    cores: list[tuple[int, ...]]
    rank: int
    gpus: int
    why: str

    def units(self, layout: _Layout) -> list[_Unit]:
        return layout.units(self.cores[:len(self.cores) - self.gpus])


def _gpu_pools(gpus, mask, cores_of, layout, sets=1, slot=0) -> dict[str, _Pool]:
    """Each GPU's pool. A GPU with allowed CPUs on its node picks among those
    by its rank on the node; the GPUs with none pick, by their rank among
    themselves, from the cores those ranks leave. A pool holds ``sets`` copies
    of its GPUs' ranks, and these GPUs take copy ``slot``."""
    nodes: dict[str | None, list[HostGpu]] = {}
    for gpu in gpus:
        nodes.setdefault(gpu.node, []).append(gpu)
    pools, claimed = {}, set()
    for members in nodes.values():
        local = cores_of(members[0].local & mask) if members[0].local else []
        if not local:
            continue
        count = len(members)
        where = format_cpu_list(c for core in local for c in core)
        for rank, gpu in enumerate(members):
            pools[gpu.address] = _Pool(local, slot * count + rank, count * sets,
                                       f"GPU {rank + 1} of {count} on its NUMA node, over its allowed CPUs {where}")
        units = pools[members[0].address].units(layout)
        if len(units) >= count * sets:
            claimed.update(core for unit in units[:count * sets] for core in unit.cores)
            claimed.update(local[len(local) - count * sets:])
    spare = [core for core in cores_of(mask) if core not in claimed]
    elsewhere = [gpu for gpu in gpus if gpu.address not in pools]
    for rank, gpu in enumerate(elsewhere):
        pools[gpu.address] = _Pool(
            spare, slot * len(elsewhere) + rank, len(elsewhere) * sets,
            f"no allowed CPU on its NUMA node, so GPU {rank + 1} of {len(elsewhere)} without one, over the cores "
            "the node-local GPUs leave")
    return pools


def _by_rank(setting, addresses, mask, cores_of, layout, pci_root, reachable, own=(), slot=0):
    """``(workers, result, why, sets)``, or why nothing can be pinned."""
    cores = cores_of(mask)
    if not cores or not addresses:
        return "no CPU or no worker to place"
    gpus = next((found for a in addresses if a and (found := host_gpus(a, pci_root))), [])
    unplaced = _Pool(cores, 0, 1, "no sysfs entry for its device, so over the process mask")

    def short_of_cores(group, sets=1, slot=0):
        pools = _gpu_pools(group, mask, cores_of, layout, sets, slot)
        homes = [pools.get(a, unplaced) for a in addresses]
        return homes, next((home for home in homes if len(home.units(layout)) < home.gpus), None)

    group = gpus
    homes, short = short_of_cores(group)
    within = [gpu for gpu in gpus if gpu.address in reachable]
    if short is not None and within and len(within) < len(group):
        group = within
        homes, short = short_of_cores(group)
    # Short of that, size by this process's own GPUs. Servers in one container
    # that hold the same rank among their own GPUs then pick the same cores,
    # so the mask is cut into as many sets as it holds and the core locks
    # (``claim_cores``) give each server its own.
    mine = [gpu for gpu in gpus if gpu.address in own]
    sets, rule = 1, None
    if short is not None and mine and len(mine) < len(group) and {a for a in addresses if a} <= set(own):
        group = mine
        homes, short = short_of_cores(group)
        while short is None and sets < len(cores) and short_of_cores(group, sets + 1)[1] is None:
            sets += 1
        rule = f"sized by the GPUs this process uses, set {slot + 1} of {sets}"
    if short is not None:
        return (f"{short.why}: {len(short.cores)} core(s) there for {short.gpus} GPU(s), and each needs "
                f"{layout.need}; widen the mask or name a CPU")
    if slot >= sets:
        return f"core set {slot + 1} asked for, and the mask holds {sets} for this deployment"
    if slot:
        homes, _ = short_of_cores(group, sets, slot)
    elif sets > 1:
        homes, _ = short_of_cores(group, sets)
    first = homes[0]
    result_core = first.cores[len(first.cores) - 1 - first.rank]
    taken = {result_core}
    workers = []
    for i, home in enumerate(homes):
        # workers of this deployment on one device (a split config) step by the
        # pool's GPU count, so each GPU's first worker keeps its cores
        nth = sum(a == addresses[i] for a in addresses[:i])
        units = home.units(layout)[home.rank + nth * home.gpus:]
        unit = next((u for u in units if not taken.intersection(u.cores)), None)
        if unit is None:
            return f"{len(addresses)} worker(s) and the API server need more cores than the mask has"
        taken.update(unit.cores)
        workers.append(StepCpus(frozenset(unit.cpus), f"{setting}: {home.why}" + (f", {rule}" if rule else "")))
    why = f"{setting}: {first.why} (the first worker's device), from the top; {rule or 'every reachable GPU fits'}"
    return workers, result_core[0], why, sets


def device_addresses(devices: Sequence[str]) -> list[str | None]:
    """The PCI address of each worker's device (``cuda:1``); None for a CPU
    worker, or where the platform doesn't report one."""
    from mstar.platforms import current_platform

    addresses: list[str | None] = []
    for device in devices:
        kind, _, index = device.partition(":")
        addresses.append(current_platform.pci_address(int(index)) if index and kind != "cpu" else None)
    return addresses


def devices_plan(setting: str | int, devices: Sequence[str], idle_hold: str = "off",
                 slot: int | None = None) -> CpuPlan:
    """``deployment_plan`` for the workers' device strings, in the conductor's
    worker order: the same call in the API server and in every worker. A
    worker takes the core set its API server locked (``CORE_SET_ENV``)."""
    # auto, which pins nothing, never asks torch about the devices
    if setting not in PLACED:
        return deployment_plan(setting, [None] * len(devices), idle_hold=idle_hold)
    from mstar.platforms import current_platform

    if slot is None:
        inherited = os.environ.get(CORE_SET_ENV, "")
        slot = int(inherited) if inherited.isdigit() else 0
    return deployment_plan(setting, device_addresses(devices), idle_hold=idle_hold,
                           reachable=current_platform.reachable_pci_addresses(),
                           own=current_platform.visible_pci_addresses(), slot=slot)


# ── one server per GPU ──────────────────────────────────────────────────────

_gpu_locks: dict[str, int] = {}


def _open_lock(path: Path, writable: bool = False) -> int:
    access = os.O_RDWR if writable else os.O_RDONLY
    try:
        fd = os.open(path, access | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        # another user's file in a sticky directory refuses O_CREAT, and flock
        # needs no write access
        try:
            return os.open(path, access | os.O_NOFOLLOW)
        except PermissionError:
            return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    # The umask narrows the mode above, and every user's servers open this file
    # to lock it (and write their pid into a writable one).
    os.fchmod(fd, 0o666 if writable else 0o644)
    return fd


def claim_gpus(addresses: Iterable[str | None]) -> str | None:
    """Hold a lock on each GPU for the life of this process; if one can't be
    had, hold none of them and say why.

    Placement depends only on the PCI address and the mask, so two servers on
    one GPU (two MIG instances report one address) would take the same CPUs.
    The kernel drops an ``flock`` however the process exits, so a dead server
    leaves nothing to clean up. The file is never unlinked: another process may
    already hold an fd on this inode, and removing it would let a third lock a
    different one. The lock only reaches servers that share ``_GPU_LOCK_DIR``.
    """
    taken: dict[str, int] = {}
    refused = None
    for address in sorted({a for a in addresses if a} - _gpu_locks.keys()):
        path = _GPU_LOCK_DIR / f"mstar-gpu-{address}.lock"
        fd = None
        try:
            fd = _open_lock(path)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refused = f"another server holds GPU {address} (its lock is {path})"
        except OSError as e:
            refused = f"can't lock GPU {address}: {e}"
        if refused is not None:
            for held in [*taken.values(), *([] if fd is None else [fd])]:
                os.close(held)
            return refused
        taken[address] = fd
    _gpu_locks.update(taken)
    return None


# ── one server per core ─────────────────────────────────────────────────────

_core_locks: dict[Path, int] = {}


def _pid_in(fd: int) -> int | None:
    text = os.pread(fd, 32, 0).decode(errors="replace").strip()
    return int(text) if text.isdigit() else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _lock_cores(cores: Iterable[int]) -> str | None:
    """Lock each physical core (by its first CPU) for the life of this
    process, writing its pid into the file; if one is held, hold none of them
    and say by whom."""
    taken: dict[Path, int] = {}
    for core in sorted(cores):
        path = _GPU_LOCK_DIR / f"mstar-core-{core}.lock"
        if path in _core_locks:
            continue
        fd = None
        try:
            fd = _open_lock(path, writable=True)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = _pid_in(fd)
            holder = f"pid {pid}" if pid is not None and _alive(pid) else "another process"
            refused = f"core {core} is held by {holder} (its lock is {path})"
        except OSError as e:
            refused = f"can't lock core {core}: {e}"
        else:
            taken[path] = fd
            continue
        for held in [*taken.values(), *([] if fd is None else [fd])]:
            os.close(held)
        return refused
    for path, fd in taken.items():
        left = _pid_in(fd)
        if left is not None and left != os.getpid() and not _alive(left):
            logger.info("step_thread_core: %s names pid %d, which has exited; taking it over", path, left)
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        except OSError:
            pass  # a file this user can't write: locked all the same, without the pid
    _core_locks.update(taken)
    return None


def claim_cores(first: CpuPlan, build: Callable[[int], CpuPlan],
                siblings: Callable[[int], frozenset[int]] | None = None) -> CpuPlan:
    """The first of the deployment's core sets (``first`` is set 0's plan,
    ``build(slot)`` gives the others) whose physical cores this process can
    lock, holding those locks for its life; else a plan that pins nothing and
    says who holds them.

    Every plan that pins goes through this, a CPU number and ``siblings``
    included, so no two servers that share ``_GPU_LOCK_DIR`` pin threads on
    one core. The kernel drops an ``flock`` however the process exits, and the
    pid in the file is only for the next server's log: a file naming a process
    that has exited is taken over. Like the GPU lock, it doesn't reach a server
    in another container with its own ``/tmp``."""
    siblings = siblings or smt_siblings
    plan, refusals = first, []
    for slot in range(first.sets):
        if slot:
            plan = build(slot)
            if not plan.pinned:
                refusals.append(f"set {slot + 1}: {plan.why}")
                continue
        cores = {min(siblings(cpu)) for cpu in plan.pinned}
        refused = _lock_cores(cores)
        if refused is None:
            return replace(plan, locked=format_cpu_list(cores))
        refusals.append(f"set {slot + 1} of {first.sets}: {refused}")
        if slot + 1 < first.sets:
            logger.info("step_thread_core: %s; trying the next set", refusals[-1])
    return CpuPlan(first.mask, first.usable,
                   why=f"not pinned: no core set left in the mask ({'; '.join(refusals)})")
