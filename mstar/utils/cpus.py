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

Placement under ``auto`` and ``always`` is by the device's rank among the
host's GPUs on its NUMA node, as sysfs lists them (every GPU of the host, not
only the ones this process was given), so servers on different GPUs pick
different CPUs without knowing about each other. A worker takes the core at its
device's rank from the bottom of the node's allowed CPUs, the API server the
core at its first device's rank from the top, so the cores a GPU picks from must
hold two per GPU. A mask too small for that across the host may still do for the
GPUs this container can open, which are the only ones that can share it, so those
rank instead; failing that, nothing is pinned. Servers on one GPU can't be told
apart by rank, so the first holds a lock on it (``claim_gpus``) and the next
pins nothing.

Pinning pays where a thread woken on an idle core waits for that core to leave
a deep idle state. Without one it costs throughput, since the threads only queue
on their CPUs, so ``auto`` reads the idle states sysfs lists for the mask's CPUs
and pins nothing unless one reaches ``DEEP_IDLE_US`` (``deepest_idle``). It also
refuses to give a worker a single CPU, where the step threads queue behind each
other. ``always`` skips both checks.
"""
from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PCI_DEVICES = Path("/sys/bus/pci/devices")
_CPU_TOPOLOGY = "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list"
_CPU_CLUSTER = "/sys/devices/system/cpu/cpu{}/topology/cluster_cpus_list"
_CPU_ROOT = Path("/sys/devices/system/cpu")
# not TMPDIR: every server on the host has to find the same file
_GPU_LOCK_DIR = Path("/tmp")

PINNABLE = hasattr(os, "sched_setaffinity")

# Exit latency, in µs, from which a handoff to a sleeping core costs more than
# queueing on a pinned one. VM-exposed and C1-class states sit well below it,
# the C6 class well above.
DEEP_IDLE_US = 100
# settings that place by the GPU's rank
PLACED = ("auto", "always")


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
    # nothing pinned because the host doesn't call for it, which is no fault of
    # the deployment's
    shallow: bool = False

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


def _paired(cores: list[tuple[int, ...]], cluster: Callable[[int], frozenset[int]]) -> list[tuple[int, ...]]:
    """``cores`` with each run of one-CPU cores of an L2 cluster taken two at a
    time, in the same order. Where a core brings no second hardware thread (no
    SMT, or the sibling is outside the mask) two cores of a cluster stand in
    for it: the step threads don't queue on one CPU, and a wake inside an awake
    cluster costs a fraction of one that wakes the cluster."""
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


def _step_cpus(core: tuple[int, ...], why: str) -> StepCpus:
    """A worker's step threads on all of ``core``'s CPUs. On one CPU the three
    queue behind each other (a Qwen3-TTS worker's main thread alone keeps about
    0.6 of a CPU busy at c=1); on a core's two hardware threads, or two cores
    of a cluster (``_paired``), there is room, and a woken thread still finds
    the core or the cluster awake."""
    return StepCpus(frozenset(core), why)


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
) -> CpuPlan:
    """The plan for a deployment whose workers' devices sit at PCI
    ``addresses``, in the conductor's worker order (None for a device sysfs
    doesn't place, such as a CPU worker; only ``auto`` and ``always`` read
    them), and ``reachable`` the GPUs this container can open.

    ``off`` pins nothing. A CPU number is the first worker's CPU; the next
    workers take the next cores above it, and the API server the last core of
    the mask; a number outside the mask, or too near its top for every worker,
    raises ``CpuPlanError``. ``always`` places by device rank (module
    docstring), and pins nothing, saying why in ``CpuPlan.why``, when the mask
    is too small. ``auto`` does the same where the mask's CPUs have a deep idle
    state and every worker gets two CPUs, and otherwise pins nothing, saying so.
    Either way at least one CPU of the mask stays unpinned.
    """
    mask = frozenset(process_mask() if mask is None else mask)
    usable = usable_cpus() if usable is None else usable
    if setting == "off":
        return CpuPlan(mask, usable, why="step_thread_core off")
    if not PINNABLE:
        return CpuPlan(mask, usable, why="not pinned: no sched_setaffinity here")
    if setting == "auto":
        shallow = _no_deep_idle(mask, cpu_root)
        if shallow is not None:
            return CpuPlan(mask, usable, why=f"not pinned: {shallow}", shallow=True)
    cores = _cores(mask, siblings)
    if setting in PLACED:
        def cores_of(cpus: Collection[int]) -> list[tuple[int, ...]]:
            return _paired(_cores(cpus, siblings), cluster)

        placed = _auto(setting, addresses, mask, cores_of, pci_root, reachable)
        if isinstance(placed, str):
            return CpuPlan(mask, usable, why=f"not pinned: {placed}")
        workers, result, why = placed
    else:
        workers, result, why = _explicit(int(setting), len(addresses), mask, cores)
    out = CpuPlan(mask, usable, tuple(workers), result, why)
    if setting == "auto" and any(len(step.cpus) < 2 for step in workers):
        return CpuPlan(mask, usable, why=f"not pinned: {_one_cpu_worker(workers, mask, siblings)}")
    if len(out.pinned) >= len(mask):
        reason = (f"{len(addresses)} worker(s) and the API server would pin all {len(mask)} CPU(s) "
                  f"of the process mask {format_cpu_list(mask)}, leaving none for the rest")
        if setting in PLACED:
            return CpuPlan(mask, usable, why=f"not pinned: {reason}")
        raise CpuPlanError(f"step_thread_core {setting}: {reason}")
    return out


def _no_deep_idle(mask: frozenset[int], cpu_root: Path) -> str | None:
    """Why the mask's CPUs don't pay for pinning; None when they do."""
    deepest = deepest_idle(mask, cpu_root)
    if deepest is None:
        where = "sysfs lists no idle state"
    elif deepest.latency_us < DEEP_IDLE_US:
        where = f"the deepest idle state is {deepest.name} at {deepest.latency_us} µs"
    else:
        return None
    return (f"{where} on CPUs {format_cpu_list(mask)}, under the {DEEP_IDLE_US} µs that pays for pinning "
            "(step_thread_core always pins anyway)")


def _one_cpu_worker(workers: Sequence[StepCpus], mask: frozenset[int], siblings) -> str:
    """Why ``auto`` refuses a worker a single CPU, and the mask that would do."""
    wider = mask.union(*(siblings(cpu) for step in workers for cpu in step.cpus))
    needs = (f"try {format_cpu_list(wider)}, which adds their hardware threads" if len(wider) > len(mask)
             else "give it two cores of one L2 cluster")
    return (f"the mask {format_cpu_list(mask)} leaves a worker's main, GPU and plan threads one CPU to queue on, "
            f"which costs a host-bound step more than the pin saves; {needs} (step_thread_core always pins anyway)")


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
    """The cores some GPUs pick from. A GPU's worker takes the core at ``rank``
    from the bottom and its API server the one at ``rank`` from the top, so
    they need two cores per GPU to stay apart."""
    cores: list[tuple[int, ...]]
    rank: int
    gpus: int
    why: str


def _gpu_pools(gpus, mask, cores_of) -> dict[str, _Pool]:
    """Each host GPU's pool. A GPU with allowed CPUs on its node picks among
    those by its rank on the node; the GPUs with none pick, by their rank among
    themselves, from the cores those ranks leave."""
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
            pools[gpu.address] = _Pool(
                local, rank, count, f"GPU {rank + 1} of {count} on its NUMA node, over its allowed CPUs {where}")
        if len(local) >= 2 * count:
            claimed.update(local[:count] + local[-count:])
    spare = [core for core in cores_of(mask) if core not in claimed]
    elsewhere = [gpu for gpu in gpus if gpu.address not in pools]
    for rank, gpu in enumerate(elsewhere):
        pools[gpu.address] = _Pool(
            spare, rank, len(elsewhere),
            f"no allowed CPU on its NUMA node, so GPU {rank + 1} of {len(elsewhere)} without one, over the cores "
            "the node-local GPUs leave")
    return pools


def _auto(setting, addresses, mask, cores_of, pci_root, reachable):
    """``(workers, result, why)``, or why nothing can be pinned."""
    cores = cores_of(mask)
    if not cores or not addresses:
        return "no CPU or no worker to place"
    gpus = next((found for a in addresses if a and (found := host_gpus(a, pci_root))), [])
    unplaced = _Pool(cores, 0, 1, "no sysfs entry for its device, so over the process mask")

    def short_of_cores(group):
        pools = _gpu_pools(group, mask, cores_of)
        homes = [pools.get(a, unplaced) for a in addresses]
        return homes, next((home for home in homes if len(home.cores) < 2 * home.gpus), None)

    homes, short = short_of_cores(gpus)
    within = [gpu for gpu in gpus if gpu.address in reachable]
    if short is not None and within and len(within) < len(gpus):
        homes, short = short_of_cores(within)
    if short is not None:
        return (f"{short.why}: {len(short.cores)} core(s) there for {short.gpus} GPU(s), and each needs two; "
                "widen the mask or name a CPU")
    first = homes[0]
    result_core = first.cores[len(first.cores) - 1 - first.rank]
    taken = {result_core}
    workers = []
    for i, home in enumerate(homes):
        # workers of this deployment on one device (a split config) step by the
        # pool's GPU count, so each GPU's first worker keeps its core; the top
        # ``gpus`` cores are those GPUs' API servers'
        nth = sum(a == addresses[i] for a in addresses[:i])
        steps = home.cores[home.rank + nth * home.gpus:len(home.cores) - home.gpus]
        core = next((c for c in steps if c not in taken), None)
        if core is None:
            return f"{len(addresses)} worker(s) and the API server need more cores than the mask has"
        taken.add(core)
        workers.append(_step_cpus(core, f"{setting}: {home.why}"))
    return workers, result_core[0], f"{setting}: {first.why} (the first worker's device), from the top"


def device_addresses(devices: Sequence[str]) -> list[str | None]:
    """The PCI address of each worker's device (``cuda:1``); None for a CPU
    worker, or where the platform doesn't report one."""
    from mstar.platforms import current_platform

    addresses: list[str | None] = []
    for device in devices:
        kind, _, index = device.partition(":")
        addresses.append(current_platform.pci_address(int(index)) if index and kind != "cpu" else None)
    return addresses


def devices_plan(setting: str | int, devices: Sequence[str]) -> CpuPlan:
    """``deployment_plan`` for the workers' device strings, in the conductor's
    worker order: the same call in the API server and in every worker."""
    # a host auto leaves alone never asks torch about its devices
    if setting not in PLACED or (setting == "auto" and _no_deep_idle(process_mask(), _CPU_ROOT)):
        return deployment_plan(setting, [None] * len(devices))
    from mstar.platforms import current_platform

    return deployment_plan(setting, device_addresses(devices), reachable=current_platform.reachable_pci_addresses())


# ── one server per GPU ──────────────────────────────────────────────────────

_gpu_locks: dict[str, int] = {}


def _open_lock(path: Path) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        # another user's file in a sticky directory refuses O_CREAT, and flock
        # needs no write access
        return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    # The umask narrows the mode above, and every user's servers open this file to lock it.
    os.fchmod(fd, 0o644)
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
