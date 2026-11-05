"""Pinning the step threads (``step_thread_core``): the CPU plan every process
builds (``mstar.utils.cpus``) from sysfs and the process mask, the lock that
keeps a second server off a GPU, the device list the API server and the
conductor share, the startup failures, the pool sizes, the thread helpers
(``mstar.utils.threads``) and the ruff rule that enforces them, and the real
``Worker.run`` moving its three step threads onto their CPUs. The knob itself is
in ``test_knobs``."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from mstar.conductor.workers import deployment_workers  # noqa: E402
from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, interface  # noqa: E402
from mstar.utils import cpus, threads  # noqa: E402
from mstar.utils.cpus import CpuPlan, CpuPlanError, StepCpus, deployment_plan, parse_cpu_list  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NODES = [parse_cpu_list(f"{32 * n}-{32 * n + 31},{128 + 32 * n}-{128 + 32 * n + 31}") for n in range(4)]
HOST = frozenset(range(256))
# this test host's 8 GPUs, two per NUMA node, by PCI address
GPUS = {"0000:1a:00.0": 0, "0000:40:00.0": 0, "0000:62:00.0": 1, "0000:73:00.0": 1,
        "0000:9a:00.0": 2, "0000:da:00.0": 2, "0000:f3:00.0": 3, "0000:fa:00.0": 3}


def smt(cpu: int) -> frozenset[int]:
    """CPU n and n + 128 share a core."""
    return frozenset({cpu % 128, cpu % 128 + 128})


def _sysfs(tmp_path, gpus=GPUS, nodes=NODES) -> Path:
    """A fake /sys/bus/pci/devices: the GPUs, plus a BMC display adapter and a
    NIC that are not GPUs."""
    root = tmp_path / "pci"
    devices = {address: ("0x10de", "0x030200", node) for address, node in gpus.items()}
    devices |= {"0000:2c:00.0": ("0x1a03", "0x030000", 0), "0000:3b:00.0": ("0x15b3", "0x020700", 0)}
    for address, (vendor, cls, node) in devices.items():
        d = root / address
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "class").write_text(cls + "\n")
        (d / "numa_node").write_text(f"{node}\n")
        local = cpus.format_cpu_list(nodes[node]) if node >= 0 else cpus.format_cpu_list(HOST)
        (d / "local_cpulist").write_text(local + "\n")
    return root


def _idle(tmp_path, cpus_, states) -> Path:
    """A fake /sys/devices/system/cpu: ``states`` is (name, exit latency, disabled) per state of each CPU."""
    root = tmp_path / "cpu"
    for cpu in cpus_:
        for i, (name, latency, disabled) in enumerate(states):
            d = root / f"cpu{cpu}" / "cpuidle" / f"state{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "name").write_text(name + "\n")
            (d / "latency").write_text(f"{latency}\n")
            (d / "disable").write_text(f"{int(disabled)}\n")
    return root


XEON = [("POLL", 0, False), ("C1", 1, False), ("C1E", 2, False), ("C6", 170, False), ("C6P", 210, False)]
SIERRA = [("POLL", 0, False), ("C1", 1, False), ("C1E", 2, False), ("C6S", 270, False), ("C6SP", 310, False)]
VM = [("POLL", 0, False), ("C1", 2, False)]


@pytest.fixture(autouse=True)
def _deep_idle_host(tmp_path_factory, monkeypatch):
    """``devices_plan`` reads the idle states of the process mask: hosts here have deep ones unless a test says
    otherwise, whether or not the machine running the tests exposes cpuidle."""
    monkeypatch.setattr(cpus, "_CPU_ROOT", _idle(tmp_path_factory.mktemp("sysfs"), [min(cpus.process_mask())], XEON))


INTEL = "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\n\nprocessor\t: 1\nvendor_id\t: GenuineIntel\n"
AMD = INTEL.replace("GenuineIntel", "AuthenticAMD")
ARM = "processor\t: 0\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\n\nprocessor\t: 1\n"


def _cpuinfo(tmp_path, text) -> Path:
    path = tmp_path / "cpuinfo"
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def _intel_host(tmp_path_factory, monkeypatch):
    """``auto`` pins only on the CPU vendors it is measured on: hosts here are Intel unless a test says otherwise,
    whatever CPU runs the tests."""
    monkeypatch.setattr(cpus, "_CPUINFO", _cpuinfo(tmp_path_factory.mktemp("proc"), INTEL))


def one_cpu_cluster(cpu: int) -> frozenset[int]:
    """An L2 per core, as on a P-core Xeon."""
    return frozenset({cpu})


def _plan(setting, addresses, mask, root, usable=None, idle=None, reachable=(), siblings=smt, cluster=one_cpu_cluster,
          idle_hold="off", own=(), slot=0):
    # placement is what most of this file is about, so a host with deep idle unless a test says otherwise
    cpu_root = idle if idle is not None else _idle(root.parent / "deep", [min(mask)], XEON)
    return deployment_plan(setting, addresses, mask=mask, usable=usable or len(mask), pci_root=root,
                           siblings=siblings, cluster=cluster, cpu_root=cpu_root, reachable=reachable,
                           idle_hold=idle_hold, own=own, slot=slot)


def _one(cpu) -> StepCpus:
    return StepCpus(frozenset({cpu}), "")


def _cpus(plan: CpuPlan) -> list[int | None]:
    """Each worker's first step CPU (its core's first hardware thread)."""
    return [None if w is None else min(w.cpus) for w in plan.workers]


# ── where a GPU sits ────────────────────────────────────────────────────────

def test_the_hosts_gpus_are_the_functions_of_one_vendor_and_class_by_address(tmp_path):
    root = _sysfs(tmp_path)
    gpus = cpus.host_gpus("0000:40:00.0", root)
    # the display adapter on node 0 (another vendor and class) and the NIC are not GPUs
    assert [g.address for g in gpus] == list(GPUS)
    assert gpus[1] == cpus.HostGpu("0000:40:00.0", "0", NODES[0])
    assert cpus.host_gpus("0000:ff:00.0", root) == []   # no such device
    assert cpus.host_gpus("0000:40:00.0", tmp_path / "absent") == []


def test_reachable_gpus_are_the_ones_with_a_device_node(tmp_path):
    proc, dev = tmp_path / "gpus", tmp_path / "dev"
    dev.mkdir()
    for address, minor in (("0000:9a:00.0", 4), ("0000:f3:00.0", 6), ("0000:fa:00.0", 7)):
        (proc / address).mkdir(parents=True)
        (proc / address / "information").write_text(f"Model: \t B300\nDevice Minor: \t {minor}\nGPU Excluded:\t No\n")
    (proc / "0000:01:00.0").mkdir()    # a function without its information file
    (dev / "nvidia6").touch()
    (dev / "nvidia7").touch()
    (dev / "nvidiactl").touch()
    assert interface._nvidia_reachable(proc, dev) == {"0000:f3:00.0", "0000:fa:00.0"}
    assert interface._nvidia_reachable(tmp_path / "absent", dev) == frozenset()
    assert CpuPlatform().reachable_pci_addresses() == frozenset()


def test_pci_addresses_come_from_torch_on_cuda_and_rocm(monkeypatch):
    props = {0: SimpleNamespace(pci_domain_id=0, pci_bus_id=0x40, pci_device_id=0)}

    def get_props(index):
        if index not in props:
            raise AssertionError("Invalid device id")
        return props[index]

    monkeypatch.setattr(interface.torch.cuda, "get_device_properties", get_props)
    for platform in (CudaPlatform(), RocmPlatform()):
        assert platform.pci_address(0) == "0000:40:00.0"
        assert platform.pci_address(5) is None
    assert CpuPlatform().pci_address(0) is None


def test_a_processs_own_gpus_are_the_devices_torch_counts(monkeypatch):
    """``--gpus 7`` leaves torch one device, whatever device nodes the container has."""
    monkeypatch.setattr(interface.torch.accelerator, "device_count", lambda: 2)
    monkeypatch.setattr(CudaPlatform, "pci_address", lambda self, i: ["0000:fa:00.0", None][i])
    assert CudaPlatform().visible_pci_addresses() == {"0000:fa:00.0"}
    assert CpuPlatform().visible_pci_addresses() == frozenset()


# ── always: three cores a worker, one hardware thread of each ───────────────

def _one_cpu_a_core(plan: CpuPlan) -> bool:
    """No two pinned CPUs, a worker's or the API server's, are hardware threads of one core."""
    return len({smt(cpu) for cpu in plan.pinned}) == len(plan.pinned)


def test_always_takes_three_cores_at_the_gpus_rank_on_its_node(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("always", ["0000:40:00.0"], HOST, root)
    # GPU 2 of 2 on node 0: the worker the second three cores from the bottom,
    # the API server the second core from the top, one hardware thread of each
    assert plan.workers[0].cpus == {3, 4, 5} and plan.result == 30
    assert "GPU 2 of 2 on its NUMA node" in plan.workers[0].why and _one_cpu_a_core(plan)
    plan = _plan("always", ["0000:62:00.0"], HOST, root)
    assert plan.workers[0].cpus == {32, 33, 34} and plan.result == 63


@pytest.mark.parametrize(("mask", "wide"), [
    (HOST, True), (parse_cpu_list("32-39,160-167"), False), (parse_cpu_list("32-39"), False),
    (parse_cpu_list("32-63,160-191"), True), (parse_cpu_list("0-7,32-47,128-135,160-175"), True)])
def test_no_two_pinned_threads_share_a_core(tmp_path, mask, wide):
    """Whatever the mask holds of each core; on a wide mask also across GPUs and a split config's workers."""
    root = _sysfs(tmp_path)
    wider = [["0000:40:00.0", "0000:62:00.0"], ["0000:73:00.0", "0000:73:00.0"]]
    for addresses in [["0000:62:00.0"], [None]] + wider * wide:
        plan = _plan("always", addresses, mask, root, reachable=tuple(GPUS))
        assert plan.pinned, (addresses, plan.why)
        assert _one_cpu_a_core(plan) and all(len(w.cpus) == cpus.STEP_CORES for w in plan.workers)


@pytest.mark.parametrize(("mask", "a", "b"), [
    (HOST, "0000:1a:00.0", "0000:40:00.0"),                          # one node, no taskset
    (NODES[0] | NODES[1], "0000:40:00.0", "0000:62:00.0"),           # two nodes under one taskset
    (parse_cpu_list("0-19,128-147"), "0000:1a:00.0", "0000:40:00.0"),  # a 40-CPU taskset on their node
    (parse_cpu_list("32-47,96-111"), "0000:40:00.0", "0000:fa:00.0"),  # one GPU's node left out
    (parse_cpu_list("32-63"), "0000:40:00.0", "0000:fa:00.0"),         # both left out
])
def test_two_servers_one_per_gpu_under_one_taskset_take_different_cpus(tmp_path, mask, a, b):
    """Each server knows only its own GPU, and no two pick a CPU, or a core, in common."""
    root = _sysfs(tmp_path)
    first, second = _plan("always", [a], mask, root), _plan("always", [b], mask, root)
    assert first.pinned and second.pinned
    assert not {smt(c) for c in first.pinned} & {smt(c) for c in second.pinned}


def test_a_vm_without_numa_nodes_still_ranks_its_gpus(tmp_path):
    root = _sysfs(tmp_path, gpus={"0000:07:00.0": -1, "0000:08:00.0": -1})
    mask = parse_cpu_list("0-39")
    one, two = (_plan("always", [a], mask, root) for a in ("0000:07:00.0", "0000:08:00.0"))
    assert (one.workers[0].cpus, one.result, two.workers[0].cpus, two.result) == ({0, 1, 2}, 39, {3, 4, 5}, 38)


def test_workers_of_one_server_on_one_gpu_take_distinct_cores(tmp_path):
    root = _sysfs(tmp_path)
    # a split config (two workers on GPU 0) plus a worker on the node's other GPU
    plan = _plan("always", ["0000:40:00.0", "0000:40:00.0", "0000:1a:00.0"], HOST, root)
    assert _cpus(plan) == [3, 9, 0] and plan.result == 30 and _one_cpu_a_core(plan)


def test_a_mask_off_the_gpus_node_falls_back_to_its_rank_among_the_gpus_off_theirs(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("always", ["0000:40:00.0"], parse_cpu_list("32-63"), root)
    # node 1's two GPUs would take 32-37, 62 and 63; six GPUs have no allowed
    # CPU on their node, and the second of them takes from 38-61
    assert plan.workers[0].cpus == {41, 42, 43} and plan.result == 60
    assert "no allowed CPU on its NUMA node, so GPU 2 of 6 without one" in plan.workers[0].why
    # a device sysfs doesn't place (a CPU worker) takes the mask from the bottom
    plan = _plan("always", [None, None], parse_cpu_list("32-39"), root)
    assert _cpus(plan) == [32, 35] and plan.result == 39


def test_always_pins_nothing_when_the_mask_is_too_small(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("always", [None], {32, 33, 34, 35}, root)   # a worker and the API server would take all four
    assert plan.workers == () and plan.result is None
    assert plan.why.startswith("not pinned: ") and "leaving none for the rest" in plan.why
    plan = _plan("always", [None] * 3, parse_cpu_list("32-38"), root)
    assert plan.pinned == frozenset() and "more cores than the mask has" in plan.why


@pytest.mark.parametrize(("mask", "address", "gpus"), [
    # node 1's GPUs take all of 32-39 and leave nothing for the 6 others
    (parse_cpu_list("32-39"), "0000:1a:00.0", 6),
    (parse_cpu_list("32-39"), "0000:9a:00.0", 6),
    (parse_cpu_list("32-39,160-167"), "0000:40:00.0", 6),   # a core's sibling is the same core
    (parse_cpu_list("32-38"), "0000:62:00.0", 2),           # on their node: 7 cores for 2 GPUs
    (parse_cpu_list("32-38"), "0000:73:00.0", 2),
])
def test_always_pins_nothing_with_fewer_than_four_cores_per_gpu_in_a_pool(tmp_path, mask, address, gpus):
    plan = _plan("always", [address], mask, _sysfs(tmp_path))
    assert plan.pinned == frozenset() and plan.why.startswith("not pinned: ")
    assert f"for {gpus} GPU(s), and each needs 4 (3 for its worker's step threads, 1 for its API server)" in plan.why


def test_four_cores_per_gpu_is_enough(tmp_path):
    root = _sysfs(tmp_path)
    assert _plan("always", ["0000:62:00.0"], parse_cpu_list("32-39"), root).pinned == {32, 33, 34, 39}
    assert _plan("always", ["0000:73:00.0"], parse_cpu_list("32-39"), root).pinned == {35, 36, 37, 38}
    # 24 cores left for the 6 GPUs off their nodes
    assert _plan("always", ["0000:1a:00.0"], parse_cpu_list("32-63"), root).pinned == {38, 39, 40, 61}
    assert _plan("always", ["0000:fa:00.0"], parse_cpu_list("32-63"), root).pinned == {53, 54, 55, 56}


@pytest.mark.parametrize("mask", [
    HOST, NODES[1], parse_cpu_list("32-39"), parse_cpu_list("32-63"), parse_cpu_list("32-35"),
    parse_cpu_list("32-47,96-111"), parse_cpu_list("0-3,32-63"), parse_cpu_list("0-2,32-63"),
])
@pytest.mark.parametrize("setting", ["always", "siblings"])
def test_no_two_gpus_of_the_host_share_a_core_under_one_mask(tmp_path, mask, setting):
    """Whichever GPU a server gets, and whichever others are in use."""
    root = _sysfs(tmp_path)
    plans = [_plan(setting, [address], mask, root) for address in GPUS]
    assert any(plan.pinned for plan in plans) or len(mask) < 8
    for i, one in enumerate(plans):
        for other in plans[i + 1:]:
            assert not {smt(c) for c in one.pinned} & {smt(c) for c in other.pinned}


def test_workers_that_outgrow_their_pool_pin_nothing_rather_than_wrap(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-47")
    # GPU 1 of 2 on the node: 32-34, then 38-40; 35-37 and 41-43 are the other GPU's
    assert _cpus(_plan("always", ["0000:62:00.0"] * 2, mask, root)) == [32, 38]
    plan = _plan("always", ["0000:62:00.0"] * 3, mask, root)
    assert plan.pinned == frozenset() and "more cores than the mask has" in plan.why


def test_always_leaves_the_siblings_of_its_cpus_unpinned(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("always", [None], parse_cpu_list("32-39,160-167"), root)
    assert plan.workers[0].cpus == {32, 33, 34} and plan.result == 39
    assert not plan.pinned & parse_cpu_list("160-167") and plan.left == 12
    # a mask without the siblings gives the same CPUs: no core is short of a partner
    assert _plan("always", [None], parse_cpu_list("32-39"), root).pinned == plan.pinned


# ── siblings: a worker on one core's hardware threads ───────────────────────

def test_siblings_puts_a_worker_on_both_hardware_threads_of_its_core(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-35,160-163")
    plan = _plan("siblings", [None, None], mask, root)
    assert [w.cpus for w in plan.workers] == [{32, 160}, {33, 161}] and plan.result == 35
    assert plan.pinned == {32, 33, 35, 160, 161} and plan.left == 3
    # a mask with one hardware thread of the core: that one
    assert [w.cpus for w in _plan("siblings", [None], parse_cpu_list("32-35"), root).workers] == [{32}]
    # by the GPU's rank, as always places
    plan = _plan("siblings", ["0000:40:00.0"], HOST, root)
    assert plan.workers[0].cpus == {1, 129} and plan.result == 30 and plan.why.startswith("siblings: GPU 2 of 2")


def test_siblings_needs_two_cores_per_gpu(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("siblings", ["0000:62:00.0"], parse_cpu_list("32-34,160-162"), root)
    assert plan.pinned == frozenset() and "3 core(s) there for 2 GPU(s), and each needs 2, one for" in plan.why
    assert _plan("siblings", ["0000:73:00.0"], parse_cpu_list("32-35,160-163"), root).pinned == {33, 161, 34}


def test_siblings_is_never_what_auto_or_always_take(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-39,160-167")
    for setting in ("auto", "always"):
        plan = _plan(setting, [POD[0]], mask, root, reachable=POD)
        assert all(len(w.cpus) == cpus.STEP_CORES for w in plan.workers) and _one_cpu_a_core(plan)
    assert _plan("siblings", [POD[0]], mask, root, reachable=POD).workers[0].cpus == {32, 160}


# ── a container given a few of the host's GPUs ──────────────────────────────

POD = ("0000:40:00.0", "0000:fa:00.0")     # both off the node of the mask below


@pytest.mark.parametrize("address", POD)
def test_a_mask_too_small_for_the_host_fits_the_gpus_a_container_can_open(tmp_path, address):
    """Eight cores can't give each of the host's GPUs four, but this container
    has two GPUs, so nothing else can take a core from them."""
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-39")
    assert _plan("always", [address], mask, root).pinned == frozenset()
    plan = _plan("always", [address], mask, root, reachable=POD)
    assert plan.pinned and "of 2 without one" in plan.workers[0].why


def test_the_gpus_of_one_container_stay_apart_under_one_small_mask(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-39")
    first, second = (_plan("always", [a], mask, root, reachable=POD) for a in POD)
    assert first.pinned == {32, 33, 34, 39} and second.pinned == {35, 36, 37, 38}
    # the same mask with the siblings: the same CPUs, the siblings left alone
    mask = parse_cpu_list("32-39,160-167")
    first, second = (_plan("always", [a], mask, root, reachable=POD) for a in POD)
    assert first.workers[0].cpus == {32, 33, 34} and first.result == 39
    assert second.workers[0].cpus == {35, 36, 37} and second.result == 38
    # and on the GPUs' own node
    mask = parse_cpu_list("0-7,128-135")
    first = _plan("always", ["0000:40:00.0"], mask, root, reachable=("0000:1a:00.0", "0000:40:00.0"))
    assert first.workers[0].cpus == {3, 4, 5} and first.result == 6


def test_a_mask_that_fits_the_host_places_as_before_whatever_the_container_can_open(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-63")
    assert _plan("always", ["0000:40:00.0"], mask, root, reachable=POD).pinned == {41, 42, 43, 60}
    assert _plan("always", ["0000:40:00.0"], mask, root).pinned == {41, 42, 43, 60}


def test_a_mask_too_small_for_the_container_too_says_what_it_needs(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("always", [POD[0]], parse_cpu_list("32-38"), root, reachable=POD)
    assert plan.pinned == frozenset() and "for 2 GPU(s), and each needs 4" in plan.why
    assert "widen the mask" in plan.why
    # every GPU reachable (bare metal): the host's rule, unchanged
    plan = _plan("always", [POD[0]], parse_cpu_list("32-39"), root, reachable=tuple(GPUS))
    assert plan.pinned == frozenset() and "for 6 GPU(s), and each needs 4" in plan.why


# ── a host without SMT ──────────────────────────────────────────────────────

# 2x96 E-cores, four to an L2 cluster, no SMT, four GPUs a socket
ECORE_NODES = [parse_cpu_list("0-95"), parse_cpu_list("96-191")]
ECORE_GPUS = {"0000:1a:00.0": 0, "0000:5d:00.0": 0, "0000:76:00.0": 0, "0000:7d:00.0": 0,
              "0000:9a:00.0": 1, "0000:da:00.0": 1, "0000:f3:00.0": 1, "0000:fa:00.0": 1}
ECORE_POD = ("0000:f3:00.0", "0000:fa:00.0")


def no_smt(cpu: int) -> frozenset[int]:
    return frozenset({cpu})


def four_to_a_cluster(cpu: int) -> frozenset[int]:
    return frozenset(range(cpu - cpu % 4, cpu - cpu % 4 + 4))


def _ecore(setting, address, mask, tmp_path, reachable=ECORE_POD):
    root = tmp_path / "pci"
    if not root.exists():
        _sysfs(tmp_path, gpus=ECORE_GPUS, nodes=ECORE_NODES)
    return _plan(setting, [address], mask, root, reachable=reachable, siblings=no_smt, cluster=four_to_a_cluster)


def test_without_smt_always_takes_three_cores_too(tmp_path):
    mask = parse_cpu_list("96-103")
    first, second = (_ecore("always", a, mask, tmp_path) for a in ECORE_POD)
    assert first.workers[0].cpus == {96, 97, 98} and first.result == 103
    assert second.workers[0].cpus == {99, 100, 101} and second.result == 102


def test_siblings_without_smt_takes_two_cores_of_one_cluster(tmp_path):
    mask = parse_cpu_list("96-103")
    first = _ecore("siblings", ECORE_POD[0], mask, tmp_path)
    assert first.workers[0].cpus == {96, 97} and first.result == 103
    assert four_to_a_cluster(96) >= first.workers[0].cpus
    second = _ecore("siblings", ECORE_POD[1], mask, tmp_path)
    assert second.workers[0].cpus == {98, 99} and second.result == 102
    assert not first.pinned & second.pinned


def test_pairs_stay_inside_a_cluster(tmp_path):
    cores = [(c,) for c in (98, 99, 100, 101, 102)]
    assert cpus._paired(cores, four_to_a_cluster) == [(98, 99), (100, 101), (102,)]
    # two hardware threads already make the pair; a lone core keeps its one CPU
    assert cpus._paired([(1, 129), (2,)], four_to_a_cluster) == [(1, 129), (2,)]


def test_siblings_on_a_core_short_of_its_sibling_and_no_cluster_stays_on_one_cpu(tmp_path):
    root = _sysfs(tmp_path)
    plan = _plan("siblings", [POD[0]], parse_cpu_list("32-39"), root, reachable=POD)
    assert [w.cpus for w in plan.workers] == [{32}] and plan.result == 39


def test_bare_metal_counts_cores_against_every_gpu_of_the_node(tmp_path):
    plan = _ecore("always", ECORE_POD[0], parse_cpu_list("96-103"), tmp_path, reachable=tuple(ECORE_GPUS))
    assert plan.pinned == frozenset() and "8 core(s) there for 4 GPU(s), and each needs 4" in plan.why
    assert _ecore("always", ECORE_POD[0], parse_cpu_list("96-111"), tmp_path, reachable=tuple(ECORE_GPUS)).pinned


def test_a_cpu_number_keeps_to_one_core_without_smt(tmp_path):
    root = _sysfs(tmp_path, gpus=ECORE_GPUS, nodes=ECORE_NODES)
    plan = _plan(98, [None], parse_cpu_list("96-103"), root, siblings=no_smt, cluster=four_to_a_cluster)
    assert plan.workers[0].cpus == {98} and plan.result == 103


# ── every shipped config ────────────────────────────────────────────────────

SHIPPED = sorted((REPO / "configs").glob("*.yaml"))


def _shipped_addresses(path: Path, gpus) -> list[str]:
    config = yaml.safe_load(path.read_text())
    host = list(gpus)
    return [host[int(device.partition(":")[2] or 0) % len(host)]
            for device in deployment_workers(config, "cuda").devices]


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
@pytest.mark.parametrize("setting", ["always", "siblings"])
def test_a_shipped_config_gets_a_sound_plan_on_each_host(path, tmp_path, setting):
    """No config may stop a server or put two threads' CPUs in one place, on a
    host with SMT or without, under a wide or a tight mask; under ``always`` no
    two pinned CPUs share a core either."""
    addresses = _shipped_addresses(path, GPUS)
    smt_root = _sysfs(tmp_path)
    (tmp_path / "e").mkdir()
    ecore_root = _sysfs(tmp_path / "e", gpus=ECORE_GPUS, nodes=ECORE_NODES)
    cases = [(HOST, smt_root, smt, one_cpu_cluster, addresses),
             (parse_cpu_list("32-39,160-167"), smt_root, smt, one_cpu_cluster, addresses),
             (parse_cpu_list("0-191"), ecore_root, no_smt, four_to_a_cluster, _shipped_addresses(path, ECORE_GPUS)),
             (parse_cpu_list("96-103"), ecore_root, no_smt, four_to_a_cluster, _shipped_addresses(path, ECORE_GPUS))]
    for mask, root, siblings, cluster, placed in cases:
        plan = _plan(setting, placed, mask, root, siblings=siblings, cluster=cluster)
        if not plan.pinned:
            assert plan.why.startswith("not pinned: ")
            continue
        steps = [w.cpus for w in plan.workers]
        assert len(steps) == len(placed)
        assert sum(map(len, steps)) == len(frozenset().union(*steps)), "two workers share a CPU"
        assert plan.result not in frozenset().union(*steps)
        assert plan.pinned <= mask and len(plan.pinned) < len(mask)
        if setting == "always":
            assert len({siblings(c) for c in plan.pinned}) == len(plan.pinned), "two pinned CPUs share a core"
    # auto pins none of them
    assert not _plan("auto", addresses, HOST, smt_root).pinned


# ── auto pins nothing, and says why ─────────────────────────────────────────

DEEP_INTEL = "on an Intel host whose cores sleep deep, the step threads on 3 cores of their own measured no faster"


@pytest.mark.parametrize("states", [XEON, SIERRA, [("C1", 1, False), ("C6", 100, False)]])
def test_auto_on_a_deep_idle_intel_host_says_what_was_measured_there(tmp_path, states):
    mask = parse_cpu_list("32-47,160-175")
    root, idle = _sysfs(tmp_path), _idle(tmp_path, mask, states)
    plan = _plan("auto", ["0000:62:00.0"], mask, root, idle=idle)
    assert plan.pinned == frozenset() and plan.workers == () and plan.declined and plan.left == len(mask)
    assert plan.why.startswith("not pinned: " + DEEP_INTEL)
    assert "step_thread_core siblings" in plan.why and plan.why.endswith("(step_thread_core always pins anyway)")
    # always and siblings pin that host
    assert _plan("always", ["0000:62:00.0"], mask, root, idle=idle).pinned == {32, 33, 34, 47}
    assert _plan("siblings", ["0000:62:00.0"], mask, root, idle=idle).pinned == {32, 160, 47}


@pytest.mark.parametrize(("states", "why"), [
    (VM, "the deepest idle state is C1 at 2 µs"),
    ([("C1", 1, False), ("C6", 99, False)], "the deepest idle state is C6 at 99 µs"),
    ([("C1E", 2, False), ("C6", 170, True), ("C6P", 210, True)], "the deepest idle state is C1E at 2 µs"),  # disabled
    ([], "sysfs lists no idle state"),
])
def test_auto_on_a_host_without_deep_idle_says_so(tmp_path, states, why):
    mask = parse_cpu_list("32-47")
    root, idle = _sysfs(tmp_path), _idle(tmp_path, mask, states)
    plan = _plan("auto", ["0000:62:00.0"], mask, root, idle=idle)
    assert plan.pinned == frozenset() and plan.workers == () and plan.declined
    assert plan.why.startswith("not pinned: " + why)
    assert "on CPUs 32-47" in plan.why and "always pins anyway" in plan.why
    assert plan.left == len(mask)
    assert _plan("always", ["0000:62:00.0"], mask, root, idle=idle).pinned == {32, 33, 34, 47}


def test_auto_only_reads_the_servers_cpus(tmp_path):
    root, idle = _sysfs(tmp_path), _idle(tmp_path, range(0, 32), XEON)
    assert "the deepest idle state" not in _plan("auto", ["0000:62:00.0"], parse_cpu_list("31-47,160-175"), root,
                                                 idle=idle).why
    assert "sysfs lists no idle state" in _plan("auto", ["0000:62:00.0"], parse_cpu_list("32-47,160-175"), root,
                                                idle=idle).why


@pytest.mark.parametrize(("states", "vendor"), [(VM, INTEL), (XEON, AMD), (XEON, INTEL)])
def test_auto_never_asks_for_device_addresses(tmp_path, monkeypatch, states, vendor):
    def boom(devices):
        raise AssertionError("auto asked torch about its devices")
    monkeypatch.setattr(cpus, "device_addresses", boom)
    monkeypatch.setattr(cpus, "_CPU_ROOT", _idle(tmp_path, cpus.process_mask(), states))
    monkeypatch.setattr(cpus, "_CPUINFO", _cpuinfo(tmp_path, vendor))
    # deployment_plan's own default is the real sysfs, so read the fake one there too
    real = deployment_plan
    monkeypatch.setattr(cpus, "deployment_plan", lambda s, a, **kw: real(s, a, cpu_root=cpus._CPU_ROOT, **kw))
    plan = cpus.devices_plan("auto", ["cuda:0"])
    assert plan.declined and plan.pinned == frozenset()


def test_one_cpu_with_a_deep_state_is_enough(tmp_path):
    mask = parse_cpu_list("32-47,160-175")
    root, idle = _sysfs(tmp_path), _idle(tmp_path, range(32, 47), VM)
    _idle(tmp_path, [47], XEON)
    assert DEEP_INTEL in _plan("auto", ["0000:62:00.0"], mask, root, idle=idle).why


def test_the_real_sysfs_reader_tolerates_a_host_without_cpuidle(tmp_path):
    assert cpus.deepest_idle([0, 1], tmp_path / "nothing") is None
    root = _idle(tmp_path, [0], [("C6", 170, False)])
    (root / "cpu0" / "cpuidle" / "state0" / "latency").write_text("not a number\n")
    assert cpus.deepest_idle([0], root) is None


@pytest.mark.parametrize(("text", "this"), [(AMD, "this host's is AuthenticAMD"),
                                            (ARM, "/proc/cpuinfo names no vendor here")])
def test_auto_on_a_cpu_vendor_it_is_not_measured_on_says_so(tmp_path, monkeypatch, text, this):
    mask = parse_cpu_list("32-39,160-167")
    root = _sysfs(tmp_path)
    monkeypatch.setattr(cpus, "_CPUINFO", _cpuinfo(tmp_path, text))
    # deep idle states and room for every thread: the vendor is what the line names
    plan = _plan("auto", [POD[0]], mask, root, reachable=POD)
    assert plan.pinned == frozenset() and plan.workers == () and plan.declined
    assert plan.why == (f"not pinned: auto is measured on Intel hosts only and {this} "
                        "(step_thread_core always pins anyway)")
    assert plan.left == len(mask)
    # always, siblings and a number place as they do on Intel
    assert _plan("always", [POD[0]], mask, root, reachable=POD).pinned == {32, 33, 34, 39}
    assert _plan("siblings", [POD[0]], mask, root, reachable=POD).pinned == {32, 160, 39}
    assert _plan(33, [POD[0]], mask, root).workers[0].cpus == {33, 161}


def test_auto_under_an_idle_hold_says_so_first(tmp_path, monkeypatch):
    mask = parse_cpu_list("32-39,160-167")
    root = _sysfs(tmp_path)
    for vendor in (INTEL, AMD):
        monkeypatch.setattr(cpus, "_CPUINFO", _cpuinfo(tmp_path, vendor))
        plan = _plan("auto", [POD[0]], mask, root, reachable=POD, idle_hold="qos")
        assert plan.pinned == frozenset() and plan.workers == () and plan.declined
        assert plan.why == ("not pinned: idle_hold qos keeps the cores out of deep idle, where pinning measured no "
                            "faster than none (step_thread_core always pins anyway)")
    # always pins under the hold
    assert _plan("always", [POD[0]], mask, root, reachable=POD, idle_hold="qos").pinned == {32, 33, 34, 39}


def test_the_api_server_and_its_workers_read_the_idle_hold(api_plan, monkeypatch, gpu_locks, caplog):
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    assert DEEP_INTEL in api_plan().why
    monkeypatch.setenv("MSTAR_IDLE_HOLD", "qos")
    plan = api_plan()
    assert plan.pinned == frozenset() and plan.declined and "idle_hold qos" in plan.why
    assert cpus._gpu_locks == {}, "a plan that pins nothing takes no lock"
    worker = _bare_worker("worker_0", "cuda:0")
    worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    assert worker._step_cpus is None


def test_the_vendor_is_the_first_processors(tmp_path):
    assert cpus.cpu_vendor(_cpuinfo(tmp_path, INTEL)) == "GenuineIntel"
    assert cpus.cpu_vendor(_cpuinfo(tmp_path, AMD)) == "AuthenticAMD"
    assert cpus.cpu_vendor(_cpuinfo(tmp_path, ARM)) is None
    assert cpus.cpu_vendor(tmp_path / "absent") is None


# ── the PM QoS hold ─────────────────────────────────────────────────────────

@pytest.fixture
def idle_hold():
    yield
    if cpus._idle_hold_fd is not None:
        os.close(cpus._idle_hold_fd)
    cpus._idle_hold_fd = None


def test_the_hold_writes_the_latency_and_keeps_the_fd_open(tmp_path, idle_hold):
    device = tmp_path / "cpu_dma_latency"
    device.write_bytes(b"")
    said = cpus.hold_idle(device)
    assert int.from_bytes(device.read_bytes(), sys.byteorder) == cpus.IDLE_HOLD_US == 10
    assert len(device.read_bytes()) == 4, "the device takes a 32-bit value"
    assert f"holding {device} at 10 µs" in said
    fd = cpus._idle_hold_fd
    assert os.readlink(f"/proc/self/fd/{fd}") == str(device), "the request lasts as long as the fd"
    assert "already holding" in cpus.hold_idle(device) and cpus._idle_hold_fd == fd


def test_the_hold_says_when_the_device_is_absent(tmp_path, idle_hold):
    device = tmp_path / "cpu_dma_latency"
    said = cpus.hold_idle(device)
    assert "nothing held" in said and "is absent" in said and cpus._idle_hold_fd is None


def test_the_hold_says_when_the_device_is_not_writable(tmp_path, idle_hold, monkeypatch):
    def refuse(*args):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(cpus.os, "open", refuse)
    said = cpus.hold_idle(tmp_path / "cpu_dma_latency")
    assert "nothing held" in said and "Permission denied" in said and cpus._idle_hold_fd is None


def test_a_refused_write_closes_the_fd(tmp_path, idle_hold, monkeypatch):
    device = tmp_path / "cpu_dma_latency"
    device.write_bytes(b"")
    closed = []
    real_close = os.close

    def refuse(*args):
        raise OSError(22, "Invalid argument")
    monkeypatch.setattr(cpus.os, "write", refuse)
    monkeypatch.setattr(cpus.os, "close", lambda fd: (closed.append(fd), real_close(fd))[1])
    said = cpus.hold_idle(device)
    assert "nothing held" in said and "Invalid argument" in said and cpus._idle_hold_fd is None and closed


# ── a CPU number, and off ───────────────────────────────────────────────────

def test_a_number_is_the_first_workers_cpu_and_the_api_takes_the_last_core(tmp_path):
    root = _sysfs(tmp_path)
    mask = parse_cpu_list("32-39")
    plan = _plan(34, ["0000:40:00.0", "0000:40:00.0"], mask, root)
    assert _cpus(plan) == [34, 35] and plan.result == 39
    # the number names a core: the worker takes its hardware threads in the mask
    assert _plan(160, [None], HOST, root).workers[0].cpus == {32, 160}


@pytest.mark.parametrize(("setting", "n", "message"), [
    (8, 1, "step_thread_core 8 is outside the process mask 32-39"),
    (38, 2, "step_thread_core 38 leaves 1 core(s) at or above it for 2 worker(s) in the process mask 32-39 "
            "(the last, 39, is the API server's)"),
    (39, 1, "leaves 0 core(s)"),
])
def test_a_number_the_mask_cannot_honour_fails_at_startup(tmp_path, setting, n, message):
    with pytest.raises(CpuPlanError, match=message.replace("(", r"\(").replace(")", r"\)")):
        _plan(setting, [None] * n, parse_cpu_list("32-39"), _sysfs(tmp_path))


def test_off_pins_nothing_and_the_pools_get_every_usable_cpu(tmp_path):
    plan = _plan("off", ["0000:40:00.0"], parse_cpu_list("32-39"), _sysfs(tmp_path))
    assert plan == CpuPlan(parse_cpu_list("32-39"), 8, why="step_thread_core off")
    assert plan.left == 8


# ── pool sizes ──────────────────────────────────────────────────────────────

def test_the_pools_are_sized_from_the_cpus_left_after_the_pins():
    mask = parse_cpu_list("32-47")
    free = CpuPlan(mask, usable=16)
    pinned = CpuPlan(mask, usable=16, workers=(_one(32), _one(33)), result=47)
    assert (free.left, free.preprocess_threads, free.png_threads, free.encode_workers) == (16, 4, 8, 2)
    assert (pinned.left, pinned.preprocess_threads, pinned.png_threads, pinned.encode_workers) == (13, 3, 8, 1)
    # the cgroup quota caps the count; a pinned CPU outside the mask isn't taken from it
    assert CpuPlan(mask, usable=6, workers=(_one(32),), result=99).left == 5


def test_the_api_server_reads_the_deployments_plan_from_its_config(monkeypatch):
    from mstar.api_server import entrypoint

    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")

    seen = {}

    def plan(setting, addresses, **kw):
        seen.update(setting=setting, addresses=addresses)
        return CpuPlan(frozenset(range(8)), 8)

    monkeypatch.setattr(entrypoint.cpus, "deployment_plan", plan)
    monkeypatch.setattr(entrypoint.cpus, "device_addresses", lambda devices: [f"at {d}" for d in devices])
    monkeypatch.delenv("MSTAR_STEP_THREAD_CORE", raising=False)
    config = {"knobs": {"step_thread_core": "always"}, "rank_devices": {2: 0},
              "node_groups": [{"node_names": ["a"], "ranks": [2]}, {"node_names": ["b"], "ranks": [0, 1]}]}
    entrypoint.deployment_cpu_plan(config)
    # ranks in order, rank 2 placed on device 0 as the conductor places it
    assert seen == {"setting": "always", "addresses": ["at cuda:0", "at cuda:1", "at cuda:0"]}
    entrypoint.deployment_cpu_plan({**config, "knobs": {"step_thread_core": 33}})
    assert seen == {"setting": 33, "addresses": [None, None, None]}
    entrypoint.knobs.configure(None)


def test_device_addresses_read_the_worker_device_strings(monkeypatch):
    from mstar import platforms

    monkeypatch.setattr(platforms.current_platform, "pci_address", lambda index: f"gpu {index}", raising=False)
    assert cpus.device_addresses(["cuda:0", "xpu:3", "cuda", "cpu"]) == ["gpu 0", "gpu 3", None, None]


# ── one list of workers ─────────────────────────────────────────────────────

def _shipped(name) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text())


def test_deployment_workers_skips_what_the_conductor_reports():
    groups = [3, {"ranks": "x"}, {"ranks": [2, "a", 0]}, {"node_names": []}]
    assert deployment_workers({"node_groups": groups}, "cuda") == ([0, 2], ["cuda:0", "cuda:2"])
    assert deployment_workers({"node_groups": "x"}, "cuda") == ([], [])
    assert deployment_workers({}, "cuda") == ([], [])
    assert deployment_workers({"node_groups": [{"ranks": [1]}], "rank_devices": {1: 0}}, "cpu").devices == ["cpu"]


@pytest.mark.parametrize(("name", "ranks", "devices"), [
    ("qwen3tts_1p7b_split.yaml", [0, 1], ["cuda:0", "cuda:0"]),                  # a split config
    ("qwen3_5_27b_tp4.yaml", [0, 1, 2, 3], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]),  # TP
])
def test_the_api_server_and_the_conductors_workers_build_one_plan(
    name, ranks, devices, tmp_path, monkeypatch, gpu_locks,
):
    """From the shipped config: the real conductor's worker list and what it
    hands each worker, the real API server plan and the real worker plan. No
    worker's step core holds the API server's result CPU."""
    from test_startup_graph_validation import _llm, _start_conductor, _ToyModel

    from mstar.api_server import entrypoint
    from mstar.conductor import conductor as conductor_mod
    from mstar.conductor.conductor import Conductor

    config = _shipped(name)
    groups = config["node_groups"]
    nodes = {node: group.get("graph_walks") or [f"{node} walk"] for group in groups for node in group["node_names"]}
    model = _ToyModel(lambda: {walk: _llm(node) for node, walks in nodes.items() for walk in walks},
                      tp_enabled_nodes=[node for group in groups if group.get("tp_size", 1) > 1
                                        for node in group["node_names"]])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(
        {"max_seq_len": 64, "node_groups": groups, "rank_devices": config.get("rank_devices")}))

    launch = Conductor._launch_workers
    monkeypatch.setattr(conductor_mod, "accelerator_type", lambda: "cuda")
    conductor = _start_conductor(model, str(path), tmp_path, monkeypatch)
    assert conductor._sorted_ranks == ranks and conductor._worker_devices == devices
    # every rank has graphs: the list from the config is what the walks use
    assert ranks == sorted({rank for wg in conductor.worker_graphs.values() for rank in wg.ranks})

    spawned = []

    class Process:
        def __init__(self, target, kwargs, daemon):
            spawned.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(
        conductor_mod, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace(Process=Process)))
    monkeypatch.setattr(conductor_mod, "atexit", SimpleNamespace(register=lambda fn: None))
    launch(conductor)
    assert [kw["device"] for kw in spawned] == devices

    root = _sysfs(tmp_path)
    at = {f"cuda:{i}": address for i, address in enumerate(list(GPUS)[:4])}
    plan = deployment_plan

    def placed(setting, addresses, **kw):
        return plan(setting, addresses, mask=HOST, usable=len(HOST), pci_root=root, siblings=smt,
                    cluster=one_cpu_cluster, cpu_root=_idle(root.parent / "deep", [0], XEON), **kw)

    monkeypatch.setattr(cpus, "device_addresses", lambda ds: [at[d] for d in ds])
    monkeypatch.setattr(cpus, "deployment_plan", placed)
    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "always")
    api = entrypoint.deployment_cpu_plan(config)
    assert len(api.workers) == len(ranks) and api.result is not None
    assert all(api.result not in step.cpus for step in api.workers)
    for kw, step in zip(spawned, api.workers, strict=True):
        worker = _bare_worker(kw["worker_id"], kw["device"])
        worker._plan_step_cpus(kw["worker_ids"], kw["local_worker_devices"])
        assert cpus.plan() == api and worker._step_cpus == step
    cpus.install(None)
    entrypoint.knobs.configure(None)


# ── one server per GPU ──────────────────────────────────────────────────────

# what a second server does: take the locks, say how that went, stay up until told to go
_OTHER_SERVER = """
import sys
from pathlib import Path
from mstar.utils import cpus
cpus._GPU_LOCK_DIR = Path(sys.argv[1])
print(cpus.claim_gpus(sys.argv[2:]), flush=True)
sys.stdin.read()
"""


@pytest.fixture
def gpu_locks(tmp_path, monkeypatch):
    """Locks in a directory of the test's, held by this process until it ends."""
    held: dict[str, int] = {}
    cores: dict[Path, int] = {}
    monkeypatch.setattr(cpus, "_GPU_LOCK_DIR", tmp_path)
    monkeypatch.setattr(cpus, "_gpu_locks", held)
    monkeypatch.setattr(cpus, "_core_locks", cores)
    yield tmp_path
    for fd in [*held.values(), *cores.values()]:
        os.close(fd)


@pytest.fixture
def other_server(gpu_locks):
    """Starts a process that claims the GPUs it's given; gives back the process
    and what its claim returned."""
    started = []

    def start(*addresses):
        p = subprocess.Popen(
            [sys.executable, "-c", _OTHER_SERVER, str(gpu_locks), *addresses],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        started.append(p)
        return p, p.stdout.readline().strip()

    yield start
    for p in started:
        p.kill()
        p.communicate()


A, B = "0000:40:00.0", "0000:fa:00.0"


def test_a_second_server_on_a_gpu_is_refused_until_the_first_exits(other_server):
    first, claimed = other_server(A)
    assert claimed == "None"
    refused = cpus.claim_gpus([A])
    assert f"another server holds GPU {A}" in refused and cpus._gpu_locks == {}
    # another GPU is free, and so is this one once the first has exited
    assert cpus.claim_gpus([B]) is None
    first.communicate("")
    assert first.returncode == 0
    assert cpus.claim_gpus([A]) is None


def test_a_killed_server_does_not_block_the_next(other_server):
    first, claimed = other_server(A)
    assert claimed == "None"
    first.kill()
    first.communicate()
    assert cpus.claim_gpus([A]) is None
    # and now it's this process that holds it
    assert "another server holds" in other_server(A)[1]


def test_a_refused_claim_holds_nothing(other_server):
    assert other_server(B)[1] == "None"
    assert f"another server holds GPU {B}" in cpus.claim_gpus([A, B])
    assert cpus._gpu_locks == {}
    assert other_server(A)[1] == "None"   # the lock on A was let go


def test_one_server_claims_its_gpus_once(gpu_locks):
    assert cpus.claim_gpus([A, None, A]) is None   # a split config, and a device sysfs doesn't place
    fd = cpus._gpu_locks[A]
    assert cpus.claim_gpus([A]) is None and cpus._gpu_locks == {A: fd}
    assert [path.name for path in gpu_locks.iterdir()] == [f"mstar-gpu-{A}.lock"]
    # a child doesn't inherit the fd, so it can't keep a dead server's lock
    assert not os.get_inheritable(fd)


def test_a_lock_that_cannot_be_opened_is_a_refusal(gpu_locks, monkeypatch):
    monkeypatch.setattr(cpus, "_GPU_LOCK_DIR", gpu_locks / "absent")
    assert f"can't lock GPU {A}" in cpus.claim_gpus([A])


def test_an_existing_lock_file_of_another_user_still_locks(gpu_locks, monkeypatch):
    real_open = os.open

    def refuse_create(path, flags, *mode):
        # the kernel answers O_EXCL on an existing file before the sticky check
        if flags & os.O_CREAT and not flags & os.O_EXCL:
            raise PermissionError("protected_regular")
        return real_open(path, flags, *mode)

    (gpu_locks / f"mstar-gpu-{A}.lock").touch()
    monkeypatch.setattr(cpus.os, "open", refuse_create)
    assert cpus.claim_gpus([A]) is None


def test_the_lock_file_is_readable_by_other_users_whatever_the_umask(gpu_locks):
    umask = os.umask(0o077)
    try:
        assert cpus.claim_gpus([A]) is None
    finally:
        os.umask(umask)
    mode = (gpu_locks / f"mstar-gpu-{A}.lock").stat().st_mode & 0o777
    assert mode == 0o644, "other users' servers couldn't open the lock, so none of them would pin on this GPU"


@pytest.fixture
def api_plan(tmp_path, monkeypatch):
    """``entrypoint.deployment_cpu_plan`` for a deployment on GPU ``A``, on the
    fake host, from a mask of 32 cores and their siblings."""
    from mstar.api_server import entrypoint

    root = _sysfs(tmp_path)
    plan = deployment_plan

    def placed(setting, addresses, **kw):
        return plan(setting, addresses, mask=parse_cpu_list("32-63,160-191"), usable=64, pci_root=root, siblings=smt,
                    cluster=one_cpu_cluster, cpu_root=_idle(root.parent / "deep", [32], XEON), **kw)

    monkeypatch.setattr(cpus, "device_addresses", lambda devices: [A] * len(devices))
    monkeypatch.setattr(cpus, "deployment_plan", placed)
    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")
    config = {"node_groups": [{"node_names": ["llm"], "ranks": [0]}]}
    # back to this after the test: the refusal below writes the knob's env var
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "always")
    yield lambda: entrypoint.deployment_cpu_plan(config)
    cpus.install(None)
    entrypoint.knobs.configure(None)


def test_the_first_server_pins_and_the_second_pins_nothing_with_a_warning(other_server, api_plan, caplog):
    other, claimed = other_server(A)
    assert claimed == "None"
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        second = api_plan()
    assert second.pinned == frozenset() and f"not pinned: another server holds GPU {A}" in second.why
    assert f"nothing pinned, another server holds GPU {A}" in caplog.text
    # the pools get the whole mask
    assert second.left == 64
    # its workers read the knob, so they pin nothing either
    worker = _bare_worker("worker_0", "cuda:0")
    worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    assert worker._step_cpus is None and cpus.plan().pinned == frozenset()

    other.communicate("")
    os.environ["MSTAR_STEP_THREAD_CORE"] = "always"
    first = api_plan()
    assert first.pinned == {41, 42, 43, 60} and A in cpus._gpu_locks
    worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    assert worker._step_cpus.cpus == {41, 42, 43}


def test_always_takes_the_lock_too(other_server, api_plan):
    other, claimed = other_server(A)
    assert claimed == "None"
    os.environ["MSTAR_STEP_THREAD_CORE"] = "always"
    assert api_plan().pinned == frozenset()
    assert os.environ["MSTAR_STEP_THREAD_CORE"] == "off", "its workers read the knob"
    other.communicate("")


def test_a_host_without_deep_idle_says_so_in_one_line(monkeypatch, caplog, tmp_path):
    """No warning: nothing is wrong with the deployment, and the workers stay quiet."""
    from api_server_helpers import make_test_api_server

    mask = parse_cpu_list("32-39")
    shallow = CpuPlan(mask, 8, why="not pinned: sysfs lists no idle state", declined=True)
    monkeypatch.setattr(cpus, "_plan", shallow)
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    with caplog.at_level(logging.INFO):
        make_test_api_server()._take_result_cpu()
        worker = _bare_worker("worker_0", "cuda:0")
        monkeypatch.setattr(cpus, "devices_plan", lambda setting, devices, idle_hold: shallow)
        worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    lines = [r for r in caplog.records if "not pinned" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "sysfs lists no idle state" in lines[0].getMessage()


def test_only_a_placed_plan_that_pins_takes_the_lock(api_plan, monkeypatch, gpu_locks):
    for setting in ("off", "auto"):
        monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", setting)
        assert api_plan().pinned == frozenset()
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "36")
    assert api_plan().pinned
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "always")
    monkeypatch.setattr(cpus, "device_addresses", lambda devices: ["0000:ff:00.0"] * len(devices))
    real = cpus.deployment_plan
    refused = CpuPlan(frozenset({32}), 1, why="not pinned: too small")
    monkeypatch.setattr(cpus, "deployment_plan", lambda s, a, **kw: refused)
    assert api_plan().pinned == frozenset()
    monkeypatch.setattr(cpus, "deployment_plan", real)
    assert cpus._gpu_locks == {} and not list(gpu_locks.glob("mstar-gpu-*"))


# ── one server per core ─────────────────────────────────────────────────────

# a node of four GPUs, as on an 8-GPU host with two sockets, and a job's 8 cores on it
NODE4 = {"0000:18:00.0": 0, "0000:2a:00.0": 0, "0000:3a:00.0": 0, "0000:5d:00.0": 0,
         "0000:9a:00.0": 1, "0000:ab:00.0": 1, "0000:ba:00.0": 1, "0000:db:00.0": 1}
ON_NODE = ("0000:9a:00.0", "0000:ab:00.0", "0000:ba:00.0", "0000:db:00.0")
JOB = "32-39,160-167"

# another server in this container: plan from the fake host, claim the cores, say how that went, stay up until told
_CORE_SERVER = """
import json, sys
from pathlib import Path
from mstar.utils import cpus
lock_dir, root, setting, mask, address = sys.argv[1:6]
cpus._GPU_LOCK_DIR = Path(lock_dir)
def smt(cpu):
    return frozenset({cpu % 128, cpu % 128 + 128})
def build(slot):
    return cpus.deployment_plan(int(setting) if setting.isdigit() else setting, [address],
                                mask=cpus.parse_cpu_list(mask), usable=64, pci_root=Path(root), siblings=smt,
                                cluster=lambda cpu: frozenset({cpu}), reachable=sys.argv[6:], own=[address], slot=slot)
plan = cpus.claim_cores(build(0), build, siblings=smt)
print(json.dumps({"pinned": sorted(plan.pinned), "why": plan.why}), flush=True)
sys.stdin.read()
"""


def _node4(tmp_path, setting, address, mask=JOB, slot=0, own=None):
    base = tmp_path / "node4"
    root = base / "pci" if base.exists() else _sysfs(base, NODE4)
    return _plan(setting, [address], parse_cpu_list(mask), root,
                 reachable=tuple(NODE4), own=(address,) if own is None else own, slot=slot)


@pytest.fixture
def core_server(gpu_locks, tmp_path):
    """Starts a server process on the fake 4-GPU node that claims its cores;
    gives back the process and what it pinned."""
    import json

    root = _sysfs(tmp_path / "other", NODE4)
    started = []

    def start(address, setting="always", mask=JOB):
        p = subprocess.Popen(
            [sys.executable, "-c", _CORE_SERVER, str(gpu_locks), str(root), setting, mask, address, *NODE4],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        started.append(p)
        return p, json.loads(p.stdout.readline())

    yield start
    for p in started:
        p.kill()
        p.communicate()


def _claim(tmp_path, setting, address, mask=JOB):
    def build(slot):
        return _node4(tmp_path, setting, address, mask, slot)

    return cpus.claim_cores(build(0), build, siblings=smt)


def test_a_mask_too_small_for_every_reachable_gpu_is_sized_by_the_processs_own(tmp_path):
    """Eight cores can't give each of the node's four reachable GPUs four, but
    this process uses one of them: three cores and an API core, and the mask
    holds a second such set."""
    before = _node4(tmp_path, "always", ON_NODE[3], own=())
    assert before.pinned == frozenset() and "GPU 4 of 4 on its NUMA node" in before.why
    assert "8 core(s) there for 4 GPU(s)" in before.why
    plan = _node4(tmp_path, "always", ON_NODE[3])
    assert plan.workers[0].cpus == {32, 33, 34} and plan.result == 39 and (plan.sets, plan.slot) == (2, 0)
    assert plan.why.endswith("sized by the GPUs this process uses, set 1 of 2")
    assert "sized by the GPUs this process uses" in plan.workers[0].why
    second = _node4(tmp_path, "always", ON_NODE[3], slot=1)
    assert second.workers[0].cpus == {35, 36, 37} and second.result == 38
    assert "set 3 asked for" in _node4(tmp_path, "always", ON_NODE[3], slot=2).why
    # a GPU the process wasn't given doesn't size it
    assert _node4(tmp_path, "always", ON_NODE[3], own=ON_NODE[:3]).pinned == frozenset()


def test_a_mask_every_reachable_gpu_fits_places_as_before_whatever_the_process_uses(tmp_path):
    for own in ((), (ON_NODE[3],)):
        plan = _node4(tmp_path, "always", ON_NODE[3], mask="32-47,160-175", own=own)
        assert plan.workers[0].cpus == {41, 42, 43} and plan.result == 44 and plan.sets == 1
        assert plan.why.endswith("every reachable GPU fits")
    # the same in a container that opens two GPUs
    plan = _plan("always", [POD[0]], parse_cpu_list("32-39"), _sysfs(tmp_path / "pod"), reachable=POD, own=POD[:1])
    assert plan.pinned == {32, 33, 34, 39} and plan.sets == 1 and plan.why.endswith("every reachable GPU fits")


def test_two_servers_in_one_container_take_disjoint_cores_then_a_third_is_refused(core_server, tmp_path, caplog):
    first, placed = core_server(ON_NODE[0])
    assert placed["pinned"] == [32, 33, 34, 39]
    with caplog.at_level(logging.INFO, logger="mstar.utils.cpus"):
        plan = _claim(tmp_path, "always", ON_NODE[3])
    assert plan.workers[0].cpus == {35, 36, 37} and plan.result == 38 and plan.slot == 1 and plan.locked == "35-38"
    assert f"set 1 of 2: core 32 is held by pid {first.pid}" in caplog.text and "trying the next set" in caplog.text
    _, refused = core_server(ON_NODE[1])
    assert refused["pinned"] == [] and refused["why"].startswith("not pinned: no core set left in the mask")
    assert f"core 32 is held by pid {first.pid}" in refused["why"]
    assert f"core 35 is held by pid {os.getpid()}" in refused["why"]


def test_a_core_lock_goes_with_its_server_and_a_dead_pid_is_taken_over(core_server, tmp_path, gpu_locks, caplog):
    for stop in ("exit", "kill"):
        first, placed = core_server(ON_NODE[0])
        assert placed["pinned"] == [32, 33, 34, 39]
        if stop == "exit":
            first.communicate("")
        else:
            first.kill()
            first.communicate()
        with caplog.at_level(logging.INFO, logger="mstar.utils.cpus"):
            plan = _claim(tmp_path, "always", ON_NODE[3])
        assert plan.pinned == {32, 33, 34, 39} and plan.slot == 0
        assert f"mstar-core-32.lock names pid {first.pid}, which has exited; taking it over" in caplog.text
        assert (gpu_locks / "mstar-core-32.lock").read_text() == f"{os.getpid()}\n"
        for fd in cpus._core_locks.values():
            os.close(fd)
        cpus._core_locks.clear()
        caplog.clear()
    # this process's own pid left in a file is no takeover
    with caplog.at_level(logging.INFO, logger="mstar.utils.cpus"):
        assert _claim(tmp_path, "always", ON_NODE[3]).slot == 0
    assert "taking it over" not in caplog.text


def test_one_server_locks_its_cores_once_and_a_refusal_holds_none(core_server, tmp_path, gpu_locks):
    plan = _claim(tmp_path, "always", ON_NODE[3])
    held = dict(cpus._core_locks)
    assert sorted(p.name for p in held) == [f"mstar-core-{c}.lock" for c in (32, 33, 34, 39)]
    assert all(not os.get_inheritable(fd) for fd in held.values())
    assert cpus.claim_cores(plan, None, siblings=smt).locked == "32-34,39" and cpus._core_locks == held
    mode = (gpu_locks / "mstar-core-32.lock").stat().st_mode & 0o777
    assert mode == 0o666, "another user's server couldn't write its pid"
    # a set one of whose cores is held is let go whole
    for fd in held.values():
        os.close(fd)
    cpus._core_locks.clear()
    other, placed = core_server(ON_NODE[0], mask="34-39,162-167")
    assert 34 in placed["pinned"]
    assert "core 34 is held" in cpus._lock_cores([32, 33, 34]) and cpus._core_locks == {}
    assert cpus._lock_cores([32, 33]) is None


def test_siblings_and_a_cpu_number_go_through_the_lock(core_server, tmp_path):
    # six cores: short of two for each of the node's four GPUs, three sets for this process's one
    first, placed = core_server(ON_NODE[0], "siblings", "32-37,160-165")
    assert placed["pinned"] == [32, 37, 160]
    plan = _claim(tmp_path, "siblings", ON_NODE[3], "32-37,160-165")
    assert plan.workers[0].cpus == {33, 161} and plan.result == 36 and (plan.sets, plan.slot) == (3, 1)
    # a number names one set: the second server to name it pins nothing
    numbered, placed = core_server(ON_NODE[0], "40", "40-47,168-175")
    assert placed["pinned"] == [40, 47, 168]
    refused = _claim(tmp_path, 40, ON_NODE[3], "40-47,168-175")
    assert refused.pinned == frozenset() and f"core 40 is held by pid {numbered.pid}" in refused.why


def test_the_api_server_takes_the_next_free_set_and_its_workers_follow(core_server, tmp_path, monkeypatch, caplog):
    """The workers rebuild the plan, so they learn which set their API server
    locked from the environment it starts them with."""
    from mstar.api_server import entrypoint

    root = _sysfs(tmp_path / "api", NODE4)
    plan = deployment_plan

    def placed(setting, addresses, **kw):
        kw.update(reachable=tuple(NODE4), own=[ON_NODE[3]])
        return plan(setting, addresses, mask=parse_cpu_list(JOB), usable=16, pci_root=root, siblings=smt,
                    cluster=one_cpu_cluster, cpu_root=_idle(root.parent / "deep", [32], XEON), **kw)

    monkeypatch.setattr(cpus, "device_addresses", lambda devices: [ON_NODE[3]] * len(devices))
    monkeypatch.setattr(cpus, "deployment_plan", placed)
    monkeypatch.setattr(cpus, "smt_siblings", smt)
    monkeypatch.setattr(entrypoint, "accelerator_type", lambda: "cuda")
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "always")
    monkeypatch.delenv(cpus.CORE_SET_ENV, raising=False)
    config = {"node_groups": [{"node_names": ["llm"], "ranks": [0]}]}
    other, _ = core_server(ON_NODE[0])
    api = entrypoint.deployment_cpu_plan(config)
    assert api.workers[0].cpus == {35, 36, 37} and api.result == 38 and os.environ[cpus.CORE_SET_ENV] == "1"
    worker = _bare_worker("worker_0", "cuda:0")
    worker._plan_step_cpus(["worker_0"], ["cuda:0"])
    assert cpus.plan() == api and worker._step_cpus.cpus == {35, 36, 37}
    from api_server_helpers import make_test_api_server

    cpus.install(api)
    with caplog.at_level(logging.INFO, logger="mstar.api_server.entrypoint"):
        make_test_api_server()._take_result_cpu()
    assert ("on CPU 38 (always: GPU 1 of 1 on its NUMA node, over its allowed CPUs 32-39,160-167 (the first "
            "worker's device), from the top; sized by the GPUs this process uses, set 2 of 2; lock held on cores "
            "35-38; mask 32-39,160-167)") in caplog.text
    # once another server has set 2 as well, it pins nothing, and nor do its workers
    for fd in cpus._core_locks.values():
        os.close(fd)
    cpus._core_locks.clear()
    third, placed_third = core_server(ON_NODE[1])
    assert placed_third["pinned"] == [35, 36, 37, 38]
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        refused = entrypoint.deployment_cpu_plan(config)
    assert refused.pinned == frozenset() and os.environ["MSTAR_STEP_THREAD_CORE"] == "off"
    assert f"core 32 is held by pid {other.pid}" in caplog.text and f"core 35 is held by pid {third.pid}" in caplog.text
    # a number goes through the same lock
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "32")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        assert entrypoint.deployment_cpu_plan(config).pinned == frozenset()
    assert f"step_thread_core 32: nothing pinned, no core set left in the mask (set 1 of 1: core 32 is held by pid " \
           f"{other.pid}" in caplog.text
    cpus.install(None)
    entrypoint.knobs.configure(None)


# ── worker ──────────────────────────────────────────────────────────────────

def _bare_worker(worker_id="worker_1", device="cuda:1"):
    w = Worker.__new__(Worker)
    w.worker_id = worker_id
    w.device = torch.device(device)
    return w


def test_the_worker_takes_its_row_and_logs_it(tmp_path, monkeypatch, caplog):
    import mstar.worker.worker as worker_mod

    root = _sysfs(tmp_path)
    addresses = {"cuda:0": "0000:1a:00.0", "cuda:1": "0000:40:00.0", "cuda:2": "0000:62:00.0"}

    def plan(setting, addrs, **kw):
        return deployment_plan(setting, addrs, mask=HOST, usable=54, pci_root=root, siblings=smt,
                               cluster=one_cpu_cluster, cpu_root=_idle(root.parent / "deep", [0], XEON), **kw)

    monkeypatch.setattr(worker_mod.cpus, "device_addresses", lambda devices: [addresses.get(d) for d in devices])
    monkeypatch.setattr(worker_mod.cpus, "deployment_plan", plan)
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "always")
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._plan_step_cpus(["worker_0", "worker_1", "worker_2"], ["cuda:0", "cuda:1", "cuda:2"])
    assert w._step_cpus.cpus == {3, 4, 5} and w._process_cpus == HOST
    # three workers' three CPUs and the API server's CPU leave 54 - 10 for the pools
    assert cpus.plan().result == 31 and cpus.plan().left == 54 - 10
    assert ("step threads (main, GPU, plan) on CPUs 3-5 (always: GPU 2 of 2 on its NUMA node, over its allowed "
            "CPUs 0-31,128-159; cuda:1, worker 2 of 3, mask 0-255)") in caplog.text

    # standalone (no conductor list): the only worker
    w = _bare_worker()
    w._plan_step_cpus(["worker_1"], None)
    assert w._step_cpus.cpus == {3, 4, 5}

    # a number the mask can't honour stops the worker
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "300")
    with pytest.raises(CpuPlanError, match="outside the process mask"):
        _bare_worker()._plan_step_cpus(["worker_1"], None)

    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "off")
    w = _bare_worker()
    with caplog.at_level(logging.INFO, logger="mstar.worker.worker"):
        w._plan_step_cpus(["worker_1"], None)
    assert w._step_cpus is None and "step threads not pinned (step_thread_core off)" in caplog.text
    w._pin_step_thread()  # a no-op
    cpus.install(None)


# ── threads a pinned thread starts ─────────────────────────────────────────

needs_affinity = pytest.mark.skipif(not cpus.PINNABLE, reason="needs os.sched_setaffinity (Linux)")


@pytest.fixture
def intra_op_pool():
    """At least two intra-op threads, so a pool exists to start, under any
    OMP_NUM_THREADS (CI sets 1)."""
    before = torch.get_num_threads()
    torch.set_num_threads(max(2, before))
    yield
    torch.set_num_threads(before)


def _two_cpus():
    allowed = frozenset(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two allowed CPUs")
    return allowed, max(allowed)


def _in_thread(target):
    """Run ``target`` on a fresh thread: pinning pins the caller, and pytest's
    own thread must keep its mask."""
    errors = []

    def run():
        try:
            target()
        except BaseException as e:  # noqa: BLE001 — reported below
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join(60)
    assert not t.is_alive() and not errors, errors


@needs_affinity
def test_threads_and_pools_a_pinned_thread_starts_run_on_the_process_mask():
    """A new thread copies its creator's mask: a bare pool started after the
    pin sits on the pinned CPU, the helpers' threads do not."""
    allowed, cpu = _two_cpus()
    seen = {}

    def pinned():
        threads.pin_thread({cpu}, allowed)
        with ThreadPoolExecutor(1) as pool:
            seen["bare pool"] = pool.submit(os.sched_getaffinity, 0).result()
        with threads.ThreadPool(1, initializer=lambda: seen.setdefault("init", os.sched_getaffinity(0))) as pool:
            seen["pool"] = pool.submit(os.sched_getaffinity, 0).result()
        t = threads.Thread(target=lambda: seen.setdefault("thread", os.sched_getaffinity(0)))
        t.start()
        t.join()

    _in_thread(pinned)
    assert seen == {"bare pool": {cpu}, "init": allowed, "pool": allowed, "thread": allowed}


@needs_affinity
def test_a_thread_runs_on_the_process_mask_whatever_its_run_is():
    """No cooperation asked of the subclass; ``run`` called directly is the
    caller's own and stays on its mask."""
    allowed, cpu = _two_cpus()
    seen = {}

    class Override(threads.Thread):
        def run(self):
            seen["override"] = os.sched_getaffinity(0)

    class Below(Override):
        def run(self):
            seen["below"] = os.sched_getaffinity(0)

    class Mixin:
        def run(self):
            seen["mixin"] = os.sched_getaffinity(0)

    class Mixed(Mixin, threads.Thread):
        pass

    assigned = threads.Thread()
    assigned.run = lambda: seen.setdefault("assigned", os.sched_getaffinity(0))

    def pinned():
        threads.pin_thread({cpu}, allowed)
        for thread in (Override(), Below(), Mixed(), assigned):
            thread.start()
            thread.join()
        Override().run()
        seen["called directly"] = seen.pop("override")

    _in_thread(pinned)
    assert seen == {"below": allowed, "mixin": allowed, "assigned": allowed, "called directly": {cpu}}


@needs_affinity
def test_an_override_started_from_a_pinned_thread_is_on_the_process_mask():
    allowed, cpu = _two_cpus()
    seen = {}

    class Override(threads.Thread):
        def run(self):
            seen["override"] = os.sched_getaffinity(0)

    def pinned():
        threads.pin_thread({cpu}, allowed)
        thread = Override()
        thread.start()
        thread.join()

    _in_thread(pinned)
    assert seen == {"override": allowed}


def _tids() -> set[int]:
    return set(map(int, os.listdir("/proc/self/task")))


def _confined_to(cpus: int | frozenset[int], besides: set[int]) -> list[int]:
    """This process's threads, other than ``besides``, whose mask is just
    ``cpus``. ``besides`` holds the threads a test started with: an earlier
    test's idle pool threads may still be on their way out."""
    stray = []
    for tid in map(int, os.listdir("/proc/self/task")):
        try:
            if tid not in besides and os.sched_getaffinity(tid) == ({cpus} if isinstance(cpus, int) else cpus):
                stray.append(tid)
        except OSError:  # exited meanwhile
            pass
    return stray


@needs_affinity
def test_the_pin_starts_the_intra_op_pool_on_the_process_mask(intra_op_pool):
    """Without the warm-up op in ``pin_thread`` the pinned thread's first
    parallel op would start its pool on the pinned CPU."""
    allowed, cpu = _two_cpus()
    seen = {}
    before = _tids()

    def pinned():
        threads.pin_thread({cpu}, allowed)
        torch.ones(torch.get_num_threads() * 65536).sum()
        # the pool lives as long as its master thread, so look now
        seen["stray"] = _confined_to(cpu, before | {threading.get_native_id()})

    _in_thread(pinned)
    assert seen == {"stray": []}


# ── the ruff rule ───────────────────────────────────────────────────────────

def _ruff() -> str:
    path = shutil.which("ruff")
    if path is None:
        try:
            from ruff.__main__ import find_ruff_bin
            path = find_ruff_bin()
        except (ImportError, FileNotFoundError):
            pytest.skip("ruff is not installed")
    return path


@pytest.mark.parametrize(("path", "flagged"), [
    ("mstar/worker/new_module.py", True),
    ("mstar/utils/threads.py", False),             # the helpers
    ("mstar/model/foo/tests/test_foo.py", False),  # tests
    ("test/modular/test_foo.py", False),
    ("benchmark/foo.py", False),
])
def test_bare_threads_are_refused_under_mstar_only(path, flagged):
    source = ("import threading\nfrom concurrent.futures import ThreadPoolExecutor\n\n"
              "threading.Thread(target=print).start()\nThreadPoolExecutor(1).shutdown()\n")
    out = subprocess.run([_ruff(), "check", "--no-cache", "--select", "TID251", "--stdin-filename", path, "-"],
                         input=source, capture_output=True, text=True, cwd=REPO, check=False)
    assert ("TID251" in out.stdout) == flagged, out.stdout + out.stderr


# ── API server ─────────────────────────────────────────────────────────────

@needs_affinity
def test_the_api_message_thread_and_the_bridge_move_to_the_result_cpu(monkeypatch):
    from api_server_helpers import StandInSocket, make_test_api_server

    from mstar.api_server.rust_frontend import RustFrontendBridge
    from mstar.communication.event import EventWakeup

    allowed, cpu = _two_cpus()
    exited = SimpleNamespace(pid=5, is_alive=lambda: False, exitcode=1)
    server = make_test_api_server(
        conductor_proc=exited, _liveness_interval_s=0.0, _result_cpu=cpu, _process_cpus=allowed,
        communicator=StandInSocket(), preprocess_worker=SimpleNamespace(results_wakeup=EventWakeup()),
    )
    seen = {}

    def message_thread():
        server._process_messages()  # finds the conductor gone and returns
        seen["message"] = os.sched_getaffinity(0)

    async def serve():
        seen["bridge"] = os.sched_getaffinity(0)

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = server
    monkeypatch.setattr(bridge, "_serve", serve, raising=False)
    for target in (message_thread, bridge.run):
        _in_thread(target)
    assert seen == {"message": {cpu}, "bridge": {cpu}}
    # a server built without a CPU (tests, ``off``, macOS) pins nothing
    make_test_api_server().pin_result_thread()


def test_the_api_server_takes_its_cpu_from_the_plan_and_logs_it(monkeypatch, caplog):
    from api_server_helpers import make_test_api_server

    mask = parse_cpu_list("32-39")
    monkeypatch.setattr(cpus, "_plan", CpuPlan(mask, 8, (_one(33),), 38, "auto: test"))
    server = make_test_api_server()
    with caplog.at_level(logging.INFO, logger="mstar.api_server.entrypoint"):
        server._take_result_cpu()
    assert (server._result_cpu, server._process_cpus) == (38, mask)
    assert "message thread and frontend bridge on CPU 38 (auto: test; mask 32-39)" in caplog.text
    monkeypatch.setattr(cpus, "_plan", CpuPlan(mask, 8, why="not pinned: too small"))
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "auto")
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        server._take_result_cpu()
    assert server._result_cpu is None and "result threads not pinned (not pinned: too small)" in caplog.text


class _StopLoopError(BaseException):
    """Ends ``Worker.run``, whose loop handles only ``Exception``."""


@needs_affinity
@pytest.mark.parametrize("width", [1, 2])
def test_run_moves_the_main_gpu_and_plan_threads_to_their_cpus(monkeypatch, intra_op_pool, width):
    """The real ``Worker.run``, every collaborator stubbed: one scheduled step,
    then a speculated one the plan thread pre-plans. Each of the three threads
    reports its mask from inside its own work; a thread one of them starts
    afterwards for intra-op work keeps the process mask."""
    import mstar.worker.worker as worker_mod
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import Speculation

    allowed, cpu = _two_cpus()
    step = frozenset({cpu, min(allowed)} if width == 2 else {cpu})
    before = _tids()
    monkeypatch.setenv("MSTAR_PRE_PLAN_SPEC", "1")
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0))
    seen: dict[str, tuple[int, set[int]]] = {}
    schedule = [("a", 2), None]
    iteration = [0]

    def mask(name, stray=()):
        seen[name] = threading.get_native_id(), os.sched_getaffinity(0), stray

    def scheduled(name, rows):
        return ScheduledBatch(node_name="LLM", graph_walk=name,
                              request_to_worker_graph={f"{name}.{i}": 0 for i in range(rows)})

    def executing(batch):
        return ExecutingBatch(
            node_name="LLM",
            per_request_info={rid: SimpleNamespace(rid=rid, n=1) for rid in batch.request_to_worker_graph},
            step_context=StepContext(request_ids=tuple(batch.request_to_worker_graph),
                                     graph_walk=batch.graph_walk, slot=0, capture=False),
        )

    def next_batch(request_state, exclude_target=None):
        mask("main")
        if not schedule:
            raise _StopLoopError
        step = schedule.pop(0)
        return None if step is None else scheduled(*step)

    def gpu(batch, node_batch, plan_future):
        mask("gpu")
        if plan_future is not None:
            plan_future.result()
        if node_batch.launch_started_event is not None:
            node_batch.launch_started_event.set()
        return {}

    def preplan(pending, spec):
        # a pool this thread started before moving keeps the process mask; one
        # it started now would sit on its CPU
        torch.ones(torch.get_num_threads() * 65536).sum()
        # by name: the GPU thread may not have reported yet
        gpu_thread = {t.native_id for t in threading.enumerate() if t.name.startswith("mstar-gpu-")}
        steps = before | {seen["main"][0], threading.get_native_id()} | gpu_thread
        mask("plan", _confined_to(step, steps))
        return True

    def build_speculation(pending, consecutive, fair):
        iteration[0] += 1
        if iteration[0] != 2:
            return None, None
        batch = scheduled("b", 1)
        return Speculation(
            scheduled_batch=batch, node_batch=executing(batch), consumed_edges=set(),
            continuing_rids=set(), partition="p", is_new_iter=False, is_same_node=False,
            is_yield_away=True,
        ), None

    def stub(*a, **k):
        return None

    w = Worker.__new__(Worker)
    w.worker_id = "worker_0"
    w.device = torch.device("cpu")
    w.enable_nvtx = False
    w._phase_period, w._phase_buf = 0, {}
    w._max_consecutive_spec, w._launch_wait_s = 1024, 0.0
    w.tp_async_sched, w.is_tp_follower = False, False
    w._in_flight_rids = set()
    w._metrics = None
    w._step_cpus = StepCpus(step, "test")
    w._process_cpus = allowed
    w.parallel_groups = SimpleNamespace(barrier_all=stub, restore_default_timeout=stub)
    w.engine_manager = SimpleNamespace(warmup_all=stub, get_engine=lambda node: None)
    w.communicator = SimpleNamespace(send=stub, wait_for_work=stub)
    w.wakeup_event = SimpleNamespace(register_future=stub)
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p")
    w.scheduler = SimpleNamespace(
        take_admit_errors=dict, get_next_batch=next_batch, clear_batch_limit=stub, note_ran=stub,
    )
    for name in ("_verify_tp_async_sched_agrees", "_await_admit_settled",
                 "_apply_pending_removes_safe_to_drop", "_fail_requests", "_apply_pending_drains",
                 "_process_messages", "_apply_removes_whose_step_landed", "_check_ready_tensors",
                 "_poll_stream_buffers", "_set_in_flight_flag", "_clear_in_flight_flag",
                 "_settle_speculation", "_drop_rows_whose_loop_stopped", "_handle_admit_failure",
                 "_thread_outputs_to_speculative", "_postprocess_batch"):
        setattr(w, name, stub)
    w.maybe_send_zmq_to_tp_followers = lambda node_batch: -1
    w._build_executing_batch = executing
    w._build_speculation = build_speculation
    w._plans_ahead = lambda spec: True
    w._preplan_spec = preplan
    w._execute_on_gpu_thread = gpu

    def fail_loudly(error, *args):
        raise error

    w._handle_main_loop_error = fail_loudly
    errors = []

    def run():
        try:
            w.run()
        except _StopLoopError:
            pass
        except BaseException as e:  # noqa: BLE001 — reported below
            errors.append(e)

    # run() pins the thread that calls it; keep that off pytest's own thread
    main = threading.Thread(target=run)
    main.start()
    main.join(60)
    assert not main.is_alive() and not errors, errors
    assert {name: m for name, (_, m, _) in seen.items()} == {"main": step, "gpu": step, "plan": step}
    assert len({tid for tid, _, _ in seen.values()}) == 3
    # no other thread of the process landed on a step CPU, the intra-op pool
    # the plan thread just used included (looked at while it was alive)
    assert seen["plan"][2] == []
