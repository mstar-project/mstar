"""Pinning a worker to its GPU's NUMA node.

The worker's main loop is host-bound, so where its threads run decides how
well it keeps the GPU fed. The risk in pinning is doing it *wrong* — widening
affinity, fighting an operator's own `taskset`, or guessing when the topology
is unknown — so that is what these cover.
"""
from __future__ import annotations

import os

import pytest
import torch

from mstar.utils.numa import (
    _parse_cpulist,
    local_cpus_for_device,
    pin_to_device_numa_node,
)


def test_parses_the_sysfs_cpulist_format():
    assert _parse_cpulist("0-3") == {0, 1, 2, 3}
    assert _parse_cpulist("0-1,4") == {0, 1, 4}
    # the shape this box actually reports
    assert _parse_cpulist("64-127,192-255") == set(range(64, 128)) | set(
        range(192, 256)
    )
    assert _parse_cpulist("") == set()


def test_cpu_device_is_left_alone():
    cpu = torch.device("cpu")
    assert local_cpus_for_device(cpu) is None
    assert pin_to_device_numa_node(cpu) is None


def test_env_disables_it(monkeypatch):
    monkeypatch.setenv("MSTAR_NUMA_PIN", "0")
    assert pin_to_device_numa_node(torch.device("cuda", 0)) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_pins_then_leaves_a_narrower_affinity_alone():
    """Idempotent, and never widens: a second call is a no-op, which is also
    what protects an operator's own `taskset`/`numactl` from being undone."""
    device = torch.device("cuda", 0)
    local = local_cpus_for_device(device)
    if local is None:
        pytest.skip("no NUMA topology for this device")
    before = os.sched_getaffinity(0)
    try:
        first = pin_to_device_numa_node(device)
        after = os.sched_getaffinity(0)
        assert after <= before, "pinning must never widen affinity"
        if first is not None:
            assert after == before & local
            assert pin_to_device_numa_node(device) is None
    finally:
        os.sched_setaffinity(0, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_every_visible_device_resolves_to_one_node():
    """Each GPU's CPU list must be non-empty and a real subset of the box."""
    total = os.cpu_count() or 1
    for i in range(torch.cuda.device_count()):
        cpus = local_cpus_for_device(torch.device("cuda", i))
        if cpus is None:
            continue
        assert cpus, f"cuda:{i} resolved to an empty CPU set"
        assert max(cpus) < total


def _patch_topology(monkeypatch, *, local, current, online):
    """Fake box: `local` CPUs next to cuda:0, `current` the process's mask."""
    from mstar.utils import numa

    calls: list[set[int]] = []
    monkeypatch.setattr(numa, "local_cpus_for_device", lambda device: set(local))
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda pid: set(current))
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: calls.append(set(cpus)))
    monkeypatch.setattr(numa.os, "cpu_count", lambda: online)
    monkeypatch.delenv("MSTAR_NUMA_PIN", raising=False)
    return calls


def test_pins_only_a_process_that_may_still_run_anywhere(monkeypatch):
    calls = _patch_topology(monkeypatch, local=range(0, 8), current=range(0, 16), online=16)
    assert pin_to_device_numa_node(torch.device("cuda", 0)) is not None
    assert calls == [set(range(0, 8))]


def test_a_cpuset_that_straddles_the_sockets_is_left_alone(monkeypatch):
    """A Slurm or cgroup allocation of 8 of 16 CPUs, four on each socket: an
    operator's placement, not ours to cut in half."""
    calls = _patch_topology(
        monkeypatch, local=range(0, 8), current={0, 1, 2, 3, 8, 9, 10, 11}, online=16,
    )
    assert pin_to_device_numa_node(torch.device("cuda", 0)) is None
    assert calls == []


def test_already_local_is_a_no_op(monkeypatch):
    calls = _patch_topology(monkeypatch, local=range(0, 8), current=range(0, 16), online=16)
    monkeypatch.setattr("mstar.utils.numa.os.sched_getaffinity", lambda pid: set(range(0, 4)))
    assert pin_to_device_numa_node(torch.device("cuda", 0)) is None
    assert calls == []
