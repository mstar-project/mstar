"""Confining a worker to the CPUs local to its GPU.

A decode step is host-bound — the worker's main loop has to keep the GPU fed —
so where its threads run matters. Unpinned, they float across every core on the
box, including the NUMA node the GPU is *not* attached to, and every migration
costs cache locality and cross-socket latency on the pinned host buffers the
D2H path uses.

Affinity also steers allocation: Linux places a page on the node that first
touches it, so pinning before the engine builds its buffers keeps them near.

Best-effort throughout. Anything unknown — no sysfs, no NUMA, a device that
reports no node — leaves affinity alone rather than guessing.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_ENV = "MSTAR_NUMA_PIN"


def _parse_cpulist(text: str) -> set[int]:
    """``0-63,128-191`` -> the set it denotes."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def local_cpus_for_device(device: torch.device) -> set[int] | None:
    """The CPUs sharing a NUMA node with ``device``, or None if unknown."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        # torch reports these as ints; sysfs names the device in hex, and the
        # domain is 4 digits there where nvidia-smi prints 8
        bdf = (
            f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}"
            f":{props.pci_device_id:02x}.0"
        )
        with open(f"/sys/bus/pci/devices/{bdf}/local_cpulist") as f:
            cpus = _parse_cpulist(f.read())
    except (OSError, AttributeError, ValueError) as err:
        logger.debug("NUMA: could not resolve CPUs for %s: %s", device, err)
        return None
    return cpus or None


def pin_to_device_numa_node(device: torch.device) -> str | None:
    """Restrict this process to ``device``'s local CPUs.

    Only acts when the process may still run on every online CPU; any prior
    narrowing (an operator's `taskset`, a cgroup or Slurm cpuset) is left as
    it is. Returns what it did, for logging, or None if it left affinity
    alone. Set ``MSTAR_NUMA_PIN=0`` to disable.
    """
    if os.environ.get(_ENV, "1").lower() in ("0", "false", "no"):
        return None
    local = local_cpus_for_device(device)
    if local is None:
        return None
    try:
        current = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None

    # Someone already narrowed us (taskset, numactl, a cgroup, a Slurm cpuset)
    # — that is a deliberate choice about placement, so don't second-guess it,
    # even when the mask happens to straddle the GPU's node: cutting a
    # sixteen-CPU allocation down to the eight on one socket leaves every
    # worker thread fighting over those eight.
    online = os.cpu_count() or len(current)
    if len(current) < online:
        logger.debug(
            "NUMA: affinity already narrowed to %d of %d CPUs; leaving it alone",
            len(current), online,
        )
        return None
    if current <= local:
        return None
    target = current & local
    if not target:
        logger.warning(
            "NUMA: %s is local to CPUs this process may not use; "
            "leaving affinity alone", device,
        )
        return None
    try:
        os.sched_setaffinity(0, target)
    except OSError as err:
        logger.debug("NUMA: sched_setaffinity failed: %s", err)
        return None
    return f"{len(target)} CPUs local to {device} (was {len(current)})"
