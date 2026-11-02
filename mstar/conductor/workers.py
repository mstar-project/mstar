"""The workers a deployment config starts, and the device each runs on.

The conductor starts its workers from this list, and the API server places its
result threads from it before the conductor exists, so the two can't disagree
about which CPUs the workers take (``mstar.utils.cpus``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch


def accelerator_type() -> str:
    """The device type workers run on: the accelerator's, else ``cpu``."""
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    return accelerator.type if accelerator is not None else "cpu"


def worker_device(device_type: str, rank: int, rank_devices: dict[int, int] | None = None) -> str:
    """The device string a worker rank runs on: its own number unless the
    deployment's ``rank_devices`` maps it elsewhere (several ranks may share a
    GPU); CPU deployments ignore the mapping."""
    if device_type == "cpu":
        return "cpu"
    index = (rank_devices or {}).get(rank, rank)
    return f"{device_type}:{index}"


class DeploymentWorkers(NamedTuple):
    """One entry per worker, in rank order."""
    ranks: list[int]
    devices: list[str]


def deployment_workers(config: Mapping[str, Any], device_type: str) -> DeploymentWorkers:
    """Every rank ``node_groups`` names gets a worker (``check_placement``
    refuses a rank no walk runs on), on the device ``worker_device`` gives it.

    The optional ``rank_devices: {rank: device_index}`` places a worker rank on
    a device other than the one its number implies, e.g. two workers sharing one
    GPU so a light node's steps stop interleaving with a heavy node's on the
    same worker loop.

    The API server reads this before the conductor's ``validate_deployment``
    has seen the config, so a malformed entry is skipped here and reported there.
    """
    groups = config.get("node_groups")
    ranks = sorted({
        rank
        for group in (groups if isinstance(groups, list) else [])
        if isinstance(group, dict) and isinstance(group.get("ranks"), list)
        for rank in group["ranks"]
        if isinstance(rank, int)
    })
    rank_devices = {int(rank): int(index) for rank, index in (config.get("rank_devices") or {}).items()}
    return DeploymentWorkers(ranks, [worker_device(device_type, rank, rank_devices) for rank in ranks])
