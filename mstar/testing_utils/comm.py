"""A tensor-parallel group simulated on one CPU: a thread per rank, the
collectives meeting at a barrier, so a component's own parallel layers run
sharded, unmodified, in a CPU test."""

from __future__ import annotations

import threading
from collections.abc import Callable

import torch

from mstar.distributed.communication import CommGroup
from mstar.utils.threads import Thread


class _Exchange:
    def __init__(self, world: int):
        self.barrier = threading.Barrier(world)
        self.slots: list = [None] * world

    def share(self, rank: int, value):
        self.slots[rank] = value
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


class ThreadGroup(CommGroup):
    """One rank of a simulated group: collectives across the ranks' threads, summed in
    rank order (deterministic)."""

    def __init__(self, rank: int, world: int, exchange: _Exchange):
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world)))
        self._exchange = exchange

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        parts = self._exchange.share(self.rank, input_)
        out = parts[0].clone()
        for p in parts[1:]:
            out += p
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return torch.cat(self._exchange.share(self.rank, input_), dim=dim)


def run_ranks(world: int, fn: Callable[[int, CommGroup], object]) -> list:
    """``fn(rank, group)`` on every rank of a simulated group, concurrently; the results
    in rank order. A rank's exception aborts the others' collectives and is raised."""
    exchange = _Exchange(world)
    results: list = [None] * world
    errors: list = []

    def body(rank: int):
        try:
            results[rank] = fn(rank, ThreadGroup(rank, world, exchange))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
            exchange.barrier.abort()

    threads = [Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results
