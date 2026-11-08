"""Which (node, walk) a worker's next step goes to.

The micro-scheduler asks a ``SchedulingPolicy`` two things: the order in which to try
the (node, walk) keys that have ready rows (``order``), and whether a speculation
chain gives the step after its own to other ready work (``chain_yields``). A key is
the scheduler's: a real walk, or a combined walk its rows share.

``RoundRobin`` is the default: least recently scheduled first, and a chain yields
only after it has had a step of its own. ``Priority`` serves keys a model declares
latency-sensitive (``Model.get_walk_priorities``) first, at the cost of every other
key's steps, which wait while one of them is ready: a walk that is always ready
starves the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

Key = tuple[str, str]


class ReadyPeek(Protocol):
    def has_ready_excluding(self, exclude_target: Key | None, among: Collection[Key] | None = None) -> bool:
        """Whether a key other than ``exclude_target`` (and, given ``among``, one of
        those) has a row the engine would take. Its readiness checks are real ones,
        so a policy asks only when it would yield on the answer."""


class SchedulingPolicy(ABC):
    @abstractmethod
    def order(self, ready: Mapping[Key, Sequence], last_scheduled: Mapping[Key, int]) -> list[Key]:
        """The keys of ``ready`` (each key's ready rows, in scan order) in the order the
        scheduler tries them. ``last_scheduled``: each key's last step number, absent
        for one never scheduled."""

    @abstractmethod
    def chain_yields(self, chain_key: Key, consecutive: int, mid_prompt: bool, peek: ReadyPeek) -> bool:
        """Whether a speculation chain into ``chain_key`` gives the next step to other
        ready work. ``consecutive``: the chain's steps since it last yielded;
        ``mid_prompt``: the chain is cutting a prompt into chunks."""


class RoundRobin(SchedulingPolicy):
    def order(self, ready, last_scheduled):
        # stable: ties keep the scan order
        return sorted(ready, key=lambda key: last_scheduled.get(key, 0))

    def chain_yields(self, chain_key, consecutive, mid_prompt, peek):
        # a chain keeps its first step, then takes turns with other ready work
        return (consecutive >= 1 or mid_prompt) and peek.has_ready_excluding(chain_key)


class Priority(RoundRobin):
    """Higher ``priorities`` first; round robin among equals, and for a key unlisted (0).

    A chain also gives up its first step to a ready key above its own. Where round robin
    would ask anyway, its peek covers those keys too, so a pass makes at most one."""

    def __init__(self, priorities: Mapping[Key, int]):
        self.priorities = dict(priorities)
        # each priority a chain can have -> the keys above it; fixed, so no pass builds it
        levels = set(self.priorities.values()) | {0}
        self._above = {
            level: frozenset(key for key, p in self.priorities.items() if p > level) for level in levels
        }

    def priority(self, key: Key) -> int:
        return self.priorities.get(key, 0)

    def order(self, ready, last_scheduled):
        return sorted(super().order(ready, last_scheduled), key=lambda key: -self.priority(key))

    def chain_yields(self, chain_key, consecutive, mid_prompt, peek):
        if consecutive >= 1 or mid_prompt:
            return peek.has_ready_excluding(chain_key)
        above = self._above[self.priority(chain_key)]
        return bool(above) and peek.has_ready_excluding(chain_key, among=above)
