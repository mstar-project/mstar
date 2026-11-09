"""Which (node, walk) a worker's next step goes to.

The micro-scheduler asks a ``SchedulingPolicy`` two things: the order in which to try
the (node, walk) keys that have ready rows (``order``), and whether a speculation
chain gives the step after its own to other ready work (``chain_yields``). A key is
the scheduler's: a real walk, or a combined walk its rows share.

``RoundRobin`` is the default: least recently scheduled first, and a chain yields
only after it has had a step of its own. ``Priority`` serves keys a model or a
deployment declares latency-sensitive (``WalkPriority``) first. Its cost falls on
every other key's steps, which wait while a higher one is ready, so a key's
``max_wait`` bounds that wait: once a ready key has gone that many steps since it
last ran, it goes first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from typing import NamedTuple, Protocol

Key = tuple[str, str]


class WalkPriority(NamedTuple):
    """One entry of a model's or a deployment's priorities (``Model.get_walk_priorities``,
    the serving config's ``scheduling.walk_priorities``). It applies to every (node,
    walk) it names: a node's walks, a walk's nodes, or one pair. Where entries overlap,
    the most specific decides: a pair over a walk over a node.

    ``priority``: higher first, unlisted 0. ``max_wait``: the most of the worker's chain
    steps the key may go without running while it has ready rows, before it goes first;
    None falls back to the policy's default, which may itself be none (no bound)."""

    node: str | None = None
    walk: str | None = None
    priority: int = 0
    max_wait: int | None = None


class ReadyPeek(Protocol):
    """What a policy reads off the scheduler."""

    batch_number: int
    node_and_walk_to_last_batch_num: Mapping[Key, int]
    # chain-lane steps taken, and each scheduling key's last
    chain_steps: int
    chain_step_of: Mapping[Key, int]

    def has_ready_excluding(self, exclude_target: Key | None, among: Collection[Key] | None = None) -> bool:
        """Whether a key other than ``exclude_target`` (and, given ``among``, one of
        those) has a row the engine would take. Its readiness checks are real ones,
        so a policy asks only when it would yield on the answer."""


class SchedulingPolicy(ABC):
    @abstractmethod
    def order(self, ready: Mapping[Key, Sequence], peek: ReadyPeek) -> list[Key]:
        """The keys of ``ready`` (each key's ready rows, in scan order) in the order the
        scheduler tries them."""

    @abstractmethod
    def chain_yields(self, chain_key: Key, consecutive: int, mid_prompt: bool, peek: ReadyPeek) -> bool:
        """Whether a speculation chain into ``chain_key`` gives the next step to other
        ready work. ``consecutive``: the chain's steps since it last yielded;
        ``mid_prompt``: the chain is cutting a prompt into chunks."""


class RoundRobin(SchedulingPolicy):
    def order(self, ready, peek):
        last = peek.node_and_walk_to_last_batch_num
        # stable: ties keep the scan order
        return sorted(ready, key=lambda key: last.get(key, 0))

    def chain_yields(self, chain_key, consecutive, mid_prompt, peek):
        # a chain keeps its first step, then takes turns with other ready work
        return (consecutive >= 1 or mid_prompt) and peek.has_ready_excluding(chain_key)


class Priority(RoundRobin):
    """Higher ``priorities`` first, round robin among equals and for an unlisted key (0);
    ahead of both, the keys that have waited their ``max_wait``, longest waiting first.

    A chain also gives up its first step to a ready key above its own or one that has
    waited its bound. Where round robin would ask anyway, its peek covers those keys
    too, so a pass makes at most one. A key's wait is the chain steps this worker has
    taken since that key last ran, so one that comes ready after an idle spell may go
    first once."""

    def __init__(self, priorities: Mapping[Key, int], max_waits: Mapping[Key, int]):
        self.priorities = dict(priorities)
        self.max_waits = dict(max_waits)
        # each priority a chain can have -> the keys above it; fixed, so no pass builds it
        levels = set(self.priorities.values()) | {0}
        self._above = {
            level: frozenset(key for key, p in self.priorities.items() if p > level) for level in levels
        }

    def priority(self, key: Key) -> int:
        return self.priorities.get(key, 0)

    def _waited(self, peek: ReadyPeek) -> list[tuple[int, Key]]:
        """(steps waited, key) of the bounded keys at or past their bound."""
        if not self.max_waits:
            return []
        now, last = peek.chain_steps, peek.chain_step_of
        return [(now - last.get(key, 0), key) for key, bound in self.max_waits.items()
                if now - last.get(key, 0) >= bound]

    def order(self, ready, peek):
        by_priority = sorted(super().order(ready, peek), key=lambda key: -self.priority(key))
        overdue = [key for _, key in sorted(self._waited(peek), key=lambda wk: -wk[0]) if key in ready]
        if not overdue:
            return by_priority
        return overdue + [key for key in by_priority if key not in overdue]

    def chain_yields(self, chain_key, consecutive, mid_prompt, peek):
        if consecutive >= 1 or mid_prompt:
            return peek.has_ready_excluding(chain_key)
        among = self._above[self.priority(chain_key)]
        overdue = [key for _, key in self._waited(peek) if key != chain_key]
        if overdue:
            among = among | frozenset(overdue)
        return bool(among) and peek.has_ready_excluding(chain_key, among=among)


def resolve_walk_priorities(
    entries: Sequence[WalkPriority], walk_nodes: Mapping[str, Collection[str]], owner: str,
) -> dict[Key, WalkPriority]:
    """Each (node, walk) the entries reach (``walk_nodes``: walk -> the nodes it runs),
    as the most specific entry that names it: a pair, then a walk, then a node. Refuses
    an entry naming neither, a node or walk no walk runs, a pair the walk doesn't run,
    a negative priority, a ``max_wait`` under 1, and two entries for the same target.
    ``owner`` names where the entries came from, in the error."""
    nodes_anywhere = {node for nodes in walk_nodes.values() for node in nodes}
    seen: set[tuple[str | None, str | None]] = set()
    by_rank: list[dict[Key, WalkPriority]] = [{}, {}, {}]  # node, walk, pair
    for entry in entries:
        if not isinstance(entry, WalkPriority):
            raise ValueError(f"{owner}: {entry!r} is not a WalkPriority")
        node, walk = entry.node, entry.walk
        if node is None and walk is None:
            raise ValueError(f"{owner}: {entry!r} names neither a node nor a walk")
        if walk is not None and walk not in walk_nodes:
            raise ValueError(f"{owner}: no walk is named {walk!r}")
        if node is not None and node not in (walk_nodes[walk] if walk is not None else nodes_anywhere):
            where = f"walk {walk!r}" if walk is not None else "any walk"
            raise ValueError(f"{owner}: {node!r} is not a node of {where}")
        if not isinstance(entry.priority, int) or isinstance(entry.priority, bool) or entry.priority < 0:
            raise ValueError(f"{owner}: priority must be a non-negative int, got {entry.priority!r} in {entry!r}")
        if entry.max_wait is not None and (
            not isinstance(entry.max_wait, int) or isinstance(entry.max_wait, bool) or entry.max_wait < 1
        ):
            raise ValueError(f"{owner}: max_wait must be a positive int or none, got {entry.max_wait!r}")
        if (node, walk) in seen:
            raise ValueError(f"{owner}: two entries for node {node!r}, walk {walk!r}")
        seen.add((node, walk))
        rank = 2 if node is not None and walk is not None else 1 if walk is not None else 0
        for w, nodes in walk_nodes.items():
            if walk is not None and w != walk:
                continue
            for n in nodes:
                if node is None or n == node:
                    by_rank[rank][(n, w)] = entry
    resolved: dict[Key, WalkPriority] = {}
    for level in by_rank:
        resolved.update(level)
    return resolved
