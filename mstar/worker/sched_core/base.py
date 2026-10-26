"""A scheduling pass split at the engine's readiness check: plan -> check -> commit.

The check (``Engine.check_ready``) is per-rank resource code with side effects
(a KV reload, a transfer kickoff), so it stays Python. Everything around it --
which rows to check and in what order, and what to do with the verdicts -- is
the scheduler core's. A pass ``begin``s, and each time it stops with a
``CheckRequest`` the caller answers with one verdict per row
(``MicroScheduler._run_checks``) and ``resume``s it, until it returns the
step. ``SchedCore`` is that interface,
with the backlog queries the rest of the scheduler needs; ``PythonSchedCore``
and ``RustSchedCore`` (the graph runtime's) implement it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from enum import IntEnum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from mstar.worker.engine_manager import EngineManager
    from mstar.worker.micro_scheduler import ScheduledBatch

class Verdict(IntEnum):
    """One per requested row; ``CheckResult.verdicts`` holds them as bytes."""
    NOT_READY = 0
    READY = 1
    FAILED = 2
    # past the request's limit
    UNCHECKED = 3
    # ready, but outside the step's capture group (``CheckRequest.by_anchor``)
    OFF_GROUP = 4


# ``CheckResult.groups`` for a row whose node does not split by capture key
NO_GROUP = -1


class CheckRequest(NamedTuple):
    """Rows whose engine readiness a scheduling pass needs, in check order."""
    rids: list[int]
    # row i is node ``nodes[node_idx[i]]``; None when every row is ``nodes[0]``
    node_idx: list[int] | None
    nodes: list[str]
    # stop once this many rows were found ready (and counted); -1 checks all.
    # Rows past the stop are UNCHECKED.
    limit: int = -1
    # The scheduling key's walk per node, under which a node that splits
    # batches by capture key groups its rows; None asks for no groups.
    walks: list[str] | None = None
    # The row whose capture group the step must share; None: the first ready row.
    anchor: int | None = None
    # Whether a ready row outside the anchor's group is OFF_GROUP rather than
    # READY, and so does not count toward ``limit``.
    by_anchor: bool = False


class CheckResult(NamedTuple):
    """A check's answer: one verdict per row, and for a capture-split node
    each ready row's capture group (as an id, ``NO_GROUP`` otherwise) and the
    anchor's."""
    verdicts: bytes
    groups: list[int] | None = None
    anchor_group: int | None = None


class PassRequest(NamedTuple):
    """One scheduling pass's arguments (see ``MicroScheduler.get_next_batch``)
    and the rows it must leave alone."""
    target: tuple[str, str] | None
    exclude_target: tuple[str, str] | None
    pre_existing_batch_size: int
    capture_group_of: int | None
    # rows that wait without a check: deferred removes and unexpired holds
    waiting: frozenset[int]
    # the scheduler's failed rows, live: a check can fail a row mid-pass
    failed: AbstractSet[int]


class PassStep(NamedTuple):
    """Where a pass stands: the rows it needs checked next, or (``check`` is
    None) the step it built, if any."""
    check: CheckRequest | None
    batch: ScheduledBatch | None = None


class SchedKeys:
    """Scheduling keys and their caps, shared by the scheduler and its core:
    a (node, walk)'s key is its combined walk if it has one."""

    def __init__(
        self, engine_manager: EngineManager,
        combined_walk_of: dict[tuple[str, str], str] | None = None,
        speculation_mixes_walks: Mapping[str, Any] | None = None,
    ):
        self.engine_manager = engine_manager
        # (node, real walk) -> combined walk
        self.combined_walk_of = combined_walk_of or {}
        self.walks_of_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (node, walk), combined in self.combined_walk_of.items():
            self.walks_of_key[(node, combined)].append(walk)
        # (node, walk) whose speculation chain merges the other walks of its
        # combined walk; see `speculation_mixes`
        self.mixing_chains = self._parse_speculation_mixes_walks(speculation_mixes_walks or {})
        self._max_bs: dict[tuple[str, str], int | None] = {}
        self._walk_caps_of_key: dict[tuple[str, str], dict[str, int | None]] = {}
        self._row_caps_of_key: dict[tuple[str, str], dict[str, int]] = {}

    def _parse_speculation_mixes_walks(
        self, setting: Mapping[str, Any],
    ) -> frozenset[tuple[str, str]]:
        """The serving config's ``speculation_mixes_walks``: per node, one bool
        for every walk of its combined walks, or a mapping of walk -> bool."""
        walks_of_node: dict[str, set[str]] = defaultdict(set)
        for node, walk in self.combined_walk_of:
            walks_of_node[node].add(walk)
        on = set()
        for node, value in setting.items():
            if node not in walks_of_node:
                raise ValueError(
                    f"speculation_mixes_walks names {node!r}, which has no combined walk"
                )
            per_walk = value if isinstance(value, Mapping) else dict.fromkeys(walks_of_node[node], value)
            for walk, mixes in per_walk.items():
                if walk not in walks_of_node[node]:
                    raise ValueError(
                        f"speculation_mixes_walks names walk {walk!r} of {node!r}, "
                        "which is in none of its combined walks"
                    )
                if not isinstance(mixes, bool):
                    raise ValueError(
                        f"speculation_mixes_walks for {node}/{walk}: {mixes!r} is not a bool"
                    )
                if mixes:
                    on.add((node, walk))
        return frozenset(on)

    def speculation_mixes(self, node_name: str, graph_walk: str) -> bool:
        """Whether a speculation chain on ``(node_name, graph_walk)`` merges
        ready rows of the other walks of its combined walk. Off (the default),
        the chain takes only its own walk's rows, and another walk's ready rows
        make it yield; steps then mix walks only when the scheduler builds one
        fresh, which plans inline."""
        return (node_name, graph_walk) in self.mixing_chains

    def key(self, node_name: str, graph_walk: str) -> tuple[str, str]:
        """The scheduling key: the walk's combined walk if it has one."""
        return node_name, self.combined_walk_of.get((node_name, graph_walk), graph_walk)

    def label_for(
        self, node_name: str, request_walks: dict[int, str],
    ) -> tuple[str, dict[int, str]]:
        """A batch's walk label and ``request_walks`` for rows with these real
        walks. The batch takes ``request_walks`` itself, not a copy."""
        walks = set(request_walks.values())
        if len(walks) == 1:
            return walks.pop(), {}
        return self.key(node_name, next(iter(walks)))[1], request_walks

    def walk_caps(self, key: tuple[str, str]) -> dict[str, int | None]:
        """``BatchBuildRequest.walk_caps`` for ``key``: its walk's cap, or a
        combined key's walks' caps and its own under its label."""
        caps = self._walk_caps_of_key.get(key)
        if caps is None:
            walks = self.walks_of_key.get(key, ())
            caps = {walk: self.max_batch_size(key[0], walk) for walk in walks}
            caps[key[1]] = self.max_batch_size(*key)
            self._walk_caps_of_key[key] = caps
        return caps

    def walk_row_caps(self, key: tuple[str, str]) -> dict[str, int]:
        """``BatchBuildRequest.walk_row_caps`` for ``key``: empty unless it is
        a combined walk. Fixed by the model, so asked once."""
        caps = self._row_caps_of_key.get(key)
        if caps is None:
            engine = self.engine_manager.get_engine(key[0])
            caps = {}
            for walk in self.walks_of_key.get(key, ()):
                cap = engine.get_max_rows_in_combined(key[0], walk)
                if cap is not None:
                    caps[walk] = cap
            self._row_caps_of_key[key] = caps
        return caps

    def max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one. A combined
        walk's cap is the model's to declare, via its captures or max_batch_size.
        Fixed once the graphs are captured, so asked once."""
        key = (node_name, graph_walk)
        if key not in self._max_bs:
            self._max_bs[key] = self.engine_manager.get_engine(node_name).get_max_batch_size(
                node_name, graph_walk
            )
        return self._max_bs[key]

    def walk_set_cap(self, key: tuple[str, str], walks: set[str]) -> int | None:
        """The cap of a mixed ``key`` step holding ``walks``, as the batch
        builder composes one: the combined walk's own where it has one, else
        the smallest of the walks present."""
        own = self.max_batch_size(*key)
        if own is not None:
            return own
        caps = [self.max_batch_size(key[0], walk) for walk in walks]
        return min((cap for cap in caps if cap is not None), default=None)


class RoundRobin:
    """Which scheduling key ran least recently."""

    def __init__(self):
        self.batch_number = 0
        self.last_batch_num: dict[tuple[str, str], int] = {}

    def note(self, key: tuple[str, str]) -> None:
        self.batch_number += 1
        self.last_batch_num[key] = self.batch_number

    def pick(self, ready: Iterable[tuple[str, str]]) -> tuple[str, str] | None:
        """The least recently scheduled key; ties go to the first given."""
        return min(ready, default=None, key=lambda key: self.last_batch_num.get(key, 0))


class SchedCore(ABC):
    """The scheduling pass after the TP-follow FIFO, and the backlog it keeps:
    the rows of a ready set too big for one step, waiting their turn."""

    @abstractmethod
    def begin(self, request: PassRequest) -> PassStep:
        """Start a pass; it runs until it needs rows checked, or is done."""

    @abstractmethod
    def resume(self, result: CheckResult) -> PassStep:
        """Continue the pass with the answer to its last check."""

    @abstractmethod
    def room_for_continuing(self, target: tuple[str, str]) -> int | None:
        """How many of a speculative batch's own rids fit in ``target``'s step
        beside its backlog; see ``BaseBatchBuilder.room_for_continuing``."""

    @abstractmethod
    def backlog_has_schedulable(
        self, exclude_key: tuple[str, str] | None, exclude_walk: str | None,
        waiting: Iterable[int], failed: Iterable[int],
    ) -> bool:
        """Whether a backlogged row outside ``exclude_key`` (or, with
        ``exclude_walk``, outside that walk of it) could run."""

    @abstractmethod
    def backlog_rows(self, exclude: Iterable[int]) -> list[tuple[str, int]]:
        """``(node, rows)`` per backlogged key, leaving out ``exclude``."""

    @abstractmethod
    def drop_rids(self, rids: Iterable[int]) -> None:
        """Take requests out of the backlog: the only place holding them."""

    @abstractmethod
    def live_backlog(
        self, key: tuple[str, str], waiting: Iterable[int], failed: Iterable[int],
    ) -> list[int]:
        """``key``'s backlogged rows that are not failed or waiting, in order."""
