"""Batch composition: which ready rows of one (node, walk) run this step.

The ``MicroScheduler`` picks the (node, walk) and owns the mechanics (popping
ready queues, the backlog, holds, TP follow); a ``BaseBatchBuilder`` decides
which of the popped rows go into the step, which stay parked in the backlog
and which go back to their ready queues.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from mstar.worker.micro_scheduler import ScheduledBatch


@dataclass
class BatchBuildRequest:
    node_name: str
    graph_walk: str
    # Both popped: `backlog` out of the scheduler's backlog, `fresh` off the
    # ready queues. Whatever the builder neither schedules nor returns must come
    # back as `BatchBuildResult.backlog`.
    backlog: "ScheduledBatch | None" = None
    fresh: "ScheduledBatch | None" = None
    # Rows of `backlog` that cannot run this step; they stay parked.
    blocked_rids: set[int] = field(default_factory=set)
    # Rows left in the step once the caller's own are counted (> 0); None is uncapped.
    max_batch_size: int | None = None
    # The rid whose capture group the step must share, else its first row's.
    capture_group_of: int | None = None
    # `Engine.capture_group` for one rid of this (node, walk).
    capture_group: Callable[[int], Any] = lambda _: None


class BatchBuildResult(NamedTuple):
    scheduled: "ScheduledBatch | None"
    # Parked under this (node, walk) for a later step.
    backlog: "ScheduledBatch | None" = None
    # Popped rows handed back to their ready queues: rid -> worker graph id.
    returned: dict[int, int] | None = None


def off_group_rids(
    batch: "ScheduledBatch", capture_group: Callable[[int], Any],
    exclude_rids: set[int] = frozenset(), anchor: int | None = None,
) -> set[int]:
    """The rids of ``batch`` outside ``anchor``'s capture group, or outside
    the first eligible rid's when no anchor is given."""
    rids = [
        rid for rid in batch.request_to_worker_graph
        if rid not in exclude_rids
    ]
    if not rids:
        return set()
    # `is not None`: handle 0 is a real rid
    wanted = capture_group(rids[0] if anchor is None else anchor)
    return {rid for rid in rids if capture_group(rid) != wanted}


def _merged(
    backlog: "ScheduledBatch | None", fresh: "ScheduledBatch | None",
) -> "ScheduledBatch | None":
    """``backlog`` with ``fresh`` folded in behind it."""
    if not fresh:
        return backlog
    if backlog is None:
        return fresh
    backlog.merge(fresh)
    return backlog


class BaseBatchBuilder(ABC):
    @abstractmethod
    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        """One step's rows out of ``request``'s backlog and entries."""

    def room_for_continuing(
        self, cap: int | None, backlog: "ScheduledBatch | None",
    ) -> int | None:
        """How many of a speculation chain's own rids fit in a step capped at
        ``cap`` when ``backlog`` waits under the same (node, walk).

        The chain only ever continues its own rids, so at the cap a backlogged
        chunk would never be reached. Giving the backlog first claim costs the
        displaced rids one step — their nodes go ready again when the in-flight
        batch lands. None for an uncapped node: nothing is displaced.
        """
        if cap is None:
            return None
        return max(0, cap - (0 if backlog is None else len(backlog)))

    def chain_must_yield(
        self, backlog: "ScheduledBatch", live_rids: set[int], anchor: int,
        capture_group: Callable[[int], Any],
    ) -> bool:
        """Whether a speculation chain anchored at ``anchor`` has to stop so
        ``backlog``'s live rows can run: the chain's merge (a build with
        ``capture_group_of=anchor``) would never take them."""
        exclude = set(backlog.request_to_worker_graph) - live_rids
        return bool(off_group_rids(
            backlog, capture_group, exclude_rids=exclude, anchor=anchor,
        ))


class FIFOBatchBuilder(BaseBatchBuilder):
    """Backlog first, then fresh rows, in arrival order, up to the cap; one
    capture group per step. Everything past the cap stays in the backlog."""

    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        max_bs = request.max_batch_size
        batch = _merged(request.backlog, request.fresh)
        if batch is None:
            return BatchBuildResult(None)

        exclude = request.blocked_rids | off_group_rids(
            batch, request.capture_group,
            exclude_rids=request.blocked_rids, anchor=request.capture_group_of,
        )
        scheduled, remainder = batch.split_off_first(max_bs, exclude_rids=exclude)
        return BatchBuildResult(scheduled, backlog=remainder)


class BatchBuilderType(Enum):
    FIFO = "fifo"


def make_batch_builder(kind: BatchBuilderType) -> BaseBatchBuilder:
    return {BatchBuilderType.FIFO: FIFOBatchBuilder}[kind]()
