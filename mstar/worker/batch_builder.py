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
    # The rid whose capture group the step must share, else its first row's.
    capture_group_of: int | None = None
    # Each walk's cap, before the caller's rows are counted; None (or absent)
    # is uncapped. Under `graph_walk` for a key of one walk. Under a combined
    # walk, each real walk's, plus the combined walk's own under `graph_walk`,
    # which caps a step mixing walks.
    walk_caps: dict[str, int | None] = field(default_factory=dict)
    # The caller's own rows (a speculation merge's continuing ones) and their walk.
    pre_existing_batch_size: int = 0
    pre_existing_walk: str | None = None
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
    capture group per step. Everything past the cap stays in the backlog.

    Under a combined walk a step of one walk takes that walk's cap and a mixed
    step the smallest of its walks' and the combined walk's. Rows join in
    order; one whose walk would cap the step at or below the rows already in
    it is left out, and is first in line next step.
    """

    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        batch = _merged(request.backlog, request.fresh)
        if batch is None:
            return BatchBuildResult(None)

        exclude = request.blocked_rids | off_group_rids(
            batch, request.capture_group,
            exclude_rids=request.blocked_rids, anchor=request.capture_group_of,
        )
        walk = _single_walk(batch, request)
        if walk is None:
            max_bs, left_out = _compose_walks(batch, exclude, request)
            exclude = exclude | left_out
        else:
            cap = request.walk_caps.get(walk)
            max_bs = None if cap is None else cap - request.pre_existing_batch_size
        if max_bs is not None and max_bs <= 0:
            return BatchBuildResult(None, backlog=batch)  # the caller's rows fill it
        scheduled, remainder = batch.split_off_first(max_bs, exclude_rids=exclude)
        return BatchBuildResult(scheduled, backlog=remainder)


def _single_walk(batch: "ScheduledBatch", request: BatchBuildRequest) -> str | None:
    """The one walk of every row, the caller's included, else None."""
    if not batch.request_walks:
        return request.graph_walk
    walks = set(batch.request_walks.values())  # rows and request_walks share keys
    if request.pre_existing_walk is not None:
        walks.add(request.pre_existing_walk)
    return walks.pop() if len(walks) == 1 else None


def _min_cap(*caps: int | None) -> int | None:
    return min((cap for cap in caps if cap is not None), default=None)


def _compose_walks(
    batch: "ScheduledBatch", exclude: set[int], request: BatchBuildRequest,
) -> tuple[int | None, set[int]]:
    """Rows left for the step after the caller's, and the rows left out."""
    caps = request.walk_caps
    n = request.pre_existing_batch_size
    walks = {request.pre_existing_walk} if request.pre_existing_walk else set()
    cap = _min_cap(*(caps.get(walk) for walk in walks))
    left_out = set()
    for rid in batch.request_to_worker_graph:
        if rid in exclude:
            continue
        walk = batch.walk_of(rid)
        new_cap = cap
        if walk not in walks:
            mixed_cap = caps.get(request.graph_walk) if walks else None
            new_cap = _min_cap(cap, caps.get(walk), mixed_cap)
        if new_cap is not None and new_cap <= n:
            left_out.add(rid)
            continue
        walks.add(walk)
        cap = new_cap
        n += 1
    return (None if cap is None else cap - request.pre_existing_batch_size), left_out


class BatchBuilderType(Enum):
    FIFO = "fifo"


def make_batch_builder(kind: BatchBuilderType) -> BaseBatchBuilder:
    return {BatchBuilderType.FIFO: FIFOBatchBuilder}[kind]()
