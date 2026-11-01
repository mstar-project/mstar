"""Batch composition: which ready rows of one (node, walk) run this step.

The ``MicroScheduler`` picks the (node, walk) and owns the mechanics (popping
ready queues, the backlog, holds, TP follow); a batch builder decides which of
the popped rows go into the step, which stay parked in the backlog and which
go back to their ready queues.

A builder has a Python backend (a ``BaseBatchBuilder``, for the Python
scheduler core), a Rust one (``rust/src/batch_builder.rs``, for the Rust
core), or both; ``BATCH_BUILDERS`` says which.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

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
    # Rows that cannot run this step (waiting, not ready, or outside the
    # step's capture group); they stay parked.
    blocked_rids: set[int] = field(default_factory=set)
    # Backlogged rows past the readiness check's stop: their readiness is
    # unknown, so they must not run this step either, but they keep their
    # place in the backlog ahead of the blocked rows.
    deferred_rids: set[int] = field(default_factory=set)
    # Each real walk's cap (None or absent: uncapped); under `graph_walk`, the
    # cap of a step mixing walks.
    walk_caps: dict[str, int | None] = field(default_factory=dict)
    # Under a combined walk: how many rows of each real walk a step mixing
    # walks may hold (absent is uncapped). A step of one walk is not bound.
    walk_row_caps: dict[str, int] = field(default_factory=dict)
    # The caller's own rows (a speculation merge's continuing ones) and their walk.
    pre_existing_batch_size: int = 0
    pre_existing_walk: str | None = None


class BatchBuildResult(NamedTuple):
    scheduled: "ScheduledBatch | None"
    # Parked under this (node, walk) for a later step.
    backlog: "ScheduledBatch | None" = None
    # Popped rows handed back to their ready queues: rid -> worker graph id.
    returned: dict[int, int] | None = None


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


class FIFOBatchBuilder(BaseBatchBuilder):
    """Backlog first, then fresh rows, in arrival order, up to the cap; one
    capture group per step.

    Under a combined walk a step of one walk takes that walk's cap and a mixed
    step the combined walk's own, or with none the smallest of the caps of the
    walks present in it; a walk with no row in the step does not cap it. Rows join in
    order; one whose walk would cap the step at or below the rows already in
    it is left out, and is first in line next step.

    Where this differs from the scheduler it replaces, and why. main had two
    paths: with a backlog it drained that alone, never touching the ready
    queues, so the backlog strictly shrank; with none it popped a fresh set,
    capped it and parked the remainder. This builder always does both, which
    is better -- a backlog one row short of the cap gets topped up instead of
    running short -- but it means a step that drains a backlog also pops fresh
    rows, and parking their overflow would refill the backlog from the queues
    on every pass. It would then never drain, and since a key holding a
    backlog is served before any other node, that node starves the rest: on
    ``wan22.yaml`` with ``dit`` capped at 1 a fifth request's
    ``text_encoder`` waited 48.7 s for its first step against 0.2 s on main.

    So a fresh row the cap pushed out goes back to its ready queue **only on
    a step that served a backlog**, which is the case main never created. With
    no backlog it parks, exactly as main did, and the remainder keeps its
    arrival order instead of being re-popped and reordered. A row held back
    for any other reason -- OOM backoff, pending remove, a foreign capture
    group, a walk the combined cap left out -- always parks: it cannot run
    this step whoever holds it.
    """

    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        # both read before `_merged`, which folds `fresh` into `backlog` in place
        had_backlog = request.backlog is not None
        fresh_rids = (
            set(request.fresh.request_to_worker_graph) if request.fresh else set()
        )
        batch = _merged(request.backlog, request.fresh)
        if batch is None:
            return BatchBuildResult(None)

        exclude = set(request.blocked_rids)
        deferred = request.deferred_rids
        walk = _single_walk(batch, request)
        if walk is None:
            max_bs, left_out = _compose_walks(
                batch, exclude | deferred if deferred else exclude, request,
            )
            exclude = exclude | left_out
        else:
            cap = request.walk_caps.get(walk)
            max_bs = None if cap is None else cap - request.pre_existing_batch_size
        if max_bs is not None and max_bs <= 0:
            return BatchBuildResult(None, backlog=batch)  # the caller's rows fill it
        scheduled, remainder = batch.split_off_first(
            max_bs, exclude_rids=exclude, deferred_rids=deferred,
        )
        if remainder is None:
            return BatchBuildResult(scheduled)
        give_back = {
            rid for rid in remainder.request_to_worker_graph
            if rid in fresh_rids and rid not in exclude
        } if had_backlog else set()
        if not give_back:
            return BatchBuildResult(scheduled, backlog=remainder)
        parked, returned = remainder.split_off_first(None, exclude_rids=give_back)
        return BatchBuildResult(
            scheduled, backlog=parked,
            returned=dict(returned.request_to_worker_graph) if returned else None,
        )


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
    """Rows left for the step after the caller's, and the rows left out.
    Rows in ``exclude`` are neither counted nor left out."""
    caps = request.walk_caps
    row_caps = request.walk_row_caps
    n = request.pre_existing_batch_size
    walks = {request.pre_existing_walk} if request.pre_existing_walk else set()
    counts = {request.pre_existing_walk: n} if request.pre_existing_walk else {}
    cap = _min_cap(*(caps.get(walk) for walk in walks))
    left_out = set()
    for rid in batch.request_to_worker_graph:
        if rid in exclude:
            continue
        walk = batch.walk_of(rid)
        new_cap = cap
        if walk not in walks:
            mixed_cap = caps.get(request.graph_walk) if walks else None
            # a mixed step: the combined walk's own cap where it has one,
            # else the smallest of the walks present
            new_cap = mixed_cap if mixed_cap is not None else _min_cap(cap, caps.get(walk))
        if new_cap is not None and new_cap <= n:
            left_out.add(rid)
            continue
        if row_caps and walks:
            if walk not in walks:
                # the row makes the step mix walks: every walk's rows count
                over = any(
                    counts.get(w, 0) + (w == walk) > row_caps[w]
                    for w in (*walks, walk) if w in row_caps
                )
            else:
                # already mixed or not, only this walk's count moves
                cap_w = row_caps.get(walk)
                over = len(walks) > 1 and cap_w is not None and counts[walk] + 1 > cap_w
            if over:
                # the step would mix walks past one's row cap
                left_out.add(rid)
                continue
        walks.add(walk)
        counts[walk] = counts.get(walk, 0) + 1
        cap = new_cap
        n += 1
    return (None if cap is None else cap - request.pre_existing_batch_size), left_out


class BatchBuilderType(Enum):
    FIFO = "fifo"


class BatchBuilderBackends(NamedTuple):
    """A builder's implementations; either may be missing."""
    python: "type[BaseBatchBuilder] | None"
    # the name ``rust/src/batch_builder.rs``'s ``by_name`` knows it by
    rust: str | None


BATCH_BUILDERS: dict[BatchBuilderType, BatchBuilderBackends] = {
    BatchBuilderType.FIFO: BatchBuilderBackends(python=FIFOBatchBuilder, rust="fifo"),
}
