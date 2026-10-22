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
    # Each real walk's cap (None or absent: uncapped); under `graph_walk`, the
    # cap of a step mixing walks.
    walk_caps: dict[str, int | None] = field(default_factory=dict)
    # Under a combined walk: how many rows of each real walk a step mixing
    # walks may hold (absent is uncapped). A step of one walk is not bound.
    walk_row_caps: dict[str, int] = field(default_factory=dict)
    # The caller's own rows (a speculation merge's continuing ones) and their walk.
    pre_existing_batch_size: int = 0
    pre_existing_walk: str | None = None
    # Token budget for the step; None for none. With it, each row's remaining
    # tokens, and the rows that may be cut into chunks.
    max_batch_tokens: int | None = None
    # Under a combined walk: walk -> the most tokens that walk's rows may run
    # within the step's budget (can_override_mixed_max_tokens); None for none.
    walk_max_tokens: dict[str, int] | None = None
    row_tokens: dict[int, int] = field(default_factory=dict)
    chunkable: set[int] = field(default_factory=set)
    # `Engine.capture_group` for one rid of this (node, walk).
    capture_group: Callable[[int], Any] = lambda _: None


class BatchBuildResult(NamedTuple):
    scheduled: "ScheduledBatch | None"
    # Parked under this (node, walk) for a later step.
    backlog: "ScheduledBatch | None" = None
    # Popped rows handed back to their ready queues: rid -> worker graph id.
    returned: dict[int, int] | None = None
    # Rows cut short this step: rid -> tokens to run.
    chunk_tokens: dict[int, int] | None = None


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
        # A fresh row the cap pushed out goes back to its ready queue only on
        # a step that served a backlog; with none it parks, as main did. The
        # token budget below does the same for the rows it drops, so both
        # kinds of leftover accumulate into `returned`.
        returned: dict[int, int] | None = None
        if remainder is not None and had_backlog:
            give_back = {
                rid for rid in remainder.request_to_worker_graph
                if rid in fresh_rids and rid not in exclude
            }
            if give_back:
                remainder, back = remainder.split_off_first(
                    None, exclude_rids=give_back,
                )
                returned = dict(back.request_to_worker_graph) if back else None

        if scheduled is None or request.max_batch_tokens is None:
            return BatchBuildResult(scheduled, backlog=remainder, returned=returned)

        take, dropped = fill_token_budget(
            list(scheduled.request_to_worker_graph), request.row_tokens,
            request.chunkable, request.max_batch_tokens,
        )
        if request.walk_max_tokens:
            _cap_walks(scheduled, take, dropped, request.chunkable, request.walk_max_tokens)
        chunk_tokens = {
            rid: n for rid, n in take.items() if n < request.row_tokens.get(rid, n)
        }
        if dropped:
            # a fresh row that missed the budget goes back to its queue; a backlogged one stays parked
            scheduled, out = scheduled.split_off_first(None, exclude_rids=dropped)
            backlogged = () if request.backlog is None else request.backlog.request_to_worker_graph
            back, fresh_out = out.split_off_first(
                None, exclude_rids=set(out.request_to_worker_graph).difference(backlogged),
            )
            if back is not None:
                remainder = back if remainder is None else _merged(remainder, back)
            if fresh_out is not None:
                # accumulated, not replaced: the cap may already have given rows back
                returned = (returned or {}) | dict(fresh_out.request_to_worker_graph)
        return BatchBuildResult(
            scheduled, backlog=remainder, returned=returned, chunk_tokens=chunk_tokens,
        )


def fill_token_budget(
    rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
) -> tuple[dict[int, int], set[int]]:
    """Max-min fair share of ``budget`` over ``rids``: rows under an equal share
    go in whole, the rest split what is left (chunkable rows) or sit out.
    Never empty while a row exists: the first row runs over budget if it must.
    Returns (rid -> tokens taken, rids left out)."""
    take: dict[int, int] = {}
    left = budget
    pending = list(rids)
    while pending:
        share = left // len(pending)
        fits = [rid for rid in pending if tokens.get(rid, 0) <= share]
        if not fits:
            break
        for rid in fits:
            take[rid] = tokens.get(rid, 0)
            left -= take[rid]
        pending = [rid for rid in pending if rid not in take]
    dropped: set[int] = set()
    share = left // len(pending) if pending else 0
    for rid in pending:
        if rid in chunkable and share >= 1:
            take[rid] = share
        else:
            dropped.add(rid)
    if not take and rids:
        head = rids[0]
        need = tokens.get(head, 0)
        take[head] = min(need, max(budget, 1)) if head in chunkable else need
        dropped.discard(head)
    return take, dropped


def _cap_walks(
    batch: "ScheduledBatch", take: dict[int, int], dropped: set[int],
    chunkable: set[int], caps: dict[str, int],
) -> None:
    """Hold each capped walk's rows to its cap as well as the step's budget:
    those rows share the cap as ``fill_token_budget`` shares a budget, from
    what the step's budget gave them. A token one gives up is not handed to
    another walk. Updates ``take`` and ``dropped`` in place."""
    for walk, cap in caps.items():
        rids = [rid for rid in batch.request_to_worker_graph if rid in take and batch.walk_of(rid) == walk]
        if sum(take[rid] for rid in rids) <= cap:
            continue
        walk_take, walk_dropped = fill_token_budget(rids, {rid: take[rid] for rid in rids}, chunkable, cap)
        for rid in rids:
            del take[rid]
        take.update(walk_take)
        dropped |= walk_dropped


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


def make_batch_builder(kind: BatchBuilderType) -> BaseBatchBuilder:
    return {BatchBuilderType.FIFO: FIFOBatchBuilder}[kind]()
