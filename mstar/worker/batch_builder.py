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


# The fewest tokens `FIT_FIRST` and `ARRIVAL` cut a row to, one KV page at the default page size.
# Without a floor, a prompt behind decode rows that fill the budget would be cut to nothing.
MIN_CHUNK = 128


class BudgetSplit(Enum):
    """How a step's token budget is shared among its rows.

    ``FIT_FIRST`` and ``ARRIVAL`` take the rows that cannot be cut first, then
    the rows partway through their prefill, then the rest, each in arrival
    order. They cut one row a step, to ``MIN_CHUNK`` tokens or more even over
    budget, and keep one prompt of a node partway through at a time. Until
    that prompt finishes, ``FIT_FIRST`` starts only prompts that fit whole and
    ``ARRIVAL`` starts none.
    """
    # an equal share each: rows under it go in whole and the rest split what is left
    MAX_MIN = "max_min"
    # every row that fits goes in whole, and the first that does not takes the leftover
    FIT_FIRST = "fit_first"
    # the first row that does not fit takes the rest, and the rows behind it sit out
    ARRIVAL = "arrival"


@dataclass
class BatchBuildRequest:
    node_name: str
    graph_walk: str
    # Both popped: `backlog` out of the scheduler's backlog, `fresh` off the
    # ready queues. Whatever the builder neither schedules nor returns must come
    # back as `BatchBuildResult.backlog`.
    backlog: "ScheduledBatch | None" = None
    fresh: "ScheduledBatch | None" = None
    # Rows that cannot run this step; a backlogged one stays parked.
    blocked_rids: set[int] = field(default_factory=set)
    # The caller's rows whose step is in flight. They are not popped, and they go first in line.
    # The scheduler neither parks nor returns one that is left out of the step.
    continuing: "ScheduledBatch | None" = None
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
    # Token budget for the step; None for none. With it, each row's remaining
    # tokens, and the rows that may be cut into chunks.
    max_batch_tokens: int | None = None
    row_tokens: dict[int, int] = field(default_factory=dict)
    chunkable: set[int] = field(default_factory=set)
    # How the budget is split. Every split but max-min also reads the node's
    # rows partway through their prefill, in this step or not, and each row's
    # place in arrival order.
    budget_split: BudgetSplit = BudgetSplit.MAX_MIN
    partial_rids: set[int] = field(default_factory=set)
    arrival: dict[int, int] = field(default_factory=dict)
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
    """The caller's ``continuing`` rows, then the backlog, then fresh rows, in
    arrival order, up to the cap; one capture group per step. Everything past
    the cap stays in the backlog.

    Under a combined walk a step of one walk takes that walk's cap and a mixed
    step the smallest of the combined walk's and the caps of the walks present
    in it; a walk with no row in the step does not cap it. Rows join in
    order; one whose walk would cap the step at or below the rows already in
    it is left out, and is first in line next step.
    """

    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        batch = _merged(_merged(request.continuing, request.backlog), request.fresh)
        if batch is None:
            return BatchBuildResult(None)

        exclude = request.blocked_rids | off_group_rids(
            batch, request.capture_group,
            exclude_rids=request.blocked_rids, anchor=request.capture_group_of,
        )
        backlogged = set() if request.backlog is None else set(request.backlog.request_to_worker_graph)
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
        if scheduled is None or request.max_batch_tokens is None:
            return BatchBuildResult(scheduled, backlog=remainder)
        if sum(request.row_tokens.values()) <= request.max_batch_tokens and (
            request.budget_split is not BudgetSplit.ARRIVAL or not request.chunkable
        ):
            # Every measured row fits, so the split takes each row whole. Only `ARRIVAL` can hold back one that fits.
            return BatchBuildResult(scheduled, backlog=remainder, chunk_tokens={})

        rids = list(scheduled.request_to_worker_graph)
        if request.budget_split is not BudgetSplit.MAX_MIN:
            rids.sort(key=lambda rid: request.arrival.get(rid, 0))
        take, dropped = fill_token_budget(
            rids, request.row_tokens, request.chunkable, request.max_batch_tokens,
            request.budget_split, request.partial_rids,
        )
        chunk_tokens = {
            rid: n for rid, n in take.items() if n < request.row_tokens.get(rid, n)
        }
        returned = None
        if dropped:
            # a fresh row that missed the budget goes back to its queue; a backlogged one stays parked
            scheduled, out = scheduled.split_off_first(None, exclude_rids=dropped)
            back, fresh_out = out.split_off_first(
                None, exclude_rids=set(out.request_to_worker_graph) - backlogged,
            )
            if back is not None:
                remainder = back if remainder is None else _merged(remainder, back)
            if fresh_out is not None:
                returned = dict(fresh_out.request_to_worker_graph)
        return BatchBuildResult(
            scheduled, backlog=remainder, returned=returned, chunk_tokens=chunk_tokens,
        )


def fill_token_budget(
    rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
    split: BudgetSplit = BudgetSplit.MAX_MIN, partial: set[int] = frozenset(),
) -> tuple[dict[int, int], set[int]]:
    """Max-min fair share of ``budget`` over ``rids``: rows under an equal share
    go in whole, the rest split what is left (chunkable rows) or sit out.
    Any other ``split`` shares it as described on ``BudgetSplit``, and ``partial``
    is the node's rows partway through their prefill.
    Empty only when every row is a prompt that waits for the one partway
    through. Otherwise the first row runs over budget if it must.
    Returns (rid -> tokens taken, rids left out)."""
    if split is not BudgetSplit.MAX_MIN:
        return _fill_in_order(rids, tokens, chunkable, budget, split, partial)
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


def _fill_in_order(
    rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
    split: BudgetSplit, partial: set[int],
) -> tuple[dict[int, int], set[int]]:
    """`fill_token_budget` for the splits that cut one row a step."""
    ordered = (
        [rid for rid in rids if rid not in chunkable]
        + [rid for rid in rids if rid in chunkable and rid in partial]
    )
    # While a prompt partway through is not in this step, no other prompt is cut, and under ARRIVAL none
    # starts. On a tight pool, prompts started that way each hold pages the others need to finish.
    rid_set = set(rids)
    waiting = not partial <= rid_set
    if split is BudgetSplit.FIT_FIRST or not waiting:
        ordered += [rid for rid in rids if rid in chunkable and rid not in partial]
    take: dict[int, int] = {}
    left = budget
    cut = None
    for rid in ordered:
        need = tokens.get(rid, 0)
        if need <= left:
            take[rid] = need
            left -= need
        elif cut is None and rid in chunkable and (rid in partial or not waiting):
            cut = rid
            if split is BudgetSplit.ARRIVAL:
                break
    if cut is not None:
        take[cut] = min(tokens[cut], max(left, MIN_CHUNK))
    elif not take and ordered and ordered[0] not in chunkable:
        take[ordered[0]] = tokens.get(ordered[0], 0)
    return take, rid_set - take.keys()


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
