"""Batch composition: which ready rows of one (node, walk) run this step.

The ``MicroScheduler`` picks the (node, walk) and owns the mechanics (popping
ready queues, the backlog, holds, TP follow); a ``BaseBatchBuilder`` decides
which rows the step takes out of a speculation chain's own rows, the backlog
and the ready queues (``pick``), and splits what was popped into the step, the
rows that stay parked in the backlog and the ones that go back to their ready
queues (``build_batch``).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from mstar.worker.micro_scheduler import ScheduledBatch


class LineOrder(NamedTuple):
    """How rows line up for a step that can't take them all."""
    # sort key, lowest first
    key: Callable[[int], Any]
    # Whether a row of another capture group holds the rows behind it in line
    # back (first come, first served), or lets them pass (it is first in line
    # for the next step anyway).
    strict: bool


@dataclass
class StepCandidates:
    """Every row that could run one step of a scheduling key, for ``pick``."""
    # the key's walk: a combined walk's label where the key has one
    graph_walk: str
    # A speculation chain's own rows, all ready and of `own_walk`; empty off a chain.
    continuing: Sequence[int] = ()
    own_walk: str | None = None
    # The live backlogged rows, then the rows on the ready queues, each in its own order.
    backlog: Sequence[int] = ()
    queued: Sequence[int] = ()
    walk_of: Callable[[int], str] = lambda _: ""
    # The walk of every row, when the caller knows they share one.
    one_walk: str | None = None
    # Engine readiness of a backlogged or queued row. Asked once per row, and
    # only of the rows the step reaches.
    ready: Callable[[int], bool] = lambda _: True
    # `Engine.capture_group` for one rid of this key; None when the key's
    # steps don't split by capture key. The chain's rows share one group.
    capture_group: Callable[[int], Any] | None = None
    # Each real walk's cap (None or absent: uncapped); under `graph_walk`, the
    # cap of a step mixing walks.
    walk_caps: dict[str, int | None] = field(default_factory=dict)
    # Under a combined walk: how many rows of each real walk a step mixing
    # walks may hold (absent is uncapped). A step of one walk is not bound.
    walk_row_caps: dict[str, int] = field(default_factory=dict)
    # The node's `StepOrder` as a `LineOrder`, for when the rows don't all
    # fit; not asked otherwise.
    line_order: Callable[[], LineOrder] = lambda: LineOrder(lambda _: 0, False)


class StepPick(NamedTuple):
    # every row the step takes, the chain's included, in the order it took them
    taken: list[int]
    # the chain's rows, those that keep their place first, then the rest in line
    continuing: list[int]
    # how many of `continuing` keep their place; None when nothing caps the step
    room: int | None
    # backlogged or queued rows it passed over for a walk cap or another
    # capture group, which can't run this step whoever holds it
    held_back: set[int]
    # True when the rows didn't all fit and went in the order key's order
    ordered: bool


@dataclass
class BatchBuildRequest:
    node_name: str
    graph_walk: str
    # Both popped: `backlog` out of the scheduler's backlog, `fresh` off the
    # ready queues. Whatever the step doesn't take comes back as
    # `BatchBuildResult.backlog` or `returned`.
    backlog: "ScheduledBatch | None" = None
    fresh: "ScheduledBatch | None" = None
    # `pick`'s verdict over these rows.
    pick: StepPick | None = None
    # The caller's rows whose current chunk is still in flight (not popped):
    # first in line for their next chunk. One left out of the step is neither
    # parked nor returned; it goes back to its queue when its chunk lands.
    continuing: "ScheduledBatch | None" = None
    # Token budget for the step; None for none. With it, each row's remaining
    # tokens, and the rows that may be cut into chunks.
    max_batch_tokens: int | None = None
    # Under a combined walk: walk -> the most tokens that walk's rows may run
    # within the step's budget (can_override_mixed_max_tokens); None for none.
    walk_max_tokens: dict[str, int] | None = None
    row_tokens: dict[int, int] = field(default_factory=dict)
    chunkable: set[int] = field(default_factory=set)
    # The fewest tokens a cut row runs (the node's floor for its configured
    # budget; ``max_batch_tokens`` above is what is left of that budget).
    min_chunk_tokens: int = 1
    # Rows already mid-prefill -> when their first chunk was cut; lower is older.
    start_order: dict[int, int] = field(default_factory=dict)


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
    def pick(self, candidates: StepCandidates) -> StepPick:
        """Who runs one step: the one place a capped step's rows are chosen,
        for a speculation chain and for a step the scheduler builds fresh."""

    @abstractmethod
    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        """One step's rows out of ``request``'s backlog and entries."""

    def chain_must_yield(
        self, backlog: "ScheduledBatch", live_rids: set[int], anchor: int,
        capture_group: Callable[[int], Any],
    ) -> bool:
        """Whether a speculation chain anchored at ``anchor`` has to stop so
        ``backlog``'s live rows can run: a step sharing the chain's capture
        group would never take them."""
        exclude = set(backlog.request_to_worker_graph) - live_rids
        return bool(off_group_rids(
            backlog, capture_group, exclude_rids=exclude, anchor=anchor,
        ))


class FIFOBatchBuilder(BaseBatchBuilder):
    """One capture group per step, up to the cap; who goes first when the
    rows don't all fit is the node's `StepOrder`.

    When every row fits, the step takes them in the order backlog, the chain's
    own rows, then the queues: nothing is left out, so the order only decides
    the capture group, which the chain's or else the first row's sets, and a
    row of another group waits for a step of its own. When they don't all
    fit, every row goes in the order's sort and the first sets the group. A
    row of another group then waits for a step of its own: under TURNS the
    rows behind it pass it, since it leads the next step; under ARRIVAL they
    don't, or rows that came later would join the step's holders and keep it
    from it.

    Under a combined walk a step of one walk takes that walk's cap and a mixed
    step the combined walk's own, or with none the smallest of the caps of the
    walks present in it; a walk with no row in the step does not cap it. Rows
    join in order; one whose walk would cap the step at or below the rows
    already in it is left out.

    Of the rows ``build_batch`` gets, a fresh row the cap pushed out goes back
    to its ready queue **only on a step that served a backlog**. Parking it
    would refill the backlog from the queues on every pass, so it would never
    drain, and since a key holding a backlog is served before any other node,
    that node starves the rest: on ``wan22.yaml`` with ``dit`` capped at 1 a
    fifth request's ``text_encoder`` waited 48.7 s for its first step against
    0.2 s before backlogs were topped up. With no backlog it parks, so a split
    set drains before anything new starts. A row held back for any other
    reason -- OOM backoff, pending remove, a foreign capture group, a walk the
    combined cap left out -- always parks: it cannot run this step whoever
    holds it.

    Under a token budget, of the rows the pick took, prompts are served in
    the order they started (``fill_token_budget_in_start_order``), and the
    caller's rows mid-prefill come first. A row the budget leaves out always
    parks, ahead of the cap's leftovers, because a budgeted walk's backlog is
    its line: the Rust runtime scans ready rows in handle order, so a row
    sent back rejoins behind rows that arrived after it. Such a backlog can
    stay non-empty under sustained load; the worker's fairness peek still
    hands the step after a prompt step to other ready work.
    """

    def pick(self, candidates: StepCandidates) -> StepPick:
        c = candidates
        chain = set(c.continuing)
        rows = [*c.backlog, *c.continuing, *c.queued]
        ordered = not _all_fit(rows, chain, c)
        strict = False
        if ordered:
            line = c.line_order()
            rows.sort(key=line.key)  # stable: ties keep the order above
            strict = line.strict
        group_of = c.capture_group
        if not ordered and group_of is None and c.one_walk is not None:
            # every row fits and nothing splits them: the chain keeps its rows
            # as they are, and the others join if ready
            ready = c.ready
            taken = [rid for rid in c.backlog if ready(rid)]
            taken += c.continuing
            taken += (rid for rid in c.queued if rid not in chain and ready(rid))
            cap = c.walk_caps.get(c.one_walk)
            return StepPick(
                taken=taken, continuing=list(c.continuing),
                room=len(c.continuing) if cap is not None else None,
                held_back=set(), ordered=False,
            )
        chain_group = _NO_GROUP
        if group_of is not None and c.continuing:
            chain_group = group_of(c.continuing[0])
        # every row fits: the chain's group, else the first row's
        group = _NO_GROUP if ordered else chain_group
        step = None if c.one_walk is not None else _Composition(c)
        cap = c.walk_caps.get(c.one_walk) if step is None else None
        taken: list[int] = []
        held_back: set[int] = set()
        for rid in rows:
            in_chain = rid in chain
            if step is None:
                if cap is not None and len(taken) >= cap:
                    break
            else:
                if step.is_full():
                    break
                walk = c.own_walk if in_chain else c.walk_of(rid)
                verdict = step.admits(walk)
                if verdict is not _JOINS:
                    if verdict is _WALK_LEFT_OUT:
                        held_back.add(rid)
                    continue
            if not in_chain and not c.ready(rid):
                continue
            if group_of is not None:
                rid_group = chain_group if in_chain else group_of(rid)
                if group is _NO_GROUP:
                    group = rid_group
                elif rid_group != group:
                    if strict:
                        break
                    held_back.add(rid)
                    continue
            if step is not None:
                step.add(walk)
            taken.append(rid)
        kept = [rid for rid in taken if rid in chain] if chain else []
        behind = []
        if len(kept) < len(chain):
            kept_set = set(kept)
            behind = [rid for rid in rows if rid in chain and rid not in kept_set]
        capped = cap is not None if step is None else step.capped()
        return StepPick(
            taken=taken,
            continuing=kept + behind,
            room=len(kept) if capped or behind else None,
            held_back=held_back - chain if chain else held_back,
            ordered=ordered,
        )

    def build_batch(self, request: BatchBuildRequest) -> BatchBuildResult:
        # both read before `_merged`, which folds `fresh` into `backlog` in place
        had_backlog = request.backlog is not None
        fresh_rids = (
            set(request.fresh.request_to_worker_graph) if request.fresh else set()
        )
        batch = _merged(_merged(request.continuing, request.backlog), request.fresh)
        if batch is None:
            return BatchBuildResult(None)
        pick = request.pick
        rows = batch.request_to_worker_graph
        taken = [rid for rid in pick.taken if rid in rows]
        if not taken:
            return BatchBuildResult(None, backlog=batch)
        scheduled, remainder = batch.split_off_rids(taken)
        returned: dict[int, int] | None = None
        if remainder is not None and had_backlog:
            give_back = {
                rid for rid in remainder.request_to_worker_graph
                if rid in fresh_rids and rid not in pick.held_back
            }
            if give_back:
                remainder, back = remainder.split_off_first(None, exclude_rids=give_back)
                returned = dict(back.request_to_worker_graph) if back else None
        if request.max_batch_tokens is None:
            return BatchBuildResult(scheduled, backlog=remainder, returned=returned)

        take, dropped = self.share_token_budget(
            list(scheduled.request_to_worker_graph), request.row_tokens,
            request.chunkable, request.max_batch_tokens, request.min_chunk_tokens,
            request.start_order,
        )
        if request.walk_max_tokens:
            _cap_walks(
                scheduled, take, dropped, request.chunkable, request.walk_max_tokens,
                self.share_token_budget, request.min_chunk_tokens, request.start_order,
            )
        chunk_tokens = {
            rid: n for rid, n in take.items() if n < request.row_tokens.get(rid, n)
        }
        if dropped:
            # parked ahead of the cap's leftovers (see the class docstring)
            scheduled, out = scheduled.split_off_first(None, exclude_rids=dropped)
            remainder = _merged(out, remainder)
        return BatchBuildResult(
            scheduled, backlog=remainder, returned=returned, chunk_tokens=chunk_tokens,
        )

    @staticmethod
    def share_token_budget(
        rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
        min_chunk: int, start_order: dict[int, int],
    ) -> tuple[dict[int, int], set[int]]:
        """How the step's rows share its token budget: (rid -> tokens taken,
        rids left out)."""
        return fill_token_budget_in_start_order(
            rids, tokens, chunkable, budget, min_chunk, start_order,
        )


class FairShareBatchBuilder(FIFOBatchBuilder):
    """``FIFOBatchBuilder`` with the token budget shared max-min fair
    (``fill_token_budget``): every waiting prompt advances each step, so a
    long prompt's first token waits on every other prompt's prefill too."""

    @staticmethod
    def share_token_budget(
        rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
        min_chunk: int, start_order: dict[int, int],
    ) -> tuple[dict[int, int], set[int]]:
        return fill_token_budget(rids, tokens, chunkable, budget, min_chunk)


def _take_whole(
    rids: list[int], tokens: dict[int, int], budget: int,
) -> tuple[dict[int, int], int, list[int]]:
    """The rows of ``rids`` under an equal share of ``budget``, taken whole,
    the share recomputed over the rest until none fits. Returns (rid ->
    tokens taken, the budget left, the rows not taken)."""
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
    return take, left, pending


def _run_head(
    rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
    min_chunk: int, take: dict[int, int], dropped: set[int],
) -> None:
    """Never an empty step while a row exists: the first of ``rids`` runs
    over budget if it must. Updates ``take`` and ``dropped`` in place."""
    if take or not rids:
        return
    head = rids[0]
    need = tokens.get(head, 0)
    take[head] = min(need, max(budget, min_chunk, 1)) if head in chunkable else need
    dropped.discard(head)


def fill_token_budget_in_start_order(
    rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
    min_chunk: int = 1, start_order: dict[int, int] | None = None,
) -> tuple[dict[int, int], set[int]]:
    """``budget`` over ``rids`` first come, first served. Rows under an equal
    share go in whole, and then any row that can finish in what is left,
    shortest first, so a short prompt never waits behind a long one. The rest
    follow in start order -- prompts already mid-prefill by when they started
    (``start_order``), then the others as listed -- each taking all it needs
    of what is left, so a prompt finishes before the next one starts and its
    first token does not wait on every other prompt's prefill. The first that
    does not fit is cut to what is left if that is at least ``min_chunk``
    tokens, and otherwise waits for the next step; one that cannot be cut sits
    out without holding up the rows behind it. Never empty while a row
    exists: the first row runs over budget if it must.

    The oldest prompt mid-prefill keeps ``min_chunk`` of the budget out of
    the finishing pass, so a steady stream of short prompts cannot starve it.
    Returns (rid -> tokens taken, rids left out)."""
    take, left, pending = _take_whole(rids, tokens, budget)
    start_order = start_order or {}
    oldest = min(
        (rid for rid in pending if rid in start_order and rid in chunkable),
        key=start_order.__getitem__, default=None,
    )
    reserved = min(min_chunk, tokens.get(oldest, 0)) if oldest is not None and left >= min_chunk else 0
    left -= reserved
    for rid in sorted(pending, key=lambda rid: tokens.get(rid, 0)):
        need = tokens.get(rid, 0)
        if need > left or rid == oldest:
            continue
        take[rid] = need
        left -= need
    left += reserved
    pending = [rid for rid in pending if rid not in take]
    started = sorted((rid for rid in pending if rid in start_order), key=start_order.__getitem__)
    order = started + [rid for rid in pending if rid not in start_order]
    dropped: set[int] = set()
    cut = False
    for rid in order:
        need = tokens.get(rid, 0)
        if need <= left:
            take[rid] = need
            left -= need
        elif rid in chunkable and not cut and left >= min_chunk:
            # the one row cut. Not to the floor past the budget: a step over
            # its budget pays for the next capture bucket up
            cut = True
            take[rid] = left
            left = 0
        else:
            dropped.add(rid)
    _run_head(order, tokens, chunkable, budget, min_chunk, take, dropped)
    return take, dropped


def fill_token_budget(
    rids: list[int], tokens: dict[int, int], chunkable: set[int], budget: int,
    min_chunk: int = 1,
) -> tuple[dict[int, int], set[int]]:
    """Max-min fair share of ``budget`` over ``rids``: rows under an equal share
    go in whole, the rest split what is left (chunkable rows) or sit out.
    A cut row takes at least ``min_chunk`` tokens, or all it has left if that
    is fewer: when the share is smaller, only as many rows as what is left
    holds are cut, and if it holds none, they wait for the next step.
    Never empty while a row exists: the first row runs over budget if it must.
    Returns (rid -> tokens taken, rids left out)."""
    take, left, pending = _take_whole(rids, tokens, budget)
    dropped = {rid for rid in pending if rid not in chunkable}
    cut = [rid for rid in pending if rid in chunkable]
    share = left // len(pending) if pending else 0
    if cut and share < min_chunk:
        n = min(len(cut), max(left, 0) // min_chunk)
        share = max(left // n, min_chunk) if n else 0
        dropped.update(cut[n:])
        cut = cut[:n]
    for rid in cut:
        if share >= 1:
            take[rid] = min(tokens.get(rid, 0), share)
        else:
            dropped.add(rid)
    _run_head(rids, tokens, chunkable, budget, min_chunk, take, dropped)
    return take, dropped


def _cap_walks(
    batch: "ScheduledBatch", take: dict[int, int], dropped: set[int],
    chunkable: set[int], caps: dict[str, int],
    share: Callable[..., tuple[dict[int, int], set[int]]],
    min_chunk: int = 1, start_order: dict[int, int] | None = None,
) -> None:
    """Hold each capped walk's rows to its cap as well as the step's budget:
    those rows share the cap as ``share`` (the builder's) shares a budget,
    from what the step's budget gave them. A token one gives up is not handed
    to another walk. Updates ``take`` and ``dropped`` in place."""
    for walk, cap in caps.items():
        rids = [rid for rid in batch.request_to_worker_graph if rid in take and batch.walk_of(rid) == walk]
        if sum(take[rid] for rid in rids) <= cap:
            continue
        walk_take, walk_dropped = share(
            rids, {rid: take[rid] for rid in rids}, chunkable, cap, min(min_chunk, cap),
            start_order or {},
        )
        for rid in rids:
            del take[rid]
        take.update(walk_take)
        dropped |= walk_dropped


_NO_GROUP = object()
# `_Composition.admits` verdicts
_JOINS, _FULL, _WALK_LEFT_OUT = object(), object(), object()


def _min_cap(*caps: int | None) -> int | None:
    return min((cap for cap in caps if cap is not None), default=None)


def _all_fit(rows: list[int], chain: set[int], c: StepCandidates) -> bool:
    """Whether a step taking ``rows`` in this order would hold them all."""
    if c.one_walk is not None:
        cap = c.walk_caps.get(c.one_walk)
        return cap is None or len(rows) <= cap
    step = _Composition(c)
    for rid in rows:
        walk = c.own_walk if rid in chain else c.walk_of(rid)
        if step.admits(walk) is not _JOINS:
            return False
        step.add(walk)
    return True


class _Composition:
    """The rows a step has taken so far, by walk, and whether a row of a
    given walk can join them: the composition rules of ``FIFOBatchBuilder``."""

    def __init__(self, c: StepCandidates):
        self.caps = c.walk_caps
        self.row_caps = c.walk_row_caps
        self.label = c.graph_walk
        self.walks: set[str] = set()
        self.counts: dict[str, int] = {}
        self.cap: int | None = None
        self.n = 0
        caps = list(self.caps.values())
        # no step of any mix of walks holds more
        self._most = None if not caps or None in caps else max(caps)
        self._next_cap: int | None = None

    def admits(self, walk: str) -> object:
        """``_JOINS``; ``_FULL`` when the step is at its cap; or
        ``_WALK_LEFT_OUT`` when the row's walk can't join a step like this one
        (it would lower the cap to the rows already in it, or pass its row cap)."""
        new_cap = self.cap
        if walk not in self.walks:
            mixed_cap = self.caps.get(self.label) if self.walks else None
            # a mixed step: the combined walk's own cap where it has one,
            # else the smallest of the walks present
            new_cap = mixed_cap if mixed_cap is not None else _min_cap(self.cap, self.caps.get(walk))
        if new_cap is not None and new_cap <= self.n:
            return _FULL if walk in self.walks else _WALK_LEFT_OUT
        if self.row_caps and self.walks:
            if walk not in self.walks:
                # the row makes the step mix walks: every walk's rows count
                over = any(
                    self.counts.get(w, 0) + (w == walk) > self.row_caps[w]
                    for w in (*self.walks, walk) if w in self.row_caps
                )
            else:
                # already mixed or not, only this walk's count moves
                cap_w = self.row_caps.get(walk)
                over = len(self.walks) > 1 and cap_w is not None and self.counts[walk] + 1 > cap_w
            if over:
                return _WALK_LEFT_OUT
        self._next_cap = new_cap
        return _JOINS

    def add(self, walk: str) -> None:
        """Take a row of ``walk``; ``admits(walk)`` came just before."""
        self.walks.add(walk)
        self.counts[walk] = self.counts.get(walk, 0) + 1
        self.cap = self._next_cap
        self.n += 1

    def is_full(self) -> bool:
        return self._most is not None and self.n >= self._most

    def capped(self) -> bool:
        return self.cap is not None


class BatchBuilderType(Enum):
    FIFO = "fifo"
    FAIR_SHARE = "fair_share"


def make_batch_builder(kind: BatchBuilderType) -> BaseBatchBuilder:
    return {
        BatchBuilderType.FIFO: FIFOBatchBuilder,
        BatchBuilderType.FAIR_SHARE: FairShareBatchBuilder,
    }[kind]()
