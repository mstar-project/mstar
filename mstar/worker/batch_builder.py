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
        batch = _merged(request.backlog, request.fresh)
        if batch is None:
            return BatchBuildResult(None)
        pick = request.pick
        rows = batch.request_to_worker_graph
        taken = [rid for rid in pick.taken if rid in rows]
        if not taken:
            return BatchBuildResult(None, backlog=batch)
        scheduled, remainder = batch.split_off_rids(taken)
        if remainder is None:
            return BatchBuildResult(scheduled)
        give_back = {
            rid for rid in remainder.request_to_worker_graph
            if rid in fresh_rids and rid not in pick.held_back
        } if had_backlog else set()
        if not give_back:
            return BatchBuildResult(scheduled, backlog=remainder)
        parked, returned = remainder.split_off_first(None, exclude_rids=give_back)
        return BatchBuildResult(
            scheduled, backlog=parked,
            returned=dict(returned.request_to_worker_graph) if returned else None,
        )


_NO_GROUP = object()
# `_Composition.admits` verdicts
_JOINS, _FULL, _WALK_LEFT_OUT = object(), object(), object()


def _min_cap(*caps: int | None) -> int | None:
    return min((cap for cap in caps if cap is not None), default=None)


def _all_fit(rows: list[int], chain: set[int], c: StepCandidates) -> bool:
    """Whether one step can take every row in ``rows``, whatever their order:
    within the cap of the walks they hold, and of each walk's rows under a
    step mixing walks."""
    caps = c.walk_caps
    if c.one_walk is not None:
        cap = caps.get(c.one_walk)
        return cap is None or len(rows) <= cap
    counts: dict[str, int] = {}
    for rid in rows:
        walk = c.own_walk if rid in chain else c.walk_of(rid)
        counts[walk] = counts.get(walk, 0) + 1
    if len(counts) == 1:
        (walk,) = counts
        cap = caps.get(walk)
        return cap is None or len(rows) <= cap
    cap = caps.get(c.graph_walk)
    if cap is None:
        cap = _min_cap(*(caps.get(walk) for walk in counts))
    if cap is not None and len(rows) > cap:
        return False
    row_caps = c.walk_row_caps
    return all(n <= row_caps.get(walk, n) for walk, n in counts.items())


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


def make_batch_builder(kind: BatchBuilderType) -> BaseBatchBuilder:
    return {BatchBuilderType.FIFO: FIFOBatchBuilder}[kind]()
