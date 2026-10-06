"""Future-peak planning for KV admission, in pages. Pure Python and numpy.

The summed test the pool starts from admits a request only if every admitted
request could take all it reserved at once. That is safe for any order of
progress, and it refuses sets whose reservations never coincide: a short
request that would be gone before a long one reaches its peak is turned away
because the sums add up. This module asks the other question: given how many
decode rounds each admitted request has left, what is the most pages the set
can hold at any one round?

Each request is a ``PlanEntry``. Its footprint at round ``t`` from now is

    min(max(claim, held), held + now + growth * ceil(t / page_size))   for t < rounds

and 0 from ``rounds`` on, when it is gone. ``now`` is the prompt not yet
allocated, which the plan puts at round 0 (the prefill runs first). ``growth``
is the number of decode labels, each adding a token a round, so a label's
pages grow by ``ceil(t / page_size)`` at most. The cap is the reservation: a
request never takes more from the free list than it reserved. This is an upper
bound on the pages the request holds at round ``t`` when every request
advances one token a round together; it does not hold when they drift apart,
which is what ``banker_safe`` is for. The plan decides *how much to admit*;
the safe-state test decides *whether a page may be handed out*, whatever the
order of progress.

``rounds == 0`` is a request that has finished and not been removed: its
release time is unknown, so it holds ``min(cap, held + now)`` for ever.

A candidate or a head always runs at least one round (``max(1, rounds)``).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class PlanEntry(NamedTuple):
    # fresh pages held now: what it gives back when it finishes
    held: int
    # most fresh pages it may hold over its life (its reservation)
    claim: int
    # pages it takes at once on top of ``held``: its prompt, not yet allocated
    now: int
    # decode labels: pages added per ``page_size`` rounds
    growth: int
    # decode rounds left; 0 = finished, not yet removed
    rounds: int

    @property
    def need(self) -> int:
        """Pages it may still take, as the safe-state test counts them."""
        return max(0, self.claim - self.held)


def footprint(entry: PlanEntry, t: int, page_size: int) -> int:
    """The pages ``entry`` holds ``t`` rounds from now, by the bound above."""
    if entry.rounds > 0 and t >= entry.rounds:
        return 0
    cap = max(entry.claim, entry.held)
    grown = 0 if entry.rounds <= 0 else entry.growth * (-(-t // page_size))
    return min(cap, entry.held + entry.now + grown)


def banker_safe(free: int, held: Sequence[int], need: Sequence[int]) -> bool:
    """Whether the requests can finish one after another, in some order.

    A request can finish when what it may still take is within the free pages
    plus what the ones before it gave back, and finishing returns everything
    it holds. Finishing never lowers the pages available, so any request that
    can finish can go first, and trying them in order of need is exact: if the
    one needing least cannot go, none can. A safe state always has a request
    that can finish, so no set of admitted requests is ever stuck.
    """
    n = len(need)
    if n == 0:
        return free >= 0
    need_a = np.asarray(need, dtype=np.int64)
    held_a = np.asarray(held, dtype=np.int64)
    order = np.argsort(need_a, kind="stable")
    sorted_held = held_a[order]
    # what is free when the k-th request goes: the free pages, and every one before it
    avail = free + np.cumsum(sorted_held) - sorted_held
    return bool((need_a[order] <= avail).all())


class PeakPlanner:
    """The most pages a set of admitted requests can hold at any round.

    Built once per change to the set, then a candidate is a vectorized pass
    over the cached usage at each release time. ``capacity`` is the pages the
    set may use.
    """

    def __init__(
        self, entries: Sequence[PlanEntry], capacity: int, page_size: int,
    ):
        self.capacity = capacity
        self.page_size = page_size
        self.entries = tuple(entries)
        live = [e for e in self.entries if e.rounds > 0]
        self.frozen = sum(
            min(max(e.claim, e.held), e.held + e.now)
            for e in self.entries if e.rounds <= 0
        )
        self._base = np.array([e.held + e.now for e in live], dtype=np.int64)
        self._cap = np.array([max(e.claim, e.held) for e in live], dtype=np.int64)
        self._growth = np.array([e.growth for e in live], dtype=np.int64)
        self._rounds = np.array([e.rounds for e in live], dtype=np.int64)
        # usage only falls at a release and never falls between two, so its
        # maximum is at round 0 or the round before some request leaves
        self.times = np.unique(np.concatenate(([0], self._rounds - 1)))
        self.usage = self._usage(self.times)
        # the most from each time on
        self._later = np.maximum.accumulate(self.usage[::-1])[::-1]
        self.held_total = sum(e.held for e in self.entries)
        self._glance: tuple[list[int], list[int], list[int]] | None = None

    @classmethod
    def from_arrays(
        cls, held: np.ndarray, claim: np.ndarray, now: np.ndarray,
        growth: np.ndarray, rounds: np.ndarray, capacity: int, page_size: int,
    ) -> PeakPlanner:
        """The planner for the set whose `PlanEntry` fields are these int64 arrays,
        one slot a request, with every answer the list constructor gives for them.

        For a set kept as arrays, so no entry is made to build it: it has no ``entries``.
        """
        self = cls.__new__(cls)
        self.capacity = capacity
        self.page_size = page_size
        cap = np.maximum(claim, held)
        base = held + now
        if np.count_nonzero(rounds) == len(rounds):
            self.frozen = 0
            self._base, self._cap, self._growth, self._rounds = base, cap, growth, rounds
        else:
            live = rounds > 0
            self.frozen = int(np.minimum(cap, base)[~live].sum())
            self._base = base[live]
            self._cap = cap[live]
            self._growth = growth[live]
            self._rounds = rounds[live]
        # `np.unique(np.concatenate(([0], rounds - 1)))`: sorted by Python's, which is quicker for what is a few dozen
        self.times = np.array(sorted({0, *(self._rounds - 1).tolist()}), dtype=np.int64)
        # `_usage(times)`, in fewer calls
        steps = (self.times + (page_size - 1)) // page_size
        grown = np.multiply.outer(self._growth, steps)
        grown += self._base[:, None]
        np.minimum(grown, self._cap[:, None], out=grown)
        alive = np.greater.outer(self._rounds, self.times)
        self.usage = self.frozen + grown.sum(axis=0, where=alive)
        self._later = np.maximum.accumulate(self.usage[::-1])[::-1]
        self.held_total = int(held.sum())
        self._glance = None
        return self

    # ── usage ───────────────────────────────────────────────────────────

    def _usage(self, ts: np.ndarray) -> np.ndarray:
        steps = -(-ts // self.page_size)
        grown = self._base[:, None] + self._growth[:, None] * steps[None, :]
        pages = np.minimum(self._cap[:, None], grown)
        alive = self._rounds[:, None] > ts[None, :]
        return self.frozen + (pages * alive).sum(axis=0)

    def usage_at(self, t: int) -> int:
        """The set's pages ``t`` rounds from now."""
        return int(self._usage(np.array([t], dtype=np.int64))[0])

    def peak(self) -> int:
        """The most pages the set holds at any round (frozen ones included)."""
        return int(self._later[0])

    # ── a request added to the set ──────────────────────────────────────

    def peak_from(self, entry: PlanEntry, start: int = 0) -> int:
        """The set's most pages with ``entry`` added and running from round ``start``.

        ``entry`` is counted from ``start``; every request in the set is aged
        by it. Only the rounds the plan can peak at are looked at: the ones
        just before a release, and the round before the entry leaves.
        """
        rounds = max(1, entry.rounds)
        cap = max(entry.claim, entry.held)
        base = entry.held + entry.now
        ps = self.page_size
        end = start + rounds
        lo = int(np.searchsorted(self.times, start, side="left"))
        hi = int(np.searchsorted(self.times, end, side="left"))
        peak = 0
        if hi > lo:
            ages = self.times[lo:hi] - start
            mine = np.minimum(cap, base + entry.growth * (-(-ages // ps)))
            peak = int((self.usage[lo:hi] + mine).max())
        if hi == lo or int(self.times[hi - 1]) != end - 1:
            mine_last = min(cap, base + entry.growth * (-(-(rounds - 1) // ps)))
            peak = max(peak, self.usage_at(end - 1) + mine_last)
        # once it has gone, the set alone
        peak = max(peak, int(self._later[hi]) if hi < len(self.times) else self.frozen)
        return peak

    def exceeds(self, entry: PlanEntry, capacity: int) -> bool:
        """Whether ``peak_from(entry)`` is over ``capacity``, when two rounds say so.

        Only ever True when it is: each round looked at is one of the terms
        ``peak_from`` takes the most of. They are the last round the plan
        stores before ``entry`` is gone, and the round of most usage up to it
        (the set's peak, if ``entry`` is still there then): where a request
        that does not fit is usually over. False means nothing; ask ``peak_from``.
        Costs a search and a few sums, where ``peak_from`` is a pass over the plan.
        """
        if self._glance is None:
            times, usage = self.times.tolist(), self.usage.tolist()
            # the index of the most usage at or before each round
            best, at = [], 0
            for j, used in enumerate(usage):
                if used > usage[at]:
                    at = j
                best.append(at)
            self._glance = (times, usage, best)
        times, usage, best = self._glance
        # `times[0]` is 0, and `entry` runs at least a round: there is always one before it is gone
        last = bisect_left(times, max(1, entry.rounds)) - 1
        cap = max(entry.claim, entry.held)
        base = entry.held + entry.now
        for j in (last, best[last]):
            mine = min(cap, base + entry.growth * (-(-times[j] // self.page_size)))
            if usage[j] + mine > capacity:
                return True
        return False

    def fits(self, entry: PlanEntry, capacity: int | None = None) -> bool:
        cap = self.capacity if capacity is None else capacity
        return self.peak_from(entry) <= cap

    # ── EASY backfill ───────────────────────────────────────────────────

    def shadow(
        self, head: PlanEntry, capacity: int | None = None,
    ) -> tuple[int, int] | None:
        """When the head could start, and the room it would leave.

        ``(s, extra)``: ``s`` is the earliest round, among now and each release
        in the set, at which the head passes the peak test against the set aged
        by ``s``; ``extra`` is the pages spare while it runs from ``s``. None
        if no release makes room (the set keeps pages for ever, as finished
        requests do until they are removed).
        """
        cap = self.capacity if capacity is None else capacity
        starts = [0] + sorted({int(r) for r in self._rounds})
        for start in starts:
            peak = self.peak_from(head, start)
            if peak <= cap:
                return start, cap - peak
        return None


def shadow_sum(
    entries: Sequence[PlanEntry], supply: int, head_need: int,
) -> tuple[int, int] | None:
    """``PeakPlanner.shadow`` for the summed test.

    The head fits when what admitted requests may still take, plus its own,
    is within the supply. A request that is gone by round ``s`` is no longer
    owed anything, and returns what it holds; finished ones never leave. So
    the earliest ``s`` is the first release at which the slack covers the
    head, and ``extra`` is the slack it leaves.
    """
    owed = sum(e.need for e in entries)
    releases = sorted({e.rounds for e in entries if e.rounds > 0})
    for start in [0, *releases]:
        gone = sum(e.held + e.need for e in entries if 0 < e.rounds <= start)
        slack = supply - owed + gone
        if slack >= head_need:
            return start, slack - head_need
    return None


def easy_allows(
    rounds: int, cost: int, shadow: tuple[int, int] | None,
) -> bool:
    """Whether a request behind the head may be admitted without delaying it.

    It may if it is gone by the head's start, or if the pages it can hold fit
    in what the head leaves spare. With no start in sight (``shadow`` is None)
    there is nothing to protect yet.
    """
    if shadow is None:
        return True
    start, extra = shadow
    return rounds <= start or cost <= extra
