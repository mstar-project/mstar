"""Which request holds which row slot of a persistent batch, and what changes per step.

A persistent (node, walk) keeps each request's per-row state in a fixed row
slot across steps, so a step only has to describe what changed. Slots stay
dense: the rows of a step of ``n`` requests are slots ``[0, n)``, so the
capture bucket follows the row count and every slot past it is padding.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_EMPTY = np.empty(0, dtype=np.int64)


@dataclass(slots=True)
class RowDiff:
    """One step's change to a persistent batch, as columns.

    Applied in three phases, because their slots overlap: leaving rows are
    read out first (``leave_slots``), then rows past the new end move into
    the holes (``move_src`` -> ``move_dst``), then joining rows are written
    and the vacated tail ``[n_active, n_before)`` is reset to padding.
    ``order`` is the step's requests in slot order. ``dropped`` are the
    leaving requests that were removed while they held a row: their state is
    gone, so nothing is read out for them.
    """
    base_version: int
    n_before: int
    n_active: int
    order: list[int]
    join_slots: np.ndarray
    join_rids: list[int]
    leave_slots: np.ndarray
    leave_rids: list[int]
    move_src: np.ndarray
    move_dst: np.ndarray
    dropped: frozenset[int] = frozenset()

    def empty(self) -> bool:
        """Nothing to stage: the same requests in the same slots."""
        return (
            self.n_active == self.n_before
            and not self.join_rids and not self.leave_rids
        )


class PersistentBatch:
    """Slot assignment for one persistent (node, walk).

    ``propose`` describes a step without changing anything; the step's
    ``commit`` (once it is launched) or ``discard`` (refused or stale) settles
    it. Proposals are against the committed state, and a commit checks that
    nothing committed in between, since a diff only applies to the state it
    was taken from. Assignment is a pure function of the committed state and
    the step's requests, so ranks fed the same steps agree on every slot.
    """

    def __init__(self, max_rows: int):
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        self.max_rows = max_rows
        self._order: list[int] = []  # the committed rows, in slot order
        self._slot_of: dict[int, int] = {}
        self._version = 0
        # removed requests whose row has not left yet; their handle may
        # already name a new request
        self._forgotten: set[int] = set()

    @property
    def n_active(self) -> int:
        return len(self._order)

    @property
    def order(self) -> Sequence[int]:
        return self._order

    def slot_of(self, rid: int) -> int | None:
        return self._slot_of.get(rid)

    @property
    def version(self) -> int:
        """Bumped by every commit that changes the rows."""
        return self._version

    def is_checked_out(self, rid: int) -> bool:
        """Whether ``rid`` holds a row it will be checked in from."""
        return rid in self._slot_of and rid not in self._forgotten

    def forget(self, rid: int) -> bool:
        """``rid`` was removed while holding a row. Its row leaves at the
        next commit with nothing read out; if the handle comes back in a step
        before then, it names a new request and joins afresh. True if it held
        a row."""
        if rid not in self._slot_of:
            return False
        self._forgotten.add(rid)
        return True

    def propose(self, rids: Sequence[int]) -> RowDiff:
        """The diff that takes the committed rows to ``rids``.

        Rows that stay keep their slot unless it falls past the new end, in
        which case they move into a hole; joining rows fill the remaining
        holes in order.
        """
        n_before, n = len(self._order), len(rids)
        if n > self.max_rows:
            raise ValueError(f"{n} rows exceed the persistent batch's {self.max_rows}")
        if not self._forgotten and list(rids) == self._order:
            return RowDiff(
                self._version, n_before, n, self._order, _EMPTY, [], _EMPTY, [], _EMPTY, _EMPTY,
            )
        slot_of = self._slot_of
        forgotten = self._forgotten
        wanted = set(rids)
        if len(wanted) != n:
            raise ValueError("a step names a request twice")
        leave_rids = [
            rid for rid in self._order if rid not in wanted or rid in forgotten
        ]
        leave_slots = [slot_of[rid] for rid in leave_rids]
        order: list[int | None] = [None] * n
        movers, joins = [], []
        for rid in rids:
            slot = None if rid in forgotten else slot_of.get(rid)
            if slot is None:
                joins.append(rid)
            elif slot < n:
                order[slot] = rid
            else:
                movers.append((slot, rid))
        holes = [slot for slot in range(n) if order[slot] is None]
        move_src, move_dst = [], []
        movers.sort()
        for (src, rid), dst in zip(movers, holes, strict=False):
            order[dst] = rid
            move_src.append(src)
            move_dst.append(dst)
        join_slots = holes[len(movers):]
        for rid, slot in zip(joins, join_slots, strict=True):
            order[slot] = rid
        return RowDiff(
            base_version=self._version,
            n_before=n_before,
            n_active=n,
            order=order,
            join_slots=np.asarray(join_slots, dtype=np.int64),
            join_rids=joins,
            leave_slots=np.asarray(leave_slots, dtype=np.int64),
            leave_rids=leave_rids,
            move_src=np.asarray(move_src, dtype=np.int64),
            move_dst=np.asarray(move_dst, dtype=np.int64),
            dropped=frozenset(rid for rid in leave_rids if rid in forgotten),
        )

    def without(self, rids: Sequence[int]) -> RowDiff:
        """The diff that checks ``rids`` in without running a step: for a
        request being removed, offloaded, or run by a step outside this batch."""
        gone = set(rids) | self._forgotten
        return self.propose([rid for rid in self._order if rid not in gone])

    def commit(self, diff: RowDiff) -> np.ndarray:
        """Make ``diff`` the committed state. Returns the slots that requests
        removed while holding a row left, including any forgotten after the
        diff was taken. Slots, not handles: a handle may name a new request."""
        if diff.base_version != self._version:
            raise RuntimeError(
                f"a diff taken at version {diff.base_version} committed at {self._version}"
            )
        if diff.empty():
            return _EMPTY
        released = []
        for rid, slot in zip(diff.leave_rids, diff.leave_slots.tolist(), strict=True):
            del self._slot_of[rid]
            if rid in self._forgotten:
                self._forgotten.discard(rid)
                released.append(slot)
        if self._forgotten:
            joined = set(diff.join_rids)
            stale = [rid for rid in diff.order if rid in self._forgotten and rid not in joined]
            if stale:
                # the worker never removes a request a step it has planned still runs
                raise RuntimeError(f"a step keeps rows of removed requests {stale}")
        self._order = list(diff.order)
        for slot, rid in enumerate(self._order):
            self._slot_of[rid] = slot
        self._version += 1
        return np.asarray(released, dtype=np.int64)
