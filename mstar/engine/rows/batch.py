"""Which request holds which row slot of a persistent batch, and what changes per step.

A persistent (node, walk) keeps each live request's per-row state in a
resident slot from the request's first step until it is removed or evicted,
so a step only has to say which slots it runs (``RowDiff.idx``, in step
order) and which rows joined or left. The captured graphs read a step's rows
from a dense view the engine gathers them into (``RowSpace.gather``), so the
capture bucket follows the step's row count, and the slots stay put however
the scheduler rotates the requests it runs.
"""
from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from mstar_rust import RowSlots as _RustRowSlots
except ImportError:
    _RustRowSlots = None

_EMPTY = np.empty(0, dtype=np.int64)


@dataclass(slots=True)
class RowDiff:
    """One step's change to a persistent batch, as columns.

    Leaving rows are read out first (``leave_slots``), then joining rows are
    written (``join_slots``), which may reuse a slot a row left in the same
    step. ``order`` is the step's requests in step order and ``idx`` their
    slots. ``dropped`` are the leaving requests that were removed while they
    held a row: their state is gone, so nothing is read out for them.
    ``same_step``: the committed step ran the same slots in the same order,
    so nothing changes on the device.
    """
    base_version: int
    n_active: int
    order: list[int]
    idx: np.ndarray
    join_slots: np.ndarray
    join_rids: list[int]
    leave_slots: np.ndarray
    leave_rids: list[int]
    same_step: bool = False
    dropped: frozenset[int] = frozenset()

    def empty(self) -> bool:
        """Nothing to stage: the same requests in the same slots and order."""
        return self.same_step


class PyPersistentBatch:
    """Slot assignment for one persistent (node, walk).

    ``capacity`` slots hold rows; a step runs at most ``max_rows`` of them. A
    joining row takes the lowest free slot, and when none is free, the slot
    of the row that ran least recently and isn't in the step, which is
    checked in (evicted). ``propose`` describes a step without changing
    anything; the step's ``commit`` (once it is launched) settles it.
    Proposals are against the committed state, and a commit checks that
    nothing committed in between, since a diff only applies to the state it
    was taken from. Assignment is a pure function of the committed state and
    the step's requests, so ranks fed the same steps agree on every slot.
    Requests are worker handles (ints), as the engine keys its steps.
    ``RustPersistentBatch`` is the same, slot for slot.
    """

    def __init__(self, max_rows: int, capacity: int | None = None):
        capacity = max_rows if capacity is None else capacity
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        if capacity < max_rows:
            raise ValueError(f"capacity {capacity} is below the {max_rows} rows a step may run")
        self.max_rows = max_rows
        self.capacity = capacity
        self._slot_of: dict[int, int] = {}
        self._free = list(range(capacity))  # a heap: the lowest free slot first
        # the version each slot last ran at, for choosing what to evict
        self._last_run = np.zeros(capacity, dtype=np.int64)
        self._order: list[int] = []  # the committed step's requests, in step order
        self._idx = _EMPTY
        self._version = 0
        # removed requests whose row has not left yet; their handle may
        # already name a new request
        self._forgotten: set[int] = set()

    @property
    def n_held(self) -> int:
        """Rows held, whether or not the last step ran them."""
        return len(self._slot_of)

    @property
    def order(self) -> Sequence[int]:
        """The committed step's requests, in step order."""
        return self._order

    def slot_of(self, rid: int) -> int | None:
        return self._slot_of.get(rid)

    def held(self) -> dict[int, int]:
        """Every held row's slot, by request, removed requests' included."""
        return dict(self._slot_of)

    def last_run(self, slot: int) -> int:
        """The version ``slot`` last ran at."""
        return int(self._last_run[slot])

    @property
    def version(self) -> int:
        """Bumped by every commit that changes the rows or the step's slots."""
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
        """The diff of a step running ``rids``, in that order."""
        n = len(rids)
        if n > self.max_rows:
            raise ValueError(f"{n} rows exceed the persistent batch's {self.max_rows}")
        forgotten = self._forgotten
        if not forgotten and list(rids) == self._order:
            return RowDiff(
                self._version, n, self._order, self._idx, _EMPTY, [], _EMPTY, [], same_step=True,
            )
        wanted = set(rids)
        if len(wanted) != n:
            raise ValueError("a step names a request twice")
        slot_of = self._slot_of
        joins = [rid for rid in rids if rid in forgotten or rid not in slot_of]
        leave_rids = sorted(forgotten, key=slot_of.__getitem__)
        short = len(joins) - len(self._free) - len(leave_rids)
        if short > 0:
            idle = [
                rid for rid in slot_of
                if rid not in wanted and rid not in forgotten
            ]
            last_run = self._last_run
            idle.sort(key=lambda rid: (last_run[slot_of[rid]], slot_of[rid]))
            leave_rids += idle[:short]
        leave_slots = [slot_of[rid] for rid in leave_rids]
        take = min(len(joins), len(self._free))
        join_slots = heapq.nsmallest(take, self._free) + leave_slots[:len(joins) - take]
        joined = dict(zip(joins, join_slots, strict=True))
        idx = [joined[rid] if rid in joined else slot_of[rid] for rid in rids]
        return RowDiff(
            base_version=self._version,
            n_active=n,
            order=list(rids),
            idx=np.asarray(idx, dtype=np.int64),
            join_slots=np.asarray(join_slots, dtype=np.int64),
            join_rids=joins,
            leave_slots=np.asarray(leave_slots, dtype=np.int64),
            leave_rids=leave_rids,
            dropped=frozenset(rid for rid in leave_rids if rid in forgotten),
        )

    def without(self, rids: Sequence[int]) -> RowDiff:
        """The diff that checks ``rids`` in without running a step: for a
        request being removed, offloaded, or run by a step outside this batch.
        The committed step keeps its other rows, in order."""
        slot_of, forgotten = self._slot_of, self._forgotten
        gone = {rid for rid in rids if rid in slot_of} | forgotten
        leave_rids = sorted(gone, key=slot_of.__getitem__)
        order = [rid for rid in self._order if rid not in gone]
        return RowDiff(
            base_version=self._version,
            n_active=len(order),
            order=order,
            idx=np.asarray([slot_of[rid] for rid in order], dtype=np.int64),
            join_slots=_EMPTY,
            join_rids=[],
            leave_slots=np.asarray([slot_of[rid] for rid in leave_rids], dtype=np.int64),
            leave_rids=leave_rids,
            dropped=frozenset(rid for rid in leave_rids if rid in forgotten),
        )

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
        slot_of, forgotten = self._slot_of, self._forgotten
        if forgotten:
            # checked before anything changes: the worker never removes a
            # request a step it has planned still runs
            moved = set(diff.join_rids) | set(diff.leave_rids)
            stale = [rid for rid in diff.order if rid in forgotten and rid not in moved]
            if stale:
                raise RuntimeError(f"a step keeps rows of removed requests {stale}")
        released = []
        for rid, slot in zip(diff.leave_rids, diff.leave_slots.tolist(), strict=True):
            del slot_of[rid]
            if rid in forgotten:
                forgotten.discard(rid)
                released.append(slot)
        reused = set()
        for rid, slot in zip(diff.join_rids, diff.join_slots.tolist(), strict=True):
            slot_of[rid] = slot
            reused.add(slot)
        # the joins took the lowest free slots first, then slots rows left
        for _ in range(min(len(diff.join_rids), len(self._free))):
            heapq.heappop(self._free)
        for slot in diff.leave_slots.tolist():
            if slot not in reused:
                heapq.heappush(self._free, slot)
        self._order = list(diff.order)
        self._idx = diff.idx
        self._version += 1
        self._last_run[diff.idx] = self._version
        return np.asarray(released, dtype=np.int64)


def _int64s(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int64) if raw else _EMPTY


class RustPersistentBatch:
    """``PyPersistentBatch`` on the Rust slot assignment (``rust/src/row_slots.rs``)."""

    def __init__(self, max_rows: int, capacity: int | None = None):
        capacity = max_rows if capacity is None else capacity
        self.max_rows = max_rows
        self.capacity = capacity
        self._core = _RustRowSlots(max_rows, capacity)
        self._order: list[int] = []
        self._idx = _EMPTY

    @property
    def n_held(self) -> int:
        return self._core.n_held

    @property
    def order(self) -> Sequence[int]:
        return self._order

    def slot_of(self, rid: int) -> int | None:
        return self._core.slot_of(rid)

    def held(self) -> dict[int, int]:
        return self._core.held()

    def last_run(self, slot: int) -> int:
        return self._core.last_run(slot)

    @property
    def version(self) -> int:
        return self._core.version

    def is_checked_out(self, rid: int) -> bool:
        return self._core.is_checked_out(rid)

    def forget(self, rid: int) -> bool:
        return self._core.forget(rid)

    def _diff(self, order, cols) -> RowDiff:
        same_step, idx, join_rids, join_slots, leave_rids, leave_slots, dropped = cols
        if same_step:
            return RowDiff(
                self.version, len(self._order), self._order, self._idx, _EMPTY, [], _EMPTY, [], same_step=True,
            )
        return RowDiff(
            base_version=self.version,
            n_active=len(order),
            order=order,
            idx=_int64s(idx),
            join_slots=_int64s(join_slots),
            join_rids=join_rids,
            leave_slots=_int64s(leave_slots),
            leave_rids=leave_rids,
            dropped=frozenset(dropped),
        )

    def propose(self, rids: Sequence[int]) -> RowDiff:
        rids = list(rids)
        return self._diff(rids, self._core.propose(rids))

    def without(self, rids: Sequence[int]) -> RowDiff:
        order, cols = self._core.without(list(rids))
        return self._diff(order, cols)

    def commit(self, diff: RowDiff) -> np.ndarray:
        released = self._core.commit(
            diff.base_version, diff.same_step, diff.order, diff.idx, diff.join_rids, diff.join_slots,
            diff.leave_rids, diff.leave_slots,
        )
        if not diff.same_step:
            self._order = list(diff.order)
            self._idx = diff.idx
        return np.asarray(released, dtype=np.int64) if released else _EMPTY


PersistentBatch = PyPersistentBatch if _RustRowSlots is None else RustPersistentBatch
