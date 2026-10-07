"""The admitted requests as a plan sees them, kept as arrays by slot.

A plan is rebuilt whenever the pool's counters move, which under load is every
decision, and making each admitted request's `PlanEntry` for it means reading
every one of its streams again. Most of an entry only moves when the manager
says so: what the request holds (``held``), what it reserved (``claim``) and
what its prompt still has to allocate (``now``) are set as they change. What
follows the committed length of a stream (``growth`` and ``rounds``) moves with
every token, so `fields` reads it again, for every request at once.

Rows are in the order the requests reserved, as the manager's own are.
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import attrgetter

import numpy as np

from mstar.engine.resources.kv.peak_plan import PlanEntry


class _Absent:
    """In place of a stream the request has not opened: nothing committed to it."""

    stored_len = 0


ABSENT = _Absent()
_stored = attrgetter("stored_len")


class PlanTable:
    """``held``, ``claim``, ``now`` and the decode labels of each admitted request.

    A decode label is ``(stream, prompt tokens, decode tokens)``, as `_life`
    counts them, with the stream it is on, or `ABSENT`. A request with fewer
    labels than the most any has is padded with ones that decode nothing, which
    never count as a growing label.

    The arrays handed out are the table's own, and good until it next changes.
    """

    def __init__(self) -> None:
        self.rids: list[str] = []
        self._at: dict[str, int] = {}
        self._width = 1
        # the stream of each label of each row, `_width` to a row
        self._streams: list = []
        room = 64
        self._held = np.zeros(room, dtype=np.int64)
        self._claim = np.zeros(room, dtype=np.int64)
        self._now = np.zeros(room, dtype=np.int64)
        # what the request may still take: claim less held
        self._need = np.zeros(room, dtype=np.int64)
        self._decode = np.zeros((room, self._width), dtype=np.int64)
        # prompt + decode: the committed length a label is done at
        self._end = np.zeros((room, self._width), dtype=np.int64)
        self.held_total = 0
        self.need_total = 0

    def __len__(self) -> int:
        return len(self.rids)

    def __contains__(self, rid: str) -> bool:
        return rid in self._at

    def append(self, rid: str) -> None:
        """A row, empty, after the others."""
        at = len(self.rids)
        # room for this row, and the spare slot after it (`with_spare`)
        if at + 1 >= len(self._held):
            self._grow(2 * len(self._held))
        self._at[rid] = at
        self.rids.append(rid)
        self._streams.extend([ABSENT] * self._width)
        self._held[at] = self._claim[at] = self._now[at] = self._need[at] = 0
        self._decode[at] = self._end[at] = 0

    def drop(self, rid: str) -> None:
        """The row of ``rid``; the rows behind it close up."""
        at = self._at.pop(rid)
        n, width = len(self.rids), self._width
        self.held_total -= int(self._held[at])
        self.need_total -= int(self._need[at])
        del self.rids[at], self._streams[at * width:(at + 1) * width]
        for column in (self._held, self._claim, self._now, self._need, self._decode, self._end):
            column[at:n - 1] = column[at + 1:n]
        for later in range(at, n - 1):
            self._at[self.rids[later]] = later

    def set(
        self, rid: str, held: int, claim: int, now: int,
        decoding: Sequence[tuple[object, int, int]],
    ) -> None:
        at = self._at[rid]
        if len(decoding) > self._width:
            self._widen(len(decoding))
        need = max(0, claim - held)
        self.held_total += held - int(self._held[at])
        self.need_total += need - int(self._need[at])
        self._held[at] = held
        self._claim[at] = claim
        self._now[at] = now
        self._need[at] = need
        width = self._width
        streams = [ABSENT] * width
        for k, (stream, prompt, decode) in enumerate(decoding):
            streams[k] = stream
            self._decode[at, k] = decode
            self._end[at, k] = prompt + decode
        for k in range(len(decoding), width):
            self._decode[at, k] = self._end[at, k] = 0
        self._streams[at * width:(at + 1) * width] = streams

    def fields(self) -> tuple[np.ndarray, ...]:
        """``held``, ``claim``, ``now``, ``growth`` and ``rounds``, a slot per request.

        The rounds are the longest any decode label has left of its tokens, by the
        committed length of its stream now, as `_plan_entry` counts them.
        """
        n, width = len(self.rids), self._width
        stored = np.fromiter(map(_stored, self._streams), dtype=np.int64, count=n * width)
        decode, end = self._decode[:n], self._end[:n]
        if width == 1:
            decode, end = decode[:, 0], end[:, 0]
        else:
            stored = stored.reshape(n, width)
        # what a label has left: its decode tokens, less what is committed past its prompt
        left = np.maximum(np.minimum(decode, end - stored), 0)
        if width == 1:
            return self._held[:n], self._claim[:n], self._now[:n], np.minimum(left, 1), left
        return (
            self._held[:n], self._claim[:n], self._now[:n],
            np.minimum(left, 1).sum(axis=1), left.max(axis=1),
        )

    def with_spare(self) -> tuple[np.ndarray, np.ndarray]:
        """``held`` and ``need``, a slot per request and one more, which is no request's."""
        n = len(self.rids)
        return self._held[:n + 1], self._need[:n + 1]

    def _grow(self, room: int) -> None:
        for name in ("_held", "_claim", "_now", "_need", "_decode", "_end"):
            old = getattr(self, name)
            new = np.zeros((room, *old.shape[1:]), dtype=np.int64)
            new[:len(old)] = old
            setattr(self, name, new)

    def _widen(self, width: int) -> None:
        for name in ("_decode", "_end"):
            old = getattr(self, name)
            new = np.zeros((len(old), width), dtype=np.int64)
            new[:, :old.shape[1]] = old
            setattr(self, name, new)
        n, was = len(self.rids), self._width
        self._streams = [
            self._streams[row * was + k] if k < was else ABSENT
            for row in range(n) for k in range(width)
        ]
        self._width = width


def plan_entries(
    held: np.ndarray, claim: np.ndarray, now: np.ndarray, growth: np.ndarray, rounds: np.ndarray,
) -> list[PlanEntry]:
    """The `PlanEntry` of each slot of what `PlanTable.fields` gives."""
    return [
        PlanEntry(*row) for row in zip(
            held.tolist(), claim.tolist(), now.tolist(), growth.tolist(), rounds.tolist(),
            strict=True,
        )
    ]
