"""Which verify results each request still has to apply, and when.

A verify step's commit counts all ``k + 1`` tokens of every row (the reserved
length), before its result is known; the graph keeps ``accepted + 1`` on the
device. Postprocess records the result, and the runner applies it at the head
of a later admit (`StepRunner._take_back`), lowering each resource by
``counted - (accepted + 1)``, exactly once.

Steps are numbered by the node's commits, which every rank of a lockstep
instance runs in the same order, so the ranks apply the same results at the
same step without a collective. A settle outside an admit is rank 0's alone, so
it reaches the followers through the resident delta (`OffloadDelta`). Only the
admitted step's own rows are taken back: theirs are the lengths the step reads,
and a row a step holds can't be torn down under the pass. A row that isn't
running keeps its results until it next runs, is picked as an offload's victim
(`settle`) or is torn down. Postprocess of step S
precedes the admit of step S + 2 on every worker path (the worker arms S + 2
only after S's postprocess), so a row's result is due two steps on
(asynchronous mode); in synchronous mode the step's rows were post-processed
before it was built, so their last result is due as well.
"""
from __future__ import annotations

import threading
from collections.abc import Collection, Mapping, Sequence

import numpy as np


class _Entry:
    __slots__ = ("seq", "rids", "labels", "counted", "accepted")

    def __init__(self, seq: int, rids: tuple[int, ...], labels: dict[str, str], counted: list[int]):
        self.seq = seq
        self.rids = rids
        # resource key -> the label of the stream it committed
        self.labels = labels
        # the span every corrected resource counted per row, 0 where one didn't (`NodeDraft.committed`)
        self.counted = counted
        # rid -> drafts accepted, once postprocess recorded the step
        self.accepted: dict[int, int] | None = None


class TakeBack:
    """One resource's share of a take-back pass, columnar."""

    __slots__ = ("rids", "counts", "label")

    def __init__(self, label: str):
        self.label = label
        self.rids: list[int] = []
        self.counts: list[int] = []


class VerifyLedger:
    """One verifying node's committed verify steps whose results are not yet
    applied. Touched from the GPU thread (commit), the main thread (record,
    removal, offload) and the plan or GPU thread (the take-back pass), under
    one lock.

    Handle-keyed: a removed request's entries go with it (`remove_request`),
    so a recycled handle never inherits one."""

    def __init__(self, node: str, synchronous: bool):
        self.node = node
        self.synchronous = synchronous
        self._lock = threading.Lock()
        # steps of the node committed so far; the step being admitted is the next
        self._commits = 0
        self._entries: dict[int, _Entry] = {}
        # rid -> the steps holding it, and how many of them postprocess has not recorded
        self._steps_of: dict[int, set[int]] = {}
        self._unrecorded: dict[int, int] = {}

    @property
    def next_step(self) -> int:
        """The number of the node's next step to commit."""
        return self._commits + 1

    def pending(self, rid: int) -> bool:
        """``rid`` has a committed verify step whose result is not applied:
        its reserved length runs ahead of its verified one, and on a resident
        row the device holds its true length."""
        return rid in self._steps_of

    def unrecorded(self, rid: int) -> bool:
        """``rid`` has a committed verify step whose result postprocess has not
        recorded yet: only the device knows how much of it was kept."""
        return rid in self._unrecorded

    def committed(
        self, rids: Sequence[int] | None, labels: Mapping[str, str] | None = None,
        counted: Sequence[int] | None = None,
    ) -> int:
        """A step of the node committed: a verify step's real ``rids``, per
        resource key with a take-back its stream's label, and the span they
        counted per row. ``rids`` None for any other step of the node, which
        only counts. Returns the step's number."""
        with self._lock:
            self._commits += 1
            seq = self._commits
            if rids:
                self._entries[seq] = _Entry(seq, tuple(rids), dict(labels or {}), list(counted or ()))
                steps_of, unrecorded = self._steps_of, self._unrecorded
                for rid in rids:
                    steps_of.setdefault(rid, set()).add(seq)
                    unrecorded[rid] = unrecorded.get(rid, 0) + 1
            return seq

    def record(self, seq: int, request_ids: Sequence[int], accepted: np.ndarray) -> None:
        """Postprocess of step ``seq``: the drafts each row accepted, rank 0's,
        in the order the forward ran ``request_ids`` (a padded copy may run
        longer). Rows removed since the commit are dropped."""
        with self._lock:
            entry = self._entries.get(seq)
            if entry is None:
                return  # every row of the step was removed, or the step voided
            if entry.accepted is not None:
                raise RuntimeError(f"verify step {seq} on {self.node} recorded twice")
            of = dict(zip(request_ids, accepted[: len(request_ids)].tolist(), strict=True))
            missing = [rid for rid in entry.rids if rid not in of]
            if missing:
                raise RuntimeError(f"verify step {seq} on {self.node}: no result for rows {missing}")
            entry.accepted = {rid: of[rid] for rid in entry.rids}
            for rid in entry.rids:
                self._decrement(self._unrecorded, rid)

    def take_due(self, step: int, step_rids: Collection[int]) -> dict[str, TakeBack]:
        """Pop the results of ``step_rids`` due at the admit of ``step``, as each
        resource's columns: ``counted - (accepted + 1)`` per row, rows that give
        back nothing left out. Raises when one of theirs is due but was never
        recorded, which would mean postprocess no longer precedes that admit.
        Linear in the step's rows: each due entry is walked and rebuilt once."""
        with self._lock:
            steps_of = self._steps_of
            through = step - 1 if self.synchronous else step - 2
            due: set[int] = set()
            for rid in step_rids:
                held = steps_of.get(rid)
                if held:
                    due.update(seq for seq in held if seq <= through)
            if not due:
                return {}
            rows = step_rids if isinstance(step_rids, (set, frozenset)) else set(step_rids)
            out: dict[str, TakeBack] = {}
            for seq in sorted(due):
                entry = self._entries[seq]
                if entry.accepted is None:
                    raise RuntimeError(
                        f"verify step {seq} on {self.node} is due at step {step}, but its result was never recorded"
                    )
                popped = [i for i, rid in enumerate(entry.rids) if rid in rows]
                self._take(entry, popped, out)
                self._drop_rows(entry, popped)
            return out

    def settle(self, rid: int) -> dict[str, TakeBack]:
        """Pop every result of ``rid``, due or not, as the take-back columns:
        an offload snapshots the host length, which must be the verified one.
        Raises when one is unrecorded. Exactly once, as `take_due` is."""
        with self._lock:
            held = self._steps_of.get(rid)
            if not held:
                return {}
            entries = [self._entries[seq] for seq in sorted(held)]
            if any(e.accepted is None for e in entries):
                raise RuntimeError(f"settling {rid} on {self.node}: a committed verify step of it is unrecorded")
            out: dict[str, TakeBack] = {}
            for entry in entries:
                i = [entry.rids.index(rid)]
                self._take(entry, i, out)
                self._drop_rows(entry, i)
            return out

    def void(self, seq: int) -> None:
        """Step ``seq`` committed and will never be recorded (it failed between
        commit and postprocess): its rows are torn down, so it is dropped and
        nothing is taken back for it."""
        with self._lock:
            entry = self._entries.get(seq)
            if entry is None or entry.accepted is not None:
                return
            self._drop_rows(entry, range(len(entry.rids)))

    def remove_request(self, rid: int) -> None:
        """``rid`` is gone: its unapplied results go with its pages."""
        with self._lock:
            for seq in sorted(self._steps_of.get(rid, ())):
                entry = self._entries[seq]
                self._drop_rows(entry, [entry.rids.index(rid)])

    # under the lock

    def _take(self, entry: _Entry, rows: Sequence[int], out: dict[str, TakeBack]) -> None:
        """Add the take-back of ``entry``'s rows ``rows`` (recorded) to ``out``."""
        rids, counts = [], []
        for i in rows:
            span = entry.counted[i]
            if not span:
                continue  # commit dropped it: nothing was counted
            k = entry.accepted[entry.rids[i]] + 1
            n = span - k
            if not 0 <= n <= span:
                raise RuntimeError(
                    f"verify step {entry.seq} on {self.node}: {entry.rids[i]} counted {span} and kept {k}"
                )
            if n:
                rids.append(entry.rids[i])
                counts.append(n)
        for key, label in entry.labels.items():
            take = out.get(key)
            if take is None:
                take = out[key] = TakeBack(label)
            elif take.label != label:
                raise RuntimeError(f"{key!r} took back on {take.label!r} and {label!r}")
            take.rids.extend(rids)
            take.counts.extend(counts)

    def _drop_rows(self, entry: _Entry, rows: Collection[int]) -> None:
        """Take rows ``rows`` (indices) out of ``entry`` in one rebuild, and the
        entry out once it is empty."""
        if not rows:
            return
        recorded = entry.accepted is not None
        for i in rows:
            rid = entry.rids[i]
            if not recorded:
                self._decrement(self._unrecorded, rid)
            held = self._steps_of[rid]
            held.discard(entry.seq)
            if not held:
                del self._steps_of[rid]
        if len(rows) == len(entry.rids):
            del self._entries[entry.seq]
            return
        gone = set(rows)
        keep = [j for j in range(len(entry.rids)) if j not in gone]
        if recorded:
            for i in gone:
                entry.accepted.pop(entry.rids[i], None)
        entry.rids = tuple(entry.rids[j] for j in keep)
        entry.counted = [entry.counted[j] for j in keep]

    @staticmethod
    def _decrement(counts: dict[int, int], rid: int) -> None:
        left = counts[rid] - 1
        if left:
            counts[rid] = left
        else:
            del counts[rid]
