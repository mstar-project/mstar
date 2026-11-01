"""The position resource's share of a persistent walk's rows (``mstar.engine.rows``).

Each row holds its request's next position on every label the walk's steps
carry in the resource's KV cache (``RowWalk.labels``), mirrored on the host.
A step reads the rows as its position ids, and each forward steps them by its
one token (``RowSpace.buffer(step=)``), so no step builds or copies a position
vector.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from mstar.engine.rows import RowSpace, RowState, StepRows


class PositionRows(RowState):
    def __init__(
        self, space: RowSpace, labels: Iterable[str], counters: dict[str, dict[str, int]], step: int,
    ):
        self._space = space
        # the manager's, rid -> label -> next position; a held row's lives here instead
        self._counters = counters
        self.labels = {
            label: space.buffer(f"positions.{label}", (), torch.long, step=step) for label in labels
        }

    def check_out(self, rows: StepRows) -> None:
        d = rows.diff
        held = [self._counters.get(rid) or {} for rid in d.join_rids]
        for label, buf in self.labels.items():
            rows.write(buf, d.join_slots, np.array([c.get(label, 0) for c in held], dtype=np.int64))

    def check_in(self, rows: StepRows) -> None:
        d = rows.diff
        leaving = [(rid, slot) for rid, slot in zip(d.leave_rids, d.leave_slots.tolist(), strict=True)
                   if rid not in d.dropped]
        if not leaving:
            return

        def write_back(released: np.ndarray) -> None:
            # a slot in ``released`` lost its request since the proposal, and
            # its handle may already name a new one
            gone = set(released.tolist())
            for label, buf in self.labels.items():
                host = buf.host
                for rid, slot in leaving:
                    if slot not in gone:
                        self._counters.setdefault(rid, {})[label] = int(host[slot])

        rows.on_commit(write_back)

    def held(self, rid: int) -> dict[str, int] | None:
        """``rid``'s committed positions, if it holds a row here."""
        out = {}
        for label, buf in self.labels.items():
            row = self._space.committed(buf, rid)
            if row is None:
                return None
            out[label] = int(row)
        return out
