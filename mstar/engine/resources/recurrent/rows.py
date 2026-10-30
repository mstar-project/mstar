"""The recurrent state pool's share of a persistent walk's rows (``mstar.engine.rows``).

A row holds, per label its walk's steps carry in the pool (``RowWalk.labels``),
the slot its request's state lives in, mirrored on the host. A step addresses
the pool through the rows its graph gathers, so a steady decode batch builds
and uploads no addressing; what the host writes is only a joining row, or a
slot a held row's admit took fresh.

The pool's slot table stays the host truth, and a row changes no state, so the
rows need nothing read back. Rows report every slot as holding state: the pool
zeroes a slot whenever it frees or resets one, so a slot with no saved state
reads as the zeros a fresh row would start from.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch

from mstar.engine.rows import RowSpace, RowState, StepRows


class RecurrentRows(RowState):
    def __init__(self, space: RowSpace, labels: Iterable[str], pad_index: int):
        self.pad_index = pad_index
        self.slots = {
            label: space.buffer(f"recurrent.{label}.slot", (), torch.int32, pad=pad_index)
            for label in labels
        }
        self.has_state = torch.ones(space.max_rows, dtype=torch.bool, device=space.device)

    def write_joins(self, rows: StepRows, slots: Mapping) -> None:
        """The joining rows' slots, from the pool's table once admit has filled it."""
        d = rows.diff
        held = [slots.get(rid) or {} for rid in d.join_rids]
        pad = self.pad_index
        for label, buf in self.slots.items():
            index = [s.index if (s := h.get(label)) is not None else pad for h in held]
            rows.write(buf, d.join_slots, np.array(index, dtype=np.int32))

    def write_taken(self, rows: StepRows, taken: Mapping[tuple, int]) -> None:
        """Slots admit took fresh for rows already held: ``taken`` is the pool's
        (rid, label) -> slot record of them."""
        joined = set(rows.diff.join_rids)
        order = set(rows.order)
        for (rid, label), index in taken.items():
            buf = self.slots.get(label)
            if buf is None or rid in joined or rid not in order:
                continue
            rows.write(buf, np.array([rows.slot_of(rid)]), np.array([index], dtype=np.int32))
