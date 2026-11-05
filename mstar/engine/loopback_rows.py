"""Device loop-back rows: a node's loop-back values kept on the device by request.

A signal that loops back to its own node can stay on the device between steps
instead of going through the worker: each step's tail writes the step's value
into the row of each of its requests (``write``), and the next step's head
reads them back (``read``). The one host write a row gets is its first value,
routed like any other input (the prefill's output, or the iteration-0 seed).
Row 0 is the trash row: padding rows read and write it.

Rows are handed out at ``add`` and taken back at ``drop``; a table grows on the
GPU thread, the only thread that reads or writes them.
"""
from __future__ import annotations

import threading
from collections.abc import Hashable, Sequence

import torch

from mstar.utils.h2d import PinnedStager

TRASH_ROW = 0


class LoopbackRows:
    """One table ``[rows, *shape]`` per loop-back signal of a node."""

    def __init__(self, device: torch.device):
        self._device = device
        self._tables: dict[str, torch.Tensor] = {}
        self._row_of: dict[Hashable, int] = {}
        self._free: list[int] = []
        # rows handed out so far, the trash row included
        self._rows = 1
        self._lock = threading.Lock()
        self._stager = PinnedStager(torch.long)
        # the last index uploaded, by its rids: a decode batch reads and writes the same rows
        self._last: tuple[tuple, torch.Tensor] | None = None

    def add(self, rid: Hashable) -> None:
        with self._lock:
            if rid not in self._row_of:
                self._row_of[rid] = self._free.pop() if self._free else self._next_row()
            self._last = None

    def _next_row(self) -> int:
        row, self._rows = self._rows, self._rows + 1
        return row

    def drop(self, rid: Hashable) -> None:
        with self._lock:
            row = self._row_of.pop(rid, None)
            if row is not None:
                self._free.append(row)
            self._last = None

    def has_row(self, rid: Hashable) -> bool:
        return rid in self._row_of

    def _index(self, rids: Sequence[Hashable]) -> torch.Tensor:
        key = tuple(rids)
        last = self._last
        if last is not None and last[0] == key:
            return last[1]
        with self._lock:
            rows = [self._row_of.get(rid, TRASH_ROW) for rid in rids]
        index = torch.empty(len(rows), dtype=torch.long, device=self._device)
        self._stager.copy_(index, rows)
        self._last = (key, index)
        return index

    def _table(self, name: str, like: torch.Tensor | None = None) -> torch.Tensor:
        """``name``'s table, grown to every row handed out (made like ``like``)."""
        table = self._tables.get(name)
        if table is None and like is None:
            raise RuntimeError(f"no loop-back row of {name!r} was written yet")
        if table is None or table.shape[0] < self._rows:
            size = max(self._rows, 8, 0 if table is None else 2 * table.shape[0])
            proto = like if table is None else table
            grown = torch.zeros((size, *proto.shape[1:]), dtype=proto.dtype, device=self._device)
            if table is not None:
                grown[:table.shape[0]].copy_(table)
            table = self._tables[name] = grown
        return table

    def write(self, name: str, rids: Sequence[Hashable], values: torch.Tensor) -> None:
        """Row ``i`` of ``values`` into the row of ``rids[i]`` (the trash row for
        a rid with none)."""
        table = self._table(name, values)
        table.index_copy_(0, self._index(rids), values.to(table.dtype))

    def read(self, name: str, rids: Sequence[Hashable]) -> torch.Tensor:
        """``[len(rids), *shape]``: each rid's row of ``name``."""
        return self._table(name).index_select(0, self._index(rids))

    def row(self, name: str, rid: Hashable) -> torch.Tensor:
        """A view of ``rid``'s row of ``name``, valid until the next write."""
        return self._table(name)[self._row_of[rid]]
