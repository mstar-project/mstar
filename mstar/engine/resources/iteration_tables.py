"""The rows of a node's ``ITERATION_INPUTS``, on the device.

Each input (``IterationInput``) keeps one table for every request: a request's
rows and their pad go in once, when a step first reads them, and a step reads
its rows' iterations with one gather an input, on one staged index, however
many rows it has. A view a row and a stack took about 5x as long at 64 rows in
a CPU microbench (results/f6, 1082 us against 193 us); no GPU timing says what
share of a step that is.

Row 0 of a table is the input's ``capture_fill``, which padding rows read. Read
past its rows, a request reads its last: that is the pad's repeat.

A table is read outside the graphs, so it may grow, and a request never waits
for room. Streamed inputs are a step's own items, and don't come here.
"""
from __future__ import annotations

import threading
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager

if TYPE_CHECKING:
    from mstar.engine.resources.resident import ResidentStream
    from mstar.model.submodule_base import IterationInput

# a request's rows take a multiple of this, so a freed extent fits the next request of about its size
_EXTENT_ROWS = 64


@dataclass
class IterationTablesSpec(NodeResourceSpec):
    """Declared by the engine, for a node with ``ITERATION_INPUTS``: a model writes none."""

    @property
    def resource_class(self) -> type[IterationTables]:
        return IterationTables


def tables_key(node: str) -> str:
    return f"iteration_tables:{node}"


class _Extent(NamedTuple):
    """A request's rows of one table, and the state tensors they were made from."""
    source: torch.Tensor
    pad: torch.Tensor | None
    base: int
    # rows held, its pad's included
    rows: int
    size: int
    # rows of its table proper: reading past them is an error where there's no pad
    padless_rows: int | None


class _Table:
    """Doubles as requests need and never shrinks, outside any memory budget: its
    high-water mark stays, as a long-running server's concurrency peak."""

    def __init__(self, spec: IterationInput, device: torch.device):
        self.rows = torch.empty(1 + _EXTENT_ROWS, *spec.shape, dtype=spec.dtype, device=device)
        self.rows[0].fill_(spec.capture_fill)
        self.used = 1
        # extent size -> the bases free at that size
        self.free: dict[int, list[int]] = {}

    def take(self, size: int) -> int:
        if self.free.get(size):
            return self.free[size].pop()
        base, self.used = self.used, self.used + size
        if self.used > self.rows.shape[0]:
            grown = self.rows.new_empty(max(self.used, 2 * self.rows.shape[0]), *self.rows.shape[1:])
            grown[:base].copy_(self.rows[:base])
            self.rows = grown
        return base


class IterationTables(Resource):
    """One table a non-streamed ``IterationInput`` of a node."""

    def __init__(self, device: torch.device):
        self._device = torch.device(device)
        self._tables: dict[str, _Table] = {}
        self._extents: dict[Hashable, dict[str, _Extent]] = {}
        # the main thread frees what the GPU thread's steps take
        self._lock = threading.Lock()
        self._stager = PinnedStager(torch.long)

    @classmethod
    def build(cls, spec: IterationTablesSpec, info: EngineResourceInfo) -> IterationTables:
        return cls(info.device)

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        return None  # a step reads its columns before it launches

    def resident_refusal(self, key: str, streams: Mapping[str, ResidentStream]) -> str | None:
        return None

    def remove_request(self, rid: Hashable) -> None:
        with self._lock:
            for name, extent in self._extents.pop(rid, {}).items():
                self._tables[name].free.setdefault(extent.size, []).append(extent.base)

    def _extent(self, name: str, spec: IterationInput, rid: Hashable, state) -> _Extent:
        """``rid``'s rows of ``name``, made once while its source and pad are the
        same tensors (a walk change that reassigns either makes them anew)."""
        held = self._extents.setdefault(rid, {})
        source = state[spec.source]
        pad = state.get(spec.pad) if isinstance(spec.pad, str) else None
        extent = held.get(name)
        if extent is not None and extent.source is source and extent.pad is pad:
            return extent
        rows = source.reshape(-1, *spec.shape).to(spec.dtype)
        padless = spec.pad is None or (isinstance(spec.pad, str) and pad is None)
        if isinstance(spec.pad, str) and pad is not None:
            rows = torch.cat([rows, pad.reshape(-1, *spec.shape).to(spec.dtype)])
        elif not padless:
            rows = torch.cat([rows, rows.new_full((1, *spec.shape), spec.pad)])
        table = self._tables[name]
        count = rows.shape[0]
        size = max(-(-count // _EXTENT_ROWS), 1) * _EXTENT_ROWS
        with self._lock:
            if extent is not None and extent.size == size:
                base = extent.base
            else:
                if extent is not None:
                    table.free.setdefault(extent.size, []).append(extent.base)
                base = table.take(size)
            table.rows[base:base + count].copy_(rows)
            extent = held[name] = _Extent(source, pad, base, count, size, count if padless else None)
        return extent

    def read(
        self, specs: Mapping[str, IterationInput], rows: Sequence[tuple | None], states: Mapping, repeats: int,
    ) -> dict[str, torch.Tensor]:
        """Each input of ``specs``, ``[B, repeats, *shape]``: row i's ``(rid, its
        iteration, ...)`` at that iteration on, or None for a padding row, which
        (as a row with no state) reads ``capture_fill``."""
        index: list[int] = []
        names = list(specs)
        for name in names:
            spec = specs[name]
            if name not in self._tables:
                self._tables[name] = _Table(spec, self._device)
            for row in rows:
                state = None if row is None else states.get(row[0])
                if state is None:
                    index.extend((0,) * repeats)
                    continue
                extent = self._extent(name, spec, row[0], state)
                start = row[1] + spec.offset
                if extent.padless_rows is not None and start + repeats > extent.padless_rows:
                    raise ValueError(f"iteration input {name!r} reads past the {extent.rows} rows of {spec.source!r}")
                last = extent.base + extent.rows - 1
                first = extent.base + start
                if repeats == 1:
                    index.append(min(first, last))
                else:
                    index.extend(min(first + i, last) for i in range(repeats))
        staged = torch.empty(len(index), dtype=torch.long, device=self._device)
        self._stager.copy_(staged, index)
        width = len(rows) * repeats
        return {
            name: self._tables[name].rows.index_select(0, staged[i * width:(i + 1) * width]).view(
                len(rows), repeats, *specs[name].shape,
            )
            for i, name in enumerate(names)
        }
