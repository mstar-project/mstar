"""The rows of a node's ``ITERATION_INPUTS``, on the device.

Each input (``IterationInput``) keeps one table for every request: a request's
rows and their pad go in once, when a step first reads them, and a step reads
its rows' iterations with one gather an input, on one staged index, however
many rows it has. A one-row, one-iteration read is a view of that table,
with no index upload or gather.

Row 0 of a table is the input's ``capture_fill``, which padding rows read. Read
past its rows, a request reads its last: that is the pad's repeat.

The source arena is read outside graphs and may grow. Captured runs instead
read fixed banks owned here, including their snapshotted streamed inputs.
"""
from __future__ import annotations

import threading
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.device_run import IterationWindow
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.resident_hooks import (
    ResidentBinding,
    no_resident_effect,
    unbounded_repeats,
)
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import SlotLease
from mstar.utils.h2d import PinnedStager

if TYPE_CHECKING:
    from mstar.model.submodule_base import IterationInput
    from mstar.streaming.stream_buffer import StreamReservation

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


@dataclass
class StreamWindow:
    """A captured view of ordinary stream-policy decisions, owned by IterationTables.

    Selection never executes a policy: it looks up the frozen predecessor,
    available end and producer-done key. Semantic pops and model computation
    have separate predicates; an ordinary empty final flush can do only the former.
    """
    decisions: torch.Tensor
    chunks: torch.Tensor
    history: torch.Tensor
    counts: torch.Tensor
    limits: torch.Tensor
    terminal: torch.Tensor
    selected: torch.Tensor
    metadata: torch.Tensor
    ready: torch.Tensor
    compute_ready: torch.Tensor

    def select(self, available: torch.Tensor, done: torch.Tensor, live: torch.Tensor) -> None:
        choices = self.decisions
        matches = ((choices[:, :, 0] == self.history[:, None])
                   & (choices[:, :, 1] == available[:, None]) & (choices[:, :, 2] == done[:, None]))
        allowed = live & ~self.terminal & (self.counts < self.limits)
        selected = torch.where(matches & allowed[:, None], choices[:, :, 3], -1).amax(dim=1)
        self.selected.copy_(selected)
        index = selected.clamp_min(0)[:, None, None].expand(-1, 1, self.chunks.shape[2])
        self.metadata.copy_(torch.where((selected >= 0)[:, None], self.chunks.gather(1, index).squeeze(1), 0))
        self.ready.copy_((selected >= 0).any().reshape(1))
        self.compute_ready.copy_(((selected >= 0) & (self.metadata[:, 5] != 0)).any().reshape(1))

    def advance(self) -> None:
        """After selected semantic work, including a prepared zero-compute tail."""
        kept = self.selected >= 0
        self.history.copy_(torch.where(kept, self.selected, self.history))
        self.counts.add_(kept)
        self.terminal.logical_or_(kept & (self.metadata[:, 3] != 0))


class _Table:
    """Doubles as requests need and never shrinks, outside any memory budget: its
    high-water mark stays, as a long-running server's concurrency peak."""

    def __init__(self, spec: IterationInput, device: torch.device):
        self.rows = torch.empty(1 + _EXTENT_ROWS, *spec.shape, dtype=spec.dtype, device=device)
        self.rows[0].fill_(spec.capture_fill)
        self.used = 1
        # extent size -> the bases free at that size
        self.free: dict[int, list[int]] = {}
        self.extents: dict[Hashable, _Extent] = {}

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
        # the main thread frees what the GPU thread's steps take
        self._lock = threading.Lock()
        self._stager = PinnedStager(torch.long)
        # Fixed captured views, distinct from the growing outside-capture arena.
        # Canonical bucket + member-qualified slot: aliases share, other buckets do not.
        self._windows: dict[tuple, IterationWindow] = {}
        self._window_indices: dict[tuple, torch.Tensor] = {}
        self._window_fills: dict[tuple, Mapping[str, float | int]] = {}
        self._stream_windows: dict[tuple, StreamWindow] = {}

    @classmethod
    def build(cls, spec: IterationTablesSpec, info: EngineResourceInfo) -> IterationTables:
        return cls(info.device)

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        return None  # a step reads its columns before it launches

    def resident_binding(self) -> ResidentBinding:
        return ResidentBinding(
            check_streams=no_resident_effect,
            bind=no_resident_effect,
            row=no_resident_effect,
            stage=no_resident_effect,
            gather=no_resident_effect,
            run_gather=no_resident_effect,
            commit_in_graph=no_resident_effect,
            advance=no_resident_effect,
            check_horizon=self._check_horizon,
            max_repeats=unbounded_repeats,
            rewind_run=no_resident_effect,
        )

    def _check_horizon(self, step, ctx) -> None:
        lease = ctx.slot_lease
        window = None if lease is None else self._windows.get((lease.bucket, lease.buffer_slot))
        if window is None:
            raise ValueError("new iteration rows need their captured owner input bank")
        if any(ctx.repeats > column.shape[1] for column in window.columns.values()):
            raise ValueError("joining horizon exceeds its captured input bank")
        # The publisher stages a validated immutable snapshot into fresh pinned
        # resident slots before exposing its descriptor. This binding promises
        # fixed storage; stream readiness/retention belongs to that snapshot.

    @staticmethod
    def window_bytes(slots: int, limit: int, entries: int, inputs: Mapping[str, torch.Tensor], bs: int) -> int:
        """Persistent bytes per captured bank, before capture selection."""
        columns = sum(value[0].numel() * value.element_size() for value in inputs.values())
        return slots * (2 * limit * columns + 8 * (1 + entries)) + 16 * bs

    def prepare_window(
        self, lease: SlotLease, inputs: Mapping[str, torch.Tensor], limit: int,
        entries: int, slots: int, fills: Mapping[str, float | int],
    ) -> IterationWindow:
        """Allocate a selected capture's fixed bank before capture, once."""
        key = (lease.bucket, lease.buffer_slot)
        if limit < 1 or entries < 2 or slots < 1 or inputs.keys() != fills.keys():
            raise ValueError("invalid captured input-bank geometry")
        if (window := self._windows.get(key)) is not None:
            if (window.origins.shape != (slots,) or window.entries.shape != (slots, entries)
                    or window.columns.keys() != inputs.keys() or self._window_fills[key] != fills
                    or any(window.columns[name].shape != (slots, 2 * limit, *value.shape[1:])
                           or window.columns[name].dtype != value.dtype
                           or window.columns[name].device != value.device for name, value in inputs.items())):
                raise ValueError("captured input-bank geometry changed for the same capture")
            return window
        window = IterationWindow(
            {name: value.new_full((slots, 2 * limit, *value.shape[1:]), fills[name])
             for name, value in inputs.items()},
            torch.zeros(slots, dtype=torch.long, device=self._device),
            torch.full((slots, entries), -1, dtype=torch.long, device=self._device),
        )
        self._windows[key] = window
        # Ordinary staging can still be queued on the execution stream when
        # a fresh row is initialized on the independent publication stream.
        self._window_indices[key] = torch.empty((2, lease.bucket.bs), dtype=torch.long, device=self._device)
        self._window_fills[key] = dict(fills)
        return window

    def discard_window(self, lease: SlotLease) -> None:
        """Release a bank whose capture failed, before it can have readers."""
        key = (lease.bucket, lease.buffer_slot)
        self._windows.pop(key, None)
        self._window_indices.pop(key, None)
        self._window_fills.pop(key, None)
        self._stream_windows.pop(key, None)

    def prepare_stream_window(self, lease: SlotLease, decisions: int, transitions: int) -> StreamWindow:
        """One bounded policy bank per leased capture; no payload ownership changes."""
        if not 0 < decisions <= 4096 or not 0 < transitions <= 4096:
            raise ValueError("a captured stream policy needs bounded decision and transition tables")
        key, bs = (lease.bucket, lease.buffer_slot), lease.bucket.bs
        if (window := self._stream_windows.get(key)) is not None:
            if window.decisions.shape != (bs, decisions, 4) or window.chunks.shape != (bs, transitions, 6):
                raise ValueError("captured stream-policy geometry changed for the same capture")
            return window
        def zeros(*shape, dtype=torch.long):
            return torch.zeros(shape, dtype=dtype, device=self._device)
        window = StreamWindow(
            torch.full((bs, decisions, 4), -1, dtype=torch.long, device=self._device),
            zeros(bs, transitions, 6), torch.full((bs,), -1, dtype=torch.long, device=self._device),
            zeros(bs), zeros(bs), zeros(bs, dtype=torch.bool),
            torch.full((bs,), -1, dtype=torch.long, device=self._device), zeros(bs, 6), zeros(1), zeros(1),
        )
        self._stream_windows[key] = window
        return window

    def stage_stream_window(
        self, lease: SlotLease, reservations: Sequence[StreamReservation], computes: Sequence[Sequence[bool]],
    ) -> None:
        """Before launch, freeze ordinary policy and prepared-tail decisions.

        ``computes`` comes from ordinary preparation for each transition; neither
        the policy nor the captured selector infers it from an empty chunk.
        This stages a whole fixed-membership bank on the execution stream.
        """
        window = self._stream_windows[lease.bucket, lease.buffer_slot]
        if len(reservations) != len(computes) or len(reservations) > lease.bucket.bs:
            raise ValueError("stream-policy snapshots do not fit the captured rows")
        decisions, chunks, limits = [], [], []
        for reservation, flags in zip(reservations, computes, strict=True):
            if (len(flags) != len(reservation.transitions) or any(type(flag) is not bool for flag in flags)
                    or len(reservation.parents) != len(reservation.transitions)
                    or len(reservation.decisions) > window.decisions.shape[1]
                    or len(reservation.transitions) > window.chunks.shape[1]
                    or reservation.bound < 1):
                raise ValueError("stream-policy snapshot exceeds its captured bank or lacks prepared transitions")
            decisions.extend([*reservation.decisions, *[(-1, -1, -1, -1)]
                              * (window.decisions.shape[1] - len(reservation.decisions))])
            chunks.extend([*(
                (t.chunk.start_offset, t.chunk.context_items, t.chunk.num_items,
                 int(t.chunk.is_final), t.chunk.tail_offset, int(compute))
                for t, compute in zip(reservation.transitions, flags, strict=True)
            ), *[(0,) * 6] * (window.chunks.shape[1] - len(reservation.transitions))])
            limits.append(reservation.bound)
        padding = lease.bucket.bs - len(reservations)
        decisions.extend([(-1, -1, -1, -1)] * (padding * window.decisions.shape[1]))
        chunks.extend([(0,) * 6] * (padding * window.chunks.shape[1]))
        limits.extend([0] * padding)
        # All snapshots have been checked before changing any capture binding.
        self._stager.copy_(window.decisions.view(-1), [v for row in decisions for v in row])
        self._stager.copy_(window.chunks.view(-1), [v for row in chunks for v in row])
        self._stager.copy_(window.limits, limits)
        window.history.fill_(-1)
        window.counts.zero_()
        window.terminal.zero_()
        window.selected.fill_(-1)
        window.metadata.zero_()
        window.ready.zero_()
        window.compute_ready.zero_()

    def stage_window(
        self, lease: SlotLease, slots: Sequence[int], snapshot: IterationWindow,
        *, stager: PinnedStager | None = None,
    ) -> None:
        """Stage selected resident rows on the caller's stream.

        Ordinary staging follows the capture's previous execution. Publication
        supplies its independent pinned stager and uses a second index scratch;
        its caller must hold fresh resident slots until every reader finishes.
        Validate every input before any copy. Preplanning never calls this.
        """
        key = (lease.bucket, lease.buffer_slot)
        target = self._windows.get(key)
        if target is None or not isinstance(snapshot, IterationWindow):
            raise ValueError("actual-progress execution needs its prepared owner input bank")
        n, rows = len(slots), snapshot.origins.numel()
        if (len(set(slots)) != n or any(type(slot) is not int or not 0 < slot < len(target.origins) for slot in slots)
                or n > lease.bucket.bs or rows != lease.bucket.bs
                or snapshot.origins.shape != (rows,) or snapshot.origins.dtype != torch.long
                or snapshot.origins.device != target.origins.device
                or snapshot.entries.shape != (rows, target.entries.shape[1])
                or snapshot.entries.dtype != torch.long or snapshot.entries.device != target.entries.device
                or snapshot.columns.keys() != target.columns.keys()):
            raise ValueError("input window does not match its captured owner rows")
        for name, column in snapshot.columns.items():
            bank = target.columns[name]
            if (column.ndim != bank.ndim or column.shape[0] != rows or not 0 < column.shape[1] <= bank.shape[1]
                    or column.shape[2:] != bank.shape[2:]
                    or column.dtype != bank.dtype or column.device != bank.device):
                raise ValueError(f"iteration input {name!r} exceeds its captured owner bank")
        if not n:
            return
        indices = self._window_indices[key][int(stager is not None), :n]
        (self._stager if stager is None else stager).copy_(indices, slots)
        for name, column in snapshot.columns.items():
            bank = target.columns[name]
            width = column.shape[1]
            bank[:, :width].index_copy_(0, indices, column[:n])
            if width < bank.shape[1]:
                # Clear the unread tail of a reused row too; never inherit an
                # earlier request's columns from the same capture bank.
                bank[:, width:].index_fill_(0, indices, self._window_fills[key][name])
        target.origins.index_copy_(0, indices, snapshot.origins[:n])
        target.entries.index_copy_(0, indices, snapshot.entries[:n])

    def remove_request(self, rid: Hashable) -> None:
        with self._lock:
            for table in self._tables.values():
                if (extent := table.extents.pop(rid, None)) is not None:
                    table.free.setdefault(extent.size, []).append(extent.base)

    def _extent(self, table: _Table, spec: IterationInput, rid: Hashable, state) -> _Extent:
        """``rid``'s rows of ``table``, made once while its source and pad are the
        same tensors (a walk change that reassigns either makes them anew)."""
        source = state[spec.source]
        pad = state.get(spec.pad) if isinstance(spec.pad, str) else None
        extent = table.extents.get(rid)
        if extent is not None and extent.source is source and extent.pad is pad:
            return extent
        rows = source.reshape(-1, *spec.shape).to(spec.dtype)
        padless = spec.pad is None or (isinstance(spec.pad, str) and pad is None)
        if isinstance(spec.pad, str) and pad is not None:
            rows = torch.cat([rows, pad.reshape(-1, *spec.shape).to(spec.dtype)])
        elif not padless:
            rows = torch.cat([rows, rows.new_full((1, *spec.shape), spec.pad)])
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
            extent = table.extents[rid] = _Extent(source, pad, base, count, size, count if padless else None)
        return extent

    def read(
        self, specs: Mapping[str, IterationInput], rows: Sequence[tuple | None], states: Mapping, repeats: int,
        horizons: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Each input of ``specs``, ``[B, repeats, *shape]``: row i's ``(rid, its
        iteration, ...)`` at that iteration on, or None for a padding row, which
        (as a row with no state) reads ``capture_fill``."""
        if horizons is None:
            horizons = [repeats] * len(rows)
        if len(horizons) != len(rows) or any(type(n) is not int or not 0 <= n <= repeats for n in horizons):
            raise ValueError("iteration input horizons must fit each staged row")
        index: list[int] = []
        names = list(specs)
        for name in names:
            spec = specs[name]
            if name not in self._tables:
                self._tables[name] = _Table(spec, self._device)
            table = self._tables[name]
            for row, count in zip(rows, horizons, strict=True):
                state = None if row is None else states.get(row[0])
                if state is None or count == 0:
                    index.extend((0,) * repeats)
                    continue
                extent = self._extent(table, spec, row[0], state)
                start = row[1] + spec.offset
                if extent.padless_rows is not None and start + count > extent.padless_rows:
                    raise ValueError(f"iteration input {name!r} reads past the {extent.rows} rows of {spec.source!r}")
                last = extent.base + extent.rows - 1
                first = extent.base + start
                if repeats == 1:
                    index.append(first if first < last else last)
                else:
                    index.extend(min(first + i, last) if i < count else 0 for i in range(repeats))
        width = len(rows) * repeats
        if width == 1:
            # A single iteration of one row is already contiguous in its
            # owner's table: no index upload, gather kernel or new storage.
            return {
                name: self._tables[name].rows[index[i]:index[i] + 1].view(1, 1, *specs[name].shape)
                for i, name in enumerate(names)
            }
        staged = torch.empty(len(index), dtype=torch.long, device=self._device)
        self._stager.copy_(staged, index)
        return {
            name: self._tables[name].rows.index_select(0, staged[i * width:(i + 1) * width]).view(
                len(rows), repeats, *specs[name].shape,
            )
            for i, name in enumerate(names)
        }
