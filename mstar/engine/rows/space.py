"""Device row buffers of a persistent batch, kept in step with their host mirrors.

A ``RowSpace`` owns one ``PersistentBatch`` and every ``RowBuffer`` the
resources of its (node, walk) register: per-row state at a fixed address that
the captured graphs read, ``max_rows`` rows each. A step proposes its rows
(``propose``), the resources say what joining rows hold and what changed
(``StepRows``), and everything that changed reaches the device as one
host-to-device copy plus one segment-copy kernel per phase; a step that
changes nothing costs neither.

A mirrored buffer's host copy is the truth: proposals are copy-on-write, so a
step that is refused or goes stale leaves it untouched.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

import numpy as np
import torch

from mstar.engine.rows.batch import PersistentBatch, RowDiff
from mstar.engine.rows.copy import copy_segments
from mstar.engine.rows.stage import new_stage
from mstar.utils.h2d import _H2D, _cuda_memcpy_async


class RowBuffer:
    """``max_rows`` rows of one per-row field, at a fixed device address."""

    def __init__(
        self, space: RowSpace, name: str, shape: tuple[int, ...],
        dtype: torch.dtype, pad, mirrored: bool,
    ):
        self.space = space
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.pad = pad
        self.device = torch.full(
            (space.max_rows, *self.shape), pad, dtype=dtype, device=space.device,
        )
        self.row_bytes = self.device[0].numel() * self.device.element_size()
        self._ptr = self.device.data_ptr()
        pad_row = torch.full(self.shape, pad, dtype=dtype)
        self.pad_bytes = pad_row.reshape(-1).view(torch.uint8).numpy().copy()
        # the committed mirror; None for state with no host truth (it is
        # checked out from a device store instead)
        self.host: np.ndarray | None = (
            np.full((space.max_rows, *self.shape), pad, dtype=pad_row.numpy().dtype)
            if mirrored else None
        )

    def rows(self, n: int) -> torch.Tensor:
        """The first ``n`` rows: what a step of ``n`` requests reads."""
        if n > self.space.max_rows:
            # a slice would silently hand the step fewer rows
            raise ValueError(f"{self}: a step of {n} rows, but the space holds {self.space.max_rows}")
        return self.device[:n]

    def row_ptr(self, slot: int) -> int:
        return self._ptr + slot * self.row_bytes

    def __repr__(self) -> str:
        return f"RowBuffer({self.name}, {self.shape}, {self.dtype})"


class StepRows:
    """One proposed step: its diff, and what the resources write for it.

    Joining rows must be written in full (``write`` or ``copy_in``) for every
    buffer; ``stage`` refuses a proposal that leaves one stale. Leaving rows
    are read out with ``copy_out`` before anything overwrites their slots.
    """

    def __init__(self, space: RowSpace, diff: RowDiff):
        self.space = space
        self.diff = diff
        self._mirrors: dict[RowBuffer, np.ndarray] = {}
        # buffer -> (slots, row values) written from the host
        self._rows: dict[RowBuffer, list[tuple[np.ndarray, np.ndarray]]] = {}
        # buffer -> (slots, flat element indices) written into the mirror
        self._elements: dict[RowBuffer, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._copy_in: list[tuple[RowBuffer, np.ndarray, torch.Tensor, np.ndarray]] = []
        self._copy_out: list[tuple[RowBuffer, np.ndarray, torch.Tensor, np.ndarray]] = []
        self._advance: list[tuple[RowBuffer, object]] = []
        self._on_commit: list[Callable[[np.ndarray], None]] = []
        self._staged: _Block | None = None

    @property
    def n_active(self) -> int:
        return self.diff.n_active

    @property
    def order(self) -> list[int]:
        return self.diff.order

    def mirror(self, buf: RowBuffer) -> np.ndarray:
        """``buf``'s host rows as this step will have them (before ``advance``).

        Read-only unless through ``write``/``write_at``: on a step that moves
        nothing it is the committed array itself.
        """
        mirror = self._mirrors.get(buf)
        if mirror is not None:
            return mirror
        if buf.host is None:
            raise ValueError(f"{buf.name} keeps no host mirror")
        if self.diff.empty():
            return buf.host
        return self._own(buf)

    def _own(self, buf: RowBuffer) -> np.ndarray:
        """A private copy of the committed rows with this step's moves and
        clears applied, made on first need."""
        mirror = self._mirrors.get(buf)
        if mirror is None:
            d = self.diff
            mirror = buf.host.copy()
            if len(d.move_src):
                mirror[d.move_dst] = buf.host[d.move_src]
            if d.n_before > d.n_active:
                mirror[d.n_active:d.n_before] = buf.pad
            self._mirrors[buf] = mirror
        return mirror

    def write(self, buf: RowBuffer, slots: np.ndarray, values: np.ndarray) -> None:
        """Whole rows from the host: ``buf[slots[i]] = values[i]``."""
        slots = np.asarray(slots, dtype=np.int64)
        if buf.host is not None:
            self._own(buf)[slots] = values
            self._rows.setdefault(buf, []).append((slots, None))
        else:
            values = np.ascontiguousarray(values).reshape(len(slots), -1)
            self._rows.setdefault(buf, []).append((slots, values.view(np.uint8)))

    def write_at(self, buf: RowBuffer, slots: np.ndarray, index: tuple, values) -> None:
        """Single elements of mirrored rows: ``buf[slots[i], *index_i] = values[i]``,
        with ``index`` a tuple of per-element index arrays over the row shape."""
        if buf.host is None:
            raise ValueError(f"{buf.name}: write_at needs a mirrored buffer")
        slots = np.asarray(slots, dtype=np.int64)
        mirror = self._own(buf)
        mirror[(slots, *index)] = values
        flat = np.ravel_multi_index((slots, *index), mirror.shape)
        self._elements.setdefault(buf, []).append((slots, flat))

    def copy_in(self, buf: RowBuffer, slots: np.ndarray, src: torch.Tensor, src_rows: np.ndarray) -> None:
        """Whole rows from a device store: ``buf[slots[i]] = src[src_rows[i]]``.
        Only into a buffer with no host mirror, which would not see them."""
        if buf.host is not None:
            raise ValueError(f"{buf.name}: copy_in into a mirrored buffer; write its rows from the host")
        _check_store(buf, src)
        self._copy_in.append((buf, np.asarray(slots, dtype=np.int64), src, np.asarray(src_rows, dtype=np.int64)))

    def copy_out(self, buf: RowBuffer, slots: np.ndarray, dst: torch.Tensor, dst_rows: np.ndarray) -> None:
        """Leaving rows back to a device store: ``dst[dst_rows[i]] = buf[slots[i]]``,
        read before this step moves or clears anything."""
        _check_store(buf, dst)
        self._copy_out.append((buf, np.asarray(slots, dtype=np.int64), dst, np.asarray(dst_rows, dtype=np.int64)))

    def advance(self, buf: RowBuffer, delta) -> None:
        """Add ``delta`` to the active rows of ``buf``'s mirror at commit: the
        host side of what the replay does to the device rows."""
        if buf.host is None:
            raise ValueError(f"{buf.name} keeps no host mirror")
        if any(b is buf for b, _ in self._advance):
            # a resource planned this step twice (pre-plan, then exec) and
            # wrote both times: it has to write on one of them
            raise ValueError(f"{buf.name} advanced twice in one step")
        self._advance.append((buf, delta))

    def on_commit(self, fn: Callable[[np.ndarray], None]) -> None:
        """Run ``fn`` when this step commits, before its mirrors replace the
        committed ones (so ``buf.host`` still holds the leaving rows). It is
        passed the slots whose requests were removed while holding them, which
        have no host state left to update."""
        self._on_commit.append(fn)

    def stage(self) -> None:
        """Upload what this step changes, on the current stream. Nothing for a
        step that changes nothing."""
        self._check_joins()
        with self.space._ring_lock:
            self._staged = self.space._stage(self)

    def commit(self) -> np.ndarray:
        """The step is launching: its rows become the committed state. Refused
        if anything committed since the proposal. Returns the slots freed by
        removed requests."""
        return self.space._commit(self)

    def apply(self) -> None:
        """Apply the staged change to the device rows, on the current stream:
        after the previous step's replay, before this one's."""
        if self._staged is not None:
            self.space._apply(self._staged)

    @property
    def stale(self) -> bool:
        """Something committed since this step was proposed."""
        return self.diff.base_version != self.space.batch.version

    def _check_joins(self) -> None:
        joins = self.diff.join_slots
        if not len(joins):
            return
        written: dict[RowBuffer, set[int]] = {}
        for buf, writes in self._rows.items():
            for slots, _ in writes:
                written.setdefault(buf, set()).update(slots.tolist())
        for buf, slots, _, _ in self._copy_in:
            written.setdefault(buf, set()).update(slots.tolist())
        need = set(joins.tolist())
        for buf in self.space.buffers:
            missing = need - written.get(buf, set())
            if missing:
                raise RuntimeError(
                    f"{buf.name}: joining rows in slots {sorted(missing)} were not written"
                )


def _check_store(buf: RowBuffer, store: torch.Tensor) -> None:
    if store.device != buf.device.device or store.dtype != buf.dtype:
        raise ValueError(f"{buf.name}: a store must match the buffer's device and dtype")
    if not store.is_contiguous() or store[0].numel() * store.element_size() != buf.row_bytes:
        raise ValueError(f"{buf.name}: a store must be contiguous rows of the buffer's shape")


class _Block:
    """One staged step: where its segment tables sit in a device block."""

    __slots__ = ("entry", "phases")

    def __init__(self, entry: _RingEntry, phases: list[tuple[int, int, int]]):
        self.entry = entry
        self.phases = phases  # (table byte offset, segments, longest segment) per phase


class _RingEntry:
    """A pinned host block and its device twin; the events say when each may
    be rewritten."""

    def __init__(self, nbytes: int, device: torch.device):
        accel = device.type != "cpu"
        self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=accel)
        self.view = self.host.numpy()
        self.device = torch.empty(nbytes, dtype=torch.uint8, device=device) if accel else self.host
        self.uploaded: torch.Event | None = None  # the H2D out of `host` is done
        self.applied: torch.Event | None = None   # the kernels reading `device` are done


class RowSpace:
    """The persistent rows of one (node, walk)."""

    def __init__(self, max_rows: int, device: torch.device, depth: int = 3, block_bytes: int = 64 << 10):
        self.max_rows = max_rows
        self.device = torch.device(device)
        self.batch = PersistentBatch(max_rows)
        self.buffers: list[RowBuffer] = []
        # the active row count, for in-graph updates that must skip padding
        self.n_active = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._n_bytes = np.zeros(1, dtype=np.int32).view(np.uint8)
        # the committed state, against a step on another thread
        self._lock = threading.Lock()
        # the staging ring: the plan thread and the gpu thread both stage
        self._ring_lock = threading.Lock()
        self._ring = [_RingEntry(block_bytes, self.device) for _ in range(depth)]
        self._next = 0
        self._memcpy = _cuda_memcpy_async() if self.device.type == "cuda" else None

    def buffer(
        self, name: str, shape: Sequence[int], dtype: torch.dtype, pad=0, mirrored: bool = True,
    ) -> RowBuffer:
        buf = RowBuffer(self, name, tuple(shape), dtype, pad, mirrored)
        self.buffers.append(buf)
        return buf

    def propose(self, rids: Sequence[int]) -> StepRows:
        with self._lock:
            return StepRows(self, self.batch.propose(rids))

    def without(self, rids: Sequence[int]) -> StepRows:
        with self._lock:
            return StepRows(self, self.batch.without(rids))

    def holds(self, rid: int) -> bool:
        """Whether ``rid`` is checked out here."""
        return self.batch.is_checked_out(rid)

    def forget(self, rid: int) -> bool:
        """See ``PersistentBatch.forget``; safe against a step proposing or
        committing on another thread."""
        with self._lock:
            return self.batch.forget(rid)

    def check_mirrors(self) -> None:
        """Raise if a mirrored buffer's device rows differ from its mirror.
        Synchronizes: for tests and debugging only."""
        for buf in self.buffers:
            if buf.host is None:
                continue
            device = buf.device.cpu().numpy()
            bad = np.flatnonzero((device != buf.host).reshape(self.max_rows, -1).any(axis=1))
            if len(bad):
                raise AssertionError(
                    f"{buf.name}: device rows {bad.tolist()} differ from the mirror"
                )

    # staging

    def _stage(self, step: StepRows) -> _Block | None:
        d = step.diff
        if (
            d.empty() and not step._rows and not step._elements
            and not step._copy_in and not step._copy_out
        ):
            return None
        # phase 0: leaving rows to their stores; 1: rows past the end into
        # holes; 2: joins, element writes, the cleared tail and the row count
        stage = new_stage(3)
        for buf, slots, dst, dst_rows in step._copy_out:
            rb = buf.row_bytes
            stage.copy_rows(0, buf._ptr, rb, slots, dst.data_ptr(), rb, dst_rows, rb)
        for buf in self.buffers:
            rb = buf.row_bytes
            if len(d.move_src):
                stage.copy_rows(1, buf._ptr, rb, d.move_src, buf._ptr, rb, d.move_dst, rb)
            if d.n_before > d.n_active:
                cleared = np.arange(d.n_active, d.n_before, dtype=np.int64)
                stage.write(buf._ptr, rb, cleared, np.tile(buf.pad_bytes, len(cleared)), rb)
        for buf, slots, src, src_rows in step._copy_in:
            rb = buf.row_bytes
            stage.copy_rows(2, src.data_ptr(), rb, src_rows, buf._ptr, rb, slots, rb)
        for buf, entries in step._rows.items():
            for slots, values in entries:
                if values is None:
                    values = step._mirrors[buf][slots].reshape(len(slots), -1).view(np.uint8)
                stage.write(buf._ptr, buf.row_bytes, slots, np.ascontiguousarray(values).reshape(-1), buf.row_bytes)
        for buf, entries in step._elements.items():
            mirror = step._mirrors[buf].reshape(-1)
            item = mirror.itemsize
            for _, flat in entries:
                # each element its own one-element row of the flat buffer
                stage.write(buf._ptr, item, flat, mirror[flat].view(np.uint8), item)
        if d.n_active != d.n_before:
            self._n_bytes.view(np.int32)[0] = d.n_active
            stage.put(self._n_bytes.copy(), self.n_active.data_ptr())
        return self._upload(stage)

    def _upload(self, stage) -> _Block:
        """The stage's tables and payload to a device block, as one copy."""
        total = stage.size()
        entry = self._take(total)
        base = entry.device.data_ptr()
        placed = stage.pack(entry.view, base)
        if self._memcpy is not None:
            stream = torch.cuda.current_stream(self.device)
            if entry.applied is not None:
                stream.wait_event(entry.applied)
            err = self._memcpy(base, entry.host.data_ptr(), total, _H2D, stream.cuda_stream)
            if err:
                raise RuntimeError(f"cudaMemcpyAsync failed with error {err}")
            entry.uploaded = stream.record_event()
        elif self.device.type != "cpu":
            stream = torch.accelerator.current_stream(self.device)
            if entry.applied is not None:
                stream.wait_event(entry.applied)
            entry.device[:total].copy_(entry.host[:total], non_blocking=True)
            entry.uploaded = stream.record_event()
        return _Block(entry, placed)

    def _take(self, nbytes: int) -> _RingEntry:
        """The next ring entry, grown to ``nbytes`` if it must be; its host
        block is free once its last upload has gone."""
        entry = self._ring[self._next]
        self._next = (self._next + 1) % len(self._ring)
        if entry.uploaded is not None:
            entry.uploaded.synchronize()
        if entry.host.numel() < nbytes:
            if entry.applied is not None:
                entry.applied.synchronize()
            grown = _RingEntry(max(nbytes, 2 * entry.host.numel()), self.device)
            self._ring[self._ring.index(entry)] = grown
            entry = grown
        return entry

    def _apply(self, block: _Block) -> None:
        entry = block.entry
        if self.device.type != "cpu":
            stream = torch.accelerator.current_stream(self.device)
            if entry.uploaded is not None:
                stream.wait_event(entry.uploaded)
        for at, count, longest in block.phases:
            if count:
                copy_segments(entry.device, entry.view, at, count, longest)
        if self.device.type != "cpu":
            entry.applied = torch.accelerator.current_stream(self.device).record_event()

    def _commit(self, step: StepRows) -> np.ndarray:
        with self._lock:
            released = self.batch.commit(step.diff)
            for fn in step._on_commit:
                fn(released)
            if not step.diff.empty():
                # a mirror the step never touched still takes its moves and clears
                for buf in self.buffers:
                    if buf.host is not None:
                        step._own(buf)
            for buf, mirror in step._mirrors.items():
                buf.host = mirror
            n = step.diff.n_active
            for buf, delta in step._advance:
                buf.host[:n] += delta
            return released
