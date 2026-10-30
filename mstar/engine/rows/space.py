"""Device row buffers of a persistent batch, kept in step with their host mirrors.

A ``RowSpace`` owns one ``PersistentBatch`` and every ``RowBuffer`` the
resources of its (node, walk) register. Each buffer has two levels on the
device: resident rows, one slot per live request (``capacity`` of them, plus
a padding slot and a trash slot), and a dense step view of ``max_rows`` rows
at a fixed address, which the captured graphs read. Before a step's forward
its rows are gathered into the dense view (``gather``); after it, what the
forward changed is scattered back (``after_forward``). A step proposes its
rows (``propose``), the resources say what joining rows hold and what changed
(``StepRows``), and everything that changed reaches the device as one
host-to-device copy of a segment table and its payload, which one copy
kernel runs (two when leaving rows are read out); a step that changes
nothing costs nothing.

A mirrored buffer's host copy is the truth: proposals are copy-on-write, so a
step that is refused or goes stale leaves it untouched. The mirrored buffers
share one slab, which a step that writes any of them uploads whole: it is a
few bytes a row, and a whole copy costs less than working out which rows
changed. Buffers without a mirror hold device state, moved row by row.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from mstar.engine.rows.batch import PersistentBatch, RowDiff
from mstar.engine.rows.copy import RowTable, copy_segments, move_rows
from mstar.engine.rows.stage import new_stage, step_mirrors
from mstar.utils.h2d import _H2D, _cuda_memcpy_async

logger = logging.getLogger(__name__)

_ALIGN = 16


def _round(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


_NO_NUMPY = (torch.bfloat16, *(getattr(torch, name) for name in (
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
) if hasattr(torch, name)))


class RowBuffer:
    """One per-row field: resident rows by slot, and the dense step view a
    step of ``n`` requests reads as its first ``n`` rows, both at fixed
    device addresses. The rows are allocated when the space is sealed, at its
    first use. An unmoved buffer has no dense view: a step addresses its
    resident rows by slot (``RowSpace.gather_slots``)."""

    def __init__(
        self, space: RowSpace, name: str, shape: tuple[int, ...],
        dtype: torch.dtype, pad, mirrored: bool, step: int = 0, moved: bool = True,
    ):
        self.space = space
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.mirrored = mirrored
        self.step = step
        self.moved = moved
        # one value, or one per element of the row
        if dtype in _NO_NUMPY:
            raise ValueError(f"{name}: numpy has no {dtype} for the rows' host values")
        pad_row = torch.as_tensor(pad, dtype=dtype).expand(self.shape).contiguous()
        self.pad = pad_row.numpy()
        self.pad_bytes = pad_row.reshape(-1).view(torch.uint8).numpy().copy()
        self.row_bytes = self.pad_bytes.nbytes
        self._device: torch.Tensor | None = None  # the dense step view
        self._resident: torch.Tensor | None = None
        self._ptr = 0  # the resident rows', which host writes and copies address
        # the last store checked for this buffer (``_store_ptr``)
        self._store: torch.Tensor | None = None
        self._store_ptr = 0
        self._offset = 0  # a mirrored buffer's place in the slab

    @property
    def device(self) -> torch.Tensor:
        """The dense step view, ``max_rows`` rows."""
        if not self.moved:
            raise RuntimeError(f"{self} stays resident: address it by slot")
        if self._device is None:
            self.space.seal()
        return self._device

    @property
    def resident(self) -> torch.Tensor:
        """The resident rows, by slot."""
        if self._resident is None:
            self.space.seal()
        return self._resident

    @property
    def host(self) -> np.ndarray | None:
        """The committed mirror of the resident rows, by slot (read-only);
        None for state with no host truth, which is checked out from a
        device store instead."""
        if not self.mirrored:
            return None
        self.space.seal()
        return self.space._host_view(self, self.space._host)

    def rows(self, n: int) -> torch.Tensor:
        """The first ``n`` rows of the dense view: what a step of ``n`` requests reads."""
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
    Slots are resident slots (``diff.join_slots``, ``slot_of``).
    """

    def __init__(self, space: RowSpace, diff: RowDiff):
        self.space = space
        self.diff = diff
        # this step's mirrors: a private copy of the committed slab, made on
        # first write, and the forwards it had taken (``RowSpace._steps``)
        self._slab: np.ndarray | None = None
        self._slab_steps = 0
        # the step's copies and unmirrored host writes, as registered: phase 0
        # reads leaving rows out, phase 1 writes (possibly into the slots they left)
        self._stage = new_stage(2)
        # rows written per buffer, for the check that every join was
        self._written: dict[RowBuffer, int] = {}
        # the separate tensors, alive until the copies are done (``_RingEntry.keep``)
        self._keep: list[torch.Tensor] = []
        self._on_commit: list[Callable[[np.ndarray], None]] = []
        self._staged: _Block | None = None
        self._gather: np.ndarray | None = None
        # the forwards committed when a step that changes rows first read the
        # live mirrors: one committing since makes what it read stale
        self._read_steps: int | None = None
        self.committed = False

    @property
    def n_active(self) -> int:
        return self.diff.n_active

    @property
    def order(self) -> list[int]:
        return self.diff.order

    def slot_of(self, rid: int) -> int:
        """The resident slot of a request this step runs."""
        return int(self.diff.idx[self.diff.order.index(rid)])

    def mirror(self, buf: RowBuffer) -> np.ndarray:
        """``buf``'s host rows as this step's dense view will hold them (before
        its forward's ``step``): its rows in step order, then padding up to
        ``max_rows``. A copy."""
        if not buf.mirrored:
            raise ValueError(f"{buf.name} keeps no host mirror")
        space = self.space
        slab = self._slab
        if slab is None:
            slab = space._host
            if self._read_steps is None and not self.diff.same_step:
                self._read_steps = space._steps
        return space._host_view(buf, slab)[self._gather_idx()]

    def _gather_idx(self) -> np.ndarray:
        if self._gather is None:
            space, d = self.space, self.diff
            idx = np.full(space.max_rows, space.pad_slot, np.int64)
            idx[:d.n_active] = d.idx
            self._gather = idx
        return self._gather

    def _own(self) -> np.ndarray:
        slab = self._slab
        if slab is None:
            space = self.space
            with space._lock:
                slab, self._slab_steps = space._host.copy(), space._steps
            self._slab = slab
        return slab

    def _count(self, buf: RowBuffer, n: int) -> None:
        self._written[buf] = self._written.get(buf, 0) + n

    def write(self, buf: RowBuffer, slots: np.ndarray, values: np.ndarray) -> None:
        """Whole rows from the host: ``buf[slots[i]] = values[i]``."""
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        if buf.mirrored:
            self.space._host_view(buf, self._own())[slots] = values
        else:
            values = np.ascontiguousarray(values, dtype=buf.pad.dtype).reshape(len(slots), -1)
            rb = buf.row_bytes
            self._stage.write(buf._ptr, rb, slots, values.view(np.uint8), rb)
        self._count(buf, len(slots))

    def write_at(self, buf: RowBuffer, slots: np.ndarray, index: tuple, values) -> None:
        """Single elements of rows: ``buf[slots[i], *index_i] = values[i]``,
        with ``index`` a tuple of per-element index arrays over the row shape."""
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        if buf.mirrored:
            self.space._host_view(buf, self._own())[(slots, *index)] = values
            return
        flat = np.ravel_multi_index(tuple(np.asarray(i, dtype=np.int64) for i in index), buf.shape)
        values = np.ascontiguousarray(np.broadcast_to(values, slots.shape), dtype=buf.pad.dtype)
        size = buf.pad.itemsize
        offsets = np.ascontiguousarray(flat * size, dtype=np.int64)
        self._stage.write(buf._ptr, buf.row_bytes, slots, values.view(np.uint8), size, offsets)

    def copy_in(self, buf: RowBuffer, slots: np.ndarray, src: torch.Tensor, src_rows: np.ndarray) -> None:
        """Whole rows from a device store: ``buf[slots[i]] = src[src_rows[i]]``.
        Only into a buffer with no host mirror, which would not see them."""
        if buf.mirrored:
            raise ValueError(f"{buf.name}: copy_in into a mirrored buffer; write its rows from the host")
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        rb = buf.row_bytes
        self._stage.copy_rows(
            1, _store_ptr(buf, src), rb, np.ascontiguousarray(src_rows, dtype=np.int64), buf._ptr, rb, slots, rb,
        )
        self._count(buf, len(slots))

    def copy_out(self, buf: RowBuffer, slots: np.ndarray, dst: torch.Tensor, dst_rows: np.ndarray) -> None:
        """Leaving rows back to a device store: ``dst[dst_rows[i]] = buf[slots[i]]``,
        read before this step writes anything."""
        rb = buf.row_bytes
        self._stage.copy_rows(
            0, buf._ptr, rb, np.ascontiguousarray(slots, dtype=np.int64),
            _store_ptr(buf, dst), rb, np.ascontiguousarray(dst_rows, dtype=np.int64), rb,
        )

    def copy_in_each(self, buf: RowBuffer, slots: np.ndarray, srcs: Sequence[torch.Tensor]) -> None:
        """Whole rows from separate device tensors, one a row: ``buf[slots[i]] = srcs[i]``."""
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        ptrs = np.array([_tensor_ptr(buf, t) for t in srcs], dtype=np.int64)
        rb = buf.row_bytes
        self._stage.copy_rows(1, 0, 1, ptrs, buf._ptr, rb, slots, rb)
        self._keep += srcs
        self._count(buf, len(slots))

    def copy_out_each(self, buf: RowBuffer, slots: np.ndarray, dsts: Sequence[torch.Tensor]) -> None:
        """Leaving rows back to separate device tensors: ``dsts[i] = buf[slots[i]]``,
        read before this step writes anything."""
        ptrs = np.array([_tensor_ptr(buf, t) for t in dsts], dtype=np.int64)
        rb = buf.row_bytes
        self._stage.copy_rows(0, buf._ptr, rb, np.ascontiguousarray(slots, dtype=np.int64), 0, 1, ptrs, rb)
        self._keep += dsts

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

    def commit(self, forward: bool = True) -> np.ndarray:
        """The step is launching: its rows become the committed state, with
        the ``step`` of each buffer that declares one added to the step's
        rows of the mirror if the step runs the walk's forward (``forward``),
        which adds it on the device. Refused if anything committed since the
        proposal. Returns the slots freed by removed requests."""
        return self.space._commit(self, forward)

    def apply(self) -> None:
        """Apply the staged change to the device rows, on the stream that
        staged it: after the previous step's replay, before this one's."""
        if self._staged is not None:
            self.space._apply(self._staged)

    @property
    def stale(self) -> bool:
        """Something committed since this step was proposed, or a forward
        stepped the rows since its mirrors were copied."""
        return self.diff.base_version != self.space.batch.version or self._steps_moved()

    def _steps_moved(self) -> bool:
        steps = self.space._steps
        return (self._slab is not None and self._slab_steps != steps) or (
            self._read_steps is not None and self._read_steps != steps
        )

    def _check_joins(self) -> None:
        joins = len(self.diff.join_slots)
        if not joins:
            return
        missing = [buf.name for buf in self.space.buffers if self._written.get(buf, 0) < joins]
        if missing:
            raise RuntimeError(f"{', '.join(missing)}: joining rows were not written")


def _store_ptr(buf: RowBuffer, store: torch.Tensor) -> int:
    """``store``'s address, once it is checked to be rows of ``buf``'s kind;
    checked on first use only, as stores are long-lived."""
    if store is buf._store:
        return buf._store_ptr
    if store.device != buf.resident.device or store.dtype != buf.dtype:
        raise ValueError(f"{buf.name}: a store must be {buf.dtype} on {buf.resident.device}")
    if not store.is_contiguous() or store[0].numel() * store.element_size() != buf.row_bytes:
        raise ValueError(f"{buf.name}: a store must be contiguous rows of the buffer's shape")
    buf._store, buf._store_ptr = store, store.data_ptr()
    return buf._store_ptr


def _tensor_ptr(buf: RowBuffer, t: torch.Tensor) -> int:
    """``t``'s address, once it is checked to be one row of ``buf``."""
    if t.device != buf.resident.device or t.dtype != buf.dtype:
        raise ValueError(f"{buf.name}: a row's tensor must be {buf.dtype} on {buf.resident.device}")
    if not t.is_contiguous() or t.numel() * t.element_size() != buf.row_bytes:
        raise ValueError(f"{buf.name}: a row's tensor must be contiguous and of the row's shape")
    return t.data_ptr()


class _Block:
    """One staged step: where its segment tables sit in a device block."""

    __slots__ = ("entry", "phases", "stream")

    def __init__(self, entry: _RingEntry, phases: list[tuple[int, int, int]], stream: int | None):
        self.entry = entry
        self.phases = phases  # (table byte offset, segments, longest segment) per phase
        self.stream = stream  # the raw stream it was staged on


class _RingEntry:
    """A pinned host block and its device twin, rewritable once the copies
    and kernels of the step last staged in it (``staged``, a count) are done."""

    def __init__(self, nbytes: int, device: torch.device):
        accel = device.type != "cpu"
        self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=accel)
        self.view = self.host.numpy()
        self.device = torch.empty(nbytes, dtype=torch.uint8, device=device) if accel else self.host
        self.words = self.device.view(torch.int64)
        self.staged = 0
        # tensors the staged step copies to or from, held until it is done
        self.keep: list[torch.Tensor] = []


class RowSpace:
    """The persistent rows of one (node, walk)."""

    # an event is recorded every this many stagings (one costs as much as a
    # kernel launch); the ring is deep enough that reusing an entry waits on
    # one recorded after its step, long done
    _EVENT_EVERY = 4

    def __init__(
        self, max_rows: int, device: torch.device, depth: int = 8, block_bytes: int = 16 << 10,
        labels: Mapping[str, tuple[str, ...]] | None = None, capacity: int | None = None,
    ):
        self.max_rows = max_rows
        self.device = torch.device(device)
        # KV cache key -> the stream labels the walk's steps carry (``RowWalk``)
        self.labels: Mapping[str, tuple[str, ...]] = labels or {}
        self.batch = PersistentBatch(max_rows, capacity)
        self.capacity = self.batch.capacity
        # a slot always holding the pad row, which padding rows gather, and
        # one padding rows scatter into
        self.pad_slot = self.capacity
        self.trash_slot = self.capacity + 1
        self.buffers: list[RowBuffer] = []
        self._sealed = False
        # the device rows may have missed a forward's step: upload the mirrors whole
        self._resync = False
        # the committed state, against a step on another thread; reentrant
        # for the commit hooks, which read the mirrors
        self._lock = threading.RLock()
        # forwards committed, against a private slab copied before one
        self._steps = 0
        # the staging ring: the plan thread and the gpu thread both stage
        self._ring_lock = threading.Lock()
        self._ring = [_RingEntry(block_bytes, self.device) for _ in range(depth)]
        self._next = 0
        self._staged = 0  # stagings so far
        # (stagings it covers, event), oldest first
        self._events: deque[tuple[int, torch.Event]] = deque()
        self._memcpy = _cuda_memcpy_async() if self.device.type == "cuda" else None
        self._device_index = self.device.index if self.device.index is not None else (
            torch.cuda.current_device() if self._memcpy is not None else -1)

    def buffer(
        self, name: str, shape: Sequence[int], dtype: torch.dtype, pad=0, mirrored: bool = True,
        step: int = 0, moved: bool = True,
    ) -> RowBuffer:
        """Rows of ``shape``. Each forward of the walk adds ``step`` to the
        step's rows (a position, say): on the device after the forward, in
        its graph (``after_forward``), and on the mirror as the step commits,
        so a steady step uploads nothing.

        ``moved=False`` keeps the rows out of the per-step gather and scatter,
        for wide rows a step reads or changes only in part (a vocabulary-wide
        mask): the forward addresses them by slot instead."""
        if self._sealed:
            raise RuntimeError(f"{name}: the rows are allocated; register buffers before the first step")
        if step and dtype.is_floating_point:
            raise ValueError(f"{name}: only integer rows can step")
        if not moved and (mirrored or step):
            raise ValueError(f"{name}: unmoved rows are device state that doesn't step")
        buf = RowBuffer(self, name, tuple(shape), dtype, pad, mirrored, step, moved)
        self.buffers.append(buf)
        return buf

    @property
    def gather_slots(self) -> torch.Tensor:
        """``[max_rows]`` int32 on the device: each dense row's resident slot
        to read (padding rows: the pad slot), for unmoved buffers."""
        self.seal()
        return self._gather_at

    @property
    def scatter_slots(self) -> torch.Tensor:
        """``[max_rows]`` int32 on the device: each dense row's resident slot
        to write (padding rows: the trash slot), for unmoved buffers."""
        self.seal()
        return self._scatter_at

    @property
    def n_active(self) -> torch.Tensor:
        """``[1]`` int32 on the device: the rows the step being run holds, for
        in-graph updates that must skip padding."""
        self.seal()
        return self._n_active

    def seal(self) -> None:
        """Allocate every buffer's rows. Once all are registered, and before
        anything (a capture) records their addresses; the first step seals."""
        if self._sealed:
            return
        self._sealed = True
        mirrored = [b for b in self.buffers if b.mirrored]
        state = [b for b in self.buffers if not b.mirrored]
        slots = self.capacity + 2  # with the pad and trash slots
        rows = self.max_rows

        # the mirrored slab of resident rows
        size = 0
        for buf in mirrored:
            buf._offset = size
            size = _round(size + slots * buf.row_bytes)
        self._slab_bytes = size
        # what a commit steps on the mirror (``stage.step_mirrors``)
        self._step_desc = np.array(
            [(b._offset, b.pad.itemsize, b.pad.size, b.step) for b in mirrored if b.step], dtype=np.int64,
        ).reshape(-1, 4)
        self._host = np.zeros(size, dtype=np.uint8)
        for buf in mirrored:
            self._host_view(buf, self._host)[:] = buf.pad
        self._slab = torch.from_numpy(self._host).to(self.device)
        for buf in mirrored:
            buf._resident = self._slab[buf._offset:buf._offset + slots * buf.row_bytes].view(
                buf.dtype).view(slots, *buf.shape)
        for buf in state:
            buf._resident = torch.from_numpy(buf.pad).to(self.device).expand(slots, *buf.shape).contiguous()
        for buf in self.buffers:
            buf._ptr = buf._resident.data_ptr()
            if buf.moved:
                buf._device = torch.from_numpy(buf.pad).to(self.device).expand(rows, *buf.shape).contiguous()

        # the step header: its row count, then each dense row's slot to gather
        # from and to scatter to (padding rows: the pad and trash slots)
        header = np.empty(1 + 2 * rows, dtype=np.int32)
        header[0], header[1:1 + rows], header[1 + rows:] = 0, self.pad_slot, self.trash_slot
        self._header = torch.from_numpy(header).to(self.device)
        self._n_active = self._header[0:1]
        self._gather_at = self._header[1:1 + rows]
        self._scatter_at = self._header[1 + rows:]
        moved = [b for b in self.buffers if b.moved]
        self._gather_table = RowTable([(b._resident, b._device) for b in moved], self.device)
        # what a forward may change: device state, and stepped rows
        self._scatter_table = RowTable(
            [(b._resident, b._device) for b in moved if not b.mirrored or b.step], self.device,
        )

        self._stepped = [b for b in self.buffers if b.step]
        if self._stepped:
            self._arange = torch.arange(rows, dtype=torch.int32, device=self.device)
            self._active = torch.empty(rows, dtype=torch.bool, device=self.device)
        resident = sum(b.row_bytes for b in self.buffers) * slots
        logger.info(
            "persistent rows: %d slots of %d buffers (%.1f MiB resident), %d a step",
            self.capacity, len(self.buffers), resident / 2**20, rows,
        )

    def gather(self, n: int) -> None:
        """Fill the first ``n`` rows of every dense view from the slots of the
        step being run, padding past its rows: once the step's rows are
        applied and before its preprocess, which may read them (a loop-back
        token) and overwrite some, outside the step's graph."""
        move_rows(self._gather_table, self._gather_at, min(n, self.max_rows), gather=True)

    def after_forward(self) -> None:
        """What the walk's forward owes its rows, run after it, inside its
        graph when captured: each stepped buffer's ``step`` added to the
        step's rows, then every row the forward may have changed back to its
        slot. Never outside a forward of the walk."""
        if self._stepped:
            torch.lt(self._arange, self._n_active, out=self._active)
            for buf in self._stepped:
                buf._device.add_(self._active.view(-1, *(1,) * len(buf.shape)), alpha=buf.step)
        move_rows(self._scatter_table, self._scatter_at, self.max_rows, gather=False)

    def resync(self) -> None:
        """A step's rows committed but its forward may not have run: the next
        step uploads the mirrors whole, which puts the device rows back."""
        if any(not b.mirrored for b in self._stepped):
            raise RuntimeError("device-only stepped rows can't be resynced from the host")
        self._resync = True

    def _host_view(self, buf: RowBuffer, slab: np.ndarray) -> np.ndarray:
        slots = self.capacity + 2
        n = slots * buf.row_bytes
        return slab[buf._offset:buf._offset + n].view(buf.pad.dtype).reshape(slots, *buf.shape)

    def propose(self, rids: Sequence[int]) -> StepRows:
        self.seal()
        with self._lock:
            return StepRows(self, self.batch.propose(rids))

    def without(self, rids: Sequence[int]) -> StepRows:
        self.seal()
        with self._lock:
            return StepRows(self, self.batch.without(rids))

    def holds(self, rid: int) -> bool:
        """Whether ``rid`` is checked out here."""
        return self.batch.is_checked_out(rid)

    def committed(self, buf: RowBuffer, rid: int) -> np.ndarray | None:
        """A copy of ``rid``'s row of ``buf``'s committed mirror, or None if
        it holds none (a removed request's row, still to leave, names it no
        longer); safe against a step committing on another thread."""
        with self._lock:
            if not self.batch.is_checked_out(rid):
                return None
            return self._host_view(buf, self._host)[self.batch.slot_of(rid)].copy()

    def forget(self, rid: int) -> bool:
        """See ``PersistentBatch.forget``; safe against a step proposing or
        committing on another thread."""
        with self._lock:
            return self.batch.forget(rid)

    def check_mirrors(self) -> None:
        """Raise if a mirrored buffer's resident rows (the trash slot aside)
        differ from its mirror. Synchronizes: for tests and debugging only."""
        for buf in self.buffers:
            if not buf.mirrored:
                continue
            device = buf.resident[:self.trash_slot].cpu().numpy()
            host = buf.host[:self.trash_slot]
            bad = np.flatnonzero((device != host).reshape(len(host), -1).any(axis=1))
            if len(bad):
                raise AssertionError(
                    f"{buf.name}: device rows {bad.tolist()} differ from the mirror"
                )

    # staging

    def _stage(self, step: StepRows) -> _Block | None:
        d = step.diff
        stage = step._stage
        if self._resync:
            step._own()
        # the mirrors straight to their rows, which only this stream's steps read
        if step._slab is not None:
            stage.put(step._slab, self._slab.data_ptr())
        if not d.same_step:
            stage.header(d.idx, self.max_rows, self.pad_slot, self.trash_slot, self._header.data_ptr())
        if stage.is_empty():
            return None
        return self._upload(stage, step._keep)

    def _upload(self, stage, keep: list[torch.Tensor]) -> _Block:
        """The stage's tables and payload to a device block, as one copy."""
        total = stage.size()
        entry = self._take(total)
        entry.keep = keep
        base = entry.device.data_ptr()
        placed = stage.pack(entry.view, base)
        stream = None
        if self._memcpy is not None:
            # the raw handle directly: `torch.cuda.current_stream` costs ~20x more
            stream = torch._C._cuda_getCurrentRawStream(self._device_index)
            self._h2d(base, entry.host.data_ptr(), total, stream)
        elif self.device.type != "cpu":
            entry.device[:total].copy_(entry.host[:total], non_blocking=True)
        return _Block(entry, placed, stream)

    def _h2d(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        err = self._memcpy(dst, src, nbytes, _H2D, stream)
        if err:
            raise RuntimeError(f"cudaMemcpyAsync failed with error {err}")

    def _take(self, nbytes: int) -> _RingEntry:
        """The next ring entry, grown to ``nbytes`` if it must be, once the
        step last staged in it is done."""
        entry = self._ring[self._next]
        if entry.staged and self.device.type != "cpu":
            self._wait(entry.staged)
        if entry.host.numel() < nbytes:
            entry = self._ring[self._next] = _RingEntry(max(nbytes, 2 * entry.host.numel()), self.device)
        self._next = (self._next + 1) % len(self._ring)
        self._staged += 1
        entry.staged = self._staged
        return entry

    def _wait(self, staged: int) -> None:
        """Until the work of staging ``staged`` (and all before it) is done."""
        events = self._events
        while events and events[0][0] < staged:
            events.popleft()
        if not events:
            # nothing recorded since (its step was discarded, say): record now
            self._record()
        events[0][1].synchronize()

    def _record(self) -> None:
        stream = torch.accelerator.current_stream(self.device)
        self._events.append((self._staged, stream.record_event()))

    def _apply(self, block: _Block) -> None:
        entry = block.entry
        for at, count, longest in block.phases:
            copy_segments(entry.words, entry.view, at, count, longest, block.stream)
        if self.device.type != "cpu" and (
            not self._events or self._staged - self._events[-1][0] >= self._EVENT_EVERY
        ):
            self._record()

    def _commit(self, step: StepRows, forward: bool) -> np.ndarray:
        with self._lock:
            # the copy (or the read) took every step committed before it: none may have since
            if step._steps_moved():
                raise RuntimeError("a step's mirrors were copied before a forward that committed since")
            d = step.diff
            released = self.batch.commit(d)
            step.committed = True
            for fn in step._on_commit:
                fn(released)
            if step._slab is not None:
                self._host = step._slab
                self._resync = False
            if forward:
                self._steps += 1
                if d.n_active and len(self._step_desc):
                    step_mirrors(self._host, self._step_desc, self.capacity + 2, d.idx)
            return released

