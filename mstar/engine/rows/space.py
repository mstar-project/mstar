"""Device row buffers of a persistent batch, kept in step with their host mirrors.

A ``RowSpace`` owns one ``PersistentBatch`` and every ``RowBuffer`` the
resources of its (node, walk) register. Each buffer has two levels on the
device: resident rows, one slot per live request (``capacity`` of them, plus
a padding slot and a trash slot), and a dense step view of ``max_rows`` rows
at a fixed address, which the captured graphs read and the forward updates
in place. Row ``i`` of the dense view holds the request the last committed
step ran at ``i``, and is that request's state; its resident slot is where
the state waits while the request is out of the steps. So a step that runs
the same requests in the same order moves nothing, and a step that changes
some moves only those: a row whose request leaves its place goes back to
its slot, and a request taking a place is copied in from its slot, or
written whole if it joins. A step proposes its rows (``propose``), the
resources say what joining rows hold and what changed (``StepRows``), and
everything reaches the device as one host-to-device copy of segment tables
and their payload, which a copy kernel a phase runs.

A mirrored buffer's host copy, by slot, is the truth: proposals are
copy-on-write, so a step that is refused or goes stale leaves it untouched.
Buffers without a mirror hold device state.

A step outside the walk that runs a held request (a mixed step, say) reads
its state where it keeps it outside the rows: the request's row is lent
(``lend``), its device state read back, and it keeps its slot and place
until a step of the walk runs it again and writes it whole (a resync).
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.rows.batch import PersistentBatch, RowDiff
from mstar.engine.rows.copy import StagedTables, copy_segments
from mstar.engine.rows.stage import new_stage, step_mirrors
from mstar.utils.h2d import _H2D, _cuda_memcpy_async

logger = logging.getLogger(__name__)

_ALIGN = 16
_NO_SLOTS = np.empty(0, dtype=np.int64)

# a step's copy phases: rows leaving their place go back to their slots, then
# rows taking a place come in and leaving or lent rows are read back, then the
# host's writes (the last phase, where a stage puts them)
_UNSEAT, _READ, _WRITE = 0, 1, 2
_PHASES = 3


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
        self._dense_ptr = 0
        self._resident: torch.Tensor | None = None
        self._ptr = 0  # the resident rows'
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


class ElementWrites(NamedTuple):
    """A step's element writes into one unmirrored buffer (``StepRows.element_writes``):
    ``stage.write(base, row_bytes, slots, values, elem_bytes, offsets)``."""
    stage: object
    base: int
    row_bytes: int


class StepRows:
    """One proposed step: its diff, and what the resources write for it.

    Joining rows (resyncing ones among them) must be written in full
    (``write`` or ``copy_in``) for every buffer; ``stage`` refuses a proposal
    that leaves one stale. Leaving rows are read out with ``copy_out`` before
    anything overwrites them. Rows are named by resident slot
    (``diff.join_slots``, ``slot_of``); a write lands where the row's state is
    while the step runs it, its place in a moved buffer's dense view.
    """

    def __init__(
        self, space: RowSpace, diff: RowDiff, lent: set[int] | frozenset = frozenset(), runs: bool = True,
    ):
        self.space = space
        # the batch's diff, which commits; ``diff`` is what the resources write
        # from: resyncing rows join in place, and lent rows leave with nothing read out
        self._batch_diff = diff
        # the committed step's slots: what the dense rows hold now
        self._base_idx = space.batch.idx
        self._lends = space._lends
        self._resync_rids: list[int] = []
        if lent:
            # a step that runs its rows (not a check-in or a lend) resyncs the lent ones
            at = [i for i, rid in enumerate(diff.order) if rid in lent] if runs else []
            left = frozenset(rid for rid in diff.leave_rids if rid in lent)
            if at or left:
                self._resync_rids = [diff.order[i] for i in at]
                diff = dataclasses.replace(
                    diff, join_rids=diff.join_rids + self._resync_rids,
                    join_slots=np.concatenate([diff.join_slots, diff.idx[at]]),
                    same_step=False, dropped=diff.dropped | left,
                )
        self.diff = diff
        # a lend's rows (``RowSpace.lend``): read back, keeping their slots
        self.lent_rids: list[int] = []
        self.lent_slots = _NO_SLOTS
        self._pos: np.ndarray | None = None
        self._kept: np.ndarray | None = None
        self._resynced = False
        self._applied = False
        # this step's mirrors: a private copy of the committed slab, made on
        # first write, and the forwards it had taken (``RowSpace._steps``)
        self._slab: np.ndarray | None = None
        self._slab_steps = 0
        # the step's copies and host writes, by phase (``_UNSEAT`` ...)
        self._stage = new_stage(_PHASES)
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

    def _positions(self, slots: np.ndarray) -> np.ndarray:
        """Where the rows of ``slots``, all run by this step, sit in its dense view."""
        pos = self._pos
        if pos is None:
            pos = self._pos = np.full(self.space.capacity + 2, -1, np.int64)
            pos[self.diff.idx] = np.arange(self.diff.n_active)
        at = pos[slots]
        if len(at) and at.min() < 0:
            raise ValueError(f"slots {slots[at < 0].tolist()} are not rows of the step")
        return at

    def _write_to(self, buf: RowBuffer, slots: np.ndarray) -> tuple[int, np.ndarray]:
        """Where writes into rows of the step land: a moved buffer's dense view
        (the row's state while the step runs it), an unmoved one's slots."""
        if buf.moved:
            return buf._dense_ptr, self._positions(slots)
        return buf._ptr, slots

    def _read_from(self, buf: RowBuffer, slots: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Where rows are read back from, as ``(base, rows, which of slots)``:
        the dense view for a row that keeps its place through the step (a lent
        one), its slot otherwise, where a row leaving its place has gone by then."""
        everything = np.arange(len(slots))
        if not buf.moved:
            return [(buf._ptr, slots, everything)]
        kept = self._kept
        if kept is None:
            old, new = self._base_idx, self.diff.idx
            m = min(len(old), len(new))
            same = np.flatnonzero(old[:m] == new[:m])
            kept = self._kept = np.full(self.space.capacity + 2, -1, np.int64)
            kept[old[same]] = same
        place = kept[slots]
        dense = place >= 0
        if not dense.any():
            return [(buf._ptr, slots, everything)]
        if dense.all():
            return [(buf._dense_ptr, place, everything)]
        return [(buf._dense_ptr, place[dense], everything[dense]), (buf._ptr, slots[~dense], everything[~dense])]

    def mirror(self, buf: RowBuffer) -> np.ndarray:
        """``buf``'s host rows as this step's dense view will hold them (before
        its forward's ``step``): its rows in step order, then padding up to
        ``max_rows``. A copy."""
        return self.mirror_by_slot(buf)[self._gather_idx()]

    def mirror_by_slot(self, buf: RowBuffer) -> np.ndarray:
        """``buf``'s host rows by resident slot, as ``mirror`` reads them. A
        view, to read before this step writes the buffer or commits."""
        if not buf.mirrored:
            raise ValueError(f"{buf.name} keeps no host mirror")
        space = self.space
        slab = self._slab
        if slab is None:
            slab = space._host
            if self._read_steps is None and not self.diff.same_step:
                self._read_steps = space._steps
        return space._host_view(buf, slab)

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
        """Whole rows of the step from the host: ``buf[slots[i]] = values[i]``."""
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        if buf.mirrored:
            self.space._host_view(buf, self._own())[slots] = values
        values = np.ascontiguousarray(
            np.broadcast_to(np.asarray(values, dtype=buf.pad.dtype), (len(slots), *buf.shape)),
        ).reshape(len(slots), -1)
        rb = buf.row_bytes
        base, at = self._write_to(buf, slots)
        self._stage.write(base, rb, at, values.view(np.uint8), rb)
        self._count(buf, len(slots))

    def write_at(self, buf: RowBuffer, slots: np.ndarray, index: tuple, values) -> None:
        """Single elements of rows of the step: ``buf[slots[i], *index_i] = values[i]``,
        with ``index`` a tuple of per-element index arrays over the row shape."""
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        if buf.mirrored:
            self.space._host_view(buf, self._own())[(slots, *index)] = values
        flat = np.ravel_multi_index(tuple(np.asarray(i, dtype=np.int64) for i in index), buf.shape)
        values = np.ascontiguousarray(np.broadcast_to(values, slots.shape), dtype=buf.pad.dtype)
        size = buf.pad.itemsize
        offsets = np.ascontiguousarray(np.broadcast_to(flat * size, slots.shape), dtype=np.int64)
        base, at = self._write_to(buf, slots)
        self._stage.write(base, buf.row_bytes, at, values.view(np.uint8), size, offsets)

    def element_writes(self, buf: RowBuffer) -> ElementWrites:
        """Where a caller registers ``write_at``'s writes into ``buf`` itself
        (`PageArena.decode_rows`, say): the step's stage and the buffer's dense
        view, whose rows are the step's, by position."""
        if buf.mirrored or not buf.moved:
            raise ValueError(f"{buf.name}: only a moved buffer without a mirror takes element writes")
        return ElementWrites(self._stage, buf._dense_ptr, buf.row_bytes)

    def copy_in(self, buf: RowBuffer, slots: np.ndarray, src: torch.Tensor, src_rows: np.ndarray) -> None:
        """Whole rows of the step from a device store: ``buf[slots[i]] = src[src_rows[i]]``.
        Only into a buffer with no host mirror, which would not see them."""
        if buf.mirrored:
            raise ValueError(f"{buf.name}: copy_in into a mirrored buffer; write its rows from the host")
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        rb = buf.row_bytes
        base, at = self._write_to(buf, slots)
        self._stage.copy_rows(
            _WRITE, _store_ptr(buf, src), rb, np.ascontiguousarray(src_rows, dtype=np.int64), base, rb, at, rb,
        )
        self._count(buf, len(slots))

    def copy_out(self, buf: RowBuffer, slots: np.ndarray, dst: torch.Tensor, dst_rows: np.ndarray) -> None:
        """Leaving or lent rows back to a device store: ``dst[dst_rows[i]] = buf[slots[i]]``,
        read before this step writes anything."""
        rb = buf.row_bytes
        dst_ptr, dst_rows = _store_ptr(buf, dst), np.ascontiguousarray(dst_rows, dtype=np.int64)
        for base, at, which in self._read_from(buf, np.ascontiguousarray(slots, dtype=np.int64)):
            self._stage.copy_rows(_READ, base, rb, at, dst_ptr, rb, np.ascontiguousarray(dst_rows[which]), rb)

    def copy_in_each(self, buf: RowBuffer, slots: np.ndarray, srcs: Sequence[torch.Tensor]) -> None:
        """Whole rows of the step from separate device tensors, one a row: ``buf[slots[i]] = srcs[i]``."""
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        ptrs = np.array([_tensor_ptr(buf, t) for t in srcs], dtype=np.int64)
        rb = buf.row_bytes
        base, at = self._write_to(buf, slots)
        self._stage.copy_rows(_WRITE, 0, 1, ptrs, base, rb, at, rb)
        self._keep += srcs
        self._count(buf, len(slots))

    def copy_out_each(self, buf: RowBuffer, slots: np.ndarray, dsts: Sequence[torch.Tensor]) -> None:
        """Leaving or lent rows back to separate device tensors: ``dsts[i] = buf[slots[i]]``,
        read before this step writes anything."""
        ptrs = np.array([_tensor_ptr(buf, t) for t in dsts], dtype=np.int64)
        rb = buf.row_bytes
        for base, at, which in self._read_from(buf, np.ascontiguousarray(slots, dtype=np.int64)):
            self._stage.copy_rows(_READ, base, rb, at, 0, 1, np.ascontiguousarray(ptrs[which]), rb)
        self._keep += dsts

    def on_commit(self, fn: Callable[[np.ndarray], None]) -> None:
        """Run ``fn`` when this step commits, before its mirrors replace the
        committed ones (so ``buf.host`` still holds the leaving rows). It is
        passed the slots whose requests were removed while holding them, which
        have no host state left to update."""
        self._on_commit.append(fn)

    def stage(self, in_graph: bool = False) -> None:
        """Upload what this step changes, on the current stream. Nothing for a
        step that changes nothing. ``in_graph``: the step replays a graph
        captured with ``RowSpace.prologue``, which applies the upload itself,
        so ``apply`` launches nothing."""
        self._check_joins()
        with self.space._ring_lock:
            self._staged = self.space._stage(self, in_graph)

    def commit(self, forward: bool = True) -> np.ndarray:
        """The step is launching: its rows become the committed state, with
        the ``step`` of each buffer that declares one added to the step's
        rows of the mirror if the step runs the walk's forward (``forward``),
        which adds it on the device. Refused if anything committed since the
        proposal. Returns the slots freed by removed requests."""
        return self.space._commit(self, forward)

    def apply(self) -> None:
        """Apply the staged change to the device rows, on the stream that
        staged it: after the previous step's replay, before this one's. Once:
        a move applied twice (two rows trading places) would undo itself."""
        if self._staged is not None and not self._applied:
            self._applied = True
            self.space._apply(self._staged)

    def apply_ahead(self) -> None:
        """Apply an upload staged for the step's graph now, on the current
        stream, so writes into the rows made before the replay land after it
        (the replay then applies nothing). For the rare step that writes them."""
        if self._staged is not None and self._staged.in_graph:
            self.space._tables.apply()

    @property
    def stale(self) -> bool:
        """Something committed since this step was proposed, rows were lent
        or resynced since, or a forward stepped the rows since its mirrors
        were copied."""
        return (
            self.diff.base_version != self.space.batch.version or self._lends != self.space._lends
            or self._steps_moved()
        )

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

    __slots__ = ("entry", "phases", "stream", "in_graph")

    def __init__(
        self, entry: _RingEntry, phases: list[tuple[int, int, int]], stream: int | None, in_graph: bool = False,
    ):
        self.entry = entry
        self.phases = phases  # (table byte offset, segments, longest segment) per phase
        self.stream = stream  # the raw stream it was staged on
        self.in_graph = in_graph  # staged into the space's StagedTables, for the graph to apply


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
        staged_bytes: int = 256 << 10,
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
        # the device rows may have missed a forward's step: write the mirrors whole
        self._resync = False
        # held requests whose rows are lent (``lend``), and lends and resyncs so far
        self._lent: set[int] = set()
        self._lends = 0
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
        # where a captured step's upload lands, for its graph to apply; a larger
        # one is applied eagerly from a ring block before the replay
        self._staged_bytes = staged_bytes
        self._tables: StagedTables | None = None
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
                buf._dense_ptr = buf._device.data_ptr()

        # the step header: its row count, then each dense row's slot to gather
        # from and to scatter to (padding rows: the pad and trash slots)
        header = np.empty(1 + 2 * rows, dtype=np.int32)
        header[0], header[1:1 + rows], header[1 + rows:] = 0, self.pad_slot, self.trash_slot
        self._header = torch.from_numpy(header).to(self.device)
        self._n_active = self._header[0:1]
        self._gather_at = self._header[1:1 + rows]
        self._scatter_at = self._header[1 + rows:]
        self._moved = [b for b in self.buffers if b.moved]

        if self._memcpy is not None:
            self._tables = StagedTables(self._staged_bytes, self.device)
        self._dense_ptrs = frozenset(b._device.data_ptr() for b in self.buffers if b.moved)

        self._stepped = [b for b in self.buffers if b.step]
        if self._stepped:
            self._arange = torch.arange(rows, dtype=torch.int32, device=self.device)
            self._active = torch.empty(rows, dtype=torch.bool, device=self.device)
        resident = sum(b.row_bytes for b in self.buffers) * slots
        logger.info(
            "persistent rows: %d slots of %d buffers (%.1f MiB resident), %d a step",
            self.capacity, len(self.buffers), resident / 2**20, rows,
        )

    def after_forward(self) -> None:
        """What the walk's forward owes its rows, run after it, inside its
        graph when captured: each stepped buffer's ``step`` added to the
        step's rows. Never outside a forward of the walk."""
        if self._stepped:
            torch.lt(self._arange, self._n_active, out=self._active)
            for buf in self._stepped:
                buf._device.add_(self._active.view(-1, *(1,) * len(buf.shape)), alpha=buf.step)

    def prologue(self) -> None:
        """What a captured step's graph runs before its forward: the step's
        staged upload (``StepRows.stage(in_graph=True)``)."""
        if self._tables is None:
            raise RuntimeError("rows staged in a graph need CUDA")
        self._tables.apply()

    def owns(self, t: torch.Tensor) -> bool:
        """Whether ``t`` is (the start of) a dense step view: a captured graph
        fills it in its prologue, so it can be the graph's input as it is."""
        self.seal()
        return t.data_ptr() in self._dense_ptrs

    def resync(self) -> None:
        """A step's rows committed but its forward may not have run: the next
        step writes the mirrors whole, which puts the device rows back."""
        if any(not b.mirrored for b in self._stepped):
            raise RuntimeError("device-only stepped rows can't be resynced from the host")
        self._resync = True

    def drop_staged(self) -> None:
        """A step staged for its graph may not have replayed: drop its upload,
        on the current stream, so no later replay applies it."""
        if self._tables is not None:
            self._tables.clear()

    def _host_view(self, buf: RowBuffer, slab: np.ndarray) -> np.ndarray:
        slots = self.capacity + 2
        n = slots * buf.row_bytes
        return slab[buf._offset:buf._offset + n].view(buf.pad.dtype).reshape(slots, *buf.shape)

    def propose(self, rids: Sequence[int]) -> StepRows:
        self.seal()
        with self._lock:
            return StepRows(self, self.batch.propose(rids), self._lent)

    def without(self, rids: Sequence[int]) -> StepRows:
        self.seal()
        with self._lock:
            return StepRows(self, self.batch.without(rids), self._lent, runs=False)

    def lend(self, rids: Sequence[int]) -> StepRows | None:
        """The step that lends the rows ``rids`` hold here and aren't lent
        already (``StepRows.lent_rids``), for a step outside the walk that
        runs them; None if there are none. Each keeps its slot and place, and
        is written whole (resynced) by the next step of the walk that runs it."""
        self.seal()
        with self._lock:
            batch, lent = self.batch, self._lent
            rids = [rid for rid in rids if rid not in lent and batch.is_checked_out(rid)]
            if not rids:
                return None
            rows = StepRows(self, batch.without(()), lent, runs=False)
            rows.lent_rids = rids
            rows.lent_slots = np.array([batch.slot_of(rid) for rid in rids], dtype=np.int64)
            return rows

    def holds(self, rid: int) -> bool:
        """Whether ``rid`` is checked out here."""
        return self.batch.is_checked_out(rid)

    def holds_live(self, rid: int) -> bool:
        """Whether ``rid`` is checked out here and its row isn't lent."""
        return rid not in self._lent and self.batch.is_checked_out(rid)

    def committed(self, buf: RowBuffer, rid: int) -> np.ndarray | None:
        """A copy of ``rid``'s row of ``buf``'s committed mirror, or None if
        it holds none or its row is lent (a removed request's row, still to
        leave, names it no longer); safe against a step committing on
        another thread."""
        with self._lock:
            if rid in self._lent or not self.batch.is_checked_out(rid):
                return None
            return self._host_view(buf, self._host)[self.batch.slot_of(rid)].copy()

    def forget(self, rid: int) -> bool:
        """See ``PersistentBatch.forget``; safe against a step proposing or
        committing on another thread."""
        with self._lock:
            if rid in self._lent:
                self._lent.discard(rid)
                self._lends += 1
            return self.batch.forget(rid)

    def check_mirrors(self) -> None:
        """Raise if a mirrored buffer's device rows differ from its mirror:
        the dense view's for the committed step's rows, the resident slots'
        for the other held ones and the pad slot (lent rows aside, which are
        stale by design). Synchronizes: for tests and debugging only."""
        held = self.batch.held()
        lent = {held[rid] for rid in self._lent if rid in held}
        idx = self.batch.idx
        seated = set(idx.tolist())
        waiting = np.array(
            sorted(s for s in held.values() if s not in seated and s not in lent) + [self.pad_slot], dtype=np.int64,
        )
        live = np.array([i for i, s in enumerate(idx.tolist()) if s not in lent], dtype=np.int64)
        for buf in self.buffers:
            if not buf.mirrored:
                continue
            host = buf.host
            checks = [("slots", waiting, buf.resident.cpu().numpy()[waiting], host[waiting])]
            if buf.moved and len(live):
                checks.append(("places", live, buf.device.cpu().numpy()[live], host[idx[live]]))
            for what, at, device, mirror in checks:
                bad = at[(device != mirror).reshape(len(at), -1).any(axis=1)]
                if len(bad):
                    raise AssertionError(f"{buf.name}: device {what} {bad.tolist()} differ from the mirror")

    # staging

    def _stage(self, step: StepRows, in_graph: bool) -> _Block | None:
        d = step._batch_diff
        stage = step._stage
        if not d.same_step:
            self._stage_moves(step)
            stage.header(d.idx, self.max_rows, self.pad_slot, self.trash_slot, self._header.data_ptr())
        if self._resync:
            # the step's rows from the mirrors in their places, the rest in their slots
            step._resynced = True
            slab = step._own()
            stage.put(slab, self._slab.data_ptr())
            if d.n_active:
                at = np.arange(d.n_active)
                for buf in self._moved:
                    if buf.mirrored:
                        rows = np.ascontiguousarray(self._host_view(buf, slab)[d.idx]).reshape(d.n_active, -1)
                        stage.write(buf._dense_ptr, buf.row_bytes, at, rows.view(np.uint8), buf.row_bytes)
        if stage.is_empty():
            return None
        return self._upload(stage, step._keep, in_graph)

    def _stage_moves(self, step: StepRows) -> None:
        """The rows whose place changes: the committed step's row there goes
        back to its slot, and the new one comes in from its slot, but for
        joining and resyncing rows, which the resources write whole; places
        the step leaves get the pad row back."""
        old, new = step._base_idx, step._batch_diff.idx
        m = min(len(old), len(new))
        changed = np.flatnonzero(old[:m] != new[:m])
        out = np.concatenate([changed, np.arange(m, len(old))])
        into = np.concatenate([changed, np.arange(m, len(new))])
        written = step.diff.join_slots
        if len(written) and len(into):
            into = into[~np.isin(new[into], written)]
        pads = np.arange(len(new), len(old))
        if not (len(out) or len(into)):
            return
        stage = step._stage
        pad = np.full(len(pads), self.pad_slot, dtype=np.int64)
        for buf in self._moved:
            rb, dense, resident = buf.row_bytes, buf._dense_ptr, buf._ptr
            if len(out):
                stage.copy_rows(_UNSEAT, dense, rb, out, resident, rb, np.ascontiguousarray(old[out]), rb)
            if len(into):
                stage.copy_rows(_READ, resident, rb, np.ascontiguousarray(new[into]), dense, rb, into, rb)
            if len(pads):
                stage.copy_rows(_READ, resident, rb, pad, dense, rb, pads, rb)

    def _upload(self, stage, keep: list[torch.Tensor], in_graph: bool) -> _Block:
        """The stage's tables and payload to a device block, as one copy:
        for a graph's prologue, the space's StagedTables, header first."""
        total = stage.size()
        tables = self._tables
        in_graph = in_graph and tables is not None and StagedTables.HEADER + total <= tables.nbytes
        head = StagedTables.HEADER if in_graph else 0
        entry = self._take(head + total)
        entry.keep = keep
        base = tables.payload_base if in_graph else entry.device.data_ptr()
        placed = stage.pack(entry.view[head:], base)
        if in_graph:
            entry.view[:head] = tables.header(placed).view(np.uint8)
        stream = None
        if self._memcpy is not None:
            # the raw handle directly: `torch.cuda.current_stream` costs ~20x more
            stream = torch._C._cuda_getCurrentRawStream(self._device_index)
            self._h2d(base - head, entry.host.data_ptr(), head + total, stream)
        elif self.device.type != "cpu":
            entry.device[:total].copy_(entry.host[:total], non_blocking=True)
        return _Block(entry, placed, stream, in_graph)

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

    def _record(self, covers: int | None = None) -> None:
        stream = torch.accelerator.current_stream(self.device)
        self._events.append((self._staged if covers is None else covers, stream.record_event()))

    def _apply(self, block: _Block) -> None:
        entry = block.entry
        # a block staged for a graph is read by its replay, still to come:
        # an event recorded now covers only the stagings before it
        covers = self._staged - 1 if block.in_graph else self._staged
        if not block.in_graph:
            for at, count, longest in block.phases:
                copy_segments(entry.words, entry.view, at, count, longest, block.stream)
        if self.device.type != "cpu" and (
            not self._events or covers - self._events[-1][0] >= self._EVENT_EVERY
        ):
            self._record(covers)

    def _commit(self, step: StepRows, forward: bool) -> np.ndarray:
        with self._lock:
            # the copy (or the read) took every step committed before it: none may have since
            if step._steps_moved():
                raise RuntimeError("a step's mirrors were copied before a forward that committed since")
            if step._lends != self._lends:
                raise RuntimeError("a step proposed before rows were lent or resynced since")
            d = step._batch_diff
            released = self.batch.commit(d)
            step.committed = True
            for fn in step._on_commit:
                fn(released)
            if step._slab is not None:
                self._host = step._slab
            if step._resynced:
                self._resync = False
            lent = self._lent
            if step.lent_rids or (lent and (step._resync_rids or d.leave_rids)):
                lent.update(step.lent_rids)
                lent.difference_update(step._resync_rids)
                lent.difference_update(d.leave_rids)
                self._lends += 1
            if forward:
                self._steps += 1
                if d.n_active and len(self._step_desc):
                    step_mirrors(self._host, self._step_desc, self.capacity + 2, d.idx)
            return released

