"""Request state kept on the device across decode steps.

Two steps of one batch differ by "every length +1" and "a row crossed a page
boundary", yet a host plan rebuilds each row's write location, position and
recurrent slot every step, and the plan thread shares a core with the other two
step threads. Here each request holds a slot, a row in fixed-address tables on
the device. The owner of a state keeps its own table (the KV cache its block
table and lengths, the recurrent pool its slot, positions their offset), and a
captured step reads them by a per-step row of slots, updates them itself, and
needs from the host only what changed.

Every table write lands on the node's stream in step order: the graph's own
scatters, and the row copies `ResidentPlanner.flush` issues right before a
replay. The plan stream only ever writes a step's own staging buffers. So a row
is never rewritten under a graph that reads it, an abandoned pre-plan leaves the
tables alone, and nothing on the host path needs a kernel: a slot's fields sit
side by side in one row of one store, so rewriting it is one H2D.

The host stays the source of truth. A row is written whole the first time a
step reads it and whenever something other than a resident commit changed it
(a host-planned step, a fork); see `mark_stale`. An offload gives the slot up,
so a reload starts over as a new request does.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import chain
from typing import NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

from mstar.engine.resources.base import Resource
from mstar.engine.resources.step import BucketKey, SubmoduleStep
from mstar.utils import knobs
from mstar.utils.h2d import PinnedStager
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)

# Row 0 of every table, never a request's. Padding rows read and write it: its
# cells stay zero, which every owner's table takes to mean the sink.
TRASH_SLOT = 0


# The reads of buffers a resident step's graph fills, after `preprocess` has run.
_TABLE_READS: ContextVar[list[str] | None] = ContextVar("resident_table_reads", default=None)


def table_read(name: str) -> None:
    """Called by an accessor of a buffer that a resident step's tables fill in
    its graph. A `preprocess` that reads one would read the last step's value;
    the engine watches for it (`watch_table_reads`). Free outside a watch."""
    reads = _TABLE_READS.get()
    if reads is not None:
        reads.append(name)


@contextmanager
def watch_table_reads() -> Iterator[list[str]]:
    """The names `table_read` records on this thread while open."""
    reads: list[str] = []
    token = _TABLE_READS.set(reads)
    try:
        yield reads
    finally:
        _TABLE_READS.reset(token)


# What `watched_counts` records a read of the loop counter as. It matters only
# to a step that repeats, whose later replays don't run its prepare_inputs or
# preprocess again; a resident step's tables are a separate question.
LOOP_COUNTER = "the loop counter"


class _WatchedCounts(dict):
    """A row's loop counters, recording each read (`table_read`)."""

    def __getitem__(self, key):
        table_read(LOOP_COUNTER)
        return super().__getitem__(key)

    def get(self, key, default=None):
        table_read(LOOP_COUNTER)
        return super().get(key, default)

    def __contains__(self, key):
        table_read(LOOP_COUNTER)
        return super().__contains__(key)

    def __iter__(self):
        table_read(LOOP_COUNTER)
        return super().__iter__()

    def items(self):
        table_read(LOOP_COUNTER)
        return super().items()

    def values(self):
        table_read(LOOP_COUNTER)
        return super().values()


def resident_asked(submodule) -> bool:
    """Whether the serving config asks ``submodule``'s looping walks for
    resident planning: ``on`` every node, ``auto`` one that keeps it by default."""
    mode = knobs.resident_planning()
    return mode == "on" or (mode == "auto" and submodule.resident_by_default)


def watched_counts(counts: Mapping[str, int]) -> Mapping[str, int]:
    """``counts`` as a step's code reads them: inside a watch, every read is
    recorded as `LOOP_COUNTER`. The engine's own reads (an iteration input's
    cut) take ``counts`` directly."""
    return counts if _TABLE_READS.get() is None else _WatchedCounts(counts)


# What a captured step records: a small kernel apiece. The torch ops for the same
# work take several times the GPU time, and the replay's launch on the GPU thread.

@triton.jit
def _scatter_cells_kernel(cells_ptr, values_ptr, store_ptr):
    i = tl.program_id(0)
    tl.store(store_ptr + tl.load(cells_ptr + i), tl.load(values_ptr + i))


@triton.jit
def _gather_column_kernel(
    slots_ptr, store_ptr, out_ptr, plus_ptr, STRIDE: tl.constexpr, COL: tl.constexpr, PLUS: tl.constexpr,
):
    i = tl.program_id(0)
    value = tl.load(store_ptr + tl.load(slots_ptr + i) * STRIDE + COL)
    if PLUS:
        value += tl.load(plus_ptr + i)
    tl.store(out_ptr + i, value.to(out_ptr.dtype.element_ty))


@triton.jit
def _advance_column_kernel(slots_ptr, store_ptr, STRIDE: tl.constexpr, COL: tl.constexpr, TRASH: tl.constexpr):
    i = tl.program_id(0)
    slot = tl.load(slots_ptr + i)
    cell = slot * STRIDE + COL
    tl.store(store_ptr + cell, tl.load(store_ptr + cell) + tl.where(slot != TRASH, 1, 0))


def scatter_cells(store: torch.Tensor, cells: torch.Tensor) -> None:
    """``store[cells[0]] = cells[1]``."""
    if store.is_cuda:
        _scatter_cells_kernel[(cells.shape[1],)](cells[0], cells[1], store, num_warps=1)
    else:
        store.index_copy_(0, cells[0], cells[1])


def gather_column(
    store: torch.Tensor, slots: torch.Tensor, stride: int, col: int, out: torch.Tensor,
    plus: torch.Tensor | None = None,
) -> None:
    """``out[i] = store[slots[i], col]``, plus ``plus[i]``."""
    n = slots.numel()
    if store.is_cuda:
        _gather_column_kernel[(n,)](
            slots, store, out, plus if plus is not None else slots,
            STRIDE=stride, COL=col, PLUS=plus is not None, num_warps=1,
        )
        return
    value = store.index_select(0, slots * stride + col)
    out[:n].copy_(value if plus is None else value + plus)


def advance_column(store: torch.Tensor, slots: torch.Tensor, stride: int, col: int) -> None:
    """``store[slots[i], col] += 1`` for every row but the trash row's."""
    if store.is_cuda:
        _advance_column_kernel[(slots.numel(),)](slots, store, STRIDE=stride, COL=col, TRASH=TRASH_SLOT, num_warps=1)
    else:
        cells = slots * stride + col
        store.index_copy_(0, cells, store.index_select(0, cells) + (slots != TRASH_SLOT))


class ResidentTable:
    """``width`` columns of the planner's store, one cell per slot per column."""

    __slots__ = ("col", "width", "stride", "view")

    def __init__(self, width: int):
        self.width = width
        # set when the planner seals the store
        self.col = 0
        self.stride = 0
        self.view: torch.Tensor | None = None  # `[slots, width]`

    def cell(self, slot: int, col: int = 0) -> int:
        """A column's cell in the flat store."""
        return slot * self.stride + self.col + col


class ResidentStream(NamedTuple):
    """One resource's stream on a resident walk's steps, fixed at load by the
    step its capture template declares."""

    label: str
    # declared at span 0: the step reads the stream and appends nothing
    reads: bool


class ResidentWrites:
    """What one step changes. ``rows`` are runs (slot, first column, cells)
    that `flush` writes whole; ``cells`` and ``values`` are single cells, which
    ride the graph's own scatter. ``streams`` is each resource's stream, by
    resource key. ``repeats`` is how many replays the step runs (`StepContext.repeats`)."""

    __slots__ = ("streams", "rows", "cells", "values", "kv_len", "repeats")

    def __init__(self, streams: Mapping[str, ResidentStream], repeats: int = 1):
        self.repeats = repeats
        self.streams = streams
        self.rows: list[tuple[int, int, list[int]]] = []
        self.cells: list[int] = []
        self.values: list[int] = []
        # each KV pool's stored length of the row being rewritten, for the owners after it
        self.kv_len: dict[str, int] = {}

    def put(self, cell: int, value: int) -> None:
        self.cells.append(cell)
        self.values.append(value)

    def run(self, slot: int, col: int, cells: list[int]) -> None:
        """``slot``'s row from column ``col``. A run that starts where the one
        before it ends joins it, so a row of narrow cells and one block table
        is still one copy. Takes ``cells``, which a later run may extend."""
        rows = self.rows
        if rows:
            last_slot, last_col, last = rows[-1]
            if last_slot == slot and last_col + len(last) == col:
                last.extend(cells)
                return
        rows.append((slot, col, cells))


@dataclass(slots=True)
class ResidentBuffers:
    """What one (bucket, slot) capture reads per step. ``cells`` is flat cells
    then values, so a step's single-cell writes are one copy: a row has at most
    a length and a new column to write per KV pool. Unused entries write zero
    to cell zero, a trash cell, and ``dirty`` says whether any entry is in use."""
    slots: torch.Tensor
    cells: torch.Tensor
    # the row the host stages: ``slots``, unless a capture that drops stopped
    # rows (``ResidentPlanner.drop_rows``) writes ``slots`` from it each replay
    rows: torch.Tensor
    # each KV pool's stored lengths of the rows this step, by resource key: the
    # graph writes them, for what reads them after
    kv_len: dict[str, torch.Tensor]
    # the host row ``slots`` holds: a steady batch keeps one array, so no H2D and no compare
    staged: np.ndarray | None = None
    dirty: bool = False


@dataclass(slots=True)
class ResidentPlan:
    """One step's row of slots and what it changes."""
    key: tuple
    rids: tuple[int, ...]
    slots: np.ndarray
    seq: int
    commits: int
    writes: ResidentWrites
    # rows written whole by this plan, and rows whose length alone it refreshes
    taken: set[int]
    moved: set[int]
    # each KV pool's (rows, pages after the step) of its crossings, for `resident_advance`
    crossed: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass(slots=True)
class CapturedResidentStep:
    """What one capture's kernels share, at warmup and capture only."""
    bucket: BucketKey
    slot: int
    slots: torch.Tensor
    buffers: ResidentBuffers
    store: torch.Tensor
    # each resource's stream, by resource key
    streams: Mapping[str, ResidentStream]
    # each KV pool's block table, by resource key, for attention that reads a row's pages itself
    block_tables: dict[str, ResidentTable] = field(default_factory=dict)
    # How each attention read its pages: off the tables alone, or through a
    # host plan of the step. Nothing plans the bucket only if none needed one.
    reads_tables: bool = False
    host_planned: bool = False
    member: str = ""

    @property
    def buffer_slot(self) -> int | tuple[str, int]:
        return (self.member, self.slot) if self.member else self.slot


class ResidentPlanner:
    """One node's request slots, tables and per-step rows.

    Slots are a cache, not something a request can run out of. Every cell of a
    row mirrors host state, so a request can lose its slot whenever no step needs
    it, and it takes another with a whole-row rewrite when it next runs, as a new
    request does. ``2 * max_bs`` slots, ``max_bs`` being the widest capture of a
    resident walk, always hold a step, on every rank and whatever the commit
    timing: a slot is free, cooling or held, and a step can take any of them but
    its own rows, at most ``max_bs``, and those of the one earlier step that may
    still commit, at most ``max_bs``. Slot numbers are private to a rank's
    planner, and no collective reads one.
    """

    def __init__(
        self, node: str, walks: Mapping[str, Mapping[str, ResidentStream]], resources: Mapping[str, Resource],
        device: torch.device, max_bs: int,
    ):
        self.node = node
        self.walks = frozenset(walks)
        # each walk's stream per resource key, which every bucket of it declares
        self._walk_streams = dict(walks)
        self.device = device
        # request rows plus the trash row
        self.max_slots = 2 * max_bs + 1
        self._keys = tuple(resources)
        self._owners = tuple(resources.values())
        self._tables: list[ResidentTable] = []
        # the KV pools whose stored lengths a capture gathers, as they bind
        self._length_keys: list[str] = []
        self._store: torch.Tensor | None = None
        # columns of the one-cell tables, which a rewritten row always covers
        self.narrow_cols = 0
        self._buffers: dict[tuple[BucketKey, int], ResidentBuffers] = {}
        # each resource's stream on a bucket's steps, by bucket then resource key, from its capture
        self._streams: dict[BucketKey, Mapping[str, ResidentStream]] = {}
        # buckets whose steps need no host plan at all
        self.host_plan_free: set[BucketKey] = set()

        self._lock = threading.Lock()
        # rid -> slot. Handle-keyed: `release` purges it.
        self._slot_of: dict[int, int] = {}
        self._free: list[int] = list(range(self.max_slots - 1, TRASH_SLOT, -1))
        # (slot, seq of the last step that read it), freed under a step not yet committed
        self._cooling: list[tuple[int, int]] = []
        self._last_read = np.zeros(self.max_slots, dtype=np.int64)
        # Handle-keyed, like `_slot_of`. Rows to write whole, and rows a
        # host-planned step only advanced as a resident one would have.
        self._stale: set[int] = set()
        self._moved: set[int] = set()
        self._plan_seq = 0
        self._committed_seq = 0
        self._commits = 0
        self._staged: ResidentPlan | None = None
        # the last batch's rows, so a steady batch skips the per-row lookup
        self._last_rids: tuple[int, ...] | None = None
        self._last_slots: np.ndarray | None = None

        for key, resource in resources.items():
            resource.bind_resident(self, key)
        self._seal()
        # the largest flush up front, plus a cells row: growing the stager waits
        # out its copies, which on the GPU thread would be a sync in the step path
        self._stager = PinnedStager(torch.long, numel=max_bs * self._stride + 2 * self._cells_width(max_bs))

    # Tables

    def table(self, width: int = 1) -> ResidentTable:
        """A table of ``width`` columns. Only from `bind_resident`."""
        if self._store is not None:
            raise RuntimeError("resident tables are fixed once the planner is built")
        table = ResidentTable(width)
        self._tables.append(table)
        return table

    def length_table(self, key: str) -> ResidentTable:
        """KV pool ``key``'s table of stored lengths, which each capture
        gathers into a buffer of its own (`ResidentBuffers.kv_len`)."""
        self._length_keys.append(key)
        return self.table()

    def _seal(self) -> None:
        # The narrow tables first, then a span per wide table, so a row's used
        # cells are a run per span and the narrow cells join the first one.
        wide = [t for t in self._tables if t.width > 1]
        narrow = [t for t in self._tables if t.width == 1]
        self.narrow_cols = len(narrow)
        stride = self._stride = sum(t.width for t in self._tables)
        # int64: what the KV write addressing and the positions are, and index ops take
        self._store = torch.zeros(self.max_slots * stride, dtype=torch.long, device=self.device)
        rows = self._store.view(self.max_slots, stride)
        col = 0
        for t in (*narrow, *wide):
            t.col, t.stride = col, stride
            t.view = rows[:, col: col + t.width]
            col += t.width
        logger.info(
            "resident planning on %s: %d slots of %d cells, %.1f MiB, walks %s",
            self.node, self.max_slots - 1, stride, self._store.nbytes / 2**20, sorted(self.walks),
        )

    @property
    def store(self) -> torch.Tensor:
        return self._store

    def covers(self, bucket: BucketKey) -> bool:
        return bucket.graph_walk in self.walks

    # Slots

    def slot_of(self, rid: int) -> int | None:
        return self._slot_of.get(rid)

    def _take_slots(self, n: int, seq: int) -> list[int]:
        """``n`` slots for the new rows of plan ``seq``, whose held rows carry
        its stamp already. Caller holds the lock. A slot is taken only once the
        last step that read it has committed, or can't: until then that step's
        commit still advances the slot's host lengths. A step is planned behind
        the commit of the one before it, so only the newest earlier plan can
        still be waiting on one, however many before it never ran."""
        settled = max(self._committed_seq, seq - 2)
        if self._cooling:
            ready = [slot for slot, read in self._cooling if read <= settled]
            if ready:
                self._cooling = [(slot, read) for slot, read in self._cooling if read > settled]
                self._free.extend(ready)
        free = self._free
        taken = [free.pop() for _ in range(min(n, len(free)))]
        if len(taken) < n:
            taken += self._displace(n - len(taken), settled)
        return taken

    def _displace(self, n: int, settled: int) -> list[int]:
        """The slots of the ``n`` least recently run requests. Each loses its
        slot and takes another when it next runs; the slot's host mirrors go to
        the new tenant. Ordered by plan, not by clock or slot number, and a
        step's rows in the order they took their slots, so a rank with the same
        history picks the same requests. Nothing depends on that: a displaced
        row is host state and one H2D, and no collective sees it."""
        held = self._slot_of
        reads = self._last_read[np.fromiter(held.values(), dtype=np.int64, count=len(held))]
        open_rows = np.flatnonzero(reads <= settled)
        if open_rows.size < n:
            raise RuntimeError(
                f"resident planning on {self.node}: {n} rows found no slot among {self.max_slots - 1}, "
                f"{open_rows.size} of the {len(held)} held open. A step's rows and those of the one step "
                "behind it can't hold them all, so the slot accounting is wrong"
            )
        rids = list(held)
        slots = []
        for i in open_rows[np.argsort(reads[open_rows], kind="stable")][:n].tolist():
            rid = rids[i]
            slots.append(held.pop(rid))
            self._stale.discard(rid)
            self._moved.discard(rid)
        return slots

    def release(self, rid: int) -> None:
        """The request is gone, or offloaded: a reload is written whole anyway.
        Its slot cools until the last step that read it has committed; on the
        device the row is only read again behind that step, by stream order."""
        with self._lock:
            self._stale.discard(rid)
            self._moved.discard(rid)
            slot = self._slot_of.pop(rid, None)
            if slot is None:
                return
            # a handle comes back as another request, under a batch tuple equal to the old one
            self._last_rids = None
            if self._staged is not None and rid in self._staged.rids:
                self._staged.commits = -1
            seq = int(self._last_read[slot])
            if seq <= self._committed_seq:
                self._free.append(slot)
            else:
                self._cooling.append((slot, seq))

    def mark_stale(self, rid: int) -> None:
        """Something other than a resident commit changed ``rid``'s row."""
        if rid in self._slot_of:
            with self._lock:
                self._stale.add(rid)

    # The step

    def _cells_width(self, bs: int) -> int:
        """Single-cell writes a step of ``bs`` rows can queue: a length and a
        new column a row per KV pool."""
        return 2 * bs * max(1, len(self._length_keys))

    def buffers(self, bucket: BucketKey, slot: int) -> ResidentBuffers:
        """The per-step rows of one capture, every slot at the trash row until staged."""
        key = (bucket, slot)
        found = self._buffers.get(key)
        if found is None:
            bs, device = bucket.bs, self.device
            slots = torch.full((bs,), TRASH_SLOT, dtype=torch.long, device=device)
            found = self._buffers[key] = ResidentBuffers(
                slots=slots, rows=slots,
                cells=torch.zeros((2, self._cells_width(bs)), dtype=torch.long, device=device),
                kv_len={pool: torch.zeros(bs, dtype=torch.long, device=device) for pool in self._length_keys},
            )
        return found

    def drop_rows(self, bucket: BucketKey, slot: int) -> ResidentBuffers:
        """A capture of ``bucket`` that sends a stopped row's reads and writes to
        the trash slot: the host stages ``rows``, and the capture writes
        ``slots``, which every owner reads, from it each replay."""
        buffers = self.buffers(bucket, slot)
        if buffers.rows is buffers.slots:
            buffers.rows = buffers.slots.clone()
        return buffers

    def plan(self, step: SubmoduleStep, ctx) -> None:
        """Stage this step's rows; on the plan thread for a pre-plan. Its admit
        has run, so every page the step writes is on its stream."""
        lease = ctx.slot_lease
        rids = tuple(ctx.request_ids)
        key = (rids, lease.bucket, lease.slot)
        staged = self._staged
        if staged is not None:
            self._staged = None
            if not ctx.is_preplan and staged.key == key and staged.commits == self._commits:
                ctx.resident_plan = staged
                return
            # the step it was for never ran, or something ran first
            self._restore(staged)
        plan = self._build(rids, key, self._streams[lease.bucket], step, ctx)
        ctx.resident_plan = plan
        if ctx.is_preplan:
            self._staged = plan

    def drop_stage(self) -> None:
        """The pages a pre-plan's stage names were given back (a refused admit,
        an abandoned pre-plan): a retry over the same rows reserves its own, so
        it plans afresh instead of promoting cells that name the old ones."""
        staged, self._staged = self._staged, None
        if staged is not None:
            self._restore(staged)

    def _restore(self, plan: ResidentPlan) -> None:
        """Hand back the rows of a plan that won't run: it wrote nothing but its
        staging. The next plan stages its row again, even an equal one: this
        one's copy is on the plan stream, and the replacing step waits on no
        event for it."""
        with self._lock:
            self._stale |= {rid for rid in plan.taken if rid in self._slot_of}
            self._moved |= {rid for rid in plan.moved if rid in self._slot_of}
        buffers = self._buffers.get(plan.key[1:])
        if buffers is not None:
            buffers.staged = None

    def _build(
        self, rids: tuple[int, ...], key: tuple, streams: Mapping[str, ResidentStream], step: SubmoduleStep, ctx,
    ) -> ResidentPlan:
        lease = ctx.slot_lease
        laps = PhaseLaps(f"{ctx.plan_phase}.resident.")
        with self._lock:
            self._plan_seq += 1
            seq = self._plan_seq
            if rids == self._last_rids:
                slots = self._last_slots
            else:
                get = self._slot_of.get
                row = [get(rid) for rid in rids]
                if None in row:
                    # stamped first: a row of this step is no one's victim
                    self._last_read[[slot for slot in row if slot is not None]] = seq
                    new = [i for i, slot in enumerate(row) if slot is None]
                    for i, slot in zip(new, self._take_slots(len(new), seq), strict=True):
                        row[i] = self._slot_of[rids[i]] = slot
                        self._stale.add(rids[i])
                slots = np.array(row, dtype=np.int64)
                self._last_rids, self._last_slots = rids, slots
            stale = self._stale
            taken = stale.intersection(rids) if stale else set()
            stale -= taken
            moved = self._moved
            refreshed = moved.intersection(rids) if moved else set()
            moved -= refreshed
            refreshed -= taken
            self._last_read[slots] = seq
            rows = [(rid, self._slot_of[rid]) for rid in taken]
        if taken and (refusal := _row_refusal(step, streams, taken)) is not None:
            # the tables advance each row on its walk's streams, every replay; the step's rows fail
            raise ValueError(f"resident walk {ctx.graph_walk!r} on {self.node}: {refusal}")
        plan = ResidentPlan(
            key, rids, slots, seq, self._commits, ResidentWrites(streams, ctx.repeats), taken, refreshed,
        )
        laps.lap("rows")
        try:
            for rid, slot in rows:
                self.rewrite(rid, slot, plan.writes)
            laps.lap("rewrites")
            buffers = self.buffers(lease.bucket, lease.slot)
            if buffers.staged is not slots:
                self._stager.copy_(buffers.rows, slots, pad_value=TRASH_SLOT)
                buffers.staged = slots
            laps.lap("slot_row")
            for resource in self._owners:
                resource.resident_stage(plan)
            laps.lap("stage")
            self._stage_cells(plan, buffers)
            laps.lap("cells")
        except BaseException:
            # nothing wrote these rows, so the next step that reads them must
            self._restore(plan)
            raise
        return plan

    def rewrite(self, rid: int, slot: int, writes: ResidentWrites) -> None:
        """Queue ``rid``'s row whole: the narrow columns as one run, which the
        owners fill, and each wide table's used cells as a run its owner adds."""
        row = [0] * self.narrow_cols
        if row:
            writes.run(slot, 0, row)
        for resource in self._owners:
            resource.resident_row(rid, slot, row, writes)

    def _stage_cells(self, plan: ResidentPlan, buffers: ResidentBuffers) -> None:
        """One copy, on the plan stream: the graph scatters these first thing."""
        cells = plan.writes.cells
        if not cells and not buffers.dirty:
            return
        n = buffers.cells.shape[1]
        staged = np.zeros(2 * n, dtype=np.int64)
        staged[: len(cells)] = cells
        staged[n: n + len(cells)] = plan.writes.values
        buffers.dirty = bool(cells)
        self._stager.copy_(buffers.cells, staged)

    def flush(self, plan: ResidentPlan) -> None:
        """The rows `plan` rewrites, on the step's stream: behind every earlier
        step, ahead of this one. On the GPU thread, right before the replay."""
        rows = plan.writes.rows
        if not rows:
            return
        stride, store = self._stride, self._store
        dsts = [store[slot * stride + col: slot * stride + col + len(cells)] for slot, col, cells in rows]
        self._stager.copy_runs_(dsts, np.fromiter(chain.from_iterable(c for _, _, c in rows), dtype=np.int64))

    def commit(self, step: SubmoduleStep, ctx) -> None:
        """After the resources' commit, before the step's commit event: the
        next pre-plan reads what this leaves."""
        plan = ctx.resident_plan if ctx.resident is not None else None
        self._commits += 1
        if plan is None:
            self._note_host_step(step, ctx)
            return
        for resource in self._owners:
            resource.resident_advance(plan)
        self._committed_seq = plan.seq

    def _note_host_step(self, step: SubmoduleStep, ctx) -> None:
        """A host-planned step moved its rows without their tables. A
        one-token row moved as a resident step would have, and needs only its
        length and any new column; anything else is written whole."""
        held = self._slot_of
        if not held:
            return
        segments = step.segments or [seg for s in step.steps.values() for seg in s.segments or ()]
        one = [seg.request_id for seg in segments if seg.span == 1 and seg.request_id in held]
        rest = [seg.request_id for seg in segments if seg.span != 1 and seg.request_id in held]
        with self._lock:
            self._moved.update(one)
            self._stale.update(rest)

    # The capture

    def capture_step(self, bucket: BucketKey, slot: int, step: SubmoduleStep) -> CapturedResidentStep:
        """What a capture of ``bucket`` records around its forward. Its walk
        passed `step_refusal` at load on its widest bucket; the others declare
        the same step or the worker stops."""
        streams = self._streams.get(bucket)
        if streams is None:
            want = self._walk_streams[bucket.graph_walk]
            refusal = step_refusal(self._keys, step)
            if refusal is None and declared_streams(self._keys, step) != want:
                refusal = f"its streams {declared_streams(self._keys, step)} aren't its walk's {dict(want)}"
            if refusal is not None:
                raise SystemExit(f"resident walk {bucket.graph_walk!r} on {self.node}: bucket {bucket} {refusal}")
            # once per bucket: each of its captures, and every step it replays, reads these
            streams = self._streams[bucket] = want
        buffers = self.buffers(bucket, slot)
        lease = step.ctx.slot_lease if step.ctx is not None else None
        return CapturedResidentStep(bucket, slot, buffers.slots, buffers, self._store, streams,
                                    member=lease.member if lease is not None else "")

    def gather(self, step: CapturedResidentStep) -> None:
        if self._length_keys:
            # before the owners' reads: a row that crossed writes into its new
            # page. Only a KV pool queues single cells.
            scatter_cells(self._store, step.buffers.cells)
        for resource in self._owners:
            resource.resident_gather(step)
        if step.reads_tables and not step.host_planned:
            self.host_plan_free.add(step.bucket)

    def run_gather(self, step: CapturedResidentStep) -> None:
        for resource in self._owners:
            resource.resident_run_gather(step)

    def commit_in_graph(self, step: CapturedResidentStep) -> None:
        for resource in self._owners:
            resource.resident_commit(step)


# what a resource's part of a step may declare that a captured step can't carry
_UNCARRIED = ("pre_forks", "post_forks", "retention", "sink", "advance", "pos_ids", "fork_rids")


def step_refusal(keys: Collection[str], step: SubmoduleStep | None) -> str | None:
    """Why ``step`` can't be a resident step, or None: for each resource of
    ``keys``, one stream that it appends one token a row to (span 1) or only
    reads (span 0), each row advancing alone."""
    if step is None:
        return "it declares no step"
    for key, resource_step in step.steps.items():
        uncarried = next((name for name in _UNCARRIED if getattr(resource_step, name, None)), None)
        if uncarried is None and getattr(resource_step, "commit", True) is False:
            # a step that keeps no tokens: the table's commit would advance past its stream
            uncarried = "commit=False"
        # `retention` (bounded KV) frees a stream's front pages at commit, a step
        # behind the device: a row planned past it would point at freed pages
        if uncarried is not None:
            return f"resource {key!r} declares {uncarried}, which a resident step can't carry"
    for key in keys:
        segments = _segments(key, step)
        if not segments:
            continue  # a resource with no segments has no stream to refuse
        labels = {seg.label for seg in segments}
        spans = {seg.span for seg in segments}
        if len(labels) != 1 or spans not in ({1}, {0}):
            return (
                f"resource {key!r} declares {len(segments)} segments over labels {sorted(labels)} and spans "
                f"{sorted(spans)}, not one token a row in one stream or a stream it only reads"
            )
    return None


def declared_streams(keys: Collection[str], step: SubmoduleStep) -> dict[str, ResidentStream]:
    """Each resource's stream on a step that passed `step_refusal`: the label
    of its own segments, or of the step's when it names none. A resource with
    neither has no stream."""
    streams = {}
    for key in keys:
        segments = _segments(key, step)
        if segments:
            streams[key] = ResidentStream(segments[0].label, reads=segments[0].span == 0)
    return streams


def _row_refusal(
    step: SubmoduleStep, streams: Mapping[str, ResidentStream], rids: Collection[int],
) -> str | None:
    """Why a row of ``rids`` in ``step`` declares other than its walk's
    ``streams``. Asked of the rows a step writes whole, which every row is once."""
    for key, want in streams.items():
        span = 0 if want.reads else 1
        for seg in _segments(key, step):
            if seg.request_id in rids and (seg.label != want.label or seg.span != span):
                return (
                    f"row {seg.request_id} declares {seg.label!r} at span {seg.span} on resource {key!r}, "
                    f"not its walk's {want.label!r} at span {span}"
                )
    return None


def _segments(key: str, step: SubmoduleStep) -> list:
    resource_step = step.steps.get(key)
    if getattr(resource_step, "streamless", False):
        return []
    return list((resource_step.segments if resource_step is not None else None) or step.segments or ())


# a capability's verdict when the serving config doesn't ask for it
NOT_ASKED = "not asked for"


def build_planner(
    node: str, steps: Mapping[str, SubmoduleStep | None], resources: Mapping[str, Resource],
    resource_nodes: Mapping[str, Collection[str]], device: torch.device,
    max_capture_bs: Callable[[Collection[str]], int], probe: Callable[[str], str | None] = lambda walk: None,
) -> tuple[ResidentPlanner | None, dict[str, str]]:
    """A planner for the walks of ``steps`` that can run on resident tables,
    or None, and why each other walk can't. ``steps``: each walk's step as
    its widest capture declares it; ``probe``: the last check, of a walk that
    passed the others. Every resource of the node must be able to, since a
    capture is one graph: one host-planned resource would need the row a
    resident step no longer stages."""
    refusal = None if resources else "node: no resource of it keeps a table"
    for key in resources:
        if refusal is None and len(resource_nodes.get(key, ())) > 1:
            refusal = f"resource {key!r}: it is shared by nodes {sorted(resource_nodes[key])}"
    if refusal is not None:
        return None, dict.fromkeys(steps, refusal)
    refusals: dict[str, str] = {}
    walks: dict[str, dict[str, ResidentStream]] = {}
    for walk, step in steps.items():
        refusal = step_refusal(resources, step)
        if refusal is not None:
            refusal = f"node: {refusal}"
        else:
            streams = declared_streams(resources, step)
            refusal = next((
                f"resource {key!r}: {why}" for key, resource in resources.items()
                if (why := resource.resident_refusal(key, streams)) is not None
            ), None)
        if refusal is None:
            refusal = probe(walk)
        if refusal is None:
            walks[walk] = streams
        else:
            refusals[walk] = refusal
    if not walks:
        return None, refusals
    return ResidentPlanner(node, walks, resources, device, max_capture_bs(walks)), refusals
