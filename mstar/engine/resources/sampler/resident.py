"""The sampler on the node's resident rows (`resources/resident.py`).

Each request's per-row scalars (temperature, top-k, top-p, seed, penalty and
whatever else the spec enables) are narrow cells of the planner's rows,
written whenever its row is written whole: a new, displaced or stale row, and
a config update marks it stale. A captured step gathers them itself, off the
row of planner slots, so a resident step stages none of them on the host.

What only the device knows stays by the sampler's own row, which a request
keeps from registration to removal (`SamplerBuffers`): the RNG offset, the
last token, loop-back state, the seen-token mask and the generation history.
A slot is a cache and a request can lose it, so none of that can live on the
planner's rows. A resident step reads them by an index row of its sampler
rows, gathered on the host from what each planner slot holds and uploaded
when the planner's slot row changes, not every step. The offset is gathered
off its master, advanced and scattered back inside the graph through that
row, the in-graph scatter host-planned steps use, with no copy to keep in step.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import triton
import triton.language as tl

from mstar.engine.resources.resident import TRASH_SLOT
from mstar.engine.resources.sampler.utils import SamplingConfig, ScalarField, scalar_values
from mstar.utils.h2d import H2DMirror, PinnedStager

if TYPE_CHECKING:
    from mstar.engine.resources.resident import CapturedResidentStep, ResidentPlan, ResidentPlanner
    from mstar.engine.resources.sampler.utils import AcceleratorGraphableSampler, SamplerBuffers
    from mstar.engine.resources.step import BucketKey

# the sampler's trash row: padding rows read its defaults and write their draws to it
SAMPLER_TRASH_ROW = 0


@triton.jit
def _gather_cells_kernel(
    slots_ptr, store_ptr, out_ptr, STRIDE: tl.constexpr, COL: tl.constexpr,
    STARTS: tl.constexpr, SIZES: tl.constexpr, DEFAULTS: tl.constexpr, TRASH: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(slots_ptr + i)
    for j in tl.static_range(len(SIZES)):
        value = tl.load(store_ptr + slot * STRIDE + COL + j)
        # the trash row stays zero, the sink to every other owner
        value = tl.where(slot == TRASH, DEFAULTS[j], value)
        at = out_ptr + STARTS[j]
        if SIZES[j] == 8:
            tl.store(at.to(tl.pointer_type(tl.int64)) + i, value)
        elif SIZES[j] == 4:
            tl.store(at.to(tl.pointer_type(tl.int32)) + i, value.to(tl.int32))
        else:
            tl.store(at.to(tl.pointer_type(tl.int8)) + i, value.to(tl.int8))


def _encoder(dtype: torch.dtype) -> Callable[[float], int]:
    """A scalar as an int64 cell: a float as its 32-bit pattern."""
    if dtype == torch.float32:
        return lambda value: int(np.float32(value).view(np.int32))
    if dtype == torch.bool:
        return lambda value: 1 if value else 0
    return int


def _decode(cells: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return cells.to(torch.int32).view(torch.float32)
    if dtype == torch.bool:
        return cells != 0
    return cells.to(dtype)


class _ScalarRow:
    """A resident bucket's per-step scalars: one byte row, a typed region per
    field. Only its graph writes it, so its captures share one."""

    __slots__ = ("row", "starts", "views")

    def __init__(self, fields: tuple[ScalarField, ...], sizes: tuple[int, ...], bs: int, device: torch.device):
        self.starts: list[int] = []
        at = 0
        for size in sizes:
            self.starts.append(at)
            at += -(-bs * size // 8) * 8
        self.row = torch.zeros(at, dtype=torch.uint8, device=device)
        self.views = {
            f.name: self.row[start: start + bs * size].view(f.dtype)
            for f, start, size in zip(fields, self.starts, sizes, strict=True)
        }


@dataclass(slots=True)
class _IndexRow:
    """One (bucket, slot) capture's sampler rows: the gather half, then the
    scatter half the in-graph scatter writes by, where a dropped draw's row
    is the trash row."""
    rows: torch.Tensor  # [2 * bs]
    host: np.ndarray
    mirror: H2DMirror = field(default_factory=H2DMirror)
    # the planner's slot row these are the sampler rows of; a steady batch keeps one array
    staged: np.ndarray | None = None
    dropped: bool = False
    # rows -> the gather half's first ``rows``, for the scatters after a replay
    real: dict[int, torch.Tensor] = field(default_factory=dict)


class SamplerCells:
    """A sampler's scalars as cells of one node's resident rows, and the
    index rows its resident steps read the sampler's own rows by."""

    def __init__(self, planner: ResidentPlanner, fields: tuple[ScalarField, ...]):
        self._planner = planner
        self.fields = fields
        # one narrow table a field, side by side: a row's cells join its narrow run
        self._tables = [planner.table() for _ in fields]
        self._encoders = [_encoder(f.dtype) for f in fields]
        self._defaults = [enc(f.default) for enc, f in zip(self._encoders, fields, strict=True)]
        # for the torch path, made once: a capture can't copy one up
        self._default_cells = torch.tensor(self._defaults, dtype=torch.long, device=planner.device)
        self._sizes = tuple(torch.empty((), dtype=f.dtype).element_size() for f in fields)
        # rid -> its cells, set on the thread that registers or updates it.
        # Handle-keyed: `drop` purges it.
        self._cells: dict[int, list[int]] = {}
        # planner slot -> the sampler row of the request it was last written for
        self._row_of_slot = np.zeros(planner.max_slots, dtype=np.int64)
        self._rows: dict[BucketKey, _ScalarRow] = {}
        self._index: dict[tuple[BucketKey, int], _IndexRow] = {}
        self._stager = PinnedStager(torch.long)

    @property
    def node(self) -> str:
        return self._planner.node

    # The cells

    def set_config(self, rid: int, cfg: SamplingConfig, windowed: bool, stale: bool = False) -> None:
        """``rid``'s cells for ``cfg``; with ``stale``, its row is rewritten by
        the next step that reads it. Set before the mark, so a rewrite racing
        the update reads the new cells or is followed by one that does."""
        values = scalar_values(cfg, windowed)
        self._cells[rid] = [enc(values[f.name]) for enc, f in zip(self._encoders, self.fields, strict=True)]
        if stale:
            self._planner.mark_stale(rid)

    def drop(self, rid: int) -> None:
        self._cells.pop(rid, None)

    def row(self, rid: int, slot: int, sampler_row: int, row: list[int]) -> None:
        """``rid``'s cells into its row, and the sampler row its slot now stands for."""
        for table, cell in zip(self._tables, self._cells.get(rid, self._defaults), strict=True):
            row[table.col] = cell
        self._row_of_slot[slot] = sampler_row

    # The step's sampler rows

    def index_row(self, bucket: BucketKey, slot: int) -> _IndexRow:
        key = (bucket, slot)
        index = self._index.get(key)
        if index is None:
            # every row the trash row until staged
            rows = torch.full((2 * bucket.bs,), SAMPLER_TRASH_ROW, dtype=torch.long, device=self._planner.device)
            index = self._index[key] = _IndexRow(rows, np.full(2 * bucket.bs, SAMPLER_TRASH_ROW, dtype=np.int64))
        return index

    def stage(self, plan: ResidentPlan) -> None:
        """The step's sampler rows, from what each of its planner slots holds,
        both halves in one copy; nothing when the planner's slot row didn't
        change. Host work only, after the step's rows were written."""
        _, bucket, slot = plan.key
        index = self.index_row(bucket, slot)
        if index.staged is plan.slots and not index.dropped:
            return
        bs, n, host = bucket.bs, len(plan.slots), index.host
        np.take(self._row_of_slot, plan.slots, out=host[:n])
        host[n:bs] = SAMPLER_TRASH_ROW
        host[bs:] = host[:bs]
        self._stager.copy_(index.rows, host, mirror=index.mirror)
        index.staged, index.dropped = plan.slots, False

    def drop_draws(self, bucket: BucketKey, slot: int, request_ids, kept: frozenset) -> None:
        """Send the draws ``kept`` leaves out to the trash row in the scatter
        half, so their offsets and tokens stay as they were. After `stage`."""
        index = self.index_row(bucket, slot)
        keep = np.fromiter((rid in kept for rid in request_ids), dtype=bool, count=len(request_ids))
        if keep.all():
            return
        bs, n, host = bucket.bs, len(request_ids), index.host
        host[bs: bs + n] = np.where(keep, host[:n], SAMPLER_TRASH_ROW)
        self._stager.copy_(index.rows, host, mirror=index.mirror)
        index.dropped = True

    def gather_view(self, bucket: BucketKey, slot: int, rows: int | None = None) -> torch.Tensor:
        """The gather half, or its first ``rows``: a step's real rows."""
        index = self.index_row(bucket, slot)
        rows = bucket.bs if rows is None else rows
        view = index.real.get(rows)
        if view is None:
            view = index.real[rows] = index.rows[:rows]
        return view

    # The capture

    def scalar_row(self, bucket: BucketKey) -> _ScalarRow:
        row = self._rows.get(bucket)
        if row is None:
            row = self._rows[bucket] = _ScalarRow(self.fields, self._sizes, bucket.bs, self._planner.device)
        return row

    def sampler_for(self, buffers: SamplerBuffers, bucket: BucketKey, slot: int) -> AcceleratorGraphableSampler:
        """The captured sampler of a resident (bucket, slot): its scalars from
        the bucket's row, its offsets and tokens by the slot's index row."""
        return buffers.resident_sampler_for(
            bucket, slot, bucket.bs, self.scalar_row(bucket).views, self.index_row(bucket, slot).rows,
        )

    def gather(self, step: CapturedResidentStep) -> None:
        """Record the cells' gather into the bucket's scalar row."""
        tables = self._tables
        col = tables[0].col
        if any(t.col != col + j for j, t in enumerate(tables)):
            raise RuntimeError(f"resident planning on {self.node}: the sampler's cells aren't side by side")
        row = self.scalar_row(step.bucket)
        gather_cells(
            step.store, step.slots, tables[0].stride, col, row, self.fields, self._sizes,
            self._defaults, self._default_cells,
        )


def gather_cells(
    store: torch.Tensor, slots: torch.Tensor, stride: int, col: int, row: _ScalarRow,
    fields: tuple[ScalarField, ...], sizes: tuple[int, ...], defaults: list[int], default_cells: torch.Tensor,
) -> None:
    """Each field's cell of ``slots``' rows into its region of ``row``, the
    trash row reading ``defaults`` (``default_cells`` on the device)."""
    n = slots.numel()
    if store.is_cuda:
        _gather_cells_kernel[(n,)](
            slots, store, row.row, STRIDE=stride, COL=col, STARTS=tuple(row.starts),
            SIZES=sizes, DEFAULTS=tuple(defaults), TRASH=TRASH_SLOT, num_warps=1,
        )
        return
    cells = store.view(-1, stride)[:, col: col + len(fields)].index_select(0, slots)
    cells = torch.where((slots == TRASH_SLOT).unsqueeze(1), default_cells, cells)
    for j, f in enumerate(fields):
        row.views[f.name][:n].copy_(_decode(cells[:, j], f.dtype))
