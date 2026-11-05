"""Device loop-back rows: a node's loop-back values kept on the device by request.

A signal that loops back to its own node can stay on the device between steps
and between the replays of a run: a captured step reads each row's value at its
head and writes it at its tail, inside the graph (``head``, ``tail``), through a
device row of table indices, which a resident step gathers from the planner's
slots in the graph and the host stages for a host-planned one. A replay
therefore needs no host before it. The one host write a row gets is its first
value, routed like any other input. Row 0 is the trash row: padding rows read
and write it, and a dropped row (``device_stop: drop_rows``) lands there too.

Captured graphs hold the tables' addresses, so a table has a fixed number of
rows and a request takes one at admission (``ask``, ``reserve``) and gives it
back at removal, as it does the sampler's. An eager step reads and writes by
request (``read``, ``write``).

A streamed signal whose consumer is on this GPU (``step_repeats: stream_queue``)
keeps its items in a ring of the same rows, ``[rows, depth, *item]``: the node's
steps write each row's items at its write count (``ring_write``), its consumer
gathers a chunk by position (``ring_read``) and gives positions back as it moves
past them (``ring_release``). A row whose ring is full is not ready
(``admit_retrieve``) and caps no step it is in (``ring_room``).
"""
from __future__ import annotations

import threading
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.resident import gather_column
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ADMIT_OK, ASKS, AdmitOutcome, BucketKey, ResourceStep, StepContext
from mstar.utils import knobs
from mstar.utils.h2d import PinnedStager

if TYPE_CHECKING:
    from mstar.engine.resources.resident import (
        CapturedResidentStep,
        ResidentPlanner,
        ResidentStream,
        ResidentTable,
        ResidentWrites,
    )

TRASH_ROW = 0
# a table's most bytes: past it the node serves fewer requests at once
_TABLE_BYTES = 1 << 30


@dataclass(frozen=True)
class LoopbackRowsStep(ResourceStep):
    """What a step takes of the rows: nothing but its place in the sweep."""
    # a row is its request's, whatever streams the step declares
    streamless: ClassVar[bool] = True


@dataclass
class LoopbackRowsSpec(NodeResourceSpec):
    """Declared by the engine, for a node that keeps rows: a model writes none."""

    @property
    def resource_class(self) -> type[LoopbackRows]:
        return LoopbackRows


def rows_key(node: str) -> str:
    return f"loopback_rows:{node}"


class LoopbackRows(Resource):
    """One table ``[rows, *shape]`` per loop-back signal of a node."""

    def __init__(self, device: torch.device, capacity: int | None = None, decides: bool = True):
        self._device = device
        # request rows, the trash row excluded
        self.capacity = knobs.sampler_slots() if capacity is None else capacity
        # admission decides at one rank only, or ranks could split on a request
        self._decides = decides
        self._max_bs = 1
        self._tables: dict[str, torch.Tensor] = {}
        self._row_of: dict[Hashable, int] = {}
        self._free: list[int] = []
        # rows handed out so far, the trash row included
        self._rows = 1
        self._lock = threading.Lock()
        self._stager = PinnedStager(torch.long)
        # the last eager index uploaded, by its rids: a batch reads and writes the same rows
        self._last: tuple[tuple, torch.Tensor] | None = None
        # (bucket, cg slot) -> a capture's device row of table indices, and the rids it holds
        self._captured: dict[tuple[BucketKey, int], torch.Tensor] = {}
        self._staged: dict[tuple[BucketKey, int], tuple] = {}
        self._placeholders: dict[tuple[str, int], torch.Tensor] = {}
        self._resident: ResidentPlanner | None = None
        self._resident_rows: ResidentTable | None = None
        # a ring per streamed signal: its depth, and by (signal, rid) the items
        # written and the position its consumer has moved past
        self._depths: dict[str, int] = {}
        self._written: dict[tuple[str, Hashable], int] = {}
        self._consumed: dict[tuple[str, Hashable], int] = {}

    @classmethod
    def build(cls, spec: LoopbackRowsSpec, info: EngineResourceInfo) -> LoopbackRows:
        group = info.joint_comm_group
        return cls(info.device, decides=group is None or group.world_size == 1)

    # Requests

    def add(self, rid: Hashable) -> None:
        with self._lock:
            if rid not in self._row_of:
                if not self._free and self._rows > self.capacity:
                    raise RuntimeError(f"loop-back rows: all {self.capacity} are held (MSTAR_SAMPLER_SLOTS)")
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
            for name in self._depths:
                self._written.pop((name, rid), None)
                self._consumed.pop((name, rid), None)

    def has_row(self, rid: Hashable) -> bool:
        return rid in self._row_of

    @property
    def free_rows(self) -> int:
        return len(self._free) + self.capacity + 1 - self._rows

    def ingest_request(self, rid: int, overrides=None) -> None:
        if not self._decides:
            self.add(rid)

    def remove_request(self, rid: int) -> None:
        self.drop(rid)

    def ask(self, rid: int, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        return ASKS if self._decides and rid not in self._row_of else None

    def reserve(self, rid: int, node_name: str, graph_walk: str, published: None) -> int | None:
        room = self.free_rows
        if room <= 0:
            return room
        self.add(rid)
        return None

    def unreserve(self, rid: int) -> None:
        self.drop(rid)

    def grew_since(self, mark: int) -> bool:
        return self.free_rows > mark

    def admit_retrieve(self, rid: int, node_name: str, graph_walk: str, published=None) -> AdmitOutcome:
        """Not ready while a ring of its row is full: its consumer frees it."""
        if self._depths and any(self.ring_room(name, rid) <= 0 for name in self._depths):
            return AdmitOutcome(ok=True, ready=False)
        return ADMIT_OK

    # Rings

    def ask_ring(self, name: str, depth: int) -> None:
        """Before serving: keep ``name``'s items in a ring ``depth`` deep."""
        self._depths[name] = depth

    def ring_room(self, name: str, rid: Hashable) -> int:
        """The items ``rid``'s ring of ``name`` takes before its consumer frees any."""
        with self._lock:
            return self._depths[name] - self._written.get((name, rid), 0) + self._consumed.get((name, rid), 0)

    def ring_write(self, name: str, rids: Sequence[Hashable], items: torch.Tensor) -> list[int]:
        """Row ``i`` of ``items`` (``[rows, width, *item]``) into ``rids[i]``'s
        ring at its write count; each rid's first position. One copy for the
        step, on the stream behind it."""
        depth = self._depths[name]
        width = items.shape[1]
        table = self._tables.get(name)
        if table is None:
            row_bytes = max(depth * items[0, 0].numel() * items.element_size(), 1)
            self.capacity = max(min(self.capacity, _TABLE_BYTES // row_bytes), self._max_bs)
            table = self._tables[name] = torch.zeros(
                (self.capacity + 1, depth, *items.shape[2:]), dtype=items.dtype, device=self._device,
            )
        with self._lock:
            bases = [self._written.get((name, rid), 0) for rid in rids]
            for rid, base in zip(rids, bases, strict=True):
                self._written[(name, rid)] = base + width
            rows = [self._row_of.get(rid, TRASH_ROW) for rid in rids]
        n = len(rids) * width
        at = torch.empty(2 * n, dtype=torch.long, device=self._device)
        self._stager.copy_(at, [
            *(row for row in rows for _ in range(width)),
            *((base + i) % depth for base in bases for i in range(width)),
        ])
        table.index_put_((at[:n], at[n:]), items[:len(rids)].reshape(-1, *table.shape[2:]).to(table.dtype))
        return bases

    def ring_read(self, name: str, rid: Hashable, positions: Sequence[int]) -> torch.Tensor:
        """``[len(positions), *item]``: ``rid``'s ring of ``name`` at ``positions``."""
        depth = self._depths[name]
        at = torch.empty(len(positions), dtype=torch.long, device=self._device)
        self._stager.copy_(at, [position % depth for position in positions])
        return self._tables[name][self._row_of[rid]].index_select(0, at)

    def ring_release(self, name: str, rid: Hashable, until: int) -> None:
        """The consumer has moved past every position before ``until``."""
        with self._lock:
            key = (name, rid)
            if key in self._written:
                self._consumed[key] = max(self._consumed.get(key, 0), until)

    # Steps

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        return None  # a replay reads and writes its rows in the graph

    def resident_refusal(self, key: str, streams: Mapping[str, ResidentStream]) -> str | None:
        return None  # a request keeps its row, so a slot's cell only names it

    def build_accelerator_graph_buffers(self, slots, max_bs: int, max_seq_len: int) -> None:
        self._max_bs = max(self._max_bs, max_bs)

    def indices(self, bucket: BucketKey, slot: int) -> torch.Tensor:
        """The device row of table indices a capture of ``bucket`` on ``slot`` reads."""
        found = self._captured.get((bucket, slot))
        if found is None:
            found = self._captured[(bucket, slot)] = torch.full(
                (bucket.bs,), TRASH_ROW, dtype=torch.long, device=self._device,
            )
        return found

    def plan(self, step: LoopbackRowsStep, ctx: StepContext) -> None:
        lease = ctx.slot_lease
        if lease is None or ctx.resident is not None:
            return  # an eager step goes by request; a resident one gathers in its graph
        key = (lease.bucket, lease.slot)
        rids = tuple(ctx.padded_request_ids)
        if self._staged.get(key) != rids:
            self._staged[key] = rids
            with self._lock:
                rows = [self._row_of.get(rid, TRASH_ROW) for rid in rids]
            self._stager.copy_(self.indices(*key)[:len(rows)], rows)

    def bind_resident(self, planner: ResidentPlanner, key: str) -> None:
        self._resident = planner
        self._resident_rows = planner.table()

    def resident_row(self, rid: int, slot: int, row: list[int], writes: ResidentWrites) -> None:
        row[self._resident_rows.col] = self._row_of.get(rid, TRASH_ROW)

    def resident_gather(self, step: CapturedResidentStep) -> None:
        # a padding row reads the trash slot's zero, which is the trash row
        table = self._resident_rows
        gather_column(self._resident.store, step.slots, table.stride, table.col, self.indices(step.bucket, step.slot))

    def head(self, bucket: BucketKey, slot: int, name: str, target: torch.Tensor) -> bool:
        """In a capture, before the forward: each row's ``name`` into the
        static input ``target``. False while no table holds it (the first warm-up)."""
        table = self._tables.get(name)
        if table is None or table[0].numel() * target.shape[0] != target.numel():
            return False
        rows = table.index_select(0, self.indices(bucket, slot)[:target.shape[0]])
        target.copy_(rows.reshape(target.shape))
        return True

    def tail(self, bucket: BucketKey, slot: int, name: str, value: torch.Tensor) -> bool:
        """In a capture, after the forward: each row's ``name`` from ``value``.
        False for a value of another shape than the table's."""
        table = self._table(name, value)
        if table.shape[1:] != value.shape[1:]:
            return False
        table.index_copy_(0, self.indices(bucket, slot)[:value.shape[0]], value.to(table.dtype))
        return True

    def _table(self, name: str, like: torch.Tensor | None = None) -> torch.Tensor:
        table = self._tables.get(name)
        if table is None:
            if like is None:
                raise RuntimeError(f"no loop-back row of {name!r} was written yet")
            row_bytes = max(like[0].numel() * like.element_size(), 1)
            self.capacity = max(min(self.capacity, _TABLE_BYTES // row_bytes), self._max_bs)
            table = self._tables[name] = torch.zeros(
                (self.capacity + 1, *like.shape[1:]), dtype=like.dtype, device=self._device,
            )
        return table

    # By request: an eager step, a mixed step's read, a row's first write

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

    def write(self, name: str, rids: Sequence[Hashable], values: torch.Tensor) -> None:
        """Row ``i`` of ``values`` into the row of ``rids[i]`` (the trash row for a rid with none)."""
        table = self._table(name, values)
        table.index_copy_(0, self._index(rids), values.to(table.dtype))

    def write_rows(
        self, name: str, rids: Sequence[Hashable], values: torch.Tensor, positions: Sequence[int],
    ) -> None:
        """Row ``positions[i]`` of ``values`` into the row of ``rids[i]``."""
        table = self._table(name)
        values = values.reshape(values.shape[0], *table.shape[1:])
        if list(positions) != list(range(len(positions))):
            picked = torch.empty(len(positions), dtype=torch.long, device=values.device)
            self._stager.copy_(picked, list(positions))
            values = values.index_select(0, picked)
        self.write(name, rids, values[:len(positions)])

    def read(self, name: str, rids: Sequence[Hashable]) -> torch.Tensor:
        """``[len(rids), *shape]``: each rid's row of ``name``."""
        return self._table(name).index_select(0, self._index(rids))

    def row(self, name: str, rid: Hashable) -> torch.Tensor:
        """A view of ``rid``'s row of ``name``, valid until the next write."""
        return self._table(name)[self._row_of[rid]]

    def placeholder(self, name: str, rows: int) -> torch.Tensor:
        """``[rows, *shape]`` zeros, for an input a capture's ``head`` fills."""
        found = self._placeholders.get((name, rows))
        if found is None:
            table = self._table(name)
            found = self._placeholders[(name, rows)] = table.new_zeros((rows, *table.shape[1:]))
        return found
