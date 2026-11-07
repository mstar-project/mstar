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
steps reserve each row's positions before they launch (``ring_reserve``) and
write the items there behind the step (``ring_write``); its consumer gathers a
chunk by position (``ring_read``) and gives positions back as it moves past them
(``ring_release``), over any it was never handed (``ring_skip``: a run that ended
short, a withheld item). A row whose ring is full is not ready
(``admit_retrieve``) and caps each step it is in (``ring_room``).
"""
from __future__ import annotations

import threading
from collections.abc import Hashable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.resident import gather_column
from mstar.engine.resources.resident_hooks import (
    ResidentBinding,
    no_resident_effect,
    unbounded_repeats,
)
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ADMIT_OK, ASKS, AdmitOutcome, BucketKey, ResourceStep, StepContext
from mstar.graph.loop_indices import LoopLayout
from mstar.utils import knobs
from mstar.utils.h2d import PinnedStager

if TYPE_CHECKING:
    from mstar.engine.resources.resident import (
        CapturedResidentStep,
        ResidentPlanner,
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

    def __init__(self, device: torch.device, capacity: int | None = None):
        self._device = device
        # request rows, the trash row excluded
        self.capacity = knobs.sampler_slots() if capacity is None else capacity
        self._max_bs = 1
        self._tables: dict[Hashable, torch.Tensor] = {}
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
        # positions reserved but never handed to the consumer, which it moves past unread
        self._skipped: dict[tuple[str, Hashable], set[int]] = {}
        # the side stream a consumer's ring reads run on (``ring_read``)
        self._reads: torch.cuda.Stream | None = None

    @classmethod
    def build(cls, spec: LoopbackRowsSpec, info: EngineResourceInfo) -> LoopbackRows:
        return cls(info.device)

    # Requests

    def add(self, rid: Hashable) -> None:
        with self._lock:
            if rid not in self._row_of:
                if not self._free and self._rows > self.capacity:
                    raise RuntimeError(f"loop-back rows: all {self.capacity} are held (MSTAR_SAMPLER_SLOTS)")
                self._row_of[rid] = self._free.pop() if self._free else self._next_row()
            self._last = None
            self._staged.clear()  # a handle is reused: one tuple of rids can name other rows

    def _next_row(self) -> int:
        row, self._rows = self._rows, self._rows + 1
        return row

    def drop(self, rid: Hashable) -> None:
        with self._lock:
            row = self._row_of.pop(rid, None)
            if row is not None:
                self._free.append(row)
            self._last = None
            self._staged.clear()  # a handle is reused: one tuple of rids can name other rows
            for name in self._depths:
                self._written.pop((name, rid), None)
                self._consumed.pop((name, rid), None)
                self._skipped.pop((name, rid), None)

    def has_row(self, rid: Hashable) -> bool:
        return rid in self._row_of

    @property
    def free_rows(self) -> int:
        return len(self._free) + self.capacity + 1 - self._rows

    def remove_request(self, rid: int) -> None:
        self.drop(rid)

    def ask(self, rid: int, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        return ASKS if rid not in self._row_of else None

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

    def ring_table(self, name: str, item: torch.Tensor) -> str | None:
        """Before serving, from a capture's output: allocate ``name``'s ring for
        items like ``item``'s rows; why it can't, or None. It takes as many
        rows as the other tables and never fewer."""
        depth = self._depths[name]
        rows = self.capacity + 1
        size = rows * depth * item[0].numel() * item.element_size()
        if size > _TABLE_BYTES:
            return (
                f"rows: a ring {depth} deep for {rows} rows is {size >> 20} MiB, over the "
                f"{_TABLE_BYTES >> 20} MiB a table may take"
            )
        self._tables[name] = torch.zeros((rows, depth, *item.shape[1:]), dtype=item.dtype, device=self._device)
        return None

    def ring_reserve(self, name: str, rids: Sequence[Hashable], width: int) -> list[int]:
        """Before a step launches: ``width`` positions for each of ``rids``,
        each rid's first. Reserved first, so a step built while this one runs
        sees the room it takes."""
        with self._lock:
            bases = [self._written.get((name, rid), 0) for rid in rids]
            for rid, base in zip(rids, bases, strict=True):
                self._written[(name, rid)] = base + width
        return bases

    def ring_write(
        self, name: str, rids: Sequence[Hashable], items: torch.Tensor, bases: Sequence[int],
    ) -> None:
        """Row ``i`` of ``items`` (``[rows, width, *item]``) into ``rids[i]``'s
        ring from ``bases[i]``. One copy for the step, on the stream behind it."""
        depth = self._depths[name]
        width = items.shape[1]
        table = self._tables[name]
        with self._lock:
            rows = [self._row_of.get(rid, TRASH_ROW) for rid in rids]
        n = len(rids) * width
        at = torch.empty(2 * n, dtype=torch.long, device=self._device)
        self._stager.copy_(at, [
            *(row for row in rows for _ in range(width)),
            *((base + i) % depth for base in bases for i in range(width)),
        ])
        table.index_put_((at[:n], at[n:]), items[:len(rids)].reshape(-1, *table.shape[2:]).to(table.dtype))

    def ring_read(self, name: str, rid: Hashable, positions: Sequence[int]) -> torch.Tensor:
        """``[len(positions), *item]``: ``rid``'s ring of ``name`` at ``positions``,
        complete on return. The main thread reads it for a consumer that runs
        on another stream (a codec's lane), which nothing orders after this
        read, so it runs on a side stream and the host waits for it alone: the
        positions were written by a step that has already landed."""
        depth = self._depths[name]
        stream = self._read_stream()
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            at = torch.empty(len(positions), dtype=torch.long, device=self._device)
            self._stager.copy_(at, [position % depth for position in positions])
            window = self._tables[name][self._row_of[rid]].index_select(0, at)
        if stream is not None:
            stream.synchronize()
        return window

    def _read_stream(self) -> "torch.cuda.Stream | None":
        if self._device.type != "cuda":
            return None
        if self._reads is None:
            self._reads = torch.cuda.Stream(device=self._device)
        return self._reads

    def ring_release(self, name: str, rid: Hashable, until: int) -> None:
        """The consumer has moved past every position before ``until``."""
        with self._lock:
            key = (name, rid)
            if key in self._written:
                self._consumed[key] = max(self._consumed.get(key, 0), until)
                self._past_skipped(key)

    def ring_skip(self, name: str, rid: Hashable, positions: Sequence[int]) -> None:
        """Reserved positions the consumer is never handed: it moves past them
        as it reaches them, so they hold no room."""
        if not positions:
            return
        with self._lock:
            key = (name, rid)
            if key in self._written:
                self._skipped.setdefault(key, set()).update(positions)
                self._past_skipped(key)

    def _past_skipped(self, key: tuple[str, Hashable]) -> None:
        skipped = self._skipped.get(key)
        consumed = self._consumed.get(key, 0)
        if skipped:
            skipped.difference_update([p for p in skipped if p < consumed])
        while skipped and consumed in skipped:
            skipped.discard(consumed)
            consumed += 1
        self._consumed[key] = consumed

    # Steps

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        return None  # a replay reads and writes its rows in the graph

    def resident_binding(self) -> ResidentBinding:
        return ResidentBinding(
            check_streams=no_resident_effect,
            bind=self._bind_resident,
            row=self._resident_row,
            stage=no_resident_effect,
            gather=self._resident_gather,
            run_gather=no_resident_effect,
            commit_in_graph=no_resident_effect,
            advance=no_resident_effect,
            check_horizon=no_resident_effect,
            max_repeats=unbounded_repeats,
            rewind_run=no_resident_effect,
        )

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

    def _bind_resident(self, planner: ResidentPlanner, key: str) -> None:
        self._resident = planner
        self._resident_rows = planner.table()

    def _resident_row(self, rid: int, slot: int, row: list[int], writes: ResidentWrites) -> None:
        row[self._resident_rows.col] = self._row_of.get(rid, TRASH_ROW)

    def _resident_gather(self, step: CapturedResidentStep) -> None:
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

    def _table(self, name: Hashable, like: torch.Tensor | None = None) -> torch.Tensor:
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

    def iteration_rows(self, step: CapturedResidentStep) -> torch.Tensor:
        """The entry's owner rows before stopped rows are sent to trash.

        Only this column is gathered ahead of the active-row mask; the normal
        resident gather still runs once for all resources afterwards.
        """
        table = self._resident_rows
        indices = self.indices(step.bucket, step.slot)
        gather_column(self._resident.store, step.buffers.rows, table.stride, table.col, indices)
        return indices

    def prepare_iteration(self, walk: str, loop: LoopLayout) -> torch.Tensor:
        """Allocate the entry record before capture, never inside a replay body."""
        key = ("iteration", walk, loop.name)
        table = self._tables.get(key)
        if table is None:
            item = torch.empty(1, 4 + len(loop.ancestors), dtype=torch.long, device=self._device)
            table = self._table(key, item)
            table[:, 2:].fill_(-1)  # generation zero is a valid first request
        return table

    def iteration_head(
        self, walk: str, loop: LoopLayout, indices: torch.Tensor, entry: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Actual positions and live rows for one body, before any resource writes.

        ``entry`` stages request generation, graph pass, then strict ancestor
        counts. Planning never writes this table. Only an executed entry resets
        its position and stop; a skipped body never calls this operation.
        """
        if entry.shape != (len(indices), 2 + len(loop.ancestors)):
            raise ValueError("a Loop entry needs generation, graph pass and strict ancestor counts")
        table = self.prepare_iteration(walk, loop)
        previous = table.index_select(0, indices)
        requested = indices != TRASH_ROW
        if active is not None:
            requested = requested & active
        entering = requested & (previous[:, 2:] != entry).any(dim=1)
        position = torch.where(entering, torch.zeros_like(previous[:, 0]), previous[:, 0])
        stopped = (previous[:, 1] != 0) & ~entering
        live = requested & ~stopped
        current = torch.cat((position[:, None], stopped[:, None].to(entry.dtype), entry), dim=1)
        table.index_copy_(0, indices, torch.where(requested[:, None], current, previous))
        return position, live

    def iteration_tail(
        self, walk: str, loop: LoopLayout, indices: torch.Tensor, position: torch.Tensor,
        active: torch.Tensor, stopped: torch.Tensor | None = None,
    ) -> None:
        """Advance executed rows and remember their terminal iteration.

        Resource gathers may already have mapped inactive indices to trash;
        neither their old position nor their stop may alter that padding row.
        """
        real = indices != TRASH_ROW
        live = real & active
        table = self._tables[("iteration", walk, loop.name)]
        table[:, 0].index_copy_(0, indices, torch.where(real, position + live.to(position.dtype), 0))
        if stopped is not None:
            done = table[:, 1].index_select(0, indices).bool() | (live & stopped)
            table[:, 1].index_copy_(0, indices, (done & real).to(table.dtype))

    def finish_iteration(
        self, walk: str, loop: LoopLayout, rids: Sequence[Hashable], entry: torch.Tensor, stopped: torch.Tensor,
    ) -> None:
        """Publish a settled eager step through the same cursor as captured runs."""
        indices = self._index(rids)
        position, live = self.iteration_head(walk, loop, indices, entry)
        self.iteration_tail(walk, loop, indices, position, live, stopped)

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

    def input_copies(
        self, name: str, rids: Sequence[Hashable], target: torch.Tensor,
        copies: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Redirect copies into a captured head's input to its owning rows.

        The head overwrites its static input, so an earlier member or round
        boundary must write the table it reads. Return views without writing:
        an unsubmitted group may still refuse or need to snapshot its seeds.
        """
        table = self._table(name)
        width = table[0].numel()
        if (not target.is_contiguous() or target.numel() != len(rids) * width
                or target.dtype != table.dtype or target.device != table.device):
            raise ValueError("a captured row input does not match its owning rows")
        with self._lock:
            rows = [self._row_of.get(rid, TRASH_ROW) for rid in rids]
        redirected = []
        for source, destination in copies:
            start = destination.storage_offset() - target.storage_offset()
            if (source.numel() != destination.numel() or not destination.is_contiguous()
                    or destination.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()
                    or start < 0 or start + destination.numel() > target.numel()):
                raise ValueError("a captured row copy falls outside its input")
            source = source.reshape(-1)
            offset = 0
            while offset < source.numel():
                row, column = divmod(start + offset, width)
                size = min(width - column, source.numel() - offset)
                redirected.append((source.narrow(0, offset, size), table[rows[row]].view(-1).narrow(0, column, size)))
                offset += size
        return redirected

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
