"""Bounded membership shared by consecutive device runs.

The engine reserves graph/resource ownership before publishing a descriptor.
This resource only publishes identity and immutable owner bindings; it neither
selects ready requests nor reserves their KV, RNG or loop progress.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.resident_hooks import ResidentBinding, no_resident_effect, unbounded_repeats
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager


def run_rows_key(node: str) -> str:
    return f"__run_rows_{node}"


# One control word names both the descriptor generation and its publication phase.
_OFFERED, _ACTIVE, _CANCELLED, _RETIRED = range(4)


@triton.jit(do_not_specialize=["row", "expected", "replacement"],
            do_not_specialize_on_alignment=["row", "expected", "replacement"])
def _change(Control, row: tl.int32, expected: tl.int64, replacement: tl.int64):
    tl.atomic_cas(Control + row, expected, replacement, sem="acq_rel", scope="gpu")


@triton.jit
def _boundary(Control, Bindings, Copied, Ids, Out, CAPACITY: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    word = tl.atomic_add(Control + row, 0, sem="acquire", scope="gpu")
    handle = word >> 2
    if handle != 0:
        phase = word & 3
        if phase == 0:
            found = tl.atomic_cas(Control + row, word, word | 1, sem="acq_rel", scope="gpu")
            word = tl.where(found == word, word | 1, found)
            phase = word & 3
        if phase == 2:
            tl.atomic_cas(Control + row, word, (word & ~3) | 3, sem="acq_rel", scope="gpu")
        active = phase == 1
        handle = tl.where(active, word >> 2, 0)
    tl.store(Ids + row, handle)
    for col in range(WIDTH):
        value = tl.load(Bindings + ((handle - 1) % CAPACITY) * WIDTH + col, handle != 0, 0)
        tl.store(Out + row * WIDTH + col, value)
    if handle != 0:
        # Distinct from acceptance: all members' bindings have now been copied.
        tl.atomic_xchg(Copied + (handle - 1) % CAPACITY, handle, sem="release", scope="gpu")


@triton.jit
def _stop(Control, Ids, Stops):
    row = tl.program_id(0)
    handle = tl.load(Ids + row)
    if handle != 0 and tl.load(Stops + row):
        active = (handle << 2) | 1
        found = tl.atomic_cas(Control + row, active, (handle << 2) | 3, sem="acq_rel", scope="gpu")
        if found == ((handle << 2) | 2):
            tl.atomic_cas(Control + row, found, (handle << 2) | 3, sem="acq_rel", scope="gpu")


@triton.jit
def _drain(Control):
    row = tl.program_id(0)
    word = tl.atomic_add(Control + row, 0, sem="acquire", scope="gpu")
    if word != 0:
        tl.atomic_cas(Control + row, word, (word & ~3) | 3, sem="acq_rel", scope="gpu")


@dataclass(frozen=True)
class RunRow:
    """An immutable publication, independent of a recycled request handle."""
    handle: int
    request_id: int
    request_generation: int
    row: int
    bindings: tuple[int, ...]


class RunRowsSnapshot:
    """One retained capture's view of an atomic membership boundary.

    Every member consumes the same identity change, before any body executes.
    The resource may serve another retained capture, so this scratch belongs
    to the parent capture rather than to the resource's publication table.
    """

    def __init__(self, owner: RunRows, rows: int):
        if not 0 < rows <= owner.rows or owner.binding_width % 3:
            raise ValueError("run members need (resident slot, immunity, bound) bindings")
        self.owner = owner
        self.ids = torch.zeros(rows, dtype=torch.long, device=owner.device)
        self.bindings = torch.zeros((rows, owner.binding_width), dtype=torch.long, device=owner.device)
        self.changed = torch.zeros(rows, dtype=torch.bool, device=owner.device)
        self._previous = torch.zeros_like(self.ids)

    def accept(self) -> None:
        self._previous.copy_(self.ids)
        self.owner.boundary(self.ids, self.bindings)
        torch.ne(self.ids, self._previous, out=self.changed)

    def copy_from(self, accepted: RunRowsSnapshot) -> None:
        """Install one parent's accepted view into another retained member.

        Captured bodies keep their own fixed scratch addresses. Only the
        parent accepts publication; every member consumes its same change mask.
        """
        if accepted.owner is not self.owner or accepted.ids.shape != self.ids.shape:
            raise ValueError("captured members must share one membership owner and row capacity")
        self.ids.copy_(accepted.ids)
        self.bindings.copy_(accepted.bindings)
        self.changed.copy_(accepted.changed)

    def retire(self, completed: torch.Tensor) -> None:
        self.owner.stop(self.ids, completed)


@dataclass
class RunRowsSpec(NodeResourceSpec):
    rows: int = 1
    capacity: int = 2
    binding_width: int = 1

    @property
    def resource_class(self) -> type[RunRows]:
        return RunRows


class RunRows(Resource):
    """Membership descriptors retained until publication and output readers end.

    Host operations have one owner. Publication and cancellation run on
    ``publication_stream``; the body boundary runs on the execution stream.
    The caller orders owner initialization before publication, and waits for
    every submitted window before draining a closed publication window.
    """

    def __init__(self, device: torch.device, rows: int, capacity: int, binding_width: int):
        if rows < 1 or capacity < rows or binding_width < 1:
            raise ValueError("run membership requires positive rows/bindings and capacity >= rows")
        self.device, self.rows, self.capacity = device, rows, capacity
        self.binding_width = binding_width
        self.control = torch.zeros(rows, dtype=torch.long, device=device)
        self.bindings = torch.zeros((capacity, binding_width), dtype=torch.long, device=device)
        self.copied = torch.zeros(capacity, dtype=torch.long, device=device)
        self.publication_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        if self.publication_stream is not None:
            self.publication_stream.wait_stream(torch.cuda.current_stream(device))
        self.stager = PinnedStager(torch.long, numel=capacity * binding_width)
        self._free = list(range(capacity - 1, -1, -1))
        self._generation = [0] * capacity
        self._descriptors: dict[int, RunRow] = {}
        self._readers: dict[int, int] = {}
        self._row_handles = [0] * rows
        self._observed = [0] * rows
        self._retired: set[int] = set()
        self._published: set[int] = set()
        self._control_copy = (torch.empty(rows, dtype=torch.long, pin_memory=True)
                              if device.type == "cuda" else None)
        self._control_ready = None
        if self.publication_stream is not None:
            # Compilation alone leaves CUDA module loading until first launch,
            # which can wait for the active WHILE. Load the one scalar ABI now;
            # -1 cannot match a fresh zero control word, so this changes no row.
            with torch.cuda.stream(self.publication_stream):
                self._change(0, -1, 0)

    @classmethod
    def build(cls, spec: RunRowsSpec, info: EngineResourceInfo) -> RunRows:
        return cls(info.device, spec.rows, spec.capacity, spec.binding_width)

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        return None  # only the captured boundary changes membership

    def resident_binding(self) -> ResidentBinding:
        return ResidentBinding(
            check_streams=no_resident_effect, bind=no_resident_effect, row=no_resident_effect,
            stage=no_resident_effect, gather=no_resident_effect, run_gather=no_resident_effect,
            commit_in_graph=no_resident_effect, advance=no_resident_effect,
            check_horizon=self._check_horizon, max_repeats=unbounded_repeats, rewind_run=no_resident_effect,
        )

    def _check_horizon(self, step, ctx) -> None:
        if ctx.slot_lease is None or ctx.slot_lease.bucket.bs > self.rows:
            raise ValueError("joining membership does not cover this capture")

    def available_rows(self, width: int) -> tuple[int, ...]:
        """Observe only a completed control copy; never wait for an active run.

        The next copy runs on the publication stream, independent of WHILE.
        A stale observation can postpone admission but cannot retire a newer
        descriptor: acknowledgement checks the full generation word.
        """
        if not 0 < width <= self.rows:
            raise ValueError("membership polling exceeds its capture prefix")
        if self.publication_stream is None:
            self.acknowledge(self.control.tolist())
        else:
            if self._control_ready is not None and self._control_ready.query():
                self.acknowledge(self._control_copy.tolist())
                self._control_ready = None
            if self._control_ready is None:
                with torch.cuda.stream(self.publication_stream):
                    self._control_copy.copy_(self.control, non_blocking=True)
                    self._control_ready = torch.cuda.Event()
                    self._control_ready.record(self.publication_stream)
        return tuple(row for row, handle in enumerate(self._row_handles[:width])
                     if (not handle or handle in self._retired) and self._free)

    @property
    def idle(self) -> bool:
        """Every publication and output reader from the previous window ended."""
        return not self._descriptors

    def reserve_row(
        self, request_id: int, request_generation: int, row: int, bindings: Sequence[int],
    ) -> RunRow | None:
        """Take a descriptor credit; exhaustion leaves ordinary readiness intact."""
        if not 0 <= row < self.rows or len(bindings) != self.binding_width:
            raise ValueError("run row does not fit the captured membership layout")
        previous = self._row_handles[row]
        if previous and previous not in self._retired:
            return None
        if not self._free:
            return None
        slot = self._free.pop()
        self._generation[slot] += 1
        handle = self._generation[slot] * self.capacity + slot + 1
        descriptor = RunRow(handle, request_id, request_generation, row, tuple(bindings))
        self._descriptors[handle] = descriptor
        self._readers[handle] = 1
        self._row_handles[row] = handle
        return descriptor

    def retain(self, descriptor: RunRow) -> None:
        self._check(descriptor)
        self._readers[descriptor.handle] += 1

    def release(self, descriptor: RunRow) -> bool:
        """Return a credit only after retirement and the last reader's release."""
        self._check(descriptor)
        handle = descriptor.handle
        readers = self._readers[handle]
        if readers == 1 and handle not in self._retired:
            raise ValueError("run row still has an unacknowledged publication")
        if readers > 1:
            self._readers[handle] = readers - 1
            return False
        del self._readers[handle], self._descriptors[handle]
        self._retired.remove(handle)
        self._published.discard(handle)
        if self._row_handles[descriptor.row] == handle:
            self._row_handles[descriptor.row] = 0
        self._free.append((handle - 1) % self.capacity)
        return True

    def _check(self, descriptor: RunRow) -> None:
        if self._descriptors.get(descriptor.handle) is not descriptor:
            raise ValueError("run row descriptor was released or belongs to another resource")

    def _change(self, row: int, expected: int, replacement: int) -> None:
        if self.device.type == "cuda":
            _change[(1,)](self.control, row, expected, replacement, num_warps=1)
        elif int(self.control[row]) == expected:
            self.control[row] = replacement

    def _check_publication_stream(self) -> None:
        if self.publication_stream is not None and torch.cuda.current_stream(self.device) != self.publication_stream:
            raise ValueError("run membership writes must use its publication stream")

    def publish_row(self, descriptor: RunRow) -> None:
        """On the publication stream, after all new owner writes are queued."""
        self._check(descriptor)
        self._check_publication_stream()
        handle = descriptor.handle
        if handle in self._published or handle in self._retired:
            raise ValueError("run row was already published or withdrawn")
        slot = (handle - 1) % self.capacity
        self.stager.copy_(self.bindings[slot], descriptor.bindings)
        self._change(descriptor.row, self._observed[descriptor.row], handle << 2)
        self._published.add(handle)

    def cancel(self, descriptor: RunRow) -> None:
        self._check(descriptor)
        self._check_publication_stream()
        handle = descriptor.handle
        if handle not in self._published:
            self._retired.add(handle)
            return
        for phase in (_OFFERED, _ACTIVE):
            self._change(descriptor.row, (handle << 2) | phase, (handle << 2) | _CANCELLED)

    def boundary(self, ids: torch.Tensor, bindings: torch.Tensor) -> None:
        """Snapshot one membership generation for all members of this iteration."""
        if (ids.ndim != 1 or not 0 < len(ids) <= self.rows or bindings.shape != (len(ids), self.binding_width)
                or ids.dtype != torch.long or bindings.dtype != torch.long
                or ids.device != self.device or bindings.device != self.device
                or not ids.is_contiguous() or not bindings.is_contiguous()):
            raise ValueError("run boundary buffers do not match its membership layout")
        if self.device.type == "cuda":
            _boundary[(len(ids),)](
                self.control, self.bindings, self.copied, ids, bindings,
                self.capacity, self.binding_width, num_warps=1,
            )
            return
        for row, word in enumerate(self.control[:len(ids)].tolist()):
            handle, phase = word >> 2, word & 3
            if handle and phase == _OFFERED:
                self.control[row] = (handle << 2) | _ACTIVE
                phase = _ACTIVE
            elif handle and phase == _CANCELLED:
                self.control[row] = (handle << 2) | _RETIRED
            active = bool(handle and phase == _ACTIVE)
            ids[row] = handle if active else 0
            if active:
                slot = (handle - 1) % self.capacity
                bindings[row].copy_(self.bindings[slot])
                self.copied[slot] = handle
            else:
                bindings[row].zero_()

    def stop(self, ids: torch.Tensor, stops: torch.Tensor) -> None:
        if (ids.ndim != 1 or not 0 < len(ids) <= self.rows or stops.shape != ids.shape
                or ids.dtype != torch.long or stops.dtype != torch.bool
                or ids.device != self.device or stops.device != self.device
                or not ids.is_contiguous() or not stops.is_contiguous()):
            raise ValueError("run stop buffers do not match its membership layout")
        if self.device.type == "cuda":
            _stop[(len(ids),)](self.control, ids, stops, num_warps=1)
        else:
            for row, (handle, stop) in enumerate(zip(ids.tolist(), stops.tolist(), strict=True)):
                if handle and stop:
                    for phase in (_ACTIVE, _CANCELLED):
                        self._change(row, (handle << 2) | phase, (handle << 2) | _RETIRED)

    def drain(self) -> None:
        """On the publication stream after every submitted window completes."""
        self._check_publication_stream()
        if self.device.type == "cuda":
            _drain[(self.rows,)](self.control, num_warps=1)
        else:
            for row, word in enumerate(self.control.tolist()):
                if word:
                    self.control[row] = (word & ~3) | _RETIRED

    def acknowledge(self, completed_control: Sequence[int]) -> None:
        """Consume a completed asynchronous control copy, never a timeout."""
        if len(completed_control) != self.rows:
            raise ValueError("run membership acknowledgement has the wrong row count")
        for row, word in enumerate(completed_control):
            handle = word >> 2
            if handle == self._row_handles[row] and word & 3 == _RETIRED:
                self._retired.add(handle)
                self._observed[row] = word
