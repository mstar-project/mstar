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
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager

# One control word names both the descriptor generation and its publication phase.
_OFFERED, _ACTIVE, _CANCELLED, _RETIRED = range(4)


@triton.jit
def _change(Control, row, expected, replacement):
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
        self._stager = PinnedStager(torch.long, numel=capacity * binding_width)
        self._free = list(range(capacity - 1, -1, -1))
        self._generation = [0] * capacity
        self._descriptors: dict[int, RunRow] = {}
        self._readers: dict[int, int] = {}
        self._row_handles = [0] * rows
        self._observed = [0] * rows
        self._retired: set[int] = set()
        self._published: set[int] = set()

    @classmethod
    def build(cls, spec: RunRowsSpec, info: EngineResourceInfo) -> RunRows:
        return cls(info.device, spec.rows, spec.capacity, spec.binding_width)

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
        self._stager.copy_(self.bindings[slot], descriptor.bindings)
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
        if (ids.shape != (self.rows,) or bindings.shape != (self.rows, self.binding_width)
                or ids.dtype != torch.long or bindings.dtype != torch.long
                or ids.device != self.device or bindings.device != self.device
                or not ids.is_contiguous() or not bindings.is_contiguous()):
            raise ValueError("run boundary buffers do not match its membership layout")
        if self.device.type == "cuda":
            _boundary[(self.rows,)](
                self.control, self.bindings, self.copied, ids, bindings,
                self.capacity, self.binding_width, num_warps=1,
            )
            return
        for row, word in enumerate(self.control.tolist()):
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
        if (ids.shape != (self.rows,) or stops.shape != (self.rows,)
                or ids.dtype != torch.long or stops.dtype != torch.bool
                or ids.device != self.device or stops.device != self.device
                or not ids.is_contiguous() or not stops.is_contiguous()):
            raise ValueError("run stop buffers do not match its membership layout")
        if self.device.type == "cuda":
            _stop[(self.rows,)](self.control, ids, stops, num_warps=1)
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
