"""Host memory a worker copies request state into, so a request can leave the device and come back."""

from __future__ import annotations

import bisect
from typing import NamedTuple

import torch

from mstar.utils.device import DeviceCapability

# Slices start on a multiple of the widest dtype a copy takes, so any of them can view a slice.
_ALIGN = 8


def aligned(nbytes: int) -> int:
    """``nbytes`` rounded up to where the next slice can start."""
    return -(-nbytes // _ALIGN) * _ALIGN


class HostSlice(NamedTuple):
    offset: int
    nbytes: int


class HostCopy(NamedTuple):
    """A device tensor copied into the pool: a view of where it went, and the event that records the copy, None
    when it copied in place."""
    view: torch.Tensor
    landed: torch.Event | None

    def has_landed(self) -> bool:
        return self.landed is None or self.landed.query()


class HostPool:
    """``nbytes`` of host memory allocated once, pinned where ``device`` takes pinned memory, and handed out
    first fit. Copies run on a stream of the pool's own, so they never wait behind a later step."""

    def __init__(self, nbytes: int, device: torch.device) -> None:
        self.device = device
        self._memory = torch.empty(nbytes, dtype=torch.uint8, pin_memory=DeviceCapability.of(device).pin_memory)
        # sorted by offset, with no two adjacent
        self._free = [HostSlice(0, nbytes)]
        self._stream: torch.Stream | None = None

    def take(self, nbytes: int) -> HostSlice | None:
        size = aligned(nbytes)
        for i, free in enumerate(self._free):
            if free.nbytes >= size:
                if free.nbytes == size:
                    del self._free[i]
                else:
                    self._free[i] = HostSlice(free.offset + size, free.nbytes - size)
                return HostSlice(free.offset, size)
        return None

    def give(self, piece: HostSlice) -> None:
        i = bisect.bisect(self._free, piece)
        start, end = piece.offset, piece.offset + piece.nbytes
        if i < len(self._free) and self._free[i].offset == end:
            end += self._free.pop(i).nbytes
        if i and sum(self._free[i - 1]) == start:
            i -= 1
            start = self._free.pop(i).offset
        self._free.insert(i, HostSlice(start, end - start))

    def view(self, into: HostSlice, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
        """The start of ``into`` as a tensor of ``dtype`` and ``shape``."""
        nbytes = shape.numel() * dtype.itemsize
        assert nbytes <= into.nbytes, f"a copy of {nbytes} bytes doesn't fit the {into.nbytes} bytes it was given"
        return self._memory[into.offset:into.offset + nbytes].view(dtype).view(shape)

    def copy_out(
        self, src: torch.Tensor, into: HostSlice, after: torch.Event | None, dtype: torch.dtype | None = None,
    ) -> HostCopy:
        """Copy ``src``, in ``dtype`` when given, to the start of ``into`` once ``after`` (the step that wrote it)
        completes."""
        dtype = src.dtype if dtype is None else dtype
        view = self.view(into, dtype, src.shape)
        if after is None:
            view.copy_(src)
            return HostCopy(view, None)
        if self._stream is None:
            self._stream = torch.Stream(device=self.device)
        self._stream.wait_event(after)
        with self._stream:
            view.copy_(src.to(dtype), non_blocking=True)
        # the step's inputs may be freed before the copy reads them
        src.record_stream(self._stream)
        landed = torch.Event(device=self.device)
        landed.record(self._stream)
        return HostCopy(view, landed)
