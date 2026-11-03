"""Host memory a worker copies request state into, so a request can leave the device and come back."""

from __future__ import annotations

import bisect
from typing import NamedTuple

import torch

from mstar.utils.device import DeviceCapability

# Slices start on a multiple of the widest dtype a copy takes, so any of them can view a slice.
_ALIGN = 8


class HostSlice(NamedTuple):
    offset: int
    nbytes: int


class HostCopy(NamedTuple):
    """A device tensor copied into the pool: its slice, a view of the slice, and the event that records the
    copy, None when it copied in place."""
    slice: HostSlice
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
        size = -(-nbytes // _ALIGN) * _ALIGN
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

    def copy_out(
        self, src: torch.Tensor, after: torch.Event | None, dtype: torch.dtype | None = None,
    ) -> HostCopy | None:
        """Copy ``src``, in ``dtype`` when given, into a slice once ``after`` (the step that wrote it) completes.
        None when no free slice holds it."""
        dtype = src.dtype if dtype is None else dtype
        piece = self.take(src.numel() * dtype.itemsize)
        if piece is None:
            return None
        view = self._memory[piece.offset:piece.offset + src.numel() * dtype.itemsize].view(dtype).view(src.shape)
        if after is None:
            view.copy_(src)
            return HostCopy(piece, view, None)
        if self._stream is None:
            self._stream = torch.Stream(device=self.device)
        self._stream.wait_event(after)
        with self._stream:
            view.copy_(src.to(dtype), non_blocking=True)
        # the step's inputs may be freed before the copy reads them
        src.record_stream(self._stream)
        landed = torch.Event(device=self.device)
        landed.record(self._stream)
        return HostCopy(piece, view, landed)
