"""Copy-only host-to-device staging beside a live accelerator graph.

Pre-planning builds step N+1's small tensors on the plan stream while step N's
graph replays. Any SM work it issues (a kernel, a ``fill_``, a D2D copy) slows
the graph; H2D copies and event waits do not. So host values go to the device
in ONE copy from pinned memory (``PinnedStager``), padding written on the host,
never via a device-side staging tensor. An ``H2DMirror`` skips re-copying
values a destination already holds, as in a steady decode batch.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import torch


class H2DMirror:
    """What the stager last copied into one device destination.

    Lets ``PinnedStager.copy_`` skip a copy whose values the destination
    already holds. Only sound for a destination nothing but that stager writes,
    and that lives as long as its mirror -- hence an object the owner keeps
    with it, not a cache keyed by address (the allocator reuses addresses).
    """

    __slots__ = ("_key", "_values")

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._values: np.ndarray | None = None

    def matches(self, key: tuple, values: np.ndarray) -> bool:
        return (
            self._key == key and self._values is not None
            and np.array_equal(self._values, values)
        )

    def record(self, key: tuple, values: np.ndarray) -> None:
        self._key = key
        self._values = values.copy()

    def invalidate(self) -> None:
        """The destination was written some other way."""
        self._key = self._values = None


class PinnedStager:
    """A small ring of persistent pinned buffers for one dtype.

    ``copy_`` writes values into the next buffer and issues one non-blocking
    H2D copy from it on the current stream. Each buffer records an event after
    its copy and is not rewritten until that event has completed, so a copy
    still in flight never sees its source change. Thread-safe: the plan thread
    and the GPU thread may share one.
    """

    def __init__(self, dtype: torch.dtype, numel: int = 256, depth: int = 8):
        self.dtype = dtype
        self._np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        self._depth = depth
        self._numel = 0
        self._bufs: list[torch.Tensor] = []
        self._views: list[np.ndarray] = []
        self._events: list[torch.Event | None] = [None] * depth
        self._next = 0
        self._lock = threading.Lock()
        self._pinned = torch.accelerator.is_available()
        self._grow(numel)

    def _grow(self, numel: int) -> None:
        # Nothing may still be copying out of the buffers being replaced.
        for ev in self._events:
            if ev is not None:
                ev.synchronize()
        self._numel = max(numel, 2 * self._numel, 1)
        self._bufs = [
            torch.empty(self._numel, dtype=self.dtype, pin_memory=self._pinned)
            for _ in range(self._depth)
        ]
        self._views = [b.numpy() for b in self._bufs]

    def copy_(
        self,
        dst: torch.Tensor,
        values: Sequence | np.ndarray,
        pad_value=None,
        mirror: H2DMirror | None = None,
    ) -> None:
        """``dst[:len(values)] = values`` in one non-blocking H2D copy.

        With ``pad_value``, the rest of ``dst`` is set to it by the same copy
        (the padding is written on the host), so ``dst`` is fully overwritten
        without a device-side fill. ``dst`` must be contiguous.

        With ``mirror`` (``dst``'s, see ``H2DMirror``), nothing is issued when
        ``dst`` already holds exactly this: the copy that put it there is
        ahead of anything that reads ``dst`` after this call.
        """
        n = len(values)
        total = dst.numel() if pad_value is not None else n
        if n > total:
            raise ValueError(f"{n} values do not fit a destination of {total}")
        if total == 0:
            return
        if mirror is not None:
            values = np.asarray(values, dtype=self._np_dtype)
            key = (total, pad_value)
        if dst.device.type not in {"cuda", "xpu"} or not self._pinned:
            host = np.asarray(values)
            dst.view(-1)[:n].copy_(torch.as_tensor(host, dtype=self.dtype))
            if pad_value is not None:
                dst.view(-1)[n:].fill_(pad_value)
            return
        with self._lock:
            if mirror is not None and mirror.matches(key, values):
                return
            if total > self._numel:
                self._grow(total)
            i = self._next
            self._next = (i + 1) % self._depth
            ev = self._events[i]
            if ev is not None:
                ev.synchronize()  # the previous copy out of this buffer is done
            view = self._views[i]
            view[:n] = values
            if pad_value is not None:
                view[n:total] = pad_value
            dst.view(-1)[:total].copy_(self._bufs[i][:total], non_blocking=True)
            if ev is None:
                ev = self._events[i] = torch.Event(device=dst.device)
            ev.record(torch.accelerator.current_stream(dst.device))
            if mirror is not None:
                mirror.record(key, values)
