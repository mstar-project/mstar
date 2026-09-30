"""Copy-only host-to-device staging, for work that runs beside a live CUDA graph.

Pre-planning builds step N+1's small index and parameter tensors on the plan
stream while step N's graph replays on the default stream. Anything it puts on
the GPU besides H2D or D2H copies, which also runs on the SMs, executes
concurrently with the graph and slows down the in-graph gap between kernels.
H2D copies (the copy engine) and event waits leave it alone.

So values that come from host bookkeeping go host -> device in ONE copy, with
any padding written on the host as part of the same buffer rather than by a
``fill_`` on the device, and never through a device-side staging tensor.
``PinnedStager`` owns the pinned memory that takes, reused across steps.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import torch


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
        self._depth = depth
        self._numel = 0
        self._bufs: list[torch.Tensor] = []
        self._views: list[np.ndarray] = []
        self._events: list[torch.cuda.Event | None] = [None] * depth
        self._next = 0
        self._lock = threading.Lock()
        self._pinned = torch.cuda.is_available()
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
    ) -> None:
        """``dst[:len(values)] = values`` in one non-blocking H2D copy.

        With ``pad_value``, the rest of ``dst`` is set to it by the same copy
        (the padding is written on the host), so ``dst`` is fully overwritten
        without a device-side fill. ``dst`` must be contiguous.
        """
        n = len(values)
        total = dst.numel() if pad_value is not None else n
        if n > total:
            raise ValueError(f"{n} values do not fit a destination of {total}")
        if total == 0:
            return
        if dst.device.type != "cuda":
            host = np.asarray(values)
            dst.view(-1)[:n].copy_(torch.as_tensor(host, dtype=self.dtype))
            if pad_value is not None:
                dst.view(-1)[n:].fill_(pad_value)
            return
        with self._lock:
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
                ev = self._events[i] = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(dst.device))
