"""Utilities for NVTX range annotations for profiling with nsys."""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter, thread_time
from typing import Iterator

import torch

# Per-phase wall-clock samples (MSTAR_PHASE_TIMING), shared process-wide so the
# engine's threads can record into the same buffer the worker flushes.
PHASE_PERIOD = int(os.environ.get("MSTAR_PHASE_TIMING", "0") or "0")
_PHASE_BUF: dict[str, list[float]] = defaultdict(list)


def nvtx_enabled(
    requested: bool, device: torch.device | None = None,
) -> bool:
    """Enable NVTX only for a CUDA deployment or an explicitly CUDA device."""
    if not requested:
        return False
    if device is None:
        device = torch.accelerator.current_accelerator(check_available=True)
    return device is not None and device.type == "cuda"


def phase_record(name: str, dt: float) -> None:
    if PHASE_PERIOD:
        _PHASE_BUF[name].append(dt)


def phase_start() -> tuple[float, float] | None:
    """Start stamp for ``phase_end``: None unless MSTAR_PHASE_TIMING is set."""
    return (perf_counter(), thread_time()) if PHASE_PERIOD else None


def phase_end(name: str, start: tuple[float, float] | None) -> None:
    """Record wall time as ``name`` and this thread's CPU time as
    ``name.cpu``. Wall minus CPU is time the thread spent blocked -- on the
    GIL, an event or a lock -- which wall time alone cannot tell from work.
    Three threads share the GIL, so that gap is often large."""
    if start is not None:
        _PHASE_BUF[name].append(perf_counter() - start[0])
        _PHASE_BUF[name + ".cpu"].append(thread_time() - start[1])


class PhaseLaps:
    """Consecutive stage timings: each ``lap(stage)`` records the wall and CPU
    time since the previous lap (or construction) as ``prefix + stage`` and
    ``prefix + stage + ".cpu"``. A no-op unless MSTAR_PHASE_TIMING is set;
    ``skip()`` restarts the clock without recording, for a stretch timed
    elsewhere."""

    __slots__ = ("_prefix", "_t", "_c")

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix
        self.skip()

    def lap(self, stage: str) -> None:
        if PHASE_PERIOD:
            now, cpu = perf_counter(), thread_time()
            name = self._prefix + stage
            _PHASE_BUF[name].append(now - self._t)
            _PHASE_BUF[name + ".cpu"].append(cpu - self._c)
            self._t, self._c = now, cpu

    def skip(self) -> None:
        if PHASE_PERIOD:
            self._t, self._c = perf_counter(), thread_time()


def phase_buffer() -> dict[str, list[float]]:
    """The shared sample buffer; ``Worker.run`` reads and clears it."""
    return _PHASE_BUF


def _sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def range_push(name: str, *, synchronize: bool = False) -> None:
    """Push an NVTX range, optionally syncing before the marker.

    Default is ``synchronize=False`` so adding NVTX markers doesn't
    serialize the execution. Set ``synchronize=True`` only when the
    caller specifically wants the range to extend over the GPU work it
    wraps (e.g. an ad-hoc benchmark of one kernel) — and remember that
    each ``synchronize=True`` call drains the *entire* default stream
    via ``torch.cuda.synchronize()``, not just the wrapped kernel.
    """
    if synchronize:
        _sync_if_available()

    torch.cuda.nvtx.range_push(name)


def range_pop(*, synchronize: bool = False) -> None:
    """Pop the current NVTX range, optionally syncing before the marker.

    Same semantics as ``range_push`` — default is ``synchronize=False``.
    """
    if synchronize:
        _sync_if_available()

    torch.cuda.nvtx.range_pop()


def mark(name: str) -> None:
    """Emit an instant NVTX marker without CUDA synchronization."""
    torch.cuda.nvtx.mark(name)


@contextmanager
def nvtx_range(name: str, *, synchronize: bool = False) -> Iterator[None]:
    """Convenience context manager for `range_push`/`range_pop`."""
    range_push(name, synchronize=synchronize)
    try:
        yield
    finally:
        range_pop(synchronize=synchronize)
