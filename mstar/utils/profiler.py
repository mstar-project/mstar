"""Utilities for NVTX range annotations for profiling with nsys."""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch

# Per-phase wall-clock samples (MSTAR_PHASE_TIMING), shared process-wide so the
# engine's threads can record into the same buffer the worker flushes.
PHASE_PERIOD = int(os.environ.get("MSTAR_PHASE_TIMING", "0") or "0")
_PHASE_BUF: dict[str, list[float]] = defaultdict(list)


def phase_record(name: str, dt: float) -> None:
    if PHASE_PERIOD:
        _PHASE_BUF[name].append(dt)


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
