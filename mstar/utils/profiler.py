"""Utilities for NVTX range annotations for profiling with nsys."""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter, thread_time, time
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


class _HopStamps:
    """Per-request clock stamps along the path to a request's first token,
    logged by each process as ``hop-stamps wall_offset=<s>: <key> <hop> <t>; ...`` lines that
    ``benchmark.worker_phases.hops`` joins by key. The key is the engine's
    request key (``<request_id>#<generation>``), the one name every process
    has for a request; the clock is ``perf_counter`` (CLOCK_MONOTONIC on
    Linux), which every process on the host shares. A no-op unless
    MSTAR_PHASE_TIMING is set; a stamp is a tuple append, and the lines are
    written every ``_FLUSH_N`` stamps or after ``_FLUSH_S`` seconds."""

    _FLUSH_N = 64
    _FLUSH_S = 1.0

    def __init__(self) -> None:
        self._buf: list[tuple[str, str, float]] = []
        self._lock = threading.Lock()
        self._last_flush = perf_counter()

    def stamp(self, key: str, hop: str, t: float | None = None) -> None:
        if not PHASE_PERIOD:
            return
        now = perf_counter()
        with self._lock:
            self._buf.append((key, hop, now if t is None else t))
            if len(self._buf) < self._FLUSH_N and now - self._last_flush < self._FLUSH_S:
                return
            buf, self._buf, self._last_flush = self._buf, [], now
        # the wall clock minus perf_counter, so a reader can place stamps in a run's time window
        _HOP_LOGGER.info(
            "hop-stamps wall_offset=%.6f: %s", time() - perf_counter(),
            "; ".join(f"{k} {h} {v:.6f}" for k, h, v in buf),
        )


_HOP_LOGGER = logging.getLogger("mstar.hops")
_HOPS = _HopStamps()


def hop_stamp(key: str, hop: str, t: float | None = None) -> None:
    r"""Stamp ``key``'s arrival at ``hop`` (``[\w.]+``); see ``_HopStamps``."""
    _HOPS.stamp(key, hop, t)


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
