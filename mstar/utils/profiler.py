"""Utilities for NVTX range annotations for profiling with nsys, and the
per-kernel step trace window (``StepKernelTrace``)."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch

# Per-phase wall-clock samples (MSTAR_PHASE_TIMING), shared process-wide so the
# engine's threads can record into the same buffer the worker flushes.
PHASE_PERIOD = int(os.environ.get("MSTAR_PHASE_TIMING", "0") or "0")
_PHASE_BUF: dict[str, list[float]] = defaultdict(list)

logger = logging.getLogger(__name__)


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


class StepKernelTrace:
    """A torch.profiler window over a run of GPU-thread executes.

    ``MSTAR_PROFILE_STEPS="<first>:<count>"`` opens a CPU+CUDA profiler before
    execute ``first`` (0-based, per worker process) and closes it after execute
    ``first + count - 1``. CUPTI records the kernels a replayed CUDA graph
    launches individually, so a captured step decomposes into kernels and gaps
    — what the phase timers cannot see. Output per worker in
    ``MSTAR_PROFILE_DIR`` (default ``$TMPDIR`` or ``/tmp``): ``step-trace-
    <worker>.json`` (chrome trace; see ``env/kernel_trace_summary.py``) and
    ``.txt`` (``key_averages``). ``MSTAR_PROFILE_NSYS=1`` brackets the window
    with cudaProfilerStart/Stop instead, for ``nsys -c cudaProfilerApi``.
    Cost when unset: one integer compare per execute. Never quote tok/s from
    a run that had this on.
    """

    def __init__(self, worker_id: str, device: torch.device | None = None) -> None:
        spec = os.environ.get("MSTAR_PROFILE_STEPS", "")
        self.enabled = bool(spec)
        self._n = 0
        self._prof = None
        if not self.enabled:
            return
        first, _, count = spec.partition(":")
        self.first = int(first)
        self.count = max(1, int(count or "1"))
        self.nsys = os.environ.get("MSTAR_PROFILE_NSYS", "0") == "1"
        self.out_dir = (
            os.environ.get("MSTAR_PROFILE_DIR") or os.environ.get("TMPDIR") or "/tmp"
        )
        self.worker_id = worker_id
        if not self.nsys and device is not None and device.type == "cuda":
            # Initialise CUPTI now, on this worker's GPU, so opening the real
            # window mid-benchmark costs milliseconds rather than the ~1 s
            # first-time setup landing in one request's ITL.
            from torch.profiler import ProfilerActivity, profile

            with torch.cuda.device(device):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
                    torch.ones(1, device=device).add_(1)
                    torch.cuda.synchronize(device)
            logger.info("StepKernelTrace: armed for executes %d..%d (CUPTI warm)",
                        self.first, self.first + self.count - 1)

    def before_execute(self) -> None:
        if not self.enabled or self._n != self.first:
            return
        if self.nsys:
            torch.cuda.cudart().cudaProfilerStart()
        else:
            from torch.profiler import ProfilerActivity, profile

            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            )
            self._prof.__enter__()
        logger.info(
            "StepKernelTrace: window open at execute %d for %d execute(s)",
            self._n, self.count,
        )

    def after_execute(self) -> None:
        if not self.enabled:
            return
        n = self._n
        self._n += 1
        if n != self.first + self.count - 1:
            return
        # The last step's kernels must land inside the window.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.enabled = False
        if self.nsys:
            torch.cuda.cudart().cudaProfilerStop()
            logger.info("StepKernelTrace: cudaProfilerStop after execute %d", n)
            return
        prof, self._prof = self._prof, None
        prof.__exit__(None, None, None)
        base = os.path.join(self.out_dir, f"step-trace-{self.worker_id}")
        prof.export_chrome_trace(base + ".json")
        with open(base + ".txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=80))
        logger.info("StepKernelTrace: wrote %s.json and .txt", base)
