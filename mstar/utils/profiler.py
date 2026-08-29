"""Utilities for NVTX range annotations for profiling with nsys, and the
per-kernel step trace window (``StepKernelTrace``)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


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

    ``MSTAR_PROFILE_STEPS="<first>:<count>"`` opens a CPU+CUDA profiler right
    before execute number ``first`` (0-based, counted per worker process) and
    closes it after execute ``first + count - 1``. CUPTI records the kernels a
    replayed CUDA graph launches *individually*, so a captured decode step
    decomposes into its kernels and the gaps between them — the thing the
    phase timers (``MSTAR_PHASE_TIMING``, the glm52 step timer) cannot see.

    Output per worker, in ``MSTAR_PROFILE_DIR`` (default ``$TMPDIR`` or
    ``/tmp``): ``step-trace-<worker>.json`` (chrome trace; feed it to
    ``env/kernel_trace_summary.py`` or chrome://tracing) and
    ``step-trace-<worker>.txt`` (``key_averages`` by CUDA time).
    ``MSTAR_PROFILE_NSYS=1`` brackets the same window with
    ``cudaProfilerStart/Stop`` instead, for
    ``nsys profile -c cudaProfilerApi --cuda-graph-trace=node``.

    Cost when unset: one integer compare per execute. Never quote tok/s from
    a run that had this on — CUPTI adds microseconds per kernel and the
    window ends with a device synchronize.
    """

    def __init__(self, worker_id: str) -> None:
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
