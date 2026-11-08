"""Per-phase timing of the worker's threads, and the NVTX range that goes with
each phase. ``benchmark/worker_phases`` reads what this writes out.

Set ``MSTAR_PHASE_TIMING`` to a positive period (in passes) to turn it on; see
``docs/environment_variables.rst``.
"""
import gc
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

import torch

from mstar.utils.profiler import PHASE_PERIOD, phase_buffer, range_pop, range_push

logger = logging.getLogger(__name__)

_GC_PHASES = ("gc.gen0", "gc.gen1", "gc.gen2")


def _thread_role() -> str:
    """The worker thread this runs on, by the executors' name prefixes."""
    name = threading.current_thread().name
    if name.startswith("mstar-gpu-"):
        return "gpu"
    if name.startswith("mstar-plan-"):
        return "plan"
    return "main" if threading.current_thread() is threading.main_thread() else "other"


@cache  # (node, walk) pairs are few
def step_key(node_name: str, graph_walk: str) -> str:
    """``step.<node>.<walk>``, as a phase name (``benchmark/worker_phases``
    reads names of word characters and dots)."""
    return "step." + re.sub(r"\W", "_", node_name) + "." + re.sub(r"\W", "_", graph_walk)


class PhaseTimer:
    """Phase samples from the main, GPU and plan threads, written out once per
    ``period`` passes of the main loop.

    ``period`` and ``buf`` are read directly on hot paths, behind ``if
    timer.period``, so that an untimed step does not pay for a call.
    """

    def __init__(self, worker_id: str = "", enable_nvtx: bool = False, period: int = PHASE_PERIOD):
        self.worker_id = worker_id
        self.enable_nvtx = enable_nvtx
        self.period = period
        self.buf = phase_buffer()
        # (start, end, step key) CUDA events per step, drained once they land
        self._gpu_spans: deque = deque(maxlen=64)
        # a per-thread start stamp: collections on two threads can interleave
        self._gc_start = threading.local()
        self._passes = 0
        # Rows in the step each pass submitted. Reported beside the timings
        # because it separates ramp-up, steady state and drain: a phase mean
        # averaged across those is not a measurement of anything.
        # ``benchmark/worker_phases`` segments on it.
        self._rows: list[int] = []

    @contextmanager
    def span(self, name: str, cpu: str | None = None) -> Iterator[None]:
        """One NVTX range plus one phase sample, same name.

        ``cpu`` names a second sample: this thread's CPU time over the span.
        Wall minus CPU is time spent blocked -- on an event, a lock or the
        GIL -- which a wall-clock span alone cannot tell from work.

        Safe off the main thread: the append is the only shared mutation and
        ``flush`` snapshots before it clears.
        """
        nvtx = self.enable_nvtx
        if nvtx:
            range_push(name, synchronize=False)
        timed = self.period
        t0 = time.perf_counter() if timed else 0.0
        c0 = time.thread_time() if timed and cpu else 0.0
        try:
            yield
        finally:
            if timed:
                self.buf[name].append(time.perf_counter() - t0)
                if cpu:
                    self.buf[cpu].append(time.thread_time() - c0)
            if nvtx:
                range_pop(synchronize=False)

    def record(self, name: str, dt: float) -> None:
        """A sample timed elsewhere."""
        if self.period > 0:
            self.buf[name].append(dt)

    def record_gpu_span(self, start: torch.Event, end: torch.Event, key: str) -> None:
        """A step's device time, as two timing events recorded around it.
        Drained here rather than at ``flush``: the deque holds fewer steps than
        a period usually runs."""
        self._gpu_spans.append((start, end, key))
        while self._gpu_spans:
            start, end, key = self._gpu_spans[0]
            # reading a pair still in flight would block the host on the GPU
            if not end.query():
                return
            self._gpu_spans.popleft()
            # elapsed_time is ms; the phase buffer holds seconds
            elapsed = start.elapsed_time(end) / 1000
            # drained while another step runs: not that step's sample
            self.buf.unkeyed("gpu_exec").append(elapsed)
            self.buf.unkeyed(f"{key}.gpu_exec").append(elapsed)

    def track_gc(self) -> None:
        """Time collections from now on, once warmup is out of the way."""
        if self.period:
            gc.callbacks.append(self._gc_callback)

    def _gc_callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_start.t = time.perf_counter()
            return
        t0 = getattr(self._gc_start, "t", None)
        if t0 is not None:
            dt = time.perf_counter() - t0
            name = _GC_PHASES[info["generation"]]
            self.buf[name].append(dt)
            # and by thread: a collection is charged to whatever span allocated
            # when it triggered, so this says whose spans carry it
            # (gc.gen0.main / .gpu / .plan)
            self.buf[f"{name}.{_thread_role()}"].append(dt)

    def pass_done(self, rows: int, started: float, started_cpu: float) -> None:
        """A pass of the main loop submitted a step of ``rows`` rows. Only
        called with timing on."""
        self.record("iter_total", time.perf_counter() - started)
        self.record("iter_total.cpu", time.thread_time() - started_cpu)
        self._rows.append(rows)
        self._passes += 1
        if self.period > 0 and self._passes % self.period == 0:
            self.flush()

    def flush(self) -> None:
        """Log the window's samples and start a new one."""
        # list() first: the GPU and plan threads append to this while we read
        # it, and iterating the live dict would raise on resize.
        samples = sorted((k, v) for k, v in list(self.buf.items()) if v)
        parts = []
        for name, vs in samples:
            vs = sorted(vs)
            n = len(vs)
            # a trailing '#' marks a count, not a duration
            scale, unit = (1, "") if name.endswith("#") else (1000, "ms")
            p50 = vs[n // 2] * scale
            p95 = vs[min(n - 1, int(n * 0.95))] * scale
            mean = (sum(vs) / n) * scale
            parts.append(
                f"{name}: p50={p50:.3f}{unit} p95={p95:.3f}{unit} "
                f"mean={mean:.3f}{unit} n={n}"
            )
        bs = (sum(self._rows) / len(self._rows)) if self._rows else 0.0
        logger.info(
            "Worker %s phase-timing iter=%d bs=%.2f: %s",
            self.worker_id, self._passes, bs, " | ".join(parts),
        )
        self.buf.clear()
        self._rows.clear()
