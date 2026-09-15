"""Env-gated ``torch.profiler`` windows over engine steps (``MSTAR_TORCH_PROFILE``).

``MSTAR_TORCH_PROFILE=<start>:<count>[,<start>:<count>...]`` makes the worker profile
``count`` consecutive steps of ``MSTAR_TORCH_PROFILE_WALK`` (default ``decode``) from the
``start``-th such step (absolute, 0-based) for each window, then log one per-kernel table per
window (CUDA time and launches per step, the batch sizes seen) and write a Chrome trace to
``MSTAR_TORCH_PROFILE_DIR`` (default ``/tmp``). ``MSTAR_TORCH_PROFILE_RANKS`` (default ``0``,
comma-separated worker indices, ``all``) selects the workers that profile. Kernels launched by
CUDA-graph replays are recorded individually, so the tables show the real served step. A window
costs one device synchronize when it closes; unset, the hook is a counter increment per step.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


class StepProfiler:
    def __init__(self, windows: list[tuple[int, int]], graph_walk: str, tag: str, out_dir: str):
        self.windows = sorted(windows)  # (start step, count), absolute step indices of ``graph_walk``
        self.graph_walk, self.tag, self.out_dir = graph_walk, tag, out_dir
        self._seen = 0
        self._prof: torch.profiler.profile | None = None
        self._profiled = 0
        self._batch_sizes: list[int] = []
        self._window = 0

    @property
    def done(self) -> bool:
        return self._window >= len(self.windows)

    @classmethod
    def from_env(cls, worker_id: str, env: dict | None = None) -> StepProfiler | None:
        env = os.environ if env is None else env
        spec = env.get("MSTAR_TORCH_PROFILE", "")
        if not spec:
            return None
        ranks = env.get("MSTAR_TORCH_PROFILE_RANKS", "0")
        idx = worker_id.rsplit("_", 1)[-1]
        if ranks != "all" and idx not in {r.strip() for r in ranks.split(",")}:
            return None
        windows = []
        for w in spec.split(","):
            start, count = w.split(":")[:2]
            windows.append((int(start), int(count)))
        walk = env.get("MSTAR_TORCH_PROFILE_WALK", "decode")
        return cls(windows, walk, worker_id, env.get("MSTAR_TORCH_PROFILE_DIR", "/tmp"))

    @contextmanager
    def step(self, graph_walk: str | None, batch_size: int | None = None):
        """Wrap one engine step of ``graph_walk``; opens the next window at its start step and
        closes it ``count`` steps later."""
        if self.done or graph_walk != self.graph_walk:
            yield
            return
        start, count = self.windows[self._window]
        if self._prof is None and self._seen >= start:
            acts = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                acts.append(torch.profiler.ProfilerActivity.CUDA)
            self._prof = torch.profiler.profile(activities=acts)
            self._prof.__enter__()
            self._profiled, self._batch_sizes = 0, []
            logger.info("%s: profiling %d %s steps from step %d", self.tag, count, self.graph_walk, self._seen)
        self._seen += 1
        try:
            yield
        finally:
            if self._prof is not None:
                self._profiled += 1
                if batch_size is not None:
                    self._batch_sizes.append(batch_size)
                if self._profiled >= count:
                    self._close()

    def _close(self) -> None:
        prof = self._prof
        self._prof = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # the last replayed step must have finished to be recorded
        prof.__exit__(None, None, None)
        self.report(prof)
        self._window += 1

    def report(self, prof) -> None:
        n = max(self._profiled, 1)
        cuda_us = launches = 0
        for ev in prof.key_averages():
            dev = getattr(ev, "device_time_total", None)
            if dev is None:
                dev = getattr(ev, "cuda_time_total", 0)
            if getattr(ev, "device_type", None) is not None and str(ev.device_type).endswith("CUDA"):
                cuda_us += dev
                launches += ev.count
        table = prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                                          row_limit=45)
        bs = f"batch sizes {min(self._batch_sizes)}..{max(self._batch_sizes)}" if self._batch_sizes else "batch sizes unknown"
        logger.info("%s: window %d: %d %s steps profiled (%s): %.3f ms of CUDA time and %.1f kernel launches per step\n%s",
                    self.tag, self._window + 1, n, self.graph_walk, bs, cuda_us / n / 1000.0, launches / n, table)
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, f"mstar_{self.tag}_{self.graph_walk}_w{self._window + 1}_trace.json")
            prof.export_chrome_trace(path)
            logger.info("%s: trace written to %s", self.tag, path)
        except Exception as ex:  # the table is the deliverable; the trace is best effort
            logger.warning("%s: could not write the trace: %s", self.tag, ex)
