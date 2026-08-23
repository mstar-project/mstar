"""Per-engine-step trace: one JSON record per executed batch.

``--log-stats`` reports per-request aggregates: a request's decode steps
collapse into one ``GraphTiming`` with an ``exec_count``. That is the right
granularity for a request report and the wrong one for building a cost model,
which needs to know what each individual step's shape and duration were.

This module records the missing granularity. Every batch the worker
postprocesses emits one record with:

* the executed shape — real and padded batch/token counts, and whether the
  step ran as a captured graph replay, an eager batched forward, or a
  per-request sequential loop;
* true GPU time from the CUDA event pair, plus the engine's CPU phases;
* the KV context the step ran against.

Two consumers: :mod:`mstar.sim.harvest` turns these into stepdb rows, and the
validation harness compares them against simulated steps one-for-one.

Enabled by ``MSTAR_STEP_LOG=/path/to/steps.jsonl`` (the worker appends its
rank to the filename, so ranks never interleave writes). Off by default and
gated on ``enable_prof``, so the serving path pays nothing.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from typing import Any

SCHEMA_VERSION = 1

_ENV_VAR = "MSTAR_STEP_LOG"


class StepLogWriter:
    """Appends step records to a per-worker JSONL file.

    Buffers and flushes in batches: a decode loop emits a record every few
    milliseconds, and one ``write`` syscall per step would show up in the
    very measurement this is meant to capture.
    """

    def __init__(self, path: str, flush_every: int = 64):
        self.path = path
        self.flush_every = flush_every
        self._buf: list[str] = []
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Truncate once at open so a re-run doesn't append to a stale trace.
        with open(self.path, "w"):
            pass
        # Workers have no shutdown hook around their main loop, so the tail
        # of the buffer would otherwise be lost on every run. atexit covers
        # normal exit and SIGINT-driven teardown; a SIGKILL still loses the
        # tail, which is why readers tolerate a truncated final line.
        atexit.register(self.close)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        with open(self.path, "a") as fh:
            fh.write("\n".join(self._buf) + "\n")
        self._buf.clear()

    def close(self) -> None:
        self.flush()


def writer_for_worker(worker_id: str) -> StepLogWriter | None:
    """Build a writer if ``MSTAR_STEP_LOG`` is set, else None.

    The env var names a base path; each worker writes
    ``<base>.<worker_id>.jsonl`` so ranks don't contend.
    """
    base = os.environ.get(_ENV_VAR)
    if not base:
        return None
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".jsonl"
    return StepLogWriter(f"{root}.{worker_id}{ext}")


def make_record(
    *,
    worker_id: str,
    node: str,
    graph_walk: str,
    request_ids: list[str],
    timings,
    model: str = "",
    tp_size: int = 1,
    sp_size: int = 1,
    requires_cfg: bool = False,
    wall_start: float | None = None,
    wall_end: float | None = None,
) -> dict[str, Any]:
    """Render one step record from a batch's :class:`ExecTimings`."""
    return {
        "schema_version": SCHEMA_VERSION,
        "worker": worker_id,
        "model": model,
        "node": node,
        "graph_walk": graph_walk,
        "bs": len(request_ids),
        "real_bs": timings.real_bs,
        "real_num_tokens": timings.real_num_tokens,
        "padded_bs": timings.padded_bs,
        "padded_num_tokens": timings.padded_num_tokens,
        "mode": timings.mode,
        "kv_len_total": timings.kv_len_total,
        "num_sub_batches": timings.num_sub_batches,
        "tp_size": tp_size,
        "sp_size": sp_size,
        "requires_cfg": requires_cfg,
        # Seconds. gpu_s is the CUDA-event measurement; the rest are CPU.
        "gpu_s": timings.gpu_time,
        "prepare_s": timings.prepare_s,
        "plan_s": timings.plan_s,
        "launch_s": timings.launch_s,
        "sample_s": timings.sample_s,
        "total_s": (
            None if (timings.start is None or wall_end is None)
            else wall_end - timings.start
        ),
        "t_start": timings.start,
        "t_end": wall_end,
    }


def read_step_log(path: str) -> list[dict[str, Any]]:
    """Read one step-log file, skipping blank and truncated lines."""
    out: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
