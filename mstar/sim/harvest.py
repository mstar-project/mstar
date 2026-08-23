"""Build a stepdb from per-step traces produced by real serving runs.

The step log (:mod:`mstar.sim.steplog`) records every batch a worker executed,
with its padded shape and CUDA-event GPU time. This module aggregates those
observations into the table the simulator queries.

Why aggregate rather than store raw: one serving run produces thousands of
steps that land on a few dozen distinct (shape, KV) points. Repeated
observations of the same point are the same measurement taken many times, so
they are combined into a robust central estimate.

The estimator is the **median**, not the mean. A step's measured GPU time has
a long right tail — the first replay of a bucket pays cache warmup, an
occasional step collides with another process on the GPU, and CPU-bound
stretches inflate the event window when the GPU went idle mid-step. Those are
all one-sided, so a mean drifts upward with them while a median doesn't.

KV lengths are bucketed before aggregation: decode KV grows by one token per
step, so raw KV values are nearly all distinct and would defeat aggregation
entirely, leaving one noisy sample per row.
"""

from __future__ import annotations

import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from mstar.sim.stepdb import StepDB, StepKey, StepSample
from mstar.sim.steplog import read_step_log

#: KV lengths are grouped into buckets this many tokens wide before
#: aggregation. Attention cost varies smoothly and slowly in KV length, so a
#: few hundred tokens of quantization is far below the noise floor, while the
#: aggregation it enables is what makes the estimate stable.
DEFAULT_KV_BUCKET = 512


def bucket_kv(kv_len: int | None, width: int = DEFAULT_KV_BUCKET) -> int:
    """Round a KV length to the center of its bucket."""
    if not kv_len:
        return 0
    idx = kv_len // width
    return idx * width + width // 2


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


@dataclass
class HarvestReport:
    """What a harvest run produced, for the operator to sanity-check."""

    records_read: int = 0
    records_used: int = 0
    rows_written: int = 0
    skipped_no_gpu_time: int = 0
    skipped_no_shape: int = 0
    skipped_split: int = 0
    #: (node, walk) -> number of usable observations, to spot thin coverage.
    per_node_counts: dict[tuple[str, str], int] | None = None

    def summary(self) -> str:
        lines = [
            f"read {self.records_read} step records, "
            f"used {self.records_used}, wrote {self.rows_written} rows",
        ]
        skipped = (
            self.skipped_no_gpu_time + self.skipped_no_shape + self.skipped_split
        )
        if skipped:
            lines.append(
                f"  skipped {skipped}: "
                f"{self.skipped_no_gpu_time} without GPU time, "
                f"{self.skipped_no_shape} without a recorded shape, "
                f"{self.skipped_split} split across sub-batches"
            )
        for (node, walk), n in sorted((self.per_node_counts or {}).items()):
            lines.append(f"  {node}/{walk}: {n} observations")
        return "\n".join(lines)


def harvest_records(
    records: Iterable[dict[str, Any]],
    db: StepDB,
    model: str,
    kv_bucket: int = DEFAULT_KV_BUCKET,
    min_observations: int = 1,
) -> HarvestReport:
    """Aggregate step records into ``db``.

    ``min_observations`` drops points measured too few times to trust; the
    default keeps everything, since a missing row degrades a prediction more
    visibly (it is flagged) than a slightly noisy one.
    """
    report = HarvestReport(per_node_counts=defaultdict(int))
    grouped: dict[tuple[StepKey, int], list[dict]] = defaultdict(list)

    for rec in records:
        report.records_read += 1
        if rec.get("gpu_s") is None:
            report.skipped_no_gpu_time += 1
            continue
        if rec.get("mode") is None or rec.get("padded_bs") is None:
            report.skipped_no_shape += 1
            continue
        if (rec.get("num_sub_batches") or 1) > 1:
            # A split forward has one GPU window covering several shapes;
            # attributing it to any single shape would be wrong.
            report.skipped_split += 1
            continue

        key = StepKey(
            model=model,
            node=rec["node"],
            graph_walk=rec["graph_walk"],
            padded_bs=rec["padded_bs"],
            padded_num_tokens=rec["padded_num_tokens"],
            tp_size=rec.get("tp_size", 1),
            sp_size=rec.get("sp_size", 1),
            requires_cfg=bool(rec.get("requires_cfg", False)),
            mode=rec["mode"],
        )
        grouped[(key, bucket_kv(rec.get("kv_len_total"), kv_bucket))].append(rec)
        report.records_used += 1
        report.per_node_counts[(rec["node"], rec["graph_walk"])] += 1

    samples = []
    for (key, kv), recs in grouped.items():
        if len(recs) < min_observations:
            continue
        gpu = [r["gpu_s"] for r in recs]
        samples.append(StepSample(
            key=key,
            kv_len_total=kv,
            gpu_s=_median(gpu),
            prepare_s=_median([r.get("prepare_s") or 0.0 for r in recs]),
            plan_s=_median([r.get("plan_s") or 0.0 for r in recs]),
            launch_s=_median([r.get("launch_s") or 0.0 for r in recs]),
            sample_s=_median([r.get("sample_s") or 0.0 for r in recs]),
            gpu_s_stdev=_stdev(gpu),
            repeats=len(recs),
            provenance={"source": "steplog", "kv_bucket": kv_bucket},
        ))

    db.add_many(samples)
    report.rows_written = len(samples)
    report.per_node_counts = dict(report.per_node_counts)
    return report


def harvest_paths(
    paths: Iterable[str],
    db_path: str,
    model: str,
    kv_bucket: int = DEFAULT_KV_BUCKET,
    gpu_name: str | None = None,
) -> HarvestReport:
    """Harvest every step-log file named by ``paths``.

    Each path may be a file or a glob; a directory is read as ``<dir>/*.jsonl``.
    Step logs are written one per worker, so a run is normally a glob.
    """
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
        elif any(ch in p for ch in "*?["):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)

    records: list[dict] = []
    for f in files:
        records.extend(read_step_log(f))

    with StepDB(db_path, gpu_name=gpu_name) as db:
        return harvest_records(records, db, model=model, kv_bucket=kv_bucket)
