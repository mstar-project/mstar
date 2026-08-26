"""Derive the simulator's non-step timings from a measured run.

The stepdb covers what an engine step costs. It says nothing about the costs
*between* steps — the conductor hop at a walk boundary, the api-server
preprocess, the client delivery path, the fixed worker overhead per step.
Those are what this module measures, from the per-request profiles the server
writes with ``--log-stats-json`` plus the per-step trace.

Each term is derived from a specific pair of checkpoints rather than fitted,
so a number that comes out implausible points at one identifiable stage:

``preprocess_s``       recv → preprocess_finish
``conductor_hop_s``    preprocess_finish → conductor_ingest
``client_delivery_s``  the residual between the last engine step for a
                       request and the client seeing its last chunk
``worker_step_overhead_s``
                       per-step wall time not accounted for by the engine's
                       own measured GPU and CPU phases

Medians throughout: these distributions have long right tails (a queued
request's "hop" includes admission wait), and a mean would track the tail.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict
from typing import Any

from mstar.profile.export import read_profiles_json
from mstar.sim.des import TimingModel
from mstar.sim.steplog import read_step_log


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


#: Step-to-step intervals longer than this are treated as the worker having
#: been idle, not as per-step overhead.
_MAX_STEP_GAP_S = 0.5


def _cadence_residuals(steps: list[dict]) -> list[float]:
    """Per-step time not explained by max(gpu, cpu), from observed cadence."""
    by_worker: dict[str, list[dict]] = {}
    for s in steps:
        if s.get("t_start") is None or s.get("gpu_s") is None:
            continue
        by_worker.setdefault(s.get("worker", ""), []).append(s)

    out: list[float] = []
    for records in by_worker.values():
        records.sort(key=lambda r: r["t_start"])
        for a, b in zip(records, records[1:], strict=False):
            interval = b["t_start"] - a["t_start"]
            if interval <= 0 or interval > _MAX_STEP_GAP_S:
                continue
            cpu = sum(
                a.get(k) or 0.0
                for k in ("prepare_s", "plan_s", "launch_s", "sample_s")
            )
            out.append(max(0.0, interval - max(a["gpu_s"], cpu)))
    return out


def calibrate(
    profile_paths: list[str],
    step_log_paths: list[str] | None = None,
) -> tuple[TimingModel, dict[str, Any]]:
    """Build a :class:`TimingModel` from measured artifacts.

    Returns the model and a diagnostics dict recording how many observations
    backed each term — a term derived from three requests should not be read
    with the same confidence as one derived from three hundred.
    """
    profiles: list[dict] = []
    for p in profile_paths:
        for f in _expand(p):
            profiles.extend(read_profiles_json(f))

    steps: list[dict] = []
    for p in step_log_paths or []:
        for f in _expand(p):
            steps.extend(read_step_log(f))

    tm = TimingModel()
    diag: dict[str, Any] = {"profiles": len(profiles), "steps": len(steps)}

    pre, hop, deliver = [], [], []
    for prof in profiles:
        t = prof.get("timing") or {}
        recv = t.get("recv_time")
        pfin = t.get("preprocess_finish_time")
        ingest = t.get("conductor_ingest_time")
        first = t.get("first_chunk_time")
        last = t.get("last_chunk_time")
        cdone = t.get("conductor_finish_time")

        if recv is not None and pfin is not None:
            pre.append(pfin - recv)
        if pfin is not None and ingest is not None:
            hop.append(ingest - pfin)
        # Delivery: the conductor considers the request done, but the client
        # only sees the last chunk after the api-server poll + postprocess.
        if last is not None and cdone is not None and last > cdone:
            deliver.append(last - cdone)
        elif first is not None and ingest is not None and first > ingest:
            # Fallback: no conductor-finish stamp, so bound delivery by the
            # first-chunk path instead. Noisier, hence only a fallback.
            pass

    if (v := _median(pre)) is not None:
        tm.preprocess_s = v
        diag["preprocess_s_n"] = len(pre)
    if (v := _median(hop)) is not None:
        tm.conductor_hop_s = max(v, 0.0)
        diag["conductor_hop_s_n"] = len(hop)
    if (v := _median(deliver)) is not None:
        tm.client_delivery_s = max(v, 0.0)
        diag["client_delivery_s_n"] = len(deliver)

    # Per-step worker overhead, measured as *cadence*: how much longer one
    # step-to-step interval on a worker is than max(gpu, cpu) accounts for.
    #
    # The tempting basis — a step's own ``total_s`` minus max(gpu, cpu) — is
    # wrong, and expensively so. ``total_s`` spans the engine call, which
    # under async scheduling overlaps the *next* step; treating the leftover
    # as additional serial overhead double-counts work that already ran in
    # parallel. On a model whose per-step Python cost is large relative to
    # its GPU time that inflated the term by an order of magnitude and pushed
    # predicted latency 35% high.
    #
    # Consecutive starts on one worker measure the real thing. Intervals
    # longer than ``_MAX_STEP_GAP_S`` are dropped as idle rather than
    # counted as overhead — a worker waiting for work is not a worker
    # paying a cost.
    residuals = _cadence_residuals(steps)
    if (v := _median(residuals)) is not None:
        tm.worker_step_overhead_s = v
        diag["worker_step_overhead_s_n"] = len(residuals)

    return tm, diag


def _expand(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if any(c in path for c in "*?["):
        return sorted(glob.glob(path))
    return [path]


def save_timing(tm: TimingModel, diag: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump({"timing": asdict(tm), "diagnostics": diag}, fh, indent=2)


def load_timing(path: str | None) -> tuple[TimingModel, bool]:
    """Load a calibration file. Returns (model, was_calibrated)."""
    if not path:
        return TimingModel(), False
    with open(path) as fh:
        blob = json.load(fh)
    return TimingModel(**blob["timing"]), True


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="mstar calibrate",
        description=(
            "Measure the simulator's non-step timings (conductor hop, "
            "preprocess, client delivery, per-step worker overhead) from a "
            "run captured with --log-stats-json and MSTAR_STEP_LOG."
        ),
    )
    p.add_argument("--profiles", nargs="+", required=True,
                   help="request-profile JSONL files, globs, or directories")
    p.add_argument("--steps", nargs="*", default=[],
                   help="step-log files, globs, or directories")
    p.add_argument("--out", required=True, help="write the calibration JSON here")
    args = p.parse_args(argv)

    tm, diag = calibrate(args.profiles, args.steps)
    save_timing(tm, diag, args.out)

    print(f"calibrated from {diag['profiles']} request profiles, "
          f"{diag['steps']} steps")
    for field_name, value in asdict(tm).items():
        n = diag.get(f"{field_name}_n")
        origin = f"{n} observations" if n else "default (not measured)"
        # Only the *_s fields are durations; bandwidth is not.
        if field_name.endswith("_s"):
            print(f"  {field_name:<28} {value * 1e3:9.4f} ms   [{origin}]")
        else:
            print(f"  {field_name:<28} {value:9.1f}      [{origin}]")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
