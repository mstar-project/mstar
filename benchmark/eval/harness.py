"""Runs a suite at one load point: warmup, then ``repeats`` measured passes
over the items, each scored, into one ``RunResult``."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

from benchmark.eval import stats
from benchmark.eval.datasets import Item
from benchmark.eval.load import LoadSpec, poisson_offsets, run_load, warmup
from benchmark.eval.schema import RepeatResult, RunResult
from benchmark.eval.suites.base import Suite, error_row

SAVE_AUDIO = ("none", "last", "all")


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


async def measure(
    suite: Suite,
    url: str,
    items: list[Item],
    spec: LoadSpec,
    repeats: int,
    warmup_requests: int,
    audio_dir: Path | None = None,
    save_audio: str = "last",
    log=print,
) -> list[RepeatResult]:
    url = url.rstrip("/")
    connector = aiohttp.TCPConnector(limit=max(100, spec.concurrency + 10))
    async with aiohttp.ClientSession(connector=connector, read_bufsize=2**20) as session:
        wall_start = 0.0
        planned = poisson_offsets(len(items), spec.rate, spec.seed) if spec.mode == "open" else None
        timed = False

        async def one(index: int, item: Item, t0: float) -> dict[str, Any]:
            try:
                row = await suite.request(session, url, index, item, t0)
            except Exception as exc:  # noqa: BLE001 - a suite bug must not hide the other rows
                row = error_row(item, f"{type(exc).__name__}: {exc}")
            row["start_s"] = t0 - wall_start
            if planned is not None and timed:
                # Open loop is only open while the client keeps the schedule.
                row["dispatch_lateness_s"] = row["start_s"] - planned[index]
            return row

        if warmup_requests:
            log(f"[{suite.name}] warmup: {warmup_requests} request(s) at {spec.describe()}")
            await warmup(items, one, spec, warmup_requests)
        out: list[RepeatResult] = []
        timed = True
        for rep in range(repeats):
            wall_start = time.perf_counter()
            rows, wall = await run_load(items, one, spec)
            keep = save_audio == "all" or (save_audio == "last" and rep == repeats - 1)
            rep_dir = audio_dir / f"r{rep}" if (audio_dir is not None and keep) else None
            metrics = suite.score(items, rows, wall, rep_dir)
            if planned is not None:
                metrics["dispatch_lateness_s_p99"] = stats.percentile([r["dispatch_lateness_s"] for r in rows], 99)
            out.append(RepeatResult(rep, wall, metrics, [public_row(r) for r in rows]))
            log(f"[{suite.name}] {spec.describe()} repeat {rep}: " + _brief(metrics))
    return out


def _brief(m: dict[str, Any]) -> str:
    keys = [k for k in m if k not in ("requests",)][:7]
    return ", ".join(f"{k}={m[k]:.4g}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in keys)


def run_point(
    suite: Suite,
    url: str,
    items: list[Item],
    spec: LoadSpec,
    repeats: int,
    warmup_requests: int,
    label: str,
    config: dict[str, Any],
    meta: dict[str, Any],
    audio_dir: Path | None = None,
    save_audio: str = "last",
) -> RunResult:
    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    reps = asyncio.run(measure(suite, url, items, spec, repeats, warmup_requests, audio_dir, save_audio, log))
    cfg = {
        **config,
        "load": {"mode": spec.mode, "concurrency": spec.concurrency, "rate": spec.rate, "seed": spec.seed},
        "repeats": repeats,
        "warmup": warmup_requests,
        "suite": suite.config(),
    }
    return RunResult(suite.name, label, cfg, meta, reps).finalize()
