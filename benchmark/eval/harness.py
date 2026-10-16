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
from benchmark.eval.suites.base import Suite, exception_row

SAVE_AUDIO = ("none", "last", "all")
CLIENT_KEEPALIVE_S = 2.0


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
    error_log: Path | None = None,
) -> list[RepeatResult]:
    url = url.rstrip("/")
    # Uncapped open loop must not queue behind a client connection pool either.
    uncapped = spec.mode == "open" and spec.concurrency <= 0

    def new_session() -> aiohttp.ClientSession:
        # A fresh pool per pass: scoring between repeats outlasts the server's keep-alive
        # (uvicorn closes idle connections after 5 s), and a reused dead connection fails
        # the first wave of the next repeat. The client also drops idle ones first.
        connector = aiohttp.TCPConnector(limit=0 if uncapped else max(100, spec.concurrency + 10),
                                         keepalive_timeout=CLIENT_KEEPALIVE_S)
        return aiohttp.ClientSession(connector=connector, read_bufsize=2**20)

    session: aiohttp.ClientSession | None = None
    wall_start = 0.0
    planned = poisson_offsets(len(items), spec.rate, spec.seed) if spec.mode == "open" else None
    timed = False

    async def one(index: int, item: Item, t0: float) -> dict[str, Any]:
        # t0 is the arrival (the latency clock); now is when the request is actually sent,
        # later than t0 only while an open-loop cap holds it back.
        sent = time.perf_counter()
        try:
            row = await suite.request(session, url, index, item, t0)
        except Exception as exc:  # noqa: BLE001 - a suite bug must not hide the other rows
            row = exception_row(item, exc)
        row["start_s"] = t0 - wall_start
        if planned is not None and timed:
            # Open loop is only open while the client keeps the schedule.
            row["arrival_lateness_s"] = row["start_s"] - planned[index]
            row["slot_wait_s"] = sent - t0
        return row

    if warmup_requests:
        log(f"[{suite.name}] warmup: {warmup_requests} request(s) at {spec.describe()}")
        async with new_session() as session:
            await warmup(items, one, spec, warmup_requests)
    out: list[RepeatResult] = []
    timed = True
    for rep in range(repeats):
        async with new_session() as session:
            wall_start = time.perf_counter()
            rows, wall = await run_load(items, one, spec)
        keep = save_audio == "all" or (save_audio == "last" and rep == repeats - 1)
        rep_dir = audio_dir / f"r{rep}" if (audio_dir is not None and keep) else None
        metrics = suite.score(items, rows, wall, rep_dir)
        if planned is not None:
            metrics["arrival_lateness_s_p99"] = stats.percentile([r["arrival_lateness_s"] for r in rows], 99)
            metrics["slot_wait_s_p99"] = stats.percentile([r["slot_wait_s"] for r in rows], 99)
        out.append(RepeatResult(rep, wall, metrics, [public_row(r) for r in rows]))
        failed = [r for r in rows if not r["ok"]]
        if failed and error_log is not None:
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with error_log.open("a", encoding="utf-8") as f:
                for r in failed:
                    f.write(f"repeat {rep} {r['id']} {r['error']}: {r.get('_detail') or ''}\n")
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
    error_log: Path | None = None,
) -> RunResult:
    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    reps = asyncio.run(measure(suite, url, items, spec, repeats, warmup_requests, audio_dir, save_audio, log,
                               error_log))
    cfg = {
        **config,
        "load": {"mode": spec.mode, "concurrency": spec.concurrency, "rate": spec.rate, "seed": spec.seed},
        "repeats": repeats,
        "warmup": warmup_requests,
        "suite": suite.config(),
    }
    return RunResult(suite.name, label, cfg, meta, reps).finalize()
