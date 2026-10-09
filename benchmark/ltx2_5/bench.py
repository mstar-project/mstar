#!/usr/bin/env python3
"""Closed-loop benchmark of a running LTX-2.5 M* server: ``--concurrency`` requests in
flight, ``--num-requests`` measured after ``--warmup-waves`` waves of warmup.

    python benchmark/ltx2_5/bench.py --port 8123 --concurrency 1 2 4 --out <dir>

Reports per-request latency (mean / p50 / p90) and throughput in videos per minute,
and fails loudly on any failed request.
"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompts import PROMPTS  # noqa: E402

from mstar.client.client import MStarClient  # noqa: E402


def one(port, prompt, seed, args):
    client = MStarClient(f"http://127.0.0.1:{port}", timeout=3600)
    t0 = time.perf_counter()
    res = client.generate(
        text=prompt, output_modalities=("video", "audio"), seed=seed,
        height=args.height, width=args.width, num_frames=args.num_frames, recipe=args.recipe,
    )
    elapsed = time.perf_counter() - t0
    ok = any(c.get("modality") == "video" and c["bytes"] for c in res.raw) and res.audio is not None
    return elapsed, ok


def run_level(args, concurrency):
    n_warm = concurrency * args.warmup_waves
    n = max(args.num_requests, concurrency * args.min_waves)
    jobs = [(PROMPTS[i % len(PROMPTS)], i) for i in range(n_warm + n)]
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(lambda j: one(args.port, *j, args), jobs[:n_warm]))
        t0 = time.perf_counter()
        results = list(pool.map(lambda j: one(args.port, *j, args), jobs[n_warm:]))
        wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in results)
    failed = sum(not r[1] for r in results)
    return {
        "concurrency": concurrency, "completed": len(results) - failed, "failed": failed,
        "mean_s": statistics.mean(lat), "p50_s": lat[len(lat) // 2], "p90_s": lat[int(len(lat) * 0.9) - 1],
        "throughput_videos_per_min": 60.0 * len(results) / wall, "wall_s": wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8123)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num-requests", type=int, default=8)
    p.add_argument("--min-waves", type=int, default=3)
    p.add_argument("--warmup-waves", type=int, default=1)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--recipe", default="single", choices=("single", "two_stage"))
    p.add_argument("--label", default="mstar")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in args.concurrency:
        row = {"system": args.label, **run_level(args, c)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if row["failed"]:
            raise SystemExit(f"{row['failed']} requests failed at concurrency {c}")
    (out / f"{args.label}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
