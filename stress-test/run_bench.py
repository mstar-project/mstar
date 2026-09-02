#!/usr/bin/env python
"""Runs benchmark/runner.py's own Benchmark end to end and dumps the
per-request fields the analysis needs.

runner.py's --output-dir results.json carries only jct_* aggregates; the ITL
metric is defined on `chunk_arrivals` (per-request median inter-chunk gap,
then the cell median), which that payload drops. So this calls the same
`parse_args()` / `Benchmark.run()` path with the same argv and serialises the
RequestMetrics objects it returns. Nothing in either checkout is modified.

--harness-image-size WxH injects `width`/`height` into the BAGEL T2I model
kwargs. benchmark/request.py hardcodes `"size": "1024x1024"` in the
/v1/images/generations payload and BagelAdapter.image_to_request ignores
`size` outright, so 512x512 is only reachable through the `width`/`height`
passthrough (bagel_model.py overridable_keys). The patch is applied here, in
the harness, identically for both commits.

Usage: run_bench.py --harness-out <path.json> [--harness-tag T]
                    [--harness-image-size 512x512] <normal runner.py args>
"""
import argparse
import asyncio
import json
import statistics
import sys
import time

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--harness-out", required=True)
parser.add_argument("--harness-tag", default="")
parser.add_argument("--harness-image-size", default=None)
known, rest = parser.parse_known_args()
sys.argv = [sys.argv[0]] + rest

from benchmark.base import Bagel, RequestType  # noqa: E402
from benchmark.runner import Benchmark, parse_args  # noqa: E402

if known.harness_image_size:
    _W, _H = (int(v) for v in known.harness_image_size.lower().split("x"))
    _orig_gmk = Bagel.get_model_kwargs

    def _gmk(self, request_type):
        kw = _orig_gmk(self, request_type)
        if request_type in (RequestType.T2I, RequestType.I2I):
            kw = {**kw, "width": _W, "height": _H}
        return kw

    Bagel.get_model_kwargs = _gmk


def _stats(d):
    if d is None:
        return None
    return {"mean": d.mean, "p50": d.p50, "p95": getattr(d, "p95", None), "p99": getattr(d, "p99", None)}


async def main():
    config = parse_args()
    bench = Benchmark(config)
    t0 = time.time()
    metrics, agg = await bench.run()
    wall = time.time() - t0

    per = []
    for m in metrics:
        gaps = m.chunk_gaps
        per.append({
            "request_id": str(m.request_id),
            "status": m.status.value if hasattr(m.status, "value") else str(m.status),
            "error": m.error,
            "e2e_latency": m.e2e_latency,
            "ttft": dict(m.ttft),
            "chunk_arrivals": {k: list(v) for k, v in m.chunk_arrivals.items()},
            # per-request ITL = median of that request's inter-chunk gaps
            "itl_median": {k: statistics.median(v) for k, v in gaps.items() if v},
            "itl_gaps": {k: list(v) for k, v in gaps.items()},
            "n_gaps": {k: len(v) for k, v in gaps.items()},
            "response_chunks": dict(m.response_chunks),
            "output_bytes": dict(m.output_bytes),
            "output_text_tokens": m.output_text_tokens,
        })

    ok = [p for p in per if p["error"] is None and p["e2e_latency"] is not None]
    payload = {
        "tag": known.harness_tag,
        "argv": rest,
        "image_size": known.harness_image_size,
        "wall_time_s": wall,
        "n_requests": len(per),
        "n_success": len(ok),
        "n_failed": len(per) - len(ok),
        "agg": {
            "request_throughput": agg.request_throughput,
            "text_token_throughput": agg.text_token_throughput,
            "total_text_tokens": agg.total_text_tokens,
            "wall_time": agg.wall_time,
            "profiling_type": agg.profiling_type,
            "max_concurrency": agg.max_concurrency,
            "ttft": {k: _stats(v) for k, v in agg.ttft.items()},
            "itl_pooled": {k: _stats(v) for k, v in agg.itl.items()},
            "e2e_latency": _stats(agg.e2e_latency),
        },
        "per_request": per,
    }
    with open(known.harness_out, "w") as f:
        json.dump(payload, f)
    print(f"\n[harness] wrote {known.harness_out}: {len(ok)}/{len(per)} ok, wall={wall:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
