#!/usr/bin/env python3
"""Pi0.5 batch benchmark.

Sends a fixed pool of requests across batch sizes [1, 2, 3, 4], enforcing
concurrency via a semaphore. Reports per-request and aggregate latency stats.

Usage::

    # Run with defaults (server on localhost:20002, 24 requests, all batch sizes)
    python benchmark_pi05.py

    # Custom port / pool size / batch sizes
    python benchmark_pi05.py --port 20002 --pool 24 --batch-sizes 1 2 4 8

    # Specify server URL directly
    python benchmark_pi05.py --url http://my-robot-server:20002/infer
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import numpy as np
from _env import get_server_url
from PIL import Image

# ── constants matching Pi0.5 defaults ────────────────────────────────────────
ACTION_HORIZON = 50
ACTION_DIM = 32
IMAGE_SIZE = 224  # model expects 224×224 3-channel RGB

DEFAULT_BATCH_SIZES = [1, 2, 3, 4]
DEFAULT_POOL_SIZE = 24
DEFAULT_TEXT = "pick up the block"


# ── image helpers ─────────────────────────────────────────────────────────────

def _make_random_png(seed: int) -> bytes:
    """Deterministic 224×224 RGB PNG (fast, small, reproducible)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


# Pre-generate a small palette of images so we're not regenerating every call.
_IMAGE_PALETTE: List[bytes] = [_make_random_png(s) for s in range(32)]


def _get_image(idx: int) -> bytes:
    return _IMAGE_PALETTE[idx % len(_IMAGE_PALETTE)]


# ── request builder ───────────────────────────────────────────────────────────

def _build_form_data(request_idx: int, text: str = DEFAULT_TEXT) -> aiohttp.FormData:
    """Build the multipart/form-data body expected by the Pi0.5 inference server."""
    form = aiohttp.FormData()

    # Three camera images: base, left wrist, right wrist
    for cam_idx in range(3):
        seed_offset = request_idx * 3 + cam_idx
        form.add_field(
            "files",
            _get_image(seed_offset),
            filename=f"cam{cam_idx}_{request_idx}.png",
            content_type="image/png",
        )

    form.add_field("text", text)
    form.add_field("input_modalities", "image,text")
    form.add_field("output_modalities", "action")
    form.add_field("streaming", "false")
    form.add_field(
        "model_kwargs",
        json.dumps({"robot_state": [0.0] * ACTION_DIM}),
    )
    return form


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_idx: int
    batch_size: int          # concurrency level this request ran under
    latency_s: float
    success: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    action_shape: Optional[tuple] = None


# ── async worker ─────────────────────────────────────────────────────────────

async def _send_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    request_idx: int,
    batch_size: int,
) -> RequestResult:
    async with semaphore:
        form = _build_form_data(request_idx)
        t0 = time.perf_counter()
        try:
            async with session.post(url, data=form) as resp:
                latency = time.perf_counter() - t0
                body = await resp.read()
                if not resp.ok:
                    return RequestResult(
                        request_idx=request_idx,
                        batch_size=batch_size,
                        latency_s=latency,
                        success=False,
                        status_code=resp.status,
                        error=body[:200].decode(errors="replace"),
                    )
                payload = json.loads(body)
                chunks = payload.get("outputs", {}).get("action", [])
                shape = None
                if chunks:
                    raw = base64.b64decode(chunks[0]["data"])
                    arr = np.frombuffer(raw, dtype=np.float32)
                    n_steps = arr.size // ACTION_DIM
                    shape = (n_steps, ACTION_DIM)
                return RequestResult(
                    request_idx=request_idx,
                    batch_size=batch_size,
                    latency_s=latency,
                    success=True,
                    status_code=resp.status,
                    action_shape=shape,
                )
        except Exception as exc:
            latency = time.perf_counter() - t0
            return RequestResult(
                request_idx=request_idx,
                batch_size=batch_size,
                latency_s=latency,
                success=False,
                error=str(exc),
            )


# ── benchmark runner ──────────────────────────────────────────────────────────

async def run_batch_benchmark(
    url: str,
    pool_size: int,
    batch_sizes: List[int],
    timeout_s: float = 120.0,
) -> dict[int, List[RequestResult]]:
    """
    For each batch_size, fire ``pool_size`` requests with at most
    ``batch_size`` in-flight concurrently and record E2E latencies.
    """
    connector = aiohttp.TCPConnector(limit=max(batch_sizes) * 2)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    all_results: dict[int, List[RequestResult]] = {}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for bs in batch_sizes:
            print(f"\n{'─'*60}")
            print(f"  Batch size = {bs}  ({pool_size} requests, ≤{bs} concurrent)")
            print(f"{'─'*60}")
            semaphore = asyncio.Semaphore(bs)
            tasks = [
                _send_one(session, semaphore, url, idx, bs)
                for idx in range(pool_size)
            ]
            wall_t0 = time.perf_counter()
            results: List[RequestResult] = await asyncio.gather(*tasks)
            wall_elapsed = time.perf_counter() - wall_t0
            all_results[bs] = results

            # ── per-batch-size summary ────────────────────────────────────
            ok = [r for r in results if r.success]
            fail = [r for r in results if not r.success]
            latencies = [r.latency_s for r in ok]

            if latencies:
                print(f"  Requests : {len(results)} total, {len(ok)} ok, {len(fail)} failed")
                print(f"  Wall time: {wall_elapsed:.3f}s  |  "
                      f"Throughput: {len(ok)/wall_elapsed:.2f} req/s")
                print(f"  Latency  : "
                      f"min={min(latencies):.3f}s  "
                      f"p50={statistics.median(latencies):.3f}s  "
                      f"p95={_percentile(latencies, 95):.3f}s  "
                      f"p99={_percentile(latencies, 99):.3f}s  "
                      f"max={max(latencies):.3f}s")
                print(f"  Mean±std : {statistics.mean(latencies):.3f}s ± "
                      f"{statistics.stdev(latencies) if len(latencies) > 1 else 0:.3f}s")
                # Show action shape from first successful result
                for r in ok:
                    if r.action_shape:
                        print(f"  Action   : shape={r.action_shape}")
                        break
            else:
                print("  All requests FAILED.")

            if fail:
                print(f"\n  Sample errors ({min(3, len(fail))} of {len(fail)}):")
                for r in fail[:3]:
                    print(f"    req#{r.request_idx:02d}  "
                          f"status={r.status_code}  err={r.error}")

    return all_results


def _percentile(data: List[float], p: int) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: dict[int, List[RequestResult]], pool_size: int) -> None:
    print(f"\n{'═'*72}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═'*72}")
    header = (
        f"{'Batch':>6}  {'OK':>4}  {'Fail':>4}  {'min':>7}  {'p50':>7}  "
        f"{'p95':>7}  {'p99':>7}  {'max':>7}  {'req/s':>7}"
    )
    print(header)
    print(f"{'─'*72}")
    for bs in sorted(all_results.keys()):
        results = all_results[bs]
        ok = [r for r in results if r.success]
        fail = [r for r in results if not r.success]
        latencies = [r.latency_s for r in ok]
        if latencies:
            # Estimate wall time: sum of latencies / batch_size approximation
            # (better: use per-batch wall time, but we store results only)
            # Use sum/bs as a proxy for throughput
            throughput = len(ok) / (sum(latencies) / bs) if latencies else 0
            print(
                f"{bs:>6}  {len(ok):>4}  {len(fail):>4}  "
                f"{min(latencies):>7.3f}  "
                f"{statistics.median(latencies):>7.3f}  "
                f"{_percentile(latencies, 95):>7.3f}  "
                f"{_percentile(latencies, 99):>7.3f}  "
                f"{max(latencies):>7.3f}  "
                f"{throughput:>7.2f}"
            )
        else:
            print(f"{bs:>6}  {'0':>4}  {len(fail):>4}  {'ALL FAILED':>51}")
    print(f"{'═'*72}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pi0.5 batch benchmark — measures E2E latency at varying concurrency levels."
    )
    parser.add_argument(
        "--pool", type=int, default=DEFAULT_POOL_SIZE,
        help=f"Total requests to send per batch size (default: {DEFAULT_POOL_SIZE}).",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
        help=f"Concurrency levels to test (default: {DEFAULT_BATCH_SIZES}).",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT,
        help=f"Task prompt sent with every request (default: '{DEFAULT_TEXT}').",
    )
    args = parser.parse_args()

    url = get_server_url()
    batch_sizes = sorted(set(args.batch_sizes))

    print("Pi0.5 Batch Benchmark")
    print(f"  Server  : {url}")
    print(f"  Pool    : {args.pool} requests per batch size")
    print(f"  Batches : {batch_sizes}")
    print(f"  Prompt  : {args.text!r}")
    print(f"  Images  : {IMAGE_SIZE}×{IMAGE_SIZE} random RGB PNGs (3 per request)")

    all_results = asyncio.run(
        run_batch_benchmark(url, args.pool, batch_sizes, args.timeout)
    )

    print_summary(all_results, args.pool)


if __name__ == "__main__":
    main()
