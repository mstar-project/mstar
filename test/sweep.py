#!/usr/bin/env python3
"""Batch-size sweep runner.

Calls benchmark.py repeatedly across a set of batch sizes, accumulates all
results, and writes one combined JSON.  Also prints a compact comparison table.

Usage:
  # Qwen3-Omni A2T sweep vs vLLM (run from repo root)
  python test/sweep.py \\
      --system mminf --model qwen3_omni --task A2T \\
      --audio test/qwen3-omni/audio.wav \\
      --batch-sizes 1 2 4 8 16 \\
      --num-requests 32 \\
      --output results/qwen3_omni_A2T_sweep.json

  # BAGEL I2T sweep
  python test/sweep.py \\
      --system mminf --model bagel --task I2T \\
      --image test/bagel/bagel.png \\
      --batch-sizes 1 2 4 8 16 \\
      --output results/bagel_I2T_sweep.json

The output JSON has the structure:
  {
    "system": "mminf",
    "model": "bagel",
    "task": "I2T",
    "sweep": [
      { "batch_size": 1, "summary": {...}, "per_request": [...] },
      { "batch_size": 4, "summary": {...}, "per_request": [...] },
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import (
    MODEL_TASKS,
    TASK_SPEC,
    _default_url,
    compute_summary,
    parse_args as _bench_parse_args,
    print_summary,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_sweep_table(sweep_results: list[dict], model: str, task: str) -> None:
    header = (
        f"{'BS':>4}  {'TTFT_mean':>10}  {'TTFT_p95':>9}  "
        f"{'E2E_mean':>9}  {'ITL_mean':>9}  "
        f"{'Tput(req/s)':>11}  {'Tput(tok/s)':>11}  {'Succ':>6}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(f"  Sweep: {model.upper()} — {task}")
    print(sep)
    print(f"  {header}")
    print(f"  {sep}")

    def _f(v: float, d: int = 3) -> str:
        return f"{v:.{d}f}" if v == v else "  N/A  "

    for row in sweep_results:
        s = row["summary"]
        bs = s["batch_size"]
        t = s["ttft"]
        e = s["e2e"]
        i = s["itl"]
        print(
            f"  {bs:>4}  "
            f"{_f(t['mean']):>10}  {_f(t['p95']):>9}  "
            f"{_f(e['mean']):>9}  {_f(i['mean']):>9}  "
            f"{_f(s['throughput_req_s'], 2):>11}  "
            f"{_f(s['throughput_tok_s'], 1):>11}  "
            f"{s['n_success']:>3}/{s['n_total']:<3}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Namespace bridge
# ---------------------------------------------------------------------------

def _make_bench_args(
    sweep_args: argparse.Namespace,
    batch_size: int,
) -> argparse.Namespace:
    """Build a benchmark.Namespace from sweep args + one batch size."""
    import argparse as _ap

    ns = _ap.Namespace()
    ns.system = sweep_args.system
    ns.model = sweep_args.model
    ns.task = sweep_args.task
    ns.url = sweep_args.url
    ns.timeout = sweep_args.timeout
    ns.num_requests = sweep_args.num_requests
    ns.batch_size = batch_size
    ns.wave_delay = sweep_args.wave_delay
    ns.prompt = sweep_args.prompt
    ns.image = sweep_args.image
    ns.audio = sweep_args.audio
    ns.video = sweep_args.video
    ns.model_kwargs = sweep_args.model_kwargs
    ns.output = None    # we collect ourselves
    ns.quiet = True     # suppress per-run prints; we print the table
    return ns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-size sweep over benchmark.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--system", default="mminf")
    p.add_argument("--model", required=True, choices=list(MODEL_TASKS))
    p.add_argument("--task", required=True,
                   choices=list(TASK_SPEC) + ["mixture"])
    p.add_argument("--url", default=None)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--batch-sizes", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16],
                   metavar="N", help="Batch sizes to sweep (default: 1 2 4 8 16)")
    p.add_argument("--num-requests", type=int, default=32,
                   help="Total requests per batch-size run (default: 32)")
    p.add_argument("--wave-delay", type=float, default=0.0)
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", default=None, metavar="PATH")
    p.add_argument("--audio", default=None, metavar="PATH")
    p.add_argument("--video", default=None, metavar="PATH")
    p.add_argument("--model-kwargs", default=None, metavar="JSON")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Write combined JSON to this file")

    args = p.parse_args()
    if args.url is None:
        args.url = _default_url()

    # Validate media requirements
    if args.task != "mixture":
        in_mods, _ = TASK_SPEC[args.task]
        if "image" in in_mods and not args.image:
            p.error(f"Task '{args.task}' requires --image")
        if "audio" in in_mods and not args.audio:
            p.error(f"Task '{args.task}' requires --audio")
        if "video" in in_mods and not args.video:
            p.error(f"Task '{args.task}' requires --video")

    return args


async def _run_sweep(args: argparse.Namespace) -> list[dict]:
    sweep_results = []
    for bs in args.batch_sizes:
        print(f"\n[sweep] batch_size={bs}  ({args.num_requests} requests total) ...", flush=True)
        bench_ns = _make_bench_args(args, batch_size=bs)
        result = await run_benchmark(bench_ns)
        sweep_results.append(result)
        # Show a one-line summary immediately
        s = result["summary"]
        t = s["ttft"]
        print(
            f"  Done — TTFT_mean={t['mean']:.3f}s  "
            f"E2E_mean={s['e2e']['mean']:.3f}s  "
            f"ITL_mean={s['itl']['mean']:.4f}s  "
            f"tput={s['throughput_tok_s']:.1f}tok/s  "
            f"{s['n_success']}/{s['n_total']} ok",
            flush=True,
        )
    return sweep_results


def main() -> None:
    args = parse_args()

    sweep_results = asyncio.run(_run_sweep(args))

    print_sweep_table(sweep_results, args.model, args.task)

    output_doc = {
        "system": args.system,
        "model": args.model,
        "task": args.task,
        "batch_sizes_swept": args.batch_sizes,
        "num_requests_per_bs": args.num_requests,
        "sweep": sweep_results,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output_doc, indent=2))
        print(f"\nSweep results saved to {args.output}")


if __name__ == "__main__":
    main()
