"""GPU utilisation during benchmark runs, from ``nvidia-smi dmon -s um`` logs.

``notes``-style stage scripts sample ``nvidia-smi dmon -s um -d 1 -o T``
next to each served run; this summarises every ``dmon_*.log`` under a
results directory: mean and median SM utilisation, memory-controller
utilisation and used memory over the sampled seconds, skipping the idle
head and tail (server start-up and shutdown) below an SM threshold.

Usage::

    python benchmark/chatterbox/gpu_util.py --results results/2026-09-25 [--idle 5]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path


def parse_dmon(path: Path) -> list[tuple[float, float, float]]:
    """(sm %, mem-controller %, used MiB) per sample; header lines skipped."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # "-o T" puts the time first, then gpu index, sm, mem, enc, dec, ..., fb (used MiB) ...
        try:
            sm, mem = float(parts[2]), float(parts[3])
            used = float(parts[-2]) if len(parts) > 8 else float("nan")
        except (ValueError, IndexError):
            continue
        rows.append((sm, mem, used))
    return rows


def trim_idle(rows: list[tuple[float, float, float]], idle: float) -> list[tuple[float, float, float]]:
    """Drop the idle head and tail (SM below ``idle`` %)."""
    lo, hi = 0, len(rows)
    while lo < hi and rows[lo][0] < idle:
        lo += 1
    while hi > lo and rows[hi - 1][0] < idle:
        hi -= 1
    return rows[lo:hi]


def summarize(path: Path, idle: float) -> str:
    rows = parse_dmon(path)
    active = trim_idle(rows, idle)
    if not active:
        return f"{path.stem} | {len(rows)} | - | - | - | -"
    sm = [r[0] for r in active]
    mem = [r[1] for r in active]
    used = [r[2] for r in active if r[2] == r[2]]
    busy = sum(1 for v in sm if v >= idle) / len(sm)
    return (
        f"{path.stem} | {len(active)} | {statistics.fmean(sm):.0f} / {statistics.median(sm):.0f} | "
        f"{statistics.fmean(mem):.0f} | {100 * busy:.0f} % | {statistics.fmean(used) / 1024:.1f}" if used else
        f"{path.stem} | {len(active)} | {statistics.fmean(sm):.0f} / {statistics.median(sm):.0f} | "
        f"{statistics.fmean(mem):.0f} | {100 * busy:.0f} % | -"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--idle", type=float, default=5.0, help="SM %% below which a second counts as idle")
    parser.add_argument("--out")
    args = parser.parse_args()
    lines = ["log | active s | SM % mean / p50 | mem-ctl % | seconds with SM ≥ idle | used GiB",
             "--- | --- | --- | --- | --- | ---"]
    for path in sorted(Path(args.results).glob("dmon_*.log")):
        lines.append(summarize(path, args.idle))
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
