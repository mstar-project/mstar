#!/usr/bin/env python3
"""PSNR between images (PNG / JPEG / WebP).

    psnr.py ref.png other.png [more.png ...]
    psnr.py --dirs REF_DIR OTHER_DIR --ref-pattern 'sdpa_{:03d}.png' --pattern 'mstar_{:03d}.png' \\
        --start 0 --count 100 [--threshold 40] [--json out.json]

Prices a serving-path deviation against a bit-exact reference: e.g. the compiled + CUDA-graph
path's output against the SDPA/eager path's image for the same prompt, seed and size. The
directory mode compares index-aligned pairs and prints the distribution (min / p5 / median /
max, and which indices fall below the threshold), which is what a "PSNR > 40 dB" claim needs
across a prompt set rather than on one image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def load(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def compare_dirs(ref_dir: Path, other_dir: Path, ref_pattern: str, pattern: str,
                 start: int, count: int) -> tuple[dict[int, float], list[int]]:
    """``{index: psnr}`` for every index whose two files exist, plus the indices with a file missing."""
    values: dict[int, float] = {}
    missing: list[int] = []
    for i in range(start, start + count):
        ref, other = ref_dir / ref_pattern.format(i), other_dir / pattern.format(i)
        if not (ref.exists() and other.exists()):
            missing.append(i)
            continue
        values[i] = psnr(load(ref), load(other))
    return values, missing


def summarize(values: dict[int, float], threshold: float) -> dict:
    """Distribution of a PSNR set: exact (inf) count, min / p5 / median / max, indices below ``threshold``."""
    if not values:
        return {"n": 0}
    arr = np.array(sorted(values.values()), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "n": int(arr.size),
        "exact": int(arr.size - finite.size),
        "min": float(arr[0]),
        "p5": float(np.percentile(arr, 5, method="lower")),
        "median": float(np.median(arr)),
        "max": float(arr[-1]),
        "threshold": threshold,
        "below": sorted(i for i, v in values.items() if v < threshold),
    }


def _fmt(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", nargs="?")
    ap.add_argument("others", nargs="*")
    ap.add_argument("--dirs", nargs=2, metavar=("REF_DIR", "OTHER_DIR"))
    ap.add_argument("--ref-pattern", default="sdpa_{:03d}.png")
    ap.add_argument("--pattern", default="mstar_{:03d}.png")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=40.0)
    ap.add_argument("--json", default="", help="write per-index values and the summary here")
    args = ap.parse_args()
    if args.dirs:
        values, missing = compare_dirs(Path(args.dirs[0]), Path(args.dirs[1]), args.ref_pattern, args.pattern,
                                       args.start, args.count)
        for i, v in values.items():
            print(f"{i:03d} {_fmt(v)} dB")
        summary = summarize(values, args.threshold)
        if missing:
            print(f"missing pairs: {missing}")
        if summary["n"]:
            print(f"n={summary['n']} exact={summary['exact']} min={_fmt(summary['min'])} p5={_fmt(summary['p5'])} "
                  f"median={_fmt(summary['median'])} max={_fmt(summary['max'])} "
                  f"below {args.threshold:g} dB: {len(summary['below'])} {summary['below']}")
        if args.json:
            payload = {"values": {str(i): (None if math.isinf(v) else v) for i, v in values.items()},
                       "missing": missing, "summary": summary,
                       "dirs": list(args.dirs), "patterns": [args.ref_pattern, args.pattern]}
            Path(args.json).write_text(json.dumps(payload, indent=1))
        return
    if not args.reference or not args.others:
        ap.error("give ref.png other.png [...] or --dirs REF_DIR OTHER_DIR")
    ref = load(args.reference)
    for other in args.others:
        value = psnr(ref, load(other))
        print(f"{Path(other).name:40s} PSNR vs {Path(args.reference).name}: {_fmt(value)} dB")


if __name__ == "__main__":
    main()
