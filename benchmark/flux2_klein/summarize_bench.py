#!/usr/bin/env python3
"""Render the benchmark protocol's image table from ``bench_images.py`` result files.

    python benchmark/flux2_klein/summarize_bench.py results/2026-09-18/*.json

Rows are grouped by ``(tag, model)``: the latency file supplies the B=1 median / p95, the
throughput file the images/s at each concurrency and the peak VRAM. Missing pieces print as
``n/a`` rather than being invented.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(paths: list[str | Path]) -> dict[tuple[str, str], dict]:
    """``{(tag, model): {"latency": ..., "throughput": ...}}`` from result JSON files."""
    rows: dict[tuple[str, str], dict] = defaultdict(dict)
    for path in paths:
        data = json.loads(Path(path).read_text())
        rows[(data["tag"], data["model"])][data["mode"]] = data
    return rows


def _fmt_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} s"


def _fmt_rate(entry: dict | None) -> str:
    if entry is None:
        return "n/a"
    spread = ""
    if "images_per_s_min" in entry:
        spread = f" ({entry['images_per_s_min']:.2f}-{entry['images_per_s_max']:.2f})"
    return f"{entry['images_per_s']:.2f}{spread}"


def _fmt_vram(*entries: dict | None) -> str:
    peaks = [e["peak_vram_mib"] for e in entries if e and e.get("peak_vram_mib") is not None]
    return "n/a" if not peaks else f"{max(peaks) / 1024:.1f} GiB"


def render_table(rows: dict[tuple[str, str], dict], concurrencies: tuple[int, ...] = (4, 8, 16)) -> str:
    """The protocol's markdown table for the image row set."""
    head = ["System", "model", "size / steps", "output", "B=1 latency median", "p95"]
    head += [f"images/s @{c}" for c in concurrencies] + ["peak VRAM", "notes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (tag, model), parts in sorted(rows.items()):
        lat, thr = parts.get("latency"), parts.get("throughput")
        by_c = (thr or {}).get("by_concurrency", {})
        any_part = lat or thr or {}
        size = f"{any_part.get('size', 'n/a')} / {any_part.get('steps', 'n/a')}"
        notes = []
        if lat:
            notes.append(f"n={lat['n']}")
        if thr and by_c:
            first = next(iter(by_c.values()))
            notes.append(f"{first.get('images', '?')} images x {first.get('repeats', 1)} repeats per level")
        output = any_part.get("output_format") or "server default"
        cells = [tag, model, size, output, _fmt_s(lat and lat["median_s"]), _fmt_s(lat and lat["p95_s"])]
        cells += [_fmt_rate(by_c.get(str(c))) for c in concurrencies]
        cells += [_fmt_vram(lat, *by_c.values()), "; ".join(notes)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="bench_images.py JSON files")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()
    print(render_table(load_results(args.results), tuple(args.concurrency)))


if __name__ == "__main__":
    main()
