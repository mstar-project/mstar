#!/usr/bin/env python3
"""Side-by-side comparison of robotics benchmark results.

Reads ``results.json`` files written by:
  - our HTTP harness (``benchmark/runner.py`` with ``--output-dir`` set)
  - the HF transformers baseline (``benchmark/hf_vjepa2_ac.py``)
  - the openpi baseline (``benchmark/openpi_pi05.py``)

All three writers emit a common JCT/throughput schema (``jct_mean_ms``,
``jct_p95_ms``, ``request_throughput``, ...), so this script just loads,
normalizes, and prints a table.

Usage
-----
    # Two-system pi0.5 comparison (our system vs openpi)
    python benchmark/compare_robotics.py \
        --label-a "mstar"  --json-a results/mstar_pi05/results.json \
        --label-b "openpi" --json-b results/openpi_pi05/results.json

    # Two-system vjepa2-ac comparison (our system vs HF)
    python benchmark/compare_robotics.py \
        --label-a "mstar" --json-a results/mstar_vjepa2/results.json \
        --label-b "hf"    --json-b results/hf_vjepa2/results.json

    # Three-way (e.g. our system vs HF vs openpi for a paper figure)
    python benchmark/compare_robotics.py \
        --label-a "mstar"  --json-a results/mstar/results.json \
        --label-b "hf"     --json-b results/hf/results.json \
        --label-c "openpi" --json-c results/openpi/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METRIC_ORDER = [
    ("JCT mean (ms)",     "jct_mean_ms",        "{:.1f}"),
    ("JCT median (ms)",   "jct_median_ms",      "{:.1f}"),
    ("JCT p90 (ms)",      "jct_p90_ms",         "{:.1f}"),
    ("JCT p95 (ms)",      "jct_p95_ms",         "{:.1f}"),
    ("JCT p99 (ms)",      "jct_p99_ms",         "{:.1f}"),
    ("Throughput (req/s)", "request_throughput", "{:.2f}"),
    ("Actions/sec",       "actions_per_sec",    "{:.2f}"),
    ("Rollout-steps/sec", "rollout_steps_per_sec", "{:.2f}"),
    ("Completed",         "completed",          "{}"),
    ("Failed",            "failed",             "{}"),
]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _delta(base: float, comp: float) -> str:
    """Return percent change of comp relative to base, or '' when not meaningful."""
    if not base or base == 0:
        return ""
    pct = (comp - base) / base * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _row(metric_label: str, fmt: str, values: list[float | int | None]) -> str:
    cells: list[str] = []
    for v in values:
        if v is None or v == "":
            cells.append("—")
        elif isinstance(v, str):
            cells.append(v)
        else:
            try:
                cells.append(fmt.format(v))
            except Exception:
                cells.append(str(v))
    return cells


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side robotics benchmark comparison")
    p.add_argument("--label-a", required=True, help="Label for the first system (e.g. 'mstar')")
    p.add_argument("--json-a", required=True, help="Path to results.json for system A")
    p.add_argument("--label-b", required=True, help="Label for the second system")
    p.add_argument("--json-b", required=True, help="Path to results.json for system B")
    p.add_argument("--label-c", default=None, help="Optional third system label")
    p.add_argument("--json-c", default=None, help="Optional third system results.json")
    p.add_argument(
        "--baseline",
        default=None,
        help="Which label to use as baseline for delta columns (default: --label-a).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    systems: list[tuple[str, dict]] = [
        (args.label_a, _load(args.json_a)),
        (args.label_b, _load(args.json_b)),
    ]
    if args.label_c and args.json_c:
        systems.append((args.label_c, _load(args.json_c)))

    baseline = args.baseline or args.label_a
    base_idx = next((i for i, (lbl, _) in enumerate(systems) if lbl == baseline), 0)

    # Header summary
    print("=" * 80)
    print("Robotics benchmark comparison")
    print("=" * 80)
    for lbl, data in systems:
        sysname = data.get("system", "?")
        model = data.get("model", "?")
        n_ok = data.get("completed", 0)
        n_total = data.get("num_requests", 0)
        warmup = data.get("num_warmup", 0)
        print(f"  {lbl:10s} : system={sysname:18s} model={model:12s} "
              f"completed={n_ok}/{n_total} warmup={warmup}")
    print(f"  baseline   : {systems[base_idx][0]} (delta columns measure relative to this)")
    print()

    # Build rows
    labels = [lbl for lbl, _ in systems]
    delta_labels = [f"Δ{lbl}" for i, lbl in enumerate(labels) if i != base_idx]
    cols = ["Metric"] + labels + delta_labels

    rows: list[list[str]] = []
    for label, key, fmt in METRIC_ORDER:
        raw = [data.get(key, None) for _, data in systems]
        # Skip the row if every system is missing or zero for this metric
        # (e.g. actions_per_sec is only meaningful for pi0.5 results).
        if all(v is None or v == 0 or v == 0.0 for v in raw):
            continue

        cells = _row(label, fmt, raw)
        # Delta cells (relative to baseline)
        base_val = raw[base_idx]
        deltas: list[str] = []
        for i, v in enumerate(raw):
            if i == base_idx:
                continue
            if base_val is None or v is None:
                deltas.append("—")
            else:
                deltas.append(_delta(float(base_val), float(v)))
        rows.append([label] + cells + deltas)

    # Pretty-print
    col_widths = [max(len(str(c)) for c in [col] + [r[i] for r in rows])
                  for i, col in enumerate(cols)]
    header = " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in col_widths)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(c.ljust(col_widths[i]) for i, c in enumerate(r)))
    print()


if __name__ == "__main__":
    main()
