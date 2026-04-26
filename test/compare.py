#!/usr/bin/env python3
"""Compare benchmark JSON results across systems (mminf vs vllm vs sglang).

Accepts multiple --results files produced by benchmark.py or sweep.py,
and prints a side-by-side comparison table.  Optionally saves a matplotlib
figure.

Usage:
  # Single-task comparison
  python test/compare.py \\
      --results results/mminf_qwen3_A2T.json \\
               results/vllm_qwen3_A2T.json \\
               results/sglang_qwen3_A2T.json \\
      --metric ttft e2e itl throughput_tok_s \\
      --plot results/qwen3_A2T_compare.png

  # Batch-sweep throughput comparison (from sweep.py outputs)
  python test/compare.py \\
      --results results/mminf_bagel_I2T_sweep.json \\
               results/vllm_bagel_I2T_sweep.json \\
      --sweep --metric throughput_tok_s throughput_req_s \\
      --plot results/bagel_I2T_tput.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

METRICS = ["ttft", "e2e", "itl", "throughput_req_s", "throughput_tok_s"]
STAT_KEYS = ["mean", "median", "p95", "p99"]   # for latency metrics


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _get_summary(doc: dict) -> dict:
    """Works for both benchmark.py and sweep.py output (uses first sweep entry)."""
    if "sweep" in doc:
        return doc["sweep"][0]["summary"]
    return doc["summary"]


def _get_sweep_summaries(doc: dict) -> list[tuple[int, dict]]:
    """Return [(batch_size, summary), ...] for sweep docs."""
    if "sweep" not in doc:
        s = doc["summary"]
        return [(s["batch_size"], s)]
    return [(entry["summary"]["batch_size"], entry["summary"]) for entry in doc["sweep"]]


def _extract_metric(summary: dict, metric: str, stat: str = "mean") -> float:
    if metric in ("throughput_req_s", "throughput_tok_s"):
        return summary.get(metric, float("nan"))
    # latency sub-dict
    sub = summary.get(metric, {})
    return sub.get(stat, float("nan"))


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _f(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def print_comparison_table(
    docs: list[dict],
    metrics: list[str],
    stat: str = "mean",
) -> None:
    """Single batch-size comparison table."""
    systems = [d.get("system", f"sys{i}") for i, d in enumerate(docs)]
    summaries = [_get_summary(d) for d in docs]

    # Header
    col_w = max(12, max(len(s) for s in systems) + 2)
    metric_labels = {
        "ttft": f"TTFT_{stat}(s)",
        "e2e":  f"E2E_{stat}(s)",
        "itl":  f"ITL_{stat}(s)",
        "throughput_req_s": "Tput(req/s)",
        "throughput_tok_s": "Tput(tok/s)",
    }

    header_parts = [f"{'Metric':<20}"] + [f"{s:>{col_w}}" for s in systems]
    print("\n" + "─" * (20 + col_w * len(systems) + len(systems)))
    print("  " + "  ".join(header_parts))
    print("  " + "─" * (20 + col_w * len(systems) + len(systems)))

    for m in metrics:
        label = metric_labels.get(m, m)
        vals = [_extract_metric(s, m, stat) for s in summaries]
        # Bold/mark the best value
        if m in ("throughput_req_s", "throughput_tok_s"):
            best_idx = max(range(len(vals)), key=lambda i: vals[i] if not math.isnan(vals[i]) else -1)
        else:
            best_idx = min(range(len(vals)), key=lambda i: vals[i] if not math.isnan(vals[i]) else 1e18)

        row_parts = [f"{label:<20}"]
        for idx, v in enumerate(vals):
            s_val = _f(v, 4 if m == "itl" else 3)
            marker = " *" if idx == best_idx and not math.isnan(v) else "  "
            row_parts.append(f"{(s_val + marker):>{col_w}}")
        print("  " + "  ".join(row_parts))

    print("  " + "─" * (20 + col_w * len(systems) + len(systems)))
    print("  (* = best value for that metric)\n")


def print_sweep_comparison_table(
    docs: list[dict],
    metric: str,
    stat: str = "mean",
) -> None:
    """Side-by-side sweep table (rows = batch sizes, cols = systems)."""
    systems = [d.get("system", f"sys{i}") for i, d in enumerate(docs)]
    all_bs = sorted({bs for d in docs for bs, _ in _get_sweep_summaries(d)})

    is_tput = metric in ("throughput_req_s", "throughput_tok_s")
    metric_label = {
        "throughput_req_s": "Tput(req/s)",
        "throughput_tok_s": "Tput(tok/s)",
        "ttft": f"TTFT_{stat}(s)",
        "e2e":  f"E2E_{stat}(s)",
        "itl":  f"ITL_{stat}(s)",
    }.get(metric, metric)

    col_w = max(12, max(len(s) for s in systems) + 2)
    print(f"\n  Sweep comparison — metric: {metric_label}")
    header = f"{'BS':>4}  " + "  ".join(f"{s:>{col_w}}" for s in systems)
    print("  " + "─" * len(header))
    print("  " + header)
    print("  " + "─" * len(header))

    for bs in all_bs:
        vals = []
        for d in docs:
            sweep = dict(_get_sweep_summaries(d))
            s = sweep.get(bs, {})
            vals.append(_extract_metric(s, metric, stat) if s else float("nan"))

        best_idx = (
            max(range(len(vals)), key=lambda i: vals[i] if not math.isnan(vals[i]) else -1)
            if is_tput else
            min(range(len(vals)), key=lambda i: vals[i] if not math.isnan(vals[i]) else 1e18)
        )
        parts = [f"{bs:>4}  "]
        for idx, v in enumerate(vals):
            s_val = _f(v, 2 if is_tput else 3)
            marker = " *" if idx == best_idx and not math.isnan(v) else "  "
            parts.append(f"{(s_val + marker):>{col_w}}")
        print("  " + "  ".join(parts))

    print("  " + "─" * len(header))
    print("  (* = best per row)\n")


# ---------------------------------------------------------------------------
# Matplotlib plots (optional)
# ---------------------------------------------------------------------------

def _try_plot_sweep(
    docs: list[dict],
    metrics: list[str],
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not available — skipping plot", file=sys.stderr)
        return

    systems = [d.get("system", f"sys{i}") for i, d in enumerate(docs)]
    is_tput_metrics = [m for m in metrics if m in ("throughput_req_s", "throughput_tok_s")]
    latency_metrics = [m for m in metrics if m not in ("throughput_req_s", "throughput_tok_s")]

    all_metrics = latency_metrics + is_tput_metrics
    ncols = len(all_metrics)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for ax, metric in zip(axes, all_metrics):
        is_tput = metric in ("throughput_req_s", "throughput_tok_s")
        ylabel = {
            "throughput_req_s": "Throughput (req/s)",
            "throughput_tok_s": "Throughput (tok/s)",
            "ttft": "TTFT mean (s)",
            "e2e":  "E2E mean (s)",
            "itl":  "ITL mean (s)",
        }.get(metric, metric)

        for idx, d in enumerate(docs):
            pairs = _get_sweep_summaries(d)
            xs = [bs for bs, _ in pairs]
            ys = [_extract_metric(s, metric) for _, s in pairs]
            ax.plot(xs, ys, marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)], label=systems[idx])

        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if not is_tput:
            ax.set_ylim(bottom=0)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


def _try_plot_bar(
    docs: list[dict],
    metrics: list[str],
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[compare] matplotlib/numpy not available — skipping plot", file=sys.stderr)
        return

    systems = [d.get("system", f"sys{i}") for i, d in enumerate(docs)]
    n_sys = len(systems)
    n_met = len(metrics)
    if n_met == 0:
        return

    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 4))
    if n_met == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, metric in zip(axes, metrics):
        is_tput = metric in ("throughput_req_s", "throughput_tok_s")
        summaries = [_get_summary(d) for d in docs]
        vals = [_extract_metric(s, metric) for s in summaries]
        label = {
            "throughput_req_s": "Tput (req/s)",
            "throughput_tok_s": "Tput (tok/s)",
            "ttft": "TTFT mean (s)",
            "e2e":  "E2E mean (s)",
            "itl":  "ITL mean (s)",
        }.get(metric, metric)

        xs = np.arange(n_sys)
        bars = ax.bar(xs, vals, color=[colors[i % len(colors)] for i in range(n_sys)])
        ax.set_xticks(xs)
        ax.set_xticklabels(systems, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate bar values
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare mminf benchmark results across systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results", nargs="+", required=True, metavar="PATH",
                   help="JSON result files from benchmark.py or sweep.py")
    p.add_argument("--metric", nargs="+", default=["ttft", "e2e", "itl", "throughput_tok_s"],
                   choices=METRICS, dest="metrics",
                   help="Metrics to compare (default: ttft e2e itl throughput_tok_s)")
    p.add_argument("--stat", default="mean", choices=STAT_KEYS,
                   help="Which statistic to show for latency metrics (default: mean)")
    p.add_argument("--sweep", action="store_true",
                   help="Treat inputs as sweep.py output; show table per batch size")
    p.add_argument("--plot", default=None, metavar="PATH",
                   help="Save comparison plot to this path (requires matplotlib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    docs = [load(p) for p in args.results]

    if args.sweep:
        for m in args.metrics:
            print_sweep_comparison_table(docs, m, args.stat)
        if args.plot:
            _try_plot_sweep(docs, args.metrics, args.plot)
    else:
        print_comparison_table(docs, args.metrics, args.stat)
        if args.plot:
            _try_plot_bar(docs, args.metrics, args.plot)


if __name__ == "__main__":
    main()
