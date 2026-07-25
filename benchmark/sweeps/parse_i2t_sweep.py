#!/usr/bin/env python3
"""
Parse a block of closed-loop image-to-text benchmark outputs (one config, N
trials) and emit N data rows plus an average row as TSV.

The metric columns mirror benchmark/convenience/parse_i2t_table.py (TTFT / E2E /
ITL-in-ms / text tok/s / mean output tokens). The config columns (max
concurrency, max-token range, warmup, num requests, run) are supplied as flags
since they don't appear in the raw output.

Warmup convention (see run_bagel_i2t.sh): only trial 1 is warmed up, so the
"Num warmup" column is `--num-warmup-first` on Run 1 and 0 on the rest. The
average row leaves warmup blank and sets Run to "avg".

Usage:
    python parse_i2t_sweep.py run1.txt run2.txt ... run5.txt \
        --max-con 4 --max-tok-range 64/256 --num-requests 30 \
        --num-warmup-first 2
    python parse_i2t_sweep.py ... --no-header    # rows only (later blocks)
"""

import argparse
import re
import sys


def find_stat(text, label, stat):
    """Find a stat (e.g. 'mean', 'p95') on the line whose metric label matches.

    Spacing in `label` is collapsed to \\s+ and regex-special chars (parens)
    are escaped. Returns float or None.
    """
    label_pat = r"\s+".join(re.escape(tok) for tok in label.split())
    line_re = re.compile(r"^\s*" + label_pat + r"[^\n]*", re.MULTILINE)
    m = line_re.search(text)
    if not m:
        return None
    line = m.group(0)
    val_re = re.compile(r"\b" + re.escape(stat) + r"\s*=\s*([0-9]*\.?[0-9]+)")
    vm = val_re.search(line)
    return float(vm.group(1)) if vm else None


def find_throughput_tok(text):
    """Throughput: 75.32 text tok/s"""
    m = re.search(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*text\s*tok/s", text)
    return float(m.group(1)) if m else None


def find_mean_output_tokens(text):
    """Text tokens: 2456 total (122.8 avg/req) -> 122.8"""
    m = re.search(r"Text tokens:\s*[0-9]+\s*total\s*\(([0-9]*\.?[0-9]+)\s*avg/req\)", text)
    return float(m.group(1)) if m else None


# The 14 metric columns, in order. ITL values are seconds in the raw output and
# converted to ms here so downstream formatting is uniform.
def parse_metrics(text):
    itl = [find_stat(text, "ITL (text)", s) for s in ("mean", "p50", "p95", "p99")]
    return [
        find_stat(text, "TTFT (text)", "mean"),
        find_stat(text, "TTFT (text)", "p50"),
        find_stat(text, "TTFT (text)", "p95"),
        find_stat(text, "TTFT (text)", "p99"),
        find_stat(text, "E2E", "mean"),
        find_stat(text, "E2E", "p50"),
        find_stat(text, "E2E", "p95"),
        find_stat(text, "E2E", "p99"),
        *[(v * 1000.0 if v is not None else None) for v in itl],
        find_throughput_tok(text),
        find_mean_output_tokens(text),
    ]


def fmt(v):
    return "" if v is None else f"{v:g}"


def mean(values):
    xs = [v for v in values if v is not None]
    return (sum(xs) / len(xs)) if xs else None


HEADER = [
    "Max con.", "max tok range", "Num warmup", "Num requests", "Run",
    "TTFT (mean)", "TTFT (p50)", "TTFT (p95)", "TTFT (p99)",
    "E2E (mean)", "E2E (p50)", "E2E (p95)", "E2E (p99)",
    "ITL (avg, ms)", "ITL (p50, ms)", "ITL (p95, ms)", "ITL (p99, ms)",
    "Tpt (text tok/s)", "num tok",
]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="Raw benchmark outputs, in trial (Run) order")
    p.add_argument("--max-con", required=True)
    p.add_argument("--max-tok-range", required=True)
    p.add_argument("--num-requests", required=True)
    p.add_argument("--num-warmup-first", required=True, help="Warmup used on Run 1 (0 on the rest)")
    p.add_argument("--no-header", action="store_true")
    args = p.parse_args(argv[1:])

    per_trial = []
    for path in args.files:
        with open(path, "r", encoding="utf-8") as f:
            per_trial.append(parse_metrics(f.read()))

    rows = []
    for i, metrics in enumerate(per_trial):
        run = i + 1
        warmup = args.num_warmup_first if run == 1 else "0"
        cfg = [args.max_con, args.max_tok_range, warmup, args.num_requests, str(run)]
        rows.append(cfg + [fmt(v) for v in metrics])

    # Average row: column-wise mean across trials.
    avg_metrics = [mean([t[c] for t in per_trial]) for c in range(len(HEADER) - 5)]
    avg_cfg = [args.max_con, args.max_tok_range, "", args.num_requests, "avg"]
    rows.append(avg_cfg + [fmt(v) for v in avg_metrics])

    if not args.no_header:
        print("\t".join(HEADER))
    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main(sys.argv)
