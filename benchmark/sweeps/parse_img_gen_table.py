#!/usr/bin/env python3
"""
Parse closed-loop image-generation benchmark output and emit a single TSV row.

JCT (job completion time) is taken from the E2E latency line; throughput from
the "Throughput: X req/s" line. Config columns (max concurrency, warmup, num
requests, run) are supplied as flags since they don't appear in the raw output.

Usage:
    python parse_img_gen_table.py results.txt --max-con 1 --num-warmup 5 \
        --num-requests 10 --run 1
    some_command | python parse_img_gen_table.py --max-con 1 ...
    python parse_img_gen_table.py results.txt ... --no-header   # row only

Columns (tab-separated):
    Max con.  Num warmup  Num requests  Run
    JCT (mean)  JCT (p50)  JCT (p95)  JCT (p99)   [seconds]
    Tpt (req/s)

cf. benchmark/convenience/parse_i2t_table.py
"""

import argparse
import re
import sys


def find_stat(text, label, stat):
    """
    Find a stat (e.g. 'mean', 'p50') on the line whose metric label matches.

    Spacing in `label` is collapsed to \\s+ and regex-special chars are escaped.
    Returns float or None. Trailing units (e.g. the 's' on '1.234s') are ignored.
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


def find_throughput_req(text):
    """Throughput: 0.50 req/s (successful only)"""
    m = re.search(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*req/s", text)
    return float(m.group(1)) if m else None


def fmt(v):
    """Format a value for output; blank if missing."""
    return "" if v is None else f"{v:g}"


def parse(text):
    return [
        fmt(find_stat(text, "E2E", "mean")),
        fmt(find_stat(text, "E2E", "p50")),
        fmt(find_stat(text, "E2E", "p95")),
        fmt(find_stat(text, "E2E", "p99")),
        fmt(find_throughput_req(text)),
    ]


HEADER = [
    "Max con.", "Num warmup", "Num requests", "Run",
    "JCT (mean)", "JCT (p50)", "JCT (p95)", "JCT (p99)",
    "Tpt (req/s)",
]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", help="Raw benchmark output file (default: stdin)")
    p.add_argument("--max-con", required=True)
    p.add_argument("--num-warmup", required=True)
    p.add_argument("--num-requests", required=True)
    p.add_argument("--run", default="1")
    p.add_argument("--no-header", action="store_true")
    args = p.parse_args(argv[1:])

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    row = [args.max_con, args.num_warmup, args.num_requests, args.run] + parse(text)

    if not args.no_header:
        print("\t".join(HEADER))
    print("\t".join(row))


if __name__ == "__main__":
    main(sys.argv)
