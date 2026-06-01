#!/usr/bin/env python3
"""
Parse offline text-to-image benchmark output and emit a single TSV row.

Usage:
    python img_bench_to_tsv.py < results.txt
    python img_bench_to_tsv.py results.txt
    some_command | python img_bench_to_tsv.py
    python img_bench_to_tsv.py results.txt --no-header   # row only

Columns (tab-separated):
    JCT (mean)  JCT (p50)  JCT (p95)  JCT (p99)
    Throughput (req/s)

JCT (job completion time) is taken from the E2E line.
"""

import re
import sys


def find_stat(text, label, stat):
    """
    Find a stat (e.g. 'mean', 'p50') on the line whose metric label matches.

    Spacing in `label` is collapsed to \\s+ and regex-special chars (parens)
    are escaped, so 'E2E' or 'TTFT (image)' both work. Returns float or None.
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
    """Throughput: 0.06 req/s (successful only)"""
    m = re.search(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*req/s", text)
    return float(m.group(1)) if m else None


def fmt(v):
    """Format a value for output; blank if missing."""
    return "" if v is None else f"{v:g}"


def parse(text):
    # E2E == job completion time (JCT).
    jct_mean = find_stat(text, "E2E", "mean")
    jct_p50 = find_stat(text, "E2E", "p50")
    jct_p95 = find_stat(text, "E2E", "p95")
    jct_p99 = find_stat(text, "E2E", "p99")

    throughput_req = find_throughput_req(text)

    return [
        fmt(jct_mean),
        fmt(jct_p50),
        fmt(jct_p95),
        fmt(jct_p99),
        fmt(throughput_req),
    ]


HEADER = [
    "JCT (mean)", "JCT (p50)", "JCT (p95)", "JCT (p99)",
    "Throughput (req/s)",
]


def main(argv):
    args = [a for a in argv[1:] if a != "--no-header"]
    no_header = "--no-header" in argv[1:]

    if args:
        with open(args[0], "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    row = parse(text)

    if not no_header:
        print("\t".join(HEADER))
    print("\t".join(row))


if __name__ == "__main__":
    main(sys.argv)