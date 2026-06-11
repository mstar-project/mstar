#!/usr/bin/env python3
"""
Parse closed-loop image-to-text benchmark output and emit a single TSV row.

Usage:
    python i2t_bench_to_tsv.py < results.txt
    python i2t_bench_to_tsv.py results.txt
    some_command | python i2t_bench_to_tsv.py
    python i2t_bench_to_tsv.py results.txt --no-header   # row only

Columns (tab-separated):
    TTFT (mean) TTFT (p50) TTFT (p95) TTFT (p99)
    E2E (mean)  E2E (p50)  E2E (p95)  E2E (p99)
    ITL (mean, ms) ITL (p50, ms) ITL (p95, ms) ITL (p99, ms)
    Throughput (text tok/s)

TTFT and ITL are taken from the text lines; ITL is converted to ms.
"""

import re
import sys


def find_stat(text, label, stat):
    """
    Find a stat (e.g. 'mean', 'p50') on the line whose metric label matches.

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
    """Text tokens: 2456 total (122.8 avg/req) -> 122.8

    The mean output length per request. With --ignore-eos this should match
    across systems (proving equal decode work); without it, it documents the
    output-length disparity that otherwise confounds the tok/s comparison.
    """
    m = re.search(r"Text tokens:\s*[0-9]+\s*total\s*\(([0-9]*\.?[0-9]+)\s*avg/req\)", text)
    return float(m.group(1)) if m else None


def fmt(v):
    """Format a value for output; blank if missing."""
    return "" if v is None else f"{v:g}"


def to_ms(v_seconds):
    """Convert seconds to milliseconds, trimming trailing zeros."""
    if v_seconds is None:
        return ""
    return f"{v_seconds * 1000:g}"


def parse(text):
    ttft_mean = find_stat(text, "TTFT (text)", "mean")
    ttft_p50 = find_stat(text, "TTFT (text)", "p50")
    ttft_p95 = find_stat(text, "TTFT (text)", "p95")
    ttft_p99 = find_stat(text, "TTFT (text)", "p99")

    e2e_mean = find_stat(text, "E2E", "mean")
    e2e_p50 = find_stat(text, "E2E", "p50")
    e2e_p95 = find_stat(text, "E2E", "p95")
    e2e_p99 = find_stat(text, "E2E", "p99")

    # ITL reported in seconds; convert to ms.
    itl_mean = find_stat(text, "ITL (text)", "mean")
    itl_p50 = find_stat(text, "ITL (text)", "p50")
    itl_p95 = find_stat(text, "ITL (text)", "p95")
    itl_p99 = find_stat(text, "ITL (text)", "p99")

    throughput_tok = find_throughput_tok(text)
    mean_out_tokens = find_mean_output_tokens(text)

    return [
        fmt(ttft_mean), fmt(ttft_p50), fmt(ttft_p95), fmt(ttft_p99),
        fmt(e2e_mean), fmt(e2e_p50), fmt(e2e_p95), fmt(e2e_p99),
        to_ms(itl_mean), to_ms(itl_p50), to_ms(itl_p95), to_ms(itl_p99),
        fmt(throughput_tok),
        fmt(mean_out_tokens),
    ]


HEADER = [
    "TTFT (mean)", "TTFT (p50)", "TTFT (p95)", "TTFT (p99)",
    "E2E (mean)", "E2E (p50)", "E2E (p95)", "E2E (p99)",
    "ITL (mean, ms)", "ITL (p50, ms)", "ITL (p95, ms)", "ITL (p99, ms)",
    "Throughput (text tok/s)",
    "Out tok/req (avg)",
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
