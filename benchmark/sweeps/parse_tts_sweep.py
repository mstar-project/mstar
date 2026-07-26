#!/usr/bin/env python3
"""
Parse one closed-loop text-to-speech benchmark output (one config, one trial)
and emit a single TSV data row.

Column names map onto the raw output as:
    RTF        -> the "RTF" line (dimensionless, E2E / audio duration)
    SV         -> "Audio SV" (streaming viability; higher is better)
    TTFA       -> "TTFT (audio)" (time to first audio chunk)
    Text ITL   -> "ITL (text)", converted from seconds to ms
    Throughput -> "Throughput: ... audio sec/s"

Config columns (max concurrency, warmup, num requests, run) are supplied as
flags since they don't appear in the raw output.

Usage:
    python parse_tts_sweep.py run.txt --max-con 4 --num-requests 24 --num-warmup 2
    python parse_tts_sweep.py ... --no-header    # row only (later configs)
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


def find_audio_throughput(text):
    """Throughput: 12.34 audio sec/s (synthesized audio per wall second)"""
    m = re.search(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*audio\s*sec/s", text)
    return float(m.group(1)) if m else None


def parse_metrics(text):
    itl_text = find_stat(text, "ITL (text)", "mean")
    return [
        find_stat(text, "RTF", "mean"),
        find_stat(text, "RTF", "p50"),
        find_stat(text, "RTF", "p95"),
        find_stat(text, "RTF", "p99"),
        find_stat(text, "Audio SV", "p50"),
        find_stat(text, "Audio SV", "mean"),
        find_stat(text, "TTFT (audio)", "mean"),
        find_stat(text, "TTFT (audio)", "p50"),
        find_stat(text, "TTFT (audio)", "p95"),
        find_stat(text, "TTFT (audio)", "p99"),
        itl_text * 1000.0 if itl_text is not None else None,
        find_audio_throughput(text),
    ]


def fmt(v):
    return "" if v is None else f"{v:g}"


HEADER = [
    "Max con.", "Num warmup", "Num requests", "Run",
    "RTF (mean)", "RTF (p50)", "RTF (p95)", "RTF (p99)",
    "SV (p50)", "SV (mean)",
    "TTFA (mean)", "TTFA (p50)", "TTFA (p95)", "TTFA (p99)",
    "Text ITL (mean, ms)", "Throughput (audio sec / sec)",
]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("file", help="Raw benchmark output for one config")
    p.add_argument("--max-con", required=True)
    p.add_argument("--num-requests", required=True)
    p.add_argument("--num-warmup", required=True)
    p.add_argument("--run", default="1")
    p.add_argument("--no-header", action="store_true")
    args = p.parse_args(argv[1:])

    with open(args.file, "r", encoding="utf-8") as f:
        metrics = parse_metrics(f.read())

    cfg = [args.max_con, args.num_warmup, args.num_requests, args.run]

    if not args.no_header:
        print("\t".join(HEADER))
    print("\t".join(cfg + [fmt(v) for v in metrics]))


if __name__ == "__main__":
    main(sys.argv)
