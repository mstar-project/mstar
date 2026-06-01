#!/usr/bin/env python3
"""
Parse closed-loop TTS benchmark output and emit a single TSV row.

Usage:
    python bench_to_tsv.py < results.txt
    python bench_to_tsv.py results.txt
    some_command | python bench_to_tsv.py
    python bench_to_tsv.py results.txt --no-header   # row only

Columns (tab-separated):
    RTF (mean)  RTF (p50)  RTF (p95)  RTF (p99)
    SV (p50)    SV (mean)
    TTFA (mean) TTFA (p50) TTFA (p95) TTFA (p99)
    Text ITL (mean, ms)
    Throughput (audio sec / sec)
"""

import re
import sys


def find_stat(text, label, stat):
    """
    Find a stat (e.g. 'mean', 'p50') on the line whose metric label matches.

    `label` is matched at the start of a line, ignoring spaces/parens so that
    'TTFT (audio)' or 'RTF' work. Returns a float, or None if not found.
    """
    # Locate the line that begins with the given label (up to the ':' separator).
    # Collapse runs of whitespace in the label to \s+ so spacing differences
    # don't matter, and escape regex-special chars like parentheses.
    label_pat = r"\s+".join(re.escape(tok) for tok in label.split())
    line_re = re.compile(r"^\s*" + label_pat + r"[^\n]*", re.MULTILINE)
    m = line_re.search(text)
    if not m:
        return None
    line = m.group(0)
    # Pull "stat=number" off that line, tolerating an optional trailing unit (s, ms).
    val_re = re.compile(r"\b" + re.escape(stat) + r"\s*=\s*([0-9]*\.?[0-9]+)")
    vm = val_re.search(line)
    return float(vm.group(1)) if vm else None


def find_throughput_audio(text):
    """Throughput: 12.36 audio sec/s (synthesized audio per wall second)"""
    m = re.search(r"Throughput:\s*([0-9]*\.?[0-9]+)\s*audio\s*sec/s", text)
    return float(m.group(1)) if m else None


def fmt(v, ndigits=None):
    """Format a value for output; blank if missing."""
    if v is None:
        return ""
    if ndigits is not None:
        return f"{v:.{ndigits}f}"
    return f"{v:g}"


def parse(text):
    rtf_mean = find_stat(text, "RTF", "mean")
    rtf_p50 = find_stat(text, "RTF", "p50")
    rtf_p95 = find_stat(text, "RTF", "p95")
    rtf_p99 = find_stat(text, "RTF", "p99")

    # "Audio SV" line. Only p50 and mean are requested.
    sv_p50 = find_stat(text, "Audio SV", "p50")
    sv_mean = find_stat(text, "Audio SV", "mean")

    # TTFA == time to first audio == TTFT (audio).
    ttfa_mean = find_stat(text, "TTFT (audio)", "mean")
    ttfa_p50 = find_stat(text, "TTFT (audio)", "p50")
    ttfa_p95 = find_stat(text, "TTFT (audio)", "p95")
    ttfa_p99 = find_stat(text, "TTFT (audio)", "p99")

    # Text ITL mean is reported in seconds; convert to ms.
    text_itl_mean_s = find_stat(text, "ITL  (text)", "mean")
    if text_itl_mean_s is None:
        text_itl_mean_s = find_stat(text, "ITL (text)", "mean")
    text_itl_mean_ms = text_itl_mean_s * 1000 if text_itl_mean_s is not None else None

    throughput_audio = find_throughput_audio(text)

    return [
        fmt(rtf_mean),
        fmt(rtf_p50),
        fmt(rtf_p95),
        fmt(rtf_p99),
        fmt(sv_p50),
        fmt(sv_mean),
        fmt(ttfa_mean),
        fmt(ttfa_p50),
        fmt(ttfa_p95),
        fmt(ttfa_p99),
        fmt(text_itl_mean_ms, 3).rstrip("0").rstrip(".") if text_itl_mean_ms is not None else "",
        fmt(throughput_audio),
    ]


HEADER = [
    "RTF (mean)", "RTF (p50)", "RTF (p95)", "RTF (p99)",
    "SV (p50)", "SV (mean)",
    "TTFA (mean)", "TTFA (p50)", "TTFA (p95)", "TTFA (p99)",
    "Text ITL (mean, ms)",
    "Throughput (audio sec / sec)",
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