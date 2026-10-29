"""One step's kernel sequence with each position's median duration.

    python -m benchmark.kernels.trace.kseq TRACE.json ANCHOR_REGEX [FIRST LAST]

Steps are cut at ``ANCHOR_REGEX`` as in ``ktrace``; only steps of the most
common kernel count are kept, and position i is the i-th kernel of each.
"""

from __future__ import annotations

import collections
import json
import re
import sys


def sequence(trace: str, anchor: str) -> list[tuple[float, str]]:
    """(median duration in us, kernel name) per position of the common step."""
    with open(trace) as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("cat") == "kernel"]
    events.sort(key=lambda e: e["ts"])
    rx = re.compile(anchor)
    starts = [i for i, e in enumerate(events) if rx.search(e["name"])]
    if len(starts) < 2:
        raise SystemExit(f"anchor {anchor!r} matched {len(starts)} kernels")
    bounds = list(zip(starts, starts[1:], strict=False))
    n = collections.Counter(b - a for a, b in bounds).most_common(1)[0][0]
    steps = [events[a:b] for a, b in bounds if b - a == n]
    return [(sorted(s[i]["dur"] for s in steps)[len(steps) // 2], steps[0][i]["name"]) for i in range(n)]


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 4):
        raise SystemExit(__doc__)
    rows = sequence(args[0], args[1])
    lo, hi = (int(args[2]), int(args[3])) if len(args) == 4 else (0, len(rows))
    for i in range(lo, min(hi, len(rows))):
        print(f"{i:4d} {rows[i][0]:8.1f}  {rows[i][1][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
