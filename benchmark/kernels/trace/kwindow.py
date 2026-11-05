"""Whole-window kernel totals by launch kind and name, for traces without a
clean step anchor.

    python -m benchmark.kernels.trace.kwindow TRACE.json [TOP]
"""

from __future__ import annotations

import collections
import json
import sys

from benchmark.kernels.trace.ktrace import launch_kind


def totals(trace: str, top: int = 25) -> str:
    with open(trace) as f:
        events = json.load(f)["traceEvents"]
    api = {e["args"]["correlation"]: e["name"] for e in events
           if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    runtime = collections.Counter(e["name"] for e in events if e.get("cat") == "cuda_runtime")
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        return "no kernels"
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e["dur"] for e in kernels)
    by_kind = collections.defaultdict(lambda: [0, 0.0])
    by_name = collections.defaultdict(lambda: [0, 0.0])
    for e in kernels:
        kind = launch_kind(api.get(e["args"].get("correlation")))
        for table, key in ((by_kind, kind), (by_name, (kind, e["name"][:80]))):
            table[key][0] += 1
            table[key][1] += e["dur"]
    out = [f"window {(t1 - t0) / 1e3:.1f} ms, kernels {len(kernels)}, "
           f"busy-sum {sum(e['dur'] for e in kernels) / 1e3:.1f} ms",
           f"runtime calls: {dict(runtime.most_common(8))}"]
    out += [f"  {k:6s} {n:7d} kernels {d / 1e3:9.1f} ms"
            for k, (n, d) in sorted(by_kind.items(), key=lambda kv: -kv[1][1])]
    out += [f"  {k:6s} {n:6d} {d / 1e3:8.2f} ms {d / n:7.1f} us  {name}"
            for (k, name), (n, d) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:top]]
    return "\n".join(out)


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(__doc__)
    print(totals(args[0], int(args[1]) if len(args) > 1 else 25))
    return 0


if __name__ == "__main__":
    sys.exit(main())
