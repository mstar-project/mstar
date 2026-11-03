"""Aggregate ``--log-stats-file`` request profiles into one timeline table.

Where does a request's time go outside the worker? Serve with
``--log-stats-file FILE``, drive any client (c=1 gives the cleanest split),
then::

    python -m benchmark.request_timeline.report FILE [--skip 2] [--client-e2e-ms 812]

Every checkpoint is placed by its mean offset from ``recv`` across requests,
and each stage is the gap between consecutive checkpoints in that order. The
worker's share is its span (first step's launch to last step's postprocess,
over every node, so meaningful when the workers share one host); everything
else is outside it.

It parses ``pretty_print_profile``'s text, the only form the stats file has,
and raises on a timeline line it doesn't know, so a change to that format
fails here rather than dropping a stage from the table.
"""
from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass, field

_SEG = re.compile(r"^\s{3}(.+?) → (.+?)\s+(-?[\d.]+) ms$")
_TOTAL = re.compile(r"^\s{3}total\s+(-?[\d.]+) ms$")
_SPAN = re.compile(r"^\s{3}worker span \(same-host clocks\)\s+(-?[\d.]+) ms$")
_LAST_READ = re.compile(r"^\s{3}last chunk after last output read\s+(-?[\d.]+) ms$")
_POST = re.compile(r"^\s{3}output postprocess \(sum\)\s+([\d.]+) ms$")
_NODE = re.compile(r"^\s{3}(\S+)\s*$")
_WALK = re.compile(r"^\s{5}(\S+)\s+n=(\d+)\s+([\d.]+)\s+\(")


@dataclass
class Profile:
    rid: str
    offsets: dict[str, float] = field(default_factory=dict)  # checkpoint -> ms after the first one
    total_ms: float | None = None
    worker_span_ms: float | None = None
    postprocess_ms: float = 0.0
    node_busy_ms: dict[str, float] = field(default_factory=dict)
    node_steps: dict[str, int] = field(default_factory=dict)


def parse(text: str) -> list[Profile]:
    profiles: list[Profile] = []
    cur: Profile | None = None
    section = None
    node = None
    for line in text.splitlines():
        if line.startswith(" Request profile: "):
            cur = Profile(rid=line.split(": ", 1)[1].strip())
            profiles.append(cur)
            section = None
            continue
        if cur is None:
            continue
        if line.startswith(" Timeline:"):
            section = "timeline"
            continue
        if line.startswith(" Graph timings"):
            section = "graph"
            continue
        if line.startswith(" Tensor transfer") or line.startswith("-" * 10) or line.startswith("=" * 10):
            section = None if not line.startswith(" Tensor") else "transfer"
            continue
        if section == "timeline":
            if m := _SEG.match(line):
                a, b, ms = m.group(1), m.group(2), float(m.group(3))
                cur.offsets.setdefault(a, 0.0)
                cur.offsets[b] = cur.offsets[a] + ms
            elif m := _TOTAL.match(line):
                cur.total_ms = float(m.group(1))
            elif m := _SPAN.match(line):
                cur.worker_span_ms = float(m.group(1))
            elif m := _POST.match(line):
                cur.postprocess_ms = float(m.group(1))
            elif line.strip() and not _LAST_READ.match(line):
                raise ValueError(f"unrecognised timeline line in request {cur.rid}: {line!r}")
        elif section == "graph":
            if m := _WALK.match(line):
                key = f"{node}/{m.group(1)}"
                cur.node_busy_ms[key] = cur.node_busy_ms.get(key, 0.0) + float(m.group(3))
                cur.node_steps[key] = cur.node_steps.get(key, 0) + int(m.group(2))
            elif m := _NODE.match(line):
                node = m.group(1)
    return profiles


def _stats(xs: list[float]) -> str:
    if not xs:
        return f"{'-':>9} {'-':>9}"
    return f"{statistics.fmean(xs):9.1f} {statistics.median(xs):9.1f}"


def render(profiles: list[Profile], client_e2e_ms: float | None = None) -> str:
    profiles = [p for p in profiles if p.offsets]
    if not profiles:
        return "no request profiles with a timeline"
    labels: dict[str, list[float]] = {}
    for p in profiles:
        for label, off in p.offsets.items():
            labels.setdefault(label, []).append(off)
    order = sorted(labels, key=lambda k: statistics.fmean(labels[k]))
    out = [f"{len(profiles)} requests; ms (mean, p50)", "",
           f"{'stage':<44} {'mean':>9} {'p50':>9}"]
    for a, b in zip(order, order[1:], strict=False):
        gaps = [p.offsets[b] - p.offsets[a] for p in profiles if a in p.offsets and b in p.offsets]
        out.append(f"{a + ' → ' + b:<44} {_stats(gaps)}")
    totals = [p.total_ms for p in profiles if p.total_ms is not None]
    out.append(f"{'total (recv → last checkpoint)':<44} {_stats(totals)}")
    span = [p.worker_span_ms for p in profiles if p.worker_span_ms is not None]
    outside = [p.total_ms - p.worker_span_ms for p in profiles
               if p.total_ms is not None and p.worker_span_ms is not None]
    out += ["", f"{'worker span (start → end, all nodes)':<44} {_stats(span)}",
            f"{'outside the worker span (server side)':<44} {_stats(outside)}",
            f"{'  of which output postprocess (encode)':<44} {_stats([p.postprocess_ms for p in profiles])}"]
    if client_e2e_ms is not None and totals:
        out.append(f"{'client e2e − server total (HTTP, parse, reply)':<44} "
                   f"{client_e2e_ms - statistics.fmean(totals):9.1f}")
    nodes = sorted({k for p in profiles for k in p.node_busy_ms})
    if nodes:
        out += ["", f"{'worker busy per node/walk (sum of steps)':<44} {'mean':>9} {'p50':>9} {'steps':>6}"]
        for k in nodes:
            xs = [p.node_busy_ms[k] for p in profiles if k in p.node_busy_ms]
            steps = statistics.fmean([p.node_steps[k] for p in profiles if k in p.node_steps])
            out.append(f"{k:<44} {_stats(xs)} {steps:6.1f}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats_file")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N requests (warmup)")
    ap.add_argument("--client-e2e-ms", type=float, default=None,
                    help="the client's mean e2e for the same requests, to show HTTP/reply time")
    args = ap.parse_args()
    with open(args.stats_file) as f:
        profiles = parse(f.read())[args.skip:]
    print(render(profiles, args.client_e2e_ms))


if __name__ == "__main__":
    main()
