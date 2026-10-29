"""Time to first token, hop by hop, from ``MSTAR_PHASE_TIMING`` hop stamps.

With phase timing on, the API server, its data worker, the conductor and the
worker stamp each request at the points it passes on its way to the first
token (``mstar.utils.profiler.hop_stamp``) and log them as::

    <ts> INFO [worker_0] mstar.hops: hop-stamps wall_offset=1761753600.123456: \
<key> wk.ingest 1234.567890; <key> wk.ready 1234.568012; ...

The key is the submission key every process has for a request and the stamp
clock is ``perf_counter`` (CLOCK_MONOTONIC), shared by every process on the
host, so stamps from different processes subtract. ``wall_offset`` maps them to
the wall clock, which places a request in a benchmark run's time window.

    python -m benchmark.worker_phases.hops SERVER_LOG [--window T0,T1]

prints one row per hop: the time from the previous hop the request reached
(p50, mean, p95 over requests, in ms) and how many requests had both. The
first row's origin is the HTTP handler; the last hop is the first text delta
handed to the HTTP layer, so the hops sum to the server's share of TTFT (the
client's own TTFT adds the HTTP round trip on top).
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass

# The order a text request passes them. A hop a request never reached (the
# Rust frontend has no ``api.http_in``; a cached prompt may skip a read) is
# left out of that request's row, and the next hop is timed from the last one
# it did reach.
HOPS = (
    "api.http_in",    # chat handler entered (body parsed)
    "api.submit",     # submission key made, handed to the data worker
    "dw.start",       # data worker took it
    "dw.tokenized",   # process_prompt done
    "dw.sent",        # NEW_REQUEST handed to the conductor's socket
    "cond.arrival",   # conductor read it
    "cond.dispatch",  # NEW_REQUEST sent to the worker
    "wk.ingest",      # worker read it, handle minted
    "wk.ready",       # its input tensors read, node schedulable
    "wk.submit",      # its first step handed to the GPU thread
    "wk.launch",      # GPU thread: prepared, about to launch
    "wk.enqueued",    # GPU thread: kernels queued
    "wk.gpu_done",    # main thread: the step's completion event passed
    "wk.sent",        # main thread: outputs sent
    "api.result_in",  # API message thread read the result
    "dw.read",        # data worker read the output tensor
    "api.chunk",      # detokenized chunk back on the message thread
    "api.first_out",  # first text delta handed to the HTTP layer
)
_RANK = {h: i for i, h in enumerate(HOPS)}

_LINE = re.compile(r"hop-stamps wall_offset=(?P<off>-?[\d.]+): (?P<body>.+)$")


@dataclass
class Request:
    key: str
    stamps: dict[str, float]  # hop -> perf_counter, earliest seen
    wall_offset: float

    def wall(self, hop: str) -> float | None:
        t = self.stamps.get(hop)
        return None if t is None else t + self.wall_offset


def parse(text: str) -> dict[str, Request]:
    """Every request's stamps, keyed by submission key; a hop stamped more
    than once keeps its earliest time."""
    out: dict[str, Request] = {}
    for line in text.splitlines():
        m = _LINE.search(line)
        if m is None:
            continue
        off = float(m.group("off"))
        for entry in m.group("body").split("; "):
            parts = entry.split(" ")
            if len(parts) != 3:
                continue
            key, hop, raw = parts
            try:
                t = float(raw)
            except ValueError:
                continue
            req = out.get(key)
            if req is None:
                req = out[key] = Request(key, {}, off)
            prev = req.stamps.get(hop)
            if prev is None or t < prev:
                req.stamps[hop] = t
    return out


def in_window(requests: dict[str, Request], t0: float, t1: float) -> dict[str, Request]:
    """The requests whose first stamp falls in the wall-clock window."""
    sel = {}
    for k, r in requests.items():
        first = min(r.stamps.values()) + r.wall_offset
        if t0 <= first <= t1:
            sel[k] = r
    return sel


@dataclass(frozen=True)
class HopRow:
    hop: str
    p50_ms: float
    mean_ms: float
    p95_ms: float
    n: int


def _stats(hop: str, xs: list[float]) -> HopRow:
    xs = sorted(xs)
    n = len(xs)
    return HopRow(hop, xs[n // 2] * 1e3, statistics.fmean(xs) * 1e3, xs[min(n - 1, int(n * 0.95))] * 1e3, n)


def hop_table(requests: dict[str, Request]) -> list[HopRow]:
    """Per hop, the time since the previous hop each request reached, and a
    last ``total`` row: first hop to last, over the requests that have both
    ``api.http_in`` and ``api.first_out`` (or whichever ends they reached)."""
    deltas: dict[str, list[float]] = {h: [] for h in HOPS}
    totals: list[float] = []
    for r in requests.values():
        seen = sorted((h for h in r.stamps if h in _RANK), key=_RANK.__getitem__)
        for a, b in zip(seen, seen[1:], strict=False):
            deltas[b].append(r.stamps[b] - r.stamps[a])
        if len(seen) >= 2 and seen[-1] == HOPS[-1]:
            totals.append(r.stamps[seen[-1]] - r.stamps[seen[0]])
    rows = [_stats(h, xs) for h, xs in deltas.items() if xs]
    if totals:
        rows.append(_stats("total", totals))
    return rows


def render(rows: list[HopRow], title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{'hop (time since the previous one)':36s} {'p50_ms':>8s} {'mean_ms':>8s} {'p95_ms':>8s} {'n':>6s}")
    lines.append("-" * 70)
    for r in rows:
        lines.append(f"{r.hop:36s} {r.p50_ms:8.3f} {r.mean_ms:8.3f} {r.p95_ms:8.3f} {r.n:6d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--window", help="T0,T1: only requests whose first stamp is in this wall-clock window")
    args = ap.parse_args(argv)
    with open(args.log, errors="replace") as f:
        reqs = parse(f.read())
    if args.window:
        t0, t1 = (float(x) for x in args.window.split(","))
        reqs = in_window(reqs, t0, t1)
    print(render(hop_table(reqs), f"{len(reqs)} requests"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
