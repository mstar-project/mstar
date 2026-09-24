"""Parse and summarise ``MSTAR_PHASE_TIMING`` output.

Pure functions, no I/O and no server: everything here can be exercised from a
string, which is what makes the segmentation reviewable (see ``test_phases.py``).

A worker logs one record per ``MSTAR_PHASE_TIMING`` iterations::

    <ts> INFO [worker_0] mstar.worker.worker: Worker worker_0 phase-timing \
iter=300 bs=15.94: await_gpu: p50=0.00ms p95=0.01ms mean=0.00ms n=100 | ...

``bs`` is the mean number of in-flight requests over the window. It is the
reason this module exists: a closed-loop run ramps up, holds, then drains, and
a phase mean taken across all three describes none of them. Records are grouped
into SEGMENTS of similar batch size, and each segment is summarised separately.
"""
from __future__ import annotations

import datetime as dt
import re
import statistics
from dataclasses import dataclass, field

# "Worker worker_0 phase-timing iter=300 bs=15.94: <body>". bs is optional so
# logs from before it was added still parse (they simply cannot be segmented).
_RECORD = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:[,.]\d+)?)?.*?"
    r"Worker\s+(?P<worker>\S+)\s+phase-timing\s+iter=(?P<iter>\d+)"
    r"(?:\s+bs=(?P<bs>[\d.]+))?:\s*(?P<body>.+)$"
)
_FIELD = re.compile(
    r"(?P<name>[\w.]+): p50=(?P<p50>[\d.]+)ms p95=(?P<p95>[\d.]+)ms "
    r"mean=(?P<mean>[\d.]+)ms n=(?P<n>\d+)"
)


@dataclass(frozen=True)
class Phase:
    p50: float
    p95: float
    mean: float
    n: int


@dataclass(frozen=True)
class Record:
    """One flush from one worker."""

    worker: str
    iter: int
    bs: float
    phases: dict[str, Phase]
    ts: float | None = None


def parse_line(line: str) -> Record | None:
    """A ``Record`` if the line is a phase-timing flush, else None."""
    m = _RECORD.match(line.rstrip("\n"))
    if m is None:
        return None
    phases = {
        g["name"]: Phase(float(g["p50"]), float(g["p95"]),
                         float(g["mean"]), int(g["n"]))
        for g in (d.groupdict() for d in _FIELD.finditer(m.group("body")))
    }
    if not phases:
        return None
    ts = None
    if m.group("ts"):
        raw = m.group("ts").replace(",", ".")
        fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in raw else "%Y-%m-%d %H:%M:%S"
        ts = dt.datetime.strptime(raw, fmt).timestamp()
    return Record(
        worker=m.group("worker"),
        iter=int(m.group("iter")),
        bs=float(m.group("bs")) if m.group("bs") else 0.0,
        phases=phases,
        ts=ts,
    )


def parse_log(text: str) -> list[Record]:
    out = [parse_line(ln) for ln in text.splitlines()]
    return [r for r in out if r is not None]


@dataclass
class Segment:
    """Consecutive records from one worker at a comparable batch size."""

    worker: str
    records: list[Record] = field(default_factory=list)

    @property
    def iters(self) -> tuple[int, int]:
        return self.records[0].iter, self.records[-1].iter

    @property
    def mean_bs(self) -> float:
        return statistics.fmean(r.bs for r in self.records)

    @property
    def bs_range(self) -> tuple[float, float]:
        return min(r.bs for r in self.records), max(r.bs for r in self.records)

    @property
    def request_steps(self) -> float:
        """Work the segment did: sum over records of ``bs`` x iterations.

        One worker iteration advances ``bs`` requests by a step. Per-iteration
        means are not comparable between two runs that interleave a different
        mix of iterations -- orpheus runs the snac decoder alongside the AR
        loop, so both the batch size and what one iteration means shift -- and
        dividing a phase's total time by this is throughput normalisation:
        cost per unit of output rather than per iteration.
        """
        total = 0.0
        for rec in self.records:
            it = rec.phases.get("iter_total")
            if it is not None and rec.bs:
                total += rec.bs * it.n
        return total

    def summary(self) -> dict[str, dict[str, float]]:
        """Per phase: n-weighted mean (exact), median p50, max p95, samples.

        Only the mean recombines exactly across records -- p50/p95 are already
        summarised per record, so they are reported as median-of-p50 and
        worst-p95 rather than pretending to be true percentiles.
        """
        acc: dict[str, dict] = {}
        for rec in self.records:
            for name, ph in rec.phases.items():
                a = acc.setdefault(
                    name, {"sum": 0.0, "n": 0, "p50s": [], "p95s": []})
                a["sum"] += ph.mean * ph.n
                a["n"] += ph.n
                a["p50s"].append(ph.p50)
                a["p95s"].append(ph.p95)
        work = self.request_steps
        return {
            name: {
                "mean_ms": a["sum"] / a["n"] if a["n"] else 0.0,
                "p50_ms": statistics.median(a["p50s"]),
                "p95_ms": max(a["p95s"]),
                "samples": a["n"],
                "records": len(a["p50s"]),
                # us of this phase per 1000 request-steps; 0.0 when the log
                # has no bs (older workers), which the renderer omits.
                "us_per_1k": a["sum"] / work * 1e6 if work else 0.0,
            }
            for name, a in sorted(acc.items())
        }


def segment(
    records: list[Record],
    *,
    skip_warmup: int = 0,
    bs_tolerance: float = 0.15,
    min_records: int = 1,
) -> list[Segment]:
    """Split each worker's records into runs of comparable batch size.

    ``skip_warmup`` drops that many leading records per worker -- the first
    flushes of a run include cold caches and the concurrency ramp.

    A record opens a new segment when its ``bs`` differs from the running mean
    of the current one by more than ``bs_tolerance`` (a fraction, so 0.15 means
    15%). That is what separates steady state from the drain without anyone
    having to guess a time window per workload. Segments shorter than
    ``min_records`` are dropped, which discards the brief transitions.

    With no ``bs`` in the log (older workers) every record reads bs=0.0 and
    this degenerates to one segment per worker, which is the old behaviour.
    """
    by_worker: dict[str, list[Record]] = {}
    for r in records:
        by_worker.setdefault(r.worker, []).append(r)

    out: list[Segment] = []
    for worker, recs in by_worker.items():
        recs = sorted(recs, key=lambda r: r.iter)[skip_warmup:]
        cur: Segment | None = None
        run_sum = 0.0
        for r in recs:
            if cur is not None:
                mean = run_sum / len(cur.records)
                ref = max(mean, 1e-9)
                if abs(r.bs - mean) / ref > bs_tolerance:
                    out.append(cur)
                    cur = None
            if cur is None:
                cur, run_sum = Segment(worker=worker), 0.0
            cur.records.append(r)
            run_sum += r.bs
        if cur is not None:
            out.append(cur)
    return [s for s in out if len(s.records) >= min_records]


def render(segments: list[Segment], phases: list[str] | None = None) -> str:
    """A table per segment. ``phases`` filters by substring."""
    buf: list[str] = []
    for seg in segments:
        lo, hi = seg.iters
        bmin, bmax = seg.bs_range
        buf.append(
            f"\n=== {seg.worker}  iters {lo}-{hi}  "
            f"records={len(seg.records)}  batch={seg.mean_bs:.2f} "
            f"(range {bmin:.1f}-{bmax:.1f}) ==="
        )
        rows = seg.summary()
        if phases:
            rows = {k: v for k, v in rows.items()
                    if any(p in k for p in phases)}
        work = seg.request_steps
        if work:
            buf.append(f"    work: {work:,.0f} request-steps "
                       f"(bs x iters) -- us/1k is cost per unit of output, "
                       f"comparable across runs with a different iteration mix")
        head = (f"{'phase':<40}{'mean_ms':>10}{'p50_ms':>9}"
                f"{'p95_ms':>9}{'samples':>9}"
                + (f"{'us/1k':>11}" if work else ""))
        buf.append(head)
        buf.append("-" * len(head))
        for name, v in rows.items():
            buf.append(
                f"{name:<40}{v['mean_ms']:>10.3f}{v['p50_ms']:>9.3f}"
                f"{v['p95_ms']:>9.3f}{v['samples']:>9}"
                + (f"{v['us_per_1k']:>11.1f}" if work else "")
            )
    return "\n".join(buf)
