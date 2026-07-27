"""Aggregate ``--log-stats`` request profiles across many requests.

``mstar/profile/display.py`` pretty-prints one block per finished request.
This module parses those blocks back into structured rows and reports
mean / p50 / p95 / p99 over a whole run, so a benchmark's overheads can be
read off directly instead of eyeballing hundreds of blocks.

Three ways in:

    # 1. aggregate a log the server already wrote
    python -m mstar.profile.aggregate stats.log

    # 2. aggregate a live stream
    mstar serve orpheus --log-stats | python -m mstar.profile.aggregate -

    # 3. wrap the server: tee its output through, summarize when it exits
    python -m mstar.profile.aggregate --wrap -- mstar serve orpheus --log-stats

Note on precision: the display layer rounds (0.1 ms above 10 ms, 0.01 ms
below; byte counts to 1 decimal in binary units), so aggregates inherit that
quantization. It is well below the noise floor for everything the timeline
and per-node tables are used for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, TextIO

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BYTE_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}

_RE_START = re.compile(r"^\s*Request profile:\s*(\S+)\s*$")
_RE_RULE = re.compile(r"^[=-]{10,}\s*$")
_RE_SECTION = re.compile(r"^ (Inputs|Outputs|Timeline|Graph timings|Tensor transfer)\b")
# "   text         x1          31 B"
_RE_IO = re.compile(r"^ {3}(\S+)\s+x(\d+)\s+([\d.]+)\s+(B|KiB|MiB|GiB|TiB)\s*$")
# "   recv → preprocess done                      349.9 ms"
_RE_STAGE = re.compile(r"^ {3}(\S.*?)\s{2,}(-?[\d.]+)\s*ms\s*$")
# "   LLM"  (a node header inside the graph-timings table)
_RE_GRAPH_NODE = re.compile(r"^ {3}(\S+)\s*$")
# "     decode     n=266      1463.9 (   5.50)    1228.2 (   4.62) ..."
_RE_GRAPH_WALK = re.compile(r"^ {5}(\S+)\s+n=(\d+)\s+(.*)$")
_RE_GRAPH_CELL = re.compile(r"([\d.]+)\s*\(\s*([\d.]+)\s*\)")
# "     api_server_preprocess_worker → worker_0"
_RE_XFER_PAIR = re.compile(r"^ {5}(\S+)\s*→\s*(\S+)\s*$")
# "     worker_0"
_RE_XFER_SRC = re.compile(r"^ {5}(\S+)\s*$")
# "       audio_chunk             144.0 KiB      8.99 ms  (x36)"
_RE_XFER_ROW = re.compile(
    r"^ {7}(\S+)\s+([\d.]+)\s+(B|KiB|MiB|GiB|TiB)\s+([\d.]+)\s*ms\s+\(x(\d+)\)\s*$"
)
_RE_RX_HEADER = re.compile(r"^ {3}rx\b")
_RE_TX_HEADER = re.compile(r"^ {3}tx\b")


@dataclass
class IoRow:
    modality: str
    count: int
    total_bytes: int


@dataclass
class GraphRow:
    node: str
    walk: str
    exec_count: int
    # totals over the request, milliseconds
    all_ms: float
    fwd_ms: float
    pre_ms: float
    post_ms: float


@dataclass
class XferRow:
    edge: str
    source: str
    dest: str | None  # None for tx rows (the sender doesn't know the reader)
    count: int
    num_bytes: int
    ms: float


@dataclass
class ParsedProfile:
    """One request's profile, recovered from its printed block."""

    rid: str
    # stage label ("recv → preprocess done", "total") -> milliseconds
    stages: dict[str, float] = field(default_factory=dict)
    # checkpoint name -> ms since the first recorded checkpoint. Derived by
    # walking the printed stages in order, which is robust to the checkpoint
    # races that let display.py reorder segments between requests.
    offsets: dict[str, float] = field(default_factory=dict)
    graph: list[GraphRow] = field(default_factory=list)
    rx: list[XferRow] = field(default_factory=list)
    tx: list[XferRow] = field(default_factory=list)
    inputs: list[IoRow] = field(default_factory=list)
    outputs: list[IoRow] = field(default_factory=list)

    @property
    def total_ms(self) -> float | None:
        return self.stages.get("total")


def _bytes_of(value: str, unit: str) -> int:
    return int(round(float(value) * _BYTE_UNITS[unit]))


def _derive_offsets(ordered_stages: list[tuple[str, float]]) -> dict[str, float]:
    """Turn consecutive "a → b  <ms>" segments into offsets from the first
    checkpoint. ``display.py`` sorts segments by wall-clock time, so two
    requests can print the same checkpoints in different orders (the
    conductor's ``done`` and the api server's ``last chunk`` race). Offsets
    are stable under that reordering; raw stage labels are not."""
    offsets: dict[str, float] = {}
    cursor = 0.0
    for label, ms in ordered_stages:
        if "→" not in label:
            continue
        start, _, end = label.partition("→")
        start, end = start.strip(), end.strip()
        if not offsets:
            offsets[start] = 0.0
            cursor = 0.0
        cursor = offsets.get(start, cursor) + ms
        offsets[end] = cursor
    return offsets


def parse_stream(lines: Iterable[str]) -> Iterator[ParsedProfile]:
    """Yield a :class:`ParsedProfile` for each complete block in ``lines``.

    Tolerant by design: unrecognized lines (interleaved server logs, partial
    blocks at EOF) are skipped, and any ``Request profile:`` line starts a
    fresh block even if the previous one never terminated.
    """
    prof: ParsedProfile | None = None
    section = ""
    ordered_stages: list[tuple[str, float]] = []
    node = ""
    xfer_dir = ""  # "rx" | "tx"
    xfer_src = ""
    xfer_dst: str | None = None
    seen_open_rule = False

    def flush() -> Iterator[ParsedProfile]:
        nonlocal prof
        if prof is not None:
            prof.offsets = _derive_offsets(ordered_stages)
            yield prof
            prof = None

    for raw in lines:
        line = raw.rstrip("\n")

        start = _RE_START.match(line)
        if start:
            yield from flush()
            prof = ParsedProfile(rid=start.group(1))
            section = ""
            ordered_stages = []
            node = ""
            xfer_dir = xfer_src = ""
            xfer_dst = None
            seen_open_rule = False
            continue

        if prof is None:
            continue

        if _RE_RULE.match(line):
            if line.startswith("="):
                # The first "=" rule closes the header; the second closes the
                # block. (Section separators inside the block are "-" rules.)
                if seen_open_rule:
                    yield from flush()
                else:
                    seen_open_rule = True
            section = "" if line.startswith("=") else section
            continue

        sec = _RE_SECTION.match(line)
        if sec:
            section = sec.group(1)
            node = ""
            xfer_dir = xfer_src = ""
            xfer_dst = None
            continue

        if section in ("Inputs", "Outputs"):
            m = _RE_IO.match(line)
            if m:
                row = IoRow(m.group(1), int(m.group(2)), _bytes_of(m.group(3), m.group(4)))
                (prof.inputs if section == "Inputs" else prof.outputs).append(row)
            continue

        if section == "Timeline":
            m = _RE_STAGE.match(line)
            if m:
                label, ms = m.group(1).strip(), float(m.group(2))
                prof.stages[label] = ms
                if label != "total":
                    ordered_stages.append((label, ms))
            continue

        if section == "Graph timings":
            m = _RE_GRAPH_WALK.match(line)
            if m:
                cells = _RE_GRAPH_CELL.findall(m.group(3))
                if len(cells) >= 4 and node:
                    totals = [float(c[0]) for c in cells[:4]]
                    prof.graph.append(
                        GraphRow(node, m.group(1), int(m.group(2)), *totals)
                    )
                continue
            m = _RE_GRAPH_NODE.match(line)
            if m:
                node = m.group(1)
            continue

        if section == "Tensor transfer":
            if _RE_RX_HEADER.match(line):
                xfer_dir, xfer_src, xfer_dst = "rx", "", None
                continue
            if _RE_TX_HEADER.match(line):
                xfer_dir, xfer_src, xfer_dst = "tx", "", None
                continue
            m = _RE_XFER_ROW.match(line)
            if m and xfer_dir and xfer_src:
                row = XferRow(
                    edge=m.group(1),
                    source=xfer_src,
                    dest=xfer_dst,
                    count=int(m.group(5)),
                    num_bytes=_bytes_of(m.group(2), m.group(3)),
                    ms=float(m.group(4)),
                )
                (prof.rx if xfer_dir == "rx" else prof.tx).append(row)
                continue
            m = _RE_XFER_PAIR.match(line)
            if m:
                xfer_src, xfer_dst = m.group(1), m.group(2)
                continue
            m = _RE_XFER_SRC.match(line)
            if m:
                xfer_src, xfer_dst = m.group(1), None
            continue

    yield from flush()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class Stat:
    n: int
    mean: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    min: float | None
    max: float | None
    total: float


def summarize(values: list[float]) -> Stat:
    if not values:
        return Stat(0, None, None, None, None, None, None, 0.0)
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = (p / 100.0) * (len(ordered) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    return Stat(
        n=len(ordered),
        mean=sum(ordered) / len(ordered),
        p50=pct(50),
        p95=pct(95),
        p99=pct(99),
        min=ordered[0],
        max=ordered[-1],
        total=sum(ordered),
    )


@dataclass
class Metric:
    """A named per-request series plus its summary."""

    group: tuple[str, ...]  # e.g. ("LLM", "decode") or ("worker_0", "audio_chunk")
    name: str  # e.g. "all (ms/exec)"
    stat: Stat


class Aggregator:
    """Accumulates parsed profiles and renders the cross-request summary."""

    def __init__(self, warmup: int = 0, rids: set[str] | None = None):
        self.warmup = warmup
        # When set, only these request ids are aggregated. Essential when one
        # long-lived server writes a single --log-stats-file across a whole
        # sweep: without it every cell aggregates all requests since server
        # start, not just its own.
        self.rids = rids
        self.seen = 0
        self.skipped_rid = 0
        self.profiles: list[ParsedProfile] = []

    def add(self, prof: ParsedProfile) -> bool:
        """Record ``prof``; returns False if it was filtered out."""
        if self.rids is not None and prof.rid not in self.rids:
            self.skipped_rid += 1
            return False
        self.seen += 1
        if self.seen <= self.warmup:
            return False
        self.profiles.append(prof)
        return True

    # -- series builders ---------------------------------------------------

    def _series(self, extract) -> dict[tuple[tuple[str, ...], str], list[float]]:
        out: dict[tuple[tuple[str, ...], str], list[float]] = {}
        for prof in self.profiles:
            for group, name, value in extract(prof):
                out.setdefault((group, name), []).append(value)
        return out

    def timeline_metrics(self) -> list[Metric]:
        def extract(prof: ParsedProfile):
            for label, ms in prof.stages.items():
                yield ((), label, ms)

        return _as_metrics(self._series(extract))

    def offset_metrics(self) -> list[Metric]:
        def extract(prof: ParsedProfile):
            for name, ms in prof.offsets.items():
                yield ((), name, ms)

        return _as_metrics(self._series(extract))

    def graph_metrics(self) -> list[Metric]:
        def extract(prof: ParsedProfile):
            for row in prof.graph:
                g = (row.node, row.walk)
                n = max(row.exec_count, 1)
                yield (g, "execs/req", float(row.exec_count))
                yield (g, "all (ms/exec)", row.all_ms / n)
                yield (g, "fwd (ms/exec)", row.fwd_ms / n)
                yield (g, "pre (ms/exec)", row.pre_ms / n)
                yield (g, "post (ms/exec)", row.post_ms / n)
                yield (g, "all (ms/req)", row.all_ms)

        return _as_metrics(self._series(extract))

    def transfer_metrics(self, direction: str) -> list[Metric]:
        def extract(prof: ParsedProfile):
            rows = prof.rx if direction == "rx" else prof.tx
            for row in rows:
                who = f"{row.source} → {row.dest}" if row.dest else row.source
                g = (who, row.edge)
                n = max(row.count, 1)
                yield (g, "count/req", float(row.count))
                yield (g, "KiB/req", row.num_bytes / 1024.0)
                yield (g, "ms/req", row.ms)
                yield (g, "KiB/xfer", row.num_bytes / 1024.0 / n)
                yield (g, "ms/xfer", row.ms / n)
                if row.ms > 0:
                    yield (g, "MiB/s", row.num_bytes / 1024.0**2 / (row.ms / 1e3))

        return _as_metrics(self._series(extract))

    def io_metrics(self, which: str) -> list[Metric]:
        def extract(prof: ParsedProfile):
            rows = prof.inputs if which == "inputs" else prof.outputs
            for row in rows:
                g = (row.modality,)
                yield (g, "count/req", float(row.count))
                yield (g, "KiB/req", row.total_bytes / 1024.0)

        return _as_metrics(self._series(extract))

    # -- rendering ---------------------------------------------------------

    def render(self, width: int = 96) -> str:
        if not self.profiles:
            return "no request profiles found\n"

        out: list[str] = []
        head = f" Aggregate over {len(self.profiles)} requests"
        if self.warmup:
            head += f" (dropped first {min(self.warmup, self.seen)} as warmup)"
        if self.rids is not None:
            head += f" [rid-filtered; {self.skipped_rid} others in log ignored]"
        out.append("=" * width)
        out.append(head)
        out.append("=" * width)

        out += _table("Timeline stages (ms)", self.timeline_metrics(), [])
        out += _table(
            "Checkpoint offsets from first checkpoint (ms)",
            self.offset_metrics(),
            [],
        )
        out += _table("Graph timings", self.graph_metrics(), ["node", "walk"])
        out += _table("Tensor rx", self.transfer_metrics("rx"), ["source → dest", "edge"])
        out += _table("Tensor tx", self.transfer_metrics("tx"), ["source", "edge"])
        out += _table("Inputs", self.io_metrics("inputs"), ["modality"])
        out += _table("Outputs", self.io_metrics("outputs"), ["modality"])
        out.append("=" * width)
        return "\n".join(out) + "\n"

    def to_dict(self) -> dict:
        def pack(metrics: list[Metric]) -> list[dict]:
            return [
                {"group": list(m.group), "metric": m.name, **asdict(m.stat)}
                for m in metrics
            ]

        return {
            "num_requests": len(self.profiles),
            "num_seen": self.seen,
            "warmup_dropped": min(self.warmup, self.seen),
            "rid_filtered_out": self.skipped_rid if self.rids is not None else None,
            "timeline": pack(self.timeline_metrics()),
            "offsets": pack(self.offset_metrics()),
            "graph": pack(self.graph_metrics()),
            "rx": pack(self.transfer_metrics("rx")),
            "tx": pack(self.transfer_metrics("tx")),
            "inputs": pack(self.io_metrics("inputs")),
            "outputs": pack(self.io_metrics("outputs")),
        }

    def request_rows(self) -> list[dict]:
        """Flat per-request records, for callers doing their own analysis."""
        rows = []
        for prof in self.profiles:
            rows.append(
                {
                    "rid": prof.rid,
                    "total_ms": prof.total_ms,
                    "stages": prof.stages,
                    "offsets": prof.offsets,
                    "graph": [asdict(g) for g in prof.graph],
                    "rx": [asdict(x) for x in prof.rx],
                    "tx": [asdict(x) for x in prof.tx],
                    "inputs": [asdict(i) for i in prof.inputs],
                    "outputs": [asdict(o) for o in prof.outputs],
                }
            )
        return rows


def _as_metrics(series: dict[tuple[tuple[str, ...], str], list[float]]) -> list[Metric]:
    return [
        Metric(group=group, name=name, stat=summarize(values))
        for (group, name), values in series.items()
    ]


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "0"
    mag = abs(v)
    if mag >= 1000:
        return f"{v:.1f}"
    if mag >= 10:
        return f"{v:.2f}"
    if mag >= 0.1:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _table(title: str, metrics: list[Metric], group_headers: list[str]) -> list[str]:
    if not metrics:
        return []
    # Stable ordering: by group, then by the order metrics were first produced
    # (so "execs/req" precedes the per-exec breakdown).
    name_order: dict[str, int] = {}
    for m in metrics:
        name_order.setdefault(m.name, len(name_order))
    metrics = sorted(metrics, key=lambda m: (m.group, name_order[m.name]))

    headers = [*group_headers, "metric", "n", "mean", "p50", "p95", "p99", "min", "max"]
    rows: list[list[str]] = []
    for m in metrics:
        s = m.stat
        rows.append(
            [
                *[g for g in m.group],
                m.name,
                str(s.n),
                _fmt(s.mean),
                _fmt(s.p50),
                _fmt(s.p95),
                _fmt(s.p99),
                _fmt(s.min),
                _fmt(s.max),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    n_left = len(group_headers) + 1  # group columns + "metric" are left-aligned
    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.ljust(widths[i]) if i < n_left else cell.rjust(widths[i]))
        return "  " + "  ".join(parts).rstrip()

    lines = ["", f" {title}", fmt_row(headers), "  " + "-" * (sum(widths) + 2 * (len(widths) - 1))]
    # Blank line between groups so blocks read as units.
    last_group: tuple[str, ...] | None = None
    for m, row in zip(metrics, rows, strict=True):
        if last_group is not None and m.group != last_group:
            lines.append("")
        lines.append(fmt_row(row))
        last_group = m.group
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_rids(paths: list[str]) -> set[str]:
    """Read request ids from plain-text, JSONL, or benchmark-runner JSON files.

    Accepts whatever the driver happened to write: one id per line, the
    runner's ``--rids-out`` list, its full ``--json`` result (ids are pulled
    out of the measured waves), or this module's own ``--dump-requests``
    JSONL.
    """
    rids: set[str] = set()
    for path in paths:
        text = open(path).read().strip()
        if not text:
            continue
        if text[0] in "[{":
            try:
                rids |= _rids_from_json(json.loads(text))
                continue
            except json.JSONDecodeError:
                # JSONL: one object per line
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rids |= _rids_from_json(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                continue
        rids |= {ln.strip() for ln in text.splitlines() if ln.strip()}
    return rids


def _rids_from_json(obj) -> set[str]:
    """Pull every ``rid`` / ``request_id`` out of a nested JSON structure.

    A bare list of strings is taken as a list of ids (``--rids-out``); inside
    objects only the two id keys count, so config strings elsewhere in a
    runner result file are not mistaken for request ids.
    """
    found: set[str] = set()
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in ("rid", "request_id"):
                value = node.get(key)
                if isinstance(value, str):
                    found.add(value)
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    found.add(item)
                else:
                    stack.append(item)
    return found


def _consume(
    lines: Iterable[str],
    agg: Aggregator,
    tee: TextIO | None = None,
    every: int = 0,
) -> None:
    """Feed ``lines`` into ``agg``, optionally echoing them to ``tee``."""

    def tapped() -> Iterator[str]:
        for line in lines:
            if tee is not None:
                tee.write(line)
                tee.flush()
            yield line

    for prof in parse_stream(tapped()):
        agg.add(prof)
        if every and len(agg.profiles) and len(agg.profiles) % every == 0:
            sys.stderr.write(agg.render())
            sys.stderr.flush()


def _run_wrapped(cmd: list[str], agg: Aggregator, every: int) -> int:
    """Run ``cmd``, pass its output through, and parse profiles from stdout."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit: server logs stay on stderr, untouched
        text=True,
        bufsize=1,
        env=env,
    )

    # Forward Ctrl-C to the child and let it shut down; we still summarize.
    def forward(signum, _frame):
        proc.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, forward)
        except ValueError:
            pass  # not on the main thread

    assert proc.stdout is not None
    try:
        _consume(proc.stdout, agg, tee=sys.stdout, every=every)
    finally:
        proc.stdout.close()
    return proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mstar-profile-aggregate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="log files containing --log-stats blocks ('-' or empty reads stdin)",
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="treat the args after -- as a server command to run, tee, and summarize",
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="drop the first N requests (torch.compile)",
    )
    parser.add_argument(
        "--every", type=int, default=0,
        help="print a rolling summary to stderr every N requests",
    )
    parser.add_argument("--json", default=None, help="write the summary as JSON here")
    parser.add_argument(
        "--dump-requests", default=None,
        help="write one JSON object per request here (JSONL), for custom analysis",
    )
    parser.add_argument(
        "--rids", action="append", default=[],
        help="only aggregate these request ids. Takes a text file (one id per "
             "line), a benchmark runner --json / --rids-out file, or a "
             "--dump-requests JSONL. Repeatable. Use this whenever one server "
             "writes a single stats file across several runs, otherwise every "
             "run's summary includes all earlier requests too.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the human-readable summary"
    )
    args, rest = parser.parse_known_args(argv)

    rids = load_rids(args.rids) if args.rids else None
    if rids is not None and not rids:
        parser.error("--rids matched no request ids")
    agg = Aggregator(warmup=args.warmup, rids=rids)
    rc = 0

    if args.wrap:
        cmd = [a for a in rest if a != "--"] or args.files
        if not cmd:
            parser.error("--wrap needs a command, e.g. --wrap -- mstar serve orpheus --log-stats")
        rc = _run_wrapped(cmd, agg, args.every)
    else:
        sources = args.files or ["-"]
        for path in sources:
            if path == "-":
                _consume(sys.stdin, agg, every=args.every)
            else:
                with open(path, "r", errors="replace") as fh:
                    _consume(fh, agg, every=args.every)

    if not args.quiet:
        sys.stderr.write(agg.render())
        sys.stderr.flush()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(agg.to_dict(), fh, indent=2)
    if args.dump_requests:
        with open(args.dump_requests, "w") as fh:
            for row in agg.request_rows():
                fh.write(json.dumps(row) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
