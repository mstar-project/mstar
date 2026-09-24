"""Where a Chatterbox request's time goes, from the server's ``--log-stats`` profiles.

``mstar-serve --log-stats --log-stats-file stats.log`` writes one "Request
profile" block per request (``bench_all.sh`` keeps it as ``<run>/stats.log``).
This reads those blocks and reports, per run, the API preprocessing time,
the conductor's ingest-to-first-chunk and ingest-to-done spans, and the
worker's CPU time around every graph walk of every node (T3 prefill and
decode, S3Gen chunks, voice encoder) with per-execution averages. The
"residual" is ingest-to-done minus the request's own node time: queueing,
scheduling and the other requests' batches it waited behind.

Node times are wall time around the batches a request took part in, so at
concurrency > 1 a request's decode total covers steps it shared with
others; per-step averages stay comparable across concurrencies.

Usage::

    python benchmark/chatterbox/node_breakdown.py --results results/2026-09-18
    python benchmark/chatterbox/node_breakdown.py --stats results/2026-09-18/mstar_chatterbox_c8/stats.log --detail
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

_SIZE = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}
_IO_RE = re.compile(r"^\s+(\w+)\s+x(\d+)\s+([\d.]+) (B|KiB|MiB|GiB)\s*$")
_TIMELINE_RE = re.compile(r"^\s+(.+?) → (.+?)\s{2,}([\d.]+) ms\s*$")
_NODE_RE = re.compile(r"^   (\S+)\s*$")
_WALK_RE = re.compile(
    r"^\s+(\S+)\s+n=(\d+)\s+([\d.]+) \(\s*([\d.]+)\)\s+([\d.]+) \(\s*([\d.]+)\)"
    r"\s+([\d.]+) \(\s*([\d.]+)\)\s+([\d.]+) \(\s*([\d.]+)\)"
)
_XFER_RE = re.compile(r"^\s+(\w+)\s+([\d.]+) (B|KiB|MiB|GiB)\s+([\d.]+) ms\s+\(x(\d+)\)")

PCM_BYTES_PER_SECOND = 2 * 24_000  # 16-bit mono at 24 kHz, the server's output format


@dataclass
class WalkTime:
    n: int
    total: float  # ms, "all" column
    fwd: float
    pre: float
    post: float


@dataclass
class RequestProfile:
    request_id: str
    text_bytes: int = 0
    audio_bytes: int = 0
    audio_chunks: int = 0
    checkpoints: dict[str, float] = field(default_factory=dict)  # ms since recv
    walks: dict[tuple[str, str], WalkTime] = field(default_factory=dict)
    rx_ms: float = 0.0
    tx_ms: float = 0.0

    def span(self, start: str, end: str) -> float | None:
        if start in self.checkpoints and end in self.checkpoints:
            return self.checkpoints[end] - self.checkpoints[start]
        return None

    @property
    def node_total_ms(self) -> float:
        return sum(w.total for w in self.walks.values())

    @property
    def audio_seconds(self) -> float:
        return self.audio_bytes / PCM_BYTES_PER_SECOND


def parse_stats(path: Path) -> list[RequestProfile]:
    profiles: list[RequestProfile] = []
    current: RequestProfile | None = None
    section = None
    node = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip("\n")
        if line.startswith(" Request profile:"):
            current = RequestProfile(line.split(":", 1)[1].strip())
            profiles.append(current)
            current.checkpoints["recv"] = 0.0
            section = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("Inputs:"):
            section = "inputs"
        elif stripped.startswith("Outputs:"):
            section = "outputs"
        elif stripped.startswith("Timeline:"):
            section = "timeline"
        elif stripped.startswith("Graph timings"):
            section = "graph"
        elif stripped.startswith("Tensor transfer"):
            section = "xfer"
        elif stripped.startswith("rx (received"):
            section = "rx"
        elif stripped.startswith("tx (registered"):
            section = "tx"
        elif stripped.startswith("* post overlaps") or stripped.startswith("---"):
            if section == "graph":
                section = None
        elif section in ("inputs", "outputs"):
            m = _IO_RE.match(line)
            if m:
                size = float(m.group(3)) * _SIZE[m.group(4)]
                if section == "inputs" and m.group(1) == "text":
                    current.text_bytes = int(size)
                elif section == "outputs" and m.group(1) == "audio":
                    current.audio_bytes = int(size)
                    current.audio_chunks = int(m.group(2))
        elif section == "timeline":
            m = _TIMELINE_RE.match(line)
            if m:
                start, end, ms = m.group(1).strip(), m.group(2).strip(), float(m.group(3))
                if start in current.checkpoints:
                    current.checkpoints[end] = current.checkpoints[start] + ms
        elif section == "graph":
            m = _WALK_RE.match(line)
            if m and node is not None:
                current.walks[(node, m.group(1))] = WalkTime(
                    n=int(m.group(2)), total=float(m.group(3)), fwd=float(m.group(5)),
                    pre=float(m.group(7)), post=float(m.group(9)),
                )
                continue
            m = _NODE_RE.match(line)
            if m and "n=" not in line and not line.strip().startswith("all"):
                node = m.group(1)
        elif section in ("rx", "tx"):
            m = _XFER_RE.match(line)
            if m:
                if section == "rx":
                    current.rx_ms += float(m.group(4))
                else:
                    current.tx_ms += float(m.group(4))
    return profiles


def _p50(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _spans(profiles: list[RequestProfile], start: str, end: str) -> list[float]:
    return [s for p in profiles if (s := p.span(start, end)) is not None]


def walk_summary(profiles: list[RequestProfile]) -> dict[tuple[str, str], dict[str, float]]:
    """Per (node, walk): mean executions per request, mean total ms per
    request, and the per-execution mean (weighted by executions)."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    keys = sorted({k for p in profiles for k in p.walks})
    for key in keys:
        rows = [p.walks[key] for p in profiles if key in p.walks]
        n_total = sum(w.n for w in rows)
        out[key] = {
            "requests": len(rows),
            "n_per_request": n_total / len(profiles),
            "total_per_request": sum(w.total for w in rows) / len(profiles),
            "per_exec": sum(w.total for w in rows) / n_total if n_total else float("nan"),
            "fwd_per_exec": sum(w.fwd for w in rows) / n_total if n_total else float("nan"),
            "pre_per_exec": sum(w.pre for w in rows) / n_total if n_total else float("nan"),
            "post_per_exec": sum(w.post for w in rows) / n_total if n_total else float("nan"),
        }
    return out


def summarize(profiles: list[RequestProfile]) -> dict[str, float]:
    ingest_done = _spans(profiles, "conductor ingest", "conductor done")
    node_totals = [p.node_total_ms for p in profiles]
    residual = [d - n for d, n in zip(ingest_done, node_totals, strict=False)]
    return {
        "requests": len(profiles),
        "audio_s": _mean([p.audio_seconds for p in profiles]),
        "chunks": _mean([p.audio_chunks for p in profiles]),
        "total_p50": _p50(_spans(profiles, "recv", "finish")),
        "preprocess": _mean(_spans(profiles, "recv", "preprocess done")),
        "first_chunk_p50": _p50(_spans(profiles, "recv", "first chunk")),
        "ingest_done_p50": _p50(ingest_done),
        "node_total": _mean(node_totals),
        "residual": _mean(residual),
        "xfer": _mean([p.rx_ms + p.tx_ms for p in profiles]),
    }


def _walk(ws, node: str, walk_prefix: str) -> dict[str, float] | None:
    for (n, w), v in ws.items():
        if n == node and w.startswith(walk_prefix):
            return v
    return None


def _f(v: float | None, digits: int = 1) -> str:
    if v is None or math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def _triple(v: dict[str, float], n_digits: int, exec_digits: int) -> str:
    """``executions / ms per execution / ms per request`` for one walk."""
    return f"{_f(v['n_per_request'], n_digits)} / {_f(v['per_exec'], exec_digits)} / {_f(v['total_per_request'], 0)}"


def run_row(name: str, profiles: list[RequestProfile]) -> str:
    s = summarize(profiles)
    ws = walk_summary(profiles)
    prefill = _walk(ws, "T3", "prefill")
    decode = _walk(ws, "T3", "decode")
    s3gen = _walk(ws, "s3gen", "s3gen")
    voice = _walk(ws, "voice_encoder", "prefill")
    return " | ".join([
        name, str(s["requests"]), _f(s["audio_s"], 2), _f(s["total_p50"], 0), _f(s["preprocess"]),
        _f(s["first_chunk_p50"], 0), _f(s["ingest_done_p50"], 0),
        _f(prefill["per_exec"]) if prefill else "-",
        _triple(decode, 0, 2) if decode else "-",
        _triple(s3gen, 1, 0) if s3gen else "-",
        _f(voice["total_per_request"]) if voice else "-",
        _f(s["residual"], 0),
    ])


HEADER = (
    "run | req | audio s | total p50 ms | preprocess ms | first chunk p50 ms | ingest→done p50 ms | "
    "T3 prefill ms | T3 decode n / ms per step / ms | S3Gen n / ms per chunk / ms | VE ms | residual ms"
)


def detail(name: str, profiles: list[RequestProfile]) -> str:
    s = summarize(profiles)
    lines = [f"### {name}", "",
             f"{s['requests']} requests, mean audio {_f(s['audio_s'], 2)} s in {_f(s['chunks'], 1)} chunks; "
             f"total p50 {_f(s['total_p50'], 0)} ms, preprocess {_f(s['preprocess'])} ms, "
             f"recv→first chunk p50 {_f(s['first_chunk_p50'], 0)} ms, "
             f"ingest→done p50 {_f(s['ingest_done_p50'], 0)} ms, "
             f"node time {_f(s['node_total'], 0)} ms per request, residual {_f(s['residual'], 0)} ms, "
             f"tensor transport {_f(s['xfer'], 2)} ms", "",
             "node | walk | requests | execs per request | ms per request | ms per exec | fwd | pre | post",
             "--- | --- | --- | --- | --- | --- | --- | --- | ---"]
    for (node, walk), v in walk_summary(profiles).items():
        lines.append(
            f"{node} | {walk} | {v['requests']} | {_f(v['n_per_request'], 1)} | {_f(v['total_per_request'])} | "
            f"{_f(v['per_exec'], 2)} | {_f(v['fwd_per_exec'], 2)} | {_f(v['pre_per_exec'], 2)} | "
            f"{_f(v['post_per_exec'], 2)}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", help="results directory; every */stats.log becomes a row")
    parser.add_argument("--stats", help="one stats.log")
    parser.add_argument("--detail", action="store_true", help="per-walk table for each run")
    parser.add_argument("--skip", type=int, default=0, help="drop the first N requests (warm-up) of each run")
    parser.add_argument("--out", help="write the markdown here instead of stdout")
    args = parser.parse_args()
    if not args.results and not args.stats:
        parser.error("--results or --stats is required")

    # run dirs set aside by the stage scripts carry a dotted suffix (``<run>.failed.HHMMSS``) and are skipped
    files = [Path(args.stats)] if args.stats else sorted(
        p for p in Path(args.results).glob("*/stats.log") if "." not in p.parent.name
    )
    blocks = []
    rows = [HEADER, " | ".join(["---"] * (HEADER.count("|") + 1))]
    for path in files:
        profiles = parse_stats(path)[args.skip:]
        if not profiles:
            continue
        name = path.parent.name
        rows.append(run_row(name, profiles))
        if args.detail:
            blocks.append(detail(name, profiles))
    text = "\n".join(rows) + ("\n\n" + "\n\n".join(blocks) if blocks else "") + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
