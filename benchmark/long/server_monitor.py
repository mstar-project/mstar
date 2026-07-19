"""Server-side SHM-arena monitor (wrap the API server, tail its arena stats).

Launches the API server command as-is, tees its stderr through unchanged, and in
parallel parses the ``ARENA stats: {...}`` lines that ``--log-stats`` emits (one
per producer entity: ``api_server`` data worker + each ``worker_N``). For each
sample it derives the fragmentation / occupancy metrics, aggregates SHM usage
across entities, and writes a timestamped JSONL that lines up with the client
soak harness's ``--metrics-jsonl`` (both stamp ``t_wall``).

    python -m benchmark.long.server_monitor \
        --stats-jsonl shm_server.jsonl --shm-size-gb 64 -- \
        bash test/qwen3-omni/launch_server.sh

Everything after ``--`` is the server launch command, run verbatim.

Per-entity arenas: the arena is named ``mstar_arena_{entity_id}`` and created
once per entity (`mstar/communication/arena.py`), so **/dev/shm usage is
`segments x segment_size` PER entity** — the configured max
(`MSTAR_SHM_ARENA_MAX_SEGMENTS x MSTAR_SHM_ARENA_SEGMENT_MB`) multiplies by the
number of producer entities (workers + the api-server data worker). Pinned host
RAM is a separate axis and also per-process (own + peer segments, capped by
`MSTAR_SHM_ARENA_PIN_MAX_MB` each). This monitor surfaces both the per-entity and
the node-aggregate totals so those multipliers are visible.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import signal
import subprocess
import sys
import threading
import time

# 2026-... INFO [worker_0] mstar.communication.arena: ARENA stats: {'segments': 1, ...}
_ARENA_RE = re.compile(r"\[(?P<source>[^\]]+)\][^\n]*ARENA stats:\s*(?P<body>\{[^\n]*\})")

_MIB = 1 << 20
_GIB = 1 << 30


class ShmMonitor:
    def __init__(
        self,
        stats_jsonl: str | None,
        report_interval_s: float,
        shm_size_gb: float | None,
        frag_warn: float,
    ):
        self._lock = threading.Lock()
        self._sources: dict[str, dict] = {}  # entity -> latest derived record
        self._jsonl = open(stats_jsonl, "a") if stats_jsonl else None
        self._report_interval_s = report_interval_s
        self._shm_size = shm_size_gb * _GIB if shm_size_gb else None
        self._frag_warn = frag_warn
        self._start = time.time()
        self._frag_warned: set[str] = set()
        self._shm_warned = False
        # Peaks for the final summary.
        self._peak_segments: dict[str, int] = {}
        self._peak_pinned: dict[str, int] = {}
        self._peak_node_shm = 0
        self._worst_frag: dict[str, float] = {}

    # -- parsing ---------------------------------------------------------

    def handle_line(self, line: str) -> None:
        m = _ARENA_RE.search(line)
        if not m:
            return
        try:
            stats = ast.literal_eval(m.group("body"))
        except (ValueError, SyntaxError):
            return
        if "total_bytes" not in stats:
            return
        source = m.group("source")
        rec = self._derive(source, stats)
        with self._lock:
            self._sources[source] = rec
            self._peak_segments[source] = max(
                self._peak_segments.get(source, 0), rec["segments"])
            self._peak_pinned[source] = max(
                self._peak_pinned.get(source, 0), rec["pinned_bytes"])
            node_shm = sum(r["total_bytes"] for r in self._sources.values())
            self._peak_node_shm = max(self._peak_node_shm, node_shm)
            # Track worst (lowest) fragmentation headroom while free is healthy.
            if rec["free_over_total"] > 0.2:  # only when there's real free space
                self._worst_frag[source] = min(
                    self._worst_frag.get(source, 1.0), rec["largest_over_free"])
            if self._jsonl:
                self._jsonl.write(json.dumps(rec) + "\n")
                self._jsonl.flush()
            self._maybe_warn_locked(rec, node_shm)

    def _derive(self, source: str, s: dict) -> dict:
        total = s["total_bytes"]
        free = s["free_bytes"]
        largest = s["largest_free_block"]
        return {
            "t_wall": time.time(),
            "elapsed_s": round(time.time() - self._start, 2),
            "source": source,
            "segments": s["segments"],
            "total_bytes": total,
            "free_bytes": free,
            "largest_free_block": largest,
            "pinned_bytes": s.get("pinned_bytes", 0),
            # occupancy
            "free_over_total": round(free / total, 4) if total else 1.0,
            # fragmentation gauge: largest contiguous block as a fraction of all
            # free space. Collapsing toward 0 while free_over_total stays high is
            # the fragmentation signature.
            "largest_over_free": round(largest / free, 4) if free else 1.0,
        }

    def _maybe_warn_locked(self, rec: dict, node_shm: int) -> None:
        src = rec["source"]
        if (rec["largest_over_free"] < self._frag_warn
                and rec["free_over_total"] > 0.3
                and src not in self._frag_warned):
            print(
                f"[shm-monitor] WARN fragmentation on {src}: largest_free_block "
                f"is {rec['largest_over_free']:.0%} of free while "
                f"{rec['free_over_total']:.0%} of the arena is free "
                f"(largest={rec['largest_free_block'] // _MIB} MiB, "
                f"free={rec['free_bytes'] // _MIB} MiB)",
                file=sys.stderr, flush=True)
            self._frag_warned.add(src)
        if (self._shm_size and node_shm > 0.8 * self._shm_size
                and not self._shm_warned):
            print(
                f"[shm-monitor] WARN node /dev/shm usage {node_shm / _GIB:.1f} GiB "
                f"is >80% of --shm-size {self._shm_size / _GIB:.1f} GiB "
                f"(sum over {len(self._sources)} entities)",
                file=sys.stderr, flush=True)
            self._shm_warned = True

    # -- reporting -------------------------------------------------------

    def report(self) -> None:
        with self._lock:
            srcs = dict(self._sources)
        if not srcs:
            return
        node_shm = sum(r["total_bytes"] for r in srcs.values())
        node_pinned = sum(r["pinned_bytes"] for r in srcs.values())
        node_free = sum(r["free_bytes"] for r in srcs.values())
        elapsed = time.time() - self._start
        lines = [
            f"[shm-monitor t=+{elapsed:7.1f}s] entities={len(srcs)}  "
            f"node_shm={node_shm / _GIB:.2f}GiB  node_free/total="
            f"{node_free / node_shm:.2%}  node_pinned={node_pinned / _GIB:.2f}GiB"
        ]
        for src in sorted(srcs):
            r = srcs[src]
            lines.append(
                f"    {src:28s} seg={r['segments']:3d}  "
                f"shm={r['total_bytes'] // _MIB:6d}MiB  "
                f"free/total={r['free_over_total']:.2%}  "
                f"frag(largest/free)={r['largest_over_free']:.2%}  "
                f"pinned={r['pinned_bytes'] // _MIB:6d}MiB"
            )
        print("\n".join(lines), file=sys.stderr, flush=True)

    def reporter_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self._report_interval_s):
            self.report()

    def final_summary(self) -> None:
        with self._lock:
            print("\n[shm-monitor] ==== PEAKS ====", file=sys.stderr)
            for src in sorted(self._peak_segments):
                print(
                    f"    {src:28s} peak_segments={self._peak_segments[src]}  "
                    f"peak_pinned={self._peak_pinned[src] // _MIB}MiB  "
                    f"worst_frag(largest/free)="
                    f"{self._worst_frag.get(src, 1.0):.2%}",
                    file=sys.stderr)
            print(f"    node peak /dev/shm = {self._peak_node_shm / _GIB:.2f} GiB",
                  file=sys.stderr, flush=True)
        if self._jsonl:
            self._jsonl.close()


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        raise SystemExit(
            "usage: server_monitor.py [options] -- <server launch command>")
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def main() -> None:
    my_argv, cmd = _split_argv(sys.argv[1:])
    p = argparse.ArgumentParser(
        description="Wrap the API server and tail its SHM arena stats.")
    p.add_argument("--stats-jsonl", default=None,
                   help="append per-sample derived stats here (aligns with the "
                        "soak client's --metrics-jsonl via t_wall)")
    p.add_argument("--report-interval-s", type=float, default=15.0)
    p.add_argument("--shm-size-gb", type=float, default=None,
                   help="/dev/shm size for an >80%% usage warning (see `df -h /dev/shm`)")
    p.add_argument("--frag-warn", type=float, default=0.5,
                   help="warn when largest_free_block/free drops below this while "
                        "the arena still has free space (fragmentation)")
    args = p.parse_args(my_argv)

    if not cmd:
        raise SystemExit("no server launch command after --")

    mon = ShmMonitor(
        stats_jsonl=args.stats_jsonl,
        report_interval_s=args.report_interval_s,
        shm_size_gb=args.shm_size_gb,
        frag_warn=args.frag_warn,
    )
    print(f"[shm-monitor] launching: {' '.join(cmd)}", file=sys.stderr, flush=True)

    # stderr piped (parse + passthrough); stdout inherits straight to the terminal.
    proc = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, stdout=None, text=True, bufsize=1)

    # Forward Ctrl-C / TERM to the child so the server shuts down cleanly.
    def _forward(signum, _frame):
        proc.send_signal(signum)
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    stop = threading.Event()
    reporter = threading.Thread(
        target=mon.reporter_loop, args=(stop,), daemon=True)
    reporter.start()
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            sys.stderr.write(line)  # passthrough — server logs still visible
            mon.handle_line(line)
    finally:
        stop.set()
        proc.wait()
        mon.report()
        mon.final_summary()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
