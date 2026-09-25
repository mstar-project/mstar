"""Run an mstar server under ``MSTAR_PHASE_TIMING`` and tabulate the result.

    python -m benchmark.worker_phases.server \
        --command "mstar serve orpheus --config configs/orpheus_tp2.yaml \
--gpus 0,1 --port 8100" \
        --period 100 --server-log /tmp/orpheus.log

The command is taken verbatim as a bare string; this only adds the phase
instrumentation to its environment. The server's own stdout/stderr goes to
``--server-log`` so the terminal carries just readiness and the tables.

Readiness is reported from the HEALTH ENDPOINT, never the log: when another
process still holds the port, uvicorn logs "startup complete" and only then
fails with "address already in use", so a log grep sails straight past it.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from .phases import Record, parse_line, render, segment


def _port_of(command: str, default: int = 8000) -> int:
    m = re.search(r"--port[= ]+(\d+)", command)
    return int(m.group(1)) if m else default


def _healthy(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _watch_health(port: int, proc: subprocess.Popen, timeout: int) -> None:
    """Announce readiness on the terminal, or say why it never came."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"[worker_phases] server exited ({proc.returncode}) before "
                  f"becoming ready", file=sys.stderr, flush=True)
            return
        if _healthy(port):
            print(f"[worker_phases] SERVER READY on port {port} "
                  f"({time.strftime('%H:%M:%S')}) -- start the client now",
                  flush=True)
            return
        time.sleep(2)
    print(f"[worker_phases] server not healthy after {timeout}s",
          file=sys.stderr, flush=True)


class Tabulator:
    """Buffers records and emits a table every ``every`` records per worker.

    Segmentation runs over the whole buffer each time, so a table always
    reflects the batch-size grouping rather than an arbitrary cut.
    """

    def __init__(self, out, every: int, skip_warmup: int,
                 bs_tolerance: float, min_records: int,
                 phases: list[str] | None):
        self.out, self.every = out, every
        self.skip_warmup, self.bs_tolerance = skip_warmup, bs_tolerance
        self.min_records, self.phases = min_records, phases
        self.records: list[Record] = []
        self._since = 0

    def add(self, rec: Record) -> None:
        self.records.append(rec)
        self._since += 1
        if self._since >= self.every:
            self._since = 0
            self.emit()

    def emit(self, final: bool = False) -> None:
        segs = segment(
            self.records, skip_warmup=self.skip_warmup,
            bs_tolerance=self.bs_tolerance, min_records=self.min_records,
        )
        if not segs:
            return
        tag = "FINAL" if final else f"after {len(self.records)} records"
        print(f"\n{'=' * 78}\n[worker_phases] {tag}\n{'=' * 78}",
              file=self.out, flush=True)
        print(render(segs, self.phases), file=self.out, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run an mstar server with phase timing and tabulate it.")
    ap.add_argument("--command", required=True,
                    help="the mstar command, verbatim, as one string")
    ap.add_argument("--period", type=int, default=100,
                    help="MSTAR_PHASE_TIMING: iterations per flush")
    ap.add_argument("--server-log", default="worker_phases_server.log",
                    help="where the server's own stdout/stderr goes")
    ap.add_argument("--out", default="-",
                    help="where tables go ('-' for stdout)")
    ap.add_argument("--every", type=int, default=20,
                    help="emit a table every N records")
    ap.add_argument("--skip-warmup", type=int, default=3,
                    help="drop this many leading records per worker")
    ap.add_argument("--bs-tolerance", type=float, default=0.15,
                    help="fractional batch-size change that starts a new "
                         "segment (0.15 = 15%%)")
    ap.add_argument("--min-records", type=int, default=3,
                    help="drop segments shorter than this (brief transitions)")
    ap.add_argument("--phases", nargs="*", default=None,
                    help="only show phases containing one of these")
    ap.add_argument("--health-timeout", type=int, default=3600,
                    help="seconds to wait for /health before giving up")
    args = ap.parse_args(argv)

    env = dict(os.environ, MSTAR_PHASE_TIMING=str(args.period))
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    port = _port_of(args.command)

    print(f"[worker_phases] MSTAR_PHASE_TIMING={args.period}; server log -> "
          f"{args.server_log}", flush=True)
    log = open(args.server_log, "w")
    proc = subprocess.Popen(
        args.command, shell=True, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        # setsid, so the whole tree can be signalled -- as a flag rather than
        # preexec_fn=os.setsid, which runs Python between fork and exec and
        # can deadlock on a lock another thread held at fork. The watcher
        # thread below is exactly the hazard.
        start_new_session=True,
    )
    threading.Thread(
        target=_watch_health, args=(port, proc, args.health_timeout),
        daemon=True,
    ).start()

    tab = Tabulator(out, args.every, args.skip_warmup, args.bs_tolerance,
                    args.min_records, args.phases)
    try:
        for line in proc.stdout:
            log.write(line)
            rec = parse_line(line)
            if rec is not None:
                tab.add(rec)
    except KeyboardInterrupt:
        pass
    finally:
        log.flush()
        tab.emit(final=True)
        if proc.poll() is None:
            # SIGINT to the group: killing the parent alone orphans the
            # conductor and workers, which keep their GPU memory and their
            # IPC handles and then break the next server.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=120)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log.close()
        if out is not sys.stdout:
            out.close()
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
