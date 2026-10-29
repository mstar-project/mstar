"""Worker phases of one mstar row under a table workload, one window per concurrency.

    python -m benchmark.amd.phase_cells --row whisper_large --workload asr-libri --concurrency 8,32
        --gpus 0 --out DIR --worktree TREE --numa [--config YAML] [--ktrace-ms 1500] [--correctness]

One server with MSTAR_PHASE_TIMING (long_prompt.phase_server) gets one benchmark.runner run per concurrency, with the
workload's first-repeat warmup, and the phase records are cut by each run's time window: phases.txt has one table
per window, summary.json the phase means beside the client's numbers. worker.postprocess.event_sync is the only GPU
wait (benchmark/worker_phases/README.md), so where it stays near zero the steps are host-bound.

With --ktrace-ms, each measured window is followed by a second client run of the same cell that is not
measured: partway through it the worker's torch.profiler (benchmark/kernels/trace/proftrigger) records that
many ms, and benchmark.kernels.trace.ktrace cuts the trace into steps (ktrace-c<N>/ktrace.txt). The tracer
slows the host, which is why it stays out of the phase windows. --correctness sends the sweep's greedy text
pass (benchmark.amd.correctness.text_pass, one request at a time) right after the server starts and scores it
once the server is down (correctness/summary.json).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark.amd import catalog, correctness, long_prompt, sweep, tables
from benchmark.amd.launch import REPO, Launch, gpu_list

GAP_S = 5  # between windows, so no phase record straddles two
TRACE_AT = 0.4  # how far into the traced run, as a share of the measured run's length, the profiler starts


def traced_run(cmd: list[str], env: dict, rdir: Path, trigger: Path, window_ms: int, measured_s: float) -> str:
    """Run the cell again with the profiler opened partway through; returns the ktrace report, or why not."""
    rdir.mkdir(exist_ok=True)
    with open(rdir / "bench.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        time.sleep(max(2.0, TRACE_AT * measured_s))
        for f in trigger.parent.glob(trigger.name + ".done.*"):
            f.unlink()
        trigger.with_name(trigger.name + ".on").write_text(str(window_ms))
        p.wait()
    done = list(trigger.parent.glob(trigger.name + ".done.*"))
    if not done:
        return f"no trace: the profiler did not finish inside the run ({rdir / 'bench.log'})"
    for f in trigger.parent.glob(trigger.name + ".*.json"):
        f.rename(rdir / f.name)
    from benchmark.kernels.trace import ktrace

    try:
        text, result = ktrace.report(str(rdir))
    except SystemExit as e:
        return f"no table: {e}"
    (rdir / "ktrace.txt").write_text(text + "\n")
    (rdir / "ktrace.json").write_text(json.dumps(result, indent=1))
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", required=True, choices=sorted(catalog.ROWS))
    ap.add_argument("--workload", required=True, choices=sorted(tables.WORKLOADS))
    ap.add_argument("--concurrency", required=True, help="comma list, one window each, in this order")
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worktree", default=str(REPO))
    ap.add_argument("--period", type=int, default=20, help="MSTAR_PHASE_TIMING iterations per record")
    ap.add_argument("--numa", action="store_true", help="bind the server and the client to the GPU's NUMA node")
    ap.add_argument("--config", default=None, help="serving config instead of the row's (e.g. a resources override)")
    ap.add_argument("--ktrace-ms", type=int, default=0, help="trace each cell in an extra, unmeasured run")
    ap.add_argument("--correctness", action="store_true", help="greedy text pass after startup, scored at the end")
    ap.add_argument("--waves", type=int, default=None, help="requests per cell = waves x concurrency (the workload's"
                    " own otherwise); more waves make a short cell less sensitive to one slow step")
    args = ap.parse_args(argv)

    row, arm, wl = catalog.ROWS[args.row], catalog.ARMS["mstar"], tables.WORKLOADS[args.workload]
    if args.waves:
        wl = dataclasses.replace(wl, waves=args.waves)
    if wl.family != row.family:
        raise SystemExit(f"{wl.name} is a {wl.family} workload and {row.name} a {row.family} row")
    conc = [int(c) for c in args.concurrency.split(",")]
    gpus = gpu_list(args.gpus)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    numa = sweep._numa_of(gpus[0]) if args.numa else None
    trigger = out / "ktrace" / "trig"
    if args.ktrace_ms:
        # read by the server's environment (Launch.environment copies os.environ)
        trigger.parent.mkdir(exist_ok=True)
        hook = Path(args.worktree).resolve() / "benchmark" / "kernels" / "trace" / "proftrigger"
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), str(hook)]))
        os.environ["KTRACE_TRIGGER"] = str(trigger)
    launch = Launch(arm=arm, row=row, gpus=gpus, port=args.port, run_dir=out / "server", worktree=Path(args.worktree),
                    numa_node=numa, config=Path(args.config).resolve() if args.config else None)
    windows: dict[str, tuple[float, float]] = {}
    plan = SimpleNamespace(correctness_src=correctness.prepare_source(out, None), worktree=launch.worktree)
    with long_prompt.phase_server(launch, out, args.period) as startup_s:
        if args.correctness:
            # one request at a time, so arms that should agree token for token can be compared
            res = correctness.text_pass(plan, row, arm, args.port, out / "correctness", concurrency=1)
            print(f"correctness pass: {res}", flush=True)
            time.sleep(GAP_S)
        for c in conc:
            rdir = out / f"c{c}"
            rdir.mkdir(exist_ok=True)
            cmd = sweep.runner_command(arm, row, wl, c, args.port, rdir, wl.warmup_waves_first)
            if numa is not None:
                cmd = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", *cmd]
            t0 = time.time()
            rc, _ = sweep.run_client(cmd, sweep.client_env(arm, row, gpus), rdir / "bench.log")
            windows[f"c{c}"] = (t0, time.time())
            print(f"c={c}: runner rc {rc}", flush=True)
            time.sleep(GAP_S)
            if args.ktrace_ms:
                tdir = out / f"ktrace-c{c}"
                tcmd = sweep.runner_command(arm, row, wl, c, args.port, tdir, wl.warmup_waves_first)
                if numa is not None:
                    tcmd = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", *tcmd]
                text = traced_run(tcmd, sweep.client_env(arm, row, gpus), tdir, trigger, args.ktrace_ms,
                                  windows[f"c{c}"][1] - t0)
                print(f"c={c} trace:\n" + "\n".join(text.splitlines()[:40]), flush=True)
                time.sleep(GAP_S)
    cut = long_prompt.cut_by_window((out / "server.log").read_text(errors="replace"), windows)
    (out / "phases.txt").write_text("\n\n".join(f"### {k}\n{v['table']}" for k, v in cut.items()) + "\n")
    secs = sweep.input_seconds_of(row, wl)
    summary = {"row": row.name, "workload": wl.name, "gpus": gpus, "numa_node": numa,
               "config": str(launch.mstar_config()), "startup_s": startup_s, "windows": windows,
               "cells": {k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in cut.items()}}
    for c in conc:
        res = out / f"c{c}" / "results.json"
        if res.is_file():
            summary["cells"].setdefault(f"c{c}", {})["client"] = sweep.repeat_stats(
                json.loads(res.read_text()), row.family, secs, wl.output_len)
    if args.correctness and (out / "correctness" / "pass.json").is_file():
        C = correctness.load(plan.correctness_src)
        summary["correctness"] = correctness.score_dir(C, launch.worktree, out / "correctness", "cuda:0")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v.get("phases") for k, v in summary["cells"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
