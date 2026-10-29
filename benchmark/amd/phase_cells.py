"""Worker phases of one mstar row under a table workload, one window per concurrency.

    python -m benchmark.amd.phase_cells --row whisper_large --workload asr-libri --concurrency 8,32
        --gpus 0 --out DIR --worktree TREE --numa [--arm mstar-split]

One server with MSTAR_PHASE_TIMING (long_prompt.phase_server) gets one benchmark.runner run per concurrency, with the
workload's first-repeat warmup, and the phase records are cut by each run's time window: phases.txt has one table
per window, summary.json the phase means beside the client's numbers. worker.postprocess.event_sync is the only GPU
wait (benchmark/worker_phases/README.md), so where it stays near zero the steps are host-bound.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmark.amd import catalog, long_prompt, sweep, tables
from benchmark.amd.launch import REPO, Launch, gpu_list

GAP_S = 5  # between windows, so no phase record straddles two


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", required=True, choices=sorted(catalog.ROWS))
    mstar_arms = sorted(name for name, arm in catalog.ARMS.items() if arm.engine == "mstar")
    ap.add_argument("--arm", default="mstar", choices=mstar_arms,
                    help="which mstar arm's config and env to serve (default: mstar)")
    ap.add_argument("--workload", required=True, choices=sorted(tables.WORKLOADS))
    ap.add_argument("--concurrency", required=True, help="comma list, one window each, in this order")
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worktree", default=str(REPO))
    ap.add_argument("--period", type=int, default=20, help="MSTAR_PHASE_TIMING iterations per record")
    ap.add_argument("--numa", action="store_true", help="bind the server and the client to the GPU's NUMA node")
    args = ap.parse_args(argv)

    row, arm, wl = catalog.ROWS[args.row], catalog.ARMS[args.arm], tables.WORKLOADS[args.workload]
    if wl.family != row.family:
        raise SystemExit(f"{wl.name} is a {wl.family} workload and {row.name} a {row.family} row")
    conc = [int(c) for c in args.concurrency.split(",")]
    gpus = gpu_list(args.gpus)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    numa = sweep._numa_of(gpus[0]) if args.numa else None
    launch = Launch(arm=arm, row=row, gpus=gpus, port=args.port, run_dir=out / "server", worktree=Path(args.worktree),
                    numa_node=numa)
    windows: dict[str, tuple[float, float]] = {}
    with long_prompt.phase_server(launch, out, args.period) as startup_s:
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
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v.get("phases") for k, v in summary["cells"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
