"""Worker phases of mstar's Qwen3.5 when ~1k-token prompts arrive together.

    python -m benchmark.amd.long_prompt --row qwen3_5_0.8b --gpus 4 --port 8700 --out DIR --worktree TREE

In the MI300X dry runs (functional only), mstar's text-long-1k cell at concurrency 4 had a TTFT p50 of
430 ms and p95 of 1.9 s, and fewer tok/s than at concurrency 1, where vLLM went from 126 to 182 ms.
The guess to test: prompts past the 512-token captured-prefill bound of the GDN layers
(linear_attn.max_captured_prefill_tokens) prefill outside a graph, those steps are bound by kernel
launches on the host, and the other requests' steps wait behind them.

One server with MSTAR_PHASE_TIMING (benchmark.worker_phases.server) gets four cells from
benchmark.runner, and the phase records are cut by each cell's time window:

    prefill-long    ~1k-token prompts, 1 output token, concurrency 1: every step is a long prefill
    prefill-short   the same prompts cut to ~400 tokens, under the bound, so the prefill is captured
    burst-long      ~1k-token prompts, 64 output tokens, concurrency 4: the slow cell
    burst-short     short chat prompts, 64 output tokens, concurrency 4: control

phases.txt has one table per cell; summary.json has the numbers below. Reading them:

  - The guess holds if, in prefill-long, worker.gpu_thread.exec (host time spent launching) is most of
    iter_total while worker.postprocess.event_sync (the only GPU wait) stays small, exec per step is
    several times prefill-short's (well past the 2.5x token ratio), and server.log has "runs eagerly"
    warnings for the prefill buckets above 512 tokens.
  - If event_sync is most of prefill-long's iter_total instead, the prefill kernels themselves are slow:
    profile them (rocprofv3) rather than the host.
  - If both prefills look alike per token but burst-long's TTFT still climbs with a request's place in
    the burst, prefills are not batched together: read the per-request TTFTs in burst-long's results.json
    and the batch size (bs) of its phase records.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmark.amd import catalog, sweep, tables
from benchmark.amd.launch import REPO, Launch, collect_versions, gpu_list
from benchmark.worker_phases.phases import parse_log, render, segment

SHORT_TOKENS = 400  # body tokens; with the chat template the prompt stays under the 512-token bound
PHASES = ("iter_total", "worker.gpu_thread.exec", "worker.gpu_thread.prepare_inputs",
          "worker.postprocess.event_sync", "gpu_exec", "await_gpu")


def cells(short_prompts: Path) -> list[tuple[tables.Workload, int]]:
    long_p = tables.LONG_PROMPTS
    return [
        (tables.Workload("prefill-long", "text", "text_to_text", (1,), long_p, output_len=1, unique_prompts=True,
                         min_requests=24, warmup_waves_first=3), 1),
        (tables.Workload("prefill-short", "text", "text_to_text", (1,), str(short_prompts), output_len=1,
                         unique_prompts=True, min_requests=24, warmup_waves_first=3), 1),
        (tables.Workload("burst-long", "text", "text_to_text", (4,), long_p, output_len=64, unique_prompts=True,
                         min_requests=40, warmup_waves_first=2), 4),
        (tables.Workload("burst-short", "text", "text_to_text", (4,), tables.TEXT_PROMPTS, output_len=64,
                         min_requests=40, warmup_waves_first=2), 4),
    ]


def make_short_prompts(src: Path, dest: Path, tokenizer_id: str, tokens: int = SHORT_TOKENS) -> Path:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dest.write_text("\n".join(tok.decode(tok(ln).input_ids[:tokens]) for ln in lines) + "\n", encoding="utf-8")
    return dest


def cut_by_window(log_text: str, windows: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """Phase table and per-phase means of worker_0's longest segment, per cell time window."""
    records = parse_log(log_text)
    out = {}
    for name, (t0, t1) in windows.items():
        recs = [r for r in records if r.ts is not None and t0 <= r.ts <= t1]
        segs = segment(recs, skip_warmup=1, min_records=1)
        if not segs:
            out[name] = {"table": "(no phase records in this window)", "phases": {}, "bs": None}
            continue
        main = max([s for s in segs if s.worker == "worker_0"] or segs, key=lambda s: len(s.records))
        summ = main.summary()
        out[name] = {"table": render(segs), "bs": round(main.mean_bs, 2), "records": len(main.records),
                     "phases": {p: {k: round(summ[p][k], 4) for k in ("mean_ms", "p50_ms", "p95_ms")}
                                for p in PHASES if p in summ}}
    return out


def readout(cut: dict[str, dict], refusals: int) -> dict:
    """The numbers the reading rules in the module docstring look at."""
    def mean(cell, phase):
        return (cut.get(cell, {}).get("phases", {}).get(phase) or {}).get("mean_ms")

    res = {"prefill_buckets_running_eagerly": refusals}
    for cell in ("prefill-long", "prefill-short", "burst-long", "burst-short"):
        it, ex, sync = mean(cell, "iter_total"), mean(cell, "worker.gpu_thread.exec"), \
            mean(cell, "worker.postprocess.event_sync")
        res[cell] = {"iter_total_ms": it, "exec_share": round(ex / it, 3) if it and ex is not None else None,
                     "event_sync_share": round(sync / it, 3) if it and sync is not None else None}
    long_ex, short_ex = mean("prefill-long", "worker.gpu_thread.exec"), mean("prefill-short", "worker.gpu_thread.exec")
    res["exec_ratio_long_over_short"] = round(long_ex / short_ex, 2) if long_ex and short_ex else None
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", default="qwen3_5_0.8b", choices=[r for r, v in catalog.ROWS.items() if v.family == "text"])
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worktree", default=str(REPO))
    ap.add_argument("--period", type=int, default=20, help="MSTAR_PHASE_TIMING iterations per record")
    ap.add_argument("--config", default=None, help="serving config instead of the row's (e.g. a resources override)")
    args = ap.parse_args(argv)

    row, arm = catalog.ROWS[args.row], catalog.ARMS["mstar"]
    gpus = gpu_list(args.gpus)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    launch = Launch(arm=arm, row=row, gpus=gpus, port=args.port, run_dir=out / "server", worktree=Path(args.worktree),
                    config=Path(args.config).resolve() if args.config else None)
    launch.run_dir.mkdir(parents=True, exist_ok=True)
    short = make_short_prompts(REPO / tables.LONG_PROMPTS, out / "prompts_400.txt", row.hf_id)
    launch._take_locks()
    proc = None
    windows: dict[str, tuple[float, float]] = {}
    try:
        launch.check_free()
        for d in (launch.ipc_dir, launch.upload_dir):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
        wrapper = [sys.executable, "-m", "benchmark.worker_phases.server", "--command", shlex.join(launch.command()),
                   "--period", str(args.period), "--server-log", str(out / "server.log"),
                   "--out", str(out / "phases_live.txt"), "--skip-warmup", "1"]
        (out / "command.txt").write_text(shlex.join(wrapper) + "\n")
        versions = collect_versions(launch, launch.command(), launch.environment())
        versions["config_text"] = launch.mstar_config().read_text()
        (launch.run_dir / "versions.json").write_text(json.dumps(versions, indent=1, default=str))
        t_start = time.monotonic()
        with open(out / "wrapper.log", "w") as log:
            proc = subprocess.Popen(wrapper, cwd=REPO, env=launch.environment(), stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        deadline = time.monotonic() + 3600
        while "SERVER READY" not in (out / "wrapper.log").read_text(errors="replace"):
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"server did not come up; see {out / 'wrapper.log'}")
            time.sleep(5)
        startup_s = round(time.monotonic() - t_start, 1)
        for wl, c in cells(short):
            rdir = out / wl.name
            rdir.mkdir(exist_ok=True)
            cmd = sweep.runner_command(arm, row, wl, c, args.port, rdir, wl.warmup_waves_first)
            t0 = time.time()
            rc, _ = sweep.run_client(cmd, sweep.client_env(arm, row, gpus), rdir / "bench.log")
            windows[wl.name] = (t0, time.time())
            print(f"{wl.name}: runner rc {rc}", flush=True)
            time.sleep(5)  # a gap between windows, so no record straddles two cells
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)  # the wrapper stops the server's whole group
            try:
                proc.wait(180)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        launch.stop()
    log_text = (out / "server.log").read_text(errors="replace")
    cut = cut_by_window(log_text, windows)
    refusals = sum(1 for ln in log_text.splitlines() if "runs eagerly" in ln)
    (out / "phases.txt").write_text("\n\n".join(f"### {k}\n{v['table']}" for k, v in cut.items()) + "\n")
    summary = {"row": row.name, "gpus": gpus, "config": str(launch.mstar_config()), "startup_s": startup_s,
               "windows": windows, "readout": readout(cut, refusals),
               "cells": {k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in cut.items()}}
    for wl, _ in cells(short):
        res = out / wl.name / "results.json"
        if res.is_file():
            summary["cells"].setdefault(wl.name, {})["client"] = sweep.repeat_stats(
                json.loads(res.read_text()), "text", output_len=wl.output_len)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["readout"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

