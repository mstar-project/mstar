"""Run one benchmark table on a held node: every (row, arm, workload, concurrency) cell.

    python -m benchmark.amd.sweep --table qwen35-dense --node-type mi325x --gpus 0,1,2,3,4,5,6,7 \\
        --worktree $WORK/amd/worktrees/amd-native

Per table: a node health check (a FAIL aborts), then each row's arms back to back on the same
GPUs. Per arm: start the server (ready time and first request are the cold start), a greedy pass
on the correctness runner's sample set for text rows, then the timed cells, with health snapshots
around them and a 1 Hz monitor during them. A cell is warmup plus ``repeats`` timed runs of
benchmark.runner. A cell whose spread is over the table's limit is re-run once and flagged.
Everything goes under <run-root>/<table>/<run-id>/, and table.py turns it into table.md and
table.json.

Timings only count from an exclusive full node (mi3258x or mi3008x) with a passing check. --dry
runs the same path anywhere and labels the table as a functional run.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import fcntl
import fnmatch
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.amd import catalog, health, tables
from benchmark.amd.catalog import Arm, Row
from benchmark.amd.launch import (
    REPO,
    VRAM_BUSY_MB,
    Launch,
    _entries,
    amd_smi_json,
    arm_worktree,
    gpu_list,
    gpu_occupancy,
    host_record,
    smi_gpu,
    start_on_free_port,
    tree_list,
)

WORK = Path("/work1/stephaniewang/atindra")
# A cell is client-bound when the benchmark client's event loop was busy more than this share of the timed window.
CLIENT_BOUND_FRAC = 0.9
DEFAULT_RUN_ROOT = WORK / "amd/results/bench"
SETS_ROOT = WORK / "amd/results/correctness/data"
BENCH_PARTITIONS = {"mi3258x": "mi325x", "mi3008x": "mi300x"}
NODE_TYPES = {"mi325x": "MI325X", "mi300x": "MI300X"}


def client_for(arm: Arm, row: Row) -> str:
    # Every ASR arm is timed on the same OpenAI transcription route.
    if arm.engine == "mstar" and row.family == "asr":
        return "openai_transcribe"
    return arm.client


def workload_set_dir(row: Row, wl: tables.Workload, arm_dir: Path, arm: str | None = None) -> Path | None:
    """The sample set a row's cells send: the workload's, or for an image-to-video workload on a row with its own
    first frames (Row.i2v_frames) a copy of its manifest that points at those frames, unless the arm is one of the
    row's i2v_own_frames_arms. None for a text-file set."""
    if not wl.prompts.startswith("set:"):
        return None
    src = SETS_ROOT / "sets" / wl.prompts[4:]
    if wl.request_type != "image_to_video" or not row.i2v_frames or arm in row.i2v_own_frames_arms:
        return src
    manifest = json.loads((src / "manifest.json").read_text())
    for it in manifest["items"]:
        frame = Path(row.i2v_frames) / f"{it['id']}.png"
        if not frame.is_file():
            raise FileNotFoundError(frame)
        it["files"] = {**(it.get("files") or {}), "image": str(frame)}
    out = arm_dir / f"set-{wl.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


# Requests in a row that may fail before a run stops sending (at least the concurrency): a server that answers
# nothing then costs a few request timeouts instead of every request's.
ABORT_AFTER_FAILURES = 4


def runner_command(arm: Arm, row: Row, wl: tables.Workload, c: int, port: int, out_dir: Path, warmup: int,
                   sets_root: Path = SETS_ROOT, request_kwargs: dict | None = None,
                   set_dir: Path | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "benchmark.runner", "--url", f"http://127.0.0.1:{port}",
           "--model", row.runner_model, "--inference-system", client_for(arm, row),
           "--request-type", wl.request_type, "--profiling-type", "closed_loop", "--max-concurrency", str(c),
           "--num-requests", str(wl.num_requests(c)), "--num-warmup", str(warmup), "--output-dir", str(out_dir),
           "--local-cache", str(out_dir.parent / ".runner-cache"),
           "--abort-after-failures", str(max(ABORT_AFTER_FAILURES, c))]
    if row.runner_model_id:
        cmd += ["--model-id", row.hf_id]
    if wl.prompts.startswith("set:"):
        cmd += ["--dataset", "sample_set", "--sample-set-dir", str(set_dir or sets_root / "sets" / wl.prompts[4:])]
    else:
        cmd += ["--dataset", "text", "--request-txt-file", str(REPO / wl.prompts)]
    if wl.output_len:
        cmd += ["--ignore-eos", "--output-len-min", str(wl.output_len), "--output-len-max", str(wl.output_len)]
    if wl.unique_prompts:
        cmd += ["--unique-prompts"]
    if row.family == "text":
        cmd += ["--chat-template-kwargs", json.dumps({"enable_thinking": False})]
    if request_kwargs:
        cmd += ["--request-kwargs", json.dumps(request_kwargs)]
    if row.family == "omni" and client_for(arm, row) == "vllm_omni":
        # Raw PCM like mstar's chat stream, instead of vllm-omni's default WAV segments.
        cmd += ["--chat-audio-format", "pcm"]
    return cmd


# --- cell statistics ------------------------------------------------------------------------

def _p(stats: dict | None, key: str):
    return (stats or {}).get(key)


def repeat_stats(results: dict, family: str, input_seconds: list[float] | None = None,
                 output_len: int | None = None) -> dict:
    """The numbers one runner invocation contributes to a cell. With a fixed output length, tok/s
    counts at most that many tokens per request (mstar sends one token past max_tokens)."""
    agg = results.get("aggregate") or {}
    out = {"completed": results.get("completed"), "failed": results.get("failed"),
           "wall_s": results.get("wall_time_s"), "req_s": results.get("request_throughput"),
           "e2e_p50_s": _p(agg.get("e2e_latency_s"), "p50"), "e2e_p95_s": _p(agg.get("e2e_latency_s"), "p95")}
    per = results.get("per_request") or []
    if family == "text":
        ttft = (agg.get("ttft_s") or {}).get("text")
        itl = (agg.get("itl_s") or {}).get("text")
        tpots = []
        for r in per:
            n, t0 = r.get("output_text_tokens") or 0, (r.get("ttft_s") or {}).get("text")
            if n > 1 and t0 is not None:
                tpots.append((r["jct_ms"] / 1000.0 - t0) / (n - 1))
        tok_s = agg.get("text_token_throughput")
        wall = results.get("wall_time_s")
        if output_len and per and wall:
            tok_s = sum(min(r.get("output_text_tokens") or 0, output_len) for r in per) / wall
        out.update(ttft_p50_s=_p(ttft, "p50"), ttft_p95_s=_p(ttft, "p95"), itl_p50_s=_p(itl, "p50"),
                   itl_p95_s=_p(itl, "p95"), tpot_p50_s=statistics.median(tpots) if tpots else None,
                   tok_s=tok_s, tok_s_raw=agg.get("text_token_throughput"), out_tokens=agg.get("total_text_tokens"),
                   out_tokens_per_req=(agg.get("total_text_tokens") or 0) / max(len(per), 1),
                   in_tokens_per_req=(agg.get("total_input_tokens") or 0) / max(len(per), 1))
    elif family == "tts":
        ttfa = (agg.get("ttft_s") or {}).get("audio")
        out.update(ttfa_p50_s=_p(ttfa, "p50"), ttfa_p95_s=_p(ttfa, "p95"), rtf_p50=_p(agg.get("rtf"), "p50"),
                   audio_s_per_s=agg.get("audio_seconds_throughput"),
                   audio_s_per_req=agg.get("audio_duration_mean_s"))
    elif family == "asr":
        if input_seconds and per and results.get("wall_time_s"):
            secs = sum(input_seconds[r["request_id"] % len(input_seconds)] for r in per)
            out["rtfx"] = secs / results["wall_time_s"]
    elif family == "omni":
        ttfa, ttft = (agg.get("ttft_s") or {}).get("audio"), (agg.get("ttft_s") or {}).get("text")
        out.update(ttfa_p50_s=_p(ttfa, "p50"), ttfa_p95_s=_p(ttfa, "p95"), text_ttft_p50_s=_p(ttft, "p50"),
                   rtf_p50=_p(agg.get("rtf"), "p50"), audio_s_per_s=agg.get("audio_seconds_throughput"),
                   audio_s_per_req=agg.get("audio_duration_mean_s"))
    elif family == "image":
        out["images_per_s"] = results.get("request_throughput")
    elif family == "video":
        rate = results.get("request_throughput")
        out["clips_per_min"] = 60.0 * rate if rate else None
    return out


def summarize_repeats(repeats: list[dict]) -> dict:
    """Median and min/max of every numeric field over the repeats."""
    keys = sorted({k for r in repeats for k, v in r.items() if isinstance(v, (int, float)) and v is not None})
    out = {}
    for k in keys:
        vals = [r[k] for r in repeats if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def spread(summary: dict, key: str) -> float | None:
    s = summary.get(key)
    if not s or not s["median"]:
        return None
    return (s["max"] - s["min"]) / abs(s["median"])


def cotenants(job: str | None) -> dict:
    """Who else is on this node: workstream markers for the job and our python/apptainer processes."""
    users_dir = WORK / "amd/jobs/active" / f"{job}.users"
    markers = sorted(p.name for p in users_dir.iterdir()) if job and users_dir.is_dir() else []
    p = subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-af", "python|apptainer"], capture_output=True,
                       text=True, check=False)
    procs = [ln for ln in p.stdout.splitlines() if "pgrep" not in ln]
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "markers": markers, "loadavg": os.getloadavg(),
            "processes": procs}


# --- the driver ----------------------------------------------------------------------------

# Image and video rows running at once on a node, counted across every sweep there (dry runs too): more than two
# diffusion servers side by side throttled MI300X GPUs for up to a fifth of each arm's window.
MEDIA_LANES = 2
MEDIA_LANE_DIR = Path("/tmp") / os.environ.get("USER", "user") / "wsj-media-lanes"


@contextlib.contextmanager
def media_lane(n: int, log=print, poll_s: float = 10.0):
    """Hold one of the node's ``n`` media lanes while the block runs (a lock on one of n files, which the kernel
    drops with the process)."""
    MEDIA_LANE_DIR.mkdir(parents=True, exist_ok=True)
    waited = False
    while True:
        for i in range(n):
            f = open(MEDIA_LANE_DIR / f"lane{i}", "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        if not waited:
            log(f"waiting for one of the node's {n} media lanes")
            waited = True
        time.sleep(poll_s)


@dataclass
class Plan:
    table: tables.Table
    rows: list[Row]
    arms: list[str]
    workloads: list[tables.Workload]
    gpus: list[int]
    run_dir: Path
    worktree: Path
    port_base: int
    repeats: int
    lanes: int = 1
    dry: bool = False
    correctness: bool = True
    correctness_src: Path | None = None
    numa: bool = False
    resume: bool = False
    expect_gpus: int | None = None
    measured: list[str] = field(default_factory=list)  # row/arm of every arm this invocation ran
    trees: dict = field(default_factory=dict)  # named mstar trees (--tree) for arms that serve one
    fixed_concurrency: bool = False  # --concurrency given: it wins over a row's own per-workload concurrency
    skips: list = field(default_factory=list)  # --skip ROW/ARM[/WORKLOAD] patterns (fnmatch), e.g. from a dry run
    media_lanes: int = MEDIA_LANES  # image and video rows that may run at once on the node, across sweeps; 0: any
    log_lock: threading.Lock = field(default_factory=threading.Lock)

    def log(self, msg: str) -> None:
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
        with self.log_lock:
            print(line, flush=True)
            with open(self.run_dir / "sweep.log", "a") as f:
                f.write(line + "\n")


def _numa_of(gpu: int) -> int | None:
    """NUMA node of HIP device ``gpu``. amd-smi 26 wraps its JSON in {"gpu_data": [...]}, older ones don't."""
    for e in _entries(amd_smi_json("static", "--numa", "-g", str(smi_gpu(gpu)))):
        node = (e.get("numa") or {}).get("node") if isinstance(e, dict) else None
        if isinstance(node, int):
            return node
    return None


def client_env(arm: Arm, row: Row, gpus: list[int]) -> dict:
    """Environment of the benchmark client. It sees no GPU unless the arm decodes on one."""
    path = [str(REPO), *arm.client_path, os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in path if p), "HF_HUB_OFFLINE": "1"}
    if arm.client_gpu:
        server = gpus[: arm.server_gpus] if arm.server_gpus else gpus
        g = str((gpus[len(server):] or gpus)[0])
        env.update(HIP_VISIBLE_DEVICES=g, CUDA_VISIBLE_DEVICES=g)
    else:
        env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    return env


def run_client(cmd: list[str], env: dict, log_path: Path, timeout_s: float | None = None) -> tuple[int, float]:
    """Run one benchmark.runner invocation; returns its exit code and CPU seconds (user + system). With
    timeout_s, a runner still going after that long is killed and TimeoutError raised."""
    fired = threading.Event()

    def kill(pid: int) -> None:
        fired.set()
        os.kill(pid, 9)

    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        timer = threading.Timer(timeout_s, kill, (p.pid,)) if timeout_s else None
        if timer:
            timer.start()
        try:
            _, status, usage = os.wait4(p.pid, 0)
        finally:
            if timer:
                timer.cancel()
        p.returncode = os.waitstatus_to_exitcode(status)
    if fired.is_set():
        raise TimeoutError(f"the client was still waiting after {timeout_s:.0f} s")
    return p.returncode, usage.ru_utime + usage.ru_stime


def input_seconds_of(row: Row, wl: tables.Workload) -> list[float] | None:
    """Audio seconds of each item of an ASR row's sample set, for RTFx."""
    if row.family != "asr" or not wl.prompts.startswith("set:"):
        return None
    man = json.loads((SETS_ROOT / "sets" / wl.prompts[4:] / "manifest.json").read_text())
    return [float(it["params"]["duration_s"]) for it in man["items"]]


def run_cell(plan: Plan, arm: Arm, row: Row, wl: tables.Workload, c: int, port: int, arm_dir: Path,
             gpus: list[int], numa: int | None = None, request_kwargs: dict | None = None) -> dict:
    cell_dir = arm_dir / "cells" / wl.name / f"c{c}"
    done = cell_dir / "cell.json"
    if plan.resume and done.is_file() and json.loads(done.read_text()).get("status") != "error":
        return json.loads(done.read_text())
    cell_dir.mkdir(parents=True, exist_ok=True)
    input_seconds = input_seconds_of(row, wl)
    attempts = []
    all_failed = None
    for attempt in (1, 2):
        reps = []
        for k in range(1, plan.repeats + 1):
            rdir = cell_dir / f"a{attempt}r{k}"
            rdir.mkdir(parents=True, exist_ok=True)
            warm = wl.warmup_waves_first if k == 1 else wl.warmup_waves_next
            cmd = runner_command(arm, row, wl, c, port, rdir, warm, request_kwargs=request_kwargs,
                                 set_dir=workload_set_dir(row, wl, arm_dir, arm.name))
            if numa is not None:
                cmd = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", *cmd]
            (rdir / "cotenants.json").write_text(json.dumps(cotenants(os.environ.get("SLURM_JOB_ID")), indent=1))
            (rdir / "command.txt").write_text(shlex.join(cmd) + "\n")
            env = client_env(arm, row, gpus)
            t0 = time.monotonic()
            rc, cpu_s = run_client(cmd, env, rdir / "bench.log")
            wall = time.monotonic() - t0
            res_path = rdir / "results.json"
            if rc != 0 or not res_path.is_file():
                reps.append({"error": f"runner rc {rc}", "dir": str(rdir)})
                continue
            res = json.loads(res_path.read_text())
            st = repeat_stats(res, row.family, input_seconds, wl.output_len)
            # The runner process's CPU over its whole life (imports, tokenizer, warmup included), and its event
            # loop's CPU over the timed window alone, which is what says whether the client limited the cell.
            loop_cpu, timed = res.get("client_loop_cpu_s"), res.get("wall_time_s")
            st.update(dir=str(rdir), client_wall_s=round(wall, 1), client_cpu_s=round(cpu_s, 1),
                      client_cpu_frac=round(cpu_s / wall, 3) if wall > 0 else None,
                      client_loop_frac=round(loop_cpu / timed, 3) if loop_cpu is not None and timed else None)
            if st.get("failed"):
                st["error"] = f"{st['failed']} failed requests"
            reps.append(st)
            if st.get("failed") and not st.get("completed"):
                # Nothing came back: further repeats would only wait out the same timeouts.
                errs = res.get("errors") or []
                all_failed = str(errs[0].get("error"))[:200] if errs else "no request completed"
                break
        good = [r for r in reps if "error" not in r]
        summary = summarize_repeats(good)
        sp = spread(summary, wl.primary[0])
        attempts.append({"attempt": attempt, "repeats": reps, "summary": summary, "spread": sp})
        if len(good) == len(reps) and (sp is None or sp <= plan.table.spread_limit or plan.repeats < 2):
            break
        if len(good) < len(reps):
            break  # failures are not noise; a re-run would not fix them
        plan.log(f"  {arm.name} {row.name} {wl.name} c={c}: spread {sp:.1%} > {plan.table.spread_limit:.0%}, re-run")
    last = attempts[-1]
    flags = []
    if any((r.get("client_loop_frac") or 0) > CLIENT_BOUND_FRAC for r in last["repeats"]):
        flags.append("client-bound")
    if len(attempts) > 1:
        flags.append("rerun")
        if last["spread"] is not None and last["spread"] > plan.table.spread_limit:
            flags.append("noisy")
    errors = [r["error"] for r in last["repeats"] if "error" in r]
    cell = {"status": "error" if errors else "done", "arm": arm.name, "row": row.name, "workload": wl.name,
            "concurrency": c, "num_requests": wl.num_requests(c), "summary": last["summary"],
            "spread": last["spread"], "primary": wl.primary[0], "flags": flags, "errors": errors,
            "attempts": attempts}
    if all_failed:
        cell["all_failed"] = all_failed
    done.write_text(json.dumps(cell, indent=1))
    return cell


def run_arm(plan: Plan, row: Row, arm: Arm, gpus: list[int], port: int, lane: str) -> dict:
    arm_dir = plan.run_dir / row.name / arm.name
    summary_path = arm_dir / "arm.json"
    if plan.resume and summary_path.is_file():
        prev = json.loads(summary_path.read_text())
        if prev.get("status") == "done" and ((prev.get("health") or {}).get("verdict") or {}).get("status") != "FAIL":
            plan.log(f"{row.name}/{arm.name}: done earlier, skipped")
            return prev
        if prev.get("status") == "done":
            # Its cells are void (thermal throttling or new ECC errors during the arm): measure them again, and
            # keep the voided run beside the new one.
            void = arm_dir.with_name(f"{arm.name}.health-fail-{time.strftime('%Y%m%dT%H%M%S')}")
            arm_dir.rename(void)
            plan.log(f"{row.name}/{arm.name}: failed its health check earlier, re-run (old run in {void.name})")
    plan.measured.append(f"{row.name}/{arm.name}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"arm": arm.name, "row": row.name, "gpus": gpus, "port": port, "lane": lane, "status": "running",
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": []}
    numa = _numa_of(gpus[0]) if plan.numa else None
    server = None
    hlth = health.ArmHealth(arm_dir / "health", health.smi_gpus(gpus))
    try:
        if arm.engine != "reference":
            tree = arm_worktree(arm, plan.worktree, plan.trees)
            plan.log(f"{row.name}/{arm.name}: starting on GPUs {gpus} port {port}")
            server, info["startup"] = start_on_free_port(
                lambda p: Launch(arm=arm, row=row, gpus=gpus, port=p, run_dir=arm_dir / "server", worktree=tree,
                                 numa_node=numa, lane=lane), port, plan.log)
            port = info["port"] = server.port
            plan.log(f"{row.name}/{arm.name}: ready in {info['startup']['ready_s']} s, first request "
                     f"{info['startup']['first_request_s']} s")
            if plan.correctness and row.family in ("text", "omni"):
                from benchmark.amd import correctness
                passes = {"text": correctness.text_pass, "omni": correctness.omni_pass}
                try:
                    info["correctness_pass"] = passes[row.family](plan, row, arm, port, arm_dir / "correctness",
                                                                  worktree=tree)
                except Exception:
                    info["correctness_pass"] = {"error": traceback.format_exc()[-3000:]}
                    plan.log(f"{row.name}/{arm.name}: correctness pass failed, see arm.json")
        workloads = [wl for wl in row_workloads(plan, row) if not skipped(plan, row, arm, wl)]
        kwargs_by_wl: dict = {}
        if row.family in ("image", "video"):
            from benchmark.amd import correctness
            # The same steps, guidance, size and frames the correctness reference was made with.
            for wl in workloads:
                kwargs_by_wl[wl.name] = correctness.media_request_kwargs(plan, row, arm, wl,
                                                                         arm_worktree(arm, plan.worktree, plan.trees))
            info["request_kwargs"] = kwargs_by_wl
            if arm.engine != "reference":
                info["first_size"] = {wl.name: first_size_requests(plan, arm, row, wl, port, arm_dir, gpus, numa,
                                                                   kwargs_by_wl[wl.name]) for wl in workloads}
                plan.log(f"{row.name}/{arm.name}: first request per size {info['first_size']}")
                failed = {w: v for w, sizes in info["first_size"].items() for v in sizes.values() if isinstance(v, str)}
                if failed:
                    workloads = [wl for wl in workloads if wl.name not in failed]
                    info["error"] = "first request failed, its cells left out: " + "; ".join(
                        f"{w} {v}" for w, v in failed.items())
                    plan.log(f"{row.name}/{arm.name}: {info['error'][:300]}")
        hlth.begin()
        if arm.engine == "reference":
            from benchmark.amd import reference
            info["reference"] = reference.run_arm(plan, row, arm, gpus, arm_dir)
        else:
            for wl in workloads:
                request_kwargs = kwargs_by_wl.get(wl.name)
                for c in wl.concurrency if plan.fixed_concurrency else row.concurrency.get(wl.name, wl.concurrency):
                    cell = run_cell(plan, arm, row, wl, c, port, arm_dir, gpus, numa, request_kwargs)
                    s = cell["summary"].get(wl.primary[0], {})
                    plan.log(f"{row.name}/{arm.name} {wl.name} c={c}: {wl.primary[0]} "
                             f"{s.get('median', float('nan')):.4g} spread "
                             f"{cell['spread'] if cell['spread'] is not None else float('nan'):.1%} "
                             f"{' '.join(cell['flags'])} {'; '.join(cell['errors'])[:200]}")
                    info["cells"].append({k: cell[k] for k in ("workload", "concurrency", "status", "flags")})
                    if cell.get("all_failed"):
                        raise RuntimeError(f"every request of {wl.name} at c={c} failed ({cell['all_failed']}); "
                                           "the arm's other cells were not run")
        info["health"] = hlth.end()
        hlth = None
        info["status"] = "done"
    except Exception:
        info["status"] = "error"
        info["error"] = traceback.format_exc()[-5000:]
        plan.log(f"{row.name}/{arm.name}: ERROR {info['error'].strip().splitlines()[-1][:300]}")
        if hlth and hlth.monitor:
            info["health"] = hlth.end()
    finally:
        if server is not None:
            info["stop"] = server.stop()
            if info["stop"].get("leftover_pids"):
                plan.log(f"{row.name}/{arm.name}: leftover processes after stop: {info['stop']['leftover_pids']}")
        info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        summary_path.write_text(json.dumps(info, indent=1, default=str))
    return info


def skipped(plan: Plan, row: Row, arm: Arm, wl: tables.Workload | None = None) -> bool:
    """Whether a --skip pattern leaves out this arm on this row: for one workload, or with wl=None for every one."""
    for pat in getattr(plan, "skips", None) or []:
        r, a, w = (pat.split("/") + ["*"])[:3]
        if fnmatch.fnmatch(row.name, r) and fnmatch.fnmatch(arm.name, a) and (
                fnmatch.fnmatch(wl.name, w) if wl is not None else w == "*"):
            return True
    return False


def _runs(plan: Plan, row: Row, arm: Arm) -> bool:
    """Whether an arm runs on a row: its tree was given, and --skip leaves it at least one workload."""
    if arm.tree and arm.tree not in plan.trees:
        return False
    if not getattr(plan, "skips", None):
        return True
    wls = row_workloads(plan, row)
    return not skipped(plan, row, arm) and not (wls and all(skipped(plan, row, arm, wl) for wl in wls))


def row_workloads(plan: Plan, row: Row) -> list[tables.Workload]:
    """The run's workloads this row takes: its family's, limited to Row.workloads when the row names some."""
    return [wl for wl in plan.workloads if wl.family == row.family and (not row.workloads or wl.name in row.workloads)]


# Generous for an engine's first request at a size (JIT, autotune, torch.compile of a video model), short of the
# hour a video job may take before the client gives up on it.
FIRST_SIZE_TIMEOUT_S = 1800.0


def first_size_requests(plan: Plan, arm: Arm, row: Row, wl: tables.Workload, port: int, arm_dir: Path,
                        gpus: list[int], numa: int | None, request_kwargs: dict | None) -> dict:
    """One untimed request per size of the workload's set, after startup and before any cell: an engine's first
    request at a new size can pay minutes of JIT, autotune and VAE builds, which would otherwise land in the first
    cell's warmup. Each is the row's first-size cost at that size (client e2e seconds, or the error)."""
    set_dir = workload_set_dir(row, wl, arm_dir, arm.name)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    pinned = bool(request_kwargs and request_kwargs.get("height") and request_kwargs.get("width"))
    need = "image" if wl.request_type == "image_to_video" else None
    picks: dict = {}
    for it in manifest["items"]:
        if need and need not in (it.get("files") or {}):
            continue
        p = it.get("params") or {}
        size = (f"{request_kwargs['width']}x{request_kwargs['height']}" if pinned
                else f"{p.get('width')}x{p.get('height')}")
        picks.setdefault(size, it)
    out = {}
    for k, (size, it) in enumerate(picks.items()):
        d = arm_dir / "first_size" / wl.name / f"s{k}"
        sdir = d / "set"
        sdir.mkdir(parents=True, exist_ok=True)
        files = {m: str((set_dir / f).resolve()) for m, f in (it.get("files") or {}).items()}
        (sdir / "manifest.json").write_text(json.dumps({"name": f"first-size-{wl.name}",
                                                       "items": [{**it, "files": files}]}))
        one = dataclasses.replace(wl, min_requests=1, waves=1)
        cmd = runner_command(arm, row, one, 1, port, d, 0, request_kwargs=request_kwargs)
        cmd[cmd.index("--sample-set-dir") + 1] = str(sdir)
        if numa is not None:
            cmd = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}", *cmd]
        try:
            rc, _ = run_client(cmd, client_env(arm, row, gpus), d / "bench.log", timeout_s=FIRST_SIZE_TIMEOUT_S)
        except TimeoutError:
            # The server is most likely wedged on it, so the arm ends here rather than in its cells' timeouts.
            raise RuntimeError(f"the first {wl.name} request at {size} did not finish in {FIRST_SIZE_TIMEOUT_S:.0f} s"
                               f" (see {d})") from None
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").is_file() else {}
        per, errs = res.get("per_request") or [], res.get("errors") or []
        why = str(errs[0].get("error"))[:200] if errs else f"rc {rc}"
        out[size] = round(per[0]["jct_ms"] / 1000.0, 2) if rc == 0 and per else f"failed ({why}, see {d})"
    return out


def run_row(plan: Plan, row: Row, index: int, gpus: list[int], port: int) -> None:
    """Every arm of one row, back to back on the same GPUs."""
    arms = [a for a in catalog.arms_for(row, plan.arms) if _runs(plan, row, a)]
    names = {a.name for a in arms}
    lead = [a for a in arms if a.follows not in names]
    # Rotate the arm order per row so no arm always runs first (or last) on a warm GPU. An arm that follows
    # another runs right after it.
    k = index % len(lead) if lead else 0
    order = [x for a in lead[k:] + lead[:k] for x in [a, *(f for f in arms if f.follows == a.name)]]
    # Each arm gets its own port: right after a server stops, its port can sit in TIME_WAIT, and a server that
    # binds without SO_REUSEADDR (sglang-omni) then refuses it.
    for j, arm in enumerate(order):
        run_arm(plan, row, arm, gpus, port + j, "")


def run_rows(plan: Plan, rows: list[Row], gpus: list[int], parallel: int, port_base: int) -> None:
    """Run rows on the GPU pool, at most ``parallel`` at a time, each on as many GPUs as it needs.
    Wide and expensive rows start first so the tail is short."""
    free = list(gpus)
    cond = threading.Condition()
    running = [0]
    threads = []

    def one(row, index, mine, port):
        try:
            n = getattr(plan, "media_lanes", 0)
            with media_lane(n, plan.log) if n and row.family in ("image", "video") else contextlib.nullcontext():
                run_row(plan, row, index, mine, port)
        finally:
            with cond:
                free.extend(mine)
                running[0] -= 1
                cond.notify_all()

    order = sorted(rows, key=lambda r: (-r.gpus, -r.cost))
    for i, row in enumerate(order):
        if row.gpus > len(gpus):
            plan.log(f"{row.name}: needs {row.gpus} GPUs, the run has {len(gpus)}; skipped")
            continue
        with cond:
            while len(free) < row.gpus or running[0] >= parallel:
                cond.wait()
            mine = sorted(free)[: row.gpus]
            for g in mine:
                free.remove(g)
            running[0] += 1
        t = threading.Thread(target=one, args=(row, rows.index(row), mine, port_base + 10 * i), name=row.name)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def _merged(order, *lists) -> list:
    """Every name in ``lists`` once, in ``order`` where it has one."""
    names = [x for lst in lists for x in lst]
    out = [x for x in order if x in names]
    for x in names:
        if x not in out:
            out.append(x)
    return out


def resumed_meta(meta: dict, prev: dict, table: tables.Table) -> dict:
    """sweep.json for a resume of ``prev``. The header keeps the first placement and every row and arm of the
    table, whatever part this resume runs; the resume records its own placement (and, at the end, what it
    measured)."""
    out = dict(meta, started=prev.get("started", meta["started"]))
    resume = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": meta["node"].get("host"), "job": meta["job"],
              "gpus": meta["gpus"], "lanes": meta["lanes"], "numa": meta["numa"], "driver": meta["driver"]}
    out["resumed"] = [*prev.get("resumed", []), resume]
    for k in ("gpus", "lanes", "numa"):
        out[k] = prev.get(k, meta[k])
    out["rows"] = _merged(table.rows, prev.get("rows", []), meta["rows"])
    out["arms"] = _merged(table.arms, prev.get("arms", []), meta["arms"])
    out["workloads"] = _merged(table.workloads, prev.get("workloads", []), meta["workloads"])
    out["concurrency"] = {**prev.get("concurrency", {}), **meta["concurrency"]}
    return out


def wait_for_free_vram(gpus: list[int], log, timeout_s: float = 180) -> None:
    """Before the health gate: a server the previous table or prewarm just stopped takes a few seconds
    to hand its memory back, and the gate would count that as an occupied GPU."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        busy = {g: o.get("vram_used_mb") for g, o in gpu_occupancy(gpus).items()
                if (o.get("vram_used_mb") or 0) > VRAM_BUSY_MB}
        if not busy:
            return
        log(f"waiting for VRAM on {busy} to come back")
        time.sleep(10)


def refuse_mixed_chips(prev: dict, node: dict, run_dir: Path) -> None:
    """A resumed run continues on the chip it started on, so a table never holds two."""
    was, now = prev.get("node_type"), node.get("node_type")
    was_gpus, now_gpus = set((prev.get("gpu_names") or {}).values()), set((node.get("gpu_names") or {}).values())
    if was != now or (was_gpus and now_gpus and was_gpus != now_gpus):
        raise SystemExit(f"{run_dir} ran on {was} {sorted(map(str, was_gpus))}; this node is {now} "
                         f"{sorted(map(str, now_gpus))}. Start a new run id instead.")


def check_node(plan: Plan, node_type: str) -> dict:
    part = os.environ.get("SLURM_JOB_PARTITION", "")
    rec = host_record(plan.gpus)
    names = {g: (v or {}).get("market_name") for g, v in rec.get("gpus", {}).items()}
    info = {"partition": part, "host": socket.gethostname(), "gpu_names": names, "node_type": node_type}
    want = NODE_TYPES.get(node_type, node_type)
    wrong = {g: n for g, n in names.items() if not n or want.upper() not in str(n).upper()}
    if not plan.dry:
        if BENCH_PARTITIONS.get(part) != node_type:
            raise SystemExit(f"partition {part!r} is not a {node_type} benchmark partition; use --dry for functional "
                             "runs")
        if wrong or len(names) < len(plan.gpus):
            raise SystemExit(f"GPUs are not all {want}: {names}")
    elif wrong:
        info["warning"] = f"GPUs not reported as {want}: {wrong}"
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, choices=sorted(tables.TABLES))
    ap.add_argument("--node-type", required=True, choices=sorted(NODE_TYPES))
    ap.add_argument("--gpus", required=True, help="GPUs this run owns, e.g. 0,1,2,3")
    ap.add_argument("--worktree", default=str(REPO), help="mstar checkout the mstar arm serves")
    ap.add_argument("--arms", default=None, help="comma list (default: the table's)")
    ap.add_argument("--rows", default=None, help="comma list (default: the table's)")
    ap.add_argument("--workloads", default=None, help="comma list (default: the table's)")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--concurrency", default=None, help="comma list replacing every workload's (dry runs)")
    ap.add_argument("--lanes", type=int, default=1, help="rows that may run at once, each on its own GPUs")
    ap.add_argument("--media-lanes", type=int, default=MEDIA_LANES,
                    help="image and video rows that may run at once on the node, across sweeps (0: no cap)")
    ap.add_argument("--numa", action="store_true", help="bind each server to its GPU's NUMA node")
    ap.add_argument("--port-base", type=int, default=8400)
    ap.add_argument("--expect-gpus", type=int, default=None, help="node GPU count for the health check")
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true", help="keep finished arms and cells of an existing run-id")
    ap.add_argument("--dry", action="store_true", help="functional run on any node; the table says so")
    ap.add_argument("--no-correctness", action="store_true")
    ap.add_argument("--correctness-src", default=None,
                    help="checkout or git ref with benchmark/correctness (default: this tree)")
    ap.add_argument("--skip-health", action="store_true", help="dry runs only")
    ap.add_argument("--skip", action="append", default=[], metavar="ROW/ARM[/WORKLOAD]",
                    help="leave an arm out on matching rows (and workloads), fnmatch patterns: '*/vllm-omni/video-i2v'")
    ap.add_argument("--requests", type=int, default=None,
                    help="dry runs only: every cell sends this many requests, with no warmup")
    ap.add_argument("--tree", action="append", default=[], metavar="NAME=PATH",
                    help="a named mstar tree for the arms that serve one (base=TREE for the mstar-base arms)")
    args = ap.parse_args(argv)

    table = tables.TABLES[args.table]
    rows = [catalog.ROWS[r] for r in (args.rows.split(",") if args.rows else table.rows)]
    arms = args.arms.split(",") if args.arms else list(table.arms)
    wls = [tables.WORKLOADS[w] for w in (args.workloads.split(",") if args.workloads else table.workloads)]
    if args.concurrency:
        conc = tuple(int(x) for x in args.concurrency.split(","))
        wls = [dataclasses.replace(w, concurrency=conc) for w in wls]
    if args.requests:
        if not args.dry:
            raise SystemExit("--requests is for --dry runs only")
        wls = [dataclasses.replace(w, min_requests=args.requests, waves=0, warmup_waves_first=0,
                                   warmup_waves_next=0) for w in wls]
    gpus = gpu_list(args.gpus)
    if args.skip_health and not args.dry:
        raise SystemExit("--skip-health is for --dry runs only")
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    run_id = args.run_id or f"{datetime.datetime.now():%Y%m%dT%H%M%S}-{socket.gethostname().split('.')[0]}-{job}"
    run_dir = Path(args.run_root) / table.name / run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"{run_dir} exists; pass --resume to continue it")
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = Plan(table=table, rows=rows, arms=arms, workloads=wls, gpus=gpus, run_dir=run_dir,
                worktree=Path(args.worktree).resolve(), port_base=args.port_base,
                repeats=args.repeats or table.repeats, lanes=args.lanes, dry=args.dry,
                correctness=not args.no_correctness, numa=args.numa, resume=args.resume,
                expect_gpus=args.expect_gpus, trees=tree_list(args.tree), fixed_concurrency=bool(args.concurrency),
                skips=list(args.skip), media_lanes=args.media_lanes)
    if plan.skips:
        plan.log(f"left out by --skip: {', '.join(plan.skips)}")
    skipped = sorted({a for a in arms if catalog.ARMS[a].tree and catalog.ARMS[a].tree not in plan.trees})
    if skipped:
        plan.log(f"no --tree for {', '.join(skipped)}; those arms are skipped")
    node = check_node(plan, args.node_type)
    prev = json.loads((run_dir / "sweep.json").read_text()) if (run_dir / "sweep.json").is_file() else None
    if prev:
        refuse_mixed_chips(prev.get("node") or {}, node, run_dir)
        if bool(prev.get("dry")) != args.dry:
            raise SystemExit(f"{run_dir} is a {'dry' if prev.get('dry') else 'timed'} run; resume it the same way")
    if plan.correctness:
        from benchmark.amd import correctness
        plan.correctness_src = correctness.prepare_source(run_dir, args.correctness_src)
    meta = {"table": table.name, "note": table.note, "run_id": run_id, "dry": args.dry, "node": node,
            "job": job, "gpus": gpus, "rows": [r.name for r in rows], "arms": arms, "workloads": [w.name for w in wls],
            "concurrency": {w.name: list(w.concurrency) for w in wls},
            "repeats": plan.repeats, "spread_limit": table.spread_limit, "lanes": args.lanes,
            "media_lanes": args.media_lanes, "numa": args.numa,
            "worktree": str(plan.worktree), "trees": {k: str(v) for k, v in plan.trees.items()},
            "driver": str(REPO), "argv": sys.argv,
            "correctness_src": str(plan.correctness_src) if plan.correctness_src else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if prev:
        meta = resumed_meta(meta, prev, table)
    (run_dir / "sweep.json").write_text(json.dumps(meta, indent=1, default=str))
    plan.log(f"table {table.name} -> {run_dir}")

    if not args.skip_health:
        wait_for_free_vram(gpus, plan.log)
        # A resumed run keeps the first check and adds one per resume.
        name = f"check_resume{len(meta['resumed'])}" if meta.get("resumed") else "check_before"
        res = health.check(run_dir / "health", args.expect_gpus, gpus, name=name)
        plan.log(f"node health: {res['verdict']['status']} {'; '.join(res['verdict']['reasons'])[:400]}")
        if res["verdict"]["status"] == "FAIL":
            if not args.dry:
                (run_dir / "ABORTED").write_text("node health check failed:\n" + "\n".join(res["verdict"]["reasons"]))
                plan.log("aborted: node health FAIL")
                return 1
            plan.log("dry run: continuing past the health FAIL")
        health.snapshot(run_dir / "health" / "snapshot_table_before.txt")

    run_rows(plan, rows, gpus, args.lanes, args.port_base)

    if not args.skip_health:
        health.snapshot(run_dir / "health" / "snapshot_table_after.txt")
        health.check(run_dir / "health", args.expect_gpus, gpus, name="check_after")
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if meta.get("resumed"):
        meta["resumed"][-1].update(finished=meta["finished"], measured=plan.measured)
    (run_dir / "sweep.json").write_text(json.dumps(meta, indent=1, default=str))
    if plan.correctness:
        from benchmark.amd import correctness
        plan.log("scoring correctness")
        correctness.score_run(run_dir, gpu=gpus[0], log=plan.log)
    from benchmark.amd import table as table_writer
    table_writer.write(run_dir)
    plan.log(f"wrote {run_dir / 'table.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
