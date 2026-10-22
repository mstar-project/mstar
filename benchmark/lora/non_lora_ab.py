"""Models served without LoRA, this branch against a base: does the LoRA-capable shared
linears' hook (and the rest of the LoRA stack) cost a deployment with no ``lora:`` block
anything? GPU; nothing here runs in CI.

Three models whose layers the change touches, no LoRA block anywhere:

* ``qwen3_5_0.8b``: column / row parallel linears, the fused delta-net projection, the
  vocab-parallel embedding; text out, closed loop at concurrency 32 (a small model is
  where CPU overhead shows).
* ``orpheus``: ``ParallelAttention`` / ``ParallelGatedMLP`` and the declare_step that
  merges ``lora_steps()``; speech out at concurrency 16.
* ``kokoro``: ``FusedColumnLinear`` inside piecewise-captured regions; speech out at 8.

Each (model, ref) is served from its own git worktree (``setup``), alternating the order
per round (ABBA), with ``MSTAR_PHASE_TIMING`` on so every run also writes the worker
phase tables (the profile pass). Results land per run under ``--out``; a run whose JSON
exists is skipped, so an interrupted sweep resumes. ``compare`` prints, per model, the
median request throughput of each ref over the rounds and the branch/base ratio, and
the per-phase means of ``iter_total``, ``worker.gpu_thread.exec``, ``engine.declare_step``
and ``worker.postprocess.event_sync`` side by side.

    python -m benchmark.lora.non_lora_ab --phase setup --work ~/ab --base 44ab46571 --branch lora/generic
    python -m benchmark.lora.non_lora_ab --phase run --work ~/ab --out ~/ab/results --rounds 3
    python -m benchmark.lora.non_lora_ab --phase compare --out ~/ab/results

Expected: ratios within run-to-run noise (about 2%; see the benchmarking skill), and
``engine.declare_step`` unchanged within noise. Compare against ``1976242c2`` (the LoRA
stack before the generic path) too, to separate this branch's hook from the stack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

MODELS = {
    "qwen3_5_0.8b": {
        "config": "configs/qwen3_5_0.8b.yaml",
        "client": ["--model", "qwen3.5", "--request-type", "text_to_text", "--max-concurrency", "32",
                   "--num-requests", "512", "--num-warmup", "5"],
    },
    "orpheus": {
        "config": "configs/orpheus.yaml",
        "client": ["--model", "orpheus", "--request-type", "text_to_speech", "--dataset", "seed_tts",
                   "--max-concurrency", "16", "--num-requests", "256", "--num-warmup", "5"],
    },
    "kokoro": {
        "config": "configs/kokoro.yaml",
        "client": ["--model", "kokoro", "--request-type", "text_to_speech", "--max-concurrency", "8",
                   "--num-requests", "256", "--num-warmup", "5"],
    },
}
PHASES = ("iter_total", "worker.gpu_thread.exec", "engine.declare_step", "worker.postprocess.event_sync")


def setup(args) -> None:
    repo = Path(__file__).resolve().parents[2]
    for name, ref in (("base", args.base), ("branch", args.branch)):
        tree = args.work / name
        if not tree.exists():
            subprocess.run(["git", "-C", str(repo), "worktree", "add", "-f", str(tree), ref], check=True)
        print(f"{name}: {tree} at {ref}")


def _run_one(args, model: str, ref: str, round_: int) -> None:
    from benchmark.eval.server import ManagedServer
    from benchmark.worker_phases.phases import parse_log, segment

    out = args.out / model / f"{ref}_r{round_}"
    if (out / "result.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    tree = args.work / ref
    spec = MODELS[model]
    env = {"MSTAR_PHASE_TIMING": str(args.period), "PYTHONPATH": str(tree)}
    log = out / "server.log"
    with ManagedServer(model, config=str(tree / spec["config"]), env=env, log_path=log, cwd=str(tree)) as server:
        bench = subprocess.run(
            [sys.executable, "-m", "benchmark.runner", "--url", server.url, "--profiling-type", "closed_loop",
             "--output-dir", str(out / "runner"), *spec["client"]],
            check=False, capture_output=True, text=True, cwd=str(tree),
            env={**os.environ, "PYTHONPATH": str(tree)},
        )
    (out / "bench.log").write_text(bench.stdout + bench.stderr)
    runner = json.loads((out / "runner" / "results.json").read_text())
    segments = segment(parse_log(log.read_text(errors="replace")))
    result = {
        "request_throughput": runner.get("request_throughput"),
        "failed": runner.get("failed"),
        "phases_text": _render(segments),
        "phase_means": _phase_means(log.read_text(errors="replace")),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"{model} {ref} r{round_}: {result['request_throughput']} req/s, failed {result['failed']}", flush=True)


def _render(segments) -> str:
    from benchmark.worker_phases.phases import render

    return render(segments)


def _phase_means(log: str) -> dict[str, float]:
    """Mean ms of each tracked phase in the rendered tables' steady segment (the one
    whose ``iter_total`` has the most samples)."""
    from benchmark.worker_phases.phases import parse_log, render, segment

    best: dict[str, float] = {}
    best_samples = -1
    for table in render(segment(parse_log(log))).split("==="):
        rows = {}
        for line in table.splitlines():
            m = re.match(r"^([\w.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$", line.strip())
            if m:
                rows[m.group(1)] = (float(m.group(2)), int(m.group(5)))
        samples = rows.get("iter_total", (0, -1))[1]
        if samples > best_samples:
            best_samples = samples
            best = {p: rows[p][0] for p in PHASES if p in rows}
    return best


def run(args) -> None:
    for round_ in range(args.rounds):
        order = ("base", "branch") if round_ % 2 == 0 else ("branch", "base")
        for model in args.models:
            for ref in order:
                _run_one(args, model, ref, round_)


def compare(args) -> int:
    for model in args.models:
        rows = {"base": [], "branch": []}
        phases = {"base": {}, "branch": {}}
        for path in sorted((args.out / model).glob("*/result.json")):
            ref = path.parent.name.split("_r")[0]
            data = json.loads(path.read_text())
            rows[ref].append(data["request_throughput"] or 0.0)
            for phase, ms in data["phase_means"].items():
                phases[ref].setdefault(phase, []).append(ms)
        if not rows["base"] or not rows["branch"]:
            print(f"{model}: incomplete")
            continue
        b, n = statistics.median(rows["base"]), statistics.median(rows["branch"])
        print(f"{model}: base {b:.3f} req/s {rows['base']}, branch {n:.3f} req/s {rows['branch']}, "
              f"ratio {n / b:.3f}")
        for phase in PHASES:
            pb, pn = phases["base"].get(phase), phases["branch"].get(phase)
            if pb and pn:
                print(f"    {phase:36s} base {statistics.median(pb):.3f} ms  branch {statistics.median(pn):.3f} ms")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("setup", "run", "compare"), required=True)
    p.add_argument("--work", type=Path, default=Path.home() / "non_lora_ab")
    p.add_argument("--out", type=Path, default=Path.home() / "non_lora_ab" / "results")
    p.add_argument("--base", default="44ab46571")
    p.add_argument("--branch", default="lora/generic")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--period", type=int, default=100, help="MSTAR_PHASE_TIMING iterations per flush")
    p.add_argument("--models", default=",".join(MODELS))
    args = p.parse_args()
    args.models = args.models.split(",")
    {"setup": setup, "run": run, "compare": compare}[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
