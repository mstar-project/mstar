"""Qwen3.5 serving cost of LoRA, eager and captured, through ``benchmark/worker_phases``.

GPU only. One server per (LoRA mode, graphs, pass), each driven at concurrency 1, 32 and
64 by ``benchmark.lora.load``:

* LoRA mode ``off`` (the stock config), ``base`` (a ``resources: lora:`` block, every
  request on the base model) or ``adapters`` (every request on an adapter, round-robin
  over three, all resident).
* graphs ``captured`` (as served) or ``eager`` (``benchmark/lora/eager_site`` on
  ``PYTHONPATH``: no CUDA graphs, no ``torch.compile`` for the LLM node).
* pass ``throughput`` (no instrumentation) and ``profile`` (``MSTAR_PHASE_TIMING``), as
  separate servers: instrumentation costs throughput, so the two are never mixed.

Each run writes under ``--out/<mode>-<graphs>-<pass>/``: the server log, one JSON per
concurrency, and for the profile pass the worker_phases tables. A finished run is
skipped when rerun, so a lost box resumes where it stopped; copy ``--out`` off the box
after each run. ``summary`` (also printed at the end) tabulates every finished run.

    python -m benchmark.lora.phase_plan --out /tmp/lora_phases
    python -m benchmark.lora.phase_plan --out /tmp/lora_phases --only adapters-captured
    python -m benchmark.lora.phase_plan --out /tmp/lora_phases --summary
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from benchmark.lora.load import closed_loop
from benchmark.lora.qwen3_5_parity import DEFAULT_ADAPTERS, _adapter_dir

REPO = Path(__file__).resolve().parents[2]
EAGER_SITE = Path(__file__).resolve().parent / "eager_site"
MODES = ("off", "base", "adapters")
GRAPHS = ("captured", "eager")
PASSES = ("throughput", "profile")
# requests per concurrency: enough steady state at each batch size
REQUESTS = {1: 32, 32: 256, 64: 512}
# the phases the summary quotes (mean ms, in the steady-state segment of each concurrency)
PHASES = (
    "iter_total", "worker.postprocess.event_sync", "await_gpu", "worker.gpu_thread.exec",
    "engine.plan.fresh", "engine.admit", "res.plan.lora", "res.admit.lora",
)


def _config(model: str, workdir: Path, mode: str, adapters: dict[str, str]) -> Path:
    config = yaml.safe_load((REPO / "configs" / f"{model}.yaml").read_text())
    # text-only load; the vision warmup is not what is measured
    config.pop("warmup_requests", None)
    if mode != "off":
        config.setdefault("resources", {})["lora"] = {
            "max_slots": len(adapters), "max_rank": 16,
            "adapters": {
                name: str(_adapter_dir(spec) / "adapter_model.safetensors")
                for name, spec in adapters.items()
            },
        }
    path = workdir / f"{model}_{mode}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run_one(args, mode: str, graphs: str, pass_: str) -> None:
    from benchmark.eval.server import ManagedServer

    cell = args.out / f"{mode}-{graphs}-{pass_}"
    if (cell / "done.json").exists():
        print(f"{cell.name}: done, skipped", flush=True)
        return
    cell.mkdir(parents=True, exist_ok=True)
    env = {}
    if pass_ == "profile":
        env["MSTAR_PHASE_TIMING"] = str(args.period)
    if graphs == "eager":
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(EAGER_SITE), str(REPO), os.environ.get("PYTHONPATH")) if p
        )
    models = list(args.adapters) if mode == "adapters" else [args.model]
    with tempfile.TemporaryDirectory(prefix="lora-phases-") as tmp:
        config = _config(args.model, Path(tmp), mode, args.adapters)
        server = ManagedServer(args.model, config=str(config), env=env, log_path=cell / "server.log")
        results = {}
        with server:
            for c in args.concurrency:
                stats = closed_loop(
                    server.url, models, c, REQUESTS.get(c, 8 * c), args.max_tokens, warmup=c,
                )
                (cell / f"load_c{c}.json").write_text(json.dumps(stats, indent=1))
                results[c] = stats
                print(f"{cell.name} c={c}: {stats['output_tok_per_s']:.0f} tok/s, "
                      f"p50 {stats['latency_p50_s']:.2f} s, {stats['failed']} failed", flush=True)
    log = (cell / "server.log").read_text(errors="replace")
    if graphs == "eager" and "torch.compile disabled for LLM" not in log:
        raise SystemExit(f"{cell.name}: the eager patch did not take (see {cell / 'server.log'})")
    if pass_ == "profile":
        from benchmark.worker_phases.phases import parse_log, render, segment

        (cell / "phases.txt").write_text(render(segment(parse_log(log))))
    (cell / "done.json").write_text(json.dumps({str(c): r for c, r in results.items()}))


def summary(args) -> str:
    from benchmark.worker_phases.phases import parse_log, segment

    lines = [f"{'run':<24} {'c':>3} {'tok/s':>8} {'p50 s':>7}  " + "  ".join(PHASES)]
    for mode in MODES:
        for graphs in GRAPHS:
            tp = args.out / f"{mode}-{graphs}-throughput"
            prof = args.out / f"{mode}-{graphs}-profile"
            segs = []
            if (prof / "done.json").exists():
                segs = segment(parse_log((prof / "server.log").read_text(errors="replace")))
            for c in args.concurrency:
                path = tp / f"load_c{c}.json"
                if not path.exists():
                    continue
                stats = json.loads(path.read_text())
                phases = ""
                if segs:
                    # the longest segment near this concurrency is its steady state
                    near = [s for s in segs if abs(s.mean_bs - c) <= max(1.0, 0.2 * c)]
                    rows = max(near, key=lambda s: len(s.records)).summary() if near else {}
                    phases = "  ".join(
                        f"{rows[p]['mean_ms']:.3f}" if p in rows else "-" for p in PHASES
                    )
                lines.append(
                    f"{mode + '-' + graphs:<24} {c:>3} {stats['output_tok_per_s']:>8.0f} "
                    f"{stats['latency_p50_s']:>7.2f}  {phases}"
                )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3_5_0.8b")
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=REPO_OR_DIR")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 32, 64])
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--period", type=int, default=100, help="MSTAR_PHASE_TIMING iterations per record")
    p.add_argument("--only", nargs="*", default=None, metavar="MODE-GRAPHS",
                   help="e.g. adapters-captured; default every combination")
    p.add_argument("--summary", action="store_true", help="only print the summary of finished runs")
    args = p.parse_args()
    args.adapters = dict(a.split("=", 1) for a in args.adapter) or DEFAULT_ADAPTERS.get(args.model)
    if not args.adapters:
        p.error(f"no default adapters for {args.model}; pass --adapter name=repo")
    if not args.summary:
        for mode in MODES:
            for graphs in GRAPHS:
                if args.only and f"{mode}-{graphs}" not in args.only:
                    continue
                for pass_ in PASSES:
                    run_one(args, mode, graphs, pass_)
    text = summary(args)
    (args.out / "summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
