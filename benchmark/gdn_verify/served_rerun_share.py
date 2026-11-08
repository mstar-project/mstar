"""What share of a Qwen3.5 + MTP verify step the GDN closed-gate rerun takes,
on the checkpoint's real weights, with the node's own captured verify graphs
(RFC "Recurrent state").

For each model and batch size it builds the LLM node on the real Engine
(``Qwen3_5DenseModel`` with ``draft: {method: mtp}``, its weights from the
local Hugging Face cache), captures its decode buckets, prefills ``rows``
prompts and runs verify steps, synchronous, one batch of ``rows``. It does
that twice: as served, and with ``GDNManager.settle_verify_step`` a no-op (the
rerun left out of the captured graph; the shapes, and so the rest of the
step's work, are the same). Each arm's GPU time per verify step is the sum
of its kernels' device time under ``torch.profiler`` (kernels inside graph
replays included), over ``--steps`` steps after ``--warmup``. The rerun's
share is the difference over the served arm.

The verdict line says whether the share stays within 10% at rows <= 16.

    python -m benchmark.gdn_verify.served_rerun_share --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B \\
        --rows 1 4 16 --k 3 --out served_rerun.json

The driver is the CPU tests' (`test/modular/qwen3_5_cpu.py`), on the device.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RERUN_LIMIT_PCT, RULE_MAX_ROWS = 10.0, 16


def _engine(model, rows: int):
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES = sorted({1, rows})
    sub = model.get_submodule("LLM", device="cuda", autocast_dtype=torch.bfloat16)
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.max_num_pages = 4096
    eng = Engine(graph_runtime=None, autocast_dtype=torch.bfloat16)
    eng.load_model(
        {"LLM": sub}, specs, WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cuda", torch.cuda.current_device()),
        transfer_engine_info=None, kv_cache_type=torch.bfloat16, model=model,
    )
    eng.warmup()
    return eng


def _gpu_ms_per_step(eng, model, rows: int, args) -> float:
    sys.path.insert(0, str(REPO / "test" / "modular"))
    from qwen3_5_cpu import Driver

    gen = torch.Generator().manual_seed(0)
    prompts = {rid: torch.randint(1000, 20000, (args.prompt,), generator=gen).tolist() for rid in range(1, rows + 1)}
    driver = Driver(eng, model)
    for rid, prompt in prompts.items():
        # every step a full batch: no row stops before the measurement ends
        driver.add(rid, prompt, (args.warmup + args.steps) * (args.k + 1) + 8, ignore_eos=True)
    driver.prefill(list(prompts), prompts)
    live = list(prompts)
    for _ in range(args.warmup):
        driver.step(live)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(args.steps):
            driver.step(live)
        torch.cuda.synchronize()
    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    for rid in live:
        eng.remove_request(rid)
    return total_us / 1000 / args.steps


def one_arm(args, model_path: str, rows: int, arm: str) -> float:
    """One arm's GPU ms per verify step, in this process (`measure` runs each in its own)."""
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.linear_attn.gdn import GDNManager
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    sys.path.insert(0, str(REPO / "test" / "modular"))
    from qwen3_cpu import _StubTransfer

    # an engine outside a worker has no transfer engine for its KV pool (this process is the arm's own)
    manager_mod.KVTransferManager = _StubTransfer
    if arm == "no_rerun":
        # benchmark only: the captured graph leaves the rerun out (the state is then wrong,
        # which changes tokens, not the step's shapes)
        GDNManager.settle_verify_step = lambda self, result: None
    draft = {"method": "mtp", "num_draft_tokens": args.k, "mode": "synchronous"}
    model = Qwen3_5DenseModel(model_path_hf=model_path, draft=draft, prefill_token_buckets=[args.prompt * rows])
    return _gpu_ms_per_step(_engine(model, rows), model, rows, args)


def measure(args, model_path: str, rows: int) -> dict:
    """Both arms, each in a fresh process: no engine, compile cache or memory carries
    from one build to the next."""
    times = {}
    for arm in ("served", "no_rerun"):
        command = [
            sys.executable, "-m", "benchmark.gdn_verify.served_rerun_share", "--one", model_path, str(rows), arm,
            "--k", str(args.k), "--prompt", str(args.prompt), "--warmup", str(args.warmup), "--steps", str(args.steps),
        ]
        done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        if done.returncode:
            sys.stderr.write(done.stdout[-4000:] + done.stderr[-8000:])
            raise RuntimeError(f"{model_path} rows {rows} {arm}: the arm failed (output above)")
        times[arm] = json.loads(done.stdout.strip().splitlines()[-1])["ms"]
    rerun = times["served"] - times["no_rerun"]
    return {"model": model_path, "rows": rows, "k": args.k, "served_ms": times["served"],
            "no_rerun_ms": times["no_rerun"], "rerun_ms": rerun, "rerun_pct": 100 * rerun / times["served"]}


def verdict(results: list[dict]) -> str:
    ruled = [r for r in results if r["rows"] <= RULE_MAX_ROWS]
    if not ruled:
        return f"no measurement at rows <= {RULE_MAX_ROWS}: the rule is undecided"
    worst = max(ruled, key=lambda r: r["rerun_pct"])
    keep = worst["rerun_pct"] <= RERUN_LIMIT_PCT
    return (f"worst rerun at rows <= {RULE_MAX_ROWS}: {worst['rerun_pct']:.1f}% of the served verify step "
            f"({worst['model']}, rows {worst['rows']}): "
            + ("keep the rerun" if keep else f"over {RERUN_LIMIT_PCT:.0f}%"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-9B"])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--one", nargs=3, metavar=("MODEL", "ROWS", "ARM"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        model_path, rows, arm = args.one
        print(json.dumps({"ms": one_arm(args, model_path, int(rows), arm)}))
        return
    results = []
    print(f"{'model':>18} {'rows':>4} {'served_ms':>9} {'no_rerun_ms':>11} {'rerun_ms':>8} {'rerun%':>7}")
    for model in args.model:
        for rows in args.rows:
            r = measure(args, model, rows)
            results.append(r)
            print(f"{model[-18:]:>18} {rows:>4} {r['served_ms']:>9.3f} {r['no_rerun_ms']:>11.3f} "
                  f"{r['rerun_ms']:>8.3f} {r['rerun_pct']:>6.1f}%", flush=True)
            if args.out:
                Path(args.out + ".partial").write_text(json.dumps(results, indent=1))
    print(verdict(results))
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "results": results, "verdict": verdict(results)},
                                             indent=1, default=str))
        Path(args.out + ".partial").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
