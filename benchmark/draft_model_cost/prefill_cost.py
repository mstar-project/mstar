"""The TTFT cost of a draft model's prefill (RFC #1186, "Phases"): Qwen3-0.6B's prompt prefill
against Qwen3-8B's, batch 1, at 1k, 8k and 32k prompt tokens, on the checkpoints' real weights, bf16, and the ratio.

A draft model prefills the prompt too, before the first verify step, so its prefill adds to time to first token
unless it overlaps the target's. Both run eager here: a prompt of 1k tokens and up is compute-bound, and the
prefill capture ladder stops at 8k. Each length runs ``--warmup`` prompts, then ``--reps`` timed ones, each a fresh
request; device time per prefill two ways, as `common.step_times` gives a decode step's.

    python -m benchmark.draft_model_cost.prefill_cost --tokens 1024 8192 32768 --out prefill_cost.json

Hardware: one H100 80GB (``CUDA_VISIBLE_DEVICES`` 0 or 1). Each model runs in its own fresh process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmark.draft_model_cost import common

MODULE = "benchmark.draft_model_cost.prefill_cost"


def one(arm: dict) -> dict:
    model = common.qwen3_model(arm["model"], None)
    eng = common.engine(model, [1], max_num_pages=-(-(max(arm["tokens"]) + 64) // 128) + 8, captured=False)
    drv = common.driver(eng, model)
    out = {"model": arm["model"], "tokens": {}}
    rid = 0
    for n in arm["tokens"]:
        def prefill(n=n):
            nonlocal rid
            rid += 1
            prompt = common.prompts(1, n, seed=rid)[1]
            drv.add(rid, prompt, 2)
            drv.prefill([rid], {rid: prompt})
            return rid

        for _ in range(arm["warmup"]):
            eng.remove_request(prefill())
        torch.cuda.synchronize()
        walls = []
        for _ in range(arm["reps"]):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            done = prefill()
            end.record()
            torch.cuda.synchronize()
            walls.append(start.elapsed_time(end))
            eng.remove_request(done)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(arm["reps"]):
                eng.remove_request(prefill())
            torch.cuda.synchronize()
        kernels = sum(e.self_device_time_total for e in prof.key_averages()) / 1000 / arm["reps"]
        out["tokens"][n] = {"wall_ms": sorted(walls)[len(walls) // 2], "kernel_ms": kernels}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1024, 8192, 32768])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target", default=common.TARGET)
    ap.add_argument("--draft", default=common.DRAFT)
    ap.add_argument("--out", type=Path, default=Path("prefill_cost.json"))
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(json.loads(args.one))))
        return
    base = {"tokens": args.tokens, "warmup": args.warmup, "reps": args.reps}
    target = common.run_arm(MODULE, {**base, "model": args.target})
    draft = common.run_arm(MODULE, {**base, "model": args.draft})
    rows = []
    for n in args.tokens:
        t, d = target["tokens"][str(n)], draft["tokens"][str(n)]
        row = {"tokens": n, **{f"target_{h}": t[h] for h in t}, **{f"draft_{h}": d[h] for h in d},
               "ratio_wall": d["wall_ms"] / t["wall_ms"], "ratio_kernel": d["kernel_ms"] / t["kernel_ms"]}
        rows.append(row)
        print(f"{n:6d} tokens: target {t['wall_ms']:.1f} ms, draft {d['wall_ms']:.1f} ms "
              f"({row['ratio_wall']:.1%} of the target's; kernels {row['ratio_kernel']:.1%})", flush=True)
    args.out.write_text(json.dumps({"target": args.target, "draft": args.draft, "rows": rows}, indent=1))
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
