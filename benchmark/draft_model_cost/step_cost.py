"""Draft-model step cost: Qwen3-0.6B as a drafter against Qwen3-8B's verify step (RFC #1186, "Phases"), on the
checkpoints' real weights, captured, bf16, synchronous, one batch of ``rows``.

Per cell (rows, k):

- target: one Qwen3-8B verify step, ``k + 1`` rows a request (the node verifying through a stand-in drafter that
  costs nothing, `common.NullDrafterDecl`);
- drafter (a): ``k`` chained Qwen3-0.6B passes over the full ``k + 1``-row block each, under one plan (PR 1133's
  shape): ``k`` times one 0.6B verify-shaped step, measured the same way as the target's;
- drafter (b): ``k`` chained passes of one query row each: ``k`` times one 0.6B decode step. Its ``k - 1`` plans past
  (a)'s one are counted here, and priced by `plan_cost.py`.

Device time per step is given two ways (`common.step_times`): summed kernel time, and CUDA-event wall time over
the steps. A pass's ratio is the drafter's total over the target's step. Bytes are per step: weights read once a
pass (bf16; the 0.6B ties its head to its embedding, read once as the head) and KV over the context.

    python -m benchmark.draft_model_cost.step_cost --rows 1 4 16 32 64 --k 3 4 7 --out step_cost.json

Hardware: one H100 80GB (``CUDA_VISIBLE_DEVICES`` 0 or 1). Each (model, step kind, k) builds its engine in a
fresh process and measures every batch size there, on fresh requests per batch size.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.draft_model_cost import common

MODULE = "benchmark.draft_model_cost.step_cost"


def one(arm: dict) -> dict:
    """One engine: ``arm["model"]``, verifying ``arm["k"]`` drafts a step (or decoding plainly, ``k`` None);
    every batch size in ``arm["rows"]``."""
    k = arm["k"]
    model = common.qwen3_model(arm["model"], k)
    eng = common.engine(model, arm["rows"])
    per_step = 1 if k is None else k + 1
    out = {"model": arm["model"], "k": k, "weight_bytes": common.weight_bytes(model), "rows": {}}
    for rows in arm["rows"]:
        drv = common.driver(eng, model)
        prompts = common.prompts(rows, arm["prompt"])
        for rid, prompt in prompts.items():
            drv.add(rid, prompt, (arm["warmup"] + 2 * arm["steps"]) * per_step + 8)
        drv.prefill(list(prompts), prompts)
        times = common.step_times(drv, list(prompts), arm["warmup"], arm["steps"])
        # the context a measured step reads, at the measurement's midpoint
        context = arm["prompt"] + (arm["warmup"] + arm["steps"]) * per_step
        times["kv_bytes"] = common.kv_bytes(model, rows, context)
        out["rows"][rows] = times
        for rid in prompts:
            eng.remove_request(rid)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 7])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--target", default=common.TARGET)
    ap.add_argument("--draft", default=common.DRAFT)
    ap.add_argument("--out", type=Path, default=Path("step_cost.json"))
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(json.loads(args.one))))
        return
    base = {"rows": args.rows, "prompt": args.prompt, "warmup": args.warmup, "steps": args.steps}
    decode = common.run_arm(MODULE, {**base, "model": args.draft, "k": None})
    cells = []
    for k in args.k:
        target = common.run_arm(MODULE, {**base, "model": args.target, "k": k})
        block = common.run_arm(MODULE, {**base, "model": args.draft, "k": k})
        for rows in args.rows:
            t, a, b = (arm["rows"][str(rows)] for arm in (target, block, decode))
            cell = {"rows": rows, "k": k}
            for how in ("kernel_ms", "wall_ms"):
                cell[f"target_{how}"] = t[how]
                cell[f"drafter_a_{how}"] = k * a[how]
                cell[f"drafter_b_{how}"] = k * b[how]
                cell[f"ratio_a_{how}"] = k * a[how] / t[how]
                cell[f"ratio_b_{how}"] = k * b[how] / t[how]
            # plans a step: the target's one; (a) one for its k passes; (b) one a pass
            cell["drafter_a_plans"], cell["drafter_b_plans"] = 1, k
            cell["target_bytes"] = target["weight_bytes"] + t["kv_bytes"]
            cell["drafter_a_bytes"] = k * (block["weight_bytes"] + a["kv_bytes"])
            cell["drafter_b_bytes"] = k * (decode["weight_bytes"] + b["kv_bytes"])
            cell["weight_read_ratio"] = k * block["weight_bytes"] / target["weight_bytes"]
            cells.append(cell)
            print(f"rows {rows:3d} k {k}: target {t['kernel_ms']:.3f} ms; drafter (a) "
                  f"{cell['drafter_a_kernel_ms']:.3f} ms ({cell['ratio_a_kernel_ms']:.2f}x), (b) "
                  f"{cell['drafter_b_kernel_ms']:.3f} ms ({cell['ratio_b_kernel_ms']:.2f}x, {k - 1} extra plans); "
                  f"weights read "
                  f"{cell['weight_read_ratio']:.1%} of the target's", flush=True)
    args.out.write_text(json.dumps({"target": args.target, "draft": args.draft, "prompt": args.prompt,
                                    "cells": cells}, indent=1))
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
