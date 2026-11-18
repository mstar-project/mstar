"""The host cost of planning a second attention binding a step (RFC #1186, "Phases"): a draft
model attends over the target's KV pool through an attention binding of its own, planned every step beside the
target's.

Two measurements:

- engine: each attention resource's ``plan`` host time a verify step, as the runner calls it, on Qwen3-8B with
  one binding (the target's, through `common.NullDrafterDecl`) against two (the DSpark node, whose
  drafter attends through its own binding over the pool), in synchronous mode (FlashInfer's planned verify
  attention) and asynchronous mode (FA3's verify kernel on resident tables, whose step needs no host plan). The
  second binding's cost a step is the DSpark binding's ``plan`` time. DSpark's binding has the target's query heads;
- flashinfer: FlashInfer's own ``BatchPrefillWithPagedKVCacheWrapper.plan`` host time for ``k + 1`` query rows a
  request over the pool's pages, for the target's geometry (32 query, 8 KV heads, head dim 128) and Qwen3-0.6B's
  (16 query, 8 KV, 128). M* couldn't plan a binding with its own query heads over the target's pool before layer sets
  of their own geometry, so this is how that geometry is priced; one binding against two is the sum.

worker_phases gives per-phase times, not per binding, so it doesn't answer this directly; the engine arm's
two-binding total can be checked against it.

    python -m benchmark.draft_model_cost.plan_cost --rows 1 4 16 32 64 --k 3 4 7 --out plan_cost.json

Hardware: one H100 80GB (``CUDA_VISIBLE_DEVICES`` 0 or 1); host times on that node's CPU. Each engine arm is a
fresh process.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import torch

from benchmark.draft_model_cost import common

MODULE = "benchmark.draft_model_cost.plan_cost"
DSPARK = "deepseek-ai/dspark_qwen3_8b_block7"


def _time_plans(eng) -> dict[str, list[float]]:
    """Wrap every attention resource's ``plan``: its host seconds, a list a resource key."""
    from mstar.engine.resources.attn.base import AttentionManager

    spent: dict[str, list[float]] = {}
    for key, resource in eng._resources.items():
        if not isinstance(resource, AttentionManager):
            continue
        real = resource.plan

        def plan(step, ctx, _real=real, _key=key):
            t0 = time.perf_counter()
            try:
                return _real(step, ctx)
            finally:
                spent.setdefault(_key, []).append(time.perf_counter() - t0)

        resource.plan = plan
    return spent


def one(arm: dict) -> dict:
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    k, mode = arm["k"], arm["mode"]
    if arm["drafter"] == "dspark":
        draft = {"method": "dspark", "model": DSPARK, "num_draft_tokens": k, "mode": mode}
        model = Qwen3Model(model_path_hf=common.TARGET, draft=draft)
    else:
        model = common.qwen3_model(common.TARGET, k, mode=mode)
    eng = common.engine(model, arm["rows"])
    out = {"drafter": arm["drafter"], "mode": mode, "k": k, "rows": {}}
    for rows in arm["rows"]:
        drv = common.driver(eng, model)
        prompts = common.prompts(rows, arm["prompt"])
        for rid, prompt in prompts.items():
            drv.add(rid, prompt, (arm["warmup"] + arm["steps"]) * (k + 1) + 8)
        drv.prefill(list(prompts), prompts)
        live = list(prompts)
        for _ in range(arm["warmup"]):
            drv.step(live)
        spent = _time_plans(eng)
        for _ in range(arm["steps"]):
            drv.step(live)
        out["rows"][rows] = {key: 1000 * sum(v) / arm["steps"] for key, v in spent.items()}
        # unwrap for the next batch size
        for key in spent:
            del eng._resources[key].plan
        for rid in prompts:
            eng.remove_request(rid)
    return out


def flashinfer_plans(rows_list, ks, context: int, page: int, reps: int) -> list[dict]:
    import flashinfer

    workspace = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    out = []
    for qo, kv, dim, name in ((32, 8, 128, "target"), (16, 8, 128, "draft")):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        for k in ks:
            for rows in rows_list:
                span = k + 1
                pages = -(-(context + span) // page)
                qo_indptr = torch.arange(0, (rows + 1) * span, span, dtype=torch.int32)
                kv_indptr = torch.arange(0, (rows + 1) * pages, pages, dtype=torch.int32)
                kv_indices = torch.arange(rows * pages, dtype=torch.int32, device="cuda")
                last = torch.full((rows,), context + span - (pages - 1) * page, dtype=torch.int32)

                plan = functools.partial(
                    wrapper.plan, qo_indptr, kv_indptr, kv_indices, last, qo, kv, dim, page, causal=True,
                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                )

                for _ in range(10):
                    plan()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    plan()
                ms = 1000 * (time.perf_counter() - t0) / reps
                out.append({"binding": name, "qo_heads": qo, "kv_heads": kv, "head_dim": dim, "k": k, "rows": rows,
                            "plan_ms": ms})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 7])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--page", type=int, default=128)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("plan_cost.json"))
    args = ap.parse_args()
    if args.one:
        print(json.dumps(one(json.loads(args.one))))
        return
    base = {"rows": args.rows, "prompt": args.prompt, "warmup": args.warmup, "steps": args.steps}
    engine_arms = []
    for k in args.k:
        for mode in ("synchronous", "asynchronous"):
            for drafter in ("null", "dspark"):
                arm = common.run_arm(MODULE, {**base, "k": k, "mode": mode, "drafter": drafter})
                engine_arms.append(arm)
                for rows, keys in arm["rows"].items():
                    print(f"engine k {k} {mode:12s} {drafter:6s} rows {rows:>3}: plan ms a step "
                          + ", ".join(f"{key} {ms:.3f}" for key, ms in sorted(keys.items())), flush=True)
    raw = flashinfer_plans(args.rows, args.k, args.prompt, args.page, args.reps)
    for row in raw:
        print(f"flashinfer {row['binding']:6s} ({row['qo_heads']}/{row['kv_heads']}/{row['head_dim']}) k {row['k']} "
              f"rows {row['rows']:3d}: plan {row['plan_ms']:.3f} ms", flush=True)
    args.out.write_text(json.dumps({"engine": engine_arms, "flashinfer": raw}, indent=1))
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
