"""Where a Qwen3.5 + MTP verify step's GPU time goes, on the checkpoint's real
weights, with the node's own captured graphs.

Four arms, each in a fresh process, synchronous, one batch of ``rows``:

- ``served``: the verify step as served;
- ``no_rerun``: `GDNManager.settle_verify_step` a no-op (the rerun out of the graph);
- ``no_draft``: the MTP head stubbed out: ``propose`` repeats each row's last
  token, ``observe`` keeps the anchor row's hidden state (no head pass, no
  drafter LM head). The block is still ``k + 1`` tokens a row;
- ``plain``: the same model without drafting, one decode token a row.

Each arm's kernels are summed by device time under ``torch.profiler`` (kernels
inside graph replays included) over ``--steps`` steps, and binned by kernel
name. The drafter is ``served - no_draft`` and the rerun ``served - no_rerun``;
the bins split the rest (GEMMs serve every projection, the MLP and the LM head,
so they stay one bin). Kernel sums miss the gaps between a graph's nodes, so
each arm's step is also timed between CUDA events (``wall_ms``), outside the
profiler, and the splits are given both ways.

    python -m benchmark.gdn_verify.verify_step_profile --model Qwen/Qwen3.5-9B --rows 16 --out profile.json

Shares `served_rerun_share`'s engine and driver.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from benchmark.gdn_verify.served_rerun_share import REPO, _engine

ARMS = ("served", "no_rerun", "no_draft", "plain")
# first match wins
BINS = (
    # FlashInfer's sampling and norm kernels carry "flashinfer" and "cutlass" in their names
    ("sampler", r"flashinfer::sampling|argmax|softmax|sampl|topk|top_k|top_p|cumsum"),
    ("norm", r"rmsnorm|layernorm"),
    ("gdn", r"gdn|gated_delta|delta_rule|recurrent|causal_conv|conv1d|conv_update"),
    ("attention", r"fmha|flash::|flashattn|attn|attention|BatchDecode|BatchPrefill"),
    ("gemm", r"gemm|nvjet|xmma|cutlass|cublas|matmul|sm90_|sm100_|splitK"),
    ("copy", r"memcpy|memset|copy|index|gather|scatter|cat_|CatArray"),
    ("fused_elementwise", r"triton_|elementwise|vectorized|reduce|norm"),
)


def _bin(name: str) -> str:
    low = name.lower()
    for label, pattern in BINS:
        if re.search(pattern.lower(), low):
            return label
    return "other"


def _kernels(eng, model, rows: int, args) -> dict:
    sys.path.insert(0, str(REPO / "test" / "modular"))
    from qwen3_5_cpu import Driver

    gen = torch.Generator().manual_seed(0)
    prompts = {rid: torch.randint(1000, 20000, (args.prompt,), generator=gen).tolist() for rid in range(1, rows + 1)}
    driver = Driver(eng, model)
    per_step = 1 if model.draft is None else args.k + 1
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, (args.warmup + 2 * args.steps) * per_step + 8, ignore_eos=True)
    driver.prefill(list(prompts), prompts)
    live = list(prompts)
    for _ in range(args.warmup):
        driver.step(live)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        driver.step(live)
    end.record()
    torch.cuda.synchronize()
    wall_ms = start.elapsed_time(end) / args.steps
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            driver.step(live)
        torch.cuda.synchronize()
    table = {}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            table[e.key] = {"us": e.self_device_time_total / args.steps, "count": e.count / args.steps}
    return {"kernels": table, "wall_ms": wall_ms}


def one_arm(args, model_path: str, rows: int, arm: str) -> dict:
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.linear_attn.gdn import GDNManager
    from mstar.model.qwen3_5.components import mtp as mtp_mod
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    sys.path.insert(0, str(REPO / "test" / "modular"))
    from qwen3_cpu import _StubTransfer

    manager_mod.KVTransferManager = _StubTransfer
    if arm == "no_rerun":
        GDNManager.settle_verify_step = lambda self, result: None
    if arm == "no_draft":
        k = args.k
        mtp_mod.Qwen3_5MTPDrafter.propose = lambda self, inputs: (
            inputs.input_ids.reshape(-1, 1).expand(-1, k).long())
        mtp_mod.Qwen3_5MTPDrafter.observe = lambda self, block, result, hidden: (
            hidden.view(block.shape[0], block.shape[1], -1)[:, 0].to(mtp_mod.CARRY_DTYPE))
    draft = None if arm == "plain" else {"method": "mtp", "num_draft_tokens": args.k, "mode": "synchronous"}
    model = Qwen3_5DenseModel(model_path_hf=model_path, draft=draft, prefill_token_buckets=[args.prompt * rows])
    return _kernels(_engine(model, rows), model, rows, args)


def summarize(tables: dict[str, dict]) -> dict:
    out = {}
    for arm, arm_out in tables.items():
        bins: dict[str, float] = {}
        for name, row in arm_out["kernels"].items():
            bins[_bin(name)] = bins.get(_bin(name), 0.0) + row["us"]
        out[arm] = {"total_ms": sum(bins.values()) / 1000, "wall_ms": arm_out["wall_ms"],
                    "bins_ms": {b: v / 1000 for b, v in sorted(bins.items())}}
    split = {}
    for how in ("total_ms", "wall_ms"):
        served = out["served"][how]
        rerun = served - out["no_rerun"][how]
        split[how] = {
            "drafter (served - no_draft)": served - out["no_draft"][how],
            "gdn rerun (served - no_rerun)": rerun,
            "gdn rerun share of served": rerun / served,
            "spec / plain step": served / out["plain"][how],
        }
    out["split_ms"] = split
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--one", nargs=2, metavar=("ARM", "FILE"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        arm, path = args.one
        Path(path).write_text(json.dumps(one_arm(args, args.model, args.rows, arm)))
        return
    tables = {}
    for arm in ARMS:
        path = Path(args.out or "verify_step_profile.json").with_suffix(f".{arm}.json")
        command = [sys.executable, "-m", "benchmark.gdn_verify.verify_step_profile", "--one", arm, str(path),
                   "--model", args.model, "--rows", str(args.rows), "--k", str(args.k), "--prompt", str(args.prompt),
                   "--warmup", str(args.warmup), "--steps", str(args.steps)]
        done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        if done.returncode:
            sys.stderr.write(done.stdout[-4000:] + done.stderr[-8000:])
            raise RuntimeError(f"{arm}: the arm failed (output above)")
        tables[arm] = json.loads(path.read_text())
    summary = summarize(tables)
    for arm in ARMS:
        s = summary[arm]
        print(f"{arm:>9} kernels {s['total_ms']:7.3f} ms  wall {s['wall_ms']:7.3f} ms  "
              + "  ".join(f"{b} {v:.3f}" for b, v in s["bins_ms"].items()))
    for how, split in summary["split_ms"].items():
        for label, value in split.items():
            print(f"{how} {label}: {value:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "summary": summary, "tables": tables}, indent=1))


if __name__ == "__main__":
    main()
