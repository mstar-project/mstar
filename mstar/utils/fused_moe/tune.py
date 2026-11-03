"""Tune fused-MoE Triton tile configs for one expert shape on the local GPU.

Writes a JSON file that :func:`mstar.utils.fused_moe.kernels.get_config`
picks up for matching ``(E, N, K)`` on the same device name. Each launch is
timed inside a CUDA graph with a different routing per launch, so expert
weights come from HBM rather than L2, as they do in a real decode step.

    python -m mstar.utils.fused_moe.tune --experts 128 --inter 1024 --hidden 4096 --top-k 8
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch
import triton.language as tl

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import config_path, invoke_fused_moe_kernel

BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512]


def _candidates(m: int):
    block_m = [16, 32, 64] if m * 8 >= 256 else [16]
    for bm, bn, bk, warps, stages in itertools.product(
        block_m, [32, 64, 128, 256], [64, 128, 256], [4, 8], [2, 3, 4, 5],
    ):
        if bn * bk > 256 * 128 or (bm == 64 and bn == 256 and bk == 256):
            continue
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
               "GROUP_SIZE_M": 1 if m <= 64 else 8, "num_warps": warps, "num_stages": stages}


def _time_graph(launch, repeats: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats * 1000  # us per graph


def tune(experts: int, inter: int, hidden: int, top_k: int, rotations: int = 16) -> dict:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    w1 = torch.randn(experts, 2 * inter, hidden, device=device, dtype=dtype) * 0.02
    w2 = torch.randn(experts, hidden, inter, device=device, dtype=dtype) * 0.02
    results = {}
    for m in BATCH_SIZES:
        x = torch.randn(m, hidden, device=device, dtype=dtype)
        routings = [
            torch.stack([torch.randperm(experts, device=device)[:top_k] for _ in range(m)]).to(torch.int32)
            for _ in range(rotations)
        ]
        weights = torch.full((m, top_k), 1.0 / top_k, device=device, dtype=dtype)
        cache1 = torch.empty(m * top_k, 2 * inter, device=device, dtype=dtype)
        cache2 = torch.randn(m * top_k, inter, device=device, dtype=dtype)
        cache3 = torch.empty(m * top_k, hidden, device=device, dtype=dtype)
        aligned = {}
        best = {"up": {}, "down": {}}
        for cfg in _candidates(m):
            bm = cfg["BLOCK_SIZE_M"]
            if bm not in aligned:
                aligned[bm] = [moe_align_block_size(ids, bm, experts) for ids in routings]
            for gemm, (a, b, c, k) in {
                "up": (x, w1, cache1, top_k), "down": (cache2, w2, cache3, 1),
            }.items():
                def launch(a=a, b=b, c=c, k=k, cfg=cfg, gemm=gemm, weights=weights,
                           batches=list(zip(routings, aligned[bm], strict=True))):
                    for ids, (sorted_ids, expert_ids, padded) in batches:
                        invoke_fused_moe_kernel(
                            A=a, B=b, C=c, topk_weights=weights, topk_ids=ids,
                            sorted_token_ids=sorted_ids, expert_ids=expert_ids,
                            num_tokens_post_padded=padded, mul_routed_weight=gemm == "down",
                            top_k=k, config=cfg, compute_type=tl.bfloat16,
                        )
                try:
                    us = _time_graph(launch, repeats=10) / rotations
                except Exception:  # noqa: BLE001 -- configs that fail to compile or exceed smem
                    continue
                if us < best[gemm].get((bm, "us"), float("inf")):
                    best[gemm][(bm, "us")] = us
                    best[gemm][bm] = cfg
        choice = min(
            {bm for bm in aligned if bm in best["up"] and bm in best["down"]},
            key=lambda bm: best["up"][(bm, "us")] + best["down"][(bm, "us")],
        )
        up, down = best["up"][choice], best["down"][choice]
        results[str(m)] = {**up, "down": {k: v for k, v in down.items() if k != "BLOCK_SIZE_M"}}
        print(f"M={m}: up {best['up'][(choice, 'us')]:.1f}us down {best['down'][(choice, 'us')]:.1f}us "
              f"{results[str(m)]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--inter", type=int, required=True, help="per-rank expert intermediate size")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    args = parser.parse_args()
    results = tune(args.experts, args.inter, args.hidden, args.top_k)
    path = config_path(args.experts, 2 * args.inter, args.hidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
