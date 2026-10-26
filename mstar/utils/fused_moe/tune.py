"""Tune fused-MoE Triton launch configs for one expert shape on the local GPU.

Writes a JSON file that :func:`mstar.utils.fused_moe.kernels.get_config`
picks up for matching ``(E, N, K)`` on the same device name: per batch size,
the gate/up GEMM's config and a ``"down"`` block with the down GEMM's (they
share BLOCK_SIZE_M). Each launch is timed inside a CUDA graph with a
different routing per launch, so expert weights come from HBM rather than
L2, as they do in a real decode step. On AMD GPUs the best few tiles of each
GEMM are timed again across Triton's HIP compiler options (waves_per_eu,
matrix_instr_nonkdim, kpack).

    python -m mstar.utils.fused_moe.tune --experts 128 --inter 1024 --hidden 4096 --top-k 8
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch
import triton.language as tl

from mstar.platforms import current_platform
from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import config_path, device_config_name, invoke_fused_moe_kernel

BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512]
AMD_OPTIONS = {"waves_per_eu": [0, 1, 2, 4], "matrix_instr_nonkdim": [16, 32], "kpack": [1, 2]}
AMD_SWEPT_TILES = 3  # per GEMM and BLOCK_SIZE_M


def _candidates(m: int, amd: bool):
    block_m = [16, 32, 64] if m * 8 >= 256 else [16]
    # AMD recommends two stages for Triton GEMMs on CDNA3 (more can run out of
    # LDS), and small tiles often run best on one or two of its 64-wide warps.
    num_warps, num_stages = ([1, 2, 4, 8], [2]) if amd else ([4, 8], [2, 3, 4, 5])
    for bm, bn, bk, warps, stages in itertools.product(
        block_m, [32, 64, 128, 256], [64, 128, 256], num_warps, num_stages,
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


def _amd_variants(cfg: dict):
    for waves, nonk, kpack in itertools.product(*AMD_OPTIONS.values()):
        if nonk <= min(cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]):
            yield {**cfg, "waves_per_eu": waves, "matrix_instr_nonkdim": nonk, "kpack": kpack}


def _tune_batch(m: int, w1: torch.Tensor, w2: torch.Tensor, top_k: int, rotations: int) -> dict:
    experts, two_inter, hidden = w1.shape
    device, dtype = w1.device, w1.dtype
    x = torch.randn(m, hidden, device=device, dtype=dtype)
    routings = [
        torch.stack([torch.randperm(experts, device=device)[:top_k] for _ in range(m)]).to(torch.int32)
        for _ in range(rotations)
    ]
    weights = torch.full((m, top_k), 1.0 / top_k, device=device, dtype=dtype)
    operands = {  # A, B, C and top_k of each GEMM
        "up": (x, w1, torch.empty(m * top_k, two_inter, device=device, dtype=dtype), top_k),
        "down": (torch.randn(m * top_k, two_inter // 2, device=device, dtype=dtype), w2,
                 torch.empty(m * top_k, hidden, device=device, dtype=dtype), 1),
    }
    aligned = {}
    timings = {"up": {}, "down": {}}  # GEMM -> BLOCK_SIZE_M -> [(us, config)]

    def measure(gemm: str, cfg: dict) -> None:
        bm = cfg["BLOCK_SIZE_M"]
        if bm not in aligned:
            aligned[bm] = [moe_align_block_size(ids, bm, experts) for ids in routings]
        a, b, c, k = operands[gemm]

        def launch():
            for ids, (sorted_ids, expert_ids, padded) in zip(routings, aligned[bm], strict=True):
                invoke_fused_moe_kernel(
                    A=a, B=b, C=c, topk_weights=weights, topk_ids=ids,
                    sorted_token_ids=sorted_ids, expert_ids=expert_ids,
                    num_tokens_post_padded=padded, mul_routed_weight=gemm == "down",
                    top_k=k, config=cfg, compute_type=tl.bfloat16,
                )

        try:
            us = _time_graph(launch, repeats=10) / rotations
        except Exception:  # noqa: BLE001 -- configs that fail to compile or exceed smem
            return
        timings[gemm].setdefault(bm, []).append((us, cfg))

    amd = current_platform.is_rocm()
    for cfg in _candidates(m, amd):
        for gemm in operands:
            measure(gemm, cfg)
    if amd:
        for gemm, by_bm in timings.items():
            for runs in list(by_bm.values()):
                for _, cfg in sorted(runs, key=lambda run: run[0])[:AMD_SWEPT_TILES]:
                    for variant in _amd_variants(cfg):
                        measure(gemm, variant)
    best = {gemm: {bm: min(runs, key=lambda run: run[0]) for bm, runs in by_bm.items()}
            for gemm, by_bm in timings.items()}
    choice = min(best["up"].keys() & best["down"].keys(), key=lambda bm: best["up"][bm][0] + best["down"][bm][0])
    (up_us, up), (down_us, down) = best["up"][choice], best["down"][choice]
    entry = {**up, "down": {k: v for k, v in down.items() if k != "BLOCK_SIZE_M"}}
    print(f"M={m}: up {up_us:.1f}us down {down_us:.1f}us {entry}", flush=True)
    return entry


def tune(experts: int, inter: int, hidden: int, top_k: int, batch_sizes=BATCH_SIZES, rotations: int = 16) -> dict:
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    return {str(m): _tune_batch(m, w1, w2, top_k, rotations) for m in batch_sizes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--inter", type=int, required=True, help="per-rank expert intermediate size")
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    args = parser.parse_args()
    results = tune(args.experts, args.inter, args.hidden, args.top_k, args.batch_sizes)
    path = config_path(args.experts, 2 * args.inter, args.hidden, device_config_name())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
