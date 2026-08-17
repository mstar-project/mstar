"""End-to-end A/B of one full ``fused_experts`` dispatch.

``bench_fused_moe.py`` times the two grouped GEMMs in isolation with the
alignment memoised outside the timed region, which is what you want when
tuning tiles.  This one times everything a real dispatch does -- align,
gate+up GEMM, SwiGLU, down GEMM, top-k sum-reduce -- so the alignment op and
the launch overhead land where they actually land.

Three configurations:

``legacy``  the state before this branch, reproduced here so the comparison
            does not depend on checking out an old revision: torch-fallback
            alignment, the worst-case launch grid, and one heuristic config
            shared by both GEMMs.
``tuned``   what ``fused_experts`` does today.
``graph``   the same, captured into a CUDA graph and replayed.

``legacy`` has no graph number by construction. The torch alignment fallback
sizes a ``repeat_interleave`` from a device tensor, which forces a
device-to-host sync, so capture fails -- that is the point of the comparison,
not an omission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstar.utils.fused_moe import fused_experts  # noqa: E402
from mstar.utils.fused_moe.align import (  # noqa: E402
    _cuda_op_available,
    _moe_align_block_size_torch,
)
from mstar.utils.fused_moe.kernels import (  # noqa: E402
    act_and_mul_triton,
    fused_moe_kernel,
    get_default_config,
    moe_sum_reduce_triton,
)
from perf_testing.moe_bench_common import (  # noqa: E402
    DEFAULT_M,
    SHAPES,
    bench,
    roofline_us,
    touched_experts,
)
from perf_testing.tune_fused_moe import Problem  # noqa: E402


def legacy_fused_experts(p: Problem) -> None:
    """One dispatch as it ran before this branch.

    Kept in the benchmark rather than in the library: it is a reference point,
    not a code path anyone should be able to select at runtime.
    """
    shape = p.shape
    E, top_k = shape.num_experts, shape.top_k
    hidden, inter = shape.hidden, shape.inter
    hs, w1, w2 = p.hidden_states, p.w1, p.w2
    topk_ids, topk_weights = p.topk_ids, p.topk_weights
    m = p.m
    ct = tl.bfloat16 if p.dtype == torch.bfloat16 else tl.float16

    cfg = get_default_config(M=m, E=E, N=2 * inter, K=hidden, top_k=top_k)
    block_m = cfg["BLOCK_SIZE_M"]

    # 1. alignment, via the torch fallback that the broken JIT build forced
    n = topk_ids.numel()
    max_padded = n + E * (block_m - 1)
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(triton.cdiv(max_padded, block_m), dtype=torch.int32, device="cuda")
    npp = torch.empty(1, dtype=torch.int32, device="cuda")
    _moe_align_block_size_torch(topk_ids, block_m, E, sorted_ids, expert_ids, npp)

    m_topk = m * top_k
    cache1 = torch.empty(m_topk, 2 * inter, device="cuda", dtype=p.dtype)
    cache2 = torch.empty(m_topk, inter, device="cuda", dtype=p.dtype)
    cache3 = torch.empty(m, top_k, hidden, device="cuda", dtype=p.dtype)

    def gemm(A, B, C, mul, tk):
        K = B.shape[2]
        # the pre-fix grid: sized from the worst-case padded length
        grid = (
            triton.cdiv(sorted_ids.shape[0], cfg["BLOCK_SIZE_M"])
            * triton.cdiv(B.shape[1], cfg["BLOCK_SIZE_N"]),
        )
        fused_moe_kernel[grid](
            A, B, C, topk_weights, sorted_ids, expert_ids, npp,
            B.shape[1], K, sorted_ids.shape[0], n,
            A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
            C.stride(-2), C.stride(-1),
            MUL_ROUTED_WEIGHT=mul, top_k=tk, compute_type=ct,
            even_Ks=(K % cfg["BLOCK_SIZE_K"]) == 0, **cfg,
        )

    gemm(hs, w1, cache1, False, top_k)
    act_and_mul_triton(cache1, cache2, activation="silu")
    gemm(cache2, w2, cache3.view(m_topk, hidden), True, 1)
    out = torch.empty_like(hs)
    moe_sum_reduce_triton(cache3, out, routed_scaling_factor=1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="thinker", choices=sorted(SHAPES))
    ap.add_argument("--m", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    shape = SHAPES[args.shape]
    print(f"# {shape.name}: hidden={shape.hidden} inter={shape.inter} "
          f"E={shape.num_experts} top_k={shape.top_k}  device={torch.cuda.get_device_name(0)}")
    print(f"# CUDA align op available: {_cuda_op_available()}")
    print("# one full fused_experts dispatch: align + gemm1 + swiglu + gemm2 + reduce")
    print(f"{'M':>6} {'exp':>4} {'legacy':>10} {'tuned':>10} {'graph':>10} "
          f"{'eager x':>8} {'graph x':>8} {'roofline':>9} {'% roof':>7}")

    for m in args.m or list(DEFAULT_M):
        p = Problem(shape, m)
        args_e = (p.hidden_states, p.w1, p.w2, p.topk_weights, p.topk_ids)

        t_legacy = bench(lambda p=p: legacy_fused_experts(p),
                         warm_s=0.1, windows=4, iters=10, graph=False).us
        t_tuned = bench(lambda a=args_e: fused_experts(*a),
                        warm_s=0.1, windows=4, iters=10, graph=False).us
        try:
            t_graph = bench(lambda a=args_e: fused_experts(*a),
                            warm_s=0.1, windows=4, iters=10, graph=True).us
            graph_s = f"{t_graph:9.2f}u"
            graph_x = f"{t_legacy / t_graph:7.2f}x"
            roof_pct = f"{roofline_us(shape, touched_experts(p.topk_ids, shape.num_experts)) / t_graph * 100:6.0f}%"
        except Exception as e:  # noqa: BLE001 -- capture can legitimately fail
            graph_s, graph_x, roof_pct = f"  {type(e).__name__[:7]}", "      --", "     --"

        roof = roofline_us(shape, touched_experts(p.topk_ids, shape.num_experts))
        print(f"{m:6d} {touched_experts(p.topk_ids, shape.num_experts):4d} "
              f"{t_legacy:9.2f}u {t_tuned:9.2f}u {graph_s} "
              f"{t_legacy / t_tuned:7.2f}x {graph_x} {roof:8.1f}u {roof_pct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
