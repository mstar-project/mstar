"""A/B the fused-MoE grouped GEMM: original launch vs the current one.

Reports, per token count, the time of one gate+up GEMM and one down GEMM
under three settings:

``legacy``  the launch as it was before tuning work: grid sized from
            ``len(sorted_token_ids)``, tile from ``get_default_config``, and
            Triton's default ``num_warps`` / ``num_stages``.
``grid``    same tile, but the tightened grid bound.
``tuned``   the tightened grid plus whatever ``get_default_config`` now
            returns (i.e. the offline config table, once it is populated).

Everything is timed under CUDA-graph replay -- see ``moe_bench_common.bench``
for why an eager loop cannot resolve these kernels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstar.utils.fused_moe.kernels import (  # noqa: E402
    fused_moe_kernel,
    get_default_config,
    get_moe_configs,
)
from perf_testing.moe_bench_common import (  # noqa: E402
    DEFAULT_M,
    SHAPES,
    bench,
    roofline_us,
    touched_experts,
)
from perf_testing.tune_fused_moe import Problem  # noqa: E402


def legacy_launch(p: Problem, gemm: int, config: dict):
    """The pre-fix launch path, reproduced here so production code stays clean."""
    block_m = config["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, num_post = p.align(block_m)
    ct = tl.bfloat16 if p.dtype == torch.bfloat16 else tl.float16
    if gemm == 1:
        A, B, C, mul, tk = p.hidden_states, p.w1, p.cache1, False, p.shape.top_k
    else:
        A, B, C, mul, tk = p.cache2, p.w2, p.cache3, True, 1
    tile = {k: v for k, v in config.items() if k.startswith(("BLOCK_", "GROUP_"))}
    K = B.shape[2]

    def run():
        grid = (
            triton.cdiv(sorted_ids.shape[0], tile["BLOCK_SIZE_M"])
            * triton.cdiv(B.shape[1], tile["BLOCK_SIZE_N"]),
        )
        fused_moe_kernel[grid](
            A, B, C, p.topk_weights, sorted_ids, expert_ids, num_post,
            B.shape[1], K, sorted_ids.shape[0], p.topk_ids.numel(),
            A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
            C.stride(-2), C.stride(-1),
            MUL_ROUTED_WEIGHT=mul, top_k=tk, compute_type=ct,
            even_Ks=(K % tile["BLOCK_SIZE_K"]) == 0, **tile,
        )

    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="thinker", choices=sorted(SHAPES))
    ap.add_argument("--m", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    shape = SHAPES[args.shape]
    m_list = args.m or list(DEFAULT_M)
    print(f"# {shape.name}: hidden={shape.hidden} inter={shape.inter} "
          f"E={shape.num_experts} top_k={shape.top_k}  device={torch.cuda.get_device_name(0)}")
    print("# times are gemm1 + gemm2 for one dispatch; roofline = time to stream the")
    print("# touched experts' w1+w2 once at HBM peak, the floor for this kernel.")
    print(f"{'M':>6} {'exp':>4} {'legacy':>10} {'grid':>10} {'tuned':>10} "
          f"{'grid x':>7} {'tuned x':>8} {'roofline':>9} {'% roof':>7}")

    for m in m_list:
        p = Problem(shape, m)
        legacy_cfg = get_default_config(
            M=m, E=shape.num_experts, N=shape.gemm1[0], K=shape.hidden, top_k=shape.top_k
        )
        tuned1, tuned2 = get_moe_configs(
            M=m, E=shape.num_experts, hidden=shape.hidden, inter=shape.inter, top_k=shape.top_k
        )
        totals = {}
        for label, launch, cfgs in (
            ("legacy", legacy_launch, (legacy_cfg, legacy_cfg)),
            ("grid", p.launch, (legacy_cfg, legacy_cfg)),
            ("tuned", p.launch, (tuned1, tuned2)),
        ):
            total = 0.0
            for gemm, cfg in zip((1, 2), cfgs, strict=True):
                fn = launch(p, gemm, cfg) if launch is legacy_launch else launch(gemm, cfg)
                total += bench(fn, warm_s=0.1, windows=4, iters=20).us
            totals[label] = total

        touched = touched_experts(p.topk_ids, shape.num_experts)
        roof = roofline_us(shape, touched)
        print(f"{m:6d} {touched:4d} {totals['legacy']:9.2f}u {totals['grid']:9.2f}u "
              f"{totals['tuned']:9.2f}u {totals['legacy'] / totals['grid']:6.2f}x "
              f"{totals['legacy'] / totals['tuned']:7.2f}x {roof:8.1f}u "
              f"{roof / totals['tuned'] * 100:6.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
