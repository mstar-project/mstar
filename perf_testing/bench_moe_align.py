"""Cost of moe_align_block_size: vendored CUDA op vs the torch fallback.

The op is JIT-built on first use and silently falls back to a pure-torch
implementation when the toolchain cannot compile it. That fallback ran for
every call in-tree until the build was fixed, so this quantifies what the
fallback was costing per MoE layer.

Both are timed *eagerly* (``graph=False``), which is the only way to compare
them: the fallback cannot be CUDA-graph captured at all. It sizes
``repeat_interleave`` from a device tensor of per-expert block counts, which
forces a device-to-host sync, and capture aborts with
``cudaErrorStreamCaptureInvalidated``. So the fallback is not merely slower --
it takes the whole fused-MoE path out of the graph. The CUDA op has no such
restriction; that difference does not show up in the numbers below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstar.utils.fused_moe.align import (  # noqa: E402
    _cuda_op_available,
    _moe_align_block_size_torch,
)
from perf_testing.moe_bench_common import DEFAULT_M, SHAPES, bench  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="thinker", choices=sorted(SHAPES))
    ap.add_argument("--m", type=int, nargs="*", default=None)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    have_cuda_op = _cuda_op_available()
    print(f"# CUDA align op available: {have_cuda_op}")
    if not have_cuda_op:
        print("# nothing to compare -- only the fallback can run here")
        return 1

    shape = SHAPES[args.shape]
    E, top_k = shape.num_experts, shape.top_k
    bs = args.block_size
    print(f"# {shape.name}: E={E} top_k={top_k} block_size={bs}  "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"{'M':>6} {'cuda op':>10} {'torch':>10} {'speedup':>8}")

    for m in args.m or list(DEFAULT_M):
        torch.manual_seed(0)
        ids = (
            torch.stack([torch.randperm(E)[:top_k] for _ in range(m)])
            .to("cuda", torch.int32)
            .contiguous()
        )
        n = ids.numel()
        max_padded = n + E * (bs - 1)
        sorted_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
        expert_ids = torch.empty(triton.cdiv(max_padded, bs), dtype=torch.int32, device="cuda")
        npp = torch.empty(1, dtype=torch.int32, device="cuda")

        buffers = (sorted_ids, expert_ids, npp)
        t_cuda = bench(
            lambda ids=ids, b=buffers: torch.ops._mstar_moe_C.moe_align_block_size(
                ids, E, bs, *b
            ),
            warm_s=0.1, windows=4, iters=20, graph=False,
        ).us
        t_torch = bench(
            lambda ids=ids, b=buffers: _moe_align_block_size_torch(ids, bs, E, *b),
            warm_s=0.1, windows=4, iters=20, graph=False,
        ).us
        print(f"{m:6d} {t_cuda:9.2f}u {t_torch:9.2f}u {t_torch / t_cuda:7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
