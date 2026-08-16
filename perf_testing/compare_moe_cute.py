"""Head-to-head: mstar's Triton fused-MoE GEMMs vs a CuTe DSL grouped GEMM.

What is being compared
----------------------
mstar's ``fused_moe_kernel`` is a *gather* grouped GEMM: tokens are never
materialised in expert order, the kernel reads ``sorted_token_ids`` and pulls
each row of A through an indirection.  CUTLASS's Hopper grouped GEMM is a
*contiguous* grouped GEMM: every group's A rows must already be adjacent in
memory, and the kernel is a warp-specialised TMA + WGMMA persistent kernel
driven by a tile scheduler.

So the honest comparison charges CuTe for the permutation it needs and does
not charge Triton for it.  Two CuTe numbers are reported:

``cute``        the grouped GEMM alone
``cute+gather`` the same plus the ``index_select`` that builds the permuted A

Caveats, stated because they bound what the numbers mean:

* The Hopper example supports fp16 but not bf16 (``is_valid_dtypes``), so
  **both** sides run in fp16 here.  On H100 fp16 and bf16 share the same
  tensor-core rate, so this is fair for speed; mstar itself runs bf16.
* The example's M tile is at minimum 64.  At decode a group holds 1--8 rows,
  so most of each tile is padding -- that is a property of the kernel being
  compared, not a handicap imposed by this harness, and it is the main thing
  the numbers show.
* CuTe's own ``testing.benchmark`` is used for the CuTe side (it is what the
  example ships); the Triton side uses the CUDA-graph timing in
  ``moe_bench_common``.  Both settle the clock first.

Usage
-----
    PYTHONPATH=/home/xikaim/cutlass_pkgs:/home/xikaim/cutlass_pkgs/nvidia_cutlass_dsl/dsl_packages \\
    python perf_testing/compare_moe_cute.py --shape thinker --m 1 8 64 512 4096
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_testing.moe_bench_common import SHAPES, bench  # noqa: E402
from perf_testing.tune_fused_moe import Problem  # noqa: E402

DEFAULT_EXAMPLE = "/home/xikaim/cutlass_examples/grouped_gemm.py"


def load_cute_example(path: str):
    """Import the CUTLASS example module from a file path."""
    spec = importlib.util.spec_from_file_location("cutlass_grouped_gemm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def expert_counts(p: Problem) -> list[int]:
    """Rows landing on each expert, dropping the empty ones."""
    counts = torch.bincount(
        p.topk_ids.reshape(-1).to(torch.int64), minlength=p.shape.num_experts
    )
    return [int(c) for c in counts.tolist() if c > 0]


# The tile/cluster shapes the Hopper example accepts.  Both sides get tuned;
# comparing a tuned Triton kernel against a default-configured CuTe kernel
# would say more about the defaults than about the two approaches.
CUTE_TILES = ((128, 256), (128, 128), (128, 64), (64, 64))
CUTE_CLUSTERS = ((1, 1), (2, 1))


def cute_time_us(mod, problem_sizes, tile_shape=(128, 128), cluster=(1, 1), iters: int = 50) -> float:
    """Run the example's grouped GEMM on ``problem_sizes``; return microseconds."""
    import cutlass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        us = mod.run(
            len(problem_sizes),
            problem_sizes,
            cutlass.Float16,  # a
            cutlass.Float16,  # b
            cutlass.Float16,  # c
            cutlass.Float32,  # acc
            "k",
            "k",
            "n",
            tile_shape,
            cluster,
            mod.utils.TensorMapUpdateMode.SMEM,
            1e-1,
            5,  # warmup iterations
            iters,
            True,  # skip_ref_check -- correctness is covered by the pytest suite
            False,  # use_cold_l2
        )
    return float(us)


def cute_best(mod, problem_sizes, verbose: bool = False) -> tuple[float, tuple, tuple] | None:
    """Best (time, tile, cluster) over the example's legal configurations."""
    best = None
    for tile in CUTE_TILES:
        for cluster in CUTE_CLUSTERS:
            try:
                us = cute_time_us(mod, problem_sizes, tile_shape=tile, cluster=cluster)
            except Exception as e:  # noqa: BLE001 -- unsupported combination
                if verbose:
                    print(f"    cute skip {tile}/{cluster}: {type(e).__name__}")
                continue
            if verbose:
                print(f"    cute {tile}/{cluster}: {us:.2f} us")
            if best is None or us < best[0]:
                best = (us, tile, cluster)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="thinker", choices=sorted(SHAPES))
    ap.add_argument("--m", type=int, nargs="*", default=[1, 8, 64, 512, 4096])
    ap.add_argument("--example", default=os.environ.get("CUTE_GROUPED_GEMM", DEFAULT_EXAMPLE))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    if not Path(args.example).exists():
        print(f"CuTe example not found at {args.example}; pass --example or set "
              f"CUTE_GROUPED_GEMM")
        return 1

    mod = load_cute_example(args.example)
    shape = SHAPES[args.shape]
    print(f"# {shape.name}: hidden={shape.hidden} inter={shape.inter} "
          f"E={shape.num_experts} top_k={shape.top_k}  fp16, both sides tuned")
    print(f"# device={torch.cuda.get_device_name(0)}")
    print(f"{'M':>6} {'gemm':>5} {'groups':>7} {'triton':>10} {'cute':>10} "
          f"{'gather':>9} {'cute+g':>10} {'cute cfg':>18} {'speedup':>9}")

    for m in args.m:
        p = Problem(shape, m, dtype=torch.float16)
        counts = expert_counts(p)

        # The permuted-A gather CuTe would need, timed once per M.
        idx = p.topk_ids.reshape(-1).argsort().to(torch.int64) // shape.top_k
        src = p.hidden_states
        dst = torch.empty(m * shape.top_k, shape.hidden, device="cuda", dtype=torch.float16)
        t_gather = bench(
            lambda src=src, idx=idx, dst=dst: torch.index_select(src, 0, idx, out=dst),
            warm_s=0.1, windows=4, iters=20,
        ).us

        for gemm in (1, 2):
            n, k = shape.gemm1 if gemm == 1 else shape.gemm2
            from mstar.utils.fused_moe.kernels import get_moe_configs

            # The tuned table was built in bf16; on H100 the two 16-bit types
            # share a tensor-core rate and a bytes-per-element, so the tile
            # choice carries over and the Triton side is not handicapped by
            # running fp16 to match what the CuTe example supports.
            cfg1, cfg2 = get_moe_configs(
                M=m, E=shape.num_experts, hidden=shape.hidden, inter=shape.inter,
                top_k=shape.top_k, dtype="bfloat16",
            )
            cfg = cfg1 if gemm == 1 else cfg2
            t_triton = bench(p.launch(gemm, cfg), warm_s=0.1, windows=4, iters=20).us

            sizes = [(c, n, k, 1) for c in counts]
            best = cute_best(mod, sizes, verbose=args.verbose)
            if best is None:
                print(f"{m:6d} {gemm:5d} {len(sizes):7d} {t_triton:9.2f}u  "
                      f"no legal cute configuration")
                continue
            t_cute, tile, cluster = best

            # >1 means Triton is faster.  The fair denominator includes the
            # gather, since CuTe cannot consume the unpermuted rows.
            speedup = (t_cute + t_gather) / t_triton
            print(f"{m:6d} {gemm:5d} {len(sizes):7d} {t_triton:9.2f}u {t_cute:9.2f}u "
                  f"{t_gather:8.2f}u {t_cute + t_gather:9.2f}u "
                  f"{str(tile) + '/' + str(cluster):>18} {speedup:8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
