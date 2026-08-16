"""Autotune the fused-MoE grouped GEMM and emit a config table.

Why this exists
---------------
The fused MoE used to pick tile sizes from a two-branch heuristic inherited
from sglang (now ``kernels.py:_heuristic_config``), never set ``num_warps`` or
``num_stages`` at all, so every launch ran at Triton's defaults.  On top of
that, ``runner.py`` computed the config **once** from the gate+up GEMM's
``(N, K) = (2*inter, hidden)`` and reused it for the down GEMM, whose shape is
the transposed-ish ``(hidden, inter)``.  Those are different problems and they
do not want the same tile.

We cannot fix this with ``@triton.autotune`` at runtime: the fused MoE is
called from inside CUDA-graph capture (``mstar/engine/cuda_graph_runner.py``),
and autotune's first-call benchmarking sweep both launches un-captured work and
allocates, which corrupts the graph.  So we tune offline and ship a lookup
table, the same way vLLM and sglang do.

Usage
-----
    python perf_testing/tune_fused_moe.py --shape thinker --quick
    python perf_testing/tune_fused_moe.py --shape thinker --save

Writes
``mstar/utils/fused_moe/configs/E=<E>,hidden=<H>,inter=<I>,dtype=<dt>,device=<gpu>.json``
mapping ``str(M) -> {BLOCK_SIZE_M, gemm1: {...}, gemm2: {...}}``.  Both GEMMs
share ``BLOCK_SIZE_M`` because it is the alignment granularity of
``moe_align_block_size``, which runs once per dispatch; the rest is per-GEMM.
Existing entries are merged, not overwritten.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mstar.utils.fused_moe.align import moe_align_block_size  # noqa: E402
from mstar.utils.fused_moe.kernels import invoke_fused_moe_kernel  # noqa: E402
from perf_testing.moe_bench_common import DEFAULT_M, SHAPES, bench  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "mstar" / "utils" / "fused_moe" / "configs"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# BLOCK_SIZE_M is the alignment granularity for moe_align_block_size, so both
# GEMMs in a dispatch must agree on it (otherwise we would have to run the
# align kernel twice per layer).  It is therefore searched jointly and the
# remaining knobs are searched per-GEMM given a fixed BLOCK_SIZE_M.
BLOCK_M_CHOICES = (16, 32, 64, 128)
BLOCK_N_CHOICES = (16, 32, 64, 128, 256)
BLOCK_K_CHOICES = (32, 64, 128, 256)
GROUP_M_CHOICES = (1, 4, 8, 16, 32)
NUM_WARPS_CHOICES = (2, 4, 8)
NUM_STAGES_CHOICES = (2, 3, 4, 5)

QUICK = {
    "BLOCK_N": (32, 64, 128),
    "BLOCK_K": (32, 64, 128),
    "GROUP_M": (1, 8),
    "num_warps": (4, 8),
    "num_stages": (3, 4),
}

# H100 SMEM per SM, minus a small allowance for the driver's reserved bytes.
SMEM_LIMIT = 227 * 1024


def _smem_bytes(bm: int, bn: int, bk: int, stages: int, elem: int = 2) -> int:
    """Pipelined A+B tile footprint -- the dominant shared-memory user."""
    return stages * (bm * bk + bk * bn) * elem


def candidate_configs(block_m: int, n: int, k: int, quick: bool) -> list[dict[str, Any]]:
    """Legal (BLOCK_N, BLOCK_K, GROUP_M, warps, stages) tuples for one GEMM."""
    bn_all = QUICK["BLOCK_N"] if quick else BLOCK_N_CHOICES
    bk_all = QUICK["BLOCK_K"] if quick else BLOCK_K_CHOICES
    gm_all = QUICK["GROUP_M"] if quick else GROUP_M_CHOICES
    w_all = QUICK["num_warps"] if quick else NUM_WARPS_CHOICES
    s_all = QUICK["num_stages"] if quick else NUM_STAGES_CHOICES

    out: list[dict[str, Any]] = []
    for bn, bk, gm, warps, stages in itertools.product(bn_all, bk_all, gm_all, w_all, s_all):
        # A tile wider than the problem is pure waste; one power-of-two of
        # overshoot is allowed because the mask makes it cheap and it can still
        # win on occupancy.
        if bn > 2 * n or bk > 2 * k:
            continue
        if _smem_bytes(block_m, bn, bk, stages) > SMEM_LIMIT:
            continue
        # One warp per >=16x16 output sub-tile, loosely; reject the extremes
        # that Triton will reject or that spill.
        if block_m * bn < warps * 256:
            continue
        out.append(
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm,
                "num_warps": warps,
                "num_stages": stages,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


class Problem:
    """Buffers for one (shape, M) point, reusable across configs."""

    def __init__(self, shape, m: int, dtype=torch.bfloat16, device="cuda", seed: int = 0):
        torch.manual_seed(seed)
        self.shape = shape
        self.m = m
        self.dtype = dtype
        E, top_k = shape.num_experts, shape.top_k
        h, inter = shape.hidden, shape.inter

        self.hidden_states = torch.randn(m, h, device=device, dtype=dtype)
        self.w1 = torch.randn(E, 2 * inter, h, device=device, dtype=dtype) / (h**0.5)
        self.w2 = torch.randn(E, h, inter, device=device, dtype=dtype) / (inter**0.5)

        logits = torch.randn(m, E, device=device, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        w, i = torch.topk(probs, top_k, dim=-1)
        self.topk_weights = (w / w.sum(-1, keepdim=True)).to(dtype).contiguous()
        self.topk_ids = i.to(torch.int32).contiguous()

        self.cache1 = torch.empty(m * top_k, 2 * inter, device=device, dtype=dtype)
        self.cache2 = torch.randn(m * top_k, inter, device=device, dtype=dtype)
        self.cache3 = torch.empty(m * top_k, h, device=device, dtype=dtype)

        self._align_cache: dict[int, tuple] = {}

    def align(self, block_m: int):
        """``moe_align_block_size`` output, memoized per BLOCK_SIZE_M."""
        if block_m not in self._align_cache:
            self._align_cache[block_m] = moe_align_block_size(
                self.topk_ids, block_m, self.shape.num_experts
            )
        return self._align_cache[block_m]

    def launch(self, gemm: int, config: dict[str, Any]):
        """Return a zero-arg callable running one GEMM under ``config``."""
        block_m = config["BLOCK_SIZE_M"]
        sorted_ids, expert_ids, num_post = self.align(block_m)
        tile = {k: v for k, v in config.items() if k.startswith(("BLOCK_", "GROUP_"))}
        launch_meta = {k: config[k] for k in ("num_warps", "num_stages") if k in config}
        ct = tl.bfloat16 if self.dtype == torch.bfloat16 else tl.float16

        if gemm == 1:
            A, B, C, mul, tk = self.hidden_states, self.w1, self.cache1, False, self.shape.top_k
        else:
            A, B, C, mul, tk = self.cache2, self.w2, self.cache3, True, 1

        def run():
            invoke_fused_moe_kernel(
                A=A,
                B=B,
                C=C,
                topk_weights=self.topk_weights,
                topk_ids=self.topk_ids,
                sorted_token_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_post,
                mul_routed_weight=mul,
                top_k=tk,
                config={**tile, **launch_meta},
                compute_type=ct,
            )

        return run


# ---------------------------------------------------------------------------
# Reference + correctness
# ---------------------------------------------------------------------------


def reference_gemm(p: Problem, gemm: int) -> torch.Tensor:
    """Per-slot reference for the grouped GEMM.

    Done expert-by-expert rather than by gathering ``w[ids]``: the gathered
    form is ``(M*top_k, N, K)`` and at M=1024 that is 96 GiB.  One matmul per
    expert over the slots routed to it is the same arithmetic in O(weights)
    memory.
    """
    top_k = p.shape.top_k
    ids = p.topk_ids.reshape(-1).to(torch.int64)  # (M*topk,) slot -> expert
    W = p.w1 if gemm == 1 else p.w2
    n = W.shape[1]
    out = torch.zeros(p.m * top_k, n, device=ids.device, dtype=torch.float32)
    src = p.hidden_states if gemm == 1 else p.cache2

    for e in torch.unique(ids).tolist():
        slots = (ids == e).nonzero(as_tuple=True)[0]
        rows = src[slots // top_k] if gemm == 1 else src[slots]
        out[slots] = rows.float() @ W[e].float().T

    if gemm == 2:
        out = out * p.topk_weights.reshape(-1, 1).float()
    return out.to(p.dtype)


def check(p: Problem, gemm: int, config: dict[str, Any], ref: torch.Tensor) -> float:
    """Run one config and return the max relative error against ``ref``."""
    out = p.cache1 if gemm == 1 else p.cache3
    out.zero_()
    p.launch(gemm, config)()
    torch.cuda.synchronize()
    denom = ref.abs().float().max().clamp_min(1e-6)
    return ((out.float() - ref.float()).abs().max() / denom).item()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_one(
    p: Problem, gemm: int, block_m: int, quick: bool, verbose: bool
) -> list[tuple[float, dict]]:
    """Benchmark every legal config for one GEMM at a fixed BLOCK_SIZE_M."""
    n, k = p.shape.gemm1 if gemm == 1 else p.shape.gemm2
    results: list[tuple[float, dict]] = []
    for cfg in candidate_configs(block_m, n, k, quick):
        try:
            t = bench(p.launch(gemm, cfg), warm_s=0.05, windows=3, iters=10)
        except Exception as e:  # noqa: BLE001 -- illegal tiles are expected
            if verbose:
                print(f"    skip {cfg}: {type(e).__name__}")
            continue
        results.append((t.us, cfg))
    results.sort(key=lambda r: r[0])
    return results


def best_verified(
    p: Problem, gemm: int, results: list[tuple[float, dict]], ref: torch.Tensor, top: int
) -> tuple[float, dict] | None:
    """Walk the ranked configs and return the fastest that is also correct."""
    for us, cfg in results[:top]:
        err = check(p, gemm, cfg, ref)
        if err < 2e-2:
            return us, cfg
        print(f"    reject {cfg} rel_err={err:.3g}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="thinker", choices=sorted(SHAPES))
    ap.add_argument("--m", type=int, nargs="*", default=None, help="token counts to tune")
    ap.add_argument("--quick", action="store_true", help="small search space")
    ap.add_argument("--save", action="store_true", help="write the config JSON")
    ap.add_argument("--top", type=int, default=5, help="candidates to verify / print")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    shape = SHAPES[args.shape]
    m_list = args.m or list(DEFAULT_M)
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    print(f"# shape={shape.name} E={shape.num_experts} top_k={shape.top_k} "
          f"hidden={shape.hidden} inter={shape.inter}")
    print(f"# gemm1 (N,K)={shape.gemm1}  gemm2 (N,K)={shape.gemm2}  device={device_name}")

    table: dict[str, dict] = {}
    for m in m_list:
        p = Problem(shape, m)
        ref1, ref2 = reference_gemm(p, 1), reference_gemm(p, 2)

        # BLOCK_SIZE_M is shared by both GEMMs (it is the align granularity),
        # so pick the one minimising the *sum* of the two GEMM times rather
        # than tuning each in isolation and hoping they agree.
        best: tuple[float, int, dict, dict] | None = None
        for block_m in BLOCK_M_CHOICES:
            r1 = sweep_one(p, 1, block_m, args.quick, args.verbose)
            r2 = sweep_one(p, 2, block_m, args.quick, args.verbose)
            v1 = best_verified(p, 1, r1, ref1, args.top)
            v2 = best_verified(p, 2, r2, ref2, args.top)
            if v1 is None or v2 is None:
                if args.verbose:
                    print(f"    BM={block_m}: no verified config")
                continue
            total = v1[0] + v2[0]
            if args.verbose:
                print(f"    BM={block_m:3d}: {v1[0]:7.2f} + {v2[0]:7.2f} = {total:7.2f} us")
            if best is None or total < best[0]:
                best = (total, block_m, v1[1], v2[1])

        if best is None:
            print(f"M={m:5d}: no verified config")
            continue
        total, block_m, cfg1, cfg2 = best
        rest = lambda c: {k: v for k, v in c.items() if k != "BLOCK_SIZE_M"}  # noqa: E731
        n1, k1 = shape.gemm1
        n2, k2 = shape.gemm2
        rows = m * shape.top_k
        flops = 2 * rows * (n1 * k1 + n2 * k2)
        print(f"M={m:5d}: {total:8.2f} us  {flops / (total * 1e-6) / 1e12:7.1f} TF/s  "
              f"BLOCK_SIZE_M={block_m}")
        print(f"        gemm1 {rest(cfg1)}")
        print(f"        gemm2 {rest(cfg2)}")
        table[str(m)] = {
            "BLOCK_SIZE_M": block_m,
            "gemm1": rest(cfg1),
            "gemm2": rest(cfg2),
        }

    if args.save:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = CONFIG_DIR / (
            f"E={shape.num_experts},hidden={shape.hidden},inter={shape.inter},"
            f"dtype=bfloat16,device={device_name}.json"
        )
        # Merge, so a rerun over a few token counts tops up an existing table
        # instead of dropping every M it did not visit.
        merged: dict[str, dict] = {}
        if path.exists():
            merged.update(json.loads(path.read_text()))
        merged.update(table)
        path.write_text(
            json.dumps(merged, indent=2, sort_keys=True, default=int) + "\n"
        )
        print(f"wrote {path} ({len(merged)} entries)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    raise SystemExit(main())
