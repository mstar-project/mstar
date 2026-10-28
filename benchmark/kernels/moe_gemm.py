"""``moe_gemm``: fused_experts' grouped GEMMs (gate/up, then down), swept per token count.

Each bucket times the gate/up and the down GEMM separately for every candidate tile config,
inside a CUDA graph that launches once per routing in a rotation, so expert weights come from
HBM rather than L2, as in a decode step. The two GEMMs share one alignment, so a row takes the
BLOCK_SIZE_M whose best gate/up plus best down is fastest. A candidate competes only when its
output matches the default config's to within bf16 accumulation-order noise.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
import triton.language as tl

from benchmark.kernels.timing import time_graph
from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import get_config, get_default_config, invoke_fused_moe_kernel

# Decode batch sizes, then prefill chunk sizes; a lookup takes the nearest row.
DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)
DECODE_MAX = 128


def candidates(m: int, top_k: int, quick: bool = False) -> Iterator[dict]:
    """Tile configs worth timing at ``m`` tokens. Up to DECODE_MAX tokens the GEMMs are
    bandwidth-bound and want narrow row blocks; above, they are compute-bound and want wide ones."""
    if m <= DECODE_MAX:
        block_m = (16, 32, 64) if m * top_k >= 256 else (16,)
        grid = itertools.product(block_m, (32, 64, 128, 256), (64, 128, 256), (4, 8), (2, 3, 4, 5), (1,))
    else:
        grid = itertools.product((32, 64, 128), (64, 128, 256), (64, 128), (4, 8), (2, 3, 4), (1, 8))
    for bm, bn, bk, warps, stages, group in grid:
        if bn * bk > 256 * 128 or bm * bn > 128 * 256 or (bm == 64 and bn == 256 and bk == 256):
            continue
        if quick and (warps == 8 or stages == 2):
            continue
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
               "GROUP_SIZE_M": group if m > DECODE_MAX else (1 if m <= 64 else 8),
               "num_warps": warps, "num_stages": stages}


@dataclass
class BucketResult:
    bucket: int
    current_us: float
    default_us: float
    best_us: float
    row: dict  # the best candidate's
    current_row: dict  # what runs now, as a row
    timed: int
    rejected: int


def as_row(up: Mapping, down: Mapping) -> dict:
    """A (gate/up, down) tile pair as a table row: the gate/up tiles, and the down GEMM's
    differences under "down" (never BLOCK_SIZE_M, which the two share)."""
    return {**up, "down": {k: v for k, v in down.items() if k != "BLOCK_SIZE_M" and up.get(k) != v}}


class MoeGemmSweep:
    op = "moe_gemm"
    backend = "triton"
    default_buckets = DEFAULT_BUCKETS

    def __init__(self, shape: Mapping[str, int], dtype: torch.dtype, device: torch.device,
                 quick: bool = False):
        self.E, self.N, self.K, self.top_k = (shape[k] for k in ("E", "N", "K", "top_k"))
        if self.N % 2:
            raise ValueError(f"N is gate/up's width, 2 * the expert intermediate size; got {self.N}")
        self.dtype, self.device, self.quick = dtype, device, quick
        torch.manual_seed(0)
        self.w1 = torch.randn(self.E, self.N, self.K, device=device, dtype=dtype) * 0.02
        self.w2 = torch.randn(self.E, self.K, self.N // 2, device=device, dtype=dtype) * 0.02
        self.compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    def current(self, m: int) -> tuple[dict, dict]:
        """What fused_experts would launch now: the table's row, or the default."""
        return get_config(m, self.E, self.N, self.K, self.top_k, dtype=self.dtype, device=self.device)

    def run_bucket(self, m: int) -> BucketResult:
        E, N, K, top_k, dev, dt = self.E, self.N, self.K, self.top_k, self.device, self.dtype
        rotations = 16 if m <= DECODE_MAX else 2
        x = torch.randn(m, K, device=dev, dtype=dt)
        routings = [torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(m)]).to(torch.int32)
                    for _ in range(rotations)]
        weights = torch.full((m, top_k), 1.0 / top_k, device=dev, dtype=dt)
        up_out = torch.empty(m * top_k, N, device=dev, dtype=dt)
        down_in = torch.randn(m * top_k, N // 2, device=dev, dtype=dt)
        down_out = torch.empty(m * top_k, K, device=dev, dtype=dt)
        aligned: dict[int, list] = {}
        gemms = {"up": (x, self.w1, up_out, top_k), "down": (down_in, self.w2, down_out, 1)}

        def launch_all(cfg, gemm, only_first=False):
            bm = cfg["BLOCK_SIZE_M"]
            if bm not in aligned:
                aligned[bm] = [moe_align_block_size(ids, bm, E) for ids in routings]
            a, b, c, k = gemms[gemm]
            for ids, (sorted_ids, expert_ids, padded) in list(zip(routings, aligned[bm], strict=True))[
                    : 1 if only_first else None]:
                invoke_fused_moe_kernel(
                    A=a, B=b, C=c, topk_weights=weights, topk_ids=ids, sorted_token_ids=sorted_ids,
                    expert_ids=expert_ids, num_tokens_post_padded=padded,
                    mul_routed_weight=gemm == "down", top_k=k, config=cfg, compute_type=self.compute_type)

        default = get_default_config(m, E, N, K, top_k)
        refs = {}
        for gemm in gemms:
            launch_all(default, gemm, only_first=True)
            refs[gemm] = gemms[gemm][2].clone()

        def matches(cfg, gemm) -> bool:
            launch_all(cfg, gemm, only_first=True)
            out, ref = gemms[gemm][2], refs[gemm]
            return bool((out.float() - ref.float()).abs().max() <= 1e-2 * ref.float().abs().max() + 1e-3)

        def timed(cfg, gemm) -> float | None:
            try:
                if not matches(cfg, gemm):
                    return None
                return time_graph(lambda: launch_all(cfg, gemm), replays=10 if m <= DECODE_MAX else 5) / rotations
            except Exception:  # noqa: BLE001 - a tile Triton rejects, or one out of shared memory
                return None

        def pair_us(up, down) -> float:
            times = timed(up, "up"), timed(down, "down")
            return float("nan") if None in times else sum(times)

        current = self.current(m)
        current_us = pair_us(*current)
        default_us = pair_us(default, default)
        best: dict[str, dict[int, tuple[float, dict]]] = {"up": {}, "down": {}}
        tried = rejected = 0
        for cfg in candidates(m, top_k, self.quick):
            for gemm in gemms:
                tried += 1
                us = timed(cfg, gemm)
                if us is None:
                    rejected += 1
                elif us < best[gemm].get(cfg["BLOCK_SIZE_M"], (float("inf"),))[0]:
                    best[gemm][cfg["BLOCK_SIZE_M"]] = (us, cfg)
        bm = min((b for b in best["up"] if b in best["down"]),
                 key=lambda b: best["up"][b][0] + best["down"][b][0])
        (up_us, up), (down_us, down) = best["up"][bm], best["down"][bm]
        return BucketResult(bucket=m, current_us=current_us, default_us=default_us, best_us=up_us + down_us,
                            row=as_row(up, down), current_row=as_row(*current), timed=tried, rejected=rejected)
