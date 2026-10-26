"""``moe_gemm``: fused_experts' grouped GEMMs (gate/up, then down), swept per token count.

Each bucket times the gate/up and the down GEMM separately for every candidate tile config,
inside a CUDA graph that launches once per routing in a rotation, so expert weights come from
HBM rather than L2, as in a decode step. The two GEMMs share one alignment, so a row takes the
BLOCK_SIZE_M whose best gate/up plus best down is fastest. A candidate, and the row that runs now,
competes only when its output is within the implementation's tolerance of the same kernel's at
its default tiles (the reference), which allows bf16 accumulation-order noise. On AMD GPUs the
tiles run at two stages with one to eight warps, and the best few tiles of each GEMM and
BLOCK_SIZE_M are timed again across Triton's HIP compiler options (waves_per_eu,
matrix_instr_nonkdim, kpack), which ride in the row's config.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping

import torch
import triton.language as tl

from benchmark.kernels.sweep import BucketResult
from benchmark.kernels.timing import time_graph
from mstar.kernels import Choice, TableEntry
from mstar.platforms import current_platform
from mstar.utils.fused_moe import kernels
from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import MOE_GEMM, get_default_config, invoke_fused_moe_kernel

# Decode batch sizes, then prefill chunk sizes; a row takes the counts up to its bucket.
DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)
DECODE_MAX = 128
# Triton's HIP compiler options, swept for the best tiles on AMD GPUs
AMD_OPTIONS = {"waves_per_eu": (0, 1, 2, 4), "matrix_instr_nonkdim": (16, 32), "kpack": (1, 2)}
AMD_SWEPT_TILES = 3  # per GEMM and BLOCK_SIZE_M


def candidates(m: int, top_k: int, quick: bool = False, amd: bool | None = None) -> Iterator[dict]:
    """Tile configs worth timing at ``m`` tokens. Up to DECODE_MAX tokens the GEMMs are
    bandwidth-bound and want narrow row blocks; above, they are compute-bound and want wide ones.
    AMD recommends two stages for Triton GEMMs on CDNA3 (more can run out of LDS), and small
    tiles often run best on one or two of its 64-wide warps."""
    if amd is None:
        amd = current_platform.is_rocm()
    warps, stages = ((1, 2, 4, 8), (2,)) if amd else ((4, 8), None)
    if m <= DECODE_MAX:
        block_m = (16, 32, 64) if m * top_k >= 256 else (16,)
        grid = itertools.product(block_m, (32, 64, 128, 256), (64, 128, 256), warps, stages or (2, 3, 4, 5), (1,))
    else:
        grid = itertools.product((32, 64, 128), (64, 128, 256), (64, 128), warps, stages or (2, 3, 4), (1, 8))
    for bm, bn, bk, warps, stages, group in grid:
        if bn * bk > 256 * 128 or bm * bn > 128 * 256 or (bm == 64 and bn == 256 and bk == 256):
            continue
        if quick and (warps == 8 or (stages == 2 and not amd)):
            continue
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
               "GROUP_SIZE_M": group if m > DECODE_MAX else (1 if m <= 64 else 8),
               "num_warps": warps, "num_stages": stages}


def amd_variants(cfg: Mapping) -> Iterator[dict]:
    """``cfg`` under each combination of the HIP options its tiles allow."""
    for waves, nonk, kpack in itertools.product(*AMD_OPTIONS.values()):
        if nonk <= min(cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]):
            yield {**cfg, "waves_per_eu": waves, "matrix_instr_nonkdim": nonk, "kpack": kpack}


def as_row(up: Mapping, down: Mapping) -> Choice:
    """A (gate/up, down) tile pair as a table row: the gate/up tiles, and the down GEMM's
    differences under "down" (never BLOCK_SIZE_M, which the two share)."""
    return Choice("triton", {**up, "down": {k: v for k, v in down.items()
                                            if k != "BLOCK_SIZE_M" and up.get(k) != v}})


class MoeGemmSweep:
    op = "moe_gemm"
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

    def tiles(self, choice: Choice) -> tuple[dict, dict]:
        """A row's (gate/up, down) tiles, as fused_experts launches them."""
        up = dict(choice.config)
        down_overrides = up.pop("down", {})
        return up, {**up, **down_overrides, "BLOCK_SIZE_M": up["BLOCK_SIZE_M"]}

    def check(self, entry: TableEntry) -> None:
        """Refuse an entry fused_experts would refuse."""
        kernels._entry_tiles(entry)

    def run_bucket(self, m: int, current: Choice | None) -> BucketResult:
        """``current`` is the table's row for ``m``; None, the default tiles."""
        default = get_default_config(m, self.E, self.N, self.K, self.top_k)
        default_choice = as_row(default, default)
        current = default_choice if current is None else current
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

        refs = {}
        for gemm in gemms:
            launch_all(default, gemm, only_first=True)
            refs[gemm] = gemms[gemm][2].clone()

        tolerance = MOE_GEMM.impl("triton").tolerance

        def matches(cfg, gemm) -> bool:
            launch_all(cfg, gemm, only_first=True)
            return tolerance.error(gemms[gemm][2], refs[gemm]) <= 1.0

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

        current_us = pair_us(*self.tiles(current))
        default_us = pair_us(*self.tiles(default_choice))
        amd = current_platform.is_rocm()
        # GEMM -> BLOCK_SIZE_M -> the fastest tiles, best first (several on AMD, for the option sweep)
        ranked: dict[str, dict[int, list[tuple[float, dict]]]] = {"up": {}, "down": {}}
        tried = rejected = 0

        def consider(cfg, gemm, keep) -> None:
            nonlocal tried, rejected
            tried += 1
            us = timed(cfg, gemm)
            if us is None:
                rejected += 1
                return
            top = ranked[gemm].setdefault(cfg["BLOCK_SIZE_M"], [])
            top.append((us, cfg))
            top.sort(key=lambda t: t[0])
            del top[keep:]

        for cfg in candidates(m, top_k, self.quick, amd):
            for gemm in gemms:
                consider(cfg, gemm, AMD_SWEPT_TILES if amd else 1)
        if amd:
            for gemm in gemms:
                for top in list(ranked[gemm].values()):
                    for _, cfg in list(top):
                        for variant in amd_variants(cfg):
                            consider(variant, gemm, AMD_SWEPT_TILES)
        best = {gemm: {bm: top[0] for bm, top in by_bm.items()} for gemm, by_bm in ranked.items()}
        result = BucketResult(bucket=m, current=current, current_us=current_us, default_us=default_us,
                              best=None, best_us=float("nan"), timed=tried - rejected, rejected=rejected)
        shared = [b for b in best["up"] if b in best["down"]]
        if shared:
            bm = min(shared, key=lambda b: best["up"][b][0] + best["down"][b][0])
            (up_us, up), (down_us, down) = best["up"][bm], best["down"][bm]
            result.best, result.best_us = as_row(up, down), up_us + down_us
            result.impl_us["triton"] = result.best_us
        return result
