"""Decode-sized linears (``x @ w.T`` with a few rows): the vendor GEMM behind
``F.linear`` against AMD's skinny kernel where AITER is installed. Weights are
the traffic, so the floor is one pass over them at HBM bandwidth."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from benchmark.kernels.candidates import Candidate, is_amd
from benchmark.kernels.spec import KernelSpec, shape_product
from mstar.kernels.numerics import Tolerance
from mstar.platforms.facts import DeviceFacts

# Qwen3.5 9B decode: GDN in_proj and out_proj, attention qkv and o, MLP gate_up
# and down, and the vocab head, as (n, k)
QWEN3_5_9B = ((12352, 4096), (4096, 4096), (10240, 4096), (24576, 4096), (4096, 12288), (248320, 4096))
SHAPES = tuple(
    {"rows": s["rows"], "n": n, "k": k}
    for (n, k) in QWEN3_5_9B for s in shape_product(rows=(1, 2, 4, 8, 16))
)
# AMD's wvSpltK reads the weight once across every CU; it serves few rows
SKINNY_MAX_ROWS = 4


def _inputs(shape, dtype, device, generator=None):
    x = torch.randn(shape["rows"], shape["k"], dtype=dtype, device=device, generator=generator)
    w = torch.randn(shape["n"], shape["k"], dtype=dtype, device=device, generator=generator)
    return x, w * shape["k"] ** -0.5


def _reference(x, w):
    return x @ w.T


def _bytes(shape, dtype) -> float:
    size = torch.empty((), dtype=dtype).element_size()
    return (shape["n"] * shape["k"] + shape["rows"] * (shape["k"] + shape["n"])) * size


def _flops(shape, dtype) -> float:
    return 2.0 * shape["rows"] * shape["n"] * shape["k"]


def _torch():
    return F.linear


@functools.cache
def _cu_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _aiter_skinny():
    from aiter.ops.custom import wvSpltK

    def linear(x, w):
        if x.shape[0] > SKINNY_MAX_ROWS:
            raise ValueError(f"wvSpltK serves up to {SKINNY_MAX_ROWS} rows")
        out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
        wvSpltK(w, x.contiguous(), out, x.shape[0], _cu_count(x.device.index or 0))
        return out

    return linear


# AITER's bf16 asm GEMMs for gfx942 (hsa/gfx942/bf16gemm), by tile; "auto" lets AITER pick. Their split-K
# variants reduce through a 16x64 semaphore array, so a grid wider than that (any vocab head) runs unsplit.
ASM_TILES = ("auto", "32x64_pf3_splitk", "48x64_pf3_splitk", "64x64_pf3_splitk", "96x64_pf3_splitk",
             "64x64_splitk_clean", "96x64_splitk_clean")
ASM_SPLITS = (1, 2, 4)


def _aiter_asm(tile: str, split_k: int):
    def build():
        from aiter.ops.gemm_op_a16w16 import ASM_SPLITK_MAX_GRID, gemm_a16w16_asm

        name = None if tile == "auto" else f"bf16gemm_fp32bf16_tn_{tile}"
        name = None if name is None else f"_ZN5aiter{len(name)}{name}E"

        def linear(x, w):
            tile_m = 32 if tile == "auto" else int(tile.split("x", maxsplit=1)[0])
            if split_k > 1 and -(-x.shape[0] // tile_m) * -(-w.shape[0] // 64) > ASM_SPLITK_MAX_GRID:
                raise ValueError("grid wider than the split-K semaphore array")
            out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
            return gemm_a16w16_asm(x.contiguous(), w, out, None, split_k, name, False)

        return linear

    return build


def _aiter_triton():
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    return lambda x, w: gemm_a16w16(x, w, dtype=x.dtype)


def _tunableop():
    """hipBLASLt / rocBLAS through torch's TunableOp: the first (eager) call per shape times every solution and
    keeps the fastest, which the captured replays then run. On only around this call, so the other candidates'
    F.linear stays the default solution."""
    from torch.cuda import tunable

    tunable.tuning_enable(True)
    tunable.set_max_tuning_iterations(30)

    def linear(x, w):
        tunable.enable(True)
        try:
            return F.linear(x, w)
        finally:
            tunable.enable(False)

    return linear


# solutions chosen by `hipblaslt_best`, by (rows, n, k): (index, quick us, default us, solutions tried)
HIPBLASLT_CHOICES: dict[tuple[int, int, int], tuple[int, float, float, int]] = {}


def _quick_us(fn, x, weights, reps: int = 10) -> float:
    """Median us of ``fn(x, w)`` over ``reps`` calls cycling ``weights`` (eager, events)."""
    times = []
    for i in range(reps + 2):
        w = weights[i % len(weights)]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x, w)
        end.record()
        end.synchronize()
        if i >= 2:
            times.append(start.elapsed_time(end) * 1e3)
    return sorted(times)[len(times) // 2]


def _hipblaslt_best():
    """hipBLASLt at the fastest of all its solutions for a shape (AITER's ``hipb_findallsols``), picked on
    the first call per shape by a quick eager timing over weight copies that outgrow the MALL; the
    captured timings then run that solution. The choice is kept in ``HIPBLASLT_CHOICES``."""
    import aiter

    aiter.hipb_create_extension()
    chosen: dict[tuple[int, int, int], int] = {}

    def linear(x, w):
        key = (x.shape[0], w.shape[0], w.shape[1])
        if key not in chosen:
            copies = max(1, min(8, -(-(600 << 20) // (w.numel() * w.element_size()))))
            weights = [w] + [w.clone() for _ in range(copies - 1)]
            sols = aiter.hipb_findallsols(x, w.t())
            timed = []
            for sol in sols:
                try:
                    timed.append((_quick_us(lambda a, b, sol=sol: aiter.hipb_mm(a, b.t(), sol), x, weights), sol))
                except Exception:  # noqa: BLE001  a solution that cannot run this shape
                    continue
            best_us, best = min(timed)
            default_us = _quick_us(F.linear, x, weights)
            chosen[key] = best
            HIPBLASLT_CHOICES[key] = (best, best_us, default_us, len(sols))
            print(f"hipblaslt_best rows={key[0]} n={key[1]} k={key[2]}: solution {best} {best_us:.1f} us "
                  f"(default {default_us:.1f} us, {len(sols)} solutions)", flush=True)
            del weights
        return aiter.hipb_mm(x, w.t(), chosen[key])

    return linear


def candidates(facts: DeviceFacts, budget: int | None = None, seeds=None) -> list[Candidate]:
    found = [Candidate("torch", {}, _torch)]
    if is_amd(facts):
        found.append(Candidate("aiter_skinny", {}, _aiter_skinny))
        found.append(Candidate("aiter_triton", {}, _aiter_triton))
        found += [Candidate("aiter_asm", {"tile": t, "split_k": k}, _aiter_asm(t, k))
                  for t in ASM_TILES for k in ASM_SPLITS if t != "auto" or k == 1]
        found.append(Candidate("tunableop", {}, _tunableop))
        found.append(Candidate("hipblaslt_best", {}, _hipblaslt_best))
    return found


def spec(shapes=None) -> KernelSpec:
    return KernelSpec(
        name="linear",
        shapes=tuple(shapes or SHAPES),
        dtypes=(torch.bfloat16,),
        inputs=_inputs,
        reference=_reference,
        # one bf16 rounding of a long fp32 sum, plus summation order
        tolerance=Tolerance.accumulated(),
        bytes_moved=_bytes,
        flops=_flops,
        candidates=candidates,
        # a vocab head is 2 GB of bf16 weight: check it in fp32 on the device
        oracle_on_device=True,
        shape_class=lambda shape: "decode" if shape["rows"] <= 16 else None,
    )
