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


def candidates(facts: DeviceFacts, budget: int | None = None, seeds=None) -> list[Candidate]:
    found = [Candidate("torch", {}, _torch)]
    if is_amd(facts):
        found.append(Candidate("aiter_skinny", {}, _aiter_skinny))
        found.append(Candidate("aiter_triton", {}, _aiter_triton))
        found += [Candidate("aiter_asm", {"tile": t, "split_k": k}, _aiter_asm(t, k))
                  for t in ASM_TILES for k in ASM_SPLITS if t != "auto" or k == 1]
        found.append(Candidate("tunableop", {}, _tunableop))
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
