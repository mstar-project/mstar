"""``mstar::decode_gemv`` and ``mstar::decode_gemv_silu_mul``: decode linears, a
weight stream, on the Triton kernel over its knob space (up to 16 rows) against
hipBLASLt (``F.linear``), hipBLASLt at its best solution for the shape, and
AMD's skinny kernel. The floor is one pass over the weight at HBM bandwidth.

The Triton kernel comes as two families, ``triton_fma`` (fp32 multiply-adds,
a reduction after the loop; for a few rows) and ``triton_dot`` (matrix cores,
rows padded to 16), so a search can run either alone. The fused gated MLP
projection is timed against the unfused pairs the served graph can run: a
linear, then inductor's one ``silu(gate) * up`` kernel.

``registry`` is what this tree serves for the shape, as the graph passes route
it, and ``decode_linear`` the tree's tuned hipBLASLt solution where it has one
(``F.linear`` elsewhere), so a search also reads what it would replace.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.kernels.candidates import Candidate, is_amd, search_order, space
from benchmark.kernels.spec import KernelSpec, shape_product
from benchmark.kernels.specs.linear import chosen_solution, hipblaslt_best
from mstar.kernels.numerics import Tolerance
from mstar.platforms.facts import DeviceFacts

ROWS = (1, 2, 4, 8, 16)
# decode buckets past the Triton kernel, which only hipBLASLt serves
WIDE_ROWS = (32, 48, 64)
# Qwen3.5 decode (n, k): GDN in_proj and out_proj, attention qkv and o, MLP
# gate_up and down, the vocab head
QWEN3_5_9B = ((12352, 4096), (4096, 4096), (10240, 4096), (24576, 4096), (4096, 12288), (248320, 4096))
QWEN3_5_2B = ((8224, 2048), (2048, 2048), (5120, 2048), (12288, 2048), (2048, 6144), (248320, 2048))
QWEN3_5_0_8B = ((8224, 1024), (1024, 2048), (5120, 1024), (7168, 1024), (1024, 3584), (248320, 1024))
SHAPES = tuple(
    {"rows": s["rows"], "n": n, "k": k}
    for (n, k) in QWEN3_5_9B + QWEN3_5_2B + QWEN3_5_0_8B for s in shape_product(rows=ROWS + WIDE_ROWS)
)
# the gate_up projections, stacked [gate; up]
SILU_MUL_SHAPES = tuple(
    {"rows": rows, "n": n, "k": k} for (n, k) in ((24576, 4096), (12288, 2048), (7168, 1024)) for rows in ROWS
)
# AMD's wvSpltK serves up to four rows
SKINNY_MAX_ROWS = 4
TRITON_MAX_ROWS = max(ROWS)

# Weight elements one thread loads per step along K stay between one 16-byte
# vector and 64: fewer leaves loads too short to stream, more spills. Pipeline
# stages only change the matrix-core loop. Split-K costs a second kernel, so it
# pays only where a narrow weight leaves CUs idle.
FMA_SPACE = {
    "block_n": (2, 4, 8, 16, 32), "block_k": (256, 512, 1024, 2048, 4096), "num_warps": (1, 2, 4, 8),
    "split_k": (1, 2, 4), "xcd_remap": (False, True), "waves_per_eu": (0, 2),
}
DOT_SPACE = {
    "block_n": (16, 32, 64), "block_k": (64, 128, 256, 512), "num_warps": (2, 4, 8),
    "split_k": (1, 2, 4, 8), "xcd_remap": (False, True), "waves_per_eu": (0, 2), "num_stages": (1, 2),
}


def _per_thread(point, facts) -> float:
    warp = facts.warp_size or 64
    return point["block_n"] * point["block_k"] / (point["num_warps"] * warp)


def _fma_ok(point, facts) -> bool:
    return 8 <= _per_thread(point, facts) <= 64


def _dot_ok(point, facts) -> bool:
    return 4 <= _per_thread(point, facts) <= 64


def _inputs(shape, dtype, device, generator=None):
    x = torch.randn(shape["rows"], shape["k"], dtype=dtype, device=device, generator=generator)
    w = torch.randn(shape["n"], shape["k"], dtype=dtype, device=device, generator=generator)
    return x, w * shape["k"] ** -0.5


def _reference(x, w):
    return x @ w.T


def _silu_mul_reference(x, w):
    gate, up = (x @ w.T).chunk(2, dim=-1)
    return F.silu(gate) * up


def _bytes(shape, dtype) -> float:
    size = torch.empty((), dtype=dtype).element_size()
    return (shape["n"] * shape["k"] + shape["rows"] * (shape["k"] + shape["n"])) * size


def _flops(shape, dtype) -> float:
    return 2.0 * shape["rows"] * shape["n"] * shape["k"]


@functools.cache
def _cu_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _aiter_linear():
    from aiter.ops.custom import wvSpltK

    def linear(x, w):
        if x.shape[0] > SKINNY_MAX_ROWS:
            raise ValueError(f"wvSpltK serves up to {SKINNY_MAX_ROWS} rows")
        out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
        wvSpltK(w, x.contiguous(), out, x.shape[0], _cu_count(x.device.index or 0))
        return out

    return linear


def _triton(knobs, use_dot):
    from mstar.kernels.triton.gemv import decode_gemv

    knobs = {**knobs, "use_dot": use_dot}

    def linear(x, w):
        if x.shape[0] > TRITON_MAX_ROWS:
            raise ValueError(f"the Triton kernel serves up to {TRITON_MAX_ROWS} rows")
        return decode_gemv(x, w, None, knobs)

    return linear


def _triton_silu_mul(knobs, use_dot):
    from mstar.kernels.triton.gemv import decode_gemv_silu_mul

    knobs = {**knobs, "use_dot": use_dot}
    return lambda x, w: decode_gemv_silu_mul(x, w, knobs)


def _decode_linear():
    return lambda x, w: torch.ops.mstar.decode_linear(x, w, None)


def _registry():
    """The linear as this tree's graph passes route it."""
    from mstar.platforms.rocm_gemm import gemv_impl

    def linear(x, w):
        impl = gemv_impl(x.shape[0], w.shape[0], x.shape[1])
        if impl == "aiter":
            return torch.ops.mstar.skinny_gemm(x, w, None)
        if impl == "triton":
            return torch.ops.mstar.decode_gemv(x, w, None)
        return torch.ops.mstar.decode_linear(x, w, None)

    return linear


def _registry_silu_mul():
    from mstar.platforms.rocm_gemm import fuses_silu_mul

    pair = _unfused(_registry())

    def run(x, w):
        if fuses_silu_mul(x.shape[0], w.shape[0], x.shape[1]):
            return torch.ops.mstar.decode_gemv_silu_mul(x, w)
        return pair(x, w)

    return run


def _unfused(linear):
    """``linear`` then inductor's one pointwise kernel, as the served graph runs them."""

    def act(gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up

    act = torch.compile(act, dynamic=False)

    def run(x, w):
        return act(linear(x, w))

    run.chosen = getattr(linear, "chosen", None)
    return run


def _triton_candidates(facts, budget, seeds, build) -> list[Candidate]:
    found = []
    for family, knobs_space, ok, use_dot in (
        ("triton_fma", FMA_SPACE, _fma_ok, False), ("triton_dot", DOT_SPACE, _dot_ok, True),
    ):
        points = space(knobs_space, facts, keep=lambda p, ok=ok: ok(p, facts))
        for knobs in search_order(points, seeds.get(family, ()), budget):
            found.append(Candidate(family, knobs, functools.partial(build, knobs, use_dot)))
    return found


def candidates(facts: DeviceFacts, budget: int | None = None, seeds=None) -> list[Candidate]:
    found = [Candidate("torch", {}, lambda: F.linear)]
    if is_amd(facts):
        found.extend((
            Candidate("registry", {}, _registry),
            Candidate("aiter_skinny", {}, _aiter_linear),
            Candidate("decode_linear", {}, _decode_linear),
            Candidate("hipblaslt_tuned", {}, hipblaslt_best, shape_knobs=chosen_solution),
        ))
    return found + _triton_candidates(facts, budget, seeds or {}, _triton)


def silu_mul_candidates(facts: DeviceFacts, budget: int | None = None, seeds=None) -> list[Candidate]:
    found = [Candidate("torch", {}, lambda: _unfused(F.linear))]
    if is_amd(facts):
        found.extend((
            Candidate("registry", {}, _registry_silu_mul),
            Candidate("aiter_skinny", {}, lambda: _unfused(_aiter_linear())),
            Candidate("decode_linear", {}, lambda: _unfused(_decode_linear())),
            Candidate("hipblaslt_tuned", {}, lambda: _unfused(hipblaslt_best()), shape_knobs=chosen_solution),
        ))
    return found + _triton_candidates(facts, budget, seeds or {}, _triton_silu_mul)


def _shape_class(shape) -> str | None:
    rows = shape["rows"]
    for name, limit in (("rows1", 1), ("rows2", 2), ("rows4", 4), ("rows8", 8), ("rows16", 16)):
        if rows <= limit:
            return name
    return None


def _spec(name, shapes, reference, cands) -> KernelSpec:
    return KernelSpec(
        name=name,
        shapes=tuple(shapes),
        dtypes=(torch.bfloat16,),
        inputs=_inputs,
        reference=reference,
        # one bf16 rounding of a long fp32 sum, plus summation order
        tolerance=Tolerance.accumulated(),
        bytes_moved=_bytes,
        flops=_flops,
        candidates=cands,
        # a vocab head is 2 GB of bf16 weight: check it in fp32 on the device
        oracle_on_device=True,
        shape_class=_shape_class,
    )


def spec(shapes=None) -> KernelSpec:
    return _spec("decode_gemv", shapes or SHAPES, _reference, candidates)


# what a candidate family ships as, in the `decode_gemm` table
LINEAR_IMPLS = {
    "torch": ("hipblaslt", {}), "aiter_skinny": ("aiter", {}), "hipblaslt_tuned": ("hipblaslt_tuned", {}),
    "triton_fma": ("triton", {"use_dot": False}), "triton_dot": ("triton", {"use_dot": True}),
}
GATED_IMPLS = {
    "torch": ("unfused", {}), "aiter_skinny": ("unfused", {}), "hipblaslt_tuned": ("unfused", {}),
    "triton_fma": ("fused", {"use_dot": False}), "triton_dot": ("fused", {"use_dot": True}),
}
# what a shape nobody measured runs
_CLASS_DEFAULTS = {"rows1": {"impl": "aiter"}, "rows2": {"impl": "aiter"}}


def promote(store, facts, margin: float, root=None, provenance=None) -> Path:
    """The ``decode_gemm`` table from the stored ``decode_gemv`` and
    ``decode_gemv_silu_mul`` records: every measured shape's fastest kernel, a
    kernel of this repo or a tuned solution leaving the default by ``margin``.
    A gate/up projection stays fused unless the best pair beats it by ``margin``:
    a tie goes to the fold, which launches one kernel less."""
    common = dict(table="decode_gemm", margin=margin, root=root, provenance=provenance, shape_class=_shape_class)
    store.promote_best(facts, "decode_gemv", LINEAR_IMPLS.get, tuned=("triton", "hipblaslt_tuned"),
                       class_default=lambda klass: _CLASS_DEFAULTS.get(klass, {"impl": "hipblaslt"}), **common)
    return store.promote_best(facts, "decode_gemv_silu_mul", GATED_IMPLS.get, tuned=("fused",),
                              class_default=lambda klass: {"impl": "fused"}, extra={"gated": 1}, **common)


def silu_mul_spec(shapes=None) -> KernelSpec:
    return _spec("decode_gemv_silu_mul", shapes or SILU_MUL_SHAPES, _silu_mul_reference, silu_mul_candidates)
