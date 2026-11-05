"""``mstar::fused_add_rms_norm``: the shipped op (FlashInfer's ROCm fork, or
whatever the registry runs here) against Inductor's fused codegen and the
Triton kernel over its knob space. Both outputs are written in place; the
oracle checks both against the fp64 reference.

Shapes carry ``gemma`` (0 or 1) beside rows and hidden, since Qwen3.5 calls
the ``1 + weight`` form and most other models the plain one.
"""

from __future__ import annotations

import functools

import torch

from benchmark.kernels.candidates import Candidate, is_amd, search_order, space
from benchmark.kernels.spec import KernelSpec, shape_product
from mstar.platforms.facts import DeviceFacts

# the rms_norm grid (decode rows up to the largest decode bucket, prompt-sized
# rows; Qwen3.5 0.8B, 2B and 9B hidden sizes), Gemma form first
SHAPES = (
    shape_product(rows=(1, 4, 16, 64, 128, 512, 2048), hidden=(1024, 2048, 4096), gemma=(1,))
    + shape_product(rows=(1, 16, 128, 2048), hidden=(4096,), gemma=(0,))
)
TRITON_SPACE = {"elems_per_thread": (2, 4, 8, 16, 32), "rows_per_program": (1, 2, 4, 8)}
TRITON_AMD_SPACE = {**TRITON_SPACE, "waves_per_eu": (0, 1, 2, 4)}


def _inputs(shape, dtype, device, generator=None):
    rows, hidden = shape["rows"], shape["hidden"]
    hidden_states = torch.randn(rows, hidden, dtype=dtype, device=device, generator=generator)
    residual = torch.randn(rows, hidden, dtype=dtype, device=device, generator=generator) * 4
    weight = torch.randn(hidden, dtype=dtype, device=device, generator=generator) * 0.5
    return hidden_states, residual, weight, 1e-6, bool(shape.get("gemma", 0))


def _bytes(shape, dtype) -> float:
    size = torch.empty((), dtype=dtype).element_size()
    # read both inputs, write both outputs, read the weight
    return (4 * shape["rows"] + 1) * shape["hidden"] * size


def _registry():
    return torch.ops.mstar.fused_add_rms_norm


def _fork():
    from mstar.engine.resources.rms_norm.rocm import fused_add_rmsnorm

    return fused_add_rmsnorm


def _inductor():
    from mstar.engine.resources.rms_norm.rms_norm import _fused_add_rms_norm_reference

    return torch.compile(_fused_add_rms_norm_reference, dynamic=False)


def _triton(knobs):
    from mstar.kernels.triton.norm import fused_add_rms_norm

    return functools.partial(fused_add_rms_norm, knobs=dict(knobs))


def candidates(facts: DeviceFacts, budget: int | None = None, seeds=None) -> list[Candidate]:
    seeds = seeds or {}
    found = [Candidate("registry", {}, _registry), Candidate("inductor", {}, _inductor)]
    if is_amd(facts):
        found.append(Candidate("flashinfer_rocm", {}, _fork))
    points = space(TRITON_AMD_SPACE if is_amd(facts) else TRITON_SPACE, facts)
    for knobs in search_order(points, seeds.get("triton", ()), budget):
        found.append(Candidate("triton", knobs, functools.partial(_triton, knobs)))
    return found


def spec(shapes=None) -> KernelSpec:
    return KernelSpec.for_op(
        "fused_add_rms_norm",
        shapes=tuple(
            {**s, "gemma": s.get("gemma", 1)} for s in (shapes or SHAPES)
        ),
        dtypes=(torch.bfloat16,),
        inputs=_inputs,
        bytes_moved=_bytes,
        flops=lambda shape, dtype: 0.0,
        candidates=candidates,
    )
