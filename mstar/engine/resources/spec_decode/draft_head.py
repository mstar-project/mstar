"""A draft's vocabulary head in fp8.

A draft pass reads the head for one argmax, and the target verifies the result,
so rounding the weights to 8 bits costs a little acceptance and no correctness.
The head is the largest kernel of a draft pass and all of it is the weight
stream, so one byte a weight in place of two halves that kernel.

``Fp8DraftHead`` keeps a copy of the head at one scale. Its logits are only
good for an argmax over a row: each row carries its own scale.

``mstar::fp8_skinny_gemm`` is ``x @ weight.T`` in fp8 on AITER's split-K kernel,
which reads the weight once across every CU. It is 1.7-1.9x the bf16 kernel at
one or two rows of a 248k x 4K head on MI325X, and from three rows on an order
of magnitude slower, so wider steps read the bf16 head.
"""

from __future__ import annotations

import functools

import torch
from torch import nn

from mstar.kernels import KernelOp, ShapePolicy, Tolerance
from mstar.kernels.fp8_gemm import quantize_rows
from mstar.platforms import current_platform

# where AITER's fp8 split-K kernel beats the bf16 one
FP8_SKINNY_MAX_ROWS = 2


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=device)


def _aiter_skinny():
    from aiter.ops.custom import wvSplitKQ

    def fp8_skinny_gemm(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=out_dtype, device=x.device)
        one = _unit_scale(x.device)
        wvSplitKQ(weight, x, out, one, one, _cu_count(x.device.index or 0))
        return out

    return fp8_skinny_gemm


def _reference(x, weight, out_dtype):
    return (x.to(torch.float32) @ weight.to(torch.float32).T).to(out_dtype)


def _fake(x, weight, out_dtype):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=out_dtype)


def _shape(x, weight, out_dtype) -> dict[str, int]:
    return {"rows": x.shape[0], "n": weight.shape[0], "k": x.shape[1]}


def _inputs(shape, dtype, device, generator=None):
    fp8 = current_platform.fp8_dtype() or torch.float8_e4m3fn
    x = torch.randn(shape["rows"], shape["k"], device=device, generator=generator)
    weight = torch.randn(shape["n"], shape["k"], device=device, generator=generator) * shape["k"] ** -0.5
    return quantize_rows(x, fp8)[0], quantize_rows(weight, fp8)[0], dtype


FP8_SKINNY_GEMM = KernelOp(
    "fp8_skinny_gemm",
    "(Tensor x, Tensor weight, ScalarType out_dtype) -> Tensor",
    fake=_fake,
    reference=_reference,
    impls={f"rocm:{arch}": _aiter_skinny for arch in ("gfx942", "gfx950")},
    shapes=ShapePolicy(
        dims=("rows", "n", "k"),
        key=_shape,
        inputs=_inputs,
        warm=({"rows": 1, "n": 2048, "k": 1024}, {"rows": 2, "n": 2048, "k": 1024}),
    ),
    # one rounding of a long fp32 sum of exact fp8 products, plus summation order
    tolerance=Tolerance.accumulated(),
)


class Fp8DraftHead(nn.Module):
    """``weight`` ``[vocab, hidden]`` rounded to fp8, for argmaxes over its rows."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        fp8 = current_platform.fp8_dtype()
        if fp8 is None:
            raise ValueError(f"{current_platform.name} has no fp8 GEMM for a draft head")
        scale = weight.float().abs().amax() / torch.finfo(fp8).max
        self.register_buffer("weight", (weight.float() / scale).to(fp8), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FP8_SKINNY_GEMM(quantize_rows(x, self.weight.dtype)[0], self.weight, x.dtype)
