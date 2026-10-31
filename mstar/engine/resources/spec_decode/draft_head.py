"""A draft's vocabulary head in fp8.

A draft pass reads the head for one argmax, and the target verifies the result,
so rounding the weights to 8 bits costs a little acceptance and no correctness.
The head is the largest kernel of a draft pass and all of it is the weight
stream, so one byte a weight in place of two halves that kernel.

``Fp8DraftHead`` keeps a copy of the head at one scale. Its logits are only
good for an argmax over a row: each row carries its own scale.

``mstar::fp8_head`` quantises each row of ``x`` and reads the head in fp8. At one
or two rows AITER's split-K kernel reads the weight once across every CU, 1.7-1.9x
the bf16 kernel on a 248k x 4K head on MI325X; from three rows on that kernel is
an order of magnitude slower and hipBLASLt's fp8 GEMM takes over, 1.2-1.6x. The
quantisation is inside the op because inductor does not get through compiling
an fp8 cast over a whole row.
"""

from __future__ import annotations

import functools

import torch
from torch import nn

from mstar.kernels import KernelOp, ShapePolicy, Tolerance
from mstar.kernels.fp8_gemm import quantize_rows
from mstar.platforms import current_platform

# where AITER's fp8 split-K kernel beats hipBLASLt's
SKINNY_MAX_ROWS = 2


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _unit_scales(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Scales of 1, for GEMMs that take them. Cached, so the first call of a
    shape, an eager warmup's, allocates them outside any captured graph."""
    return torch.ones(shape, dtype=torch.float32, device=device)


def _aiter_impl():
    from aiter.ops.custom import wvSplitKQ
    from aiter.ops.triton.quant import dynamic_per_token_quant_fp8_i8

    def fp8_head(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        rows, vocab, device = x.shape[0], weight.shape[0], x.device
        xq = torch.empty(x.shape, dtype=weight.dtype, device=device)
        dynamic_per_token_quant_fp8_i8(xq, x, torch.empty(rows, dtype=torch.float32, device=device))
        if rows > SKINNY_MAX_ROWS:
            return torch._scaled_mm(
                xq, weight.T, scale_a=_unit_scales((rows, 1), device),
                scale_b=_unit_scales((1, vocab), device), out_dtype=x.dtype,
            )
        out = torch.empty((rows, vocab), dtype=x.dtype, device=device)
        one = _unit_scales((1,), device)
        wvSplitKQ(weight, xq, out, one, one, _cu_count(device.index or 0))
        return out

    return fp8_head


def _reference(x, weight):
    xq = quantize_rows(x, weight.dtype)[0]
    return (xq.to(torch.float32) @ weight.to(torch.float32).T).to(x.dtype)


def _fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _shape(x, weight) -> dict[str, int]:
    return {"rows": x.shape[0], "n": weight.shape[0], "k": x.shape[1]}


def _inputs(shape, dtype, device, generator=None):
    fp8 = current_platform.fp8_dtype() or torch.float8_e4m3fn
    x = torch.randn(shape["rows"], shape["k"], device=device, generator=generator).to(dtype)
    weight = torch.randn(shape["n"], shape["k"], device=device, generator=generator) * shape["k"] ** -0.5
    return x, quantize_rows(weight, fp8)[0]


FP8_HEAD = KernelOp(
    "fp8_head",
    "(Tensor x, Tensor weight) -> Tensor",
    fake=_fake,
    reference=_reference,
    impls={f"rocm:{arch}": _aiter_impl for arch in ("gfx942", "gfx950")},
    shapes=ShapePolicy(
        dims=("rows", "n", "k"),
        key=_shape,
        inputs=_inputs,
        warm=({"rows": 1, "n": 2048, "k": 1024}, {"rows": 2, "n": 2048, "k": 1024},
              {"rows": 8, "n": 2048, "k": 1024}),
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

    def _apply(self, fn, recurse=True):
        # the engine moves a whole submodule to its dtype, which would widen the copy
        fp8 = self.weight.dtype
        return super()._apply(lambda t: fn(t).to(fp8), recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FP8_HEAD(x, self.weight)
