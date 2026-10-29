"""``mstar::fp8_rowwise_mm``: an fp8 GEMM with one scale per row of each operand.

``out[m, n] = x_scale[m] * weight_scale[n] * sum_k x[m, k] * weight[n, k] (+ bias[n])``

``x`` and ``weight`` are e4m3 in the platform's format (``Platform.fp8_dtype``):
activations quantised per token, weights per output channel. The product
accumulates in fp32 and is rounded once to ``out_dtype``. Quantising the
activations is left to the caller, in plain torch, so a compiled forward fuses
it into whatever produced them (``mstar.model.components.fp8_linear``).

On both GPU platforms this is ``torch._scaled_mm`` with rowwise scales, i.e.
hipBLASLt on AMD and cuBLASLt on NVIDIA.
"""

from __future__ import annotations

import torch

from mstar.kernels import KernelOp, ShapePolicy, Tolerance
from mstar.platforms import current_platform


def _reference(x, x_scale, weight, weight_scale, bias, out_dtype):
    compute = torch.float64 if x_scale.dtype == torch.float64 else torch.float32
    out = (x.to(compute) * x_scale.to(compute)) @ (weight.to(compute) * weight_scale.to(compute).T).T
    if bias is not None:
        out = out + bias.to(compute)
    return out.to(out_dtype)


def _fake(x, x_scale, weight, weight_scale, bias, out_dtype):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=out_dtype)


def _scaled_mm():
    if current_platform.fp8_dtype() is None:
        raise RuntimeError(f"{current_platform.name} {current_platform.arch()} has no fp8 GEMM")
    if current_platform.is_cuda() and not current_platform.has_cuda_capability(9, 0):
        # Ada has fp8 tensor cores, but torch's rowwise-scaled fp8 GEMM starts at Hopper
        raise RuntimeError(f"rowwise-scaled fp8 GEMMs need sm_90 or newer, not {current_platform.arch()}")

    def fp8_rowwise_mm(x, x_scale, weight, weight_scale, bias, out_dtype):
        return torch._scaled_mm(x, weight.T, scale_a=x_scale, scale_b=weight_scale, bias=bias, out_dtype=out_dtype)

    return fp8_rowwise_mm


def _shape(x, x_scale, weight, weight_scale, bias, out_dtype) -> dict[str, int]:
    return {"rows": x.shape[0], "n": weight.shape[0], "k": x.shape[1], "bias": int(bias is not None)}


def quantize_rows(t: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` with ``t ~= q * scale``: one fp32 scale per row of the 2-D ``t``,
    which maps the row's largest magnitude to the format's largest finite value."""
    fp8_max = torch.finfo(fp8_dtype).max
    # an all-zero row keeps a finite scale
    scale = (t.abs().amax(dim=-1, keepdim=True).float() / fp8_max).clamp(min=1e-12)
    return (t.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype), scale


def _inputs(shape, dtype, device, generator=None):
    fp8 = current_platform.fp8_dtype() or torch.float8_e4m3fn
    x = torch.randn(shape["rows"], shape["k"], device=device, generator=generator)
    weight = torch.randn(shape["n"], shape["k"], device=device, generator=generator) * shape["k"] ** -0.5
    xq, x_scale = quantize_rows(x, fp8)
    wq, w_scale = quantize_rows(weight, fp8)
    bias = torch.randn(shape["n"], device=device, generator=generator).to(dtype) if shape["bias"] else None
    return xq, x_scale, wq, w_scale.T.contiguous(), bias, dtype


FP8_ROWWISE_MM = KernelOp(
    "fp8_rowwise_mm",
    "(Tensor x, Tensor x_scale, Tensor weight, Tensor weight_scale, Tensor? bias, ScalarType out_dtype) -> Tensor",
    fake=_fake,
    reference=_reference,
    impls={"cuda": _scaled_mm, "rocm": _scaled_mm},
    shapes=ShapePolicy(
        dims=("rows", "n", "k", "bias"),
        key=_shape,
        inputs=_inputs,
        # small enough for the CPU reference; the DiT shapes are measured by the harness
        warm=({"rows": 1024, "n": 3072, "k": 1024, "bias": 0}, {"rows": 300, "n": 1024, "k": 512, "bias": 1}),
    ),
    # one rounding of a long fp32 sum of exact fp8 products, plus summation order
    tolerance=Tolerance.accumulated(),
)
