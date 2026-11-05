"""``linear_dtype: fp8``: e4m3 weights with one scale per output channel, activations
quantised per token on the fly, the product on the platform's fp8 GEMM
(``mstar::fp8_rowwise_mm``, hipBLASLt on AMD GPUs, cuBLASLt on NVIDIA).

The format every fp8 GEMM takes natively (MI300X/MI325X, Hopper and up), where the
block-scaled ones (``block_scaled.py``) need Blackwell. The per-token quantisation is
plain torch, so a compiled forward fuses it into the producer of the activations,
usually the modulated norm, and the GEMM is the one opaque op per linear.

The output moves from the bf16 model's within the error of e4m3 rounding (3 mantissa
bits). Whether that is acceptable is a per-model quality call, which is why it is
opt-in.
"""

from __future__ import annotations

import torch

from mstar.kernels.fp8_gemm import FP8_ROWWISE_MM, quantize_rows
from mstar.model.components.quant_linear import LinearFormat, register_format
from mstar.platforms import current_platform

# hipBLASLt's fp8 kernels need both GEMM dimensions of the weight in multiples of 16
_ALIGN = 16

# An uncompiled forward (a Wan2.2 size outside compile_dit_sizes, say) would otherwise quantise in
# five eager kernels per linear, which cost more than the fp8 GEMM saves; this runs it as one.
_quantize_rows_fused = torch.compile(quantize_rows, dynamic=True)


def _quantize_tokens(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling() or x.device.type == "cpu":
        # traced into the caller's graph, where it fuses with whatever produced x
        return quantize_rows(x, fp8_dtype)
    return _quantize_rows_fused(x, fp8_dtype)


class Fp8RowwiseFormat(LinearFormat):
    """e4m3 rows in the GPU's native fp8 dtype (``Platform.fp8_dtype``: FNUZ on gfx942, OCP
    elsewhere), one fp32 scale per output channel, stored ``[1, N]`` as the GEMM takes it."""

    name = "fp8"
    buffers = ("weight", "weight_scale")

    def unavailable_reason(self, device: torch.device) -> str | None:
        # resolves the GEMM's implementation, so this runs at load, never inside a capture
        if current_platform.fp8_dtype(device) is None:
            arch = current_platform.arch()
            return f"{current_platform.name}{f' {arch}' if arch else ''} has no fp8 GEMM"
        return FP8_ROWWISE_MM.unavailable_reason()

    def check(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        if dtype != torch.bfloat16:
            raise ValueError(f"{self.name} replaces a bf16 GEMM; this linear's weight is {dtype}")
        if out_features % _ALIGN or in_features % _ALIGN:
            raise ValueError(
                f"{self.name} needs N and K multiples of {_ALIGN}; this linear is {out_features}x{in_features}"
            )

    def quantize_weight(self, weight: torch.Tensor, input_amax: float | None) -> dict[str, torch.Tensor | None]:
        del input_amax  # the activation scale is per token, computed on the fly
        data, scale = quantize_rows(weight, current_platform.fp8_dtype(weight.device))
        return {"weight": data, "weight_scale": scale.T.contiguous()}

    def linear(self, x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
        xq, x_scale = _quantize_tokens(x, weight.dtype)
        return FP8_ROWWISE_MM(xq, x_scale, weight, weight_scale, None, x.dtype)


FP8 = register_format(Fp8RowwiseFormat())
