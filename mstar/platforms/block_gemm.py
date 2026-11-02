"""The block-scaled GEMM and its activation quantizer, per platform.

On a GPU with cuBLASLt's block-scaled GEMMs (``Platform.has_block_scaled_gemm``) these are ``torch._scaled_mm`` and
the Triton quantizer. Anywhere else they are the reference, which computes what the kernels stand for from the host-side
quantizer in fp32. A deployment doesn't run the reference: a model on such a platform keeps its bf16 GEMMs (see
``model.components.linear.apply_linear_precision``). Tests and parity checks do.
"""

from __future__ import annotations

import torch

from mstar.platforms import current_platform
from mstar.platforms.dispatch import PlatformDispatch
from mstar.utils.fp8_block import MXFP8, BlockFormat, BlockQuant, dequantize_block, quantize_block


def _reference_gemm(x: BlockQuant, weight: BlockQuant, fmt: BlockFormat) -> torch.Tensor:
    return (dequantize_block(x, fmt) @ dequantize_block(weight, fmt).t()).to(torch.bfloat16)


def _reference_quant(x: torch.Tensor, fmt: BlockFormat) -> BlockQuant:
    return quantize_block(x, fmt, tensor_scale=1.0)


def _scaled_mm(x: BlockQuant, weight: BlockQuant, fmt: BlockFormat) -> torch.Tensor:
    """``weight`` row-major ``[N, K // packing]``, as ``nn.Linear`` keeps it: its transpose is the col-major
    ``[K, N]`` the GEMM reads."""
    if fmt is MXFP8:
        return torch._scaled_mm(
            x.data, weight.data.t(), scale_a=x.scale, scale_b=weight.scale, out_dtype=torch.bfloat16,
        )
    recipe = [torch.nn.functional.ScalingType.BlockWise1x16.value, torch.nn.functional.ScalingType.TensorWise.value]
    swizzle = [torch.nn.functional.SwizzleType.SWIZZLE_32_4_4.value, torch.nn.functional.SwizzleType.NO_SWIZZLE.value]
    return torch._scaled_mm_v2(
        x.data, weight.data.t(), [x.scale, x.tensor_scale], recipe, swizzle,
        [weight.scale, weight.tensor_scale], recipe, swizzle, None, torch.bfloat16,
    )


def _cuda_gemm():
    return _scaled_mm if current_platform.has_block_scaled_gemm() else _reference_gemm


def _cuda_quant():
    if not current_platform.has_block_scaled_gemm():
        return _reference_quant
    from mstar.utils.quant_fp8 import quant_block

    return quant_block


block_scaled_gemm = PlatformDispatch("block_scaled_gemm", cuda=_cuda_gemm, default=lambda: _reference_gemm)
block_quant = PlatformDispatch("block_quant", cuda=_cuda_quant, default=lambda: _reference_quant)
