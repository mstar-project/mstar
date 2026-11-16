"""Load-time 128x128 e4m3 quantization for the dense FP8 linear kernel.

This format uses the same weight and fp32 scale layout as the dense FP8
checkpoint linears. It quantizes a loaded bf16 weight once; the existing
``fp8_block_linear`` kernel handles the forward and its CPU reference.
"""

import torch

from mstar.model.components.fp8_linear import fp8_block_linear
from mstar.model.components.quant_linear import LinearFormat, register_format
from mstar.platforms import current_platform
from mstar.utils.fp8_block import FP8_DTYPE


class DenseFp8Format(LinearFormat):
    name = "dense_fp8"
    buffers = ("weight", "weight_scale_inv")
    block_size = (128, 128)

    def unavailable_reason(self, device: torch.device) -> str | None:
        if current_platform.has_cuda_capability(9, 0, device):
            return None
        return "dense FP8 linears need an NVIDIA SM90 or newer GPU"

    def check(self, out_features: int, in_features: int, dtype: torch.dtype) -> None:
        if dtype != torch.bfloat16:
            raise ValueError(f"dense_fp8 replaces a bf16 GEMM; this linear's weight is {dtype}")
        if out_features % 128 or in_features % 128:
            raise ValueError(
                f"dense_fp8 needs N and K multiples of 128; this linear is ({out_features}, {in_features})"
            )

    def quantize_weight(self, weight: torch.Tensor, input_amax: float | None) -> dict[str, torch.Tensor]:
        del input_amax  # the dense kernel uses bf16 input for short rows and quantizes it for long rows
        n, k = weight.shape
        blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
        scale = blocks.abs().amax(dim=(1, 3)).div_(448.0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quant = (blocks / scale[:, None, :, None]).reshape(n, k).to(FP8_DTYPE)
        if weight.is_cuda:
            # Probe the fast implementations and reserve split-K counters before
            # compilation/capture, as Fp8Linear.process_weights_after_loading does.
            from mstar.utils import dense_fp8

            dense_fp8.reserve(weight.device)
        return {"weight": quant.view(torch.uint8).contiguous(), "weight_scale_inv": scale.contiguous()}

    def linear(self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return fp8_block_linear(x, weight, scale, self.block_size)


DENSE_FP8 = register_format(DenseFp8Format())
