"""Stand-in for ``flashinfer.gemm``: the SM90 fp8 block-scale GEMM."""

from __future__ import annotations

import torch

from mstar_sim import clock


def _cost(input, weight, *args, **kwargs):
    m, k = input.shape[-2:] if input.dim() > 1 else (1, input.shape[-1])
    n = weight.shape[0]
    return 2.0 * m * n * k, float(input.numel() * input.element_size() + weight.numel() * weight.element_size()
                                  + m * n * 2)


@clock.kernel_fn("flashinfer.fp8_blockscale_gemm_sm90", cost=_cost, static=True)
def fp8_blockscale_gemm_sm90(input, weight, input_scale=None, weight_scale=None, out=None, out_dtype=None):
    """``input [M, K]`` (bf16, quantized per 128 columns inside, or fp8 with ``input_scale``)
    times ``weight [N, K]`` (fp8, 128x128 block scales): ``[M, N]``."""
    if out is not None:
        return out
    dtype = out_dtype or (input.dtype if input.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16)
    return torch.empty(*input.shape[:-1], weight.shape[0], dtype=dtype, device=input.device)
