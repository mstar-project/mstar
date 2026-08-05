"""Test-support helpers for the GLM-5.2 fp8-block path.

NOT part of the serving path (the real checkpoint arrives pre-quantized).
These fabricate a synthetic fp8-block checkpoint and its exact bf16
reference so goldens can assert the load path (``quantization.py`` +
``weight_loader.py``) reproduces the reference bit-for-bit — the same role
``kimi_k2_7/_testing.py`` plays for compressed-tensors.
"""
from __future__ import annotations

import torch

from mstar.model.glm52.quantization import FP8_DTYPE, dequantize_fp8_block_weight


def fake_quantize_fp8_block(
    weight: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fp8 weight, fp32 scale_inv, exact bf16 dequantized reference).

    Per block: scale = amax / 448 (e4m3 max normal), quantize w / scale to
    e4m3, store scale as ``weight_scale_inv`` (the multiply-back convention).
    """
    out_f, in_f = weight.shape
    bo, bi = block_size
    n_bo, n_bi = -(-out_f // bo), -(-in_f // bi)

    w = weight.to(torch.float32)
    padded = torch.zeros(n_bo * bo, n_bi * bi, dtype=torch.float32)
    padded[:out_f, :in_f] = w
    blocks = padded.view(n_bo, bo, n_bi, bi)
    amax = blocks.abs().amax(dim=(1, 3))  # (n_bo, n_bi)
    scale_inv = amax / 448.0  # e4m3 max normal value
    scale_inv = torch.where(scale_inv == 0, torch.ones_like(scale_inv), scale_inv)

    scale_bc = scale_inv.repeat_interleave(bo, dim=0)[:out_f]
    scale_bc = scale_bc.repeat_interleave(bi, dim=1)[:, :in_f]
    w_fp8 = (w / scale_bc).to(FP8_DTYPE)

    dequant = dequantize_fp8_block_weight(w_fp8, scale_inv, block_size=block_size)
    return w_fp8, scale_inv, dequant
