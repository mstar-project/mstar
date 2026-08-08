"""Test-support helpers for the Kimi-K2.7 compressed-tensors path.

NOT part of the serving path. These helpers exist only to let the test suite
fabricate a synthetic quantized checkpoint and its exact bf16 reference, so a
golden can assert the real load path (:mod:`mstar.model.components.quantization`
+ ``weight_loader.py``) reproduces the reference bit-for-bit. Nothing here is
imported by the model or the loader at serve time.

The real load-path primitives (``pack_int32`` / ``unpack_int32`` /
``dequantize_weight`` / ``dequant_compressed_tensors_stream`` /
``CompressedTensorsQuantConfig``) live in
:mod:`mstar.model.components.quantization.compressed_tensors`; this module builds
on them.
"""
from __future__ import annotations

import torch

from mstar.model.components.quantization import dequantize_weight, pack_int32


def fake_quantize_weight(
    weight: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed weights, stored scales, and the exact dequantized reference."""
    if not symmetric:
        raise NotImplementedError("fake_quantize_weight: only symmetric is implemented")
    out_f, in_f = weight.shape
    gs = in_f if group_size in (-1, None) else group_size
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {gs}")

    qmax = (1 << (num_bits - 1)) - 1  # 7 for INT4
    qmin = -(1 << (num_bits - 1))     # -8
    w = weight.to(torch.float32).reshape(out_f, in_f // gs, gs)
    scale = w.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale).clamp(qmin, qmax)  # signed [qmin, qmax], fp32 scale

    q_unsigned = (q + float(1 << (num_bits - 1))).reshape(out_f, in_f).to(torch.int64)
    packed = pack_int32(q_unsigned, num_bits)
    scale_2d = scale.squeeze(-1).to(scale_dtype)
    dequant = dequantize_weight(
        packed, scale_2d, num_bits=num_bits, group_size=group_size, symmetric=symmetric,
    )
    return packed, scale_2d, dequant
