"""MXFP4 (compressed-tensors ``mxfp4-pack-quantized``) reference codec (spec H).

Storage for a linear weight ``W[N, K]``:

* ``weight_packed``: ``uint8 [N, K // 2]``, two E2M1 codes per byte along the input
  dimension, **low nibble = element 2j, high nibble = element 2j + 1**.
* ``weight_scale``: ``uint8 [N, K // 32]``, one E8M0 exponent per group of 32 input
  elements; ``scale = 2 ** (code - 127)``.

E2M1 code (4 bits: sign, 2 exponent, 1 mantissa) -> value: ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``
with the sign bit. ``W[n, k] = fp4(code) * 2 ** (scale[n, k // 32] - 127)``.

The dequantized values are exactly representable in bf16, so ``dequant_mxfp4(...)`` is
lossless with respect to the stored weights.
"""
from __future__ import annotations

import torch

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
MXFP4_GROUP = 32


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """``uint8 [..., K/2]`` -> ``uint8 [..., K]`` of 4-bit codes, low nibble first."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unpack_fp4_codes`."""
    codes = codes.to(torch.uint8)
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    return (even | (odd << 4)).to(torch.uint8)


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale.to(torch.float32) - 127.0)


def dequant_mxfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``[N, K/2] uint8, [N, K/32] uint8 -> [N, K] dtype``."""
    n, half_k = weight_packed.shape
    k = half_k * 2
    assert weight_scale.shape == (n, k // MXFP4_GROUP), (weight_scale.shape, (n, k // MXFP4_GROUP))
    codes = unpack_fp4_codes(weight_packed).long()  # [N, K]
    vals = E2M1_VALUES.to(weight_packed.device)[codes]  # fp32 [N, K]
    scales = e8m0_to_float(weight_scale)  # [N, K/32]
    vals = vals.view(n, k // MXFP4_GROUP, MXFP4_GROUP) * scales.unsqueeze(-1)
    return vals.view(n, k).to(dtype)


def quant_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest MXFP4 quantizer (for tests/tiny models only; not the training
    quantizer). Scale per group of 32 = largest power of two such that
    ``max|w| / scale <= 6``."""
    n, k = weight.shape
    assert k % MXFP4_GROUP == 0
    w = weight.float().view(n, k // MXFP4_GROUP, MXFP4_GROUP)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    exp = torch.floor(torch.log2(amax / 6.0)).clamp(-127, 128)
    # make sure the max element fits after rounding (ceil when amax/scale would exceed 6)
    scale = torch.exp2(exp)
    scale = torch.where(amax / scale > 6.0, scale * 2, scale)
    exp = torch.log2(scale)
    scaled = (w / scale).view(n, k)
    # nearest E2M1 value
    table = E2M1_VALUES[:8].to(weight.device)
    mag = scaled.abs().unsqueeze(-1)
    idx = (mag - table).abs().argmin(dim=-1)  # 0..7
    # ties: prefer even mantissa like the hardware rounding is unknowable here; argmin is
    # deterministic (first minimum), good enough for tests
    sign = (scaled < 0) | ((scaled == 0) & torch.signbit(scaled))
    codes = idx + 8 * sign.long()
    packed = pack_fp4_codes(codes)
    e8m0 = (exp.view(n, k // MXFP4_GROUP) + 127).round().clamp(0, 255).to(torch.uint8)
    return packed, e8m0
