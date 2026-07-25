"""Test-support helpers for the Kimi-K2.7 compressed-tensors path.

NOT part of the serving path. These helpers exist only to let the test suite
fabricate a synthetic quantized checkpoint and its exact bf16 reference, so a
golden can assert the real load path (``quantization.py`` +
``weight_loader.py``) reproduces the reference bit-for-bit. Nothing here is
imported by the model or the loader at serve time.

The real load-path primitives (``pack_int32`` / ``unpack_int32`` /
``dequantize_weight`` / ``dequant_compressed_tensors_stream`` /
``CompressedTensorsQuantConfig``) live in ``quantization.py``; this module builds
on them.
"""
from __future__ import annotations

import torch

from mstar.model.kimi_k2_7.quantization import dequantize_weight, pack_int32


def fake_quantize_weight(
    weight: torch.Tensor,
    *,
    num_bits: int,
    group_size: int,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` group-wise and return ``(packed, scale, dequant)``.

    A test/harness helper (not used at serve time): produces the on-disk
    compressed-tensors tensors *and* the exact bf16 result they dequantize back
    to, so a golden can assert the loader reproduces ``dequant`` bit-for-bit. Only
    symmetric INT-style quantization is implemented (Kimi's scheme).

    ``dequant`` is derived from the *returned* ``scale`` via :func:`dequantize_weight`,
    so it stays consistent with whatever ``scale_dtype`` the scale is stored at.
    Pass ``scale_dtype=torch.bfloat16`` to match a real compressed-tensors
    checkpoint (whose ``weight_scale`` is stored in the model dtype) — otherwise
    the loader's bf16-scale dequant would differ from an fp32-scale reference in
    the low bits.
    """
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
