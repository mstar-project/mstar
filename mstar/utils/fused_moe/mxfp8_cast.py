"""The bf16 rounding of an fp32 GEMM's output columns and the MXFP8 quantization of some of them, in one kernel.

K3.1's packed MoE-input GEMM writes fp32 ``[logits | z | gate_up]`` rows; the routed experts on TRT-LLM's
kernels take ``z`` as MXFP8 (e4m3 values, one ue8m0 scale per 32 along the row, linear layout), the shared
experts take ``gate_up`` in bf16. torch rounds the two to bf16 and FlashInfer's ``mxfp8_quantize`` then
quantizes ``z``: three launches a layer at decode. ``split_cast_mxfp8`` does all of it in one, ``z``'s bf16
values never stored.

The quantization is TRT-LLM's ``cvt_warp_fp16_to_mxfp8`` step by step: a group's scale is its bf16 values'
amax times ``rcp.approx.ftz(448)``, rounded up to a power of two (``cvt.rp.satfinite.ue8m0``, done here on the
bits: the biased exponent plus one where the mantissa is not 0; a value at or below 2^-127, zero included,
gives 2^-127); the values are multiplied by ``rcp.approx.ftz`` of the scale (exact for a power of two), or by 0
where the scale is 2^-127: its float is denormal, which TRT-LLM's ``SFValue != 0`` test flushes to zero under
ftz, so an all-zero or tiny block (amax * rcp(448) at or below 2^-127) comes out as zeros (B300 slot 10 caught
this: the first version multiplied by +inf and saturated such blocks to 448); 0 also for 2^127, whose reciprocal
the ftz flushes; then clamped to +-448 and rounded to e4m3 (``cvt.rn.satfinite``).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

GROUP = 32  # elements per ue8m0 scale


if triton is not None:

    @triton.jit
    def _rcp_approx_ftz(x):
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32,
                                         is_pure=True, pack=1)

    @triton.jit
    def _split_cast_mxfp8_kernel(h, s_h, q, sf, gu, LATENT: tl.constexpr, W: tl.constexpr,
                                 BLOCK: tl.constexpr):
        """Row t, columns [c0, c0 + BLOCK) of ``h``'s ``[z (LATENT) | gate_up]`` (W wide): gate_up's in bf16
        into ``gu``, z's as MXFP8 into ``q`` and ``sf``."""
        t = tl.program_id(0).to(tl.int64)
        c0 = tl.program_id(1) * BLOCK
        c = c0 + tl.arange(0, BLOCK)
        m = c < W
        x = tl.load(h + t * s_h + c, mask=m, other=0.0).to(tl.bfloat16)
        tl.store(gu + t * (W - LATENT) + (c - LATENT), x, mask=m & (c >= LATENT))
        if c0 < LATENT:
            NG: tl.constexpr = BLOCK // 32
            v = tl.reshape(x.to(tl.float32), [NG, 32])
            amax = tl.max(tl.abs(v), axis=1)
            sfv = amax * _rcp_approx_ftz(tl.full([NG], 448.0, tl.float32))
            bits = sfv.to(tl.int32, bitcast=True)
            exp = (bits >> 23) & 0xFF
            mant = bits & 0x7FFFFF
            e = tl.where(exp == 0, tl.where(mant > 0x400000, 1, 0), exp + (mant != 0).to(tl.int32))
            e = tl.minimum(e, 254)
            # rcp.approx.ftz of 2^(e - 127): 2^(127 - e) from its bits; 0 at e == 0 (TRT-LLM's denormal scale
            # tests equal to zero under ftz) and at e == 254 (the reciprocal flushes)
            scale = tl.where(e == 0, 0.0, ((254 - e) << 23).to(tl.float32, bitcast=True))
            y = v * scale[:, None]
            y = tl.maximum(tl.minimum(y, 448.0), -448.0)
            y = tl.reshape(y, [BLOCK]).to(tl.float8e4nv)
            tl.store(q + t * LATENT + c, y, mask=c < LATENT)
            g = c0 // 32 + tl.arange(0, NG)
            tl.store(sf + t * (LATENT // 32) + g, e.to(tl.uint8), mask=g < LATENT // 32)


def supported(x: torch.Tensor, latent: int) -> bool:
    return triton is not None and x.is_cuda and x.dtype == torch.float32 and latent % GROUP == 0


@torch.library.custom_op("mstar::split_cast_mxfp8", mutates_args=())
def split_cast_mxfp8(h: torch.Tensor, start: int, latent: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(z_q, z_sf, gate_up)`` from fp32 ``h`` (unit stride on columns): z = ``h[:, start:start + latent]``
    rounded to bf16 and quantized as ``mxfp8_quantize(z, is_sf_swizzled_layout=False)`` quantizes it (``z_q``
    e4m3 ``(T, latent)``, ``z_sf`` uint8 ue8m0 ``(T, latent / 32)``), gate_up = ``h[:, start + latent:]`` in bf16,
    contiguous."""
    T = h.shape[0]
    width = h.shape[1] - start
    q = torch.empty(T, latent, dtype=torch.float8_e4m3fn, device=h.device)
    sf = torch.empty(T, latent // GROUP, dtype=torch.uint8, device=h.device)
    gu = torch.empty(T, width - latent, dtype=torch.bfloat16, device=h.device)
    if T:
        assert h.stride(1) == 1 and latent % GROUP == 0
        block = 512
        _split_cast_mxfp8_kernel[(T, triton.cdiv(width, block))](
            h[:, start:], h.stride(0), q, sf, gu, LATENT=latent, W=width, BLOCK=block, num_warps=4)
    return q, sf, gu


@split_cast_mxfp8.register_fake
def _(h, start, latent):
    T, width = h.shape[0], h.shape[1] - start
    return (h.new_empty(T, latent, dtype=torch.float8_e4m3fn), h.new_empty(T, latent // GROUP, dtype=torch.uint8),
            h.new_empty(T, width - latent, dtype=torch.bfloat16))
