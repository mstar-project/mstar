"""Triton kernels for the FastConformer encoder's memory-bound stretches.

On a 32 x 1024-frame Parakeet batch about half the encoder's GPU time went to
elementwise kernels around the GEMMs. Two stretches are fused here:

- the conv module's GLU, padding mask, depthwise conv over time, its bias and
  SiLU: six kernels and two transposes become one, in the ``(B, T, C)`` layout
  the pointwise convs use;
- the subsampling's bias add, padding mask and ReLU after each conv that a
  ReLU follows: three passes over the largest activations become one.

Each rounds to the activation dtype where the eager ops do (after GLU, after
the depthwise conv, after the bias add), so they agree with the eager path up
to the conv's accumulation order and ``exp``'s last bit. Off CUDA, or without
Triton, the callers keep the eager ops.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def use_triton(x: torch.Tensor) -> bool:
    return _TRITON_AVAILABLE and x.is_cuda


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["T"])
    def _glu_dwconv_kernel(
        x_ptr,  # (B, T, 2C): values, then gates
        valid_ptr,  # (B, T) bool, read when HAS_VALID
        w_ptr,  # (C, K)
        b_ptr,  # (C,)
        out_ptr,  # (B, T, C)
        T,
        C: tl.constexpr,
        K: tl.constexpr,
        PAD: tl.constexpr,
        HAS_VALID: tl.constexpr,
        SILU: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """One ``(BLOCK_T, BLOCK_C)`` output tile of one row. Each of the K
        taps reloads its shifted input tile (from cache after the first) and
        recomputes the GLU, rather than staging a halo."""
        b = tl.program_id(2).to(tl.int64)
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C
        x_row = x_ptr + b * T * (2 * C)
        # the bias starts the sum, as in PyTorch's depthwise kernel
        acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        acc += tl.load(b_ptr + c, mask=c_ok, other=0.0).to(tl.float32)[None, :]
        for k in tl.static_range(K):
            s = t + (k - PAD)
            s_ok = (s >= 0) & (s < T)
            if HAS_VALID:
                s_ok = s_ok & (tl.load(valid_ptr + b * T + s, mask=s_ok, other=0) != 0)
            m = s_ok[:, None] & c_ok[None, :]
            off = s[:, None] * (2 * C) + c[None, :]
            a = tl.load(x_row + off, mask=m, other=0.0).to(tl.float32)
            g = tl.load(x_row + off + C, mask=m, other=0.0).to(tl.float32)
            # F.glu rounds to the activation dtype before the conv reads it
            glu = (a / (1.0 + tl.exp(-g))).to(out_ptr.dtype.element_ty).to(tl.float32)
            acc += glu * tl.load(w_ptr + c * K + k, mask=c_ok, other=0.0).to(tl.float32)[None, :]
        y = acc.to(out_ptr.dtype.element_ty)
        if SILU:
            yf = y.to(tl.float32)
            y = (yf / (1.0 + tl.exp(-yf))).to(out_ptr.dtype.element_ty)
        out = out_ptr + b * T * C + t[:, None] * C + c[None, :]
        tl.store(out, y, mask=(t < T)[:, None] & c_ok[None, :])

    @triton.jit(do_not_specialize=["TF"])
    def _bias_relu_mask_kernel(
        x_ptr,  # (B, C, T, F), in place
        b_ptr,  # (C,)
        len_ptr,  # (B,) frames per row, read when HAS_LEN
        C,
        F,
        TF,
        HAS_LEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One ``BLOCK`` chunk of one ``(row, channel)`` plane of T x F."""
        bc = tl.program_id(1)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < TF
        ptrs = x_ptr + bc.to(tl.int64) * TF + offs
        x = tl.load(ptrs, mask=ok, other=0.0).to(tl.float32)
        # the conv's output plus its bias, rounded as eager's separate add is
        y = (x + tl.load(b_ptr + bc % C).to(tl.float32)).to(x_ptr.dtype.element_ty)
        y = tl.maximum(y, 0.0)
        if HAS_LEN:
            y = tl.where(offs < tl.load(len_ptr + bc // C) * F, y, 0.0)
        tl.store(ptrs, y, mask=ok)


def glu_depthwise_conv(
    x: torch.Tensor, valid: torch.Tensor | None, weight: torch.Tensor, bias: torch.Tensor,
    left_pad: int, silu: bool,
) -> torch.Tensor:
    """``(B, T, 2C)`` -> ``(B, T, C)``: ``F.glu``, frames where ``valid`` is
    False zeroed, a depthwise conv over time with ``weight`` ``(C, 1, K)``,
    ``left_pad`` zero frames before the clip and ``K - 1 - left_pad`` after,
    its ``bias``, then SiLU when ``silu``."""
    b, t, c2 = x.shape
    c, k = weight.shape[0], weight.shape[-1]
    assert c2 == 2 * c
    x = x.contiguous()
    out = torch.empty(b, t, c, dtype=x.dtype, device=x.device)
    block_t, block_c = 16, 128
    _glu_dwconv_kernel[(triton.cdiv(t, block_t), triton.cdiv(c, block_c), b)](
        x, valid.view(torch.uint8) if valid is not None else x, weight, bias, out, t,
        C=c, K=k, PAD=left_pad, HAS_VALID=valid is not None, SILU=silu,
        BLOCK_T=block_t, BLOCK_C=block_c,
    )
    return out


def bias_relu_mask_(x: torch.Tensor, bias: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """On a bias-free conv output ``(B, C, T, F)``, in place when it is
    contiguous: add the conv's ``bias``, ReLU, and zero frames at or past
    ``lengths``."""
    b, c, t, f = x.shape
    x = x.contiguous()
    block = 1024
    _bias_relu_mask_kernel[(triton.cdiv(t * f, block), b * c)](
        x, bias, lengths if lengths is not None else x, c, f, t * f,
        HAS_LEN=lengths is not None, BLOCK=block,
    )
    return x
