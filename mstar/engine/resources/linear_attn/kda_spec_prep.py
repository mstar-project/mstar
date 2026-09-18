"""One launch prepares a KDA layer's verify step for the checkpoint recurrence (plan section 8.3
item 6). Per row and channel block: the checkpoint window and the pending prefix are read by
slot, the prefix's convolution runs over them and the block's over the window after the prefix
(the last ``W - 1`` inputs before the block, at the prefix's accepted length) and the block, in fp32
with the taps in tap order and SiLU like the reference; prefix positions past the accepted length
come out as no-op tokens (k = v = 0, raw gate and beta -1e4: decay 1, beta 0); the window after the
prefix is written back to the pool, the block saved as the next pending prefix with its raw gates
and betas. The outputs are packed ``prefix + block`` rows for ``kda_recurrent_checkpoint``.

Replaces some sixty torch launches per layer (gathers, tap-by-tap products, masks, concatenations)
that a verify step of Kimi K3's 69 KDA layers turned into thousands of tiny kernels per step.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _silu(x):
    # torch's CUDA silu, operation for operation: x / (1 + expf(-x)) with IEEE division
    return libdevice.div_rn(x, 1.0 + libdevice.exp(-x))


@triton.jit
def _load_combined(conv_state, prefix, slot, c, cm, idx, P: tl.constexpr, K1: tl.constexpr, W: tl.constexpr):
    """The row's combined pre-conv inputs (checkpoint window then pending prefix) at positions
    ``idx [T, 1]`` for channels ``c [BP]``: fp32 ``[T, BP]``, zero outside ``[0, W - 1 + K1)``."""
    cc = c[None, :]
    m_w = (idx >= 0) & (idx < W - 1) & cm[None, :]
    m_p = (idx >= W - 1) & (idx < W - 1 + K1) & cm[None, :]
    win = tl.load(conv_state + slot * (3 * P) * (W - 1) + cc * (W - 1) + idx, mask=m_w, other=0.0).to(tl.float32)
    pre = tl.load(prefix + (slot * K1 + (idx - (W - 1))) * (3 * P) + cc, mask=m_p, other=0.0).to(tl.float32)
    return win + pre


@triton.jit
def _kda_verify_prep_kernel(
    qkv, g_raw, beta_raw, conv_state, prefix, spec_g, spec_beta, spec_len, conv_w,
    out_qkv, out_g, out_beta, out_ckpt, slot_ids, rows,
    P: tl.constexpr, H: tl.constexpr, K1: tl.constexpr, TK: tl.constexpr, W: tl.constexpr, NW: tl.constexpr,
    BP: tl.constexpr, BH: tl.constexpr,
):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    c = pb * BP + tl.arange(0, BP)
    cm = c < 3 * P
    slot = tl.load(slot_ids + r).to(tl.int64)
    plen = tl.load(spec_len + slot)
    # padding rows address the sink, whose length is garbage: clamp before it indexes anything
    plen = tl.minimum(tl.maximum(plen, 0), K1)
    t = tl.arange(0, TK)[:, None]  # prefix / block positions
    tm = t < K1
    T2: tl.constexpr = 2 * K1
    acc_pre = tl.zeros([TK, BP], dtype=tl.float32)
    acc_blk = tl.zeros([TK, BP], dtype=tl.float32)
    for j in tl.static_range(W):
        wj = tl.load(conv_w + c * W + j, mask=cm, other=0.0).to(tl.float32)[None, :]
        acc_pre += _load_combined(conv_state, prefix, slot, c, cm, t + j, P, K1, W) * wj
        # the block's inputs: the window after the prefix (combined[plen : plen + W - 1]) then the block
        i = t + j
        from_win = _load_combined(conv_state, prefix, slot, c, cm, plen + i, P, K1, W)
        from_blk = tl.load(qkv + (r * K1 + (i - (W - 1))) * (3 * P) + c[None, :],
                           mask=(i >= W - 1) & (i < W - 1 + K1) & cm[None, :], other=0.0).to(tl.float32)
        acc_blk += tl.where(i < W - 1, from_win, from_blk) * wj
    y_pre = _silu(acc_pre)
    y_blk = _silu(acc_blk)
    jw = tl.arange(0, NW)[:, None]
    win_after = _load_combined(conv_state, prefix, slot, c, cm, plen + jw, P, K1, W)  # [NW, BP]
    blk = tl.load(qkv + (r * K1 + t) * (3 * P) + c[None, :], mask=tm & cm[None, :], other=0.0)
    gm = c < P
    g_pre = tl.load(spec_g + (slot * K1 + t) * P + c[None, :], mask=tm & gm[None, :], other=0.0).to(tl.float32)
    g_blk = tl.load(g_raw + (r * K1 + t) * P + c[None, :], mask=tm & gm[None, :], other=0.0)
    # every read of the pool is done: the writes
    which = c // P
    o_base = which[None, :] * (rows * T2 * P) + (c - which * P)[None, :]
    out_dt = out_qkv.dtype.element_ty
    tl.store(out_qkv + o_base + (r * T2 + t) * P, tl.where(t < plen, y_pre, 0.0).to(out_dt), mask=tm & cm[None, :])
    tl.store(out_qkv + o_base + (r * T2 + K1 + t) * P, y_blk.to(out_dt), mask=tm & cm[None, :])
    tl.store(conv_state + slot * (3 * P) * (W - 1) + c[None, :] * (W - 1) + jw,
             win_after.to(conv_state.dtype.element_ty), mask=(jw < W - 1) & cm[None, :])
    tl.store(prefix + (slot * K1 + t) * (3 * P) + c[None, :], blk.to(prefix.dtype.element_ty), mask=tm & cm[None, :])
    g_dt = out_g.dtype.element_ty
    tl.store(out_g + (r * T2 + t) * P + c[None, :], tl.where(t < plen, g_pre, -1e4).to(g_dt), mask=tm & gm[None, :])
    tl.store(out_g + (r * T2 + K1 + t) * P + c[None, :], g_blk.to(g_dt), mask=tm & gm[None, :])
    tl.store(spec_g + (slot * K1 + t) * P + c[None, :], g_blk.to(spec_g.dtype.element_ty), mask=tm & gm[None, :])
    if pb == 0:
        hh = tl.arange(0, BH)[None, :]
        hm = hh < H
        b_pre = tl.load(spec_beta + (slot * K1 + t) * H + hh, mask=tm & hm, other=0.0).to(tl.float32)
        b_blk = tl.load(beta_raw + (r * K1 + t) * H + hh, mask=tm & hm, other=0.0)
        b_dt = out_beta.dtype.element_ty
        tl.store(out_beta + (r * T2 + t) * H + hh, tl.where(t < plen, b_pre, -1e4).to(b_dt), mask=tm & hm)
        tl.store(out_beta + (r * T2 + K1 + t) * H + hh, b_blk.to(b_dt), mask=tm & hm)
        tl.store(spec_beta + (slot * K1 + t) * H + hh, b_blk.to(spec_beta.dtype.element_ty), mask=tm & hm)
        tl.store(out_ckpt + r, plen - 1)


def kda_verify_prep(
    qkv: torch.Tensor, g_raw: torch.Tensor, beta_raw: torch.Tensor, conv_state: torch.Tensor, spec,
    slot_ids: torch.Tensor, conv_w: torch.Tensor, rows: int, k1: int, h: int, d: int,
) -> tuple[torch.Tensor, ...]:
    """``qkv [rows * k1, 3P]`` (pre-conv), ``g_raw [rows * k1, P]``, ``beta_raw [rows * k1, H]`` of the
    rows' blocks; ``conv_state [slots, 3P, W - 1]`` and ``spec`` (``SpecBlocks``) are the pool's blocks,
    rewritten in place; ``slot_ids [rows]`` int32; ``conv_w [3P, W]``. Returns ``(q, k, v, g, beta,
    checkpoint_pos)``: ``[rows * 2 k1, P]`` each (``beta [rows * 2 k1, H]``) in ``qkv``'s dtype, each
    row's prefix then its block, and ``checkpoint_pos [rows]`` int32 (accepted length - 1). Static
    shapes, device tensors only: capturable."""
    p = h * d
    w = conv_w.shape[1]
    t2 = 2 * k1
    # the raw gates arrive as [T, H, D] and the betas as [T, H]: the kernel reads them flat
    qkv, g_raw, beta_raw = qkv.reshape(rows * k1, 3 * p), g_raw.reshape(rows * k1, p), beta_raw.reshape(rows * k1, h)
    assert conv_state.shape[1:] == (3 * p, w - 1) and spec.prefix.shape[1:] == (k1, 3 * p)
    assert spec.g.shape[1:] == (k1, h, d) and spec.beta.shape[1:] == (k1, h) and spec.length.shape[1:] == (1,)
    for x in (qkv, g_raw, beta_raw, conv_state, spec.prefix, spec.g, spec.beta, spec.length, slot_ids, conv_w):
        assert x.is_contiguous(), "contiguous inputs"
    assert slot_ids.numel() == rows and slot_ids.dtype == torch.int32
    out_qkv = torch.empty(3, rows * t2, p, dtype=qkv.dtype, device=qkv.device)
    out_g = torch.empty(rows * t2, p, dtype=qkv.dtype, device=qkv.device)
    out_beta = torch.empty(rows * t2, h, dtype=qkv.dtype, device=qkv.device)
    ckpt = torch.empty(rows, dtype=torch.int32, device=qkv.device)
    bp = min(256, triton.next_power_of_2(3 * p))
    grid = (rows, triton.cdiv(3 * p, bp))
    _kda_verify_prep_kernel[grid](
        qkv, g_raw, beta_raw, conv_state, spec.prefix, spec.g, spec.beta, spec.length, conv_w,
        out_qkv, out_g, out_beta, ckpt, slot_ids, rows,
        P=p, H=h, K1=k1, TK=triton.next_power_of_2(k1), W=w, NW=triton.next_power_of_2(w - 1),
        BP=bp, BH=triton.next_power_of_2(h), num_warps=4, enable_fp_fusion=False,
    )
    return out_qkv[0], out_qkv[1], out_qkv[2], out_g, out_beta, ckpt
