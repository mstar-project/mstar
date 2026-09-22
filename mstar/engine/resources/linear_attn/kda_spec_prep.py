"""One launch prepares a KDA layer's verify step for the checkpoint recurrence (plan section 8.3
item 6). Per row and channel block: the checkpoint window and the pending prefix are read by
slot, the prefix's convolution runs over them and the block's over the window after the prefix
(the last ``W - 1`` inputs before the block, at the prefix's accepted length) and the block, in fp32
with the taps in tap order and SiLU like the reference; prefix positions past the accepted length
come out as no-op tokens (k = v = 0, raw gate and beta -1e4: decay 1, beta 0); the window after the
prefix (after the block, for a step that commits its block) is written back to the pool, the block
saved as the next pending prefix with its raw gates and betas. The outputs are packed ``prefix + block``
rows for ``kda_recurrent_checkpoint``: the prefix
part has one slot per pool prefix position (``KP`` = the largest block + 1), the block ``K1`` tokens, so a
block shorter than the pool's (a block length that follows the batch size, down to one token) packs the
same way.

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
def _load_combined(conv_state, prefix, slot, c, cm, idx, P: tl.constexpr, KP: tl.constexpr, W: tl.constexpr):
    """The row's combined pre-conv inputs (checkpoint window then pending prefix, ``KP`` slots) at
    positions ``idx [T, 1]`` for channels ``c [BP]``: fp32 ``[T, BP]``, zero outside ``[0, W - 1 + KP)``."""
    cc = c[None, :]
    m_w = (idx >= 0) & (idx < W - 1) & cm[None, :]
    m_p = (idx >= W - 1) & (idx < W - 1 + KP) & cm[None, :]
    win = tl.load(conv_state + slot * (3 * P) * (W - 1) + cc * (W - 1) + idx, mask=m_w, other=0.0).to(tl.float32)
    pre = tl.load(prefix + (slot * KP + (idx - (W - 1))) * (3 * P) + cc, mask=m_p, other=0.0).to(tl.float32)
    return win + pre


@triton.jit
def _kda_verify_prep_kernel(
    qkv, g_raw, beta_raw, conv_state, prefix, spec_g, spec_beta, spec_len, conv_w,
    out_qkv, out_g, out_beta, out_ckpt, out_plen, slot_ids, rows, stride_g_row, stride_b_row,
    P: tl.constexpr, H: tl.constexpr, K1: tl.constexpr, KP: tl.constexpr, TK: tl.constexpr, W: tl.constexpr,
    NW: tl.constexpr, BP: tl.constexpr, BH: tl.constexpr, COMMIT: tl.constexpr,
):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    c = pb * BP + tl.arange(0, BP)
    cm = c < 3 * P
    slot = tl.load(slot_ids + r).to(tl.int64)
    plen = tl.load(spec_len + slot)
    # padding rows address the sink, whose length is garbage: clamp before it indexes anything
    plen = tl.minimum(tl.maximum(plen, 0), KP)
    t = tl.arange(0, TK)[:, None]  # prefix slots / block positions
    tp = t < KP
    tm = t < K1
    T2: tl.constexpr = KP + K1
    acc_pre = tl.zeros([TK, BP], dtype=tl.float32)
    acc_blk = tl.zeros([TK, BP], dtype=tl.float32)
    for j in tl.static_range(W):
        wj = tl.load(conv_w + c * W + j, mask=cm, other=0.0).to(tl.float32)[None, :]
        acc_pre += _load_combined(conv_state, prefix, slot, c, cm, t + j, P, KP, W) * wj
        # the block's inputs: the window after the prefix (combined[plen : plen + W - 1]) then the block
        i = t + j
        from_win = _load_combined(conv_state, prefix, slot, c, cm, plen + i, P, KP, W)
        from_blk = tl.load(qkv + (r * K1 + (i - (W - 1))) * (3 * P) + c[None, :],
                           mask=(i >= W - 1) & (i < W - 1 + K1) & cm[None, :], other=0.0).to(tl.float32)
        acc_blk += tl.where(i < W - 1, from_win, from_blk) * wj
    y_pre = _silu(acc_pre)
    y_blk = _silu(acc_blk)
    jw = tl.arange(0, NW)[:, None]
    if COMMIT:
        # the step commits its block: the pool's window is the last W - 1 inputs before the block's end,
        # drawn from the window after the prefix and then the block itself
        iw = K1 + jw
        from_win = _load_combined(conv_state, prefix, slot, c, cm, plen + iw, P, KP, W)
        from_blk = tl.load(qkv + (r * K1 + (iw - (W - 1))) * (3 * P) + c[None, :],
                           mask=(iw >= W - 1) & (iw < W - 1 + K1) & cm[None, :], other=0.0).to(tl.float32)
        win_after = tl.where(iw < W - 1, from_win, from_blk)  # [NW, BP]
    else:
        win_after = _load_combined(conv_state, prefix, slot, c, cm, plen + jw, P, KP, W)  # [NW, BP]
    blk = tl.load(qkv + (r * K1 + t) * (3 * P) + c[None, :], mask=tm & cm[None, :], other=0.0)
    gm = c < P
    g_pre = tl.load(spec_g + (slot * KP + t) * P + c[None, :], mask=tp & gm[None, :], other=0.0).to(tl.float32)
    g_blk = tl.load(g_raw + (r * K1 + t) * stride_g_row + c[None, :], mask=tm & gm[None, :], other=0.0)
    # every read of the pool is done: the writes (the prefix part fills KP slots, the block K1)
    which = c // P
    o_base = which[None, :] * (rows * T2 * P) + (c - which * P)[None, :]
    out_dt = out_qkv.dtype.element_ty
    tl.store(out_qkv + o_base + (r * T2 + t) * P, tl.where(t < plen, y_pre, 0.0).to(out_dt), mask=tp & cm[None, :])
    tl.store(out_qkv + o_base + (r * T2 + KP + t) * P, y_blk.to(out_dt), mask=tm & cm[None, :])
    tl.store(conv_state + slot * (3 * P) * (W - 1) + c[None, :] * (W - 1) + jw,
             win_after.to(conv_state.dtype.element_ty), mask=(jw < W - 1) & cm[None, :])
    tl.store(prefix + (slot * KP + t) * (3 * P) + c[None, :], blk.to(prefix.dtype.element_ty), mask=tm & cm[None, :])
    g_dt = out_g.dtype.element_ty
    tl.store(out_g + (r * T2 + t) * P + c[None, :], tl.where(t < plen, g_pre, -1e4).to(g_dt), mask=tp & gm[None, :])
    tl.store(out_g + (r * T2 + KP + t) * P + c[None, :], g_blk.to(g_dt), mask=tm & gm[None, :])
    tl.store(spec_g + (slot * KP + t) * P + c[None, :], g_blk.to(spec_g.dtype.element_ty), mask=tm & gm[None, :])
    if pb == 0:
        hh = tl.arange(0, BH)[None, :]
        hm = hh < H
        b_pre = tl.load(spec_beta + (slot * KP + t) * H + hh, mask=tp & hm, other=0.0).to(tl.float32)
        b_blk = tl.load(beta_raw + (r * K1 + t) * stride_b_row + hh, mask=tm & hm, other=0.0)
        b_dt = out_beta.dtype.element_ty
        tl.store(out_beta + (r * T2 + t) * H + hh, tl.where(t < plen, b_pre, -1e4).to(b_dt), mask=tp & hm)
        tl.store(out_beta + (r * T2 + KP + t) * H + hh, b_blk.to(b_dt), mask=tm & hm)
        tl.store(spec_beta + (slot * KP + t) * H + hh, b_blk.to(spec_beta.dtype.element_ty), mask=tm & hm)
        if COMMIT:
            tl.store(out_ckpt + r, KP + K1 - 1)  # the state is stored after the block's last token
        else:
            tl.store(out_ckpt + r, plen - 1)
        tl.store(out_plen + r, plen)


def kda_verify_prep(
    qkv: torch.Tensor, g_raw: torch.Tensor, beta_raw: torch.Tensor, conv_state: torch.Tensor, spec,
    slot_ids: torch.Tensor, conv_w: torch.Tensor, rows: int, k1: int, h: int, d: int, commit_block: bool = False,
) -> tuple[torch.Tensor, ...]:
    """``qkv [rows * k1, 3P]`` (pre-conv), ``g_raw [rows * k1, P]`` and ``beta_raw [rows * k1, H]`` (both
    read with their row stride, so the layer's projection slices need no copy) of the rows' blocks;
    ``conv_state [slots, 3P, W - 1]`` and ``spec`` (``SpecBlocks``) are the pool's blocks,
    rewritten in place; ``slot_ids [rows]`` int32; ``conv_w [3P, W]``. Returns ``(q, k, v, g, beta,
    checkpoint_pos, prefix_len)``: ``[rows * (kp + k1), P]`` each (``beta [rows * (kp + k1), H]``) in
    ``qkv``'s dtype, each row's prefix padded to the pool's ``kp`` slots then its ``k1`` block tokens,
    ``checkpoint_pos [rows]`` int32 (the last prefix token, accepted length - 1; with ``commit_block``
    the block's last token, for a step that commits its block: a one-token block is the bonus token,
    always accepted, so nothing need pend) and ``prefix_len [rows]`` int32 (the real prefix tokens, the
    recurrence skips the slots after them). Static shapes, device tensors only: capturable."""
    p = h * d
    w = conv_w.shape[1]
    kp = spec.prefix.shape[1]
    assert 1 <= k1 <= kp, (k1, kp)
    t2 = kp + k1
    # the raw gates arrive as [T, H, D] (heads and dims dense) and the betas as [T, H]: the kernel reads
    # them as rows of P and H with the row strides they have
    qkv = qkv.reshape(rows * k1, 3 * p)
    g_raw = g_raw.view(rows * k1, p) if g_raw.dim() == 3 else g_raw
    beta_raw = beta_raw.view(rows * k1, h)
    assert g_raw.shape == (rows * k1, p) and g_raw.stride(1) == 1 and beta_raw.stride(1) == 1
    assert conv_state.shape[1:] == (3 * p, w - 1) and spec.prefix.shape[1:] == (kp, 3 * p)
    assert spec.g.shape[1:] == (kp, h, d) and spec.beta.shape[1:] == (kp, h) and spec.length.shape[1:] == (1,)
    for x in (qkv, conv_state, spec.prefix, spec.g, spec.beta, spec.length, slot_ids, conv_w):
        assert x.is_contiguous(), "contiguous inputs"
    assert slot_ids.numel() == rows and slot_ids.dtype == torch.int32
    out_qkv = torch.empty(3, rows * t2, p, dtype=qkv.dtype, device=qkv.device)
    out_g = torch.empty(rows * t2, p, dtype=qkv.dtype, device=qkv.device)
    out_beta = torch.empty(rows * t2, h, dtype=qkv.dtype, device=qkv.device)
    ckpt = torch.empty(rows, dtype=torch.int32, device=qkv.device)
    plen = torch.empty(rows, dtype=torch.int32, device=qkv.device)
    bp = min(256, triton.next_power_of_2(3 * p))
    grid = (rows, triton.cdiv(3 * p, bp))
    _kda_verify_prep_kernel[grid](
        qkv, g_raw, beta_raw, conv_state, spec.prefix, spec.g, spec.beta, spec.length, conv_w,
        out_qkv, out_g, out_beta, ckpt, plen, slot_ids, rows, g_raw.stride(0), beta_raw.stride(0),
        P=p, H=h, K1=k1, KP=kp, TK=triton.next_power_of_2(kp), W=w, NW=triton.next_power_of_2(w - 1),
        BP=bp, BH=triton.next_power_of_2(h), COMMIT=bool(commit_block), num_warps=4, enable_fp_fusion=False,
    )
    return out_qkv[0], out_qkv[1], out_qkv[2], out_g, out_beta, ckpt, plen
