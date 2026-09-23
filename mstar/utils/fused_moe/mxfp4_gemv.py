"""Small-batch routed-expert kernels that read FlashInfer's SM90 mixed-GEMM MXFP4 layout.

For one to a few decode tokens the CUTLASS grouped GEMM is occupancy-bound (16 active experts
give it too few tiles) and reaches ~10% of HBM bandwidth. These Triton kernels stream each
(token, expert) pair's weights with one program per 16-row block, which fills the GPU at any
batch size, and they read the *interleaved* weights in place, so no second weight copy exists.

Layout facts (derived and verified by ``bench/kernels/derive_sm90_interleave.py`` and
``sm90_word_map.py``): the interleave is a bijection on 16-row x 64-column tiles; the packed
``[rows, K/2]`` bytes keep row-major addressing, but the 32-bit word at (dst row ``r'``, word
column ``j``) holds the 8 logical values ``(r, c), (r, c+1), (r, c+8), (r, c+9)`` for
``r in {r' % 8, r' % 8 + 8}`` with ``c = (j // 4) * 32 + (r' // 8) * 16 + (j % 4) * 2``, at the
fixed bit positions below. E8M0 scales use the folded layout of
``interleave_moe_scales_for_sm90_mixed_gemm``: bytes ``[E, rows/64, K/128, 16, 16]`` with
``scale(row, group) = S[e, row // 64, group // 4, row % 16, ((row % 64) // 16) * 4 + group % 4]``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4(w, p0: tl.constexpr, p1: tl.constexpr, p2: tl.constexpr, p3: tl.constexpr):
    """E2M1 value (unscaled, fp32) of one slot of an interleaved int32 word: bit p0 = mantissa,
    p1/p2 = exponent low/high, p3 = sign. Builds the fp32 bit pattern directly: normal values
    are (1 + m/2) * 2^(e-1) -> exponent field 126 + e, mantissa field m << 22; e == 0 gives
    0 or 0.5 (0x3F000000 = 0.5)."""
    m = (w >> p0) & 1
    e = ((w >> p1) & 1) | (((w >> p2) & 1) << 1)
    s = (w >> p3) & 1
    bits = tl.where(e == 0, m * 0x3F000000, ((126 + e) << 23) | (m << 22)) | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _block_dot(w_ptr, s_ptr, x_ptr, rblk, K: tl.constexpr, KB: tl.constexpr, BQ: tl.constexpr,
               q_begin, q_end):
    """``acc[16] = W[rblk : rblk + 16, :] @ x`` for one expert whose interleaved bytes (int32
    view) start at ``w_ptr`` and folded scale bytes at ``s_ptr``; ``x_ptr``: K bf16 values.
    ``BQ`` 64-column partitions are decoded per iteration. Returns the two 8-row halves
    (logical rows 0..7 and 8..15 of the block)."""
    rp = tl.arange(0, 16)[:, None, None]
    qq = tl.arange(0, BQ)[None, :, None]
    wc = tl.arange(0, 8)[None, None, :]
    words_per_row: tl.constexpr = K // 8
    acc_lo = tl.zeros((8,), dtype=tl.float32)
    acc_hi = tl.zeros((8,), dtype=tl.float32)
    mslice = (rblk % 64) // 16
    B = rblk // 64
    fm_lo = rp % 8
    for q0 in range(q_begin, q_end, BQ):
        q = q0 + qq
        w = tl.load(w_ptr + (rblk + rp) * words_per_row + q * 8 + wc)  # [16, BQ, 8] int32
        col = q * 64 + (wc // 4) * 32 + (rp // 8) * 16 + (wc % 4) * 2
        s_off = (B * KB + q // 2) * 256 + mslice * 4 + 2 * (q % 2) + wc // 4
        f_lo = tl.exp2((tl.load(s_ptr + s_off + fm_lo * 16).to(tl.int32) - 127).to(tl.float32))
        f_hi = tl.exp2((tl.load(s_ptr + s_off + (fm_lo + 8) * 16).to(tl.int32) - 127).to(tl.float32))
        x0 = tl.load(x_ptr + col).to(tl.float32)
        x1 = tl.load(x_ptr + col + 1).to(tl.float32)
        x8 = tl.load(x_ptr + col + 8).to(tl.float32)
        x9 = tl.load(x_ptr + col + 9).to(tl.float32)
        c_lo = (_fp4(w, 6, 7, 8, 15) * x0 + _fp4(w, 22, 23, 24, 31) * x1
                + _fp4(w, 0, 1, 2, 9) * x8 + _fp4(w, 16, 17, 18, 25) * x9) * f_lo
        c_hi = (_fp4(w, 3, 4, 5, 12) * x0 + _fp4(w, 19, 20, 21, 28) * x1
                + _fp4(w, 13, 10, 11, 14) * x8 + _fp4(w, 29, 26, 27, 30) * x9) * f_hi
        # dst rows r' and r' + 8 hold the two column halves of the same logical rows
        r_lo = tl.sum(tl.sum(c_lo, axis=2), axis=1)  # [16]
        r_hi = tl.sum(tl.sum(c_hi, axis=2), axis=1)
        acc_lo += tl.sum(tl.reshape(r_lo, (2, 8)), axis=0)
        acc_hi += tl.sum(tl.reshape(r_hi, (2, 8)), axis=0)
    return acc_lo, acc_hi


@triton.jit
def _gate_up_kernel(x_ptr, idx_ptr, w_ptr, s_ptr, gu_ptr, TOPK: tl.constexpr, K: tl.constexpr,
                    I: tl.constexpr, BQ: tl.constexpr, SPLIT: tl.constexpr):
    """gu[split, p, blk*16 : +16] = W13[e, blk*16 : +16, K-slice] @ x[t, K-slice] for pair
    p = (token, j), one 16-row block of the ``[up (I) | gate (I)]`` fc1 and one of SPLIT
    K-slices (fp32 partials; the activation kernel sums the slices)."""
    p = tl.program_id(0)
    blk = tl.program_id(1)
    split = tl.program_id(2)
    t = p // TOPK
    e = tl.load(idx_ptr + p).to(tl.int64)
    KB: tl.constexpr = K // 128
    rows: tl.constexpr = 2 * I
    QPS: tl.constexpr = (K // 64) // SPLIT
    we = w_ptr + e * (rows * (K // 8))
    se = s_ptr + e * ((rows // 64) * KB * 256)
    lo, hi = _block_dot(we, se, x_ptr + t.to(tl.int64) * K, blk * 16, K, KB, BQ, split * QPS, (split + 1) * QPS)
    r8 = tl.arange(0, 8)
    base = gu_ptr + (split * tl.num_programs(0) + p).to(tl.int64) * rows + blk * 16
    tl.store(base + r8, lo)
    tl.store(base + 8 + r8, hi)


@triton.jit
def _situ_kernel(gu_ptr, h_ptr, I: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr, P, beta, lin_beta):
    """h[p, i] = SiTU(gate = sum_s gu[s, p, I + i], up = sum_s gu[s, p, i])."""
    p = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < I
    u = tl.zeros((BLOCK,), dtype=tl.float32)
    g = tl.zeros((BLOCK,), dtype=tl.float32)
    for sp in tl.static_range(SPLIT):
        row = (sp * P + p).to(tl.int64) * (2 * I)
        u += tl.load(gu_ptr + row + off, mask=mask, other=0.0)
        g += tl.load(gu_ptr + row + I + off, mask=mask, other=0.0)
    a = beta * (2.0 / (1.0 + tl.exp(-2.0 * g / beta)) - 1.0) * (1.0 / (1.0 + tl.exp(-g)))
    u2 = lin_beta * (2.0 / (1.0 + tl.exp(-2.0 * u / lin_beta)) - 1.0)
    tl.store(h_ptr + p.to(tl.int64) * I + off, (a * u2).to(tl.bfloat16), mask=mask)


@triton.jit
def _down_kernel(h_ptr, idx_ptr, w_ptr, s_ptr, out_ptr, I: tl.constexpr, N: tl.constexpr,
                 BQ: tl.constexpr, NB: tl.constexpr):
    """out[p, blk*16 : +16] = W2[e, blk*16 : +16, :] @ h[p] for NB consecutive blocks."""
    p = tl.program_id(0)
    e = tl.load(idx_ptr + p).to(tl.int64)
    KB: tl.constexpr = I // 128
    we = w_ptr + e * (N * (I // 8))
    se = s_ptr + e * ((N // 64) * KB * 256)
    r8 = tl.arange(0, 8)
    for b in tl.static_range(NB):
        blk = tl.program_id(1) * NB + b
        lo, hi = _block_dot(we, se, h_ptr + p.to(tl.int64) * I, blk * 16, I, KB, BQ, 0, I // 64)
        tl.store(out_ptr + p.to(tl.int64) * N + blk * 16 + r8, lo)
        tl.store(out_ptr + p.to(tl.int64) * N + blk * 16 + 8 + r8, hi)


@triton.jit
def _reduce_kernel(part_ptr, w_ptr, out_ptr, TOPK: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    """out[t, n] = sum_j w[t, j] * part[t*TOPK + j, n] (deterministic order)."""
    t = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        wj = tl.load(w_ptr + t * TOPK + j)
        acc += wj * tl.load(part_ptr + (t * TOPK + j).to(tl.int64) * N + off, mask=mask, other=0.0)
    tl.store(out_ptr + t.to(tl.int64) * N + off, acc.to(tl.bfloat16), mask=mask)


def sm90_moe_gemv(x: torch.Tensor, w13_il: torch.Tensor, s13_il: torch.Tensor, w2_il: torch.Tensor,
                  s2_il: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
                  beta: float, lin_beta: float, bq1: int = 4, bq2: int = 2, nb2: int = 1,
                  num_warps: int = 2, split: int = 1, num_stages: int = 3) -> torch.Tensor:
    """Routed experts for a few tokens on the interleaved MXFP4 weights.

    ``x [T, K]`` bf16; ``w13_il [E, 2I, K/2]`` / ``w2_il [E, K, I/2]`` uint8 interleaved;
    ``s13_il`` / ``s2_il`` the folded scales (any dtype view of the byte layout);
    ``topk_idx [T, k]`` ints, ``topk_weight [T, k]`` fp32. Returns ``[T, K]`` in ``x.dtype``.
    Requires ``K % 128 == 0``, ``I % 128 == 0`` (the folded scale layout) and ``I % 16 == 0``.
    """
    T, K = x.shape
    E, rows, _ = w13_il.shape
    I = rows // 2
    k = topk_idx.shape[1]
    assert K % 128 == 0 and I % 128 == 0, (K, I)
    assert (K // 64) % bq1 == 0 and (I // 64) % bq2 == 0 and (K // 16) % nb2 == 0, (bq1, bq2, nb2)
    idx = topk_idx.reshape(-1).to(torch.int32).contiguous()
    x = x.contiguous()
    assert (K // 64) % (split * bq1) == 0, (split, bq1)
    P = T * k
    gu = torch.empty(split * P, rows, dtype=torch.float32, device=x.device)
    _gate_up_kernel[(P, rows // 16, split)](x, idx, w13_il.view(torch.int32), s13_il.view(torch.uint8), gu,
                                            TOPK=k, K=K, I=I, BQ=bq1, SPLIT=split, num_warps=num_warps,
                                            num_stages=num_stages)
    h = torch.empty(P, I, dtype=torch.bfloat16, device=x.device)
    _situ_kernel[(P, triton.cdiv(I, 256))](gu, h, I=I, BLOCK=256, SPLIT=split, P=P, beta=float(beta),
                                           lin_beta=float(lin_beta))
    partial = torch.empty(P, K, dtype=torch.float32, device=x.device)
    _down_kernel[(P, K // 16 // nb2)](h, idx, w2_il.view(torch.int32), s2_il.view(torch.uint8), partial,
                                      I=I, N=K, BQ=bq2, NB=nb2, num_warps=num_warps, num_stages=num_stages)
    out = torch.empty(T, K, dtype=torch.bfloat16, device=x.device)
    _reduce_kernel[(T, triton.cdiv(K, 1024))](partial, topk_weight.to(torch.float32).contiguous(), out,
                                              TOPK=k, N=K, BLOCK=1024)
    return out.to(x.dtype)
