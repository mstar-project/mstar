"""Small-batch path of a block-fp8 MoE with a bf16 shared expert: a sigmoid top-k router,
then the routed and shared experts, where the grouped-GEMM runner would mostly multiply padding.

Up to a per-GPU token count (``_Tuning.pair_max_tokens``) gate/up and down are GEMV kernels
over (token, expert) pairs; above it experts repeat across the batch, and grouped kernels read
each expert's weights once. ``swiglu_limit`` picks the clamped SwiGLU or the plain one.
"""
from __future__ import annotations

import dataclasses
import functools

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait


@dataclasses.dataclass(frozen=True)
class _Tuning:
    """One GPU's launch choices. Up to ``pair_max_tokens`` tokens the experts run as per-pair
    GEMVs, above it grouped by expert. ``gate_up`` / ``down`` are the per-pair tiles as
    ``(max tokens, tiles)`` rows, the first row that fits wins (None fits all): few pairs
    want narrow row blocks with long K steps so enough programs are in flight, more pairs
    wider blocks. bjs / bks tile the shared expert (tl.dot needs bjs >= 16). ``router``: the
    logits kernel's K step, most K splits and warps."""

    pair_max_tokens: int
    gate_up: tuple[tuple[int | None, dict], ...]
    down: tuple[tuple[int | None, dict], ...]
    router: dict


# Tiles from sweeps on each GPU: the H200 at hidden 4096, the H100 at hidden 6144.
_H200 = _Tuning(
    pair_max_tokens=10,
    gate_up=((3, dict(bj=8, bk=1024, bjs=32, bks=256, warps=4)),
             (None, dict(bj=16, bk=256, bjs=16, bks=128, warps=4))),
    down=((2, dict(bn=32, bc=256, warps=8)), (None, dict(bn=32, bc=128, warps=4))),
    router=dict(bk=256, splits=16, warps=2),
)
_H100 = _Tuning(
    pair_max_tokens=7,
    gate_up=((1, dict(bj=8, bk=512, bjs=32, bks=256, warps=2)),
             (3, dict(bj=8, bk=1024, bjs=32, bks=256, warps=4)),
             (None, dict(bj=16, bk=256, bjs=16, bks=128, warps=4))),
    down=((3, dict(bn=16, bc=256, warps=8)), (None, dict(bn=32, bc=128, warps=4))),
    router=dict(bk=128, splits=12, warps=4),
)


@functools.cache
def _tuning(device: torch.device) -> _Tuning:
    """The H100 table on an H100, the H200 one everywhere else."""
    if device.type == "cuda" and "H100" in torch.cuda.get_device_name(device):
        return _H100
    return _H200


@triton.jit
def _swiglu(g, u, limit, CLAMP: tl.constexpr):
    if CLAMP:
        g = tl.minimum(g, limit)
        u = tl.minimum(tl.maximum(u, -limit), limit)
    return g * tl.sigmoid(g) * u


@triton.jit
def _routed_gate_up(x, ids, w13, s13, act, limit, q, stride_we, stride_se,
                    K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr,
                    BI: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, CLAMP: tl.constexpr):
    """Program ``q``: BJ columns of one routed (token, slot) pair, fp8 rows scaled per
    (BO, BI) block."""
    p = q // (I // BJ)
    j0 = (q % (I // BJ)) * BJ
    j = j0 + tl.arange(0, BJ)
    kk = tl.arange(0, BK)
    e = tl.load(ids + p).to(tl.int64)
    wg = w13 + e * stride_we + j[:, None] * H + kk[None, :]
    wu = wg + I * H
    sg = s13 + e * stride_se + (j0 // BO) * (H // BI)
    su = s13 + e * stride_se + ((I + j0) // BO) * (H // BI)
    xr = x + (p // K) * H
    acc_g = tl.zeros([BJ, BK], tl.float32)
    acc_u = tl.zeros([BJ, BK], tl.float32)
    for k0 in range(0, H, BK):
        xv = tl.load(xr + k0 + kk).to(tl.float32)
        g = tl.load(wg + k0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        u = tl.load(wu + k0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        acc_g += g * (xv * tl.load(sg + (k0 + kk) // BI))[None, :]
        acc_u += u * (xv * tl.load(su + (k0 + kk) // BI))[None, :]
    a = _swiglu(tl.sum(acc_g, 1), tl.sum(acc_u, 1), limit, CLAMP)
    tl.store(act + p * I + j, a)


@triton.jit
def _shared_gate_up(x, sw13, act, limit, T, q,
                    K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, TB: tl.constexpr,
                    NTB: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, CLAMP: tl.constexpr):
    """Program ``q``: BJ columns of the bf16 shared expert for token block ``q % NTB`` (TB
    tokens). The programs of one column block are adjacent, so its weights come from HBM
    about once."""
    t = (q % NTB) * TB + tl.arange(0, TB)
    j = (q // NTB) * BJ + tl.arange(0, BJ)
    kk = tl.arange(0, BK)
    live = t[:, None] < T
    acc_g = tl.zeros([TB, BJ], tl.float32)
    acc_u = tl.zeros([TB, BJ], tl.float32)
    for k0 in range(0, H, BK):
        xt = tl.load(x + t[:, None] * H + k0 + kk[None, :], mask=live, other=0.0)
        wg = tl.load(sw13 + j[:, None] * H + k0 + kk[None, :])
        wu = tl.load(sw13 + (I + j[:, None]) * H + k0 + kk[None, :])
        acc_g = tl.dot(xt, tl.trans(wg), acc_g)
        acc_u = tl.dot(xt, tl.trans(wu), acc_u)
    tl.store(act + (T * K + t[:, None]) * I + j[None, :], _swiglu(acc_g, acc_u, limit, CLAMP),
             mask=live)


@triton.jit
def _gate_up_kernel(
    x, ids, w13, s13, sw13, act, limit, T,
    stride_we, stride_se,
    K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
    BJ: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, TB: tl.constexpr,
    BJS: tl.constexpr, BKS: tl.constexpr, CLAMP: tl.constexpr, PDL: tl.constexpr,
):
    """``act = silu(clamp(x @ gate.T)) * clamp(x @ up.T)`` in fp32 (no clamp without
    CLAMP). Rows ``[0, T*K)`` of ``act`` are the routed (token, slot) pairs, rows
    ``[T*K, T*K+T)`` the shared expert per token; the NS shared programs come first in the
    grid."""
    if PDL:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    if pid < NS:
        _shared_gate_up(x, sw13, act, limit, T, pid, K, I, H, TB, 1, BJS, BKS, CLAMP)
    else:
        if PDL:
            gdc_wait()
        _routed_gate_up(x, ids, w13, s13, act, limit, pid - NS, stride_we, stride_se,
                        K, I, H, BO, BI, BJ, BK, CLAMP)


@triton.jit
def _down_kernel(
    act, ids, topk_w, w2, s2, sw2, out, T,
    stride_we, stride_se,
    K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
    BN: tl.constexpr, BC: tl.constexpr, PDL: tl.constexpr,
):
    """One program per (token, BN hidden rows): ``sum_k topk_w[k] * act[k] @ down[e_k].T``
    plus the shared expert's ``act @ down.T``."""
    if PDL:
        gdc_wait()
    t = tl.program_id(0).to(tl.int64)
    n0 = tl.program_id(1) * BN
    n = n0 + tl.arange(0, BN)
    c = tl.arange(0, BC)
    acc = tl.zeros([BN, BC], tl.float32)
    for k in tl.static_range(K):
        p = t * K + k
        e = tl.load(ids + p).to(tl.int64)
        wt = tl.load(topk_w + p)
        ws = w2 + e * stride_we + n[:, None] * I + c[None, :]
        ss = s2 + e * stride_se + (n0 // BO) * (I // BI)
        for c0 in tl.static_range(0, I, BC):
            a = tl.load(act + p * I + c0 + c) * tl.load(ss + (c0 + c) // BI) * wt
            w = tl.load(ws + c0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            acc += w * a[None, :]
    p = T * K + t
    ws = sw2 + n[:, None] * I + c[None, :]
    for c0 in tl.static_range(0, I, BC):
        acc += tl.load(ws + c0).to(tl.float32) * tl.load(act + p * I + c0 + c)[None, :]
    tl.store(out + t * H + n, tl.sum(acc, 1).to(out.dtype.element_ty))


@triton.jit
def _router_logits_kernel(x, w, part, T, E: tl.constexpr, H: tl.constexpr, TB: tl.constexpr,
                          BE: tl.constexpr, S: tl.constexpr, BK: tl.constexpr,
                          IEEE: tl.constexpr, PDL: tl.constexpr):
    """Program (expert block, K split): fp32 partial router logits ``x @ w.T`` for all
    tokens. IEEE: an fp32 operand, so the dot runs in fp32 instead of bf16."""
    if PDL:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    e = (pid // S) * BE + tl.arange(0, BE)
    k_lo = (pid % S) * (H // S)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros([TB, BE], tl.float32)
    for k0 in range(k_lo, k_lo + H // S, BK):
        xt = tl.load(x + t[:, None] * H + k0 + kk[None, :], mask=t[:, None] < T, other=0.0)
        wt = tl.load(w + e[:, None] * H + k0 + kk[None, :], mask=e[:, None] < E, other=0.0)
        if IEEE:
            acc = tl.dot(xt.to(tl.float32), tl.trans(wt.to(tl.float32)), acc,
                         input_precision="ieee")
        else:
            acc = tl.dot(xt, tl.trans(wt), acc)
    tl.store(part + ((pid % S) * T + t[:, None]) * E + e[None, :], acc,
             mask=(t[:, None] < T) & (e[None, :] < E))


@triton.jit
def _router_topk_kernel(part, bias, w_out, id_out, T, scale,
                        E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                        S: tl.constexpr, NORM: tl.constexpr, PDL: tl.constexpr):
    """One program per token: sum the S partials in a fixed order, then sigmoid -> + bias
    -> top-K (ties to the lower expert) -> gather the unbiased scores -> normalize. The
    ``1e-20`` guards an all-zero top-K and rounds away for any sum above 2^-42."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    t = tl.program_id(0)
    e = tl.arange(0, EP)
    valid = e < E
    logit = tl.zeros([EP], tl.float32)
    for s in tl.static_range(S):
        logit += tl.load(part + (s * T + t) * E + e, mask=valid, other=0.0)
    score = tl.sigmoid(logit)
    biased = tl.where(valid, score + tl.load(bias + e, mask=valid, other=0.0), float("-inf"))
    kk = tl.arange(0, KP)
    ids = tl.zeros([KP], tl.int32)
    for j in tl.static_range(K):
        idx = tl.argmax(biased, 0)
        ids = tl.where(kk == j, idx, ids)
        biased = tl.where(e == idx, float("-inf"), biased)
    ws = tl.sum(tl.where(e[None, :] == ids[:, None], score[None, :], 0.0), 1)
    ws = tl.where(kk < K, ws, 0.0)
    if NORM:
        ws = ws / (tl.sum(ws, 0) + 1e-20)
    tl.store(w_out + t * K + kk, ws * scale, mask=kk < K)
    tl.store(id_out + t * K + kk, ids.to(tl.int64), mask=kk < K)


@triton.jit
def _split_x(x, xq, xs, t, T, H: tl.constexpr, HP: tl.constexpr):
    """``x[t] = xs[t] * (xq[0, t] + xq[1, t] / 16)``: two e4m3 terms under a power of two
    ``xs[t] >= max|x[t]| / 448``. The first keeps each element's top 4 significant bits, the
    second the other 4 of bf16's 8, so the split is exact for ``|x| >= 2^-6 xs[t]``; smaller
    elements are off by at most ``2^-14 xs[t]``. HP: H rounded up to a power of two."""
    k = tl.arange(0, HP)
    live = k < H
    v = tl.load(x + t * H + k, mask=live, other=0.0).to(tl.float32)
    s = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(v), 0), 1e-30) / 448.0)))
    v = v / s
    hi = v.to(tl.float8e4nv)
    tl.store(xq + t * H + k, hi, mask=live)
    tl.store(xq + (T + t) * H + k, ((v - hi.to(tl.float32)) * 16.0).to(tl.float8e4nv),
             mask=live)
    tl.store(xs + t, s)


@triton.jit
def _plan(ids, pairs, tiles, TK, E: tl.constexpr, EP: tl.constexpr, NP: tl.constexpr,
          BM: tl.constexpr, MT: tl.constexpr, NT: tl.constexpr, NTP: tl.constexpr):
    """Group the TK (token, slot) pairs by expert into tiles of at most BM. ``pairs`` holds
    each expert's pairs in ascending order, experts in order; tile i is
    ``pairs[tiles[1, i]:][:tiles[2, i]]``, all routed to expert ``tiles[0, i]``. Tiles past
    the last one have count 0."""
    p = tl.arange(0, NP)
    e_of = tl.load(ids + p, mask=p < TK, other=E).to(tl.int32)
    tl.store(pairs + p, tl.sort(e_of * NP + p) % NP)
    ee = tl.arange(0, EP)
    c = tl.where(ee < E, tl.histogram(e_of, EP), 0)
    off = tl.cumsum(c, 0) - c
    te = (c + BM - 1) // BM
    t0 = tl.cumsum(te, 0) - te
    for mt in tl.static_range(MT):
        live = mt < te
        tl.store(tiles + t0 + mt, ee, mask=live)
        tl.store(tiles + NT + t0 + mt, off + mt * BM, mask=live)
        tl.store(tiles + 2 * NT + t0 + mt, tl.minimum(c - mt * BM, BM), mask=live)
    i = tl.arange(0, NTP)
    tl.store(tiles + 2 * NT + i, 0, mask=(i >= tl.sum(te, 0)) & (i < NT))


@triton.jit
def _plan_kernel(ids, pairs, tiles, x, xq, xs, T, TK, E: tl.constexpr, EP: tl.constexpr,
                 NP: tl.constexpr, BM: tl.constexpr, MT: tl.constexpr, NT: tl.constexpr,
                 NTP: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, PDL: tl.constexpr):
    """Program 0: ``_plan``. Programs 1..T: ``_split_x`` of one token each, for the gate/up;
    they need no routing, so they run beside the plan."""
    if tl.program_id(0) > 0:
        if PDL:
            gdc_launch_dependents()
        _split_x(x, xq, xs, tl.program_id(0) - 1, T, H, HP)
    else:
        if PDL:
            gdc_wait()
            gdc_launch_dependents()
        _plan(ids, pairs, tiles, TK, E, EP, NP, BM, MT, NT, NTP)


@triton.jit
def _grouped_gate_up(xq, xs, pairs, tiles, w13, s13, act, limit, q, T, stride_we, stride_se,
                     K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr,
                     BI: tl.constexpr, BJ: tl.constexpr, BM: tl.constexpr, NT: tl.constexpr,
                     CLAMP: tl.constexpr):
    """BJ gate and up columns of tile ``q // (I // BJ)``'s expert for its pairs (real tiles
    come first, so do their programs), as one fp8 tl.dot with the gate rows stacked on the
    up rows. The e4m3 weights meet x's two e4m3 terms (``_split_x``), so every product is
    exact; each 32-deep slice of the dot is summed into fp32."""
    t = q // (I // BJ)
    n = tl.load(tiles + 2 * NT + t)
    if n > 0:
        e = tl.load(tiles + t).to(tl.int64)
        j0 = (q % (I // BJ)) * BJ
        r = tl.arange(0, 2 * BJ)
        kk = tl.arange(0, BI)
        w = w13 + e * stride_we + tl.where(r < BJ, j0 + r, I + j0 + r - BJ)[:, None] * H
        sg = s13 + e * stride_se + (j0 // BO) * (H // BI)
        su = s13 + e * stride_se + ((I + j0) // BO) * (H // BI)
        m = tl.arange(0, BM)
        live = m < n
        p = tl.load(pairs + tl.load(tiles + NT + t) + m, mask=live, other=0)
        tok = (p // K).to(tl.int64)
        xr = xq + tok[:, None] * H + kk[None, :]
        acc = tl.zeros([2 * BJ, BM], tl.float32)
        for k0 in range(0, H, BI):
            wk = tl.load(w + k0 + kk[None, :])
            lo = tl.load(xr + T * H + k0, mask=live[:, None], other=0.0)
            hi = tl.load(xr + k0, mask=live[:, None], other=0.0)
            d = tl.dot(wk, tl.trans(lo), max_num_imprecise_acc=32)
            d = tl.dot(wk, tl.trans(hi), d * 0.0625, max_num_imprecise_acc=32)
            acc += d * tl.where(r < BJ, tl.load(sg + k0 // BI), tl.load(su + k0 // BI))[:, None]
        acc = acc * tl.load(xs + tok, mask=live, other=0.0)[None, :]
        g, u = tl.split(tl.permute(tl.reshape(acc, [2, BJ, BM]), [1, 2, 0]))
        j = j0 + tl.arange(0, BJ)
        tl.store(act + p[None, :] * I + j[:, None], _swiglu(g, u, limit, CLAMP),
                 mask=live[None, :])


@triton.jit
def _grouped_gate_up_kernel(
    x, xq, xs, pairs, tiles, w13, s13, sw13, act, limit, T, stride_we, stride_se,
    K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
    BJ: tl.constexpr, BM: tl.constexpr, NT: tl.constexpr, NS: tl.constexpr, TB: tl.constexpr,
    NTB: tl.constexpr, BJS: tl.constexpr, BKS: tl.constexpr, CLAMP: tl.constexpr,
    PDL: tl.constexpr,
):
    """``_gate_up_kernel`` with the routed rows grouped by expert (same ``act`` layout)."""
    if PDL:
        gdc_launch_dependents()
    pid = tl.program_id(0)
    if pid < NS:
        _shared_gate_up(x, sw13, act, limit, T, pid, K, I, H, TB, NTB, BJS, BKS, CLAMP)
    else:
        if PDL:
            gdc_wait()
        _grouped_gate_up(xq, xs, pairs, tiles, w13, s13, act, limit, pid - NS, T, stride_we,
                         stride_se, K, I, H, BO, BI, BJ, BM, NT, CLAMP)


@triton.jit
def _grouped_down(act, pairs, tiles, topk_w, w2, s2, part, q, stride_we, stride_se,
                  I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
                  BN: tl.constexpr, NG: tl.constexpr, BM: tl.constexpr, NT: tl.constexpr,
                  CLAMP: tl.constexpr):
    """NG blocks of BN hidden rows of tile ``q // (H // (BN * NG))``'s expert: ``part[pair]
    = topk_w * act @ down.T``. The SwiGLU clamp bounds |act| by limit^2, so act can enter
    the dot as fp16 (11-bit mantissa, 3 more than bf16) against fp16 weights (exact for
    e4m3); unclamped it has no bound fp16 would hold and enters as bf16."""
    dt: tl.constexpr = tl.float16 if CLAMP else tl.bfloat16
    t = q // (H // (BN * NG))
    n = tl.load(tiles + 2 * NT + t)
    if n > 0:
        e = tl.load(tiles + t).to(tl.int64)
        c = tl.arange(0, BI)
        nn = tl.arange(0, BN)
        g0 = (q % (H // (BN * NG))) * (BN * NG)
        m = tl.arange(0, BM)
        live = m < n
        p = tl.load(pairs + tl.load(tiles + NT + t) + m, mask=live, other=0)
        tw = tl.load(topk_w + p, mask=live, other=0.0)
        for g in range(NG):
            n0 = g0 + g * BN
            ws = w2 + e * stride_we + (n0 + nn)[:, None] * I + c[None, :]
            ss = s2 + e * stride_se + (n0 // BO) * (I // BI)
            acc = tl.zeros([BM, BN], tl.float32)
            if I <= 1024:
                for c0 in tl.static_range(0, I, BI):
                    a = tl.load(act + p[:, None] * I + c0 + c[None, :], mask=live[:, None],
                                other=0.0).to(dt)
                    w = tl.load(ws + c0).to(tl.float8e4nv, bitcast=True).to(dt)
                    acc += tl.dot(a, tl.trans(w)) * tl.load(ss + c0 // BI)
            else:  # a runtime loop: unrolled, the pipeliner stages every block of a wide expert
                for c0 in tl.range(0, I, BI):
                    a = tl.load(act + p[:, None] * I + c0 + c[None, :], mask=live[:, None],
                                other=0.0).to(dt)
                    w = tl.load(ws + c0).to(tl.float8e4nv, bitcast=True).to(dt)
                    acc += tl.dot(a, tl.trans(w)) * tl.load(ss + c0 // BI)
            tl.store(part + p[:, None] * H + (n0 + nn)[None, :], acc * tw[:, None],
                     mask=live[:, None])


@triton.jit
def _shared_down(act, sw2, part, T, q, K: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
                 BI: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr):
    """BN hidden rows of the shared expert's down for BM tokens. Its weights are x.dtype, so
    act enters as bf16 hi + lo (about 16 mantissa bits). The weight rows sit on the dot's M
    side, so both operands stay in shared memory and the program holds little more than its
    accumulator: the routed down programs in the same kernel keep their occupancy."""
    c = tl.arange(0, BI)
    n = (q % (H // BN)) * BN + tl.arange(0, BN)
    t = (q // (H // BN)) * BM + tl.arange(0, BM)
    live = t < T
    acc = tl.zeros([BN, BM], tl.float32)
    for c0 in tl.static_range(0, I, BI):
        a = tl.load(act + (T * K + t[:, None]) * I + c0 + c[None, :], mask=live[:, None],
                    other=0.0)
        w = tl.load(sw2 + n[:, None] * I + c0 + c[None, :])
        hi = a.to(w.dtype)
        acc = tl.dot(w, tl.trans(hi), acc)
        acc = tl.dot(w, tl.trans((a - hi.to(tl.float32)).to(w.dtype)), acc)
    tl.store(part + (T * K + t[None, :]) * H + n[:, None], acc, mask=live[None, :])


@triton.jit
def _grouped_down_kernel(
    act, pairs, tiles, topk_w, w2, s2, sw2, part, T, stride_we, stride_se,
    K: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BO: tl.constexpr, BI: tl.constexpr,
    BN: tl.constexpr, NG: tl.constexpr, BM: tl.constexpr, NT: tl.constexpr,
    NSD: tl.constexpr, BNS: tl.constexpr, CLAMP: tl.constexpr, PDL: tl.constexpr,
):
    """fp32 partials: rows ``[0, T*K)`` of ``part`` per routed pair, ``[T*K, T*K+T)`` the
    shared expert per token; the NSD shared programs come first in the grid."""
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    pid = tl.program_id(0)
    if pid < NSD:
        _shared_down(act, sw2, part, T, pid, K, I, H, BI, BNS, BM)
    else:
        _grouped_down(act, pairs, tiles, topk_w, w2, s2, part, pid - NSD, stride_we,
                      stride_se, I, H, BO, BI, BN, NG, BM, NT, CLAMP)


@triton.jit
def _sum_kernel(part, out, T, K: tl.constexpr, H: tl.constexpr, BN: tl.constexpr,
                PDL: tl.constexpr):
    """Program (token, BN hidden rows): the K routed partials in slot order plus the shared
    one, rounded to ``out.dtype`` once."""
    if PDL:
        gdc_wait()
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], tl.float32)
    for k in tl.static_range(K):
        acc += tl.load(part + (t * K + k) * H + n)
    acc += tl.load(part + (T * K + t) * H + n)
    tl.store(out + t * H + n, acc.to(out.dtype.element_ty))


def _tiles(T, tuning):
    """(gate/up, down) per-pair tiles for T tokens."""
    def pick(rows):
        return next(tiles for bound, tiles in rows if bound is None or T <= bound)

    return pick(tuning.gate_up), pick(tuning.down)


def _router_split(H, tuning):
    """(K splits, K step) of the router logits kernel: the most splits up to the table's that
    cut H into whole K steps."""
    bk = min(tuning.router["bk"], H)
    while H % bk:
        bk //= 2
    steps = H // bk
    splits = max(s for s in range(1, min(tuning.router["splits"], steps) + 1) if steps % s == 0)
    return splits, bk


# The router's logits kernel holds every token in one tile: past this many its operands
# outgrow shared memory on some tile tables. The models route larger batches elsewhere.
MAX_ROUTER_TOKENS = 64


def _check_router_tokens(T: int) -> None:
    if T > MAX_ROUTER_TOKENS:
        raise ValueError(f"the decode router takes at most {MAX_ROUTER_TOKENS} tokens, got {T}")


def _launch_router(x, w, bias, part, topk_w, topk_ids, scale, normalize, pdl, tuning, be=32):
    T, H = x.shape
    E, K = w.shape[0], topk_ids.shape[1]
    splits, bk = _router_split(H, tuning)
    _router_logits_kernel[(triton.cdiv(E, be) * splits,)](
        x, w, part, T, E=E, H=H, TB=max(16, triton.next_power_of_2(T)), BE=be, S=splits,
        BK=bk, IEEE=torch.float32 in (x.dtype, w.dtype), PDL=pdl,
        num_warps=tuning.router["warps"])
    _router_topk_kernel[(T,)](
        part, bias, topk_w, topk_ids, T, float(scale), E=E, EP=triton.next_power_of_2(E),
        K=K, KP=triton.next_power_of_2(K), S=splits, NORM=normalize, PDL=pdl, num_warps=1,
        launch_pdl=pdl)


def _launch_gate_up(x, ids, w13, s13, sw13, act, block_size, limit, pdl, bj, bk, bjs, bks, warps):
    T, H = x.shape
    K = ids.shape[1]
    I = w13.shape[1] // 2
    bo, bi = block_size
    bj, bk, bks = min(bj, bo), min(bk, H), min(bks, H)
    assert bo % bj == 0 and I % bo == 0 and H % bk == 0 and I % bjs == 0 and H % bks == 0
    _gate_up_kernel[(I // bjs + T * K * (I // bj),)](
        x, ids, w13, s13, sw13, act, 0.0 if limit is None else float(limit), T, w13.stride(0),
        s13.stride(0), K=K, I=I, H=H, BO=bo, BI=bi, BJ=bj, BK=bk, NS=I // bjs,
        TB=max(16, triton.next_power_of_2(T)), BJS=bjs, BKS=bks, CLAMP=limit is not None,
        PDL=pdl, num_warps=warps, launch_pdl=pdl)


def _launch_down(act, ids, tw, w2, s2, sw2, out, block_size, pdl, bn, bc, warps):
    T, H = out.shape
    K = ids.shape[1]
    I = w2.shape[2]
    bo, bi = block_size
    bn, bc = min(bn, bo), min(bc, I)
    assert bo % bn == 0 and H % bn == 0 and I % bc == 0
    _down_kernel[(T, H // bn)](
        act, ids, tw, w2, s2, sw2, out, T, w2.stride(0), s2.stride(0),
        K=K, I=I, H=H, BO=bo, BI=bi, BN=bn, BC=bc, PDL=pdl, num_warps=warps, launch_pdl=pdl)


def _grouped_tiles(T):
    """Grouped-path tiles. gate/up: BJ gate + BJ up rows (one fp8 dot of M = 2 * BJ) x BM
    pairs, long K loop. down: NG blocks of BN rows per program, so its routing-index loads
    are paid once per 256 rows; two stages keep it at 57 KB of shared memory (4 programs
    per SM, 2 with 3 stages of 4 blocks).
    tbs / bjs / bks / bns tile the shared expert: its gate/up programs take all tokens up to
    32, above that 16 each, since a 64-row x tile would size the whole kernel's shared memory
    (3 programs per SM instead of 6). sum_bn tiles the final sum."""
    tbs = max(16, triton.next_power_of_2(T)) if T <= 32 else 16
    return dict(bj=32, bm=16, gu_warps=4, gu_stages=3, bn=128, ng=2, dn_warps=4, dn_stages=2,
                tbs=tbs, bjs=16, bks=128, bns=64, sum_bn=1024)


def _launch_grouped(x, ids, tw, w13, s13, w2, s2, sw13, sw2, buf, block_size, limit, pdl):
    """Experts grouped by expert: a one-program plan sorts the pairs into tiles of up to BM
    pairs of one expert, each gate/up and down program takes one tile's pairs as the rows of
    a ``tl.dot`` (fp32 accumulate, each fp8 block scale applied to its partial product), and
    a last kernel sums each token's partials. The gate/up meets the fp8 weights with x split
    exactly into two e4m3 terms, so its dot runs on fp8 tensor cores."""
    T, H = x.shape
    K = ids.shape[1]
    E, I = w13.shape[0], w13.shape[1] // 2
    bo, bi = block_size
    t = _grouped_tiles(T)
    bj, bn, bns = min(t["bj"], bo), min(t["bn"], bo), min(t["bns"], H)
    ng, sum_bn = min(t["ng"], H // bn), min(t["sum_bn"], H)
    clamp = limit is not None
    # x is split exactly into e4m3 only from bf16 (8 significant bits); a clamped act enters
    # the down dot as fp16, |act| <= limit^2.
    assert x.dtype == torch.bfloat16
    assert not clamp or limit * limit < torch.finfo(torch.float16).max
    assert bo % bj == 0 and bo % bn == 0 and I % bo == 0 and I % bi == 0 and H % bi == 0
    assert H % (bn * ng) == 0 and H % bns == 0 and H % sum_bn == 0 and I % t["bjs"] == 0
    bm = t["bm"]
    nt = buf["tiles"].shape[1]
    plan = buf["pairs"], buf["tiles"]
    _plan_kernel[(1 + T,)](
        ids, *plan, x, buf["xq"], buf["xs"], T, T * K, E=E, EP=triton.next_power_of_2(E + 1),
        NP=buf["pairs"].numel(), BM=bm, MT=triton.cdiv(T, bm), NT=nt,
        NTP=triton.next_power_of_2(nt), H=H, HP=triton.next_power_of_2(H), PDL=pdl,
        num_warps=4, launch_pdl=pdl)
    ntb = triton.cdiv(T, t["tbs"])
    ns = ntb * (I // t["bjs"])
    w13 = w13.view(torch.float8_e4m3fn)
    _grouped_gate_up_kernel[(ns + nt * (I // bj),)](
        x, buf["xq"], buf["xs"], *plan, w13, s13, sw13, buf["act"],
        0.0 if limit is None else float(limit), T, w13.stride(0), s13.stride(0),
        K=K, I=I, H=H, BO=bo, BI=bi, BJ=bj, BM=bm, NT=nt, NS=ns, TB=t["tbs"], NTB=ntb,
        BJS=t["bjs"], BKS=min(t["bks"], H), CLAMP=clamp, PDL=pdl, num_warps=t["gu_warps"],
        num_stages=t["gu_stages"], launch_pdl=pdl)
    nsd = triton.cdiv(T, bm) * (H // bns)
    _grouped_down_kernel[(nsd + nt * (H // (bn * ng)),)](
        buf["act"], *plan, tw, w2, s2, sw2, buf["part"], T, w2.stride(0), s2.stride(0),
        K=K, I=I, H=H, BO=bo, BI=bi, BN=bn, NG=ng, BM=bm, NT=nt, NSD=nsd, BNS=bns,
        CLAMP=clamp, PDL=pdl, num_warps=t["dn_warps"], num_stages=t["dn_stages"],
        launch_pdl=pdl)
    _sum_kernel[(T, H // sum_bn)](
        buf["part"], buf["out"], T, K=K, H=H, BN=sum_bn, PDL=pdl, num_warps=4, launch_pdl=pdl)


def _check_experts(x, w13, s13, w2, s2, shared_w13, shared_w2, block_size):
    H, I = x.shape[1], w13.shape[1] // 2
    E, (bo, bi) = w13.shape[0], block_size
    assert x.is_contiguous() and w13.is_contiguous() and w2.is_contiguous()
    assert shared_w13.is_contiguous() and shared_w2.is_contiguous()
    assert shared_w13.dtype == shared_w2.dtype == x.dtype and shared_w2.shape == (H, I)
    # the kernels index the scales as dense blocks, past their leading stride
    assert s13.is_contiguous() and s2.is_contiguous(), "block scales must be contiguous"
    assert s13.shape == (E, 2 * I // bo, H // bi) and s2.shape == (E, H // bo, I // bi), (
        f"block scales {tuple(s13.shape)}, {tuple(s2.shape)} do not tile the weights")


def _expert_buffers(x, K, E, I, pair_max_tokens, out=None):
    """Everything the expert kernels write: ``act`` (routed pair rows, then one shared row
    per token) and the output (``out`` if given); above ``pair_max_tokens`` also the routing
    plan, x's e4m3 terms and scales, and the fp32 partials, one row per routed pair then one
    per token for the shared expert."""
    T, H = x.shape
    dev = x.device
    if out is None:
        out = torch.empty(T, H, dtype=x.dtype, device=dev)
    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
    buf = dict(act=torch.empty(T * K + T, I, dtype=torch.float32, device=dev), out=out)
    if T > pair_max_tokens:
        # Tiles: each hit expert's pairs in chunks of bm; at most one partial tile per expert.
        bm = _grouped_tiles(T)["bm"]
        i32 = dict(dtype=torch.int32, device=dev)
        buf.update(pairs=torch.empty(triton.next_power_of_2(T * K), **i32),
                   tiles=torch.empty(3, min(E, T * K) + T * K // bm, **i32),
                   xq=torch.empty(2, T, H, dtype=torch.float8_e4m3fn, device=dev),
                   xs=torch.empty(T, dtype=torch.float32, device=dev),
                   part=torch.empty(T * K + T, H, dtype=torch.float32, device=dev))
    return buf


def _launch_experts(x, ids, tw, w13, s13, w2, s2, sw13, sw2, buf, block_size, limit, pdl,
                    tuning):
    """Up to ``tuning.pair_max_tokens`` tokens, gate/up and down are GEMVs over (token,
    expert) pairs: the fp8 weights meet the activations in fp32, and the down kernel folds
    in the routing weights and the top-k sum. The shared expert runs in the same two
    launches. Above it, ``_launch_grouped``."""
    T = x.shape[0]
    if T > tuning.pair_max_tokens:
        _launch_grouped(x, ids, tw, w13, s13, w2, s2, sw13, sw2, buf, block_size, limit, pdl)
        return
    gu, dn = _tiles(T, tuning)
    _launch_gate_up(x, ids, w13, s13, sw13, buf["act"], block_size, limit, pdl, **gu)
    _launch_down(buf["act"], ids, tw, w2, s2, sw2, buf["out"], block_size, pdl, **dn)


def route(x, w, bias, *, top_k, scale, normalize):
    """``(topk_weights fp32, topk_ids int64)``, both ``(T, top_k)``, of the sigmoid router
    on ``x (T, H) @ w (E, H).T``: a split-K logits kernel, then one program per token."""
    T, H = x.shape
    _check_router_tokens(T)
    assert x.is_contiguous() and w.is_contiguous()
    tuning = _tuning(x.device)
    part = torch.empty(_router_split(H, tuning)[0], T, w.shape[0], dtype=torch.float32,
                       device=x.device)
    topk_w = torch.empty(T, top_k, dtype=torch.float32, device=x.device)
    topk_ids = torch.empty(T, top_k, dtype=torch.int64, device=x.device)
    if T:
        _launch_router(x, w, bias, part, topk_w, topk_ids, scale, normalize, False, tuning)
    return topk_w, topk_ids


@torch.library.custom_op("mstar::moe_router_topk", mutates_args=())
def router_topk(logits: torch.Tensor, bias: torch.Tensor, top_k: int, scale: float,
                normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``route``'s top-k program on precomputed fp32 ``logits (T, E)``. An op opaque to
    dynamo, so a compiled gate keeps it whole."""
    T, E = logits.shape
    assert logits.dtype == torch.float32 and logits.is_contiguous()
    topk_w = torch.empty(T, top_k, dtype=torch.float32, device=logits.device)
    topk_ids = torch.empty(T, top_k, dtype=torch.int64, device=logits.device)
    if T:
        _router_topk_kernel[(T,)](
            logits, bias, topk_w, topk_ids, T, float(scale), E=E,
            EP=triton.next_power_of_2(E), K=top_k, KP=triton.next_power_of_2(top_k), S=1,
            NORM=normalize, PDL=False, num_warps=1)
    return topk_w, topk_ids


@router_topk.register_fake
def _(logits, bias, top_k, scale, normalize):
    T = logits.shape[0]
    return logits.new_empty(T, top_k), logits.new_empty(T, top_k, dtype=torch.int64)


def experts(x, w13, s13, w2, s2, shared_w13, shared_w2, topk_w, topk_ids, *,
            block_size, swiglu_limit=None):
    """Routed plus shared expert output ``(T, H)`` in ``x.dtype``.

    ``w13 (E, 2I, H)`` / ``w2 (E, H, I)`` are e4m3 bytes (uint8 views are fine) with fp32
    ``s13`` / ``s2`` scales per ``block_size`` block; ``shared_w13 (2I, H)`` /
    ``shared_w2 (H, I)`` are in ``x.dtype``; ``topk_w`` fp32, ``topk_ids`` int, distinct
    experts per token. ``swiglu_limit`` clamps gate from above and up to +-limit before the
    SwiGLU; None: no clamp. Above ``pair_max_tokens`` tokens ``x`` must be bf16."""
    _check_experts(x, w13, s13, w2, s2, shared_w13, shared_w2, block_size)
    tuning = _tuning(x.device)
    buf = _expert_buffers(x, topk_ids.shape[1], w13.shape[0], w13.shape[1] // 2,
                          tuning.pair_max_tokens)
    if x.shape[0]:
        ids, tw = topk_ids.contiguous(), topk_w.float().contiguous()
        _launch_experts(x, ids, tw, w13, s13, w2, s2, shared_w13, shared_w2, buf, block_size,
                        swiglu_limit, False, tuning)
    return buf["out"]


def forward(x, gate_w, bias, w13, s13, w2, s2, shared_w13, shared_w2, *, top_k, scale,
            normalize, block_size, swiglu_limit=None, out=None):
    """``route`` then ``experts`` as one chain of programmatic dependent launches (PDL,
    sm90+); returns the ``(T, H)`` MoE output, written into ``out`` if given.

    Each kernel is scheduled while its predecessor drains and waits (``gdc_wait``) before
    reading its results, except the shared expert's gate/up, which needs no routing and so
    overlaps the router's top-k. So every buffer the chain writes is allocated before the
    first launch and stays referenced until the last: no launch may reuse memory that a
    still-running kernel reads."""
    _check_experts(x, w13, s13, w2, s2, shared_w13, shared_w2, block_size)
    assert gate_w.is_contiguous()
    T, H = x.shape
    _check_router_tokens(T)
    tuning = _tuning(x.device)
    part = torch.empty(_router_split(H, tuning)[0], T, gate_w.shape[0], dtype=torch.float32,
                       device=x.device)
    topk_w = torch.empty(T, top_k, dtype=torch.float32, device=x.device)
    topk_ids = torch.empty(T, top_k, dtype=torch.int64, device=x.device)
    buf = _expert_buffers(x, top_k, w13.shape[0], w13.shape[1] // 2, tuning.pair_max_tokens,
                          out)
    if T:
        pdl = torch.cuda.get_device_capability(x.device)[0] >= 9
        _launch_router(x, gate_w, bias, part, topk_w, topk_ids, scale, normalize, pdl, tuning)
        _launch_experts(x, topk_ids, topk_w, w13, s13, w2, s2, shared_w13, shared_w2, buf,
                        block_size, swiglu_limit, pdl, tuning)
    return buf["out"]
