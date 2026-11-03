"""Qwen3.5's 4-layer block (3 gated-delta-net layers, then 1 full-attention layer, each followed by its
MLP) as one persistent kernel, decode rows 1-4.

The stages are the gdn, attn and mlp kernels' phases back to back in one launch:
    gdn P1 P2 P3 | mlp | gdn P1 P2 P3 | mlp | gdn P1 P2 P3 | mlp | attn P1 P2 P3 | mlp
Each layer has its own counters and accumulators, so nothing is reused within a launch. A CTA walks
the stages in order and waits only on earlier stages, never on later ones, so no wait can block a
CTA that another waits on (the argument holds whether or not every CTA is resident).

The residual stream between stages is never finished by one CTA: a mixer's output and an MLP's
output are fp32 accumulators (relaxed atomics), and each consumer rebuilds what it needs from the
last materialized residual and those accumulators (the add+norm runs redundantly per CTA, a few KB
from L2). CTA 0 materializes each layer's input residual inside the next mixer's first phase, before
that phase's arrival, which every later reader is ordered after.

The layer loop is a runtime loop on purpose: unrolled, four layers of code overflow the SM's
instruction cache and the later layers run far slower than the first.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from mstar.kernels.megakernel.sync import row as _row
from mstar.kernels.megakernel.sync import wait_geq

# counters per layer: [0] mixer P1, [1:65] mixer P2 per (row, head or kv head), [70] mixer P3, [71] MLP
CSTRIDE = 128


@triton.jit
def _span(n, pid, G):
    per = n // G
    extra = n % G
    return pid * per + tl.minimum(pid, extra), per + (pid < extra).to(tl.int32)


@triton.jit
def _gemv_rows(w_ptr, start, count, x, bi, B: tl.constexpr, BB: tl.constexpr, H: tl.constexpr,
               RB: tl.constexpr, RP: tl.constexpr, NS: tl.constexpr):
    """y[b, r] = W[start + r] @ x[b] for r < count, as a [BB, RP] tile (bf16-rounded)."""
    rb = tl.arange(0, RB)
    rp = tl.arange(0, RP)
    hk = tl.arange(0, H)
    yall = tl.zeros([BB, RP], dtype=tl.float32)
    for r0 in tl.range(0, count, RB, num_stages=NS):
        live = (r0 + rb) < count
        rows = start + r0 + rb
        w = tl.load(w_ptr + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        for b in tl.static_range(BB):
            if b < B:
                y = tl.sum(w * _row(x, bi, b)[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
                yall += tl.where((bi[:, None] == b) & (rp[None, :] >= r0) & (rp[None, :] < r0 + RB),
                                 tl.sum(tl.where(rp[None, :] == (r0 + rb)[:, None], y[:, None], 0.0), axis=0)[None, :],
                                 0.0)
    return yall


@triton.jit
def _stream_in(res_ptr, a_ptr, m_ptr, wn_ptr, out_res_ptr, pid, eps, bi, bmask, H: tl.constexpr):
    """This layer's input from the previous layer's materialized residual and its two accumulators:
    r1 = res + attn_out (rounded, as the post-attention add stores it), r2 = r1 + mlp_out; returns
    norm(r2) * (1 + w) and, from CTA 0, stores r2 as the next residual."""
    hk = tl.arange(0, H)
    m2 = bmask[:, None]
    res = tl.load(res_ptr + bi[:, None] * H + hk[None, :], mask=m2, other=0.0, cache_modifier=".cg").to(tl.float32)
    a = tl.load(a_ptr + bi[:, None] * H + hk[None, :], mask=m2, other=0.0, cache_modifier=".cg")
    m = tl.load(m_ptr + bi[:, None] * H + hk[None, :], mask=m2, other=0.0, cache_modifier=".cg")
    r1 = (res + a.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    r2 = r1 + m.to(tl.bfloat16).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(r2 * r2, axis=1) / H + eps)
    wn = tl.load(wn_ptr + hk).to(tl.float32)
    if pid == 0:
        tl.store(out_res_ptr + bi[:, None] * H + hk[None, :], r2.to(tl.bfloat16), mask=m2)
    return (r2 * rstd[:, None] * (1.0 + wn)[None, :]).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _block_kernel(
    # activations in/out, residual buffers, accumulators, scratch, counters
    h_in_ptr, res_in_ptr, res_out_ptr, h_out_ptr, res_mid_ptr, acc_a_ptr, acc_m_ptr, ssq_ptr, cnt_ptr,
    qkv_s_ptr, z_s_ptr, ab_s_ptr, core_s_ptr, y_s_ptr, part_o_ptr, part_ml_ptr,
    # GDN weights, stacked [3, ...], and the pool's state/conv blocks for the block's GDN layers
    win_ptr, convw_ptr, alog_ptr, dtb_ptr, gnw_ptr, gwo_ptr, convst_ptr, state_ptr, slots_ptr,
    conv_layer_stride, conv_slot_stride, state_layer_stride, state_slot_stride,
    # attention weights, cache, table
    wqkv_ptr, qnw_ptr, knw_ptr, awo_ptr, cos_ptr, sin_ptr, kv_ptr, table_ptr, lens_ptr, page_stride, table_stride,
    # MLP weights stacked [4, ...]
    wgu_ptr, wdt_ptr, wpost_ptr, wnext_ptr, w_in0_ptr,
    eps, scale,
    B: tl.constexpr, BB: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
    NIN: tl.constexpr, CONV: tl.constexpr, KD: tl.constexpr, VD: tl.constexpr, HV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, A_OFF: tl.constexpr, B_OFF: tl.constexpr,
    QH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr, P: tl.constexpr,
    CH: tl.constexpr, MAXC: tl.constexpr, CB: tl.constexpr,
    RB: tl.constexpr, NS: tl.constexpr, VB: tl.constexpr, RO: tl.constexpr, ROA: tl.constexpr,
    RP_G: tl.constexpr, RP_A: tl.constexpr, RB_M: tl.constexpr, NS_M: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    bi = tl.arange(0, BB)
    bmask = bi < B
    hk = tl.arange(0, H)
    NQKV: tl.constexpr = QH * 2 * D + 2 * KVH * D
    NVB: tl.constexpr = DV // VB
    NRB: tl.constexpr = H // RO
    NRBA: tl.constexpr = H // ROA
    GQ: tl.constexpr = QH // KVH
    gin_start, gin_count = _span(NIN, pid, G)
    ain_start, ain_count = _span(NQKV, pid, G)
    m_start, m_count = _span(I, pid, G)

    for layer in range(4):
        cnt = cnt_ptr + layer * 128
        acc_a = acc_a_ptr + layer * 4 * H
        acc_m = acc_m_ptr + layer * 4 * H
        res_l = res_mid_ptr + (layer - 1) * 4 * H
        if layer == 0:
            res_l = res_in_ptr

        # ---------------- mixer input
        if layer == 0:
            x = tl.load(h_in_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0).to(tl.float32)
        else:
            prev = cnt_ptr + (layer - 1) * 128
            wait_geq(prev + 71, G)
            tl.debug_barrier()
            res_p = res_mid_ptr + (layer - 2) * 4 * H
            if layer == 1:
                res_p = res_in_ptr
            x = _stream_in(res_p, acc_a_ptr + (layer - 1) * 4 * H, acc_m_ptr + (layer - 1) * 4 * H,
                           wnext_ptr + (layer - 1) * H, res_l, pid, eps, bi, bmask, H)
        if layer < 3:
            # ================= gated delta net, layer `layer`
            win = win_ptr + layer * NIN * H
            convst = convst_ptr + layer * conv_layer_stride
            state = state_ptr + layer * state_layer_stride
            qkv_s = qkv_s_ptr + layer * 4 * CONV
            z_s = z_s_ptr + layer * 4 * VD
            ab_s = ab_s_ptr + layer * 4 * 2 * HV
            core_s = core_s_ptr + layer * 4 * VD
            ssq = ssq_ptr + layer * 4 * HV
            rp = tl.arange(0, RP_G)
            prow = gin_start + rp
            plive = rp < gin_count
            is_conv = plive & (prow < CONV)
            cw = convw_ptr + layer * CONV * 4
            cw0 = tl.load(cw + prow * 4 + 0, mask=is_conv, other=0.0).to(tl.float32)
            cw1 = tl.load(cw + prow * 4 + 1, mask=is_conv, other=0.0).to(tl.float32)
            cw2 = tl.load(cw + prow * 4 + 2, mask=is_conv, other=0.0).to(tl.float32)
            cw3 = tl.load(cw + prow * 4 + 3, mask=is_conv, other=0.0).to(tl.float32)
            slot_v = tl.load(slots_ptr + bi, mask=bmask, other=0)
            sb = convst + slot_v[:, None].to(tl.int64) * conv_slot_stride + prow[None, :] * 3
            cmask = bmask[:, None] & is_conv[None, :]
            s0 = tl.load(sb + 0, mask=cmask, other=0.0).to(tl.float32)
            s1 = tl.load(sb + 1, mask=cmask, other=0.0)
            s2 = tl.load(sb + 2, mask=cmask, other=0.0)
            yall = _gemv_rows(win, gin_start, gin_count, x, bi, B, BB, H, RB, RP_G, NS)
            pre = (cw0[None, :] * s0 + cw1[None, :] * s1.to(tl.float32) + cw2[None, :] * s2.to(tl.float32)
                   + cw3[None, :] * yall)
            act = pre * tl.sigmoid(pre)
            tl.store(qkv_s + bi[:, None] * CONV + prow[None, :], act.to(tl.bfloat16), mask=cmask)
            tl.store(sb + 0, s1, mask=cmask)
            tl.store(sb + 1, s2, mask=cmask)
            tl.store(sb + 2, yall.to(tl.bfloat16), mask=cmask)
            is_z = plive & (prow >= CONV) & (prow < CONV + VD)
            tl.store(z_s + bi[:, None] * VD + (prow - CONV)[None, :], yall.to(tl.bfloat16),
                     mask=bmask[:, None] & is_z[None, :])
            is_a = plive & (prow >= A_OFF) & (prow < A_OFF + HV)
            tl.store(ab_s + bi[:, None] * 2 * HV + (prow - A_OFF)[None, :], yall, mask=bmask[:, None] & is_a[None, :])
            is_b = plive & (prow >= B_OFF) & (prow < B_OFF + HV)
            tl.store(ab_s + bi[:, None] * 2 * HV + HV + (prow - B_OFF)[None, :], yall,
                     mask=bmask[:, None] & is_b[None, :])
            tl.debug_barrier()
            tl.atomic_add(cnt + 0, 1, sem="release", scope="gpu")

            # P2: delta rule
            vr = tl.arange(0, VB)
            kk = tl.arange(0, DK)
            n2 = B * HV * NVB
            for t in range(pid, n2, G):
                b = t // (HV * NVB)
                h = (t // NVB) % HV
                j = t % NVB
                slot = tl.load(slots_ptr + b)
                sp = state + slot * state_slot_stride + h * DV * DK + (j * VB + vr)[:, None] * DK + kk[None, :]
                s = tl.load(sp)
                wait_geq(cnt + 0, G)
                tl.debug_barrier()
                q = tl.load(qkv_s + b * CONV + h * DK + kk, cache_modifier=".cg").to(tl.float32)
                k = tl.load(qkv_s + b * CONV + KD + h * DK + kk, cache_modifier=".cg").to(tl.float32)
                v = tl.load(qkv_s + b * CONV + 2 * KD + h * DV + j * VB + vr, cache_modifier=".cg").to(tl.float32)
                a_raw = tl.load(ab_s + b * 2 * HV + h, cache_modifier=".cg")
                b_raw = tl.load(ab_s + b * 2 * HV + HV + h, cache_modifier=".cg")
                xg = a_raw + tl.load(dtb_ptr + layer * HV + h)
                sp_x = tl.where(xg <= 20.0, tl.log(1.0 + tl.exp(xg)), xg)
                g = tl.exp(-tl.exp(tl.load(alog_ptr + layer * HV + h)) * sp_x)
                beta = 1.0 / (1.0 + tl.exp(-b_raw))
                q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (1.0 / tl.sqrt(DK * 1.0))
                k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
                hs = s * g
                v_new = (v - tl.sum(hs * k[None, :], axis=1)) * beta
                hs = hs + k[None, :] * v_new[:, None]
                o = tl.sum(hs * q[None, :], axis=1)
                tl.store(sp, hs)
                ob = o.to(tl.bfloat16)
                tl.store(core_s + b * VD + h * DV + j * VB + vr, ob)
                obf = ob.to(tl.float32)
                tl.atomic_add(ssq + b * HV + h, tl.sum(obf * obf, axis=0), sem="relaxed")
                tl.debug_barrier()
                tl.atomic_add(cnt + 1 + b * HV + h, 1, sem="release", scope="gpu")

            # P3: gated norm + out projection
            ro = tl.arange(0, RO)
            vv = tl.arange(0, DV)
            for t in range(pid, HV * NRB, G):
                h = t // NRB
                rblk = t % NRB
                wrow = gwo_ptr + layer * H * VD + (rblk * RO + ro)[:, None] * VD
                wo = tl.load(wrow + (h * DV + vv)[None, :]).to(tl.float32)
                nw = tl.load(gnw_ptr + layer * DV + vv)
                for b in tl.static_range(BB):
                    if b < B:
                        wait_geq(cnt + 1 + b * HV + h, NVB)
                tl.debug_barrier()
                part = tl.zeros([BB, RO], dtype=tl.float32)
                for b in tl.static_range(BB):
                    if b < B:
                        c = tl.load(core_s + b * VD + h * DV + vv, cache_modifier=".cg").to(tl.float32)
                        sq = tl.load(ssq + b * HV + h, cache_modifier=".cg")
                        zz = tl.load(z_s + b * VD + h * DV + vv, cache_modifier=".cg").to(tl.float32)
                        xn = (c * tl.rsqrt(sq / DV + eps)).to(tl.bfloat16).to(tl.float32)
                        go = (nw * xn * (zz * tl.sigmoid(zz))).to(tl.bfloat16).to(tl.float32)
                        pr = tl.sum(wo * go[None, :], axis=1)
                        part += tl.where(bi[:, None] == b, pr[None, :], 0.0)
                tl.atomic_add(acc_a + bi[:, None] * H + (rblk * RO + ro)[None, :], part, mask=bmask[:, None],
                              sem="relaxed")
                tl.debug_barrier()
                tl.atomic_add(cnt + 70, 1, sem="release", scope="gpu")
            n3 = HV * NRB
        else:
            # ================= full attention
            rp = tl.arange(0, RP_A)
            yall = _gemv_rows(wqkv_ptr, ain_start, ain_count, x, bi, B, BB, H, RB, RP_A, NS)
            tl.store(y_s_ptr + bi[:, None] * NQKV + (ain_start + rp)[None, :], yall.to(tl.bfloat16),
                     mask=bmask[:, None] & (rp < ain_count)[None, :])
            tl.debug_barrier()
            tl.atomic_add(cnt + 0, 1, sem="release", scope="gpu")
            dd = tl.arange(0, D)
            half: tl.constexpr = ROT // 2
            partner = tl.where(dd < half, dd + half, tl.where(dd < ROT, dd - half, dd))
            sign = tl.where(dd < half, -1.0, 1.0)
            tt = tl.arange(0, CH)
            gq = tl.arange(0, 16)
            gmask = gq < GQ
            # tasks over the step's longest row only: MAXC covers the longest a row can be
            live = (tl.load(lens_ptr) + CH - 1) // CH
            for bb in tl.static_range(1, BB):
                if bb < B:
                    live = tl.maximum(live, (tl.load(lens_ptr + bb) + CH - 1) // CH)
            for t in range(pid, B * KVH * live, G):
                b = t // (KVH * live)
                g = (t // live) % KVH
                c = t % live
                L = tl.load(lens_ptr + b)
                nchunks = (L + CH - 1) // CH
                if c < nchunks:
                    tok = c * CH + tt
                    tvalid = tok < L
                    page = tl.load(table_ptr + b * table_stride + tok // P, mask=tvalid, other=0)
                    base = (kv_ptr + page.to(tl.int64)[:, None] * page_stride + (tok % P)[:, None] * (KVH * D)
                            + g * D + dd[None, :])
                    kt = tl.load(base, mask=tvalid[:, None], other=0.0)
                    vt = tl.load(base + P * KVH * D, mask=tvalid[:, None], other=0.0)
                    cs = tl.load(cos_ptr + b * ROT + dd, mask=dd < ROT, other=1.0).to(tl.float32)
                    sn = tl.load(sin_ptr + b * ROT + dd, mask=dd < ROT, other=0.0).to(tl.float32)
                    qn = tl.load(qnw_ptr + dd).to(tl.float32)
                    qnp = tl.load(qnw_ptr + partner).to(tl.float32)
                    kn = tl.load(knw_ptr + dd).to(tl.float32)
                    knp = tl.load(knw_ptr + partner).to(tl.float32)
                    wait_geq(cnt + 0, G)
                    tl.debug_barrier()
                    yb = y_s_ptr + b * NQKV
                    qh = g * GQ + gq
                    q = tl.load(yb + (qh * 2 * D)[:, None] + dd[None, :], mask=gmask[:, None], other=0.0,
                                cache_modifier=".cg").to(tl.float32)
                    qp = tl.load(yb + (qh * 2 * D)[:, None] + partner[None, :], mask=gmask[:, None], other=0.0,
                                 cache_modifier=".cg").to(tl.float32)
                    rq = tl.rsqrt(tl.sum(q * q, axis=1) / D + eps)[:, None]
                    nq = (q * rq * (1.0 + qn)[None, :]).to(tl.bfloat16).to(tl.float32)
                    nqp = (qp * rq * (1.0 + qnp)[None, :]).to(tl.bfloat16).to(tl.float32)
                    q = tl.where((dd < ROT)[None, :], nq * cs[None, :] + sign[None, :] * nqp * sn[None, :], nq)
                    q = q.to(tl.bfloat16)
                    kb = yb + QH * 2 * D + g * D
                    kr = tl.load(kb + dd, cache_modifier=".cg").to(tl.float32)
                    krp = tl.load(kb + partner, cache_modifier=".cg").to(tl.float32)
                    rk = tl.rsqrt(tl.sum(kr * kr, axis=0) / D + eps)
                    nk = (kr * rk * (1.0 + kn)).to(tl.bfloat16).to(tl.float32)
                    nkp = (krp * rk * (1.0 + knp)).to(tl.bfloat16).to(tl.float32)
                    knew = tl.where(dd < ROT, nk * cs + sign * nkp * sn, nk).to(tl.bfloat16)
                    vnew = tl.load(yb + QH * 2 * D + KVH * D + g * D + dd, cache_modifier=".cg")
                    is_new = tok == L - 1
                    kt = tl.where(is_new[:, None], knew[None, :], kt)
                    vt = tl.where(is_new[:, None], vnew[None, :], vt)
                    if c == nchunks - 1:
                        lp = tl.load(table_ptr + b * table_stride + (L - 1) // P)
                        wb = kv_ptr + lp.to(tl.int64) * page_stride + ((L - 1) % P) * (KVH * D) + g * D + dd
                        tl.store(wb, knew)
                        tl.store(wb + P * KVH * D, vnew)
                    sc = tl.dot(q, tl.trans(kt)) * scale
                    sc = tl.where(tvalid[None, :], sc, float("-inf"))
                    mx = tl.max(sc, axis=1)
                    p = tl.exp(sc - mx[:, None])
                    l = tl.sum(p, axis=1)
                    o = tl.dot(p.to(tl.bfloat16), vt)
                    ob = (b * QH + g * GQ + gq) * MAXC + c
                    tl.store(part_o_ptr + ob[:, None] * D + dd[None, :], o, mask=gmask[:, None])
                    tl.store(part_ml_ptr + ob * 2, mx, mask=gmask)
                    tl.store(part_ml_ptr + ob * 2 + 1, l, mask=gmask)
                    tl.debug_barrier()
                    tl.atomic_add(cnt + 1 + b * KVH + g, 1, sem="release", scope="gpu")
            roa = tl.arange(0, ROA)
            cb = tl.arange(0, CB)
            for t in range(pid, QH * NRBA, G):
                h = t // NRBA
                rblk = t % NRBA
                g = h // GQ
                wo = tl.load(awo_ptr + (rblk * ROA + roa)[:, None] * (QH * D) + (h * D + dd)[None, :]).to(tl.float32)
                part = tl.zeros([BB, ROA], dtype=tl.float32)
                for b in tl.static_range(BB):
                    if b < B:
                        L = tl.load(lens_ptr + b)
                        nchunks = (L + CH - 1) // CH
                        wait_geq(cnt + 1 + b * KVH + g, nchunks)
                        tl.debug_barrier()
                        mx = tl.max(tl.full([CB], float("-inf"), tl.float32), axis=0)
                        den = tl.sum(tl.zeros([CB], dtype=tl.float32), axis=0)
                        num = tl.zeros([D], dtype=tl.float32)
                        for c0 in range(0, nchunks, CB):
                            cm = (c0 + cb) < nchunks
                            ob = (b * QH + h) * MAXC + c0 + cb
                            m = tl.load(part_ml_ptr + ob * 2, mask=cm, other=float("-inf"), cache_modifier=".cg")
                            l = tl.load(part_ml_ptr + ob * 2 + 1, mask=cm, other=0.0, cache_modifier=".cg")
                            po = tl.load(part_o_ptr + ob[:, None] * D + dd[None, :], mask=cm[:, None], other=0.0,
                                         cache_modifier=".cg")
                            mn = tl.maximum(mx, tl.max(m, axis=0))
                            sc2 = tl.exp(mx - mn)
                            wgt = tl.where(cm, tl.exp(m - mn), 0.0)
                            num = num * sc2 + tl.sum(wgt[:, None] * po, axis=0)
                            den = den * sc2 + tl.sum(wgt * l, axis=0)
                            mx = mn
                        o = (num / den).to(tl.bfloat16).to(tl.float32)
                        gate = tl.load(y_s_ptr + b * NQKV + h * 2 * D + D + dd, cache_modifier=".cg").to(tl.float32)
                        go = (o * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
                        pr = tl.sum(wo * go[None, :], axis=1)
                        part += tl.where(bi[:, None] == b, pr[None, :], 0.0)
                tl.atomic_add(acc_a + bi[:, None] * H + (rblk * ROA + roa)[None, :], part, mask=bmask[:, None],
                              sem="relaxed")
                tl.debug_barrier()
                tl.atomic_add(cnt + 70, 1, sem="release", scope="gpu")
            n3 = QH * NRBA
        # ================= MLP of layer `layer`
        wait_geq(cnt + 70, n3)
        tl.debug_barrier()
        res = tl.load(res_l + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0,
                      cache_modifier=".cg").to(tl.float32)
        av = tl.load(acc_a + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0, cache_modifier=".cg")
        r1 = res + av.to(tl.bfloat16).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(r1 * r1, axis=1) / H + eps)
        wp = tl.load(wpost_ptr + layer * H + hk).to(tl.float32)
        xm = (r1 * rstd[:, None] * (1.0 + wp)[None, :]).to(tl.bfloat16).to(tl.float32)
        out = tl.zeros([BB, H], dtype=tl.float32)
        rbm = tl.arange(0, RB_M)
        wgu = wgu_ptr + layer * 2 * I * H
        wdt = wdt_ptr + layer * I * H
        for r0 in tl.range(0, m_count, RB_M, num_stages=NS_M):
            live = (r0 + rbm) < m_count
            rows = m_start + r0 + rbm
            wg = tl.load(wgu + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            wu = tl.load(wgu + (I + rows)[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            wd = tl.load(wdt + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            for b in tl.static_range(BB):
                if b < B:
                    xb = _row(xm, bi, b)
                    gg = tl.sum(wg * xb[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
                    uu = tl.sum(wu * xb[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
                    aa = (gg * tl.sigmoid(gg) * uu).to(tl.bfloat16).to(tl.float32)
                    contrib = tl.sum(aa[:, None] * wd, axis=0)
                    out += tl.where(bi[:, None] == b, contrib[None, :], 0.0)
        tl.atomic_add(acc_m + bi[:, None] * H + hk[None, :], out, mask=bmask[:, None], sem="relaxed")
        tl.debug_barrier()
        if layer < 3:
            tl.atomic_add(cnt + 71, 1, sem="release", scope="gpu")
        else:
            done = tl.atomic_add(cnt + 71, 1, sem="acq_rel", scope="gpu")
            if done == G - 1:
                # the block's output: residual after layer 3 and the next layer's normed input
                mv = tl.load(acc_m + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0,
                             cache_modifier=".cg")
                r1b = r1.to(tl.bfloat16).to(tl.float32)
                r2 = r1b + mv.to(tl.bfloat16).to(tl.float32)
                rstd2 = tl.rsqrt(tl.sum(r2 * r2, axis=1) / H + eps)
                wn = tl.load(wnext_ptr + 3 * H + hk).to(tl.float32)
                tl.store(res_out_ptr + bi[:, None] * H + hk[None, :], r2.to(tl.bfloat16), mask=bmask[:, None])
                tl.store(h_out_ptr + bi[:, None] * H + hk[None, :],
                         (r2 * rstd2[:, None] * (1.0 + wn)[None, :]).to(tl.bfloat16), mask=bmask[:, None])
                z4 = tl.arange(0, 16 * H)
                tl.store(acc_a_ptr + z4, tl.zeros([16 * H], dtype=tl.float32))
                tl.store(acc_m_ptr + z4, tl.zeros([16 * H], dtype=tl.float32))
                zs = tl.arange(0, 16 * HV)
                tl.store(ssq_ptr + zs, tl.zeros([16 * HV], dtype=tl.float32))
                zc = tl.arange(0, 4 * 128)
                tl.store(cnt_ptr + zc, tl.zeros([4 * 128], dtype=tl.int32))


class BlockGeometry(NamedTuple):
    """The sizes one block launch compiles for (all but the row count)."""
    H: int
    I: int
    HV: int
    DK: int
    DV: int
    KD: int
    VD: int
    CONV: int
    NIN: int
    A_OFF: int
    B_OFF: int
    QH: int
    KVH: int
    D: int

    @classmethod
    def of(cls, hidden_size: int, intermediate_size: int, gdn, attn) -> "BlockGeometry":
        """From one gated-delta-net mixer and one attention mixer of the stack."""
        conv, vd, hv = gdn.conv_dim, gdn.value_dim, gdn.num_v_heads
        nin = gdn.in_proj_fused.weight.shape[0]
        a_off = conv + vd
        # `b` sits after `a` and the alignment pad between them
        b_off = a_off + hv + (nin - (conv + vd + 2 * hv))
        return cls(
            H=hidden_size, I=intermediate_size, HV=hv, DK=gdn.head_k_dim, DV=gdn.head_v_dim,
            KD=gdn.key_dim, VD=vd, CONV=conv, NIN=nin, A_OFF=a_off, B_OFF=b_off,
            QH=attn.num_heads, KVH=attn.num_kv_heads, D=attn.head_dim,
        )


class BlockWeights(NamedTuple):
    """One block's weights: the three GDN layers' and four MLPs' stacked by layer, the attention
    layer's as they are. ``wdt`` is the down projections transposed (``[4, I, H]``)."""
    win: torch.Tensor
    convw: torch.Tensor
    alog: torch.Tensor
    dtb: torch.Tensor
    gnw: torch.Tensor
    gwo: torch.Tensor
    wqkv: torch.Tensor
    qnw: torch.Tensor
    knw: torch.Tensor
    awo: torch.Tensor
    wgu: torch.Tensor
    wdt: torch.Tensor
    wpost: torch.Tensor
    wnext: torch.Tensor


class BlockScratch(NamedTuple):
    """What one launch keeps between its stages. Counters and accumulators must be zero when a
    launch starts; the launch leaves them so, so one set serves launches back to back on a stream
    (the blocks of a step, and step after step), never two launches at once."""
    res_mid: torch.Tensor
    acc_a: torch.Tensor
    acc_m: torch.Tensor
    ssq: torch.Tensor
    cnt: torch.Tensor
    qkv_s: torch.Tensor
    z_s: torch.Tensor
    ab_s: torch.Tensor
    core_s: torch.Tensor
    y_s: torch.Tensor
    part_o: torch.Tensor
    part_ml: torch.Tensor

    @classmethod
    def allocate(cls, geom: BlockGeometry, rows: int, max_chunks: int, zeros) -> "BlockScratch":
        """``zeros(name, shape, dtype)`` hands out zero-filled buffers. The per-layer buffers have
        room for ``MAX_ROWS`` rows (the kernel strides them so), the attention partials for
        ``rows`` rows of ``max_chunks`` chunks."""
        f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
        R = 4
        return cls(
            res_mid=zeros("res_mid", (3, R, geom.H), bf16),
            acc_a=zeros("acc_a", (4, R, geom.H), f32),
            acc_m=zeros("acc_m", (4, R, geom.H), f32),
            ssq=zeros("ssq", (4, R, geom.HV), f32),
            cnt=zeros("cnt", (4 * CSTRIDE,), i32),
            qkv_s=zeros("qkv_s", (3, R, geom.CONV), bf16),
            z_s=zeros("z_s", (3, R, geom.VD), bf16),
            ab_s=zeros("ab_s", (3, R, 2 * geom.HV), f32),
            core_s=zeros("core_s", (3, R, geom.VD), bf16),
            y_s=zeros("y_s", (R, geom.QH * 2 * geom.D + 2 * geom.KVH * geom.D), bf16),
            part_o=zeros("part_o", (rows * geom.QH * max_chunks, geom.D), f32),
            part_ml=zeros("part_ml", (rows * geom.QH * max_chunks, 2), f32),
        )


class LaunchConfig(NamedTuple):
    """Tile sizes and pipelining (the pilot's tuned values at 0.8B), and the grid (the SM count)."""
    grid: int
    ch: int = 32
    rb: int = 8
    ns: int = 3
    vb: int = 16
    ro: int = 128
    roa: int = 64
    rb_m: int = 4
    ns_m: int = 3
    num_warps: int = 4


def max_chunks(max_len: int, ch: int) -> int:
    """Split-KV chunks a row of ``max_len`` tokens can need, as the kernel strides its partials."""
    return triton.next_power_of_2(-(-max_len // ch))


def run_block(geom: BlockGeometry, w: BlockWeights, scratch: BlockScratch, cfg: LaunchConfig,
              h_in, res_in, res_out, h_out, conv_pool, state_pool, gdn_idx0, slots, cos, sin, kv_layer,
              table, lens, eps, chunks):
    """One block over the step's rows: ``h_in``/``res_in`` the first layer's normed input and the
    residual, written to ``h_out``/``res_out``. ``conv_pool``/``state_pool`` are the recurrent pool's
    whole blocks (``[layers, slots, ...]``), the block's GDN layers starting at ``gdn_idx0``;
    ``kv_layer`` is the attention layer's NHD pages and ``table``/``lens`` the rows' block table and
    lengths (the new token included); ``chunks`` is what ``scratch`` was sized for."""
    B = h_in.shape[0]
    _block_kernel[(cfg.grid,)](
        h_in, res_in, res_out, h_out, scratch.res_mid, scratch.acc_a, scratch.acc_m, scratch.ssq, scratch.cnt,
        scratch.qkv_s, scratch.z_s, scratch.ab_s, scratch.core_s, scratch.y_s, scratch.part_o, scratch.part_ml,
        w.win, w.convw, w.alog, w.dtb, w.gnw, w.gwo,
        conv_pool[gdn_idx0], state_pool[gdn_idx0], slots,
        conv_pool.stride(0), conv_pool.stride(1), state_pool.stride(0), state_pool.stride(1),
        w.wqkv, w.qnw, w.knw, w.awo, cos, sin, kv_layer, table, lens, kv_layer.stride(0),
        table.stride(0),
        w.wgu, w.wdt, w.wpost, w.wnext, w.wnext,
        eps, geom.D ** -0.5,
        B=B, BB=triton.next_power_of_2(B), H=geom.H, I=geom.I, NIN=geom.NIN, CONV=geom.CONV, KD=geom.KD,
        VD=geom.VD, HV=geom.HV, DK=geom.DK, DV=geom.DV, A_OFF=geom.A_OFF, B_OFF=geom.B_OFF,
        QH=geom.QH, KVH=geom.KVH, D=geom.D, ROT=cos.shape[-1], P=kv_layer.shape[2], CH=cfg.ch, MAXC=chunks,
        CB=min(16, chunks), RB=cfg.rb, NS=cfg.ns, VB=cfg.vb, RO=cfg.ro, ROA=cfg.roa,
        RP_G=triton.next_power_of_2(-(-geom.NIN // cfg.grid)),
        RP_A=triton.next_power_of_2(-(-(geom.QH * 2 * geom.D + 2 * geom.KVH * geom.D) // cfg.grid)),
        RB_M=cfg.rb_m, NS_M=cfg.ns_m, num_warps=cfg.num_warps,
    )


def stack_block_weights(gdn_layers, attn_layer, mlps, post_norms, next_norms) -> BlockWeights:
    """A block's weights as copies (the benchmarks' and tests' way; a served model re-points its
    parameters into the stacks instead, see ``mstar.model.qwen3_5.fused_decode``)."""
    st = lambda ts: torch.stack([t.detach() for t in ts]).contiguous()  # noqa: E731
    a = attn_layer
    return BlockWeights(
        win=st([m.in_proj_fused.weight for m in gdn_layers]),
        convw=st([m.conv1d.weight.squeeze(1) for m in gdn_layers]),
        alog=st([m.A_log for m in gdn_layers]),
        dtb=st([m.dt_bias for m in gdn_layers]),
        gnw=st([m.norm.weight for m in gdn_layers]),
        gwo=st([m.out_proj.weight for m in gdn_layers]),
        wqkv=a.qkv_proj.weight.detach(), qnw=a.q_norm.weight.detach(), knw=a.k_norm.weight.detach(),
        awo=a.o_proj.weight.detach(),
        wgu=st([m.gate_up_proj.weight for m in mlps]),
        wdt=st([m.down_proj.weight.t() for m in mlps]),
        wpost=st([n.weight for n in post_norms]),
        wnext=st([n.weight for n in next_norms]),
    )


class BlockMegakernel:
    """Stacked weights and scratch for one 4-layer block (layers ``first`` .. ``first + 3``)."""

    def __init__(self, cfg, gdn_layers, attn_layer, mlps, post_norms, next_norms, max_len, max_rows=4,
                 device="cuda", rb=8, ns=3, vb=16, ro=128, roa=64, ch=32, rb_m=4, ns_m=3, num_warps=4,
                 grid=None):
        self.geom = BlockGeometry.of(cfg["hidden_size"], cfg["intermediate_size"], gdn_layers[0], attn_layer)
        self.weights = stack_block_weights(gdn_layers, attn_layer, mlps, post_norms, next_norms)
        self.MAXC = max_chunks(max_len, ch)
        self.scratch = BlockScratch.allocate(
            self.geom, max_rows, self.MAXC,
            lambda name, shape, dtype: torch.zeros(shape, dtype=dtype, device=device),
        )
        self.launch = LaunchConfig(
            grid=grid or torch.cuda.get_device_properties(device).multi_processor_count,
            ch=ch, rb=rb, ns=ns, vb=vb, ro=ro, roa=roa, rb_m=rb_m, ns_m=ns_m, num_warps=num_warps,
        )

    @property
    def cnt(self) -> torch.Tensor:
        return self.scratch.cnt

    def __call__(self, h_in, res_in, res_out, h_out, conv_pool, state_pool, gdn_idx0, slots, cos, sin, kv_layer,
                 table, lens, eps):
        run_block(self.geom, self.weights, self.scratch, self.launch, h_in, res_in, res_out, h_out, conv_pool,
                  state_pool, gdn_idx0, slots, cos, sin, kv_layer, table, lens, eps, self.MAXC)
