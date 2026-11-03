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
            for t in range(pid, B * KVH * MAXC, G):
                b = t // (KVH * MAXC)
                g = (t // MAXC) % KVH
                c = t % MAXC
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
class BlockMegakernel:
    """Stacked weights and scratch for one 4-layer block (layers ``first`` .. ``first + 3``)."""

    def __init__(self, cfg, gdn_layers, attn_layer, mlps, post_norms, next_norms, max_len, max_rows=4,
                 device="cuda", rb=8, ns=3, vb=16, ro=128, roa=64, ch=32, rb_m=4, ns_m=3, num_warps=4,
                 grid=None):
        H, I = cfg["hidden_size"], cfg["intermediate_size"]
        self.H, self.I = H, I
        g0 = gdn_layers[0]
        self.HV, self.DK, self.DV = g0.num_v_heads, g0.head_k_dim, g0.head_v_dim
        self.KD, self.VD = g0.key_dim, g0.value_dim
        self.CONV = g0.conv_dim
        self.NIN = g0.in_proj_fused.weight.shape[0]
        self.A_OFF = self.CONV + self.VD
        self.B_OFF = self.A_OFF + self.HV + (self.NIN - (self.CONV + self.VD + 2 * self.HV))
        st = lambda ts: torch.stack([t.detach() for t in ts]).contiguous()  # noqa: E731
        self.win = st([m.in_proj_fused.weight for m in gdn_layers])
        self.convw = st([m.conv1d.weight.squeeze(1) for m in gdn_layers])
        self.alog = st([m.A_log for m in gdn_layers])
        self.dtb = st([m.dt_bias for m in gdn_layers])
        self.gnw = st([m.norm.weight for m in gdn_layers])
        self.gwo = st([m.out_proj.weight for m in gdn_layers])
        a = attn_layer
        self.QH, self.KVH, self.D = a.num_heads, a.num_kv_heads, a.head_dim
        self.wqkv, self.qnw, self.knw, self.awo = (a.qkv_proj.weight.detach(), a.q_norm.weight.detach(),
                                                   a.k_norm.weight.detach(), a.o_proj.weight.detach())
        self.wgu = st([m.gate_up_proj.weight for m in mlps])
        self.wdt = st([m.down_proj.weight.t() for m in mlps])
        self.wpost = st([n.weight for n in post_norms])
        self.wnext = st([n.weight for n in next_norms])
        self.CH = ch
        self.MAXC = triton.next_power_of_2(-(-max_len // ch))
        self.NQKV = self.QH * 2 * self.D + 2 * self.KVH * self.D
        f = dict(device=device)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, **f)  # noqa: E731
        self.res_mid = z(3, 4, H, dt=torch.bfloat16)
        self.acc_a, self.acc_m = z(4, 4, H), z(4, 4, H)
        self.ssq = z(4, 4, self.HV)
        self.cnt = z(4 * 128, dt=torch.int32)
        self.qkv_s = z(3, 4, self.CONV, dt=torch.bfloat16)
        self.z_s = z(3, 4, self.VD, dt=torch.bfloat16)
        self.ab_s = z(3, 4, 2 * self.HV)
        self.core_s = z(3, 4, self.VD, dt=torch.bfloat16)
        self.y_s = z(4, self.NQKV, dt=torch.bfloat16)
        self.part_o = z(4 * self.QH * self.MAXC, self.D)
        self.part_ml = z(4 * self.QH * self.MAXC, 2)
        self.grid = grid or torch.cuda.get_device_properties(device).multi_processor_count
        self.cfg = dict(rb=rb, ns=ns, vb=vb, ro=ro, roa=roa, rb_m=rb_m, ns_m=ns_m, num_warps=num_warps)
        self.rot = None

    def __call__(self, h_in, res_in, res_out, h_out, conv_pool, state_pool, gdn_idx0, slots, cos, sin, kv_layer,
                 table, lens, eps):
        B = h_in.shape[0]
        c = self.cfg
        rot = cos.shape[-1]
        _block_kernel[(self.grid,)](
            h_in, res_in, res_out, h_out, self.res_mid, self.acc_a, self.acc_m, self.ssq, self.cnt,
            self.qkv_s, self.z_s, self.ab_s, self.core_s, self.y_s, self.part_o, self.part_ml,
            self.win, self.convw, self.alog, self.dtb, self.gnw, self.gwo,
            conv_pool[gdn_idx0], state_pool[gdn_idx0], slots,
            conv_pool.stride(0), conv_pool.stride(1), state_pool.stride(0), state_pool.stride(1),
            self.wqkv, self.qnw, self.knw, self.awo, cos, sin, kv_layer, table, lens, kv_layer.stride(0),
            table.stride(0),
            self.wgu, self.wdt, self.wpost, self.wnext, self.wnext,
            eps, self.D ** -0.5,
            B=B, BB=triton.next_power_of_2(B), H=self.H, I=self.I, NIN=self.NIN, CONV=self.CONV, KD=self.KD,
            VD=self.VD, HV=self.HV, DK=self.DK, DV=self.DV, A_OFF=self.A_OFF, B_OFF=self.B_OFF,
            QH=self.QH, KVH=self.KVH, D=self.D, ROT=rot, P=kv_layer.shape[2], CH=self.CH, MAXC=self.MAXC,
            CB=min(16, self.MAXC), RB=c["rb"], NS=c["ns"], VB=c["vb"], RO=c["ro"], ROA=c["roa"],
            RP_G=triton.next_power_of_2(-(-self.NIN // self.grid)),
            RP_A=triton.next_power_of_2(-(-self.NQKV // self.grid)), RB_M=c["rb_m"], NS_M=c["ns_m"],
            num_warps=c["num_warps"],
        )
