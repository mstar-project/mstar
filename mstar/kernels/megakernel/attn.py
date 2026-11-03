"""One Qwen3.5 full-attention mixer as a single persistent kernel, decode rows 1-4.

    [q|gate] per head, k, v = x @ W_qkv^T
    q, k = gemma RMSNorm per head (q_norm / k_norm), partial RoPE on the first ROT dims (rotate-half,
           cos/sin as the step's rope tables give them)
    K/V of the new token written into the paged cache (NHD pages [page, 2, P, KVH, D]) at len - 1
    o = softmax(q K^T * D^-0.5) V over the row's pages (block table + lengths, as the fmha wrapper holds)
    out = (o * sigmoid(gate)) @ W_o^T

Phases (a CTA waits only on earlier phases, so it cannot deadlock):
  P1 (every CTA): its run of W_qkv rows -> y scratch.
  P2 (tasks (row, kv head, CH-token chunk)): loads its K/V tile before waiting on all of P1; the chunk
      holding the new token writes it to the cache and uses it in registers; stores (max, sum, o) per
      q head of the group.
  P3 (tasks (q head, RO-row block of W_o)): waits for its group's chunks, merges them, gates, and adds
      its head's share of the o projection into an fp32 accumulator; the last arrival writes the
      bf16 output and zeroes counters and accumulator.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.kernels.megakernel.sync import row as _row
from mstar.kernels.megakernel.sync import wait_geq


@triton.jit
def _norm_rope(v, vp, w, wp, cos, sin, dd, eps, D: tl.constexpr, ROT: tl.constexpr):
    """Gemma RMSNorm over the last dim of [n, D] (bf16), then rotate-half RoPE on dims < ROT. ``vp`` is
    ``v`` gathered at each dim's rotate-half partner and ``wp`` the norm weight there."""
    rstd = tl.rsqrt(tl.sum(v * v, axis=1) / D + eps)[:, None]
    nv = (v * rstd * (1.0 + w)[None, :]).to(tl.bfloat16).to(tl.float32)
    npv = (vp * rstd * (1.0 + wp)[None, :]).to(tl.bfloat16).to(tl.float32)
    half: tl.constexpr = ROT // 2
    sign = tl.where(dd < half, -1.0, 1.0)
    out = tl.where((dd < ROT)[None, :], nv * cos[None, :] + sign[None, :] * npv * sin[None, :], nv)
    return out.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _attn_kernel(
    x_ptr, wqkv_ptr, qnw_ptr, knw_ptr, cos_ptr, sin_ptr, kv_ptr, table_ptr, lens_ptr, wo_ptr,
    y_ptr, part_o_ptr, part_ml_ptr, acc_ptr, out_ptr, cnt_ptr,
    page_stride, table_stride, eps, scale,
    B: tl.constexpr, BB: tl.constexpr, H: tl.constexpr, QH: tl.constexpr, KVH: tl.constexpr,
    D: tl.constexpr, ROT: tl.constexpr, P: tl.constexpr, CH: tl.constexpr, MAXC: tl.constexpr,
    RB: tl.constexpr, NS: tl.constexpr, RO: tl.constexpr, RP: tl.constexpr, CB: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    NQKV: tl.constexpr = QH * 2 * D + 2 * KVH * D
    GQ: tl.constexpr = QH // KVH
    bi = tl.arange(0, BB)
    bmask = bi < B
    hk = tl.arange(0, H)
    x = tl.load(x_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0).to(tl.float32)
    # ---- P1: W_qkv rows
    per = NQKV // G
    extra = NQKV % G
    start = pid * per + tl.minimum(pid, extra)
    count = per + (pid < extra).to(tl.int32)
    rb = tl.arange(0, RB)
    rp = tl.arange(0, RP)
    yall = tl.zeros([BB, RP], dtype=tl.float32)
    for r0 in tl.range(0, count, RB, num_stages=NS):
        live = (r0 + rb) < count
        rows = start + r0 + rb
        w = tl.load(wqkv_ptr + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        for b in tl.static_range(BB):
            if b < B:
                y = tl.sum(w * _row(x, bi, b)[None, :], axis=1)
                yall += tl.where((bi[:, None] == b) & (rp[None, :] >= r0) & (rp[None, :] < r0 + RB),
                                 tl.sum(tl.where(rp[None, :] == (r0 + rb)[:, None], y[:, None], 0.0), axis=0)[None, :],
                                 0.0)
    tl.store(y_ptr + bi[:, None] * NQKV + (start + rp)[None, :], yall.to(tl.bfloat16),
             mask=bmask[:, None] & (rp < count)[None, :])
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr + 0, 1, sem="release", scope="gpu")

    # ---- P2: split-KV decode attention, task = (row b, kv head g, chunk c)
    dd = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    partner = tl.where(dd < half, dd + half, tl.where(dd < ROT, dd - half, dd))
    tt = tl.arange(0, CH)
    gq = tl.arange(0, 16)
    gmask = gq < GQ
    n2 = B * KVH * MAXC
    for t in range(pid, n2, G):
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
            kt = tl.load(base, mask=tvalid[:, None], other=0.0)                     # [CH, D] bf16
            vt = tl.load(base + P * KVH * D, mask=tvalid[:, None], other=0.0)
            cs = tl.load(cos_ptr + b * ROT + dd, mask=dd < ROT, other=1.0).to(tl.float32)
            sn = tl.load(sin_ptr + b * ROT + dd, mask=dd < ROT, other=0.0).to(tl.float32)
            qn = tl.load(qnw_ptr + dd).to(tl.float32)
            qnp = tl.load(qnw_ptr + partner).to(tl.float32)
            kn = tl.load(knw_ptr + dd).to(tl.float32)
            knp = tl.load(knw_ptr + partner).to(tl.float32)
            wait_geq(cnt_ptr + 0, G)
            tl.debug_barrier()
            yb = y_ptr + b * NQKV
            qh = g * GQ + gq
            q = tl.load(yb + (qh * 2 * D)[:, None] + dd[None, :], mask=gmask[:, None], other=0.0,
                        cache_modifier=".cg").to(tl.float32)
            qp = tl.load(yb + (qh * 2 * D)[:, None] + partner[None, :], mask=gmask[:, None], other=0.0,
                         cache_modifier=".cg").to(tl.float32)
            q = _norm_rope(q, qp, qn, qnp, cs, sn, dd, eps, D, ROT)               # [GQ, D]
            kb = yb + QH * 2 * D + g * D
            knew = tl.load(kb + dd, cache_modifier=".cg").to(tl.float32)
            knewp = tl.load(kb + partner, cache_modifier=".cg").to(tl.float32)
            knew = tl.sum(_norm_rope(knew[None, :], knewp[None, :], kn, knp, cs, sn, dd, eps, D, ROT), axis=0)
            vnew = tl.load(yb + QH * 2 * D + KVH * D + g * D + dd, cache_modifier=".cg").to(tl.float32)
            is_new = tok == L - 1
            kt = tl.where(is_new[:, None], knew.to(tl.bfloat16)[None, :], kt)
            vt = tl.where(is_new[:, None], vnew.to(tl.bfloat16)[None, :], vt)
            if c == nchunks - 1:
                lp = tl.load(table_ptr + b * table_stride + (L - 1) // P)
                wb = kv_ptr + lp.to(tl.int64) * page_stride + ((L - 1) % P) * (KVH * D) + g * D + dd
                tl.store(wb, knew.to(tl.bfloat16))
                tl.store(wb + P * KVH * D, vnew.to(tl.bfloat16))
            # [16 (q heads of the group, padded), CH] scores, then [16, D] = P V, on MMA tiles
            sc = tl.dot(q.to(tl.bfloat16), tl.trans(kt)) * scale
            sc = tl.where(tvalid[None, :], sc, float("-inf"))
            m = tl.max(sc, axis=1)
            p = tl.exp(sc - m[:, None])
            l = tl.sum(p, axis=1)
            o = tl.dot(p.to(tl.bfloat16), vt)
            ob = (b * QH + g * GQ + gq) * MAXC + c
            tl.store(part_o_ptr + ob[:, None] * D + dd[None, :], o, mask=gmask[:, None])
            tl.store(part_ml_ptr + ob * 2, m, mask=gmask)
            tl.store(part_ml_ptr + ob * 2 + 1, l, mask=gmask)
            tl.debug_barrier()
            tl.atomic_add(cnt_ptr + 1 + b * KVH + g, 1, sem="release", scope="gpu")
    # ---- P3: merge, gate, o projection, task = (q head h, RO-row block)
    NRB: tl.constexpr = H // RO
    n3 = QH * NRB
    ro = tl.arange(0, RO)
    cb = tl.arange(0, CB)
    for t in range(pid, n3, G):
        h = t // NRB
        rblk = t % NRB
        g = h // GQ
        wo = tl.load(wo_ptr + (rblk * RO + ro)[:, None] * (QH * D) + (h * D + dd)[None, :]).to(tl.float32)
        part = tl.zeros([BB, RO], dtype=tl.float32)
        for b in tl.static_range(BB):
            if b < B:
                L = tl.load(lens_ptr + b)
                nchunks = (L + CH - 1) // CH
                wait_geq(cnt_ptr + 1 + b * KVH + g, nchunks)
                tl.debug_barrier()
                # online merge of the group's chunk partials, CB at a time
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
                    sc = tl.exp(mx - mn)
                    wgt = tl.where(cm, tl.exp(m - mn), 0.0)
                    num = num * sc + tl.sum(wgt[:, None] * po, axis=0)
                    den = den * sc + tl.sum(wgt * l, axis=0)
                    mx = mn
                o = num / den
                o = o.to(tl.bfloat16).to(tl.float32)
                gate = tl.load(y_ptr + b * NQKV + h * 2 * D + D + dd, cache_modifier=".cg").to(tl.float32)
                go = (o * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
                pr = tl.sum(wo * go[None, :], axis=1)
                part += tl.where(bi[:, None] == b, pr[None, :], 0.0)
        tl.atomic_add(acc_ptr + bi[:, None] * H + (rblk * RO + ro)[None, :], part, mask=bmask[:, None],
                      sem="relaxed")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + 1 + 4 * KVH, 1, sem="acq_rel", scope="gpu")
        if done == n3 - 1:
            accv = tl.load(acc_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0,
                           cache_modifier=".cg")
            tl.store(out_ptr + bi[:, None] * H + hk[None, :], accv.to(tl.bfloat16), mask=bmask[:, None])
            tl.store(acc_ptr + bi[:, None] * H + hk[None, :], tl.zeros([BB, H], dtype=tl.float32),
                     mask=bmask[:, None])
            kz = tl.arange(0, 4 * KVH)
            tl.store(cnt_ptr + 1 + kz, tl.zeros([4 * KVH], dtype=tl.int32))
            tl.store(cnt_ptr + 0, 0)
            tl.atomic_xchg(cnt_ptr + 1 + 4 * KVH, 0, sem="release", scope="gpu")
class AttnMegakernel:
    def __init__(self, hidden, q_heads, kv_heads, head_dim, rot_dim, page_size, max_len, max_rows=4,
                 device="cuda", rb=8, ns=3, ch=32, ro=64, num_warps=4, grid=None):
        self.H, self.QH, self.KVH, self.D, self.ROT, self.P = hidden, q_heads, kv_heads, head_dim, rot_dim, page_size
        self.CH = ch
        self.MAXC = triton.next_power_of_2(-(-max_len // ch))
        self.NQKV = q_heads * 2 * head_dim + 2 * kv_heads * head_dim
        f = dict(device=device)
        self.y = torch.zeros(max_rows, self.NQKV, dtype=torch.bfloat16, **f)
        self.part_o = torch.zeros(max_rows * q_heads * self.MAXC, head_dim, dtype=torch.float32, **f)
        self.part_ml = torch.zeros(max_rows * q_heads * self.MAXC, 2, dtype=torch.float32, **f)
        self.acc = torch.zeros(max_rows, hidden, dtype=torch.float32, **f)
        self.cnt = torch.zeros(2 + 4 * kv_heads, dtype=torch.int32, **f)
        self.grid = grid or torch.cuda.get_device_properties(device).multi_processor_count
        self.rp = triton.next_power_of_2(-(-self.NQKV // self.grid))
        self.rb, self.ns, self.ro, self.num_warps = rb, ns, ro, num_warps

    def __call__(self, x, w_qkv, q_norm_w, k_norm_w, cos, sin, kv_layer, table, lens, w_o, out, eps):
        """``kv_layer`` is one layer's NHD pages ``[pages, 2, page, kv_heads, head_dim]``; ``table`` /
        ``lens`` the rows' block table and lengths counting this step's token (the fmha wrapper's)."""
        B = x.shape[0]
        _attn_kernel[(self.grid,)](
            x, w_qkv, q_norm_w, k_norm_w, cos, sin, kv_layer, table, lens, w_o,
            self.y, self.part_o, self.part_ml, self.acc, out, self.cnt,
            kv_layer.stride(0), table.stride(0), eps, self.D ** -0.5,
            B=B, BB=triton.next_power_of_2(B), H=self.H, QH=self.QH, KVH=self.KVH, D=self.D, ROT=self.ROT,
            P=self.P, CH=self.CH, MAXC=self.MAXC, CB=min(16, self.MAXC), RB=self.rb, NS=self.ns, RO=self.ro, RP=self.rp,
            num_warps=self.num_warps,
        )
