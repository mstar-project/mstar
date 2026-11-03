"""One gated-delta-net mixer (Qwen3.5 SPLIT layout) as a single persistent kernel, decode rows 1-4.

    y = x @ W_in^T                         [q|k|v|z|a|b] (pad block, if any, skipped)
    qkv = silu(conv(conv_state[slot], y[:conv]));  conv_state[slot] shifts in y[:conv]
    per head: the gated delta rule on state[slot, head] (FlashInfer's pretranspose decode: [V, K]
    fp32, q/k L2-normed, q scaled by K^-0.5)
    o = RMSNormGated(core, z);  out = o @ W_out^T

Three phases, separated by counters; a CTA runs its tasks phase by phase, so it waits only on
earlier phases, which no CTA waits to finish (no deadlock with every CTA resident or not):
  P1 (every CTA): its run of W_in rows, RB at a time; conv epilogue for q/k/v rows, z/a/b stored.
  P2 (tasks (row, head, VB-row block of V)): waits for all of P1, updates its state rows in place,
      stores core (bf16) and adds the block's sum of squares into the head's.
  P3 (tasks (head, RO-row block of W_out)): waits for its head's P2 blocks, applies the gated
      norm and adds its head's share of the out projection into an fp32 accumulator.
The last P3 arrival writes the bf16 output and zeroes the counters and accumulators.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.kernels.megakernel.sync import row as _row
from mstar.kernels.megakernel.sync import wait_geq


@triton.jit
def _gdn_kernel(
    x_ptr, win_ptr, convw_ptr, convst_ptr, slots_ptr, alog_ptr, dtb_ptr, state_ptr, normw_ptr, wout_ptr,
    qkv_ptr, z_ptr, ab_ptr, core_ptr, ssq_ptr, acc_ptr, out_ptr, cnt_ptr,
    conv_stride_slot, state_stride_slot, eps,
    B: tl.constexpr, BB: tl.constexpr, H: tl.constexpr, NIN: tl.constexpr, CONV: tl.constexpr,
    KD: tl.constexpr, VD: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    A_OFF: tl.constexpr, B_OFF: tl.constexpr, RB: tl.constexpr, NS: tl.constexpr,
    VB: tl.constexpr, RO: tl.constexpr, RP: tl.constexpr,
):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    bi = tl.arange(0, BB)
    bmask = bi < B
    hk = tl.arange(0, H)
    x = tl.load(x_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0).to(tl.float32)

    # ---- P1: in_proj rows; the conv epilogue runs once after the loop, on inputs loaded before it
    per = NIN // G
    extra = NIN % G
    start = pid * per + tl.minimum(pid, extra)
    count = per + (pid < extra).to(tl.int32)
    rb = tl.arange(0, RB)
    rp = tl.arange(0, RP)
    prow = start + rp
    plive = rp < count
    is_conv = plive & (prow < CONV)
    cw0 = tl.load(convw_ptr + prow * 4 + 0, mask=is_conv, other=0.0).to(tl.float32)
    cw1 = tl.load(convw_ptr + prow * 4 + 1, mask=is_conv, other=0.0).to(tl.float32)
    cw2 = tl.load(convw_ptr + prow * 4 + 2, mask=is_conv, other=0.0).to(tl.float32)
    cw3 = tl.load(convw_ptr + prow * 4 + 3, mask=is_conv, other=0.0).to(tl.float32)
    slot_v = tl.load(slots_ptr + bi, mask=bmask, other=0)
    sb = convst_ptr + slot_v[:, None].to(tl.int64) * conv_stride_slot + prow[None, :] * 3
    cmask = bmask[:, None] & is_conv[None, :]
    s0 = tl.load(sb + 0, mask=cmask, other=0.0).to(tl.float32)
    s1 = tl.load(sb + 1, mask=cmask, other=0.0)
    s2 = tl.load(sb + 2, mask=cmask, other=0.0)
    yall = tl.zeros([BB, RP], dtype=tl.float32)
    for r0 in tl.range(0, count, RB, num_stages=NS):
        live = (r0 + rb) < count
        rows = start + r0 + rb
        w = tl.load(win_ptr + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        for b in tl.static_range(BB):
            if b < B:
                y = tl.sum(w * _row(x, bi, b)[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
                yall += tl.where((bi[:, None] == b) & (rp[None, :] >= r0) & (rp[None, :] < r0 + RB),
                                 tl.sum(tl.where(rp[None, :] == (r0 + rb)[:, None], y[:, None], 0.0), axis=0)[None, :],
                                 0.0)
    pre = cw0[None, :] * s0 + cw1[None, :] * s1.to(tl.float32) + cw2[None, :] * s2.to(tl.float32) + cw3[None, :] * yall
    act = pre * tl.sigmoid(pre)
    tl.store(qkv_ptr + bi[:, None] * CONV + prow[None, :], act.to(tl.bfloat16), mask=cmask)
    tl.store(sb + 0, s1, mask=cmask)
    tl.store(sb + 1, s2, mask=cmask)
    tl.store(sb + 2, yall.to(tl.bfloat16), mask=cmask)
    is_z = plive & (prow >= CONV) & (prow < CONV + VD)
    tl.store(z_ptr + bi[:, None] * VD + (prow - CONV)[None, :], yall.to(tl.bfloat16),
             mask=bmask[:, None] & is_z[None, :])
    is_a = plive & (prow >= A_OFF) & (prow < A_OFF + HV)
    tl.store(ab_ptr + bi[:, None] * 2 * HV + (prow - A_OFF)[None, :], yall, mask=bmask[:, None] & is_a[None, :])
    is_b = plive & (prow >= B_OFF) & (prow < B_OFF + HV)
    tl.store(ab_ptr + bi[:, None] * 2 * HV + HV + (prow - B_OFF)[None, :], yall, mask=bmask[:, None] & is_b[None, :])
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr + 0, 1, sem="release", scope="gpu")

    # ---- P2: the delta rule, task = (row b, head h, V block j)
    NVB: tl.constexpr = DV // VB
    n2 = B * HV * NVB
    vr = tl.arange(0, VB)
    kk = tl.arange(0, DK)
    for t in range(pid, n2, G):
        b = t // (HV * NVB)
        h = (t // NVB) % HV
        j = t % NVB
        slot = tl.load(slots_ptr + b)
        sp = state_ptr + slot * state_stride_slot + h * DV * DK + (j * VB + vr)[:, None] * DK + kk[None, :]
        s = tl.load(sp)  # independent of P1: issued before the wait
        wait_geq(cnt_ptr + 0, G)
        tl.debug_barrier()
        q = tl.load(qkv_ptr + b * CONV + h * DK + kk, cache_modifier=".cg").to(tl.float32)
        k = tl.load(qkv_ptr + b * CONV + KD + h * DK + kk, cache_modifier=".cg").to(tl.float32)
        v = tl.load(qkv_ptr + b * CONV + 2 * KD + h * DV + j * VB + vr, cache_modifier=".cg").to(tl.float32)
        a_raw = tl.load(ab_ptr + b * 2 * HV + h, cache_modifier=".cg")
        b_raw = tl.load(ab_ptr + b * 2 * HV + HV + h, cache_modifier=".cg")
        xg = a_raw + tl.load(dtb_ptr + h)
        sp_x = tl.where(xg <= 20.0, tl.log(1.0 + tl.exp(xg)), xg)
        g = tl.exp(-tl.exp(tl.load(alog_ptr + h)) * sp_x)
        beta = 1.0 / (1.0 + tl.exp(-b_raw))
        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (1.0 / tl.sqrt(DK * 1.0))
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        hs = s * g
        v_new = (v - tl.sum(hs * k[None, :], axis=1)) * beta
        hs = hs + k[None, :] * v_new[:, None]
        o = tl.sum(hs * q[None, :], axis=1)
        tl.store(sp, hs)
        ob = o.to(tl.bfloat16)
        tl.store(core_ptr + b * VD + h * DV + j * VB + vr, ob)
        obf = ob.to(tl.float32)
        tl.atomic_add(ssq_ptr + b * HV + h, tl.sum(obf * obf, axis=0), sem="relaxed")
        tl.debug_barrier()
        tl.atomic_add(cnt_ptr + 1 + b * HV + h, 1, sem="release", scope="gpu")
    # ---- P3: gated norm + out projection, task = (head h, RO-row block of W_out)
    NRB: tl.constexpr = H // RO
    n3 = HV * NRB
    ro = tl.arange(0, RO)
    vv = tl.arange(0, DV)
    for t in range(pid, n3, G):
        h = t // NRB
        rblk = t % NRB
        wo = tl.load(wout_ptr + (rblk * RO + ro)[:, None] * VD + (h * DV + vv)[None, :]).to(tl.float32)
        nw = tl.load(normw_ptr + vv)
        for b in tl.static_range(BB):
            if b < B:
                wait_geq(cnt_ptr + 1 + b * HV + h, NVB)
        tl.debug_barrier()
        part = tl.zeros([BB, RO], dtype=tl.float32)
        for b in tl.static_range(BB):
            if b < B:
                c = tl.load(core_ptr + b * VD + h * DV + vv, cache_modifier=".cg").to(tl.float32)
                ssq = tl.load(ssq_ptr + b * HV + h, cache_modifier=".cg")
                zz = tl.load(z_ptr + b * VD + h * DV + vv, cache_modifier=".cg").to(tl.float32)
                xn = (c * tl.rsqrt(ssq / DV + eps)).to(tl.bfloat16).to(tl.float32)
                go = (nw * xn * (zz * tl.sigmoid(zz))).to(tl.bfloat16).to(tl.float32)
                pr = tl.sum(wo * go[None, :], axis=1)
                part += tl.where(bi[:, None] == b, pr[None, :], 0.0)
        tl.atomic_add(acc_ptr + bi[:, None] * H + (rblk * RO + ro)[None, :], part, mask=bmask[:, None],
                      sem="relaxed")
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + 1 + 4 * HV, 1, sem="acq_rel", scope="gpu")
        if done == n3 - 1:
            accv = tl.load(acc_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0,
                           cache_modifier=".cg")
            tl.store(out_ptr + bi[:, None] * H + hk[None, :], accv.to(tl.bfloat16), mask=bmask[:, None])
            tl.store(acc_ptr + bi[:, None] * H + hk[None, :], tl.zeros([BB, H], dtype=tl.float32),
                     mask=bmask[:, None])
            hv_ = tl.arange(0, 4 * HV)
            tl.store(ssq_ptr + hv_, tl.zeros([4 * HV], dtype=tl.float32))
            tl.store(cnt_ptr + 1 + hv_, tl.zeros([4 * HV], dtype=tl.int32))
            tl.store(cnt_ptr + 0, 0)
            tl.atomic_xchg(cnt_ptr + 1 + 4 * HV, 0, sem="release", scope="gpu")
class GdnMegakernel:
    """Scratch for one mixer launch (reused across layers: launches are stream-ordered)."""

    def __init__(self, hidden, num_v_heads, head_k, head_v, num_k_heads, in_rows, max_rows=4, device="cuda",
                 rb=8, ns=3, vb=16, ro=128, num_warps=4, grid=None):
        self.H, self.HV, self.DK, self.DV = hidden, num_v_heads, head_k, head_v
        self.KD = num_k_heads * head_k
        self.VD = num_v_heads * head_v
        self.CONV = 2 * self.KD + self.VD
        self.NIN = in_rows
        pad = in_rows - (self.CONV + self.VD + 2 * num_v_heads)
        self.A_OFF = self.CONV + self.VD
        self.B_OFF = self.A_OFF + num_v_heads + pad
        assert num_v_heads == num_k_heads, "v heads per k head > 1 not handled here"
        f = dict(device=device)
        self.qkv = torch.zeros(max_rows, self.CONV, dtype=torch.bfloat16, **f)
        self.z = torch.zeros(max_rows, self.VD, dtype=torch.bfloat16, **f)
        self.ab = torch.zeros(max_rows, 2 * num_v_heads, dtype=torch.float32, **f)
        self.core = torch.zeros(max_rows, self.VD, dtype=torch.bfloat16, **f)
        self.ssq = torch.zeros(4 * num_v_heads, dtype=torch.float32, **f)
        self.acc = torch.zeros(max_rows, hidden, dtype=torch.float32, **f)
        self.cnt = torch.zeros(2 + 4 * num_v_heads, dtype=torch.int32, **f)
        self.grid = grid or torch.cuda.get_device_properties(device).multi_processor_count
        self.rb, self.ns, self.vb, self.ro, self.num_warps = rb, ns, vb, ro, num_warps

    def __call__(self, x, w_in, conv_w, conv_state, slots, a_log, dt_bias, state, norm_w, w_out, out, eps):
        """``conv_state`` / ``state`` are one layer's pool views, ``[slots, conv_dim, width - 1]`` bf16 and
        ``[slots, HV, V, K]`` fp32, updated in place for the rows' ``slots``."""
        B = x.shape[0]
        _gdn_kernel[(self.grid,)](
            x, w_in, conv_w, conv_state, slots, a_log, dt_bias, state, norm_w, w_out,
            self.qkv, self.z, self.ab, self.core, self.ssq, self.acc, out, self.cnt,
            conv_state.stride(0), state.stride(0), eps,
            B=B, BB=triton.next_power_of_2(B), H=self.H, NIN=self.NIN, CONV=self.CONV, KD=self.KD, VD=self.VD,
            HV=self.HV, DK=self.DK, DV=self.DV, A_OFF=self.A_OFF, B_OFF=self.B_OFF, RB=self.rb, NS=self.ns,
            VB=self.vb, RO=self.ro, RP=triton.next_power_of_2(-(-self.NIN // self.grid)),
            num_warps=self.num_warps,
        )
