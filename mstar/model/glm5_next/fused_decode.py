"""Fused Triton kernels for the glm5_next layer: mHC, the router and the SwiGLU.

Each replaces a chain of small torch ops that is kernel-count bound at decode batch sizes.
The math and the bf16 rounding points follow the torch reference in mhc.py and moe.py,
which runs wherever the kernels cannot (CPU, or ``_ENABLED`` off in tests).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    _HAS_TRITON = True
except ModuleNotFoundError:  # no triton: the torch reference covers this install
    _HAS_TRITON = False

# tests switch this off to run the torch reference
_ENABLED = True


def fused_available(t: torch.Tensor) -> bool:
    return _ENABLED and _HAS_TRITON and t.is_cuda


if _HAS_TRITON:

    @triton.jit
    def _hc_mix_partial_kernel(
        x, h_in, post_in, comb_in, x_out, fn, part, T,
        HID: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr, MIXP: tl.constexpr,
        BT: tl.constexpr, CH: tl.constexpr, NSPLIT: tl.constexpr, UPDATE: tl.constexpr,
        LOOP: tl.constexpr,
    ):
        """Split-K partials for BT tokens over LOOP CH-wide slices of the hidden dim, all
        streams in one tensor-core dot per slice: the mHC mix ``streams @ fn.T`` in columns
        < MIX and ``sum(streams^2)`` in column MIX. With UPDATE, first applies the previous
        sublayer's ``update_streams`` to ``x`` (the residual) and writes the result."""
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        tmask = (t < T)[:, None, None]
        t = t.to(tl.int64)[:, None, None]
        s = tl.arange(0, HC)[None, :, None]
        k = tl.arange(0, HC * CH)
        m = tl.arange(0, MIXP)
        acc = tl.zeros([BT, MIXP], tl.float32)
        ssq = tl.zeros([BT], tl.float32)
        for step in tl.static_range(LOOP):
            c0 = (tl.program_id(1) * LOOP + step) * CH
            i = c0 + tl.arange(0, CH)[None, None, :]
            w = tl.load(fn + m[None, :] * (HC * HID) + ((k // CH) * HID + k % CH)[:, None]
                        + c0, mask=(m < MIX)[None, :], other=0.0)
            src = x
            if UPDATE:
                hv = tl.load(h_in + t * HID + i, mask=tmask, other=0.0)
                val = tl.load(post_in + t * HC + s, mask=tmask, other=0.0) * hv.to(tl.float32)
                for j in tl.static_range(HC):
                    r = tl.load(x + t * (HC * HID) + j * HID + i, mask=tmask, other=0.0)
                    c = tl.load(comb_in + t * (HC * HC) + j * HC + s, mask=tmask, other=0.0)
                    val += c * r.to(tl.float32)
                tl.store(x_out + t * (HC * HID) + s * HID + i, val.to(tl.bfloat16), mask=tmask)
                # Read the slice back rather than reuse val: the dot wants its own layout, and
                # Triton would otherwise recompute the whole update in it.
                tl.debug_barrier()
                src = x_out
            xs = tl.load(src + t * (HC * HID) + s * HID + i, mask=tmask, other=0.0)
            xs = tl.reshape(xs, [BT, HC * CH])
            if fn.dtype.element_ty == tl.float32:
                acc += tl.dot(xs.to(tl.float32), w, input_precision="ieee")
            else:
                acc += tl.dot(xs.to(w.dtype), w)
            xf = xs.to(tl.float32)
            ssq += tl.sum(xf * xf, 1)
        acc = tl.where(m[None, :] == MIX, ssq[:, None], acc)
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        tl.store(part + (t.to(tl.int64)[:, None] * NSPLIT + tl.program_id(1)) * MIXP
                 + m[None, :], acc, mask=(t < T)[:, None])

    @triton.jit
    def _update_fma(post_in, comb_in, t, s, hv, r0, r1, r2, r3, HC: tl.constexpr):
        """Stream s of ``update_streams``: ``post * h``, then one fma per stream j in order."""
        c = comb_in + t * (HC * HC) + s
        val = tl.load(post_in + t * HC + s) * hv
        val = tl.fma(tl.load(c), r0, val)
        val = tl.fma(tl.load(c + HC), r1, val)
        val = tl.fma(tl.load(c + 2 * HC), r2, val)
        return tl.fma(tl.load(c + 3 * HC), r3, val)

    @triton.jit
    def _hc_mix_prefill_kernel(
        x, h_in, post_in, comb_in, x_out, fn, part, T,
        HID: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr, MIXP: tl.constexpr,
        BT: tl.constexpr, CH: tl.constexpr, NSPLIT: tl.constexpr, UPDATE: tl.constexpr,
        LOOP: tl.constexpr, EV_IN: tl.constexpr, EV_KEEP: tl.constexpr,
    ):
        """``_hc_mix_partial_kernel`` at prefill sizes. The update runs on [BT, CH] tiles
        per stream (the [BT, HC, CH] form is slower here), and the streams the finalize
        reads next are kept in L2 (EV_KEEP) while the inputs are not (EV_IN). No masks:
        rows past T read row T - 1 and write the padding of ``x_out`` and ``part``
        (masked stores run 1.3x slower whenever T % 16 != 0)."""
        tl.static_assert(HC == 4)
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        tr = tl.minimum(t, T - 1).to(tl.int64)[:, None]
        t = t.to(tl.int64)[:, None]
        m = tl.arange(0, MIXP)
        k = tl.arange(0, HC * CH)
        acc = tl.zeros([BT, MIXP], tl.float32)
        ssq = tl.zeros([BT, 1], tl.float32)
        src, t_src = x, tr
        if UPDATE:
            src, t_src = x_out, t
        for step in tl.static_range(LOOP):
            c0 = (tl.program_id(1) * LOOP + step) * CH
            if UPDATE:
                i = c0 + tl.arange(0, CH)[None, :]
                row = tr * (HC * HID) + i
                hv = tl.load(h_in + tr * HID + i, eviction_policy=EV_IN)
                r0 = tl.load(x + row, eviction_policy=EV_IN)
                r1 = tl.load(x + row + HID, eviction_policy=EV_IN)
                r2 = tl.load(x + row + 2 * HID, eviction_policy=EV_IN)
                r3 = tl.load(x + row + 3 * HID, eviction_policy=EV_IN)
                for s in tl.static_range(HC):
                    val = _update_fma(post_in, comb_in, tr, s, hv.to(tl.float32),
                                      r0.to(tl.float32), r1.to(tl.float32), r2.to(tl.float32),
                                      r3.to(tl.float32), HC)
                    tl.store(x_out + t * (HC * HID) + i + s * HID, val.to(tl.bfloat16),
                             eviction_policy=EV_KEEP)
                # As in the decode kernel: read the slice back for the dot.
                tl.debug_barrier()
            i3 = c0 + tl.arange(0, CH)[None, None, :]
            s3 = tl.arange(0, HC)[None, :, None]
            xs = tl.load(src + t_src[:, :, None] * (HC * HID) + s3 * HID + i3,
                         eviction_policy=EV_KEEP)
            xs = tl.reshape(xs, [BT, HC * CH])
            w = tl.load(fn + m[None, :] * (HC * HID) + ((k // CH) * HID + k % CH)[:, None]
                        + c0, mask=(m < MIX)[None, :], other=0.0)
            if fn.dtype.element_ty == tl.float32:
                acc += tl.dot(xs.to(tl.float32), w, input_precision="ieee")
            else:
                acc += tl.dot(xs.to(w.dtype), w)
            xf = xs.to(tl.float32)
            ssq += tl.sum(xf * xf, 1, keep_dims=True)
        acc = tl.where(m[None, :] == MIX, ssq, acc)
        tl.store(part + (t * NSPLIT + tl.program_id(1)) * MIXP + m[None, :], acc)

    @triton.jit
    def _hc_mix(part, t, rms_eps, HID: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr,
                MIXP: tl.constexpr, NSPLIT: tl.constexpr):
        """One token's partials summed over the splits and RMS-scaled: the site's mix."""
        m = tl.arange(0, MIXP)
        rows = part + (t * NSPLIT + tl.arange(0, NSPLIT)) * MIXP
        mix = tl.sum(tl.load(rows[:, None] + m[None, :]), 0)
        ss = tl.sum(tl.where(m == MIX, mix, 0.0), 0)
        return mix * tl.rsqrt(ss / (HC * HID) + rms_eps), m

    @triton.jit
    def _hc_finalize_kernel(
        x, part, hc_scale, hc_base, norm_w, post_out, comb_out, y,
        rms_eps, hc_eps, norm_eps,
        HID: tl.constexpr, HC: tl.constexpr, MIX: tl.constexpr, MIXP: tl.constexpr,
        NSPLIT: tl.constexpr, ITERS: tl.constexpr, EV: tl.constexpr,
    ):
        """Two programs per token that run side by side. Role 0 builds pre, collapses the
        streams with it and applies the site's weighted RMSNorm; role 1 builds post and the
        Sinkhorn comb, which only the next ``update_streams`` reads."""
        t = tl.program_id(0).to(tl.int64)
        r = tl.arange(0, HC)
        if tl.program_id(1) == 0:
            i = tl.arange(0, HID)
            rx = tl.arange(0, HC)
            xs = tl.load(x + t * HC * HID + rx[:, None] * HID + i[None, :], eviction_policy=EV)
            mix, m = _hc_mix(part, t, rms_eps, HID, HC, MIX, MIXP, NSPLIT)
            pre_w = tl.sum(tl.where(m[None, :] == r[:, None], mix[None, :], 0.0), 1)
            pre = tl.sigmoid(pre_w * tl.load(hc_scale + 0) + tl.load(hc_base + r)) + hc_eps

            # Collapse with pre (rounded to bf16 like the reference), then RMSNorm. pre is
            # moved onto the rows of xs through scalars, so the compiler never re-lays-out xs.
            pre_x = tl.zeros([HC], tl.float32)
            for s in tl.static_range(HC):
                pre_x += tl.where(rx == s, tl.sum(tl.where(r == s, pre, 0.0), 0), 0.0)
            c = tl.sum(pre_x[:, None] * xs.to(tl.float32), 0).to(tl.bfloat16).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(c * c, 0) / HID + norm_eps)
            w = tl.load(norm_w + i).to(tl.float32)
            tl.store(y + t * HID + i, (c * rrms * w).to(y.dtype.element_ty))
        else:
            mix, m = _hc_mix(part, t, rms_eps, HID, HC, MIX, MIXP, NSPLIT)
            post_w = tl.sum(tl.where(m[None, :] == (HC + r)[:, None], mix[None, :], 0.0), 1)
            cidx = 2 * HC + r[:, None] * HC + r[None, :]
            comb_w = tl.sum(tl.where(m[None, None, :] == cidx[:, :, None], mix[None, None, :],
                                     0.0), 2)
            post = 2.0 * tl.sigmoid(post_w * tl.load(hc_scale + 1) + tl.load(hc_base + HC + r))
            logits = comb_w * tl.load(hc_scale + 2) + tl.load(hc_base + cidx)
            e = tl.exp(logits - tl.max(logits, 1)[:, None])
            comb = e / tl.sum(e, 1)[:, None] + hc_eps
            comb = (comb / (tl.sum(comb, 0)[None, :] + hc_eps)).to(tl.float32)
            for _ in range(ITERS - 1):
                comb = (comb / (tl.sum(comb, 1)[:, None] + hc_eps)).to(tl.float32)
                comb = (comb / (tl.sum(comb, 0)[None, :] + hc_eps)).to(tl.float32)
            tl.store(post_out + t * HC + r, post)
            tl.store(comb_out + t * HC * HC + r[:, None] * HC + r[None, :], comb)

    @triton.jit
    def _swiglu_kernel(gate_up, out, rows, limit, I: tl.constexpr, BR: tl.constexpr,
                       BI: tl.constexpr):
        """``silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)`` for BR rows x BI columns
        of ``gate_up (rows, 2 I)``, with torch's roundings: silu is x / (1 + exp(-x)) in fp32
        (libdevice exp, round-to-nearest division) rounded to bf16, then the product."""
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        j = tl.program_id(1) * BI + tl.arange(0, BI)
        live = (r < rows)[:, None] & (j < I)[None, :]
        src = gate_up + r.to(tl.int64)[:, None] * (2 * I) + j[None, :]
        g = tl.minimum(tl.load(src, mask=live, other=0.0).to(tl.float32), limit)
        u = tl.load(src + I, mask=live, other=0.0).to(tl.float32)
        u = tl.minimum(tl.maximum(u, -limit), limit)
        act = libdevice.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
        tl.store(out + r.to(tl.int64)[:, None] * I + j[None, :], (act * u).to(out.dtype.element_ty),
                 mask=live)

    @triton.jit
    def _update_streams_kernel(x, h_in, post_in, comb_in, x_out,
                               HID: tl.constexpr, HC: tl.constexpr, CH: tl.constexpr,
                               FMA: tl.constexpr):
        """``post[s] * h + sum_j comb[j, s] * x[j]`` per stream, one CH slice per program.
        FMA: one explicit fma per j, as the prefill mix does; without it the compiler picks
        which product to fuse, element by element."""
        t = tl.program_id(0).to(tl.int64)
        i = tl.program_id(1) * CH + tl.arange(0, CH)
        hv = tl.load(h_in + t * HID + i).to(tl.float32)
        for s in tl.static_range(HC):
            val = tl.load(post_in + t * HC + s) * hv
            for j in tl.static_range(HC):
                r = tl.load(x + t * HC * HID + j * HID + i).to(tl.float32)
                if FMA:
                    val = tl.fma(tl.load(comb_in + t * HC * HC + j * HC + s), r, val)
                else:
                    val += tl.load(comb_in + t * HC * HC + j * HC + s) * r
            tl.store(x_out + t * HC * HID + s * HID + i, val.to(x_out.dtype.element_ty))

    @triton.jit
    def _router_topk_kernel(logits, bias, w_out, id_out, scale,
                            E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr, NORM: tl.constexpr):
        """sigmoid -> + correction bias -> top-K -> gather unbiased scores -> normalize."""
        t = tl.program_id(0).to(tl.int64)
        e = tl.arange(0, EP)
        valid = e < E
        s = tl.sigmoid(tl.load(logits + t * E + e, mask=valid, other=0.0))
        biased = tl.where(valid, s + tl.load(bias + e, mask=valid, other=0.0), float("-inf"))
        kk = tl.arange(0, K)
        ids = tl.zeros([K], tl.int64)
        ws = tl.zeros([K], tl.float32)
        for j in tl.static_range(K):
            top = tl.max(biased, 0)
            # A NaN row (a captured prefill's padding) matches nothing: keep it an expert.
            idx = tl.minimum(tl.min(tl.where(biased == top, e, EP), 0), E - 1)
            ids = tl.where(kk == j, idx.to(tl.int64), ids)
            ws = tl.where(kk == j, tl.sum(tl.where(e == idx, s, 0.0), 0), ws)
            biased = tl.where(e == idx, float("-inf"), biased)
        if NORM:
            ws = ws / (tl.sum(ws, 0) + 1e-20)
        tl.store(w_out + t * K + kk, ws * scale)
        tl.store(id_out + t * K + kk, ids)


# Above this many tokens (prefill) hc_pre takes the prefill mix kernel, whose fused update
# pays once the layer carries its closing update into the next site. L2 hints: mix inputs,
# kept, finalize.
_PREFILL_TOKENS = 256
_PREFILL_L2 = ("evict_first", "evict_last", "evict_first")


def prefill_sized(streams) -> bool:
    """hc_pre and update_streams take their prefill-size path for these streams."""
    _, T, HC, HID = streams.shape
    return T > _PREFILL_TOKENS and HID % 128 == 0 and HC == 4


def hc_pre(streams, fn, hc_scale, hc_base, norm_w, *, rms_eps, hc_eps, iters, norm_eps, update=None):
    """One mHC site plus the norm that follows it, on ``streams (1, T, HC, HID)``.

    Returns ``(post (1,T,HC), comb (1,T,HC,HC), normed (T,HID), streams)``. With
    ``update=(h, post, comb)`` the input is the residual: the previous sublayer's
    ``update_streams`` is applied first and the updated streams are returned."""
    _, T, HC, HID = streams.shape
    prefill = prefill_sized(streams)
    mix = fn.shape[0]
    mixp = triton.next_power_of_2(mix + 1)  # + the sum-of-squares column
    # Narrow slices spread the fn read over more CTAs at decode sizes; wide ones keep the
    # partials small once the token count alone fills the GPU.
    bt, ch, loop = (32, 64, 2) if prefill else (16, min(32 if T <= 32 else 128, HID), 1)
    ev_in, ev_keep, ev_fin = _PREFILL_L2 if prefill else ("", "", "")
    nsplit = HID // (ch * loop)
    assert HID & (HID - 1) == 0 and HID >= 16, HID
    x = streams.reshape(T, HC * HID)
    rows = triton.cdiv(T, bt) * bt if prefill else T  # the prefill kernel writes whole tiles
    part = torch.empty(rows, nsplit, mixp, dtype=torch.float32, device=x.device)
    if update is None:
        x_out, h_in, post_in, comb_in = x, x, x, x
    else:
        h_in, post_in, comb_in = update
        x_out = torch.empty(rows, HC * HID, dtype=x.dtype, device=x.device)[:T]
    args = (x, h_in, post_in, comb_in, x_out, fn, part, T)
    kw = dict(HID=HID, HC=HC, MIX=mix, MIXP=mixp, BT=bt, CH=ch, NSPLIT=nsplit,
              UPDATE=update is not None, LOOP=loop, num_warps=4)
    if T and prefill:
        _hc_mix_prefill_kernel[(triton.cdiv(T, bt), nsplit)](
            *args, EV_IN=ev_in, EV_KEEP=ev_keep, **kw)
    elif T:
        _hc_mix_partial_kernel[(triton.cdiv(T, bt), nsplit)](*args, **kw)
    post = torch.empty(1, T, HC, dtype=torch.float32, device=x.device)
    comb = torch.empty(1, T, HC, HC, dtype=torch.float32, device=x.device)
    y = torch.empty(T, HID, dtype=streams.dtype, device=x.device)
    if T:
        _hc_finalize_kernel[(T, 2)](
            x_out, part, hc_scale, hc_base, norm_w, post, comb, y,
            float(rms_eps), float(hc_eps), float(norm_eps),
            HID=HID, HC=HC, MIX=mix, MIXP=mixp, NSPLIT=nsplit, ITERS=iters,
            EV=ev_fin, num_warps=4,
        )
    return post, comb, y, x_out.view(1, T, HC, HID)


def update_streams(residual, h, post, comb):
    """Fused ``mhc.update_streams`` for ``residual (1, T, HC, HID)``, ``h (T, HID)``."""
    _, T, HC, HID = residual.shape
    out = torch.empty_like(residual)
    ch = min(256, HID)
    if T:
        _update_streams_kernel[(T, HID // ch)](
            residual, h, post, comb, out, HID=HID, HC=HC, CH=ch,
            FMA=prefill_sized(residual), num_warps=4)
    return out


def swiglu(gate_up, limit):
    """The clamped SwiGLU of a gated MLP's ``gate_up (T, 2 I)`` bf16 output, ``(T, I)``."""
    T, I2 = gate_up.shape
    I = I2 // 2
    assert gate_up.is_contiguous()
    out = torch.empty(T, I, dtype=gate_up.dtype, device=gate_up.device)
    bi = min(triton.next_power_of_2(I), 1024)
    if T:
        _swiglu_kernel[(triton.cdiv(T, 4), triton.cdiv(I, bi))](
            gate_up, out, T, float(limit), I=I, BR=4, BI=bi, num_warps=4)
    return out


def router_topk(logits, bias, *, top_k, scale, normalize):
    """``(topk_weights fp32, topk_ids int64)`` from fp32 router ``logits (T, E)``."""
    T, E = logits.shape
    w = torch.empty(T, top_k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(T, top_k, dtype=torch.int64, device=logits.device)
    if T:
        _router_topk_kernel[(T,)](
            logits, bias, w, ids, float(scale),
            E=E, EP=triton.next_power_of_2(E), K=top_k, NORM=normalize, num_warps=4)
    return w, ids
