"""Fused Triton kernels for Kimi delta attention against a recurrent state pool.

Both entry points take the layer's merged in-projection row, q | k | v | beta | f_a | g_a,
and fuse the whole layer between that projection and ``o_proj``: the causal conv, the
forget gate's up-projection, the delta rule on the pooled fp32 state and the gated
RMSNorm. ``kda_decode`` is one program per (token, head). ``kda_prefill`` runs every span
of a step in one varlen pass per layer. ``kda_verify`` runs a speculative verify block per
row: it replays the accepted part of the row's last block from the state before it, then the
new block. The pool holds each head's state V-first, ``[V, K]``, as ``DeltaNetGeometry``
declares it and fla's kernels read it.

Prefill, per head: a span is cut into chunks of C tokens. With G the in-chunk cumsum of the
log-decay and S the ``[K, V]`` fp32 state, the chunked delta rule is

    A[i, j]   = beta_i * sum_d k_i k_j exp(G_i - G_j)   (j < i)
    Aqk[i, j] = sum_d q_i k_j exp(G_i - G_j)           (j <= i)
    [W U]     = (I + A)^-1 [beta K exp(G), beta V]
    per chunk: V' = U - W S;  O = (Q exp(G)) S + Aqk V';  S = exp(G_last) S + (K exp(G_last - G))^T V'

Four kernels, the first two chunk-parallel: ``_kda_conv_kernel`` (the causal conv, the
l2norm factors and the gates' in-chunk cumsum), ``_kda_intra_kernel`` (A, Aqk, the solve, W, U),
``_kda_scan_kernel`` (serial over each span's chunks, state in place in the slot pool) and
``_kda_out_norm_kernel`` (output gate + gated RMSNorm). The math and the bf16 rounding points
follow the model's torch reference; ``PrefillConfig`` holds the
tile and precision choices.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ModuleNotFoundError:  # no triton: a torch reference covers this install
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _conv_silu(row, cstate, conv_w, c):
        """Width-4 causal conv + SiLU for channels ``c``. Returns the output and the rolled
        state; the caller stores the state after a barrier (the vectors are replicated
        across warps, so an early store would race with another warp's load)."""
        s0 = tl.load(cstate + c * 3 + 0).to(tl.float32)
        s1 = tl.load(cstate + c * 3 + 1)
        s2 = tl.load(cstate + c * 3 + 2)
        x = tl.load(row + c)
        acc = (s0 * tl.load(conv_w + c * 4 + 0) + s1.to(tl.float32) * tl.load(conv_w + c * 4 + 1)
               + s2.to(tl.float32) * tl.load(conv_w + c * 4 + 2)
               + x.to(tl.float32) * tl.load(conv_w + c * 4 + 3))
        return (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32), s1, s2, x

    @triton.jit
    def _conv_roll(cstate, c, s1, s2, x):
        tl.store(cstate + c * 3 + 0, s1.to(cstate.dtype.element_ty))
        tl.store(cstate + c * 3 + 1, s2.to(cstate.dtype.element_ty))
        tl.store(cstate + c * 3 + 2, x.to(cstate.dtype.element_ty))

    @triton.jit
    def _kda_decode_kernel(
        proj, stride_proj, conv_pool, stride_conv, conv_w, f_b, g_b, dt_bias, a_log, norm_w,
        state_pool, stride_state, slot_ids, out, stride_out,
        lower_bound, scale, norm_eps,
        H: tl.constexpr, D: tl.constexpr,
    ):
        """One program per (token, head): conv, gates, l2norm, the delta-rule step on the
        pooled fp32 state ``[v][k]``, and the gated RMSNorm. ``proj`` rows are
        q | k | v | b | f_a | g_a from the merged in-projection."""
        n = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        P = H * D
        slot = tl.load(slot_ids + n).to(tl.int64)
        d = tl.arange(0, D)
        row = proj + n * stride_proj
        cstate = conv_pool + slot * stride_conv

        # Both gate up-projections first: they read only the projections, so their weight
        # loads are issued before any store and overlap each other.
        f_a = tl.load(row + 3 * P + H + d).to(tl.float32)
        fb = tl.load(f_b + (h * D + d)[:, None] * D + d[None, :]).to(tl.float32)
        g_a = tl.load(row + 3 * P + H + D + d).to(tl.float32)
        gb = tl.load(g_b + (h * D + d)[:, None] * D + d[None, :]).to(tl.float32)
        # Forget gate: f_b(f_a) per head, + dt_bias, lower_bound * sigmoid(exp(A) * g).
        g = tl.sum(fb * f_a[None, :], 1).to(tl.bfloat16).to(tl.float32)
        g = g + tl.load(dt_bias + h * D + d)
        decay = tl.exp(lower_bound * tl.sigmoid(tl.exp(tl.load(a_log + h)) * g))
        gate = tl.sum(gb * g_a[None, :], 1).to(tl.bfloat16).to(tl.float32)
        beta = tl.sigmoid(tl.load(row + 3 * P + h).to(tl.float32))
        beta = beta.to(tl.bfloat16).to(tl.float32)

        q, qs1, qs2, qx = _conv_silu(row, cstate, conv_w, h * D + d)
        k, ks1, ks2, kx = _conv_silu(row, cstate, conv_w, P + h * D + d)
        v, vs1, vs2, vx = _conv_silu(row, cstate, conv_w, 2 * P + h * D + d)
        tl.debug_barrier()
        _conv_roll(cstate, h * D + d, qs1, qs2, qx)
        _conv_roll(cstate, P + h * D + d, ks1, ks2, kx)
        _conv_roll(cstate, 2 * P + h * D + d, vs1, vs2, vx)
        q = q / tl.sqrt(tl.sum(q * q, 0) + 1e-6) * scale
        k = k / tl.sqrt(tl.sum(k * k, 0) + 1e-6)

        # s[v, k]: the decay runs along k, the reads and the update reduce over it
        sp = state_pool + slot * stride_state + h * D * D + d[:, None] * D + d[None, :]
        s = tl.load(sp) * decay[None, :]
        delta = (v - tl.sum(s * k[None, :], 1)) * beta
        s = s + delta[:, None] * k[None, :]
        o = tl.sum(s * q[None, :], 1).to(tl.bfloat16).to(tl.float32)
        tl.store(sp, s)

        # The gated RMSNorm over the head.
        y = o * tl.rsqrt(tl.sum(o * o, 0) / D + norm_eps)
        y = y * tl.load(norm_w + d).to(tl.float32) * tl.sigmoid(gate)
        tl.store(out + n * stride_out + h * D + d, y.to(out.dtype.element_ty))


def kda_decode(proj, conv_pool, conv_w, f_b, g_b, dt_bias, a_log, norm_w, state_pool, slot_ids,
               *, num_heads, head_dim, lower_bound, scale, norm_eps):
    """Fused KDA decode for ``proj (N, 3P + H + 2D)``; returns the gated-normed ``(N, P)``.
    ``conv_pool (S, 3P, 3)`` and ``state_pool (S, H, V, K)`` fp32 are updated at ``slot_ids``."""
    n = proj.shape[0]
    assert conv_pool.shape[-1] == 3 and conv_w.shape[-1] == 4, "width-4 conv only"
    assert state_pool.is_contiguous() and conv_pool.is_contiguous() and proj.stride(1) == 1
    out = torch.empty(n, num_heads * head_dim, dtype=proj.dtype, device=proj.device)
    if n:
        _kda_decode_kernel[(n, num_heads)](
            proj, proj.stride(0), conv_pool, conv_pool.stride(0), conv_w, f_b, g_b, dt_bias,
            a_log, norm_w, state_pool, state_pool.stride(0), slot_ids, out, out.stride(0),
            float(lower_bound), float(scale), float(norm_eps),
            H=num_heads, D=head_dim, num_warps=4,
        )
    return out


@dataclass
class PrefillConfig:
    """Tile and precision choices; the defaults are tuned on H200 at one TP8 rank of
    GLM-5.3 (8 heads of 128)."""
    chunk: int = 64
    # rows per decay-reference block of A; exp(|lower_bound| * row_block / 2) must fit fp32
    row_block: int = 32
    slice: int = 64  # head-dim channels per step of the intra kernel
    conv_groups: int = 16  # row groups the conv walks in parallel, C / groups rows each
    conv_warps: int = 4
    intra_warps: int = 4
    # V columns of the state per scan program; 0 picks 32 while the grid fits one wave of
    # SMs (few spans: more programs on the serial path) and 64 above it
    scan_bv: int = 0
    scan_warps: int = 4
    scan_stages: int = 3
    # tl.dot precision: "bf16" rounds the operands; "tf32" / "tf32x3" / "ieee" keep fp32.
    # bf16 everywhere is as accurate as tf32 here, and tf32 in the scan is much slower.
    solve_prec: str = "bf16"
    wu_prec: str = "bf16"
    scan_prec: str = "bf16"
    scratch_dtype: torch.dtype = torch.bfloat16


CONFIG = PrefillConfig()


@functools.cache
def _num_sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def supported(head_dim: int) -> bool:
    """Head dims the fused prefill is validated for. At 16, Triton 3.7 miscompiles the bf16
    T @ X dot that follows the tf32 solve (rows past 16 come out wrong); GLM-5.3 uses 128."""
    return head_dim >= 32 and head_dim & (head_dim - 1) == 0


@dataclass
class KdaVarlen:
    """Where each span's tokens and state live, and the (span, chunk) table, on device."""
    num_spans: int
    num_tokens: int
    num_chunks: int
    chunk: int
    starts: torch.Tensor  # int32 [N]
    lens: torch.Tensor  # int32 [N]
    slots: torch.Tensor  # int32 [N]
    chunk_offsets: torch.Tensor  # int32 [N]: index of each span's first chunk
    chunks: torch.Tensor  # int32 [num_chunks, 2]: (span, chunk within the span)


def static_rows(num_spans: int, num_tokens: int, chunk: int | None = None) -> int:
    """Chunk-table rows that hold any split of ``num_tokens`` over ``num_spans`` spans."""
    return num_tokens // (chunk or CONFIG.chunk) + num_spans


def varlen_layout(lens, num_tokens: int | None = None, chunk: int | None = None) -> list[int]:
    """The spans' starts, lengths and first chunks, then the (span, chunk) table, as host ints.

    With ``num_tokens`` (a capture bucket's) the table is filled out to ``static_rows`` with a
    chunk past span 0's end, which every kernel masks out, so a captured graph reads any split
    of those tokens at the same offsets."""
    chunk = chunk or CONFIG.chunk
    starts, offsets, table, start = [], [], [], 0
    for n, length in enumerate(lens):
        starts.append(start)
        start += length
        offsets.append(len(table) // 2)
        for c in range(-(-length // chunk)):
            table += (n, c)
    if num_tokens is not None:
        rows = static_rows(len(lens), num_tokens, chunk)
        assert len(table) // 2 <= rows and start <= num_tokens, (len(table), rows, num_tokens)
        table += (0, -(-lens[0] // chunk)) * (rows - len(table) // 2)
    return [*starts, *lens, *offsets, *table]


def varlen_view(layout: torch.Tensor, slots: torch.Tensor, num_spans: int, num_tokens: int,
                chunk: int | None = None) -> KdaVarlen:
    """A ``varlen_layout`` on device, with each span's slot, as the kernels' ``KdaVarlen``."""
    n = num_spans
    return KdaVarlen(
        num_spans=n, num_tokens=num_tokens, num_chunks=(layout.numel() - 3 * n) // 2,
        chunk=chunk or CONFIG.chunk, starts=layout[:n], lens=layout[n:2 * n], slots=slots[:n],
        chunk_offsets=layout[2 * n:3 * n], chunks=layout[3 * n:],
    )


if _HAS_TRITON:

    @triton.jit
    def _mm(a, b, PREC: tl.constexpr):
        if PREC == "bf16":
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision=PREC)

    @triton.jit
    def _sigmoid(x):
        """1 / (1 + exp(-x)) as two MUFU ops (ex2, rcp.approx): within ~2 ulp of fp32, and
        every use rounds to bf16 or sums afterwards."""
        den = 1.0 + tl.exp2(x * -1.4426950408889634)
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [den],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _chunk_rows(chunks, starts, lens, ic, C: tl.constexpr):
        """The chunk's span-relative positions, validity, token indices and span start."""
        n = tl.load(chunks + 2 * ic)
        c = tl.load(chunks + 2 * ic + 1)
        bos = tl.load(starts + n)
        pos = c * C + tl.arange(0, C)
        valid = pos < tl.load(lens + n)
        return n, c, pos, valid, (bos + pos).to(tl.int64)

    @triton.jit
    def _conv_input(proj, stride_proj, cstate, bos, L, ch, pos):
        """The raw conv input at span positions ``pos`` [R] for channels ``ch`` [D], fp32:
        the projection row, or the slot's conv tail (column 2 the newest) before the span."""
        x = tl.load(proj + (bos + pos).to(tl.int64)[:, None] * stride_proj + ch[None, :],
                    mask=((pos >= 0) & (pos < L))[:, None], other=0.0).to(tl.float32)
        old = tl.load(cstate + ch[None, :] * 3 + (3 + pos)[:, None],
                      mask=(pos < 0)[:, None], other=0.0).to(tl.float32)
        return x + old

    @triton.jit
    def _new_tail(proj, stride_proj, cstate, ch, bos, L, m):
        """Column ``m`` of the conv tail after a span of ``L`` tokens: span position L - 3 + m,
        or the old tail's column L + m when that position is before the span."""
        src = L - 3 + m
        live = m < 3
        x = tl.load(proj + (bos + src).to(tl.int64)[None, :] * stride_proj + ch[:, None],
                    mask=(live & (src >= 0))[None, :], other=0.0)
        old = tl.load(cstate + ch[:, None] * 3 + (3 + src)[None, :],
                      mask=(live & (src < 0))[None, :], other=0.0)
        return x + old

    # T (the step's token count) is not specialized on: one binary per config, never a
    # recompile when a step's T changes divisibility (every T-derived address is scaled by D).
    @triton.jit(do_not_specialize=["T"])
    def _kda_conv_kernel(
        proj, stride_proj, conv_pool, stride_conv, conv_w, f_b, dt_bias, a_log,
        starts, lens, slots, chunks,
        qkv_s, g_s, glast_s, beta_s, T, lower_bound,
        H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
    ):
        """One program per (chunk, part, head). Parts 0-2 (q | k | v): the causal conv + SiLU
        of the head's channels, stored bf16 (like the reference's) head-major in
        ``qkv_s [3, H, T, D]``. The chunk's rows are walked in R groups of C / R consecutive
        rows, each keeping its last three inputs in registers, so every input is loaded once.
        Part 3: the forget gate's in-chunk cumsum ``g_s [H, T, D]`` (fp32, log2 units), its
        last row per chunk and beta."""
        ic = tl.program_id(0).to(tl.int64)
        part = tl.program_id(1)
        h = tl.program_id(2)
        P = H * D
        d = tl.arange(0, D)
        if part < 3:
            n = tl.load(chunks + 2 * ic)
            c = tl.load(chunks + 2 * ic + 1)
            bos = tl.load(starts + n)
            L = tl.load(lens + n)
            cstate = conv_pool + tl.load(slots + n).to(tl.int64) * stride_conv
            ch = (part * H + h) * D + d
            w0 = tl.load(conv_w + ch * 4 + 0)[None, :]
            w1 = tl.load(conv_w + ch * 4 + 1)[None, :]
            w2 = tl.load(conv_w + ch * 4 + 2)[None, :]
            w3 = tl.load(conv_w + ch * 4 + 3)[None, :]
            start = c * C + tl.arange(0, R) * (C // R)  # each group's first span position
            x1 = _conv_input(proj, stride_proj, cstate, bos, L, ch, start - 3)
            x2 = _conv_input(proj, stride_proj, cstate, bos, L, ch, start - 2)
            x3 = _conv_input(proj, stride_proj, cstate, bos, L, ch, start - 1)
            out = qkv_s + (part * H + h).to(tl.int64) * T * D + d[None, :]
            for t in tl.static_range(C // R):
                live = (start + t < L)[:, None]
                row_t = (bos + start + t).to(tl.int64)[:, None]
                x4 = tl.load(proj + row_t * stride_proj + ch[None, :], mask=live, other=0.0)
                x4 = x4.to(tl.float32)
                acc = w0 * x1 + w1 * x2 + w2 * x3 + w3 * x4
                tl.store(out + row_t * D, (acc * _sigmoid(acc)).to(tl.bfloat16), mask=live)
                x1, x2, x3 = x2, x3, x4
        else:
            n, c, pos, valid, tok = _chunk_rows(chunks, starts, lens, ic, C)
            vmask = valid[:, None]
            row = proj + tok * stride_proj
            # f_b(f_a) + dt_bias -> lower_bound * sigmoid(exp(A) g), summed down the chunk.
            f_a = tl.load(row[:, None] + 3 * P + H + d[None, :], mask=vmask, other=0.0)
            fb = tl.load(f_b + (h * D + d)[:, None] * D + d[None, :])
            g = tl.dot(f_a, tl.trans(fb)).to(tl.bfloat16).to(tl.float32)
            g = g + tl.load(dt_bias + h * D + d)[None, :]
            g = lower_bound * _sigmoid(tl.exp(tl.load(a_log + h)) * g)
            G = tl.cumsum(tl.where(vmask, g, 0.0), axis=0) * 1.4426950408889634
            base = h * T + tok
            tl.store(g_s + base[:, None] * D + d[None, :], G, mask=vmask)
            last = tl.arange(0, C)[:, None] == C - 1
            tl.store(glast_s + (ic * H + h) * D + d, tl.sum(tl.where(last, G, 0.0), 0))
            beta = _sigmoid(tl.load(row + 3 * P + h, mask=valid, other=0.0).to(tl.float32))
            tl.store(beta_s + base, beta.to(tl.bfloat16).to(tl.float32), mask=valid)

    @triton.jit
    def _load_g(g_s, glast_s, out, vmask, ic, h, ch, H: tl.constexpr, D: tl.constexpr):
        """G for channels ``ch``, and its last row; past the span end G stays at the last
        row, which is what the cumsum over zero gates leaves there."""
        g_last = tl.load(glast_s + (ic * H + h) * D + ch)
        G = tl.load(g_s + out, mask=vmask, other=0.0)
        return tl.where(vmask, G, g_last[None, :]), g_last

    @triton.jit(do_not_specialize=["T"])
    def _kda_intra_kernel(
        qkv_s, g_s, glast_s, beta_s, starts, lens, chunks,
        w_s, u_s, qg_s, kg_s, aqk_s, T, scale,
        H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, LOG_C: tl.constexpr,
        BR: tl.constexpr, DS: tl.constexpr, SOLVE_PREC: tl.constexpr, WU_PREC: tl.constexpr,
    ):
        """One program per (chunk, head): everything the scan needs that does not depend on
        the state."""
        ic = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        n, c, pos, valid, tok = _chunk_rows(chunks, starts, lens, ic, C)
        vmask = valid[:, None]
        r = tl.arange(0, C)
        cols = tl.arange(0, C)
        e = tl.arange(0, DS)
        blk = r // BR
        base = h * T + tok
        plane = H * T * D
        beta = tl.load(beta_s + base, mask=valid, other=0.0)

        # Per slice: the slice's share of A and Aqk, from the raw (not yet l2-normed) q and k,
        # and of their squared norms; the norms are applied to A and Aqk as row and column scales.
        # exp(G_i - G_j) = exp(G_i - G_m) * exp(G_m - G_j) with G_m the middle row of the BR-row
        # block holding i: both factors stay within exp(|lb| * BR / 2) for i, j in the block and
        # the right one is <= 1 left of it. Columns right of the block get a zero exponent;
        # entries with j > i may still overflow and are masked after the dot.
        a = tl.zeros([C, C], tl.float32)
        aqk = tl.zeros([C, C], tl.float32)
        ssq_q = tl.zeros([C], tl.float32)
        ssq_k = tl.zeros([C], tl.float32)
        for s in range(0, D, DS):
            ch = s + e
            out = base[:, None] * D + ch[None, :]
            G, g_last = _load_g(g_s, glast_s, out, vmask, ic, h, ch, H, D)
            k = tl.load(qkv_s + plane + out, mask=vmask, other=0.0).to(tl.float32)
            q = tl.load(qkv_s + out, mask=vmask, other=0.0).to(tl.float32)
            ssq_q += tl.sum(q * q, 1)
            ssq_k += tl.sum(k * k, 1)
            g_row = tl.zeros([C, DS], tl.float32)
            for b in tl.static_range(C // BR):
                g_b = tl.sum(tl.where(r[:, None] == b * BR + BR // 2, G, 0.0), 0)
                g_row = tl.where((blk == b)[:, None], g_b[None, :], g_row)
            left = tl.exp2(G - g_row)
            kl = (k * left).to(tl.bfloat16)
            ql = (q * left).to(tl.bfloat16)
            for b in tl.static_range(C // BR):
                g_b = tl.sum(tl.where(r[:, None] == b * BR + BR // 2, G, 0.0), 0)
                right = tl.exp2(tl.where((blk <= b)[:, None], g_b[None, :] - G, 0.0))
                kr = tl.trans((k * right).to(tl.bfloat16))
                in_b = (blk == b)[:, None]
                a += tl.dot(tl.where(in_b, kl, tl.zeros_like(kl)), kr)
                aqk += tl.dot(tl.where(in_b, ql, tl.zeros_like(ql)), kr)
        q_nrm = scale / tl.sqrt(ssq_q + 1e-6)
        k_nrm = 1.0 / tl.sqrt(ssq_k + 1e-6)
        a = tl.where(cols[None, :] < r[:, None], a * (beta * k_nrm)[:, None] * k_nrm[None, :], 0.0)
        aqk = tl.where(cols[None, :] <= r[:, None], aqk * q_nrm[:, None] * k_nrm[None, :], 0.0)
        tl.store(aqk_s + base[:, None] * C + cols[None, :], aqk.to(aqk_s.dtype.element_ty),
                 mask=vmask)

        # T = (I + A)^-1 by recursive doubling of the diagonal blocks: once T holds the inverses
        # of the size-s blocks, each pair (B1, B2) of neighbours becomes one size-2s block with
        # T[B2, B1] = -T[B2, B2] A[B2, B1] T[B1, B1]. log2(C) levels of two dots each.
        t_mat = tl.where(r[:, None] == cols[None, :], 1.0, 0.0)
        for lvl in tl.static_range(LOG_C):
            size = 1 << lvl
            pair = (r // (2 * size))[:, None] == (cols // (2 * size))[None, :]
            lower = ((r // size) % 2 == 1)[:, None] & ((cols // size) % 2 == 0)[None, :] & pair
            x = _mm(tl.where(lower, a, 0.0), t_mat, SOLVE_PREC)
            t_mat = t_mat - _mm(t_mat, x, SOLVE_PREC)

        # Per slice: the scan's Q exp(G) and K exp(G_last - G), W = T (beta K exp(G)) and
        # U = T (beta V), with q and k l2-normed.
        for s in range(0, D, DS):
            ch = s + e
            out = base[:, None] * D + ch[None, :]
            G, g_last = _load_g(g_s, glast_s, out, vmask, ic, h, ch, H, D)
            k = tl.load(qkv_s + plane + out, mask=vmask, other=0.0).to(tl.float32) * k_nrm[:, None]
            q = tl.load(qkv_s + out, mask=vmask, other=0.0).to(tl.float32) * q_nrm[:, None]
            e_g = tl.exp2(G)
            tl.store(qg_s + out, (q * e_g).to(qg_s.dtype.element_ty), mask=vmask)
            tl.store(kg_s + out, (k * tl.exp2(g_last[None, :] - G)).to(kg_s.dtype.element_ty),
                     mask=vmask)
            x_w = k * beta[:, None] * e_g
            tl.store(w_s + out, _mm(t_mat, x_w, WU_PREC).to(w_s.dtype.element_ty), mask=vmask)
            v = tl.load(qkv_s + 2 * plane + out, mask=vmask, other=0.0).to(tl.float32)
            u = _mm(t_mat, v * beta[:, None], WU_PREC)
            tl.store(u_s + out, u.to(u_s.dtype.element_ty), mask=vmask)

    @triton.jit(do_not_specialize=["T"])
    def _kda_scan_kernel(
        w_s, u_s, qg_s, kg_s, aqk_s, glast_s, state_pool, stride_state,
        proj, stride_proj, conv_pool, stride_conv, o_out,
        starts, lens, slots, chunk_offsets, T,
        H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BV: tl.constexpr,
        PREC: tl.constexpr, STAGES: tl.constexpr,
    ):
        """One program per (V block, span, head), serial over the span's chunks. The fp32
        state ``[K, V]`` is read from the span's slot, stored V-first, and written back."""
        iv = tl.program_id(0)
        n = tl.program_id(1)
        h = tl.program_id(2)
        P = H * D
        bos = tl.load(starts + n)
        L = tl.load(lens + n)
        slot = tl.load(slots + n).to(tl.int64)
        c0 = tl.load(chunk_offsets + n).to(tl.int64)
        k = tl.arange(0, D)
        v = iv * BV + tl.arange(0, BV)
        r = tl.arange(0, C)
        sp = state_pool + slot * stride_state + h * D * D + k[:, None] + v[None, :] * D
        s = tl.load(sp)
        for c in tl.range(0, tl.cdiv(L, C), num_stages=STAGES):
            pos = c * C + r
            valid = (pos < L)[:, None]
            base = h * T + bos + pos
            w = tl.load(w_s + base[:, None] * D + k[None, :], mask=valid, other=0.0)
            qg = tl.load(qg_s + base[:, None] * D + k[None, :], mask=valid, other=0.0)
            kg = tl.load(kg_s + base[:, None] * D + k[None, :], mask=valid, other=0.0)
            u = tl.load(u_s + base[:, None] * D + v[None, :], mask=valid, other=0.0)
            aqk = tl.load(aqk_s + base[:, None] * C + r[None, :], mask=valid, other=0.0)
            decay = tl.exp2(tl.load(glast_s + ((c0 + c) * H + h) * D + k))
            v_new = u.to(tl.float32) - _mm(w, s, PREC)
            o = _mm(qg, s, PREC) + _mm(aqk, v_new, PREC)
            tl.store(o_out + (bos + pos)[:, None] * P + h * D + v[None, :],
                     o.to(o_out.dtype.element_ty), mask=valid)
            s = s * decay[:, None] + _mm(tl.trans(kg), v_new, PREC)
        tl.store(sp, s)

        if iv == 0:
            # The new conv tail: the span's last three raw inputs, older ones from the old
            # tail when the span is shorter than that. Loads before the barrier, stores after.
            m = tl.arange(0, 4)
            cstate = conv_pool + slot * stride_conv
            tq = _new_tail(proj, stride_proj, cstate, h * D + k, bos, L, m)
            tk = _new_tail(proj, stride_proj, cstate, P + h * D + k, bos, L, m)
            tv = _new_tail(proj, stride_proj, cstate, 2 * P + h * D + k, bos, L, m)
            tl.debug_barrier()
            live = (m < 3)[None, :]
            tl.store(cstate + (h * D + k)[:, None] * 3 + m[None, :], tq, mask=live)
            tl.store(cstate + (P + h * D + k)[:, None] * 3 + m[None, :], tk, mask=live)
            tl.store(cstate + (2 * P + h * D + k)[:, None] * 3 + m[None, :], tv, mask=live)

    @triton.jit(do_not_specialize=["T"])
    def _kda_out_norm_kernel(o_in, proj, stride_proj, g_b, norm_w, out, T, norm_eps,
                             H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
        """Output gate g_b(g_a) and the gated RMSNorm over each head, for BT tokens."""
        rows = tl.program_id(0) * BT + tl.arange(0, BT)
        h = tl.program_id(1)
        P = H * D
        d = tl.arange(0, D)
        valid = (rows < T)[:, None]
        rows = rows.to(tl.int64)
        o = tl.load(o_in + rows[:, None] * P + h * D + d[None, :], mask=valid, other=0.0).to(tl.float32)
        g_a = tl.load(proj + rows[:, None] * stride_proj + 3 * P + H + D + d[None, :], mask=valid, other=0.0)
        gb = tl.load(g_b + (h * D + d)[:, None] * D + d[None, :])
        gate = tl.dot(g_a, tl.trans(gb)).to(tl.bfloat16).to(tl.float32)
        y = o * tl.rsqrt(tl.sum(o * o, 1) / D + norm_eps)[:, None]
        y = y * tl.load(norm_w + d).to(tl.float32)[None, :] * _sigmoid(gate)
        tl.store(out + rows[:, None] * P + h * D + d[None, :], y.to(out.dtype.element_ty), mask=valid)


def kda_prefill(proj, conv_pool, conv_w, f_b, g_b, dt_bias, a_log, norm_w, state_pool, varlen,
                *, num_heads, head_dim, lower_bound, scale, norm_eps, cfg: PrefillConfig | None = None):
    """Fused KDA prefill for ``proj (T, 3P + H + 2D)`` over ``varlen``'s spans; returns the
    gated-normed ``(T, P)``. The spans' slots in ``conv_pool (S, 3P, 3)`` and
    ``state_pool (S, H, V, K)`` fp32 are updated in place."""
    cfg = cfg or CONFIG
    H, D, C = num_heads, head_dim, varlen.chunk
    T, P = proj.shape[0], num_heads * head_dim
    assert varlen.num_tokens == T, (varlen.num_tokens, T)
    assert conv_pool.shape[-1] == 3 and conv_w.shape[-1] == 4, "width-4 conv only"
    assert state_pool.is_contiguous() and conv_pool.is_contiguous() and proj.stride(1) == 1
    br = min(cfg.row_block, C)
    assert C % br == 0 and C & (C - 1) == 0 and D >= 16
    # The in-block decay factors reach exp(|lb| * br / 2); fp32 tops out at exp(88.7).
    assert abs(lower_bound) * (br // 2) <= 80.0, (lower_bound, br)
    dev, dt = proj.device, cfg.scratch_dtype
    out = torch.empty(T, P, dtype=proj.dtype, device=dev)
    if T == 0:
        return out
    qkv_s = torch.empty(3, H, T, D, dtype=torch.bfloat16, device=dev)
    g_s = torch.empty(H, T, D, dtype=torch.float32, device=dev)
    glast_s = torch.empty(varlen.num_chunks, H, D, dtype=torch.float32, device=dev)
    beta_s = torch.empty(H, T, dtype=torch.float32, device=dev)
    _kda_conv_kernel[(varlen.num_chunks, 4, H)](
        proj, proj.stride(0), conv_pool, conv_pool.stride(0), conv_w, f_b, dt_bias, a_log,
        varlen.starts, varlen.lens, varlen.slots, varlen.chunks,
        qkv_s, g_s, glast_s, beta_s, T, float(lower_bound),
        H=H, D=D, C=C, R=min(cfg.conv_groups, C), num_warps=cfg.conv_warps,
    )
    w_s, u_s, qg_s, kg_s = (torch.empty(H, T, D, dtype=dt, device=dev) for _ in range(4))
    aqk_s = torch.empty(H, T, C, dtype=dt, device=dev)
    _kda_intra_kernel[(varlen.num_chunks, H)](
        qkv_s, g_s, glast_s, beta_s, varlen.starts, varlen.lens, varlen.chunks,
        w_s, u_s, qg_s, kg_s, aqk_s, T, float(scale),
        H=H, D=D, C=C, LOG_C=C.bit_length() - 1, BR=br, DS=min(cfg.slice, D),
        SOLVE_PREC=cfg.solve_prec, WU_PREC=cfg.wu_prec, num_warps=cfg.intra_warps,
    )
    o = torch.empty(T, P, dtype=proj.dtype, device=dev)
    bv = cfg.scan_bv or (32 if varlen.num_spans * H * D // 32 <= _num_sms(dev) else 64)
    bv = min(bv, D)
    _kda_scan_kernel[(D // bv, varlen.num_spans, H)](
        w_s, u_s, qg_s, kg_s, aqk_s, glast_s, state_pool, state_pool.stride(0),
        proj, proj.stride(0), conv_pool, conv_pool.stride(0), o,
        varlen.starts, varlen.lens, varlen.slots, varlen.chunk_offsets, T,
        H=H, D=D, C=C, BV=bv, PREC=cfg.scan_prec, STAGES=cfg.scan_stages,
        num_warps=cfg.scan_warps,
    )
    _kda_out_norm_kernel[(triton.cdiv(T, 64), H)](
        o, proj, proj.stride(0), g_b, norm_w, out, T, float(norm_eps),
        H=H, D=D, BT=64, num_warps=4,
    )
    return out


BV_VERIFY = 32  # value columns per verify program


if _HAS_TRITON:

    @triton.jit
    def _conv_tok(x, s0, s1, s2, w0, w1, w2, w3):
        """Width-4 causal conv + SiLU of one token, rounded like the decode kernel's."""
        acc = s0 * w0 + s1 * w1 + s2 * w2 + x * w3
        return (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _kda_verify_kernel(
        proj, stride_proj, gpre, stride_gpre,
        prefix, sp_slot, sp_side, sp_tok, gc, sg_slot, sg_side, sg_tok,
        bc, sb_slot, sb_side, sb_tok, win, sw_slot, sw_side, prefix_len, side_of,
        conv_pool, stride_conv, conv_w, dt_bias, a_log,
        state_pool, stride_state, slot_ids, o_out, stride_o,
        lower_bound, scale,
        T: tl.constexpr, KMAX: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        BV: tl.constexpr,
    ):
        """One program per (row, head, BV value columns). Replays the row's accepted prefix
        from the checkpoint and stores the state there, then runs its ``T`` block rows of
        ``proj`` (the gate pre-activations in ``gpre``), writing their raw outputs and caching
        their inputs on the slot's other side, which no program of this step reads."""
        n = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        iv = tl.program_id(2)
        P = H * D
        d = tl.arange(0, D)
        dv = iv * BV + tl.arange(0, BV)
        slot = tl.load(slot_ids + n).to(tl.int64)
        # Padding rows share the sink slot, whose count is garbage: clamp it.
        plen = tl.minimum(tl.maximum(tl.load(prefix_len + slot), 0), KMAX)
        side = (tl.load(side_of + slot) & 1).to(tl.int64)

        cq = h * D + d
        ck = P + h * D + d
        cv = 2 * P + h * D + dv
        qw0 = tl.load(conv_w + cq * 4 + 0)
        qw1 = tl.load(conv_w + cq * 4 + 1)
        qw2 = tl.load(conv_w + cq * 4 + 2)
        qw3 = tl.load(conv_w + cq * 4 + 3)
        kw0 = tl.load(conv_w + ck * 4 + 0)
        kw1 = tl.load(conv_w + ck * 4 + 1)
        kw2 = tl.load(conv_w + ck * 4 + 2)
        kw3 = tl.load(conv_w + ck * 4 + 3)
        vw0 = tl.load(conv_w + cv * 4 + 0)
        vw1 = tl.load(conv_w + cv * 4 + 1)
        vw2 = tl.load(conv_w + cv * 4 + 2)
        vw3 = tl.load(conv_w + cv * 4 + 3)

        # The conv window before the prefix: the pool's on a request's first verify step
        # (right after its prefill, the only time the prefix is empty), else the cached one.
        if plen == 0:
            cw = conv_pool + slot * stride_conv
        else:
            cw = win + slot * sw_slot + side * sw_side
        qs0 = tl.load(cw + cq * 3 + 0).to(tl.float32)
        qs1 = tl.load(cw + cq * 3 + 1).to(tl.float32)
        qs2 = tl.load(cw + cq * 3 + 2).to(tl.float32)
        ks0 = tl.load(cw + ck * 3 + 0).to(tl.float32)
        ks1 = tl.load(cw + ck * 3 + 1).to(tl.float32)
        ks2 = tl.load(cw + ck * 3 + 2).to(tl.float32)
        vs0 = tl.load(cw + cv * 3 + 0).to(tl.float32)
        vs1 = tl.load(cw + cv * 3 + 1).to(tl.float32)
        vs2 = tl.load(cw + cv * 3 + 2).to(tl.float32)

        bias = tl.load(dt_bias + h * D + d)
        a_exp = tl.exp(tl.load(a_log + h))
        # s[v, k] as the pool holds it: the decay runs along k, the reads reduce over it
        sp = state_pool + slot * stride_state + h * D * D + dv[:, None] * D + d[None, :]
        s = tl.load(sp)

        # The prefix: the last block's accepted tokens. No outputs.
        pr = prefix + slot * sp_slot + side * sp_side
        gr = gc + slot * sg_slot + side * sg_side + h * D
        br = bc + slot * sb_slot + side * sb_side + h
        for i in range(plen):
            xq = tl.load(pr + i * sp_tok + cq).to(tl.float32)
            xk = tl.load(pr + i * sp_tok + ck).to(tl.float32)
            xv = tl.load(pr + i * sp_tok + cv).to(tl.float32)
            g = tl.load(gr + i * sg_tok + d).to(tl.float32) + bias
            beta = tl.sigmoid(tl.load(br + i * sb_tok).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            k = _conv_tok(xk, ks0, ks1, ks2, kw0, kw1, kw2, kw3)
            v = _conv_tok(xv, vs0, vs1, vs2, vw0, vw1, vw2, vw3)
            qs0, qs1, qs2 = qs1, qs2, xq
            ks0, ks1, ks2 = ks1, ks2, xk
            vs0, vs1, vs2 = vs1, vs2, xv
            k = k / tl.sqrt(tl.sum(k * k, 0) + 1e-6)
            s = s * tl.exp(lower_bound * tl.sigmoid(a_exp * g))[None, :]
            s = s + ((v - tl.sum(s * k[None, :], 1)) * beta)[:, None] * k[None, :]
        if plen > 0:
            tl.store(sp, s)  # the checkpoint

        # The window before the block, for the next step's prefix.
        ww = win + slot * sw_slot + (1 - side) * sw_side
        wt = win.dtype.element_ty
        if iv == 0:
            tl.store(ww + cq * 3 + 0, qs0.to(wt))
            tl.store(ww + cq * 3 + 1, qs1.to(wt))
            tl.store(ww + cq * 3 + 2, qs2.to(wt))
            tl.store(ww + ck * 3 + 0, ks0.to(wt))
            tl.store(ww + ck * 3 + 1, ks1.to(wt))
            tl.store(ww + ck * 3 + 2, ks2.to(wt))
        tl.store(ww + cv * 3 + 0, vs0.to(wt))
        tl.store(ww + cv * 3 + 1, vs1.to(wt))
        tl.store(ww + cv * 3 + 2, vs2.to(wt))

        pw = prefix + slot * sp_slot + (1 - side) * sp_side
        gw = gc + slot * sg_slot + (1 - side) * sg_side + h * D
        bw = bc + slot * sb_slot + (1 - side) * sb_side + h
        pt = prefix.dtype.element_ty
        for j in tl.static_range(T):
            row = proj + (n * T + j) * stride_proj
            xq = tl.load(row + cq).to(tl.float32)
            xk = tl.load(row + ck).to(tl.float32)
            xv = tl.load(row + cv).to(tl.float32)
            b = tl.load(row + 3 * P + h)
            gp = tl.load(gpre + (n * T + j) * stride_gpre + h * D + d)
            beta = tl.sigmoid(b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            q = _conv_tok(xq, qs0, qs1, qs2, qw0, qw1, qw2, qw3)
            k = _conv_tok(xk, ks0, ks1, ks2, kw0, kw1, kw2, kw3)
            v = _conv_tok(xv, vs0, vs1, vs2, vw0, vw1, vw2, vw3)
            qs0, qs1, qs2 = qs1, qs2, xq
            ks0, ks1, ks2 = ks1, ks2, xk
            vs0, vs1, vs2 = vs1, vs2, xv
            q = q / tl.sqrt(tl.sum(q * q, 0) + 1e-6) * scale
            k = k / tl.sqrt(tl.sum(k * k, 0) + 1e-6)
            s = s * tl.exp(lower_bound * tl.sigmoid(a_exp * (gp.to(tl.float32) + bias)))[None, :]
            s = s + ((v - tl.sum(s * k[None, :], 1)) * beta)[:, None] * k[None, :]
            o = tl.sum(s * q[None, :], 1).to(tl.bfloat16)
            tl.store(o_out + (n * T + j) * stride_o + h * D + dv, o.to(o_out.dtype.element_ty))
            if iv == 0:
                tl.store(pw + j * sp_tok + cq, xq.to(pt))
                tl.store(pw + j * sp_tok + ck, xk.to(pt))
                tl.store(gw + j * sg_tok + d, gp.to(gc.dtype.element_ty))
                tl.store(bw + j * sb_tok, b.to(bc.dtype.element_ty))
            tl.store(pw + j * sp_tok + cv, xv.to(pt))


def kda_verify(proj, gpre, conv_pool, conv_w, g_b, dt_bias, a_log, norm_w, state_pool, slot_ids,
               spec, *, block, num_heads, head_dim, lower_bound, scale, norm_eps):
    """Fused KDA for a verify step over ``proj (N * block, 3P + H + 2D)``, rows grouped by
    request, with ``gpre (N * block, P)`` the forget gate's ``f_b(f_a)``; returns the
    gated-normed ``(N * block, P)``. ``state_pool (S, H, V, K)`` fp32 holds the checkpoint and
    advances over each row's accepted prefix; ``conv_pool (S, 3P, 3)`` is read on a request's
    first verify step only; ``spec`` is the layer's ``SpecBlocks``."""
    rows = proj.shape[0]
    H, D = num_heads, head_dim
    kmax = spec.prefix.shape[2]
    assert rows % block == 0 and block <= kmax, (rows, block, kmax)
    assert conv_pool.shape[-1] == 3 and conv_w.shape[-1] == 4, "width-4 conv only"
    assert state_pool.is_contiguous() and conv_pool.is_contiguous() and proj.stride(1) == 1
    assert all(t.is_contiguous() for t in spec) and D % BV_VERIFY == 0
    assert spec.conv.dtype == conv_pool.dtype == spec.g.dtype == torch.bfloat16
    o = torch.empty(rows, H * D, dtype=proj.dtype, device=proj.device)
    out = torch.empty_like(o)
    if rows:
        assert gpre.shape == (rows, H * D) and gpre.stride(1) == 1 and gpre.dtype == torch.bfloat16
        _kda_verify_kernel[(rows // block, H, D // BV_VERIFY)](
            proj, proj.stride(0), gpre, gpre.stride(0),
            spec.prefix, *spec.prefix.stride()[:3], spec.g, *spec.g.stride()[:3],
            spec.beta, *spec.beta.stride()[:3], spec.conv, *spec.conv.stride()[:2],
            spec.prefix_len, spec.side,
            conv_pool, conv_pool.stride(0), conv_w, dt_bias, a_log,
            state_pool, state_pool.stride(0), slot_ids, o, o.stride(0),
            float(lower_bound), float(scale),
            T=block, KMAX=kmax, H=H, D=D, BV=BV_VERIFY, num_warps=4,
        )
        _kda_out_norm_kernel[(triton.cdiv(rows, 64), H)](
            o, proj, proj.stride(0), g_b, norm_w, out, rows, float(norm_eps),
            H=H, D=D, BT=64, num_warps=4,
        )
    return out


def _merged_row(qkv, g, beta, gate, p) -> torch.Tensor:
    """``qkv`` as the head of the projection row the kernels read beta, f_a and g_a from,
    past its end. All four must be column slices of that one row."""
    if p.f_b is None or p.g_b is None or p.norm_weight is None or gate is None:
        raise ValueError(
            "the fused KDA kernels apply both gates' up-projections and the gated RMSNorm: "
            "pass params.f_b, params.g_b, params.norm_weight and the output gate"
        )
    if p.lower_bound is None:
        raise ValueError("the fused KDA kernels take the bounded forget gate (lower_bound)")
    size = qkv.element_size()
    for name, t, col in (("beta", beta, qkv.shape[1]), ("g", g, qkv.shape[1] + p.num_heads),
                         ("gate", gate, qkv.shape[1] + p.num_heads + p.head_dim)):
        if t.data_ptr() != qkv.data_ptr() + col * size or t.stride() != (qkv.stride(0), 1):
            raise ValueError(
                f"{name} is not the column slice at {col} of qkv's projection row; the fused "
                "KDA kernels read q | k | v | beta | f_a | g_a as one row"
            )
    return qkv


class TritonKDAKernels:
    """``kda_decode``, ``kda_prefill`` and ``kda_verify`` as a ``KDAManager`` kernel bundle.

    Both fuse the forget gate's up-projection and the output gate with the gated RMSNorm, so
    they need ``params.f_b``, ``g_b`` and ``norm_weight``, take the low-rank gates, and return
    the normed ``[T, P]``. Every address comes from the plan's tensors, so a captured step
    replays with new slots and spans.
    """

    def __init__(self, config: PrefillConfig | None = None):
        self.config = config or CONFIG

    def layout(self, spans, num_tokens: int, fixed: bool) -> list[int]:
        """The prefill's span layout; ``fixed`` sizes it by ``num_tokens`` for a capture."""
        return varlen_layout(spans, num_tokens if fixed else None, self.config.chunk)

    @torch.compiler.disable
    def run_paged(self, qkv, g, beta, plan, conv_state, rec_state, p, gate=None):
        """``qkv [T, 3P]`` pre-conv, ``g`` and ``gate`` the low-rank forget and output gates
        ``[T, D]``, ``beta [T, H]`` raw; the pool's layer blocks ``conv_state [S, 3P, W - 1]``
        and ``rec_state [S, H, V, K]`` are updated in place. Returns ``[T, P]``."""
        proj = _merged_row(qkv, g, beta, gate, p)
        kwargs = dict(num_heads=p.num_heads, head_dim=p.head_dim, lower_bound=p.lower_bound,
                      scale=p.scale, norm_eps=p.norm_eps)
        if plan.is_decode:
            return kda_decode(proj, conv_state, p.conv_weight, p.f_b, p.g_b, p.dt_bias, p.A_log,
                              p.norm_weight, rec_state, plan.slot_ids, **kwargs)
        varlen = plan.cached("kda_triton", lambda: varlen_view(
            plan.layout, plan.slot_ids, plan.num_rows, plan.num_tokens, self.config.chunk))
        return kda_prefill(proj, conv_state, p.conv_weight, p.f_b, p.g_b, p.dt_bias, p.A_log,
                           p.norm_weight, rec_state, varlen, cfg=self.config, **kwargs)

    @torch.compiler.disable
    def run_verify(self, qkv, g, beta, plan, conv_state, rec_state, spec, p, gate=None):
        """``run_paged``'s arguments for a verify step (``plan.is_verify``), plus the layer's
        ``SpecBlocks``; the state advances over each row's accepted prefix only."""
        proj = _merged_row(qkv, g, beta, gate, p)
        # the forget gate's up-projection as one GEMM; the kernel caches it for the replay
        gpre = F.linear(g, p.f_b)
        return kda_verify(proj, gpre, conv_state, p.conv_weight, p.g_b, p.dt_bias, p.A_log,
                          p.norm_weight, rec_state, plan.slot_ids, spec, block=plan.block,
                          num_heads=p.num_heads, head_dim=p.head_dim, lower_bound=p.lower_bound,
                          scale=p.scale, norm_eps=p.norm_eps)
