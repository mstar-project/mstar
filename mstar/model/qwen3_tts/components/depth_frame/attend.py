"""`cp_attend` for 4 (token, head) pairs at a time, one per warp, in Gluon.

Today's kernel runs one pair per 1-warp program. Here each warp holds one
pair in exactly the layout that program uses (from its TTGIR: [64] halves in
[2]/[32], [16, 64] keys in [1, 8]/[4, 8], [16, 128] values in [1, 8]/[2, 16],
[128] rows in [4]/[32]), with the pair index added as a leading dimension
across the 4 warps. No reduction crosses that dimension, so every reduction
runs the same in-thread and shuffle order as today's and the outputs match
bit for bit. Decode only: one depth position per step (SEQ 1).
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.jit
def rope_table_kernel(T, theta, D: tl.constexpr, HALF_D: tl.constexpr):
    """Row p: cos then sin of RoPE angle p * inv_freq, written as `rms_norm_rope` computes them."""
    p = tl.program_id(0)
    offs_half = tl.arange(0, HALF_D)
    pos = p.to(tl.float32)
    inv_freq = tl.exp(-tl.log(theta) * (offs_half.to(tl.float32) * 2.0 / D))
    angle = pos * inv_freq
    tl.store(T + p * D + offs_half, tl.cos(angle))
    tl.store(T + p * D + HALF_D + offs_half, tl.sin(angle))


def rope_table(positions: int, theta: float, head_dim: int, device):
    import torch
    t = torch.empty(positions, head_dim, device=device, dtype=torch.float32)
    rope_table_kernel[(positions,)](t, theta, D=head_dim, HALF_D=head_dim // 2)
    return t


def layouts():
    L1 = gl.BlockedLayout([1, 2], [1, 32], [4, 1], [1, 0])              # [4, 64]
    L3 = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])              # [4, 128]
    K3 = gl.BlockedLayout([1, 1, 8], [1, 4, 8], [4, 1, 1], [2, 1, 0])   # [4, 16, 64]
    V3 = gl.BlockedLayout([1, 1, 8], [1, 2, 16], [4, 1, 1], [2, 1, 0])  # [4, 16, 128]
    return L1, L3, K3, V3


@gluon.jit
def _norm_rope4(x1, x2, w1, w2, cos, sin, eps,
                D: gl.constexpr, HALF_D: gl.constexpr, L1: gl.constexpr, IS_K: gl.constexpr):
    """`rms_norm_rope` on 4 heads at once: x1, x2 [4, HALF_D] fp32, the norm
    weight's halves w1, w2 [HALF_D] fp32, each row's RoPE cos and sin [4, HALF_D].

    The rotation is written as the FMAs today's compiled kernel ends up with
    (its SASS): left to the compiler, which product gets fused depends on the
    surrounding code, and where the two products cancel, one ulp moves."""
    offs = gl.arange(0, HALF_D, layout=gl.SliceLayout(0, L1))
    sumsq = gl.sum(x1 * x1, axis=1) + gl.sum(x2 * x2, axis=1)
    inv_rms = 1.0 / gl.sqrt(sumsq / D + eps)
    x1 = x1 * inv_rms[:, None] * w1[None, :]
    x2 = x2 * inv_rms[:, None] * w2[None, :]
    if IS_K:
        return gl.fma(x1, cos, -(x2 * sin)), gl.fma(x1, sin, x2 * cos)
    return gl.fma(x1, cos, -(x2 * sin)), gl.fma(x2, cos, x1 * sin)


@gluon.jit
def attend4(
    QKV, s_qkv, Wq, Wk, Pos, ROPE, KV, s_kv_b, s_kv_c, s_kv_n, s_kv_h, O, s_o,
    m0, h, M, cache_pos, eps_q, eps_k, sm_scale,
    H_Q: gl.constexpr, H_KV: gl.constexpr, D: gl.constexpr, HALF_D: gl.constexpr, BLOCK_N: gl.constexpr,
    L1: gl.constexpr, L3: gl.constexpr, K3: gl.constexpr, V3: gl.constexpr,
):
    """Tokens m0 .. m0 + 3, q head h: q/k norm, RoPE, the cache write (first q
    head of each kv head) and attention over depth positions 0 .. cache_pos."""
    elem: gl.constexpr = gl.bfloat16
    hk = h // (H_Q // H_KV)
    # [4, *] row tensors
    r1 = m0 + gl.arange(0, 4, layout=gl.SliceLayout(1, L1))
    r3 = m0 + gl.arange(0, 4, layout=gl.SliceLayout(1, L3))
    offs = gl.arange(0, HALF_D, layout=gl.SliceLayout(0, L1))
    d3 = gl.arange(0, D, layout=gl.SliceLayout(0, L3))
    ok1 = (r1 < M)[:, None]
    ok3 = (r3 < M)[:, None]
    row1 = QKV + r1[:, None] * s_qkv
    row3 = QKV + r3[:, None] * s_qkv
    kr = m0 + gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, K3)))
    kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(2, K3)))
    kd = gl.arange(0, HALF_D, layout=gl.SliceLayout(0, gl.SliceLayout(1, K3)))
    vr = m0 + gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, V3)))
    vn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(2, V3)))
    vd = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(1, V3)))
    kbase = KV + kr[:, None, None] * s_kv_b + hk * s_kv_h
    kmask = (kn < cache_pos)[None, :, None] & (kr < M)[:, None, None]
    k_off = kn[None, :, None] * s_kv_n + kd[None, None, :]
    vbase = KV + vr[:, None, None] * s_kv_b + hk * s_kv_h
    vmask = (vn < cache_pos)[None, :, None] & (vr < M)[:, None, None]

    # every global load first: a layout conversion is a CTA barrier, and no load moves across one
    pos = gl.load(Pos + r1, mask=r1 < M, other=0)
    cos = gl.load(ROPE + pos[:, None] * D + offs[None, :])
    sin = gl.load(ROPE + pos[:, None] * D + HALF_D + offs[None, :])
    xq1 = gl.load(row1 + h * D + offs[None, :], mask=ok1, other=0.0).to(gl.float32)
    xq2 = gl.load(row1 + h * D + HALF_D + offs[None, :], mask=ok1, other=0.0).to(gl.float32)
    xk1 = gl.load(row1 + (H_Q + hk) * D + offs[None, :], mask=ok1, other=0.0).to(gl.float32)
    xk2 = gl.load(row1 + (H_Q + hk) * D + HALF_D + offs[None, :], mask=ok1, other=0.0).to(gl.float32)
    wq1 = gl.load(Wq + offs).to(gl.float32)
    wq2 = gl.load(Wq + offs + HALF_D).to(gl.float32)
    wk1 = gl.load(Wk + offs).to(gl.float32)
    wk2 = gl.load(Wk + offs + HALF_D).to(gl.float32)
    vj = gl.load(row3 + (H_Q + H_KV + hk) * D + d3[None, :], mask=ok3, other=0.0).to(elem)
    k1 = gl.load(kbase + k_off, mask=kmask, other=0.0).to(gl.float32)
    k2 = gl.load(kbase + k_off + HALF_D, mask=kmask, other=0.0).to(gl.float32)
    v = gl.load(vbase + s_kv_c + vn[None, :, None] * s_kv_n + vd[None, None, :], mask=vmask, other=0.0).to(gl.float32)

    q1, q2 = _norm_rope4(xq1, xq2, wq1, wq2, cos, sin, eps_q, D, HALF_D, L1, False)
    q1 = gl.convert_layout(q1.to(elem), gl.SliceLayout(1, K3)).to(gl.float32) * sm_scale
    q2 = gl.convert_layout(q2.to(elem), gl.SliceLayout(1, K3)).to(gl.float32) * sm_scale
    # this step's own key and value, from qkv (no wait on another program's cache write)
    kj1, kj2 = _norm_rope4(xk1, xk2, wk1, wk2, cos, sin, eps_k, D, HALF_D, L1, True)
    kj1 = kj1.to(elem)
    kj2 = kj2.to(elem)
    if h % (H_Q // H_KV) == 0:
        dst1 = KV + r1[:, None] * s_kv_b + hk * s_kv_h + cache_pos * s_kv_n
        gl.store(dst1 + offs[None, :], kj1, mask=ok1)
        gl.store(dst1 + HALF_D + offs[None, :], kj2, mask=ok1)
        dst3 = KV + r3[:, None] * s_kv_b + hk * s_kv_h + cache_pos * s_kv_n
        gl.store(dst3 + s_kv_c + d3[None, :], vj, mask=ok3)
    khere = (kn == cache_pos)[None, :, None]
    vhere = (vn == cache_pos)[None, :, None]
    k1 = gl.where(khere, gl.convert_layout(kj1, gl.SliceLayout(1, K3)).to(gl.float32)[:, None, :], k1)
    k2 = gl.where(khere, gl.convert_layout(kj2, gl.SliceLayout(1, K3)).to(gl.float32)[:, None, :], k2)
    v = gl.where(vhere, gl.convert_layout(vj, gl.SliceLayout(1, V3)).to(gl.float32)[:, None, :], v)

    valid = (kn <= cache_pos)[None, :]
    qk = gl.where(valid, gl.sum(k1 * q1[:, None, :], 2) + gl.sum(k2 * q2[:, None, :], 2), float("-inf"))
    p = gl.exp(qk - gl.max(qk, 1)[:, None])
    pv = gl.convert_layout(p, gl.SliceLayout(2, V3))
    num = gl.sum(pv[:, :, None] * v, 1)
    den = gl.convert_layout(gl.sum(p, 1), gl.SliceLayout(1, gl.SliceLayout(1, V3)))
    out = num / den[:, None]
    orow = m0 + gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(1, V3)))
    gl.store(O + orow[:, None] * s_o + h * D + vd[None, :], out.to(O.dtype.element_ty), mask=(orow < M)[:, None])
