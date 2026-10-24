"""Code-predictor attention prologue in one launch.

Per layer and depth step the code predictor normalized and roped q and k with one
kernel each (plus an int32 cast of the positions before each), then copied k and v
into its dense KV cache with two strided copies: six launches of a few microseconds
around a ~1 us job, 16 x 5 times per Talker step. This does all of it in one, with
the same per-element arithmetic as ``mstar.utils.attention._qk_norm_rope_kernel``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cp_qk_norm_rope_kv_kernel(
    QKV, Wq, Wk, Pos, KV,
    eps_q, eps_k, rope_theta, seq_len, cache_pos,
    s_qkv,
    s_kv_b, s_kv_c, s_kv_n, s_kv_h,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, HALF_D)
    row = QKV + m * s_qkv
    # programs [0, H_Q) take the q heads, [H_Q, H_Q + H_KV) the k heads and their v
    base = row + h * D
    if h < H_Q:
        w = Wq
        eps = eps_q
    else:
        w = Wk
        eps = eps_k

    x_first = tl.load(base + offs).to(tl.float32)
    x_second = tl.load(base + offs + HALF_D).to(tl.float32)
    sumsq = tl.sum(x_first * x_first, axis=0) + tl.sum(x_second * x_second, axis=0)
    inv_rms = 1.0 / tl.sqrt(sumsq / D + eps)
    x_first = x_first * inv_rms * tl.load(w + offs).to(tl.float32)
    x_second = x_second * inv_rms * tl.load(w + offs + HALF_D).to(tl.float32)

    pos = tl.load(Pos + m).to(tl.float32)
    inv_freq = tl.exp(-tl.log(rope_theta) * (offs.to(tl.float32) * 2.0 / D))
    angle = pos * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    out_first = x_first * cos - x_second * sin
    out_second = x_second * cos + x_first * sin

    if h < H_Q:
        tl.store(base + offs, out_first.to(QKV.dtype.element_ty))
        tl.store(base + offs + HALF_D, out_second.to(QKV.dtype.element_ty))
    else:
        hk = h - H_Q
        b = m // seq_len
        n = cache_pos + m % seq_len
        k_dst = KV + b * s_kv_b + n * s_kv_n + hk * s_kv_h
        tl.store(k_dst + offs, out_first.to(KV.dtype.element_ty))
        tl.store(k_dst + offs + HALF_D, out_second.to(KV.dtype.element_ty))
        d = tl.arange(0, D)
        v = tl.load(row + (H_Q + H_KV + hk) * D + d)
        tl.store(k_dst + s_kv_c + d, v.to(KV.dtype.element_ty))


def cp_qk_norm_rope_kv_write(
    qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, pos: torch.Tensor,
    kv_layer: torch.Tensor, cache_pos: int, seq_len: int, eps_q: float, eps_k: float,
    rope_theta: float, n_q_heads: int, n_kv_heads: int, head_dim: int,
) -> None:
    """QK-norm + RoPE the q heads of ``qkv (tokens, (Hq + 2 Hkv) * D)`` in place, and
    write the normed, roped k and the raw v of token ``b * seq_len + s`` into
    ``kv_layer (bs, 2, S, Hkv, D)`` at sequence position ``cache_pos + s``. ``pos`` holds
    one position per token."""
    tokens = qkv.shape[0]
    assert qkv.stride(1) == 1 and kv_layer.stride(-1) == 1
    _cp_qk_norm_rope_kv_kernel[(tokens, n_q_heads + n_kv_heads)](
        qkv, q_norm_w, k_norm_w, pos, kv_layer,
        eps_q, eps_k, rope_theta, seq_len, cache_pos,
        qkv.stride(0),
        kv_layer.stride(0), kv_layer.stride(1), kv_layer.stride(2), kv_layer.stride(3),
        H_Q=n_q_heads, H_KV=n_kv_heads, D=head_dim, HALF_D=head_dim // 2,
        num_warps=4, num_stages=1,
    )


@triton.jit
def _cp_embed_accumulate_kernel(
    Tok, W, Emb, Sum, Codes, group, s_w, s_emb, s_sum, s_codes,
    H: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    tok = tl.load(Tok + b)
    offs = tl.arange(0, BLOCK)
    for h0 in tl.static_range(0, H, BLOCK):
        e = tl.load(W + tok * s_w + h0 + offs)
        tl.store(Emb + b * s_emb + h0 + offs, e)
        acc = tl.load(Sum + b * s_sum + h0 + offs)
        tl.store(Sum + b * s_sum + h0 + offs, acc + e.to(acc.dtype))
    tl.store(Codes + b * s_codes + group, tok.to(Codes.dtype.element_ty))


def cp_embed_accumulate(
    tokens: torch.Tensor, weight: torch.Tensor, emb_sum: torch.Tensor,
    all_codes: torch.Tensor, group: int,
) -> torch.Tensor:
    """For a depth step's sampled ``tokens (bs,)``: record them as ``all_codes[:, group]``,
    look up their embeddings in ``weight (V, H)`` and add those to ``emb_sum (bs, H)`` in
    its dtype. Returns the embeddings, the next depth step's input. One launch for what
    was an index_select, a copy and a promoting add."""
    bs, hidden = tokens.shape[0], weight.shape[1]
    assert weight.stride(1) == 1 and emb_sum.stride(1) == 1 and hidden % 256 == 0
    emb = torch.empty(bs, hidden, dtype=weight.dtype, device=weight.device)
    _cp_embed_accumulate_kernel[(bs,)](
        tokens, weight, emb, emb_sum, all_codes, group,
        weight.stride(0), emb.stride(0), emb_sum.stride(0), all_codes.stride(0),
        H=hidden, BLOCK=256, num_warps=4,
    )
    return emb



@triton.jit
def _cp_attn_kernel(Q, K, V, O, cache_pos, sm_scale,
                    sq_b, sq_s, sq_h, sk_b, sk_n, sk_h, so_b, so_s, so_h,
                    GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    hk = h // GROUP
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + b * sq_b + s * sq_s + h * sq_h + d).to(tl.float32) * sm_scale
    valid = n <= cache_pos + s  # causal within the depth sequence
    kv = b * sk_b + n[:, None] * sk_n + hk * sk_h + d[None, :]
    k = tl.load(K + kv, mask=valid[:, None], other=0.0).to(tl.float32)
    qk = tl.where(valid, tl.sum(k * q[None, :], 1), float("-inf"))
    p = tl.exp(qk - tl.max(qk, 0))
    v = tl.load(V + kv, mask=valid[:, None], other=0.0).to(tl.float32)
    out = tl.sum(p[:, None] * v, 0) / tl.sum(p, 0)
    tl.store(O + b * so_b + s * so_s + h * so_h + d, out.to(O.dtype.element_ty))


def cp_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_pos: int) -> torch.Tensor:
    """Causal attention of the code predictor's ``q (bs, S, Hq, D)``, whose tokens sit
    at depth positions ``cache_pos .. cache_pos + S - 1``, over its dense cache
    ``(bs, max_depth, Hkv, D)``. The depth sequence is at most 16 long, so one
    program holds a (batch row, head, token)'s whole key range."""
    bs, seq, hq, d = q.shape
    hkv, max_depth = k_cache.shape[2], k_cache.shape[1]
    assert k_cache.stride() == v_cache.stride() and q.stride(-1) == 1 and k_cache.stride(-1) == 1
    out = torch.empty(bs, seq, hq, d, device=q.device, dtype=q.dtype)
    _cp_attn_kernel[(bs, hq, seq)](
        q, k_cache, v_cache, out, cache_pos, d ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        GROUP=hq // hkv, D=d, BLOCK_N=triton.next_power_of_2(max_depth), num_warps=4,
    )
    return out
