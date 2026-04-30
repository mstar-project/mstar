import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attn_nhd_kernel(
    Q_ptr,          # (B, H_q, D)        - query, q_len=1 squeezed out
    K_ptr,          # (B, S, H_kv, D)    - NHD layout
    V_ptr,          # (B, S, H_kv, D)    - NHD layout
    O_ptr,          # (B, H_q, D)        - output
    cache_len,      # int: actual filled length (attend to [0, cache_len))
    sm_scale,       # float: 1/sqrt(D) typically
    # Q strides
    sq_b, sq_h, sq_d,
    # K strides (NHD)
    sk_b, sk_n, sk_h, sk_d,
    # V strides (NHD)
    sv_b, sv_n, sv_h, sv_d,
    # O strides
    so_b, so_h, so_d,
    H_q: tl.constexpr,
    H_kv: tl.constexpr,
    GROUP: tl.constexpr,        # H_q // H_kv
    D: tl.constexpr,            # head dim, power of 2
    BLOCK_N: tl.constexpr,      # KV chunk size along sequence
):
    # One program per (batch, q_head).
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_hkv = pid_hq // GROUP

    offs_d = tl.arange(0, D)

    # Load query vector: (D,)
    q = tl.load(Q_ptr + pid_b * sq_b + pid_hq * sq_h + offs_d * sq_d).to(tl.float32)
    q = q * sm_scale

    # Online softmax accumulators
    m_i = -float("inf")                       # running max
    l_i = 0.0                                  # running sum of exp
    acc = tl.zeros((D,), dtype=tl.float32)     # running weighted sum of V

    # Base pointers for this batch + kv head
    K_base = K_ptr + pid_b * sk_b + pid_hkv * sk_h
    V_base = V_ptr + pid_b * sv_b + pid_hkv * sv_h

    # Iterate over the KV sequence in chunks of BLOCK_N
    for start_n in range(0, cache_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < cache_len

        # Load K block: (BLOCK_N, D)
        k_ptrs = K_base + offs_n[:, None] * sk_n + offs_d[None, :] * sk_d
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # qk: (BLOCK_N,)  =  k @ q
        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(mask_n, qk, -float("inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)                 # (BLOCK_N,)

        # Load V block: (BLOCK_N, D)
        v_ptrs = V_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Update accumulator
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # Finalize and store
    out = acc / l_i
    o_ptrs = O_ptr + pid_b * so_b + pid_hq * so_h + offs_d * so_d
    tl.store(o_ptrs, out)


@torch.compiler.disable
def decode_attn_nhd(q, k_cache, v_cache, cache_len, sm_scale=None):
    """
    Decode-only SDPA for NHD-format KV cache.

    q:        (B, 1, H_q, D)  or  (B, H_q, D)        - fp32
    k_cache:  (B, S_max, H_kv, D)                    - fp32
    v_cache:  (B, S_max, H_kv, D)                    - fp32
    cache_len: int, number of valid KV positions

    returns: (B, 1, H_q, D) matching q's input shape
    """
    squeeze_qlen = False
    if q.dim() == 4:
        assert q.shape[1] == 1, "decode kernel requires q_len == 1"
        q = q.squeeze(1)
        squeeze_qlen = True

    B, H_q, D = q.shape
    _, _, H_kv, _ = k_cache.shape
    assert H_q % H_kv == 0, "H_q must be divisible by H_kv"
    assert D in (16, 32, 64, 128, 256), f"head_dim {D} should be a power of 2"
    assert q.is_contiguous() or True  # strides are passed explicitly; contiguity not required

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(q)

    GROUP = H_q // H_kv
    # BLOCK_N tuning: 64 or 128 is usually a good default for decode.
    # TODO: do an autotune instead
    BLOCK_N = 64

    grid = (B, H_q)
    _decode_attn_nhd_kernel[grid](
        q, k_cache, v_cache, out,
        cache_len, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        H_q=H_q, H_kv=H_kv, GROUP=GROUP, D=D, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    return out.unsqueeze(1) if squeeze_qlen else out


@triton.jit
def _qk_norm_rope_kernel(
    X_ptr, W_norm_ptr, Pos_ptr,
    eps, rope_theta,
    sx_m, sx_h, sx_d,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = X_ptr + pid_m * sx_m + pid_h * sx_h

    offs_half = tl.arange(0, HALF_D)

    # Load both halves — the compiler will coalesce these into one transaction
    # since they're contiguous in the inner dim.
    x_first  = tl.load(base + offs_half * sx_d).to(tl.float32)
    x_second = tl.load(base + (offs_half + HALF_D) * sx_d).to(tl.float32)

    # RMS computed from both halves
    sumsq = tl.sum(x_first * x_first, axis=0) + tl.sum(x_second * x_second, axis=0)
    inv_rms = 1.0 / tl.sqrt(sumsq / D + eps)

    w_first  = tl.load(W_norm_ptr + offs_half).to(tl.float32)
    w_second = tl.load(W_norm_ptr + offs_half + HALF_D).to(tl.float32)

    x_first  = x_first  * inv_rms * w_first
    x_second = x_second * inv_rms * w_second

    pos = tl.load(Pos_ptr + pid_m).to(tl.float32)
    inv_freq = tl.exp(-tl.log(rope_theta) * (offs_half.to(tl.float32) * 2.0 / D))
    angle = pos * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    out_first  = x_first * cos - x_second * sin
    out_second = x_second * cos + x_first * sin

    tl.store(base + offs_half * sx_d,             out_first)
    tl.store(base + (offs_half + HALF_D) * sx_d,  out_second)


def fused_qk_norm_rope(x, w_norm, pos, eps, rope_theta):
    """
    x:      (M, H, D)  fp32 — modified in place
    w_norm: (D,)       fp32
    pos:    (M,)       int32 positions
    """
    M, H, D = x.shape
    assert D % 2 == 0
    assert x.is_contiguous() or x.stride(-1) == 1

    grid = (M, H)
    _qk_norm_rope_kernel[grid](
        x, w_norm, pos.to(torch.int32),
        eps, rope_theta,
        x.stride(0), x.stride(1), x.stride(2),
        D=D, HALF_D=D // 2,
        num_warps=4, num_stages=1,
    )
    return x