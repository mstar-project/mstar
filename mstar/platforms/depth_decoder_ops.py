"""Fused kernels for the depth loop of residual-codebook speech decoders.

Qwen3-Omni's and Qwen3-TTS's code predictors walk 16 codec groups per audio frame
through a 5-layer transformer with a dense, frame-local KV cache. At decode batch
sizes every kernel in that loop costs about the same fixed launch time on a captured
graph whatever it computes, so the loop is bound by its kernel count. These ops cut
it without changing the math:

- ``mstar::depth_attention``: q/k RMSNorm, rotary embedding at the frame-local
  position, the K/V append and attention over the positions so far, in one kernel
  per layer.
- ``mstar::codec_embed_accumulate``: one group's embedding lookup, the running sum
  of the frame's embeddings and the code write, in one kernel.
- ``mstar::silu_and_mul``: SwiGLU's activation in fp32 with one rounding, as
  Inductor computes it, for loops that run outside ``torch.compile``.

Each op has a portable torch reference (the CPU kernel and the fallback) and a
Triton implementation that ``PlatformDispatch`` picks on AMD GPUs. They move to the
kernel registry once it exists; until then they follow ``mstar::rms_norm``.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from mstar.platforms.dispatch import PlatformDispatch

# ---------------------------------------------------------------------------
# depth_attention
# ---------------------------------------------------------------------------


def _norm_rope(x: torch.Tensor, weight: torch.Tensor, eps: float, position: int, rope_theta: float) -> torch.Tensor:
    """RMSNorm over the last dim, then split-half rope at one position, in fp32."""
    d = x.shape[-1]
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    inv_freq = torch.exp(
        -math.log(rope_theta) * (torch.arange(d // 2, device=x.device, dtype=torch.float32) * 2.0 / d)
    )
    angle = float(position) * inv_freq
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def depth_attention_reference(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_pos: int,
    num_heads: int,
    num_kv_heads: int,
    eps: float,
    rope_theta: float,
) -> torch.Tensor:
    """Attention of one new position per row over a dense per-row KV cache.

    ``qkv`` is ``[rows, (num_heads + 2 * num_kv_heads) * D]``, a QKV projection's
    output; ``k_cache`` and ``v_cache`` are ``[rows, positions, num_kv_heads, D]``.
    q and k are RMS-normalized per head and rotated (split-half rope) at position
    ``cache_pos`` in fp32, then rounded once; k and v are written at ``cache_pos``
    and the new query attends to positions ``0..cache_pos``. Returns
    ``[rows, num_heads * D]`` in ``qkv``'s dtype.
    """
    rows = qkv.shape[0]
    d = k_cache.shape[-1]
    q, k, v = qkv.split((num_heads * d, num_kv_heads * d, num_kv_heads * d), dim=-1)
    q = _norm_rope(q.reshape(rows, num_heads, d), q_norm_weight, eps, cache_pos, rope_theta).to(qkv.dtype)
    k = _norm_rope(k.reshape(rows, num_kv_heads, d), k_norm_weight, eps, cache_pos, rope_theta)
    k_cache[:, cache_pos] = k.to(k_cache.dtype)
    v_cache[:, cache_pos] = v.reshape(rows, num_kv_heads, d).to(v_cache.dtype)
    keys = k_cache[:, : cache_pos + 1].float().transpose(1, 2)
    values = v_cache[:, : cache_pos + 1].float().transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q.float().unsqueeze(2), keys, values, scale=1.0 / math.sqrt(d), enable_gqa=True,
    )
    return out.reshape(rows, num_heads * d).to(qkv.dtype)


depth_attention = torch.library.custom_op(
    "mstar::depth_attention", mutates_args=("k_cache", "v_cache"),
)(depth_attention_reference)


@depth_attention.register_fake
def _depth_attention_fake(qkv, q_norm_weight, k_norm_weight, k_cache, v_cache, cache_pos, num_heads,
                          num_kv_heads, eps, rope_theta):
    return qkv.new_empty((qkv.shape[0], num_heads * k_cache.shape[-1]))


@triton.jit(do_not_specialize=["cache_pos"])
def _depth_attention_kernel(
    QKV, Out, KCache, VCache, QNormW, KNormW,
    cache_pos, eps, rope_theta, sm_scale,
    s_qkv, s_out, s_cb, s_cn, s_ch,
    NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr, GROUP_P2: tl.constexpr,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per (row, kv head): it normalizes and rotates the head's new k once,
    # appends k and v, and serves every query head of its group. The expressions
    # follow mstar.utils.attention's qk-norm/rope and decode-attention kernels.
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_d = tl.arange(0, HALF)
    offs_g = tl.arange(0, GROUP_P2)
    head_ok = offs_g < GROUP
    base = QKV + row * s_qkv

    inv_freq = tl.exp(-tl.log(rope_theta) * (offs_d.to(tl.float32) * 2.0 / D))
    angle = cache_pos * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    k_ptr = base + NUM_HEADS * D + kv_head * D + offs_d
    k1 = tl.load(k_ptr).to(tl.float32)
    k2 = tl.load(k_ptr + HALF).to(tl.float32)
    k_inv = 1.0 / tl.sqrt((tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)) / D + eps)
    k1 = k1 * k_inv * tl.load(KNormW + offs_d).to(tl.float32)
    k2 = k2 * k_inv * tl.load(KNormW + HALF + offs_d).to(tl.float32)
    new_k1 = (k1 * cos - k2 * sin).to(KCache.dtype.element_ty)
    new_k2 = (k2 * cos + k1 * sin).to(KCache.dtype.element_ty)
    v_ptr = base + (NUM_HEADS + NUM_KV_HEADS) * D + kv_head * D + offs_d
    new_v1 = tl.load(v_ptr).to(VCache.dtype.element_ty)
    new_v2 = tl.load(v_ptr + HALF).to(VCache.dtype.element_ty)
    slot = row * s_cb + cache_pos * s_cn + kv_head * s_ch + offs_d
    tl.store(KCache + slot, new_k1)
    tl.store(KCache + slot + HALF, new_k2)
    tl.store(VCache + slot, new_v1)
    tl.store(VCache + slot + HALF, new_v2)

    # the group's query heads, rounded to the activation dtype as a stored q would be
    q_heads = kv_head * GROUP + offs_g
    q_ptr = base + q_heads[:, None] * D + offs_d[None, :]
    q1 = tl.load(q_ptr, mask=head_ok[:, None], other=0.0).to(tl.float32)
    q2 = tl.load(q_ptr + HALF, mask=head_ok[:, None], other=0.0).to(tl.float32)
    q_inv = 1.0 / tl.sqrt((tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)) / D + eps)
    q1 = q1 * q_inv[:, None] * tl.load(QNormW + offs_d).to(tl.float32)[None, :]
    q2 = q2 * q_inv[:, None] * tl.load(QNormW + HALF + offs_d).to(tl.float32)[None, :]
    rq1 = (q1 * cos[None, :] - q2 * sin[None, :]).to(QKV.dtype.element_ty).to(tl.float32) * sm_scale
    rq2 = (q2 * cos[None, :] + q1 * sin[None, :]).to(QKV.dtype.element_ty).to(tl.float32) * sm_scale

    # positions before cache_pos come from the cache, cache_pos itself from registers
    offs_n = tl.arange(0, BLOCK_N)
    past = (offs_n < cache_pos)[:, None]
    now = (offs_n == cache_pos)[:, None]
    kv_ptr = row * s_cb + offs_n[:, None] * s_cn + kv_head * s_ch + offs_d[None, :]
    key1 = tl.where(now, new_k1.to(tl.float32)[None, :], tl.load(KCache + kv_ptr, mask=past, other=0.0).to(tl.float32))
    key2 = tl.where(now, new_k2.to(tl.float32)[None, :],
                    tl.load(KCache + kv_ptr + HALF, mask=past, other=0.0).to(tl.float32))
    val1 = tl.where(now, new_v1.to(tl.float32)[None, :], tl.load(VCache + kv_ptr, mask=past, other=0.0).to(tl.float32))
    val2 = tl.where(now, new_v2.to(tl.float32)[None, :],
                    tl.load(VCache + kv_ptr + HALF, mask=past, other=0.0).to(tl.float32))

    scores = tl.sum(rq1[:, None, :] * key1[None, :, :], axis=2) + tl.sum(rq2[:, None, :] * key2[None, :, :], axis=2)
    scores = tl.where((offs_n <= cache_pos)[None, :], scores, -float("inf"))
    p = tl.exp(scores - tl.max(scores, axis=1)[:, None])
    denom = tl.sum(p, axis=1)
    out1 = tl.sum(p[:, :, None] * val1[None, :, :], axis=1) / denom[:, None]
    out2 = tl.sum(p[:, :, None] * val2[None, :, :], axis=1) / denom[:, None]
    out_ptr = Out + row * s_out + q_heads[:, None] * D + offs_d[None, :]
    tl.store(out_ptr, out1.to(Out.dtype.element_ty), mask=head_ok[:, None])
    tl.store(out_ptr + HALF, out2.to(Out.dtype.element_ty), mask=head_ok[:, None])


def _depth_attention_triton(
    qkv: torch.Tensor, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor, cache_pos: int, num_heads: int, num_kv_heads: int,
    eps: float, rope_theta: float,
) -> torch.Tensor:
    d = k_cache.shape[-1]
    if (d & (d - 1) or d < 32 or qkv.dim() != 2 or qkv.stride(-1) != 1 or k_cache.stride(-1) != 1
            or k_cache.stride() != v_cache.stride() or not q_norm_weight.is_contiguous()
            or not k_norm_weight.is_contiguous()):
        return depth_attention_reference(qkv, q_norm_weight, k_norm_weight, k_cache, v_cache, cache_pos,
                                         num_heads, num_kv_heads, eps, rope_theta)
    rows = qkv.shape[0]
    group = num_heads // num_kv_heads
    out = qkv.new_empty((rows, num_heads * d))
    _depth_attention_kernel[(rows, num_kv_heads)](
        qkv, out, k_cache, v_cache, q_norm_weight, k_norm_weight,
        cache_pos, eps, float(rope_theta), 1.0 / math.sqrt(d),
        qkv.stride(0), out.stride(0), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, GROUP=group, GROUP_P2=triton.next_power_of_2(group),
        D=d, HALF=d // 2, BLOCK_N=triton.next_power_of_2(k_cache.shape[1]),
        num_warps=4,
    )
    return out


_depth_attention_impl = PlatformDispatch(
    "mstar::depth_attention", rocm=lambda: _depth_attention_triton, default=lambda: depth_attention_reference,
)


# torch calls AMD GPUs "cuda" devices too, so the kernel picks by platform
@depth_attention.register_kernel("cuda")
def _depth_attention_cuda(qkv, q_norm_weight, k_norm_weight, k_cache, v_cache, cache_pos, num_heads,
                          num_kv_heads, eps, rope_theta):
    return _depth_attention_impl(qkv, q_norm_weight, k_norm_weight, k_cache, v_cache, cache_pos, num_heads,
                                 num_kv_heads, eps, rope_theta)


# ---------------------------------------------------------------------------
# codec_embed_accumulate
# ---------------------------------------------------------------------------


def codec_embed_accumulate_reference(
    codes: torch.Tensor,
    table: torch.Tensor,
    embed: torch.Tensor,
    embed_sum: torch.Tensor,
    all_codes: torch.Tensor,
    group: int,
    accumulate: bool,
) -> None:
    """One codec group of a frame: ``all_codes[:, group] = codes``, ``embed =
    table[codes]`` and ``embed_sum += embed`` (``= embed`` without ``accumulate``),
    added in fp32 and stored in ``embed_sum``'s dtype."""
    all_codes[:, group] = codes
    embed.copy_(F.embedding(codes, table))
    if accumulate:
        embed_sum.copy_(embed_sum.float() + embed.float())
    else:
        embed_sum.copy_(embed)


codec_embed_accumulate = torch.library.custom_op(
    "mstar::codec_embed_accumulate", mutates_args=("embed", "embed_sum", "all_codes"),
)(codec_embed_accumulate_reference)


@codec_embed_accumulate.register_fake
def _codec_embed_accumulate_fake(codes, table, embed, embed_sum, all_codes, group, accumulate):
    return None


@triton.jit(do_not_specialize=["group"])
def _codec_embed_accumulate_kernel(
    Codes, Table, Embed, Sum, AllCodes,
    group, width, s_table, s_embed, s_sum, s_codes,
    BLOCK: tl.constexpr, ACCUMULATE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    code = tl.load(Codes + row)
    if block == 0:
        tl.store(AllCodes + row * s_codes + group, code.to(AllCodes.dtype.element_ty))
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < width
    vec = tl.load(Table + code.to(tl.int64) * s_table + offs, mask=mask).to(Embed.dtype.element_ty)
    tl.store(Embed + row * s_embed + offs, vec, mask=mask)
    total = vec.to(tl.float32)
    if ACCUMULATE:
        total += tl.load(Sum + row * s_sum + offs, mask=mask).to(tl.float32)
    tl.store(Sum + row * s_sum + offs, total.to(Sum.dtype.element_ty), mask=mask)


_EMBED_BLOCK = 1024


def _codec_embed_accumulate_triton(
    codes: torch.Tensor, table: torch.Tensor, embed: torch.Tensor, embed_sum: torch.Tensor,
    all_codes: torch.Tensor, group: int, accumulate: bool,
) -> None:
    if (codes.dim() != 1 or table.stride(1) != 1 or embed.stride(1) != 1 or embed_sum.stride(1) != 1
            or not codes.is_contiguous()):
        codec_embed_accumulate_reference(codes, table, embed, embed_sum, all_codes, group, accumulate)
        return
    width = table.shape[1]
    _codec_embed_accumulate_kernel[(codes.shape[0], triton.cdiv(width, _EMBED_BLOCK))](
        codes, table, embed, embed_sum, all_codes,
        group, width, table.stride(0), embed.stride(0), embed_sum.stride(0), all_codes.stride(0),
        BLOCK=_EMBED_BLOCK, ACCUMULATE=accumulate, num_warps=4,
    )


_codec_embed_accumulate_impl = PlatformDispatch(
    "mstar::codec_embed_accumulate", rocm=lambda: _codec_embed_accumulate_triton,
    default=lambda: codec_embed_accumulate_reference,
)


@codec_embed_accumulate.register_kernel("cuda")
def _codec_embed_accumulate_cuda(codes, table, embed, embed_sum, all_codes, group, accumulate):
    _codec_embed_accumulate_impl(codes, table, embed, embed_sum, all_codes, group, accumulate)


# ---------------------------------------------------------------------------
# silu_and_mul
# ---------------------------------------------------------------------------


def silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` for ``x = [gate | up]`` on the last dim, computed in
    fp32 and rounded once."""
    gate, up = x.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


silu_and_mul = torch.library.custom_op("mstar::silu_and_mul", mutates_args=())(silu_and_mul_reference)


@silu_and_mul.register_fake
def _silu_and_mul_fake(x):
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


@triton.jit
def _silu_and_mul_kernel(X, Out, width, s_x, s_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < width
    gate = tl.load(X + row * s_x + offs, mask=mask).to(tl.float32)
    up = tl.load(X + row * s_x + width + offs, mask=mask).to(tl.float32)
    tl.store(Out + row * s_out + offs, (gate * tl.sigmoid(gate) * up).to(Out.dtype.element_ty), mask=mask)


_SILU_BLOCK = 1024


def _silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2 or x.stride(-1) != 1:
        return silu_and_mul_reference(x)
    rows, width = x.shape[0], x.shape[1] // 2
    out = x.new_empty((rows, width))
    _silu_and_mul_kernel[(rows, triton.cdiv(width, _SILU_BLOCK))](
        x, out, width, x.stride(0), out.stride(0), BLOCK=_SILU_BLOCK, num_warps=4,
    )
    return out


_silu_and_mul_impl = PlatformDispatch(
    "mstar::silu_and_mul", rocm=lambda: _silu_and_mul_triton, default=lambda: silu_and_mul_reference,
)


@silu_and_mul.register_kernel("cuda")
def _silu_and_mul_cuda(x):
    return _silu_and_mul_impl(x)


# ---------------------------------------------------------------------------
# decode_linear
# ---------------------------------------------------------------------------


def _skinny_linear():
    from mstar.platforms.rocm_gemm import SKINNY_MAX_ROWS, skinny_gemm_passes

    if skinny_gemm_passes() is None:
        return F.linear

    def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        rows, k = x.shape
        if (bias is None and rows <= SKINNY_MAX_ROWS and x.dtype == weight.dtype
                and x.dtype in (torch.bfloat16, torch.float16) and k % 8 == 0 and weight.shape[0] > 8
                and weight.is_contiguous()):
            return torch.ops.mstar.skinny_gemm(x, weight, None)
        return F.linear(x, weight, bias)

    return linear


# ``linear(x, weight, bias)`` for 2-D ``x`` outside torch.compile: AITER's skinny GEMM
# for one or two rows without a bias on AMD GPUs, as the captured decode buckets get
# from ``RocmPlatform.inductor_overrides``, and the platform's BLAS otherwise.
decode_linear = PlatformDispatch("decode_linear", rocm=_skinny_linear, default=lambda: F.linear)
