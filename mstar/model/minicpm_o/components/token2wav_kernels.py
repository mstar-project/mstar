"""Triton kernels for the token2wav DiT on the slot pool (``RingKV``).

The DiT runs 10 Euler steps x 16 blocks per window, so its per-block cost is dominated by
memory movement and small kernels rather than FLOPs. These kernels read the caches where
they live (the slot ring by slot index, the voice's tail, this window's frames) instead of
gathering and concatenating them, and fuse the elementwise work around the GEMMs. Inputs
are float32; dot products use TF32, as the rest of the serving process does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attend_segment(
    q, m_i, l_i, acc, base, stride_pos, d_off, start, count, sm_scale,
    BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    """Online-softmax accumulation over ``count`` frames at ``base + (start + j) * stride_pos``,
    keys in ``[0, D)`` and values in ``[D, 2D)`` of each frame (offset ``d_off``)."""
    offs_d = tl.arange(0, D)
    for j0 in range(0, count, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < count
        ptrs = base + (start + offs_n)[:, None] * stride_pos + offs_d[None, :]
        k = tl.load(ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(ptrs + d_off, mask=mask_n[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="tf32") * sm_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32")
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _ring_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    ring_ptr, slots_ptr, voice_ptr,
    T, H, ring_len, tail_start, tail_len, sm_scale,
    s_qn, s_qh, s_qt,
    s_rs, s_rc, s_rh, s_rp,
    s_vc, s_vh, s_vp,
    s_on, s_ot, s_oh,
    FRESH: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nh = tl.program_id(1)
    n = pid_nh // H
    h = pid_nh % H
    b = n // 2
    c = n % 2
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T
    q = tl.load(q_ptr + n * s_qn + h * s_qh + offs_m[:, None] * s_qt + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # this window's own frames (k and v are separate [N, H, T, D] tensors)
    for j0 in range(0, T, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T
        kp = k_ptr + n * s_qn + h * s_qh + offs_n[:, None] * s_qt + offs_d[None, :]
        k = tl.load(kp, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptr + n * s_qn + h * s_qh + offs_n[:, None] * s_qt + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="tf32") * sm_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32")
        m_i = m_new

    # the ring (the voice's head on a request's first window)
    if FRESH:
        ring_base = voice_ptr + c * s_vc + h * s_vh
        m_i, l_i, acc = _attend_segment(q, m_i, l_i, acc, ring_base, s_vp, D, 0, ring_len, sm_scale,
                                        BLOCK_N, D)
    else:
        slot = tl.load(slots_ptr + b)
        ring_base = ring_ptr + slot * s_rs + c * s_rc + h * s_rh
        m_i, l_i, acc = _attend_segment(q, m_i, l_i, acc, ring_base, s_rp, D, 0, ring_len, sm_scale,
                                        BLOCK_N, D)
    # the voice's tail
    voice_base = voice_ptr + c * s_vc + h * s_vh
    m_i, l_i, acc = _attend_segment(q, m_i, l_i, acc, voice_base, s_vp, D, tail_start, tail_len, sm_scale,
                                    BLOCK_N, D)

    out = acc / l_i[:, None]
    tl.store(o_ptr + n * s_on + offs_m[:, None] * s_ot + h * s_oh + offs_d[None, :], out, mask=mask_m[:, None])


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ring: torch.Tensor,
    slots: torch.Tensor,
    voice: torch.Tensor,
    ring_len: int,
    tail_len: int,
    fresh: bool,
) -> torch.Tensor:
    """``q, k, v [2B, H, T, D]`` (contiguous) attend over themselves, each row's ring
    (``ring [S, 2, H, >= ring_len, 2D]`` at ``slots [B]``; the voice's head when ``fresh``)
    and the voice's last ``tail_len`` frames before ``ring_len`` (``voice [2, H, >=
    ring_len, 2D]``). Returns ``[2B, T, H, D]``."""
    n, h, t, d = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert ring.stride(-1) == 1 and voice.stride(-1) == 1
    out = torch.empty(n, t, h, d, device=q.device, dtype=q.dtype)
    # small batches: more, smaller query tiles so the grid fills the GPU
    block_m = 16 if n * h < 128 else 64
    grid = (triton.cdiv(t, block_m), n * h)
    _ring_attention_kernel[grid](
        q, k, v, out, ring, slots, voice,
        t, h, ring_len, ring_len - tail_len, tail_len, d ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        ring.stride(0), ring.stride(1), ring.stride(2), ring.stride(3),
        voice.stride(0), voice.stride(1), voice.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        FRESH=fresh, BLOCK_M=block_m, BLOCK_N=64, D=d,
    )
    return out


@triton.jit
def _ring_store_kernel(
    k_ptr, v_ptr, ring_ptr, slots_ptr, heads_ptr,
    T, H, ring_len,
    s_kn, s_kh, s_kt,
    s_rs, s_rc, s_rh, s_rp,
    FRESH: tl.constexpr, BLOCK_T: tl.constexpr, D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_nh = tl.program_id(1)
    n = pid_nh // H
    h = pid_nh % H
    b = n // 2
    c = n % 2
    slot = tl.load(slots_ptr + b)
    if FRESH:
        head = 0
    else:
        head = tl.load(heads_ptr + slot).to(tl.int64)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    mask = offs_t < T
    pos = (head - T + offs_t + ring_len) % ring_len
    src = n * s_kn + h * s_kh + offs_t[:, None] * s_kt + offs_d[None, :]
    k = tl.load(k_ptr + src, mask=mask[:, None])
    v = tl.load(v_ptr + src, mask=mask[:, None])
    dst = ring_ptr + slot * s_rs + c * s_rc + h * s_rh + pos[:, None] * s_rp + offs_d[None, :]
    tl.store(dst, k, mask=mask[:, None])
    tl.store(dst + D, v, mask=mask[:, None])


def ring_store(
    k: torch.Tensor,
    v: torch.Tensor,
    ring: torch.Tensor,
    slots: torch.Tensor,
    heads: torch.Tensor,
    ring_len: int,
    fresh: bool,
) -> None:
    """This window's ``k, v [2B, H, T, D]`` into each row's ring at ``[head - T, head)`` mod
    ``ring_len`` (``head`` 0 when ``fresh``)."""
    n, h, t, d = k.shape
    assert k.is_contiguous() and v.is_contiguous() and ring.stride(-1) == 1
    block_t = 64
    grid = (triton.cdiv(t, block_t), n * h)
    _ring_store_kernel[grid](
        k, v, ring, slots, heads, t, h, ring_len,
        k.stride(0), k.stride(1), k.stride(2),
        ring.stride(0), ring.stride(1), ring.stride(2), ring.stride(3),
        FRESH=fresh, BLOCK_T=block_t, D=d,
    )


# ---------------------------------------------------------------------------
# Row kernels around the GEMMs
# ---------------------------------------------------------------------------


@triton.jit
def _res_adaln_kernel(
    x_ptr, y_ptr, gate_ptr, shift_ptr, scale_ptr, x_out_ptr, h_ptr,
    T, mod_stride_n, eps,
    HAS_RES: tl.constexpr, C: tl.constexpr,
):
    """One ``[C]`` row: ``x += gate * y`` (when ``HAS_RES``), then ``h = LN(x) * scale + shift``
    (LayerNorm without affine; ``scale`` is adaLN's ``1 + scale``)."""
    row = tl.program_id(0)
    n = row // T
    offs = tl.arange(0, C)
    x = tl.load(x_ptr + row * C + offs)
    mod = n * mod_stride_n + offs
    if HAS_RES:
        y = tl.load(y_ptr + row * C + offs)
        x = x + tl.load(gate_ptr + mod) * y
        tl.store(x_out_ptr + row * C + offs, x)
    mean = tl.sum(x, axis=0) / C
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / C
    h = xc * tl.rsqrt(var + eps) * tl.load(scale_ptr + mod) + tl.load(shift_ptr + mod)
    tl.store(h_ptr + row * C + offs, h)


def _mod_rows(mod: torch.Tensor, n: int) -> tuple[torch.Tensor, int]:
    """A modulation vector ``[r, 1, C]`` (r = 1, or one per row) and its per-row stride."""
    mod = mod.reshape(mod.shape[0], -1)
    assert mod.shape[0] in (1, n) and mod.stride(-1) == 1
    return mod, (0 if mod.shape[0] == 1 else mod.stride(0))


def res_adaln(
    x: torch.Tensor,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [N, T, C]`` (contiguous); with ``y`` returns ``(x + gate * y, adaLN(x + gate * y))``,
    else ``(x, adaLN(x))``. Modulations are ``[1 or N, 1, C]``."""
    n, t, c = x.shape
    assert x.is_contiguous()
    shift, stride = _mod_rows(shift, n)
    scale, _ = _mod_rows(scale, n)
    h = torch.empty_like(x)
    if y is None:
        _res_adaln_kernel[(n * t,)](x, x, x, shift, scale, x, h, t, stride, eps, HAS_RES=False, C=c)
        return x, h
    gate, _ = _mod_rows(gate, n)
    y = y.contiguous()
    x_out = torch.empty_like(x)
    _res_adaln_kernel[(n * t,)](x, y, gate, shift, scale, x_out, h, t, stride, eps, HAS_RES=True, C=c)
    return x_out, h


@triton.jit
def _qk_norm_split_kernel(
    qkv_ptr, q_ptr, k_ptr, v_ptr, qw_ptr, qb_ptr, kw_ptr, kb_ptr,
    T, H, eps, D: tl.constexpr,
):
    """One (row, head): q and k LayerNorm'd over the head (with affine), v copied, each to
    ``[N, H, T, D]``."""
    row = tl.program_id(0)
    h = tl.program_id(1)
    n = row // T
    t = row % T
    offs = tl.arange(0, D)
    width = 3 * H * D
    base = qkv_ptr + row * width + h * D + offs
    dst = ((n * H + h) * T + t) * D + offs
    for which in tl.static_range(2):
        val = tl.load(base + which * H * D)
        mean = tl.sum(val, axis=0) / D
        vc = val - mean
        var = tl.sum(vc * vc, axis=0) / D
        normed = vc * tl.rsqrt(var + eps)
        if which == 0:
            tl.store(q_ptr + dst, normed * tl.load(qw_ptr + offs) + tl.load(qb_ptr + offs))
        else:
            tl.store(k_ptr + dst, normed * tl.load(kw_ptr + offs) + tl.load(kb_ptr + offs))
    tl.store(v_ptr + dst, tl.load(base + 2 * H * D))


def qk_norm_split(
    qkv: torch.Tensor, heads: int, q_norm: torch.nn.LayerNorm, k_norm: torch.nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkv [N, T, 3 H D]`` -> normed ``q``, normed ``k`` and ``v``, each ``[N, H, T, D]``."""
    n, t, width = qkv.shape
    d = width // (3 * heads)
    assert qkv.is_contiguous() and q_norm.eps == k_norm.eps
    q = torch.empty(n, heads, t, d, device=qkv.device, dtype=qkv.dtype)
    k, v = torch.empty_like(q), torch.empty_like(q)
    _qk_norm_split_kernel[(n * t, heads)](
        qkv, q, k, v, q_norm.weight, q_norm.bias, k_norm.weight, k_norm.bias, t, heads, q_norm.eps, D=d,
    )
    return q, k, v


@triton.jit
def _ln_mish_kernel(x_ptr, w_ptr, b_ptr, out_ptr, eps, C: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, C)
    x = tl.load(x_ptr + row * C + offs)
    mean = tl.sum(x, axis=0) / C
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / C
    y = xc * tl.rsqrt(var + eps) * tl.load(w_ptr + offs) + tl.load(b_ptr + offs)
    softplus = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))
    tanh = 1.0 - 2.0 / (tl.exp(2.0 * softplus) + 1.0)
    tl.store(out_ptr + row * C + offs, y * tanh)


def ln_mish(x: torch.Tensor, norm: torch.nn.LayerNorm) -> torch.Tensor:
    """``Mish(LayerNorm(x))`` over the last dim of a contiguous ``[N, T, C]``."""
    n, t, c = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    _ln_mish_kernel[(n * t,)](x, norm.weight, norm.bias, out, norm.eps, C=c)
    return out


# ---------------------------------------------------------------------------
# The DiT forward on these kernels
# ---------------------------------------------------------------------------


def _fused_weights(module: torch.nn.Module, name: str, make):
    """Weights rearranged for the fused path, built on first use (after loading) and kept on
    the module. They are copies, so they are rebuilt if the module moves to another device."""
    cache = module.__dict__.setdefault("_fused_cache", {})
    hit = cache.get(name)
    if hit is None or hit[0].device != next(module.parameters()).device:
        hit = make()
        cache[name] = hit
    return hit


def _attention(attn, h: torch.Tensor, kv) -> torch.Tensor:
    w, b = _fused_weights(attn, "qkv", lambda: (
        torch.cat([attn.to_q.weight, attn.to_k.weight, attn.to_v.weight]).contiguous(),
        torch.cat([attn.to_q.bias, attn.to_k.bias, attn.to_v.bias]).contiguous(),
    ))
    n, t, _ = h.shape
    q, k, v = qk_norm_split(torch.nn.functional.linear(h, w, b), attn.num_heads, attn.q_norm, attn.k_norm)
    out = kv.attend(q, k, v)  # [N, T, H, D]
    return attn.proj(out.reshape(n, t, -1))


def _causal_conv(conv: torch.nn.Conv1d, x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Width-3 causal conv over time-major ``x [N, T, C]`` with ``cache [N, C, 2]`` (updated):
    a channels-last conv2d, so cuDNN reads the activations as they are."""
    (w,) = _fused_weights(conv, "conv2d", lambda: (
        conv.weight.unsqueeze(2).contiguous(memory_format=torch.channels_last),
    ))
    n, t, c = x.shape
    x = torch.cat([cache.transpose(1, 2), x], dim=1)  # [N, T + 2, C]
    cache.copy_(x[:, -cache.shape[-1]:].transpose(1, 2))
    out = torch.nn.functional.conv2d(x.unsqueeze(1).permute(0, 3, 1, 2), w, conv.bias)
    return out.permute(0, 2, 3, 1).reshape(n, t, c)


def _conv_block(block, x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    c = block.channels
    y = _causal_conv(block.conv1, x, cache[:, :c])
    return _causal_conv(block.conv2, ln_mish(y, block.norm), cache[:, c:])


def _mlp(mlp, h: torch.Tensor) -> torch.Tensor:
    n, t, c = h.shape
    a = torch._addmm_activation(mlp.fc1.bias, h.reshape(n * t, c), mlp.fc1.weight.t(), use_gelu=True)
    return mlp.fc2(a).reshape(n, t, -1)


def dit_forward(dit, x, mu, mods, final_mod, spks, cond, cnn_cache, kv) -> torch.Tensor:
    """``DiT.forward`` over ``RingKV`` caches on CUDA: the same blocks, with each residual
    add fused into the next adaLN (the MLP's into the next block's), the attention over the
    ring, channels-last causal convs, and bias + GELU in the fc1 GEMM's epilogue."""
    t = x.shape[-1]
    h = torch.cat([x, mu, spks.unsqueeze(-1).expand(-1, -1, t), cond], dim=1)
    x = dit.in_proj(h.transpose(1, 2)).contiguous()
    pending = (None, None)
    for i, block in enumerate(dit.blocks):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_conv, scale_conv, gate_conv) = mods[i]
        x, h = res_adaln(x, *pending, shift_msa, scale_msa, block.norm1.eps)
        x, h = res_adaln(x, _attention(block.attn, h, kv[i]), gate_msa, shift_conv, scale_conv, block.norm3.eps)
        x, h = res_adaln(x, _conv_block(block.conv, h, cnn_cache[i]), gate_conv, shift_mlp, scale_mlp,
                         block.norm2.eps)
        pending = (_mlp(block.mlp, h), gate_mlp)
    shift, scale = final_mod
    _, h = res_adaln(x, *pending, shift, scale, dit.final_layer.norm_final.eps)
    return dit.final_layer.linear(h).transpose(1, 2)
