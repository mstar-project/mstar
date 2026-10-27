"""Triton kernels for the token2wav DiT on the batched path.

The DiT runs 10 Euler steps x 16 blocks per window, so its per-block cost is dominated by
memory movement and small kernels rather than FLOPs. These fuse the elementwise work around
the GEMMs; attention runs on the bounded KV resource. Inputs are float32; dot products use
TF32, as the rest of the serving process does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


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
