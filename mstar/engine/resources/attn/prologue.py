"""The attention prologue of a decode step as one kernel.

A packed ``[tokens, q(+gate) | k | v]`` row is split, q and k are RMS-normalised
per head (gemma style: scaled by ``1 + weight``), the leading rotary dims of
each are rotated (NeoX halves, read from cos/sin tables at the step's
positions), and k and v are written into the paged cache at one flat slot per
token. A decode step otherwise runs a norm-and-rope kernel per tensor and a
cache scatter for each attention layer.

``mstar::qkv_norm_rope_cache`` runs AITER's ``fused_qkv_split_qk_norm_rope_cache``
on AMD GPUs. Elsewhere it has no implementation, and layers keep their own path
(``available()``).
"""

from __future__ import annotations

import functools

import torch

from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import KernelOp, ShapePolicy


def _split(qkv, heads, kv_heads, head_dim, gated):
    tokens = qkv.shape[0]
    q_size = heads * head_dim * (2 if gated else 1)
    kv_size = kv_heads * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    if gated:
        # the checkpoint interleaves query and gate per head
        q, gate = q.reshape(tokens, heads, 2 * head_dim).chunk(2, dim=-1)
        gate = gate.contiguous()
    else:
        q, gate = q.reshape(tokens, heads, head_dim), qkv.new_empty(0)
    return q, gate, k.reshape(tokens, kv_heads, head_dim), v.reshape(tokens, kv_heads, head_dim)


def _reference(qkv, q_weight, k_weight, cos, sin, positions, key_cache, value_cache, slots,
               heads, kv_heads, head_dim, eps, gated):
    q, gate, k, v = _split(qkv, heads, kv_heads, head_dim, gated)
    half = cos.shape[-1]
    c, s = cos[positions].float()[:, None, :], sin[positions].float()[:, None, :]

    def norm_rope(x, weight):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())
        x1, x2 = xf[..., :half], xf[..., half:2 * half]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, xf[..., 2 * half:]), dim=-1).to(x.dtype)

    q, k = norm_rope(q, q_weight), norm_rope(k, k_weight)
    block = key_cache.shape[1]
    pages, offsets = slots // block, slots % block
    key_cache[pages, offsets] = k.to(key_cache.dtype)
    value_cache[pages, offsets] = v.to(value_cache.dtype)
    # fresh outputs: a one-token row's v slice is contiguous, and would alias qkv
    return q, gate, k, v.clone(memory_format=torch.contiguous_format)


def _fake(qkv, q_weight, k_weight, cos, sin, positions, key_cache, value_cache, slots,
          heads, kv_heads, head_dim, eps, gated):
    tokens = qkv.shape[0]
    gate = qkv.new_empty((tokens, heads, head_dim)) if gated else qkv.new_empty(0)
    return (qkv.new_empty((tokens, heads, head_dim)), gate,
            qkv.new_empty((tokens, kv_heads, head_dim)), qkv.new_empty((tokens, kv_heads, head_dim)))


def _aiter():
    from aiter.ops.triton.rope.fused_qkv_split_qk_norm_rope_cache import fused_qkv_split_qk_norm_rope_cache

    def qkv_norm_rope_cache(qkv, q_weight, k_weight, cos, sin, positions, key_cache, value_cache, slots,
                            heads, kv_heads, head_dim, eps, gated):
        out = fused_qkv_split_qk_norm_rope_cache(
            qkv, q_weight, k_weight, cos, sin, positions, key_cache, value_cache, slots,
            heads, kv_heads, head_dim, is_neox=True, reuse_freqs_front_part=True,
            attn_output_gate=gated, eps=eps, gated_qkv_layout="interleaved", kv_cache_layout="NHD",
        )
        if gated:
            return out
        q, k, v = out
        return q, qkv.new_empty(0), k, v

    return qkv_norm_rope_cache


def _inputs(shape, dtype, device, generator=None):
    tokens, heads, kv_heads, head_dim, half = (shape[d] for d in ("tokens", "heads", "kv_heads", "head_dim", "half"))
    page_size, pages = 16, 8

    def rand(*size):
        return torch.randn(*size, dtype=dtype, device=device, generator=generator)

    qkv = rand(tokens, (2 * heads + 2 * kv_heads) * head_dim)
    steps = torch.arange(1, half + 1, device=device)
    angle = torch.arange(64, device=device, dtype=torch.float32)[:, None] * 0.01 * steps
    cache = torch.zeros(pages, 2, page_size, kv_heads, head_dim, dtype=dtype, device=device)
    slots = torch.randperm(pages * page_size, device=device, generator=generator)[:tokens]
    positions = torch.randint(0, 64, (tokens,), device=device, generator=generator)
    return (qkv, rand(head_dim) * 0.1, rand(head_dim) * 0.1, angle.cos().to(dtype), angle.sin().to(dtype),
            positions, cache[:, 0], cache[:, 1], slots, heads, kv_heads, head_dim, 1e-6, True)


QKV_NORM_ROPE_CACHE = KernelOp(
    "qkv_norm_rope_cache",
    "(Tensor qkv, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, Tensor positions, "
    "Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor slots, int heads, int kv_heads, int head_dim, "
    "float eps, bool gated) -> (Tensor, Tensor, Tensor, Tensor)",
    fake=_fake,
    reference=_reference,
    impls={"rocm": _aiter},
    shapes=ShapePolicy(
        dims=("tokens", "heads", "kv_heads", "head_dim", "half"),
        key=lambda qkv, q_weight, k_weight, cos, *rest: {
            "tokens": qkv.shape[0], "heads": rest[-5], "kv_heads": rest[-4], "head_dim": rest[-3],
            "half": cos.shape[-1]},
        inputs=_inputs,
        warm=({"tokens": 1, "heads": 16, "kv_heads": 4, "head_dim": 256, "half": 32},
              {"tokens": 64, "heads": 16, "kv_heads": 4, "head_dim": 256, "half": 32}),
        dtypes=(torch.bfloat16,),
    ),
    # fp32 norm and rotation, one rounding (or two: the kernel rounds the normalised q before rotating)
    tolerance=Tolerance.accumulated(rtol=1.6e-2, atol_scale=1e-2),
)


@functools.cache
def available() -> bool:
    """Whether this platform runs the fused prologue (its implementation loads)."""
    return QKV_NORM_ROPE_CACHE.unavailable_reason() is None
