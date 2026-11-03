"""``mstar::qk_norm_rope``: per-head RMSNorm of q and k and rotate-half RoPE, in one kernel.

Takes a packed ``[tokens, q | k | v]`` projection row, RMS-normalises each q
and k head with its weight (fp32 math, rounded to the input dtype as the norm
kernel stores it), rotates the halves of each head with the step's per-token
``cos`` / ``sin`` tables (``[tokens, head_dim]``, NeoX layout, already
interleaved for multimodal RoPE) in fp32, and writes q and k contiguous,
rounded once. Unfused, a decode step runs a norm kernel and a rope kernel for
each of q and k per attention layer, plus a copy of each to make the strided
q and k slices contiguous for the norm once there is more than one token.

AMD GPUs run a Triton kernel; other platforms have no implementation and keep
their unfused path (``qk_norm_rope_kernel`` returns None there).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.kernels import KernelOp, ShapePolicy, Tolerance
from mstar.platforms.dispatch import PlatformDispatch


def _reference(qkv, q_weight, k_weight, cos, sin, heads, kv_heads, head_dim, eps):
    tokens = qkv.shape[0]
    q, k, _ = qkv.split([heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1)
    c = cos.float()[:, None, :]
    s = sin.float()[:, None, :]

    def norm_rope(x, n_heads, weight):
        x = x.reshape(tokens, n_heads, head_dim).float()
        normed = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()).to(qkv.dtype).float()
        x1, x2 = normed.chunk(2, dim=-1)
        return (normed * c + torch.cat((-x2, x1), dim=-1) * s).to(qkv.dtype)

    return norm_rope(q, heads, q_weight), norm_rope(k, kv_heads, k_weight)


def _fake(qkv, q_weight, k_weight, cos, sin, heads, kv_heads, head_dim, eps):
    tokens = qkv.shape[0]
    return qkv.new_empty((tokens, heads, head_dim)), qkv.new_empty((tokens, kv_heads, head_dim))


@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr, q_weight_ptr, k_weight_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr,
    stride_qkv, stride_cos, stride_sin, eps,
    HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HALF: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, HALF)
    src = qkv_ptr + token * stride_qkv + head * 2 * HALF
    x1 = tl.load(src + offs).to(tl.float32)
    x2 = tl.load(src + HALF + offs).to(tl.float32)
    if head < HEADS:
        w_ptr = q_weight_ptr
        dst = q_ptr + (token * HEADS + head) * 2 * HALF
    else:
        w_ptr = k_weight_ptr
        dst = k_ptr + (token * KV_HEADS + head - HEADS) * 2 * HALF
    w1 = tl.load(w_ptr + offs).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + offs).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / (2 * HALF) + eps)
    # the norm's output is rounded as the separate norm kernel stores it
    n1 = (x1 * rstd * w1).to(q_ptr.dtype.element_ty).to(tl.float32)
    n2 = (x2 * rstd * w2).to(q_ptr.dtype.element_ty).to(tl.float32)
    c1 = tl.load(cos_ptr + token * stride_cos + offs).to(tl.float32)
    c2 = tl.load(cos_ptr + token * stride_cos + HALF + offs).to(tl.float32)
    s1 = tl.load(sin_ptr + token * stride_sin + offs).to(tl.float32)
    s2 = tl.load(sin_ptr + token * stride_sin + HALF + offs).to(tl.float32)
    tl.store(dst + offs, (n1 * c1 - n2 * s1).to(q_ptr.dtype.element_ty))
    tl.store(dst + HALF + offs, (n2 * c2 + n1 * s2).to(q_ptr.dtype.element_ty))


def _triton():
    def run(qkv, q_weight, k_weight, cos, sin, heads, kv_heads, head_dim, eps):
        tokens = qkv.shape[0]
        q = qkv.new_empty((tokens, heads, head_dim))
        k = qkv.new_empty((tokens, kv_heads, head_dim))
        if tokens:
            _qk_norm_rope_kernel[(tokens, heads + kv_heads)](
                qkv, q_weight, k_weight, cos, sin, q, k,
                qkv.stride(0), cos.stride(0), sin.stride(0), eps,
                HEADS=heads, KV_HEADS=kv_heads, HALF=head_dim // 2, num_warps=1,
            )
        return q, k

    return run


def _key(qkv, q_weight, k_weight, cos, sin, heads, kv_heads, head_dim, eps) -> dict[str, int]:
    return {"tokens": qkv.shape[0], "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim}


def _inputs(shape, dtype, device, generator=None):
    tokens, heads, kv_heads, head_dim = shape["tokens"], shape["heads"], shape["kv_heads"], shape["head_dim"]

    def randn(*size):
        return torch.randn(*size, dtype=torch.float32, device=device, generator=generator)

    qkv = (randn(tokens, (heads + 2 * kv_heads) * head_dim) * 3).to(dtype)
    angles = randn(tokens, head_dim // 2) * 10
    cos = torch.cat([angles.cos()] * 2, dim=-1).to(dtype)
    sin = torch.cat([angles.sin()] * 2, dim=-1).to(dtype)
    return qkv, (randn(head_dim) * 0.3 + 1).to(dtype), (randn(head_dim) * 0.3 + 1).to(dtype), cos, sin, \
        heads, kv_heads, head_dim, 1e-6


QK_NORM_ROPE = KernelOp(
    "qk_norm_rope",
    "(Tensor qkv, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, int heads, int kv_heads, "
    "int head_dim, float eps) -> (Tensor, Tensor)",
    fake=_fake,
    reference=_reference,
    impls={"rocm": _triton},
    shapes=ShapePolicy(
        dims=("tokens", "heads", "kv_heads", "head_dim"),
        key=_key,
        inputs=_inputs,
        warm=({"tokens": 1, "heads": 32, "kv_heads": 4, "head_dim": 128},
              {"tokens": 32, "heads": 32, "kv_heads": 4, "head_dim": 128}),
    ),
    # the norm is rounded to bf16 before the rotation, as the unfused pair does,
    # which the fp64 reference does not; a bf16 ulp of slack covers it
    tolerance=Tolerance.accumulated(rtol=1.6e-2, atol_scale=1e-2),
)


def _fused():
    return torch.ops.mstar.qk_norm_rope


# The op on AMD GPUs, None elsewhere (the caller keeps its unfused path).
qk_norm_rope_kernel = PlatformDispatch("mstar::qk_norm_rope", rocm=_fused, default=lambda: None)
