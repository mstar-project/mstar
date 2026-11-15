"""Half-rotation RoPE and bidirectional attention over short, complete rows.

For stacks that attend within rows of a few tokens each and cache nothing
(VoxCPM2's local DiT and patch encoder: 11 and 5 tokens a row), one program
holds a row's head in registers: it rotates q and k in fp32, rounded to the
activation dtype as the eager path does, runs softmax attention and writes the
output in the layout the output projection reads. Eager, the same step is a
dozen elementwise kernels, the transposes around SDPA and the copy after it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_attention_reference(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rows: int, seq: int,
    num_heads: int, num_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    """The eager path: ``[rows * seq, (heads + 2 kv) * head_dim]`` packed
    q/k/v to the attention output ``[rows * seq, heads * head_dim]``."""
    q_dim, kv_dim = num_heads * head_dim, num_kv_heads * head_dim
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
    q = q.reshape(rows, seq, num_heads, head_dim)
    k = k.reshape(rows, seq, num_kv_heads, head_dim)
    v = v.reshape(rows, seq, num_kv_heads, head_dim)
    c, s = cos[:seq].unsqueeze(1), sin[:seq].unsqueeze(1)
    qf, kf = q.float(), k.float()
    q = (qf * c + _rotate_half(qf) * s).to(qkv.dtype)
    k = (kf * c + _rotate_half(kf) * s).to(qkv.dtype)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=num_heads != num_kv_heads,
    )
    return out.transpose(1, 2).reshape(rows * seq, q_dim)


if triton is not None:

    @triton.jit
    def _rope_attention_kernel(
        qkv_ptr, cos_ptr, sin_ptr, out_ptr, seq, qkv_stride, out_stride, rope_stride, sm_scale,
        NUM_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        kv_head = head // (NUM_HEADS // NUM_KV_HEADS)
        HALF: tl.constexpr = HEAD_DIM // 2
        s = tl.arange(0, BLOCK_S)
        d = tl.arange(0, HALF)
        valid = s < seq
        tokens = (row * seq + s)[:, None]
        in_row = qkv_ptr + tokens * qkv_stride
        lo, hi = d[None, :], HALF + d[None, :]
        mask = valid[:, None]

        c_lo = tl.load(cos_ptr + s[:, None] * rope_stride + lo, mask=mask, other=0.0)
        c_hi = tl.load(cos_ptr + s[:, None] * rope_stride + hi, mask=mask, other=0.0)
        s_lo = tl.load(sin_ptr + s[:, None] * rope_stride + lo, mask=mask, other=0.0)
        s_hi = tl.load(sin_ptr + s[:, None] * rope_stride + hi, mask=mask, other=0.0)

        q_base = in_row + head * HEAD_DIM
        q1 = tl.load(q_base + lo, mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_base + hi, mask=mask, other=0.0).to(tl.float32)
        q_lo = (q1 * c_lo - q2 * s_lo).to(qkv_ptr.dtype.element_ty)
        q_hi = (q2 * c_hi + q1 * s_hi).to(qkv_ptr.dtype.element_ty)

        k_base = in_row + (NUM_HEADS + kv_head) * HEAD_DIM
        k1 = tl.load(k_base + lo, mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(k_base + hi, mask=mask, other=0.0).to(tl.float32)
        k_lo = (k1 * c_lo - k2 * s_lo).to(qkv_ptr.dtype.element_ty)
        k_hi = (k2 * c_hi + k1 * s_hi).to(qkv_ptr.dtype.element_ty)

        # "ieee": float32 products for a float32 row (eager SDPA's, not TF32's);
        # half-precision rows ignore it, and every backend accepts it
        scores = tl.dot(q_lo, tl.trans(k_lo), input_precision="ieee") + tl.dot(
            q_hi, tl.trans(k_hi), input_precision="ieee"
        )
        scores = tl.where(valid[None, :], scores * sm_scale, float("-inf"))
        scores = scores - tl.max(scores, axis=1)[:, None]
        p = tl.exp(scores)
        p = p / tl.sum(p, axis=1)[:, None]

        full = tl.arange(0, HEAD_DIM)[None, :]
        v_base = in_row + (NUM_HEADS + NUM_KV_HEADS + kv_head) * HEAD_DIM
        v = tl.load(v_base + full, mask=mask, other=0.0)
        out = tl.dot(p.to(v.dtype), v, input_precision="ieee")
        tl.store(out_ptr + tokens * out_stride + head * HEAD_DIM + full, out.to(out_ptr.dtype.element_ty), mask=mask)


def rope_attention(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rows: int, seq: int,
    num_heads: int, num_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    """Half-rotation RoPE on q/k, then bidirectional attention within each row
    of ``seq`` tokens: ``[rows * seq, (heads + 2 kv) * head_dim]`` packed
    q/k/v to ``[rows * seq, heads * head_dim]``. ``cos``/``sin`` are fp32
    ``[>= seq, head_dim]``."""
    # tl.dot needs each half of a head to span 16 or more, and a power of two;
    # the kernel reads each row's q/k/v as contiguous columns
    if (
        triton is None or not qkv.is_cuda or seq > 16 or head_dim < 32 or head_dim & (head_dim - 1)
        or qkv.stride(-1) != 1
        # one row stride read for both tables, and whole groups of query heads per KV head
        or sin.stride() != cos.stride() or num_heads % num_kv_heads
    ):
        return rope_attention_reference(qkv, cos, sin, rows, seq, num_heads, num_kv_heads, head_dim)
    out = qkv.new_empty(rows * seq, num_heads * head_dim)
    _rope_attention_kernel[(rows, num_heads)](
        qkv, cos, sin, out, seq, qkv.stride(0), out.stride(0), cos.stride(0), head_dim ** -0.5,
        NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim, BLOCK_S=16, num_warps=4,
    )
    return out
