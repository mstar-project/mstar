"""Attention over a bounded KV step's ranges, and its write-back (Triton).

Keys and values share a ``2 * head_dim`` last axis in the slot and the source:
keys in ``[0, D)``, values in ``[D, 2D)``. Every row attends to its retained
ranges and to this step's own tokens, unmasked; dot products use TF32 unless
``ieee``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.engine.resources.kv.bounded.layout import NUM_READS, NUM_WRITES, ROW_INTS


@triton.jit
def _bounded_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, cache_ptr, source_ptr, table_ptr,
    T, H, sm_scale,
    s_qn, s_qh, s_qt,
    s_cs, s_cr, s_ch, s_cp,
    s_sr, s_sh, s_sp,
    s_on, s_ot, s_oh,
    ROWS: tl.constexpr, HAS_SOURCE: tl.constexpr, ROW_INTS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_nh = tl.program_id(1)
    n = pid_nh // H
    h = pid_nh % H
    b = n // ROWS
    c = n % ROWS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < T
    q = tl.load(q_ptr + n * s_qn + h * s_qh + offs_m[:, None] * s_qt + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # this step's own tokens (separate k and v tensors, laid out like q)
    for j0 in range(0, T, BLOCK_N):
        offs_n = j0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T
        kv_off = n * s_qn + h * s_qh + offs_n[:, None] * s_qt + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptr + kv_off, mask=mask_n[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * sm_scale
        s = tl.where(mask_n[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=PRECISION)
        m_i = m_new

    row = table_ptr + b * ROW_INTS
    slot = tl.load(row).to(tl.int64)
    slot_base = cache_ptr + slot * s_cs + c * s_cr + h * s_ch
    source_base = source_ptr + c * s_sr + h * s_sh
    # The five ranges (source, slot, source, slot, slot: layout.READ_FROM_SOURCE)
    # tiled as one sequence, so short ranges do not each pay a masked tile.
    s0 = tl.load(row + 1)
    e1 = tl.load(row + 2)
    s1 = tl.load(row + 3)
    e2 = e1 + tl.load(row + 4)
    s2 = tl.load(row + 5)
    e3 = e2 + tl.load(row + 6)
    s3 = tl.load(row + 7)
    e4 = e3 + tl.load(row + 8)
    s4 = tl.load(row + 9)
    total = e4 + tl.load(row + 10)
    for j0 in range(0, total, BLOCK_N):
        j = j0 + tl.arange(0, BLOCK_N)
        valid = j < total
        in_source = (j < e1) | ((j >= e2) & (j < e3))
        pos = tl.where(j < e1, s0 + j,
              tl.where(j < e2, s1 + j - e1,
              tl.where(j < e3, s2 + j - e2,
              tl.where(j < e4, s3 + j - e3, s4 + j - e4))))
        ptrs = slot_base + pos[:, None] * s_cp + offs_d[None, :]
        if HAS_SOURCE:
            source_ptrs = source_base + pos[:, None] * s_sp + offs_d[None, :]
            ptrs = tl.where(in_source[:, None], source_ptrs, ptrs)
        k = tl.load(ptrs, mask=valid[:, None], other=0.0)
        v = tl.load(ptrs + D, mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * sm_scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=PRECISION)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(o_ptr + n * s_on + offs_m[:, None] * s_ot + h * s_oh + offs_d[None, :], out,
             mask=mask_m[:, None])


@triton.jit
def _bounded_store_kernel(
    k_ptr, v_ptr, cache_ptr, table_ptr,
    H,
    s_kn, s_kh, s_kt,
    s_cs, s_cr, s_ch, s_cp,
    ROWS: tl.constexpr, ROW_INTS: tl.constexpr, NUM_READS: tl.constexpr, REVERSE: tl.constexpr,
    BLOCK_T: tl.constexpr, D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_nh = tl.program_id(1)
    n = pid_nh // H
    h = pid_nh % H
    b = n // ROWS
    c = n % ROWS
    row = table_ptr + b * ROW_INTS
    slot = tl.load(row).to(tl.int64)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    for w in tl.static_range(3):
        base = row + 1 + 2 * NUM_READS + 3 * w
        offset = tl.load(base)
        start = tl.load(base + 1)
        count = tl.load(base + 2)
        mask = offs_t < count
        fresh = offset - offs_t if REVERSE else offset + offs_t
        src = n * s_kn + h * s_kh + fresh[:, None] * s_kt + offs_d[None, :]
        k = tl.load(k_ptr + src, mask=mask[:, None])
        v = tl.load(v_ptr + src, mask=mask[:, None])
        dst = cache_ptr + slot * s_cs + c * s_cr + h * s_ch + (start + offs_t)[:, None] * s_cp + offs_d[None, :]
        tl.store(dst, k, mask=mask[:, None])
        tl.store(dst + D, v, mask=mask[:, None])


def bounded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    source: torch.Tensor | None,
    table: torch.Tensor,
    ieee: bool = False,
) -> torch.Tensor:
    """``q, k, v [B * rows, H, T, D]`` (contiguous) over their retained ranges:
    ``cache [slots, rows, H, cap, 2D]`` and ``source [rows, H, L, 2D]`` per
    ``table [>= B, ROW_INTS]``. Returns ``[B * rows, T, H, D]``."""
    n, h, t, d = q.shape
    rows = cache.shape[1]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert cache.stride(-1) == 1 and table.shape[-1] == ROW_INTS
    has_source = source is not None
    if not has_source:
        source = cache
        s_src = (0, 0, 0)
    else:
        assert source.stride(-1) == 1
        s_src = (source.stride(0), source.stride(1), source.stride(2))
    out = torch.empty(n, t, h, d, device=q.device, dtype=q.dtype)
    block_m = 16 if n * h < 128 else 64
    grid = (triton.cdiv(t, block_m), n * h)
    _bounded_attention_kernel[grid](
        q, k, v, out, cache, source, table,
        t, h, d ** -0.5,
        q.stride(0), q.stride(1), q.stride(2),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        *s_src,
        out.stride(0), out.stride(1), out.stride(2),
        ROWS=rows, HAS_SOURCE=has_source, ROW_INTS=ROW_INTS,
        BLOCK_M=block_m, BLOCK_N=64, D=d, PRECISION="ieee" if ieee else "tf32",
    )
    return out


def bounded_store(
    k: torch.Tensor, v: torch.Tensor, cache: torch.Tensor, table: torch.Tensor, reverse: bool = False,
) -> None:
    """This step's ``k, v [B * rows, H, T, D]`` into the slots per ``table``'s writes
    (a range's tokens counting down from its offset when ``reverse``)."""
    n, h, t, d = k.shape
    assert k.is_contiguous() and v.is_contiguous() and cache.stride(-1) == 1
    block_t = 64
    grid = (triton.cdiv(t, block_t), n * h)
    _bounded_store_kernel[grid](
        k, v, cache, table, h,
        k.stride(0), k.stride(1), k.stride(2),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        ROWS=cache.shape[1], ROW_INTS=ROW_INTS, NUM_READS=NUM_READS, REVERSE=reverse,
        BLOCK_T=block_t, D=d,
    )


assert NUM_READS == 5 and NUM_WRITES == 3, "the kernels unroll five reads and three writes"
