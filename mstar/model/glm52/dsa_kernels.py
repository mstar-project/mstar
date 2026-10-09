"""GLM-5.2 DSA indexer scores over a paged index-key store, in one pass.

``score[r, s] = sum_h w[r, h] * relu(q[r, h] . k[s])`` for a query row ``r`` over its request's
keys ``s``, the indexer's selection score (components/indexer.py computes the same in fp32 on
a gathered history). The output is ``[rows, max_len]`` fp32, what
``flashinfer.top_k_page_table_transform`` takes; it reads a row's first ``lens`` columns only.

The key store is ``[pages, page_size, head_dim]`` bf16, one layer view of an MLA-layout KV
resource with a ``head_dim`` latent, addressed through a request's page-table row. Each dot is
bf16 x bf16 with fp32 accumulation, so every product is exact; the relu and the w-weighted
head sum run in fp32. Against the stored keys the scores differ by summation order only. The
store rounds the indexer's fp32 keys to bf16: against those the scores move ~2e-3 relative,
and a row's top-2048 changes a few picks.

Two launch shapes: decode (``decode_scores``: programs per row that stride over its key blocks,
each row its own request) and prefill (``prefill_scores``: BR rows of one request per program,
so a key block is read once per row block rather than once per row).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Decode programs per launch, about four per SM on an H100/H200.
_DECODE_PROGRAMS = 528


@triton.jit
def _decode_scores_kernel(q, w, k_pages, table, row_req, lens, out, max_len, max_pages,
                          splits, NH: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                          BK: tl.constexpr):
    """Program (row, split): row ``r`` attends keys ``[0, lens[r])``, and the split takes
    every ``splits``-th BK-key block of them (a block stays inside one page). Blocks past
    ``lens[r]`` are not visited, so the cost follows the context, not ``max_len``."""
    r = tl.program_id(0)
    n = tl.load(lens + r)
    req = tl.load(row_req + r).to(tl.int64)
    h = tl.arange(0, NH)
    d = tl.arange(0, D)
    qv = tl.load(q + (r.to(tl.int64) * NH + h[:, None]) * D + d[None, :])
    wv = tl.load(w + r.to(tl.int64) * NH + h)
    for s0 in range(tl.program_id(1) * BK, n, splits * BK):
        page = tl.load(table + req * max_pages + s0 // PAGE).to(tl.int64)
        kv = tl.load(k_pages + (page * PAGE + s0 % PAGE + tl.arange(0, BK))[:, None] * D
                     + d[None, :])
        dots = tl.maximum(tl.dot(qv, tl.trans(kv)), 0.0)                # (NH, BK)
        sc = tl.sum(dots * wv[:, None], 0)
        s = s0 + tl.arange(0, BK)
        tl.store(out + r.to(tl.int64) * max_len + s, tl.where(s < n, sc, float("-inf")),
                 mask=s < max_len)


@triton.jit
def _prefill_scores_kernel(q, w, k_pages, table, pos, out, R, max_len,
                           NH: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                           BR: tl.constexpr, BK: tl.constexpr):
    """Program (BR-row block, BK-key block): the rows' heads sit on the dot's M side; key
    ``s`` is visible to row ``r`` iff ``s <= pos[r]``."""
    rb = tl.program_id(0)
    s0 = tl.program_id(1) * BK
    r = rb * BR + tl.arange(0, BR)
    live_r = r < R
    p = tl.load(pos + r, mask=live_r, other=-1)
    s = s0 + tl.arange(0, BK)
    dst = out + r[:, None].to(tl.int64) * max_len + s[None, :]
    live = live_r[:, None] & (s[None, :] < max_len)
    if s0 > tl.max(p, 0):  # the whole block lies past every row's position
        tl.store(dst, float("-inf"), mask=live)
        return
    page = tl.load(table + s0 // PAGE).to(tl.int64)
    d = tl.arange(0, D)
    m = tl.arange(0, BR * NH)  # (row, head), row-major
    live_m = (rb * BR + m // NH) < R
    qv = tl.load(q + (rb * BR * NH + m)[:, None].to(tl.int64) * D + d[None, :],
                 mask=live_m[:, None], other=0.0)
    kv = tl.load(k_pages + (page * PAGE + s0 % PAGE + tl.arange(0, BK))[:, None] * D + d[None, :])
    dots = tl.maximum(tl.dot(qv, tl.trans(kv)), 0.0)                    # (BR * NH, BK)
    wv = tl.load(w + rb * BR * NH + m, mask=live_m, other=0.0)
    sc = tl.sum(tl.reshape(dots * wv[:, None], [BR, NH, BK]), 1)       # (BR, BK)
    tl.store(dst, tl.where(s[None, :] <= p[:, None], sc, float("-inf")), mask=live)


def _check(q, w, k_pages, bk):
    rows, nh, d = q.shape
    assert q.dtype == k_pages.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert w.shape == (rows, nh) and k_pages.shape[-1] == d and k_pages.shape[1] % bk == 0
    assert nh >= 16 and d >= 16, f"tl.dot needs 16+ index heads and dims, got {nh} x {d}"
    assert q.is_contiguous() and w.is_contiguous() and k_pages.is_contiguous()


def decode_scores(q: torch.Tensor, w: torch.Tensor, k_pages: torch.Tensor,
                  page_table: torch.Tensor, row_req: torch.Tensor, lens: torch.Tensor,
                  max_len: int) -> torch.Tensor:
    """``[rows, max_len]`` fp32 scores; a row's columns at or past its ``lens`` are left
    unwritten (top-k reads ``[0, lens)``) except inside its last key block, where they are
    ``-inf``. ``q (rows, NH, D)`` bf16 (roped), ``w (rows, NH)`` fp32 (weights_proj x
    head_dim^-0.5 x NH^-0.5), ``k_pages (P, page, D)`` bf16, ``page_table (B, max_pages)``
    int32, ``row_req (rows,)`` a row's page-table row, ``lens (rows,)`` keys a row sees
    (position + 1)."""
    rows, nh, d = q.shape
    work = rows * max_len
    bk, warps = (64, 4) if work <= 8 * 8192 else (128, 8)  # H100 sweep, prototype job 9761
    bk = min(bk, k_pages.shape[1])  # a key block stays inside one page
    _check(q, w, k_pages, bk)
    # enough programs to fill the GPU at any row count; each strides over its row's blocks
    splits = max(1, min(triton.cdiv(max_len, bk), triton.cdiv(_DECODE_PROGRAMS, rows)))
    out = torch.empty(rows, max_len, dtype=torch.float32, device=q.device)
    _decode_scores_kernel[(rows, splits)](
        q, w, k_pages, page_table, row_req, lens, out, max_len, page_table.shape[1], splits,
        NH=nh, D=d, PAGE=k_pages.shape[1], BK=bk, num_warps=warps, num_stages=1)
    return out


def prefill_scores(q: torch.Tensor, w: torch.Tensor, k_pages: torch.Tensor,
                   page_table_row: torch.Tensor, positions: torch.Tensor,
                   max_len: int) -> torch.Tensor:
    """``[rows, max_len]`` fp32 scores for one request's rows at ``positions``;
    ``page_table_row (max_pages,)`` int32 is that request's row."""
    rows, nh, d = q.shape
    br, bk = 4, 128  # H100 sweep, prototype job 9771: ~240 bf16 TFLOP/s
    bk = min(bk, k_pages.shape[1])  # a key block stays inside one page
    _check(q, w, k_pages, bk)
    out = torch.empty(rows, max_len, dtype=torch.float32, device=q.device)
    _prefill_scores_kernel[(triton.cdiv(rows, br), triton.cdiv(max_len, bk))](
        q, w, k_pages, page_table_row, positions, out, rows, max_len,
        NH=nh, D=d, PAGE=k_pages.shape[1], BR=br, BK=bk, num_warps=4, num_stages=2)
    return out
