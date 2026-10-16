"""GLM-5.3-Flash k-pool indexer kernels over the index planes of the paged MLA cache.

An index plane row ``t`` holds ``[k_t | gate_t | key of the pool ending at t | unused]``
(``D`` = index_head_dim each). Pools are ``KP`` consecutive tokens from position 0; pool ``j``
covers ``[KP*j, KP*j + KP)`` and its key lives in the row of its last token. Pages hold a
multiple of ``KP`` tokens, so a pool never straddles a page.

``pool_keys`` writes the key of every pool a step's row completes: per channel, a softmax over
the members of ``gate + ape`` (fp32, rounded to bf16), times the member keys (each product
rounded to bf16), summed in fp32 and rounded to bf16, the reference's bf16 arithmetic.

``decode_scores`` / ``prefill_scores``: ``score[r, j] = sum_h w[r, h] * relu(q[r, h] . key_j)``
over the pools row ``r`` sees (``(pos[r] + 1) // KP``), bf16 dots with fp32 accumulation. ``w``
carries the softmax scale and the head-count scale (relu commutes with a positive scale).
Columns past a row's pools are left unwritten: the top-k reads only ``[0, pools)``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pool_keys_kernel(plane, pages, page_start, row_req, pos, ape,
                      W: tl.constexpr, D: tl.constexpr, KP: tl.constexpr, PAGE: tl.constexpr):
    """Program per step row; a row at the last position of a pool writes that pool's key."""
    r = tl.program_id(0)
    p = tl.load(pos + r)
    if p % KP != KP - 1:
        return
    start = tl.load(page_start + tl.load(row_req + r))
    t0 = p - (KP - 1)
    page = tl.load(pages + start + t0 // PAGE).to(tl.int64)
    m = tl.arange(0, KP)
    d = tl.arange(0, D)
    base = (page * PAGE + t0 % PAGE + m)[:, None] * W + d[None, :]           # (KP, D)
    k = tl.load(plane + base).to(tl.float32)
    logits = tl.load(plane + base + D).to(tl.float32) + tl.load(ape + m[:, None] * D + d[None, :])
    e = tl.exp(logits - tl.max(logits, 0)[None, :])
    prob = (e / tl.sum(e, 0)[None, :]).to(tl.bfloat16).to(tl.float32)
    key = tl.sum((prob * k).to(tl.bfloat16).to(tl.float32), 0)
    tl.store(plane + (page * PAGE + p % PAGE) * W + 2 * D + d, key.to(tl.bfloat16))


@triton.jit(do_not_specialize=["splits"])
def _decode_scores_kernel(q, w, plane, pages, page_start, row_req, pos, out, max_pools, splits,
                          NH: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
                          KP: tl.constexpr, PAGE: tl.constexpr, BP: tl.constexpr):
    """Program (row, split): the split walks every ``splits``-th block of BP pools the row
    sees."""
    r = tl.program_id(0)
    n = (tl.load(pos + r) + 1) // KP
    start = tl.load(page_start + tl.load(row_req + r))
    h = tl.arange(0, NH)
    d = tl.arange(0, D)
    j = tl.arange(0, BP)
    qv = tl.load(q + (r.to(tl.int64) * NH + h[:, None]) * D + d[None, :])
    wv = tl.load(w + r.to(tl.int64) * NH + h)
    for blk in tl.range(tl.program_id(1), tl.cdiv(n, BP), splits):
        jj = blk * BP + j
        live = jj < n
        last = jj * KP + KP - 1                                                # pools' last tokens
        page = tl.load(pages + start + last // PAGE, mask=live, other=0).to(tl.int64)
        kv = tl.load(plane + (page * PAGE + last % PAGE)[:, None] * W + 2 * D + d[None, :],
                     mask=live[:, None], other=0.0)
        dots = tl.maximum(tl.dot(qv, tl.trans(kv)), 0.0)                       # (NH, BP)
        sc = tl.sum(dots * wv[:, None], 0)
        tl.store(out + r.to(tl.int64) * max_pools + jj, sc, mask=live)


@triton.jit(do_not_specialize=["splits"])
def _prefill_scores_kernel(q, w, plane, pages, start, pos, out, R, max_pools, splits,
                           NH: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
                           KP: tl.constexpr, PAGE: tl.constexpr, BR: tl.constexpr,
                           BP: tl.constexpr):
    """Program (BR rows of one request, split): the rows' heads sit on the dot's M side, and
    the split walks every ``splits``-th block of BP pools up to the rows' last, so a row
    block's queries load once."""
    rb = tl.program_id(0)
    r = rb * BR + tl.arange(0, BR)
    live_r = r < R
    n = (tl.load(pos + r, mask=live_r, other=-1) + 1) // KP
    d = tl.arange(0, D)
    m = tl.arange(0, BR * NH)                                                 # (row, head)
    live_m = (rb * BR + m // NH) < R
    qv = tl.load(q + (rb * BR * NH + m)[:, None].to(tl.int64) * D + d[None, :],
                 mask=live_m[:, None], other=0.0)
    wv = tl.load(w + rb * BR * NH + m, mask=live_m, other=0.0)
    n_max = tl.max(n, 0)
    for blk in tl.range(tl.program_id(1), tl.cdiv(n_max, BP), splits):
        j = blk * BP + tl.arange(0, BP)
        live_j = j < n_max
        last = j * KP + KP - 1
        page = tl.load(pages + start + last // PAGE, mask=live_j, other=0).to(tl.int64)
        kv = tl.load(plane + (page * PAGE + last % PAGE)[:, None] * W + 2 * D + d[None, :],
                     mask=live_j[:, None], other=0.0)                          # (BP, D)
        dots = tl.maximum(tl.dot(qv, tl.trans(kv)), 0.0)                      # (BR*NH, BP)
        sc = tl.sum(tl.reshape(dots * wv[:, None], [BR, NH, BP]), 1)          # (BR, BP)
        tl.store(out + r[:, None].to(tl.int64) * max_pools + j[None, :], sc,
                 mask=live_r[:, None] & (j[None, :] < n[:, None]))


@triton.jit
def _expand_slots_kernel(first, pos, row_req, pages, page_start, out, K, W,
                         KP: tl.constexpr, PAGE: tl.constexpr, BLOCK: tl.constexpr):
    """Program per row: the members of its selected pools (``first`` holds each pool's first
    slot, -1 past the selection), then its tail ``[KP * pools, pos]``, packed from entry 0;
    -1 after."""
    r = tl.program_id(0)
    p = tl.load(pos + r)
    pools = (p + 1) // KP
    n = tl.minimum(pools, K) * KP
    start = tl.load(page_start + tl.load(row_req + r))
    for e0 in range(0, W, BLOCK):
        e = e0 + tl.arange(0, BLOCK)
        in_pool = e < n
        f = tl.load(first + r.to(tl.int64) * K + e // KP, mask=in_pool, other=0)
        t = pools * KP + (e - n)                                    # tail position
        in_tail = (e >= n) & (t <= p)
        page = tl.load(pages + start + t // PAGE, mask=in_tail, other=0)
        val = tl.where(in_pool, f + e % KP, tl.where(in_tail, page * PAGE + t % PAGE, -1))
        tl.store(out + r.to(tl.int64) * W + e, val, mask=e < W)


def expand_slots(first: torch.Tensor, pos: torch.Tensor, row_req: torch.Tensor,
                 pages: torch.Tensor, page_start: torch.Tensor, kp: int, page_size: int,
                 width: int) -> torch.Tensor:
    """``[rows, width]`` int32 slots: each row's selected pools' members, then its tail."""
    rows, k = first.shape
    out = torch.empty(rows, width, dtype=torch.int32, device=first.device)
    _expand_slots_kernel[(rows,)](first, pos, row_req, pages, page_start, out, k, width,
                                  KP=kp, PAGE=page_size, BLOCK=1024, num_warps=4)
    return out


def _check_plane(plane: torch.Tensor, d: int, kp: int) -> None:
    assert plane.dtype == torch.bfloat16 and plane.is_contiguous()
    assert plane.shape[-1] >= 3 * d and plane.shape[1] % kp == 0


def pool_keys(plane: torch.Tensor, pages: torch.Tensor, page_start: torch.Tensor,
              row_req: torch.Tensor, pos: torch.Tensor, ape: torch.Tensor, d: int) -> None:
    """Write the key of every pool a row of ``pos`` completes into ``plane`` (in place).
    ``plane (P, page, W)`` bf16; ``pages`` int32 flat page list, ``page_start`` int32 per
    request, ``row_req`` / ``pos`` int32 per row; ``ape (KP, D)`` fp32."""
    kp = ape.shape[0]
    _check_plane(plane, d, kp)
    _pool_keys_kernel[(pos.shape[0],)](
        plane, pages, page_start, row_req, pos, ape.float().contiguous(),
        W=plane.shape[-1], D=d, KP=kp, PAGE=plane.shape[1], num_warps=4)


def decode_scores(q: torch.Tensor, w: torch.Tensor, plane: torch.Tensor, pages: torch.Tensor,
                  page_start: torch.Tensor, row_req: torch.Tensor, pos: torch.Tensor,
                  max_pools: int, kp: int) -> torch.Tensor:
    """``[rows, max_pools]`` fp32 pool scores, rows independent. ``q (rows, NH, D)`` bf16,
    ``w (rows, NH)`` fp32."""
    rows, nh, d = q.shape
    _check_plane(plane, d, kp)
    assert nh >= 16 and d >= 16 and q.is_contiguous() and w.is_contiguous()
    out = torch.empty(rows, max_pools, dtype=torch.float32, device=q.device)
    bp = 64
    # enough programs to fill the GPU however few the rows: a long row's blocks spread out.
    # A runtime argument: as a constexpr, every new count would compile mid-request
    splits = max(1, min(triton.cdiv(max_pools, bp), triton.cdiv(2048, rows)))
    _decode_scores_kernel[(rows, splits)](
        q, w, plane, pages, page_start, row_req, pos, out, max_pools, splits,
        NH=nh, D=d, W=plane.shape[-1], KP=kp, PAGE=plane.shape[1], BP=bp,
        num_warps=4, num_stages=2)
    return out


def prefill_scores(q: torch.Tensor, w: torch.Tensor, plane: torch.Tensor, pages: torch.Tensor,
                   start: int, pos: torch.Tensor, max_pools: int, kp: int) -> torch.Tensor:
    """``[rows, max_pools]`` fp32 pool scores for rows of one request whose pages begin at
    ``pages[start]``."""
    rows, nh, d = q.shape
    _check_plane(plane, d, kp)
    assert nh >= 16 and d >= 16 and q.is_contiguous() and w.is_contiguous()
    out = torch.empty(rows, max_pools, dtype=torch.float32, device=q.device)
    br, bp = 4, 64
    blocks = triton.cdiv(rows, br)
    # enough programs to fill the GPU: splits per row block, more for few rows (a runtime
    # argument, as in decode_scores)
    splits = max(1, min(triton.cdiv(max_pools, bp), triton.cdiv(1024, blocks)))
    _prefill_scores_kernel[(blocks, splits)](
        q, w, plane, pages, start, pos, out, rows, max_pools, splits,
        NH=nh, D=d, W=plane.shape[-1], KP=kp, PAGE=plane.shape[1], BR=br, BP=bp,
        num_warps=4, num_stages=2)
    return out
