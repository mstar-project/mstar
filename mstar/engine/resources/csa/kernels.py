"""CSA indexer scores over index keys held in their FP4 bytes on a paged cache's pages.

``decode_scores``: ``score[r, j] = sum_h w[r, h] * relu(q[r, h] . key_j)`` over the groups row
``r`` sees, each key read in place: two E2M1 codes a byte along its ``D`` values (the even one in
the low nibble), then one E8M0 scale byte per 32 of them (``mx_round.pack_fp4_rows``). The
arithmetic is the torch path's in bf16, step for step: a dot summed in fp32 and rounded to bf16,
rectified, times the head's weight rounded to bf16, the heads summed in fp32 and rounded to bf16.
Every product of two FP4 values is exact and so is their sum in fp32, so the rounded dot does
not depend on the order a kernel sums it in.

The DSA resource's score kernel (``dsa.kernels``) reads bf16 keys and keeps its sums in fp32,
which picks other rows than this model's reference: hence this one.

``attend``: each row's heads over its entries (window rows, then picked latents, gathered to
bf16 by the caller) in one softmax with a sink in its denominator, the arithmetic of the kernel
DeepSeek-V4.1 was served with: 64 entries at a time against the running maximum, each block's
probabilities rounded to bf16 before they weight the entries. Its dots sum in another order than
a GEMM's, so it agrees with the torch path to fp32 rounding, not bit for bit (neither does the
reference's own kernel with its torch stand-in).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# values a key's scale byte covers (``Fp4Rows(block=32)``, power-of-two scales)
SCALE_BLOCK = 32


@triton.jit
def _e2m1(codes):
    """fp32 values of E2M1 codes: magnitude e:m (3 bits) as an fp32 exponent / mantissa,
    ``(e + 126) << 23 | m << 22``, exact for e >= 1; e = 0 is 0 or 0.5. Sign in bit 3."""
    v = ((((codes & 7) << 22) + (126 << 23)) | ((codes & 8) << 28)).to(tl.float32, bitcast=True)
    small = (codes & 1).to(tl.float32) * 0.5
    small = tl.where((codes & 8) != 0, -small, small)
    return tl.where((codes & 6) == 0, small, v)


@triton.jit(do_not_specialize=["splits"])
def _decode_scores_kernel(q, w, plane, pages, page_start, row_req, seen, out, width, num_pages, splits,
                          NH: tl.constexpr, NHP: tl.constexpr, D: tl.constexpr, ROW: tl.constexpr,
                          PAGE: tl.constexpr, BP: tl.constexpr):
    """Program (row, split): the split walks every ``splits``-th block of BP groups the row
    sees. ``plane`` is the key plane's bytes, ``ROW`` of them a cache row. The ``NH`` heads sit
    in a tile of ``NHP`` (the dot's least), the spare ones zero."""
    r = tl.program_id(0)
    n = tl.load(seen + r)
    start = tl.load(page_start + tl.load(row_req + r))
    h = tl.arange(0, NHP)
    d = tl.arange(0, D)
    j = tl.arange(0, BP)
    qv = tl.load(q + (r.to(tl.int64) * NH + h[:, None]) * D + d[None, :], mask=(h < NH)[:, None], other=0.0)
    wv = tl.load(w + r.to(tl.int64) * NH + h, mask=h < NH, other=0.0).to(tl.float32)
    for blk in tl.range(tl.program_id(1), tl.cdiv(n, BP), splits):
        jj = blk * BP + j
        live = jj < n
        page = tl.load(pages + start + jj // PAGE, mask=live, other=0).to(tl.int64)
        # a request's own pages lie inside the cache; an entry read past them is masked
        page = tl.minimum(page, num_pages - 1)
        base = (page * PAGE + jj % PAGE) * ROW                                    # (BP,) byte offsets
        packed = tl.load(plane + base[:, None] + (d // 2)[None, :], mask=live[:, None], other=0).to(tl.int32)
        codes = (packed >> ((d % 2) * 4)[None, :]) & 0xF
        scale = tl.load(plane + base[:, None] + D // 2 + (d // 32)[None, :], mask=live[:, None], other=127)
        keys = (_e2m1(codes) * tl.exp2(scale.to(tl.float32) - 127.0)).to(tl.bfloat16)   # (BP, D), exact
        dots = tl.dot(qv, tl.trans(keys)).to(tl.bfloat16)                          # (NH, BP)
        dots = tl.maximum(dots.to(tl.float32), 0.0)
        weighted = (dots * wv[:, None]).to(tl.bfloat16).to(tl.float32)
        sc = tl.sum(weighted, 0).to(tl.bfloat16).to(tl.float32)
        tl.store(out + r.to(tl.int64) * width + jj, sc, mask=live)


def decode_scores(q: torch.Tensor, w: torch.Tensor, plane: torch.Tensor, pages: torch.Tensor,
                  page_start: torch.Tensor, row_req: torch.Tensor, seen: torch.Tensor,
                  width: int) -> torch.Tensor:
    """``[rows, width]`` fp32 scores, rows independent; the columns past the ``seen`` groups of
    a row are left unwritten. ``q (rows, NH, D)`` and ``w (rows, NH)`` bf16; ``plane (P, page,
    row elements)`` the cache's key plane; ``pages`` the flat page list (long), ``page_start``
    each request's first entry in it, ``row_req`` / ``seen`` per row (long)."""
    rows, nh, d = q.shape
    assert q.is_contiguous() and w.is_contiguous() and plane.is_contiguous()
    assert q.dtype == w.dtype == torch.bfloat16 and d >= 16 and d & (d - 1) == 0, (q.dtype, w.dtype, d)
    row_bytes = plane.shape[-1] * plane.element_size()
    assert row_bytes >= d // 2 + d // SCALE_BLOCK, (row_bytes, d)
    out = torch.empty(rows, width, dtype=torch.float32, device=q.device)
    bp = 64
    # enough programs to fill the GPU however few the rows: a long row's blocks spread out. A
    # runtime argument: as a constexpr, every new count would compile mid-request
    splits = max(1, min(triton.cdiv(width, bp), triton.cdiv(2048, rows)))
    _decode_scores_kernel[(rows, splits)](
        q, w, plane.view(torch.uint8), pages, page_start, row_req, seen, out, width, plane.shape[0], splits,
        NH=nh, NHP=max(16, triton.next_power_of_2(nh)), D=d, ROW=row_bytes, PAGE=plane.shape[1], BP=bp,
        num_warps=4, num_stages=2)
    return out


@triton.jit
def _attend_kernel(q, entries, valid, sink, acc, out, scale,
                   H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, N: tl.constexpr, BK: tl.constexpr,
                   BD: tl.constexpr):
    """Program per row: its ``N`` entries in blocks of BK (``valid`` says which it attends), the
    dots in BD-wide pieces of the D values. ``acc`` is the row's fp32 sums, zero at launch; the
    ``H`` heads sit in a tile of ``HP``."""
    r = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)
    live_h = h < H
    k = tl.arange(0, BK)
    d = tl.arange(0, BD)
    q_row = q + (r * H + h[:, None]) * D
    acc_row = acc + (r * HP + h[:, None]) * D
    peak = tl.full((HP,), -1e30, dtype=tl.float32)
    total = tl.zeros((HP,), dtype=tl.float32)
    for block in range(0, N // BK):
        kk = block * BK + k
        live_k = tl.load(valid + r * N + kk) != 0
        e_row = entries + (r * N + kk[:, None]) * D
        scores = tl.zeros((HP, BK), dtype=tl.float32)
        for piece in tl.static_range(D // BD):
            qv = tl.load(q_row + piece * BD + d[None, :], mask=live_h[:, None], other=0.0)
            ev = tl.load(e_row + piece * BD + d[None, :], mask=live_k[:, None], other=0.0)
            scores += tl.dot(qv, tl.trans(ev))
        scores = tl.where(live_k[None, :], scores * scale, float("-inf"))
        last = peak
        peak = tl.maximum(peak, tl.max(scores, 1))
        decay = tl.exp(last - peak)
        prob = tl.exp(scores - peak[:, None])
        total = total * decay + tl.sum(prob, 1)
        rounded = prob.to(tl.bfloat16)
        for piece in tl.static_range(D // BD):
            ev = tl.load(e_row + piece * BD + d[None, :], mask=live_k[:, None], other=0.0)
            sums = tl.load(acc_row + piece * BD + d[None, :])
            tl.store(acc_row + piece * BD + d[None, :], sums * decay[:, None] + tl.dot(rounded, ev))
    total += tl.exp(tl.load(sink + h, mask=live_h, other=0.0) - peak)
    out_row = out + (r * H + h[:, None]) * D
    for piece in tl.static_range(D // BD):
        sums = tl.load(acc_row + piece * BD + d[None, :])
        tl.store(out_row + piece * BD + d[None, :], (sums / total[:, None]).to(tl.bfloat16), mask=live_h[:, None])


def attend(q: torch.Tensor, entries: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor,
           scale: float, block: int = 64) -> torch.Tensor:
    """``[rows, heads, D]`` bf16: each row's heads over its ``entries (rows, N, D)`` bf16 where
    ``valid (rows, N)`` (bool) holds, ``N`` a multiple of ``block`` (an entry's place decides its
    block); ``sink (heads,)`` fp32. A row with no entry gets zeros."""
    rows, heads, dim = q.shape
    n = entries.shape[1]
    assert q.dtype == entries.dtype == torch.bfloat16 and sink.dtype == torch.float32
    assert valid.dtype == torch.bool and valid.shape == entries.shape[:2], (valid.dtype, valid.shape)
    assert q.is_contiguous() and entries.is_contiguous() and valid.is_contiguous() and n % block == 0, (n, block)
    assert dim >= 16 and dim & (dim - 1) == 0 and block >= 16, (dim, block)
    padded = max(16, triton.next_power_of_2(heads))
    acc = torch.zeros(rows, padded, dim, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    # a second stage loads the next block's entries ahead, in shared memory: past 32 heads that
    # is more than a Hopper GPU gives a block of threads (393 KB of 232 at 64 heads x 512)
    _attend_kernel[(rows,)](
        q, entries, valid.view(torch.uint8), sink.contiguous(), acc, out, scale,
        H=heads, HP=padded, D=dim, N=n, BK=block, BD=min(128, dim), num_warps=8,
        num_stages=2 if padded <= 32 else 1)
    return out
