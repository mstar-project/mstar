"""One Markov drafting step of the DSpark draft in two launches.

The draft's token ``i`` is the argmax over the vocabulary of ``logits[:, i] + W2 @ W1[prev]``, the
rank-r Markov bias of the previously drafted token. Spelled out in torch that is a gather, a GEMM
over the whole vocabulary, two fp32 copies of ``[rows, V]``, an add and an argmax: six launches and
about 200 us per drafted token at 163840 columns. Here the first kernel handles a chunk of the
vocabulary for every row at once (the bias by ``tl.dot`` against the gathered ``W1`` rows, the sum
in fp32, the chunk's maximum and its first position), the second reduces the chunks per row with the
lowest index winning ties, as ``torch.argmax`` does. ``W2`` is read once per step, which is the
floor. With bf16 weights the bias is rounded to bf16 before the sum, as the GEMM's output was.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_INF = float("inf")
_BIG = 2**31 - 1


@triton.jit
def _markov_partial_kernel(
    logits, w1, w2, prev, out_val, out_idx,
    V, stride_l_row, stride_w1, stride_w2, n_chunks,
    ROWS: tl.constexpr, RP: tl.constexpr, R: tl.constexpr, BV: tl.constexpr,
    ROUND_BF16: tl.constexpr, IEEE: tl.constexpr,
):
    c = tl.program_id(0)
    v0 = c * BV
    o_v = v0 + tl.arange(0, BV)
    m_v = o_v < V
    o_r = tl.arange(0, R)
    o_rows = tl.arange(0, RP)
    m_rows = o_rows < ROWS
    prev_ids = tl.load(prev + o_rows, mask=m_rows, other=0).to(tl.int64)
    e = tl.load(w1 + prev_ids[:, None] * stride_w1 + o_r[None, :], mask=m_rows[:, None], other=0.0)  # [RP, R]
    w = tl.load(w2 + o_v[:, None] * stride_w2 + o_r[None, :], mask=m_v[:, None], other=0.0)  # [BV, R]
    if IEEE:
        bias = tl.dot(w, tl.trans(e), input_precision="ieee")  # [BV, RP] fp32
    else:
        bias = tl.dot(w, tl.trans(e))
    if ROUND_BF16:
        bias = bias.to(tl.bfloat16).to(tl.float32)
    lg = tl.load(logits + o_rows[None, :] * stride_l_row + o_v[:, None], mask=m_v[:, None] & m_rows[None, :], other=0.0)
    s = lg.to(tl.float32) + bias
    s = tl.where(m_v[:, None], s, float("-inf"))
    mx = tl.max(s, axis=0)
    am = tl.argmax(s, axis=0)
    tl.store(out_val + o_rows * n_chunks + c, mx, mask=m_rows)
    tl.store(out_idx + o_rows * n_chunks + c, (v0 + am).to(tl.int32), mask=m_rows)


@triton.jit
def _markov_final_kernel(out_val, out_idx, drafts, n_chunks, stride_d_row, NC: tl.constexpr):
    r = tl.program_id(0)
    o_c = tl.arange(0, NC)
    m_c = o_c < n_chunks
    vals = tl.load(out_val + r * n_chunks + o_c, mask=m_c, other=float("-inf"))
    idxs = tl.load(out_idx + r * n_chunks + o_c, mask=m_c, other=2147483647)
    mx = tl.max(vals, axis=0)
    best = tl.min(tl.where(vals == mx, idxs, 2147483647), axis=0)
    tl.store(drafts + r * stride_d_row, best.to(drafts.dtype.element_ty))


def markov_argmax(logits: torch.Tensor, prev: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                  out: torch.Tensor, workspace: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
    """``out[r] = argmax_v(logits[r, v] + w2[v] . w1[prev[r]])`` for ``logits [rows, V]`` (any row
    stride), ``prev [rows]`` ids, ``w1 [V, R]`` and ``w2 [V, R]``; ``out`` a ``[rows]`` integer view
    (a column of the drafts) written in place. ``workspace``: the ``[rows, n_chunks]`` fp32 and int32
    partials, reused across the draft's steps."""
    rows, v = logits.shape
    r = w1.shape[1]
    assert w2.shape == (v, r) and w1.shape[0] == v and logits.stride(1) == 1 and prev.numel() == rows
    assert r % 16 == 0 and r >= 16, "the Markov rank must be a multiple of 16"
    bv = 128
    n_chunks = triton.cdiv(v, bv)
    if workspace is None:
        workspace = (torch.empty(rows, n_chunks, dtype=torch.float32, device=logits.device),
                     torch.empty(rows, n_chunks, dtype=torch.int32, device=logits.device))
    vals, idxs = workspace
    rp = max(16, triton.next_power_of_2(rows))
    _markov_partial_kernel[(n_chunks,)](
        logits, w1, w2, prev, vals, idxs, v, logits.stride(0), w1.stride(0), w2.stride(0), n_chunks,
        ROWS=rows, RP=rp, R=r, BV=bv, ROUND_BF16=w2.dtype == torch.bfloat16, IEEE=w2.dtype == torch.float32,
        num_warps=4,
    )
    _markov_final_kernel[(rows,)](vals, idxs, out, n_chunks, out.stride(0), NC=triton.next_power_of_2(n_chunks), num_warps=4)


def markov_argmax_workspace(rows: int, v: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n_chunks = triton.cdiv(v, 128)
    return (torch.empty(rows, n_chunks, dtype=torch.float32, device=device),
            torch.empty(rows, n_chunks, dtype=torch.int32, device=device))
