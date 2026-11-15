# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Copyright (c) 2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: repetition-aware sampling
# redrawn as hash-seeded Gumbel-max draws that a CUDA graph can capture.
#
# Original file (FunAudioLLM/CosyVoice, cosyvoice/utils/common.py) was
# released under Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""Repetition-aware sampling (RAS) for the CosyVoice3 speech-token LM.

The reference (``cosyvoice.utils.common.ras_sampling``) draws a token from the
nucleus: ids in descending probability while the mass before them is under
``top_p`` and fewer than ``top_k`` are taken. If that token already appears
``>= window * tau`` times among the last ``window`` generated tokens, it is
masked and a new token is drawn from the whole distribution. Without the
redraw, a nucleus that collapses onto a silence token never reaches EOS.

Here both draws are Gumbel-max over noise from a counter-based hash of
``(seed, step, id)``: one fixed-shape tensor program per batch that a CUDA
graph can capture, the same tokens whatever the batch around a request, and a
request's seed reproduces its tokens. The draws follow the reference's
distributions; they cannot follow its random stream, which is the global
``torch.multinomial`` state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Largest top_k the fixed-shape nucleus covers (the reference uses 25).
MAX_TOP_K = 128

_MASK32 = 0xFFFFFFFF
# The redraw uses its own noise stream.
_REDRAW_SALT = 0x5BD1E995


def _mulmod32(x: torch.Tensor, c: int) -> torch.Tensor:
    """``x * c mod 2**32`` for uint32 values in int64 lanes, without overflowing
    64 bits (signed overflow is undefined on the CPU)."""
    lo = x * (c & 0xFFFF)
    hi = ((x * (c >> 16)) & 0xFFFF) << 16
    return (lo + hi) & _MASK32


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """murmur3's 32-bit finaliser on int64 lanes holding uint32 values."""
    x = x ^ (x >> 16)
    x = _mulmod32(x, 0x85EBCA6B)
    x = x ^ (x >> 13)
    x = _mulmod32(x, 0xC2B2AE35)
    return x ^ (x >> 16)


def hashed_uniform(seed: torch.Tensor, step: torch.Tensor, salt: int, n: int) -> torch.Tensor:
    """``[B, n]`` uniforms in (0, 1), a function of ``(seed[b], step[b], salt, j)`` only."""
    j = torch.arange(n, device=seed.device, dtype=torch.int64)
    s = _mix32((seed & _MASK32) ^ _mulmod32((seed >> 32) & _MASK32, 0x27D4EB2F))
    t = _mix32((step & _MASK32) ^ salt)
    x = _mix32(_mulmod32(s, 0x9E3779B1) ^ t)
    x = _mix32((x[:, None] + _mulmod32(j, 0x632BE5AB)[None, :]) & _MASK32)
    return (x.to(torch.float64) + 0.5) / 4294967296.0


def _gumbel(seed: torch.Tensor, step: torch.Tensor, salt: int, n: int) -> torch.Tensor:
    u = hashed_uniform(seed, step, salt, n)
    return (-torch.log(-torch.log(u))).to(torch.float32)


@triton.jit
def _mix32_tl(x):
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    return x ^ (x >> 16)


@triton.jit
def _row_mix_tl(seed, step, salt):
    """What ``hashed_uniform`` adds each id's hash to, for one row."""
    lo = (seed & 0xFFFFFFFF).to(tl.uint32)
    hi = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    s = _mix32_tl(lo ^ (hi * 0x27D4EB2F))
    t = _mix32_tl((step & 0xFFFFFFFF).to(tl.uint32) ^ salt)
    return _mix32_tl((s * 0x9E3779B1) ^ t)


@triton.jit
def _gumbel_tl(row_mix, ids):
    """``_gumbel`` of one row's ids: uint32 multiplies are its ``_mulmod32``."""
    x = _mix32_tl(row_mix + ids * 0x632BE5AB)
    u = (x.to(tl.float64) + 0.5) / 4294967296.0
    return (-tl.log(-tl.log(u))).to(tl.float32)


@triton.jit
def _first_draw_kernel(TOP_LP, KEEP, SEED, STEP, PICK, k, BLOCK: tl.constexpr):
    """A row's Gumbel-max over its kept nucleus: ``PICK`` is the winning rank."""
    row = tl.program_id(0)
    row_mix = _row_mix_tl(tl.load(SEED + row), tl.load(STEP + row), 0)
    best = -float("inf")
    best_at = 0
    for start in range(0, k, BLOCK):
        ranks = start + tl.arange(0, BLOCK)
        mask = ranks < k
        keep = tl.load(KEEP + row * k + ranks, mask=mask, other=0)
        lp = tl.load(TOP_LP + row * k + ranks, mask=mask, other=0.0)
        score = tl.where(keep != 0, lp + _gumbel_tl(row_mix, ranks.to(tl.uint32)), -float("inf"))
        at = tl.argmax(score, 0)
        value = tl.max(score, 0)
        if value > best:
            best = value
            best_at = start + at
    tl.store(PICK + row, best_at.to(PICK.dtype.element_ty))


@triton.jit
def _redraw_kernel(LOG_PROBS, SEED, STEP, TOKEN, REDRAW, OUT, vocab, BLOCK: tl.constexpr):
    """Rows that repeated a token draw again from the whole distribution, without it.
    Most rows don't, and leave at once."""
    row = tl.program_id(0)
    token = tl.load(TOKEN + row)
    if tl.load(REDRAW + row) == 0:
        tl.store(OUT + row, token)
    else:
        row_mix = _row_mix_tl(tl.load(SEED + row), tl.load(STEP + row), 0x5BD1E995)
        best = -float("inf")
        best_at = token
        for start in range(0, vocab, BLOCK):
            ids = start + tl.arange(0, BLOCK)
            mask = ids < vocab
            lp = tl.load(LOG_PROBS + row * vocab + ids, mask=mask, other=-float("inf"))
            lp = tl.where(ids == token, -float("inf"), lp)
            score = lp + _gumbel_tl(row_mix, ids.to(tl.uint32))
            score = tl.where(mask, score, -float("inf"))
            at = tl.argmax(score, 0)
            value = tl.max(score, 0)
            if value > best:
                best = value
                best_at = (start + at).to(tl.int64)
        tl.store(OUT + row, best_at)


def _fused_draws(log_probs, top_lp, top_ids, keep, seed, step, greedy, history, window, repeat_threshold):
    """The noise and redraw of ``ras_sample`` in two launches, tokens unchanged."""
    b, k = top_lp.shape
    vocab = log_probs.shape[-1]
    pick = torch.empty(b, dtype=torch.int64, device=log_probs.device)
    keep = keep.to(torch.uint8)
    _first_draw_kernel[(b,)](top_lp, keep, seed, step, pick, k, BLOCK=64, num_warps=1, num_stages=1)
    token = top_ids.gather(1, pick[:, None]).squeeze(1)
    token = torch.where(greedy, top_ids[:, 0], token)
    width = history.shape[-1]
    recent = torch.arange(width, device=history.device)[None, :] >= width - window[:, None]
    repeats = ((history == token[:, None]) & recent).sum(dim=-1).to(repeat_threshold.dtype)
    redraw = ((repeat_threshold > 0) & (repeats >= repeat_threshold)).to(torch.uint8)
    out = torch.empty(b, dtype=torch.int64, device=log_probs.device)
    _redraw_kernel[(b,)](log_probs, seed, step, token, redraw, out, vocab, BLOCK=256, num_warps=1, num_stages=1)
    return out


def ras_sample(logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """``ras_sample_plain``, on CUDA with the draws in one launch each (same tokens)."""
    return ras_sample_plain(logits, fused=logits.is_cuda, **kwargs)


def ras_sample_plain(
    logits: torch.Tensor,
    *,
    seed: torch.Tensor,
    step: torch.Tensor,
    history: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    window: torch.Tensor,
    repeat_threshold: torch.Tensor,
    blocked: torch.Tensor,
    fused: bool = False,
) -> torch.Tensor:
    """One token per row.

    ``logits`` ``[B, V]`` float32. Per row: ``seed``/``step`` (int64) key the
    noise; ``history`` ``[B, W]`` holds the last W generated tokens (-1 where
    fewer exist); ``temperature`` (0 = greedy), ``top_p``, ``top_k`` (<=
    ``MAX_TOP_K``), ``window`` (the row's RAS window, <= W: repeats count in
    its last ``window`` history slots only) and ``repeat_threshold``
    (``window * tau``; <= 0 disables the redraw) are ``[B]``; ``blocked``
    ``[B, V]`` bool marks ids that may not be drawn this step (the
    reference's minimum-length rule).
    """
    vocab = logits.shape[-1]
    greedy = temperature <= 0
    scale = torch.where(greedy, torch.ones_like(temperature), temperature).to(logits.dtype)
    scores = (logits / scale[:, None]).masked_fill(blocked, float("-inf"))
    log_probs = torch.log_softmax(scores, dim=-1)

    k = min(MAX_TOP_K, vocab)
    top_lp, top_ids = torch.topk(log_probs, k, dim=-1)
    probs = top_lp.exp()
    mass_before = torch.cumsum(probs, dim=-1) - probs
    rank = torch.arange(k, device=logits.device)[None, :]
    keep = (mass_before < top_p[:, None]) & (rank < top_k[:, None])
    keep[:, 0] = True
    if fused:
        return _fused_draws(
            log_probs, top_lp, top_ids, keep, seed, step, greedy, history, window, repeat_threshold,
        )
    noise = _gumbel(seed, step, 0, k)
    pick = torch.where(keep, top_lp + noise, torch.full_like(top_lp, float("-inf"))).argmax(dim=-1)
    token = top_ids.gather(1, pick[:, None]).squeeze(1)
    token = torch.where(greedy, top_ids[:, 0], token)

    width = history.shape[-1]
    recent = torch.arange(width, device=history.device)[None, :] >= width - window[:, None]
    repeats = ((history == token[:, None]) & recent).sum(dim=-1).to(repeat_threshold.dtype)
    redraw = (repeat_threshold > 0) & (repeats >= repeat_threshold)
    ids = torch.arange(vocab, device=logits.device)[None, :]
    masked = log_probs.masked_fill(ids == token[:, None], float("-inf"))
    second = (masked + _gumbel(seed, step, _REDRAW_SALT, vocab)).argmax(dim=-1)
    # no other id left to draw: keep the first
    second = torch.where(torch.isfinite(masked).any(dim=-1), second, token)
    return torch.where(redraw, second, token)
