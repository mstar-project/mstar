"""The sampled leg of a verify step: speculative sampling on filtered distributions.

Both models' distributions go through the transform a plain sampled step applies
to its target: temperature, top-k, then top-p over what is left, renormalised. A
draft ``d`` drawn from the filtered draft distribution ``p`` is kept with
probability ``min(1, q(d) / p(d))``, ``q`` the filtered target; the first rejected
position draws from ``(q - p)+`` instead, and a row that keeps all k drafts draws
its bonus token from ``q`` (Leviathan et al. 2023). Every emitted token then
follows ``q`` exactly, while a row accepts ``sum min(p, q)`` of the time, not the
``q(d)`` a deterministic draft gets. Any distribution the draft token really came
from is valid here, a point mass included: a request's first drafts are.

The filters run on each row's top `CANDIDATES` tokens, which hold nearly always
everything top-k and top-p keep. FlashInfer's vocabulary-wide renorms search the
whole row for each cut (about 0.5 ms a launch at 248k tokens, with k and p
together about 1.2); here a selection, then a filter held in registers. The
target must be exact, so a row whose kept set may run past the candidates takes
the renorms after all, which leave other rows as they are at the cost of one read.
A draft need not be: `Candidates.probs` is a valid distribution even then.

Draws are keyed by the request's seed, the row and a Philox word offset. Offsets
count 32-bit words of the row's stream, so two draws overlap unless the first
uses fewer words than the distance between their offsets (`words_per_step`).

These functions are the kernels on an accelerator; a torch path serves CPU
tensors, for tests that drive the step without one.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from mstar.engine.resources.sampler import utils as sampling

# Tokens a row's filters run on. Measured on chat and 1k-token prompts at temperature
# 0.7 and top-p 0.9, top-p keeps at most 30 and at the 99th percentile 18; covers a top-k
# of up to this many with no fallback. A power of two, as the filter's kernel prefers.
CANDIDATES = 64
# What a stored draft probability is scaled by. The accept test is ``u * p(d) <
# q(d)`` with u in (0, 1], so with p = q a draw of exactly 1 would reject a draft
# that nothing can replace: the residual is empty, and the kernel then returns an
# id it never set. The margin keeps ``q - p`` positive wherever q is the larger or
# they are equal, so a rejection always has a residual to draw from. It also
# covers a greedy row, whose q is a point mass: whatever a stale p holds, the
# argmax keeps a positive residual. The error is an ulp of the acceptance rate.
DRAFT_MARGIN = 1.0 - 2.0**-23


def words_per_step(num_draft_tokens: int) -> int:
    """Stream words a verify step of k drafts uses: one per draft judged, one
    for the residual or bonus draw, and the k draft draws for the next step."""
    return 2 * num_draft_tokens + 1


class Candidates(NamedTuple):
    """A filtered distribution as each row's top tokens, ``[rows, CANDIDATES]``."""
    ids: torch.Tensor       # int64
    probs: torch.Tensor     # float32, renormalised, 0 past what the filters keep
    # [rows] bool: every token the filters keep is among ``ids``, so ``probs`` is
    # the filtered distribution itself and not a truncation of it
    complete: torch.Tensor


def _keep_reference(probs, top_k, top_p, whole: float):
    """One row's `Candidates.probs` by the kernels' rule: a token stays while fewer
    than k tokens, then less than top_p of the renormalised mass, lie strictly
    above it. ``whole``: the mass top-p is a share of, where top-k is off."""
    k = int(top_k)
    if k < probs.numel():
        floor = probs.topk(k).values[-1]
        probs = torch.where(probs >= floor, probs, 0.0)
        whole = float(probs.sum())
    if float(top_p) < 1.0:
        values = probs.unique()
        above = torch.tensor([float(probs[probs > v].sum()) for v in values], dtype=torch.float32)
        floor = values[above < float(top_p) * whole].min()
        probs = torch.where(probs >= floor, probs, 0.0)
    return probs / probs.sum()


def _softmax_reference(logits, temperature):
    out = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
    for r in range(logits.shape[0]):
        x = logits[r].float()
        if temperature[r] == 0:
            out[r, x.argmax()] = 1.0
        else:
            out[r] = torch.softmax(x / temperature[r], dim=-1)
    return out


def _softmax(logits, temperature):
    if logits.device.type == "cpu":
        return _softmax_reference(logits, temperature)
    return sampling.fused_temperature_softmax(logits, temperature, include_greedy=True)


def _filter_candidates(values, top_k, top_p, mass):
    if values.device.type == "cpu":
        return torch.stack([
            _keep_reference(values[r], top_k[r], top_p[r], float(mass[r])) for r in range(values.shape[0])
        ])
    return sampling.filter_top_k_top_p(values, top_k, top_p)


def _candidates(probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> Candidates:
    vocab, width = probs.shape[1], CANDIDATES
    values, ids = torch.topk(probs, width, dim=-1)
    mass = values.sum(dim=-1)
    off = top_k >= vocab
    if values.device.type == "cpu":
        kept = _filter_candidates(values, top_k, top_p, torch.where(off, 1.0, mass))
    else:
        # The kernel takes top-p as a share of the mass it is given. With top-k off
        # that must be the whole row's, which is 1, not what the candidates hold.
        kept = _filter_candidates(values, top_k, torch.where(off, top_p / mass, top_p), None)
    # All candidates kept, or a top-k wider than them: the kept set may go on past them
    complete = ((kept > 0).sum(dim=-1) < width) & (off | (top_k <= width))
    return Candidates(ids, kept, complete | (width == vocab))


def filtered_candidates(
    logits: torch.Tensor,       # [rows, vocab]
    temperature: torch.Tensor,  # [rows]; 0: a point mass at the argmax
    top_k: torch.Tensor,        # [rows] int32; vocab or more: off
    top_p: torch.Tensor,        # [rows]; 1: off
) -> Candidates:
    """A row's filtered distribution on its top tokens: all of it unless
    ``complete`` says otherwise, and a distribution to draw from either way."""
    with _on_device(logits):
        return _candidates(_softmax(logits, temperature), top_k, top_p)


def filtered_probs(logits, temperature, top_k, top_p) -> torch.Tensor:
    """The ``[rows, vocab]`` float32 distribution a sampled row draws from,
    exactly, whatever the filters keep."""
    with _on_device(logits):
        probs = _softmax(logits, temperature)
        cand = _candidates(probs, top_k, top_p)
        dense = torch.zeros_like(probs).scatter_(1, cand.ids, cand.probs)
        if logits.device.type == "cpu":
            full = torch.stack([
                _keep_reference(probs[r], top_k[r], top_p[r], 1.0) for r in range(probs.shape[0])
            ])
        else:
            # 1 and the vocabulary are the renorms' fast paths: only rows that need them search
            full = sampling.top_k_first_filter(
                probs, torch.where(cand.complete, probs.shape[1], top_k), torch.where(cand.complete, 1.0, top_p),
            )
        return torch.where(cand.complete[:, None], dense, full)


class _on_device:
    """``torch.cuda.device`` for a tensor on an accelerator, nothing on the CPU."""

    def __init__(self, tensor: torch.Tensor):
        self._guard = torch.cuda.device(tensor.device) if tensor.device.type != "cpu" else None

    def __enter__(self):
        if self._guard is not None:
            self._guard.__enter__()

    def __exit__(self, *exc):
        if self._guard is not None:
            self._guard.__exit__(*exc)


def _generator(seed, row: int, offset) -> torch.Generator:
    return torch.Generator().manual_seed((int(seed) * 1_000_003 + row) * 1_000_033 + int(offset))


def draw(probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """One ``[rows]`` int64 index into each row of ``probs``, at its seed and offset."""
    if probs.device.type == "cpu":
        return torch.cat([
            torch.multinomial(probs[r], 1, generator=_generator(seed[r], r, offset[r]))
            for r in range(probs.shape[0])
        ])
    with _on_device(probs):
        return sampling._sampling_from_probs(probs, deterministic=True, seed=seed, offset=offset).to(torch.int64)


def _chain_reference(draft_probs, drafts, target_probs, seed, offset):
    b, k = drafts.shape
    tokens = torch.full((b, k + 1), -1, dtype=torch.int32, device=drafts.device)
    kept = torch.zeros(b, dtype=torch.int32, device=drafts.device)
    for r in range(b):
        gen = _generator(seed[r], r, offset[r])
        n = k
        for i in range(k):
            d = drafts[r, i]
            # (0, 1], as the kernel's uniform
            u = 1.0 - torch.rand((), generator=gen)
            if u * draft_probs[r, i, d] < target_probs[r, i, d]:
                tokens[r, i] = d
            else:
                n = i
                break
        residual = target_probs[r, n]
        if n < k:
            residual = (residual - draft_probs[r, n]).clamp_min(0.0)
        tokens[r, n] = torch.multinomial(residual, 1, generator=gen)[0]
        kept[r] = n
    return tokens, kept


def accept_chain(
    draft_ids: torch.Tensor,     # [rows, k, CANDIDATES], the ids of `draft_probs`
    draft_probs: torch.Tensor,   # [rows, k, CANDIDATES]
    drafts: torch.Tensor,        # [rows, k]
    target_probs: torch.Tensor,  # [rows, k + 1, vocab]
    seed: torch.Tensor,          # [rows]
    offset: torch.Tensor,        # [rows], k + 1 words of the stream are used
) -> tuple[torch.Tensor, torch.Tensor]:
    """Judge each row's drafts in order. Returns ``tokens`` ``[rows, k + 1]``,
    the kept drafts, then the residual or bonus draw, then -1; and how many
    drafts each row kept, ``[rows]`` int32."""
    b, k, _ = draft_ids.shape
    dense = torch.zeros(b, k, target_probs.shape[-1], dtype=torch.float32, device=target_probs.device)
    # added, not written: a point mass repeats its id with nothing on the rest
    dense.scatter_add_(2, draft_ids, draft_probs * DRAFT_MARGIN)
    if drafts.device.type == "cpu":
        return _chain_reference(dense, drafts, target_probs, seed, offset)
    with _on_device(drafts):
        tokens, _, kept = sampling._chain_speculative_sampling(
            dense, drafts.to(torch.int32).contiguous(), target_probs,
            deterministic=True, seed=seed, offset=offset,
        )
    return tokens, kept
