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

The filters run on each row's top `CANDIDATES` tokens (see `candidates`), which
hold nearly always everything top-k and top-p keep. The target must be exact, so a
row whose kept set runs past them is filtered over its whole vocabulary. A draft
need not be: `Candidates.probs` is a valid distribution even then.

Draws are keyed by the request's seed, the row and a Philox word offset. Offsets
count 32-bit words of the row's stream, so two draws overlap unless the first
uses fewer words than the distance between their offsets (`words_per_step`).

These functions are the kernels on an accelerator; a torch path serves CPU
tensors, for tests that drive the step without one.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch

from mstar.engine.resources.sampler import utils as sampling
from mstar.engine.resources.spec_decode.candidates import FILTER_PAST_CANDIDATES, FILTERED_CANDIDATES

# Tokens a row's filters run on. On chat and 1k-token prompts, Qwen3.5 at temperature
# 0.7 and top-p 0.9 keeps at most 30 and at the 99th percentile 18; at 1.0 and 0.95 the
# 99th percentile is 64, so about one row in a hundred goes past them there.
CANDIDATES = 64
# What a stored draft probability is scaled by. The accept test is ``u * p(d) <
# q(d)`` with u in (0, 1], so with p = q a draw of exactly 1 would reject a draft
# that nothing can replace: the residual is empty, and the kernel then returns an
# id it never set. The margin keeps ``q - p`` positive wherever q is the larger or
# they are equal, so a rejection always has a residual to draw from. It also
# covers a greedy row, whose q is a point mass: whatever a stale p holds, the
# argmax keeps a positive residual. The error is an ulp of the acceptance rate.
DRAFT_MARGIN = 1.0 - 2.0**-23


def _on_device(tensor: torch.Tensor):
    return nullcontext() if tensor.device.type == "cpu" else torch.cuda.device(tensor.device)


def words_per_step(num_draft_tokens: int) -> int:
    """Stream words a verify step of k drafts uses: one per draft judged, one
    for the residual or bonus draw, and the k draft draws for the next step."""
    return 2 * num_draft_tokens + 1


class Candidates(NamedTuple):
    """A filtered distribution as each row's top tokens, ``[rows, CANDIDATES]``."""
    ids: torch.Tensor       # int64, 0 past what the filters keep
    probs: torch.Tensor     # float32, renormalised, 0 past what the filters keep
    # [rows] int32: nonzero where every token the filters keep is among ``ids``, so
    # ``probs`` is the filtered distribution itself and not a truncation of it
    complete: torch.Tensor


def filtered_candidates(
    logits: torch.Tensor,       # [rows, vocab]
    temperature: torch.Tensor,  # [rows]; 0: a point mass at the argmax
    top_k: torch.Tensor,        # [rows] int32; vocab or more: off
    top_p: torch.Tensor,        # [rows]; 1: off
) -> Candidates:
    """A row's filtered distribution on its top tokens: all of it unless
    ``complete`` says otherwise, and a distribution to draw from either way."""
    rows, device = logits.shape[0], logits.device
    ids = torch.empty(rows, CANDIDATES, dtype=torch.int64, device=device)
    probs = torch.empty(rows, CANDIDATES, dtype=torch.float32, device=device)
    complete = torch.empty(rows, dtype=torch.int32, device=device)
    FILTERED_CANDIDATES(logits, temperature, top_k, top_p, ids, probs, complete)
    return Candidates(ids, probs, complete)


def filtered_probs(logits, temperature, top_k, top_p) -> torch.Tensor:
    """The ``[rows, vocab]`` float32 distribution a sampled row draws from,
    exactly, whatever the filters keep."""
    cand = filtered_candidates(logits, temperature, top_k, top_p)
    probs = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
    # added, not written: ids repeat, with nothing on the repeats
    probs.scatter_add_(1, cand.ids, cand.probs)
    FILTER_PAST_CANDIDATES(logits, temperature, top_k, top_p, cand.complete, probs)
    return probs


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
