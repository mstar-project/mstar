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

Draws are keyed by the request's seed, the row and a Philox word offset. Offsets
count 32-bit words of the row's stream, so two draws overlap unless the first
uses fewer words than the distance between their offsets (`words_per_step`).

These functions are the kernels on an accelerator; a torch path serves CPU
tensors, for tests that drive the step without one.
"""

from __future__ import annotations

import torch

from mstar.engine.resources.sampler import utils as sampling

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


def _filtered_reference(logits, temperature, top_k, top_p) -> torch.Tensor:
    rows, vocab = logits.shape
    out = torch.zeros(rows, vocab, dtype=torch.float32, device=logits.device)
    for r in range(rows):
        x = logits[r].float()
        if temperature[r] == 0:
            out[r, x.argmax()] = 1.0
            continue
        probs = torch.softmax(x / temperature[r], dim=-1)
        # the kernels' rule: a token stays while fewer than k tokens, then less
        # than top_p of the renormalised mass, lie strictly above it
        k = int(top_k[r])
        if 0 < k < vocab:
            floor = probs.topk(k).values[-1]
            probs = torch.where(probs >= floor, probs, 0.0)
            probs = probs / probs.sum()
        if top_p[r] < 1.0:
            above = torch.tensor([float(probs[probs > v].sum()) for v in probs.unique()], dtype=torch.float32)
            floor = probs.unique()[above < top_p[r]].min()
            probs = torch.where(probs >= floor, probs, 0.0)
            probs = probs / probs.sum()
        out[r] = probs
    return out


def filtered_probs(
    logits: torch.Tensor,       # [rows, vocab]
    temperature: torch.Tensor,  # [rows]; 0: a point mass at the argmax
    top_k: torch.Tensor,        # [rows] int32; vocab or more: off
    top_p: torch.Tensor,        # [rows]; 1: off
) -> torch.Tensor:
    """The ``[rows, vocab]`` float32 distribution a sampled row draws from."""
    if logits.device.type == "cpu":
        return _filtered_reference(logits, temperature, top_k, top_p)
    with torch.cuda.device(logits.device):
        probs = sampling.fused_temperature_softmax(logits, temperature, include_greedy=True)
        return sampling.top_k_first_filter(probs, top_k, top_p)


def _generator(seed, row: int, offset) -> torch.Generator:
    return torch.Generator().manual_seed((int(seed) * 1_000_003 + row) * 1_000_033 + int(offset))


def draw(probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """One ``[rows]`` int64 draw per row of ``probs``, at its seed and offset."""
    if probs.device.type == "cpu":
        return torch.cat([
            torch.multinomial(probs[r], 1, generator=_generator(seed[r], r, offset[r]))
            for r in range(probs.shape[0])
        ])
    with torch.cuda.device(probs.device):
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
    draft_probs: torch.Tensor,   # [rows, k, vocab]
    drafts: torch.Tensor,        # [rows, k]
    target_probs: torch.Tensor,  # [rows, k + 1, vocab]
    seed: torch.Tensor,          # [rows]
    offset: torch.Tensor,        # [rows], k + 1 words of the stream are used
) -> tuple[torch.Tensor, torch.Tensor]:
    """Judge each row's drafts in order. Returns ``tokens`` ``[rows, k + 1]``,
    the kept drafts, then the residual or bonus draw, then -1; and how many
    drafts each row kept, ``[rows]`` int32."""
    if drafts.device.type == "cpu":
        return _chain_reference(draft_probs, drafts, target_probs, seed, offset)
    with torch.cuda.device(drafts.device):
        tokens, _, kept = sampling._chain_speculative_sampling(
            draft_probs, drafts.to(torch.int32).contiguous(), target_probs,
            deterministic=True, seed=seed, offset=offset,
        )
    return tokens, kept
