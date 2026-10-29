"""Verification of a draft block by rejection sampling (speculative sampling).

A verify row holds the last emitted token and ``k`` drafts, and the target's logits after
each. Filtered as the request samples (temperature, min-p, top-k / top-p), they give ``p_i``,
and draft ``d_i`` came from ``q_i`` (a point mass at ``d_i`` for a drafter that gives tokens
only):

- draft ``i`` stays with probability ``min(1, p_i(d_i) / q_i(d_i))``;
- the first refused one is replaced by a draw from ``max(0, p_i - q_i)``, normalised;
- with every draft kept, one more token comes from ``p_k``.

The tokens a row emits are then distributed as sampling the target one token at a time. On
CUDA this is FlashInfer's chain kernel; elsewhere a torch version of the same rule, for the
CPU tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.engine.resources.sampler.config import derive_seed
from mstar.engine.resources.sampler.utils import (
    FilterOrder,
    apply_min_p,
    filter_top_k_top_p,
    fused_temperature_softmax,
)


def verify_stride(drafts: int) -> int:
    """Philox numbers one verify row draws at most, which its RNG offset moves by: FlashInfer's
    chain kernel draws one per draft position whether or not an earlier one was refused, the
    refused position a second time (for its acceptance count), then the replacement or the
    bonus token."""
    return drafts + 2


def filtered_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor | None = None,
    order: FilterOrder | None = None,
) -> torch.Tensor:
    """``[R, V]`` float32: each row's distribution as the sampler draws from it. The
    temperature softmax (a one-hot at the argmax for a greedy row), then min-p, then top-k and
    top-p, renormalised. Capturable."""
    if logits.is_cuda:
        with torch.cuda.device(logits.device):
            probs = fused_temperature_softmax(logits, temperature, include_greedy=True)
            if min_p is not None:
                probs = apply_min_p(probs, min_p)
            return filter_top_k_top_p(probs, top_k, top_p, order)
    if order is not None:
        raise ValueError("top_p_first / top_p_min_keep are CUDA-only")
    return _filtered_probs_torch(logits, temperature, top_k, top_p, min_p)


def chain_verify(
    probs: torch.Tensor,
    drafts: torch.Tensor,
    draft_probs: torch.Tensor | None,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify each row's drafts against the target's ``probs`` ``[n, k + 1, V]``
    (``filtered_probs``). ``drafts`` ``[n, k]`` came from ``draft_probs`` ``[n, k, V]``, or are
    point masses when it is None. Draws at each row's ``seed`` and ``offset``.

    Returns the tokens ``[n, k + 1]`` and the drafts kept per row ``[n]``: a row emits
    ``tokens[:accepted + 1]``, its kept drafts then the replacement or bonus token. Entries past
    ``accepted`` are valid ids with no meaning. Capturable on CUDA."""
    n, block, vocab = probs.shape
    k = block - 1
    drafts = drafts.long()
    if probs.is_cuda:
        import flashinfer

        with torch.cuda.device(probs.device):
            if draft_probs is None:
                draft_probs = probs.new_zeros(n, k, vocab).scatter_(2, drafts[..., None], 1.0)
            out, _, accepted = flashinfer.sampling.chain_speculative_sampling(
                draft_probs.float().contiguous(), drafts.int(), probs.contiguous(),
                deterministic=True, seed=seed, offset=offset,
            )
        accepted = accepted.long()
        at = accepted[:, None]
        p_at = probs.gather(1, at[..., None].expand(n, 1, vocab)).squeeze(1)
        last = out.long().gather(1, at)
        # A draft refused at u = 1 (curand's range is (0, 1]) where the target holds all of
        # q's mass on it leaves an empty residual, and the kernel then returns an id it never
        # set. Such a draft is a sample of p: emit it.
        drawn = (last >= 0) & (last < vocab) & (p_at.gather(1, last.clamp(0, vocab - 1)) > 0)
        last = torch.where(drawn, last, _fallback(drafts, accepted, p_at)[:, None]).squeeze(1)
    else:
        uniforms = _uniforms(seed, offset, k + 1, probs.dtype)
        accepted, last = _chain_verify_torch(probs, drafts, draft_probs, uniforms)
    keep = torch.arange(block, device=probs.device)[None, :] < accepted[:, None]
    tokens = torch.where(keep, F.pad(drafts, (0, 1)), last[:, None])
    return tokens, accepted


def draw(probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """One token per row of ``probs`` ``[n, V]``, at each row's ``seed`` and ``offset`` (one
    number). Capturable on CUDA."""
    if probs.is_cuda:
        import flashinfer

        with torch.cuda.device(probs.device):
            return flashinfer.sampling.sampling_from_probs(
                probs, deterministic=True, seed=seed, offset=offset,
            ).long()
    return _inverse_cdf(probs, _uniforms(seed, offset, 1, probs.dtype)[:, 0])


def _fallback(drafts: torch.Tensor, accepted: torch.Tensor, p_at: torch.Tensor) -> torch.Tensor:
    """The token a row emits at ``accepted`` when its residual is empty: the refused draft
    (the residual is empty only where ``p`` equals ``q``), or ``p``'s argmax at the bonus."""
    k = drafts.shape[1]
    draft = drafts.gather(1, accepted.clamp(max=k - 1)[:, None]).squeeze(1)
    return torch.where(accepted < k, draft, p_at.argmax(-1))


def _filtered_probs_torch(logits, temperature, top_k, top_p, min_p):
    vocab = logits.shape[-1]
    logits = logits.float()
    greedy = (temperature <= 0)[:, None]
    probs = torch.softmax(logits / torch.where(greedy, 1.0, temperature[:, None]), -1)
    probs = torch.where(greedy, F.one_hot(logits.argmax(-1), vocab).to(probs.dtype), probs)
    if min_p is not None:
        probs = apply_min_p(probs, min_p)
    # top-k, then top-p over the renormalised top-k, as FlashInfer filters: a token stays
    # while the mass ranked above it is below p
    ranked, order = probs.sort(-1, descending=True)
    k = torch.where(top_k > 0, top_k, vocab).clamp(max=vocab)
    ranked = torch.where(torch.arange(vocab)[None, :] < k[:, None], ranked, 0.0)
    ranked = ranked / ranked.sum(-1, keepdim=True)
    keep = ((ranked.cumsum(-1) - ranked) < top_p[:, None]) | (top_p[:, None] >= 1.0)
    ranked = torch.where(keep, ranked, 0.0)
    out = torch.zeros_like(probs).scatter(-1, order, ranked)
    return out / out.sum(-1, keepdim=True)


def _chain_verify_torch(probs, drafts, draft_probs, uniforms):
    n, block, vocab = probs.shape
    k = block - 1
    rows = torch.arange(n)
    p_d = probs[:, :k].gather(2, drafts[..., None]).squeeze(-1)
    q_d = torch.ones_like(p_d) if draft_probs is None else draft_probs.gather(2, drafts[..., None]).squeeze(-1)
    # FlashInfer's test, u * q < p, on numbers in [0, 1)
    accepted = (uniforms[:, :k] * q_d < p_d).long().cumprod(1).sum(1)
    p_at = probs[rows, accepted]
    refused = accepted < k
    at = accepted.clamp(max=k - 1)
    if draft_probs is None:
        q_at = F.one_hot(drafts[rows, at], vocab).to(probs.dtype)
    else:
        q_at = draft_probs[rows, at].to(probs.dtype)
    residual = (p_at - torch.where(refused[:, None], q_at, 0.0)).clamp_min(0.0)
    total = residual.sum(-1)
    last = _inverse_cdf(residual, uniforms[:, k])
    return accepted, torch.where(total > 0, last, _fallback(drafts, accepted, p_at))


def _inverse_cdf(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The token each row's ``u`` in [0, 1) picks from its (unnormalised) ``weights``."""
    cdf = weights.cumsum(-1)
    picked = torch.searchsorted(cdf, (u.to(cdf.dtype) * cdf[:, -1])[:, None], right=True).squeeze(1)
    # never past the last token with weight, where rounding could put u * total
    held = weights > 0
    last_held = weights.shape[-1] - 1 - held.flip(-1).long().argmax(-1)
    return torch.minimum(picked, last_held)


def _uniforms(seed: torch.Tensor, offset: torch.Tensor, width: int, dtype) -> torch.Tensor:
    """``[n, width]`` numbers in [0, 1), each row's from its own generator at (seed, offset):
    the same request draws the same numbers whatever batch it rides in."""
    out = torch.empty(len(seed), width, dtype=torch.float64)
    for row, (s, o) in enumerate(zip(seed.tolist(), offset.tolist(), strict=True)):
        gen = torch.Generator().manual_seed(derive_seed(s, f"draw:{o}"))
        out[row] = torch.rand(width, generator=gen, dtype=torch.float64)
    return out.to(dtype)
