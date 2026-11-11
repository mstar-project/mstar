"""Verifying a block of draft tokens against the target (speculative sampling).

A verify row holds the last emitted token and ``k`` drafts, and the target's
logits after each. Filtered as the request samples (temperature, min-p,
top-k / top-p, in plain sampling's order), they give ``p_j``. The drafters
here draft by argmax, so a draft is a point mass and:

- draft ``d_j`` is accepted with probability ``p_j(d_j)``, while every draft
  before it was;
- at the first rejection the bonus token comes from ``p_j`` with ``d_j``
  removed and renormalised; after ``k`` acceptances, from ``p_{k+1}``.

The emitted tokens are then distributed as sampling the target one token at a
time. A greedy row's ``p`` is a one-hot at the argmax, so it accepts drafts
while they equal the target's top token and then emits that token.

Each row draws from a counter hash of its own seed and offset (`_uniforms`),
so a request's numbers are the same in any batch, on any device, and in a
captured graph.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from mstar.engine.resources.sampler.utils import (
    FUSED_FILTER_MAX_VOCAB,
    FilterOrder,
    apply_min_p,
    filter_top_k_top_p,
    fused_temperature_softmax,
)

# keys verify's numbers apart from any other stream drawn off the same seed
_VERIFY_SALT = 0x5EC0DE5
# FlashInfer's top-p for a row whose renorm is never read (`renorm_top_p`): any p in (0, 1) would do, and p = 1
# is the kernel's slow case: 3-4x its time on a greedy row, for a sampled row at Qwen's vocabularies
_UNREAD_TOP_P = 1e-6


def verify_stride(num_draft_tokens: int) -> int:
    """Numbers one verify row draws, which its offset moves by every step: one
    per draft, then one for the bonus token. A constant, so every rank and
    both modes advance alike, and a rejection's draw never meets the next
    step's."""
    return num_draft_tokens + 1


def filtered_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor | None = None,
    order: FilterOrder | None = None,
) -> torch.Tensor:
    """``[R, V]`` float32: each row's distribution as plain sampling draws
    from it. The temperature softmax (a one-hot at the argmax for a greedy
    row), then min-p, then top-k and top-p. Capturable."""
    if logits.is_cuda:
        with torch.cuda.device(logits.device):
            probs = fused_temperature_softmax(logits, temperature, include_greedy=True)
            if min_p is not None:
                probs = apply_min_p(probs, min_p)
            return filter_top_k_top_p(probs, top_k, top_p, order)
    if order is not None:
        raise ValueError("top_p_first and top_p_min_keep are CUDA-only")
    return _filtered_probs_torch(logits, temperature, top_k, top_p, min_p)


class _TwoFilters(NamedTuple):
    """A block's filtered distributions as two candidates and a choice: ``first``
    the top-k renorm, ``then`` its top-p renorm, ``use_then`` ``[n, k + 1]``
    where top-p applies (p < 1). Read at the drafts and at one row per request,
    so the choice is never made over the whole vocabulary."""

    first: torch.Tensor
    then: torch.Tensor
    use_then: torch.Tensor


def verify_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor | None = None,
    order: FilterOrder | None = None,
) -> "torch.Tensor | _TwoFilters":
    """`filtered_probs` for a verify step's ``[R, V]`` rows. For a vocabulary past
    the fused filter (FlashInfer's renorms, top-k first), the two renorms
    themselves: a top-p row's filtered distribution is the top-p renorm of its
    top-k renorm, which keeps the same prefix of tokens `filter_top_k_top_p`'s
    count does, so neither the counts nor a third renorm are needed."""
    if not (logits.is_cuda and order is None and logits.shape[-1] > FUSED_FILTER_MAX_VOCAB):
        return filtered_probs(logits, temperature, top_k, top_p, min_p, order)
    import flashinfer

    with torch.cuda.device(logits.device):
        probs = fused_temperature_softmax(logits, temperature, include_greedy=True)
        if min_p is not None:
            probs = apply_min_p(probs, min_p)
        k = torch.where(top_k > 0, top_k, probs.shape[-1]).to(torch.int32)
        first = flashinfer.sampling.top_k_renorm_probs(probs, k)
        then = flashinfer.sampling.top_p_renorm_probs(first, renorm_top_p(top_p))
    return _TwoFilters(first, then, top_p < 1.0)


def renorm_top_p(top_p: torch.Tensor) -> torch.Tensor:
    """The top-p each row's top-p renorm runs at: its own where `_TwoFilters` reads the renorm (p < 1), and
    `_UNREAD_TOP_P` where it reads ``first`` instead (p = 1 removes nothing)."""
    return torch.where(top_p < 1.0, top_p, _UNREAD_TOP_P)


def chain_verify(
    probs: torch.Tensor, drafts: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify each row's ``drafts`` ``[n, k]`` against ``probs`` ``[n, k + 1, V]``
    (`filtered_probs`), drawing at each row's ``seed`` and ``offset``.

    Returns the tokens ``[n, k + 1]`` and the drafts accepted per row ``[n]``.
    Entries past ``accepted`` are valid ids with no meaning. Capturable."""
    return _chain_verify(probs, drafts, seed, offset)


def chain_verify_rows(
    dist: "torch.Tensor | _TwoFilters", n: int, drafts: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`chain_verify` over `verify_probs`' ``[n * (k + 1), ...]`` rows."""
    if isinstance(dist, torch.Tensor):
        return _chain_verify(dist.view(n, -1, dist.shape[-1]), drafts, seed, offset)
    vocab = dist.first.shape[-1]
    return _chain_verify(
        _TwoFilters(dist.first.view(n, -1, vocab), dist.then.view(n, -1, vocab), dist.use_then.view(n, -1)),
        drafts, seed, offset,
    )


def _chain_verify(dist, drafts, seed, offset):
    two = isinstance(dist, _TwoFilters)
    probs = dist.first if two else dist
    n, block, _ = probs.shape
    k = block - 1
    drafts = drafts.long()
    uniforms = _uniforms(seed, offset, k + 1, probs.dtype)
    rows = torch.arange(n, device=probs.device)
    p_d = probs[:, :k].gather(2, drafts[..., None]).squeeze(-1)
    if two:
        p_d = torch.where(dist.use_then[:, :k], dist.then[:, :k].gather(2, drafts[..., None]).squeeze(-1), p_d)
    # FlashInfer's test, u < p, on numbers in [0, 1): a draft with p = 1 always stays
    accepted = (uniforms[:, :k] < p_d).long().cumprod(1).sum(1)
    p_at = probs[rows, accepted]
    if two:
        p_at = torch.where(dist.use_then[rows, accepted][:, None], dist.then[rows, accepted], p_at)
    refused = accepted < k
    at = accepted.clamp(max=k - 1)
    q_at = torch.zeros_like(p_at).scatter_(1, drafts[rows, at][:, None], 1.0)
    residual = (p_at - torch.where(refused[:, None], q_at, 0.0)).clamp_min(0.0)
    last = _inverse_cdf(residual, uniforms[:, k])
    # an empty residual: the refused draft held all of p (rounding), or the bonus row's p is empty
    fallback = torch.where(refused, drafts[rows, at], p_at.argmax(-1))
    last = torch.where(residual.sum(-1) > 0, last, fallback)
    keep = torch.arange(block, device=probs.device)[None, :] < accepted[:, None]
    tokens = torch.where(keep, F.pad(drafts, (0, 1)), last[:, None])
    return tokens, accepted


def _filtered_probs_torch(logits, temperature, top_k, top_p, min_p):
    vocab = logits.shape[-1]
    logits = logits.float()
    greedy = (temperature <= 0)[:, None]
    probs = torch.softmax(logits / torch.where(greedy, 1.0, temperature[:, None]), -1)
    probs = torch.where(greedy, F.one_hot(logits.argmax(-1), vocab).to(probs.dtype), probs)
    if min_p is not None:
        probs = apply_min_p(probs, min_p)
    # top-k, then top-p over the renormalised top-k, as FlashInfer filters: a
    # token stays while the mass ranked above it is below p
    ranked, order = probs.sort(-1, descending=True)
    k = torch.where(top_k > 0, top_k, vocab).clamp(max=vocab)
    ranked = torch.where(torch.arange(vocab, device=logits.device)[None, :] < k[:, None], ranked, 0.0)
    ranked = ranked / ranked.sum(-1, keepdim=True)
    keep = ((ranked.cumsum(-1) - ranked) < top_p[:, None]) | (top_p[:, None] >= 1.0)
    ranked = torch.where(keep, ranked, 0.0)
    out = torch.zeros_like(probs).scatter(-1, order, ranked)
    return out / out.sum(-1, keepdim=True)


def _inverse_cdf(weights: torch.Tensor, u: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """The token each row's ``u`` in [0, 1) picks from its unnormalised
    ``weights``, by blocks of ``block`` tokens: the block where ``u`` of the
    row's mass falls, then the token in it."""
    n, vocab = weights.shape
    pad = (-vocab) % block
    blocks = (F.pad(weights, (0, pad)) if pad else weights).view(n, -1, block)
    sums = blocks.sum(-1)
    upto = sums.cumsum(-1)
    target = u.to(upto.dtype) * upto[:, -1]
    at = torch.searchsorted(upto, target[:, None], right=True).squeeze(1)
    # never past the last block, or token, with weight, where rounding could put u * total
    at = torch.minimum(at, _last_held(sums))
    rows = torch.arange(n, device=weights.device)
    before = torch.where(at > 0, upto[rows, (at - 1).clamp(min=0)], 0.0)
    inner = blocks[rows, at]
    picked = torch.searchsorted(inner.cumsum(-1), (target - before)[:, None], right=True).squeeze(1)
    return at * block + torch.minimum(picked, _last_held(inner))


def _last_held(weights: torch.Tensor) -> torch.Tensor:
    """Each row's last index with weight (its last index when it has none)."""
    held = weights > 0
    return weights.shape[-1] - 1 - held.flip(-1).long().argmax(-1)


_M32 = 0xFFFFFFFF


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """``x * c`` mod 2^32 for ``x`` below 2^32, in int64 without overflow."""
    return (x * (c & 0xFFFF) + ((x * (c >> 16)) & 0xFFFF) * 65536) & _M32


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (lowbias32) of ``x`` below 2^32."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def _uniforms(seed: torch.Tensor, offset: torch.Tensor, width: int, dtype) -> torch.Tensor:
    """``[n, width]`` numbers in [0, 1): row ``r``'s are the ``offset[r]``-th
    onward of a stream keyed by ``seed[r]`` alone, in device ops."""
    seed, offset = seed.long(), offset.long()
    key = _hash32((seed & _M32) ^ _hash32(((seed >> 32) & _M32) ^ _VERIFY_SALT))
    counter = offset[:, None] + torch.arange(width, device=offset.device)
    bits = _hash32(_hash32(key[:, None] ^ (counter & _M32)) ^ ((counter >> 32) & _M32) ^ 0x85EBCA6B)
    # 24 bits: exact in float32, and below 1
    return ((bits >> 8).to(torch.float64) * 2.0 ** -24).to(dtype)
