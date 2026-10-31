"""Verification of a draft block by rejection sampling (speculative sampling).

A verify row holds the last emitted token and ``k`` drafts, and the target's logits after
each. Filtered as the request samples (temperature, min-p, top-k / top-p), they give ``p_i``,
and draft ``d_i`` came from ``q_i`` (a point mass at ``d_i`` for a drafter that gives tokens
only):

- draft ``i`` stays with probability ``min(1, p_i(d_i) / q_i(d_i))``;
- the first refused one is replaced by a draw from ``max(0, p_i - q_i)``, normalised;
- with every draft kept, one more token comes from ``p_k``.

The tokens a row emits are then distributed as sampling the target one token at a time. One
torch version of the rule runs on every device, each row drawing its numbers from its own seed
and offset (``_uniforms``), so a request's draws are the same in any batch: FlashInfer's chain
and sampling kernels read the first row's seed and offset and the row's place in the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.engine.resources.sampler.utils import (
    FilterOrder,
    apply_min_p,
    filter_top_k_top_p,
    fused_temperature_softmax,
)


def verify_stride(drafts: int) -> int:
    """Numbers one verify row draws, which its RNG offset moves by: one per draft position,
    then one for the replacement or the bonus token."""
    return drafts + 1


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
    ``accepted`` are valid ids with no meaning. Capturable."""
    n, block, vocab = probs.shape
    k = block - 1
    drafts = drafts.long()
    accepted, last = _chain_verify_torch(probs, drafts, draft_probs, _uniforms(seed, offset, k + 1, probs.dtype))
    keep = torch.arange(block, device=probs.device)[None, :] < accepted[:, None]
    tokens = torch.where(keep, F.pad(drafts, (0, 1)), last[:, None])
    return tokens, accepted


def draw(probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """One token per row of ``probs`` ``[n, V]``, at each row's ``seed`` and ``offset`` (one
    number). Capturable."""
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
    rows = torch.arange(n, device=probs.device)
    p_d = probs[:, :k].gather(2, drafts[..., None]).squeeze(-1)
    q_d = torch.ones_like(p_d) if draft_probs is None else draft_probs.gather(2, drafts[..., None]).squeeze(-1)
    # FlashInfer's test, u * q < p, on numbers in [0, 1)
    accepted = (uniforms[:, :k] * q_d < p_d).long().cumprod(1).sum(1)
    p_at = probs[rows, accepted]
    refused = accepted < k
    at = accepted.clamp(max=k - 1)
    if draft_probs is None:
        q_at = probs.new_zeros(n, vocab).scatter_(1, drafts[rows, at][:, None], 1.0)
    else:
        q_at = draft_probs[rows, at].to(probs.dtype)
    residual = (p_at - torch.where(refused[:, None], q_at, 0.0)).clamp_min(0.0)
    total = residual.sum(-1)
    last = _inverse_cdf(residual, uniforms[:, k])
    return accepted, torch.where(total > 0, last, _fallback(drafts, accepted, p_at))


def _inverse_cdf(weights: torch.Tensor, u: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """The token each row's ``u`` in [0, 1) picks from its (unnormalised) ``weights``, by blocks
    of ``block`` tokens: the block where ``u`` of the row's mass falls, then the token in it. A
    scan over a whole vocabulary runs one thread block per row, a few hundred microseconds."""
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
    """``[n, width]`` numbers in [0, 1), row ``r``'s the ``offset[r]``-th onward of a stream
    keyed by ``seed[r]`` alone: the same request draws the same numbers whatever batch it rides
    in, on any device. A counter hash in device ops, so a captured step draws in its graph."""
    seed, offset = seed.long(), offset.long()
    key = _hash32((seed & _M32) ^ _hash32(((seed >> 32) & _M32) ^ 0x9E3779B9))
    counter = offset[:, None] + torch.arange(width, device=offset.device)
    bits = _hash32(_hash32(key[:, None] ^ (counter & _M32)) ^ ((counter >> 32) & _M32) ^ 0x85EBCA6B)
    # 24 bits: exact in float32, and below 1
    return ((bits >> 8).to(torch.float64) * 2.0 ** -24).to(dtype)
