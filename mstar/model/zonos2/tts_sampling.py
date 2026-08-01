"""Multi-codebook TTS sampling for Zonos2.

This is a port of ``../ZONOS2/python/zonos2/tts/sampler.py``. The reference
``sample_tts`` returns Python lists, which forces a device sync.
:func:`sample_frame` returns tensors instead. The forward of the LLM submodule
calls it inside the CUDA graph, with no ``.tolist()`` sync on the GPU thread.
It maps per-codebook logits ``(B, C, V)`` to frames ``(B, C + 1)``. Each frame
holds the sampled audio codes and a text placeholder. One call handles ``B``
requests.

A stateless RNG keeps the result reproducible under batching. The last draw is
a Gumbel-max over noise. :func:`_deterministic_uniform` keys that noise only on
``(seed, step, codebook, vocab)``, not on the batch position of the request. A
request therefore draws the same frame at a given step, whatever other requests
share its batch. A stateful ``torch.Generator`` for each request cannot do
this, because it becomes position-dependent once the code vectorises it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TTSSamplingParams:
    """Sampling parameters for one request. The defaults match the reference."""

    temperature: float = 1.15
    topk: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int = 1024
    ignore_eos: bool = False
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    # The repetition penalty applies to codebooks 0 to repetition_codebooks - 1.
    # A negative value applies it to all codebooks.
    repetition_codebooks: int = 8
    seed: int | None = None


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Apply a nucleus (top-p) filter to a probability distribution."""
    if p <= 0.0 or p >= 1.0:
        return probs
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    probs = probs.scatter(-1, probs_idx, probs_sort)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Apply a min-p filter. Drop the tokens below ``min_p * max_prob``."""
    if min_p <= 0.0:
        return probs
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    probs = probs.masked_fill(probs < (min_p * top_probs), 0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def apply_repetition_penalty(
    logits: torch.Tensor,
    repetition_token_ids: torch.Tensor | None,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply the repetition penalty to each codebook.

    ``repetition_token_ids`` is ``(B, C, W)``: the recent token ids of each
    codebook. The function ignores a token id of ``-1`` or one out of range. To
    exclude a codebook from the penalty, set its ids to ``-1``.
    """
    if repetition_token_ids is None or repetition_penalty == 1.0:
        return logits
    if repetition_token_ids.numel() == 0:
        return logits

    B, C, V = logits.shape
    safe_ids = repetition_token_ids.clamp(min=0, max=V - 1).long()
    valid = (repetition_token_ids >= 0) & (repetition_token_ids < V)

    counts = torch.zeros((B, C, V), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(-1, safe_ids, valid.to(torch.int32))
    repeated = counts > 0

    penalty = max(repetition_penalty, 1.0)
    adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(repeated, adjusted, logits)


_M32 = 0xFFFFFFFF


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    """Apply the MurmurHash3 ``fmix32`` finalizer to uint32 values in int64.

    Every value stays non-negative and less than ``2**32``. The only exception
    is the transient multiply. Its overflow past int64 wraps two's-complement,
    and the code masks it back to 32 bits immediately. The result therefore
    agrees with the uint32 reference, and the ``>>`` shifts act as logical
    shifts.
    """
    h = h & _M32
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _M32
    h = h ^ (h >> 15)
    return h & _M32


def _deterministic_uniform(
    B: int, C: int, V: int,
    seed: int, steps: torch.Tensor,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return reproducible ``U[0, 1)`` noise of shape ``(B, C, V)``.

    A counter-based hash keys the noise only on ``(seed, step, codebook,
    vocab)``. It does not use the batch position. The noise for request ``b`` at
    ``steps[b]`` is therefore the same alone or in any batch. ``steps`` is the
    step index of each request, of shape ``(B,)``.
    """
    v = torch.arange(V, device=device, dtype=torch.int64).view(1, 1, V)
    c = torch.arange(C, device=device, dtype=torch.int64).view(1, C, 1)
    s = steps.to(device=device, dtype=torch.int64).view(B, 1, 1)
    base = int(seed) & _M32
    # The chained fmix32 rounds mix every field into the result.
    h = (v * 0x27D4EB2F) & _M32
    h = _fmix32(h ^ (c * 0x85EBCA77))
    h = _fmix32(h ^ (s * 0xC2B2AE3D))
    h = _fmix32(h ^ base)
    return (h.to(torch.float64) / 4294967296.0).to(dtype)


def sample_frame(
    logits: torch.Tensor,
    params: TTSSamplingParams,
    repetition_token_ids: torch.Tensor | None = None,
    text_placeholder: int = 0,
    seed: int | None = None,
    steps: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Sample one frame for each request from the per-codebook logits.

    Args:
        logits: the per-codebook logits ``(B, C, V)`` of the current step.
        params: the sampling parameters, shared across the batch.
        repetition_token_ids: the recent tokens ``(B, C, W)``, or None. A ``-1``
            marks a padded or ignored slot.
        text_placeholder: the value to write into the appended text column.
        seed: the base RNG seed, shared across the batch. ``None`` uses the
            global RNG, which is not reproducible. This matches a request with
            no seed.
        steps: the step index of each request, of shape ``(B,)``. An int or
            ``None`` maps to 0. With ``seed`` set, ``(seed, step)`` fully
            determines the draw of a request, whatever its batch position.
            Batched sampling is therefore reproducible for each request.

    Returns:
        The int64 frames ``(B, C + 1)``: ``[cb0, ..., cb_{C-1},
        text_placeholder]``.
    """
    B, C, V = logits.shape
    device = logits.device

    logits = apply_repetition_penalty(
        logits, repetition_token_ids, params.repetition_penalty
    )

    if params.temperature <= 0:
        next_ids = torch.argmax(logits, dim=-1)  # (B, C)
    else:
        logits = logits / max(params.temperature, 1e-8)

        top_k = int(params.topk)
        if 0 < top_k < V:
            values, _ = torch.topk(logits, top_k, dim=-1)
            kth = values[..., -1:].clone()
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        if 0.0 < params.top_p < 1.0:
            probs = apply_top_p(probs, params.top_p)
        if params.min_p > 0.0:
            probs = apply_min_p(probs, params.min_p)

        # Reproducible Gumbel-max. ``argmax(log p + Gumbel)`` samples in
        # proportion to ``probs``, as ``multinomial`` does. The noise comes from
        # the stateless RNG above, so this vectorises across the batch without a
        # Generator for each request.
        if steps is None:
            steps_t = torch.zeros(B, dtype=torch.int64, device=device)
        elif isinstance(steps, int):
            steps_t = torch.full((B,), steps, dtype=torch.int64, device=device)
        else:
            steps_t = steps.to(device=device, dtype=torch.int64).reshape(-1)

        if seed is None:
            u = torch.rand((B, C, V), device=device, dtype=probs.dtype)
        else:
            u = _deterministic_uniform(B, C, V, seed, steps_t, device, probs.dtype)

        eps = 1e-20
        gumbel = -torch.log(-torch.log(u.clamp(eps, 1.0 - eps)))
        # log(0) is -inf on a filtered token, and -inf plus a finite Gumbel
        # stays -inf. The argmax never selects it, and no NaN appears.
        next_ids = torch.argmax(probs.clamp_min(0).log() + gumbel, dim=-1)  # (B, C)

        # A strong filter can set a whole row to zero. The code then falls back
        # to greedy: the argmax of the filtered logits. It applies the fallback
        # unconditionally, so there is no ``bool(invalid.any())`` host sync.
        # Where no row is invalid, ``torch.where`` returns ``next_ids``
        # unchanged. The result is identical to the guarded form, and it is safe
        # for graph capture.
        invalid = probs.sum(dim=-1) <= 0  # (B, C)
        next_ids = torch.where(invalid, logits.argmax(dim=-1), next_ids)

    text_col = torch.full(
        (B, 1), text_placeholder, dtype=next_ids.dtype, device=device
    )
    return torch.cat([next_ids, text_col], dim=-1)  # (B, C + 1)
