"""Multi-codebook TTS sampling for Zonos2.

Ported from ``../ZONOS2/python/zonos2/tts/sampler.py``. The reference
``sample_tts`` samples a whole batch and returns Python lists (a device
sync). Here :func:`sample_frame` is the tensor-returning variant used
*inside* the LLM submodule's forward (no ``.tolist()`` sync on the GPU
thread): it maps per-codebook logits ``(B, C, V)`` to frames ``(B, C + 1)``
= the sampled audio codes plus a text placeholder, for a batch of ``B``
requests at once.

Reproducibility under batching is provided by a *stateless* RNG: the
terminal draw is Gumbel-max over noise keyed purely on
``(seed, step, codebook, vocab)`` (see :func:`_deterministic_uniform`),
with no dependence on a request's batch position. This replaces the old
per-request ``torch.Generator`` (stateful, and — like FlashInfer's seeded
samplers — position-dependent once vectorised) so a request draws the same
frame at a given step regardless of which requests it is co-batched with.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TTSSamplingParams:
    """Per-request sampling parameters (defaults match the reference)."""

    temperature: float = 1.15
    topk: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int = 1024
    ignore_eos: bool = False
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    # Apply repetition penalty to codebooks 0..repetition_codebooks-1;
    # a negative value applies it to all codebooks.
    repetition_codebooks: int = 8
    seed: int | None = None


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering on a probability distribution."""
    if p <= 0.0 or p >= 1.0:
        return probs
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    probs = probs.scatter(-1, probs_idx, probs_sort)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p filtering: drop tokens below ``min_p * max_prob``."""
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
    """Per-codebook repetition penalty.

    ``repetition_token_ids`` is ``(B, C, W)`` — recent token ids per
    codebook. A token id of ``-1`` (or out of range) is ignored, so
    codebooks excluded from the penalty are masked by setting them to -1.
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
    """MurmurHash3 ``fmix32`` finalizer on uint32 values held in an int64
    tensor. Every value stays non-negative and ``< 2**32`` except the
    transient multiply, whose overflow past int64 wraps two's-complement
    and is immediately masked back to 32 bits — so the result matches the
    uint32 reference exactly, and the ``>>`` shifts act as logical shifts.
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
    """Reproducible ``U[0, 1)`` noise of shape ``(B, C, V)``.

    Keyed purely on ``(seed, step, codebook, vocab)`` via a counter-based
    hash — no dependence on batch position — so the noise for request ``b``
    at ``steps[b]`` is identical whether it is sampled alone or inside any
    batch. ``steps`` is a ``(B,)`` per-request step index.
    """
    v = torch.arange(V, device=device, dtype=torch.int64).view(1, 1, V)
    c = torch.arange(C, device=device, dtype=torch.int64).view(1, C, 1)
    s = steps.to(device=device, dtype=torch.int64).view(B, 1, 1)
    base = int(seed) & _M32
    # Chained fmix32 rounds so every field avalanches into the result.
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
    """Sample one frame per request from per-codebook logits.

    Args:
        logits: ``(B, C, V)`` per-codebook logits for the current step.
        params: sampling parameters (shared across the batch).
        repetition_token_ids: ``(B, C, W)`` recent tokens (``-1`` padded /
            ignored), or None.
        text_placeholder: value written to the appended text column.
        seed: base RNG seed shared across the batch. ``None`` draws from the
            global RNG (non-reproducible), matching an unseeded request.
        steps: ``(B,)`` per-request step index (or an int / None -> 0). With
            ``seed`` set, ``(seed, step)`` fully determine a request's draw
            independent of batch position, so batched sampling stays
            bit-reproducible per request.

    Returns:
        ``(B, C + 1)`` int64 frames: ``[cb0, ..., cb_{C-1}, text_placeholder]``.
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

        # Reproducible Gumbel-max: ``argmax(log p + Gumbel)`` samples
        # proportional to ``probs`` (equivalent to ``multinomial``) but the
        # noise is the stateless per-cell RNG above, so it vectorises across
        # the batch without a per-request Generator.
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
        # log(0)=-inf on filtered tokens -> -inf + finite Gumbel = -inf (never
        # argmax'd), no NaN.
        next_ids = torch.argmax(probs.clamp_min(0).log() + gumbel, dim=-1)  # (B, C)

        # An over-aggressive filter can zero a whole row; fall back to greedy
        # (argmax of the filtered logits) there so the draw stays well-defined.
        # Applied unconditionally (no ``bool(invalid.any())`` host sync): where
        # nothing is invalid the ``torch.where`` returns ``next_ids`` unchanged,
        # so this is bit-identical to the guarded form but graph-capture-safe.
        invalid = probs.sum(dim=-1) <= 0  # (B, C)
        next_ids = torch.where(invalid, logits.argmax(dim=-1), next_ids)

    text_col = torch.full(
        (B, 1), text_placeholder, dtype=next_ids.dtype, device=device
    )
    return torch.cat([next_ids, text_col], dim=-1)  # (B, C + 1)
