"""Multi-codebook TTS sampling for Zonos2.

Ported from ``../ZONOS2/python/zonos2/tts/sampler.py``. The reference
``sample_tts`` samples a whole batch and returns Python lists (a device
sync). Here :func:`sample_frame` is the single-sequence, tensor-returning
variant used *inside* the LLM submodule's forward (no ``.tolist()`` sync
on the GPU thread): it maps per-codebook logits ``(1, C, V)`` to a frame
tensor ``(1, C + 1)`` = the sampled audio codes plus a text placeholder.
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


def sample_frame(
    logits: torch.Tensor,
    params: TTSSamplingParams,
    repetition_token_ids: torch.Tensor | None = None,
    text_placeholder: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one frame from per-codebook logits.

    Args:
        logits: ``(1, C, V)`` per-codebook logits for the current step.
        params: sampling parameters.
        repetition_token_ids: ``(1, C, W)`` recent tokens, or None.
        text_placeholder: value written to the appended text column.
        generator: optional per-request RNG for deterministic sampling.

    Returns:
        ``(1, C + 1)`` int64 frame: ``[cb0, ..., cb_{C-1}, text_placeholder]``.
    """
    B, C, V = logits.shape

    logits = apply_repetition_penalty(
        logits, repetition_token_ids, params.repetition_penalty
    )

    if params.temperature <= 0:
        next_ids = torch.argmax(logits, dim=-1)  # (B, C)
    else:
        logits = logits / max(params.temperature, 1e-8)
        flat = logits.reshape(B * C, V)

        top_k = int(params.topk)
        if 0 < top_k < V:
            values, _ = torch.topk(flat, top_k, dim=-1)
            kth = values[..., -1].unsqueeze(-1)
            flat = flat.masked_fill(flat < kth, float("-inf"))

        probs = F.softmax(flat, dim=-1)
        if 0.0 < params.top_p < 1.0:
            probs = apply_top_p(probs, params.top_p)
        if params.min_p > 0.0:
            probs = apply_min_p(probs, params.min_p)

        # An over-aggressive filter can zero an entire row; fall back to
        # greedy there so multinomial doesn't fault on an all-zero dist.
        invalid = probs.sum(dim=-1) <= 0
        if bool(invalid.any()):
            greedy = flat.argmax(dim=-1)
            fallback = torch.zeros_like(probs)
            fallback.scatter_(-1, greedy.unsqueeze(-1), 1.0)
            probs = torch.where(invalid.unsqueeze(-1), fallback, probs)

        next_ids = torch.multinomial(
            probs, num_samples=1, generator=generator
        ).view(B, C)

    text_col = torch.full(
        (B, 1), text_placeholder, dtype=next_ids.dtype, device=next_ids.device
    )
    return torch.cat([next_ids, text_col], dim=-1)  # (B, C + 1)
