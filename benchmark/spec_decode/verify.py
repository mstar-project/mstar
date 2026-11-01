"""Draft verification for speculative decoding: the sampler's distribution and the per-position accept decision
(speculative sampling, Leviathan et al. 2023 / Chen et al. 2023).

Greedy decoding is the temperature-0 case. Both distributions collapse to one-hot vectors, so a draft is accepted
exactly when it is the target's argmax and the replacement on rejection is that argmax; one code path serves both
modes and the greedy decision does not depend on what the generator draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def sampling_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The distribution a sampler with these parameters draws from, as float32 probabilities over the last dim."""
    logits = logits.float()
    if params.greedy:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(torch.float32)
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(-1)
        # Keep the smallest prefix whose mass reaches top_p, including the token that crosses it.
        sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > params.top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, order, sorted_probs)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


def sample(probs: torch.Tensor, params: SamplingParams, generator: torch.Generator | None) -> int:
    """One token from a probability vector."""
    if params.greedy:
        return int(probs.argmax().item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


@dataclass(frozen=True)
class Decision:
    accepted: bool
    # On rejection, the token drawn from the residual distribution norm(max(target - draft, 0)).
    replacement: int | None
    # min(1, target(token) / draft(token)): the probability of accepting this draft given the two distributions,
    # which is the acceptance rate's low-variance estimator.
    accept_prob: float


def accept_draft(
    token: int, draft_probs: torch.Tensor, target_probs: torch.Tensor, generator: torch.Generator | None
) -> Decision:
    """Judge one draft token against the target's distribution at the same position."""
    q = float(draft_probs[token])
    p = float(target_probs[token])
    accept_prob = 1.0 if q <= 0.0 else min(1.0, p / q)
    u = torch.rand((), device=draft_probs.device, generator=generator).item()
    if u < accept_prob:
        return Decision(True, None, accept_prob)
    residual = (target_probs - draft_probs).clamp_(min=0.0)
    total = residual.sum()
    if total <= 0.0:
        # Identical distributions can still reject on rounding; the target itself is the correct fallback then.
        residual, total = target_probs, target_probs.sum()
    replacement = int(torch.multinomial(residual / total, 1, generator=generator).item())
    return Decision(False, replacement, accept_prob)
