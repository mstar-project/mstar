"""Speculative decoding with sequential verification, over abstract target and draft models.

A step drafts k tokens, then feeds the target one token at a time: the target's distribution at a position is known
before the draft there is judged, so a rejected draft is never fed to the target and the target's state is never
rolled back. The target does exactly one forward per token it emits, which keeps every (k, sampling mode) cell cheap
enough to run as the real process rather than a replay of one continuation. Only the draft's cache is rolled back:
the positions it extended from its own hidden states are dropped and recomputed from the target's hidden states once
the step's tokens are known, as a serving engine's draft pass does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from benchmark.spec_decode.verify import Decision, SamplingParams, accept_draft, sample, sampling_probs


class Target(Protocol):
    def prefill(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the prompt. Returns the next-token logits and the hidden state of every prompt position."""

    def step(self, token: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one token. Returns the next-token logits and that position's hidden state."""


class Draft(Protocol):
    def reset(self) -> None: ...

    def extend(self, hiddens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Append positions computed from the target's hidden states, each paired with the token that followed it.
        Returns the logits of the draft for the token after the last one."""

    def chain(self, token: int) -> torch.Tensor:
        """Append one position computed from the draft's own last hidden state and a drafted token; returns the
        logits for the next draft."""

    def rollback(self, n: int) -> None:
        """Drop the last n positions (a finished step's chain positions)."""


@dataclass
class RunStats:
    """Acceptance accounting for one k, summed over prompts."""

    k: int
    prompts: int = 0
    steps: int = 0
    tokens: int = 0  # generated tokens, the first sampled token included
    accepted_total: int = 0
    # Per draft position j: steps that reached it (every earlier draft accepted), how many accepted it, and the sum
    # of min(1, p/q) there.
    evaluated: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    accept_prob_sum: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.evaluated:
            self.evaluated = [0] * self.k
            self.accepted = [0] * self.k
            self.accept_prob_sum = [0.0] * self.k

    def record(self, position: int, decision: Decision) -> None:
        self.evaluated[position] += 1
        self.accepted[position] += int(decision.accepted)
        self.accept_prob_sum[position] += decision.accept_prob

    def merge(self, other: RunStats) -> None:
        assert other.k == self.k
        self.prompts += other.prompts
        self.steps += other.steps
        self.tokens += other.tokens
        self.accepted_total += other.accepted_total
        for j in range(self.k):
            self.evaluated[j] += other.evaluated[j]
            self.accepted[j] += other.accepted[j]
            self.accept_prob_sum[j] += other.accept_prob_sum[j]

    @property
    def mean_accepted(self) -> float:
        return self.accepted_total / max(self.steps, 1)

    @property
    def tokens_per_step(self) -> float:
        """Tokens a step emits: the accepted drafts plus the target's own token (bonus or correction)."""
        return 1.0 + self.mean_accepted

    def position_rates(self) -> list[float]:
        """Acceptance at position j given the step reached it."""
        return [self.accepted[j] / self.evaluated[j] if self.evaluated[j] else float("nan") for j in range(self.k)]

    def position_expected_rates(self) -> list[float]:
        """The same, estimated from min(1, p/q) instead of the coin flips."""
        return [
            self.accept_prob_sum[j] / self.evaluated[j] if self.evaluated[j] else float("nan") for j in range(self.k)
        ]

    def reach_rates(self) -> list[float]:
        """Fraction of steps that accepted at least j+1 drafts."""
        return [self.accepted[j] / max(self.steps, 1) for j in range(self.k)]

    def summary(self) -> dict:
        return {
            "k": self.k,
            "prompts": self.prompts,
            "steps": self.steps,
            "tokens": self.tokens,
            "mean_accepted": self.mean_accepted,
            "tokens_per_step": self.tokens_per_step,
            "position_rates": self.position_rates(),
            "position_expected_rates": self.position_expected_rates(),
            "reach_rates": self.reach_rates(),
        }


def run_speculative(
    target: Target,
    draft: Draft,
    prompt: list[int],
    k: int,
    params: SamplingParams,
    max_new_tokens: int,
    stop_ids: frozenset[int] = frozenset(),
    generator: torch.Generator | None = None,
) -> tuple[list[int], RunStats]:
    """Generate up to max_new_tokens after the prompt with k drafts per step. Returns the generated tokens (the stop
    token included when one was reached) and the step statistics."""
    stats = RunStats(k=k, prompts=1)
    logits, hiddens = target.prefill(prompt)
    first = sample(sampling_probs(logits, params), params, generator)
    generated = [first]
    draft.reset()
    # Draft position p pairs the target's hidden state at p with the token at p+1.
    q_logits = draft.extend(hiddens, torch.tensor(prompt[1:] + [first], dtype=torch.long))
    pending = first  # emitted, not yet fed to the target

    def finished() -> bool:
        return len(generated) >= max_new_tokens or generated[-1] in stop_ids

    while not finished():
        drafts: list[int] = []
        draft_dists: list[torch.Tensor] = []
        for j in range(k):
            q = sampling_probs(q_logits, params)
            drafts.append(sample(q, params, generator))
            draft_dists.append(q)
            if j + 1 < k:
                q_logits = draft.chain(drafts[j])

        logits, h = target.step(pending)
        new_hiddens = [h]
        n_accepted = 0
        next_token: int | None = None
        for j in range(k):
            decision = accept_draft(drafts[j], draft_dists[j], sampling_probs(logits, params), generator)
            stats.record(j, decision)
            if not decision.accepted:
                next_token = decision.replacement
                break
            n_accepted += 1
            generated.append(drafts[j])
            if finished():
                break
            logits, h = target.step(drafts[j])
            new_hiddens.append(h)
        else:
            # Every draft accepted: the target's distribution after the last one gives a bonus token.
            next_token = sample(sampling_probs(logits, params), params, generator)
        stats.steps += 1
        stats.accepted_total += n_accepted
        if next_token is None:
            break  # a stop token or the length cap landed on an accepted draft
        generated.append(next_token)
        if finished():
            break
        draft.rollback(k - 1)
        new_tokens = generated[-(n_accepted + 1):]
        q_logits = draft.extend(torch.stack(new_hiddens), torch.tensor(new_tokens, dtype=torch.long))
        pending = next_token

    stats.tokens = len(generated)
    return generated, stats
