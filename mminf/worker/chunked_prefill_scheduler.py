"""Pure scheduler logic for Phase 2 chunked prefill.

Given lists of ready decode and prefill requests plus a per-step token
budget, produce a ChunkedStepPlan describing what to run this step.

Decode-first: each decode contributes 1 token; running them keeps tail
latency stable. Prefill chunks fill remaining budget. If a prefill's
remaining tokens fit in the budget, that chunk is "terminal" — the
request transitions to decode after this step, so we sample its output.
Non-terminal prefill chunks skip lm_head + sampling.

Pure: no torch, no IPC, no engine state. Easy to test, easy to reason
about. The MicroScheduler reads request state, constructs the input
dataclasses, calls plan_chunked_step, then turns the plan into a NodeBatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecodeReadyRequest:
    """A request that has 1 token to decode this step."""

    rid: str


@dataclass(frozen=True)
class PrefillReadyRequest:
    """A request with chunked prefill in progress."""

    rid: str
    tokens_remaining: int


@dataclass
class ChunkedStepPlan:
    """The scheduler's verdict for one step.

    decode_rids: requests that should each contribute 1 token (decode).
    prefill_allocations: rid → number of tokens to feed this step.
    terminal_prefills: rids whose prefill completes this step (last chunk).
        These need lm_head + sampling to produce the first decode token.
    """

    decode_rids: list[str] = field(default_factory=list)
    prefill_allocations: dict[str, int] = field(default_factory=dict)
    terminal_prefills: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return len(self.decode_rids) + sum(self.prefill_allocations.values())


def plan_chunked_step(
    ready_decodes: list[DecodeReadyRequest],
    ready_prefills: list[PrefillReadyRequest],
    max_step_tokens: int,
) -> ChunkedStepPlan:
    """Pack one step under the token budget.

    Decode-first because each decode is 1 token; running them keeps tail
    latency stable. Prefill fills remaining budget. If a prefill request's
    remaining tokens fit in the budget, the chunk is terminal (transitions
    the request to decode after this step).
    """
    if max_step_tokens <= 0:
        raise ValueError(f"max_step_tokens must be positive, got {max_step_tokens}")

    plan = ChunkedStepPlan()
    budget = max_step_tokens

    # Decodes first.
    for req in ready_decodes:
        if budget <= 0:
            break
        plan.decode_rids.append(req.rid)
        budget -= 1

    # Prefill fills remaining budget.
    for req in ready_prefills:
        if budget <= 0:
            break
        if req.tokens_remaining <= 0:
            continue
        chunk = min(req.tokens_remaining, budget)
        plan.prefill_allocations[req.rid] = chunk
        if chunk == req.tokens_remaining:
            plan.terminal_prefills.add(req.rid)
        budget -= chunk

    return plan
