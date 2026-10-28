"""A draft model inside the node that verifies its drafts.

The drafter is a model of its own (a small LLM; later an EAGLE-3 or DSpark head over the
target's hidden states) with its own weights, KV, positions and attention. The engine's
draft-verify core knows nothing about it: the drafter's state lives in resources the node
declares beside the target's (``DraftModel.node_resources``), and the runner rewinds every
one of them by the same verdict as the target's, at the next step's admit. Its forward runs
inside the target's verify step, so propose and verify replay as one captured graph:

    propose   k drafts from the step's input                  ``DraftLLMProposer.propose``
    agree     rank 0's drafts on every rank                   ``DraftVerifyResource.agree``
    verify    the target over [bonus | drafts], the verdict   the target, ``DraftVerifyResource.verify``
    carry     the next step's input, from the verdict         ``DraftLLMProposer.next_inputs``

``draft_verify_step`` (``mstar.engine.resources``) is that sequence; a node's forward calls it
with its target and the proposer, as it would with no drafter for an engine-side proposer.

A draft LLM's KV runs one token behind the target's. A verify step's input is the request's
last two kept tokens, ``[prev, bonus]``, and the drafter's block is ``[prev, bonus, d1 ..
d(k-1)]``, one position before the target's ``[bonus, d1 .. dk]``. Both blocks span ``k + 1``
rows, so the verdict takes the same ``k - accepted`` rows back from both, and after the
rewind the drafter holds every kept token but the last, which the next block starts with.
That is also how the drafter catches up after a full acceptance, which leaves dk's row
unwritten: the next block rewrites it, at the same width every step. The prefill fills the
drafter with the prompt but its last token.

A draft pass runs the whole block under the step's one attention plan. The plan is causal,
so a pass's output at its row reads only the rows at and before it, which the pass has just
written; the rows after it hold placeholders that later passes overwrite. That is ``k``
passes of ``k + 1`` rows each, rather than an attention plan per pass. At decode batch sizes
a pass costs the drafter's weight reads, nearly whatever its width.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch import nn

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    Drafts,
    KVSpec,
    KVStep,
    NodeResourceSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    PositionStep,
    Segment,
)
from mstar.engine.resources.step import ResourceStep
from mstar.model.base import KVLabel, LabelLayout, Span

if TYPE_CHECKING:
    from mstar.engine.resources.draft_verify.resource import DraftVerifyResource, VerdictTensors


class DraftKeys(NamedTuple):
    """The resource keys a drafter's own state is declared under."""
    kv: str = "draft_kv"
    attn: str = "draft_attn"
    pos: str = "draft_pos"


@dataclass(frozen=True)
class DraftModel:
    """A draft LLM as its node declares it: the resources, request layouts and step parts it
    adds to the target's. Plain configuration, built wherever the model is (no weights)."""
    max_drafts: int
    kv: PagedKVConfig
    keys: DraftKeys = DraftKeys()
    rope_theta: float = 10000.0
    # draw each draft as the request samples (``DraftVerifyResource.draft_sample``), which
    # a sampled verify then takes with its distribution; else the argmax, a certain draft
    sample: bool = False
    # the drafter's KV runs this many tokens behind the target's (see the module docstring)
    lag: int = 1

    def __post_init__(self):
        if self.max_drafts < 1:
            raise ValueError(f"a draft model drafts at least 1 token a step; got {self.max_drafts}")
        if self.lag != 1:
            raise ValueError("a draft LLM's KV runs exactly one token behind the target's")

    def node_resources(self, nodes: set[str]) -> list[NodeResourceSpec]:
        keys = self.keys
        return [
            KVSpec(resource_key=keys.kv, nodes=set(nodes), config=self.kv),
            AttentionSpec(resource_key=keys.attn, nodes=set(nodes), config=AttentionConfig(kv_cache=keys.kv)),
            PositionSpec(
                resource_key=keys.pos, nodes=set(nodes),
                config=PositionConfig(kv_cache=keys.kv, rope_theta=self.rope_theta),
            ),
        ]

    def kv_labels(self, walk: str, loop: str) -> dict[str, dict[str, KVLabel]]:
        return {self.keys.kv: {"main": KVLabel(growth=((walk, loop),))}}

    def check_prompt(self, length: int) -> None:
        """``ValueError`` (a 400) for a prompt the drafter can't prefill: it prefills all but
        the prompt's last token, so it needs two."""
        if length <= self.lag:
            raise ValueError(
                f"a prompt of {length} token(s); a node with a draft model needs at least {self.lag + 1}"
            )

    def kv_layout(
        self, prompt_len: int, prefill_walk: str, loop: str, lookahead: int,
    ) -> dict[str, dict[str, LabelLayout]]:
        """The drafter's KV layout beside the target's: the prompt but its last token, one
        slot per kept token, and the target's lookahead (the same rows written ahead and
        taken back)."""
        length = prompt_len - self.lag
        span = Span("rows", prefill_walk, length, length, "text_inputs")
        return {self.keys.kv: {"main": LabelLayout(spans=(span,), growth=((loop, 1),), lookahead=lookahead)}}

    def prefill_steps(self, request_ids: Sequence[int], spans: Sequence[int]) -> dict[str, ResourceStep]:
        """The drafter's part of a prefill step whose rows span ``spans``: each prompt but its
        last token."""
        segments = tuple(
            Segment(rid, "main", span - self.lag) for rid, span in zip(request_ids, spans, strict=True)
        )
        keys = self.keys
        return {
            keys.kv: KVStep(segments=segments),
            keys.attn: AttentionStep(causal=True, segments=segments),
            keys.pos: PositionStep(segments=segments),
        }

    def verify_steps(self) -> dict[str, ResourceStep]:
        """The drafter's part of a verify step, on the step's own block segments."""
        keys = self.keys
        return {keys.kv: KVStep(), keys.attn: AttentionStep(causal=True), keys.pos: PositionStep()}

    def carry_width(self) -> int:
        """Tokens a verify step's input carries: the last ``lag + 1`` kept."""
        return self.lag + 1


class DraftLM(Protocol):
    """The surface ``DraftLLMProposer`` drives (``Qwen3DenseLM``'s), bound to the drafter's
    resources."""

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor: ...

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class DraftLLMProposer(nn.Module):
    """A small LLM drafting for its node's target (same tokenizer), on the resources its
    ``DraftModel`` declares: a ``Drafter`` for ``draft_verify_step``."""

    def __init__(self, decl: DraftModel, lm: DraftLM):
        super().__init__()
        self.decl = decl
        self.lm = lm

    def prefill(self, input_ids: torch.Tensor) -> None:
        """Fill the drafter's KV from the prompts but their last tokens, packed in the order
        of the prefill step's rows."""
        self.lm(self.lm.embed(input_ids), label="main")

    def propose(
        self, inputs: torch.Tensor, dv: DraftVerifyResource, request_ids: list[int],
    ) -> Drafts:
        """``k`` drafts per row from ``inputs`` ``[n, 2]`` (prev, bonus), one pass each."""
        k = self.decl.max_drafts
        n = inputs.shape[0]
        # prev and bonus are real rows; the rest are placeholders until their drafts land
        block = torch.cat([inputs, inputs[:, 1:].expand(n, k - 1)], 1)
        tokens, probs = [], []
        for i in range(k):
            hidden = self.lm(self.lm.embed(block.reshape(-1)), label="main")
            logits = self.lm.logits(hidden.view(n, k + 1, -1)[:, i + 1]).float()
            if self.decl.sample:
                draft, dist = dv.draft_sample(request_ids, logits)
            else:
                draft, dist = logits.argmax(-1), None
            draft = draft.to(block.dtype)
            tokens.append(draft)
            probs.append(dist)
            if i + 2 <= k:
                block[:, i + 2] = draft
        stacked = torch.stack(probs, 1) if probs[0] is not None else None
        return Drafts(torch.stack(tokens, 1), stacked)

    @staticmethod
    def next_inputs(block: torch.Tensor, verdict: VerdictTensors) -> torch.Tensor:
        """``[n, 2]``: the last token the row kept from its block (its input when it accepted
        nothing), and the target's next token."""
        prev = block.gather(1, verdict.accepted.to(torch.long)[:, None])
        return torch.cat([prev, verdict.next_inputs.to(block.dtype)], 1)
