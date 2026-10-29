"""What a model declares to decode speculatively: the spec, and its step.

Kept free of the resource so a submodule can declare a step without importing
the device-side bookkeeping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class DraftKind(Enum):
    # the checkpoint's own multi-token-prediction head, chained k times
    MTP = "mtp"


@dataclass
class SpecDecodeConfig:
    # drafts per verify step (k): a verify row carries k + 1 tokens
    num_draft_tokens: int
    draft: DraftKind = DraftKind.MTP
    # requests this node can hold at once, each with one row of device registers
    max_requests: int = 256
    # Whether verify steps can sample the target (temperature, top-k, top-p),
    # not only take its argmax. A capability: the model captures a sampled
    # decode variant beside the greedy one and routes a batch by its rows.
    sampling: bool = False

    def __post_init__(self):
        if self.num_draft_tokens < 1:
            raise ValueError(
                f"num_draft_tokens={self.num_draft_tokens}: a speculative node "
                "drafts at least one token; leave the resource undeclared to "
                "decode one token a step"
            )
        if self.max_requests < 1:
            raise ValueError(f"max_requests={self.max_requests} must be >= 1")

    @property
    def tokens_per_row(self) -> int:
        return self.num_draft_tokens + 1


@dataclass
class SpecDecodeSpec(NodeResourceSpec):
    config: SpecDecodeConfig
    # the paged KV the verify rows write and the attention resource that reads
    # it; this resource rewrites their device-side addressing per pass
    kv_cache: str = "kv_cache"
    attn: str = "attn"
    # ids that end generation, masked in-graph so a step never emits past one
    stop_token_ids: tuple[int, ...] = ()

    def depends_on(self) -> set[str]:
        return {self.kv_cache, self.attn}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.spec_decode.resource import SpecDecodeResource

        return SpecDecodeResource

    def apply_yaml_overrides(self, max_requests: int | None = None):
        """The register count is a deployment's concurrency. ``k`` is not here:
        the KV layers, recurrent slots and capture buckets are sized from it at
        declaration, so it is a model kwarg."""
        if max_requests is not None:
            self.config.max_requests = max_requests


class SpecPhase(Enum):
    # prefill rows: seed each request's registers from the prompt's last token
    SEED = "seed"
    # decode rows: verify k drafts, accept a prefix, draft the next k
    VERIFY = "verify"


@dataclass(frozen=True)
class SpecDecodeStep(ResourceStep):
    phase: SpecPhase = SpecPhase.VERIFY
    # rid -> tokens the host over-counted in this request's KV stream and has
    # since learned it can give back; KV drops them in the same commit
    # (`KVStep.commit_adjust`), and this resource forgets them then
    retract: Mapping[int, int] = field(default_factory=dict)
    # seed rows: rid -> decode tokens the request may emit (`max_tokens`), and
    # the requests that run past stop ids (`ignore_eos`)
    budgets: Mapping[int, int] = field(default_factory=dict)
    ignore_stop: frozenset[int] = frozenset()
    # seed rows of a sampling node: rid -> (temperature, top_p, top_k); a row
    # left out (or at temperature 0) verifies greedily
    sampling: Mapping[int, tuple[float, float, int]] = field(default_factory=dict)


class AcceptedTokens(NamedTuple):
    """What a verify step hands back, one row per request. A NamedTuple, so a
    compiled forward takes it back from the (uncompiled) resource call as a
    plain tuple of tensors.

    ``tokens`` holds the tokens the step emits in order, -1 past the last one;
    ``accepted`` how many of the row's k + 1 positions the target took (the
    stream's real growth, which the host needs to give back the rest of its
    reservation); ``done`` whether the row emitted a stop id or reached its
    budget, so the host can end the loop without reading the tokens.
    """

    tokens: torch.Tensor    # [rows, k + 1] int64
    accepted: torch.Tensor  # [rows] int32
    done: torch.Tensor      # [rows] bool
    # the target's own token at each verify position, unmasked: the draft's
    # first pass reads them
    target: torch.Tensor    # [rows, k + 1] int64
