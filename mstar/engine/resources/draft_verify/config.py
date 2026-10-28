"""What a node declares about draft-verify (speculative) decoding: the policy of a walk, the
resource that carries a verify step's verdict, the step, and the verdict the rewinds take; the
steps of an engine-side proposer; and a deployment's ``draft`` settings.

Kept free of the resources, so a submodule can declare a step without importing them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple

from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep, SubmoduleStep

if TYPE_CHECKING:
    import torch

    from mstar.engine.resources.base import Resource

# the keys a node's draft-verify resource and its n-gram proposer are declared under, by convention
DRAFT_VERIFY = "draft_verify"
NGRAM = "ngram"


@dataclass(frozen=True)
class DraftPolicy:
    """How a walk drafts and verifies, from ``NodeSubmodule.get_draft_policy``.

    Every step of the walk verifies: a row spans ``block`` tokens, the last emitted token and
    ``max_drafts`` drafts, and emits the drafts the target agrees with plus the target's next
    token. The model keeps its drafter's forward. The engine owns the verdict, the rewind of
    the rejected rows, the tokens a row emits and its stop.
    """
    max_drafts: int
    # tokens that end a reply: a row emits up to the first of them, unless the request sets
    # ignore_eos, and a row whose input is one emits nothing (the reply ended a step earlier)
    stop_ids: frozenset[int] = frozenset()
    # the loop whose iterations verify, and which the stop ends
    loop: str = "decode_loop"
    # the output signal the emitted tokens go out on
    emit: str = "new_token"
    # tokens a request has emitted when its first verify step runs (an AR prefill samples
    # one); max_tokens counts them
    emitted_before: int = 1
    # the resource key of an engine-side proposer (``DraftProposer``) that drafts for the walk,
    # through ``DraftVerifyResource.propose``; None when the model's forward drafts (MTP)
    proposer: str | None = None

    def __post_init__(self):
        for name in ("max_drafts", "emitted_before"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DraftPolicy.{name} must be an integer >= 0; got {value!r}")
        if self.max_drafts < 1:
            raise ValueError("DraftPolicy.max_drafts must be at least 1; a walk with no drafts has no policy")
        object.__setattr__(self, "stop_ids", frozenset(int(t) for t in self.stop_ids))

    @property
    def block(self) -> int:
        """Tokens a verify row spans: ``prepare_inputs`` sets ``input_seq_len`` to it."""
        return self.max_drafts + 1

    @property
    def lookahead(self) -> int:
        """Slots a request's KV may hold past the tokens it keeps (``LabelLayout.lookahead``):
        a verify block not yet rewound, and, under the async worker, the block of one more
        step after a ``max_tokens`` stop, which runs from a length that kept every accepted
        draft past the budget."""
        return 2 * self.max_drafts + 1


@dataclass
class DraftVerifySpec(NodeResourceSpec):
    """The resource that carries a verify step's verdict from the forward to the host: the
    rewinds, the emitted tokens and the stop read it. One per node that verifies."""
    # the node's sampler, which verifies (declared with ``SamplerSpec(draft_verify=True)``)
    sampler: str = "sampler"

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.draft_verify.resource import DraftVerifyResource

        return DraftVerifyResource


@dataclass(frozen=True)
class VerifyStep(ResourceStep):
    """A verify step: each segment is a row of ``drafts + 1`` tokens."""
    drafts: int = 0
    # the key of the sampler step ``with_draft_verify`` added beside it
    sampler: str = "sampler"


def with_draft_verify(
    step: SubmoduleStep, policy: DraftPolicy, key: str = DRAFT_VERIFY, sampler: str = "sampler",
) -> SubmoduleStep:
    """``step``, a decode step of a walk with ``policy``, as a verify step: ``VerifyStep``
    added under ``key``, and a ``SamplerStep`` under ``sampler`` (the ``DraftVerifySpec``'s),
    which a sampled verify draws with. Every row, padding rows included, spans
    ``policy.block``. Each step of the submodule's with a ``verify`` field gets it set: paged KV
    then computes the block's write slots on the host, and state the forward updates in place
    waits for the verdict to settle it (``RewindKind.DEVICE``). The submodule calls this in
    ``declare_step``, on a step that declares no sampler step of its own."""
    segments = step.segments
    if segments is None:
        raise ValueError("with_draft_verify needs the step's segments on the SubmoduleStep")
    if sampler in step.steps:
        raise ValueError(
            f"a verify step's sampler step is with_draft_verify's; declare none under {sampler!r}"
        )
    for segment in segments:
        if segment.span != policy.block:
            raise ValueError(
                f"a verify step's row spans the block of {policy.block} tokens, but "
                f"{segment.request_id}/{segment.label} spans {segment.span}; set "
                "input_seq_len to policy.block in prepare_inputs"
            )
    steps = {
        name: dataclasses.replace(s, verify=True) if _has_verify_field(type(s)) else s
        for name, s in step.steps.items()
    }
    # the step's own segments, as every resource that names none gets them
    steps[key] = VerifyStep(drafts=policy.max_drafts, sampler=sampler)
    steps[sampler] = SamplerStep(apply_penalty=False, verify=True)
    if policy.proposer is not None:
        if policy.proposer in step.steps:
            raise ValueError(
                f"a verify step's proposer step is with_draft_verify's; declare none under {policy.proposer!r}"
            )
        steps[policy.proposer] = ProposeStep(drafts=policy.max_drafts)
    return SubmoduleStep(steps=steps, segments=segments, cg_key_info=step.cg_key_info)


@cache
def _has_verify_field(step_type: type) -> bool:
    # once per step type, not per step
    return dataclasses.is_dataclass(step_type) and any(f.name == "verify" for f in dataclasses.fields(step_type))


@dataclass(frozen=True, eq=False)
class ProposeStep(ResourceStep):
    """An engine-side proposer's part of a step. A verify step's (``with_draft_verify``) stages
    ``drafts`` per row for the forward; a step that only adds to the requests' histories
    (a prefill, declared by the model) names the tokens it adds."""
    drafts: int = 0
    # request -> the token ids this step adds to its history, in order (a prompt's text)
    history: Mapping[int, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftConfig:
    """A deployment's draft-verify settings: a model's ``draft`` kwarg, the YAML's
    ``model_kwargs: {draft: {...}}``, read the same way by every model that takes one.

    ``method: ngram`` drafts by prompt lookup (``NgramProposer``, engine-side): no drafter
    weights, any model whose decode verifies. ``method: draft_model`` drafts with a smaller
    checkpoint of the target's tokenizer (``model``), which the model builds inside its node
    (``mstar.model.components.draft_model``).
    """
    method: str
    # drafts per verify step (k): each row spans k + 1 tokens
    max_drafts: int = 4
    # ngram: the longest and shortest suffix the proposer matches, longest first
    max_ngram: int = 4
    min_ngram: int = 2
    # draft_model: the drafter's checkpoint (a Hugging Face id or a path), and whether it
    # draws its drafts as the request samples (``DraftVerifyResource.draft_sample``) rather
    # than taking its argmax
    model: str | None = None
    draft_sample: bool = False
    # serve temperature > 0 by rejection sampling (``SamplerSpec.enable_sampled_verify``);
    # off, the verify is an argmax and a request with a temperature is refused
    sampled_verify: bool = True

    METHODS = ("ngram", "draft_model")

    def __post_init__(self):
        if self.method not in self.METHODS:
            raise ValueError(f"draft.method {self.method!r}: one of {', '.join(self.METHODS)}")
        if not 1 <= self.min_ngram <= self.max_ngram:
            raise ValueError(f"draft n-gram sizes must satisfy 1 <= min_ngram <= max_ngram; got "
                             f"{self.min_ngram} and {self.max_ngram}")
        if (self.method == "draft_model") != (self.model is not None):
            raise ValueError("draft.model (the drafter's checkpoint) goes with method draft_model, and only with it")
        if self.draft_sample and self.method != "draft_model":
            raise ValueError("draft.draft_sample is a draft model's setting")
        DraftPolicy(max_drafts=self.max_drafts)  # checks max_drafts

    @classmethod
    def from_kwargs(cls, value: DraftConfig | Mapping[str, Any] | None) -> DraftConfig | None:
        """The ``draft`` kwarg as a config; None (or an empty mapping) drafts nothing."""
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"draft must be a mapping of settings; got {type(value).__name__}")
        if not value:
            return None
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown draft settings {unknown}; known: {sorted(known)}")
        return cls(**value)

    @property
    def proposer(self) -> str | None:
        """The engine-side proposer's resource key; None when a drafter in the forward drafts."""
        return NGRAM if self.method == "ngram" else None

    def policy(self, stop_ids=frozenset(), **kwargs) -> DraftPolicy:
        """The decode walk's policy under these settings. Its ``lookahead`` is what the walk's
        KV layouts declare (``process_prompt`` reads it here, where no submodule exists)."""
        return DraftPolicy(max_drafts=self.max_drafts, stop_ids=stop_ids, proposer=self.proposer, **kwargs)

    def resources(self, nodes: set[str], sampler: str = "sampler") -> list[NodeResourceSpec]:
        """The specs a node that drafts this way declares beside its sampler (which it declares
        with ``draft_verify=True`` and ``enable_sampled_verify=self.sampled_verify``), and
        beside a draft model's own (``DraftModel.node_resources``)."""
        specs: list[NodeResourceSpec] = [DraftVerifySpec(resource_key=DRAFT_VERIFY, nodes=set(nodes), sampler=sampler)]
        if self.method == "ngram":
            from mstar.engine.resources.draft_verify.ngram import NgramSpec

            specs.append(NgramSpec(
                resource_key=NGRAM, nodes=set(nodes), max_drafts=self.max_drafts,
                min_ngram=self.min_ngram, max_ngram=self.max_ngram,
            ))
        return specs


class VerdictBatch(NamedTuple):
    """The rows a rewind takes back, columnar: each row's request, the label its block grew
    and how many of the block's rows the verdict rejected (never 0)."""
    request_ids: list[int]
    labels: list[str]
    rejected: list[int]
