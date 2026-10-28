"""What a node declares about draft-verify (speculative) decoding: the policy of a walk, the
resource that carries a verify step's verdict, the step, and the verdict the rewinds take.

Kept free of the resource, so a submodule can declare a step without importing it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep, SubmoduleStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# the key a node's draft-verify resource is declared under, by convention
DRAFT_VERIFY = "draft_verify"


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
    return SubmoduleStep(steps=steps, segments=segments, cg_key_info=step.cg_key_info)


@cache
def _has_verify_field(step_type: type) -> bool:
    # once per step type, not per step
    return dataclasses.is_dataclass(step_type) and any(f.name == "verify" for f in dataclasses.fields(step_type))


class VerdictBatch(NamedTuple):
    """The rows a rewind takes back, columnar: each row's request, the label its block grew
    and how many of the block's rows the verdict rejected (never 0)."""
    request_ids: list[int]
    labels: list[str]
    rejected: list[int]
