"""What a node declares about draft-verify (speculative) decoding: the policy of a walk, the
resource that carries a verify step's verdict, the step, and the verdict the rewinds take; the
steps of an engine-side proposer; and a deployment's ``draft`` settings.

Kept free of the resources, so a submodule can declare a step without importing them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
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


def _gain_points(value) -> float | tuple[tuple[int, float], ...]:
    """``DraftPolicy.min_gain`` as a float, or as ``(rows, gain)`` points sorted by rows."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not value >= 0:
            raise ValueError(f"DraftPolicy.min_gain must be >= 0; got {value!r}")
        return float(value)
    try:
        points = tuple(sorted((int(rows), float(gain)) for rows, gain in value))
    except (TypeError, ValueError) as err:
        raise ValueError(f"DraftPolicy.min_gain: a number or [rows, gain] points; got {value!r}") from err
    rows = [r for r, _ in points]
    if not points or min(rows) < 1 or len(set(rows)) < len(rows) or not all(g >= 0 for _, g in points):
        raise ValueError(f"DraftPolicy.min_gain points need rows >= 1, once each, and gains >= 0; got {value!r}")
    return points


class VerifyKey(NamedTuple):
    """The capture key of a step of an adaptive walk (``DraftPolicy.capture_key``)."""
    drafts: int


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
    # A step drafts only when its rows are expected to keep enough drafts to pay for the wider
    # block, and otherwise runs one token a row (a step of no drafts, captured beside the
    # verify step). The engine decides per step, before the step is declared, from what the
    # proposer predicts of each row (``DraftProposer.expected_gain``).
    adaptive: bool = False
    # the drafts per row a step's rows must be expected to keep for it to draft: what a row of
    # the verify block costs over a row of one token, in tokens. A number, or points
    # ``((rows, gain), ...)`` interpolated by the step's rows and flat past the ends: the block
    # costs more as the batch grows (``gain_needed``)
    min_gain: float | tuple[tuple[int, float], ...] = 0.5

    def __post_init__(self):
        for name in ("max_drafts", "emitted_before"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DraftPolicy.{name} must be an integer >= 0; got {value!r}")
        if self.max_drafts < 1:
            raise ValueError("DraftPolicy.max_drafts must be at least 1; a walk with no drafts has no policy")
        if self.adaptive and self.proposer is None:
            raise ValueError(
                "an adaptive DraftPolicy needs an engine-side proposer, whose predictions decide each step"
            )
        object.__setattr__(self, "min_gain", _gain_points(self.min_gain))
        object.__setattr__(self, "stop_ids", frozenset(int(t) for t in self.stop_ids))

    @property
    def block(self) -> int:
        """Tokens a verify row spans: ``prepare_inputs`` sets ``input_seq_len`` to it. Under an
        adaptive policy the engine sets each step's own (``drafts + 1``)."""
        return self.max_drafts + 1

    @property
    def draft_options(self) -> tuple[int, ...]:
        """The drafts a step of the walk may run: ``max_drafts``, and 0 when adaptive."""
        return (0, self.max_drafts) if self.adaptive else (self.max_drafts,)

    def gain_needed(self, rows: int) -> float:
        """The drafts a step of ``rows`` rows must be expected to keep, summed, to draft."""
        points = self.min_gain
        if isinstance(points, float):
            return points * rows
        if rows <= points[0][0]:
            return points[0][1] * rows
        for (r0, g0), (r1, g1) in zip(points, points[1:], strict=False):
            if rows <= r1:
                return (g0 + (g1 - g0) * (rows - r0) / (r1 - r0)) * rows
        return points[-1][1] * rows

    def capture_key(self, drafts: int) -> VerifyKey | None:
        """The ``cg_key_info`` of a step of ``drafts`` drafts: None when the walk always drafts
        (its one capture), else ``VerifyKey(drafts)``, a capture per option."""
        return VerifyKey(drafts) if self.adaptive else None

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
    which a sampled verify draws with. Every row, padding rows included, spans one block, the
    step's input token and its drafts: ``policy.block``, or under an adaptive policy the
    block the engine chose for the step (1 for a step of no drafts). Each step of the
    submodule's with a ``verify`` field gets it set when the step drafts: paged KV then
    computes the block's write slots on the host, and state the forward updates in place waits
    for the verdict to settle it (``RewindKind.DEVICE``). The submodule calls this in
    ``declare_step``, on a step that declares no sampler step of its own."""
    segments = step.segments
    if segments is None:
        raise ValueError("with_draft_verify needs the step's segments on the SubmoduleStep")
    if sampler in step.steps:
        raise ValueError(
            f"a verify step's sampler step is with_draft_verify's; declare none under {sampler!r}"
        )
    block = segments[0].span if segments else policy.block
    drafts = block - 1
    for segment in segments:
        if segment.span != block or drafts not in policy.draft_options:
            raise ValueError(
                f"a verify step's rows span one block of 1 + {' or '.join(map(str, policy.draft_options))} "
                f"drafts, but {segment.request_id}/{segment.label} spans {segment.span} in a step "
                f"whose first row spans {block}; set input_seq_len to policy.block in prepare_inputs"
            )
    cg_key_info = step.cg_key_info
    if policy.adaptive:
        if cg_key_info is not None:
            raise ValueError("an adaptive verify step's capture key is the engine's (its drafts); declare none")
        cg_key_info = policy.capture_key(drafts)
    steps = {
        name: dataclasses.replace(s, verify=True) if drafts and _has_verify_field(type(s)) else s
        for name, s in step.steps.items()
    }
    # the step's own segments, as every resource that names none gets them
    steps[key] = VerifyStep(drafts=drafts, sampler=sampler)
    steps[sampler] = SamplerStep(apply_penalty=False, verify=True)
    if policy.proposer is not None:
        if policy.proposer in step.steps:
            raise ValueError(
                f"a verify step's proposer step is with_draft_verify's; declare none under {policy.proposer!r}"
            )
        steps[policy.proposer] = ProposeStep(drafts=drafts, verify=True)
    return SubmoduleStep(steps=steps, segments=segments, cg_key_info=cg_key_info)


def verify_captures(
    policy: DraftPolicy, walk: str, make_row: Callable[[int], Any], capture_batch_sizes: list[int] | None,
    **kwargs: Any,
) -> list:
    """The batched captures of a verifying walk: one per block the walk runs (``draft_options``),
    each keyed by ``capture_key``. ``make_row(block)`` is a template row of that span. A
    submodule returns these from ``get_accelerator_graph_configs`` for the walk."""
    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig

    return [
        BatchedAcceleratorGraphConfig(
            capture_graph_walk=walk, single_request_inputs=make_row(drafts + 1),
            capture_batch_sizes=capture_batch_sizes, additional_key_info=policy.capture_key(drafts), **kwargs,
        )
        for drafts in policy.draft_options
    ]


@cache
def _has_verify_field(step_type: type) -> bool:
    # once per step type, not per step
    return dataclasses.is_dataclass(step_type) and any(f.name == "verify" for f in dataclasses.fields(step_type))


@dataclass(frozen=True, eq=False)
class ProposeStep(ResourceStep):
    """An engine-side proposer's part of a step. A verify step's (``with_draft_verify``) stages
    ``drafts`` per row for the forward (none on an adaptive walk's plain step); a step that
    only adds to the requests' histories (a prefill, declared by the model) names the tokens
    it adds."""
    drafts: int = 0
    # request -> the token ids this step adds to its history, in order (a prompt's text)
    history: Mapping[int, torch.Tensor] = field(default_factory=dict)
    # a step of the verifying walk, which may run no drafts (an adaptive walk's plain step)
    verify: bool = False


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
    # ngram: a step drafts only when its rows are expected to keep min_gain drafts a row, and
    # otherwise decodes one token a row (``DraftPolicy.adaptive``); on by default for ngram,
    # whose lookups predict their own acceptance. A draft model drafts every step. min_gain is
    # a number or [rows, gain] points; the default is Qwen3.5-0.8B's on an H100 at 8 drafts
    # (a verify step over 1 / 8 / 32 rows cost 1.55 / 1.67 / 2.03 a step of one token)
    adaptive: bool | None = None
    min_gain: float | tuple = ((1, 0.5), (32, 1.0))

    METHODS = ("ngram", "draft_model")

    def __post_init__(self):
        if self.method not in self.METHODS:
            raise ValueError(f"draft.method {self.method!r}: one of {', '.join(self.METHODS)}")
        if self.adaptive is None:
            object.__setattr__(self, "adaptive", self.method == "ngram")
        if self.adaptive and self.method != "ngram":
            raise ValueError("draft.adaptive needs drafts whose acceptance the proposer predicts: method ngram")
        object.__setattr__(self, "min_gain", _gain_points(self.min_gain))
        if not 1 <= self.min_ngram <= self.max_ngram:
            raise ValueError(f"draft n-gram sizes must satisfy 1 <= min_ngram <= max_ngram; got "
                             f"{self.min_ngram} and {self.max_ngram}")
        if (self.method == "draft_model") != (self.model is not None):
            raise ValueError("draft.model (the drafter's checkpoint) goes with method draft_model, and only with it")
        if self.draft_sample and self.method != "draft_model":
            raise ValueError("draft.draft_sample is a draft model's setting")
        self.policy()  # checks max_drafts and min_gain

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
        return DraftPolicy(
            max_drafts=self.max_drafts, stop_ids=stop_ids, proposer=self.proposer,
            adaptive=self.adaptive, min_gain=self.min_gain, **kwargs,
        )

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
