
import itertools
import re
from collections.abc import KeysView
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch


class Segment(NamedTuple):
    """One step's addition to a request's cache stream.

    A request contributes one segment per label active for it in a step;
    the batch's ordered segment list defines the layout of per-token
    arrays. ``span`` may be 0: a zero-span segment reads its stream
    without extending it (admission reserves nothing, commit is a no-op).

    A NamedTuple, not a frozen dataclass: one is built per request per
    step, and the frozen dataclass's ``object.__setattr__``-per-field
    __init__ is the expensive way to do that.
    """
    request_id: str
    label: str
    span: int

@dataclass(frozen=True)
class ResourceStep:
    # Work for one resource, for one step
    segments: tuple[Segment, ...] | None = None


@dataclass(frozen=True)
class BucketKey:
    graph_walk: str
    bs: int
    num_tokens: int
    # Matches additional_key_info in CudaGraphConfig
    cg_key_info: Any | None = None

    def __str__(self) -> str:
        key = "" if self.cg_key_info is None else f",key={self.cg_key_info}"
        return f"{self.graph_walk}[bs={self.bs},tokens={self.num_tokens}{key}]"


@dataclass(frozen=True)
class SlotLease:
    """attention `admit` gives to inform which slot to plan and replay

    has no clean channel to plan/commit/release. see O."""
    slot: int
    bucket: BucketKey | None # None -> eagre


@dataclass
class StepContext:
    # engine-only. mutable: the engine pads `request_ids` and fills the lease
    # in once a slot is chosen, on the context the batch already carries
    request_ids: tuple[str, ...]
    graph_walk: str
    slot: int
    capture: bool

    # Preplan
    is_preplan: bool = False
    # will get populated with the output of plan as previous steps
    # complete their plan stages
    plan_results: dict[str, Any] = field(default_factory=dict)
    slot_lease: SlotLease | None = None


@dataclass(frozen=True)
class SubmoduleStep:
    steps: dict[str, ResourceStep] # resource key -> step within cumulative

    # used for convenience only: if a ResourceStep does not specify segments,
    # it gets this list
    segments: list[Segment] | None = None

    # Matches additional_key_info in CudaGraphConfig
    cg_key_info: Any | None = None

    _ctx: StepContext = None # set by the engine

    def __post_init__(self):
        if self.segments is None:
            return
        for step in self.steps.values():
            if step.segments is None:
                object.__setattr__(step, "segments", self.segments)

    @property
    def ctx(self):
        return self._ctx

    def set_ctx(self, ctx):
        # the step is frozen so a submodule can't rewrite its own declaration;
        # the engine still owns this one field
        object.__setattr__(self, "_ctx", ctx)

    def get(self, key: str) -> ResourceStep | None:
        return self.steps.get(key)

    def keys(self) -> KeysView[str]:
        return self.steps.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.steps

    def post_sample(self) -> "list[tuple[str, PostSample]]":
        """(resource key, spec) for every logits key this step asked the engine
        to sample, in declaration order. The key names the sampler resource, so
        a node with more than one sampler stays unambiguous."""
        return [
            (key, spec)
            for key, step in self.steps.items()
            if isinstance(step, SamplerStep)
            for spec in step.post_sample
        ]

"""perhaps move below to distinct files/location per resource kind"""

@dataclass(frozen=True)
class KVStep(ResourceStep):
    # write: bool # @nsagan: opting to remove this for now bc it's dead code
    commit: bool = True

    # e.g., for batched CFG
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict)
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AttentionStep(ResourceStep):
    causal: bool = True


@dataclass(frozen=True)
class PositionStep(ResourceStep):
    # `pos_ids=None` derives from stream counters; otherwise
    # label -> ids for one step
    pos_ids: "dict[str, torch.Tensor] | torch.Tensor | None" = None
    advance: tuple[int, ...] | None = None  # `advance=None` means own rule
    # no combined_labels: positions take the packing off KV's plan output,
    # so the step declares the grouping once, on KVStep


# A forward's batch-wide outputs sit at the top level of its output dict under
# a `__sentinel__` name; anything else is a per-rid entry.
_BATCHED_KEY = re.compile(r"^__\w+__$")


@dataclass
class PostSample:
    """One logits key the forward hands back for the engine to sample.

    ``batched``: ``in_key`` is a batch-wide ``[bs, vocab]`` tensor under the
    top-level sentinel convention, so the whole batch samples in one call;
    otherwise it is a per-rid entry, gathered in batch order first. Defaults to
    whichever the name implies.

    Declaring a key here also makes it the sampler's: it is consumed, never
    passed through to the node's outputs. Several specs may share an
    ``out_key`` (a forward that emits both a batched and a per-rid view of the
    same logits) — the batched one wins when the forward actually emitted it.
    """
    in_key: str
    batched: bool = None
    out_key: str = "new_token"

    def __post_init__(self):
        if self.batched is None:
            self.batched = bool(_BATCHED_KEY.match(self.in_key))


@dataclass(frozen=True)
class SamplerStep(ResourceStep):
    apply_penalty: bool = True
    # rid -> prefill tokens for the repetition penalty
    prefill_tracked_tokens: dict[str, torch.Tensor] = field(default_factory=dict)
    post_sample: list[PostSample] = field(default_factory=list)


# moved here
def group_by_plan_label(
    segments: tuple[Segment, ...],
    combined_labels: dict[tuple[str, ...], str],
) -> dict[str, list[Segment]]:
    """Segments per plan label in order of packed forward view

    combined plan concats source labels in label major order. standalone keeps
    og batch order. KV should be sole producer of this ordering and eveyrone else
    will read `plan` output of KV

    NOTE: combined key with a source label with no segments in step will cause KeyError
    """
    label_to_segments: dict[str, list[Segment]] = {}
    for segment in segments:
        label_to_segments.setdefault(segment.label, []).append(segment)

    sources = set(itertools.chain.from_iterable(combined_labels))
    grouped: dict[str, list[Segment]] = {
        plan_label: [
            segment
            for label in source_labels
            for segment in label_to_segments[label]
        ]
        for source_labels, plan_label in combined_labels.items()
    }

    grouped.update(
        (label, label_segments)
        for label, label_segments in label_to_segments.items()
        if label not in sources
    )
    return grouped


@dataclass
class AdmitFailedReason:
    message: str


@dataclass
class AllocationFailed(AdmitFailedReason):
    pages_short: int
    label: str
    request_id: str


class AdmitOutcome(NamedTuple):
    ok: bool
    ready: bool = True
    reason: AdmitFailedReason | None = None


# Every resource's admit returns this on the common path, several times a
# step; nothing reads identity, so hand back one instance rather than build it.
ADMIT_OK = AdmitOutcome(ok=True, ready=True)
