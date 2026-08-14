
from collections.abc import KeysView
from dataclasses import dataclass, field
import itertools
from typing import Any

import torch


@dataclass(frozen=True)
class Segment:
    """One step's addition to a request's cache stream.

    A request contributes one segment per label active for it in a step;
    the batch's ordered segment list defines the layout of per-token
    arrays. ``span`` may be 0: a zero-span segment reads its stream
    without extending it (admission reserves nothing, commit is a no-op).
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
    requires_cfg: bool
    bs: int
    num_tokens: int


@dataclass(frozen=True)
class SlotLease:
    """attention `admit` gives to inform which slot to plan and replay

    has no clean channel to plan/commit/release. see O."""
    slot: int
    bucket: BucketKey | None # None -> eagre
    filler: tuple[Segment, ...]


@dataclass(frozen=True)
class StepContext:
    # engine-only
    request_ids: tuple[str, ...]
    graph_walk: str
    slot: int
    capture: bool
    # will get populated with the output of plan as previous steps
    # complete their plan stages
    plan_results: dict[str, Any] = field(default_factory=dict)
    slot_lease: SlotLease | None = None


@dataclass(frozen=True)
class SubmoduleStep:
    # cumulative step over submodule::forward for all resoures
    ctx: StepContext
    segments: tuple[Segment, ...] # authoritative ordering
    steps: dict[str, ResourceStep] # resource key -> step within cumulative

    def get(self, key: str) -> ResourceStep | None:
        return self.steps.get(key)

    def keys(self) -> KeysView[str]:
        return self.steps.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.steps

    def segments_for(self, key: str) -> tuple[Segment, ...]:
        """segments covered by one resource's step

        `segments=None` means whole batch layout"""
        step = self.steps.get(key)
        if step is None or step.segments is None:
            return self.segments
        return step.segments

    def validate(self) -> None:
        """assert each resource step's sgement are order preserving
        subsequence of total batch layout

        strictly debug; to ensure no reordering occurss"""
        for key, step in self.steps.items():
            if step.segments is None:
                continue
            it = iter(self.segments)
            for segment in step.segments:
                if not any(candidate == segment for candidate in it):
                    raise ValueError(
                        f"resource step {key!r} declares segment {segment} "
                        "which is not an order-preserving subsequence of the "
                        f"batch layout {self.segments}"
                    )


"""perhaaps move below to distinct files/location per resource kind"""

@dataclass(frozen=True)
class AttentionStep(ResourceStep):
    causal: bool = True


@dataclass(frozen=True)
class PositionStep(ResourceStep):
    # `pos_ids=None` derives from stream counters; otherwise
    # label -> ids for one step
    pos_ids: "dict[str, torch.Tensor] | torch.Tensor | None" = None
    advance: tuple[int, ...] | None = None  #`advance=None` means own rule
    # no combined_labels: positions take the packing off KV's plan output,
    # so the step declares the grouping once, on KVStep


@dataclass(frozen=True)
class SamplerStep(ResourceStep):
    apply_penalty: bool = True
    # rid -> prefill tokens for the repetition penalty
    prefill_tracked_tokens: dict[str, torch.Tensor] = field(default_factory={})


@dataclass(frozen=True)
class KVStep(ResourceStep):
    # write: bool # @nsagan: opting to remove this for now bc it's dead code
    commit: bool = True

    # e.g., for batched CFG
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict)
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()

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


@dataclass(frozen=True)
class AdmitOutcome:
    ok: bool
    ready: bool = True
    reason: AdmitFailedReason | None = None
