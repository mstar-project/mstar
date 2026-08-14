
from collections.abc import KeysView
from dataclasses import dataclass, field
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
    pos_ids: torch.Tensor | None = None  # None = derive from counters
    advance: tuple[int, ...] | None = None  # None = each segment's span


@dataclass(frozen=True)
class SamplerStep(ResourceStep):
    apply_penalty: bool = True
    # rid -> prefill tokens for the repetition penalty
    prefill_tracked_tokens: dict[str, torch.Tensor] = field(default_factory={})


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
