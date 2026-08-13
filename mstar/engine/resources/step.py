
from dataclasses import dataclass, field

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
class StepContext:
    # engine-only
    request_ids: tuple[str, ...]
    graph_walk: str
    slot: int
    capture: bool # default true or false? @nsagan: no default needed


@dataclass(frozen=True)
class SubmoduleStep:
    # cumulative step over submodule::forward for all resoures
    ctx: StepContext
    segments: tuple[Segment, ...] # authoritative ordering
    steps: dict[str, ResourceStep] # resource key -> step within cumulative

    def get(self, key: str) -> ResourceStep | None: ...


"""perhaaps move below to distinct files/location per resource kind"""

@dataclass(frozen=True)
class KVStep(ResourceStep):
    write: bool
    commit: bool
    pre_forks: tuple[tuple[str, str], ...] = ()
    post_forks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AttentionStep(ResourceStep):
    causal: bool
    batched_key: str | None


@dataclass(frozen=True)
class PositionStep(ResourceStep):
    pos_ids: torch.Tensor | None
    advance: tuple[int, ...]


@dataclass(frozen=True)
class SamplerStep(ResourceStep):
    apply_penalty: bool


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