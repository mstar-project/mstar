
from collections.abc import KeysView, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple


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
    request_ids: Sequence[str]
    graph_walk: str
    slot: int
    capture: bool

    # Preplan
    is_preplan: bool = False
    # will get populated with the output of plan as previous steps
    # complete their plan stages
    plan_results: dict[str, Any] = field(default_factory=dict)
    slot_lease: SlotLease | None = None
    # `request_ids` padded with dummy rids to a capture bucket's batch size;
    # None outside a captured replay, where the two are the same
    _padded_request_ids: Sequence[str] | None = None

    @property
    def padded_request_ids(self) -> Sequence[str]:
        if self._padded_request_ids is None:
            return self.request_ids
        return self._padded_request_ids

    def set_padded_rids(self, padded_rids: Sequence[str] | None):
        self._padded_request_ids = padded_rids


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

@dataclass
class AdmitFailedReason:
    message: str


@dataclass
class AllocationFailed(AdmitFailedReason):
    pages_short: int
    label: str
    request_id: str


@dataclass
class RequestOffloading(AdmitFailedReason):
    """The request's state is moving to host memory; retry once it is back.

    Distinct from `AllocationFailed` because the answer is different: nothing
    needs evicting, the caller just re-drives the step once `reload` has run.
    """
    label: str
    request_id: str


class AdmitOutcome(NamedTuple):
    ok: bool
    ready: bool = True
    reason: AdmitFailedReason | None = None


# Every resource's admit returns this on the common path, several times a
# step; nothing reads identity, so hand back one instance rather than build it.
ADMIT_OK = AdmitOutcome(ok=True, ready=True)


class FullAdmitOutcome(NamedTuple):
    """What the runner answers with: one resource's outcome, plus which
    resource gave it.

    A resource doesn't know the key it is registered under, so the runner —
    which does — names it on the way out. The caller needs it to scope an
    eviction to the resource that actually ran out.
    """
    outcome: AdmitOutcome
    failed_resource: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome.ok

    @property
    def ready(self) -> bool:
        return self.outcome.ready

    @property
    def reason(self) -> AdmitFailedReason | None:
        return self.outcome.reason


FULL_ADMIT_OK = FullAdmitOutcome(ADMIT_OK)
