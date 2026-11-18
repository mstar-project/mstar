
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, KeysView, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, ClassVar, NamedTuple

# The graph walk on a piecewise region's step context: the region declares, plans and
# commits its own step inside the enclosing forward's (see accelerator_graph_runner).
PIECEWISE_WALK = "__piecewise__"


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
    # source labels -> the plan label one forward packs them under (batched CFG, say); the
    # resources that plan the step's layout (KV, the sequence shard) read it. Out of the hash, so a
    # step without other unhashable fields stays hashable
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict, kw_only=True, hash=False)

@dataclass(frozen=True)
class BucketKey:
    graph_walk: str
    bs: int
    num_tokens: int
    # Matches additional_key_info in CudaGraphConfig
    cg_key_info: Any | None = None
    # the keys of the resources this capture leaves running lean
    # (`Resource.lean_capture`); empty for the plain bucket every step can replay
    lean: frozenset[str] = frozenset()

    def __str__(self) -> str:
        key = "" if self.cg_key_info is None else f",key={self.cg_key_info}"
        lean = f",lean={'+'.join(sorted(self.lean))}" if self.lean else ""
        return f"{self.graph_walk}[bs={self.bs},tokens={self.num_tokens}{key}{lean}]"


def lean_subsets(keys: Sequence[str]) -> tuple[frozenset[str], ...]:
    """The non-empty subsets of ``keys``, largest first, those of one size in the
    order of ``keys``: the lean twins a step that leaves all of ``keys`` lean may
    replay, in the order it tries them."""
    return tuple(
        frozenset(subset) for size in range(len(keys), 0, -1) for subset in combinations(keys, size)
    )


@dataclass(frozen=True)
class SlotLease:
    """attention `admit` gives to inform which slot to plan and replay

    has no clean channel to plan/commit/release. see O."""
    slot: int
    bucket: BucketKey | None # None -> eagre


class SettingValue(Enum):
    UNSET = 0
    SET = 1
    PARTIAL = 2


class PerReqSetting(NamedTuple):
    """A step's flag that may hold for only some of its rows."""
    value: SettingValue
    # the rows it holds for; set iff ``value`` is PARTIAL
    set_rids: frozenset[int] | None = None

    @classmethod
    def over(
        cls, rids: Collection[int], holds: Callable[[int], bool],
    ) -> "PerReqSetting":
        set_rids = frozenset(rid for rid in rids if holds(rid))
        if len(set_rids) == len(rids):
            return ALL_SET
        if not set_rids:
            return NONE_SET
        return cls(SettingValue.PARTIAL, set_rids)

    def holds_for(self, rid: int) -> bool:
        if self.value is SettingValue.PARTIAL:
            return rid in self.set_rids
        return self.value is SettingValue.SET


ALL_SET = PerReqSetting(SettingValue.SET)
NONE_SET = PerReqSetting(SettingValue.UNSET)


@dataclass
class StepContext:
    # engine-only. mutable: the engine pads `request_ids` and fills the lease
    # in once a slot is chosen, on the context the batch already carries
    request_ids: Sequence[int]
    graph_walk: str
    slot: int
    capture: bool

    # Preplan
    is_preplan: bool = False
    # the prefix a resource's own plan laps record under, as the runner times
    # the plan: res.pre_plan, res.plan (promoting a pre-plan) or res.plan_fresh
    plan_phase: str = "res.plan"
    # will get populated with the output of plan as previous steps
    # complete their plan stages
    plan_results: dict[str, Any] = field(default_factory=dict)
    slot_lease: SlotLease | None = None
    # label -> the slot a piecewise region of this node holds for this step,
    # for regions that opt into leasing ahead of the declaration. The region
    # still declares and plans its own work; this only says who owns which
    # resource, so `declare_step` can leave those to it.
    piecewise_leases: "Mapping[str, SlotLease]" = field(default_factory=dict)
    # `request_ids` padded with dummy rids to a capture bucket's batch size;
    # None outside a captured replay, where the two are the same. Padding rows
    # carry negative handles, so this stays homogeneous with `request_ids`.
    _padded_request_ids: Sequence[int] | None = None
    # ``request_ids`` as a set, built on first use after the padded list is
    # set: the per-row tests below run for every row of every resource on two
    # threads, and a list scan made each of them linear in the batch.
    _real_rids: frozenset | None = field(default=None, repr=False)
    # rid -> real walk under a combined ``graph_walk``; empty otherwise
    request_walks: Mapping[int, str] = field(default_factory=dict)
    # plan prefill attention even when every row is one token: a packed
    # capture's forward reads per-row token offsets
    force_prefill: bool = False
    # How many times this step's capture replays: the host plans once and the
    # step counts for this many iterations. The worker asks, and `StepRunner`
    # narrows it to what the step's resources allow (`Resource.max_repeats`).
    repeats: int = 1
    # the node's `ResidentPlanner` when this step's capture reads the resident
    # tables: a resource then stages nothing per row (see `resources/resident.py`)
    resident: Any = None
    # what that planner staged for this step
    resident_plan: Any = None
    # rid -> input tokens a chunked row runs after this step
    chunk_tokens_after: Mapping[int, int] = field(default_factory=dict)
    # the rows whose walk grows its state by those tokens, so a resource may
    # reserve them now (``ChunkingPolicy.reserve_whole_prefill``). Check it
    # before reserving by ``chunk_tokens_after``.
    reserve_whole_prefill: PerReqSetting = ALL_SET
    # rows that recompute KV their request already had: the step throws their
    # samples away, and they change no per-request state but the KV
    rebuild_rids: frozenset[int] = frozenset()
    # the keys of the resources that run lean this step: its lease's
    # (`BucketKey.lean`), or for an eager step those whose `lean_for` holds for
    # every row. Set by the engine; a piecewise region's step takes its bucket's.
    lean: frozenset[str] = frozenset()

    @property
    def replays(self) -> bool:
        """Whether this step replays a captured graph, so its forward runs no Python."""
        return self.slot_lease is not None and not self.capture

    def walk_of(self, rid: int) -> str:
        return self.request_walks.get(rid, self.graph_walk)

    @property
    def padded_request_ids(self) -> Sequence[int]:
        if self._padded_request_ids is None:
            return self.request_ids
        return self._padded_request_ids

    def set_padded_rids(self, padded_rids: Sequence[int] | None):
        self._padded_request_ids = padded_rids
        self._real_rids = None

    def is_real_row(self, rid: str) -> bool:
        """Whether ``rid`` is one of this step's real requests."""
        real = self._real_rids
        if real is None:
            real = self._real_rids = frozenset(self.request_ids)
        return rid in real

    def real_rows(self) -> frozenset | None:
        """``request_ids`` as a set when the step has padding rows, else None
        (no row is padding)."""
        if self._padded_request_ids is None:
            return None
        real = self._real_rids
        if real is None:
            real = self._real_rids = frozenset(self.request_ids)
        return real

    def is_padding_row(self, rid: str) -> bool:
        """Whether ``rid`` is a replay's padding row rather than a real request.

        It carries the template span so shapes match the capture, but a resource
        should reserve, address and commit nothing on its behalf."""
        if self._padded_request_ids is None:
            return False
        real = self._real_rids
        if real is None:
            real = self._real_rids = frozenset(self.request_ids)
        return rid not in real

    def set_piecewise_leases(self, leases: "Mapping[str, SlotLease]"):
        self.piecewise_leases = leases


class StepHandle(ABC):
    """What a layer holds for something a step's ``state`` lists. The engine opens it around each step's calls."""

    @abstractmethod
    def begin_step(self, step: "SubmoduleStep") -> None:
        """Answer for ``step`` from here on. Raises if the handle is already open."""

    @abstractmethod
    def check_step(self, step: "SubmoduleStep") -> None:
        """Refuse the step's calls as a whole, before any resource commits them."""

    @abstractmethod
    def end_step(self, step: "SubmoduleStep") -> None:
        """Close the handle, on any path out of the step."""

    @abstractmethod
    def commit_step(self, step: "SubmoduleStep") -> None:
        """Apply what the handle itself holds of a step the resources just committed."""


@dataclass(frozen=True)
class SubmoduleStep:
    # resource key -> step within cumulative
    steps: dict[str, ResourceStep] = field(default_factory=dict)

    # used for convenience only: if a ResourceStep does not specify segments,
    # it gets this list
    segments: list[Segment] | None = None
    # the labels this step's forward packs together, declared once here: every ResourceStep
    # that doesn't set its own gets it, and a ResourceStep that sets any other (or any, when this is empty)
    # is refused, so the KV plan and the sequence shard agree
    combined_labels: dict[tuple[str, ...], str] = field(default_factory=dict, kw_only=True)

    # Matches additional_key_info in CudaGraphConfig
    cg_key_info: Any | None = None

    _ctx: StepContext = None # set by the engine

    # declared array or sampler -> what this step does to it, which the engine
    # expands into ``steps`` (`mstar.engine.state.adapters.state_steps`)
    state: Mapping[str, Any] = field(default_factory=dict)
    # the handles of what ``state`` lists, and of the node's sequence shard, set by that expansion
    state_handles: tuple[StepHandle, ...] = field(default=(), init=False)

    def __post_init__(self):
        if not self.combined_labels:
            for key, step in self.steps.items():
                if step.combined_labels:
                    raise ValueError(
                        f"step {key!r} declares combined_labels {step.combined_labels}, its submodule step none: "
                        "declare them once, on the SubmoduleStep"
                    )
            if self.segments is None:
                return
        for key, step in self.steps.items():
            if step.segments is None and self.segments is not None:
                object.__setattr__(step, "segments", self.segments)
            if not self.combined_labels:
                continue
            if not step.combined_labels:
                object.__setattr__(step, "combined_labels", self.combined_labels)
            elif step.combined_labels != self.combined_labels:
                raise ValueError(
                    f"step {key!r} declares combined_labels {step.combined_labels}, its submodule step "
                    f"{self.combined_labels}: declare them once, on the SubmoduleStep"
                )

    @property
    def ctx(self):
        return self._ctx

    def set_ctx(self, ctx):
        # the step is frozen so a submodule can't rewrite its own declaration;
        # the engine still owns this one field
        object.__setattr__(self, "_ctx", ctx)

    def set_state_handles(self, handles: tuple[StepHandle, ...]) -> None:
        object.__setattr__(self, "state_handles", handles)

    def begin_calls(self) -> None:
        for handle in self.state_handles:
            handle.begin_step(self)

    def check_calls(self) -> None:
        for handle in self.state_handles:
            handle.check_step(self)

    def commit_calls(self) -> None:
        for handle in self.state_handles:
            handle.commit_step(self)

    def end_calls(self) -> None:
        for handle in self.state_handles:
            handle.end_step(self)

    def get(self, key: str) -> ResourceStep | None:
        return self.steps.get(key)

    def keys(self) -> KeysView[str]:
        return self.steps.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.steps

@dataclass
class AdmitFailedReason:
    message: str
    # A terminal refusal comes back on every retry, so the caller fails the
    # request instead of re-queueing the step.
    terminal: ClassVar[bool] = False


@dataclass
class AllocationFailed(AdmitFailedReason):
    pages_short: int
    label: str
    request_id: str
    # How many of the step's requests, in its order, the free pages held, and
    # how many the pool would hold with nothing else resident, out of the
    # ``rows_sized`` it counted; None when the resource does not say. The
    # worker sizes the retry from them. A row the prefix cache served whole
    # runs no step, so it is in the worker's batch but not in ``rows_sized``.
    rows_fit: int | None = None
    rows_fit_alone: int | None = None
    rows_sized: int | None = None


@dataclass
class RequestOffloading(AdmitFailedReason):
    """The request's state is moving to host memory; retry once it is back.

    Distinct from `AllocationFailed` because the answer is different: nothing
    needs evicting, the caller just re-drives the step once `reload` has run.
    """
    label: str
    request_id: str

@dataclass
class AdmitRuntimeError(AdmitFailedReason):
    """A resource cannot serve this request at all.

    Terminal, unlike the two above: no eviction and no reload makes it go
    away, so the caller's answer is to fail the request, not to retry it.
    ``request_ids`` names the requests at fault; empty means the refusal is
    about the step as a whole.
    """
    terminal: ClassVar[bool] = True
    request_ids: frozenset[str] = frozenset()
    # the HTTP status the requests fail with: 400 when the request itself is
    # what can't be served (a grammar that doesn't compile, say)
    status: int = 500


class AdmitOutcome(NamedTuple):
    ok: bool
    ready: bool = True
    reason: AdmitFailedReason | None = None


# Every resource's admit returns this on the common path, several times a
# step; nothing reads identity, so hand back one instance rather than build it.
ADMIT_OK = AdmitOutcome(ok=True, ready=True)
# `Resource.ask`'s answer for a request that waits for the resource's capacity
ASKS = AdmitOutcome(ok=True, ready=False)


class FullAdmitOutcome(NamedTuple):
    """What the runner answers with: one resource's outcome, plus which
    resource gave it.

    A resource doesn't know the key it is registered under, so the runner —
    which does — names it on the way out. The caller needs it to scope an
    eviction to the resource that actually ran out.
    """
    outcome: AdmitOutcome
    failed_resource: str | None = None
    # waits for capacity on its node, until `StepRunner.admit_waiting` admits it
    queued: bool = False
    # dropped on its node: not ready until a rebuild from its token ids lands
    rebuild: bool = False

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
# admitted, but something it needs has not landed yet: retry, don't fail
FULL_ADMIT_NOT_READY = FullAdmitOutcome(AdmitOutcome(ok=True, ready=False))
FULL_ADMIT_QUEUED = FullAdmitOutcome(AdmitOutcome(ok=True, ready=False), queued=True)
FULL_ADMIT_REBUILD = FullAdmitOutcome(AdmitOutcome(ok=True, ready=False), rebuild=True)
