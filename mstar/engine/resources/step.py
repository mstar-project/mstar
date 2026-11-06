
from collections.abc import Callable, Collection, Iterable, Iterator, KeysView, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain, repeat
from typing import Any, ClassVar, NamedTuple, overload

# The graph walk on a piecewise region's step context: the region declares, plans and
# commits its own step inside the enclosing forward's (see accelerator_graph_runner).
PIECEWISE_WALK = "__piecewise__"


class Segment(NamedTuple):
    """One step's addition to a request's cache stream: one row of `Segments`.

    A request contributes one segment per label active for it in a step; the
    batch's ordered segments define the layout of per-token arrays. ``span``
    may be 0: a zero-span segment reads its stream without extending it
    (admission reserves nothing, commit is a no-op).
    """
    request_id: str
    label: str
    span: int


_set = object.__setattr__


def _uniform(column: tuple, rows: int, name: str):
    """``column``'s one value when every row holds it, else ``column``."""
    if len(column) != rows:
        raise ValueError(f"Segments: {len(column)} {name} for {rows} request ids")
    return column[0] if rows and column.count(column[0]) == rows else column


class Segments(Sequence[Segment]):
    """A step's segments as columns, in row order.

    Immutable, and compared and hashed by value, so resources can share one and
    a staged step can be matched against a re-declared one. A column every row
    agrees on is always held as its one value (``labels`` a ``str``, ``spans``
    an ``int``), so asking about it is O(1): ``segments.spans == 1`` is "every
    row is one token", ``== 0`` "no row writes". An empty ``Segments`` answers
    no to both; test its truth first if that matters.

    It reads as a sequence of `Segment`s for code that wants rows, which are
    built once on first use. A hot path reads the columns or ``rows()`` instead.
    """

    __slots__ = ("request_ids", "labels", "spans", "_rows")

    request_ids: tuple[int, ...]
    labels: str | tuple[str, ...]
    spans: int | tuple[int, ...]

    def __init__(
        self, request_ids: Iterable[int], labels: str | Iterable[str] = "main", spans: int | Iterable[int] = 1,
    ):
        """``labels`` and ``spans`` are one value for every row, or one per row."""
        rids = tuple(request_ids)
        rows = len(rids)
        if not isinstance(labels, str):
            labels = _uniform(tuple(labels), rows, "labels")
        if type(spans) is not int and hasattr(spans, "__iter__"):
            spans = _uniform(tuple(spans), rows, "spans")
        if not rows:
            labels, spans = "", 0
        _set(self, "request_ids", rids)
        _set(self, "labels", labels)
        _set(self, "spans", spans)
        _set(self, "_rows", None)

    @classmethod
    def of_inputs(cls, request_ids: Iterable[int], inputs: Iterable, label: str = "main") -> "Segments":
        """The usual declaration: one row per request under ``label``, each as
        long as its input's ``input_seq_len``."""
        return cls(request_ids, label, [inp.input_seq_len for inp in inputs])

    @classmethod
    def from_rows(cls, rows: "Iterable[Segment]") -> "Segments":
        """``rows`` as columns, row order kept; a `Segments` as it is."""
        if type(rows) is cls:
            return rows
        rows = tuple(rows)
        if not rows:
            return cls(())
        out = cls(*zip(*rows, strict=False))
        # the caller's own rows, so reading them back builds nothing
        if type(rows[0]) is Segment:
            _set(out, "_rows", rows)
        return out

    @classmethod
    def concat(cls, parts: "Iterable[Segments]") -> "Segments":
        """``parts`` one after another."""
        parts = [part for part in parts if part]
        if len(parts) == 1:
            return parts[0]
        return cls(
            chain.from_iterable(part.request_ids for part in parts),
            chain.from_iterable(part.label_column() for part in parts),
            chain.from_iterable(part.span_column() for part in parts),
        )

    def label_column(self) -> tuple[str, ...]:
        labels = self.labels
        return labels if type(labels) is tuple else (labels,) * len(self.request_ids)

    def span_column(self) -> tuple[int, ...]:
        spans = self.spans
        return spans if type(spans) is tuple else (spans,) * len(self.request_ids)

    def rows(self) -> Iterator[tuple[int, str, int]]:
        """``(request_id, label, span)`` per row, as plain tuples: the rows
        without building a `Segment` per row. A one-shot iterator."""
        labels, spans = self.labels, self.spans
        return zip(
            self.request_ids,
            labels if type(labels) is tuple else repeat(labels),
            spans if type(spans) is tuple else repeat(spans),
            strict=False,
        )

    def pairs(self) -> Iterator[tuple[int, str]]:
        """``(request_id, label)`` per row: the streams the rows are on."""
        labels = self.labels
        return zip(self.request_ids, labels if type(labels) is tuple else repeat(labels), strict=False)

    def group_by_label(self) -> "dict[str, Segments]":
        """Label -> its rows in row order, labels in order of first appearance."""
        labels = self.labels
        if type(labels) is not tuple:
            return {labels: self} if self.request_ids else {}
        rows_of: dict[str, list[int]] = {}
        for row, label in enumerate(labels):
            rows_of.setdefault(label, []).append(row)
        rid_at, spans = self.request_ids.__getitem__, self.spans
        return {
            label: Segments(map(rid_at, rows), label, map(spans.__getitem__, rows) if type(spans) is tuple else spans)
            for label, rows in rows_of.items()
        }

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Segments is immutable")

    def __len__(self) -> int:
        return len(self.request_ids)

    def __bool__(self) -> bool:
        return bool(self.request_ids)

    def __iter__(self) -> Iterator[Segment]:
        rows = self._rows
        if rows is None:
            # racing threads each build the same rows; either result will do
            rows = tuple(map(Segment._make, self.rows()))
            _set(self, "_rows", rows)
        return iter(rows)

    @overload
    def __getitem__(self, i: int) -> Segment: ...
    @overload
    def __getitem__(self, i: slice) -> "Segments": ...
    def __getitem__(self, i):
        labels, spans = self.labels, self.spans
        if type(i) is slice:
            return Segments(
                self.request_ids[i],
                labels[i] if type(labels) is tuple else labels,
                spans[i] if type(spans) is tuple else spans,
            )
        return Segment(
            self.request_ids[i],
            labels[i] if type(labels) is tuple else labels,
            spans[i] if type(spans) is tuple else spans,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is Segments:
            return self is other or (
                self.spans == other.spans and self.labels == other.labels and self.request_ids == other.request_ids
            )
        # rows compare by value too, and hash as the same rows in a tuple do
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self):
        return Segments, (self.request_ids, self.labels, self.spans)

    def __repr__(self) -> str:
        return f"Segments({self.request_ids!r}, {self.labels!r}, {self.spans!r})"


NO_SEGMENTS = Segments(())


@dataclass(frozen=True)
class ResourceStep:
    # Work for one resource, for one step. Rows of `Segment`s are accepted and
    # held as columns.
    segments: Segments | None = None

    def __post_init__(self):
        segments = self.segments
        if segments is not None and type(segments) is not Segments:
            _set(self, "segments", Segments.from_rows(segments))

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

    def real_rids(self) -> frozenset:
        """``request_ids`` as a set."""
        real = self._real_rids
        if real is None:
            real = self._real_rids = frozenset(self.request_ids)
        return real

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


@dataclass(frozen=True)
class SubmoduleStep:
    steps: dict[str, ResourceStep] # resource key -> step within cumulative

    # used for convenience only: if a ResourceStep does not specify segments,
    # it gets these. Rows of `Segment`s are accepted and held as columns.
    segments: Segments | None = None

    # Matches additional_key_info in CudaGraphConfig
    cg_key_info: Any | None = None

    _ctx: StepContext = None # set by the engine

    def __post_init__(self):
        segments = self.segments
        if segments is None:
            return
        if type(segments) is not Segments:
            segments = Segments.from_rows(segments)
            _set(self, "segments", segments)
        for step in self.steps.values():
            if step.segments is None:
                _set(step, "segments", segments)

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
