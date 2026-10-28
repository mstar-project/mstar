"""Per-request LoRA adapters resident in a fixed set of device slots.

The pool owns the slot weights, decides which adapter sits in which slot, and says per step
which slot each token addresses; a ``LoraBackend`` runs the kernels against that.

Slot 0 is the zero adapter (rank 0, scaling 0) and is never handed out: base-model rows and a
capture bucket's padding rows address it, and kernels skip it. A request takes the slot of
its adapter on its first admit and holds it until it is removed, so a slot whose refcount is
zero is addressed by no live request. That is what makes it safe to overwrite: the worker
defers ``remove_request`` until no in-flight step references the request, so a refcount-0
slot is read by no step still on the GPU, and a slot taken by an admit that was rolled back
was never read at all.

Under tensor parallelism every rank of a lockstep instance keeps its own pool and makes
the same slot decisions with no journal: they depend only on the order of admits, which
the leader's broadcast fixes, and on which teardowns landed between two admits, which the
worker fixes by landing each between the same two admits on every rank (``after_tp_seq``).
The one decision that can differ per rank, whether an adapter loaded, is agreed across the
instance before anything is installed.

A full pool is backpressure, not a refused step: ``admit_retrieve`` keeps a request out of
the batches the scheduler builds until a slot can take its adapter, so rows already running
never lose a step to it. Only the rank that chooses batches screens (a lockstep instance's
leader); the admit refusal stays as a backstop.

A spec over several nodes (an LLM and its vision encoder) builds a pool per node
(``LoraSpec.for_node``), and each screens only its own node's rows. A request holds a slot in each until it
is removed (none in a pool whose targets its adapter has nothing for), but where the
graph runs the encoder once, before the LLM (Qwen3.5), it takes them in that order and a
row holding an LLM slot never waits on the encoder's pool, so the two pools' waits
cannot form a cycle.

Files are read on the pool's reader thread, never on the GPU thread or under the lock: a
request names an adapter, and every rank starts reading it into the host tier as it ingests
the request; the leader's screen holds the request back until its own copy is in. A
follower whose copy is still in flight when the leader's step admits waits for it outside
the lock.

Adapters are also loaded and unloaded at runtime (``load_adapter`` / ``unload_adapter``),
which each rank applies whenever its control message arrives, not at a seq every rank
agrees on. That is safe because neither touches what an admit decides from (slot
refcounts, which slot holds an adapter, the LRU clock): a load only fills the host tier,
and an unload only forgets the adapter's name, its host copy and its name -> slot entry,
once no live request names it. An admit consults that entry only for a request naming the
adapter, and the next such request is ingested after a reload, which reaches every rank
after the unload did. The slot keeps the old weights, unheld, until LRU eviction reuses it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from mstar.engine.resources.base import CGSlotKey, EngineResourceInfo, Resource
from mstar.engine.resources.lora.backend import LoraBackend, make_backend
from mstar.engine.resources.lora.config import (
    LoraReqConfig,
    LoraSpec,
    check_expert_ids,
    describe_load_error,
)
from mstar.engine.resources.lora.store import LoadedAdapter, LoraAdapterStore
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    PIECEWISE_WALK,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    ResourceStep,
    StepContext,
)
from mstar.utils.h2d import H2DMirror, PinnedStager

logger = logging.getLogger(__name__)

ZERO_SLOT = 0
_LABEL = "lora"
_NOT_READY = AdmitOutcome(ok=True, ready=False)


@dataclass(slots=True)
class SlotState:
    index: int
    adapter: str | None = None
    # live requests holding the slot; only a slot at 0 may be given another adapter
    refcount: int = 0
    # the pool's admit count when the refcount last reached 0: eviction picks the oldest.
    # Admits, not wall clock or commits: every rank of a lockstep instance releases a
    # request between the same two admits, while a commit may land on either side of it.
    last_used_step: int = -1


@dataclass(slots=True)
class _Wait:
    """A row held back for want of a slot: since when (in the pool's admits), and the
    pool's screen count when it was last screened."""

    since: int
    screened: int


class _Served(NamedTuple):
    """Where an adapter is read from, and what its weights must digest to."""

    path: str
    digest: bytes


@dataclass(frozen=True)
class LoraAddressing:
    """Which slot each token of this step addresses, grouped by slot: the pool's plan
    output, and all a backend reads.

    Views of one fixed device buffer per capture slot (or of a reused eager buffer), sized
    to this step's tokens, which are packed in segment order. A token past every slot's
    range (a capture bucket's tail) addresses none.

    A target that sees other rows than the step's tokens (one per segment, or one per
    ``token_fold`` tokens; ``LoraTarget.row_grouping``) reads an addressing of its own,
    whose "tokens" are those rows.
    """

    token_order: torch.Tensor   # [tokens] int32, token indices grouped by slot (stable)
    slot_indptr: torch.Tensor   # [num_slots + 1] int32, each slot's range of token_order
    token_slot: torch.Tensor    # [tokens] int32, each token's slot (0 past every range)


class ExpertRows(NamedTuple):
    """How one MoE layer's (token, top-k) rows address its experts targets this step
    (``LoraLayerDelta.expert_rows``)."""

    ids: torch.Tensor     # [tokens, top_k] int32 combined ids; past every factor for none
    scale: torch.Tensor   # [tokens], each token's slot's scaling
    alignment: Any        # LoraBackend.align_experts of ids, shared by every GEMM over them


class LoraLayerDelta:
    """``y += delta(x)`` for one target linear at one stack layer, over the current step.

    A layer computes its base output, then calls this before anything consumes ``y`` —
    for a row-parallel linear, before its all-reduce. Rows on the base model address the
    zero adapter, which the kernels skip; an idle step (``StepContext.idle``: no row names
    an adapter) returns ``y`` untouched (``LoraAdapterPool.active``). The layer's slot
    weights are views resolved once, here; a call reads only the step's addressing.
    """

    def __init__(self, pool: "LoraAdapterPool", target: str, layer_idx: int):
        self._pool = pool
        self._backend = pool.backend
        self.a, self.b = pool.weights(target, layer_idx)
        self.ranks, self.scalings = pool.layer_ranks(target, layer_idx), pool.scalings
        declared = pool.targets[target]
        self.out_slices = declared.out_slices
        self._grouping = declared.row_grouping
        # an embedding's input is token ids, and its shrink a gather
        self._embedding = declared.kind == "embedding"
        # an experts target's factors, one per (slot, expert) past the zero adapter, as
        # the combined ids of `expert_rows` address them
        self.experts = declared.experts
        if self.experts > 1:
            self.a, self.b = self.a.flatten(0, 1), self.b.flatten(0, 1)

    def _live(self) -> bool:
        """Whether this step adds deltas; raises outside a planned step's compute."""
        if self._pool.outside_compute() and not torch.compiler.is_compiling():
            raise RuntimeError(
                "a LoRA delta was called outside its node's planned step (e.g. from "
                "prepare_inputs, or on an output cached from an earlier step): it would "
                "address another step's rows. Call the module's base computation there "
                "and add the delta in preprocess or the forward"
            )
        return self._pool.active

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.experts > 1:
            raise ValueError("an experts target's delta is per (token, top-k) row: add it with `grouped_`")
        if not self._live():
            # an idle step: no row names an adapter, so nothing is launched
            return y
        if self._embedding:
            x = x.reshape(-1)
        else:
            # in the weights' dtype, as autocast hands the base linear its input (a tower
            # that keeps its residual stream in fp32, e.g. Qwen3.5's vision blocks)
            x = x.reshape(-1, x.shape[-1]).to(self.a.dtype)
        self._backend.apply_(
            x, y.view(-1, y.shape[-1]), self.a, self.b,
            self._pool.addressing(self._grouping), self.ranks, self.scalings,
            self.out_slices, self._embedding,
        )
        return y

    def grouped_(
        self, x: torch.Tensor, y: torch.Tensor, ids: torch.Tensor, row_weight: torch.Tensor,
        top_k: int, alignment=None,
    ) -> None:
        """An experts target's delta, ``LoraBackend.apply_experts_`` over this layer's
        factors: ``y [rows, out]`` += ``row_weight`` times each row's delta of ``x [rows
        // top_k, in]``, for the combined ids ``ids`` of ``expert_rows`` (``alignment``
        theirs, or None)."""
        if self.experts == 1:
            raise ValueError("`grouped_` adds an experts target's delta; call a linear's delta directly")
        self._backend.apply_experts_(
            x.to(self.a.dtype), y, self.a, self.b, ids, row_weight, top_k, self.out_slices, alignment,
        )

    def expert_rows(self, topk_ids: torch.Tensor, others: Sequence[LoraLayerDelta] = ()) -> ExpertRows | None:
        """The rows of ``topk_ids`` (``[tokens, top_k]``, this rank's expert ids) for this
        experts delta and ``others``, the same MoE layer's other experts deltas; None on
        an idle step.

        Each row addresses its token's slot's A and B for its expert through one combined
        id ``(slot - 1) * experts + expert``. A row on the zero adapter, on a slot whose
        adapter adapts none of these at this layer, or on a skipped (sentinel) expert gets
        the id past every factor, which no GEMM reads.
        """
        if not self._live():
            return None
        token_slot = self._pool.addressing(1).token_slot
        if token_slot.shape[0] != topk_ids.shape[0]:
            raise ValueError(
                f"LoRA experts see {topk_ids.shape[0]} tokens but the step declared "
                f"{token_slot.shape[0]}; the segments' spans must add up to the tokens routed"
            )
        slot = token_slot.long()
        adapted = self.ranks[slot] > 0
        for other in others:
            adapted = adapted | (other.ranks[slot] > 0)
        own = adapted[:, None] & (topk_ids >= 0) & (topk_ids < self.experts)
        none = self.a.shape[0]
        ids = torch.where(own, (slot - 1)[:, None] * self.experts + topk_ids, none).to(torch.int32)
        return ExpertRows(ids, self.scalings[slot], self._backend.align_experts(ids, none))


# int32 entries per alignment unit: each array starts on a 16-byte boundary, which
# compiled kernels reading them as inputs expect
_ALIGN = 4


def _aligned(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


class _Section(NamedTuple):
    """Offsets of one row grouping's arrays: ``count`` rows, their order grouped by slot
    (``token_order``), each slot's range of it (``slot_indptr``), and each row's slot
    (``token_slot``)."""

    count: int
    order: int
    slot: int
    token_slot: int


class _Layout(NamedTuple):
    """Offsets of the addressing arrays in one packed int32 buffer, for capacities of
    ``rows`` rows and ``tokens`` tokens: a section per row grouping."""

    rows: int
    tokens: int
    sections: tuple[_Section, ...]
    size: int

    @classmethod
    def of(cls, rows: int, tokens: int, num_slots: int, groupings: tuple[int, ...]) -> "_Layout":
        at = 0
        sections = []
        for grouping in groupings:
            count = rows if grouping == 0 else tokens // grouping
            order = at
            slot = order + _aligned(count)
            token_slot = slot + _aligned(num_slots + 1)
            at = token_slot + _aligned(count)
            sections.append(_Section(count, order, slot, token_slot))
        return cls(rows, tokens, tuple(sections), at)


class _Buffers:
    """The device buffer a plan stages its addressing into, in one H2D copy.

    A capture slot's buffer has one layout for life, sized to its bucket, so the views a
    graph recorded stay valid. The eager buffer is laid out per step, packed to that
    step's rows and tokens, and copies only what it uses.
    """

    __slots__ = ("data", "layout", "mirror", "planned")

    def __init__(self, size: int, device: torch.device, layout: _Layout | None = None):
        self.data = torch.zeros(size, dtype=torch.int32, device=device)
        self.layout = layout
        self.mirror = H2DMirror()
        # (rows' slots and spans, the addressing staged for them) of the last plan here
        self.planned: tuple[tuple[bytes, bytes], dict[int, LoraAddressing]] | None = None


class LoraAdapterPool(Resource):
    # holds no per-token state, so a request whose prompt prefix was served from the cache
    # runs here exactly as one that computed it
    prefix_skip_safe = True
    # a base-model request carries no config here (`LoraSpec.request_config`), so a step
    # of them runs a graph captured without the deltas
    idle_capture = True
    # a node's rows are its segments' tokens, so it declares nothing for the pool
    steps_from_segments = True

    @classmethod
    def build(cls, spec: LoraSpec, info: EngineResourceInfo) -> "LoraAdapterPool":
        rank, world_size, agree, screens = 0, 1, None, True
        joint = info.joint_comm_group
        if joint is not None:
            if joint.sp_group.world_size > 1:
                raise ValueError(
                    f"LoRA resource {spec.resource_key!r} serves a node that runs sequence "
                    f"parallelism (SP={joint.sp_group.world_size}), which LoRA does not "
                    "support: its rows are laid out per request, while each SP rank holds "
                    "a shard of the tokens. Serve this node without SP, or drop the "
                    "'lora:' block"
                )
            # the base linears split over the TP group
            rank, world_size = joint.tp_group.rank, joint.tp_group.world_size
            if joint.world_size > 1:
                def agree(flag: bool) -> bool:
                    return bool(joint.agree([int(flag)])[0])
                # the instance's leader picks every batch; the others run what it picked
                screens = joint.rank == 0
        pool = cls(
            spec, info.device, rank=rank, world_size=world_size, dtype=info.compute_dtype,
            agree=agree, screens=screens,
        )
        # the engine brackets every planned step's compute (`enter_compute`), so a delta
        # called outside one is a bug from the start
        pool._scoped = True
        return pool

    def __init__(
        self,
        spec: LoraSpec,
        device: torch.device | str,
        rank: int = 0,
        world_size: int = 1,
        backend: LoraBackend | None = None,
        dtype: torch.dtype | None = None,
        agree: Callable[[bool], bool] | None = None,
        screens: bool = True,
    ):
        """``dtype`` is the nodes' compute dtype, which the slot weights take unless the
        spec names one. ``agree`` ANDs a flag across the pool's lockstep instance (a
        collective); None on a single rank. ``screens`` says whether this rank chooses
        batches (see ``admit_retrieve``): False on a lockstep instance's followers."""
        self.spec = spec
        self._device = torch.device(device)
        self.dtype = spec.dtype or dtype
        if self.dtype is None:
            raise ValueError(
                f"LoRA resource {spec.resource_key!r} has no dtype: its nodes run outside "
                "autocast, so the spec must name the dtype of their weights"
            )
        self.backend = backend if backend is not None else make_backend(spec.backend, self._device)
        self._agree = agree
        self._screens = screens
        self.targets = {name: t.shard(world_size) for name, t in spec.targets.items()}
        num_slots, max_rank = spec.num_slots, spec.max_rank
        # the row groupings the targets read; per token always, first
        self._groupings = (1, *sorted({t.row_grouping for t in self.targets.values()} - {1}))

        check_expert_ids(spec.resource_key, spec.max_slots, self.targets)  # as the spec checked it
        self._layer_rows: dict[str, dict[int, int]] = {}
        self._a: dict[str, torch.Tensor] = {}
        self._b: dict[str, torch.Tensor] = {}
        # [target layers, slots]: a slot's rank, or 0 where its adapter's delta is zero
        # (a module it does not adapt), so the kernels skip that layer for its rows
        self._ranks: dict[str, torch.Tensor] = {}
        for name, target in self.targets.items():
            layers = target.layers if target.layers is not None else range(spec.num_layers)
            self._layer_rows[name] = {layer: row for row, layer in enumerate(layers)}
            rows = len(self._layer_rows[name])
            # an experts target holds no zero adapter (its ids skip it) and a pair per expert
            lead = (rows, num_slots) if target.experts == 1 else (rows, spec.max_slots, target.experts)
            self._a[name] = torch.zeros(
                *lead, target.num_slices * max_rank, target.in_features,
                dtype=self.dtype, device=self._device,
            )
            self._b[name] = torch.zeros(
                *lead, target.out_features, max_rank,
                dtype=self.dtype, device=self._device,
            )
            self._ranks[name] = torch.zeros(rows, num_slots, dtype=torch.int32, device=self._device)
        self.scalings = torch.zeros(num_slots, dtype=torch.float32, device=self._device)

        self._slots = [SlotState(index=i) for i in range(num_slots)]
        self._resident: dict[str, int] = {}  # adapter -> slot
        # adapters with nothing for this spec's targets (rank 0 here, e.g. a text-only
        # adapter on a vision tower's pool): their rows address the zero adapter and
        # hold no slot. Known from the first admit that read one, on every rank alike.
        self._empty: set[str] = set()
        unpinned = sorted(set(spec.adapters) - set(spec.adapter_digests))
        if unpinned:
            # each rank would serve whatever its own read finds
            raise ValueError(
                f"LoRA adapters {unpinned} have no pinned digest: the API server resolves "
                "the serving config's adapters (resolve_lora_adapters)"
            )
        # adapter -> where it is read from: those a new request may name, and those
        # unloaded while live requests still named them, kept until the last of those
        # requests is removed
        self._serving: dict[str, _Served] = {
            name: _Served(path, spec.adapter_digests[name])
            for name, path in spec.adapters.items()
        }
        self._retiring: dict[str, _Served] = {}
        self._store = LoraAdapterStore(spec, rank=rank, world_size=world_size, dtype=self.dtype)
        # (target, layer) -> its bound delta, one per binding so compiled code guards on it
        self._layer_deltas: dict[tuple[str, int], LoraLayerDelta] = {}
        # Depth of the engine's compute scopes (a planned step's preprocess and forward,
        # a capture), and whether the engine drives them (an engine-built pool, or one a
        # scope was entered on): then a delta called outside one raises, since
        # `_current` still holds the last step's rows.
        self._compute_depth = 0
        self._scoped = False

        # Handle-keyed request state, all purged by `remove_request`.
        self._adapter_of: dict = {}   # rid -> adapter name, None for the base model
        self._row_slot: dict = {}     # rid -> slot held, read by `plan`
        self._awaiting: set = set()   # rids with an adapter but no slot yet
        self._admit_taken: dict = {}  # rid -> slot the live admit took, for rollback
        # rid -> adapter: awaiting rows `admit_retrieve` let through, counted against the
        # free slots until their admit takes one
        self._claims: dict = {}
        self._claim_count: dict[str, int] = {}  # adapter -> rows claiming it
        # rid -> when `admit_retrieve` first held it back for want of a slot; insertion
        # order is age order, so the first entry is the longest waiting
        self._waiting_since: dict = {}
        self._screen_count = 0  # screens run, the clock that tells a row out of the scans
        # (starving adapter, slot): the held slot drained for it while it starves
        self._drain: tuple[str, int] | None = None
        # Reads into the host tier run on one background thread: a host-tier miss and
        # a runtime load never read a file on the GPU thread or under the lock.
        self._reader: ThreadPoolExecutor | None = None
        self._reads: dict[str, tuple[Future, _Served]] = {}  # adapter -> read in flight
        # adapter -> why its last background read failed; a new request on it retries
        self._read_errors: dict[str, str] = {}
        # admit runs on the GPU thread, remove on the main thread, a read's completion on
        # the reader (or inline, if it completed before its callback was added: reentrant)
        self._lock = threading.RLock()
        self._admits = 0  # written by admit only; the clock of `last_used_step`

        self._stager = PinnedStager(torch.int32)
        self._eager: _Buffers | None = None
        self._cg: dict[CGSlotKey, _Buffers] = {}
        # row grouping -> this step's addressing
        self._current: dict[int, LoraAddressing] | None = None
        # whether this step adds deltas (`plan`); a plain bool, which compiled forwards
        # guard on
        self.active = True
        # a piecewise region's step plans inside its enclosing step's forward; the
        # enclosing step's addressing and `active`, for the eager code after the region
        self._enclosing: tuple[dict[int, LoraAddressing], bool] | None = None

        # read adapters ahead, so a first request finds its adapter in the host tier
        for name in list(self._serving)[:spec.cpu_capacity]:
            served = self._serving[name]
            self._store.load(name, served.path, served.digest)

        device_bytes = sum(t.numel() * t.element_size() for t in (*self._a.values(), *self._b.values()))
        # a host copy is one slot's worth, pinned, for each of up to cpu_capacity adapters
        slot_bytes = sum(
            t[:, 0].numel() * t.element_size() for t in (*self._a.values(), *self._b.values())
        )
        logger.info(
            "LoRA pool: %d slots (+ zero adapter), max rank %d, targets %s, %.2f MiB; "
            "host tier up to %d x %.2f MiB pinned",
            spec.max_slots, max_rank, sorted(self.targets), device_bytes / 2**20,
            spec.cpu_capacity, slot_bytes / 2**20,
        )

    # Storage access

    def weights(self, target: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(a [slots, slices * R, in], b [slots, out, R])`` of one target at one stack
        layer; an experts target's are ``[max_slots, experts, ...]``, no zero adapter's."""
        row = self._layer_rows[target][layer_idx]
        return self._a[target][row], self._b[target][row]

    def layer_ranks(self, target: str, layer_idx: int) -> torch.Tensor:
        """``[slots]``: each slot's rank at one target and stack layer, 0 where its
        adapter adds nothing there."""
        return self._ranks[target][self._layer_rows[target][layer_idx]]

    def layer_delta(self, target: str, layer_idx: int) -> "LoraLayerDelta":
        """What a layer calls to add ``target``'s delta at stack layer ``layer_idx``."""
        found = self._layer_deltas.get((target, layer_idx))
        if found is None:
            if layer_idx not in self._layer_rows.get(target, ()):
                raise KeyError(
                    f"the LoRA resource {self.spec.resource_key!r} has no target {target!r} "
                    f"on layer {layer_idx}; it declares {sorted(self.targets)}"
                )
            found = self._layer_deltas[(target, layer_idx)] = LoraLayerDelta(self, target, layer_idx)
        return found

    def enter_compute(self) -> None:
        self._compute_depth += 1
        self._scoped = True

    def exit_compute(self) -> None:
        self._compute_depth -= 1

    def outside_compute(self) -> bool:
        """A delta called now would run outside a planned step's compute, once the
        engine has started bracketing them (``enter_compute``)."""
        return self._scoped and not self._compute_depth

    def addressing(self, grouping: int = 1) -> LoraAddressing:
        """This step's addressing for rows of ``grouping`` (``LoraTarget.row_grouping``),
        for a backend to run against."""
        if self._current is None:
            raise RuntimeError(
                "the LoRA pool has no plan for this step; the step declaration must carry "
                "a step under this resource's key"
            )
        return self._current[grouping]

    def slot_state(self, index: int) -> SlotState:
        return self._slots[index]

    def resident_slot(self, adapter: str) -> int | None:
        return self._resident.get(adapter)

    def slot_of(self, rid) -> int | None:
        """The slot ``rid`` holds; None before its first admit."""
        return self._row_slot.get(rid)

    @property
    def store(self) -> LoraAdapterStore:
        return self._store

    # Request lifecycle

    def ingest_request(self, rid, overrides: ResourceReqConfig | None = None):
        config = overrides if isinstance(overrides, LoraReqConfig) else None
        adapter = None if config is None else config.adapter
        with self._lock:
            # idempotent: one NewRequest per partition
            if rid in self._adapter_of:
                return
            if adapter is not None:
                served = self._serving.get(adapter)
                if served is None:
                    raise ValueError(
                        f"unknown LoRA adapter {adapter!r}; this deployment serves {sorted(self._serving)}"
                    )
                if config.digest is not None and served.digest != config.digest:
                    # its prompt was keyed under the weights it was selected on
                    raise ValueError(
                        f"LoRA adapter {adapter!r} was loaded again with other weights after "
                        "this request selected it; send the request again"
                    )
            self._adapter_of[rid] = adapter
            if adapter is None:
                self._row_slot[rid] = ZERO_SLOT
            else:
                self._awaiting.add(rid)
                # every rank, so a follower has its copy when the leader's step needs it
                self._read_errors.pop(adapter, None)
                self._prefetch(adapter)

    def remove_request(self, rid):
        with self._lock:
            adapter = self._adapter_of.pop(rid, None)
            self._awaiting.discard(rid)
            self._admit_taken.pop(rid, None)
            self._unclaim(rid)
            self._waiting_since.pop(rid, None)
            slot = self._row_slot.pop(rid, None)
            if slot:
                self._release(slot)
            if self._retiring and adapter in self._retiring and adapter not in self._adapter_of.values():
                self._forget(adapter)

    # Runtime load and unload (see the module docstring for why any moment will do)

    def serves(self, adapter: str) -> bool:
        """Whether a new request may name ``adapter`` here."""
        return adapter in self._serving

    def load_adapter(self, name: str, path: str, digest: bytes) -> None:
        """``load_adapter_async``, waited on."""
        self.load_adapter_async(name, path, digest).result()

    def load_adapter_async(self, name: str, path: str, digest: bytes) -> Future:
        """Serve ``name`` from ``path``: read it on the reader thread, hold it to ``digest``
        (of the weights the API server checked against the spec), shard it for
        this rank into the host tier, and only then make it servable. The future fails if
        it is served already, is still retiring from an unload, or cannot be read; a
        refusal visible now raises here. A request that names it takes a slot on its
        first admit, as for an adapter configured at startup."""
        with self._lock:
            self._refuse_reload(name)
            return self._executor().submit(self._load, name, path, digest)

    def _load(self, name: str, path: str, digest: bytes) -> None:
        adapter = self._store.read(name, path, digest)
        with self._lock:
            self._refuse_reload(name)
            self._store.put(adapter, keep=self._wanted())
            self._serving[name] = _Served(path, digest)
        logger.info("LoRA pool: loaded %s from %s", name, path)

    def _refuse_reload(self, name: str) -> None:
        if name in self._serving:
            raise ValueError(f"LoRA adapter {name!r} is already loaded")
        if name in self._retiring:
            raise ValueError(
                f"LoRA adapter {name!r} was unloaded but requests on it are still running; "
                "load it again once they finish"
            )

    def unload_adapter(self, name: str) -> bool:
        """Stop serving ``name`` to new requests. Its host copy and its name -> slot entry
        are forgotten now if no live request names it, else when the last one is removed;
        those requests run to the end on its weights. Returns whether it was forgotten
        now. An adapter this pool does not serve is a no-op (a rolled-back load that
        failed here)."""
        with self._lock:
            served = self._serving.pop(name, None)
            if served is None:
                return False
            if name in self._adapter_of.values():
                self._retiring[name] = served
                logger.info("LoRA pool: unloaded %s; its running requests keep it until they end", name)
                return False
            self._forget(name)
            return True

    def _forget(self, name: str) -> None:
        """Caller holds the lock. The slot keeps its weights, unheld, so the slot table an
        admit decides from is unchanged; ``_install`` reuses the slot like any other."""
        self._retiring.pop(name, None)
        self._store.drop(name)
        self._empty.discard(name)
        slot = self._resident.pop(name, None)
        logger.info(
            "LoRA pool: dropped %s%s", name, "" if slot is None else f" (its slot {slot} is free to reuse)",
        )

    # Readiness

    def admit_retrieve(self, rid, node_name, graph_walk, published) -> AdmitOutcome:
        """Whether ``rid`` should go into a step now: not before its adapter can get a
        slot, so the scheduler leaves it out instead of building a step its admit would
        refuse whole (the refusal in ``_load_miss`` stays as a backstop).

        Only the rank that chooses batches screens. A lockstep instance's followers run
        the batches its leader chose and admit them from the same slot state, so they
        answer ready and keep no screening state; the leader's answers never have to
        agree with anything another rank computes.

        A row it lets through is claimed: counted against the free slots until its admit
        takes one, so two rows on different missing adapters cannot both be promised the
        last free slot. A claim is made by any readiness check, the scheduler's probes
        included, and lasts until the row's admit or removal.

        Once a row has been held back for want of a slot for ``slot_wait_steps`` admits,
        its adapter is starving: until it gets a slot, no other row may take one, and new
        rows on the held slot with the fewest holders are held back too, so that slot
        drains to refcount 0 and the starving adapter takes it (an adapter whose requests
        keep its slot held cannot starve the others). Rows on the other held slots keep
        flowing. Only rows held for want of a slot age, and only rows the scheduler still
        screens hold the others back. Ages count this pool's admits, never wall clock.
        """
        del node_name, graph_walk, published
        # the steady state: a set check, on the scheduler's hottest sweep
        if rid not in self._awaiting or rid in self._claims or not self._screens:
            return ADMIT_OK
        with self._lock:
            if rid not in self._awaiting:  # admitted or removed meanwhile
                return ADMIT_OK
            return self._screen(rid)

    def _screen(self, rid) -> AdmitOutcome:
        """Caller holds the lock."""
        name = self._adapter_of[rid]
        if name in self._empty:
            # takes no slot; and no longer waits for one, so it ages nothing
            self._waiting_since.pop(rid, None)
            return ADMIT_OK
        self._screen_count += 1
        waiting = self._waiting_since.get(rid)
        if waiting is not None:
            waiting.screened = self._screen_count
        claimed = self._claim_count
        # its admit would read the file
        unread = name not in claimed and name not in self._resident and name not in self._store
        # an adapter that failed to read fails its rows before they wait for, or drain, a slot
        error = self._read_errors.get(name) if unread else None
        if error is not None:
            self._waiting_since.pop(rid, None)
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                message=f"LoRA adapter {name!r} failed to load: {error}",
                request_ids=frozenset({rid}),
            ))
        starving = self._starving_adapter()
        if starving is not None and name != starving and not self._may_flow(name, starving):
            return _NOT_READY  # held for another adapter's drain, so not aged
        # a new claim on a missing adapter, or on one resident in an unheld slot, takes
        # a slot that no other claim may count on
        if name not in claimed and self._free_slots(name) < 0:
            if waiting is None:
                self._waiting_since[rid] = _Wait(since=self._admits, screened=self._screen_count)
            return _NOT_READY
        if unread:
            # read it in the background instead
            self._prefetch(name)
            return _NOT_READY
        self._claims[rid] = name
        claimed[name] = claimed.get(name, 0) + 1
        self._waiting_since.pop(rid, None)
        return ADMIT_OK

    def _unclaim(self, rid) -> None:
        name = self._claims.pop(rid, None)
        if name is not None:
            count = self._claim_count[name] - 1
            if count:
                self._claim_count[name] = count
            else:
                del self._claim_count[name]

    def _starving_adapter(self) -> str | None:
        """The adapter of the longest-waiting row the scheduler still screens, once it
        has waited ``slot_wait_steps``.

        A scan screens every waiting row it reaches, so between two screens of a row in
        the scans, each other unclaimed row is screened about once. A row passed over for
        twice that is out of the scans (failed and awaiting its teardown, in a backoff):
        it cannot take a slot, so it holds no one back until it is screened again. Counted
        in screens, not admits, since holding everyone back may leave nothing to admit."""
        live = self._screen_count - 2 * len(self._awaiting) - 2
        for rid, waiting in self._waiting_since.items():
            if waiting.screened < live:
                continue
            if self._admits - waiting.since < self.spec.slot_wait_steps:
                return None
            return self._adapter_of.get(rid)
        return None

    def _may_flow(self, name: str | None, starving: str) -> bool:
        """Whether a row on ``name`` may be claimed while ``starving`` waits for a slot:
        only on a slot that stays held anyway, and not the one being drained for it. A
        missing adapter, or one resident in an unheld slot, would take a slot the
        starving adapter needs."""
        slot = self._resident.get(name)
        if slot is None or not (self._slots[slot].refcount or name in self._claim_count):
            return False
        return slot != self._drain_slot(starving)

    def _drain_slot(self, starving: str) -> int | None:
        """The held slot drained for ``starving``: none while a slot is free for it, else
        the one it was given, while still held, else the held slot with the fewest
        holders (the quickest to drain)."""
        if self._free_slots(starving) >= 0:
            self._drain = None
            return None
        if self._drain is not None and self._drain[0] == starving:
            state = self._slots[self._drain[1]]
            if state.refcount or self._claim_count.get(state.adapter):
                return state.index
        holders = {}
        for state in self._slots[1:]:
            count = state.refcount + self._claim_count.get(state.adapter, 0)
            if count:
                holders[state.index] = count
        if not holders:
            return None
        slot = min(holders, key=lambda i: (holders[i], i))
        self._drain = (starving, slot)
        return slot

    def _free_slots(self, name: str) -> int:
        """Slots left once every held slot, every claimed adapter and ``name`` have one:
        negative when that promises more slots than exist."""
        pinned = {state.index for state in self._slots[1:] if state.refcount}
        misses = 0
        for adapter in self._claim_count.keys() | {name}:
            slot = self._resident.get(adapter)
            if slot is None:
                misses += 1
            else:
                pinned.add(slot)
        return self.spec.max_slots - len(pinned) - misses

    # Step lifecycle

    def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
        """Give each row whose adapter is not resident yet a slot, loading it there.

        Nothing to do once every row holds its slot, which is every step after a request's
        first, so the steady state costs one set check.

        Rows whose adapter is resident take its slot before any miss picks a victim, so a
        miss cannot evict an unheld adapter that a later row of the same step names.

        Every miss's slot is picked in that same locked section, before any wait: what
        is picked follows only from the slot table at the admit's start, which is the
        same on every rank of a lockstep instance, whatever lands while a miss waits
        for its copy or for the load agreement.
        """
        if ctx.capture:
            return ADMIT_OK
        if ctx.graph_walk != PIECEWISE_WALK:
            # the LRU clock counts steps; a piecewise region's admit is inside its
            # enclosing step's forward, whose rows already hold their slots
            self._admits += 1
        if not self._awaiting:
            return ADMIT_OK
        misses: dict[str, list] = {}
        with self._lock:
            for seg in step.segments or ():
                rid = seg.request_id
                if rid not in self._awaiting or seg.span <= 0 or ctx.is_padding_row(rid):
                    continue
                name = self._adapter_of[rid]
                slot = ZERO_SLOT if name in self._empty else self._resident.get(name)
                if slot is None:
                    misses.setdefault(name, []).append(rid)
                else:
                    self._take(rid, slot)
            if not misses:
                return ADMIT_OK
            reserved: list[int] = []
            for name, rids in misses.items():
                slot = self._pick_slot(exclude=reserved)
                if slot is None:
                    # refused on every rank alike, before any rank reaches an agreement
                    return AdmitOutcome(
                        ok=False,
                        reason=AllocationFailed(
                            message=(
                                f"LoRA pool is full: all {self.spec.max_slots} slots hold "
                                f"adapters of running requests, none free for {name!r}"
                            ),
                            pages_short=1,
                            label=_LABEL,
                            request_id=rids[0],
                        ),
                    )
                reserved.append(slot)
        for (name, rids), slot in zip(misses.items(), reserved, strict=True):
            outcome = self._load_miss(name, rids, slot)
            if not outcome.ok:
                return outcome
        return ADMIT_OK

    def rollback_admit(self, step: ResourceStep, ctx: StepContext) -> None:
        """Give back the slots this step's admit took. An adapter it loaded stays resident
        in its now-unheld slot, as a cache."""
        del step
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            for rid in [r for r in self._admit_taken if ctx.is_real_row(r)]:
                slot = self._admit_taken.pop(rid)
                if self._row_slot.get(rid) != slot:
                    continue  # removed under us; the slot went with it
                del self._row_slot[rid]
                self._awaiting.add(rid)
                if slot:
                    self._release(slot)

    def commit(self, step: ResourceStep, ctx: StepContext) -> None:
        del step
        if self._enclosing is not None and ctx.graph_walk == PIECEWISE_WALK:
            # the region is done; the eager code after it reads the enclosing plan
            (self._current, self.active), self._enclosing = self._enclosing, None
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            for rid in [r for r in self._admit_taken if ctx.is_real_row(r)]:
                del self._admit_taken[rid]

    def _take(self, rid, slot: int) -> None:
        """Caller holds the lock."""
        if slot:
            self._slots[slot].refcount += 1
        self._row_slot[rid] = slot
        self._awaiting.discard(rid)
        self._unclaim(rid)
        self._waiting_since.pop(rid, None)
        self._admit_taken[rid] = slot

    def _load_miss(self, name: str, rids: list, slot: int) -> AdmitOutcome:
        """Load ``name`` into ``slot`` (picked by ``admit``) for ``rids``. Called without
        the lock: waiting for a host copy and the load agreement both happen outside it,
        so the main thread's ingests and removes never wait on a file or a collective.

        On a lockstep instance every rank reaches the agreement for the same misses in
        the same order, into the same slots: ``admit`` picked them all from the slot
        table at its start, which is the same on every rank (see the module docstring).
        """
        with self._lock:
            served = self._serving.get(name) or self._retiring.get(name)
            loaded = self._store.get(name)
            pending = None
            if loaded is None and served is not None:
                # the leader's screen saw this copy in already; a follower may still be
                # reading its own, or (rarely) start the read here
                pending = self._reads.get(name, (None,))[0] or self._prefetch(name)
        error = None
        if served is None:
            error = f"{name!r} is not served"
        elif loaded is None and pending is None:
            error = "it is not in the host tier and no read of it is running"
        elif loaded is None:
            try:
                loaded = pending.result()
            except Exception as exc:  # a bad checkpoint fails the request, not the worker
                logger.error("LoRA adapter %s failed to load: %s", name, exc)
                error = describe_load_error(exc)
        # all ranks install it or none does: a rank refusing alone would leave the rest
        # of its instance waiting in the forward's collectives
        if self._agree is not None and not self._agree(error is None):
            error = error or "it failed to load on another rank of its instance"
        if error is not None:
            return AdmitOutcome(
                ok=False,
                reason=AdmitRuntimeError(
                    message=f"LoRA adapter {name!r} failed to load: {error}",
                    request_ids=frozenset(rids),
                ),
            )
        with self._lock:
            if loaded.rank == 0:
                # nothing for this spec: the same file, digest-pinned, on every rank, so
                # every rank leaves the slot it picked as it was
                self._empty.add(name)
                slot = ZERO_SLOT
            else:
                self._install(slot, loaded)
            for rid in rids:
                if rid in self._awaiting:
                    self._take(rid, slot)
        return ADMIT_OK

    # Host tier

    def _executor(self) -> ThreadPoolExecutor:
        if self._reader is None:
            self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lora-read")
        return self._reader

    def _prefetch(self, name: str) -> Future | None:
        """Caller holds the lock. Start reading ``name`` into the host tier unless it is
        resident, held there, empty for this pool, or already being read; returns the
        read in flight."""
        found = self._reads.get(name)
        if found is not None:
            return found[0]
        served = self._serving.get(name) or self._retiring.get(name)
        if served is None or name in self._empty or name in self._resident or name in self._store:
            return None
        future = self._executor().submit(self._store.read, name, served.path, served.digest)
        self._reads[name] = (future, served)
        future.add_done_callback(lambda f: self._read_done(name, f))
        return future

    def _read_done(self, name: str, future: Future) -> None:
        with self._lock:
            entry = self._reads.get(name)
            if entry is None or entry[0] is not future:
                return
            del self._reads[name]
            if future.cancelled():  # the pool is shutting down
                return
            served = entry[1]
            exc = future.exception()
            if exc is not None:
                logger.error("LoRA adapter %s failed to load: %s", name, exc)
                self._read_errors[name] = describe_load_error(exc)
                return
            adapter = future.result()
            current = self._serving.get(name) or self._retiring.get(name)
            if current is None or current != served:
                return  # unloaded, or reloaded with other weights, while it was read
            self._store.put(adapter, keep=self._wanted())

    def _wanted(self) -> set[str]:
        """Caller holds the lock. Adapters a request still waits on: the host tier keeps
        them past ``cpu_capacity``, so a copy read for an admit is there when it runs."""
        return {self._adapter_of[rid] for rid in self._awaiting}

    def _pick_slot(self, exclude=()) -> int | None:
        """An empty slot, else the unheld one released longest ago (ties to the lowest
        index, so the choice is the same on every rank); none of ``exclude``."""
        victim = None
        for state in self._slots[1:]:
            if state.refcount or state.index in exclude:
                continue
            if state.adapter is None:
                return state.index
            if victim is None or state.last_used_step < victim.last_used_step:
                victim = state
        return None if victim is None else victim.index

    def _install(self, slot: int, adapter: LoadedAdapter) -> None:
        """Copy ``adapter`` into ``slot`` on the current stream, so it lands ahead of this
        step's forward. Dropping the host copy from the store meanwhile is safe: torch's
        pinned-memory allocator holds a block until the copies recorded on it complete."""
        state = self._slots[slot]
        evicted = state.adapter
        if evicted is not None:
            logger.info("LoRA pool: evicting %s from slot %d for %s", evicted, slot, adapter.name)
        # an unloaded adapter's slot keeps its name with no entry in `_resident`, which a
        # reload of that name may have pointed at another slot since: then this slot was
        # not where it lived, and its claims and reads stand
        if evicted is not None and self._resident.get(evicted) == slot:
            del self._resident[evicted]
            # A row may still want it (claimed while it was resident, which the pick
            # cannot see on a follower, so does not consult anywhere): its admit would
            # read the file on the GPU thread. Screen such rows again, and read the
            # adapter back meanwhile, on every rank alike.
            for rid in [r for r, name in self._claims.items() if name == evicted]:
                self._unclaim(rid)
            if evicted in self._wanted():
                self._prefetch(evicted)
        for name in self._a:
            a, b = adapter.a.get(name), adapter.b.get(name)
            # an experts target stores no zero adapter
            index = slot if self.targets[name].experts == 1 else slot - 1
            slot_a, slot_b = self._a[name][:, index], self._b[name][:, index]
            if a is None:
                slot_a.zero_()
                slot_b.zero_()
                self._ranks[name][:, slot] = 0
            elif self.targets[name].experts > 1:
                # layer by layer: the slot's view is strided across layers, and copying it
                # whole would stage a contiguous temporary the size of the slot (GBs)
                for row in range(slot_a.shape[0]):
                    slot_a[row].copy_(a[row], non_blocking=True)
                    slot_b[row].copy_(b[row], non_blocking=True)
            else:
                slot_a.copy_(a, non_blocking=True)
                slot_b.copy_(b, non_blocking=True)
            if a is not None:
                # 0 at a layer whose A or B is all zero, which adds nothing (the store's read)
                self._ranks[name][:, slot].copy_(adapter.layer_ranks[name], non_blocking=True)
        self.scalings[slot] = adapter.scaling
        state.adapter = adapter.name
        self._resident[adapter.name] = slot

    def _release(self, slot: int) -> None:
        state = self._slots[slot]
        state.refcount -= 1
        assert state.refcount >= 0, f"LoRA slot {slot} released more often than taken"
        if state.refcount == 0:
            state.last_used_step = self._admits

    # Addressing

    def plan(self, step: ResourceStep, ctx: StepContext) -> LoraAddressing | None:
        if ctx.graph_walk == PIECEWISE_WALK and not ctx.capture and self._current is not None:
            # before the memoized return below, which a region's repeat plan takes too
            self._enclosing = (self._current, self.active)
        segments = step.segments or ()
        self.active = not ctx.idle
        if not self.active:
            # no delta reads an addressing, so none is staged
            self._current = {}
            return None
        rows = len(segments)
        if ctx.capture:
            slots = np.zeros(rows, dtype=np.int32)
        else:
            get = self._row_slot.get
            slots = np.fromiter(
                (get(seg.request_id, ZERO_SLOT) for seg in segments), dtype=np.int32, count=rows,
            )
            if ctx.padded_request_ids is not ctx.request_ids:
                padding = np.fromiter(
                    (ctx.is_padding_row(seg.request_id) for seg in segments), dtype=bool, count=rows,
                )
                slots[padding] = ZERO_SLOT
        spans = np.fromiter((seg.span for seg in segments), dtype=np.int32, count=rows)
        np.maximum(spans, 0, out=spans)
        tokens = int(spans.sum())
        num_slots = self.spec.num_slots

        buf = self._buffers(ctx, rows, tokens)
        planned = (slots.tobytes(), spans.tobytes())
        if buf.planned is not None and buf.planned[0] == planned:
            # the buffer already holds this addressing (a steady decode batch): only this
            # pool writes it, as `H2DMirror` assumes, so nothing is rebuilt or copied
            self._current = buf.planned[1]
            return self._current[1]
        layout = buf.layout or _Layout.of(rows, tokens, num_slots, self._groupings)
        host = np.zeros(layout.size, dtype=np.int32)
        data = buf.data[:layout.size]
        current = {}
        for grouping, sec in zip(self._groupings, layout.sections, strict=True):
            counts = self._grouped_spans(spans, grouping)
            count = int(counts.sum())
            token_slots = np.repeat(slots, counts)
            host[sec.token_slot:sec.token_slot + count] = token_slots
            host[sec.order:sec.order + count] = np.argsort(token_slots, kind="stable")
            np.cumsum(
                np.bincount(token_slots, minlength=num_slots),
                out=host[sec.slot + 1:sec.slot + num_slots + 1],
            )
            current[grouping] = LoraAddressing(
                token_order=data[sec.order:sec.order + count],
                slot_indptr=data[sec.slot:sec.slot + num_slots + 1],
                token_slot=data[sec.token_slot:sec.token_slot + count],
            )
        self._stager.copy_(data, host, mirror=buf.mirror)
        buf.planned = (planned, current)
        self._current = current
        return current[1]

    @staticmethod
    def _grouped_spans(spans: np.ndarray, grouping: int) -> np.ndarray:
        """Each segment's rows under ``grouping``: its tokens, one, or its folds."""
        if grouping == 1:
            return spans
        if grouping == 0:
            return np.ones_like(spans)
        if (spans % grouping).any():
            raise ValueError(
                f"LoRA segments of {spans.tolist()} tokens do not fold into rows of {grouping}"
            )
        return spans // grouping

    def _buffers(self, ctx: StepContext, rows: int, tokens: int) -> _Buffers:
        """The buffer this step fills: under a capture lease, its slot key's, laid out once
        for the bucket, so a replay reads the addresses its capture recorded; otherwise the
        one eager buffer, grown as needed."""
        lease = ctx.slot_lease
        if lease is None or lease.bucket is None:
            need = _Layout.of(rows, tokens, self.spec.num_slots, self._groupings).size
            buf = self._eager
            if buf is None or buf.data.numel() < need:
                buf = self._eager = _Buffers(_next_pow2(need), self._device)
            return buf
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=_LABEL)
        buf = self._cg.get(key)
        if buf is None:
            # built by the bucket's capture, whose step is exactly the bucket's shape
            layout = _Layout.of(
                max(lease.bucket.bs, rows), max(lease.bucket.num_tokens, tokens),
                self.spec.num_slots, self._groupings,
            )
            buf = self._cg[key] = _Buffers(layout.size, self._device, layout)
        assert buf.layout.rows >= rows and buf.layout.tokens >= tokens, (
            f"LoRA addressing for {key} holds {buf.layout.rows} rows / {buf.layout.tokens} "
            f"tokens but this step planned {rows} / {tokens}"
        )
        return buf

    @property
    def force_double_buffer(self):
        # each plan stages its addressing into a buffer a replay reads
        return True

    # Engine lifecycle

    def cleanup(self):
        if self._reader is not None:
            self._reader.shutdown(wait=False, cancel_futures=True)
        self._a.clear()
        self._b.clear()
        self._ranks.clear()
        self._cg.clear()
        self._layer_deltas.clear()
        self._eager = None
        self._current = self._enclosing = None


def _next_pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()
