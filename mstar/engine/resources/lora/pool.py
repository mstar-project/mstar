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

A full pool is backpressure, not a refused step: a request on an adapter waits at the
runner's admission (``ask`` / ``reserve``) until a slot can take its adapter, in one queue
with every other resource that asks, so it holds no KV pages meanwhile and the rows already
running never lose a step to it. A reservation counts slots and pins none: the slot an
admit picks follows from the slot table alone, and the reservations never promise more
slots than exist, so every reserved row finds one at its admit. Only the rank that chooses
batches decides (a lockstep instance's leader), so no other rank keeps reservations; the
admit refusal stays as a backstop.

A model with LoRA specs on several nodes (an LLM and its vision encoder) has a pool per
node, and each admits only its own node's rows. A request holds a slot in each until it
is removed (none in a pool whose targets its adapter has nothing for), but where the
graph runs the encoder once, before the LLM (Qwen3.5), it takes them in that order and a
row holding an LLM slot never waits on the encoder's pool, so the two pools' waits
cannot form a cycle.

Files are read on the pool's reader thread, never on the GPU thread or under the lock: a
request names an adapter, and every rank starts reading it into the host tier as it ingests
the request; the leader reserves a slot for a request only once its own copy is in. A
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
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.base import CGSlotKey, EngineResourceInfo, Resource
from mstar.engine.resources.lora.backend import LoraBackend, make_backend
from mstar.engine.resources.lora.config import (
    LoraBaseOnly,
    LoraReqConfig,
    LoraSpec,
    LoraStep,
    describe_load_error,
)
from mstar.engine.resources.lora.store import LoadedAdapter, LoraAdapterStore
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import (
    ADMIT_OK,
    ASKS,
    PIECEWISE_WALK,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    StepContext,
)
from mstar.utils.h2d import H2DMirror, PinnedStager

logger = logging.getLogger(__name__)

ZERO_SLOT = 0
_LABEL = "lora"


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


class _Served(NamedTuple):
    """Where an adapter is read from, and what its weights must digest to (None until
    its first read, for an adapter that arrived without one)."""

    path: str
    digest: bytes | None


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


class LoraLayerDelta:
    """``y += delta(x)`` for one target linear at one stack layer, over the current step.

    A layer computes its base output, then calls this before anything consumes ``y`` —
    for a row-parallel linear, before its all-reduce. Rows on the base model address the
    zero adapter, which the kernels skip; a step with no adapter row at all returns ``y``
    untouched (``LoraAdapterPool.active``). The layer's slot weights are views resolved
    once, here; a call reads only the step's addressing.
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

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # compile check first, so a traced forward installs no guard on the pool
        if not torch.compiler.is_compiling() and self._pool._current is None:
            raise RuntimeError(
                "a LoRA delta was called outside its node's planned step (e.g. from "
                "prepare_inputs, or on an output cached from an earlier step): there is "
                "no addressing for its rows. Call the module's base computation there "
                "and add the delta in preprocess or the forward"
            )
        if not self._pool.active:
            # a base-only step: no row names an adapter, so nothing is launched
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


# int32 entries per alignment unit: each array starts on a 16-byte boundary, which
# compiled kernels reading them as inputs expect
_ALIGN = 4


def _aligned(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


class _Section(NamedTuple):
    """Offsets of one row grouping's arrays: ``count`` rows, their order grouped by slot
    (``token_order``), and each slot's range of it (``slot_indptr``)."""

    count: int
    order: int
    slot: int


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
            at = slot + _aligned(num_slots + 1)
            sections.append(_Section(count, order, slot))
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

    @classmethod
    def build(cls, spec: LoraSpec, info: EngineResourceInfo) -> "LoraAdapterPool":
        rank, world_size, agree, decides = 0, 1, None, True
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
                decides = joint.rank == 0
        return cls(
            spec, info.device, rank=rank, world_size=world_size, dtype=info.compute_dtype,
            agree=agree, decides=decides,
        )

    def __init__(
        self,
        spec: LoraSpec,
        device: torch.device | str,
        rank: int = 0,
        world_size: int = 1,
        backend: LoraBackend | None = None,
        dtype: torch.dtype | None = None,
        agree: Callable[[bool], bool] | None = None,
        decides: bool = True,
    ):
        """``dtype`` is the nodes' compute dtype, which the slot weights take unless the
        spec names one. ``agree`` ANDs a flag across the pool's lockstep instance (a
        collective); None on a single rank. ``decides`` says whether this rank chooses
        batches, and so admits requests (``ask``): False on a lockstep instance's followers."""
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
        self._decides = decides
        self.targets = {name: t.shard(world_size) for name, t in spec.targets.items()}
        num_slots, max_rank = spec.num_slots, spec.max_rank
        # the row groupings the targets read; per token always, first
        self._groupings = (1, *sorted({t.row_grouping for t in self.targets.values()} - {1}))

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
            self._a[name] = torch.zeros(
                rows, num_slots, target.num_slices * max_rank, target.in_features,
                dtype=self.dtype, device=self._device,
            )
            self._b[name] = torch.zeros(
                rows, num_slots, target.out_features, max_rank,
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
        # adapter -> where it is read from: those a new request may name, and those
        # unloaded while live requests still named them, kept until the last of those
        # requests is removed
        self._serving: dict[str, _Served] = {
            name: _Served(path, spec.adapter_digests.get(name))
            for name, path in spec.adapters.items()
        }
        self._retiring: dict[str, _Served] = {}
        self._store = LoraAdapterStore(spec, rank=rank, world_size=world_size, dtype=self.dtype)
        # (target, layer) -> its bound delta, one per binding so compiled code guards on it
        self._layer_deltas: dict[tuple[str, int], LoraLayerDelta] = {}

        # Handle-keyed request state, all purged by `remove_request`.
        self._adapter_of: dict = {}   # rid -> adapter name, None for the base model
        self._adapter_rows = 0  # live rids naming an adapter
        self._row_slot: dict = {}     # rid -> slot held, read by `plan`
        self._awaiting: set = set()   # rids with an adapter but no slot yet
        # rid -> (slot the live admit took, the adapter it reserved), for rollback
        self._admit_taken: dict = {}
        # rid -> adapter: awaiting rows admission reserved for, counted against the free
        # slots until their admit takes one (the deciding rank only)
        self._reserved: dict = {}
        self._reserved_count: dict[str, int] = {}  # adapter -> rows reserved on it
        # bumped whenever a request that didn't fit may fit now (`grew_since`): a slot
        # released, a reservation given back, a read landed or failed
        self._changes = 0
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
        # row grouping -> this step's addressing, from its plan until its commit
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
            adapter = self._store.load(name, served.path, served.digest)
            self._pin_digest(name, served, adapter.digest)

        logger.info(
            "LoRA pool: %d slots (+ zero adapter), max rank %d, targets %s, %.2f MiB",
            spec.max_slots, max_rank, sorted(self.targets),
            sum(t.numel() * t.element_size() for t in (*self._a.values(), *self._b.values())) / 2**20,
        )

    # Storage access

    def weights(self, target: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(a [slots, slices * R, in], b [slots, out, R])`` of one target at one stack layer."""
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

    def addressing(self, grouping: int = 1) -> LoraAddressing:
        """This step's addressing for rows of ``grouping`` (``LoraTarget.row_grouping``),
        for a backend to run against."""
        if self._current is None:
            raise RuntimeError(
                "the LoRA pool has no plan for this step; the step declaration must carry "
                "a LoraStep under this resource's key"
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
        adapter = overrides.adapter if isinstance(overrides, LoraReqConfig) else None
        with self._lock:
            # idempotent: one NewRequest per partition
            if rid in self._adapter_of:
                return
            if adapter is not None and adapter not in self._serving:
                raise ValueError(
                    f"unknown LoRA adapter {adapter!r}; this deployment serves {sorted(self._serving)}"
                )
            self._adapter_of[rid] = adapter
            if adapter is None:
                self._row_slot[rid] = ZERO_SLOT
            else:
                self._adapter_rows += 1
                self._awaiting.add(rid)
                # every rank, so a follower has its copy when the leader's step needs it
                self._read_errors.pop(adapter, None)
                self._prefetch(adapter)

    def remove_request(self, rid):
        with self._lock:
            adapter = self._adapter_of.pop(rid, None)
            if adapter is not None:
                self._adapter_rows -= 1
            self._awaiting.discard(rid)
            self._admit_taken.pop(rid, None)
            self._unreserve(rid)
            slot = self._row_slot.pop(rid, None)
            if slot:
                self._release(slot)
            if self._retiring and adapter in self._retiring and adapter not in self._adapter_of.values():
                self._forget(adapter)

    # Runtime load and unload (see the module docstring for why any moment will do)

    def serves(self, adapter: str) -> bool:
        """Whether a new request may name ``adapter`` here."""
        return adapter in self._serving

    def load_adapter(self, name: str, path: str, digest: bytes | None = None) -> None:
        """``load_adapter_async``, waited on."""
        self.load_adapter_async(name, path, digest).result()

    def load_adapter_async(self, name: str, path: str, digest: bytes | None = None) -> Future:
        """Serve ``name`` from ``path``: read it on the reader thread, check it against the
        spec (and against ``digest``, the weights the API server checked), shard it for
        this rank into the host tier, and only then make it servable. The future fails if
        it is served already, is still retiring from an unload, or cannot be read; a
        refusal visible now raises here. A request that names it takes a slot on its
        first admit, as for an adapter configured at startup."""
        with self._lock:
            self._refuse_reload(name)
            return self._executor().submit(self._load, name, path, digest)

    def _load(self, name: str, path: str, digest: bytes | None) -> None:
        adapter = self._store.read(name, path, digest)
        with self._lock:
            self._refuse_reload(name)
            self._store.put(adapter, keep=self._wanted())
            self._serving[name] = _Served(path, adapter.digest)
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

    # Admission: a slot for a request's adapter, all or nothing with the other resources

    def ask(self, rid, node_name, graph_walk) -> AdmitOutcome | None:
        """`ASKS` for a row that will need a slot, or the outcome that fails a row whose
        adapter failed to read. None on a rank that doesn't decide, and for a base row, a
        row that holds its slot, and an adapter with nothing for this pool."""
        del node_name, graph_walk
        # the steady state: a set check
        if not self._decides or rid not in self._awaiting:
            return None
        with self._lock:
            name = self._adapter_of.get(rid)
            if name is None or name in self._empty:
                return None
            error = self._read_errors.get(name)
            if error is not None and name not in self._resident and name not in self._store:
                return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                    message=f"LoRA adapter {name!r} failed to load: {error}",
                    request_ids=frozenset({rid}),
                ))
        return ASKS

    def reserve(self, rid, node_name, graph_walk, published) -> int | None:
        """Count ``rid``'s adapter against the free slots once its host copy is in, so its
        admit never waits for a read. A copy whose read failed is not waited for: the
        admit then fails the request, on every rank alike."""
        del node_name, graph_walk, published
        with self._lock:
            name = self._adapter_of[rid]
            if self._spare_slots(name) < 0:
                return self._changes
            if (name not in self._reserved_count and name not in self._resident
                    and name not in self._store and name not in self._read_errors):
                self._prefetch(name)
                return self._changes
            self._reserved[rid] = name
            self._reserved_count[name] = self._reserved_count.get(name, 0) + 1
            return None

    def unreserve(self, rid) -> None:
        with self._lock:
            self._unreserve(rid)

    def grew_since(self, mark: int) -> bool:
        return self._changes != mark

    def _unreserve(self, rid) -> str | None:
        """Caller holds the lock."""
        name = self._reserved.pop(rid, None)
        if name is not None:
            count = self._reserved_count[name] - 1
            if count:
                self._reserved_count[name] = count
            else:
                del self._reserved_count[name]
            self._changes += 1
        return name

    def _spare_slots(self, name: str) -> int:
        """Slots left once every held slot, every reserved adapter and ``name`` have one:
        negative when that promises more slots than exist."""
        pinned = {state.index for state in self._slots[1:] if state.refcount}
        misses = 0
        for adapter in self._reserved_count.keys() | {name}:
            slot = self._resident.get(adapter)
            if slot is None:
                misses += 1
            else:
                pinned.add(slot)
        return self.spec.max_slots - len(pinned) - misses

    # Step lifecycle

    def admit(self, step: LoraStep, ctx: StepContext) -> AdmitOutcome:
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

    def rollback_admit(self, step: LoraStep, ctx: StepContext) -> None:
        """Give back the slots this step's admit took. An adapter it loaded stays resident
        in its now-unheld slot, as a cache."""
        del step
        if ctx.graph_walk != PIECEWISE_WALK:
            self._current = None
        if ctx.capture or not self._admit_taken:
            return
        with self._lock:
            for rid in [r for r in self._admit_taken if ctx.is_real_row(r)]:
                slot, reserved = self._admit_taken.pop(rid)
                if self._row_slot.get(rid) != slot:
                    continue  # removed under us; the slot went with it
                del self._row_slot[rid]
                self._awaiting.add(rid)
                if reserved is not None:
                    # admitted already, so the next admit finds its slot still counted
                    self._reserved[rid] = reserved
                    self._reserved_count[reserved] = self._reserved_count.get(reserved, 0) + 1
                if slot:
                    self._release(slot)

    def commit(self, step: LoraStep, ctx: StepContext) -> None:
        del step
        if ctx.graph_walk != PIECEWISE_WALK:
            # nothing reads the addressing past the step's forward: a delta called
            # before the next plan refuses rather than add these rows' adapters
            self._current = None
        elif self._enclosing is not None:
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
        reserved = self._reserved.get(rid)
        if reserved is not None:
            # held now, which counts it as the reservation did: no room grew
            self._reserved.pop(rid)
            count = self._reserved_count[reserved] - 1
            if count:
                self._reserved_count[reserved] = count
            else:
                del self._reserved_count[reserved]
        self._admit_taken[rid] = (slot, reserved)

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
            pending, error = None, None
            if served is None:
                error = f"{name!r} is not served"
            elif loaded is None:
                # the leader reserved once its copy was in (or its read failed); a follower
                # may still be reading its own, or (rarely) start the read here
                pending = self._reads.get(name, (None,))[0]
                if pending is None:
                    error = self._read_errors.get(name)
                if pending is None and error is None:
                    pending = self._prefetch(name)
                    if pending is None:
                        error = "it is not in the host tier and no read of it is running"
        if pending is not None:
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
            # a request waiting at admission for this copy may go ahead now
            self._changes += 1
            if exc is not None:
                logger.error("LoRA adapter %s failed to load: %s", name, exc)
                self._read_errors[name] = describe_load_error(exc)
                return
            adapter = future.result()
            current = self._serving.get(name) or self._retiring.get(name)
            if current is None or current.path != served.path or (
                current.digest is not None and current.digest != adapter.digest
            ):
                return  # unloaded, or reloaded with other weights, while it was read
            self._pin_digest(name, current, adapter.digest)
            self._store.put(adapter, keep=self._wanted())

    def _wanted(self) -> set[str]:
        """Caller holds the lock. Adapters a request still waits on: the host tier keeps
        them past ``cpu_capacity``, so a copy read for an admit is there when it runs."""
        return {self._adapter_of[rid] for rid in self._awaiting}

    def _pin_digest(self, name: str, served: _Served, digest: bytes) -> None:
        """An adapter that arrived with no digest takes its first read's, so a file
        changed after it is refused rather than served under the name."""
        if served.digest is not None:
            return
        for table in (self._serving, self._retiring):
            if table.get(name) == served:
                table[name] = served._replace(digest=digest)

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
        # not where it lived, and its reservations and reads stand
        if evicted is not None and self._resident.get(evicted) == slot:
            del self._resident[evicted]
            # A row reserved on it while it was resident (which the pick cannot see:
            # only the deciding rank reserves) now misses, and still counts one slot.
            # Read the adapter back meanwhile, so its admit finds the copy.
            if evicted in self._wanted():
                self._prefetch(evicted)
        for name in self._a:
            a, b = adapter.a.get(name), adapter.b.get(name)
            if a is None:
                self._a[name][:, slot].zero_()
                self._b[name][:, slot].zero_()
                self._ranks[name][:, slot] = 0
            else:
                self._a[name][:, slot].copy_(a, non_blocking=True)
                self._b[name][:, slot].copy_(b, non_blocking=True)
                # a layer whose A or B is all zero adds nothing; computed on the device,
                # in stream order after the copies, so nothing waits on the host
                live = (self._a[name][:, slot].flatten(1).ne(0).any(1)
                        & self._b[name][:, slot].flatten(1).ne(0).any(1))
                self._ranks[name][:, slot] = live.to(torch.int32) * adapter.rank
        self.scalings[slot] = adapter.scaling
        state.adapter = adapter.name
        self._resident[adapter.name] = slot

    def _release(self, slot: int) -> None:
        state = self._slots[slot]
        state.refcount -= 1
        assert state.refcount >= 0, f"LoRA slot {slot} released more often than taken"
        if state.refcount == 0:
            state.last_used_step = self._admits
            self._changes += 1

    # Addressing

    def runs_adapters(self, rids) -> bool:
        """Whether any of ``rids`` names an adapter: set at ingest, so the same on every
        rank of a lockstep instance."""
        if not self._adapter_rows:
            return False
        get = self._adapter_of.get
        return any(get(rid) is not None for rid in rids)

    def _adds_deltas(self, segments, ctx: StepContext) -> bool:
        """Whether this step's deltas run: under a capture lease, unless its bucket is a
        base-only one (``LoraBaseOnly``), whose graph records none; eagerly, when a row
        names an adapter."""
        lease = ctx.slot_lease
        if lease is None or lease.bucket is None:
            return self.runs_adapters(seg.request_id for seg in segments)
        if not isinstance(lease.bucket.cg_key_info, LoraBaseOnly):
            return True
        if not ctx.capture and self.runs_adapters(seg.request_id for seg in segments):
            # its rows would lose their deltas silently
            raise RuntimeError(
                f"LoRA resource {self.spec.resource_key!r}: leased base-only {lease.bucket} "
                "for a step whose rows name an adapter"
            )
        return False

    def plan(self, step: LoraStep, ctx: StepContext) -> LoraAddressing | None:
        if ctx.graph_walk == PIECEWISE_WALK and not ctx.capture and self._current is not None:
            # before the memoized return below, which a region's repeat plan takes too
            self._enclosing = (self._current, self.active)
        segments = step.segments or ()
        self.active = self._adds_deltas(segments, ctx)
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
            host[sec.order:sec.order + count] = np.argsort(token_slots, kind="stable")
            np.cumsum(
                np.bincount(token_slots, minlength=num_slots),
                out=host[sec.slot + 1:sec.slot + num_slots + 1],
            )
            current[grouping] = LoraAddressing(
                token_order=data[sec.order:sec.order + count],
                slot_indptr=data[sec.slot:sec.slot + num_slots + 1],
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
