"""SKETCH: attention and KV cache as independent ``Resource`` implementations.

Signatures and intent only. Re-cuts ``attention.py::FlashInferAttentionManager``
and the paged KV cache as two resources that know nothing of each other's
internals, with no ``BatchedCacheManager`` between them and the runner.

1. **Storage authority lives only in KV.** Attention never sees a
   ``PageAllocator``; it gets a ``SequenceView`` per segment at plan time and
   derives its index tensors from that alone.
2. **Slot authority lives in attention.** It owns the CUDA graph slots, leases
   one at ``admit``, and other resources read the slot off ``StepContext``.
   Execution still goes through ``cuda_graph_runner.py``: the resource owns the
   slot's plan state and scratch, the runner owns the captured graph and
   replays it. Departs from ``resource_execution_model.md`` ("the engine owes
   the slot") — see F., G.

Bare line refs are ``cuda_graph_runner.py``. Letter refs are the GAPS list at
the bottom of this file.
"""

# 1. At load each node has: which resources exist, and how they are keyed.
#
#     def node_resource_specs(self, ...) -> list[NodeResourceSpec]:
#         return [
#             KVSpec(label="kv", nodes={"llm"}, config=KVConfig(...)),
#             AttentionSpec(label="attn", nodes={"llm"},
#                           config=AttentionConfig(kv_cache="kv")),
#             PositionSpec(label="rope", nodes={"llm"}, config=...),
#         ]
#
#    engine calls Resource.build(spec, device, **kw) once per spec and
#    hands the node a dict[str, Resource]. AttentionSpec names its KV cache by
#    key. key is returned by depends_on() returns and what runner will topo-sort
#    on.
#
# 2. Per step: what does the step do?
#
#     def declare_step(self, graph_walk, engine_inputs, inputs):
#         spans = tuple(inp.input_seq_len for inp in inputs)
#         segments = tuple(Segment(rid, "main", n) for rid, n in zip(rids, spans))
#         return segments, {
#             "kv":   KVStep(segments, write=True, commit=True),
#             "rope": PositionStep(segments, pos_ids=None, advance=spans),
#             "attn": AttentionStep(segments, causal=True, batched_key=None),
#         }
#
#    when cfg, the model emits one segment per (request, label) and sets
#    batched_key. segment list remains ordering authority
#
# everytime submodule forward calls resource operations we must declare `ResourceStep`
# layers call device ops by key and label:
#
#     class Attention(nn.Module):
#         def forward(self, hidden, *, label):
#             q, k, v = self.qkv(hidden)
#             q, k = self.rope.apply_qk(q, k, layer_idx=self.i, label=label)
#             self.attn.write_kv(k, v, layer_idx=self.i, label=label)
#             return self.o(self.attn.attend(q, layer_idx=self.i, label=label))
#
# self.attn / self.rope resolve once at load from node_resources[key].
# (slot is iffy here; see H.)
#
# NOTE @nsagan: I had envisioned the attn, rope, sampler, etc. being passed in through
# the ModelInputsFromEngine as a dict[str, Resource] instead of having the submodule
# hold a reference to the resource---I think it works either way and this _would_ mean
# not passing in arguments everywhere; I guess I'm just used to the resources being
# passed in as arguments.
#
# how to unify capture v eager:
#
#     def execute_step(self, batch, submodule):
#         segments, steps = submodule.declare_step(walk, ei, inputs)
#         ctx  = StepContext(rids, walk, slot=None, capture=False)
#         step = SubmoduleStep(ctx, segments, steps) # ctx.slot not known yet
#
#         # sweep 1: allocatoin
#         out = runner.admit(step, ctx)
#         if not out.ok:
#             return
#         step = runner.narrow(step, out)
#
#         # sweep 2: shape. separate from sweep 1 because a capture bucket
#         # depends on the token count, which is final only after size
#         # whichever resource owns slots picks one and writes ctx.slot here
#         # ** WE MAY NEED A GLOBAL SLOT WHERE MULTIPLE SLOTFUL RESOURCES LOCKSTEP THE SAME SLOT **
#         ctx = runner.reserve(step, ctx)
#
#         # `execution_for` does not exist.
#         # it reads ctx.slot / ctx.capture and packages the four calls that differ
#         # by execution path:
#         #
#         #   pad(step)      graph only. BASIC_BATCHED pads the batch axis;
#         #                  FLASH_INFER_PACKED pads the token axis. so `pad` has
#         #                  two implementations, decided by capture. identity: noop
#         #   stage(kwargs)  graph only. copy preprocess output into the
#         #                  static buffers. identity: noop
#         #   launch(...)    graph.replay() takes nothing and gives nothing.
#         #                  outputs are read back out of static_outputs;
#         #                  forward_batched(**kwargs) returns them directly.
#         #                  launch normalizes those two retrieval shapes.
#         #   release()      give the slot back.
#         #
#         # derived mostly from kv_cache_engine.py:1077-1082
#         # (graph | batched | sequential), and the slot lookup at
#         # :1282 / :1018. this is those, given a name and an object.
#         with self.execution_for(ctx) as binding:
#             # sweep 3: plan
#             step = binding.pad(step)
#             runner.plan(step, ctx)
#
#             kwargs = submodule.preprocess(...)
#             binding.stage(kwargs)
#             out = binding.launch(submodule, kwargs)
#
#         runner.commit(step, ctx) # counters advance, fork labels, etc.
#         ... # await completion
#         runner.publish(batch.request_ids)
#         runner.release(step, ctx) # sweep; slot gets freed
#
# runner.plan is a sweep over the step's keys in topological order:
#
#     kv.plan(step["kv"])                          -> views
#     rope.plan(step["rope"], deps={"kv": views})  -> pos_ids + advance
#     attn.plan(step["attn"], deps={"kv": views})  -> indptrs, planned wrapper
#
# NOTE @nsagan: we need to figure out how this intercts with the
# speculatve pre-planning. Tentative notes below:
# I. Current Implementation:
#    1. engine.reserve_replay_slot(batch): currently assumes BASIC_BATCHED cuda graph
#       config, and then tries to find a replay slot. If we do prepare_inputs first,
#       then we will have the relevant sequence length information  (probably preferable
#       as long as there is no perf downside)
#    2. Waits for prev_advance_event, which happens after runner.commit
#    3. engine.pre_plan_for_batch -> basically calls plan_attention on a separate stream
#       and sets a cuda event for the worker to wait on. There is also reset_pre_plan_for_batch,
#       which resets the slot (e.g., frees pages for dummy rids)

# II. Most aggressive plan:
#     - Split batch execution into two parts: First, prepare_inputs through plan,
#       which can happen speculatively. Second, the forward pass through the end.
#     - Some implications of this: (1) prepare_inputs is forced to be async-friendly,
#       i.e., no .cpu or .item, etc. (2) more resources have to be double-buffered, not
#       just attention (sampler and rope will have to be; KV cache is ok as long as it's
#       thread safe, which it should already be). (3) this is more wasted work in the
#       case of failed speculation. (4) future resources will be forced to work with the
#       async worker.
#     - However,this means that speculation happens "for free" in this design; also, gathering
#       of sampler parameters is also happening asynchronously (whether that is useful or not).
#       Also feels cleaner abstraction-wise.

# III. Less aggressive (but "feels hackier") plan:
#     - We add a function onto the Resource class called pre_plan (that is a no-op for
#       anything but attention for now)
#     - We also keep the restriction to batches under the BASIC_BATCHED cuda graph config
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mstar.conductor.request_info import SequenceInfo
from mstar.engine.cpu_page_pool import CPUPagePool
from mstar.engine.kv_store import (
    KVCacheConfig,
    KVTransferEngine,
    PageAllocator,
    StoreWritePolicy,
)
from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import AttentionStep, KVStep, Segment, StepContext
from mstar.utils.flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper


@dataclass(frozen=True)
class PagedStorage:
    """KV defined by config time. whole storage surface attention gets. see C."""
    tensor: torch.Tensor
    page_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    def layer(self, layer_idx: int) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class SequenceView:
    """KV holding for one segment stream. no pool ref, see B. and D."""
    page_indices: tuple[int, ...]
    start: int
    length: int

    def last_page_len(self, page_size: int) -> int:
        ...

@dataclass(frozen=True)
class Reservation:
    resident: int
    to_compute: int
    pending: bool = False


@dataclass(frozen=True)
class AdmitOutcome:
    """admit must return this; base has `-> None`. see A."""
    ok: bool
    reason: str | None = None
    per_segment: tuple[Reservation, ...] = ()
    token: Any = None # opaque to runner, threaded back to same resource. see O.


@dataclass
class PageArena:
    """physical storage and free list management"""
    tensor: torch.Tensor
    allocator: PageAllocator
    page_size: int

    def acquire(self, n: int) -> list[int] | None:
        """None on shortage, not a raise. see N."""

    def release(self, pages: list[int]) -> None:
        ...

    def copy_pages(self, src: list[int], dst: list[int]) -> None:
        ...

@dataclass(frozen=True)
class RetentionPolicy:
    """fifo retention of `context_budget`"""
    context_budget: int


@dataclass
class CacheStream:
    """(request, label) cache stream metadata

    replaces KVRequestState minus dense_prefix_kv. see J., N."""
    page_indices: list[int] = field(default_factory=list)
    stored_len: int = 0
    position: int = 0
    released: int = 0
    retention: RetentionPolicy | None = None
    read_pending: bool = False
    offloaded: bool = False


class PagedKVCacheResource(Resource):
    """Owns physical KV, page tables, stream metadata. Handles allocation and
    retention.

    Gives `SequenceView` from `plan` and `SequenceInfo` from `publish`.

    standalone; nothing delegated to KVCachePool or PagedAllocationManager.
    see N. for what absorbing those costs.
    """

    _arena: PageArena
    _config: KVCacheConfig
    _storage: PagedStorage
    # NOTE @nsagan: would prefer {rid -> {label -> CacheStream}} so that upon cleanup
    # we can just directly clean up all labels for a request
    _streams: dict[tuple[str, str], CacheStream]
    _cpu: CPUPagePool | None
    _transfer: KVTransferEngine | None
    # NOTE @nsagan: _write_policy has been dead code, so I might prefer to not add it
    # at the moment
    _write_policy: StoreWritePolicy

    @classmethod
    def build(cls, spec: NodeResourceSpec, device: torch.device, **kw) -> "PagedKVCacheResource":
        """allocates backing tensor and inits `PageAllocator` over the pages"""

    def storage(self) -> PagedStorage:
        """handed over once at build, never via SequenceView. see C., D."""



    def ingest_request(self, rid: str, overrides: ResourceReqConfig) -> None:
        ...

    def remove_request(self, rid: str) -> None:
        ...

    def _stream(self, rid: str, label: str) -> CacheStream:
        ...

    def admit(self, step: KVStep, *, ctx: StepContext) -> AdmitOutcome:
        """reserve pages for each segment's span

        perform step.pre_forks [self._fork(rid, from, to)]. see A., N."""

    def narrow(self, step: KVStep, outcome: AdmitOutcome) -> KVStep:
        """cut down span to `to_compute`. not sure if necessary. would have to
        run before attention `admit`. see E.
        """

    def plan(self, key: str, step: KVStep, *, ctx: StepContext) -> dict[tuple[str, str], SequenceView]:
        """SequenceView per segment in segment order.

        view holds pages of stream and [released, stored_len + span]. does no allocation or mutation"""

    def commit(self, step: KVStep, *, advance: tuple[int, ...] | None = None) -> None:
        """annotates spans as computed. advances `stored_len` and `position`

        handles post_forks. can run before device work because only metadata."""

    def _fork(self, rid: str, from_label: str, to_label: str, realloc: bool = False) -> None:
        """fork one stream from another. copies pages past existing length"""




    def _flush(self, rid: str, label: str, layers: int | None = None) -> None:
        """see N."""

    def admit_retrieve(self, rid: str, published: Any) -> bool:
        """gives `True` once done with transfer. see N."""

    def publish(self, request_id: str) -> dict[str, SequenceInfo] | None:
        """SequenceInfo per stream. see N."""

    def offload_candidates(self) -> list[tuple[str, int]]:
        """computes list of eviction victims. see N."""

    def offload(self, request_id: str) -> int:
        """move all streams of rid to CPU"""

    def reload(self, request_id: str) -> bool:
        """bring back all streams into GPU"""



    def positions(self, request_id: str, label: str) -> int:
        # NOTE @nsagan: I think we can pull out PE into a differet resource;
        # I don't think it needs to ride on the paged KV resource
        """gets positions for PE planning. read only"""

    def labels(self, request_id: str) -> list[str]:
        """open streams for request"""

    @property
    def num_free_pages(self) -> int:
        ...


@dataclass
class AttentionArtifact:
    """returned by plan. for captured graph replay.

    every buffer built once at build, only written in place"""
    wrapper: FlashInferPrefillWrapper | FlashInferDecodeWrapper | None = None
    workspace: torch.Tensor | None = None
    kv_locations: torch.Tensor | None = None
    seq_lens: tuple[int, ...] = ()
    generation: int = -1 # to catch stale access


@dataclass(frozen=True)
class BucketKey:
    graph_walk: str
    requires_cfg: bool
    bs: int
    num_tokens: int


@dataclass
class AttentionSlot:
    """double-buffer slot with workspace and wrapper per-label

    see F. for ownership of captured graph or in/out buffers"""
    index: int
    bucket: BucketKey
    artifacts: dict[str, AttentionArtifact] = field(default_factory=dict)
    leased: bool = False
    graph_bound: bool = False
    plan_done_event: torch.cuda.Event | None = None   # from BatchedCacheManager


@dataclass(frozen=True)
class SlotLease:
    """attention `admit` gives to inform which slot to plan and replay

    has no clean channel to plan/commit/release. see O."""
    slot: int
    bucket: BucketKey | None # None -> eagre
    filler: tuple[Segment, ...]


class FlashInferAttentionResource(Resource):
    """owns execution for layers, static buffers, workspace, and graph slots

    to avoid nasty unpleasant entanglement, does not own storage or page table
    or request identity or allocator

    we have `admit`, `plan`, `commit`, `publish` on host. `write_kv` and `attend`
    (i.e. `invoke`) are captured"""

    _kv_key: str
    _storage: PagedStorage
    _slots: dict[BucketKey | None, list[AttentionSlot]] # eager is under key=None with transiet wrappers

    @classmethod
    def build(cls, spec: NodeResourceSpec, device: torch.device, **kw) -> "FlashInferAttentionResource":
        """takes KV's `storage()` and a workspace budget. see C."""

    def depends_on(self) -> set[str]:
        """`{self._kv_key}`"""

    def _build_slot(self, index: int, bucket: BucketKey) -> AttentionSlot:
        """pre-construction of per-(slot, label) buffers at max shape"""

    def ingest_request(self, rid: str, overrides: ResourceReqConfig) -> None:
        """noop

        not noop for dense-gen because keeps prefix. see J."""

    def remove_request(self, rid: str) -> None:
        """noop. see J."""

    def admit(self, step: AttentionStep, *, ctx: StepContext) -> AdmitOutcome:
        """lease cuda slot and reserve scratch space. `SlotLease` is in `AdmitOutcome.token`. kinda ugly

        see O. for why it is ugly and what to do."""

    def _select_bucket(self, graph_walk: str, bs: int, num_tokens: int) -> BucketKey | None:
        """finds smallest working bucket, or None if eager. we'd prefer `num_tokens` to come out of
        `KV::narrow`, otherwise we get big graphs. see E."""

    def _filler_segments(self, bucket: BucketKey, step: AttentionStep) -> tuple[Segment, ...]:
        """padds segments to bring to capture shape

        we can have fixed allocated pages for pad rid instead of transient per-step `_release_slot_step`.
        see G."""

    def plan(
        self,
        key: str,
        step: AttentionStep,
        *,
        ctx: StepContext,
        deps: dict[str, dict[tuple[str, str], SequenceView]],
        lease: SlotLease,
        stream: torch.cuda.Stream | None = None,
    ) -> AttentionArtifact:
        """build index tensors and plan wrapper

        is mostly `plan_attention` without allocation. `lease` is kind-specific
        in a method the runner calls generically, see O. signature also needs
        ctx/deps that base does not have, see A."""

    def _decode_locations(
        self, segments: tuple[Segment, ...], views: dict[tuple[str, str], SequenceView],
    ) -> torch.Tensor | None:
        """gets `[page_index, offset] per token segment for decode wrapper's in-graph KV writing

        no allocator"""

    def commit(self, step: AttentionStep, *, lease: SlotLease) -> None:
        """not strictly necessary. does lifetime bookkeeping for slots after launch

        uses the -1 flag to avoid stale. only earns its place because we own
        slots, see F."""

    def release(self, lease: SlotLease) -> None:
        """should release slot once done see F."""

    def publish(self, request_id: str) -> object | None:
        ...

    def slot_state(self, lease: SlotLease) -> dict[str, AttentionArtifact]:
        """gets artifacts from bucket. does job of CudaGraphSlot.static_cache_manager

        ensures region records against proper buffers see G."""

    def bind_graph(self, lease: SlotLease) -> None:
        """notate graph was captured against `SlotLease` buffers. replay is now valid :D

        see G."""

    def write_kv(self, k: torch.Tensor, v: torch.Tensor, *, layer_idx: int, label: str, slot: int = 0) -> None:
        """`_artifact(slot, label).wrapper.set_kv_cache(storage.layer(i), k, v)`"""

    def attend(self, q: torch.Tensor, *, layer_idx: int, label: str, slot: int = 0) -> torch.Tensor:
        """`_artifact(slot, label).wrapper.run(q, storage.layer(i))`

        tokens of segment i are rows [qo_indptr[i], qo_indptr[i+1]] from `ResourceStep`
        engine therefore can reconstruct rid -> tokens without attention knowledge of rid"""

    def qo_indptr_buf(self, label: str, slot: int = 0) -> torch.Tensor | None:
        """prefill wrapper has persistent qo_indptr. fetch for captured paths
        recovering per-request token boundaries from graph internal"""

    def _artifact(self, slot: int, label: str) -> AttentionArtifact:
        """`self._slots[...][slot].artifacts[label]`

        model does not have the slot; we do. see H."""


# gaps
#
#   A. Resource signatures are too narrow
#   B. Boundary values have no home
#   C. Attention needs the KV tensor, and only that
#   D. SequenceView must not carry a pool
#   E. admit needs an order, with narrow in the middle
#   F. CudaGraphSlot must be split
#   G. cuda_graph_runner must take an externally-owned slot
#   H. The device op needs a slot the model does not have
#   I. nvrmind; kept in here so no reindexing of alphabet needed
#   J. Dense-gen keeps per-request state
#   K. nevermind
#   L. nvrmind
#   M. nvrmind
#   N. Standing alone means absorbing four things
#   O. The slot lease has no clean channel

# A.
# base.py:37-47 gives admit(step) / plan(key, step) / commit(step). None of
# them carry StepContext (slot, capture, rids) or the dependency payload, and
# both are load-bearing: plan cannot find its slot or its views without them.
# admit also returns None though it is the only stage that can fail.


# B.
# Reservation / SequenceView / PositionPlan were dropped from base.py. We need
# these to move information from one resource to another. Runner will do so via the
# topological `depends_on` traverse.


# C.
# write_kv and wrapper.run must index storage, but nothing above KV may hold
# an allocator.
#
# Solved by defining PagedKVCacheResource.storage() -> PagedStorage. Can only
# read, and with strict semantics.


# D.
# Today's SequenceView.pool (kv_pool.py:166) exists so callers can reach
# page_size. It gives entire pool to attention along with allocator. Ewwww


# E.
#     KV.admit -> narrow -> attention.admit -> plan (topo order)
#
# Attention's bucket comes from a token count that comes from narrow.
# Need to make narrow a sweep so that Runner remains agnostic to Resources.


# F.
# Four of CudaGraphSlot five fields (:67-82) are model-wide. Attention cannot own
# slots without screwing up ownership.
#
# Fix: pair by index:
#     GraphSlot      (runner)    graph + static in/out buffers
#     AttentionSlot  (resource)  wrappers, workspaces, lease
#
# Attention remains authority on slot leasing. Gives slot number in `admit`,
# engine writes to StepContext, and positions/sampler read there. So,
# reserve_slot (:1018) and its next_slot counter must delegate to the lease,
# or the two disagree about which slot is in flight.
#
# Open: doth `release` becomes a sixth lifecycle stage of runner?


# G.
# `cuda_graph_runner` takes externally owned slot. hmmmm.
# run(slot=) already threads an index (:1257), so replay kinda works.
#
# Changes:
#   1. Drop static_cache_manager from CudaGraphSlot; capture takes
#      slot_state(lease) instead (replaces :323-397). KILL BATCHEDCACHEMANAGER
#   2. Delete _address_slot / _reset_slot_addressing / _slot_step_ids
#      (:1785-1814). They exist because the facade holds request
#      identity. these resources in `sketch` hold none.


# H.
# attend(q, layer_idx=3, label="main") comes from an nn.Module that knows
# nothing about steps, but slot-indexed artifacts key on (slot, label).


# I.
# nvrmind


# J.
# # run_dense caches the gathered frozen prefix on KVRequestState.dense_prefix_kv
# (attention.py:510-518).
#
# Split what we have now. Should live in the attention resource keyed
# (rid, label, layer), and ingest_request / remove_request are no longer
# no-ops for that backend. Unfortunately loses the invalidation that today
# comes from _new_state() zeroing the field when the prefix pages are
# reallocated, so must have an explicit signal. Per-stream generation
# counter on SequenceView, bumped by admit whenever page table change.


# K.
# nevermind


# L.
# nvrmind

# M.
# nvrmind

# N.
# PagedAllocationManager does this that we do not have:
#
#   1. Stream accounting (get_state / alloc / reset_label / add_request /
#      remove_request, kv_store.py:528-677) replaced by CacheStream (albeit
#      missing dense_prefix_kv)
#   2. Store durability (flush_to_store + the _key scheme, :513-522)
#   3. Transfers (sync_retrieve / start_async_retrieve / check_retrieve_ready
#      / wait_for_retrieves, :555-640). perhaps another object held by KV resource that
#      handles


# O.
# admit produces a SlotLease; plan, commit, release and slot_state all need it
# back. this sketch hands it off in two different ways. both feel icky.
#
#   1. AdmitOutcome.token, forward opaquely by `StepContext`. Works,
#     but the runner must hold a per-key token between two stages.
#   2. a `lease: SlotLease` keyword on plan, which violates the agnostisicm.
#
# Also has issue of engine knowing who owned it so that easy release can happen
# as a sweep. strong indication that perhaps global slot is the way to go.
