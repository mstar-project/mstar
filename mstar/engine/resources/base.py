from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.resident_hooks import ResidentHooks
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, BucketKey, ResourceStep, StepContext

if TYPE_CHECKING:
    # the config reaches back here through the submodule base, so keep the
    # import out of module exec
    from mstar.engine.accelerator_graph_config import (
        AcceleratorGraphConfig,
        AcceleratorGraphConfigType,
        PiecewiseAcceleratorGraphConfig,
        PiecewiseConfigType,
    )
    from mstar.engine.resources.attn.workspace import WorkspacePool
    from mstar.engine.resources.context_parallel import ContextParallel
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.utils import TextOutput


@dataclass(frozen=True)
class CGSlotSpec:
    bucket: BucketKey
    slot: int
    # a whole forward's capture, or one piecewise region's
    config: AcceleratorGraphConfig | PiecewiseAcceleratorGraphConfig
    config_idx: int | None = None

    @property
    def bs(self):
        return self.bucket.bs

    @property
    def num_tokens(self):
        return self.bucket.num_tokens

    def __str__(self) -> str:
        return f"{self.bucket} slot={self.slot}"


@dataclass(frozen=True)
class CGSlotKey:
    bucket: BucketKey
    slot: int
    label: str


def head_shards(group: JointGroups | None, tp_only: bool = False) -> int:
    """How many ways a node's ranks split its attention heads: its TP and SP ranks (CFG ranks run
    whole branches, every head), or TP alone (``tp_only``) where its pools place positions over the
    SP ranks (``PagedKVConfig.context_parallel``), or for a resource over pools every SP rank
    stores whole (``NodeResourceSpec.storage``)."""
    if group is None:
        return 1
    tp = group.tp_group.world_size
    return tp if tp_only else tp * group.sp_group.world_size


@dataclass(frozen=True)
class EngineResourceInfo:
    """What the engine has to offer a resource at build time.

    One struct rather than per-kind keyword arguments: a resource takes what it
    needs and ignores the rest, and a name that does not exist here is a
    TypeError rather than something silently swallowed by a ``**kwargs``.
    """
    device: torch.device
    joint_comm_group: JointGroups | None = None
    transfer_engine_info: "TransferEngineInfo | None" = None
    kv_dtype: torch.dtype = torch.bfloat16
    # the dtype this resource's nodes compute in (their autocast dtype); None when they
    # disagree or run outside autocast
    compute_dtype: torch.dtype | None = None
    # Whether this logical resource has consumers in another worker instance.
    needs_remote_transfer: bool = True
    # this resource's local nodes
    nodes: "frozenset[str] | None" = None
    # the specs this one named in `depends_on`, by resource key
    dependencies: "Mapping[str, NodeResourceSpec]" = field(
        default_factory=dict
    )
    # walk -> rows of it a step of its combined walk may hold (None uncapped),
    # for every walk of this resource's local nodes
    max_rows_in_combined: "Mapping[str, int | None]" = field(
        default_factory=dict
    )
    # the prefix caches this engine opens (`opening_prefix_caches`), decided
    # before any resource is built, for one that sizes itself by its own
    prefix_caches: frozenset[str] = frozenset()
    # whether every SP rank holds this resource's heads whole, so they split over TP alone: its
    # nodes' pools place positions over the SP ranks (``PagedKVConfig.context_parallel``), or it is
    # a pool stored ``copied`` or a resource over such pools alone (``NodeResourceSpec.storage``)
    replicated_over_sp: bool = False
    # the model's ``Model.text_output``, for a resource that reasons over its token
    # vocabulary (the sampler's grammars): a call, as it may load the tokenizer
    text_output: "Callable[[], TextOutput | None] | None" = None
    # what each step may still copy to the host once its outputs are sent, or on a pass that runs no step
    host_allowance: StepAllowance | None = None

    @property
    def head_shards(self) -> int:
        """How many ways the node's ranks split its attention heads, which each builder sizes head
        counts by (`head_shards`). A ragged resource splits over TP only, unless it attends a label
        its node's sequence shard splits (``RaggedAttentionSpec.sp_exchange``, derived): over TP x SP
        then, even where the node's pools place positions."""
        return head_shards(self.joint_comm_group, self.replicated_over_sp)

    def dependency(self, key: str) -> NodeResourceSpec:
        spec = self.dependencies.get(key)
        if spec is None:
            raise KeyError(
                f"resource {key!r} was not resolved; declare it in the "
                "spec's `depends_on`"
            )
        return spec


class CachedPrefix(NamedTuple):
    """How much of a walk's input a resource already holds for ``label``.

    ``tokens`` counts the walk's leading inputs held, and ``position`` is the
    stream's position counter once they are skipped.
    """
    label: str
    tokens: int
    position: int


@dataclass
class CachedLease:
    """Rank 0's lease of a stream's cached prefix, which a TP follower takes as given: its ``pages``, and the
    ones it ``lent``, which another request holds and so take no room."""
    pages: list[int]
    lent: list[int]


class Residency(IntEnum):
    """Where a request's state on a node is: on the device, copied to the host, or dropped for a rebuild
    from its ids. Ordered by how far the state is from the device."""
    RESIDENT = 0
    ON_HOST = 1
    DROPPED = 2


class Rebuild(NamedTuple):
    """What a dropped request's rebuild writes past its layout: the ids it ``sampled`` since the layout was laid out,
    but the newest, which the next decode step reads."""
    sampled: Sequence[int]


# What a step may queue for the host. The device has one device-to-host copy engine, which runs copies in the order
# they were queued whatever their stream or priority, and the copies queue behind the step's own output copy. This
# bounds how long a step's copies to the host hold the copy engine before the next step's output copy can run (about
# 1 ms on an H100).
_WRITE_BYTES_PER_STEP = 32 << 20


class StepAllowance:
    """What a step may still queue for the host once its outputs are sent, shared by every copy to the host on the
    device: the worker's host copies for rebuilds first, then each pool's cached pages. The engine refills it once
    the pools have taken theirs, and the worker refills it for its host copies on a pass of its loop that runs no
    step, once its last copy has landed."""

    def __init__(self, nbytes: int = _WRITE_BYTES_PER_STEP) -> None:
        self._per_step = nbytes
        self.left = nbytes

    def spend(self, nbytes: int) -> int:
        """Take up to ``nbytes`` of what is left, and return how many were taken."""
        taken = min(nbytes, self.left)
        self.left -= taken
        return taken

    def refill(self) -> None:
        self.left = self._per_step


class KeyedWalks(NamedTuple):
    """A keyed label's walks, as a resource's prefix cache takes them: the walks whose
    writes are keyed, and the growth walks whose sampled ids extend the keys."""
    writes: tuple[str, ...]
    growth: tuple[str, ...] = ()


class MoveKind(IntEnum):
    """A kind of move that rank 0 of a TP node makes on a request and each TP follower replays."""
    OFFLOAD = 0
    RELOAD = 1
    RESERVE = 2
    UNRESERVE = 3
    EARMARK = 4
    DROP = 5
    LAND = 6
    GUARANTEE = 7
    ALLOT = 8
    WAIT = 9
    FORGET = 10


class Resource(ResidentHooks, ABC):
    @classmethod
    @abstractmethod
    def build(
        cls, spec: NodeResourceSpec, info: EngineResourceInfo,
    ) -> "Resource":
        ...

    def depends_on(self) -> set[str]:
        return set()

    def can_capture(self) -> bool:
        return True

    # this resource's key among the engine's resources, set by `build_resource`
    resource_key: str | None = None

    # Whether this resource's kernels can be left out of a step whose rows all allow
    # it (``lean_for``). The engine then captures a lean twin of each bucket whose
    # step drives the resource and whose kind it accepts (``lean_captures``), keyed
    # by ``BucketKey.lean``, and names the resource in ``StepContext.lean`` for a
    # step that runs it lean, eager steps included; the resource checks that at plan.
    # Each rank makes the choice from its own state, so the resource's lean and
    # plain kernels must issue the same collectives.
    lean_capture: bool = False

    def lean_for(self, request_ids) -> bool:
        """Whether every one of ``request_ids`` lets this ``lean_capture`` resource
        run lean. Read from state every rank of a lockstep instance sets alike (its
        ingests), on the thread that leases the step's graph."""
        return False

    def lean_captures(
        self, config_type: AcceleratorGraphConfigType | PiecewiseConfigType,
    ) -> bool:
        """Whether a capture of ``config_type`` gets this resource's lean twin."""
        return True

    def outside_step(self) -> AbstractContextManager[None]:
        """Run a ``lean_capture`` resource's callers outside any planned step (a
        compile's warm-up) on this thread, as in a step that runs it lean."""
        return nullcontext()

    # Whether this resource's step on a node is the node's own segments
    # (``SubmoduleStep.segments``: a request's span of tokens each, in the order the node
    # packs its rows), so the node declares nothing for it: ``engine_step`` adds a
    # ``ResourceStep`` of them (and of its ``combined_labels``) under the resource's key (the
    # LoRA pool, whose rows are the node's tokens; the sequence shard).
    steps_from_segments: bool = False

    # stays correct when a hit skips the front of a request; required of every resource on a keyed node
    prefix_skip_safe: bool = False
    # stays correct when a pool it depends on has a `placement`; the engine refuses to load it otherwise
    supports_context_parallel: bool = False
    # stays correct on a node whose steps split over its SP ranks (`SeqShardSpec`); the engine
    # refuses to load it on one otherwise
    supports_split_steps: bool = False

    @property
    def placement(self) -> "ContextParallel | None":
        """How this resource places token positions over ranks (`PagedKVConfig.context_parallel`);
        None for one that doesn't."""
        return None

    def enable_prefix_cache(self, root: bytes, walks: dict[str, KeyedWalks] | None = None) -> None:
        """Open whatever this resource keeps across requests, under ``root``. The
        engine calls it, before any request, on the caches `opening_prefix_caches` opens."""
        return

    def close_prefix_cache(self) -> None:
        """Keep nothing across requests. The engine calls it, before any
        request, on every other resource."""
        return

    def bind_dependencies(self, resources: Mapping[str, "Resource"]) -> None:
        """The live resources this one's spec names in `depends_on`, once all are built."""
        return

    def resolve_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs: Any,
    ) -> CachedPrefix | None:
        """How much of this walk's leading inputs this resource already holds.

        None leaves the answer to other resources; 0 tokens means nothing can be skipped.
        """
        return None

    def cap_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, prefix: CachedPrefix,
    ) -> CachedPrefix:
        """The longest prefix up to ``prefix`` this resource can serve, for a
        resource that holds only some lengths."""
        return prefix

    @property
    def chunk_alignment(self) -> int:
        """Tokens a cut prefill chunk is a multiple of, so that each chunk starts on a
        position this resource can keep state at (`Engine.chunk_alignment`). 1 cuts anywhere."""
        return 1

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str,
        inputs: Any, prefix: CachedPrefix | None,
    ) -> None:
        """Take on the prefix every resource agreed to skip, or None if there is none.

        The inputs are untrimmed, so a resource that needs the tokens being
        skipped can still read them.
        """
        return

    def extend_prefix_chain(
        self, rid: str, node_name: str, graph_walk: str, outputs: Any,
    ) -> None:
        """Take this step's sampled tokens, so what was generated can be keyed."""
        return

    @property
    def keys_prefix_chains(self) -> bool:
        """Whether ``extend_prefix_chain`` can do anything right now (a prefix
        cache is open). The runner skips the per-request sweep otherwise. A
        resource that overrides ``extend_prefix_chain`` but not this is swept
        every step."""
        return False

    def after_output_copy(self, step: ResourceStep) -> None:
        """Called on the worker's main thread once ``step`` has finished on the device and its outputs are
        sent. A copy to the host queued here runs behind that step's own output copy."""
        return

    def complete_cached_walk(self, rid: str, node_name: str, graph_walk: str) -> None:
        """Called for a walk the cache served whole, since that walk never admits or commits."""
        return

    def lease_cached_prefix(self, rid: str, node_name: str, graph_walk: str) -> CachedLease | None:
        """On rank 0 of a TP node, lease the request's cached prefix as its step is sent and
        return the lease, or None."""
        return None

    def adopt_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, lease: CachedLease, caught_up: bool,
    ) -> bool:
        """On a TP follower, take rank 0's ``lease``. False until this rank holds its pages too, and
        raises if it still lacks them once ``caught_up``."""
        return True

    def release_cached_prefix(self, rid: str, node_name: str, graph_walk: str) -> None:
        """Give back what `lease_cached_prefix` or `adopt_cached_prefix` took, for a step that will not run."""
        return

    def page_trail(self) -> tuple[int, ...] | None:
        """What a TP follower checks against rank 0's before each step, such as how many page
        allocations this resource has made and a hash of their pages in order. None where it
        keeps no trail."""
        return None

    @classmethod
    def journals_resident_moves(cls) -> bool:
        """Whether rank 0 decides this resource's state for the other ranks of a lockstep
        instance outside the step that uses it (eviction, prefix leases, page trails). They
        replay each decision with the next step of the node it was journaled under, so
        validation refuses lockstep nodes on one worker sharing such a resource. Asked of
        the class a spec's ``resource_class`` names, so the hooks live there, not only on
        a subclass its ``build`` returns."""
        return any(getattr(cls, hook) is not getattr(Resource, hook)
                   for hook in ("supports_eviction", "lease_cached_prefix", "page_trail"))

    def fingerprint(self) -> bytes | None:
        """What this resource contributes to the prefix cache's root.

        Everything a stored page depends on beyond its own tokens. None leaves
        the resource out of the root, which is right only where it cannot
        change what a page holds.
        """
        return None

    # Request lifecycle

    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None):
        """Open the request's state. The worker calls this once per request, on
        its first NewRequest, though one comes per partition it serves."""
        return

    def ingest_capture_row(self, rid: str):
        """Open a CUDA graph capture padding row; by default, as a request."""
        self.ingest_request(rid, None)

    def remove_request(self, rid: str):
        return

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: "PublishedInfo | None"
    ) -> AdmitOutcome:
        """
        Takes the output of publish, possibly from another device, and kicks
        of a retrieval if needed (e.g., PD disaggregation KV transfer).
        Returns whether the retrieve has completed. Not ready also covers
        per-request state the resource is still preparing (a grammar compiling).
        """
        return ADMIT_OK

    # Admission. A resource with limited capacity answers for it, and the runner decides who
    # gets it and when: one order across resources, a request takes all it asks or none, and
    # nothing is retrieved for it before then.

    def ask(self, rid: str, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        """`ASKS` while ``rid`` waits for this resource's capacity to start on ``node_name``, None
        when it needs none, or the outcome that fails it when it can never fit."""
        return None

    def reserve(
        self, rid: str, node_name: str, graph_walk: str, published: "PublishedInfo | None",
    ) -> object | None:
        """Take what ``rid`` asked for and return None, or return a mark of what it didn't fit
        against for `grew_since`."""
        return None

    def reserve_past(
        self, rid: str, node_name: str, graph_walk: str, published: "PublishedInfo | None",
    ) -> object | None:
        """`reserve` for ``rid`` ahead of an earlier request that can't start: take what ``rid`` asked for, or
        return a mark. When this resource refused that request and hasn't grown since, it leaves that request what
        it needs. The runner calls it only while ``rid`` passes, or takes nothing, at the resource that refused that
        request, or whose queue the oldest request holds back while it waits to return. The base refuses, so a
        resource that doesn't override it admits in order.

        A resource that overrides it asks nothing more of a waiting request (`ask` returns None) only once `reserve`
        would take nothing for it, and the request may then start wherever it stands in this resource's queue."""
        return 0

    def unreserve(self, rid: str) -> None:
        """Give back what `reserve` took, because another resource ``rid`` asks didn't fit."""
        return

    def grew_since(self, mark: object) -> bool:
        """Whether a request that didn't fit against ``mark`` may fit now."""
        return True

    # Step lifecycle

    def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
        """
        Reserve space for the given step. In the case where requests in a batch must
        be executed sequentially, this may be called for all requests in a loop
        before the per-request plan -> forward -> commit cycle.
        """
        return ADMIT_OK

    def rollback_admit(self, step: ResourceStep, ctx: StepContext) -> None:
        """Give back whatever the live reservation for this step took.

        Called on a refused admit, for every resource the step reached including
        the one that refused.

        Only what was freshly taken. A page retained from a shared prefix, or a
        lease converted onto a stream, belongs to whoever else holds it.
        """
        return

    def plan(self, step: ResourceStep, ctx: StepContext) -> Any:
        """ret is immutable and opaque to runner; only gives to `ctx.plan_results`"""
        return None

    def commit(self, step: ResourceStep, ctx: StepContext) -> None:
        """record step consumption"""
        return

    def step_host_rows(self) -> dict[str, torch.Tensor]:
        """Row-addressed tensors of the step just run, copied to the host with
        the stop buffers. Read right after the forward and not cloned."""
        return {}

    def publish(self, request_id: str) -> "PublishedInfo | None":
        return None

    def publish_for_step(
        self,
        request_id: str,
        node_name: str | None,
        graph_walk: str | None,
    ) -> "PublishedInfo | None":
        """Publish after one node step, with context for selective exporters.

        Existing resources remain compatible through ``publish``; resources
        that need node/walk context can override this method.
        """
        del node_name, graph_walk
        return self.publish(request_id)

    def publish_after_stop(
        self,
        request_id: str,
        node_name: str | None,
        graph_walk: str | None,
    ) -> "PublishedInfo | None":
        """Publish state that is useful only when a dynamic loop stops."""
        del request_id, node_name, graph_walk
        return None

    def publish_snapshot_batch(
        self,
        request_ids: list[str],
        node_name: str | None,
        graph_walk: str | None,
    ) -> list:
        """``publish_snapshot_for_step`` for every request of a step, one
        entry per request in order (None for nothing to publish). A resource
        with per-request locking or lookups overrides this to do them once."""
        return [
            self.publish_snapshot_for_step(
                rid, node_name=node_name, graph_walk=graph_walk,
            )
            for rid in request_ids
        ]

    def publish_snapshot_for_step(
        self,
        request_id: str,
        node_name: str | None,
        graph_walk: str | None,
    ) -> Any:
        """A cheap record of what ``publish_for_step`` would export right now.

        The worker finishes it with ``publish_from_snapshot`` only for the
        requests whose frame or completion carries the publication, which in a
        steady decode loop is none of them. The default takes the eager
        publication itself, so a resource is correct without overriding; one
        whose publish builds something per request (a page list, a transfer
        descriptor) overrides both to record a few scalars here instead.
        """
        return self.publish_for_step(
            request_id, node_name=node_name, graph_walk=graph_walk,
        )

    def publish_from_snapshot(
        self,
        request_id: str,
        snapshot: Any,
        node_name: str | None,
        graph_walk: str | None,
    ) -> "PublishedInfo | None":
        """Finish a ``publish_snapshot_for_step`` record into the published info."""
        del request_id, node_name, graph_walk
        return snapshot

    def reset_request(self, rid: str, free: bool=False):
        """Clear a dummy rid's state: during accelerator graph capture, after each replayed step
        and when a pre-plan is discarded."""
        return

    # Pre-planning

    @property
    def supports_preplan(self):
        return False

    @property
    def force_double_buffer(self):
        """Whether this resource needs double-buffering even off the pre-plan path.

        A resource that stages a step's layout into a reused host buffer and
        issues a non-blocking H2D into a graph-read device buffer has a race the
        moment the CPU runs ahead of the GPU: plan(N+1) can overwrite the buffer
        before step N's DMA has retired, and the replay attends with N+1's data.
        The buffer need not be ours — FlashInfer's ``plan`` holds one per
        wrapper, which is why the attention managers key theirs per slot.
        The main runner already double-buffers whenever a resource
        ``supports_preplan``; this flag extends that to a runner that has no
        pre-plan path but still replays such a resource — notably the piecewise
        runner — and makes ``_exec_per_request`` fence before it reuses a slot,
        since its sub-steps all run inside one step. Off by default; a resource
        opts in only if it has this hazard.
        """
        return False

    def clear_preplan(self):
        return

    # Eviction
    #
    # A resource that holds enough per-request state to be worth reclaiming
    # (the KV cache, today) opts in here; the worker picks victims and drives
    # the move. Which requests exist and how recently they ran is the
    # scheduler's knowledge, so the resource only answers for its own state.

    @property
    def supports_eviction(self):
        return False

    # Whether `drop` discards every piece of a request's state here that a rebuild from its token ids
    # writes again, so the rebuild never runs over stale state. A node with a rebuild policy needs it of
    # every resource it holds (`Engine.load_model`).
    supports_drop = False

    def residency(self, rid: str) -> Residency:
        return Residency.RESIDENT

    def offload(self, rid: str) -> int:
        """Move the request's state off-device. Returns what was reclaimed."""
        return 0

    def check_drop(self, rid: str, tokens: int) -> str | None:
        """Why ``drop`` would refuse ``rid``, whose rebuild writes ``tokens`` tokens, or None. Changes nothing."""
        return None

    def drop(self, rid: str, tokens: int) -> int:
        """Discard the request's state that can be recomputed from the ``tokens``
        token ids its rebuild writes, and keep the rest. Returns what was reclaimed."""
        return 0

    def prepare_rebuild(self, rid: str, rebuild: Rebuild) -> None:
        """Take in what the dropped request's rebuild writes, before its first chunk runs."""
        return

    def land_rebuild(self, rid: str) -> None:
        """The dropped request's rebuild has run its last chunk, and that chunk's commit has run, so the resource
        holds everything the rebuild wrote."""
        return

    def reload(self, rid: str) -> bool:
        """Bring it back. False when it doesn't fit on device yet."""
        return True

    def can_offload(self, rid: str) -> bool:
        """Whether the request may be moved off the device to free room. If this is False, the request is no
        eviction candidate however much it holds."""
        return True

    def can_reload(self, rid: str) -> bool:
        """Whether the request can be reloaded. If this is False, the Engine
        will not attempt to reload. If True, it will attempt to reload but still
        check that the reload was successful."""
        return True

    def reclaimable(self, rid: str) -> int:
        """What `offload` would free, in whatever this resource counts.

        0 means the request holds nothing worth taking, so it is not an
        eviction candidate however cold it is. Distinct from
        `get_offload_priority`, which orders candidates rather than sizing them.
        """
        return 0

    def get_offload_priority(self, rid: str) -> float:
        """How much this resource wants ``rid`` gone, higher being more.

        Only consulted under a PRIORITY eviction policy, which names the
        resource to ask; LRU never calls it.
        """
        return 0.0

    # Engine lifecycle

    def check_captures(self, node: str, configs: list[AcceleratorGraphConfig], piecewise: Mapping) -> None:
        """Raise ``ValueError`` for a capture of ``node`` this resource can't replay, before any
        is taken: ``configs`` its whole-forward capture configs, ``piecewise`` its piecewise
        runners by region."""
        return

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec],
        max_bs: int, max_seq_len: int
    ) -> None:
        """Size whatever the captured replays will read.

        Called once per runner that captures against this node — the whole
        forward's, and one per piecewise region — so it must tolerate repeated
        calls: grow to the largest shape asked for, never clobber what an
        earlier call already sized.
        """
        # NOTE @nsagan: this should probably be refined; it was just the first
        # thing that came to mind
        return

    def seal_capture_buffers(self) -> None:
        """Every bucket on the device has been sized and none captured yet: from here on,
        what a captured graph reads keeps its address. Called once, by the engine."""
        # A no-op is safe for a resource that never reallocates what a graph reads once it
        # is allocated (at build, in `build_accelerator_graph_buffers`, or at a bucket's first
        # plan during sizing). The one kind a plan grows is a FlashInfer workspace:
        # `AttentionResource` seals the pools they come from, and the dense backend's own,
        # outside a pool, is never captured.
        return

    def post_warmup_validate(self):
        """
        For, e.g., the KV cache to check that num_free_pages is identical
        across TP ranks after cuda graph capture.

        Raises an error (fails loudly) if invalid.
        """
        return

    def cleanup(self):
        return


class AttentionResource(Resource):
    """A resource a layer stack calls per layer, under one plan label.

    Adds the label / layer-index cursors: a caller running the whole stack sets
    them once instead of threading them through every call, and an explicit
    argument still supersedes. Class-level defaults so a subclass picks them up
    without touching its __init__; subclasses that read them clear them in
    `plan`, so a step that never binds cannot inherit the previous step's.

    The KV, attention and cross-attention resources; not the sampler or the
    position resource, which are called once per step rather than per layer.
    """

    _default_label: str = "main"
    _default_layer_idx: int | None = None
    # FlashInfer's float workspaces, on a resource that plans FlashInfer. Kept here so the
    # seal is too: a workspace grows at a plan that overflows it, which past capture would
    # free memory a graph reads.
    _workspaces: WorkspacePool | None = None

    @property
    def default_label(self) -> str:
        return self._default_label

    @torch.compiler.disable
    def set_default_label(self, label: str) -> None:
        self._default_label = label

    @torch.compiler.disable
    def set_default_layer_idx(self, layer_idx: int) -> None:
        self._default_layer_idx = layer_idx

    def reset_default_cursors(self) -> None:
        self._default_label = "main"
        self._default_layer_idx = None

    def seal_capture_buffers(self) -> None:
        if self._workspaces is not None:
            self._workspaces.seal()


class PublishedInfo(ABC):
    @abstractmethod
    def update(self, other: "PublishedInfo") -> None:
        ...

    @abstractmethod
    def clone(self) -> "PublishedInfo":
        """Return independent publication metadata for deferred sending."""
        ...


def build_resource(spec: NodeResourceSpec, info: EngineResourceInfo) -> Resource:
    resource = spec.resource_class.build(spec, info)
    resource.resource_key = spec.resource_key
    return resource
