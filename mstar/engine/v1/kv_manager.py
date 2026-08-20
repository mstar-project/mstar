import threading
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CGSlotSpec, PublishedInfo, Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, ResourceType
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AllocationFailed,
    KVStep,
    Segment,
    StepContext,
    group_by_plan_label,
)
from mstar.engine.v1.cpu_page_pool import CPUPagePool
from mstar.engine.v1.kv_cache import KVCache, KVConfig, PageAllocator
from mstar.engine.v1.kv_transfer import KVTransferManager, TransferEngineInfo


@dataclass
class PageArena:
    """physical storage and free list management"""
    kv_cache: KVCache
    allocator: PageAllocator

    def acquire(self, n: int) -> list[int] | None:
        return self.allocator.try_allocate(n)

    def release(self, pages: list[int]) -> None:
        return self.allocator.free(pages)

    def copy_pages(self, src: list[int], dst: list[int]) -> None:
        self.kv_cache.copy_pages(src, dst)

    @property
    def num_free(self):
        return self.allocator.num_free


@dataclass(frozen=True)
class RetentionPolicy:
    """fifo retention of `context_budget`"""
    context_budget: int


@dataclass
class CacheStream:
    """(request, label) cache stream metadata"""
    page_indices: list[int] = field(default_factory=list)
    stored_len: int = 0
    position: int = 0
    released: int = 0
    retention: RetentionPolicy | None = None
    read_pending: bool = False
    read_future: Future | None = None
    offloaded: bool = False
    generation: int = 0

    def reset(self, freed: bool=False):
        self.stored_len = 0
        self.position = 0
        self.released = 0
        self.generation += 1

        if freed:
            self.page_indices.clear()


LabelToStream = dict[str, CacheStream]

@dataclass
class KVSequenceInfo:
    seq_len: int
    # for tracking KV cache
    latest_kv_transfer_info: Any
    page_indices: list[int] = field(default_factory=list)


@dataclass
class PublishedKVInfo(PublishedInfo):
    # {rank -> {label: SequenceInfo}}
    info: dict[int, dict[str, KVSequenceInfo]] = field(default_factory=dict)
    world_size: int = 1

    @classmethod
    def build_for_rank(
        cls, rank: int, world_size: int,
        seq_info: dict[str, KVSequenceInfo]
    ):
        return cls(
            info={rank: seq_info},
            world_size=world_size
        )

    def update(self, other: "PublishedKVInfo"):
        for key, val in other.info.items():
            if key not in self.info:
                self.info[key] = val
                continue
            self.info[key] = {
                **self.info[key],
                **val
            }

    def get(self, rank: int) -> dict[str, KVSequenceInfo]:
        return self.info.get(rank, {})


@dataclass
class AllocResult:
    success: bool = True
    error: AllocationFailed | None = None


class SequenceView(NamedTuple):
    """One (request, label) stream as this step sees it. A NamedTuple because
    it is built per segment per step and never mutated."""
    request_id: str
    label: str
    page_idxs: list[int]
    length: int # already resident + to_compute
    to_compute: int # new for this step
    start: int = 0
    generation: int = 0

    def last_page_len(self, page_size: int) -> int:
        return (self.start + self.length)  % page_size


class PagedIndptrs(NamedTuple):
    """The four int32 index tensors a FlashInfer prefill/decode wrapper's
    ``plan`` consumes, built on CPU (so wrapper.plan's ``.to("cpu")`` is a
    no-op — see ``FlashInferAttentionManager.plan``)."""

    # TODO: qo_indptr is only needed for prefill; we should avoid computing it
    # (+ doing the H2D) if possible during decode
    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor

    def to_device(self, device: torch.device):
        return PagedIndptrs(
            qo_indptr=self.qo_indptr.to(device, non_blocking=True),
            paged_kv_indptr=self.paged_kv_indptr.to(device, non_blocking=True),
            paged_kv_indices=self.paged_kv_indices.to(device, non_blocking=True),
            paged_kv_last_page_len=self.paged_kv_last_page_len.to(device, non_blocking=True),
        )

    def to_kwargs_dict(self):
        return dict(
            qo_indptr=self.qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len
        )


def build_paged_indptrs(
    segments: list[SequenceView],
    page_size: int,
) -> PagedIndptrs:
    qo_indptr = [0]
    kv_indptr = [0]
    all_pages: list[int] = []
    last_page_lens: list[int] = []
    for s in segments:
        qo_indptr.append(qo_indptr[-1] + s.to_compute)
        all_pages.extend(s.page_idxs)
        kv_indptr.append(kv_indptr[-1] + len(s.page_idxs))
        last_page_lens.append(s.last_page_len(page_size) or page_size)
    return PagedIndptrs(
        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        paged_kv_indices=torch.tensor(all_pages, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32),
    )


@dataclass
class KVPlanOutput:
    """
    Output of KVManager.plan for a single label
    """
    cpu_indptrs: PagedIndptrs
    # packing in plan order w/h 1 view per segment covered by plan
    views: list[SequenceView]
    # only the packed write addressing needs these on device, and only the
    # resource that builds them reads them; see KVManager._setup_plan_states
    cuda_indptrs: PagedIndptrs | None = None

    def get_total_len(self):
        return int(self.cpu_indptrs.qo_indptr[-1])

    @property
    def is_decode(self) -> bool:
        return all(view.to_compute == 1 for view in self.views)


# Page the padding tail of a captured step scatters into. Reserved at
# construction so no request is ever handed it; see KVPlanState.copy_.
SINK_PAGE = 0


@dataclass
class KVPlanState:
    token_to_page: torch.Tensor
    token_to_cache: torch.Tensor
    total_tokens: int | None = None

    def copy_(self, other: "KVPlanState", capture_len: int):
        """Stage a step's addressing into this captured state.

        Neutralize only ``[n:capture_len]`` — the slots the graph scatters
        beyond the real tokens (SINK_PAGE, else they hit another request's KV).
        Decode fills its bucket exactly (n == capture_len), so no-op there; only
        packed prefill pays it, over the real gap not the whole buffer.
        """
        assert other.total_tokens is not None
        n = other.total_tokens
        self.token_to_cache[:n].copy_(other.token_to_cache)
        self.token_to_page[:n].copy_(other.token_to_page)
        self.token_to_page[n:capture_len].fill_(SINK_PAGE)
        self.token_to_cache[n:capture_len].fill_(0)
        self.total_tokens = n


@dataclass
class KVReqConfig(ResourceReqConfig):
    # NOTE: this may need to be refined
    needed_labels: list[str] | None = None
    needed_labels_per_node: dict[str, list[str]] = field(default_factory=dict)
    needed_labels_per_node_walk: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    @property
    def resource_type(self):
        return ResourceType.KV_CACHE

    def apply_conductor_config(self, **kwargs):
        # does nothing rn; here because declaredb y abstract
        del kwargs

    def get_labels(self, node: str, walk: str):
        if (node, walk) in self.needed_labels_per_node_walk:
            return self.needed_labels_per_node_walk[(node, walk)]
        if node in self.needed_labels_per_node:
            return self.needed_labels_per_node[node]
        if self.needed_labels is not None:
            return self.needed_labels
        return ["main"]


@dataclass
class KVSpec(NodeResourceSpec):
    config: KVConfig

    @property
    def resource_type(self):
        return ResourceType.KV_CACHE

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        max_seq_len: int | None = None,
        cpu_offload_pages: int | None = None,
        **kwargs,
    ):
        """How much cache this deployment gets, and how it is cut up."""
        del kwargs  # keys meant for other resources
        for name, value in (
            ("max_num_pages", max_num_pages),
            ("page_size", page_size),
            ("max_seq_len", max_seq_len),
            ("cpu_offload_pages", cpu_offload_pages),
        ):
            if value is not None:
                setattr(self.config, name, value)


class KVManager(Resource):
    def __init__(
        self,
        cfg: KVConfig,
        name: str,
        joint_comm_group: JointGroups | None,
        transfer_engine_info: TransferEngineInfo,
        device: torch.device,
        dtype=torch.bfloat16,
    ):
        self.config = cfg
        if joint_comm_group is not None:
            # before the cache is allocated: it is sized off the head counts
            cfg.shard(joint_comm_group.world_size)
        self.kv_cache = KVCache(
            cfg, device, dtype
        )
        self.name = name

        self._arena = PageArena(
            kv_cache=self.kv_cache,
            allocator=PageAllocator(cfg.max_num_pages)
        )
        # take SINK_PAGE out of circulation; the allocator is FIFO from 0
        sink = self._arena.acquire(1)
        assert sink == [SINK_PAGE], f"expected page {SINK_PAGE} first, got {sink}"
        self._transfer = KVTransferManager(
            transfer_engine_info, self.kv_cache
        )
        self._cpu_pool: CPUPagePool | None = None
        if cfg.cpu_offload_pages > 0:
            self._cpu_pool = CPUPagePool(
                config=cfg, kv_cache=self.kv_cache,
                max_cpu_pages=cfg.cpu_offload_pages,
            )
        self._streams: dict[str, LabelToStream] = {}
        self._overrides: dict[str, KVReqConfig] = {}
        self._rank = joint_comm_group.rank if joint_comm_group is not None else 0
        self._world_size = joint_comm_group.world_size if joint_comm_group is not None else 1
        self._comm_group = joint_comm_group
        self._device = device
        self._lock = threading.RLock()

        # (slot, label) -> KVPlanState, sized for the largest capture bucket
        self._static_plan_states: dict[tuple[int, str], KVPlanState] = {}
        self._cg_max_seq_len = 0
        self._current_plan_states: dict[str, KVPlanState] = {}
        self._current_layer_idx: int = 0

        self._preplan_states: dict[str, KVPlanState] = {}
        self._preplanned = False
        self._cached_plan_output: dict[str, KVPlanOutput] | None = None

    @classmethod
    def build(
        cls, spec: KVSpec,
        device: torch.device,
        joint_comm_group: JointGroups | None,
        transfer_engine_info: TransferEngineInfo,
        dtype=torch.bfloat16,
        **kwargs
    ):
        return cls(
            cfg=spec.config,
            name=spec.label,
            device=device,
            joint_comm_group=joint_comm_group,
            transfer_engine_info=transfer_engine_info,
            dtype=dtype
        )

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ):
        del slots, max_bs
        # the per-(slot, label) buffers themselves are built on first plan for
        # that key (which labels a walk plans under is the step's to declare),
        # all at this one max length so they outlive any single bucket. Every
        # runner capturing against this node calls in, so keep the largest
        self._cg_max_seq_len = max(self._cg_max_seq_len, max_seq_len)

    def _static_plan_state(self, slot: int, label: str) -> KVPlanState:
        state = self._static_plan_states.get((slot, label))
        if state is None:
            state = self._static_plan_states[(slot, label)] = KVPlanState(
                token_to_cache=torch.zeros(
                    self._cg_max_seq_len, dtype=torch.long, device=self._device
                ),
                token_to_page=torch.full(
                    (self._cg_max_seq_len,), SINK_PAGE,
                    dtype=torch.long, device=self._device
                ),
            )
        return state

    def ingest_request(self, rid, overrides: KVReqConfig | None=None):
        if overrides is None:
            overrides = KVReqConfig()
        # guards `_streams`/`_overrides` against a concurrent admit/plan/commit
        # or reset/remove on another thread (see `_lock`)
        with self._lock:
            self._streams[rid] = {"main": CacheStream()}
            self._overrides[rid] = overrides

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: PublishedKVInfo | None
    ) -> AdmitOutcome:
        if published is None:
            return ADMIT_OK
    
        if published.world_size != self._world_size:
            raise RuntimeError(
                "KV cache transfer across TP world size is currently disallowed"
            )
        needed_labels = self._overrides[rid].get_labels(node_name, graph_walk)
        print(needed_labels)
        # one critical section: reading stored_len, comparing to published, and
        # firing the retrieve must be atomic against a concurrent commit/reset
        # (both non-blocking inside, so holding the lock is safe)
        with self._lock:
            for label, seq_info in published.get(self._rank).items():
                if label not in needed_labels:
                    continue
                print(label)
                if not self._check_ready(rid, label):
                    # read already in progress: admitted, just not ready yet
                    return AdmitOutcome(ok=True, ready=False)

                stream = self._ensure_label(rid, label)
                new_len = seq_info.seq_len
                old_len = stream.stored_len
                if new_len <= old_len:
                    continue

                # _alloc takes a total length, not a delta
                alloc_res = self._alloc(rid, label, new_len)
                if not alloc_res.success:
                    return AdmitOutcome(ok=False, reason=alloc_res.error)

                fut = self._transfer.start_async_retrieve(
                    start_len=old_len, end_len=new_len,
                    local_page_indices=stream.page_indices,
                    remote_page_indices=seq_info.page_indices,
                    kv_transfer_info=seq_info.latest_kv_transfer_info
                )
                stream.read_future = fut
                stream.read_pending = fut is not None
                stream.stored_len = new_len

            ready = all(self._check_ready(rid, label) for label in needed_labels)
        return AdmitOutcome(
            ok=True, ready=ready
        )

    def admit(self, step: KVStep, ctx: StepContext) -> AdmitOutcome:
        if self._preplanned and not ctx.is_preplan:
            # pages were already reserved by the preplan pass
            return ADMIT_OK
        # forks reserve here and copy later (plan for pre-, commit for post-),
        # so a step that never runs leaves pages resident but no page contents
        # moved — re-admitting it allocates nothing and re-copies nothing.
        # A post-fork copies the source *after* this step's spans land, so its
        # reservation covers them.
        growth = self._label_growth(step) if step.commit else {}
        forks = [(pre, 0) for pre in step.pre_forks] + [
            (post, growth) for post in step.post_forks
        ]
        # one critical section so the read-of-stored_len then alloc is atomic
        # against a concurrent reset/remove/commit on another thread
        with self._lock:
            for (from_label, to_label), extra in forks:
                for rid in ctx.request_ids:
                    alloc_res = self._reserve_fork(
                        rid, from_label, to_label,
                        extra=0 if not extra else extra.get((rid, from_label), 0),
                    )
                    if not alloc_res.success:
                        return AdmitOutcome(ok=False, reason=alloc_res.error)

            for segment in step.segments:
                if segment.span == 0:
                    continue
                stream = self._ensure_label(segment.request_id, segment.label)
                alloc_res = self._alloc(
                    segment.request_id,
                    segment.label,
                    segment.span + stream.stored_len
                )
                if not alloc_res.success:
                    return AdmitOutcome(ok=False, reason=alloc_res.error)
        # TODO: apply retention policy

        return ADMIT_OK

    def _sequence_views(self, segments: list[Segment]) -> list[SequenceView]:
        views = []
        for s in segments:
            stream = self._streams[s.request_id][s.label]
            views.append(SequenceView(
                request_id=s.request_id,
                label=s.label, page_idxs=stream.page_indices,
                length=s.span + stream.stored_len,
                to_compute=s.span,
                generation=stream.generation,
            ))
        return views

    def _compute_plan_state(
        self, cuda_indptrs: PagedIndptrs,
        total_tokens: int
    ) -> KVPlanState:
        qo_indptr = cuda_indptrs.qo_indptr
        paged_kv_indptr = cuda_indptrs.paged_kv_indptr
        paged_kv_last_page_len = cuda_indptrs.paged_kv_last_page_len
        paged_kv_indices = cuda_indptrs.paged_kv_indices

        # Compute per-token page and offset for vectorized KV writes
        n_req = qo_indptr.shape[0] - 1
        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)

        # Pages/lengths AFTER append
        num_pages_after = (
            paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        ).to(torch.int32)
        kv_len_after = (
            (num_pages_after - 1) * self.kv_cache.page_size + paged_kv_last_page_len
        )

        # Flatten to per-token indices
        # output_size keeps repeat_interleave from syncing to read `lens`
        seg = torch.repeat_interleave(
            torch.arange(n_req, dtype=torch.int32, device=self._device), lens,
            output_size=total_tokens
        )
        intra = torch.arange(
            total_tokens, dtype=torch.int32, device=self._device
        ) - torch.repeat_interleave(starts, lens, output_size=total_tokens)

        # Absolute KV position per token
        start_new = kv_len_after[seg] - lens[seg]
        g = start_new + intra

        # Map to page + offset
        page_off = torch.div(g, self.kv_cache.page_size, rounding_mode="floor").to(
            torch.int32
        )
        off_in_page = (g - page_off * self.kv_cache.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        return KVPlanState(
            token_to_page=paged_kv_indices[abs_page_ptr].to(torch.long),
            token_to_cache=off_in_page.to(torch.long),
            total_tokens=total_tokens
        )

    def _decode_plan_state(self, views: list[SequenceView]) -> KVPlanState:
        """Write addressing for a step appending one token per request.

        The packed path needs the indptrs on device and ~a dozen kernels to
        unpack them per token. A decode step's slot is just the end of each
        stream, so build it in the same CPU pass the views came from and send
        it over as one H2D.
        """
        page_size = self.kv_cache.page_size
        pages: list[int] = []
        offsets: list[int] = []
        for view in views:
            # off the stream's page count, not its logical length: that is what
            # `build_paged_indptrs` hands attention, so a stream holding more
            # pages than its length needs stays self-consistent
            pages.append(view.page_idxs[-1])
            offsets.append((view.last_page_len(page_size) or page_size) - 1)
        locations = torch.tensor(
            [pages, offsets], dtype=torch.long
        ).to(self._device, non_blocking=True)
        return KVPlanState(
            token_to_page=locations[0],
            token_to_cache=locations[1],
            total_tokens=len(views),
        )

    def _setup_plan_states(
        self, plan_output: dict[str, KVPlanOutput],
        ctx: StepContext
    ):
        for label, indptrs in plan_output.items():
            if indptrs.is_decode:
                plan_state = self._decode_plan_state(indptrs.views)
            else:
                indptrs.cuda_indptrs = indptrs.cpu_indptrs.to_device(self._device)
                plan_state = self._compute_plan_state(
                    indptrs.cuda_indptrs,
                    total_tokens=indptrs.get_total_len()
                )
            if ctx.slot_lease is not None:
                static_state = self._static_plan_state(ctx.slot_lease.slot, label)
                static_state.copy_(plan_state, ctx.slot_lease.bucket.num_tokens)
                plan_state = static_state
            if ctx.is_preplan:
                self._preplan_states[label] = plan_state
            else:
                self._current_plan_states[label] = plan_state


    def _plan_output(self, views: list[SequenceView]) -> KVPlanOutput:
        return KVPlanOutput(
            cpu_indptrs=build_paged_indptrs(views, self.kv_cache.page_size),
            views=views,
        )

    def plan(self, step: KVStep, ctx: StepContext) -> dict[str, KVPlanOutput]:
        """
        Returns list of sequence views per plan label
        """
        assert not (self._preplanned and ctx.is_preplan), (
            "KV preplan is already pending; clear_preplan before planning a "
            "different step ahead"
        )
        self._current_layer_idx = 0
        if self._preplanned:
            self._current_plan_states = self._preplan_states
            res = self._cached_plan_output
            # must reset here: otherwise the *next* step's admit still sees
            # `_preplanned` and skips its allocation
            self.clear_preplan()
            return res
        for (from_label, to_label) in step.pre_forks:
            for rid in ctx.request_ids:
                self._apply_fork(rid, from_label, to_label)
        res = {
            plan_label: self._plan_output(self._sequence_views(segments))
            for plan_label, segments in group_by_plan_label(
                step.segments, step.combined_labels
            ).items()
        }
        self._setup_plan_states(res, ctx)
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = res
        return res

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}
        self._cached_plan_output = None

    def commit(self, step: KVStep, ctx: StepContext):
        # atomic against admit_retrieve reading stored_len on another thread
        with self._lock:
            for segment in step.segments:
                if step.commit and segment.span > 0:
                    stream = self._streams[segment.request_id][segment.label]
                    stream.stored_len += segment.span
            # post-forks copy what this step just wrote, so they land after the
            # spans above are counted
            for (from_label, to_label) in step.post_forks:
                for rid in ctx.request_ids:
                    self._apply_fork(rid, from_label, to_label)
        # TODO: handle retention policy, free pages if not commit

    # Eviction

    @property
    def supports_eviction(self):
        return self._cpu_pool is not None

    def is_offloaded(self, rid: str) -> bool:
        return self._cpu_pool is not None and self._cpu_pool.is_offloaded(rid)

    def offload(self, rid: str) -> int:
        """Move every stream of ``rid`` to host memory. Returns pages freed.

        A stream whose pages don't fit on the host keeps them, so a partial
        offload still frees whatever did fit.
        """
        if self._cpu_pool is None or rid not in self._streams:
            return 0
        freed = 0
        for label in list(self._streams.get(rid, {})):
            with self._lock:
                stream = self._streams[rid].get(label)
                if stream is None or stream.offloaded or not stream.page_indices:
                    continue
                read_future = stream.read_future
                pages = list(stream.page_indices)
                geom = (stream.stored_len, stream.position, stream.released)
            # blocking work OUTSIDE the lock: drain the in-flight read, copy the
            # pages to host, sync so the release can't precede the copy
            if read_future is not None:
                wait([read_future])
            moved = self._cpu_pool.offload_stream(
                rid=rid, label=label,
                gpu_kv_cache=self.kv_cache.tensor,
                gpu_page_indices=pages,
                stored_len=geom[0], position=geom[1], released=geom[2],
            )
            if not moved:
                continue
            self._cpu_pool.sync()
            with self._lock:
                stream = self._streams[rid].get(label)
                if stream is None:
                    continue
                freed += len(pages)
                self._arena.release(pages)
                stream.page_indices = []
                stream.offloaded = True
                stream.reset()
        return freed

    def reload(self, rid: str) -> bool:
        """Bring every offloaded stream of ``rid`` back on device.

        False when the device can't fit them right now; nothing moves in that
        case, so the caller can evict further and try again.
        """
        if self._cpu_pool is None or not self._cpu_pool.is_offloaded(rid):
            return False
        with self._lock:
            labels = self._cpu_pool.labels(rid)
            needed = sum(self._cpu_pool.num_pages(rid, label) for label in labels)
            if needed > self._arena.num_free:
                return False
            for label in labels:
                stream = self._ensure_label(rid, label)
                pages = self._arena.acquire(
                    self._cpu_pool.num_pages(rid, label)
                )
                if pages is None:
                    # lost the race for pages against another consumer
                    return False
                state = self._cpu_pool.reload_stream(
                    rid=rid, label=label,
                    gpu_kv_cache=self.kv_cache.tensor,
                    gpu_page_indices=pages,
                )
                stream.page_indices = pages
                stream.stored_len = state.stored_len
                stream.position = state.position
                stream.released = state.released
                stream.offloaded = False
        # sync outside the lock: orders the reload H2D copies before attention
        # reads them, but the pages are already assigned so it touches no
        # shared state
        self._cpu_pool.sync()
        return True

    def get_offload_priority(self, rid: str) -> float:
        """Device pages the request is holding — the most reclaimable first."""
        if rid not in self._streams:
            return 0.0
        return float(sum(
            len(stream.page_indices) for stream in self._streams[rid].values()
        ))

    def publish(self, request_id: str):
        res = PublishedKVInfo.build_for_rank(
            rank=self._rank, world_size=self._world_size,
            seq_info={
                label: KVSequenceInfo(
                    seq_len=stream.stored_len,
                    latest_kv_transfer_info=self._transfer.get_kv_transfer_info(),
                    page_indices=stream.page_indices
                ) for label, stream in self._streams[request_id].items()
            }
        )
        return res

    def reset_request(self, rid: str, free: bool=False):
        streams = self._streams.get(rid)
        if streams is None:
            return
        # drain in-flight reads OUTSIDE the lock (their pages must not be reused
        # until they finish writing); the transfer thread doesn't touch _streams
        for stream in streams.values():
            if stream.read_future is not None:
                wait([stream.read_future])
        with self._lock:
            for stream in self._streams.get(rid, {}).values():
                if free:
                    self._arena.release(stream.page_indices)
                stream.reset(freed=free)

    def remove_request(self, rid: str):
        streams = self._streams.get(rid)
        if streams is not None:
            # drain in-flight reads outside the lock; see reset_request
            for stream in streams.values():
                if stream.read_future is not None:
                    wait([stream.read_future])
        with self._lock:
            if rid in self._streams:
                for stream in self._streams[rid].values():
                    self._arena.release(stream.page_indices)
            if self._cpu_pool is not None:
                self._cpu_pool.remove_request(rid)
            self._streams.pop(rid, None)
            self._overrides.pop(rid, None)

    def post_warmup_validate(self):
        """Assert ``num_free_pages`` is identical across every TP rank

        Catches YAML drift (e.g. ``cpu_offload_pages`` set on one rank
        but not another), allocator-init bugs, and any future code path
        that adds requests asymmetrically before ``warmup`` returns. The
        ``all_gather`` itself is synchronizing, so no extra barrier is
        needed on the success path.
        """
        if self._comm_group.world_size == 1:
            return
        local_free = self._arena.num_free
        local_t = torch.tensor(
            [local_free], dtype=torch.int64, device=self._device,
        )

        for group in [
            self._comm_group.tp_group, self._comm_group.sp_group
        ]:
            gathered = group.all_gather(local_t, dim=0)
            values = gathered.cpu().tolist()
            if any(v != values[0] for v in values):
                raise RuntimeError(
                    f"KV cache {self.name!r} has asymmetric num_free_pages "
                    f"across TP ranks: {values}. v1 requires symmetric "
                    "allocator state; check the YAML for per-rank-divergent "
                    "max_num_pages / cpu_offload_pages, and any model code "
                    "that calls add_request before warmup completes."
                )

    def cleanup(self):
        self._transfer.cleanup()

    def _ensure_label(self, rid: str, label: str) -> CacheStream:
            if label not in self._streams[rid]:
                self._streams[rid][label] = CacheStream()
            return self._streams[rid][label]

    def _check_ready(self, rid: str, label: str):
        if label not in self._streams[rid]:
            return True
        stream = self._streams[rid][label]
        if stream.read_future is not None and stream.read_future.done():
            stream.read_future.result()
            stream.read_future = None
            stream.read_pending = False
        return not stream.read_pending

    @staticmethod
    def _label_growth(step: KVStep) -> dict[tuple[str, str], int]:
        """(rid, label) -> what this step's commit adds to that stream."""
        growth: dict[tuple[str, str], int] = {}
        for segment in step.segments:
            key = (segment.request_id, segment.label)
            growth[key] = growth.get(key, 0) + segment.span
        return growth

    def _reserve_fork(
        self, rid: str, from_label: str,
        to_label: str, extra: int = 0, realloc: bool = False
    ) -> AllocResult:
        """Pages for a fork target, without moving anything into them.

        ``extra`` is how much the source still grows before the copy runs.
        """
        # TODO: handle realloc
        if from_label not in self._streams[rid]:
            return AllocResult()
        self._ensure_label(rid, to_label)
        return self._alloc(
            rid, to_label, self._streams[rid][from_label].stored_len + extra
        )

    def _apply_fork(self, rid: str, from_label: str, to_label: str) -> None:
        """Copy a stream onto its fork target, over pages `_reserve_fork` took.

        Locked (reentrant): called from plan (pre-forks, else unguarded) and
        from the already-locked commit (post-forks)."""
        with self._lock:
            if from_label not in self._streams[rid]:
                return
            from_stream = self._streams[rid][from_label]
            to_stream = self._ensure_label(rid, to_label)
            # TODO: only copy necessary pages
            # the source can hold more pages than the fork needs (pages
            # reserved for an uncommitted span), so copy only the prefix
            self._arena.copy_pages(
                from_stream.page_indices[:len(to_stream.page_indices)],
                to_stream.page_indices
            )
            to_stream.stored_len = from_stream.stored_len
            to_stream.generation += 1

    def _alloc(
        self, request_id: str, label: str, seq_len: int
    ) -> AllocResult:
        with self._lock:
            self._ensure_label(request_id, label)
            stream = self._streams[request_id][label]
            num_pages_needed = (seq_len + self.config.page_size - 1) // self.config.page_size
            num_new_pages = num_pages_needed - len(stream.page_indices)
            if num_new_pages > 0:
                new_pages = self._arena.acquire(num_new_pages)
                if new_pages is None:
                    return AllocResult(
                        success=False,
                        error=AllocationFailed(
                            pages_short=num_new_pages - self._arena.num_free,
                            request_id=request_id,
                            label=label,
                            message=(
                                f"Not enough free pages: requested {num_new_pages}, "
                                f"available {self._arena.num_free}"
                            ),
                        )
                    )
                stream.page_indices.extend(new_pages)
                stream.generation += 1
        return AllocResult()

    ### Submodule-level functionality
    def set_layer_idx(self, layer_idx: int):
        self._current_layer_idx = layer_idx

    @torch.compiler.disable
    def layer_view(self, layer_idx: int=None) -> torch.Tensor:
        """layer pages as needed by attention kernel

        handed to `AttentionManager::run`. in `kv_manager` so storage mechanics
        are opaque to layers"""
        if layer_idx is None:
            layer_idx = self._current_layer_idx
        return self.kv_cache.layer_view(layer_idx)

    @torch.compiler.disable
    def read_kv(self, layer_idx: int=None, plan_label: str="main") -> torch.Tensor:
        """
        The slots this step's plan writes, e.g. for NHD:
        [num_tokens, 2, num_kv_heads, head_dim] (K at index 0, V at 1).
        """
        if layer_idx is None:
            layer_idx = self._current_layer_idx
        plan_state = self._current_plan_states[plan_label]
        n = plan_state.total_tokens
        return self.kv_cache.read_tokens(
            layer_idx=layer_idx,
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
        )

    @torch.compiler.disable
    def write_kv(
        self, k: torch.Tensor, v: torch.Tensor,
        layer_idx: int=None, label: str="main", return_tensor: bool = False,
    ) -> torch.Tensor | None:
        """Write K, V into this step's planned slots.

        Returns nothing by default: reading the slots back is a gather no
        caller wants today, and skipping it keeps the write a pure mutation.
        """
        if layer_idx is None:
            layer_idx = self._current_layer_idx
        plan_state = self._current_plan_states[label]
        n = plan_state.total_tokens
        return self.kv_cache.write_tokens(
            layer_idx=layer_idx,
            k=k[:n], v=v[:n],
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
            return_tensor=return_tensor,
        )
