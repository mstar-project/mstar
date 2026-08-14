from concurrent.futures import Future, wait
from dataclasses import dataclass, field
import itertools
import queue
import threading
from typing import Any, NamedTuple

import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.base import CGSlotSpec, PublishedInfo, Resource
from mstar.engine.resources.spec import KVReqConfig
from mstar.engine.resources.step import AdmitOutcome, AllocationFailed, KVStep, Segment, StepContext
from mstar.engine.v1.kv_cache import KVCache, KVConfig, KVSpec
from mstar.engine.v1.kv_transfer import KVTransferManager, TransferEngineInfo


class PageAllocator:
    """Simple page allocator using a FIFO queue of free page indices.

    Thread-safe: a ``threading.Lock`` makes the qsize-then-get sequence in
    ``allocate``/``try_allocate`` atomic against concurrent ``free`` calls.
    Required by the pre-plan path, where the plan thread runs
    ``try_allocate`` while the GPU thread runs ``free`` from
    ``reset_label`` — the unlocked qsize/get pair could false-negative
    (return None when pages are about to be freed) or partially fill the
    output list under multi-consumer contention.
    """

    def __init__(self, max_num_pages: int):
        self.max_num_pages = max_num_pages
        self.free_pages: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()
        for i in range(max_num_pages):
            self.free_pages.put(i)

    def allocate(self, n: int) -> list[int]:
        with self._lock:
            if self.free_pages.qsize() < n:
                raise RuntimeError(
                    f"Not enough free pages: requested {n}, "
                    f"available {self.free_pages.qsize()}"
                )
            return [self.free_pages.get() for _ in range(n)]

    def try_allocate(self, n: int) -> list[int] | None:
        """Like allocate() but returns None instead of raising on failure."""
        with self._lock:
            if self.free_pages.qsize() < n:
                return None
            return [self.free_pages.get() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        with self._lock:
            for page in pages:
                self.free_pages.put(page)

    @property
    def num_free(self) -> int:
        return self.free_pages.qsize()


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


@dataclass
class SequenceView:
    label: str
    page_idxs: list[int]
    length: int # already resident + to_compute
    to_compute: int # new for this step
    start: int = 0

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
    cuda_indptrs: PagedIndptrs

    def get_total_len(self):
        lens = (self.cpu_indptrs.qo_indptr[1:] - self.cpu_indptrs.qo_indptr[:-1]).to(torch.int32)
        return int(lens.sum().item())


@dataclass
class KVPlanState:
    token_to_page: torch.Tensor
    token_to_cache: torch.Tensor
    total_tokens: int | None = None

    def copy_(self, other: "KVPlanState"):
        assert other.total_tokens is not None
        self.token_to_cache[:other.total_tokens].copy_(other.token_to_cache)
        self.token_to_page[:other.total_tokens].copy_(other.token_to_page)
        self.total_tokens = other.total_tokens


class KVManager(Resource):
    def __init__(
        self,
        cfg: KVConfig,
        comm_group: CommGroup | None,
        transfer_engine_info: TransferEngineInfo,
        device: torch.device,
        dtype=torch.bfloat16,
    ):
        self.config = cfg
        self.kv_cache = KVCache(
            cfg, device, dtype
        )
        
        self._arena = PageArena(
            kv_cache=self.kv_cache,
            allocator=PageAllocator(cfg.max_num_pages)
        )
        self._transfer = KVTransferManager(
            transfer_engine_info, self.kv_cache
        )
        # TODO: CPU page pool
        self._streams: dict[str, LabelToStream] = {}
        self._overrides: dict[str, KVReqConfig] = {}
        self._rank = comm_group.rank if comm_group is not None else 0
        self._world_size = comm_group.world_size if comm_group is not None else 1
        self._device = device
        self._lock = threading.RLock()

        # (slot, label) -> KVPlanState
        self._static_plan_states: dict[tuple[int, str], KVPlanState] = {}
        self._current_plan_states: dict[str, KVPlanState] = {}

    @classmethod
    def build(
        cls, spec: KVSpec,
        device: torch.device,
        comm_group: CommGroup | None,
        transfer_engine_info: TransferEngineInfo,
        dtype=torch.bfloat16,
        **kwargs
    ):
        return cls(
            cfg=spec.config,
            device=device,
            comm_group=comm_group,
            transfer_engine_info=transfer_engine_info,
            dtype=dtype
        )

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ):
        del max_bs
        slot_numbers = {
            s.slot for s in slots
        }
        labels = set(itertools.chain.from_iterable(
            s.config.labels for s in slots
        ))
        for idx in slot_numbers:
            for label in labels:
                self._static_plan_states[(idx, label)] = KVPlanState(
                    token_to_cache=torch.zeros(max_seq_len, dtype=torch.long, device=self._device),
                    token_to_page=torch.zeros(max_seq_len, dtype=torch.long, device=self._device),
                )

    def ingest_request(self, rid, overrides: KVReqConfig | None=None):
        self._streams[rid] = {
            "main": CacheStream()
        }
        if overrides is None:
            overrides = KVReqConfig()
        self._overrides[rid] = overrides

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: PublishedKVInfo
    ) -> AdmitOutcome:
        # TODO: reload from CPU

        if published.world_size != self._world_size:
            raise RuntimeError(
                "KV cache transfer across TP world size is currently disallowed"
            )
        needed_labels = self._overrides[rid].get_labels(node_name, graph_walk)
        for label, seq_info in published.get(self._rank).items():
            if label not in needed_labels:
                continue
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
                return AdmitOutcome(
                    ok=False,
                    reason=alloc_res.error
                )
            
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
        for (from_label, to_label) in step.pre_forks:
            for rid in ctx.request_ids:
                alloc_res = self._fork(rid, from_label, to_label)
                if not alloc_res.success:
                    return AdmitOutcome(
                        ok=False,
                        reason=alloc_res.error
                    )

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
                return AdmitOutcome(
                    ok=False,
                    reason=alloc_res.error
                )
        # TODO: apply retention policy

        return AdmitOutcome(ok=True)

    def _sequence_views(self, segments: list[Segment]) -> list[SequenceView]:
        views = []
        for s in segments:
            stream = self._streams[s.request_id][s.label]
            views.append(SequenceView(
                label=s.label, page_idxs=stream.page_indices,
                length=s.span + stream.stored_len,
                to_compute=s.span
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

    def _setup_plan_states(
        self, plan_output: dict[str, KVPlanOutput],
        ctx: StepContext
    ):
        for label, indptrs in plan_output.items():
            plan_state = self._compute_plan_state(
                indptrs.cuda_indptrs,
                total_tokens=indptrs.get_total_len()
            )
            if ctx.slot_lease is not None:
                static_state = self._static_plan_states[(ctx.slot_lease.slot, label)]
                static_state.copy_(plan_state)
                plan_state = static_state
            self._current_plan_states[label] = plan_state


    def _plan_output(self, views: list[SequenceView]) -> KVPlanOutput:
        indptrs = build_paged_indptrs(views, self.kv_cache.page_size)
        return KVPlanOutput(
            cpu_indptrs=indptrs,
            cuda_indptrs=indptrs.to_device(self._device)
        )

    def plan(self, step: KVStep, ctx: StepContext) -> dict[str, KVPlanOutput]:
        """
        Returns list of sequence views per plan label
        """
        label_to_segments: dict[str, list[Segment]] = {}
        combined_labels = set(itertools.chain.from_iterable(
            step.combined_labels.keys()
        ))
        standalone_labels = set()

        res = {}
        for segment in step.segments:
            label_to_segments.setdefault(segment.label, []).append(segment)
            if segment.label not in combined_labels:
                standalone_labels.add(segment.label)

        for source_labels, plan_label in step.combined_labels.items():
            views = []
            for label in source_labels:
                views.extend(self._sequence_views(label_to_segments[label]))
            res[plan_label] = self._plan_output(views)

        for label in standalone_labels:
            res[label] = self._plan_output(
                self._sequence_views(label_to_segments[label])
            )

        self._setup_plan_states(res, ctx)
        return res

    def commit(self, step: KVStep, ctx: StepContext):
        for segment in step.segments:
            if step.commit and segment.span > 0:
                stream = self._streams[segment.request_id][segment.label]
                stream.stored_len += segment.span
        # TODO: handle retention policy, free pages if not commit
    
    def publish(self, request_id: str):
        return PublishedKVInfo.build_for_rank(
            rank=self._rank, world_size=self._world_size,
            seq_info={
                label: KVSequenceInfo(
                    seq_len=stream.stored_len,
                    latest_kv_transfer_info=self._transfer.get_kv_transfer_info(),
                    page_indices=stream.page_indices
                ) for label, stream in self._streams[request_id].items()
            }
        )

    def remove_request(self, rid: str):
        if rid in self._streams:
            for stream in self._streams[rid].values():
                if stream.read_future is not None:
                    # the read's result is irrelevant here, but its pages
                    # must not be reused until it finishes writing
                    wait([stream.read_future])
                self._arena.release(stream.page_indices)

        self._streams.pop(rid, None)
        self._overrides.pop(rid, None)

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

    def _fork(
        self, rid: str, from_label: str,
        to_label: str, realloc: bool = False
    ) -> AllocResult:
        # TODO: handle realloc
        if from_label not in self._streams[rid]:
            return AllocResult()
        from_stream = self._streams[rid][from_label]
        to_stream = self._ensure_label(rid, to_label)
        alloc_res = self._alloc(rid, to_label, from_stream.stored_len)
        if alloc_res.success:
            # TODO: only copy necessary pages
            # the source can hold more pages than the fork needs (pages
            # reserved for an uncommitted span), so copy only the prefix
            self._arena.copy_pages(
                from_stream.page_indices[:len(to_stream.page_indices)],
                to_stream.page_indices
            )
            to_stream.stored_len = from_stream.stored_len
        return alloc_res

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
        return AllocResult()

    ### Submodule-level functionality
    def read_kv(self, layer_idx: int, plan_label: str) -> torch.Tensor:
        """
        The slots this step's plan writes, e.g. for NHD:
        [num_tokens, 2, num_kv_heads, head_dim] (K at index 0, V at 1).
        """
        plan_state = self._current_plan_states[plan_label]
        n = plan_state.total_tokens
        return self.kv_cache.read_tokens(
            layer_idx=layer_idx,
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
        )

    def write_kv(
        self, k: torch.Tensor, v: torch.Tensor,
        layer_idx: int, label: str
    ) -> torch.Tensor:
        """Write K, V. Returns the slice of the KV cache that was just written.
        """
        plan_state = self._current_plan_states[label]
        n = plan_state.total_tokens
        return self.kv_cache.write_tokens(
            layer_idx=layer_idx,
            k=k[:n], v=v[:n],
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
            return_tensor=True
        )