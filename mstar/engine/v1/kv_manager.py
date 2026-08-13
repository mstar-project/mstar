from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
from typing import Any

import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.base import PublishedInfo, Resource
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
    length: int
    start: int = 0

    def last_page_len(self, page_size: int) -> int:
        return (self.start + self.length)  % page_size


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
        self._lock = threading.RLock()

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

    def ingest_request(self, rid, overrides: KVReqConfig | None=None):
        self._streams[rid] = {
            "main": CacheStream()
        }
        if overrides is not None:
            overrides = KVReqConfig()
        self._overrides[rid] = overrides

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: PublishedKVInfo
    ) -> AdmitOutcome:
        del graph_walk
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
                return False # read already in progress

            stream = self._ensure_label(rid, label)
            new_len = seq_info.seq_len
            old_len = stream.stored_len
            if new_len <= old_len:
                continue

            alloc_res = self._alloc(rid, label, new_len - old_len)
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

        ready = all((self._ensure_label(rid, label) for label in needed_labels))
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
            stream = self._ensure_label(rid, segment.label)
            if segment.span == 0:
                continue
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
                length=s.span + stream.stored_len
            ))
        return views

    def plan(self, step: KVStep, ctx: StepContext) -> dict[str, list[SequenceView]]:
        """
        Returns list of sequence views per plan label
        """
        label_to_segments: dict[str, list[Segment]] = {}
        combined_labels = sum(step.combined_labels.keys(), start=set())
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
            res[plan_label] = views

        for label in standalone_labels:
            res[label] = self._sequence_views(label_to_segments[label])
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
            stream.read_future = None
            stream.read_pending = False
        return not stream.read_pending

    def _fork(
        self, rid: str, from_label: str,
        to_label: str, realloc: bool = False
    ) -> AllocResult:
        # TODO: handle realloc
        if from_label not in self._streams[rid]:
            return
        from_stream = self._streams[rid][from_label]
        to_stream = self._ensure_label(rid, to_label)
        alloc_res = self._alloc(rid, to_label, from_stream.stored_len)
        if alloc_res.success:
            # TODO: only copy necessary pages
            self._arena.copy_pages(from_stream.page_indices, to_stream.page_indices)
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
