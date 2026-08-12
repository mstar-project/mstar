from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading

import torch

from mstar.engine.kv_store import TransferEngineInfo
from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceType
from mstar.engine.v1.kv_cache import KVCache, KVConfig, KVSpec
from mstar.engine.v1.kv_transfer import KVTransferManager


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

class KVManager(Resource):
    def __init__(
        self,
        cfg: KVConfig,
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
        self._streams: dict[str, LabelToStream] = {}


    @classmethod
    def build(
        cls, spec: KVSpec,
        device: torch.device,
        transfer_engine_info: TransferEngineInfo,
        dtype=torch.bfloat16,
        **kwargs
    ):
        return cls(
            cfg=spec.config,
            device=device,
            transfer_engine_info=transfer_engine_info,
            dtype=dtype
        )

    def ingest_request(self, rid, overrides=None):
        del overrides # claude always does this for type checking...
        self._streams[rid] = {
            "main": CacheStream()
        }

    def admit_retrieve(self, rid: str, per_label_seq_info) -> bool:
        # TODO: retrieve, return whether ready
        # alloc -> call self._transfer -> set stream -> check if ready
        ...
    
    def remove_request(self, rid: str):
        # TODO: return all pages
        self.request_states.pop(rid, None)

