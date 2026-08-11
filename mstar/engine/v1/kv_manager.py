from dataclasses import dataclass, field
from enum import Enum
import queue
import threading

import torch

from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import KVSpec, NodeResourceSpec, ResourceType


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
class KVRequestState:
    """Per-request KV cache state for the AR engine."""
    page_indices: list[int] = field(default_factory=list)
    seq_len: int = 0 # includes read in progress
    read_in_progress: bool = False

    # sequence length of the in-distributed-store KV cache
    is_paused: bool = False


class KVLayout(Enum):
    NHD = "NHD"
    # TODO: can add more, like HND, MLA

@dataclass
class KVConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    layout: KVLayout=KVLayout.NHD

@dataclass
class KVSpec(NodeResourceSpec):
    config: KVConfig

    @property
    def resource_type(self):
        return ResourceType.KV_CACHE


LabelToState = dict[str, KVRequestState]

class KVManager(Resource):
    def __init__(
        self,
        cfg: KVConfig,
        device: torch.device,
        dtype=torch.bfloat16
    ):
        self.config = cfg
        if cfg.layout == KVLayout.NHD:
            self.kv_cache = torch.zeros(
                cfg.num_layers, cfg.max_num_pages, 2,
                cfg.page_size, cfg.num_kv_heads, cfg.head_dim,
                dtype=dtype, device=device,
            ).contiguous()
        else:
            raise NotImplementedError(
                f"KV layout {cfg.layout} is not recognized."
            )
        self.allocator = PageAllocator(cfg.max_num_pages)
        self.request_states: dict[str, LabelToState] = {}


    @classmethod
    def build(
        cls, spec: KVSpec,
        device: torch.device,
        dtype=torch.bfloat16,
        **kwargs
    ):
        return cls(
            cfg=spec.config,
            device=device,
            dtype=dtype
        )

    def ingest_request(self, rid, overrides=None):
        del overrides # claude always does this for type checking...
        self.request_states[rid] = {
            "main": KVRequestState()
        }

    def admit_retrieve(self, rid: str, per_label_seq_info):
        # TODO: retrieve, return whether ready
        ...
    
    def remove_request(self, rid: str):
        # TODO: return all pages
        self.request_states.pop(rid, None)

