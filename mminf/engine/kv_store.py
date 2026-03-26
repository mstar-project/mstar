from dataclasses import dataclass, field
import queue
from mooncake.store import MooncakeDistributedStore

import torch

from mminf.communication.communicator import CommProtocol


class PageAllocator:
    """Simple page allocator using a FIFO queue of free page indices."""

    def __init__(self, max_num_pages: int):
        self.max_num_pages = max_num_pages
        self.free_pages: queue.Queue[int] = queue.Queue()
        for i in range(max_num_pages):
            self.free_pages.put(i)

    def allocate(self, n: int) -> list[int]:
        if self.free_pages.qsize() < n:
            raise RuntimeError(
                f"Not enough free pages: requested {n}, "
                f"available {self.free_pages.qsize()}"
            )
        return [self.free_pages.get() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        for page in pages:
            self.free_pages.put(page)

    @property
    def num_free(self) -> int:
        return self.free_pages.qsize()


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    num_qo_heads: int | None = None  # Optional, defaults to num_kv_heads

    def __post_init__(self):
        if self.num_qo_heads is None:
            self.num_qo_heads = self.num_kv_heads


@dataclass
class KVRequestState:
    """Per-request KV cache state for the AR engine."""
    page_indices: list[int] = field(default_factory=list)
    seq_len: int = 0
    position_id_start: int = 0

    # this must be properly set by the BatchedCacheManager
    local_cache_seq_len: int = 0
    # sequence length of the in-distributed-store KV cache
    store_seq_len_per_layer: list | None = None
    is_paused: bool = False


LabelToState = dict[str, KVRequestState]

@dataclass
class StoreAllocInfo:
    key: str
    ptr: int
    nbytes: int


@dataclass
class MooncakeStoreConfig:
    hostname: str
    metadata_server: str="http://localhost:8080/metadata"
    segment_size=512*1024*1024
    local_buff_size= 28*1024*1024
    protocol: CommProtocol=CommProtocol.RDMA
    master_service: str="localhost:50051"
    

class PagedAllocationManager:
    def __init__(
        self,
        config: KVCacheConfig,
        kv_cache: torch.Tensor,
        mooncake_cfg: MooncakeStoreConfig
    ):
        self.config = config
        self.page_allocator = PageAllocator(config.max_num_pages)
        self.request_states: dict[str, LabelToState] = {}
        self.kv_cache = kv_cache
    
        self.mooncake_store = MooncakeDistributedStore()

        self.mooncake_store.setup(
            mooncake_cfg.hostname,
            mooncake_cfg.metadata_server,
            mooncake_cfg.segment_size,
            mooncake_cfg.local_buff_size,
            "tcp", # this isn't working for RDMA??
            "",
            mooncake_cfg.master_service
        )
        self.mooncake_store.register_buffer(
            self.kv_cache.data_ptr(), self.kv_cache.nbytes
        )
    
    def _key(self, request_id: str, label: str, pos: int, layer: int):
        return f"{request_id}_{label}_{pos}_{layer}"

    def flush_to_store(
        self, request_id: str, label: str, layers: int | list[int] | None=None
    ):
        if isinstance(layers, int):
            layers = [layers]
        if layers is None:
            layers = torch.arange(self.config.num_layers, dtype=torch.int32)
        state = self.request_states[request_id][label]            
        
        alloc_info: list[StoreAllocInfo] = []

        last_pos = (state.local_cache_seq_len - 1) // self.config.page_size
        for layer in layers:
            if state.store_seq_len_per_layer[layer] >= state.local_cache_seq_len:
                continue
            first_pos = state.store_seq_len_per_layer[layer] // self.config.page_size
            state.store_seq_len_per_layer[layer] = state.local_cache_seq_len

            for pos in range(first_pos, last_pos+1):
                page_idx = state.page_indices[pos]

                # TODO: the fact that we're moving full pages to and from the cache
                # is inefficient. We can instead move dynamic amounts of memory that
                # are <= a page at once, and also store metadata specifying how many
                # sequence indices are being stored at once
                alloc_info.append(StoreAllocInfo(
                    key=self._key(request_id, label, pos, layer),
                    ptr=self.kv_cache[layer, page_idx].data_ptr(),
                    nbytes=self.kv_cache[layer, page_idx].nbytes
                ))
        
        status = self.mooncake_store.batch_put_tensor_from(
            keys=[x.key for x in alloc_info],
            buffer_ptrs=[x.ptr for x in alloc_info],
            sizes=[x.nbytes for x in alloc_info]
        )
        # TODO error handling
        assert all([s == 0 for s in status])
    
    def _new_state(self):
        state = KVRequestState()
        state.store_seq_len_per_layer = torch.zeros(
            self.config.num_layers, dtype=torch.int32
        )
        return state

    def get_state(self, request_id: str, label: str):
        if label not in self.request_states[request_id]:
            
            self.request_states[request_id][label] = self._new_state()
        return self.request_states[request_id][label]

    def alloc(
        self, request_id: str, label: str, seq_len: int
    ):
        state = self.request_states[request_id][label]
        num_pages_needed = (seq_len + self.config.page_size - 1) // self.config.page_size
        num_new_pages = num_pages_needed - len(state.page_indices)
        if num_new_pages > 0:
            new_pages = self.page_allocator.allocate(num_new_pages)
            state.page_indices.extend(new_pages)

    def retrieve_from_store(
        self, request_id: str, label: str, seq_len: int
    ):
        state = self.request_states[request_id][label]
        if state.seq_len >= seq_len:
            return # nothing to do
        state.seq_len =seq_len
        state.local_cache_seq_len = seq_len
        
        first_pos = state.seq_len // self.config.page_size
        last_pos = (seq_len - 1) // self.config.page_size

        self.alloc(request_id, label, seq_len)

        alloc_info: list[StoreAllocInfo] = []
        for pos in range(first_pos, last_pos+1):
            page_idx = state.page_indices[pos]
            for layer in range(self.config.num_layers):
                alloc_info.append(StoreAllocInfo(
                    key=self._key(request_id, label, pos, layer),
                    ptr=self.kv_cache[layer, page_idx].data_ptr(),
                    nbytes=self.kv_cache[layer, page_idx].nbytes
                ))
        
        tensors = self.mooncake_store.batch_get_tensor_into(
            keys=[x.key for x in alloc_info],
            buffer_ptrs=[x.ptr for x in alloc_info],
            sizes=[x.nbytes for x in alloc_info]
        )
        # TODO error handling
        assert all([t is not None for t in tensors])
    
    def reset_label(self, request_id: str, label: str, free: bool=True):
        if label not in self.request_states[request_id]:
            self.request_states[request_id][label] = self._new_state()
            return
        if free:
            state = self.request_states[request_id][label]
            self.page_allocator.free(state.page_indices)
        self.request_states[request_id][label] = self._new_state()

    def cleanup(self):
        self.mooncake_store.unregister_buffer(
            self.kv_cache.data_ptr()
        )
    
    def add_request(self, request_id: str, labels: list[str]=[]):
        self.request_states[request_id] = {
            label: self._new_state() for label in labels
        }
    
    def remove_request(self, request_id: str):
        for label in self.request_states[request_id]:
            state = self.request_states[request_id][label]
            self.page_allocator.free(state.page_indices)
        del self.request_states[request_id]

        count = self.mooncake_store.remove_by_regex(f"^{request_id}_.*")
        # TODO error handling
        assert count >= 0