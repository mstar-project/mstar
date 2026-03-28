from dataclasses import dataclass, field
from enum import Enum
import queue
from mooncake.store import MooncakeDistributedStore
from mooncake.engine import TransferEngine

import torch

from mminf.communication.communicator import CommProtocol


# NOTE: when this is changed to be different from the page size (or perhaps when
# it is made too small), writing large amounts of data to the mooncake store is 
# sometimes very slow.
KV_STORE_CHUNK_SIZE = 128

MAX_TRANSFERS = 500


@dataclass
class SequenceInfo:
    seq_len: int
    pos_id: int

    # for tracking KV cache
    latest_entity_id: str = ""
    latest_session_id: str = ""
    kv_cache_addr: int = -1
    last_page_idx: int = -1


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
    ptr: list[int]
    nbytes: list[int]


@dataclass
class MooncakeStoreConfig:
    hostname: str
    metadata_server: str="http://localhost:8080/metadata"
    segment_size=4 * 1024*1024*1024
    local_buff_size=4 * 1024*1024*1024
    protocol: CommProtocol=CommProtocol.RDMA
    master_service: str="localhost:50051"


@dataclass
class TransferEngineInfo:
    my_entity_id: str
    my_session_id: str
    transfer_engine: TransferEngine 


class TransferType(Enum):
    GET = "get"
    PUT = "put"


class PagedAllocationManager:
    def __init__(
        self,
        config: KVCacheConfig,
        kv_cache: torch.Tensor,
        mooncake_cfg: MooncakeStoreConfig,
        transfer_engine_info: TransferEngineInfo
    ):
        self.config = config
        self.page_allocator = PageAllocator(config.max_num_pages)
        self.request_states: dict[str, LabelToState] = {}
        self.kv_cache = kv_cache
    
        self.mooncake_store = MooncakeDistributedStore()
        self.engine = transfer_engine_info.transfer_engine

        self.mooncake_store.setup(
            mooncake_cfg.hostname,
            mooncake_cfg.metadata_server,
            mooncake_cfg.segment_size,
            mooncake_cfg.local_buff_size,
            mooncake_cfg.protocol.value.lower(),
            "",
            mooncake_cfg.master_service
        )
        self.mooncake_store.register_buffer(
            self.kv_cache.data_ptr(), self.kv_cache.nbytes
        )
        self.my_entity_id = transfer_engine_info.my_entity_id
        self.my_session_id = transfer_engine_info.my_session_id

    def _key(self, request_id: str, label: str, pos: int, layer: int):
        return f"{request_id}_{label}_{pos}_{layer}"

    def _get_ptr_nbytes(
        self, layer, page_idx, token_start, token_end,
        base_ptr=None
    ):
        token_stride = self.kv_cache.stride(3)
        kv_stride = self.kv_cache.stride(2)
        page_stride = self.kv_cache.stride(1)
        layer_stride = self.kv_cache.stride(0)
        element_size = self.kv_cache.element_size()
        tokens_per_chunk = token_end - token_start

        nbytes = tokens_per_chunk * token_stride * element_size  # token_stride = num_kv_heads * head_dim

        if base_ptr is None:
            base_ptr = self.kv_cache.data_ptr()

        ptrs = [
            base_ptr + (
                    layer * layer_stride +
                    page_idx * page_stride +
                    kv_idx * kv_stride +
                    token_start * token_stride
                ) * element_size for kv_idx in [0, 1]
        ]

        return ptrs, [nbytes, nbytes]
    
    def _transfer_all(
        self, alloc_info: list[StoreAllocInfo],
        direction: TransferType
    ):
        if not alloc_info:
            return
        
        if direction == TransferType.GET:
            transfer_fn = self.mooncake_store.batch_get_into_multi_buffers
        else:
            transfer_fn = self.mooncake_store.batch_put_from_multi_buffers

        torch.cuda.default_stream().synchronize()
        for start in range(0, len(alloc_info), MAX_TRANSFERS):
            end = min(start + MAX_TRANSFERS, len(alloc_info))
            status = transfer_fn(
                keys=[alloc_info[i].key for i in range(start, end)],
                all_buffer_ptrs=[alloc_info[i].ptr for i in range(start, end)],
                all_sizes=[alloc_info[i].nbytes for i in range(start, end)]
            )
            assert all(s >= 0 for s in status)
        torch.cuda.default_stream().synchronize()

    def flush_to_store(
        self, request_id: str, label: str, layers: int | list[int] | None = None
    ):
        assert self.config.page_size % KV_STORE_CHUNK_SIZE == 0, (
            f"page_size ({self.config.page_size}) must be a multiple of "
            f"KV_STORE_CHUNK_SIZE ({KV_STORE_CHUNK_SIZE})"
        )
        tokens_per_chunk = KV_STORE_CHUNK_SIZE
        chunks_per_page = self.config.page_size // tokens_per_chunk

        if isinstance(layers, int):
            layers = [layers]
        if layers is None:
            layers = list(range(self.config.num_layers))

        state = self.request_states[request_id][label]

        # Only flush complete chunks — drop any trailing partial chunk
        num_complete_chunks = state.local_cache_seq_len // tokens_per_chunk
        flush_up_to = num_complete_chunks * tokens_per_chunk  # token boundary

        alloc_info: list[StoreAllocInfo] = []

        for layer in layers:
            if state.store_seq_len_per_layer[layer] >= flush_up_to:
                continue

            first_chunk = state.store_seq_len_per_layer[layer] // tokens_per_chunk
            last_chunk = num_complete_chunks  # exclusive

            state.store_seq_len_per_layer[layer] = flush_up_to

            for chunk_idx in range(first_chunk, last_chunk):
                page_pos = chunk_idx // chunks_per_page
                chunk_within_page = chunk_idx % chunks_per_page

                page_idx = state.page_indices[page_pos]

                # Slice the chunk out of the page tensor
                token_start = chunk_within_page * tokens_per_chunk
                token_end = token_start + tokens_per_chunk

                ptr, nbytes = self._get_ptr_nbytes(
                    layer, page_idx, token_start, token_end
                )
                alloc_info.append(StoreAllocInfo(
                    key=self._key(request_id, label, chunk_idx, layer),
                    ptr=ptr,
                    nbytes=nbytes
                ))
        
        self._transfer_all(alloc_info, TransferType.PUT)
    
    def _new_state(self):
        state = KVRequestState()
        state.store_seq_len_per_layer = [0] * self.config.num_layers
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
        self, request_id: str, label: str, seq_info: SequenceInfo
    ):
        assert self.config.page_size % KV_STORE_CHUNK_SIZE == 0, (
            f"page_size ({self.config.page_size}) must be a multiple of "
            f"KV_STORE_CHUNK_SIZE ({KV_STORE_CHUNK_SIZE})"
        )
        tokens_per_chunk = KV_STORE_CHUNK_SIZE
        chunks_per_page = self.config.page_size // tokens_per_chunk

        seq_len = seq_info.seq_len
        state = self.get_state(request_id, label)
        if state.seq_len >= seq_len:
            return  # nothing to do

        num_complete_chunks = seq_len // tokens_per_chunk
        trailing_tokens = seq_len % tokens_per_chunk

        first_chunk = state.seq_len // tokens_per_chunk

        self.alloc(request_id, label, seq_len)

        # --- Path 1: mooncake store for complete chunks ---
        alloc_info: list[StoreAllocInfo] = []
        for chunk_idx in range(first_chunk, num_complete_chunks):
            page_pos = chunk_idx // chunks_per_page
            chunk_within_page = chunk_idx % chunks_per_page
            token_start = chunk_within_page * tokens_per_chunk
            token_end = token_start + tokens_per_chunk

            page_idx = state.page_indices[page_pos]
            for layer in range(self.config.num_layers):
                ptr, nbytes = self._get_ptr_nbytes(
                    layer, page_idx, token_start, token_end
                )

                alloc_info.append(StoreAllocInfo(
                    key=self._key(request_id, label, chunk_idx, layer),
                    ptr=ptr,
                    nbytes=nbytes
                ))
        self._transfer_all(
            alloc_info, TransferType.GET
        )
        
        # --- Path 2: transfer_sync_read for trailing partial chunk ---
        if trailing_tokens > 0:
            chunk_idx = num_complete_chunks  # index of the trailing chunk
            page_pos = chunk_idx // chunks_per_page
            chunk_within_page = chunk_idx % chunks_per_page
            token_start = chunk_within_page * tokens_per_chunk
            token_end = token_start + trailing_tokens
            
            page_idx = state.page_indices[page_pos]

            local_addrs, remote_addrs, nbytes_list = [], [], []
            for layer in range(self.config.num_layers):
                ptrs, nbytes = self._get_ptr_nbytes(
                    layer, page_idx, token_start, token_end
                )
                remote_ptrs, _ = self._get_ptr_nbytes(
                    layer, seq_info.last_page_idx, token_start, token_end,
                    base_ptr=seq_info.kv_cache_addr
                )

                local_addrs.extend(ptrs)
                remote_addrs.extend(remote_ptrs)
                nbytes_list.extend(nbytes)

            # TODO: we might want to make this asynchronous to overlap
            # communication with computation, like in the main tensor mgr
            status = self.engine.batch_transfer_sync_read(
                seq_info.latest_session_id,
                local_addrs,
                remote_addrs,
                nbytes_list,
            )
            assert status == 0
            torch.cuda.default_stream().synchronize()

        state.seq_len = seq_len
        state.store_seq_len_per_layer = [
            num_complete_chunks * tokens_per_chunk for _ in range(self.config.num_layers)
        ]
        state.local_cache_seq_len = seq_len
        state.position_id_start = seq_info.pos_id

    def get_per_label_seq_info(
        self, request_id: str,
    ):
        torch.cuda.default_stream().synchronize()
        per_label_seq_info: dict[str, SequenceInfo] = {}
        for label, state in self.request_states.get(request_id, {}).items():
            state = self.get_state(request_id, label)
            per_label_seq_info[label] = SequenceInfo(
                seq_len = state.seq_len,
                pos_id = state.position_id_start,
                latest_entity_id = self.my_entity_id,
                latest_session_id = self.my_session_id,
                kv_cache_addr = self.kv_cache.data_ptr(),
            )
            if state.seq_len > 0:
                per_label_seq_info[label].last_page_idx = state.page_indices[
                    (state.seq_len - 1) // self.config.page_size]
        return per_label_seq_info
    
    def reset_label(self, request_id: str, label: str, free: bool=True, clear_store=True):
        if label in self.request_states[request_id] and free:
            state = self.request_states[request_id][label]
            self.page_allocator.free(state.page_indices)
        if clear_store:
            self.mooncake_store.remove_by_regex(f"^{request_id}_{label}.*")
        self.request_states[request_id][label] = self._new_state()

    def cleanup(self):
        self.mooncake_store.unregister_buffer(
            self.kv_cache.data_ptr()
        )
    
    def add_request(self, request_id: str, labels: list[str]=None):
        if labels is None:
            labels = []
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
