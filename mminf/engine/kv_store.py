import logging
import queue
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum

import torch

from mminf.communication.tensors import TensorTransferEngine, TransferReadInfo
from mminf.conductor.request_info import SequenceInfo

logger = logging.getLogger(__name__)


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

    def try_allocate(self, n: int) -> list[int] | None:
        """Like allocate() but returns None instead of raising on failure."""
        if self.free_pages.qsize() < n:
            return None
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
    cpu_offload_pages: int = 0  # >0 enables CPU offloading with this many CPU pages
    nodes: list[str] = None # defaults to all AR nodes

    def __post_init__(self):
        if self.num_qo_heads is None:
            self.num_qo_heads = self.num_kv_heads

    def get_node_str(self):
        if self.nodes is None:
            return "ALL_NODES"
        return "///".join(self.nodes)



@dataclass
class PositionInfo:
    full_seq_len: int = 0
    position_id_start: int = 0


@dataclass
class KVRequestState:
    """Per-request KV cache state for the AR engine."""
    page_indices: list[int] = field(default_factory=list)
    seq_len: int = 0 # includes read in progress
    position_id_start: int = 0

    read_in_progress: bool = False

    # sequence length of the in-distributed-store KV cache
    is_paused: bool = False

    def get_pos_info(self):
        return PositionInfo(
            full_seq_len=self.seq_len,
            position_id_start=self.position_id_start
        )


LabelToState = dict[str, KVRequestState]


@dataclass
class AllocationStatus:
    """Tracks the outcome of the most recent page allocation attempt."""
    success: bool = True
    pages_short: int = 0       # how many pages we couldn't allocate
    request_id: str | None = None  # which request failed
    label: str | None = None       # which cache label failed

    def reset(self):
        self.success = True
        self.pages_short = 0
        self.request_id = None
        self.label = None


@dataclass
class StoreAllocInfo:
    key: str
    ptr: list[int]
    nbytes: list[int]


@dataclass
class TransferEngineInfo:
    my_entity_id: str
    my_session_id: str
    transfer_engine: TensorTransferEngine


class StoreWritePolicy(Enum):
    ALWAYS = "always"   # disaggregated: this worker's KV may be needed elsewhere
    NEVER = "never"     # non-disaggregated: all AR graph walks on same worker


class PagedAllocationManager:
    def __init__(
        self,
        config: KVCacheConfig,
        kv_cache: torch.Tensor,
        transfer_engine_info: TransferEngineInfo
    ):
        self.config = config
        self.page_allocator = PageAllocator(config.max_num_pages)
        self.request_states: dict[str, LabelToState] = {}
        self.kv_cache = kv_cache
        self.write_policy = StoreWritePolicy.ALWAYS

        self._transfer_engine = transfer_engine_info.transfer_engine
        self._async_reader = self._transfer_engine.get_async_reader(kv_cache.device)
        self._transfer_engine.register_memory(
            self.kv_cache.data_ptr(), self.kv_cache.nbytes
        )
        self.my_entity_id = transfer_engine_info.my_entity_id
        self.my_session_id = transfer_engine_info.my_session_id

        # Stream for async GPU↔CPU page copies (Feature 3: CPU offloading)
        self._offload_stream: torch.cuda.Stream | None = None

        # Tracks the outcome of the most recent allocation attempt per batch.
        # Reset at the start of each batch by the engine.
        self.alloc_status = AllocationStatus()

        # {req_id: {label: futures}}
        self.pending_reads: dict[str, dict[str, list[Future]]] = {}

    @property
    def num_free_pages(self) -> int:
        return self.page_allocator.num_free

    @property
    def total_pages(self) -> int:
        return self.config.max_num_pages

    def _key(self, request_id: str, label: str, pos: int, layer: int):
        return f"{request_id}_{label}_{pos}_{layer}"

    def _get_ptr_nbytes(
        self, layer, page_idx,
        token_start, token_end,
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

        return ptrs, nbytes

    def flush_to_store(
        self, request_id: str, label: str, layers: int | list[int] | None = None
    ):
        # For now, is a no-op. In the future, when we have prefetching at the receiving end,
        # this function will posibly send ZMQ requests to potential receivers, who can do
        # RDMA reads on this Engine's KV cache
        return

    def _new_state(self):
        state = KVRequestState()
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
            new_pages = self.page_allocator.try_allocate(num_new_pages)
            if new_pages is None:
                self.alloc_status = AllocationStatus(
                    success=False,
                    pages_short=num_new_pages - self.page_allocator.num_free,
                    request_id=request_id,
                    label=label,
                )
                raise RuntimeError(
                    f"Not enough free pages: requested {num_new_pages}, "
                    f"available {self.page_allocator.num_free}"
                )
            state.page_indices.extend(new_pages)

    def wait_for_retrieves(
        self, request_id: str, label: str
    ):
        for future in self.pending_reads[request_id].get(label, []):
            future.result()
        state = self.get_state(request_id, label)
        state.read_in_progress = False
        self.pending_reads[request_id][label] = []

    def check_retrieve_ready(
        self, request_id: str, label: str
    ) -> bool:
        """
        Returns true if all retrieves are done
        """
        state = self.get_state(request_id, label)
        if not state.read_in_progress:
            return True
        futures = [
            fut for fut in self.pending_reads[request_id].get(label, []) \
                if not fut.done()
        ]
        in_progress = (len(futures) > 0)
        state.read_in_progress = in_progress
        self.pending_reads[request_id][label] = futures
        return not in_progress

    def sync_retrieve(
        self, request_id: str, label: str, seq_info: SequenceInfo
    ):
        self.start_async_retrieve(request_id, label, seq_info)
        self.wait_for_retrieves(request_id, label)

    def start_async_retrieve(
        self, request_id: str, label: str, seq_info: SequenceInfo
    ):
        seq_len = seq_info.seq_len
        state = self.get_state(request_id, label)
        if state.seq_len >= seq_len:
            return  # nothing to do

        first_page = state.seq_len // self.config.page_size
        last_page = (seq_len - 1) // self.config.page_size

        self.alloc(request_id, label, seq_len)

        # When _async_reader is None (e.g., SHM / single-node), KV cache data
        # is already in local GPU memory — no cross-worker transfer needed.
        if self._async_reader is not None:
            read_info = []

            for page_pos in range(first_page, last_page + 1):
                token_start = 0 if page_pos > first_page else (state.seq_len % self.config.page_size)
                token_end = self.config.page_size if page_pos != last_page else (
                    seq_len % self.config.page_size or self.config.page_size
                )

                local_page_idx = state.page_indices[page_pos]
                remote_page_idx = seq_info.page_indices[page_pos]

                for layer in range(self.config.num_layers):
                    local_ptrs, nbytes = self._get_ptr_nbytes(
                        layer, local_page_idx, token_start, token_end
                    )
                    remote_ptrs, _ = self._get_ptr_nbytes(
                        layer, remote_page_idx, token_start, token_end,
                        base_ptr=seq_info.kv_cache_addr
                    )

                    read_info.extend([
                        TransferReadInfo(
                            seq_info.latest_session_id,
                            local_ptr, remote_ptr, nbytes
                        ) for local_ptr, remote_ptr in zip(local_ptrs, remote_ptrs, strict=True)
                    ])
            future = self._async_reader.submit(read_info)
            if future is not None:
                self.pending_reads[request_id].setdefault(label, []).append(future)

        state.seq_len = seq_len
        state.position_id_start = seq_info.pos_id
        state.read_in_progress = True

    def get_per_label_seq_info(self, request_id: str):
        per_label_seq_info: dict[str, SequenceInfo] = {}
        for label, state in self.request_states.get(request_id, {}).items():
            self.wait_for_retrieves(request_id, label)

            state = self.get_state(request_id, label)
            per_label_seq_info[label] = SequenceInfo(
                seq_len = state.seq_len,
                pos_id = state.position_id_start,
                latest_entity_id = self.my_entity_id,
                latest_session_id = self.my_session_id,
                kv_cache_addr = self.kv_cache.data_ptr(),
                page_indices=state.page_indices
            )
        return per_label_seq_info

    def get_labels(self, request_id: str):
        return list(self.request_states[request_id].keys())

    def reset_label(self, request_id: str, label: str, free: bool=True):
        self.wait_for_retrieves(request_id, label)
        if label in self.request_states[request_id] and free:
            state = self.request_states[request_id][label]
            self.page_allocator.free(state.page_indices)
        self.request_states[request_id][label] = self._new_state()

    def cleanup(self):
        if self._async_reader is not None:
            self._async_reader.shutdown()
        self._transfer_engine.unregister_memory(
            self.kv_cache.data_ptr()
        )

    def add_request(self, request_id: str, labels: list[str]=None):
        if labels is None:
            labels = []
        self.request_states[request_id] = {
            label: self._new_state() for label in labels
        }
        self.pending_reads[request_id] = {
            label: [] for label in labels
        }

    def remove_request(self, request_id: str):
        for label in self.request_states[request_id]:
            self.wait_for_retrieves(request_id, label)
            state = self.request_states[request_id][label]
            self.page_allocator.free(state.page_indices)
        del self.request_states[request_id]
        del self.pending_reads[request_id]

    # ----- CPU offloading helpers -----

    def offload_request(self, request_id: str, cpu_pool) -> int:
        """Offload all labels for a request to *cpu_pool*, free GPU pages.

        Returns the total number of GPU pages freed.
        """
        freed = 0
        for label, state in self.request_states[request_id].items():
            if not state.page_indices:
                continue
            self.wait_for_retrieves(request_id, label)
            cpu_pool.offload_pages(
                request_id, label, self.kv_cache,
                state.page_indices, state.seq_len, state.position_id_start,
            )
            freed += len(state.page_indices)
            self.page_allocator.free(state.page_indices)
            state.page_indices = []
            state.seq_len = 0
        return freed

    def reload_request(self, request_id: str, cpu_pool) -> None:
        """Reload all labels for a request from *cpu_pool* back to GPU."""
        for label in list(cpu_pool.offloaded.get(request_id, {}).keys()):
            offloaded = cpu_pool.offloaded[request_id][label]
            n_pages = len(offloaded.cpu_page_indices)
            gpu_pages = self.page_allocator.allocate(n_pages)
            state = self.get_state(request_id, label)
            state.page_indices = gpu_pages
            seq_len, pos_id = cpu_pool.reload_pages(
                request_id, label, self.kv_cache, gpu_pages,
            )
            state.seq_len = seq_len
            state.position_id_start = pos_id
        cpu_pool.sync()

