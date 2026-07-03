import logging
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from mstar.communication.tensors import (
    LocalTransferEngine,
    MooncakeTransferEngine,
    TensorTransferEngine,
    TransferReadInfo,
)
from mstar.conductor.request_info import SequenceInfo
from mstar.distributed.utils import divide

logger = logging.getLogger(__name__)


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

        self._sharded = False
        self.original_num_kv_heads = self.num_kv_heads
        self.original_num_qo_heads =  self.num_qo_heads

    def get_node_str(self):
        if self.nodes is None:
            return "ALL_NODES"
        return "///".join(self.nodes)

    def shard(self, tp_size: int):
        if tp_size >= self.original_num_kv_heads:
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = divide(self.original_num_kv_heads, tp_size)
        self.num_qo_heads = divide(self.original_num_qo_heads, tp_size)
        self._sharded = True



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


class AllocationFailedError(RuntimeError):
    """Raised by ``PagedAllocationManager.alloc`` when the page pool is too
    small to satisfy a request. Carries the diagnostic payload (pages_short,
    request_id, label) on the exception itself so callers across threads can
    recover it without consulting any shared state on the manager.
    """

    def __init__(
        self,
        pages_short: int,
        request_id: str,
        label: str,
        message: str | None = None,
    ):
        super().__init__(
            message
            or f"Page allocation failed: {pages_short} page(s) short "
               f"for request {request_id!r} label {label!r}"
        )
        self.pages_short = pages_short
        self.request_id = request_id
        self.label = label


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



@dataclass
class KVReadInfo:
    layer_idx: int
    local_page_idx: int
    remote_page_idx: int
    token_start: int
    token_end: int


class KVTransferEngine(ABC):
    @abstractmethod
    def read_batched_async(
        self, remote_kv_info,
        read_info: list[KVReadInfo]
    ) -> Future | None:
        pass

    @abstractmethod
    def get_kv_transfer_info(self) -> Any:
        pass

    @abstractmethod
    def shutdown(self):
        pass


@dataclass
class MooncakeKVTransferInfo:
    entity_id: str
    session_id: str
    data_ptr: int


class MooncakeKVTransferEngine(KVTransferEngine):
    def __init__(
        self, kv_cache: torch.Tensor,
        entity_id: str,
        transfer_engine: MooncakeTransferEngine
    ):
        self._kv_cache = kv_cache
        self._transfer_engine = transfer_engine
        self._transfer_engine.register_memory(
            kv_cache.data_ptr(), kv_cache.nbytes
        )
        self._transfer_info = MooncakeKVTransferInfo(
            entity_id=entity_id,
            session_id=transfer_engine.get_session_id(),
            data_ptr=kv_cache.data_ptr()
        )
        self._async_reader = transfer_engine.get_async_reader(
            kv_cache.device
        )

    def get_kv_transfer_info(self) -> MooncakeKVTransferInfo:
        return self._transfer_info

    def _get_ptr_nbytes(
        self, kv_read_info: KVReadInfo,
        is_local: bool=True,
        base_ptr=None
    ):
        token_stride = self._kv_cache.stride(3)
        kv_stride = self._kv_cache.stride(2)
        page_stride = self._kv_cache.stride(1)
        layer_stride = self._kv_cache.stride(0)
        element_size = self._kv_cache.element_size()
        tokens_per_chunk = kv_read_info.token_end - kv_read_info.token_start

        nbytes = tokens_per_chunk * token_stride * element_size  # token_stride = num_kv_heads * head_dim

        if base_ptr is None:
            base_ptr = self._kv_cache.data_ptr()

        page_idx  = kv_read_info.local_page_idx if is_local else kv_read_info.remote_page_idx
        ptrs = [
            base_ptr + (
                    kv_read_info.layer_idx * layer_stride +
                    page_idx * page_stride +
                    kv_idx * kv_stride +
                    kv_read_info.token_start * token_stride
                ) * element_size for kv_idx in [0, 1]
        ]
        return ptrs, nbytes

    def read_batched_async(
        self, remote_kv_info: MooncakeKVTransferInfo,
        read_info: list[KVReadInfo]
    ) -> Future | None:
        mooncake_read_info: list[TransferReadInfo] = []
        for info in read_info:
            local_ptrs, nbytes = self._get_ptr_nbytes(
                kv_read_info=info, is_local=True
            )
            remote_ptrs, _ = self._get_ptr_nbytes(
                kv_read_info=info, is_local=False,
                base_ptr=remote_kv_info.data_ptr
            )
            mooncake_read_info.extend([
                TransferReadInfo(
                    remote_kv_info.session_id,
                    local_ptr, remote_ptr, nbytes
                ) for local_ptr, remote_ptr in zip(local_ptrs, remote_ptrs, strict=True)
            ])
        return self._async_reader.submit(mooncake_read_info)

    def shutdown(self):
        self._async_reader.shutdown()
        self._transfer_engine.unregister_memory(
            self._kv_cache.data_ptr()
        )


@dataclass
class CudaIpcKVTransferInfo:
    cuda_share: tuple
    size: tuple
    stride: tuple
    offset: int
    dtype: str
    requires_grad: bool


# TODO: this can also become a regular tensor transport method
class CudaIpcKVTransferEngine(KVTransferEngine):
    def __init__(
        self, kv_cache: torch.Tensor,
        max_workers=3
    ):
        storage = kv_cache.untyped_storage()
        cuda_share = storage._share_cuda_()
        self._transfer_info = CudaIpcKVTransferInfo(
            cuda_share=cuda_share,
            size=kv_cache.size(),
            stride=kv_cache.stride(),
            offset=kv_cache.storage_offset(),
            dtype=str(kv_cache.dtype),
            requires_grad=kv_cache.requires_grad
        )
        self._device = kv_cache.device
        self._kv_cache = kv_cache

        self._pending: list[Future] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_kv_transfer_info(self) -> CudaIpcKVTransferInfo:
        return self._transfer_info

    def read_batched_async(
        self, remote_kv_info: CudaIpcKVTransferInfo,
        read_info: list[KVReadInfo]
    ):
        if not read_info:
            return
        event = torch.cuda.current_stream().record_event()
        future = self._executor.submit(self._do_read, remote_kv_info, read_info, event)
        self._pending.append(future)
        # Prune completed futures to avoid unbounded growth
        self._pending = [f for f in self._pending if not f.done()]
        return future

    def _do_read(
        self, remote_kv_info: CudaIpcKVTransferInfo,
        read_info: list[KVReadInfo],
        event: torch.Event=None
    ):
        event.synchronize()
        dtype = getattr(torch, remote_kv_info.dtype.split(".")[-1])
        (
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = remote_kv_info.cuda_share

        # Note: as this is a zero-copy operation and not allocating memory
        # (just building a reference to underlying storage on the sending device),
        # it is ok that this is rebuilding the whole kv cache. What matters is that,
        # in the rest of the function, we are only copying the right pages to
        # self._device. In fact, in testing, we see it is faster to call
        # rebuild_cuda_tensor on the whole KV cache instead of just the slice we need.
        tensor = rebuild_cuda_tensor(
            torch.Tensor,
            remote_kv_info.size,
            remote_kv_info.stride,
            remote_kv_info.offset,
            torch.UntypedStorage,
            dtype,
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            remote_kv_info.requires_grad,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        )

        for info in read_info:
            slice = tensor[
                info.layer_idx, info.remote_page_idx,
                info.token_start:info.token_end
            ].to(self._device)
            self._kv_cache[
                info.layer_idx, info.local_page_idx,
                info.token_start:info.token_end
            ] = slice

    def shutdown(self):
        for fut in self._pending:
            fut.result()
        self._executor.shutdown(wait=True)

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
        # RLock guards request_states mutation against concurrent
        # plan-thread alloc vs GPU-thread reset_label/remove_request.
        # RLock (not Lock) so a future caller that nests another guarded
        # method inside ``alloc``/``reset_label`` (e.g. wrapping
        # ``start_async_retrieve``) doesn't self-deadlock.
        self._lock = threading.RLock()

        if isinstance(
            transfer_engine_info.transfer_engine, MooncakeTransferEngine
        ):
            self._kv_transfer_engine = MooncakeKVTransferEngine(
                kv_cache=kv_cache,
                entity_id=transfer_engine_info.my_entity_id,
                transfer_engine=transfer_engine_info.transfer_engine
            )
        elif isinstance(
            transfer_engine_info.transfer_engine, LocalTransferEngine
        ):
            self._kv_transfer_engine = CudaIpcKVTransferEngine(kv_cache)
        else:
            raise ValueError(f"Unsupported transfer engine type: {type(transfer_engine_info.transfer_engine)}")

        # Stream for async GPU↔CPU page copies (Feature 3: CPU offloading)
        self._offload_stream: torch.cuda.Stream | None = None

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
        with self._lock:
            state = self.request_states[request_id][label]
            num_pages_needed = (seq_len + self.config.page_size - 1) // self.config.page_size
            num_new_pages = num_pages_needed - len(state.page_indices)
            if num_new_pages > 0:
                new_pages = self.page_allocator.try_allocate(num_new_pages)
                if new_pages is None:
                    raise AllocationFailedError(
                        pages_short=num_new_pages - self.page_allocator.num_free,
                        request_id=request_id,
                        label=label,
                        message=(
                            f"Not enough free pages: requested {num_new_pages}, "
                            f"available {self.page_allocator.num_free}"
                        ),
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

        read_info = []
        for page_pos in range(first_page, last_page + 1):
            token_start = 0 if page_pos > first_page else (state.seq_len % self.config.page_size)
            token_end = self.config.page_size if page_pos != last_page else (
                seq_len % self.config.page_size or self.config.page_size
            )

            local_page_idx = state.page_indices[page_pos]
            remote_page_idx = seq_info.page_indices[page_pos]

            for layer in range(self.config.num_layers):
                read_info.append(KVReadInfo(
                    layer_idx=layer, local_page_idx=local_page_idx,
                    remote_page_idx=remote_page_idx,
                    token_start=token_start,
                    token_end=token_end
                ))
        # Important: in both the RDMA and SHM paths, we need to make sure that the KV
        # cache data is ready at the producer end before the consumer reads it. Pytorch
        # does not currently support transmitting Event objects over IPC, so we opt to
        # use the following contract: the producer always default-stream-syncs before
        # publishing seq_info (this currently happens in worker.py, right before sending
        # outputs). Once torch.Event supports the interprocess flag (it's present in the
        # function signature but currently a no-op), this path can be refatored to wait
        # on an event on the reader end instead.
        future = self._kv_transfer_engine.read_batched_async(
            remote_kv_info=seq_info.latest_kv_transfer_info,
            read_info=read_info
        )
        if future is not None:
            self.pending_reads[request_id].setdefault(label, []).append(future)

        state.seq_len = seq_len
        state.position_id_start = seq_info.pos_id
        state.read_in_progress = future is not None

    def get_per_label_seq_info(self, request_id: str):
        per_label_seq_info: dict[str, SequenceInfo] = {}
        transfer_info = self._kv_transfer_engine.get_kv_transfer_info()
        for label, state in self.request_states.get(request_id, {}).items():
            self.wait_for_retrieves(request_id, label)

            state = self.get_state(request_id, label)
            per_label_seq_info[label] = SequenceInfo(
                seq_len = state.seq_len,
                pos_id = state.position_id_start,
                latest_kv_transfer_info=transfer_info,
                page_indices=state.page_indices
            )
        return per_label_seq_info

    def get_labels(self, request_id: str):
        return list(self.request_states[request_id].keys())

    def reset_label(self, request_id: str, label: str, free: bool=True):
        self.wait_for_retrieves(request_id, label)
        with self._lock:
            if label in self.request_states[request_id] and free:
                state = self.request_states[request_id][label]
                self.page_allocator.free(state.page_indices)
            self.request_states[request_id][label] = self._new_state()

    def cleanup(self):
        self._kv_transfer_engine.shutdown()

    def add_request(self, request_id: str, labels: list[str]=None):
        if labels is None:
            labels = []
        with self._lock:
            self.request_states[request_id] = {
                label: self._new_state() for label in labels
            }
            self.pending_reads[request_id] = {
                label: [] for label in labels
            }

    def remove_request(self, request_id: str):
        for label in self.request_states[request_id]:
            self.wait_for_retrieves(request_id, label)
        with self._lock:
            for label in self.request_states[request_id]:
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
        # wait_for_retrieves can BLOCK on future.result(); release the
        # manager lock around it so the plan thread can still acquire
        # the lock for ``alloc`` while we wait. cpu_pool.offload_pages
        # also touches GPU streams and is best left outside the lock.
        with self._lock:
            labels = list(self.request_states[request_id].items())
        for label, state in labels:
            if not state.page_indices:
                continue
            self.wait_for_retrieves(request_id, label)
            cpu_pool.offload_pages(
                request_id, label, self.kv_cache,
                state.page_indices, state.seq_len, state.position_id_start,
            )
            with self._lock:
                freed += len(state.page_indices)
                self.page_allocator.free(state.page_indices)
                state.page_indices = []
                state.seq_len = 0
        return freed

    def reload_request(self, request_id: str, cpu_pool) -> None:
        """Reload all labels for a request from *cpu_pool* back to GPU."""
        with self._lock:
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

