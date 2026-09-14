"""KV transfer: moving one request's pages between engines.

``mstar.communication.tensors`` is imported lazily, inside the two places
that genuinely need it at runtime (the transfer-engine ``isinstance``
dispatch and the Mooncake read). It pulls in the conductor and the
sampling kernels behind it, and this module is otherwise free of both —
keeping the import deferred is what lets the KV layer be built and tested
without a GPU toolchain present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from mstar.engine.resources.kv.cache import KVCache, KVLayout

if TYPE_CHECKING:
    from mstar.communication.tensors import (
        MooncakeTransferEngine,
        TensorTransferEngine,
    )



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
    layout: KVLayout


class MooncakeKVTransferEngine(KVTransferEngine):
    def __init__(
        self, kv_cache: KVCache,
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
            data_ptr=kv_cache.data_ptr(),
            layout=kv_cache.layout
        )
        self._async_reader = transfer_engine.get_async_reader(
            kv_cache.device
        )
        self._shut_down = False

    def get_kv_transfer_info(self) -> MooncakeKVTransferInfo:
        return self._transfer_info

    def read_batched_async(
        self, remote_kv_info: MooncakeKVTransferInfo,
        read_info: list[KVReadInfo]
    ) -> Future | None:
        # The pointer math assumes the remote cache has our layout and shape.
        assert remote_kv_info.layout == self._kv_cache.layout, (
            f"remote KV layout {remote_kv_info.layout} != "
            f"local layout {self._kv_cache.layout}"
        )
        from mstar.communication.tensors import TransferReadInfo

        mooncake_read_info: list[TransferReadInfo] = []
        for info in read_info:
            local_ptrs, nbytes = self._kv_cache.chunk_ptrs(
                layer_idx=info.layer_idx, page_idx=info.local_page_idx,
                token_start=info.token_start, token_end=info.token_end
            )
            remote_ptrs, _ = self._kv_cache.chunk_ptrs(
                layer_idx=info.layer_idx, page_idx=info.remote_page_idx,
                token_start=info.token_start, token_end=info.token_end,
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
        # unregister_memory is the one step that isn't safely repeatable
        if self._shut_down:
            return
        self._shut_down = True
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
    layout: KVLayout



class CudaIpcKVTransferEngine(KVTransferEngine):
    def __init__(
        self, kv_cache: KVCache,
        max_workers=3
    ):
        tensor = kv_cache.tensor
        storage = tensor.untyped_storage()
        cuda_share = storage._share_cuda_()
        self._transfer_info = CudaIpcKVTransferInfo(
            cuda_share=cuda_share,
            size=tensor.size(),
            stride=tensor.stride(),
            offset=tensor.storage_offset(),
            dtype=str(tensor.dtype),
            requires_grad=tensor.requires_grad,
            layout=kv_cache.layout
        )
        self._device = kv_cache.device
        self._kv_cache = kv_cache

        self._pending: list[Future] = []
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, initializer=self._bind_device,
        )

    def _bind_device(self) -> None:
        """CUDA's current device is per-thread and defaults to 0; an unbound
        pool thread has no context to open the peer's IPC handle against."""
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

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
        # The remote tensor is sliced with our layout, so it must match.
        assert remote_kv_info.layout == self._kv_cache.layout, (
            f"remote KV layout {remote_kv_info.layout} != "
            f"local layout {self._kv_cache.layout}"
        )
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
            src = self._kv_cache.chunk_view(
                layer_idx=info.layer_idx, page_idx=info.remote_page_idx,
                token_start=info.token_start, token_end=info.token_end,
                tensor=tensor
            )
            dst = self._kv_cache.chunk_view(
                layer_idx=info.layer_idx, page_idx=info.local_page_idx,
                token_start=info.token_start, token_end=info.token_end
            )
            dst.copy_(src)

    def shutdown(self):
        for fut in self._pending:
            fut.result()
        self._executor.shutdown(wait=True)


class LocalOnlyKVTransferEngine(KVTransferEngine):
    """KV cache that remains local to its worker."""

    def read_batched_async(
        self, remote_kv_info, read_info: list[KVReadInfo]
    ) -> Future | None:
        if read_info:
            raise RuntimeError(
                "Cross-worker KV migration is unavailable for this accelerator. "
                "Use a colocated worker graph or a remote transfer engine."
            )
        return None

    def get_kv_transfer_info(self) -> None:
        return None

    def shutdown(self):
        pass


@dataclass
class TransferEngineInfo:
    my_entity_id: str
    my_session_id: str
    transfer_engine: TensorTransferEngine


class KVTransferManager:
    def __init__(
        self, transfer_engine_info: TransferEngineInfo,
        kv_cache: KVCache
    ):
        from mstar.communication.tensors import (
            LocalTransferEngine,
            MooncakeTransferEngine,
        )

        self._page_size = kv_cache.page_size
        self._num_layers = kv_cache.num_layers

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
            if kv_cache.device.type == "cuda":
                self._kv_transfer_engine = CudaIpcKVTransferEngine(kv_cache)
            else:
                self._kv_transfer_engine = LocalOnlyKVTransferEngine()
        else:
            raise ValueError(f"Unsupported transfer engine type: {type(transfer_engine_info.transfer_engine)}")


    def sync_retrieve(
        self, *args, **kwargs
    ):
        future = self.start_async_retrieve(
            *args, **kwargs
        )
        if future is not None:
            future.result()

    def start_async_retrieve(
        self, start_len: int, end_len: int,
        local_page_indices: list[int],
        remote_page_indices: list[int],
        kv_transfer_info: Any,
    ) -> Future | None:
        if start_len >= end_len:
            return

        first_page = start_len // self._page_size
        last_page = (end_len - 1) // self._page_size

        read_info = []
        for page_pos in range(first_page, last_page + 1):
            token_start = 0 if page_pos > first_page else (start_len % self._page_size)
            token_end = self._page_size if page_pos != last_page else (
                end_len % self._page_size or self._page_size
            )

            local_page_idx = local_page_indices[page_pos]
            remote_page_idx = remote_page_indices[page_pos]

            for layer in range(self._num_layers):
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
        return self._kv_transfer_engine.read_batched_async(
            remote_kv_info=kv_transfer_info,
            read_info=read_info
        )

    def cleanup(self):
        self._kv_transfer_engine.shutdown()

    def get_kv_transfer_info(self):
        """Descriptor another process needs to read this cache remotely.
        ``KVCachePool.publish`` stamps it onto every ``SequenceInfo``."""
        return self._kv_transfer_engine.get_kv_transfer_info()
