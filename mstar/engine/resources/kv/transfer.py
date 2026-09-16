"""KV transfer: moving one request's pages between engines.

``mstar.communication.tensors`` is imported lazily, inside the two places
that genuinely need it at runtime (the transfer-engine ``isinstance``
dispatch and the Mooncake read). It pulls in the conductor and the
sampling kernels behind it, and this module is otherwise free of both —
keeping the import deferred is what lets the KV layer be built and tested
without a GPU toolchain present.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
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

    def remove_request(self, request_id: str) -> None:
        """Release request-scoped transfer resources, if any."""
        del request_id

    def owns_transfer_info(
        self,
        transfer_info: Any,
        request_id: str,
        label: str,
    ) -> bool:
        del request_id, label
        return transfer_info == self.get_kv_transfer_info()

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


@dataclass(frozen=True)
class ShmKVTransferInfo:
    """One packed publication inside a producer's long-lived SHM arena."""

    path: str
    page_indices: tuple[int, ...]
    layout: KVLayout
    dtype: str
    packed_shape: tuple[int, ...]
    slot_offset: int
    data_offset: int
    data_nbytes: int
    arena_size: int
    generation: int


@dataclass(frozen=True)
class _ShmArenaSlot:
    offset: int
    size: int


class ShmKVTransferEngine(KVTransferEngine):
    """Host-staged KV transfer through one grow-on-demand arena per engine.

    A publication occupies an arena slot and contains only the physical pages
    used by one request label. The arena path is the stable, worker-scoped
    transport identity; slot offsets and generations are publication metadata.
    Slots remain immutable until request cleanup, then return to the allocator.
    """

    _GENERATION_BYTES = 8
    _SLOT_ALIGNMENT = 64

    def __init__(
        self,
        kv_cache: KVCache,
        entity_id: str,
        shm_dir: str | None = None,
    ):
        self._kv_cache = kv_cache
        root = shm_dir or os.getenv("MSTAR_KV_SHM_DIR")
        if root is None:
            root = (
                "/dev/shm/mstar_kv"
                if os.path.isdir("/dev/shm")
                else "/tmp/mstar_kv"
            )
        self._shm_dir = root
        os.makedirs(self._shm_dir, exist_ok=True)
        self._entity_id = entity_id
        identity = (
            f"{entity_id}:{os.getpid()}:{secrets.token_hex(16)}".encode()
        )
        digest = hashlib.sha256(identity).hexdigest()
        self._arena_path = os.path.join(
            self._shm_dir, f"mstar_kv_arena_{digest}.bin"
        )
        fd = os.open(
            self._arena_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o600,
        )
        os.close(fd)
        self._arena_size = 0
        self._generation = 0
        self._free_slots: list[_ShmArenaSlot] = []
        self._request_slots: dict[str, list[_ShmArenaSlot]] = {}
        self._lock = threading.RLock()
        self._shutdown = False
        self._published: dict[
            tuple[str, str],
            tuple[tuple[tuple[int, ...], int], ShmKVTransferInfo],
        ] = {}

    @classmethod
    def _aligned_slot_size(cls, data_nbytes: int) -> int:
        required = cls._GENERATION_BYTES + data_nbytes
        alignment = cls._SLOT_ALIGNMENT
        return ((required + alignment - 1) // alignment) * alignment

    def _allocate_slot(self, data_nbytes: int) -> _ShmArenaSlot:
        required = self._aligned_slot_size(data_nbytes)
        for idx, free in enumerate(self._free_slots):
            if free.size < required:
                continue
            slot = _ShmArenaSlot(offset=free.offset, size=required)
            remaining = free.size - required
            if remaining:
                self._free_slots[idx] = _ShmArenaSlot(
                    offset=free.offset + required,
                    size=remaining,
                )
            else:
                self._free_slots.pop(idx)
            return slot

        slot = _ShmArenaSlot(offset=self._arena_size, size=required)
        self._arena_size += required
        os.truncate(self._arena_path, self._arena_size)
        return slot

    def _release_slot(self, slot: _ShmArenaSlot) -> None:
        free_slots = sorted(
            [*self._free_slots, slot],
            key=lambda candidate: candidate.offset,
        )
        merged: list[_ShmArenaSlot] = []
        for candidate in free_slots:
            if (
                merged
                and merged[-1].offset + merged[-1].size == candidate.offset
            ):
                previous = merged[-1]
                merged[-1] = _ShmArenaSlot(
                    offset=previous.offset,
                    size=previous.size + candidate.size,
                )
            else:
                merged.append(candidate)
        self._free_slots = merged

    @staticmethod
    def _map_arena(path: str, size: int) -> torch.Tensor:
        return torch.from_file(
            path,
            shared=True,
            size=size,
            dtype=torch.uint8,
        )

    @classmethod
    def _read_generation(
        cls,
        arena: torch.Tensor,
        slot_offset: int,
    ) -> int:
        raw = bytes(
            arena[
                slot_offset:slot_offset + cls._GENERATION_BYTES
            ].tolist()
        )
        return int.from_bytes(raw, byteorder="little", signed=False)

    @classmethod
    def _write_generation(
        cls,
        arena: torch.Tensor,
        slot_offset: int,
        generation: int,
    ) -> None:
        raw = generation.to_bytes(
            cls._GENERATION_BYTES,
            byteorder="little",
            signed=False,
        )
        arena[
            slot_offset:slot_offset + cls._GENERATION_BYTES
        ].copy_(torch.tensor(list(raw), dtype=torch.uint8))

    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ) -> ShmKVTransferInfo | None:
        if (
            request_id is None
            or label is None
            or page_indices is None
            or seq_len is None
        ):
            return None
        with self._lock:
            if self._shutdown:
                raise RuntimeError("SHM KV transfer engine is shut down")

            pages = tuple(page_indices)
            version = (pages, seq_len)
            key = (request_id, label)
            previous = self._published.get(key)
            if previous is not None and previous[0] == version:
                return previous[1]

            if pages:
                packed = (
                    self._kv_cache.tensor[:, list(pages)]
                    .detach()
                    .cpu()
                    .contiguous()
                )
            else:
                packed = torch.empty(
                    (
                        self._kv_cache.num_layers,
                        0,
                        *self._kv_cache.tensor.shape[2:],
                    ),
                    dtype=self._kv_cache.dtype,
                )

            data_nbytes = packed.numel() * packed.element_size()
            slot = self._allocate_slot(data_nbytes)
            data_offset = slot.offset + self._GENERATION_BYTES
            arena = self._map_arena(self._arena_path, self._arena_size)
            if data_nbytes:
                destination = arena[
                    data_offset:data_offset + data_nbytes
                ].view(self._kv_cache.dtype).view(packed.shape)
                destination.copy_(packed)

            self._generation += 1
            self._write_generation(
                arena,
                slot_offset=slot.offset,
                generation=self._generation,
            )
            info = ShmKVTransferInfo(
                path=self._arena_path,
                page_indices=pages,
                layout=self._kv_cache.layout,
                dtype=str(self._kv_cache.dtype),
                packed_shape=tuple(packed.shape),
                slot_offset=slot.offset,
                data_offset=data_offset,
                data_nbytes=data_nbytes,
                arena_size=self._arena_size,
                generation=self._generation,
            )
            self._request_slots.setdefault(request_id, []).append(slot)
            self._published[key] = (version, info)
            return info

    def read_batched_async(
        self,
        remote_kv_info: ShmKVTransferInfo | None,
        read_info: list[KVReadInfo],
    ) -> Future | None:
        if not read_info:
            return None
        if remote_kv_info is None:
            raise RuntimeError("Missing SHM metadata for remote KV cache")
        if remote_kv_info.layout != self._kv_cache.layout:
            raise ValueError(
                f"remote KV layout {remote_kv_info.layout} != "
                f"local layout {self._kv_cache.layout}"
            )

        dtype = getattr(torch, remote_kv_info.dtype.split(".")[-1])
        if dtype != self._kv_cache.dtype:
            raise ValueError(
                f"remote KV dtype {dtype} != local dtype "
                f"{self._kv_cache.dtype}"
            )
        expected_shape = (
            self._kv_cache.num_layers,
            len(remote_kv_info.page_indices),
            *self._kv_cache.tensor.shape[2:],
        )
        if remote_kv_info.packed_shape != expected_shape:
            raise ValueError(
                f"remote packed KV shape {remote_kv_info.packed_shape} != "
                f"expected {expected_shape}"
            )

        arena = self._map_arena(
            remote_kv_info.path,
            remote_kv_info.arena_size,
        )
        generation = self._read_generation(
            arena,
            remote_kv_info.slot_offset,
        )
        if generation != remote_kv_info.generation:
            raise RuntimeError(
                "SHM KV arena slot was reused before retrieval: "
                f"expected generation {remote_kv_info.generation}, "
                f"found {generation}"
            )
        packed = arena[
            remote_kv_info.data_offset:
            remote_kv_info.data_offset + remote_kv_info.data_nbytes
        ].view(dtype).view(remote_kv_info.packed_shape)
        packed_page = {
            remote_page: idx
            for idx, remote_page in enumerate(remote_kv_info.page_indices)
        }
        page_copies = {
            (
                info.layer_idx,
                info.remote_page_idx,
                info.local_page_idx,
                info.token_start,
                info.token_end,
            )
            for info in read_info
        }
        for layer, remote_page, local_page, token_start, token_end in page_copies:
            source = packed[
                layer,
                packed_page[remote_page],
                :,
                token_start:token_end,
            ].to(self._kv_cache.device)
            destination = self._kv_cache.chunk_view(
                layer_idx=layer,
                page_idx=local_page,
                token_start=token_start,
                token_end=token_end,
            )
            destination.copy_(source)
        if self._kv_cache.device.type != "cpu":
            torch.accelerator.synchronize(self._kv_cache.device)
        generation_after = self._read_generation(
            arena,
            remote_kv_info.slot_offset,
        )
        if generation_after != remote_kv_info.generation:
            raise RuntimeError(
                "SHM KV arena slot changed during retrieval: "
                f"expected generation {remote_kv_info.generation}, "
                f"found {generation_after}"
            )
        return None

    def remove_request(self, request_id: str) -> None:
        with self._lock:
            keys = [key for key in self._published if key[0] == request_id]
            for key in keys:
                self._published.pop(key)
            for slot in self._request_slots.pop(request_id, []):
                self._release_slot(slot)

    def owns_transfer_info(
        self,
        transfer_info: Any,
        request_id: str,
        label: str,
    ) -> bool:
        del request_id, label
        return (
            isinstance(transfer_info, ShmKVTransferInfo)
            and transfer_info.path == self._arena_path
        )

    def shutdown(self):
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._published.clear()
            self._request_slots.clear()
            self._free_slots.clear()
            try:
                os.unlink(self._arena_path)
            except FileNotFoundError:
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
                self._kv_transfer_engine = ShmKVTransferEngine(
                    kv_cache=kv_cache,
                    entity_id=transfer_engine_info.my_entity_id,
                )
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

    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ):
        """Descriptor another process needs to read this cache remotely.
        ``KVCachePool.publish`` stamps it onto every ``SequenceInfo``."""
        if isinstance(self._kv_transfer_engine, ShmKVTransferEngine):
            return self._kv_transfer_engine.get_kv_transfer_info(
                request_id=request_id,
                label=label,
                page_indices=page_indices,
                seq_len=seq_len,
            )
        return self._kv_transfer_engine.get_kv_transfer_info()

    def remove_request(self, request_id: str) -> None:
        self._kv_transfer_engine.remove_request(request_id)

    def owns_transfer_info(
        self,
        transfer_info: Any,
        request_id: str,
        label: str,
    ) -> bool:
        return self._kv_transfer_engine.owns_transfer_info(
            transfer_info=transfer_info,
            request_id=request_id,
            label=label,
        )
