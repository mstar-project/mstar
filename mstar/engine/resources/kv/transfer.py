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
import shutil
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
    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ) -> Any:
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

    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ) -> MooncakeKVTransferInfo:
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

    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ) -> CudaIpcKVTransferInfo:
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
    """KV cache that never leaves its worker instance."""

    def read_batched_async(
        self, remote_kv_info, read_info: list[KVReadInfo]
    ) -> Future | None:
        del remote_kv_info
        if read_info:
            raise RuntimeError(
                "Cross-worker KV migration was requested for a local-only "
                "resource"
            )
        return None

    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ) -> None:
        del request_id, label, page_indices, seq_len

    def shutdown(self):
        pass


@dataclass(frozen=True)
class ShmKVTransferInfo:
    path: str
    page_indices: tuple[int, ...]
    layout: KVLayout


class ShmKVTransferEngine(KVTransferEngine):
    """Host-staged KV transfer through a shared-memory filesystem.

    Each producer publishes a packed tensor containing only the occupied
    physical pages for one request label. Consumers attach by path and copy
    the requested page/token ranges into their local accelerator cache.
    """

    def __init__(
        self,
        kv_cache: KVCache,
        entity_id: str,
        shm_dir: str,
        resource_key: str = "kv",
    ):
        if not shm_dir:
            raise ValueError(
                "shm_dir is required for shared-memory KV transfer"
            )
        self._kv_cache = kv_cache
        self._shm_dir = shm_dir
        os.makedirs(self._shm_dir, mode=0o700, exist_ok=True)
        os.chmod(self._shm_dir, 0o700)
        self._entity_id = entity_id
        self._resource_key = resource_key
        self._published: dict[
            tuple[str, str],
            tuple[tuple[tuple[int, ...], int], ShmKVTransferInfo],
        ] = {}

    def _path(self, request_id: str, label: str) -> str:
        key = (
            f"{self._entity_id}:{self._resource_key}:{request_id}:{label}"
        ).encode()
        digest = hashlib.sha256(key).hexdigest()
        return os.path.join(self._shm_dir, f"mstar_kv_{digest}.pt")

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
        pages = tuple(page_indices)
        version = (pages, seq_len)
        key = (request_id, label)
        previous = self._published.get(key)
        if previous is not None and previous[0] == version:
            return previous[1]

        path = self._path(request_id, label)
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        if pages:
            packed = (
                self._kv_cache.tensor[:, list(pages)]
                .detach()
                .cpu()
                .contiguous()
            )
        else:
            packed = torch.empty((0,), dtype=self._kv_cache.dtype)
        torch.save(packed, tmp_path)
        os.replace(tmp_path, path)
        info = ShmKVTransferInfo(
            path=path,
            page_indices=pages,
            layout=self._kv_cache.layout,
        )
        self._published[key] = (version, info)
        return info

    def read_batched_async(
        self,
        remote_kv_info: ShmKVTransferInfo | None,
        read_info: list[KVReadInfo],
    ) -> Future | None:
        try:
            return self._read_batched(remote_kv_info, read_info)
        except Exception as error:
            # SHM reads are synchronous, but the resource readiness contract
            # expects transfer failures on a Future so they are latched and
            # reported against this request rather than escaping the worker
            # scheduler loop and failing every in-flight request.
            failed = Future()
            failed.set_exception(error)
            return failed

    def _read_batched(
        self,
        remote_kv_info: ShmKVTransferInfo | None,
        read_info: list[KVReadInfo],
    ) -> None:
        if not read_info:
            return None
        if remote_kv_info is None:
            raise RuntimeError("Missing SHM metadata for remote KV cache")
        if remote_kv_info.layout != self._kv_cache.layout:
            raise ValueError(
                f"remote KV layout {remote_kv_info.layout} != "
                f"local layout {self._kv_cache.layout}"
            )

        packed = torch.load(
            remote_kv_info.path,
            map_location="cpu",
            weights_only=True,
        )
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
        return None

    def remove_request(self, request_id: str) -> None:
        keys = [key for key in self._published if key[0] == request_id]
        for key in keys:
            _, info = self._published.pop(key)
            try:
                os.unlink(info.path)
            except FileNotFoundError:
                pass

    def owns_transfer_info(
        self,
        transfer_info: Any,
        request_id: str,
        label: str,
    ) -> bool:
        return (
            isinstance(transfer_info, ShmKVTransferInfo)
            and transfer_info.path == self._path(request_id, label)
        )

    def shutdown(self):
        for request_id, _ in list(self._published):
            self.remove_request(request_id)


def make_deployment_kv_shm_dir(
    socket_path_prefix: str,
    dist_init_method: str,
    owner_pid: int | None = None,
) -> str:
    """Create a private SHM directory for one local deployment.

    The conductor PID plus its unique distributed-init endpoint separates
    concurrent servers, even when both use the default worker ids and request
    ids. On startup, directories owned by this uid whose conductor no longer
    exists are swept so killed deployments do not leak tmpfs indefinitely.
    """
    base = os.getenv("MSTAR_KV_SHM_DIR")
    if base is None:
        base = (
            "/dev/shm"
            if os.path.isdir("/dev/shm")
            else "/tmp"
        )
    os.makedirs(base, exist_ok=True)

    uid = os.getuid() if hasattr(os, "getuid") else 0
    owner_pid = os.getppid() if owner_pid is None else owner_pid
    digest = hashlib.sha256(
        f"{socket_path_prefix}\0{dist_init_method}".encode()
    ).hexdigest()[:16]
    prefix = f"mstar_kv_{uid}_"
    path = os.path.join(base, f"{prefix}{owner_pid}_{digest}")

    for entry in os.scandir(base):
        if (
            not entry.name.startswith(prefix)
            or not entry.is_dir(follow_symlinks=False)
            or entry.path == path
        ):
            continue
        suffix = entry.name[len(prefix):]
        pid_text, separator, _ = suffix.partition("_")
        if not separator or not pid_text.isdigit():
            continue
        try:
            os.kill(int(pid_text), 0)
        except ProcessLookupError:
            shutil.rmtree(entry.path, ignore_errors=True)
        except PermissionError:
            continue

    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    return path


@dataclass
class TransferEngineInfo:
    my_entity_id: str
    my_session_id: str
    transfer_engine: TensorTransferEngine
    shm_dir: str | None = None


class KVTransferManager:
    def __init__(
        self, transfer_engine_info: TransferEngineInfo,
        kv_cache: KVCache,
        resource_key: str = "kv",
        needs_remote_transfer: bool = True,
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
            elif needs_remote_transfer:
                if not transfer_engine_info.shm_dir:
                    raise ValueError(
                        "shm_dir is required for shared-memory KV transfer"
                    )
                self._kv_transfer_engine = ShmKVTransferEngine(
                    kv_cache=kv_cache,
                    entity_id=transfer_engine_info.my_entity_id,
                    shm_dir=transfer_engine_info.shm_dir,
                    resource_key=resource_key,
                )
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

    def get_kv_transfer_info(
        self,
        request_id: str | None = None,
        label: str | None = None,
        page_indices: list[int] | None = None,
        seq_len: int | None = None,
    ):
        """Descriptor another process needs to read this cache remotely.
        ``KVCachePool.publish`` stamps it onto every ``SequenceInfo``."""
        return self._kv_transfer_engine.get_kv_transfer_info(
            request_id=request_id,
            label=label,
            page_indices=page_indices,
            seq_len=seq_len,
        )

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
