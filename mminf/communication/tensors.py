import logging
import os
import platform
import struct
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

from mminf.graph.special_destinations import EMPTY_DESTINATION

try:
    from mooncake.engine import TransferEngine
except Exception as _err:
    MOONCAKE_IMPORT_ERROR = _err
    TransferEngine = None
else:
    MOONCAKE_IMPORT_ERROR = None
import torch

from mminf.communication.communicator import BaseCommunicator, CommProtocol
from mminf.graph.base import GraphEdge, TensorPointerInfo
from mminf.utils.ipc_format import TensorReceived, WorkerMessage, WorkerMessageType

logger = logging.getLogger(__name__)


@dataclass
class FutureAndPointers:
    future: Future | None
    graph_edges: list[GraphEdge]
    request_id: str = ""


@dataclass
class TensorAndReferenceInfo:
    tensor: torch.Tensor
    ref_cnt: int = 0
    persist: bool = False
    mem_registered: bool = False


NameToTensorList = dict[str, list[torch.Tensor]]
UuidToTensorAndRef = dict[str, TensorAndReferenceInfo]

class TensorStore:
    def __init__(self):
        # request ID to {UUID -> tensor}
        self.per_req_tensors: dict[str, UuidToTensorAndRef] = {}

    def get_tensor(self, request_id: str, uuid: str) -> torch.Tensor:
        return self.per_req_tensors[request_id][uuid].tensor

    def put_tensor(self, request_id: str, uuid: str, tensor: torch.Tensor):
        self.per_req_tensors.setdefault(
            request_id, {}
        )[uuid] = TensorAndReferenceInfo(tensor)

    def check_uuid_presence(self, request_id: str, uuid: str):
        return uuid in self.per_req_tensors.get(request_id, {})

    def remove_tensor(self, request_id: str, uuid: str):
        if not self.check_uuid_presence(request_id, uuid):
            return
        del self.per_req_tensors[request_id][uuid]
        if not self.per_req_tensors[request_id]:
            del self.per_req_tensors[request_id]

    def get_all_uuids(self, request_id: str) -> list[str]:
        return list(self.per_req_tensors.get(request_id, {}).keys())

    def can_gc(self, request_id: str, uuid: str)-> bool:
        if not self.check_uuid_presence(request_id, uuid):
            return False
        info = self.per_req_tensors[request_id][uuid]
        return info.ref_cnt <= 0 and not info.persist

    def is_registered(self, request_id: str, uuid: str):
        if not self.check_uuid_presence(request_id, uuid):
            return False
        return self.per_req_tensors[request_id][uuid].mem_registered

    def set_metadata(
        self, request_id: str, uuid: str,
        persist: bool | None = None,
        mem_registered: bool | None = None
    ):
        if not self.check_uuid_presence(request_id, uuid):
            return
        if persist is not None:
            self.per_req_tensors[request_id][uuid].persist = persist
        if mem_registered is not None:
            self.per_req_tensors[request_id][uuid].mem_registered = mem_registered

    def increment_ref(self, request_id: str, uuid: str, n: int=1):
        if not self.check_uuid_presence(request_id, uuid):
            return
        assert n >= 0, f"Tried to increment tensor {uuid} reference by a negative number {n}"
        self.per_req_tensors[request_id][uuid].ref_cnt += n

    def dereference(self, request_id: str, uuid: str, n: int=1):
        if not self.check_uuid_presence(request_id, uuid):
            return
        info = self.per_req_tensors[request_id][uuid]
        info.ref_cnt -= n


# ---------------------------------------------------------------------------
# TensorTransferEngine abstraction
# ---------------------------------------------------------------------------

class TensorTransferEngine(ABC):
    """Abstract interface for low-level memory registration and async reads.

    Wraps the transport-specific engine (Mooncake RDMA, local no-op, etc.)
    so that higher-level code (PagedAllocationManager, TensorCommunicationManager)
    never imports or depends on a specific transport library.
    """

    @abstractmethod
    def register_memory(self, ptr: int, nbytes: int) -> int:
        """Register a memory region for remote access. Returns 0 on success."""
        ...

    @abstractmethod
    def unregister_memory(self, ptr: int) -> int:
        """Unregister a previously registered memory region. Returns 0 on success."""
        ...

    @abstractmethod
    def get_async_reader(self, device) -> "AsyncMooncakeReader | None":
        """Return an async reader for background transfers, or None if not needed."""
        ...

    @abstractmethod
    def get_session_id(self) -> str:
        """Return the session ID for this engine (e.g., 'hostname:port')."""
        ...


# ---------------------------------------------------------------------------
# Mooncake implementation
# ---------------------------------------------------------------------------

@dataclass
class TransferReadInfo:
    source_session_id: str
    local_ptr: int
    remote_ptr: int
    nbytes: int


class AsyncMooncakeReader:
    """Background thread for non-blocking mooncake READ operations.

    Follows SGLang's pattern: caller records CUDA event on default stream,
    submits write task to thread pool. Worker thread waits on event via
    dedicated CUDA stream, then does blocking mooncake PUTs.
    The default stream is never blocked by store writes.
    """

    def __init__(self, engine, device, max_workers: int = 3, max_batch_size=500):
        self._engine = engine
        self.max_batch_size = max_batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: list[Future] = []
        if device != "cpu":
            self._copy_stream = torch.cuda.Stream(device=device)
        else:
            self._copy_stream = torch.cuda.Stream()

    def submit(self, read_info: list[TransferReadInfo]) -> Future:
        """Non-blocking: enqueue a batch of READs.

        Records a CUDA event on the current stream to ensure GPU data
        is ready before the background thread reads it.
        """
        if not read_info:
            return
        event = torch.cuda.current_stream().record_event()
        future = self._executor.submit(self._do_read, read_info, event)
        self._pending.append(future)
        # Prune completed futures to avoid unbounded growth
        self._pending = [f for f in self._pending if not f.done()]
        return future

    def _do_read(self, read_info: list["TransferReadInfo"], event: torch.cuda.Event):
        """Worker thread: wait for GPU data via CUDA event, then PUT."""
        self._copy_stream.wait_event(event)
        self._copy_stream.synchronize()

        # group read_info by session id for batch read
        grouped_read = {}
        for info in read_info:
            grouped_read.setdefault(info.source_session_id, []).append(info)

        for (session_id, infos) in grouped_read.items():
            for start in range(0, len(infos), self.max_batch_size):
                end = min(start + self.max_batch_size, len(infos))

                status = self._engine.batch_transfer_sync_read(
                    session_id,
                    [infos[i].local_ptr for i in range(start, end)],
                    [infos[i].remote_ptr for i in range(start, end)],
                    [infos[i].nbytes for i in range(start, end)],
                )
                if status < 0:
                    raise RuntimeError(f"Mooncake read failed. Status: {status}")

    def wait_all(self):
        """Block until all pending writes complete. Re-raises exceptions."""
        for f in self._pending:
            f.result()
        self._pending.clear()

    def shutdown(self):
        """Wait for pending writes and shut down the thread pool."""
        self.wait_all()
        self._executor.shutdown(wait=True)


class MooncakeTransferEngine(TensorTransferEngine):
    """Wraps mooncake.engine.TransferEngine for RDMA or TCP transport."""

    def __init__(
        self,
        hostname: str,
        protocol: CommProtocol,
        metadata_server: str = "P2PHANDSHAKE",
        tcp_transfer_device: str = "",
    ):
        if TransferEngine is None:
            detail = (
                f"{type(MOONCAKE_IMPORT_ERROR).__name__}: "
                f"{MOONCAKE_IMPORT_ERROR}"
                if MOONCAKE_IMPORT_ERROR is not None
                else "unknown import failure"
            )
            raise RuntimeError(
                "Mooncake TransferEngine is required for RDMA/TCP protocol. "
                f"Failed to load mooncake: {detail}. "
                "Install mooncake-transfer-engine or use SHM protocol."
            )

        if protocol == CommProtocol.RDMA:
            transfer_device = ""
        elif protocol == CommProtocol.TCP:
            transfer_device = tcp_transfer_device
        else:
            raise NotImplementedError(f"Unknown protocol {protocol} for mooncake")

        self._engine = TransferEngine()
        self._engine.initialize(
            hostname,
            metadata_server,
            protocol.value.lower(),
            transfer_device,
        )
        self._session_id = f"{hostname}:{self._engine.get_rpc_port()}"

    def register_memory(self, ptr: int, nbytes: int) -> int:
        return self._engine.register_memory(ptr, nbytes)

    def unregister_memory(self, ptr: int) -> int:
        return self._engine.unregister_memory(ptr)

    def get_async_reader(self, device) -> AsyncMooncakeReader:
        return AsyncMooncakeReader(self._engine, device=device)

    def get_session_id(self) -> str:
        return self._session_id


class LocalTransferEngine(TensorTransferEngine):
    """No-op engine for SHM / single-node — data is already in local GPU memory."""

    def __init__(self, hostname: str):
        self._session_id = hostname

    def register_memory(self, ptr: int, nbytes: int) -> int:
        return 0  # no-op

    def unregister_memory(self, ptr: int) -> int:
        return 0  # no-op

    def get_async_reader(self, device) -> None:
        return None  # no remote reads needed

    def get_session_id(self) -> str:
        return self._session_id


# ---------------------------------------------------------------------------
# TensorCommunicationManager base class (Comment 1: shared methods)
# ---------------------------------------------------------------------------

class TensorCommunicationManager(ABC):
    """Base class for inter-worker tensor transport.

    Holds common attributes and shared method implementations. Subclasses
    only need to override ``__init__``, ``register_for_send``,
    ``start_read_tensors``, and ``_cleanup_by_uuid``.
    """

    def __init__(
        self,
        my_entity_id: str,
        my_session_id: str,
        device: str,
        communicator: BaseCommunicator,
        transfer_engine: TensorTransferEngine,
    ):
        self.my_entity_id = my_entity_id
        self.my_session_id = my_session_id
        self.device = device
        self.communicator = communicator
        self.transfer_engine = transfer_engine
        self.tensor_store = TensorStore()
        self.pending: list[FutureAndPointers] = []

    # ---- shared: store ----

    def store_and_return_tensor_info(
        self, request_id: str, tensors: NameToTensorList,
    ) -> dict[str, list[TensorPointerInfo]]:
        if torch.cuda.is_available():
            torch.cuda.default_stream().synchronize()
        tensor_info: dict[str, list[TensorPointerInfo]] = {}
        for name, tensor_list in tensors.items():
            tensor_info[name] = []
            for tensor in tensor_list:
                tensor_uuid = str(uuid4())
                self.tensor_store.put_tensor(
                    request_id=request_id, uuid=tensor_uuid, tensor=tensor,
                )
                logger.debug("Storing tensor name %s uuid %s", name, tensor_uuid)
                tensor_info[name].append(TensorPointerInfo(
                    dims=tensor.shape,
                    dtype=tensor.dtype,
                    stride=tensor.stride(),
                    nbytes=tensor.nbytes,
                    address=tensor.data_ptr(),
                    uuid=tensor_uuid,
                    source_session_id=self.my_session_id,
                    source_entity=self.my_entity_id,
                ))
        return tensor_info

    def store_and_populate_graph_edges(
        self, request_id: str, tensors: NameToTensorList,
        graph_edges: list[GraphEdge],
    ):
        name_to_graph_edges: dict[str, list[GraphEdge]] = {}
        for edge in graph_edges:
            name_to_graph_edges.setdefault(edge.name, []).append(edge)

        graph_node_info = self.store_and_return_tensor_info(
            request_id=request_id, tensors=tensors,
        )
        for name in tensors:
            logger.debug(
                "Storing tensor %s (uuids %s) for nodes %s",
                name, str([info.uuid for info in graph_node_info[name]]),
                str([edge.name for edge in name_to_graph_edges.get(name, [])])
            )
            edges = name_to_graph_edges.get(name, [])
            for info in graph_node_info[name]:
                self.tensor_store.increment_ref(
                    request_id, info.uuid, n=len([
                        e for e in edges if e.next_node != EMPTY_DESTINATION
                    ])
                )
            for edge in edges:
                edge.tensor_info = graph_node_info[name]
        return graph_node_info

    # ---- abstract: transport-specific ----

    @abstractmethod
    def register_for_send(self, request_id: str, uuids: list[str]):
        ...

    @abstractmethod
    def start_read_tensors(self, request_id: str, graph_edges: list[GraphEdge]):
        ...

    @abstractmethod
    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        ...

    # ---- shared: polling & ACKs ----

    def _collect_and_send_acks(
        self, request_id: str, graph_edges: list[GraphEdge],
    ):
        acks: dict[str, dict[str, int]] = {}
        for edge in graph_edges:
            for info in edge.tensor_info:
                if info.source_entity not in acks:
                    acks[info.source_entity] = {}
                acks[info.source_entity][info.uuid] = acks[info.source_entity].get(
                    info.uuid, 0) + 1
        for source_entity, tensors in acks.items():
            if source_entity == self.my_entity_id:
                continue
            self.communicator.send(
                source_entity,
                WorkerMessage(
                    message_type=WorkerMessageType.TENSOR_RECEIVED,
                    body=TensorReceived(
                        request_id=request_id,
                        successful_tensors=tensors,
                        failed_tensor_ids=[],
                    ),
                ),
            )

    def get_ready_tensors(self) -> dict[str, list[GraphEdge]]:
        ready: dict[str, list[GraphEdge]] = {}
        still_pending = []
        for ep in self.pending:
            if ep.future is None or ep.future.done():
                if ep.future is not None:
                    ep.future.result()
                for edge in ep.graph_edges:
                    ready.setdefault(ep.request_id, []).append(edge)
                    logger.debug(
                        "Finished reading in %d tensors %s for graph node %s",
                        len(edge.tensor_info), edge.name, edge.next_node
                    )
            else:
                still_pending.append(ep)
        self.pending = still_pending

        for req_id, edges in ready.items():
            self._collect_and_send_acks(req_id, edges)
            for edge in edges:
                for info in edge.tensor_info:
                    self.tensor_store.dereference(req_id, info.uuid, 1)
        return ready

    # ---- shared: TensorStore delegation ----

    def get_tensor(self, request_id: str, uuid: str) -> torch.Tensor:
        return self.tensor_store.get_tensor(request_id=request_id, uuid=uuid)

    def set_persist(self, request_id: str, uuid: str, persist: bool):
        self.tensor_store.set_metadata(request_id, uuid, persist=persist)
        if self.tensor_store.can_gc(request_id, uuid):
            self._cleanup_by_uuid(request_id, uuid)

    def dereference(self, request_id: str, uuid: str, n: int = 1):
        self.tensor_store.dereference(request_id, uuid, n=n)
        if self.tensor_store.can_gc(request_id, uuid):
            self._cleanup_by_uuid(request_id, uuid)

    def increment_ref(self, request_id: str, uuid: str, n: int = 1):
        self.tensor_store.increment_ref(request_id, uuid, n=n)

    def cleanup_request(self, request_id: str):
        for uuid in self.tensor_store.get_all_uuids(request_id):
            self.tensor_store.set_metadata(request_id, uuid, persist=False)
            if not self.tensor_store.can_gc(request_id, uuid):
                logger.warning(
                    "Deferring cleanup of tensor uuid %s "
                    "(awaiting TENSOR_RECEIVED ACK)", uuid
                )
                continue
            self._cleanup_by_uuid(request_id, uuid)

        self._collect_and_send_acks(
            request_id,
            sum([ep.graph_edges for ep in self.pending if ep.request_id == request_id], start=[]),
        )
        self.pending = [ep for ep in self.pending if ep.request_id != request_id]


# ---------------------------------------------------------------------------
# MooncakeCommunicationManager
# ---------------------------------------------------------------------------

class MooncakeCommunicationManager(TensorCommunicationManager):
    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        device: str,
        communicator: BaseCommunicator,
        protocol: CommProtocol = CommProtocol.RDMA,
        metadata_server: str = "P2PHANDSHAKE",
        tcp_transfer_device: str = "",
    ):
        engine = MooncakeTransferEngine(
            hostname=hostname,
            protocol=protocol,
            metadata_server=metadata_server,
            tcp_transfer_device=tcp_transfer_device,
        )
        super().__init__(
            my_entity_id=my_entity_id,
            my_session_id=engine.get_session_id(),
            device=device,
            communicator=communicator,
            transfer_engine=engine,
        )
        self.protocol = protocol
        self._async_reader = AsyncMooncakeReader(
            engine._engine, device=device
        )

    def register_for_send(self, request_id, uuids):
        torch.cuda.default_stream().synchronize()
        for uuid in uuids:
            if self.protocol == CommProtocol.RDMA:
                if self.tensor_store.is_registered(request_id, uuid):
                    continue
                logger.debug("Registering %s for send", uuid)
                tensor = self.tensor_store.get_tensor(
                    request_id=request_id, uuid=uuid
                )
                ret_value = self.transfer_engine.register_memory(
                    tensor.data_ptr(), tensor.nbytes
                )
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake memory registration failed for request id {request_id}, uuid {uuid}."
                    )
            self.tensor_store.set_metadata(
                request_id, uuid, mem_registered=True
            )

    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        logger.debug("Deleting tensor uuid %s", uuid)
        if not self.tensor_store.check_uuid_presence(request_id, uuid):
            logger.warning("Trying to cleanup tensor %s, but uuid not found", uuid)
            return
        if self.protocol == CommProtocol.RDMA \
                and self.tensor_store.is_registered(request_id, uuid):
            ret_value = self.transfer_engine.unregister_memory(
                self.tensor_store.get_tensor(request_id, uuid).data_ptr()
            )
            if ret_value != 0:
                raise RuntimeError("Mooncake memory unregistration failed.")
        self.tensor_store.remove_tensor(request_id, uuid)

    def start_read_tensors(
        self, request_id: str, graph_edges: list[GraphEdge],
    ):
        for graph_edge in graph_edges:
            if len(graph_edge.tensor_info) == 0:
                continue

            logger.debug(
                "Starting to read in %d tensors %s for graph node %s",
                len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node
            )

            read_info = []
            for info in graph_edge.tensor_info:
                if info.source_entity == self.my_entity_id or self.tensor_store.check_uuid_presence(
                    request_id, info.uuid
                ):
                    self.tensor_store.increment_ref(
                        request_id, info.uuid, 1
                    )
                    continue
                buffer = torch.empty(
                    info.dims, dtype=info.dtype, device=self.device
                ).as_strided(info.dims, stride=info.stride)
                self.tensor_store.put_tensor(
                    request_id=request_id, uuid=info.uuid, tensor=buffer
                )
                self.tensor_store.set_metadata(
                    request_id, info.uuid, mem_registered=True
                )
                # +1 for transit (released by get_ready_tensors)
                # +1 for graph-node usage (released by _cleanup_consumed_inputs)
                self.tensor_store.increment_ref(
                    request_id, info.uuid, 2
                )

                if self.protocol == CommProtocol.RDMA:
                    self.transfer_engine.register_memory(buffer.data_ptr(), info.nbytes)

                read_info.append(TransferReadInfo(
                    source_session_id=info.source_session_id,
                    local_ptr=buffer.data_ptr(),
                    remote_ptr=info.address,
                    nbytes=info.nbytes,
                ))
                logger.debug("Started transfer read for uuid %s", info.uuid)
            fut = self._async_reader.submit(read_info)
            self.pending.append(
                FutureAndPointers(
                    future=fut, graph_edges=[graph_edge],
                    request_id=request_id
                )
            )


# ---------------------------------------------------------------------------
# Shared-memory tensor serialization helpers
# ---------------------------------------------------------------------------

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "u8",
    torch.bool: "bool",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}

# bfloat16 has no numpy equivalent — we view as uint16 for raw serialization.
_BF16_VIEW_DTYPE = torch.uint16


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes: header + contiguous raw data."""
    t = tensor.detach().contiguous().cpu()
    dtype_tag = _DTYPE_TO_STR[t.dtype].encode("ascii")

    # Header: ndim (u32) | dtype_len (u32) | dtype_tag | shape (ndim × i64) | stride (ndim × i64)
    hdr = struct.pack("<II", t.ndim, len(dtype_tag)) + dtype_tag
    for s in t.shape:
        hdr += struct.pack("<q", s)
    for s in t.stride():
        hdr += struct.pack("<q", s)

    # Raw data — bfloat16 must be viewed as uint16 for numpy conversion.
    if t.dtype == torch.bfloat16:
        raw = t.view(_BF16_VIEW_DTYPE).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()

    return hdr + raw


def _deserialize_tensor(data: bytes | memoryview, device: str) -> torch.Tensor:
    """Reconstruct a tensor from bytes produced by ``_serialize_tensor``."""
    if isinstance(data, memoryview):
        data = bytes(data)
    off = 0
    ndim, dtype_len = struct.unpack_from("<II", data, off); off += 8
    dtype_tag = data[off:off + dtype_len].decode("ascii"); off += dtype_len

    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<q", data, off)[0]); off += 8
    stride = []
    for _ in range(ndim):
        stride.append(struct.unpack_from("<q", data, off)[0]); off += 8

    dtype = _STR_TO_DTYPE[dtype_tag]
    raw = data[off:]

    if len(raw) == 0:
        t = torch.empty(shape, dtype=dtype)
    elif dtype == torch.bfloat16:
        t = torch.frombuffer(bytearray(raw), dtype=_BF16_VIEW_DTYPE).view(torch.bfloat16).reshape(shape)
    else:
        t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    if device != "cpu":
        t = t.to(device)
    return t


def _default_shm_dir() -> str:
    """Return the default shared-memory directory for the current platform."""
    if platform.system() == "Linux" and os.path.isdir("/dev/shm"):
        return "/dev/shm"
    return "/tmp/mminf_shm"


# ---------------------------------------------------------------------------
# SharedMemoryCommunicationManager
# ---------------------------------------------------------------------------

class SharedMemoryCommunicationManager(TensorCommunicationManager):
    """Tensor transport via file I/O to a tmpfs directory (``/dev/shm``)."""

    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        device: str,
        communicator: BaseCommunicator,
        shm_dir: str | None = None,
    ):
        engine = LocalTransferEngine(hostname=hostname)
        super().__init__(
            my_entity_id=my_entity_id,
            my_session_id=engine.get_session_id(),
            device=device,
            communicator=communicator,
            transfer_engine=engine,
        )

        self.shm_dir = shm_dir or _default_shm_dir()
        os.makedirs(self.shm_dir, exist_ok=True)

        # uuid → file path for sender-side cleanup
        self._shm_files: dict[str, str] = {}

    def _shm_path(self, entity_id: str, uuid: str) -> str:
        return os.path.join(self.shm_dir, f"mminf_{entity_id}_{uuid}")

    def register_for_send(self, request_id: str, uuids: list[str]):
        if torch.cuda.is_available():
            torch.cuda.default_stream().synchronize()
        for uuid in uuids:
            if self.tensor_store.is_registered(request_id, uuid):
                continue
            tensor = self.tensor_store.get_tensor(request_id, uuid)
            data = _serialize_tensor(tensor)
            path = self._shm_path(self.my_entity_id, uuid)
            with open(path, "wb") as f:
                f.write(data)
            self._shm_files[uuid] = path
            self.tensor_store.set_metadata(request_id, uuid, mem_registered=True)
            logger.debug("SHM: wrote tensor %s to %s (%d bytes)", uuid, path, len(data))

    def start_read_tensors(
        self, request_id: str, graph_edges: list[GraphEdge],
    ):
        for graph_edge in graph_edges:
            if len(graph_edge.tensor_info) == 0:
                continue
            logger.debug(
                "SHM: starting read of %d tensors %s for graph node %s",
                len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node,
            )
            for info in graph_edge.tensor_info:
                if info.source_entity == self.my_entity_id or \
                   self.tensor_store.check_uuid_presence(request_id, info.uuid):
                    self.tensor_store.increment_ref(request_id, info.uuid, 1)
                    continue
                path = self._shm_path(info.source_entity, info.uuid)
                with open(path, "rb") as f:
                    data = f.read()
                tensor = _deserialize_tensor(data, self.device)
                self.tensor_store.put_tensor(request_id, info.uuid, tensor)
                self.tensor_store.set_metadata(request_id, info.uuid, mem_registered=False)
                # +1 for transit (released by get_ready_tensors)
                # +1 for graph-node usage (released by _cleanup_consumed_inputs)
                self.tensor_store.increment_ref(request_id, info.uuid, 2)
                logger.debug("SHM: read tensor %s from %s", info.uuid, path)
            self.pending.append(
                FutureAndPointers(
                    future=None, graph_edges=[graph_edge],
                    request_id=request_id,
                )
            )

    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        logger.debug("SHM: cleaning up tensor uuid %s", uuid)
        if not self.tensor_store.check_uuid_presence(request_id, uuid):
            logger.warning("SHM: cleanup tensor %s, uuid not found", uuid)
            return
        if uuid in self._shm_files:
            path = self._shm_files.pop(uuid)
            try:
                os.unlink(path)
                logger.debug("SHM: unlinked %s", path)
            except FileNotFoundError:
                pass
        self.tensor_store.remove_tensor(request_id, uuid)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tensor_communication_manager(
    protocol: CommProtocol,
    my_entity_id: str,
    hostname: str,
    device: str,
    communicator: BaseCommunicator,
    metadata_server: str = "P2PHANDSHAKE",
    tcp_transfer_device: str = "",
    shm_dir: str | None = None,
) -> TensorCommunicationManager:
    """Select tensor transport backend based on protocol."""
    if protocol == CommProtocol.SHM:
        return SharedMemoryCommunicationManager(
            my_entity_id=my_entity_id,
            hostname=hostname,
            device=device,
            communicator=communicator,
            shm_dir=shm_dir,
        )
    return MooncakeCommunicationManager(
        my_entity_id=my_entity_id,
        hostname=hostname,
        device=device,
        communicator=communicator,
        protocol=protocol,
        metadata_server=metadata_server,
        tcp_transfer_device=tcp_transfer_device,
    )
