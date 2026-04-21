import logging
import os
import platform
import struct
import time as _time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from mminf.communication.communicator import BaseCommunicator, CommProtocol
from mminf.graph.base import GraphEdge, TensorPointerInfo
from mminf.graph.special_destinations import EMPTY_DESTINATION, SPECIAL_DESTINATIONS
from mminf.utils.ipc_format import TensorReceived, WorkerMessage, WorkerMessageType

try:
    from mooncake.engine import TransferEngine
except Exception as _err:
    MOONCAKE_IMPORT_ERROR = _err
    TransferEngine = None
else:
    MOONCAKE_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeSpec:
    """Declarative description of one NVSHMEM worker-to-worker edge captured inside a CUDA graph.

    Built by the Conductor during graph-walk registration and passed to each
    worker's ``NVSHMEMCommunicationManager.init_edges``. The Conductor emits
    one ``EdgeSpec`` per unique (producer_rank, consumer_rank, edge_id) tuple
    that resolves to NVSHMEM transport; each participating rank receives the
    same list, so ``init_edges`` is deterministic across ranks.

    ``max_bytes`` is the upper bound on payload size this edge will carry.
    The manager aligns up to 128 B and allocates one symmetric slot of that
    size per edge per rank that participates (producer or consumer).

    Self-edges (producer_rank == consumer_rank), api_server edges, and any
    edge resolving to Mooncake are NOT emitted as EdgeSpecs — they ride the
    legacy side-stream fallback path.
    """
    edge_id: int
    producer_rank: int
    consumer_rank: int
    max_bytes: int


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


class NVSHMEMTransferEngine(TensorTransferEngine):
    """No-op TensorTransferEngine for NVSHMEM-mode workers.

    NVSHMEM workers don't transfer KV cache pages cross-worker through
    Mooncake — kv_store.py's path is bypassed by the polymorphic no-ops
    below. Implementing the abstract contract this way lets NVSHMEM mode
    construct a manager via TensorCommunicationManager.__init__ without
    special-casing the engine, and lets PagedAllocationManager work
    against an NVSHMEM-mode manager unchanged.
    """

    def __init__(self, my_entity_id: str):
        self._session_id = f"nvshmem:{my_entity_id}"

    def register_memory(self, ptr: int, nbytes: int) -> int:
        return 0  # no-op

    def unregister_memory(self, ptr: int) -> int:
        return 0  # no-op

    def get_async_reader(self, device) -> None:
        return None  # NVSHMEM mode does not use the AsyncMooncakeReader path

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
    def register_for_send(
        self, request_id: str, uuids: list[str],
        skip_cuda_sync: bool = False,
    ):
        """Mark these uuids ready for remote consumers to RDMA-read.

        ``skip_cuda_sync=True`` skips the default-stream sync this call normally
        issues to ensure the source tensors' writes are visible before their
        addresses are shared with peers. Callers must have already synced on
        their own (e.g. before a batched loop) — meant to cut N serialized
        syncs to 1 when registering many uuids in a row.
        """
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
                        sender_entity_id=self.my_entity_id,
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

    def register_for_send(self, request_id, uuids, skip_cuda_sync=False):
        if not skip_cuda_sync:
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
    ndim, dtype_len = struct.unpack_from("<II", data, off)
    off += 8
    dtype_tag = data[off:off + dtype_len].decode("ascii")
    off += dtype_len

    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<q", data, off)[0])
        off += 8
    stride = []
    for _ in range(ndim):
        stride.append(struct.unpack_from("<q", data, off)[0])
        off += 8

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

    def register_for_send(
        self, request_id: str, uuids: list[str],
        skip_cuda_sync: bool = False,
    ):
        if not skip_cuda_sync and torch.cuda.is_available():
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
    """Select tensor transport backend based on protocol.

    NVSHMEM is intentionally not dispatched here — NVSHMEM workers construct
    NVSHMEMCommunicationManager directly in worker.py because it needs the
    rank/world_size/dist.ProcessGroup that the factory does not have.
    """
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


# ---------------------------------------------------------------------------
# NVSHMEM staging pack/unpack helpers (eager fallback path)
# ---------------------------------------------------------------------------

_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
    torch.int32: 3, torch.int64: 4, torch.uint8: 5, torch.bool: 6,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}
_HEADER_OVERHEAD = 4096  # bytes; generous fixed header budget


def _compute_packed_size(tensors: list[torch.Tensor]) -> int:
    return _HEADER_OVERHEAD + sum(t.nbytes for t in tensors)


def _compute_packed_size_from_infos(infos: list[TensorPointerInfo]) -> int:
    """Compute packed size for consumer side from TensorPointerInfo list."""
    return _HEADER_OVERHEAD + sum(info.nbytes for info in infos)


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse dtype string like 'torch.float32' to torch.dtype.

    Only the dtypes in _DTYPE_TO_CODE are supported; dtype strings arrive from
    a peer rank and must not be evaluated as arbitrary code.
    """
    mapping = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }
    if dtype_str in mapping:
        return mapping[dtype_str]
    raise ValueError(
        f"Unsupported dtype string {dtype_str!r}. "
        f"Supported: {list(mapping)}"
    )


def _pack_tensors_into_slot(
    tensors: list[torch.Tensor], slot_view: torch.Tensor
) -> None:
    """
    Write header + contiguous tensor data into slot_view (uint8).
    Called on the transfer stream — ops are CUDA async.
    """
    n = len(tensors)
    header = bytearray()
    header += struct.pack("<I", n)  # 4 bytes: N tensors
    offsets = []
    cur_offset = _HEADER_OVERHEAD  # data starts after header
    for t in tensors:
        ct = t.contiguous()
        ndims = ct.ndim
        if ct.dtype not in _DTYPE_TO_CODE:
            raise ValueError(
                f"Unsupported dtype {ct.dtype} for NVSHMEM staging. "
                f"Supported: {list(_DTYPE_TO_CODE.keys())}"
            )
        dtype_code = _DTYPE_TO_CODE[ct.dtype]
        header += struct.pack("<IIBB2x", cur_offset, ct.nbytes, ndims, dtype_code)
        header += struct.pack(f"<{ndims}q", *ct.shape)
        header += struct.pack(f"<{ndims}q", *ct.stride())
        offsets.append((cur_offset, ct))
        cur_offset += ct.nbytes

    # Copy header into slot (sync copy from CPU bytes to GPU tensor).
    header_bytes = bytes(header)
    header_tensor = torch.frombuffer(bytearray(header_bytes), dtype=torch.uint8)
    slot_view[:len(header_tensor)].copy_(header_tensor.to(slot_view.device))
    # Copy each tensor's data into its offset.
    # Flatten to 1D uint8 before copy to handle multi-dimensional tensors.
    for off, ct in offsets:
        flat_bytes = ct.contiguous().view(-1).view(torch.uint8)
        slot_view[off : off + ct.nbytes].copy_(flat_bytes)


def _unpack_tensors_from_slot(
    slot_view: torch.Tensor, dst_tensors: list[torch.Tensor]
) -> None:
    """
    Read header from slot_view and copy tensor data into dst_tensors.
    Called on the transfer stream after nvshmem_wait_for_signal.
    """
    header_cpu = slot_view[:_HEADER_OVERHEAD].cpu().numpy().tobytes()
    n = struct.unpack_from("<I", header_cpu, 0)[0]
    assert n == len(dst_tensors), (
        f"Header says {n} tensors but got {len(dst_tensors)} dst"
    )
    off = 4
    for _i, dst in enumerate(dst_tensors):
        data_offset, nbytes, ndims, dtype_code = struct.unpack_from("<IIBB2x", header_cpu, off)
        off += 12
        off += ndims * 8 * 2  # skip dims + strides (we already have dst shape)
        src_view = slot_view[data_offset : data_offset + nbytes]
        # Flatten dst to 1D uint8 before copy to handle multi-dimensional tensors.
        dst.contiguous().view(-1).view(torch.uint8).copy_(src_view)


# ---------------------------------------------------------------------------
# NVSHMEMStagingPool (eager fallback path)
# ---------------------------------------------------------------------------


@dataclass
class SlotDescriptor:
    """Tracks one in-flight staging slot."""
    slot_idx: int          # global slot index in the symmetric heap
    producer_rank: int
    consumer_rank: int
    sig_val: int
    nbytes: int            # actual payload bytes used in this transfer


class NVSHMEMStagingPool:
    """
    Manages the symmetric staging buffer and signal pad.

    Layout: ``world_size * num_slots_per_producer`` slots in the symmetric
    heap, each ``max_slot_bytes`` wide. Slot indices are encoded as::

        slot_idx    = producer_rank * num_slots_per_producer + slot_offset
        byte_offset = slot_idx * max_slot_bytes
        sig_pad     = sig_pad[slot_idx * 2 : slot_idx * 2 + 1]   # stride-2

    Stride-2 sig pad indexing satisfies the 8-byte alignment requirement of
    cuStreamWriteValue64 (used internally by nvshmem_put_with_signal).

    Concurrency: each producer rank can have up to ``num_slots_per_producer``
    in-flight transfers simultaneously. Slots in flight are tracked by
    ``slot_idx`` in ``_in_use``. The producer maintains a per-producer free
    pool of slot offsets in ``_free_slots[producer_rank]``.

    A single slot is held until the consumer sends back a ``TENSOR_RECEIVED``
    ACK over ZMQ; the manager then calls ``free_slot(slot_idx)``.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        group_name: str,
        max_slot_bytes: int,
        num_slots_per_producer: int = 8,
    ) -> None:
        if num_slots_per_producer < 1:
            raise ValueError(
                f"num_slots_per_producer must be >= 1, got {num_slots_per_producer}"
            )
        self.rank = rank
        self.world_size = world_size
        self.max_slot_bytes = max_slot_bytes
        self.num_slots_per_producer = num_slots_per_producer
        self.device = device

        total_slots = world_size * num_slots_per_producer
        total_bytes = total_slots * max_slot_bytes
        self.stage_buf: torch.Tensor = symm_mem.empty(
            total_bytes, dtype=torch.uint8, device=device
        )
        self.hdl = symm_mem.rendezvous(self.stage_buf, group=group_name)
        raw_pad = self.hdl.get_signal_pad(rank)
        # Stride-2 (2 uint32 per slot = 8 bytes) for cuStreamWriteValue64
        # alignment. One sig pad element per slot, indexed by the slot's
        # global slot_idx.
        _needed = total_slots * 2
        if raw_pad.numel() < _needed:
            raise RuntimeError(
                f"sig_pad.numel()={raw_pad.numel()} < {_needed} "
                f"(total_slots={total_slots} * stride=2); "
                "cannot assign 8-byte-aligned signal elements per slot. "
                "Increase NVSHMEM_SYMMETRIC_SIZE or reduce slot count."
            )
        self.sig_pad: torch.Tensor = raw_pad

        # Per-producer free pool of slot offsets [0, num_slots_per_producer).
        self._free_slots: dict[int, list[int]] = {
            r: list(range(num_slots_per_producer)) for r in range(world_size)
        }
        # Globally indexed by slot_idx → SlotDescriptor.
        self._in_use: dict[int, SlotDescriptor] = {}
        # Per-slot signal value counter (monotonic; non-zero so consumer
        # never confuses 'sig set' with 'sig zeroed').
        self._sig_counters: list[int] = [1] * total_slots

    def _global_slot_idx(self, producer_rank: int, slot_offset: int) -> int:
        return producer_rank * self.num_slots_per_producer + slot_offset

    def has_free_slot(self, producer_rank: int) -> bool:
        return len(self._free_slots[producer_rank]) > 0

    # Backward-compat shim retained so callers / tests that ask "is this
    # producer's slot pool empty?" still work. Returns False once any slot
    # is in flight.
    def is_slot_free(self, producer_rank: int) -> bool:
        return self.has_free_slot(producer_rank)

    def alloc_slot(
        self, producer_rank: int, consumer_rank: int, nbytes: int
    ) -> SlotDescriptor:
        """
        Allocate the next free slot for producer → consumer transfer.
        Caller MUST verify ``has_free_slot(producer_rank)`` before calling.
        Raises if nbytes > max_slot_bytes or no free slot is available.
        """
        if nbytes > self.max_slot_bytes:
            raise ValueError(
                f"Transfer payload {nbytes} B exceeds max_slot_bytes "
                f"{self.max_slot_bytes} B. Increase NVSHMEMCommunicationManager "
                "max_slot_bytes at construction time."
            )
        if not self._free_slots[producer_rank]:
            raise RuntimeError(
                f"No free slot for producer_rank={producer_rank}; all "
                f"{self.num_slots_per_producer} slots in flight. Caller must "
                "wait for an ACK or increase num_slots_per_producer."
            )
        slot_offset = self._free_slots[producer_rank].pop(0)
        slot_idx = self._global_slot_idx(producer_rank, slot_offset)
        sig_val = self._sig_counters[slot_idx]
        self._sig_counters[slot_idx] += 1
        desc = SlotDescriptor(
            slot_idx=slot_idx,
            producer_rank=producer_rank,
            consumer_rank=consumer_rank,
            sig_val=sig_val,
            nbytes=nbytes,
        )
        self._in_use[slot_idx] = desc
        return desc

    def free_slot(self, slot_idx: int) -> None:
        """Return a slot to its producer's free pool. Idempotent."""
        desc = self._in_use.pop(slot_idx, None)
        if desc is None:
            return
        slot_offset = slot_idx % self.num_slots_per_producer
        producer_rank = slot_idx // self.num_slots_per_producer
        free_list = self._free_slots[producer_rank]
        if slot_offset not in free_list:
            free_list.append(slot_offset)

    def get_slot_view(self, slot_idx: int, nbytes: int) -> torch.Tensor:
        """Byte-view into the staging slot for the given slot_idx."""
        start = slot_idx * self.max_slot_bytes
        return self.stage_buf[start : start + nbytes]

    def get_sig_pad_element(self, slot_idx: int) -> torch.Tensor:
        """Single-element view into sig_pad for a given slot_idx.

        Uses a stride-2 layout (2 uint32 slots per slot) so that each
        signal element is 8-byte aligned, satisfying the
        cuStreamWriteValue64 alignment requirement used internally by
        nvshmem_put_with_signal.
        """
        idx = slot_idx * 2
        return self.sig_pad[idx : idx + 1]


# ---------------------------------------------------------------------------
# _NVSHMEMPendingTransfer
# ---------------------------------------------------------------------------


@dataclass
class _NVSHMEMPendingTransfer:
    request_id: str
    graph_edges: list[GraphEdge]
    completion_event: torch.cuda.Event
    producer_rank: int  # needed to match ACK back to slot


# ---------------------------------------------------------------------------
# Captured worker-to-worker NVSHMEM transport infrastructure
# (per §3–§5 of the captured-NVSHMEM design doc)
# ---------------------------------------------------------------------------


# In-graph signal values for the captured ACK cycle (both set-to-1 per §3.1).
_NVSHMEM_DATA_VAL: int = 1
_NVSHMEM_ACK_VAL: int = 1
# 128-byte alignment for per-edge slot sizing (§4.1).
_NVSHMEM_SLOT_ALIGN: int = 128


def _align_up_slot_bytes(n: int, multiple: int = _NVSHMEM_SLOT_ALIGN) -> int:
    if n <= 0:
        return multiple
    return (n + multiple - 1) // multiple * multiple


class _CapturedTransportInfra:
    """Per-manager symmetric heap + pad bookkeeping for captured NVSHMEM edges.

    Owns three symmetric allocations, each rendezvoused once across the NVSHMEM
    process group:

      * ``slots_buf``  — contiguous uint8 heap of length
        ``sum(aligned_sizes)``. Every rank sees the same size; edge `e`
        occupies ``slots_buf[edge_byte_offset[e] : +aligned_sizes[e]]``.
        Signal pad on this allocation carries the producer→consumer data
        signal (``data_pad[e]``).
      * ``ack_src``    — 1-element int32 heap used only as a source tensor
        for the consumer's ack ``put_with_signal`` call. Signal pad on this
        allocation carries the consumer→producer ack signal (``ack_pad[e]``).

    Pad slots use stride-2 indexing per the Defect-3 fix (cuStreamWriteValue64
    needs 8-byte alignment over a torch.uint32 pad).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        group_name: str,
        edge_specs: list[EdgeSpec],
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.group_name = group_name

        # Deterministic ordering: sort by edge_id so every rank sees the same
        # per-edge offsets / pad indices (required for symmetric rendezvous).
        specs = sorted(edge_specs, key=lambda s: s.edge_id)
        if not specs:
            raise ValueError(
                "_CapturedTransportInfra: edge_specs is empty; init_edges expects at "
                "least one NVSHMEM worker-to-worker edge"
            )
        if len({s.edge_id for s in specs}) != len(specs):
            raise ValueError(
                "_CapturedTransportInfra: edge_specs contains duplicate edge_ids"
            )
        for s in specs:
            if s.edge_id < 0:
                raise ValueError(
                    f"_CapturedTransportInfra: edge_id must be non-negative, got {s.edge_id}"
                )
            if s.producer_rank == s.consumer_rank:
                raise ValueError(
                    f"_CapturedTransportInfra: self-edge (producer==consumer={s.producer_rank}) "
                    f"on edge_id={s.edge_id}; self-edges must fall through to Mooncake"
                )
            if s.max_bytes <= 0:
                raise ValueError(
                    f"_CapturedTransportInfra: edge_id={s.edge_id} has max_bytes={s.max_bytes}; "
                    "must be positive"
                )

        self.edge_specs: dict[int, EdgeSpec] = {s.edge_id: s for s in specs}
        self.edge_ids: list[int] = [s.edge_id for s in specs]
        self.aligned_sizes: dict[int, int] = {
            s.edge_id: _align_up_slot_bytes(s.max_bytes) for s in specs
        }
        self.edge_byte_offset: dict[int, int] = {}
        off = 0
        for eid in self.edge_ids:
            self.edge_byte_offset[eid] = off
            off += self.aligned_sizes[eid]
        self.total_bytes: int = off
        # Dense pad slot index per edge (stride-2 slot per edge on both
        # data_pad and ack_pad). Uniform across ranks because edge_ids is
        # sorted identically on every rank.
        self.pad_idx: dict[int, int] = {eid: i for i, eid in enumerate(self.edge_ids)}

        # Rendezvous the slot heap + ack source tensor.
        self.slots_buf: torch.Tensor = symm_mem.empty(
            self.total_bytes, dtype=torch.uint8, device=device
        )
        self.slots_hdl = symm_mem.rendezvous(self.slots_buf, group=group_name)
        self.ack_src: torch.Tensor = symm_mem.empty(
            1, dtype=torch.int32, device=device
        )
        self.ack_hdl = symm_mem.rendezvous(self.ack_src, group=group_name)

        # Per-rank signal pads. data_pad lives on the slots allocation;
        # ack_pad lives on the ack_src allocation. Each edge consumes 2
        # uint32 elements on each pad (stride-2 alignment).
        self.data_pad: torch.Tensor = self.slots_hdl.get_signal_pad(rank)
        self.ack_pad: torch.Tensor = self.ack_hdl.get_signal_pad(rank)
        needed = len(self.edge_ids) * 2
        if self.data_pad.numel() < needed:
            raise RuntimeError(
                f"_CapturedTransportInfra: data_pad.numel()={self.data_pad.numel()} < {needed} "
                f"(n_edges={len(self.edge_ids)} * stride=2). Increase "
                "NVSHMEM_SYMMETRIC_SIZE or reduce edge count."
            )
        if self.ack_pad.numel() < needed:
            raise RuntimeError(
                f"_CapturedTransportInfra: ack_pad.numel()={self.ack_pad.numel()} < {needed} "
                f"(n_edges={len(self.edge_ids)} * stride=2). Increase "
                "NVSHMEM_SYMMETRIC_SIZE or reduce edge count."
            )

    # Edge-indexed view helpers. All returned tensors are stable views into
    # the symm heap — safe to capture inside torch.cuda.graph().

    def slot_view(self, edge_id: int, nbytes: int | None = None) -> torch.Tensor:
        off = self.edge_byte_offset[edge_id]
        cap = self.aligned_sizes[edge_id]
        if nbytes is None:
            end = off + cap
        else:
            if nbytes < 0 or nbytes > cap:
                raise ValueError(
                    f"_CapturedTransportInfra.slot_view: nbytes={nbytes} out of range for "
                    f"edge_id={edge_id} (cap={cap})"
                )
            end = off + nbytes
        return self.slots_buf[off:end]

    def data_pad_slot(self, edge_id: int) -> torch.Tensor:
        i = self.pad_idx[edge_id]
        return self.data_pad[i * 2 : i * 2 + 1]

    def ack_pad_slot(self, edge_id: int) -> torch.Tensor:
        i = self.pad_idx[edge_id]
        return self.ack_pad[i * 2 : i * 2 + 1]


# ---------------------------------------------------------------------------
# NVSHMEMCommunicationManager
# ---------------------------------------------------------------------------


class NVSHMEMCommunicationManager(TensorCommunicationManager):
    def __init__(
        self,
        my_entity_id: str,
        rank: int,
        world_size: int,
        device: torch.device,
        communicator: BaseCommunicator,
        group: "dist.ProcessGroup",
        entity_id_to_rank: dict[str, int],
        max_slot_bytes: int = 32 * 1024 * 1024,  # 32 MB
        num_slots_per_producer: int = 8,
    ) -> None:
        """
        my_entity_id:           this worker's entity ID (e.g. "worker_0")
        rank:                   NVSHMEM / torch.distributed rank
        world_size:             total ranks
        device:                 this rank's CUDA device
        communicator:           ZMQ communicator (for ACK messages)
        group:                  pre-initialized dist.ProcessGroup (NCCL bootstrap)
        entity_id_to_rank:      maps entity IDs → NVSHMEM PE ranks
        max_slot_bytes:         maximum bytes per staging slot
        num_slots_per_producer: number of concurrent in-flight slots per
                                producer rank in the eager staging pool.
                                Must be >= the maximum number of cross-rank
                                NVSHMEM-bound output edges from any single
                                node in one batch. Default 8 covers BAGEL
                                CFG-parallel (4 cross-rank edges).
        """
        # Satisfy main's TensorCommunicationManager invariant: base __init__
        # sets transfer_engine / my_session_id / tensor_store / pending. The
        # NVSHMEMTransferEngine is a no-op subclass — kv_store.py's path is
        # handled polymorphically via get_async_reader() returning None.
        engine = NVSHMEMTransferEngine(my_entity_id)
        super().__init__(
            my_entity_id=my_entity_id,
            my_session_id=engine.get_session_id(),
            device=device,
            communicator=communicator,
            transfer_engine=engine,
        )

        self.rank = rank
        self.world_size = world_size
        self.entity_id_to_rank = entity_id_to_rank
        self._group = group
        self._group_name = group.group_name

        if not symm_mem.is_nvshmem_available():
            raise RuntimeError("NVSHMEM backend not available")
        symm_mem.set_backend("NVSHMEM")

        self.staging = NVSHMEMStagingPool(
            rank=rank,
            world_size=world_size,
            device=device,
            group_name=group.group_name,
            max_slot_bytes=max_slot_bytes,
            num_slots_per_producer=num_slots_per_producer,
        )

        # Dedicated CUDA stream for all NVSHMEM operations.
        self.transfer_stream = torch.cuda.Stream(device=device)

        # Warmup NVSHMEM on the transfer stream (lazy-init is not capture-safe).
        self._warmup_nvshmem()

        # Reassign self.pending to the NVSHMEM-specific transfer type. Base
        # __init__ initialized it to list[FutureAndPointers]; NVSHMEM's eager
        # path tracks list[_NVSHMEMPendingTransfer] instead.
        self.pending: list[_NVSHMEMPendingTransfer] = []

        # Map (uuid, consumer_rank) -> slot_idx, populated when an outgoing
        # PUT is registered. Cleared in _free_slot_for_ack when the
        # consumer's TENSOR_RECEIVED ACK arrives. Lets the manager free the
        # right slot when the same UUID has been broadcast to multiple
        # consumers (each PUT lives in its own slot).
        self._uuid_consumer_to_slot: dict[tuple[str, int], int] = {}
        # Map slot_idx -> set of UUIDs still awaiting ACK from the slot's
        # consumer. The slot is freed when the set empties.
        self._slot_uuids: dict[int, set[str]] = {}

        # Captured worker-to-worker transport state; populated by
        # init_edges() / warmup(). _captured is None until init_edges() is
        # called. warmup() must be called before any record_send / record_recv
        # invocation under graph capture.
        self._captured: "_CapturedTransportInfra | None" = None
        self._captured_warmed_up: bool = False
        # Metrics for §8.4: count of captured replays that have passed through
        # this manager's watchdog, plus a bounded sliding window of replay
        # latencies (microseconds) used to compute p50 on demand.
        self.captured_replays_total: int = 0
        self._captured_replay_latencies_us: list[float] = []
        self._captured_replay_latency_cap: int = 4096

        logger.info(
            "NVSHMEMCommunicationManager: initialized rank=%d world=%d device=%s",
            rank, world_size, device,
        )

    def _warmup_nvshmem(self) -> None:
        """
        Issue a no-op put+signal/wait pair on the transfer stream to trigger
        any lazy NVSHMEM initialization before the first real transfer.
        Called once during __init__.

        Uses producer_rank=0's first slot (slot_idx=0) as a known-uniform
        symmetric address across ranks. Both ranks must agree on slot_idx
        because get_slot_view / get_sig_pad_element index into the local
        symmetric heap at the same offset.
        """
        producer_rank = 0
        warmup_slot_idx = self.staging._global_slot_idx(producer_rank, 0)
        warmup_tensor = self.staging.get_slot_view(warmup_slot_idx, 4)
        sig_elem = self.staging.get_sig_pad_element(warmup_slot_idx)
        peer = (producer_rank + 1) % self.world_size

        with torch.cuda.stream(self.transfer_stream):
            if self.rank == producer_rank:
                torch.ops.symm_mem.nvshmem_put_with_signal(
                    warmup_tensor, sig_elem, 0, peer
                )
            elif self.rank == peer:
                # Wait on the same sig_pad slot the producer wrote.
                torch.ops.symm_mem.nvshmem_wait_for_signal(
                    sig_elem, 0, producer_rank
                )
        self.transfer_stream.synchronize()
        dist.barrier(group=dist.group.WORLD)
        sig_elem.zero_()
        self.transfer_stream.synchronize()

    def _free_slot_for_ack(self, sender_entity_id: str, uuid: str) -> None:
        """Free the slot that held an outgoing PUT to ``sender_entity_id``
        for ``uuid`` — and only that slot.

        Mooncake-routed UUIDs that get accidentally drained here (or ACKs
        for UUIDs we never tracked) silently no-op. The same UUID can be
        broadcast to multiple consumers, each in its own slot, so the
        (uuid, consumer_rank) key is required to free the right one.
        """
        if not sender_entity_id:
            # Older callers / Mooncake ACKs without sender info: nothing
            # to free on the NVSHMEM side. Refcount is still decremented
            # by the caller.
            return
        consumer_rank = self.entity_id_to_rank.get(sender_entity_id)
        if consumer_rank is None:
            return
        slot_idx = self._uuid_consumer_to_slot.pop((uuid, consumer_rank), None)
        if slot_idx is None:
            return
        uuids_in_slot = self._slot_uuids.get(slot_idx)
        if uuids_in_slot is not None:
            uuids_in_slot.discard(uuid)
            if not uuids_in_slot:
                self.staging.free_slot(slot_idx)
                self._slot_uuids.pop(slot_idx, None)

    def _drain_acks(self) -> None:
        """Process any incoming ZMQ messages to pick up TENSOR_RECEIVED ACKs."""
        for msg in self.communicator.get_all_new_messages():
            if (
                hasattr(msg, "message_type")
                and msg.message_type == WorkerMessageType.TENSOR_RECEIVED
            ):
                body = msg.body
                sender = getattr(body, "sender_entity_id", "")
                for uuid in body.successful_tensors:
                    self._free_slot_for_ack(sender, uuid)
                    self.tensor_store.dereference(body.request_id, uuid, n=1)

    def _wait_for_slot(self, producer_rank: int, timeout_s: float = 30.0) -> None:
        """
        Block (CPU spin) until the producer has at least one free staging
        slot in its window of the symmetric heap. In normal operation this
        returns immediately because the pool is sized for the workload.
        Raises RuntimeError after ``timeout_s`` (likely a hung consumer or
        an undersized pool).
        """
        deadline = _time.monotonic() + timeout_s
        while not self.staging.has_free_slot(producer_rank):
            self._drain_acks()
            if _time.monotonic() > deadline:
                raise RuntimeError(
                    f"NVSHMEM staging slots for producer={producer_rank} all "
                    f"in flight after {timeout_s}s. A consumer may be hung, "
                    "or num_slots_per_producer is too small for this workload."
                )
            _time.sleep(0.001)

    # ------------------------------------------------------------------
    # ABC method implementations (overrides of TensorCommunicationManager)
    # ------------------------------------------------------------------

    def store_and_return_tensor_info(
        self, request_id: str, tensors: NameToTensorList,
    ) -> dict[str, list[TensorPointerInfo]]:
        """
        Store tensors in tensor_store. Does NOT yet copy to staging or issue PUT.
        The PUT happens in store_and_populate_graph_edges where consumer_rank is known.
        Returns TensorPointerInfo with placeholder address/signal; these are filled
        when the edge is populated.
        """
        tensor_info: dict[str, list[TensorPointerInfo]] = {}
        for name, tensor_list in tensors.items():
            tensor_info[name] = []
            for tensor in tensor_list:
                tensor_uuid = str(uuid4())
                self.tensor_store.put_tensor(request_id, tensor_uuid, tensor)
                info = TensorPointerInfo(
                    dims=list(tensor.shape),
                    dtype=str(tensor.dtype),
                    stride=list(tensor.stride()),
                    nbytes=tensor.nbytes,
                    address=0,        # filled in store_and_populate_graph_edges
                    uuid=tensor_uuid,
                    source_session_id=str(self.rank),
                    source_entity=self.my_entity_id,
                    source_rank=self.rank,
                    symmetric_offset=0,
                    signal_value=0,
                )
                tensor_info[name].append(info)
        return tensor_info

    def store_and_populate_graph_edges(
        self,
        request_id: str,
        tensors: NameToTensorList,
        graph_edges: list[GraphEdge],
    ) -> None:
        # Build name → edges mapping
        name_to_edges: dict[str, list[GraphEdge]] = {}
        for edge in graph_edges:
            name_to_edges.setdefault(edge.name, []).append(edge)

        graph_node_info = self.store_and_return_tensor_info(request_id, tensors)

        for name, info_list in graph_node_info.items():
            edges = name_to_edges.get(name, [])
            for info in info_list:
                self.tensor_store.increment_ref(
                    request_id, info.uuid,
                    n=len([e for e in edges if e.next_node != EMPTY_DESTINATION])
                )
            for edge in edges:
                if edge.next_node in SPECIAL_DESTINATIONS:
                    # EMPTY_DESTINATION: no consumer rank to route to.
                    # EMIT_TO_CLIENT: api_server is not an NVSHMEM peer; the
                    # tensor is serialized inline and sent via ZMQ in
                    # Worker._send_outputs — no NVSHMEM PUT needed here.
                    # Defect 12 fix: avoids KeyError on non-worker next_nodes.
                    edge.tensor_info = info_list
                    continue
                consumer_rank = self.entity_id_to_rank[edge.next_node]
                tensors_for_edge = [
                    self.tensor_store.get_tensor(request_id, info.uuid)
                    for info in info_list
                ]
                packed_nbytes = _compute_packed_size(tensors_for_edge)

                # Block until the producer's staging window has at least
                # one free slot. With multiple slots per producer the loop
                # can issue back-to-back PUTs to different consumers
                # without waiting for ACKs in between, which is what
                # unblocks fan-out batches like combine_cfg → {LLM_cfg_text,
                # LLM_cfg_img}.
                self._wait_for_slot(self.rank)

                slot = self.staging.alloc_slot(self.rank, consumer_rank, packed_nbytes)
                slot_view = self.staging.get_slot_view(slot.slot_idx, slot.nbytes)
                sig_elem = self.staging.get_sig_pad_element(slot.slot_idx)

                with torch.cuda.stream(self.transfer_stream):
                    _pack_tensors_into_slot(tensors_for_edge, slot_view)
                    torch.ops.symm_mem.nvshmem_put_with_signal(
                        slot_view, sig_elem, slot.sig_val, consumer_rank
                    )

                # Per-edge TPI clones so each edge carries its own slot
                # address / signal. Sharing one info_list across edges
                # would let a later edge's slot info clobber an earlier
                # edge's, which corrupts broadcast routing (different
                # consumers reading from the same TPI).
                edge_infos: list[TensorPointerInfo] = []
                slot_address = slot.slot_idx * self.staging.max_slot_bytes
                slot_uuid_set = self._slot_uuids.setdefault(slot.slot_idx, set())
                for info in info_list:
                    edge_info = TensorPointerInfo(
                        dims=list(info.dims),
                        dtype=info.dtype,
                        stride=list(info.stride),
                        nbytes=info.nbytes,
                        address=slot_address,
                        uuid=info.uuid,
                        source_session_id=info.source_session_id,
                        source_entity=info.source_entity,
                        source_rank=info.source_rank,
                        symmetric_offset=slot_address,
                        signal_value=slot.sig_val,
                    )
                    edge_infos.append(edge_info)
                    self._uuid_consumer_to_slot[(info.uuid, consumer_rank)] = (
                        slot.slot_idx
                    )
                    slot_uuid_set.add(info.uuid)
                edge.tensor_info = edge_infos

    def register_for_send(
        self, request_id: str, uuids: list[str],
        skip_cuda_sync: bool = False,
    ) -> None:
        # NVSHMEM symmetric heap requires no explicit registration.
        # skip_cuda_sync is accepted for ABC compatibility but unused here.
        pass

    def start_read_tensors(
        self,
        request_id: str,
        graph_edges: list[GraphEdge],
    ) -> None:
        for edge in graph_edges:
            if not edge.tensor_info:
                continue  # signal-only edge

            # All TensorPointerInfos for an edge share the same staging slot.
            first = edge.tensor_info[0]
            if first.source_rank < 0:
                raise ValueError(
                    f"NVSHMEMCommunicationManager received TensorPointerInfo "
                    f"with source_rank={first.source_rank}; is this edge "
                    "coming from a Mooncake producer?"
                )

            producer_rank = first.source_rank
            sig_val = first.signal_value
            packed_nbytes = _compute_packed_size_from_infos(edge.tensor_info)

            # Decode the producer's slot_idx from symmetric_offset (set on the
            # producer side as ``slot_idx * max_slot_bytes``). With multiple
            # slots per producer, slot_idx is no longer simply producer_rank.
            slot_idx = first.symmetric_offset // self.staging.max_slot_bytes
            sig_elem = self.staging.get_sig_pad_element(slot_idx)
            slot_view = self.staging.get_slot_view(
                slot_idx=slot_idx,
                nbytes=packed_nbytes,
            )

            # Allocate destination tensors for each info.
            dst_tensors = []
            for info in edge.tensor_info:
                if self.tensor_store.check_uuid_presence(request_id, info.uuid):
                    dst = self.tensor_store.get_tensor(request_id, info.uuid)
                else:
                    # Parse dtype from string representation (e.g. "torch.float32")
                    dtype = _parse_dtype(info.dtype)
                    dst = torch.empty(info.dims, dtype=dtype, device=self.device)
                    self.tensor_store.put_tensor(request_id, info.uuid, dst)
                self.tensor_store.increment_ref(request_id, info.uuid, 1)
                dst_tensors.append(dst)

            # Issue the wait + unpack on the transfer stream.
            completion_event = torch.cuda.Event()
            with torch.cuda.stream(self.transfer_stream):
                torch.ops.symm_mem.nvshmem_wait_for_signal(
                    sig_elem, sig_val, producer_rank
                )
                # After wait: slot_view holds the producer's data.
                _unpack_tensors_from_slot(slot_view, dst_tensors)
                # Zero the signal element to allow reuse (before ACK).
                sig_elem.zero_()
                completion_event.record(self.transfer_stream)

            self.pending.append(_NVSHMEMPendingTransfer(
                request_id=request_id,
                graph_edges=[edge],
                completion_event=completion_event,
                producer_rank=producer_rank,
            ))

    def _cleanup_by_uuid(self, request_id: str, uuid: str) -> None:
        """Required by main's ABC. NVSHMEM overrides cleanup_request directly,
        which doesn't delegate per-uuid; this stub exists only to satisfy
        Python's ABC instantiation requirement."""
        return

    def get_ready_tensors(self) -> dict[str, list[GraphEdge]]:
        ready: dict[str, list[GraphEdge]] = {}
        still_pending: list[_NVSHMEMPendingTransfer] = []

        for transfer in self.pending:
            if transfer.completion_event.query():
                for edge in transfer.graph_edges:
                    ready.setdefault(transfer.request_id, []).append(edge)
                    logger.debug(
                        "NVSHMEM transfer complete: %d tensors for edge %s",
                        len(edge.tensor_info), edge.name,
                    )
            else:
                still_pending.append(transfer)

        self.pending = still_pending

        # Send ACKs and dereference tensors for completed transfers.
        for req_id, edges in ready.items():
            self._collect_and_send_acks(req_id, edges)
            for edge in edges:
                for info in edge.tensor_info:
                    self.tensor_store.dereference(req_id, info.uuid, 1)

        return ready

    def _collect_and_send_acks(
        self, request_id: str, graph_edges: list[GraphEdge]
    ) -> None:
        """Mirror of MooncakeCommunicationManager._collect_and_send_acks."""
        acks: dict[str, dict[str, int]] = {}
        for edge in graph_edges:
            for info in edge.tensor_info:
                if info.source_entity == self.my_entity_id:
                    continue
                acks.setdefault(info.source_entity, {})[info.uuid] = (
                    acks.get(info.source_entity, {}).get(info.uuid, 0) + 1
                )
        for source_entity, tensors in acks.items():
            self.communicator.send(
                source_entity,
                WorkerMessage(
                    message_type=WorkerMessageType.TENSOR_RECEIVED,
                    body=TensorReceived(
                        request_id=request_id,
                        successful_tensors=tensors,
                        failed_tensor_ids=[],
                        sender_entity_id=self.my_entity_id,
                    ),
                ),
            )

    def get_tensor(self, request_id: str, uuid: str) -> torch.Tensor:
        return self.tensor_store.get_tensor(request_id, uuid)

    def set_persist(self, request_id: str, uuid: str, persist: bool) -> None:
        self.tensor_store.set_metadata(request_id, uuid, persist=persist)
        if self.tensor_store.can_gc(request_id, uuid):
            self.tensor_store.remove_tensor(request_id, uuid)

    def dereference(self, request_id: str, uuid: str, n: int = 1) -> None:
        self.tensor_store.dereference(request_id, uuid, n=n)
        if self.tensor_store.can_gc(request_id, uuid):
            self.tensor_store.remove_tensor(request_id, uuid)

    def increment_ref(self, request_id: str, uuid: str, n: int = 1) -> None:
        self.tensor_store.increment_ref(request_id, uuid, n=n)

    def put_foreign(
        self,
        request_id: str,
        uuid: str,
        tensor: torch.Tensor,
        initial_ref_count: int = 1,
    ) -> None:
        """Insert an externally-built tensor into tensor_store.

        The tensor must already be on the correct device.  initial_ref_count
        should equal the number of consuming graph edges on this worker so that
        the last dereference in _cleanup_consumed_inputs triggers GC.
        """
        self.tensor_store.put_tensor(request_id, uuid, tensor)
        self.tensor_store.increment_ref(request_id, uuid, n=initial_ref_count)

    def cleanup_request(self, request_id: str) -> None:
        for uuid in self.tensor_store.get_all_uuids(request_id):
            self.tensor_store.set_metadata(request_id, uuid, persist=False)
            if not self.tensor_store.can_gc(request_id, uuid):
                logger.warning("Deferring cleanup of tensor %s (pending ACK)", uuid)
                continue
            self.tensor_store.remove_tensor(request_id, uuid)
        # Drain any pending transfers for this request (send ACKs for abandoned edges).
        leftover = [t for t in self.pending if t.request_id == request_id]
        self.pending = [t for t in self.pending if t.request_id != request_id]
        for transfer in leftover:
            self._collect_and_send_acks(request_id, transfer.graph_edges)

    def handle_ack(
        self,
        request_id: str,
        uuids: dict[str, int],
        sender_entity_id: str = "",
    ) -> None:
        """
        Called when a TENSOR_RECEIVED ACK arrives from a consumer.
        Frees the staging slot(s) the producer used for this consumer and
        dereferences the tensors.

        ``sender_entity_id`` identifies the consumer that sent the ACK, so
        the producer can free the correct (uuid, consumer) → slot mapping.
        """
        for uuid, ref_cnt in uuids.items():
            self._free_slot_for_ack(sender_entity_id, uuid)
            self.tensor_store.dereference(request_id, uuid, n=ref_cnt)

    def release_slot_for_uuid(
        self,
        request_id: str,
        uuid: str,
        ref_cnt: int = 1,
        sender_entity_id: str = "",
    ) -> None:
        """Single-UUID wrapper around handle_ack for routing via
        Worker._uuid_to_manager.

        Frees the (uuid, consumer) slot and dereferences the tensor.
        ``sender_entity_id`` is the consumer that produced the ACK; if
        empty, no slot is freed (legacy fallback) but the refcount is
        still decremented.
        """
        self.handle_ack(request_id, {uuid: ref_cnt}, sender_entity_id)

    # ------------------------------------------------------------------
    # Captured worker-to-worker transport API
    # (see §5 of the captured-NVSHMEM design doc)
    # ------------------------------------------------------------------

    def init_edges(self, edge_specs: list[EdgeSpec]) -> None:
        """Allocate the captured-transport symmetric heap + pad mappings.

        Called once per worker after graph-walk registration completes and the
        Conductor has emitted the authoritative list of NVSHMEM worker-to-worker
        edges. Every rank in the NVSHMEM process group must call this with the
        same ``edge_specs`` payload; ordering does not matter (the manager sorts
        by ``edge_id`` internally).

        Idempotent: a second call with an equal spec list is a no-op; a second
        call with a different spec list raises ``RuntimeError`` (the symmetric
        heap is rendezvoused once for the lifetime of the manager).
        """
        if self._captured is not None:
            existing = set(self._captured.edge_specs.values())
            incoming = set(edge_specs)
            if existing == incoming:
                logger.debug(
                    "NVSHMEMCommunicationManager.init_edges: idempotent no-op "
                    "(n_edges=%d already registered)", len(existing)
                )
                return
            raise RuntimeError(
                "NVSHMEMCommunicationManager.init_edges called twice with "
                "different edge_specs; captured-transport symmetric heap cannot be "
                "re-rendezvoused. Incoming diff: "
                f"added={sorted(s.edge_id for s in incoming - existing)}, "
                f"removed={sorted(s.edge_id for s in existing - incoming)}"
            )

        self._captured = _CapturedTransportInfra(
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            group_name=self._group_name,
            edge_specs=list(edge_specs),
        )
        n = len(self._captured.edge_ids)
        total_mib = self._captured.total_bytes / (1024 * 1024)
        if self._captured.total_bytes > 2 * 1024 * 1024 * 1024:
            raise RuntimeError(
                f"NVSHMEM captured-transport slot heap would exceed 2 GiB cap "
                f"(requested {total_mib:.1f} MiB across {n} edges). "
                "Reduce per-edge max_bytes or edge count."
            )
        if self._captured.total_bytes > 1024 * 1024 * 1024:
            logger.warning(
                "NVSHMEM captured-transport slot heap is %.1f MiB across %d edges (>1 GiB warning threshold)",
                total_mib, n,
            )
        logger.info(
            "NVSHMEMCommunicationManager.init_edges: rank=%d n_edges=%d total_slot_heap=%.1f MiB",
            self.rank, n, total_mib,
        )

    def warmup(self) -> None:
        """Off-stream eager put/wait per edge + pad bootstrap (§3.3).

        Forces NVSHMEM lazy init off the capture path (lazy init from inside a
        captured graph is fatal), then sets steady-state pad values:

          * producer rank `P`: ``ack_pad[e] = ACK_VAL`` (slot starts free)
          * consumer rank `C`: ``data_pad[e] = 0`` (no pending data)

        Finishes with ``dist.barrier()`` so all ranks have completed bootstrap
        before any captured replay begins. Idempotent: calling twice is a no-op.
        """
        if self._captured is None:
            raise RuntimeError(
                "NVSHMEMCommunicationManager.warmup: must call init_edges first"
            )
        if self._captured_warmed_up:
            logger.debug("NVSHMEMCommunicationManager.warmup: idempotent no-op")
            return

        info = self._captured
        warm_stream = torch.cuda.Stream(device=self.device)
        warm_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm_stream):
            for eid in info.edge_ids:
                spec = info.edge_specs[eid]
                slot = info.slot_view(eid)
                data_pad = info.data_pad_slot(eid)
                ack_pad = info.ack_pad_slot(eid)

                if self.rank == spec.producer_rank:
                    # One eager put to force NVSHMEM lazy init off the capture
                    # path. Target is the consumer's data_pad slot.
                    torch.ops.symm_mem.nvshmem_put_with_signal(
                        slot, data_pad, _NVSHMEM_DATA_VAL, spec.consumer_rank
                    )
                elif self.rank == spec.consumer_rank:
                    # Wait for the producer's eager put, then reset the pad so
                    # that the captured graph enters its first replay with
                    # data_pad=0.
                    torch.ops.symm_mem.nvshmem_wait_for_signal(
                        data_pad, _NVSHMEM_DATA_VAL, spec.producer_rank
                    )
                    data_pad.zero_()
                    # Eager ack back to the producer so it sees ack_pad=1
                    # (steady-state bootstrap: "slot initially free").
                    torch.ops.symm_mem.nvshmem_put_with_signal(
                        info.ack_src, ack_pad, _NVSHMEM_ACK_VAL, spec.producer_rank
                    )
        warm_stream.synchronize()

        # Final steady-state pad values per §3.3: producer sees ack_pad=1
        # (already set by consumer's eager ack above); consumer sees
        # data_pad=0 (already zeroed above). Nothing more to stamp; a barrier
        # ensures everyone has completed bootstrap before any capture runs.
        torch.cuda.synchronize()
        dist.barrier(group=self._group)
        self._captured_warmed_up = True
        logger.info(
            "NVSHMEMCommunicationManager.warmup: rank=%d bootstrapped %d edges",
            self.rank, len(info.edge_ids),
        )

    def record_send(
        self,
        stream: torch.cuda.Stream,
        edge_id: int,
        src: torch.Tensor,
    ) -> None:
        """Producer-side capture hook (§5.1).

        Inside a stream-capture context (the caller must already be under
        ``torch.cuda.graph(..., stream=stream)`` or an equivalent
        ``torch.cuda.stream(stream)`` on the same capture stream), records:

            wait_for(ack_pad == ACK_VAL, peer=C)
            ack_pad.zero_()
            slot[e] <- src  (device-side memcpy)
            put_with_signal(slot[e], data_pad@C, DATA_VAL, peer=C)

        ``src`` must be a contiguous CUDA tensor whose total byte size is ≤
        this edge's aligned slot capacity. The memcpy packs ``src`` into the
        per-edge symmetric slot; the consumer unpacks it inside its matching
        ``record_recv``.
        """
        info = self._require_captured_ready("record_send")
        if edge_id not in info.edge_specs:
            raise KeyError(
                f"record_send: edge_id={edge_id} not registered with this "
                "NVSHMEM manager (was it resolved to Mooncake?)"
            )
        spec = info.edge_specs[edge_id]
        if self.rank != spec.producer_rank:
            raise RuntimeError(
                f"record_send: rank={self.rank} is not the producer of "
                f"edge_id={edge_id} (producer_rank={spec.producer_rank})"
            )
        if not src.is_cuda or src.device != self.device:
            raise ValueError(
                f"record_send: src must be a CUDA tensor on device {self.device}; "
                f"got device={src.device}"
            )
        nbytes = src.numel() * src.element_size()
        cap = info.aligned_sizes[edge_id]
        if nbytes > cap:
            raise ValueError(
                f"record_send: src.nbytes={nbytes} exceeds edge_id={edge_id} "
                f"slot capacity={cap} (max_bytes={spec.max_bytes})"
            )

        slot = info.slot_view(edge_id)
        data_pad = info.data_pad_slot(edge_id)
        ack_pad = info.ack_pad_slot(edge_id)
        # uint8 view of src bytes for the pack memcpy (src may be any dtype).
        src_bytes = src.contiguous().view(torch.uint8).reshape(-1)

        with torch.cuda.stream(stream):
            # 1. Backpressure: wait until the consumer has freed the slot.
            torch.ops.symm_mem.nvshmem_wait_for_signal(
                ack_pad, _NVSHMEM_ACK_VAL, spec.consumer_rank
            )
            # 2. Reset ack pad for the *next* iteration's backpressure wait.
            ack_pad.zero_()
            # 3. Pack: memcpy src into the per-edge slot (device-side copy).
            slot[:nbytes].copy_(src_bytes)
            # 4. Ship + signal the consumer.
            torch.ops.symm_mem.nvshmem_put_with_signal(
                slot[:nbytes] if nbytes == cap else slot,
                data_pad,
                _NVSHMEM_DATA_VAL,
                spec.consumer_rank,
            )

    def record_recv(
        self,
        stream: torch.cuda.Stream,
        edge_id: int,
        dst: torch.Tensor,
    ) -> None:
        """Consumer-side capture hook (§5.1).

        Records, inside a stream-capture context:

            wait_for(data_pad == DATA_VAL, peer=P)
            data_pad.zero_()
            dst <- slot[e]  (device-side memcpy)
            put_with_signal(ack_src, ack_pad@P, ACK_VAL, peer=P)

        ``dst`` must be a contiguous CUDA tensor whose total byte size fits
        inside this edge's aligned slot capacity. The slot-free ack is folded
        into this call so engine code cannot forget to ack.
        """
        info = self._require_captured_ready("record_recv")
        if edge_id not in info.edge_specs:
            raise KeyError(
                f"record_recv: edge_id={edge_id} not registered with this "
                "NVSHMEM manager (was it resolved to Mooncake?)"
            )
        spec = info.edge_specs[edge_id]
        if self.rank != spec.consumer_rank:
            raise RuntimeError(
                f"record_recv: rank={self.rank} is not the consumer of "
                f"edge_id={edge_id} (consumer_rank={spec.consumer_rank})"
            )
        if not dst.is_cuda or dst.device != self.device:
            raise ValueError(
                f"record_recv: dst must be a CUDA tensor on device {self.device}; "
                f"got device={dst.device}"
            )
        nbytes = dst.numel() * dst.element_size()
        cap = info.aligned_sizes[edge_id]
        if nbytes > cap:
            raise ValueError(
                f"record_recv: dst.nbytes={nbytes} exceeds edge_id={edge_id} "
                f"slot capacity={cap} (max_bytes={spec.max_bytes})"
            )

        slot = info.slot_view(edge_id)
        data_pad = info.data_pad_slot(edge_id)
        ack_pad = info.ack_pad_slot(edge_id)
        dst_bytes = dst.contiguous().view(torch.uint8).reshape(-1)

        with torch.cuda.stream(stream):
            # 1. Wait for the producer's fresh put.
            torch.ops.symm_mem.nvshmem_wait_for_signal(
                data_pad, _NVSHMEM_DATA_VAL, spec.producer_rank
            )
            # 2. Reset data pad for the *next* iteration.
            data_pad.zero_()
            # 3. Unpack: memcpy slot into dst.
            dst_bytes.copy_(slot[:nbytes])
            # 4. Ack the producer ASAP so it can overlap iter (i+1)'s pack.
            torch.ops.symm_mem.nvshmem_put_with_signal(
                info.ack_src, ack_pad, _NVSHMEM_ACK_VAL, spec.producer_rank
            )

    def teardown(self) -> None:
        """Free the captured-transport symmetric heap allocation.

        After teardown, further ``record_send``/``record_recv`` calls raise
        until ``init_edges`` + ``warmup`` are called again. Mooncake-style
        cleanup semantics: idempotent, never raises on already-torn-down state.
        """
        if self._captured is None:
            return
        # Dropping the references to slots_buf / ack_src / handles releases the
        # symmetric heap via torch's symm_mem allocator. No explicit free API
        # is exposed; GC handles it.
        self._captured = None
        self._captured_warmed_up = False
        logger.info(
            "NVSHMEMCommunicationManager.teardown: rank=%d captured-transport resources released",
            self.rank,
        )

    def mark_replay(self, latency_us: float) -> None:
        """Record that one captured replay that routed through this manager
        has completed in ``latency_us`` microseconds.

        Called by the Worker's watchdog (§7.1) after ``event.query()`` returns
        True. Metrics are exposed via ``captured_replays_total`` and
        ``captured_replay_p50_us``.
        """
        self.captured_replays_total += 1
        buf = self._captured_replay_latencies_us
        buf.append(latency_us)
        # Bounded ring to avoid unbounded growth on long-running workers.
        if len(buf) > self._captured_replay_latency_cap:
            # Drop the oldest half to amortize the copy cost.
            half = self._captured_replay_latency_cap // 2
            del buf[:half]

    @property
    def captured_replay_p50_us(self) -> float:
        """Median of the recorded captured-replay latencies, or 0.0 if none.

        Computed on demand from the bounded latency ring (not a running
        estimator — the ring is small enough that sort-every-query is cheap
        relative to the hot path)."""
        buf = self._captured_replay_latencies_us
        if not buf:
            return 0.0
        s = sorted(buf)
        mid = len(s) // 2
        if len(s) % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def _require_captured_ready(self, op: str) -> "_CapturedTransportInfra":
        if self._captured is None:
            raise RuntimeError(
                f"NVSHMEMCommunicationManager.{op}: init_edges() has not been called"
            )
        if not self._captured_warmed_up:
            raise RuntimeError(
                f"NVSHMEMCommunicationManager.{op}: warmup() has not been called; "
                "captured put/wait from a cold manager deadlocks"
            )
        return self._captured
