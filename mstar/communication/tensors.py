import logging
import os
import platform
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from uuid import uuid4

from mstar.distributed.base import ShardingConfig
from mstar.graph.special_destinations import EMPTY_DESTINATION

try:
    from mooncake.engine import TransferEngine
except Exception as _err:
    MOONCAKE_IMPORT_ERROR = _err
    TransferEngine = None
else:
    MOONCAKE_IMPORT_ERROR = None
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo
from mstar.utils.ipc_format import TensorReceived, WorkerMessage, WorkerMessageType

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

@dataclass
class BufferedShards:
    total_fanin: int
    shard_dim: int
    # source rank -> tensor
    shards: dict[int, list[torch.Tensor]]

    def is_done(self):
        return len(self.shards) >= self.total_fanin

    def consolidate(self) -> list[torch.Tensor]:
        assert self.is_done(), \
            "Can't consolidate shards until all fanin contributions are received"
        keys = sorted(self.shards.keys())
        n = len(self.shards[keys[0]])
        assert all(len(self.shards[k]) == n for k in keys), (
            f"BufferedShards: source ranks contributed unequal tensor counts: "
            f"{ {k: len(self.shards[k]) for k in keys} }"
        )
        return [
            torch.cat([self.shards[k][i] for k in keys], dim=self.shard_dim)
            for i in range(n)
        ]


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
        self.read_finished: dict[str, set[str]] = {}

        # req_id -> cfg
        self.sharding_configs: dict[str, ShardingConfig] = {}
        # req_id -> {name -> shards}
        self.buffered_shards: dict[str, dict[str, BufferedShards]] = {}
        self.uuid_to_shard_dim: dict[str, int | None] = {}

    # ---- shared: store ----
    def _ensure_leading_shard_dim(self, shard_dim: int | None, tensor: torch.Tensor):
        """Move ``shard_dim`` to dim 0, preserving the relative order of the
        remaining dims. The inverse is ``_undo_leading_shard_dim``."""
        if shard_dim is None or shard_dim == 0:
            return tensor
        perm = [shard_dim] + [i for i in range(tensor.ndim) if i != shard_dim]
        return tensor.permute(*perm).contiguous()

    def _undo_leading_shard_dim(self, shard_dim: int | None, tensor: torch.Tensor):
        """Inverse of ``_ensure_leading_shard_dim``: move dim 0 back to position
        ``shard_dim``, preserving the relative order of the remaining dims."""
        if shard_dim is None or shard_dim == 0:
            return tensor
        inverse_perm = (
            list(range(1, shard_dim + 1)) + [0] + list(range(shard_dim + 1, tensor.ndim))
        )
        return tensor.permute(*inverse_perm).contiguous()

    def store_and_return_tensor_info(
        self, request_id: str, tensors: NameToTensorList,
        node_name: str | None=None,
        graph_walk: str | None=None,
        skip_cuda_sync: bool = False,
    ) -> dict[str, list[TensorPointerInfo]]:
        # CUDA sync ensures GPU writes to ``tensors`` are visible before
        # callers hand out ``tensor.data_ptr()`` to peers (RDMA register,
        # SHM serialize). With same-thread async scheduling the caller
        # has typically already synced on ``output.completion_event``
        # (which waits for *only* the producing step, not the queued next
        # step), so the unconditional default-stream sync here would
        # uselessly drain GPU(N+1). Pass ``skip_cuda_sync=True`` from
        # those call sites.
        if not skip_cuda_sync and torch.cuda.is_available():
            torch.cuda.default_stream().synchronize()
        tensor_info: dict[str, list[TensorPointerInfo]] = {}

        # register_request is a hard prereq on the worker side, but the API
        # server calls this without registering. Default to "no group" / TP=1
        # when no sharding_config is present.
        cfg = self.sharding_configs.get(request_id)
        if cfg is not None:
            source_group = cfg.get_sharding_group(node_name, graph_walk)
        else:
            source_group = None
        if source_group is not None:
            source_tp_size = source_group.tp_size
            source_tp_rank = source_group._tp_rank or 0
        else:
            source_tp_size, source_tp_rank = 1, 0

        for name, tensor_list in tensors.items():
            tensor_info[name] = []

            shard_dim = cfg.shard_dim.get(name) if cfg is not None else None
            for tensor in tensor_list:
                tensor_uuid = str(uuid4())
                # TODO: only rearrange when (1) the tensor will be sent and
                # (2) it may be split along the shard dim in transport. Doing
                # it here unconditionally so TensorPointerInfo dims/strides
                # match what receivers will read.
                canonical = self._ensure_leading_shard_dim(shard_dim, tensor)
                self.tensor_store.put_tensor(
                    request_id=request_id, uuid=tensor_uuid, tensor=canonical,
                )
                if cfg is not None:
                    self.uuid_to_shard_dim[tensor_uuid] = shard_dim

                logger.debug("Storing tensor name %s uuid %s", name, tensor_uuid)
                tensor_info[name].append(TensorPointerInfo(
                    dims=canonical.shape,
                    dtype=canonical.dtype,
                    stride=canonical.stride(),
                    nbytes=canonical.nbytes,
                    address=canonical.data_ptr(),
                    uuid=tensor_uuid,
                    source_session_id=self.my_session_id,
                    source_entity=self.my_entity_id,
                    source_tp_size=source_tp_size,
                    source_tp_rank=source_tp_rank,
                    _source_node_name=node_name,
                    _source_graph_walk=graph_walk,
                ))
        return tensor_info

    def store_and_populate_graph_edges(
        self, request_id: str, tensors: NameToTensorList,
        graph_edges: list[GraphEdge],
        node_name: str | None=None,
        graph_walk: str | None=None,
        skip_cuda_sync: bool = False,
        skip_ref_count: bool = False,
    ):
        name_to_graph_edges: dict[str, list[GraphEdge]] = {}
        for edge in graph_edges:
            name_to_graph_edges.setdefault(edge.name, []).append(edge)

        graph_node_info = self.store_and_return_tensor_info(
            request_id=request_id, tensors=tensors,
            node_name=node_name, graph_walk=graph_walk,
            skip_cuda_sync=skip_cuda_sync,
        )
        for name in tensors:
            logger.debug(
                "Storing tensor %s (uuids %s) for nodes %s",
                name, str([info.uuid for info in graph_node_info[name]]),
                str([edge.name for edge in name_to_graph_edges.get(name, [])])
            )
            edges = name_to_graph_edges.get(name, [])
            if skip_ref_count:
                # Safety hold: ref=1 prevents premature GC. The caller
                # must call set_output_ref_counts() to adjust to the real
                # fanout after routing is computed.
                for info in graph_node_info[name]:
                    self.tensor_store.increment_ref(request_id, info.uuid, n=1)
            else:
                for info in graph_node_info[name]:
                    self.tensor_store.increment_ref(
                        request_id, info.uuid, n=len([
                            e for e in edges if e.next_node != EMPTY_DESTINATION
                        ])
                    )
            for edge in edges:
                edge.tensor_info = graph_node_info[name]
        return graph_node_info

    def set_output_ref_counts(
        self,
        request_id: str,
        safety_hold_uuids: set[str],
        routed_edges: list[GraphEdge],
    ):
        """Adjust ref counts from the safety hold (1) to the actual fanout.

        Called after ``process_node_outputs`` determines the real routing.
        ``safety_hold_uuids`` is the set of UUIDs that were given ref=1 by
        ``store_and_populate_graph_edges(skip_ref_count=True)``.
        ``routed_edges`` is the flat list of all edges that will actually be
        consumed (local ingestion, remote send, persist, emit, streaming).
        """
        actual_counts: dict[str, int] = {uuid: 0 for uuid in safety_hold_uuids}
        for edge in routed_edges:
            for info in edge.tensor_info:
                if info.uuid in actual_counts:
                    actual_counts[info.uuid] += 1

        for uuid, count in actual_counts.items():
            delta = count - 1  # subtract the safety hold of 1
            if delta > 0:
                self.tensor_store.increment_ref(request_id, uuid, n=delta)
            elif delta < 0:
                self.dereference(request_id, uuid, n=-delta)

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

    def _slice_existing_tensor(
        self, request_id: str, name: str, next_node: str,
        graph_walk: str | None, info: TensorPointerInfo
    ):
        shard_dim = self.sharding_configs[request_id].shard_dim.get(name)
        dest_group = self.sharding_configs[request_id].group_mapping.get(
            NodeAndGraphWalk(next_node, graph_walk)
        )
        dest_tp_size = dest_group.tp_size if dest_group is not None else 1
        if info.source_tp_size != dest_tp_size and shard_dim is not None:
            canonical_tensor = self.tensor_store.get_tensor(request_id, info.uuid)
            # Canonical layout has shard_dim leading and is contiguous, so a
            # contiguous byte range maps to a contiguous range of rows along
            # dim 0. Slicing along dim 0 keeps the view contiguous.
            bytes_per_row = canonical_tensor[0].nbytes
            assert info.offset % bytes_per_row == 0 and info.nbytes % bytes_per_row == 0, (
                f"slice offset {info.offset} / nbytes {info.nbytes} not aligned "
                f"to canonical row size {bytes_per_row}"
            )
            start = info.offset // bytes_per_row
            end = start + info.nbytes // bytes_per_row
            slice_view = canonical_tensor[start:end]
            new_uuid = str(uuid4())
            self.tensor_store.put_tensor(request_id, new_uuid, slice_view)
            self.uuid_to_shard_dim[new_uuid] = shard_dim
            # Release this edge's stake on the producer's UUID — the slice now
            # owns it. The slice view keeps the underlying storage alive even
            # if the producer's UUID GCs.
            self.dereference(request_id, info.uuid, 1)
            info.uuid = new_uuid

    @abstractmethod
    def start_read_tensors(
        self, request_id: str, graph_edges: list[GraphEdge],
        graph_walk: str | None = None
    ) -> list[Future]:
        ...

    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        self.uuid_to_shard_dim.pop(uuid, None)

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

    def get_ready_tensors(self, graph_walk: str | None=None) -> dict[str, list[GraphEdge]]:
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

        final_ready: dict[str, list[GraphEdge]] = {}
        for req_id, edges in ready.items():
            self._collect_and_send_acks(req_id, edges)
            seen_uuids = self.read_finished.setdefault(req_id, set())
            for edge in edges:
                if not edge.tensor_info:
                    final_ready.setdefault(req_id, []).append(edge)
                    continue
                # Already-emitted edge re-surfacing (re-delivery / retry): skip
                # the buffering path and pass through unchanged.
                if any(info.uuid in seen_uuids for info in edge.tensor_info):
                    final_ready.setdefault(req_id, []).append(edge)
                    continue

                shard_dim = edge._shard_dim
                if edge.name not in self.buffered_shards.get(req_id, {}):
                    total_fanin = edge._total_fanin
                    if total_fanin > 1:
                        self.buffered_shards[req_id][edge.name] = BufferedShards(
                            total_fanin=total_fanin, shard_dim=shard_dim,
                            shards={},
                        )

                tensors: list[torch.Tensor] = []
                for info in edge.tensor_info:
                    self.tensor_store.dereference(req_id, info.uuid, 1)
                    seen_uuids.add(info.uuid)
                    self.uuid_to_shard_dim[info.uuid] = shard_dim
                    tensors.append(self.get_tensor(req_id, info.uuid))

                if edge.name in self.buffered_shards.get(req_id, {}):
                    buf = self.buffered_shards[req_id][edge.name]
                    buf.shards.setdefault(
                        edge.tensor_info[0].source_tp_rank, []
                    ).extend(tensors)
                    # Release the "+1 for graph-node usage" ref now that the
                    # tensor data has been copied into the buffer; the standard
                    # GC path drops the registered memory when refcount hits 0.
                    for info in edge.tensor_info:
                        self.dereference(req_id, info.uuid, 1)

                    if buf.is_done():
                        consolidated = buf.consolidate()
                        new_infos: list[TensorPointerInfo] = []
                        for uuid_, tensor in (
                            (str(uuid4()), t) for t in consolidated
                        ):
                            self.tensor_store.put_tensor(req_id, uuid_, tensor)
                            # +1 for graph-node usage (released by the
                            # downstream consumer via _cleanup_consumed_inputs)
                            self.tensor_store.increment_ref(req_id, uuid_, 1)
                            # consolidated tensor is in its original layout
                            # (cat happened along the real shard_dim); record
                            # None so get_tensor doesn't try to un-rearrange.
                            self.uuid_to_shard_dim[uuid_] = None
                            new_infos.append(TensorPointerInfo(
                                dims=tensor.shape, dtype=tensor.dtype,
                                nbytes=tensor.nbytes, address=tensor.data_ptr(),
                                stride=tensor.stride(), uuid=uuid_,
                                source_session_id=self.my_session_id,
                                source_entity=self.my_entity_id,
                            ))
                        edge.tensor_info = new_infos
                        del self.buffered_shards[req_id][edge.name]
                        final_ready.setdefault(req_id, []).append(edge)
                    # else: still waiting on more shards; don't emit yet
                else:
                    final_ready.setdefault(req_id, []).append(edge)
        return final_ready

    def register_request(
        self, request_id: str, sharding_config: ShardingConfig
    ):
        self.sharding_configs[request_id] = sharding_config
        self.buffered_shards[request_id] = {}

    # ---- shared: TensorStore delegation ----

    def get_tensor(self, request_id: str, uuid: str) -> torch.Tensor:
        tensor = self.tensor_store.get_tensor(request_id=request_id, uuid=uuid)
        shard_dim = self.uuid_to_shard_dim.get(uuid)
        if shard_dim is None:
            return tensor
        return self._undo_leading_shard_dim(shard_dim, tensor)

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
        self.read_finished.pop(request_id, None)
        self.buffered_shards.pop(request_id, None)
        self.sharding_configs.pop(request_id, None)
        for uuid in self.tensor_store.get_all_uuids(request_id):
            self.uuid_to_shard_dim.pop(uuid, None)
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
        super()._cleanup_by_uuid(request_id, uuid)
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
        graph_walk: str | None=None
    ) -> list[Future]:
        futures = []
        for graph_edge in graph_edges:
            if len(graph_edge.tensor_info) == 0:
                continue

            logger.debug(
                "Starting to read in %d tensors %s for graph node %s",
                len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node
            )

            read_info = []
            for info in graph_edge.tensor_info:
                if info.source_entity == self.my_entity_id:
                    self._slice_existing_tensor(
                        request_id=request_id, name=graph_edge.name,
                        next_node=graph_edge.next_node,
                        graph_walk=graph_walk, info=info
                    )
                    self.tensor_store.increment_ref(request_id, info.uuid, 1)
                    continue
                if self.tensor_store.check_uuid_presence(request_id, info.uuid):
                    self.tensor_store.increment_ref(request_id, info.uuid, 1)
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
                    remote_ptr=info.address + info.offset,
                    nbytes=info.nbytes,
                ))
                logger.debug("Started transfer read for uuid %s", info.uuid)
            fut = self._async_reader.submit(read_info)
            if fut is not None:
                futures.append(fut)
            self.pending.append(
                FutureAndPointers(
                    future=fut, graph_edges=[graph_edge],
                    request_id=request_id
                )
            )
        return futures


# ---------------------------------------------------------------------------
# Shared-memory tensor serialization helpers
# ---------------------------------------------------------------------------

def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes: header + contiguous raw data."""
    t = tensor.detach().contiguous().cpu()
    return t.view(torch.uint8).numpy().tobytes()


def _deserialize_tensor(
    data: bytes | memoryview, device: str,
    tensor_info: TensorPointerInfo
) -> torch.Tensor:
    """Reconstruct a tensor from bytes produced by ``_serialize_tensor``."""
    if isinstance(data, memoryview):
        data = bytes(data)
    shape = tensor_info.dims
    dtype = tensor_info.dtype
    if len(data) == 0:
        t = torch.empty(shape, dtype=dtype)
    else:
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype).reshape(shape)
    if device != "cpu":
        t = t.to(device)
    return t


def _default_shm_dir() -> str:
    """Return the default shared-memory directory for the current platform."""
    if platform.system() == "Linux" and os.path.isdir("/dev/shm"):
        return "/dev/shm"
    return "/tmp/mstar_shm"


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

        # Dedicated copy streams: D2H/H2D run here so they don't serialize
        # behind the next GPU step queued on the default stream.
        self._d2h_stream: torch.cuda.Stream | None = None
        self._h2d_stream: torch.cuda.Stream | None = None
        if torch.cuda.is_available() and str(device) != "cpu":
            self._d2h_stream = torch.cuda.Stream(device=device)
            self._h2d_stream = torch.cuda.Stream(device=device)

    def _shm_path(self, entity_id: str, uuid: str) -> str:
        return os.path.join(self.shm_dir, f"mstar_{entity_id}_{uuid}")

    def register_for_send(
        self, request_id: str, uuids: list[str],
        skip_cuda_sync: bool = False,
    ):
        if not skip_cuda_sync and torch.cuda.is_available():
            torch.cuda.default_stream().synchronize()
        # Producer's completion event is already waited on upstream, so the
        # source tensors are device-visible on entry. Running the D2H on a
        # dedicated stream keeps .cpu()'s host-wait bounded by the copy
        # itself instead of stalling behind GPU(N+1) on the default stream.
        ctx = (
            torch.cuda.stream(self._d2h_stream)
            if self._d2h_stream is not None
            else _nullcontext()
        )
        with ctx:
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
        graph_walk: str | None = None
    ):
        # Run H2D copies on a dedicated stream so they overlap with the
        # consumer's in-flight default-stream work. We make default wait
        # for the H2D stream at the end so downstream kernels see the data.
        h2d_did_work = False
        ctx = (
            torch.cuda.stream(self._h2d_stream)
            if self._h2d_stream is not None
            else _nullcontext()
        )
        with ctx:
            for graph_edge in graph_edges:
                if len(graph_edge.tensor_info) == 0:
                    continue
                logger.debug(
                    "SHM: starting read of %d tensors %s for graph node %s",
                    len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node,
                )
                for info in graph_edge.tensor_info:
                    if info.source_entity == self.my_entity_id:
                        self._slice_existing_tensor(
                            request_id=request_id, name=graph_edge.name,
                            next_node=graph_edge.next_node,
                            graph_walk=graph_walk, info=info
                        )
                        self.tensor_store.increment_ref(request_id, info.uuid, 1)
                        continue
                    if self.tensor_store.check_uuid_presence(request_id, info.uuid):
                        self.tensor_store.increment_ref(request_id, info.uuid, 1)
                        continue
                    path = self._shm_path(info.source_entity, info.uuid)
                    with open(path, "rb") as f:
                        f.seek(info.offset)
                        data = f.read(info.nbytes)
                    tensor = _deserialize_tensor(data, self.device, tensor_info=info)
                    h2d_did_work = True
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
        if h2d_did_work and self._h2d_stream is not None:
            torch.cuda.default_stream(self.device).wait_stream(self._h2d_stream)
        return []

    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        super()._cleanup_by_uuid(request_id, uuid)
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
