import logging
from abc import ABC, abstractmethod
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
from mminf.ipc_formats import TensorReceived, WorkerMessage, WorkerMessageType

logger = logging.getLogger(__name__)


@dataclass
class EventAndPointers:
    event: torch.cuda.Event
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


class TensorCommunicationManager(ABC):
    @abstractmethod
    def store_and_return_tensor_info(
        self, request_id: str, tensors: NameToTensorList,
    ) -> dict[str, list[TensorPointerInfo]]:
        """
        Returns tensor name to TensorPointerInfo (contains addresses, datatypes,
        num bytes, etc.) for each tensor.
        """
        pass

    @abstractmethod
    def store_and_populate_graph_edges(
        self, request_id: str, tensors: NameToTensorList,
        graph_edges: list[GraphEdge]
    ):
        """
        Updates graph_edges with required tensor info (addresses, datatypes,
        num bytes, etc.) and UUID.
        """
        pass

    @abstractmethod
    def register_for_send(
        self, request_id: str, uuids: list[str]
    ):
        """
        If relevant (e.g., mooncake rdma), registers buffers.
        """
        pass

    @abstractmethod
    def get_tensor(self, request_id: str, uuid: str=None) -> torch.Tensor:
        pass

    @abstractmethod
    def set_persist(self, request_id: str, uuid: str, persist: bool):
        pass

    @abstractmethod
    def dereference(self, request_id: str, uuid: str, n: int=1):
        pass

    @abstractmethod
    def increment_ref(self, request_id: str, uuid: str, n: int=1):
        pass

    @abstractmethod
    def cleanup_request(self, request_id: str):
        """
        Removes all tensors for a given request. Unregisters buffers if relevant.
        """
        pass

    @abstractmethod
    def start_read_tensors(
        self, request_id: str, graph_edges: list[GraphEdge]
    ) -> list[int]:
        """
        Initializes empty buffer, initializes a read. May return immediately.
        """
        pass

    @abstractmethod
    def get_ready_tensors(self) -> dict[str, list[GraphEdge]]: # uuid based
        """
        Returns request_id: list of the GraphEdges that are currently
        ready for that request

        Returns a list of local addresses for the tensors being read.
        """
        pass


class MooncakeCommunicationManager(TensorCommunicationManager):
    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        communicator: BaseCommunicator,
        protocol: CommProtocol=CommProtocol.RDMA,
        metadata_server: str="P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]

    ):
        self.my_entity_id = my_entity_id

        self.tensor_store = TensorStore()

        self.communicator = communicator
        self.protocol = protocol
        # Use hostname:port as the Mooncake session ID for RDMA handshake.
        # Each entity must use a unique port.
        self.my_session_id = hostname

        if TransferEngine is not None:
            self.engine = TransferEngine()
            self.engine.initialize(
                hostname,
                metadata_server,
                protocol.value,
                ""
            )
            self.my_session_id =f"{hostname}:{self.engine.get_rpc_port()}"
        else:
            if self.protocol == CommProtocol.RDMA:
                detail = (
                    f"{type(MOONCAKE_IMPORT_ERROR).__name__}: "
                    f"{MOONCAKE_IMPORT_ERROR}"
                    if MOONCAKE_IMPORT_ERROR is not None
                    else "unknown import failure"
                )
                raise RuntimeError(
                    "Mooncake TransferEngine is required when protocol=RDMA. "
                    f"Failed to load mooncake: {detail}. "
                    "Install mooncake-transfer-engine or set tensor protocol to IPC."
                )
            self.engine = None
        # Per-transfer pending list (allows partial tensor readiness)
        self.pending: list[EventAndPointers] = []

    def store_and_return_tensor_info(
        self, request_id: str, tensors: NameToTensorList,
    ) -> dict[str, list[TensorPointerInfo]]: # name to list[tensorPointerInfo]
        tensor_info: dict[str, list[TensorPointerInfo]] = {}
        for name, tensor_list in tensors.items():
            tensor_info[name] = []
            for tensor in tensor_list:
                tensor_uuid = str(uuid4())
                self.tensor_store.put_tensor(
                    request_id=request_id,
                    uuid=tensor_uuid,
                    tensor=tensor
                )
                logger.debug("Storing tensor name %s uuid %s", name, tensor_uuid)
                new_tensor_info = TensorPointerInfo(
                    dims=tensor.shape,
                    dtype=tensor.dtype,
                    stride=tensor.stride(),
                    nbytes=tensor.nbytes,
                    address=tensor.data_ptr(),
                    uuid=tensor_uuid,
                    source_session_id=self.my_session_id,
                    source_entity=self.my_entity_id,
                )
                tensor_info[name].append(new_tensor_info)
        return tensor_info

    def register_for_send(self, request_id, uuids):
        for uuid in uuids:
            if self.protocol == CommProtocol.RDMA:
                if self.engine is None:
                    raise RuntimeError(
                        "Cannot register tensors for RDMA send: TransferEngine is not available."
                    )
                if self.tensor_store.is_registered(request_id, uuid):
                    continue
                logger.debug("Registering %s for send", uuid)
                tensor = self.tensor_store.get_tensor(
                    request_id=request_id, uuid=uuid
                )

                torch.cuda.default_stream().synchronize()
                ret_value = self.engine.register_memory(
                    tensor.data_ptr(), tensor.nbytes
                )
                if ret_value != 0:
                    # TODO: error handling
                    raise RuntimeError(f"Mooncake memory registration failed for request id {request_id}, uuid {uuid}.")
            self.tensor_store.set_metadata(
                request_id, uuid, mem_registered=True
            )

    def store_and_populate_graph_edges(
        self, request_id: str,
        tensors: NameToTensorList,
        graph_edges: list[GraphEdge]
    ):
        # get tensor name to graph edges
        name_to_graph_edges: dict[str, list[GraphEdge]] = {}
        for edge in graph_edges:
            if edge.name not in name_to_graph_edges:
                name_to_graph_edges[edge.name] = []
            name_to_graph_edges[edge.name].append(edge)

        graph_node_info = self.store_and_return_tensor_info(
            request_id=request_id, tensors=tensors
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
                        graph_edge for graph_edge in edges if graph_edge.next_node != EMPTY_DESTINATION
                    ]) # number of nodes it will be sent to
                )
            for edge in edges:
                edge.tensor_info = graph_node_info[name]

    def _cleanup_by_uuid(
        self, request_id: str, uuid: str
    ):
        logger.debug("Deleting tensor uuid %s", uuid)
        if not self.tensor_store.check_uuid_presence(request_id, uuid):
            logger.warning("Trying to cleanup tensor %s, but uuid not found", uuid)
            return

        if self.protocol == CommProtocol.RDMA and self.engine is not None \
                and self.tensor_store.is_registered(request_id, uuid):
            ret_value = self.engine.unregister_memory(
                self.tensor_store.get_tensor(request_id, uuid).data_ptr()
            )
            if ret_value != 0:
                raise RuntimeError("Mooncake memory unregistration failed.")
        self.tensor_store.remove_tensor(request_id, uuid)

    def set_persist(self, request_id: str, uuid: str, persist: bool):
        self.tensor_store.set_metadata(
            request_id, uuid, persist=persist
        )
        if self.tensor_store.can_gc(request_id, uuid):
            self._cleanup_by_uuid(request_id, uuid)

    def dereference(self, request_id: str, uuid: str, n: int=1):
        self.tensor_store.dereference(request_id, uuid, n=n)
        if self.tensor_store.can_gc(request_id, uuid):
            self._cleanup_by_uuid(request_id, uuid)

    def increment_ref(self, request_id: str, uuid: str, n: int=1):
        self.tensor_store.increment_ref(request_id, uuid, n=n)

    def _collect_and_send_acks(
        self, request_id: str, graph_edges: list[GraphEdge]
    ):
        # Collect ACKs to send: entity_id -> {UUID -> count}
        acks:  dict[str, dict[str, int]] = {}

        for edge in graph_edges:
            for info in edge.tensor_info:
                if info.source_entity not in acks:
                    acks[info.source_entity] = {}
                acks[info.source_entity][info.uuid] = acks[info.source_entity].get(
                    info.uuid, 0) + 1

        # Send ACKs to senders
        for source_entity, tensors in acks.items():
            if source_entity == self.my_entity_id:
                continue  # local transfer, no ACK needed
            self.communicator.send(
                source_entity,
                WorkerMessage(
                    message_type=WorkerMessageType.TENSOR_RECEIVED,
                    body=TensorReceived(
                        request_id=request_id,
                        successful_tensors=tensors,
                        failed_tensor_ids=[], # TODO: handle failed transfers
                    ),
                ),
            )

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

        # remove pending transfers but send ACKs
        self._collect_and_send_acks(
            request_id, sum([
                ep.graph_edges for ep in self.pending if ep.request_id == request_id
            ], start=[])
        )
        self.pending = [
            ep for ep in self.pending if ep.request_id != request_id
        ]

    def get_tensor(self, request_id: str, uuid: str) -> torch.Tensor:
        return self.tensor_store.get_tensor(
            request_id=request_id, uuid=uuid
        )

    def get_ready_tensors(self) -> dict[str, list[GraphEdge]]:
        """
        Poll CUDA events. Return {request_id: [ready GraphEdges]}.
        Remove completed entries from self.pending.
        Sends TENSOR_RECEIVED ACKs back to senders so they can free buffers.
        """

        # request_id -> ready graph edges
        ready: dict[str, list[GraphEdge]] = {}
        still_pending = []

        for ep in self.pending:
            if ep.event.query():
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
                    self.tensor_store.dereference(
                        req_id, info.uuid, 1
                    )

        return ready

    def start_read_tensors(
        self, request_id: str, graph_edges: list[GraphEdge],
        device: str="cuda"
    ):
        """
        For each edge with tensor_info (RDMA source): allocate dst tensor,
        register memory, call engine.transfer_read_on_cuda(), record CUDA event.
        For each edge WITHOUT tensor_info (signal-only): no data to transfer.
        """
        stream = torch.cuda.Stream()
        for graph_edge in graph_edges:
            if len(graph_edge.tensor_info) == 0:
                continue  # signal-only edge, no data to transfer

            logger.debug(
                "Starting to read in %d tensors %s for graph node %s",
                len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node
            )

            for info in graph_edge.tensor_info:
                if info.source_entity == self.my_entity_id or self.tensor_store.check_uuid_presence(
                    request_id, info.uuid
                ): # we already have this tensor!
                    self.tensor_store.increment_ref(
                        request_id, info.uuid, 1 # increment reference while it is being read
                    )
                    continue
                buffer = torch.empty(info.dims, dtype=info.dtype, device=device).as_strided(
                    info.dims, stride=info.stride
                )
                self.tensor_store.put_tensor(
                    request_id=request_id, uuid=info.uuid, tensor=buffer
                )
                self.tensor_store.set_metadata(
                    request_id, info.uuid, mem_registered=True
                )
                self.tensor_store.increment_ref(
                    request_id, info.uuid, 1 # increment reference while it is being read
                )

                if self.protocol == CommProtocol.RDMA:
                    if self.engine is None:
                        raise RuntimeError(
                            "Cannot start RDMA reads: TransferEngine is not available."
                        )
                    self.engine.register_memory(buffer.data_ptr(), info.nbytes)
                else:
                    raise RuntimeError(
                        "Tensor transport for IPC is not implemented yet in this build."
                    )

                with torch.cuda.stream(stream):
                    self.engine.transfer_read_on_cuda(
                        info.source_session_id,
                        buffer.data_ptr(),
                        info.address,
                        info.nbytes,
                        stream.cuda_stream,
                    )
                logger.debug("Started transfer read for uuid %s", info.uuid)
            # For now, have one cuda event for all tensors in this graph edge
            event = torch.cuda.Event()
            event.record(stream)
            self.pending.append(
                EventAndPointers(
                    event=event, graph_edges=[graph_edge], request_id=request_id
                )
            )
