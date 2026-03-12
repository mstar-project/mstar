import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import uuid4

try:
    from mooncake.engine import TransferEngine
except Exception as _err:
    MOONCAKE_IMPORT_ERROR = _err
    TransferEngine = None
else:
    MOONCAKE_IMPORT_ERROR = None
import torch

from mminf.communication.communicator import BaseCommunicator, CommProtocol
from mminf.graph.base import GraphPointer, TensorPointerInfo
from mminf.ipc_formats import NameAndUuid, TensorReceived, WorkerMessage, WorkerMessageType

logger = logging.getLogger(__name__)


NameToTensorList = dict[str, list[torch.Tensor]]
UuidToTensor = dict[str, torch.Tensor]


@dataclass(frozen=True)
class NameAndRequestId:
    tensor_name: str
    request_id: str


@dataclass
class EventAndPointers:
    event: torch.cuda.Event
    pointers: list[GraphPointer]
    request_id: str = ""


class TensorStore:
    def __init__(self):
        self.stored_tensors: dict[NameAndRequestId, UuidToTensor] = {}

    def get_first_tensor(self, request_id: str, name: str):
        return list(self.stored_tensors[NameAndRequestId(
            tensor_name=name, request_id=request_id
        )].values())[0]

    def get_tensor(self, request_id: str, name: str, uuid: str):
        return self.stored_tensors[NameAndRequestId(
            tensor_name=name, request_id=request_id
        )][uuid]

    def put_tensor(self, request_id: str, name: str, uuid: str, tensor: torch.Tensor):
        key = NameAndRequestId(
            tensor_name=name, request_id=request_id
        )
        if key not in self.stored_tensors:
            self.stored_tensors[key] = {}
        self.stored_tensors[key][uuid] = tensor

    def check_uuid_presence(self, request_id: str, name: str, uuid: str):
        return uuid in self.stored_tensors.get(NameAndRequestId(
            tensor_name=name, request_id=request_id
        ), {})

    def check_name_presence(self, request_id: str, name: str):
        return NameAndRequestId(
            tensor_name=name, request_id=request_id
        ) in self.stored_tensors

    def remove_tensor(self, request_id: str, name: str, uuid: str):
        del self.stored_tensors[NameAndRequestId(
            tensor_name=name, request_id=request_id
        )][uuid]

    def get_all_uuids(self, request_id: str, name: str):
        return list(self.stored_tensors[NameAndRequestId(
            tensor_name=name, request_id=request_id
        )].keys())


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
        graph_pointers: list[GraphPointer]
    ):
        """
        Updates graph_pointers with required tensor info (addresses, datatypes,
        num bytes, etc.) and UUID.
        """
        pass

    @abstractmethod
    def register_for_send(
        self, request_id: str, name: str, uuids: list[str]
    ):
        """
        If relevant (e.g., mooncake rdma), registers buffers.
        """
        pass

    @abstractmethod
    def get_tensor(self, request_id: str, tensor_name: str, uuid: str=None) -> torch.Tensor:
        pass

    @abstractmethod
    def cleanup(self, request_id: str, tensor_name: str, uuids: list[str] | None=None):
        """
        Removes buffer if exists. Unregisters buffers if relevant
        """
        pass

    @abstractmethod
    def cleanup_request(self, request_id: str):
        """
        Removes all tensors for a given request. Unregisters buffers if relevant.
        """
        pass

    @abstractmethod
    def start_read_tensors(
        self, request_id: str, graph_pointers: list[GraphPointer]
    ) -> list[int]:
        """
        Initializes empty buffer, initializes a read. May return immediately.
        """
        pass

    @abstractmethod
    def get_ready_tensors(self) -> dict[str, list[GraphPointer]]: # uuid based
        """
        Returns request_id: list of the GraphPointers that are currently
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

        # NOTE: none of the designs for how to handle the values of self.tensors
        # being a list are set in stone. This is a "reasonable initial
        # implementation," but should change based on what makes sense for
        # downstream changes (right now the code is "leaving breadcrumbs
        # for a hypothetical thinker-talker", but we will have a better idea of
        # what this should look like while implementing the actual thinker-
        # talker relay)

        # internal dict is uuid to tensor. this is morally the list of tensors
        # that we keep as a dict for easy indexing
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
        # Track UUIDs registered for remote send, awaiting TENSOR_RECEIVED ACK.
        # cleanup_request() skips these so buffers stay alive until the
        # receiver confirms it finished reading.
        self.registered_for_send: dict[str, set[str]] = {}  # request_id -> set of UUIDs
        self.registered_for_recv: dict[str, set[str]] = {}

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
                    name=name,
                    uuid=tensor_uuid,
                    tensor=tensor
                )
                logger.debug("Storing tensor name %s uuid %s", name, tensor_uuid)
                new_tensor_info = TensorPointerInfo(
                    dims=tensor.shape,
                    dtype=tensor.dtype,
                    stride=tensor.stride(),
                    nbytes=tensor.element_size() * tensor.nelement(),
                    address=tensor.data_ptr(),
                    uuid=tensor_uuid,
                    source_session_id=self.my_session_id,
                    source_entity=self.my_entity_id,
                    source_tensor_name=name
                )
                tensor_info[name].append(new_tensor_info)
        return tensor_info

    def register_for_send(self, request_id, name, uuids):
        for uuid in uuids:
            if self.protocol == CommProtocol.RDMA:
                if self.engine is None:
                    raise RuntimeError(
                        "Cannot register tensors for RDMA send: TransferEngine is not available."
                    )
                tensor = self.tensor_store.get_tensor(
                    request_id=request_id, name=name, uuid=uuid
                )
                ret_value = self.engine.register_memory(
                    tensor.data_ptr(), tensor.element_size() * tensor.nelement()
                )
                if ret_value != 0:
                    # TODO: error handling
                    raise RuntimeError(f"Mooncake memory registration failed for request id {request_id}, tensor {name}, uuid {uuid}.")
            self.registered_for_send.setdefault(request_id, set()).add(uuid)

    def store_and_populate_graph_edges(
        self, request_id: str,
        tensors: NameToTensorList,
        graph_pointers: list[GraphPointer]
    ):
        # get tensor name to graph pointers
        name_to_pointers: dict[str, list[GraphPointer]] = {}
        for pointer in graph_pointers:
            if pointer.name not in name_to_pointers:
                name_to_pointers[pointer.name] = []
            name_to_pointers[pointer.name].append(pointer)

        pointer_info = self.store_and_return_tensor_info(
            request_id=request_id, tensors=tensors
        )
        for name in tensors:
            logger.debug(
                "Storing tensor %s (uuids %s) for stages %s",
                name, str([info.uuid for info in pointer_info[name]]),
                str([edge.name for edge in name_to_pointers.get(name, [])])
            )
            for pointer in name_to_pointers.get(name, []):
                pointer.tensor_info = pointer_info[name]

    def _cleanup_by_uuid(
        self, request_id: str, tensor_name: str, uuid: str
    ):
        logger.debug("Deleting tensor name %s uuid %s", tensor_name, uuid)
        req_id_name_uuid = dict(
            request_id=request_id, name=tensor_name, uuid=uuid
        )
        if not self.tensor_store.check_uuid_presence(**req_id_name_uuid):
            logger.warning("Trying to cleanup tensor %s:%s, but uuid not found", tensor_name, uuid)
            return
        if self.protocol == CommProtocol.RDMA and self.engine is not None \
                and (uuid in self.registered_for_send.get(request_id, set()) \
                    or uuid in self.registered_for_recv.get(request_id, set())):
            ret_value = self.engine.unregister_memory(
                self.tensor_store.get_tensor(**req_id_name_uuid).data_ptr()
            )
            if ret_value != 0:
                raise RuntimeError("Mooncake memory unregistration failed.")
        self.tensor_store.remove_tensor(**req_id_name_uuid)
        # Remove from send-tracking so cleanup_request() knows this UUID is done
        if request_id in self.registered_for_send:
            self.registered_for_send[request_id].discard(uuid)
            if not self.registered_for_send[request_id]:
                del self.registered_for_send[request_id]
        
        if request_id in self.registered_for_recv:
            self.registered_for_recv[request_id].discard(uuid)
            if not self.registered_for_recv[request_id]:
                del self.registered_for_recv[request_id]

    def cleanup(self, request_id: str, tensor_name: str, uuids: list[str] | None=None):
        if not self.tensor_store.check_name_presence(
            request_id=request_id, name=tensor_name
        ):
            logger.warning("Trying to cleanup tensor %s, but tensor not found", tensor_name)
            return

        # By default, cleanup all tensors with the given key, unless the address
        # argument is provided
        if uuids is None:
            uuids = self.tensor_store.get_all_uuids(
                request_id=request_id, name=tensor_name
            )
        for uuid in uuids:
            self._cleanup_by_uuid(request_id, tensor_name, uuid)

    def cleanup_request(self, request_id: str):
        pending_send_uuids = self.registered_for_send.get(request_id, set())
        names_to_remove = [
            key.tensor_name for key in self.tensor_store.stored_tensors \
                if key.request_id == request_id
        ]

        for name in names_to_remove:
            for uuid in self.tensor_store.get_all_uuids(
                request_id=request_id, name=name
            ):
                if uuid in pending_send_uuids:
                    logger.debug(
                        "Deferring cleanup of tensor %s uuid %s "
                        "(awaiting TENSOR_RECEIVED ACK)",
                        name, uuid
                    )
                    continue
                self._cleanup_by_uuid(request_id, name, uuid)
        # Also remove any pending transfers for this request
        self.pending = [
            ep for ep in self.pending if ep.request_id != request_id
        ]

    def get_tensor(self, request_id: str, tensor_name: str, uuid: str=None) -> torch.Tensor:
        if uuid is None:
            return self.tensor_store.get_first_tensor(
                request_id=request_id, name=tensor_name
            )
        return self.tensor_store.get_tensor(
            request_id=request_id, name=tensor_name, uuid=uuid
        )

    def get_ready_tensors(self) -> dict[str, list[GraphPointer]]:
        """
        Poll CUDA events. Return {request_id: [ready GraphPointers]}.
        Remove completed entries from self.pending.
        Sends TENSOR_RECEIVED ACKs back to senders so they can free buffers.
        """

        # request_id -> ready graph pointers
        ready: dict[str, list[GraphPointer]] = {}
        still_pending = []
        # Collect ACKs to send: (source_entity, request_id) -> tensor_names
        acks: dict[tuple[str, str], list[NameAndUuid]] = {}

        for ep in self.pending:
            if ep.event.query():
                for ptr in ep.pointers:
                    ready.setdefault(ep.request_id, []).append(ptr)
                    logger.debug(
                        "Finished reading in %d tensors %s for graph node %s",
                        len(ptr.tensor_info), ptr.name, ptr.next_stage
                    )

                    for tensor_info in  ptr.tensor_info:
                        key = (tensor_info.source_entity, ep.request_id)
                        acks.setdefault(key, []).append(
                            NameAndUuid(
                                tensor_id=tensor_info.source_tensor_name,
                                uuid=tensor_info.uuid
                            ))
            else:
                still_pending.append(ep)
        self.pending = still_pending

        # Send ACKs to senders
        for (source_entity, request_id), tensor_name_uuid in acks.items():
            if source_entity == self.my_entity_id:
                continue  # local transfer, no ACK needed
            self.communicator.send(
                source_entity,
                WorkerMessage(
                    message_type=WorkerMessageType.TENSOR_RECEIVED,
                    body=TensorReceived(
                        request_id=request_id,
                        successful_tensors=tensor_name_uuid,
                        failed_tensor_ids=[], # TODO: handle failed transfers
                    ),
                ),
            )

        return ready

    def start_read_tensors(
        self, request_id: str, graph_pointers: list[GraphPointer],
        device: str="cuda"
    ):
        """
        For each pointer with tensor_info (RDMA source): allocate dst tensor,
        register memory, call engine.transfer_read_on_cuda(), record CUDA event.
        For each pointer WITHOUT tensor_info (signal-only): no data to transfer.
        """
        stream = torch.cuda.Stream()
        for graph_ptr in graph_pointers:
            if len(graph_ptr.tensor_info) == 0:
                continue  # signal-only pointer, no data to transfer

            logger.debug(
                "Starting to read in %d tensors %s for graph node %s",
                len(graph_ptr.tensor_info), graph_ptr.name, graph_ptr.next_stage
            )

            for info in graph_ptr.tensor_info:
                if info.source_entity == self.my_entity_id:
                    if info.source_tensor_name == graph_ptr.name:
                        continue
                    tensor = self.tensor_store.get_tensor(
                        request_id=request_id, name=info.source_tensor_name,
                        uuid=info.uuid
                    )
                    self.tensor_store.put_tensor(
                        request_id=request_id, name=graph_ptr.name,
                        uuid=info.uuid, tensor=tensor
                    )
                    self.tensor_store.remove_tensor(
                        request_id=request_id, name=info.source_tensor_name,
                        uuid=info.uuid
                    )
                    continue
                buffer = torch.empty(info.dims, dtype=info.dtype, device=device).as_strided(
                    info.dims, stride=info.stride
                )
                self.tensor_store.put_tensor(
                    request_id=request_id, name=graph_ptr.name,
                    uuid=info.uuid, tensor=buffer
                )
                self.registered_for_recv.setdefault(request_id, set()).add(info.uuid)

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
            event.record(stream) ##TODO @Atindra : should this be placed here or up? ##
            self.pending.append(
                EventAndPointers(
                    event=event, pointers=[graph_ptr], request_id=request_id
                )
            )
