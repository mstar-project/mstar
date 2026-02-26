from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
from mooncake.engine import TransferEngine
import torch


from mminf.communication.communicator import BaseCommunicator, CommProtocol
from mminf.graph.base import GraphPointer, TensorPointerInfo


class TensorCommunicationManager(ABC):
    @abstractmethod
    def register_and_prepare_to_send(
        self, request_id: str, tensors: dict[str, torch.Tensor],
        graph_pointers: list[GraphPointer]
    ):
        """
        Updates graph_pointers with required tensor info (addresses, datatypes,
        num bytes, etc.).
        If relevant (e.g., mooncake rdma), registers buffers.
        """
        pass

    @abstractmethod
    def get_tensor(self, request_id: str, tensor_name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def cleanup(self, request_id: str, tensor_name: str):
        """
        Removes buffer if exists. Unregisters buffers if relevant
        """
        pass

    @abstractmethod
    def start_read_tensors(self, graph_pointers: list[GraphPointer]):
        """
        Initializes empty buffer, initializes a read. May return immediately. 
        """
        pass

    @abstractmethod
    def get_ready_tensors(self) -> dict[str, GraphPointer]:
        """
        Returns request_id: list of the GraphPointers that are currently
        ready for that request
        """
        pass


@dataclass
class NameAndRequestId:
    tensor_name: str
    request_id: str


@dataclass
class EventAndPointers:
    event: torch.cuda.Event
    pointers: list[GraphPointer]


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
        self.tensors: dict[NameAndRequestId, torch.Tensor]
        self.communicator = communicator
        self.protocol = protocol
        self.my_session_id = communicator.get_session_id()

        self.engine = TransferEngine()
        self.engine.initialize(
            hostname,
            metadata_server,
            protocol,
            ""
        )
        self.copy_event = None
        
        # request_id: EventAndPointers
        self.pending: dict[str, EventAndPointers] = {}

    def register_and_prepare_to_send(
        self, request_id: str, tensors: dict[str, torch.Tensor],
        graph_pointers: list[GraphPointer]
    ):
        # get tensor name to graph pointers
        name_to_pointers: dict[str, list[GraphPointer]] = {}
        for pointer in graph_pointers:
            if pointer.name not in name_to_pointers:
                name_to_pointers[pointer.name] = []
            name_to_pointers[pointer.name].append(pointer)

        for name in tensors:
            self.tensors[NameAndRequestId(
                name, request_id=request_id
            )] = tensors[name]

            pointer_info = TensorPointerInfo(
                dims=tensors[name].shape,
                dtype=tensors[name].dtype,
                nbytes=tensors[name].element_size() * tensors[name].nelement(),
                address=tensors[name].data_ptr(),
                source_session_id=self.my_session_id,
                source_entity=self.my_entity_id
            )
            for pointer in name_to_pointers[name]:
                pointer.tensor_info = pointer_info
        
            if self.protocol == CommProtocol.RDMA:
                ret_value = self.engine.register_memory(
                    pointer_info.address, pointer_info.nbytes
                )
                if ret_value != 0:
                    # TODO: error handling
                    raise RuntimeError("Mooncake memory registration failed.")
            
    def cleanup(self, request_id: str, tensor_name: str):
        if self.protocol == CommProtocol.RDMA:
            ret_value = self.engine.unregister_memory(
                self.tensors[NameAndRequestId(
                    tensor_name, request_id
                )].data_ptr()
            )
            if ret_value != 0:
                # TODO: error handling
                raise RuntimeError("Mooncake memory unregistration failed.")

        del self.tensors[NameAndRequestId(
            tensor_name, request_id
        )]
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_tensor(self, request_id: str, tensor_name: str) -> torch.Tensor:
        return self.tensors[NameAndRequestId(
            tensor_name, request_id
        )]

    def get_ready_tensors(self) -> dict[str, GraphPointer]:
        """
        Returns request_id: list of the GraphPointers that are currently
        ready for that request (by querying the events in self.pending)
        """
        pass # TODO Atindra

    def start_read_tensors(self, graph_pointers: list[GraphPointer]):
        """
        Initializes empty buffer, initializes a read. May return immediately. 

        Kicks of the engine transfer read on cuda. Updates self.pending.
        """
        pass # TODO Atindra