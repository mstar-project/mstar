from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None
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
    def start_read_tensors(
        self, request_id: str, graph_pointers: list[GraphPointer]
    ):
        """
        Initializes empty buffer, initializes a read. May return immediately.
        """
        pass

    @abstractmethod
    def get_ready_tensors(self) -> dict[str, list[GraphPointer]]:
        """
        Returns request_id: list of the GraphPointers that are currently
        ready for that request
        """
        pass


@dataclass(frozen=True)
class NameAndRequestId:
    tensor_name: str
    request_id: str


@dataclass
class EventAndPointers:
    event: torch.cuda.Event
    pointers: list[GraphPointer]
    request_id: str = ""


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
        self.tensors: dict[NameAndRequestId, torch.Tensor] = {}
        self.communicator = communicator
        self.protocol = protocol
        self.my_session_id = communicator.get_session_id()

        if TransferEngine is not None:
            self.engine = TransferEngine()
            self.engine.initialize(
                hostname,
                metadata_server,
                protocol,
                ""
            )
        else:
            self.engine = None
        # Per-transfer pending list (allows partial tensor readiness)
        self.pending: list[EventAndPointers] = []

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

    def get_ready_tensors(self) -> dict[str, list[GraphPointer]]:
        """
        Poll CUDA events. Return {request_id: [ready GraphPointers]}.
        Remove completed entries from self.pending.
        """
        ready: dict[str, list[GraphPointer]] = {}
        still_pending = []
        for ep in self.pending:
            if ep.event.query():
                for ptr in ep.pointers:
                    ready.setdefault(ep.request_id, []).append(ptr)
            else:
                still_pending.append(ep)
        self.pending = still_pending
        return ready

    def start_read_tensors(
        self, request_id: str, graph_pointers: list[GraphPointer]
    ):
        """
        For each pointer with tensor_info (RDMA source): allocate dst tensor,
        register memory, call engine.transfer_read_on_cuda(), record CUDA event.
        For each pointer WITHOUT tensor_info (signal-only): no data to transfer.
        """
        stream = torch.cuda.Stream()
        for ptr in graph_pointers:
            if ptr.tensor_info is None:
                continue  # signal-only pointer, no data to transfer

            info = ptr.tensor_info
            dst = torch.empty(info.dims, dtype=info.dtype, device="cuda")
            self.tensors[NameAndRequestId(ptr.name, request_id)] = dst

            if self.protocol == CommProtocol.RDMA:
                self.engine.register_memory(dst.data_ptr(), info.nbytes)

                with torch.cuda.stream(stream):
                    self.engine.transfer_read_on_cuda(
                        info.source_session_id,
                        dst.data_ptr(),
                        info.address,
                        info.nbytes,
                        stream.cuda_stream,
                    )
                event = torch.cuda.Event()
                event.record(stream)
                self.pending.append(
                    EventAndPointers(
                        event=event, pointers=[ptr], request_id=request_id
                    )
                )