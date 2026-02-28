from abc import ABC
from dataclasses import asdict, dataclass, field
from enum import Enum

from mminf.graph.base import GraphPointer, TensorPointerInfo


class Status(Enum):
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class MessageBody(ABC):
    def to_dict(self):
        return asdict(self)
    
    def from_dict(self, input: dict):
        return self(**input)


######################################
# Requests to workers
######################################

class WorkerMessageType(Enum):
    NEW_REQUEST = "new_request"
    REMOVE_REQUEST = "remove_request"
    INPUT_SIGNALS = "input_signals"
    TENSOR_RECEIVED = "tensor_received"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    subgraph_ids: list[str]
    subgraph_to_worker: dict[str, str]
    initial_phase: str
    initial_inputs: list[GraphPointer]


@dataclass
class RemoveRequest(MessageBody):
    request_id: str


@dataclass
class InputSignals(MessageBody):
    request_id: str
    phase: str
    inputs: list[GraphPointer]


@dataclass
class TensorReceived(MessageBody):
    request_id: str
    receiving_entity: str
    successful_tensor_ids: list[str]
    failed_tensor_ids: list[str]


@dataclass
class WorkerMessage:
    message_type: WorkerMessageType
    body: MessageBody


######################################
# Requests to conductor
######################################

class ConductorMessageType(Enum):
    NEW_REQUEST = "new_request"
    SUBGRAPHS_DONE = "subgraphs_done"


@dataclass
class NewRequestConductor(MessageBody):
    request_id: str
    initial_signals: dict[str, TensorPointerInfo]
    initial_input_modalities: list[str]
    initial_output_modalities: list[str]


@dataclass
class SubgraphsDone(MessageBody):
    request_id: str
    subgraph_ids: list[str]
    persist_signals: dict[str, TensorPointerInfo] = field(default_factory=dict)
    new_tokens: list[int] = field(default_factory=list)


@dataclass
class ConductorMessage:
    message_type: ConductorMessageType
    body: MessageBody