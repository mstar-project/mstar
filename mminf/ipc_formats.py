from abc import ABC
from dataclasses import asdict, dataclass
from enum import Enum

from mminf.graph.base import SignalToDests, SignalToDestsAndFlags
from mminf.model.base import TensorData

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
    INPUT_TENSORS = "input_tensors"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    subgraph_ids: list[str]
    subgraph_to_worker: dict[str, str]
    initial_phase: str
    initial_inputs: SignalToDestsAndFlags
    # TODO: actual tensors will be transferred via Ray. This metadata may be
    # transferred via Ray as well, but we are using ZMQ for the sake of
    # initial testing.


@dataclass
class RemoveRequest(MessageBody):
    request_id: str


@dataclass
class InputTensors(MessageBody):
    request_id: str
    phase: str
    inputs: SignalToDests
    # TODO: actual tensors to be transferred via Ray


@dataclass
class WorkerMessage:
    request_type: WorkerMessageType
    request_body: MessageBody


######################################
# Requests to conductor
######################################

class ConductorMessageType(Enum):
    NEW_REQUEST = "new_request"
    TENSORS = "tensors"
    SUBGRAPHS_DONE = "subgraphs_done"


@dataclass
class NewRequestConductor(MessageBody):
    request_id: str
    initial_inputs: dict[str, TensorData]
    initial_input_modalities: list[str]
    initial_output_modalities: list[str]
    # TODO: transfer initial input data


@dataclass
class ConductorTensors(MessageBody):
    request_id: str
    tensors: SignalToDestsAndFlags
    # TODO: transfer actual tensor data

@dataclass
class SubgraphsDone(MessageBody):
    request_id: str
    subgraph_ids: list[str]


@dataclass
class ConductorMessage:
    request_type: ConductorMessageType
    request_body: MessageBody