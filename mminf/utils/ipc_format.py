from dataclasses import asdict, dataclass, field
from enum import Enum

from mminf.conductor.request_info import CurrentForwardPassInfo, SequenceInfo
from mminf.graph.base import GraphEdge, TensorPointerInfo


class Status(Enum):
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class MessageBody:
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
    KV_TRANSFER_LAYER = "kv_transfer_layer"
    KV_TRANSFER_META = "kv_transfer_meta"
    INPUT_SIGNALS = "input_signals"
    UNPERSIST_TENSORS = "unpersist"
    TENSOR_RECEIVED = "tensor_received"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    worker_graph_ids: list[str]
    worker_graph_to_worker: dict[str, str]
    initial_inputs: list[GraphEdge]
    request_info: CurrentForwardPassInfo


@dataclass
class RemoveRequest(MessageBody):
    request_id: str


@dataclass
class InputSignals(MessageBody):
    request_id: str
    inputs: list[GraphEdge]
    request_info: CurrentForwardPassInfo


@dataclass
class TensorReceived(MessageBody):
    request_id: str
    successful_tensors: dict[str, int] # uuid -> graph edge count
    failed_tensor_ids: list[str] # uuids


@dataclass
class UnpersistTensors(MessageBody):
    request_id: str
    uuid_to_ref_count: dict[str, int]

@dataclass
class WorkerMessage:
    message_type: WorkerMessageType
    body: MessageBody


######################################
# Requests to conductor
######################################

class ConductorMessageType(Enum):
    NEW_REQUEST = "new_request"
    WORKER_GRAPHS_DONE = "worker_graphs_done"


@dataclass
class NewRequestConductor(MessageBody):
    request_id: str
    initial_signals: dict[str, list[TensorPointerInfo]]
    initial_input_modalities: list[str]
    initial_output_modalities: list[str]
    input_metadata: dict[str, list[dict]]
    model_kwargs: dict


@dataclass
class WorkerGraphsDone(MessageBody):
    request_id: str
    worker_graph_ids: list[str]
    persist_signals: dict[str, list[TensorPointerInfo]] = field(default_factory=dict)
    new_tokens: dict[str, list[int]] = field(default_factory=dict) # name to tokens
    output_signal_names: int = field(default=0)
    per_label_seq_info: dict[str, SequenceInfo] = field(default_factory=dict)


@dataclass
class ConductorMessage:
    message_type: ConductorMessageType
    body: MessageBody
