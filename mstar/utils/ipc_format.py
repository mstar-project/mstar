from dataclasses import asdict, dataclass, field
from enum import Enum

from mstar.conductor.request_info import CurrentForwardPassInfo, PerLabelSeqInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices


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
    INPUT_SIGNALS = "input_signals"
    PRODUCER_DONE = "producer_done"
    UNPERSIST_TENSORS = "unpersist"
    TENSOR_RECEIVED = "tensor_received"
    SCHEDULE_TP = "schedule_tp"
    STOP_LOOPS = "stop_loops"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    partition_worker_graph_ids: list[str]
    worker_graph_to_workers: dict[str, list[str]]
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
    partition_name: str = "default"


@dataclass
class ProducerDone(MessageBody):
    request_id: str
    partition_name: str
    producer_done: set = field(default_factory=set)


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
class StopLoops(MessageBody):
    request_id: str
    loop_names: set[str]
    partition_name: str
    loop_stop_times: dict[str, NestedLoopIndices] = field(default_factory=dict)


@dataclass
class ScheduleTPNode(MessageBody):
    node_name: str
    graph_walk: str
    request_ids: list[str]

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
    SETUP_DONE = "setup_done"


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
    is_first_tp_rank: bool
    persist_signals: dict[str, list[TensorPointerInfo]] = field(default_factory=dict)
    new_tokens: dict[str, list[int]] = field(default_factory=dict) # name to tokens
    output_signal_names: int = field(default=0)
    new_produced_edge_idx: dict[str, int] = field(default_factory=dict)
    new_consumed_edge_idx: dict[str, int] = field(default_factory=dict)
    consumer_graph_walk_transitions: dict[str, str] = field(default_factory=dict)
    per_label_seq_info: PerLabelSeqInfo = field(default_factory=PerLabelSeqInfo)
    partition_name: str = field(default="default")
    partition_done: bool = field(default=False)
    # the graph walk this partition's just-completed forward pass ran under;
    # used by the conductor to track a producer-triggered partition's walk
    partition_graph_walk: str | None = field(default=None)
    stream_tokens_consumed: dict[str, int] = field(default_factory=dict)  # edge_name -> tokens consumed from stream
    output_loop_indices: dict[str, NestedLoopIndices] = field(default_factory=dict)


@dataclass
class SetupDone(MessageBody):
    worker_id: str

@dataclass
class ConductorMessage:
    message_type: ConductorMessageType
    body: MessageBody
