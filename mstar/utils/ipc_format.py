from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import PublishedInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.profile.format import RxInfo, TxInfo
from mstar.profile.worker import GraphTimings


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
    DRAIN_REQUEST = "drain_request"
    REMOVE_REQUEST = "remove_request"
    INPUT_SIGNALS = "input_signals"
    UNPERSIST_TENSORS = "unpersist"
    TENSOR_RECEIVED = "tensor_received"
    SCHEDULE_TP = "schedule_tp"
    STOP_LOOPS = "stop_loops"
    TP_NO_SPEC = "tp_no_spec"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    partition_worker_graph_ids: list[int]
    worker_graph_to_workers: dict[int, list[str]]
    initial_inputs: list[GraphEdge]
    request_info: CurrentForwardPassInfo


class MessageSource(IntEnum):
    CONDUCTOR = 0
    TP_RANK_0 = 1
    SELF = 2

@dataclass
class RemoveRequest(MessageBody):
    request_id: str
    source: int = MessageSource.CONDUCTOR
    # Rank 0 forwarding a removal to its followers stamps the last step it had
    # broadcast, so they tear the request down at the same point in the step
    # sequence it did, maintaining KV cache state symmetry.
    after_tp_seq: int = -1


@dataclass
class DrainRequest(MessageBody):
    # Phase-1 teardown: stop reading this request and confirm no reads remain.
    # Hard cleanup (RemoveRequest) follows once every reader has ACKed via READS_DONE.
    request_id: str
    source: int = MessageSource.CONDUCTOR


@dataclass
class InputSignals(MessageBody):
    request_id: str
    inputs: list[GraphEdge]
    request_info: CurrentForwardPassInfo
    partition_name: str = "default"
    # Producer partition names. Declared, not a bare ``set``: a bare one is
    # untyped on the wire, so an empty set a Rust sender writes came back a
    # list.
    producer_done: set[str] = field(default_factory=set)


@dataclass
class TensorReceived(MessageBody):
    request_id: str
    successful_tensors: dict[int, int] # uuid -> graph edge count
    failed_tensor_ids: list[int] # uuids


@dataclass
class UnpersistTensors(MessageBody):
    request_id: str
    uuid_to_ref_count: dict[int, int]

@dataclass
class StopLoops(MessageBody):
    request_id: str
    loop_names: set[str]
    partition_name: str
    loop_stop_times: dict[str, NestedLoopIndices] = field(default_factory=dict)


@dataclass
class OffloadDelta:
    """A rank's offloads and reloads, in the order it made them.

    Ordered so the follower replays the leader's decisions in the same order: a
    sequence that worked for the leader works for the followers too.

    Holds WIRE request ids, not worker-local handles: a handle is minted per
    worker, so rank 0's would name different requests on the follower.

    A dataclass of lists rather than a NamedTuple of deques so that the wire codec
    can automatically encode it. ``pop_left`` is O(n) on a list, which costs nothing
    at the handful of entries a delta ever holds.

    ``__len__`` is the queue depth, so an empty delta is falsy -- several callers
    lean on that.
    """

    # parallel lists: rids[i] was offloaded if is_offload[i], else reloaded
    rids: list[str] = field(default_factory=list)
    is_offload: list[bool] = field(default_factory=list)

    @classmethod
    def new(cls) -> "OffloadDelta":
        return cls()

    def add_offloaded(self, rid: str):
        self.rids.append(rid)
        self.is_offload.append(True)

    def add_reloaded(self, rid: str):
        self.rids.append(rid)
        self.is_offload.append(False)

    def __len__(self):
        return len(self.rids)

    def peek_left(self) -> tuple[str, bool] | None:
        if not len(self):
            return
        return (self.rids[0], self.is_offload[0])

    def pop_left(self) -> tuple[str, bool] | None:
        if not len(self):
            return None
        return (self.rids.pop(0), self.is_offload.pop(0))

    def copy(self) -> "OffloadDelta":
        """An independent queue holding the same moves.

        One per follower: each replays at its own pace, popping as it goes.
        """
        return OffloadDelta(list(self.rids), list(self.is_offload))

    def take(self) -> "OffloadDelta":
        """Hand the queue over and leave this one empty.

        Copies and clears rather than rebinding, so a holder of this delta (the
        engine keeps its journal in a dict) sees it emptied.
        """
        taken = self.copy()
        self.rids.clear()
        self.is_offload.clear()
        return taken

    def extend(self, other: "OffloadDelta") -> None:
        self.rids.extend(other.rids)
        self.is_offload.extend(other.is_offload)


@dataclass
class ScheduleTPNode(MessageBody):
    node_name: str
    graph_walk: str
    request_ids: list[str]
    speculative: bool = False
    spec_seq: int = -1
    spec_from_seq: int = -1
    resident_delta: OffloadDelta = field(default_factory=OffloadDelta.new)
    # under a combined walk, request_ids[i] runs walks[walk_idx[i]]; empty
    # means every rid runs graph_walk
    walks: list[str] = field(default_factory=list)
    walk_idx: list[int] = field(default_factory=list)
    # per request_ids entry, the chunk this step runs ([start, end), -1 when
    # not chunked; empty when none is), and the rows that need another chunk
    chunk_starts: list[int] = field(default_factory=list)
    chunk_ends: list[int] = field(default_factory=list)
    incomplete_node_rids: list[str] = field(default_factory=list)


@dataclass
class TPNoSpeculation(MessageBody):
    node_name: str
    graph_walk: str
    spec_from_seq: int

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
    ABORT_REQUEST = "abort_request"
    FAIL_REQUESTS = "fail_requests"
    READS_DONE = "reads_done"


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
    worker_graph_ids: list[int]
    is_first_tp_rank: bool
    persist_signals: dict[str, list[TensorPointerInfo]] = field(default_factory=dict)
    new_token_counts: dict[str, int] = field(default_factory=dict) # name to token counts
    output_signal_names: list[str] = field(default_factory=list)
    resource_publish_info: dict[str, PublishedInfo] = field(default_factory=dict)
    partition_name: str = field(default="default")
    partition_done: bool = field(default=False)
    stream_tokens_consumed: dict[str, int] = field(default_factory=dict)  # edge_name -> tokens consumed from stream
    output_loop_indices: dict[str, NestedLoopIndices] = field(default_factory=dict)
    graph_timings: GraphTimings = field(default_factory=dict)
    rx_info: list[RxInfo] = field(default_factory=list)
    tx_info: list[TxInfo] = field(default_factory=list)


@dataclass
class SetupDone(MessageBody):
    worker_id: str


@dataclass
class AbortRequest(MessageBody):
    request_id: str


@dataclass
class ReadsDone(MessageBody):
    """An entity confirming it has no in-flight reads for a request and will
    start none — the conductor's gate before sending the hard RemoveRequest."""
    request_id: str
    entity_id: str


@dataclass
class FailRequests(MessageBody):
    """A worker reporting requests it can no longer serve.

    ``errors`` maps request_id -> message. It's a dict rather than a
    (rids, message) pair because per-rid stages (prepare_inputs,
    postprocess) attribute a distinct error to each request, and one
    step can fail several of them for different reasons.
    """
    errors: dict[str, str]


@dataclass
class ConductorMessage:
    message_type: ConductorMessageType
    body: MessageBody
