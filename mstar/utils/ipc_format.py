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
    # sequence it did. A teardown releases pages, and a follower that applies it
    # on the other side of a step admits that step against different page state —
    # which is a deadlock, and invisible in the resident-set delta because the
    # request is not offloaded, it is gone. ``-1`` for the unordered paths (the
    # conductor's own sends, and a rank's own deferred re-apply).
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

    Ordered, and that is the whole point: a set of offloads plus a set of
    reloads cannot say whether a request ended up resident. Offload A, reload A,
    offload A replays from sets as "resident", when the rank that recorded it has
    A on the host. Replaying the sequence cannot get that wrong.

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

    def describe(self, keep: int = 8) -> str:
        """The moves in replay order, for a log line.

        Order is the only thing that makes a delta meaningful, so this prints the
        sequence rather than counts — ``-A +A`` and ``+A -A`` leave a rank in
        opposite states. Truncated from the left: when a replay stalls it is the
        moves still owed at the front that say why.
        """
        if not len(self):
            return "empty"
        moves = [
            f"{'-' if off else '+'}{rid[:8]}"
            for rid, off in zip(self.rids, self.is_offload, strict=True)
        ]
        if len(moves) <= keep:
            return " ".join(moves)
        return " ".join(moves[:keep]) + f" (+{len(moves) - keep} more)"


@dataclass
class ScheduleTPNode(MessageBody):
    node_name: str
    graph_walk: str
    request_ids: list[str]
    speculative: bool = False
    spec_seq: int = -1
    spec_from_seq: int = -1
    resident_delta: OffloadDelta = field(default_factory=OffloadDelta.new)
    # What rank 0's page state was when it chose this step, for a follower to
    # check itself against once it has replayed ``resident_delta``. The delta
    # says what rank 0 did; these say where it should have landed.
    #
    # Both sets, because they catch different skews. ``offloaded_after`` catches
    # a delta that did not replay. ``holding_after`` catches page movement the
    # delta never describes: a request torn down releases its pages, and if the
    # ranks apply that teardown on opposite sides of this step their admits
    # disagree while their offloaded sets look identical.
    #
    # Requests that HOLD pages, not requests that are live. A request that has
    # been admitted but never run holds nothing, so it cannot shift an admit —
    # and new arrivals reach the ranks at slightly different times, so comparing
    # live sets reports that harmless skew as a fault.
    offloaded_after: tuple[str, ...] = ()
    holding_after: tuple[str, ...] = ()


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
