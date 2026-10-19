from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import NamedTuple

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
    TEARDOWN_SESSION = "teardown_session"


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
    # Rank 0's ordering stamp for the session this request hands its state back
    # to, increasing with each session that parks state. A rank reclaiming
    # parked state offloads the smallest stamp first, so taking the number from
    # rank 0 rather than counting locally is what makes every rank pick the
    # same sessions in the same order. -1 when the request has no session.
    session_idle_seq: int = -1
    # Rank 0's resident-set moves since its last broadcast, per node, replayed
    # just before the removal; see ``OffloadDelta``.
    resident_deltas: dict[str, "OffloadDelta"] = field(default_factory=dict)


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


class ResidentMove(NamedTuple):
    name: str  # a wire request id, or a session id when ``is_session``
    is_offload: bool
    is_session: bool


@dataclass
class OffloadDelta:
    """A rank's offloads and reloads, in the order it made them.

    Ordered so the follower replays the leader's decisions in the same order: a
    sequence that worked for the leader works for the followers too.

    Holds WIRE request ids, not worker-local handles: a handle is minted per
    worker, so rank 0's would name different requests on the follower. A
    session entry names the session instead: a request waiting to adopt its
    session's state has rank 0 bring that state back and hold it for it, which
    a follower can replay whether or not it has the request yet.

    Rank 0 sends what it journals with its next broadcast, or with a forwarded
    removal or teardown made before that, so the follower replays every move in
    rank 0's order against the changes those make.

    A dataclass of lists rather than a NamedTuple of deques so that the wire codec
    can automatically encode it. ``pop_left`` is O(n) on a list, which costs nothing
    at the handful of entries a delta ever holds.

    ``__len__`` is the queue depth, so an empty delta is falsy -- several callers
    lean on that.
    """

    # parallel lists: names[i] was offloaded if is_offload[i], else reloaded;
    # a session rather than a request if is_session[i]
    rids: list[str] = field(default_factory=list)
    is_offload: list[bool] = field(default_factory=list)
    is_session: list[bool] = field(default_factory=list)

    @classmethod
    def new(cls) -> "OffloadDelta":
        return cls()

    def _add(self, name: str, is_offload: bool, is_session: bool = False):
        self.rids.append(name)
        self.is_offload.append(is_offload)
        self.is_session.append(is_session)

    def add_offloaded(self, rid: str):
        self._add(rid, True)

    def add_reloaded(self, rid: str):
        self._add(rid, False)

    def add_session_held(self, session_id: str):
        """The session's parked state was brought back and held for the
        request about to adopt it."""
        self._add(session_id, False, is_session=True)

    def __len__(self):
        return len(self.rids)

    def peek_left(self) -> ResidentMove | None:
        if not len(self):
            return None
        return ResidentMove(self.rids[0], self.is_offload[0], self.is_session[0])

    def pop_left(self) -> ResidentMove | None:
        if not len(self):
            return None
        return ResidentMove(
            self.rids.pop(0), self.is_offload.pop(0), self.is_session.pop(0),
        )

    def copy(self) -> "OffloadDelta":
        """An independent queue holding the same moves.

        One per follower: each replays at its own pace, popping as it goes.
        """
        return OffloadDelta(
            list(self.rids), list(self.is_offload), list(self.is_session),
        )

    def take(self) -> "OffloadDelta":
        """Hand the queue over and leave this one empty.

        Copies and clears rather than rebinding, so a holder of this delta (the
        engine keeps its journal in a dict) sees it emptied.
        """
        taken = self.copy()
        self.rids.clear()
        self.is_offload.clear()
        self.is_session.clear()
        return taken

    def extend(self, other: "OffloadDelta") -> None:
        self.rids.extend(other.rids)
        self.is_offload.extend(other.is_offload)
        self.is_session.extend(other.is_session)


@dataclass
class TeardownSession(MessageBody):
    """Free everything a session holds. Sent to every worker that ran it; each
    ACKs with SESSION_TORN_DOWN once its state is gone.

    A TP follower ignores the conductor's copy and acts on rank 0's forward,
    stamped like a forwarded RemoveRequest: freeing the session's pages has to
    land at the same point in the step sequence on every rank."""
    session_id: str
    source: int = MessageSource.CONDUCTOR
    after_tp_seq: int = -1
    # as on RemoveRequest: replayed just before the teardown
    resident_deltas: dict[str, "OffloadDelta"] = field(default_factory=dict)


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
    TEARDOWN_SESSION = "teardown_session"
    SESSION_TORN_DOWN = "session_torn_down"


@dataclass
class NewRequestConductor(MessageBody):
    request_id: str
    initial_signals: dict[str, list[TensorPointerInfo]]
    initial_input_modalities: list[str]
    initial_output_modalities: list[str]
    input_metadata: dict[str, list[dict]]
    model_kwargs: dict
    # The session this request belongs to, as the API server validated it:
    # which one, whether it continues state that is already there, and whether
    # it is the last request in it.
    session_id: str | None = None
    resumed: bool = False
    end_session: bool = False
    # The client's request_id. ``request_id`` is the API server's key for this
    # one submission, never reused; the seed still derives from the client's.
    client_request_id: str | None = None


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
class SessionTornDown(MessageBody):
    """A worker (or the conductor, to the API server) confirming a session's
    state is gone. The API server holds a tombstone until it arrives."""
    session_id: str
    entity_id: str = ""


@dataclass
class FailRequests(MessageBody):
    """A worker reporting requests it can no longer serve.

    ``errors`` maps request_id -> message. It's a dict rather than a
    (rids, message) pair because per-rid stages (prepare_inputs,
    postprocess) attribute a distinct error to each request, and one
    step can fail several of them for different reasons.

    ``statuses`` gives the HTTP status a request fails with, where it is not
    a 500 (a session whose state was dropped is a 410).
    """
    errors: dict[str, str]
    statuses: dict[str, int] = field(default_factory=dict)


@dataclass
class ConductorMessage:
    message_type: ConductorMessageType
    body: MessageBody
