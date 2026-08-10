from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum

from mstar.conductor.request_info import CurrentForwardPassInfo, PerLabelSeqInfo
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
    REMOVE_REQUEST = "remove_request"
    INPUT_SIGNALS = "input_signals"
    UNPERSIST_TENSORS = "unpersist"
    TENSOR_RECEIVED = "tensor_received"
    SCHEDULE_TP = "schedule_tp"
    STOP_LOOPS = "stop_loops"
    # TP async scheduling: void a broadcast speculation. Separate type rather
    # than a flag on SCHEDULE_TP because it must be dispatchable without
    # touching the schedule FIFO — see CancelSpec.
    CANCEL_SPEC = "cancel_spec"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    partition_worker_graph_ids: list[str]
    worker_graph_to_workers: dict[str, list[str]]
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


@dataclass
class InputSignals(MessageBody):
    request_id: str
    inputs: list[GraphEdge]
    request_info: CurrentForwardPassInfo
    partition_name: str = "default"
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
    """Leader → followers: schedule this node for these rids.

    Deliberately THIN — ids only. Followers rebuild the batch locally from
    replicated state (``register_tp_follow`` → ``_try_schedule_tp_follow``),
    so nothing tensor-shaped rides the wire.

    ``speculative``/``spec_seq`` extend the same message for TP async
    scheduling rather than opening a second channel: a speculative head is the
    same schedule request, tagged. ``spec_seq`` is the leader's monotonic
    speculation counter, echoed by ``CancelSpec`` so a void names exactly one
    batch.

    Both fields are defaulted so every existing construction site keeps working
    unchanged and today's serial broadcasts stay non-speculative by omission.
    Note this is *source* compatibility, not wire compatibility: these travel
    as pickled dataclasses, and an old pickle would leave the new attributes
    unset rather than defaulted. That is fine here — the ranks of a TP group are
    launched together from one conductor and always run the same build — but it
    is not a mixed-version guarantee, so don't lean on it as one.
    """

    node_name: str
    graph_walk: str
    request_ids: list[str]
    speculative: bool = False
    spec_seq: int = -1


@dataclass
class CancelSpec(MessageBody):
    """Leader → followers: VOID speculation ``spec_seq`` — never retract it.

    Void means: the batch still executes on every rank, and only its *effects*
    are discarded (outputs dropped, speculatively-allocated pages freed, leader
    reschedules authoritatively afterward).

    It does NOT mean "remove it from the pending FIFO if it hasn't launched
    yet." That reading is refuted, not merely discouraged: once ``SpecBatch(S)``
    is broadcast, a rank that drops S while a faster rank already ran it posts a
    different collective sequence — invariant I1, the NCCL-hang class. The CPU
    model checker finds it as a ~16-action counterexample on the
    ``structural-cancel`` scenario (``test/modular/tp_async_sim.py``, mode
    ``B2_RETRACT``, kept as a pinned negative control).

    Under B1 (gated commit) retraction *is* safe, because the launch gate
    guarantees no rank has executed — but B1 signals commit, not cancel, so this
    message keeps void-only semantics in both variants.
    """

    spec_seq: int


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
    new_token_counts: dict[str, int] = field(default_factory=dict) # name to token counts
    output_signal_names: int = field(default=0)
    per_label_seq_info: PerLabelSeqInfo = field(default_factory=PerLabelSeqInfo)
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
