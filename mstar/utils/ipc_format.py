from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import NamedTuple

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import PublishedInfo
from mstar.engine.resources.base import CachedLease, MoveKind
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.observability.recorders import MetricSnapshot
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
    EDGE_ITEMS = "edge_items"
    UNPERSIST_TENSORS = "unpersist"
    TENSOR_RECEIVED = "tensor_received"
    SCHEDULE_TP = "schedule_tp"
    STOP_LOOPS = "stop_loops"
    TP_NO_SPEC = "tp_no_spec"
    ENABLE_METRICS = "enable_metrics"
    CLIENT_STREAM_ROUTE = "client_stream_route"
    LOAD_LORA_ADAPTER = "load_lora_adapter"
    UNLOAD_LORA_ADAPTER = "unload_lora_adapter"


@dataclass
class NewRequest(MessageBody):
    request_id: str
    partition_worker_graph_ids: list[int]
    worker_graph_to_workers: dict[int, list[str]]
    initial_inputs: list[GraphEdge]
    request_info: CurrentForwardPassInfo
    loop_caps: dict[str, int]


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
    # Phase-1 teardown: stop scheduling this request, and ACK READS_DONE once no
    # read of it is in flight, starting none after (the data worker also stops
    # reading its results and sending its client stream items). Hard cleanup
    # (RemoveRequest) follows once every reader has ACKed.
    request_id: str
    source: int = MessageSource.CONDUCTOR


@dataclass
class InputSignals(MessageBody):
    request_id: str
    inputs: list[GraphEdge]
    # None for a signal with no non-streaming input, which reads none
    request_info: CurrentForwardPassInfo | None
    partition_name: str = "default"
    # Producer partition names. Declared, not a bare ``set``: a bare one is
    # untyped on the wire, so an empty set a Rust sender writes came back a
    # list.
    producer_done: set[str] = field(default_factory=set)
    # Stream edge -> items its producer sent this worker in total. The stream
    # ends only once that many have arrived.
    producer_done_items: dict[str, int] = field(default_factory=dict)


@dataclass
class EdgeItems(MessageBody):
    """One step's streamed tensors from a producer worker to one consumer
    worker, as columns: a row per (request, streaming edge), and per row its
    items, row-major. It carries what one INPUT_SIGNALS per (request, worker)
    did for an edge that is nothing but streamed tensors -- unsharded, no
    forward-pass info -- so the receiver decodes, reads and routes a step's
    tokens once instead of once per request.

    The fields above the columns are shared by every item: a row whose
    tensors differ in them goes as INPUT_SIGNALS instead.
    """

    partition_name: str
    source_session_id: str
    source_entity: str
    tp_size: int
    tp_rank: int
    source_node_name: str | None = None
    # signal, node, walk and arena segment names the columns index
    names: list[str] = field(default_factory=list)
    # per row
    request_ids: list[str] = field(default_factory=list)
    signal_idx: list[int] = field(default_factory=list)
    next_node_idx: list[int] = field(default_factory=list)
    walk_idx: list[int] = field(default_factory=list)
    num_items: list[int] = field(default_factory=list)
    # per item
    uuids: list[int] = field(default_factory=list)
    addresses: list[int] = field(default_factory=list)
    nbytes: list[int] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    ndims: list[int] = field(default_factory=list)
    dims_flat: list[int] = field(default_factory=list)
    stride_flat: list[int] = field(default_factory=list)
    dtype_idx: list[int] = field(default_factory=list)
    dtype_names: list[str] = field(default_factory=list)
    # the arena segment each item is in (an index into names), -1 for none
    shm_segment_idx: list[int] = field(default_factory=list)
    shm_offsets: list[int] = field(default_factory=list)
    # A frame of inline items carries each item's value here instead of the
    # descriptor columns above: a token the producer took off its stop check's
    # host copy, so nothing is staged, read or acknowledged.
    values: list[int] = field(default_factory=list)

    @property
    def request_id(self) -> str | None:
        """The request of a one-row frame (one parked for a request that has
        not arrived), None for a frame of several."""
        return self.request_ids[0] if len(self.request_ids) == 1 else None


@dataclass
class TensorReceived(MessageBody):
    # Keyed by uuid, which is global, so one ack covers every request whose
    # reads from this producer finished together.
    successful_tensors: dict[int, int] # uuid -> graph edge count
    failed_tensor_ids: list[int] # uuids


@dataclass
class UnpersistTensors(MessageBody):
    request_id: str
    uuid_to_ref_count: dict[int, int]


@dataclass
class ClientStreamRoute(MessageBody):
    """The conductor telling the API server's data worker, the producer of a
    request's client streams, where each one goes: once at dispatch, and
    again with no ranks for a stream whose consuming partition has ended,
    which then takes no more items."""
    request_id: str
    # client stream edge -> the consuming node's ranks for this request
    workers: dict[str, list[str]]
    # client stream edge -> that node, from the walks as the deployment shaped
    # them; the data worker's copy of the model is the unshaped one
    nodes: dict[str, str]


@dataclass
class StopLoops(MessageBody):
    request_id: str
    loop_names: set[str]
    partition_name: str
    loop_stop_times: dict[str, NestedLoopIndices] = field(default_factory=dict)


class Move(NamedTuple):
    rid: str
    kind: MoveKind
    # A move applies to the KV resource keyed `pool`, or, when `pool` is empty, to the whole of node `node`.
    # `count` is the number of pages of a RESERVE, an EARMARK or an ALLOT, the pages a GUARANTEE or a WAIT asks for,
    # 1 for an UNRESERVE whose RESERVE made the allotment rather than grew it, and the tokens a DROP's rebuild writes.
    pool: str = ""
    count: int = 0
    victims: tuple[str, ...] = ()
    node: str = ""


@dataclass
class OffloadDelta:
    """A rank's moves, in the order the followers replay them.

    That is the order the leader made them in, so a sequence that worked for the
    leader works for the followers too, except that a node's offload goes ahead of
    the moves its pools made as it freed room (`add` with ``at``). A follower can
    replay those only after the offload.

    Holds WIRE request ids, not worker-local handles: a handle is minted per
    worker, so rank 0's would name different requests on the follower.

    A dataclass of lists rather than a NamedTuple of deques so that the wire codec
    can automatically encode it. ``pop_left`` is O(n) on a list, which costs nothing
    at the handful of entries a delta ever holds.

    ``__len__`` is the queue depth, so an empty delta is falsy -- several callers
    lean on that.
    """

    # parallel lists, one entry per `Move`
    rids: list[str] = field(default_factory=list)
    kinds: list[MoveKind] = field(default_factory=list)
    pools: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    victims: list[list[str]] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)

    @classmethod
    def new(cls) -> "OffloadDelta":
        return cls()

    def add(
        self, kind: MoveKind, rid: str, pool: str = "", count: int = 0, victims: Sequence[str] = (), node: str = "",
        at: int | None = None,
    ):
        """Append the move, or put it ahead of the moves from index ``at`` on."""
        at = len(self) if at is None else at
        self.rids.insert(at, rid)
        self.kinds.insert(at, kind)
        self.pools.insert(at, pool)
        self.counts.insert(at, count)
        self.victims.insert(at, list(victims))
        self.nodes.insert(at, node)

    def cancel(self, rid: str, kind: MoveKind, pool: str) -> bool:
        """Drop the last move if it is ``kind`` on ``rid`` in ``pool`` and moves no victims, and return whether a move
        was dropped."""
        if not (self.rids and (self.rids[-1], self.kinds[-1], self.pools[-1]) == (rid, kind, pool)) or self.victims[-1]:
            return False
        for column in (self.rids, self.kinds, self.pools, self.counts, self.victims, self.nodes):
            column.pop()
        return True

    def __len__(self):
        return len(self.rids)

    def peek_left(self) -> Move | None:
        if not len(self):
            return
        return Move(
            self.rids[0], self.kinds[0], self.pools[0], self.counts[0], tuple(self.victims[0]), self.nodes[0],
        )

    def pop_left(self) -> Move | None:
        if not len(self):
            return None
        return Move(
            self.rids.pop(0), self.kinds.pop(0), self.pools.pop(0), self.counts.pop(0), tuple(self.victims.pop(0)),
            self.nodes.pop(0),
        )

    def copy(self) -> "OffloadDelta":
        """An independent queue holding the same moves.

        One per follower: each replays at its own pace, popping as it goes.
        """
        return OffloadDelta(
            list(self.rids), list(self.kinds), list(self.pools), list(self.counts), list(self.victims),
            list(self.nodes),
        )

    def take(self) -> "OffloadDelta":
        """Hand the queue over and leave this one empty.

        Copies and clears rather than rebinding, so a holder of this delta (the
        engine keeps its journal) sees it emptied.
        """
        taken = self.copy()
        for column in (self.rids, self.kinds, self.pools, self.counts, self.victims, self.nodes):
            column.clear()
        return taken

    def extend(self, other: "OffloadDelta") -> None:
        self.rids.extend(other.rids)
        self.kinds.extend(other.kinds)
        self.pools.extend(other.pools)
        self.counts.extend(other.counts)
        self.victims.extend(other.victims)
        self.nodes.extend(other.nodes)


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
    # rank 0's prefix-cache leases, by wire request id and then resource
    prefix_hits: dict[str, dict[str, CachedLease]] = field(default_factory=dict)
    page_trails: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # A row of the step takes an input a follower reads from elsewhere: every
    # rank agrees on whose read failed before preparing the step
    read_check: bool = False
    # Per request_ids entry, the chunk the leader cut for the row: [start, end)
    # of a measured total, -1 when the row is not chunked; all empty when no
    # row is. A follower runs these rather than cutting its own, and caps a
    # row's chunk Loop on its first chunk from them.
    chunk_starts: list[int] = field(default_factory=list)
    chunk_ends: list[int] = field(default_factory=list)
    chunk_totals: list[int] = field(default_factory=list)
    # how far each chunk moves its row's chunk Loop (``ChunkRange.stride``)
    chunk_strides: list[int] = field(default_factory=list)


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
    WORKER_STATUS = "worker_status"
    ENABLE_METRICS = "enable_metrics"
    LOAD_LORA_ADAPTER = "load_lora_adapter"
    UNLOAD_LORA_ADAPTER = "unload_lora_adapter"
    LORA_ADAPTER_ACK = "lora_adapter_ack"


@dataclass
class NewRequestConductor(MessageBody):
    request_id: str
    initial_signals: dict[str, list[TensorPointerInfo]]
    initial_input_modalities: list[str]
    initial_output_modalities: list[str]
    input_metadata: dict[str, list[dict]]
    model_kwargs: dict
    # The client's request_id. ``request_id`` is the API server's key for this
    # one submission, never reused; the seed still derives from the client's.
    client_request_id: str | None = None
    # A client stream the opening leaves open: a live call, which is refused
    # rather than queued at max_concurrent_requests.
    input_open: bool = False
    streaming: bool = False
    # the prompt's token count, echoed on RequestComplete for the response's usage
    prompt_tokens: int | None = None


@dataclass
class WorkerGraphsDone(MessageBody):
    request_id: str
    worker_graph_ids: list[int]
    is_first_tp_rank: bool
    persist_signals: dict[str, list[TensorPointerInfo]] = field(default_factory=dict)
    new_token_counts: dict[str, int] = field(default_factory=dict) # name to token counts
    # one name per chunk this worker emitted to the client since its last report
    output_signal_names: list[str] = field(default_factory=list)
    resource_publish_info: dict[str, PublishedInfo] = field(default_factory=dict)
    partition_name: str = field(default="default")
    partition_done: bool = field(default=False)
    # The walk the completed pass ran under. The conductor follows it for a
    # partition whose walk the stream drives.
    graph_walk: str = field(default="")
    # Stream edge -> destination worker -> items sent since the last report.
    # Every rank reports its own.
    stream_items_sent: dict[str, dict[str, int]] = field(default_factory=dict)
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
    step can fail several of them for different reasons. ``statuses``: the
    HTTP status of each that isn't a 500 (the request's own fault, say).
    """
    errors: dict[str, str]
    statuses: dict[str, int] = field(default_factory=dict)


@dataclass
class WorkerStatus(MessageBody):
    """A worker's engine metrics, sent once per status interval only while
    metrics are on: cumulative or current values per graph node or tensor
    edge, never per-request data."""
    worker_id: str
    metrics: list[MetricSnapshot]


@dataclass
class EnableMetrics(MessageBody):
    """Start the workers' histograms and counters. The API server sends it to
    the conductor after its warmup requests; the conductor forwards it to
    every worker (``WorkerMetrics.start_counting``)."""


@dataclass
class LoadLoraAdapter(MessageBody):
    """Serve a LoRA adapter from now on (``POST /v1/load_lora_adapter``). The API server
    sends it to the conductor, which forwards it to every worker hosting the LoRA pool
    and answers once all of them have acknowledged. ``op_id`` pairs the acks and the
    answer with the request; ``lora_path`` is a safetensors file every process can read.
    ``lora_digest`` (hex) is the digest of the weights the API server checked and keys
    the prefix cache under: a worker reading other weights refuses the load.
    """
    op_id: str
    lora_name: str
    lora_path: str
    lora_digest: str


@dataclass
class UnloadLoraAdapter(MessageBody):
    """Stop serving a LoRA adapter to new requests (``POST /v1/unload_lora_adapter``);
    routed like ``LoadLoraAdapter``."""
    op_id: str
    lora_name: str


@dataclass
class LoraAdapterAck(MessageBody):
    """A worker's answer to a load or unload: ``error`` is None when it succeeded."""
    op_id: str
    worker_id: str
    error: str | None = None


@dataclass
class ConductorMessage:
    message_type: ConductorMessageType
    body: MessageBody
