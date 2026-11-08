
from dataclasses import dataclass, field


@dataclass
class InputInfo:
    modality: str
    count: int
    total_bytes: int


@dataclass
class OutputInfo:
    modality: str
    count: int
    total_bytes: int


@dataclass
class TxInfo:
    edge_name: str
    source_entity: str
    count: int = 0
    num_bytes: int = 0
    time: float = 0.0 # seconds

    def update(self, num_bytes: int, time: float, count_increment: int=1):
        self.count += count_increment
        self.num_bytes += num_bytes
        self.time += time

@dataclass
class RxInfo:
    edge_name: str
    source_entity: str
    dest_entity: str
    count: int = 0
    num_bytes: int = 0
    time: float = 0.0 # seconds

    def update(self, num_bytes: int, time: float, count_increment: int=1):
        self.count += count_increment
        self.num_bytes += num_bytes
        self.time += time


@dataclass
class GraphTiming:
    node: str
    graph_walk: str
    exec_count: int
    total_time: float # seconds
    forward_time: float # actually from fwd start to end of postprocess
    preprocess_time: float # preprocess + prepare_inputs
    postprocess_time: float # CPU-level postprocess (minus async overlap)
    # perf_counter (CLOCK_MONOTONIC, so comparable across this host's
    # processes) of the first step's start and the last step's end: the
    # worker's span over the request
    first_start: float | None = None
    last_end: float | None = None

    def __add__(self, other: "GraphTiming"):
        assert self.node == other.node and self.graph_walk == other.graph_walk
        starts = [t for t in (self.first_start, other.first_start) if t is not None]
        ends = [t for t in (self.last_end, other.last_end) if t is not None]
        return GraphTiming(
            node=self.node,
            graph_walk=self.graph_walk,
            exec_count=self.exec_count + other.exec_count,
            total_time=self.total_time + other.total_time,
            forward_time=self.forward_time + other.forward_time,
            preprocess_time=self.preprocess_time + other.preprocess_time,
            postprocess_time=self.postprocess_time + other.postprocess_time,
            first_start=min(starts) if starts else None,
            last_end=max(ends) if ends else None,
        )


@dataclass
class RequestTiming:
    recv_time: float | None = None # all are time.perf_counter
    preprocess_finish_time: float | None = None
    conductor_arrival_time: float | None = None  # only while metrics are on
    conductor_ingest_time: float | None = None
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    conductor_finish_time: float | None = None
    finish_time: float | None = None
    # the data worker: when it had the last output tensor read, and its total
    # time postprocessing (encoding) the request's outputs: ``Model.postprocess``,
    # plus a per-row share of ``postprocess_inline`` for values sent inline
    last_output_read_time: float | None = None
    output_postprocess_s: float = 0.0


@dataclass
class RequestProfile:
    rid: str
    timing: RequestTiming = field(default_factory=RequestTiming)
    graph_timings: list[GraphTiming] = field(default_factory=list)
    rx_info: list[RxInfo] = field(default_factory=list)
    tx_info: list[TxInfo] = field(default_factory=list)
    inputs: list[InputInfo] = field(default_factory=list)
    outputs: list[OutputInfo] = field(default_factory=list)
