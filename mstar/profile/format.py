
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
class GraphTiming:
    node: str
    graph_walk: str
    exec_count: int
    total_time: float # seconds
    forward_time: float # actually from fwd start to end of postprocess
    preprocess_time: float # preprocess + prepare_inputs
    postprocess_time: float # CPU-level postprocess (minus async overlap)

    def __add__(self, other: "GraphTiming"):
        assert self.node == other.node and self.graph_walk == other.graph_walk
        return GraphTiming(
            node=self.node,
            graph_walk=self.graph_walk,
            exec_count=self.exec_count + other.exec_count,
            total_time=self.total_time + other.total_time,
            forward_time=self.forward_time + other.forward_time,
            preprocess_time=self.preprocess_time + other.preprocess_time,
            postprocess_time=self.postprocess_time + other.postprocess_time,
        )


def accumulate_graph_timing(timings: list[GraphTiming]) -> list[GraphTiming]:
    """Accumulate timings by node and graph_walk."""
    acc: dict[str, GraphTiming] = {}
    for timing in timings:
        key = f"{timing.node}:{timing.graph_walk}"
        if key in acc:
            acc[key] += timing
        else:
            acc[key] = timing
    return list(acc.values())


@dataclass
class RequestTiming:
    recv_time: float | None = None # all are time.perf_counter
    preprocess_finish_time: float | None = None
    conductor_ingest_time: float | None = None
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    conductor_finish_time: float | None = None
    finish_time: float | None = None


@dataclass
class RequestProfile:
    rid: str
    timing: RequestTiming = field(default_factory=RequestTiming)
    graph_timings: list[GraphTiming] = field(default_factory=list)
    inputs: list[InputInfo] = field(default_factory=list)
    outputs: list[OutputInfo] = field(default_factory=list)