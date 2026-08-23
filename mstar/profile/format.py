
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

    # ── True GPU time (seconds), measured with a CUDA event pair ─────────
    #
    # ``forward_time`` above is a CPU launch/enqueue span: with async
    # execution it says how long the *submission* took, not how long the
    # GPU was busy, and under speculative scheduling it overlaps the next
    # step. ``gpu_time`` is the in-stream elapsed time between an event
    # recorded before the step's first launch and the step's completion
    # event, so it is the step's own GPU-busy time even when the batch was
    # queued behind its predecessor.
    #
    # None when CUDA is unavailable or profiling never stamped the pair
    # (e.g. a batch that ran no forward). Accumulates like the others.
    gpu_time: float | None = None

    # ── Engine-internal CPU phase spans (seconds) ────────────────────────
    #
    # The four hooks of BaseEngine.execute_batch, timed separately so a
    # simulator can model the CPU lane per phase instead of folding it all
    # into one launch span. ``plan_time`` is the FlashInfer/attention plan
    # cost that the worker's pre-plan thread exists to hide; ``sample_time``
    # is engine postprocess (sampling + remap) which runs after the launch.
    prepare_time: float = 0.0
    plan_time: float = 0.0
    launch_time: float = 0.0
    sample_time: float = 0.0

    def __add__(self, other: "GraphTiming"):
        assert self.node == other.node and self.graph_walk == other.graph_walk
        if self.gpu_time is None and other.gpu_time is None:
            gpu_time = None
        else:
            gpu_time = (self.gpu_time or 0.0) + (other.gpu_time or 0.0)
        return GraphTiming(
            node=self.node,
            graph_walk=self.graph_walk,
            exec_count=self.exec_count + other.exec_count,
            total_time=self.total_time + other.total_time,
            forward_time=self.forward_time + other.forward_time,
            preprocess_time=self.preprocess_time + other.preprocess_time,
            postprocess_time=self.postprocess_time + other.postprocess_time,
            gpu_time=gpu_time,
            prepare_time=self.prepare_time + other.prepare_time,
            plan_time=self.plan_time + other.plan_time,
            launch_time=self.launch_time + other.launch_time,
            sample_time=self.sample_time + other.sample_time,
        )


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
    rx_info: list[RxInfo] = field(default_factory=list)
    tx_info: list[TxInfo] = field(default_factory=list)
    inputs: list[InputInfo] = field(default_factory=list)
    outputs: list[OutputInfo] = field(default_factory=list)
