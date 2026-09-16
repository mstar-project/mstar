from dataclasses import dataclass, field

from mstar.graph.base import GraphEdge
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.multimodal import PromptPart
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.profile.worker import GraphTimings


@dataclass
class ResultChunk:
    """One chunk of generated output for a request."""
    request_id: str
    modality: str  # "text" | "image" | "audio" | "video"
    data: bytes  # raw payload (text encoded as utf-8)
    metadata: dict = field(default_factory=dict)


@dataclass
class ResultTensors:
    request_id: str
    modality: str
    graph_edge: GraphEdge
    loop_indices: NestedLoopIndices
    metadata: dict = field(default_factory=dict)


@dataclass
class InlineResults:
    """One step's small client outputs from a worker, carried in the message itself.

    A decode step emits one token per request; sending each as a tensor to be read over the
    transport costs a store, a registration, a message, a read and an ack per request per step.
    Below ``INLINE_MAX_BYTES`` the worker puts the bytes in this batch instead: ``results`` are
    the usual per-request ``ResultTensors`` (their ``graph_edge.tensor_info`` carry dims and
    dtype), ``data`` maps each tensor info's uuid to its serialized bytes.
    """
    results: list[ResultTensors]
    data: dict[str, bytes]


# per-tensor size up to which a client output travels inline in the result message
INLINE_MAX_BYTES = 4096


@dataclass
class RequestComplete:
    """Signals that a request has finished processing."""
    request_id: str
    # Maps output signal name to its final forward pass number.
    # The API server waits until all entries are received before
    # completing the request.
    final_outputs: dict[str, NestedLoopIndices]
    conductor_ingest_time: float
    conductor_finish_time: float
    graph_timings: GraphTimings = field(default_factory=dict)
    rx_info: list[RxInfo] = field(default_factory=list)
    tx_info: list[TxInfo] = field(default_factory=list)


@dataclass
class RequestFailed:
    """Signals that a request died in the engine and will produce no result.

    One message per request, so it routes through the API server's result
    loop exactly like ``RequestComplete`` does.
    """
    request_id: str
    error_message: str
    status: int = 500


@dataclass
class APIServerMessage:
    """Envelope for messages received by the API server."""
    # "result_tensors" | "request_complete" | "request_failed" | "setup_done"
    message_type: str
    body: ResultTensors | RequestComplete | RequestFailed | None = None  # None for setup_done


@dataclass
class DataWorkerProfile:
    """Profiling reported by the API-server data worker at preprocess finish:
    the timestamp at which the request was handed to the conductor and the
    per-modality sizes of the raw inputs. (The data worker's tx/rx are read
    directly from its tensor manager at request completion, not via this.)"""
    request_id: str
    preprocess_finish_time: float | None = None  # time.perf_counter
    inputs: list[InputInfo] = field(default_factory=list)


@dataclass
class PreprocessInput:
    request_id: str
    text: str | None

    # file_paths is modality: list of filenames
    file_paths: dict[str, list[str]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict

    # Ordered text/attachment sequence, when the entrypoint preserved it.
    prompt_parts: list[PromptPart] | None = None
