from dataclasses import dataclass, field

from mminf.graph.base import GraphEdge
from mminf.graph.loop_indices import NestedLoopIndices


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
class RequestComplete:
    """Signals that a request has finished processing."""
    request_id: str
    # Maps output signal name to its final forward pass number.
    # The API server waits until all entries are received before
    # completing the request.
    final_outputs: dict[str, NestedLoopIndices]


@dataclass
class APIServerMessage:
    """Envelope for messages received by the API server."""
    message_type: str  # "result_tensors" | "request_complete" | "setup_done"
    body: ResultTensors | RequestComplete | None = None  # None for setup_done message


@dataclass
class PreprocessInput:
    request_id: str
    text: str | None

    # file_paths is modality: list of filenames
    file_paths: dict[str, list[str]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict
