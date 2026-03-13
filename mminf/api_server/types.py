from dataclasses import dataclass, field

from mminf.graph.base import GraphEdge


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
    metadata: dict = field(default_factory=dict)


@dataclass
class RequestComplete:
    """Signals that a request has finished processing."""
    request_id: str


@dataclass
class APIServerMessage:
    """Envelope for messages received by the API server."""
    message_type: str  # "result_tensors" | "request_complete"
    body: ResultTensors | RequestComplete


@dataclass
class PreprocessInput:
    request_id: str
    text: str | None

    # file_paths is modality: list of filenames
    file_paths: dict[str, list[str]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict
