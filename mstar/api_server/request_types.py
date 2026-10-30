from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mstar.graph.base import GraphEdge
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.multimodal import PromptPart
from mstar.observability.recorders import MetricSnapshot
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.profile.worker import GraphTimings

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ResultChunk:
    """One chunk of generated output for a request."""
    request_id: str
    modality: str  # "text" | "image" | "audio" | "video" | "video_frame"
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
class ResultTokens:
    """One step's client-facing scalars (sampled tokens), every request in one
    frame. Replaces one ResultTensors per request for an output whose value
    the worker already holds on the host: no transport tensor, no ack."""
    request_ids: list[str]
    values: list[int]
    # per request, the loop context of the step that produced the value
    loop_indices: list[NestedLoopIndices | None]
    signal: str
    modality: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RequestComplete:
    """Signals that a request has finished processing."""
    request_id: str
    # Output signal name -> chunks the workers emitted to the client. The API
    # server waits until it has received that many of each before completing
    # the request: completion can overtake the last chunks on the wire.
    output_chunk_counts: dict[str, int]
    conductor_ingest_time: float
    conductor_finish_time: float
    graph_timings: GraphTimings = field(default_factory=dict)
    rx_info: list[RxInfo] = field(default_factory=list)
    tx_info: list[TxInfo] = field(default_factory=list)
    # When the conductor took the request in (queued or admitted), in the same
    # clock as conductor_ingest_time. Only stamped while metrics are on.
    conductor_arrival_time: float | None = None


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
class ConductorStatus:
    """The conductor's periodic status, sent only while metrics are on: one
    snapshot per family in ``catalog.STATUS_FAMILIES``, current or cumulative
    values only, never per-request data."""
    metrics: list[MetricSnapshot]


# what an ``APIServerMessage`` carries; None for ``setup_done``. The decoder
# tries branches in order and takes the first whose required fields are all
# present, so a body must lack a required field of every branch before it.
APIServerMessageBody = (
    ResultTensors | ResultTokens | RequestComplete | RequestFailed | ConductorStatus | None
)


@dataclass
class APIServerMessage:
    """Envelope for messages received by the API server."""
    # "result_tensors" | "result_tokens" | "request_complete" | "request_failed"
    # | "conductor_status" | "setup_done"
    message_type: str
    body: APIServerMessageBody = None


@dataclass
class DataWorkerProfile:
    """Profiling reported by the API-server data worker at preprocess finish
    (or failure): the timestamp at which the request was handed to the
    conductor and the per-modality sizes of the raw inputs. (The data worker's
    tx/rx are read directly from its tensor manager at request completion,
    not via this.)"""
    request_id: str
    preprocess_finish_time: float | None = None  # time.perf_counter
    inputs: list[InputInfo] = field(default_factory=list)
    # Set only with metrics on: the data worker's own numbers for the request.
    dequeue_time: float | None = None  # perf_counter, taken off the data worker's queue
    input_modalities: list[str] = field(default_factory=list)
    # modality -> size in bytes of each input file, read when it was dequeued
    upload_sizes: dict[str, list[int]] = field(default_factory=dict)
    # preprocessing raised: the client gets an error chunk instead
    failed: bool = False


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
    # The id the client knows the request by; ``request_id`` is the API
    # server's key for this one submission.
    client_request_id: str | None = None
    # the client streams the opening ends; the others stay open for appends
    ends: frozenset[str] = frozenset()


@dataclass
class ClientAppend:
    """One append to a running request's client streams, queued behind the
    request's own ``PreprocessInput`` so its chunks stay in order. The
    streams in ``ends`` end after this chunk. Stays inside the API server
    process."""
    request_id: str
    file_paths: dict[str, list[str]] | None = None
    ends: frozenset[str] = frozenset()
    text: str | None = None
    # per modality, in file_paths' order, the client stream each file names (None: unnamed)
    file_streams: dict[str, list[str | None]] | None = None
    # this append's own model_kwargs; the opening's stay the request's
    model_kwargs: dict | None = None
    # per ClientAudio stream, decoded mono float32 samples at its declared
    # rate: a route that decodes the client's codec itself appends without a file
    samples: "dict[str, np.ndarray] | None" = None
    # Answers the append as late, with an error naming why; called from the
    # preprocess thread when the model routes items to a stream that has
    # ended (by the client, or with its consuming partition). None: nobody
    # to answer.
    on_late: Callable[[str], None] | None = field(default=None, compare=False)
