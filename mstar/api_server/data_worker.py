

import hashlib
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property

import torch

from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.components.diffusion.image_io import PNG_WORKERS

try:
    import torchaudio  # noqa: F401 — probes availability; real usage in callers
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError, OSError):
    VideoDecoder = None

from mstar.api_server.request_types import (
    ClientAppend,
    DataWorkerProfile,
    PreprocessInput,
    ResultChunk,
    ResultTensors,
)
from mstar.communication.communicator import (
    BaseCommunicator,
    CommProtocol,
    OutboxCommunicator,
    make_communicator,
)
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import (
    FailedRead,
    NameToTensorList,
    create_tensor_communication_manager,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.keys import PageItem, chain, end_key, fingerprint, page_items
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.model.base import ClientChunk, Model, PrefixKeys, ProcessPromptOutput, Span, keyed_labels
from mstar.model.media_limits import (
    RequestMediaBudget,
    charge_decoded_audio,
    check_audio_duration,
    check_file_counts,
    check_media_limits,
)
from mstar.observability.catalog import STATUS_INTERVAL_S
from mstar.observability.recorders import EdgeTransfers
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.streaming.topology import (
    CLIENT,
    ClientStreamLimits,
    client_stream_limits,
)
from mstar.utils import profiler
from mstar.utils.client_errors import client_error_text
from mstar.utils.cpus import usable_cpus
from mstar.utils.ipc_format import (
    AbortRequest,
    ClientStreamRoute,
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    EnableMetrics,
    InputSignals,
    NewRequestConductor,
    ReadsDone,
    RemoveRequest,
    TensorReceived,
    UnpersistTensors,
    WorkerMessage,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)

# Outputs whose postprocess encodes a whole artifact (a PNG takes tens of ms,
# an mp4 mux hundreds) run on the encode pool, so reads, streamed chunks and
# prompt preprocessing do not wait behind them.
ARTIFACT_MODALITIES = frozenset({"image", "video"})


def pooled_output(modality: str, loop_indices: NestedLoopIndices | None) -> bool:
    """Whether an output encodes on the pool: an artifact modality that the
    walk emits outside any loop. An output a loop emits is a stream (V-JEPA2's
    video, once per iteration) and stays inline, in arrival order, as do text
    and audio, whose chunks are cheap and carry inline metadata."""
    streamed = loop_indices is not None and bool(loop_indices.loop_name_order)
    return modality in ARTIFACT_MODALITIES and not streamed


def encode_workers(cpus: int | None = None, png_workers: int | None = None) -> int:
    """Outputs encoded at once. A PNG deflates on ``png_workers`` threads of
    its own (``image_io.PNG_WORKERS``), so the pool times that stays within
    the CPUs the process may use."""
    if cpus is None:
        cpus = usable_cpus()
    if png_workers is None:
        png_workers = PNG_WORKERS
    return max(1, min(4, cpus // png_workers))


def _sha256(path: str) -> bytes:
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").digest()


def _file_state(path: str) -> tuple[int, int, int]:
    stat = os.stat(path)
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


class _FileChangedError(Exception):
    """A file was rewritten between its load and its hash."""


def _video_frame_metadata(
    tensor: torch.Tensor,
    *,
    fps: float,
    frame_index: int,
    metadata: dict | None = None,
) -> dict:
    """Describe one raw RGB24 output tensor and reject ambiguous payloads."""
    if tensor.dtype != torch.uint8 or tensor.dim() != 4 or tensor.shape[-1] != 3:
        raise ValueError(
            "video_frame output must be uint8 RGB shaped "
            f"[frame_count, height, width, 3]; got {tuple(tensor.shape)} of {tensor.dtype}"
        )
    frame_count, height, width, _ = map(int, tensor.shape)
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError(
            "video_frame output dimensions must be positive; "
            f"got {tuple(tensor.shape)}"
        )
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError(f"video_frame fps must be a positive number; got {fps!r}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(
            f"video_frame frame_index must be a non-negative int; got {frame_index!r}"
        )
    return {
        **(metadata or {}),
        "width": width,
        "height": height,
        "fps": fps,
        "pixel_format": "rgb24",
        "frame_index": frame_index,
        "frame_count": frame_count,
    }
@dataclass
class InlineToken:
    """One request's value out of a RESULT_TOKENS frame, queued behind the
    tensor results so sequences keep arrival order."""
    request_id: str
    value: int
    loop_indices: NestedLoopIndices | None
    modality: str


def _answer_late(on_late: Callable[[str], None] | None, streams: list[str]) -> None:
    """Answer a client's append as late, naming ``streams``, which take no
    more items."""
    if streams and on_late is not None:
        on_late(f"the call's {', '.join(map(repr, sorted(streams)))} input has ended")


def unlink_files(file_paths: dict[str, list[str]] | None) -> None:
    """Delete a message's uploaded files; one already gone or unremovable is skipped."""
    for paths in (file_paths or {}).values():
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


def _kv_page_sizes(model: Model, model_config: dict) -> dict[str, int]:
    """Page size per KV resource whose prefix cache is on, after this deployment's YAML has been applied."""
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, model_config)
    return {
        spec.resource_key: spec.config.page_size
        for spec in specs
        if isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig) and spec.config.prefix_cache
    }


class PreprocessWorker:
    def __init__(
        self,
        model: Model | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        tcp_transfer_device="",
        enable_prof: bool=False,
        model_config: dict | None = None,
        enable_nvtx: bool=False,
        is_pending: Callable[[str], bool] | None = None,
    ):
        check_media_limits()
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.abort_request_queue = queue.Queue()
        self.reads_done_queue = queue.Queue()
        self.discard_tensor_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.profile_queue = queue.Queue()
        # Set by report_preprocess_finish(); read by the worker thread.
        self.timing_requested = threading.Event()
        # Set by enable_worker_metrics(); the worker thread tells the conductor.
        self.worker_metrics_requested = threading.Event()
        self.stop_event = threading.Event()
        # Wakes the thread's idle wait: every producer below signals it.
        self.wakeup = EventWakeup()

        self.per_request_reading_tensors = {}
        # request id -> output name -> chunks received, for received_final_chunks
        self.received_chunk_counts: dict[str, dict[str, int]] = {}

        # Build the communicator + tensor manager here (main thread) and hand
        # them to the worker thread, rather than constructing them inside it.
        # The socket is only *used* from the worker thread, but owning the
        # tensor manager here lets the main thread read its tx/rx profiling
        # directly once a request is done (no cross-thread queue / race).
        # Sends never block the thread (OutboxCommunicator): the conductor and
        # the workers block on sends to it, so it has to keep reading.
        self.communicator = OutboxCommunicator(make_communicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
            nonblocking_send=True,
        ))  # only used to send (from the worker thread)
        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id="api_server_preprocess_worker",
            hostname=hostname,
            device="cpu",
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof,
        )

        self.thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                result_tensor_queue=self.result_tensor_input_queue,
                out_queue=self.output_queue,
                profile_queue=self.profile_queue,
                cleanup_request_queue=self.cleanup_request_queue,
                abort_request_queue=self.abort_request_queue,
                reads_done_queue=self.reads_done_queue,
                discard_tensor_queue=self.discard_tensor_queue,
                stop_event=self.stop_event,
                wakeup=self.wakeup,
                communicator=self.communicator,
                tensor_manager=self.tensor_manager,
                model=model,
                enable_prof=enable_prof,
                timing_requested=self.timing_requested,
                worker_metrics_requested=self.worker_metrics_requested,
                model_config=model_config,
                enable_nvtx=enable_nvtx,
                is_pending=is_pending,
            )
        )
        self.thread.start()

    def report_preprocess_finish(self) -> None:
        """Put each request's preprocess-finish time on the profile queue
        without ``--log-stats`` too (``mstar_request_preprocess_seconds``),
        with the data worker's own time, a failed preprocess, and the input
        files' sizes."""
        self.timing_requested.set()

    def enable_worker_metrics(self) -> None:
        """Have the conductor start the engine workers' histograms and
        counters. The thread sends it on the socket that carries every
        submission, so it reaches the conductor before any request submitted
        after this call."""
        self.worker_metrics_requested.set()
        self._signal()

    def queue_depth(self) -> int:
        """Requests submitted and not yet taken up by the data worker thread."""
        return self.request_input_queue.qsize()

    def _signal(self):
        """End the thread's idle wait: new work is queued. getattr: tests
        build this facade without its constructor."""
        wakeup = getattr(self, "wakeup", None)
        if wakeup is not None:
            wakeup.signal()

    def new_request(self, input: PreprocessInput):
        self.received_chunk_counts[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)
        self._signal()

    def stream_input(self, chunk: ClientAppend):
        """Append to a running request's client streams; the same queue as its
        ``new_request`` keeps the chunks behind it and in order."""
        self.request_input_queue.put(chunk)
        self._signal()

    def abort_request(self, request_id: str):
        # Forward the abort to the conductor and begin draining our own reads
        # (the worker thread ACKs READS_DONE once they finish). The hard cleanup
        # of the persisted input signals waits for the conductor's
        # REMOVE_REQUEST, so a worker still reading them can't be unlinked
        # out from under it. Drop only the main-thread bookkeeping here.
        self.forward_abort(request_id)
        self.received_chunk_counts.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def forward_abort(self, request_id: str):
        """:meth:`abort_request` for a request the API server still holds:
        the main-thread bookkeeping stays until :meth:`finished_reading`."""
        self.abort_request_queue.put(request_id)
        self._signal()

    def finished_reading(self, request_id: str, drained: bool = True):
        """The API server is done with this request's outputs. Tell the
        conductor via READS_DONE; it will send REMOVE_REQUEST to trigger the
        hard cleanup. ``drained`` means every chunk was delivered, so no read
        can still be in flight and the ACK can go out immediately; pass False
        when delivery was abandoned (TTL, client gone) and the ACK must wait
        for any in-flight read."""
        self.reads_done_queue.put((request_id, drained))
        self._signal()
        self.received_chunk_counts.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def new_result_tensors(self, input: ResultTensors):
        name = input.graph_edge.name
        if input.request_id not in self.received_chunk_counts:
            # Request was removed while this output was still in flight; ack the
            # tensors so the producing worker can reclaim them rather than leak.
            logger.debug("Late result_tensors for cleaned-up request %s, acking and dropping", input.request_id)
            self.discard_result_tensors(input)
            return

        counts = self.received_chunk_counts[input.request_id]
        counts[name] = counts.get(name, 0) + 1

        self.per_request_reading_tensors[input.request_id] += len(input.graph_edge.tensor_info)
        logger.debug(
            "Data worker reading queue for request %s increased to length %d",
            input.request_id,  self.per_request_reading_tensors[input.request_id]
        )
        self.result_tensor_input_queue.put(input)
        self._signal()

    def new_result_token(
        self, request_id: str, value: int,
        loop_indices: NestedLoopIndices | None, signal: str, modality: str,
        wake: bool = True,
    ):
        """A value that rode inline: same accounting as a result tensor, no
        transport read and nothing to ack. ``wake=False`` queues without
        ending the thread's idle wait: a frame's caller hands over every
        request's value first and calls ``wake`` once."""
        if request_id not in self.received_chunk_counts:
            logger.debug("Late token for cleaned-up request %s dropped", request_id)
            return
        counts = self.received_chunk_counts[request_id]
        counts[signal] = counts.get(signal, 0) + 1
        self.per_request_reading_tensors[request_id] += 1
        self.result_tensor_input_queue.put(
            InlineToken(request_id, value, loop_indices, modality)
        )
        if wake:
            self._signal()

    def wake(self):
        """End the thread's idle wait after a batch of ``wake=False`` puts."""
        self._signal()

    def discard_result_tensors(self, input: ResultTensors):
        """Ack and drop result tensors for an already-removed request.

        Routed to the worker thread (which owns the communicator) so the
        producing worker gets its TENSOR_RECEIVED ack and frees the buffers.
        """
        self.discard_tensor_queue.put(input)
        self._signal()

    def has_pending_tensors(self, request_id: str):
        return self.per_request_reading_tensors.get(request_id, 0) > 0

    def received_final_chunks(
        self, request_id: str,
        output_chunk_counts: dict[str, int],
    ):
        # Every serving walk reports at least one client-facing output, so an
        # empty dict means the walk emitted nothing (or completion raced ahead
        # of every result). Report not-done and let the TTL backstop close the
        # request rather than completing it instantly with zero chunks.
        if not output_chunk_counts:
            return False
        received = self.received_chunk_counts.get(request_id, {})
        return all(
            received.get(name, 0) >= count
            for name, count in output_chunk_counts.items()
        )

    def get_result_chunks(self)-> list[ResultChunk]:
        results = []
        while not self.output_queue.empty():
            result: ResultChunk = self.output_queue.get()
            # A request can be cleaned up (its result already returned) while a
            # late chunk is still in the queue -- common when several requests
            # complete in the same step. Mirror new_result_tensors' guard and
            # drop the straggler rather than KeyError, which would otherwise
            # abort the whole drain and lose the other requests' chunks.
            if result.request_id not in self.per_request_reading_tensors:
                logger.warning(
                    "Late result chunk for cleaned-up request %s, ignoring",
                    result.request_id,
                )
                continue
            self.per_request_reading_tensors[result.request_id] -= 1
            logger.debug(
                "Data worker reading queue for request %s decreased to length %d",
                result.request_id,  self.per_request_reading_tensors[result.request_id]
            )
            results.append(result)
        return results

    def get_profile_updates(self) -> list[DataWorkerProfile]:
        """Drain preprocess-side profiling updates emitted by the worker thread."""
        updates = []
        while not self.profile_queue.empty():
            updates.append(self.profile_queue.get())
        return updates

    def get_tx_info(self, request_id: str) -> list[TxInfo]:
        """Snapshot the data worker's send (tx) profiling for a request.

        Safe to call from the main thread once the request is done: by then the
        worker thread is no longer mutating this request's tx state, and the
        caller must read it before ``cleanup_request`` drops it.
        """
        return self.tensor_manager.get_tx_info(request_id)

    def get_rx_info(self, request_id: str) -> list[RxInfo]:
        """Snapshot the data worker's receive (rx) profiling for a request.

        Same safety contract as :meth:`get_tx_info` — read once the request's
        final chunks have all arrived, before ``cleanup_request``.
        """
        return self.tensor_manager.get_rx_info(request_id)

    def enable_transfer_metrics(self) -> EdgeTransfers:
        """Count the tensors this data worker reads (the request's results)
        from now on; the API server exports the thread's published counts."""
        return self.tensor_manager.enable_transfer_metrics()

    def cleanup_request(self, request_id: str):
        self.cleanup_request_queue.put(request_id)
        self._signal()
        self.received_chunk_counts.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def shutdown(self):
        self.stop_event.set()
        self._signal()
        if self.thread.is_alive():
            self.thread.join()


def warm_up_model(model) -> None:
    """Run the model's ``warmup_preprocess`` once; a failure is logged, not
    fatal, because the same work happens again on the first request."""
    warm = getattr(model, "warmup_preprocess", None)
    if model is None or not callable(warm):
        return
    try:
        warm()
    except Exception:  # noqa: BLE001 — warm-up is best effort
        logger.exception("model.warmup_preprocess failed; the first request will do the work")


@dataclass
class RequestOutputState:
    """One request's output ordering, held by the data worker.

    Transport reads may complete out of order. Each output takes a sequence
    when the worker notification arrives, and completed chunks are held until
    every earlier sequence has been emitted.
    """

    # tensor uuid -> (sequence, loop indices)
    order: dict[str, tuple[int, NestedLoopIndices]] = field(default_factory=dict)
    next_sequence: int = 0
    next_emit: int = 0
    # sequence -> completed chunk waiting on an earlier one
    pending: dict[int, ResultChunk] = field(default_factory=dict)
    # A video_frame chunk can carry several frames, so this advances by
    # frame_count rather than chunks.
    frame_index: int = 0


@dataclass(slots=True)
class ClientCall:
    """A call's client streams, as their producer. The data worker sends each
    item to the ranks of the node that consumes it, and then each stream's end
    with the number of items it carried."""

    # the conductor's dispatch CLIENT_STREAM_ROUTE, with a later route's rank
    # lists merged in; None until it arrives
    route: ClientStreamRoute | None = None
    # items stored before the route arrived, each with its append's on_late,
    # sent in this order once it does
    held: list[tuple[dict[str, list[TensorPointerInfo]], Callable[[str], None] | None]] = field(
        default_factory=list,
    )
    # stream edge -> items sent to each of its ranks
    items_sent: dict[str, int] = field(default_factory=dict)
    # streams that take no more items: the client ended them, or their
    # consuming partition has ended
    ended: set[str] = field(default_factory=set)
    # stream edge -> what it carried, charged against its max_items
    charged: dict[str, int] = field(default_factory=dict)
    # ClientAudio stream edge -> the audio short of a whole frame, which goes
    # with the stream's next item; a stream that ended with its partition
    # keeps it unsent
    tails: dict[str, torch.Tensor] = field(default_factory=dict)


class PreprocessWorkerThread:
    # next time the thread publishes its transfer counts; class-level so a
    # thread built without __init__ has one
    _next_transfer_publish = 0.0
    # class-level so a thread built without __init__ never sends ENABLE_METRICS
    worker_metrics_requested: threading.Event | None = None
    # None (a thread built without __init__) postprocesses every output inline
    _encode_pool: ThreadPoolExecutor | None = None
    # Appends handled per pass before output delivery gets its turn again: one
    # for each of up to 32 live calls appending at once. An append costs a file
    # decode, far less than a request's preprocess, which is one per pass.
    MAX_APPENDS_PER_PASS = 32

    def __init__(
        self,
        in_queue: queue.Queue, # for preprocessing
        result_tensor_queue: queue.Queue, # for output streaming
        out_queue: queue.Queue,
        profile_queue: queue.Queue,
        cleanup_request_queue: queue.Queue,
        abort_request_queue: queue.Queue,
        reads_done_queue: queue.Queue,
        discard_tensor_queue: queue.Queue,
        stop_event: threading.Event,
        communicator: BaseCommunicator,
        tensor_manager,
        device: str = "cpu",
        model: Model | None = None,
        enable_prof: bool=False,
        model_config: dict | None = None,
        enable_nvtx: bool=False,
        wakeup: EventWakeup | None = None,
        timing_requested: threading.Event | None = None,
        worker_metrics_requested: threading.Event | None = None,
        is_pending: Callable[[str], bool] | None = None,
    ):
        # keying a label needs the deployment's page size as well as the
        # model's declaration, so a worker built without a config keys no label
        declared = keyed_labels(model) if model is not None and model_config else {}
        # resolved as the worker does at load, so both split a prompt into the same pages
        self._prefix_page_sizes = _kv_page_sizes(model, model_config) if declared else {}
        self._keyed_labels = {
            key: by_label for key, by_label in declared.items() if key in self._prefix_page_sizes
        }
        # client stream edge -> its declaration: the partition it flows into,
        # which its items and ends name, and its ClientAudio; the consuming
        # node comes in the conductor's route
        self._client_connections = {
            conn.edge_name: conn for conn in model.get_partition_topology().client_connections()
        } if model is not None else {}
        # tensors under these names are client stream items, not prompt signals
        self._client_edges = frozenset(self._client_connections)
        self._model_config = model_config
        # per request of a model with client streams, those streams' state
        self._client_calls: dict[str, ClientCall] = {}
        self.in_queue = in_queue
        # Whether the API server still waits on a submission (see
        # _process_waiting_input); None (tests) takes everything.
        self._is_pending = is_pending or (lambda key: True)
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.abort_request_queue = abort_request_queue
        self.reads_done_queue = reads_done_queue
        self.discard_tensor_queue = discard_tensor_queue
        self.out_queue = out_queue
        self.profile_queue = profile_queue

        # Teardown drain: rids we've stopped reading for (until REMOVE_REQUEST),
        # and those we've already ACKed READS_DONE for (avoid double-ACK).
        self._draining_rids: set[str] = set()
        self._reads_done_sent: set[str] = set()
        # Requests under each rid that started here and are not removed yet
        # (by REMOVE_REQUEST, or by failing before they reached the conductor).
        # A release (finished_reading or an abort) for a rid with none must not
        # enter the sets above: its request is gone and no REMOVE_REQUEST
        # follows. A count, since a client can reuse a rid before the last
        # request under it is removed.
        self._live_requests: dict[str, int] = {}

        self.stop_event = stop_event
        self.device = device
        self.model = model
        self.enable_prof = enable_prof
        # Preprocess-finish times are reported when enable_prof or this is set,
        # the data worker's own metrics only when this is set.
        self.timing_requested = timing_requested if timing_requested is not None else threading.Event()
        self.worker_metrics_requested = (
            worker_metrics_requested if worker_metrics_requested is not None else threading.Event()
        )
        # This thread turns each output tensor into client-ready bytes. At 720p
        # that is an 11 MiB SHM read plus a postprocess copy per engine step,
        # downstream of the last worker-side NVTX range.
        self.enable_nvtx = profiler.nvtx_enabled(enable_nvtx)

        self.in_flight_requests = set()
        self.tensor_uuid_to_metadata_per_request = {}
        # The request's model_kwargs, kept so output postprocessing can
        # honor per-request parameters (e.g. the video container fps).
        self.request_model_kwargs: dict[str, dict] = {}
        self.request_output_state: dict[str, RequestOutputState] = {}

        # Owned by PreprocessWorker (main thread); used only from this thread.
        self.communicator = communicator
        self.tensor_manager = tensor_manager
        # Set by the producers (queue puts, shutdown) and by finished tensor
        # reads; run() blocks on the communicator's poll with it instead of
        # sleeping and sweeping the queues. None keeps the sleep (tests built
        # with a bare communicator).
        self.wakeup = wakeup
        self.idle_wait_ms = 20
        self._encode_pool = ThreadPoolExecutor(
            max_workers=encode_workers(), thread_name_prefix="mstar-encode",
        )
        # Finished encodes, appended by the pool's done callbacks and released
        # by this thread, which alone touches the per-request output state.
        self._encoded: deque = deque()

    def _cleanup_request_state(self, request_id: str, *, force: bool = False) -> None:
        """Release transport and postprocessing state owned by this thread.

        ``force`` unlinks the tensor SHM unconditionally, for a request that
        never reached the conductor (no remote reader will ever drain it); the
        default drain-gated path leaves the unlink to the tensor manager.
        """
        try:
            if force:
                self.tensor_manager.force_cleanup_request(request_id)
            else:
                self.tensor_manager.cleanup_request(request_id)
        finally:
            self._drop_request_state(request_id)

    def _drop_request_state(self, request_id: str) -> None:
        """Forget every per-request dict this thread keeps. Shared by both
        teardown paths so a new dict cannot be dropped by one and leaked by the
        other; a held reorder chunk can be a full 11 MiB 720p frame."""
        self.in_flight_requests.discard(request_id)
        for state_name in (
            "tensor_uuid_to_metadata_per_request",
            "request_model_kwargs",
            "request_output_state",
            "_client_calls",
        ):
            state = getattr(self, state_name, None)
            if state is not None:
                state.pop(request_id, None)

    def _load_media(
        self, file_paths: dict[str, list[str]] | None,
    ) -> tuple[NameToTensorList, dict[str, list[dict]]]:
        """Raw modality tensors and their metadata, keyed ``{modality}_inputs``.
        The files are held to one media budget together."""
        tensors: NameToTensorList = {}
        input_metadata = {}
        if not file_paths:
            return tensors, input_metadata
        counts = {modality: len(paths) for modality, paths in file_paths.items()}
        check_file_counts(counts, self.model.max_input_files())
        budget = RequestMediaBudget(sum(counts.values()))
        for modality, paths in file_paths.items():
            key = f"{modality}_inputs"
            tensors[key] = []
            # TODO: maybe make a class of tensors_and_metadata later (figure out how to use metadata)
            input_metadata[key] = []
            if modality not in ("image", "audio", "video"):
                continue
            load = getattr(self.model, f"load_{modality}")
            for filepath in paths:
                with budget.file():
                    if modality == "audio":
                        check_audio_duration(filepath)
                    out = load(filepath, self.device)
                    if modality == "audio":
                        charge_decoded_audio(out.data, out.metadata["sample_rate"])
                tensors[key].append(out.data)
                input_metadata[key].append(out.metadata)
        return tensors, input_metadata

    def _process_input(
        self, input: PreprocessInput, profile: DataWorkerProfile | None = None,
    ):
        file_states: dict[str, list[tuple[int, int, int]]] = {}
        self._live_requests[input.request_id] = self._live_requests.get(input.request_id, 0) + 1
        self.in_flight_requests.add(input.request_id)

        # First, load raw modality tensors from file_paths (images, audio, video)
        # so they can be passed to process_prompt() below.
        if input.file_paths and self._keyed_labels:
            # Record each file's state before loading it. If a file changes before it
            # is hashed, its KV would be keyed by bytes we never decoded.
            for modality, paths in input.file_paths.items():
                file_states[modality] = [_file_state(path) for path in paths]
        tensors, input_metadata = self._load_media(input.file_paths)


        # Then, tokenize the prompt and let the model augment/transform the
        # tensors dict (e.g., Qwen3-Omni needs to compute pixel_values,
        # image_grid_thw, audio_features, audio_seqlens from the raw tensors
        # loaded above).  process_prompt receives the raw multimodal tensors
        # and returns any additional tensors to merge into the final dict.
        model_kwargs = dict(input.model_kwargs or {})
        # only this worker keys a prompt: a client that sent its own could name
        # another request's pages and be served that request's KV. Likewise
        # only process_prompt sets the model's INTERNAL_KWARGS.
        internal = getattr(self.model, "INTERNAL_KWARGS", ())
        for name in ("prefix_keys", "prefix_tail", "prefix_decode", "kv_layout", *internal):
            model_kwargs.pop(name, None)
        if self.model is not None:
            prompt_tensors = self.model.process_prompt(
                input.text,
                input.input_modalities,
                input.output_modalities,
                tensors=tensors,
                input_metadata=input_metadata,
                prompt_parts=input.prompt_parts,
                **model_kwargs,
            )
            if isinstance(prompt_tensors, ProcessPromptOutput):
                model_kwargs.update(prompt_tensors.metadata)
                prompt_tensors = prompt_tensors.new_input_tensors
            if prompt_tensors:
                tensors.update(prompt_tensors)
            layouts = model_kwargs.pop("kv_layout", {})
            rows = {
                key: {label: layout.row() for label, layout in by_label.items()}
                for key, by_label in layouts.items()
            }
            if rows:
                model_kwargs["kv_layout"] = rows
            # after the update: the chain keys the tensors the request will
            # actually be prefilled with
            if model_kwargs.get("prefix_cache") is not False:
                model_kwargs.update(self._prefix_keys(
                    tensors, layouts, rows, input.file_paths, file_states,
                ))
        elif input.text is not None:
            # Fallback: encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_inputs"] = [torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )]

        # A client stream's first chunk, not a prompt signal: a one-shot upload
        # is a stream of one chunk. Stored before the request reaches the
        # conductor, so a failure here still finds no remote reader; held
        # until the conductor's route says where it goes.
        stream_items = {name: items for name in self._client_edges if (items := tensors.pop(name, None))}
        call = None
        if self._client_edges:
            call = self._client_calls[input.request_id] = ClientCall(ended=set(input.ends))
            stream_items = {name: tensors for name, tensors in
                            self._framed(call, stream_items, call.ended, per_frame=False).items() if tensors}
            self._charge_stream_items(call, stream_items)
        initial_signals = self._store_persisted(input.request_id, tensors)
        if stream_items:
            call.held.append((self._store_held(input.request_id, stream_items), None))

        self.request_model_kwargs[input.request_id] = model_kwargs
        self.request_output_state[input.request_id] = RequestOutputState()
        preprocess_only = getattr(self.model, "PREPROCESS_ONLY_KWARGS", ())
        conductor_kwargs = {k: v for k, v in model_kwargs.items() if k not in preprocess_only}
        msg = ConductorMessage(
            message_type=ConductorMessageType.NEW_REQUEST,
            body=NewRequestConductor(
                request_id=input.request_id,
                initial_signals=initial_signals,
                initial_input_modalities=input.input_modalities,
                initial_output_modalities=input.output_modalities,
                input_metadata=input_metadata,
                model_kwargs=conductor_kwargs,
                client_request_id=input.client_request_id,
                input_open=bool(self._client_edges - input.ends),
            ),
        )
        self.communicator.send("conductor", msg)

        # Record preprocess-side profiling: the moment the fully preprocessed
        # request was handed off to the conductor, plus the per-modality sizes of
        # the *raw* inputs. ``perf_counter`` is consistent here because the worker
        # runs as a thread inside the API server process. (tx/rx are snapshotted
        # directly by the main thread at request completion — see APIServer.)
        # ``profile`` was started at dequeue when metrics are on.
        if profile is None and self.enable_prof:
            profile = DataWorkerProfile(request_id=input.request_id)
        if profile is not None:
            profile.preprocess_finish_time = time.perf_counter()
            if self.enable_prof:
                profile.inputs = self._summarize_inputs(input)
            self.profile_queue.put(profile)

    @staticmethod
    def _start_profile(input: PreprocessInput) -> DataWorkerProfile:
        """The metrics side of a request's profile, taken as it leaves the
        queue: the HTTP layer deletes uploads on a timer, so their sizes are
        read before the work, not after."""
        dequeue_time = time.perf_counter()
        sizes: dict[str, list[int]] = {}
        for modality, paths in (input.file_paths or {}).items():
            # The modalities _process_input loads; it skips any other key.
            if modality not in ("image", "audio", "video"):
                continue
            found = sizes[modality] = []
            for path in paths:
                try:
                    found.append(os.stat(path).st_size)
                except (OSError, ValueError):
                    pass  # gone or not a valid path: its load fails, and the failure is counted
        return DataWorkerProfile(
            request_id=input.request_id,
            dequeue_time=dequeue_time,
            input_modalities=input.input_modalities,
            upload_sizes=sizes,
        )

    def _process_stream_input(self, chunk: ClientAppend) -> None:
        """Load an append, turn it into stream items and send them to their
        consumers. The thread owns an append's files and deletes them here."""
        request_id = chunk.request_id
        try:
            call = self._client_calls.get(request_id)
            if call is None or self._torn_down(request_id):
                return  # ended, failed or aborted: nothing will read it
            items: NameToTensorList = {}
            if chunk.file_paths or chunk.text:
                loaded = self._client_chunk(chunk)
                items = (self._client_audio_items(loaded) if self._audio_only
                         else self.model.process_stream_chunk(loaded))
                unknown = set(items) - self._client_edges
                if unknown:  # the model's bug, not the client's: a 500
                    raise RuntimeError(
                        f"process_stream_chunk returned {sorted(unknown)}, which are "
                        f"not client streams ({sorted(self._client_edges)})"
                    )
                if chunk.text and not any(items.values()):
                    raise ValueError("this model takes no text in an append")
            for name, audio in (chunk.samples or {}).items():
                items.setdefault(name, []).append(torch.from_numpy(audio))
            items = self._without_ended_streams(call, chunk, items)
            ending = chunk.ends - call.ended
            items = self._framed(call, items, ending, per_frame=True)
            self._charge_stream_items(call, items)
            items = {name: tensors for name, tensors in items.items() if tensors}
            if items:
                self._send_stream_items(request_id, call, self._store_held(request_id, items), chunk.on_late)
            call.ended |= ending
            if ending and call.route is not None:  # else they go once the route arrives
                self._send_ends(request_id, call, ending)
        finally:
            unlink_files(chunk.file_paths)

    def _without_ended_streams(
        self, call: ClientCall, chunk: ClientAppend, items: NameToTensorList,
    ) -> NameToTensorList:
        """``items`` less those for streams that ended before this append (by
        the client, or with their consuming partition), which nothing reads
        any more: they are neither charged nor sent, and the append is
        answered as late, as one that names an ended stream is. The API server
        can't tell for a file that names no stream: the model routes it."""
        late = [name for name, tensors in items.items() if tensors and name in call.ended]
        if not late:
            return items
        _answer_late(chunk.on_late, late)
        return {name: tensors for name, tensors in items.items() if name not in late}

    @cached_property
    def _audio_only(self) -> bool:
        """Every client stream declares ``ClientAudio``: this thread turns the
        model's appends into items itself, and the model has no
        ``process_stream_chunk``."""
        return bool(self._client_connections) and all(
            conn.audio is not None for conn in self._client_connections.values())

    def _client_audio_items(self, chunk: ClientChunk) -> NameToTensorList:
        """An append's files as items of the ClientAudio streams they name (or
        of the one stream, when the model has one), each already at its
        stream's rate: resampling a piece of a stream alone would leave a seam
        at every append."""
        if chunk.text:
            raise ValueError("this model takes audio appends only; got text")
        for key, files in chunk.tensors.items():
            if files and key != "audio_inputs":
                raise ValueError(f"this model takes audio appends only; got {key.removesuffix('_inputs')}")
        items: NameToTensorList = {}
        files = chunk.tensors.get("audio_inputs") or []
        metas = chunk.input_metadata.get("audio_inputs") or [{}] * len(files)
        (only, *more) = self._client_connections
        for wav, meta, name in zip(files, metas, chunk.streams.get("audio_inputs") or [], strict=True):
            if name is None and more:
                raise ValueError(f"name the client stream each audio file is for; this model takes "
                                 f"{sorted(self._client_connections)}")
            name = name or only
            rate = self._client_connections[name].audio.sample_rate
            got = meta.get("source_sample_rate", meta.get("sample_rate", rate))
            if got != rate:
                raise ValueError(f"{name} appends must be {rate} Hz audio; got {got} Hz")
            items.setdefault(name, []).append(wav)
        return items

    def _framed(
        self, call: ClientCall, items: NameToTensorList, ending: Collection[str], per_frame: bool,
    ) -> NameToTensorList:
        """``items`` with each ClientAudio stream's cut on its consumer's
        frame: what the stream holds, with the call's tail before it, goes on
        in whole frames and the rest waits for the next append, so every
        route keeps a live call on the frame grid. ``per_frame`` sends an
        append's frames as an item each, the step a live consumer is built
        for; an opening clip goes as one item. A stream that ends here takes
        its tail along, in one item."""
        out = dict(items)
        for name in (items.keys() | set(ending)) & self._client_edges:
            audio = self._client_connections[name].audio
            if audio is None:
                continue
            pieces = [call.tails.pop(name)] if name in call.tails else []
            for wav in items.get(name, ()):
                if wav.dim() != 1:
                    raise ValueError(f"{name} takes mono audio; got a {tuple(wav.shape)} tensor")
                if not torch.isfinite(wav).all():
                    raise ValueError("audio contains NaN or inf samples")
                pieces.append(wav.clamp(-1.0, 1.0))
            wav = torch.cat(pieces) if len(pieces) > 1 else (pieces[0] if pieces else torch.zeros(0))
            whole = wav.numel() if name in ending else wav.numel() - wav.numel() % audio.frame_samples
            if whole < wav.numel():
                call.tails[name] = wav[whole:].clone()
            if not whole:
                out[name] = []
            elif per_frame and name not in ending:
                out[name] = list(wav[:whole].split(audio.frame_samples))
            else:
                out[name] = [wav[:whole]]
        return out

    def _route_client_streams(self, body: ClientStreamRoute) -> None:
        """Where a call's client streams go. The first route sends what the
        call held for it, in order, then the ends its client already sent; a
        later one ends a stream whose consuming partition has ended."""
        request_id = body.request_id
        call = self._client_calls.get(request_id)
        if call is None:
            return  # removed already
        first = call.route is None
        if first:
            call.route = body
        else:
            call.route.workers.update(body.workers)
        call.ended.update(name for name, ranks in body.workers.items() if not ranks)
        if not first or self._torn_down(request_id):
            return  # torn down: REMOVE frees what is held
        held, call.held = call.held, []
        for signals, on_late in held:
            self._send_stream_items(request_id, call, signals, on_late)
        self._send_ends(request_id, call, call.ended)

    def _send_stream_items(
        self, request_id: str, call: ClientCall, signals: dict[str, list[TensorPointerInfo]],
        on_late: Callable[[str], None] | None,
    ) -> None:
        """Send stored items to their streams' consumer ranks, each rank one
        reader of them, and drop the store's own hold; before the route
        arrives, hold them. Items held for a stream the route gives no rank
        (its partition never ran, which no model's does yet) are dropped and
        their append answered as late; an append that also carried items for
        a stream already ended gets a second answer."""
        if call.route is None:
            call.held.append((signals, on_late))
            return
        late = []
        for name, infos in signals.items():
            uuids = [info.uuid for info in infos]
            ranks = call.route.workers.get(name, ())
            if ranks:
                self.tensor_manager.increment_ref_batch_uniform(uuids, len(ranks))
                edge = GraphEdge(next_node=call.route.nodes[name], name=name, tensor_info=infos, is_streaming=True)
                body = InputSignals(
                    request_id=request_id, inputs=[edge], request_info=None,
                    partition_name=self._client_connections[name].to_partition,
                )
                self.communicator.send_each(ranks, WorkerMessage(WorkerMessageType.INPUT_SIGNALS, body))
                call.items_sent[name] = call.items_sent.get(name, 0) + len(infos)
            else:
                late.append(name)
            self.tensor_manager.dereference_batch_uniform(uuids)
        _answer_late(on_late, late)

    def _send_ends(self, request_id: str, call: ClientCall, names: Collection[str]) -> None:
        """Each rank of ``names``' consumers gets one ``producer_done`` from
        ``CLIENT`` per partition, naming the streams it ends there, with the
        items each carried, on the FIFO its items took, so it can't overtake
        them."""
        ends: dict[tuple[str, str], dict[str, int]] = {}
        for name in sorted(names):
            partition = self._client_connections[name].to_partition
            for worker in call.route.workers.get(name, ()):
                ends.setdefault((worker, partition), {})[name] = call.items_sent.get(name, 0)
        for (worker, partition), counts in ends.items():
            self.communicator.send(worker, WorkerMessage(WorkerMessageType.INPUT_SIGNALS, InputSignals(
                request_id=request_id, inputs=[], request_info=None, partition_name=partition,
                producer_done={CLIENT}, producer_done_items=counts,
            )))

    @cached_property
    def _item_limits(self) -> dict[str, ClientStreamLimits]:
        """The client streams with a max_items, which this thread charges as
        their items pass. Read on first use, after the conductor's
        validate_deployment has accepted ``client_streams:``, so a bad block
        is reported with the rest."""
        if self.model is None:
            return {}
        return {
            name: limits
            for name, limits in client_stream_limits(self.model.get_partition_topology(), self._model_config).items()
            if limits.max_items is not None
        }

    @cached_property
    def _turn_tokens(self) -> tuple[int, int] | None:
        # getattr: a thread may run with no model, or a stand-in without it
        return getattr(self.model, "turn_tokens", None)

    def _turn_metadata(self, ids: list[int], loop_indices: NestedLoopIndices | None) -> dict:
        """For a model that declares ``turn_tokens``: ``step``, the iteration
        of the loop that emitted the chunk, so a client pairs its text and
        audio by step without counting chunks; and ``turn``, the marks
        (``"start"`` / ``"end"``) of the turn tokens among ``ids`` in order,
        read before ``postprocess``, which may drop them."""
        meta = {}
        if loop_indices is not None and loop_indices.loop_name_order:
            meta["step"] = loop_indices.loop_indices[loop_indices.loop_name_order[-1]]
        start, end = self._turn_tokens
        if marks := ["start" if t == start else "end" for t in ids if t in (start, end)]:
            meta["turn"] = marks
        return meta

    def _charge_stream_items(self, call: ClientCall, items: NameToTensorList) -> None:
        """Charge a call's new stream items to their streams' ``max_items``;
        an item past it fails the call with a 400."""
        for name, tensors in items.items():
            limits = self._item_limits.get(name)
            if limits is None or not tensors:
                continue
            carried = call.charged
            size = limits.item_size
            total = carried.get(name, 0) + (sum(size(t) for t in tensors) if size else len(tensors))
            if total > limits.max_items:
                raise ValueError(
                    f"the call's {name!r} stream passed {limits.max_items} {limits.unit}, "
                    f"this server's limit (client_streams.{name}.max_items)"
                )
            carried[name] = total

    def _client_chunk(self, chunk: ClientAppend) -> ClientChunk:
        """``chunk`` as ``process_stream_chunk`` takes it: its files loaded."""
        tensors, input_metadata = self._load_media(chunk.file_paths)
        named = chunk.file_streams or {}
        return ClientChunk(
            text=chunk.text or None,
            tensors=tensors,
            input_metadata=input_metadata,
            streams={
                f"{modality}_inputs": list(named.get(modality) or [None] * len(paths))
                for modality, paths in (chunk.file_paths or {}).items()
            },
            model_kwargs=chunk.model_kwargs or {},
            request_kwargs=self.request_model_kwargs.get(chunk.request_id, {}),
        )

    def _store(
        self, request_id: str, tensors: NameToTensorList,
    ) -> tuple[dict[str, list[TensorPointerInfo]], list[TensorPointerInfo]]:
        """Store tensors for workers to read; the descriptors by name, and flat."""
        signals = self.tensor_manager.store_and_return_tensor_info(
            request_id=request_id, tensors=tensors,
        )
        infos = [info for name_infos in signals.values() for info in name_infos]
        self.tensor_manager.register_for_send(request_id=request_id, tensor_infos=infos)
        return signals, infos

    def _store_persisted(
        self, request_id: str, tensors: NameToTensorList,
    ) -> dict[str, list[TensorPointerInfo]]:
        """A request's initial signals, held until the conductor releases them
        with their reader count."""
        if not tensors:
            return {}
        signals, infos = self._store(request_id, tensors)
        for info in infos:
            self.tensor_manager.set_persist(info.uuid, persist=True)
        return signals

    def _store_held(
        self, request_id: str, items: NameToTensorList,
    ) -> dict[str, list[TensorPointerInfo]]:
        """Client stream items, with one reference of the store's own until
        they are sent (``_send_stream_items``)."""
        signals, infos = self._store(request_id, items)
        self.tensor_manager.increment_ref_batch_uniform([info.uuid for info in infos], 1)
        return signals

    def _prefix_keys(
        self, tensors: dict, layouts: dict, rows: dict, file_paths: dict | None,
        file_states: dict[str, list[tuple[int, int, int]]],
    ) -> dict:
        """Key each page of every label's keyed spans, by resource and label.

        Returns the kwargs that carry the keys, the prompt tail past the last whole
        page, and the output tensor each label's sampled ids arrive in. The keys
        are unrooted. Each keyed span's row in ``rows`` gains its digest and end key.
        """
        if not self._keyed_labels:
            return {}
        keys: dict[str, dict[str, list[bytes]]] = {}
        tails: dict[str, dict[str, list[int]]] = {}
        decode: dict[str, dict[str, str]] = {}
        for resource_key, by_label in self._keyed_labels.items():
            page_size = self._prefix_page_sizes[resource_key]
            for label, kv_label in by_label.items():
                layout = layouts.get(resource_key, {}).get(label)
                keyed = () if layout is None else layout.spans[:layout.keyed_count()]
                if not keyed:
                    continue
                try:
                    digests = [
                        self._digest(span, file_paths, file_states)
                        if span.kind == "digest" else None
                        for span in keyed
                    ]
                except _FileChangedError as changed:
                    logger.warning(
                        "%s changed while it was loaded, so its request runs with the prefix cache off", changed,
                    )
                    return {"prefix_cache": False}
                pages, items = self._page_layout(
                    kv_label.keys, keyed, digests, tensors, page_size,
                )
                page_keys = chain(pages, items)
                keys.setdefault(resource_key, {})[label] = page_keys
                total = sum(span.length for span in keyed)
                tails.setdefault(resource_key, {})[label] = (
                    pages[-1] if total % page_size else []
                )
                at = 0
                span_rows = rows[resource_key][label][0][:len(keyed)]
                for row, span, digest in zip(span_rows, keyed, digests, strict=True):
                    at += span.length
                    page, end = divmod(at, page_size)
                    row += [digest, end_key(
                        page_keys[page - 1] if page else b"", pages[page], items.get(page, ()), end,
                    ) if digest is not None and end else None]
                if kv_label.keys.growth:
                    decode.setdefault(resource_key, {})[label] = kv_label.keys.tensor
        if not keys:
            return {}
        kwargs = {"prefix_keys": keys, "prefix_tail": tails}
        if decode:
            kwargs["prefix_decode"] = decode
        return kwargs

    def _page_layout(
        self, prefix_keys: PrefixKeys, layout: tuple[Span, ...],
        digests: list[bytes | None], tensors: dict, page_size: int,
    ) -> tuple[list[list[int]], dict[int, list[PageItem]]]:
        """Check the layout's spans and return each page's ids and items."""
        walks = set(prefix_keys.walks)
        entries: dict[str, int] = {}
        total = sum(span.length for span in layout)
        pages: list[list[int]] = [[] for _ in range(-(-total // page_size))]
        at = 0
        for span in layout:
            if span.kind not in ("ids", "digest"):
                raise ValueError(f"a span of kind {span.kind!r}")
            if span.walk not in walks:
                raise ValueError(
                    f"a {span.kind} span names walk {span.walk!r}, which the "
                    f"label's keys do not name; they name {sorted(walks)}"
                )
            touched = range(at // page_size, -(-(at + span.length) // page_size))
            if span.kind == "ids":
                entry = entries.get(span.source, 0)
                entries[span.source] = entry + 1
                if entry >= len(tensors.get(span.source, ())):
                    raise ValueError(f"ids span {entry} of {span.source!r} has no entry to key")
                ids = tensors[span.source][entry].flatten().tolist()
                if not len(ids) == span.length == span.advance:
                    raise ValueError(
                        f"an ids span of length {span.length} and advance "
                        f"{span.advance} over {len(ids)} ids of {span.source!r}; "
                        "an ids span fills one slot and one position per id"
                    )
                for page in touched:
                    first = max(at, page * page_size)
                    last = min(at + span.length, (page + 1) * page_size)
                    pages[page].extend(ids[first - at:last - at])
            elif span.length <= 0:
                raise ValueError("a zero-length digest span has no slot, so no page to fold into")
            at += span.length
        return pages, page_items(zip((span.length for span in layout), digests, strict=True), page_size)

    def _digest(
        self, span: Span, file_paths: dict | None,
        file_states: dict[str, list[tuple[int, int, int]]],
    ) -> bytes:
        """Hash the whole file a digest span names, with everything else that decides its KV.

        The digest is never cut short or turned into an id. Either would let two
        images share a key, and one request would be served another's KV.
        """
        if type(span.params) is not tuple or not all(
            type(param) in (type(None), bool, int, float, str, bytes)
            for param in span.params
        ):
            raise ValueError(f"digest params {span.params!r} are not a tuple of plain values, whose repr is exact")
        modality, index = span.source
        paths = (file_paths or {}).get(modality, ())
        if not 0 <= index < len(paths):
            raise ValueError(f"a digest span names {modality} {index}, but the request sent {len(paths)}")
        raw = _sha256(paths[index])
        if _file_state(paths[index]) != file_states[modality][index]:
            raise _FileChangedError(paths[index])
        return fingerprint(
            raw, modality, self.model.preprocess_fingerprint(), span.advance,
            # Use repr, not str. True and "True" can preprocess differently
            *(repr(param) for param in span.params),
        )

    @staticmethod
    def _summarize_inputs(input: PreprocessInput) -> list[InputInfo]:
        """Aggregate the *raw* (pre-decoding) inputs into per-modality sizes.

        Reports the bytes the client actually sent — uploaded file sizes on
        disk and the UTF-8 length of the prompt — rather than the much larger
        decoded tensors (e.g. a compressed JPEG vs. its raw RGB tensor), so the
        numbers line up with what a user thinks of as "input size".
        """
        infos = []
        if input.text:
            infos.append(InputInfo(
                modality="text",
                count=1,
                total_bytes=len(input.text.encode("utf-8")),
            ))
        for modality, paths in (input.file_paths or {}).items():
            total_bytes = 0
            for path in paths:
                try:
                    total_bytes += os.path.getsize(path)
                except OSError:
                    pass  # file already cleaned up / unreadable — count as 0
            infos.append(InputInfo(
                modality=modality,
                count=len(paths),
                total_bytes=total_bytes,
            ))
        return infos

    def _fail_request(
        self, request_id: str, exc: BaseException, stage: str, count: int = 1,
        sequence: int | None = None, abort: bool = True, traceback: bool = True,
    ):
        """Report a per-request data-worker failure to the API server.

        ``count`` error chunks are emitted because the API server's
        ``per_request_reading_tensors`` accounting is one decrement per chunk:
        a failure that kills N queued tensors has to answer for all N, or the
        request looks like it still has reads outstanding.

        ``sequence`` is the output slot the failed tensor held. Its error chunk
        takes that slot in the reorder buffer; put straight on ``out_queue`` it
        would leave every later sequence held in ``pending`` until the TTL.

        ``abort`` (False for a request the conductor has not seen) stops the
        request's walk, since its client already has its answer.

        ``traceback`` False logs one line, for an error already logged where
        it happened.
        """
        text, _ = client_error_text(
            exc, f"{stage} failed for request {request_id}", log=logger,
            traceback=traceback,
        )
        status = 400 if isinstance(exc, (ValueError, TypeError)) else 500
        chunks = [
            ResultChunk(
                request_id=request_id,
                modality="error",
                data=f"{stage} failed: {text}".encode("utf-8"),
                metadata={"status": status},
            )
            for _ in range(max(count, 1))
        ]
        if sequence is not None and self._sequence_unanswered(request_id, sequence):
            self._queue_completed_output(request_id, sequence, chunks.pop())
        for chunk in chunks:
            self.out_queue.put(chunk)
        if abort:
            self._abort_failed_request(request_id)

    def _torn_down(self, request_id: str) -> bool:
        """Being torn down, or already handed back by the API server: this
        thread starts no more work for it."""
        return (
            request_id in self._draining_rids
            or request_id in self._reads_done_sent
            or request_id not in self.in_flight_requests
        )

    def _abort_failed_request(self, request_id: str) -> None:
        """Stop the walk of a request this thread failed. Left running, it
        would produce every remaining output only for each to be read and
        dropped. Skipped for a request already being torn down, or already
        handed back by the API server (whose own abort or finish covers it)."""
        if self._torn_down(request_id):
            return
        logger.info("Aborting request %s: one of its outputs failed", request_id)
        self._send_abort(request_id)

    def _send_abort(self, request_id: str) -> None:
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.ABORT_REQUEST,
                body=AbortRequest(request_id=request_id),
            ),
        )
        # Begin draining our own reads immediately, overlapping with the
        # conductor round-trip (it also sends a DrainRequest).
        # None live: already removed, or still waiting for preprocessing,
        # which drops it unsent (the API server let go of it first).
        if request_id in self._live_requests:
            self._begin_drain(request_id)

    def _sequence_unanswered(self, request_id: str, sequence: int) -> bool:
        """True if no chunk has been queued for ``sequence`` yet."""
        state = self.request_output_state.get(request_id)
        if state is None:
            return True
        return sequence >= state.next_emit and sequence not in state.pending

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        result.graph_edge.name = f"{result.modality}_output"
        futures = self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )
        wakeup = getattr(self, "wakeup", None)
        if wakeup is not None and futures:
            # a finished read ends the idle wait, so the chunk goes out at once
            wakeup.register_futures(futures)
        if result.request_id not in self.tensor_uuid_to_metadata_per_request:
            self.tensor_uuid_to_metadata_per_request[result.request_id] = {}
        state = self.request_output_state.setdefault(
            result.request_id, RequestOutputState()
        )
        for tensor_info in result.graph_edge.tensor_info:
            self.tensor_uuid_to_metadata_per_request[result.request_id][
                tensor_info.uuid] = result.metadata
            state.order[tensor_info.uuid] = (state.next_sequence, result.loop_indices)
            state.next_sequence += 1

    def _emit_inline_token(self, item: InlineToken) -> None:
        """Detokenize a value that arrived inline and queue its chunk."""
        request_id = item.request_id
        state = self.request_output_state.setdefault(
            request_id, RequestOutputState()
        )
        sequence = state.next_sequence
        state.next_sequence += 1
        try:
            tensor = torch.tensor([item.value], dtype=torch.int64)
            postprocessed = self.model.postprocess(
                tensor, item.modality,
                request_kwargs=self.request_model_kwargs.get(request_id),
            )
            self._queue_completed_output(
                request_id,
                sequence,
                ResultChunk(
                    request_id=request_id,
                    modality=item.modality,
                    data=postprocessed,
                    metadata=(self._turn_metadata([item.value] if item.modality == "text" else [], item.loop_indices)
                              if self._turn_tokens else {}),
                ),
            )
        except Exception as exc:  # noqa: BLE001 — must reach the client
            self._fail_request(
                request_id, exc, f"{item.modality} output postprocessing",
                sequence=sequence,
            )

    def _queue_completed_output(
        self,
        request_id: str,
        sequence: int,
        chunk: ResultChunk,
    ) -> None:
        state = self.request_output_state.setdefault(request_id, RequestOutputState())
        if sequence in state.pending:
            raise RuntimeError(
                f"duplicate completed output sequence {sequence} for request {request_id}"
            )
        state.pending[sequence] = chunk

        while state.next_emit in state.pending:
            ready = state.pending.pop(state.next_emit)
            if ready.modality == "video_frame":
                ready.metadata["frame_index"] = state.frame_index
                state.frame_index += ready.metadata["frame_count"]
            self.out_queue.put(ready)
            state.next_emit += 1

    def _discard_result_tensor(
        self, result: ResultTensors
    ):
        # The request is gone, so don't start a read — just ack the tensors back
        # to the producing worker so it can free the source buffers.
        self.tensor_manager.ack_unread_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )

    def _fail_unread_outputs(self, failed: FailedRead) -> None:
        """Fail the request of a read that will not be delivered: one error
        chunk in the output slot of each tensor it would have delivered. The
        manager already acked or kept its buffers."""
        request_id = failed.request_id
        state = self.request_output_state.get(request_id)
        metadata = self.tensor_uuid_to_metadata_per_request.get(request_id, {})
        for edge in failed.graph_edges:
            for info in edge.tensor_info:
                metadata.pop(info.uuid, None)
                slot = state.order.pop(info.uuid, None) if state is not None else None
                self._fail_request(
                    request_id, failed.error, "output tensor read",
                    sequence=slot[0] if slot is not None else None, traceback=False,
                )

    def _postprocess_inline(
        self, request_id: str, sequence: int, modality: str, uuid: int, tensor: torch.Tensor,
        loop_indices: NestedLoopIndices | None,
    ) -> None:
        if self.enable_nvtx:
            profiler.range_push(f"dataworker.postprocess.{modality}")
        postprocessed = self.model.postprocess(
            tensor, modality,
            request_kwargs=self.request_model_kwargs.get(request_id),
        )
        if self.enable_nvtx:
            profiler.range_pop()
            profiler.mark(
                f"dataworker.postprocessed.bytes[{len(postprocessed)}]"
            )

        chunk_metadata = self.tensor_uuid_to_metadata_per_request[request_id][uuid] or {}
        if modality in ("text", "audio") and self._turn_tokens:
            ids = tensor.flatten().tolist() if modality == "text" else []
            chunk_metadata = {**chunk_metadata, **self._turn_metadata(ids, loop_indices)}
        # Audio is emitted as headerless 16-bit PCM; surface the
        # model's output sample rate + channel count so clients can
        # wrap it.
        if modality == "audio" and self.model is not None:
            chunk_metadata = {
                **chunk_metadata,
                "sample_rate": self.model.get_output_sample_rate("audio"),
                "num_channels": self.model.get_output_audio_channels("audio"),
            }
        elif modality == "video_frame" and self.model is not None:
            chunk_metadata = _video_frame_metadata(
                tensor,
                fps=self.model.get_output_frame_rate(
                    "video_frame",
                    request_kwargs=self.request_model_kwargs.get(request_id),
                ),
                # Assigned from emitted order in
                # _queue_completed_output after any earlier
                # asynchronous reads have completed.
                frame_index=0,
                metadata=chunk_metadata,
            )
            expected_bytes = (
                chunk_metadata["frame_count"]
                * chunk_metadata["height"]
                * chunk_metadata["width"]
                * 3
            )
            if len(postprocessed) != expected_bytes:
                raise ValueError(
                    "video_frame payload length does not match its RGB24 shape: "
                    f"expected {expected_bytes} bytes, got {len(postprocessed)}"
                )

        if self.enable_nvtx:
            profiler.range_push("dataworker.queue_output")
        self._queue_completed_output(
            request_id,
            sequence,
            ResultChunk(
                request_id=request_id,
                modality=modality,
                data=postprocessed,
                metadata=chunk_metadata,
            ),
        )
        if self.enable_nvtx:
            profiler.range_pop()

    def _submit_encode(
        self, request_id: str, sequence: int, modality: str, uuid: int, tensor: torch.Tensor,
    ) -> None:
        """Encode on the pool; :meth:`_release_encoded` queues the chunk. The
        transport keeps its reference to ``tensor`` until then."""
        state = self.request_output_state[request_id]
        metadata = self.tensor_uuid_to_metadata_per_request[request_id][uuid] or {}
        future = self._encode_pool.submit(
            self._encode, tensor, modality, self.request_model_kwargs.get(request_id),
        )
        # registered before the wakeup's, so the entry is queued when the thread wakes
        future.add_done_callback(lambda done: self._encoded.append(
            (done, request_id, state, sequence, modality, metadata, uuid),
        ))
        if self.wakeup is not None:
            self.wakeup.register_future(future)

    def _encode(self, tensor: torch.Tensor, modality: str, request_kwargs: dict | None) -> bytes:
        if not self.enable_nvtx:
            return self.model.postprocess(tensor, modality, request_kwargs=request_kwargs)
        profiler.range_push(f"dataworker.encode.{modality}")
        try:
            return self.model.postprocess(tensor, modality, request_kwargs=request_kwargs)
        finally:
            profiler.range_pop()

    def _release_encoded(self) -> bool:
        """Queue each finished encode's chunk in its request's output order."""
        encoded = getattr(self, "_encoded", None)
        if not encoded:
            return False
        while encoded:
            future, request_id, state, sequence, modality, metadata, uuid = encoded.popleft()
            self.tensor_manager.dereference(uuid)
            # The request was torn down while this encoded, or its rid reused since.
            if self.request_output_state.get(request_id) is not state:
                continue
            try:
                data = future.result()
            except Exception as exc:  # noqa: BLE001 — must reach the client
                self._fail_request(
                    request_id, exc, f"{modality} output postprocessing", sequence=sequence,
                )
                continue
            self._queue_completed_output(
                request_id, sequence,
                ResultChunk(request_id=request_id, modality=modality, data=data, metadata=metadata),
            )
        return True

    def _process_read_tensors(self):
        did_work = False
        ready = self.tensor_manager.get_ready_tensors()
        for failed in self.tensor_manager.take_failed_reads():
            did_work = True
            self._fail_unread_outputs(failed)
        for request_id, graph_edges in ready.items():
            did_work = True
            for graph_edge in graph_edges:
                modality = graph_edge.name.replace("_output", "")

                for tensor_info in graph_edge.tensor_info:
                    logger.debug("Reading in OUTPUT tensor %s with uuid %s", graph_edge.name, tensor_info.uuid)
                    # Reading and postprocessing an output tensor is per-request
                    # work, so a raise here is attributable: fail this request
                    # and keep draining everyone else's tensors. Letting it
                    # escape to run()'s catch-all would abandon the rest of this
                    # pass and leave the client waiting on the request timeout.
                    sequence = None
                    pooled = False
                    try:
                        sequence, loop_indices = (
                            self.request_output_state[request_id].order[tensor_info.uuid]
                        )
                        logger.debug(
                            "Postprocessing output sequence %d for request %s at %s",
                            sequence,
                            request_id,
                            loop_indices,
                        )
                        if self.enable_nvtx:
                            profiler.range_push(f"dataworker.get_tensor.{modality}")
                        tensor = self.tensor_manager.get_tensor(tensor_info.uuid)
                        if self.enable_nvtx:
                            profiler.range_pop()
                        if self._encode_pool is not None and pooled_output(modality, loop_indices):
                            self._submit_encode(request_id, sequence, modality, tensor_info.uuid, tensor)
                            pooled = True
                        else:
                            self._postprocess_inline(
                                request_id, sequence, modality, tensor_info.uuid, tensor, loop_indices,
                            )
                    except Exception as exc:  # noqa: BLE001 — must reach the client
                        self._fail_request(
                            request_id, exc, f"{modality} output postprocessing",
                            sequence=sequence,
                        )
                    self.tensor_uuid_to_metadata_per_request.get(
                        request_id, {}
                    ).pop(tensor_info.uuid, None)
                    state = self.request_output_state.get(request_id)
                    if state is not None:
                        state.order.pop(tensor_info.uuid, None)
                    if not pooled:
                        self.tensor_manager.dereference(tensor_info.uuid)
        return did_work

    def _process_messages(self):
        did_work = False
        for message in self.communicator.get_all_new_messages():
            did_work = True
            if message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                body: TensorReceived = message.body
                self.tensor_manager.dereference_batch(
                    list(body.successful_tensors),
                    list(body.successful_tensors.values()),
                )
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                body: UnpersistTensors = message.body
                for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
                    self.tensor_manager.increment_ref(uuid, n=ref_cnt)
                    self.tensor_manager.set_persist(uuid, persist=False)
            elif message.message_type == WorkerMessageType.CLIENT_STREAM_ROUTE:
                try:
                    self._route_client_streams(message.body)
                except Exception as exc:  # noqa: BLE001 — this call's failure; the batch's other messages still apply
                    self._fail_request(message.body.request_id, exc, "Client stream route")
            elif message.message_type == WorkerMessageType.DRAIN_REQUEST:
                body: DrainRequest = message.body
                self._begin_drain(body.request_id)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                body: RemoveRequest = message.body
                self._hard_cleanup(body.request_id)
        return did_work

    def _send_enable_metrics_if_requested(self) -> None:
        requested = self.worker_metrics_requested
        if requested is not None and requested.is_set():
            requested.clear()
            self.communicator.send("conductor", ConductorMessage(
                message_type=ConductorMessageType.ENABLE_METRICS, body=EnableMetrics(),
            ))

    def _finish_reading(self, request_id: str, drained: bool) -> None:
        """The API server is done with this rid's outputs. ``drained`` (every
        chunk delivered) means no read can still be in flight, so ACK straight
        away, unless a failed read's transfers are still running: its error
        chunks count as delivered, as they do for a rid this thread is already
        draining (one it failed), though no read produced them. Otherwise
        delivery was abandoned (TTL, client gone) with reads possibly still
        running, so gate the ACK on them; ACKing there would let the conductor
        unlink under a read."""
        if request_id not in self._live_requests:
            return
        if (
            drained and request_id not in self._draining_rids
            and not self.tensor_manager.has_inflight_reads(request_id)
        ):
            self._send_reads_done(request_id)
        else:
            self._begin_drain(request_id)

    def _begin_drain(self, request_id: str) -> None:
        """Stop reading this rid; ACK READS_DONE once in-flight reads finish.
        Idempotent — abort self-initiates while the conductor may also drive it."""
        self._draining_rids.add(request_id)
        self._complete_drain_if_ready(request_id)

    def _complete_drain_if_ready(self, request_id: str) -> None:
        if request_id not in self._draining_rids:
            return
        if request_id in self._reads_done_sent:
            return
        if self.tensor_manager.has_inflight_reads(request_id):
            return  # let _process_read_tensors resolve the futures; retry
        self._send_reads_done(request_id)

    def _send_reads_done(self, request_id: str) -> None:
        if request_id in self._reads_done_sent:
            return
        self._reads_done_sent.add(request_id)
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.READS_DONE,
                body=ReadsDone(
                    request_id=request_id,
                    entity_id="api_server_preprocess_worker",
                ),
            ),
        )

    def _hard_cleanup(self, request_id: str) -> None:
        """Phase-2 teardown: unconditionally drop all tensor state for the rid
        (unlink the input-signal SHM). Safe now that every reader has drained."""
        self._removed(request_id)
        self.tensor_manager.force_cleanup_request(request_id)
        self._drop_request_state(request_id)

    def _removed(self, request_id: str) -> None:
        """A request under the rid is gone: no REMOVE_REQUEST comes for it again."""
        self._draining_rids.discard(request_id)
        self._reads_done_sent.discard(request_id)
        live = self._live_requests.pop(request_id, 0) - 1
        if live > 0:
            self._live_requests[request_id] = live

    def _process_releases(self) -> bool:
        """Apply the API server's aborts and finished_reading calls. One that
        fails is logged, and the others still apply."""
        did_work = False
        while not self.abort_request_queue.empty():
            did_work = True
            rid = self.abort_request_queue.get()
            try:
                self._send_abort(rid)
            except Exception:  # noqa: BLE001 — that release alone
                logger.exception("Abort of request %r failed", rid)
        while not self.reads_done_queue.empty():
            did_work = True
            rid, drained = self.reads_done_queue.get()
            try:
                self._finish_reading(rid, drained)
            except Exception:  # noqa: BLE001 — that release alone
                logger.exception("Release of request %r failed", rid)
        return did_work

    def _publish_transfers(self, transfers: EdgeTransfers) -> None:
        """Hand the API server this thread's transfer counts (``latest``) once
        per status interval, from the thread that records them."""
        now = time.perf_counter()
        if now < self._next_transfer_publish:
            return
        self._next_transfer_publish = now + STATUS_INTERVAL_S
        transfers.publish()

    def _next_inputs(self) -> list[PreprocessInput | ClientAppend]:
        """This pass's input: up to MAX_APPENDS_PER_PASS items in arrival
        order, ending at the first new request (a preprocess each, seconds for
        media), so output delivery gets its turn between requests."""
        batch: list[PreprocessInput | ClientAppend] = []
        while len(batch) < self.MAX_APPENDS_PER_PASS:
            try:
                item = self.in_queue.get_nowait()
            except queue.Empty:
                break
            batch.append(item)
            if not isinstance(item, ClientAppend):
                break
        return batch

    def _process_waiting_input(self, pre_input: PreprocessInput | ClientAppend) -> None:
        if not self._is_pending(pre_input.request_id):
            # The API server let go of it while it waited (aborted, or its
            # call ended): nothing will read it, so nothing is sent. An abort
            # that lands after this check is sent behind what this sends,
            # from this thread through the same outbox, so the conductor
            # tears the request down as any running one.
            if isinstance(pre_input, ClientAppend):
                unlink_files(pre_input.file_paths)
            return
        if isinstance(pre_input, ClientAppend):
            try:
                self._process_stream_input(pre_input)
            except Exception as exc:  # noqa: BLE001 — must reach the client
                self._fail_request(pre_input.request_id, exc, "stream input")
            return
        profile = None
        try:
            if self.timing_requested.is_set():
                profile = self._start_profile(pre_input)
            self._process_input(pre_input, profile)
        except Exception as exc:  # noqa: BLE001 — any failure must reach the client
            if profile is not None:
                profile.failed = True
                self.profile_queue.put(profile)
            # A request whose media load or prompt processing fails never
            # reaches the conductor, so nothing downstream would ever complete
            # it; surface the failure as an error chunk instead of leaving the
            # client to hit the server timeout.
            self._fail_request(pre_input.request_id, exc, "preprocessing", abort=False)
            # Never reached the conductor, so there are no remote readers to
            # race: hard-drop the (possibly persisted) input signals directly.
            self._cleanup_request_state(pre_input.request_id, force=True)
            self._removed(pre_input.request_id)

    def _idle_wait(self):
        """How a pass that found nothing waits: on the communicator's poll,
        ended by a message, a producer's signal or a finished read, with
        ``idle_wait_ms`` as the backstop; or a 1 ms sleep when the thread has
        no wakeup or its communicator cannot poll one."""
        # getattr: teardown tests build a thread without its constructor
        wakeup = getattr(self, "wakeup", None)
        comm = getattr(self, "communicator", None)
        register = getattr(comm, "register_event_for_poll", None)
        wait = getattr(comm, "wait_for_work", None)
        if wakeup is None or not callable(register) or not callable(wait):
            return lambda: time.sleep(0.001)
        register(wakeup)
        idle_ms = getattr(self, "idle_wait_ms", 20)
        return lambda: wait(idle_ms)

    def run(self):
        # A thread told to stop before it starts (teardown tests build one with
        # only its queues) has nothing to warm up.
        if not self.stop_event.is_set():
            warm_up_model(getattr(self, "model", None))
        if not isinstance(self.communicator, OutboxCommunicator):
            self.communicator = OutboxCommunicator(self.communicator)
        idle_wait = self._idle_wait()
        while not self.stop_event.is_set():
            did_work = False
            sends_pending = False
            try:
                sends_pending = self.communicator.flush()
                did_work = self._process_messages()
                self._send_enable_metrics_if_requested()
                # Output delivery is latency-sensitive: the API server holds a
                # finished request open for its final chunks only briefly, so
                # queued read-starts / acks / cleanups must never wait behind a
                # multi-second media preprocess. Drain them fully every pass
                # before taking new input below.
                while not self.result_tensor_queue.empty():
                    did_work = True
                    result = self.result_tensor_queue.get()
                    if isinstance(result, InlineToken):
                        # Nothing to read or ack: postprocess in place, in
                        # arrival order with the tensor results.
                        if result.request_id not in self._draining_rids:
                            self._emit_inline_token(result)
                        continue
                    # Draining for teardown: don't start new reads — ack the
                    # tensors back so the producing worker can free its buffers.
                    if result.request_id in self._draining_rids:
                        self._discard_result_tensor(result)
                        continue
                    try:
                        self._read_result_tensor(result)
                    except Exception as exc:  # noqa: BLE001 — must reach the client
                        # The read never started, so none of this edge's
                        # tensors will ever produce a chunk; answer for all of
                        # them at once.
                        self._fail_request(
                            result.request_id, exc,
                            f"{result.modality} output transfer",
                            count=len(result.graph_edge.tensor_info),
                        )
                did_work = self._process_releases() or did_work
                while not self.discard_tensor_queue.empty():
                    did_work = True
                    self._discard_result_tensor(self.discard_tensor_queue.get())
                while not self.cleanup_request_queue.empty():
                    did_work = True
                    req_id = self.cleanup_request_queue.get()
                    self._cleanup_request_state(req_id)
                did_work = self._process_read_tensors() or did_work
                did_work = self._release_encoded() or did_work
                # Reads may have just resolved; ACK any drains now free of them.
                for rid in list(self._draining_rids):
                    self._complete_drain_if_ready(rid)
                # New input. None while a peer can't take what we already
                # sent: as a blocking send did, that holds new work back, so
                # the outbox stays short and inputs wait undecoded.
                for pre_input in [] if sends_pending else self._next_inputs():
                    did_work = True
                    # again here: a request submitted after the event was set
                    # sees it set, so the conductor gets ENABLE_METRICS first
                    self._send_enable_metrics_if_requested()
                    # Releases queued before this request was submitted belong
                    # to an earlier request under its id: apply them first.
                    self._process_releases()
                    if self._client_edges:
                        # Routes too, so a call's held items go out before this
                        # input, which may be a multi-second media preprocess.
                        try:
                            self._process_messages()
                        except Exception:  # noqa: BLE001 — the pass's inputs still go
                            logger.exception("PreprocessWorkerThread message handling failed")
                    try:
                        self._process_waiting_input(pre_input)
                    except Exception:  # noqa: BLE001 — the pass's other input still goes
                        logger.exception("Input for request %s failed", pre_input.request_id)
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            # getattr: tests drive this loop with stand-in tensor managers
            transfers = getattr(self.tensor_manager, "transfer_stats", None)
            if transfers is not None:
                self._publish_transfers(transfers)
            if sends_pending and not did_work:
                time.sleep(0.001)       # a peer is full: retry soon, keep reading
            elif not did_work:
                idle_wait()

        # Stopping, and nothing will send the RemoveRequest for what is still
        # tracked (the conductor is gone or going), so drop it here rather than
        # leave the input signals of in-flight requests in /dev/shm.
        for request_id in list(self.in_flight_requests):
            self._hard_cleanup(request_id)
        if self._encode_pool is not None:
            self._encode_pool.shutdown(wait=False, cancel_futures=True)
