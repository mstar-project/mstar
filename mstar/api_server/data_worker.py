

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from mstar.graph.loop_indices import NestedLoopIndices

try:
    import torchaudio  # noqa: F401 — probes availability; real usage in callers
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError, OSError):
    VideoDecoder = None

from mstar.api_server.request_types import (
    DataWorkerProfile,
    PreprocessInput,
    ResultChunk,
    ResultTensors,
)
from mstar.communication.communicator import BaseCommunicator, CommProtocol, make_communicator
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.model.base import Model, ProcessPromptOutput
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.utils import profiler
from mstar.utils.ipc_format import (
    AbortRequest,
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    NewRequestConductor,
    ReadsDone,
    RemoveRequest,
    TensorReceived,
    UnpersistTensors,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)


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


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


def _kv_page_sizes(model: Model, model_config: dict) -> dict[str, int]:
    """Page size per KV resource, after this deployment's YAML has been applied."""
    specs = model.get_node_resources()
    apply_yaml_overrides(specs, model_config)
    return {
        spec.resource_key: spec.config.page_size
        for spec in specs
        if isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig)
    }


NameToLoopIndices = dict[str, NestedLoopIndices]


class PreprocessWorker:
    def __init__(
        self,
        model: Model | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        tcp_transfer_device="",
        enable_prof: bool=False,
        enable_nvtx: bool=False,
        model_config: dict | None = None,
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.abort_request_queue = queue.Queue()
        self.reads_done_queue = queue.Queue()
        self.discard_tensor_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.profile_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.per_request_reading_tensors = {}
        self.output_loop_idxs: dict[str, NameToLoopIndices] = {}

        # Build the communicator + tensor manager here (main thread) and hand
        # them to the worker thread, rather than constructing them inside it.
        # The socket is only *used* from the worker thread, but owning the
        # tensor manager here lets the main thread read its tx/rx profiling
        # directly once a request is done (no cross-thread queue / race).
        self.communicator = make_communicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        )  # only used to send (from the worker thread)
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
                communicator=self.communicator,
                tensor_manager=self.tensor_manager,
                model=model,
                enable_prof=enable_prof,
                enable_nvtx=enable_nvtx,
                model_config=model_config,
            )
        )
        self.thread.start()

    def new_request(self, input: PreprocessInput):
        self.output_loop_idxs[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)

    def abort_request(self, request_id: str):
        # Forward the abort to the conductor and begin draining our own reads
        # (the worker thread ACKs READS_DONE once they finish). The hard cleanup
        # of the persisted input signals waits for the conductor's
        # REMOVE_REQUEST, so a worker still reading them can't be unlinked
        # out from under it. Drop only the main-thread bookkeeping here.
        self.abort_request_queue.put(request_id)
        self.output_loop_idxs.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def finished_reading(self, request_id: str, drained: bool = True):
        """The API server is done with this request's outputs. Tell the
        conductor via READS_DONE; it will send REMOVE_REQUEST to trigger the
        hard cleanup. ``drained`` means every chunk was delivered, so no read
        can still be in flight and the ACK can go out immediately; pass False
        when delivery was abandoned (TTL, client gone) and the ACK must wait
        for any in-flight read."""
        self.reads_done_queue.put((request_id, drained))
        self.output_loop_idxs.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def new_result_tensors(self, input: ResultTensors):
        name = input.graph_edge.name
        if input.request_id not in self.output_loop_idxs:
            # Request was removed while this output was still in flight; ack the
            # tensors so the producing worker can reclaim them rather than leak.
            logger.debug("Late result_tensors for cleaned-up request %s, acking and dropping", input.request_id)
            self.discard_result_tensors(input)
            return

        self.output_loop_idxs[input.request_id][name] = input.loop_indices.max(
            self.output_loop_idxs[input.request_id].get(name, None)
        )

        self.per_request_reading_tensors[input.request_id] += len(input.graph_edge.tensor_info)
        logger.debug(
            "Data worker reading queue for request %s increased to length %d",
            input.request_id,  self.per_request_reading_tensors[input.request_id]
        )
        self.result_tensor_input_queue.put(input)

    def discard_result_tensors(self, input: ResultTensors):
        """Ack and drop result tensors for an already-removed request.

        Routed to the worker thread (which owns the communicator) so the
        producing worker gets its TENSOR_RECEIVED ack and frees the buffers.
        """
        self.discard_tensor_queue.put(input)

    def has_pending_tensors(self, request_id: str):
        return self.per_request_reading_tensors.get(request_id, 0) > 0

    def received_final_chunks(
        self, request_id: str,
        final_outputs: dict[str, NestedLoopIndices],
    ):
        # Every serving walk reports at least one client-facing output, so an
        # empty dict means the walk emitted nothing (or completion raced ahead
        # of every result). Report not-done and let the TTL backstop close the
        # request rather than completing it instantly with zero chunks.
        if not final_outputs:
            return False
        return all(
            not loop_iters.label_context_gt( # recv'd loop iters is not less than the final_fwd
                self.output_loop_idxs[request_id].get(name, None)
            ) for name, loop_iters in final_outputs.items()
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

    def cleanup_request(self, request_id: str):
        self.cleanup_request_queue.put(request_id)
        self.output_loop_idxs.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def shutdown(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


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


class PreprocessWorkerThread:
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
        enable_nvtx: bool=False,
        model_config: dict | None = None,
    ):
        # keying a stream needs the deployment's page size as well as the
        # model's declaration, so a worker built without a config keys no stream
        self._prefix_streams = (
            model.prefix_key_streams()
            if model is not None and model_config else {}
        )
        # resolved as the worker does at load, so both split a prompt into the same pages
        self._prefix_page_sizes = (
            _kv_page_sizes(model, model_config) if self._prefix_streams else {}
        )
        self.in_queue = in_queue
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

        self.stop_event = stop_event
        self.device = device
        self.model = model
        self.enable_prof = enable_prof
        # This thread turns each output tensor into client-ready bytes. At 720p
        # that is an 11 MiB SHM read plus a postprocess copy per engine step,
        # downstream of the last worker-side NVTX range.
        self.enable_nvtx = enable_nvtx

        self.in_flight_requests = set()
        self.tensor_uuid_to_metadata_per_request = {}
        # The request's model_kwargs, kept so output postprocessing can
        # honor per-request parameters (e.g. the video container fps).
        self.request_model_kwargs: dict[str, dict] = {}
        self.request_output_state: dict[str, RequestOutputState] = {}

        # Owned by PreprocessWorker (main thread); used only from this thread.
        self.communicator = communicator
        self.tensor_manager = tensor_manager

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
        ):
            state = getattr(self, state_name, None)
            if state is not None:
                state.pop(request_id, None)

    def _process_input(
        self, input: PreprocessInput
    ):
        tensors: NameToTensorList = {}
        input_metadata = {}
        self.in_flight_requests.add(input.request_id)

        # First, load raw modality tensors from file_paths (images, audio, video)
        # so they can be passed to process_prompt() below.
        if input.file_paths is not None:
            for modality in input.file_paths:
                key = f"{modality}_inputs"
                tensors[key] = []
                # TODO: maybe make a class of tensors_and_metadata later (figure out how to use metadata)
                input_metadata[key] = []

                for filepath in input.file_paths[modality]:
                    # ---- Image ----
                    if modality == "image":
                        out = self.model.load_image(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)

                    # ---- Audio ----
                    elif modality == "audio":
                        out = self.model.load_audio(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)

                    # ---- Video ----
                    elif modality == "video":
                        out = self.model.load_video(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)


        # Then, tokenize the prompt and let the model augment/transform the
        # tensors dict (e.g., Qwen3-Omni needs to compute pixel_values,
        # image_grid_thw, audio_features, audio_seqlens from the raw tensors
        # loaded above).  process_prompt receives the raw multimodal tensors
        # and returns any additional tensors to merge into the final dict.
        model_kwargs = dict(input.model_kwargs or {})
        # only this worker keys a prompt: a client that sent its own could name
        # another request's pages and be served that request's KV
        for name in ("prefix_keys", "prefix_tail", "prefix_decode"):
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
            # after the update: the chain keys the tensors the request will
            # actually be prefilled with
            prefix_keys, prefix_tail, prefix_decode = self._prefix_keys(tensors)
            if prefix_keys:
                model_kwargs["prefix_keys"] = prefix_keys
                model_kwargs["prefix_tail"] = prefix_tail
                if prefix_decode:
                    model_kwargs["prefix_decode"] = prefix_decode
        elif input.text is not None:
            # Fallback: encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_inputs"] = [torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )]

        initial_signals = self.tensor_manager.store_and_return_tensor_info(
            request_id=input.request_id,
            tensors=tensors # dict(modality_input: list[tensors])
        )
        all_infos = sum(
            [infos for infos in initial_signals.values()], start=[]
        )
        self.tensor_manager.register_for_send(
            request_id=input.request_id,
            tensor_infos=all_infos,
        )
        # also persist all of the input signals
        for info in all_infos:
            self.tensor_manager.set_persist(
                input.request_id, info.uuid, persist=True
            )

        self.request_model_kwargs[input.request_id] = model_kwargs
        self.request_output_state[input.request_id] = RequestOutputState()
        msg = ConductorMessage(
            message_type=ConductorMessageType.NEW_REQUEST,
            body=NewRequestConductor(
                request_id=input.request_id,
                initial_signals=initial_signals,
                initial_input_modalities=input.input_modalities,
                initial_output_modalities=input.output_modalities,
                input_metadata=input_metadata,
                model_kwargs=model_kwargs
            ),
        )
        self.communicator.send("conductor", msg)

        # Record preprocess-side profiling: the moment the fully preprocessed
        # request was handed off to the conductor, plus the per-modality sizes of
        # the *raw* inputs. ``perf_counter`` is consistent here because the worker
        # runs as a thread inside the API server process. (tx/rx are snapshotted
        # directly by the main thread at request completion — see APIServer.)
        if self.enable_prof:
            self.profile_queue.put(DataWorkerProfile(
                request_id=input.request_id,
                preprocess_finish_time=time.perf_counter(),
                inputs=self._summarize_inputs(input),
            ))

    def _prefix_keys(self, tensors: dict) -> tuple[dict, dict, dict]:
        """Key each page of every declared stream, by resource and label.

        Returns the keys, the prompt tail past the last whole page, and the output
        tensor each stream's sampled ids arrive in. The keys are unrooted.
        """
        keys: dict[str, dict[str, list[bytes]]] = {}
        tails: dict[str, dict[str, list[int]]] = {}
        decode: dict[str, dict[str, str]] = {}
        for resource_key, by_label in self._prefix_streams.items():
            page_size = self._prefix_page_sizes[resource_key]
            for label, stream in by_label.items():
                ids = tensors.get(stream.tensor)
                if stream.keyed_by != "ids" or not ids:
                    continue
                flat = ids[0].flatten().tolist()
                pages = [
                    flat[at:at + page_size]
                    for at in range(0, len(flat), page_size)
                ]
                keys.setdefault(resource_key, {})[label] = chain(pages)
                whole = len(flat) // page_size
                tails.setdefault(resource_key, {})[label] = flat[whole * page_size:]
                if stream.decode_walk is not None:
                    decode.setdefault(resource_key, {})[label] = stream.tensor
        return keys, tails, decode

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
        sequence: int | None = None,
    ):
        """Report a per-request data-worker failure to the API server.

        ``count`` error chunks are emitted because the API server's
        ``per_request_reading_tensors`` accounting is one decrement per chunk:
        a failure that kills N queued tensors has to answer for all N, or the
        request looks like it still has reads outstanding.

        ``sequence`` is the output slot the failed tensor held. Its error chunk
        takes that slot in the reorder buffer; put straight on ``out_queue`` it
        would leave every later sequence held in ``pending`` until the TTL.
        """
        logger.exception("%s failed for request %s", stage, request_id)
        status = 400 if isinstance(exc, (ValueError, TypeError)) else 500
        chunks = [
            ResultChunk(
                request_id=request_id,
                modality="error",
                data=f"{stage} failed: {type(exc).__name__}: {exc}".encode("utf-8"),
                metadata={"status": status},
            )
            for _ in range(max(count, 1))
        ]
        if sequence is not None and self._sequence_unanswered(request_id, sequence):
            self._queue_completed_output(request_id, sequence, chunks.pop())
        for chunk in chunks:
            self.out_queue.put(chunk)

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
        self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )
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

    def _process_read_tensors(self):
        did_work = False
        for request_id, graph_edges in self.tensor_manager.get_ready_tensors().items():
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
                        tensor = self.tensor_manager.get_tensor(
                            request_id=request_id,
                            uuid=tensor_info.uuid
                        )
                        if self.enable_nvtx:
                            profiler.range_pop()
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

                        chunk_metadata = self.tensor_uuid_to_metadata_per_request[request_id][
                            tensor_info.uuid] or {}
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
                    self.tensor_manager.dereference(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
        return did_work

    def _process_messages(self):
        did_work = False
        for message in self.communicator.get_all_new_messages():
            did_work = True
            if message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                body: TensorReceived = message.body
                for (uuid, ref_cnt) in body.successful_tensors.items():
                    self.tensor_manager.dereference(
                        body.request_id, uuid, n=ref_cnt
                    )
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                body: UnpersistTensors = message.body
                for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
                    self.tensor_manager.increment_ref(
                        body.request_id, uuid, n=ref_cnt
                    )
                    self.tensor_manager.set_persist(
                        body.request_id, uuid, persist=False
                    )
            elif message.message_type == WorkerMessageType.DRAIN_REQUEST:
                body: DrainRequest = message.body
                self._begin_drain(body.request_id)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                body: RemoveRequest = message.body
                self._hard_cleanup(body.request_id)
        return did_work

    def _finish_reading(self, request_id: str, drained: bool) -> None:
        """The API server is done with this rid's outputs. ``drained`` (every
        chunk delivered) means no read can still be in flight, so ACK straight
        away — the happy path pays nothing. Otherwise delivery was abandoned
        (TTL, client gone) with reads possibly still running, so gate the ACK on
        them; ACKing there would let the conductor unlink under a read."""
        if drained:
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
        self._draining_rids.discard(request_id)
        self._reads_done_sent.discard(request_id)
        self.tensor_manager.force_cleanup_request(request_id)
        self._drop_request_state(request_id)

    def run(self):
        while not self.stop_event.is_set():
            did_work = False
            try:
                did_work = self._process_messages()
                # Output delivery is latency-sensitive: the API server holds a
                # finished request open for its final chunks only briefly, so
                # queued read-starts / acks / cleanups must never wait behind a
                # multi-second media preprocess. Drain them fully every pass
                # and take at most one preprocess item afterwards.
                while not self.result_tensor_queue.empty():
                    did_work = True
                    result = self.result_tensor_queue.get()
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
                while not self.abort_request_queue.empty():
                    did_work = True
                    rid = self.abort_request_queue.get()
                    self.communicator.send(
                        "conductor",
                        ConductorMessage(
                            message_type=ConductorMessageType.ABORT_REQUEST,
                            body=AbortRequest(request_id=rid),
                        ),
                    )
                    # Begin draining our own reads immediately, overlapping with
                    # the conductor round-trip (it also sends a DrainRequest).
                    self._begin_drain(rid)
                while not self.reads_done_queue.empty():
                    did_work = True
                    self._finish_reading(*self.reads_done_queue.get())
                while not self.discard_tensor_queue.empty():
                    did_work = True
                    self._discard_result_tensor(self.discard_tensor_queue.get())
                while not self.cleanup_request_queue.empty():
                    did_work = True
                    req_id = self.cleanup_request_queue.get()
                    self._cleanup_request_state(req_id)
                did_work = self._process_read_tensors() or did_work
                # Reads may have just resolved; ACK any drains now free of them.
                for rid in list(self._draining_rids):
                    self._complete_drain_if_ready(rid)
                if not self.in_queue.empty():
                    did_work = True
                    pre_input = self.in_queue.get()
                    try:
                        self._process_input(pre_input)
                    except Exception as exc:  # noqa: BLE001 — any failure must reach the client
                        # A request whose media load or prompt processing fails
                        # never reaches the conductor, so nothing downstream
                        # would ever complete it; surface the failure as an
                        # error chunk instead of leaving the client to hit the
                        # server timeout.
                        self._fail_request(
                            pre_input.request_id, exc, "preprocessing",
                        )
                        # Never reached the conductor, so there are no remote
                        # readers to race: hard-drop the (possibly persisted)
                        # input signals directly.
                        self._cleanup_request_state(pre_input.request_id, force=True)
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            if not did_work:
                time.sleep(0.001)

        # Stopping, and nothing will send the RemoveRequest for what is still
        # tracked (the conductor is gone or going), so drop it here rather than
        # leave the input signals of in-flight requests in /dev/shm.
        for request_id in list(self.in_flight_requests):
            self._hard_cleanup(request_id)
