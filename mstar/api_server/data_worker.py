

import collections
import logging
import os
import queue
import threading
import time

import torch

from mstar.graph.loop_indices import NestedLoopIndices

try:
    import torchaudio  # noqa: F401 — probes availability; real usage in callers
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError, OSError):
    VideoDecoder = None

from mstar.api_server.detok_proc import DetokClient
from mstar.api_server.preproc_proc import PreprocClient, preprocess_tensors
from mstar.api_server.request_types import (
    DataWorkerProfile,
    PendingDetok,
    PreprocessInput,
    ResultChunk,
    ResultTensors,
)
from mstar.communication.communicator import BaseCommunicator, CommProtocol, make_communicator
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.model.base import Model
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.utils.ipc_format import (
    AbortRequest,
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    TensorReceived,
    UnpersistTensors,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


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
        model_name: str = "dummy",
        cache_dir: str | None = None,
        model_kwargs: dict | None = None,
        log_level: str = "INFO",
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.abort_request_queue = queue.Queue()
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

        # MSTAR_DETOK_PROC (#13, default OFF): move the CPU-heavy per-chunk text
        # detokenization (``model.postprocess``) off this (uvicorn/serve) process
        # into a dedicated child, so it stops contending for the serve GIL. Read
        # ONCE here: the child is spawned now and cannot follow MSTAR_DYNFLAGS
        # (process topology is boot-time; see detok_proc.py). A/B via two boots.
        # Requires a real model with a tokenizer (postprocess); skipped for the
        # dummy/None model so tests and dummy configs are untouched.
        self.detok_client: DetokClient | None = None
        detok_on = os.environ.get("MSTAR_DETOK_PROC", "0") == "1"
        if detok_on and model is not None:
            self.detok_client = DetokClient(
                model=model,
                model_name=model_name,
                cache_dir=cache_dir,
                model_kwargs=model_kwargs,
                out_queue=self.output_queue,
                log_level=log_level,
            )
        elif detok_on:
            logger.warning(
                "MSTAR_DETOK_PROC=1 but no model with a tokenizer is available; "
                "keeping detok inline",
            )

        # MSTAR_PREPROC_PROC (default OFF): move the CPU-heavy request-side
        # multimodal preprocessing (image decode/resize/patchify via load_image +
        # process_prompt) off this (uvicorn/serve) process into a pool of child
        # processes, so a burst of image requests stops spiking the serve process
        # and starving the uvicorn/ZMQ-drain threads of the GIL. Read ONCE here
        # (process topology is boot-time; see preproc_proc.py). Requires a real
        # model with process_prompt; skipped for the dummy/None model. Composes
        # with MSTAR_DETOK_PROC (both children coexist). A/B via two boots.
        self.preproc_client: PreprocClient | None = None
        preproc_on = os.environ.get("MSTAR_PREPROC_PROC", "0") == "1"
        if preproc_on and model is not None:
            num_procs = int(os.environ.get("MSTAR_PREPROC_PROCS", "4"))
            # Per-child intra-op thread cap. Default: split the box across the
            # pool so N children * threads ~= cores (0 in the env => auto here;
            # a positive MSTAR_PREPROC_THREADS overrides). The preprocess is a
            # torch/torchvision multi-thread burst, so an uncapped pool would
            # oversubscribe and inflate latency.
            num_threads = int(os.environ.get("MSTAR_PREPROC_THREADS", "0"))
            if num_threads <= 0:
                num_threads = max(1, (os.cpu_count() or num_procs) // max(1, num_procs))
            self.preproc_client = PreprocClient(
                model_name=model_name,
                cache_dir=cache_dir,
                model_kwargs=model_kwargs,
                num_procs=num_procs,
                num_threads=num_threads,
                log_level=log_level,
            )
        elif preproc_on:
            logger.warning(
                "MSTAR_PREPROC_PROC=1 but no model with process_prompt is "
                "available; keeping preprocessing inline",
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
                discard_tensor_queue=self.discard_tensor_queue,
                stop_event=self.stop_event,
                communicator=self.communicator,
                tensor_manager=self.tensor_manager,
                model=model,
                enable_prof=enable_prof,
                detok_client=self.detok_client,
                preproc_client=self.preproc_client,
            )
        )
        self.thread.start()

    def new_request(self, input: PreprocessInput):
        self.output_loop_idxs[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)

    def abort_request(self, request_id: str):
        self.abort_request_queue.put(request_id)
        self.cleanup_request(request_id)

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
            # Tolerant decrement: a LATE chunk can land after cleanup_request
            # popped this rid's counter (in-flight SHM read completing in the
            # main-thread-pop -> worker-thread-cleanup window; ordered emit
            # widens it by holding+late-flushing chunks). A bare subscript
            # here raised KeyError and aborted the whole drain loop, dropping
            # every other request's queued chunks with it. (upstream #181 fixed
            # the same straggler race; this keeps the cnt binding the debug log
            # below reads.)
            cnt = self.per_request_reading_tensors.get(result.request_id)
            if cnt is None:
                logger.debug(
                    "Dropping late chunk for cleaned-up request %s",
                    result.request_id,
                )
                continue
            self.per_request_reading_tensors[result.request_id] = cnt - 1
            logger.debug(
                "Data worker reading queue for request %s decreased to length %d",
                result.request_id, cnt - 1,
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
        # After the worker thread has stopped submitting, tear down the detok
        # child; any still-outstanding chunks are recovered inline in shutdown().
        if self.detok_client is not None:
            self.detok_client.shutdown()
        # Tear down the preproc pool. Outstanding preproc jobs are simply
        # dropped at shutdown (the server is going down; those requests are
        # aborted/timed out by the normal shutdown path — the transport state
        # they would need is being torn down too).
        if self.preproc_client is not None:
            self.preproc_client.shutdown()


class PreprocessWorkerThread:
    def __init__(
        self,
        in_queue: queue.Queue, # for preprocessing
        result_tensor_queue: queue.Queue, # for output streaming
        out_queue: queue.Queue,
        profile_queue: queue.Queue,
        cleanup_request_queue: queue.Queue,
        abort_request_queue: queue.Queue,
        discard_tensor_queue: queue.Queue,
        stop_event: threading.Event,
        communicator: BaseCommunicator,
        tensor_manager,
        device: str = "cpu",
        model: Model | None = None,
        enable_prof: bool=False,
        detok_client: DetokClient | None = None,
        preproc_client: PreprocClient | None = None,
    ):
        self.in_queue = in_queue
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.abort_request_queue = abort_request_queue
        self.discard_tensor_queue = discard_tensor_queue
        self.out_queue = out_queue
        self.profile_queue = profile_queue

        self.stop_event = stop_event
        self.device = device
        self.model = model
        self.enable_prof = enable_prof

        # MSTAR_DETOK_PROC: when set, text chunks are built WITHOUT running
        # postprocess inline (a PendingDetok is attached instead) and emitted via
        # _emit_chunk, which hands them to this child. None => flag off / no
        # tokenizer model => every path stays exactly as before.
        self.detok_client = detok_client
        self._detok_enabled = detok_client is not None

        # MSTAR_PREPROC_PROC: when set, request-side multimodal preprocessing
        # (load_* + process_prompt) runs in the child pool and its produced
        # tensors come back here for admission. None => flag off / no model =>
        # every request preprocesses inline exactly as before.
        self.preproc_client = preproc_client
        self._preproc_enabled = preproc_client is not None
        # An audio-input request is NOT offloaded while the serve process would
        # compute the log-mel on the GPU (MSTAR_GPU_MEL default on + CUDA
        # present): a CUDA-hidden child would produce a CPU mel that is not
        # bit-identical, so such requests stay inline (byte-identical). i2t
        # (image+text) is unaffected — its preprocessing is CPU-deterministic.
        self._serve_uses_gpu_mel = (
            os.environ.get("MSTAR_GPU_MEL", "1") in ("1", "true", "True")
            and torch.cuda.is_available()
        )

        self.tensor_uuid_to_metadata_per_request = {}
        # The request's model_kwargs, kept so output postprocessing can
        # honor per-request parameters (e.g. the video container fps).
        self.request_model_kwargs: dict[str, dict] = {}

        # Owned by PreprocessWorker (main thread); used only from this thread.
        self.communicator = communicator
        self.tensor_manager = tensor_manager

        # MSTAR_ORDERED_EMIT: emit ResultChunks per (rid, modality) in ARRIVAL
        # order rather than read-completion order. Without this, a mixed
        # inline/SHM stream reorders: inline items (decode new-token ints)
        # emit synchronously in _read_result_tensor while an SHM item (the
        # prefill step's first token — its uuid also feeds the prefill→decode
        # loop-back edge, so it is excluded from the inline transport and must
        # be fetched) emits only when its async read completes. Under load the
        # fetch lands 1..k decode tokens late, so the client stream shows the
        # FIRST generated token displaced mid-sentence (or, when the request
        # finishes first, missing entirely). Default OFF = current behavior.
        self._ordered_emit = os.environ.get("MSTAR_ORDERED_EMIT", "0") == "1"
        self._ordered_emit_debug = (
            os.environ.get("MSTAR_ORDERED_EMIT_DEBUG", "0") == "1"
        )
        # (rid, modality) -> deque of entries in arrival order. Entry:
        # {"ready": bool, "uuid_order": [uuid,...], "chunks": {uuid: chunk},
        #  "pending": set[uuid]}   (inline entries: ready=True, uuid_order
        # ordered as built, pending empty).
        self._emit_fifos: dict[tuple[str, str], collections.deque] = {}
        # (rid, uuid) -> [(fifo_key, entry), ...] so read completions find every
        # entry waiting on the uuid (aliased/re-sent uuids: one read must satisfy
        # all waiters).
        self._uuid_to_emit_entry: dict[tuple[str, str], list] = {}

    def _should_offload(self, input: PreprocessInput) -> bool:
        """Whether this request's preprocessing can go to the child pool.

        Off unless the flag is on and the pool is healthy. Audio-input requests
        stay inline while the serve process uses GPU mel (byte-identity — see
        __init__). Everything else (i2t: image+text) is offloaded."""
        if not self._preproc_enabled:
            return False
        if self._serve_uses_gpu_mel and "audio" in (input.input_modalities or []):
            return False
        return True

    def _on_new_input(self, input: PreprocessInput):
        """Dispatch a new request: to the pool when eligible, else inline."""
        if self._should_offload(input) and self.preproc_client.submit(input):
            return
        # Flag off, not offloadable, or the pool has permanently failed.
        self._process_input(input)

    def _complete_preproc(self, comp: tuple):
        """Admit a request whose preprocessing finished (pool or inline recovery)."""
        kind = comp[0]
        if kind == "ok":
            _, input, tensors, input_metadata = comp
            self._admit_request(input, tensors, input_metadata)
        else:  # "inline" — child error or fallback recovery: run the full path
            self._process_input(comp[1])

    def _process_input(
        self, input: PreprocessInput
    ):
        # Inline path: load raw modality tensors + process_prompt, then admit.
        # The load/process_prompt work is the exact same implementation the
        # preproc child runs (shared preprocess_tensors — no drift, so an
        # offloaded request produces byte-identical tensors).
        tensors, input_metadata = preprocess_tensors(
            self.model, input, self.device
        )
        self._admit_request(input, tensors, input_metadata)

    def _admit_request(
        self,
        input: PreprocessInput,
        tensors: NameToTensorList,
        input_metadata: dict,
    ):
        # Transport + conductor handoff. ALWAYS runs on this worker thread (it
        # owns the tensor_manager and communicator), whether the tensors were
        # produced inline or in the child pool.
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

        self.request_model_kwargs[input.request_id] = input.model_kwargs or {}
        msg = ConductorMessage(
            message_type=ConductorMessageType.NEW_REQUEST,
            body=NewRequestConductor(
                request_id=input.request_id,
                initial_signals=initial_signals,
                initial_input_modalities=input.input_modalities,
                initial_output_modalities=input.output_modalities,
                input_metadata=input_metadata,
                model_kwargs=input.model_kwargs
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

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        result.graph_edge.name = f"{result.modality}_output"
        # Inline fast path: token values arrived in the message metadata, so
        # there is no SHM tensor to fetch and no producer ack to send. The
        # producer already released its tensor_store ref locally.
        if result.metadata and "inline_values" in result.metadata:
            if self._ordered_emit:
                # Enqueue at the FIFO tail; emits only once every earlier
                # arrival for this (rid, modality) has emitted.
                key = (result.request_id, result.modality)
                chunks = self._build_inline_chunks(result)
                entry = {
                    "ready": True,
                    "uuid_order": list(range(len(chunks))),
                    "chunks": dict(enumerate(chunks)),
                    "pending": set(),
                }
                self._emit_fifos.setdefault(key, collections.deque()).append(entry)
                if self._ordered_emit_debug:
                    logger.warning(
                        "ORDEMIT arrival INLINE rid=%s n_chunks=%d fifo_len=%d",
                        result.request_id, len(chunks),
                        len(self._emit_fifos[key]),
                    )
                self._flush_emit_fifo(key)
            else:
                self._emit_inline_result(result)
            return
        self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )
        if result.request_id not in self.tensor_uuid_to_metadata_per_request:
            self.tensor_uuid_to_metadata_per_request[result.request_id] = {}
        for tensor_info in result.graph_edge.tensor_info:
            self.tensor_uuid_to_metadata_per_request[result.request_id][
                tensor_info.uuid] = result.metadata
        if self._ordered_emit:
            key = (result.request_id, result.modality)
            uuids = [info.uuid for info in result.graph_edge.tensor_info]
            entry = {
                # A signal-only emit edge (no tensor_info — e.g. the leading
                # text_output marker) transports nothing: it is trivially
                # ready, else it wedges the FIFO head forever (no read will
                # ever complete it) and every later chunk is held until the
                # request's TTL drops them (observed: all-empty responses).
                "ready": not uuids,
                "uuid_order": uuids,
                "chunks": {},
                "pending": set(uuids),
            }
            self._emit_fifos.setdefault(key, collections.deque()).append(entry)
            for u in uuids:
                waiters = self._uuid_to_emit_entry.setdefault(
                    (result.request_id, u), []
                )
                if waiters and self._ordered_emit_debug:
                    logger.warning(
                        "ORDEMIT alias (multi-waiter) rid=%s uuid=%s n=%d",
                        result.request_id, u, len(waiters) + 1,
                    )
                # List-valued: the SAME uuid can be referenced by MULTIPLE
                # arrival entries (aliased emit edges / re-sends); one read
                # completion must satisfy every waiter or the orphaned
                # earlier entry wedges the FIFO head forever.
                waiters.append((key, entry))
            if self._ordered_emit_debug:
                logger.warning(
                    "ORDEMIT arrival SHM rid=%s uuids=%s fifo_len=%d",
                    result.request_id, uuids, len(self._emit_fifos[key]),
                )
            if entry["ready"]:
                # Signal-only entry: pop it (and any ready run) promptly so
                # it never lingers at the head.
                self._flush_emit_fifo(key)

    def _emit_chunk(self, chunk: ResultChunk):
        """Single emit choke point for every ResultChunk.

        Flag off (or a chunk with no deferred detok — e.g. audio/image): put it
        straight on the output queue, exactly as before. Flag on with a deferred
        text chunk: hand it to the detok child (it fills ``data`` off-process and
        the child's receiver thread does the ``out_queue.put`` in submit order).
        If the child is unavailable, postprocess inline right here so nothing
        hangs — identical bytes, just on this process.
        """
        dc = self.detok_client
        if dc is not None and chunk.pending_detok is not None:
            if dc.submit(chunk):
                return
            pd = chunk.pending_detok
            chunk.data = self.model.postprocess(
                torch.tensor(pd.ints, dtype=pd.dtype).reshape(pd.dims),
                chunk.modality,
            )
            chunk.pending_detok = None
        self.out_queue.put(chunk)

    def _emit_inline_result(self, result: ResultTensors):
        """Produce ResultChunk(s) directly from inline token values.

        Mirrors _process_read_tensors' emission but skips the transport
        fetch: one chunk per tensor_info entry (so per_request_reading_tensors,
        bumped by len(tensor_info) in new_result_tensors, balances exactly),
        each reconstructed as a byte-identical tensor from the inline ints
        using the tensor_info dtype/shape and run through the same postprocess.
        """
        for chunk in self._build_inline_chunks(result):
            self._emit_chunk(chunk)

    def _build_inline_chunks(self, result: ResultTensors) -> list[ResultChunk]:
        """Construct the ResultChunk list for an inline-values message
        (shared by the immediate path and MSTAR_ORDERED_EMIT's FIFO path)."""
        modality = result.graph_edge.name.replace("_output", "")
        # The producer keys inline_values by the pre-rename edge name; there is
        # exactly one entry (this edge). Fall back to the single value list.
        inline_map: dict = result.metadata["inline_values"]
        values = next(iter(inline_map.values())) if inline_map else []
        chunk_metadata = {
            k: v for k, v in (result.metadata or {}).items()
            if k != "inline_values"
        }
        # Keep parity with _process_read_tensors' audio enrichment: an audio
        # item emitted inline must carry sample_rate too, or clients fall
        # back to a hardcoded rate and mis-wrap the PCM. (Today only integer
        # text tokens ride inline, but the two chunk-assembly paths must not
        # diverge on this field.)
        if modality == "audio" and self.model is not None:
            chunk_metadata = {
                **chunk_metadata,
                "sample_rate": self.model.get_output_sample_rate("audio"),
            }
        chunks: list[ResultChunk] = []
        for tensor_info in result.graph_edge.tensor_info:
            n = 1
            for d in tensor_info.dims:
                n *= int(d)
            ints = values[:n]
            values = values[n:]
            if self._detok_enabled and modality == "text":
                # Defer detok: carry exactly what postprocess needs (ints, dtype,
                # dims). The detok child reconstructs the identical tensor, so the
                # bytes match the inline branch below. Only text is offloaded.
                chunks.append(ResultChunk(
                    request_id=result.request_id,
                    modality=modality,
                    data=b"",
                    metadata=chunk_metadata,
                    pending_detok=PendingDetok(
                        ints=list(ints),
                        dtype=tensor_info.dtype,
                        dims=tuple(tensor_info.dims),
                    ),
                ))
            else:
                tensor = torch.tensor(ints, dtype=tensor_info.dtype).reshape(
                    tensor_info.dims
                )
                postprocessed = self.model.postprocess(tensor, modality)
                chunks.append(ResultChunk(
                    request_id=result.request_id,
                    modality=modality,
                    data=postprocessed,
                    metadata=chunk_metadata,
                ))
        return chunks

    def _flush_emit_fifo(self, key: tuple[str, str]) -> None:
        """Emit the head-run of ready entries for one (rid, modality) FIFO.

        Arrival order == the producing worker's send order (single ZMQ FIFO
        per rid), so draining ready heads preserves true token order; an
        unread SHM entry at the head holds everything behind it until its
        read lands (at most the transport latency — the same latency that
        today reorders instead).
        """
        fifo = self._emit_fifos.get(key)
        if fifo is None:
            return
        n_emitted = 0
        while fifo and fifo[0]["ready"]:
            entry = fifo.popleft()
            for u in entry["uuid_order"]:
                chunk = entry["chunks"].get(u)
                if chunk is not None:
                    self._emit_chunk(chunk)
                    n_emitted += 1
        if self._ordered_emit_debug:
            logger.warning(
                "ORDEMIT flush key=%s emitted=%d held=%d head_pending=%s",
                key, n_emitted, len(fifo),
                (sorted(fifo[0]["pending"]) if fifo else None),
            )
        if not fifo:
            self._emit_fifos.pop(key, None)

    def _discard_result_tensor(
        self, result: ResultTensors
    ):
        # Inline messages carry no transported tensors: the producer never
        # registered them for send and already released its ref locally, so
        # there is nothing to ack. Acking would deref uuids the producer
        # doesn't hold — make discard a no-op for inline-only messages.
        if result.metadata and "inline_values" in result.metadata:
            return
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
                    tensor = self.tensor_manager.get_tensor(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
                    # Tolerant metadata lookup: a duplicate completion for an
                    # aliased/re-sent uuid (or a completion racing cleanup)
                    # finds the key already deleted below — a bare double
                    # subscript raised KeyError and killed the whole
                    # ready-tensor sweep for every other request. (We keep the
                    # postprocess call inline at the ResultChunk below rather
                    # than upstream's precompute, so the MSTAR detok-defer path
                    # for text can skip postprocess entirely; upstream's
                    # request_kwargs is threaded into that inline call.)
                    chunk_metadata = (
                        self.tensor_uuid_to_metadata_per_request
                        .get(request_id, {})
                        .get(tensor_info.uuid)
                    ) or {}
                    # Audio is emitted as headerless 16-bit PCM; surface the
                    # model's output sample rate + channel count so clients can
                    # wrap it.
                    if modality == "audio" and self.model is not None:
                        chunk_metadata = {
                            **chunk_metadata,
                            "sample_rate": self.model.get_output_sample_rate("audio"),
                            "num_channels": self.model.get_output_audio_channels("audio"),
                        }

                    if self._detok_enabled and modality == "text":
                        # Defer detok (SHM text path — e.g. the prefill first
                        # token, excluded from the inline transport). Ship the
                        # flat ints + dtype + dims; the child rebuilds the same
                        # tensor. Non-text (audio/image) stays inline.
                        chunk = ResultChunk(
                            request_id=request_id,
                            modality=modality,
                            data=b"",
                            metadata=chunk_metadata,
                            pending_detok=PendingDetok(
                                ints=tensor.flatten().tolist(),
                                dtype=tensor.dtype,
                                dims=tuple(tensor.shape),
                            ),
                        )
                    else:
                        chunk = ResultChunk(
                            request_id=request_id,
                            modality=modality,
                            data=self.model.postprocess(
                                tensor, modality,
                                request_kwargs=self.request_model_kwargs.get(request_id),
                            ),
                            metadata=chunk_metadata,
                        )
                    waiters = (
                        self._uuid_to_emit_entry.pop(
                            (request_id, tensor_info.uuid), None,
                        )
                        if self._ordered_emit else None
                    )
                    if self._ordered_emit and self._ordered_emit_debug:
                        logger.warning(
                            "ORDEMIT completion rid=%s uuid=%s waiters=%s",
                            request_id, tensor_info.uuid,
                            len(waiters) if waiters else 0,
                        )
                    if waiters:
                        # MSTAR_ORDERED_EMIT: attach to every arrival-ordered
                        # entry waiting on this uuid; flush each FIFO whose
                        # entry became ready at the head.
                        for key, entry in waiters:
                            entry["chunks"][tensor_info.uuid] = chunk
                            entry["pending"].discard(tensor_info.uuid)
                            if not entry["pending"]:
                                entry["ready"] = True
                                self._flush_emit_fifo(key)
                    elif not self._ordered_emit:
                        # Flag off: emit directly, as before ordered emit.
                        self._emit_chunk(chunk)
                    else:
                        # Ordered emit ON but no waiters: every SHM arrival
                        # registers waiters under the flag, so this is either
                        # a DUPLICATE completion for an aliased/re-sent uuid
                        # (the first completion already emitted for every
                        # waiter — emitting again would deliver a duplicate
                        # token) or an entry dropped by request cleanup (rid
                        # dead). Drop, don't emit.
                        logger.debug(
                            "ORDEMIT dropping waiterless completion rid=%s "
                            "uuid=%s", request_id, tensor_info.uuid,
                        )
                    self.tensor_uuid_to_metadata_per_request.get(
                        request_id, {}
                    ).pop(tensor_info.uuid, None)
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
        return did_work

    def run(self):
        while not self.stop_event.is_set():
            did_work = False
            try:
                did_work = self._process_messages()
                # Output delivery is latency-sensitive: the API server holds a
                # finished request open for its final chunks only briefly, so
                # queued read-starts / acks / cleanups must never wait behind a
                # multi-second media preprocess. Drain them fully every pass
                # (upstream #181 drain-ahead) and admit input afterwards.
                while not self.result_tensor_queue.empty():
                    did_work = True
                    self._read_result_tensor(self.result_tensor_queue.get())
                while not self.abort_request_queue.empty():
                    did_work = True
                    abort_rid = self.abort_request_queue.get()
                    # MSTAR_PREPROC_PROC: if this request is still being
                    # preprocessed in the pool (not yet admitted), cancel it so
                    # get_ready drops its completion instead of sending a
                    # NEW_REQUEST *after* this ABORT (which would leave it
                    # admitted-but-un-aborted). No-op when the flag is off.
                    if self.preproc_client is not None:
                        self.preproc_client.cancel_rid(abort_rid)
                    self.communicator.send(
                        "conductor",
                        ConductorMessage(
                            message_type=ConductorMessageType.ABORT_REQUEST,
                            body=AbortRequest(request_id=abort_rid),
                        ),
                    )
                while not self.discard_tensor_queue.empty():
                    did_work = True
                    self._discard_result_tensor(self.discard_tensor_queue.get())
                while not self.cleanup_request_queue.empty():
                    did_work = True
                    req_id = self.cleanup_request_queue.get()
                    self.tensor_manager.cleanup_request(req_id)
                    if req_id in self.tensor_uuid_to_metadata_per_request:
                        del self.tensor_uuid_to_metadata_per_request[req_id]
                    # MSTAR_DETOK_PROC: free the child's per-rid state (complete
                    # or abort). Outstanding items still return normally and are
                    # dropped as late chunks if the rid is gone (tolerant path in
                    # get_result_chunks). No-op when the flag is off.
                    if self.detok_client is not None:
                        self.detok_client.drop_rid(req_id)
                    if self._ordered_emit:
                        # Drop this rid's held-back entries and uuid refs; a
                        # late read for a dropped entry falls back to the
                        # direct out_queue path above (harmless for dead rid).
                        fifo_keys = [
                            k for k in self._emit_fifos if k[0] == req_id
                        ]
                        uuid_keys = [
                            k for k in self._uuid_to_emit_entry
                            if k[0] == req_id
                        ]
                        for key in fifo_keys:
                            if self._ordered_emit_debug:
                                fifo = self._emit_fifos.get(key)
                                logger.warning(
                                    "ORDEMIT cleanup-drop key=%s held=%d "
                                    "head_pending=%s",
                                    key, len(fifo) if fifo else 0,
                                    (sorted(fifo[0]["pending"])
                                     if fifo else None),
                                )
                            self._emit_fifos.pop(key, None)
                        for uk in uuid_keys:
                            self._uuid_to_emit_entry.pop(uk, None)
                    self.request_model_kwargs.pop(req_id, None)
                # Always call _process_read_tensors (upstream order): a plain
                # `did_work or ...` would short-circuit the call once other work
                # already flipped did_work True.
                did_work = self._process_read_tensors() or did_work
                # Input admission LAST (upstream #181 drain-ahead): result
                # delivery above drains ahead of media preprocess so a slow
                # media load can't starve output. One input per pass.
                if not self.in_queue.empty():
                    did_work = True
                    pre_input = self.in_queue.get()
                    try:
                        # MSTAR_PREPROC_PROC: _on_new_input offloads to the pool
                        # when eligible, else processes inline. No-op offload
                        # when the flag is off (falls through to _process_input).
                        self._on_new_input(pre_input)
                    except Exception as exc:  # noqa: BLE001 — any failure must reach the client
                        # A request whose media load or prompt processing fails
                        # never reaches the conductor, so nothing downstream
                        # would ever complete it; surface the failure as an
                        # error chunk instead of leaving the client to hit the
                        # server timeout.
                        logger.exception(
                            "Preprocessing failed for request %s", pre_input.request_id
                        )
                        status = 400 if isinstance(exc, (ValueError, TypeError)) else 500
                        self.out_queue.put(ResultChunk(
                            request_id=pre_input.request_id,
                            modality="error",
                            data=str(exc).encode("utf-8"),
                            metadata={"status": status},
                        ))
                        self.tensor_manager.cleanup_request(pre_input.request_id)
                        self.request_model_kwargs.pop(pre_input.request_id, None)
                # MSTAR_PREPROC_PROC: admit any requests whose off-process
                # preprocessing finished (in submit order). No-op when the flag
                # is off. Kept in the loop so completions drain promptly; each
                # wrapped so a child-side failure fails its request (error chunk)
                # rather than hanging the client (upstream #181 intent).
                if self.preproc_client is not None:
                    for comp in self.preproc_client.get_ready():
                        did_work = True
                        try:
                            self._complete_preproc(comp)
                        except Exception as exc:  # noqa: BLE001
                            rid = comp[1].request_id
                            logger.exception(
                                "Preprocessing failed for request %s", rid
                            )
                            status = 400 if isinstance(exc, (ValueError, TypeError)) else 500
                            self.out_queue.put(ResultChunk(
                                request_id=rid,
                                modality="error",
                                data=str(exc).encode("utf-8"),
                                metadata={"status": status},
                            ))
                            self.tensor_manager.cleanup_request(rid)
                            self.request_model_kwargs.pop(rid, None)
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            if not did_work:
                time.sleep(0.001)

