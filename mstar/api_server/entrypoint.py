"""FastAPI server entry point for multimodal inference requests."""

import asyncio
import base64
import collections
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from mstar.api_server.data_worker import PreprocessWorker
from mstar.api_server.request_types import APIServerMessage, PreprocessInput, ResultChunk
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.model.multimodal import PromptPart
from mstar.model.registry import HF_MODELS
from mstar.profile.display import pretty_print_profile
from mstar.profile.format import OutputInfo, RequestProfile, RequestTiming
from mstar.utils import profiler
from mstar.utils.exitcode import describe_exitcode
from mstar.utils.logging_config import quiet_noisy_loggers
from mstar.utils.orphan import watch_parent

logger = logging.getLogger(__name__)

SUPPORTED_MODALITIES = frozenset({
    "text", "image", "audio", "video", "video_frame", "action", "scalar", "tensor",
})
STREAMING_ONLY_MODALITIES = frozenset({"video_frame"})

NDJSON_STREAM_MEDIA_TYPE = "application/x-ndjson"
# Opt-in framing for raw binary payloads, requested via ``Accept``. Duplicated
# rather than shared with ``mstar.client.media`` so the SDK keeps its stdlib-only
# import contract; ``test_binary_framing.py`` asserts the two stay equal.
BINARY_STREAM_MEDIA_TYPE = "application/vnd.mstar.frames"

# Extension-based modality detection for uploaded files.
_EXT_TO_MODALITY: dict[str, str] = {}
for _mod, _exts in {
    "image": (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"),
    "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"),
    "video": (".mp4", ".avi", ".mov", ".mkv", ".webm"),
}.items():
    for _ext in _exts:
        _EXT_TO_MODALITY[_ext] = _mod


def _detect_modality(filename: str) -> str:
    return _EXT_TO_MODALITY.get(Path(filename).suffix.lower(), "unknown")


# ------------------------------------------------------------------
# Conductor process target (top-level for picklability with spawn)
# ------------------------------------------------------------------

def _conductor_process_target(
    model_name: str,
    config_path: str,
    socket_path_prefix: str,
    enable_nvtx: bool = False,
    enable_prof: bool = False,
    log_level: str = "INFO",
    cache_dir: str | None = None,
    tensor_comm_protocol=CommProtocol.RDMA,
    tcp_transfer_device=""
):
    """Runs DummyConductor.run() in a spawned process."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s [conductor] %(name)s: %(message)s",
        force=True,
    )
    quiet_noisy_loggers()
    # A server started as a background job of a non-interactive shell inherits
    # SIGINT ignored, and Python keeps an inherited ignore, so the graceful stop
    # below (and _shutdown_conductor_process's) would be a no-op. Make it real.
    # default_int_handler rather than SIG_DFL because it raises
    # KeyboardInterrupt, which unwinds conductor.run() into the finally that
    # shuts the conductor down, where SIG_DFL would kill the process on the spot.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    # Started before the model load so an API server that dies during it is
    # still caught. SIGINT is the conductor's graceful stop (run() unwinds into
    # shutdown(), terminating the workers), matching _shutdown_conductor_process.
    watch_parent("Conductor", "API server", signal.SIGINT)
    # Read yaml early to extract optional `model_kwargs:` section for the model
    # constructor. Lets a yaml override init-time model parameters (e.g.
    # Pi05's action_horizon for the DROID benchmark variant) without code
    # changes per model. Backward compatible: missing section → empty dict,
    # so existing configs see identical behavior.
    import yaml as _yaml

    from mstar.conductor.conductor import Conductor
    from mstar.model.registry import get_model_class
    with open(config_path, "r") as _f:
        _yaml_cfg = _yaml.safe_load(_f) or {}
    yaml_model_kwargs = _yaml_cfg.get("model_kwargs", {}) or {}
    if yaml_model_kwargs:
        logging.getLogger(__name__).info(
            "yaml model_kwargs from %s: %s (forwarded to %s.__init__)",
            config_path, yaml_model_kwargs, model_name,
        )
    else:
        logging.getLogger(__name__).info(
            "yaml %s has no model_kwargs section; using model defaults", config_path
        )

    model = get_model_class(model_name)(
        model_path_hf=HF_MODELS.get(model_name, {}).get("model_path_hf", ""),
        cache_dir=cache_dir,
        **yaml_model_kwargs,
    )
    conductor = Conductor(
        model=model,
        model_config_file=config_path,
        socket_path_prefix=socket_path_prefix,
        enable_nvtx=enable_nvtx,
        enable_prof=enable_prof,
        log_level=log_level,
        tensor_comm_protocol=tensor_comm_protocol,
        tcp_transfer_device=tcp_transfer_device
    )
    try:
        conductor.run()
    except KeyboardInterrupt:
        # The API parent uses SIGINT for a graceful child shutdown. Treat that
        # as the normal stop signal after allowing the conductor to unwind.
        pass
    finally:
        conductor.shutdown()


def _shutdown_conductor_process(
    conductor_proc: mp.Process,
    timeout: float = 5.0,
) -> None:
    if not conductor_proc.is_alive():
        return

    try:
        os.kill(conductor_proc.pid, signal.SIGINT)
        conductor_proc.join(timeout=timeout)
    except BaseException:
        logger.exception("Failed graceful conductor shutdown")

    if conductor_proc.is_alive():
        conductor_proc.terminate()
        conductor_proc.join(timeout=timeout)

    if conductor_proc.is_alive():
        if os.name != "nt":
            conductor_proc.kill()
            conductor_proc.join(timeout=timeout)


# ------------------------------------------------------------------
# APIServer
# ------------------------------------------------------------------


@dataclass
class PendingRequest:
    streaming: bool
    input_modalities: list[str]
    output_modalities: list[str]
    profile: RequestProfile
    event: threading.Event = field(default_factory=threading.Event)
    chunks: list[ResultChunk] = field(default_factory=list)
    final_outputs: dict = field(default_factory=dict)
    consumed_chunks: int = 0
    error: Any | None = None
    error_status: int = 500


def _chunk_to_ndjson_payload(chunk: ResultChunk) -> str:
    """Serialize one result chunk as an NDJSON line."""
    return json.dumps({
        "modality": chunk.modality,
        "data": base64.b64encode(chunk.data).decode("ascii"),
        "metadata": chunk.metadata,
    }) + "\n"


def _chunk_to_binary_frame(chunk: ResultChunk) -> tuple[bytes, bytes]:
    """Serialize one result chunk as a header line plus its untouched payload.

    ``nbytes`` lets the reader frame by length instead of by delimiter, and no
    delimiter means no escaping — which is the only reason the NDJSON form has
    to base64 the payload. A 720p video_frame chunk costs two full passes over
    ~14.7 MB in that form (base64, then ``json.dumps`` escape-scanning every
    character it just produced); here the payload is handed on by reference.

    ``json.dumps`` escapes control characters, so the header can never contain
    a raw newline and the reader's line split is always unambiguous.
    """
    header = json.dumps({
        "modality": chunk.modality,
        "nbytes": len(chunk.data),
        "metadata": chunk.metadata,
    }, separators=(",", ":"))
    return header.encode("utf-8") + b"\n", chunk.data


class DeadConductorError(RuntimeError):
    """The conductor process exited before the workers finished setup (it
    exits when a worker dies during init), so the server must not bind."""


class APIServer:
    """Accept multimodal requests, forward to conductor, collect results."""

    def __init__(
        self,
        socket_path_prefix: str = "/tmp/mstar",
        upload_dir: str = "/tmp/mstar_uploads",
        hostname: str="localhost",
        timeout_seconds: float = 600.0,
        tensor_comm_protocol=CommProtocol.RDMA,
        tcp_transfer_device="",
        model=None,
        model_name: str = "dummy",
        log_stats: bool = False,
        log_stats_file: str | None = None,
        enable_nvtx: bool = False,
        model_config: dict | None = None,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

        # The result-delivery path runs on this process, not the worker's, so its
        # cost is invisible to worker-side markers. Streaming a 720p chunk means
        # base64-encoding 11 MiB into 14.7 MiB of ASCII and copying that again
        # through json.dumps, once per engine step.
        self.enable_nvtx = enable_nvtx

        # Per-request profiling: when enabled, a RequestProfile is collected for
        # each request and pretty-printed when the request finishes. ``log_stats_file``
        # (optional) appends the report to a file instead of only stdout.
        self.log_stats = log_stats
        self.log_stats_file = log_stats_file

        # Kept so the OpenAI-compatible layer can look up the per-model adapter
        # (``model_name``) and query model-level metadata such as the audio
        # output sample rate (``model``). The instance is the lightweight,
        # tokenizer-only model the API server already builds for preprocessing.
        self.model = model
        self.model_name = model_name

        self.preprocess_worker = PreprocessWorker(
            model=model,
            model_config=model_config,
            hostname=hostname,
            socket_path_prefix=socket_path_prefix,
            tensor_comm_protocol=tensor_comm_protocol,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=self.log_stats,
            enable_nvtx=enable_nvtx,
        )

        # Concurrent request tracking
        self.pending_requests: dict[str, PendingRequest] = {}
        self.recently_completed: collections.OrderedDict[str, float] = (
            collections.OrderedDict()
        )
        self._recently_completed_ttl = 15.0
        self.request_lock = threading.Lock()
        self.running = True

        # Set by main() once the conductor is spawned. It gets polled because a
        # conductor that dies sends nothing (see finalize_setup and
        # _process_messages).
        self.conductor_proc: mp.Process | None = None
        # Non-None once the deployment is going down because the conductor
        # died. It is the error every pending (and later) request gets, and the
        # reason main() exits non-zero. on_fatal stops the HTTP server.
        self.fatal_error: str | None = None
        self.on_fatal: Callable[[], None] | None = None
        self._liveness_interval_s = 0.5

        # ZMQ channel shared with conductor / workers
        self.communicator = make_communicator(
            my_id="api_server",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        )

        # Background thread that drains results from the conductor. Started by
        # finalize_setup() once the workers report ready — before that there's
        # no traffic to drain and the HTTP server isn't up yet.
        self._msg_thread = threading.Thread(
            target=self._process_messages, daemon=True
        )

    def finalize_setup(self) -> None:
        """Block until the conductor signals that every worker has finished
        setup (weight load + warmup + CUDA-graph capture), then start draining
        results. Called before the HTTP server binds, so ``mstar`` only begins
        serving once it can actually handle requests. Raises
        ``DeadConductorError`` if the conductor exits first (it does when a
        worker fails to initialize), so the server never binds.
        """
        logger.info(
            "Waiting for workers to finish setup "
            "(loading weights, capturing CUDA graphs)..."
        )
        while True:
            for message in self.communicator.get_all_new_messages():
                if (
                    isinstance(message, APIServerMessage)
                    and message.message_type == "setup_done"
                ):
                    logger.info("All workers ready")
                    self._msg_thread.start()
                    return
                logger.warning(
                    "Unexpected message before setup_done: %s", type(message)
                )
            exited = self._conductor_exited()
            if exited is not None:
                raise DeadConductorError(
                    f"conductor process exited with {exited} before the workers "
                    "finished setup"
                )
            time.sleep(0.01)

    def _conductor_exited(self) -> str | None:
        """Words for the conductor's exit status, or None while it runs (or
        when no handle was given)."""
        proc = self.conductor_proc
        if proc is None or proc.is_alive():
            return None
        return describe_exitcode(proc.exitcode)

    def _fail_pending_for_dead_conductor(self, exited: str) -> None:
        """The conductor is gone (it exits when a worker dies, or it crashed).
        Release every waiting client with a 503 now rather than at the request
        timeout, refuse new requests, and stop the HTTP server so the process
        exits non-zero instead of serving a deployment that can't run anything.
        """
        message = f"conductor process exited with {exited}, so the server is shutting down"
        logger.error("Conductor process exited with %s, shutting the server down", exited)
        with self.request_lock:
            self.fatal_error = message
            for req in self.pending_requests.values():
                if req.error is None:
                    req.error = message
                    req.error_status = 503
                req.event.set()
            on_fatal = self.on_fatal
        if on_fatal is not None:
            on_fatal()

    def set_on_fatal(self, callback: Callable[[], None]) -> None:
        """Register what stops the HTTP server once the conductor is gone.

        The message thread runs the callback when it finds the conductor
        dead, which can be any time after finalize_setup started it, so a
        callback registered later than that runs right away. Registration
        and the thread's read share request_lock, so it runs once.
        """
        with self.request_lock:
            self.on_fatal = callback
            fatal = self.fatal_error is not None
        if fatal:
            callback()

    # ----------------------------------------------------------
    # Submitting a request
    # ----------------------------------------------------------

    def submit_request(
        self,
        *,
        text: str | None = None,
        file_paths: dict[str, list[str]] | None = None,
        input_modalities: list[str],
        output_modalities: list[str],
        model_kwargs: dict | None = None,
        prompt_parts: list[PromptPart] | None = None,
        streaming: bool = True,
        request_id: str | None = None,
    ) -> str:
        """Build a :class:`NewRequestConductor` and send it to the conductor.

        Returns the ``request_id``. If a ``request_id`` is provided by the
        caller it is used as-is (useful for deterministic-noise debugging,
        since the conductor's per-request seed is derived from
        ``hash(request_id)``); otherwise a fresh uuid4 is generated.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        for m in input_modalities + output_modalities:
            if m not in SUPPORTED_MODALITIES:
                raise ValueError(f"Unsupported modality: {m!r}")
        if "video_frame" in input_modalities:
            raise ValueError("'video_frame' is an output-only modality")
        streaming_only = STREAMING_ONLY_MODALITIES.intersection(output_modalities)
        if streaming_only and not streaming:
            names = ", ".join(sorted(streaming_only))
            raise ValueError(
                f"Output modality {names} requires streaming=True; raw frame "
                "chunks cannot be returned as an aggregated response."
            )

        # Register pending request
        with self.request_lock:
            if self.fatal_error is not None:
                raise HTTPException(status_code=503, detail=self.fatal_error)
            self.pending_requests[request_id] = PendingRequest(
                streaming=streaming,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                profile=RequestProfile(
                    rid=request_id,
                    timing=RequestTiming(recv_time=time.perf_counter()),
                ),
            )

        self.preprocess_worker.new_request(PreprocessInput(
            request_id=request_id,
            text=text,
            file_paths=file_paths,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            model_kwargs=model_kwargs,
            prompt_parts=prompt_parts,
        ))

        logger.info(
            "Request %s submitted  in=%s  out=%s",
            request_id, input_modalities, output_modalities,
        )
        return request_id

    # ----------------------------------------------------------
    # Result collection (background thread)
    # ----------------------------------------------------------

    def _prune_recently_completed(self) -> None:
        now = time.time()
        stale = []
        for rid, ts in self.recently_completed.items():
            # The client handler pops pending_requests on its own timeout (and
            # a streaming client after its final flush), so the entry can be
            # gone while the rid is still here — clean up immediately then,
            # instead of KeyError-ing the whole message-processing tick.
            req = self.pending_requests.get(rid)
            drained = (
                req is not None
                and not self.preprocess_worker.has_pending_tensors(rid)
                and self.preprocess_worker.received_final_chunks(
                    rid, req.final_outputs
                )
            )
            if drained or req is None:
                stale.append((rid, False, drained))
            elif (now - ts) >= self._recently_completed_ttl:
                stale.append((rid, True, drained))
        for rid, lost_outputs, drained in stale:
            # only set the event when there are no more pending chunks
            req = self.pending_requests.get(rid)
            if req is not None:
                if lost_outputs:
                    # The TTL is a last resort for chunks that never arrive;
                    # closing the request as a silent success would hand the
                    # client a truncated (possibly empty) response.
                    logger.error(
                        "Request %s finished but its result chunks were not "
                        "delivered within %.0fs; failing the request",
                        rid, self._recently_completed_ttl,
                    )
                    if req.error is None:
                        req.error = (
                            "result delivery timed out; response is incomplete"
                        )
                        req.error_status = 500
                req.event.set()
                # Snapshot the data worker's tx/rx now: the request is done (all
                # final chunks received), so the worker thread is no longer mutating
                # this rid's transport state, and we must read it before the hard
                # cleanup drops it. Extends so it combines with the conductor's
                # worker-side transfers (set in the request_complete handler).
                if self.log_stats:
                    profile = req.profile
                    profile.tx_info.extend(self.preprocess_worker.get_tx_info(rid))
                    profile.rx_info.extend(self.preprocess_worker.get_rx_info(rid))
            # We're done delivering this request's outputs (all chunks in, or the
            # TTL gave up): tell the conductor via READS_DONE. It drives the hard
            # cleanup with a RemoveRequest once every reader has drained. Only
            # the drained case is known to have no read left in flight; the
            # others make the data worker gate the ACK on its own reads first.
            self.preprocess_worker.finished_reading(rid, drained=drained)
            self.recently_completed.pop(rid, None)

    def _process_messages(self) -> None:
        """Drain the ZMQ pull socket and route results to pending requests.
        Also watches the conductor process. Once it exits, every pending
        request is failed and the HTTP server is told to stop."""
        next_liveness_check = 0.0
        while self.running:
            now = time.monotonic()
            if now >= next_liveness_check:
                next_liveness_check = now + self._liveness_interval_s
                exited = self._conductor_exited()
                if exited is not None:
                    self._fail_pending_for_dead_conductor(exited)
                    return
            try:
                with self.request_lock:
                    if len(self.recently_completed) > 0:
                        self._prune_recently_completed()

                for message in self.communicator.get_all_new_messages():
                    if not isinstance(message, APIServerMessage):
                        logger.warning("Unexpected message type: %s", type(message))
                        continue

                    rid = message.body.request_id

                    with self.request_lock:
                        if rid in self.pending_requests:
                            if message.message_type == "result_tensors":
                                logger.debug(
                                    "Got new tensors of %s modality for request %s",
                                    message.body.modality, rid
                                )
                                self.preprocess_worker.new_result_tensors(
                                    message.body
                                )
                            elif message.message_type == "request_failed":
                                logger.error(
                                    "Request %s failed in the engine: %s",
                                    rid, message.body.error_message,
                                )
                                req = self.pending_requests[rid]
                                # Don't clobber an earlier, more specific
                                # error (e.g. a preprocess failure) with a
                                # downstream one.
                                if req.error is None:
                                    req.error = message.body.error_message
                                    req.error_status = message.body.status
                                # Release the waiting client immediately: no
                                # result is coming, so the alternative is the
                                # blanket request timeout.
                                req.event.set()
                                # The conductor has already dropped this rid,
                                # so no abort is needed — but the data worker
                                # still holds its transport state. Parking the
                                # rid here makes _prune_recently_completed
                                # release it once the client lets go.
                                self.recently_completed[rid] = time.time()
                            elif message.message_type == "request_complete":
                                logger.info("API server received %s done", rid)
                                self.recently_completed[rid] = time.time()

                                if not message.body.final_outputs:
                                    logger.warning(
                                        "Request %s completed with no reported "
                                        "outputs; holding for late results "
                                        "until the cleanup TTL", rid,
                                    )
                                req = self.pending_requests[rid]
                                req.final_outputs = message.body.final_outputs
                                req.profile.timing.conductor_ingest_time = \
                                    message.body.conductor_ingest_time
                                req.profile.timing.conductor_finish_time = \
                                    message.body.conductor_finish_time
                                req.profile.graph_timings = list(message.body.graph_timings.values())
                                # Conductor-merged worker-side transfers; extend
                                # so they combine with the data worker's own
                                # tx/rx (applied from the profile-update queue).
                                req.profile.rx_info.extend(message.body.rx_info)
                                req.profile.tx_info.extend(message.body.tx_info)
                        elif rid in self.recently_completed:
                            if message.message_type == "result_tensors":
                                self.preprocess_worker.discard_result_tensors(message.body)
                                # The client already finished (popped the
                                # request), so these tensors have no delivery
                                # path — the completion-vs-results reorder
                                # window was hit.
                                logger.warning(
                                    "Result tensors for %s arrived after the "
                                    "client finished; dropping them", rid,
                                )
                            else:
                                logger.debug("Late message for completed %s: %s", rid, message.message_type)
                        else:
                            logger.warning(
                                "Message for unknown request %s: %s", rid, message.message_type
                            )
                            if message.message_type == "result_tensors":
                                self.preprocess_worker.discard_result_tensors(message.body)
                # Apply data-worker profiling: preprocess-finish timestamp + raw
                # input sizes. (The data worker's own tx/rx are snapshotted
                # directly in _prune_recently_completed once the request is done.)
                for update in self.preprocess_worker.get_profile_updates():
                    with self.request_lock:
                        req = self.pending_requests.get(update.request_id)
                        if req is not None:
                            req.profile.timing.preprocess_finish_time = \
                                update.preprocess_finish_time
                            req.profile.inputs = update.inputs

                for result_chunk in self.preprocess_worker.get_result_chunks():
                    logger.debug(
                        "Got result chunk of %s modality for request %s",
                        result_chunk.modality, result_chunk.request_id
                    )
                    rid = result_chunk.request_id
                    with self.request_lock:
                        req = self.pending_requests.get(rid)
                        if req is None:
                            # Client already finished (timeout/pop); don't let a
                            # late chunk KeyError abort the processing tick.
                            logger.warning(
                                "Result chunk for %s arrived after the client "
                                "finished; dropping it", rid,
                            )
                            continue
                        now = time.perf_counter()
                        if req.profile.timing.first_chunk_time is None:
                            req.profile.timing.first_chunk_time = now
                        req.profile.timing.last_chunk_time = now
                        req.chunks.append(result_chunk)

                        if result_chunk.modality == "error":
                            # The data worker failed this request (preprocess,
                            # or postprocess of a result tensor); release the
                            # waiting client with the error instead of letting
                            # it time out.
                            if req.error is None:
                                req.error = result_chunk.data.decode("utf-8", "replace")
                                req.error_status = int(
                                    (result_chunk.metadata or {}).get("status", 500)
                                )
                            req.event.set()
                            # Park the rid so _prune_recently_completed reclaims
                            # the data worker's per-request state once the
                            # client lets go of the request.
                            self.recently_completed[rid] = time.time()
            except Exception:
                if self.running:
                    logger.exception("Error in message processing loop")
                    time.sleep(0.01)
            time.sleep(0.001)

    # ----------------------------------------------------------
    # Streaming helper
    # ----------------------------------------------------------

    async def iter_result_chunks(self, request_id: str):
        """Yield raw :class:`ResultChunk` objects as they arrive.

        Shared source for both output surfaces: ``/generate`` formats each
        chunk as NDJSON (via :meth:`async_stream_results`), while the
        OpenAI-compatible endpoints translate the same chunks into SSE. The
        per-request timeout, incremental drain, and final flush behave exactly
        as before — only the yielded type changed (``ResultChunk`` instead of a
        pre-serialized line).
        """
        start = time.time()
        finished = False
        try:
            while True:
                if time.time() - start > self.timeout_seconds:
                    raise HTTPException(status_code=500, detail="Request timed out")

                new_chunks: list[ResultChunk] = []
                done = False
                with self.request_lock:
                    req = self.pending_requests.get(request_id)
                    if req:
                        avail = len(req.chunks)
                        consumed = req.consumed_chunks
                        new_chunks = req.chunks[consumed:avail]
                        req.consumed_chunks = avail
                        done = req.event.is_set()
                    else:
                        done = True

                for chunk in new_chunks:
                    if self.enable_nvtx:
                        profiler.mark("apiserver.chunk_available")
                    yield chunk

                if done:
                    logger.info("Async stream results received finish for %s", request_id)
                    # flush remaining
                    remaining: list[ResultChunk] = []
                    finished_req: PendingRequest | None = None
                    with self.request_lock:
                        req = self.pending_requests.get(request_id)
                        if req:
                            remaining = req.chunks[req.consumed_chunks:]
                            finished_req = self.pending_requests.pop(request_id, None)
                    # Profiling (incl. the optional file write) runs outside the
                    # lock; the popped request is no longer shared with other threads.
                    if finished_req is not None:
                        self._finalize_profile(finished_req)
                    for chunk in remaining:
                        yield chunk
                    # A request can fail after the stream is already open
                    # (preprocess error, result-delivery timeout); the HTTP
                    # status is committed by then, so the error must travel
                    # in-band as the final chunk.
                    if finished_req is not None and finished_req.error is not None:
                        yield ResultChunk(
                            request_id=request_id,
                            modality="error",
                            data=str(finished_req.error).encode("utf-8"),
                            metadata={"status": finished_req.error_status},
                        )
                    finished = True
                    break

                await asyncio.sleep(0.001)
        finally:
            if not finished:
                self.abort_request(request_id)

    def async_stream_results(self, request_id: str, binary: bool = False):
        """Yield the serialized body of ``/generate`` one piece at a time.

        ``binary`` selects the length-framed form negotiated through ``Accept``.
        The default stays NDJSON, so a client that did not negotiate — including
        the Rust frontend, which never reads ``Accept`` — sees today's bytes.

        Deliberately a plain ``def`` returning the chosen async generator rather
        than an ``async def`` delegating to it: the branch is per-request, not
        per-chunk, and this keeps both bodies flat.
        """
        if binary:
            return self._stream_binary(request_id)
        return self._stream_ndjson(request_id)

    async def _stream_ndjson(self, request_id: str):
        async for chunk in self.iter_result_chunks(request_id):
            line = self._chunk_to_ndjson(chunk)
            if not self.enable_nvtx:
                yield line
                continue
            profiler.mark(f"apiserver.yield_line.bytes[{len(line)}]")
            # Spans the handoff to Starlette/uvicorn: chunked-transfer framing
            # and the socket writes for ~14.7 MB, plus any transport
            # backpressure. The generator resumes only once that is done, so
            # this range is the server's share of the client's blocking read.
            profiler.range_push(f"apiserver.socket_write.bytes[{len(line)}]")
            try:
                yield line
            finally:
                profiler.range_pop()

    async def _stream_binary(self, request_id: str):
        async for chunk in self.iter_result_chunks(request_id):
            header, payload = _chunk_to_binary_frame(chunk)
            # Two yields rather than one concatenation: joining them would copy
            # the whole payload to prepend ~100 bytes, which is the class of
            # work this framing exists to remove.
            yield header
            if not self.enable_nvtx:
                yield payload
                continue
            profiler.mark(f"apiserver.yield_frame.bytes[{len(payload)}]")
            # Same range name as the NDJSON path on purpose: it is the column
            # the gap budget in STREAMING_GAP_BUDGET.md is built from, so the
            # two protocols stay directly comparable in one analyzer run.
            profiler.range_push(f"apiserver.socket_write.bytes[{len(payload)}]")
            try:
                yield payload
            finally:
                profiler.range_pop()

    def _chunk_to_ndjson(self, chunk: ResultChunk) -> str:
        if not self.enable_nvtx:
            return _chunk_to_ndjson_payload(chunk)

        # Split rather than wrapped as one range: the question these markers
        # answer is which of the two full passes over the payload dominates.
        profiler.range_push(f"apiserver.b64encode.bytes[{len(chunk.data)}]")
        encoded = base64.b64encode(chunk.data).decode("ascii")
        profiler.range_pop()

        profiler.range_push(f"apiserver.json_dumps.chars[{len(encoded)}]")
        line = json.dumps({
            "modality": chunk.modality,
            "data": encoded,
            "metadata": chunk.metadata,
        }) + "\n"
        profiler.range_pop()
        return line

    # ----------------------------------------------------------
    # Non-streaming helper
    # ----------------------------------------------------------

    async def collect_results(
        self, request_id: str, raw_request: Request | None = None
    ) -> list[ResultChunk]:
        """Wait for the request to finish (or the client to disconnect), then
        return its chunks. Disconnecting or timing out releases engine state."""
        start = time.time()
        while True:
            with self.request_lock:
                req = self.pending_requests.get(request_id)
                done = req.event.is_set() if req else True
            if done:
                break
            if time.time() - start > self.timeout_seconds:
                self.abort_request(request_id)
                raise HTTPException(status_code=500, detail="Request timed out")
            if raw_request is not None and await raw_request.is_disconnected():
                self.abort_request(request_id)
                return []
            await asyncio.sleep(0.005)

        with self.request_lock:
            req = self.pending_requests.pop(request_id, None)
        if req is None:
            return []
        if req.error is not None:
            raise HTTPException(
                status_code=req.error_status, detail=req.error
            )
        # Profiling (incl. the optional file write) runs outside the lock; the
        # popped request is no longer shared with other threads.
        self._finalize_profile(req)
        return list(req.chunks)

    def _finalize_profile(self, req: PendingRequest) -> None:
        """Stamp the finish time, aggregate output sizes, and emit the report.

        Called once per request from whichever completion path pops it
        (streaming or non-streaming), after the request has been removed from
        ``pending_requests`` so it is no longer shared — callers must NOT hold
        ``request_lock``. No-op unless ``--log-stats`` is set.
        """
        if not self.log_stats:
            return

        req.profile.timing.finish_time = time.perf_counter()

        # Aggregate the collected chunks into per-modality output info.
        by_modality: dict[str, OutputInfo] = {}
        for chunk in req.chunks:
            info = by_modality.get(chunk.modality)
            if info is None:
                info = OutputInfo(modality=chunk.modality, count=0, total_bytes=0)
                by_modality[chunk.modality] = info
            info.count += 1
            info.total_bytes += len(chunk.data)
        req.profile.outputs = list(by_modality.values())

        try:
            pretty_print_profile(req.profile, self.log_stats_file)
        except Exception:
            logger.exception("Failed to emit request profile for %s", req.profile.rid)

    def abort_request(self, request_id: str) -> None:
        """Stop GPU work for a request the client abandoned and drop its state."""
        with self.request_lock:
            active = (
                request_id in self.pending_requests
                or request_id in self.recently_completed
            )
            self.pending_requests.pop(request_id, None)
            self.recently_completed.pop(request_id, None)
        if not active:
            return
        logger.info("Client cancelled request %s; releasing resources", request_id)
        self.preprocess_worker.abort_request(request_id)

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    def cleanup(self) -> None:
        self.preprocess_worker.shutdown()
        self.running = False
        if hasattr(self, "_msg_thread") and self._msg_thread.is_alive():
            self._msg_thread.join(timeout=2)

# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

# Behind an ingress that serves the app under a sub-path -- Run:AI routes a
# workload at /<project>/<job-name>/ and does NOT strip the prefix before it
# reaches the pod -- FastAPI has to be told, or every route 404s on a path it
# considers unknown. Empty by default, so a direct deployment is unaffected.
app = FastAPI(
    title="mstar API",
    description="Multimodal Inference API",
    root_path=os.environ.get("MSTAR_ROOT_PATH", ""),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_server: APIServer | None = None

# Mount the OpenAI-compatible routes (/v1/*) alongside the native /generate.
# The router resolves the loaded model's adapter lazily per request, so models
# without an adapter simply return a 404 there and keep working via /generate.
from mstar.api_server.openai.router import router as openai_router  # noqa: E402

app.include_router(openai_router)


@app.post("/generate")
async def generate(
    request: Request,
    text: Optional[str] = Form(None),
    files: Optional[list[UploadFile]] = File(None),
    input_modalities: Optional[str] = Form(None),
    output_modalities: str = Form("text"),
    streaming: bool = Form(True),
    model_kwargs: Optional[str] = Form(None),
    request_id: Optional[str] = Form(None),
):
    """Submit a multimodal generation request.

    Args:
        text: Optional text input.
        files: Optional media files (images, audio, video).  The modality of
            each file is inferred from its extension.
        input_modalities: Comma-separated list of input modalities.  When
            omitted, modalities are auto-detected from the provided data.
        output_modalities: Comma-separated list of desired output modalities
            (default ``"text"``).
        streaming: If ``True``, return an NDJSON stream of result chunks.
        model_kwargs: Optional JSON string of model-specific parameters.
        request_id: Optional client-supplied request id. When omitted, the
            server generates a fresh uuid4. Pinning this is useful for
            deterministic-noise debugging because the conductor seeds its
            per-request RNG via ``hash(request_id)``.
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    out_mods = [m.strip() for m in output_modalities.split(",") if m.strip()]
    streaming_only = STREAMING_ONLY_MODALITIES.intersection(out_mods)
    if streaming_only and not streaming:
        names = ", ".join(sorted(streaming_only))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Output modality {names} requires streaming=true; raw frame "
                "chunks cannot be returned as an aggregated response."
            ),
        )

    # --- save uploaded files, grouped by modality ----------------
    file_paths: dict[str, list[str]] = {}
    parts: list[PromptPart] = []
    if files:
        for f in files:
            modality = _detect_modality(f.filename or "")
            if modality == "unknown":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot determine modality for file: {f.filename}",
                )
            # Sanitize: only the final path component of the client name;
            # embedded separators (../) would escape upload_dir.
            base = os.path.basename(f.filename or "") or "upload"
            save_name = f"{uuid.uuid4()}_{base}"
            save_path = api_server.upload_dir / save_name
            content = await f.read()
            await run_in_threadpool(save_path.write_bytes, content)
            paths = file_paths.setdefault(modality, [])
            parts.append(PromptPart(modality=modality, index=len(paths)))
            paths.append(str(save_path))
    if text:
        parts.append(PromptPart(modality="text", text=text))

    # --- resolve input modalities --------------------------------
    # An explicit list is then the layout on its own, so the derived parts go
    # rather than disagree with it.
    if input_modalities is not None:
        in_mods = [m.strip() for m in input_modalities.split(",") if m.strip()]
        parts = []
        # A layout with no text slot would drop the prompt on the floor; it
        # went at the end before ordering was kept, so put it back there.
        if text and "text" not in in_mods:
            in_mods.append("text")
        # And a declared text slot with no text renders to nothing, so the
        # prompt has one fewer span than the layout plans for. Drop it here,
        # where intake and the schedule builder both still read the same list.
        if not text:
            in_mods = [m for m in in_mods if m != "text"]
    else:
        in_mods = [p.modality for p in parts]

    if "video_frame" in in_mods:
        raise HTTPException(
            status_code=400,
            detail="'video_frame' is an output-only modality",
        )

    try:
        parsed_kwargs = json.loads(model_kwargs) if model_kwargs else None
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail="model_kwargs must be valid JSON",
        ) from e

    if parsed_kwargs is not None and not isinstance(parsed_kwargs, dict):
        raise HTTPException(
            status_code=400,
            detail="model_kwargs must be a JSON object",
        )

    try:
        request_id = api_server.submit_request(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=out_mods,
            model_kwargs=parsed_kwargs,
            prompt_parts=parts or None,
            streaming=streaming,
            request_id=request_id,
        )

        if streaming:
            # Substring match, not RFC 7231 q-value parsing: the value is a
            # private vendor type that appears in no other media range, and a
            # client that does not ask for it keeps the historical NDJSON body.
            binary = BINARY_STREAM_MEDIA_TYPE in request.headers.get("accept", "")
            return StreamingResponse(
                api_server.async_stream_results(request_id, binary=binary),
                media_type=BINARY_STREAM_MEDIA_TYPE if binary else NDJSON_STREAM_MEDIA_TYPE,
                headers={"Cache-Control": "no-cache", "Vary": "Accept"},
            )

        chunks = await api_server.collect_results(request_id, request)
        outputs: dict[str, list[dict]] = {}
        for chunk in chunks:
            outputs.setdefault(chunk.modality, []).append({
                "data": base64.b64encode(chunk.data).decode("ascii"),
                "metadata": chunk.metadata,
            })
        return JSONResponse({
            "request_id": request_id,
            "outputs": outputs,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Deferred cleanup of uploaded files
        if file_paths:
            def _cleanup(paths: dict[str, list[str]]) -> None:
                time.sleep(60)
                for ps in paths.values():
                    for p in ps:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except OSError:
                            pass
            threading.Thread(
                target=_cleanup, args=(file_paths,), daemon=True
            ).start()


@app.get("/health")
async def health_check():
    # Report unhealthy while shutting down, so a load balancer stops sending here.
    if api_server is not None and api_server.fatal_error is not None:
        raise HTTPException(status_code=503, detail=api_server.fatal_error)
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    if api_server is not None:
        api_server.cleanup()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main(argv: list[str] | None = None):
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="mstar — launch API server and conductor from a config file"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mooncake-port", type=int, default=8080)
    parser.add_argument(
        "--socket-path-prefix", type=str, default="/tmp/mstar",
        help="ZMQ IPC socket path prefix (shared with conductor/workers)",
    )
    parser.add_argument(
        "--upload-dir", type=str, default="/tmp/mstar_uploads",
        help="Directory for temporary uploaded files",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--enable-nvtx",
        action="store_true",
        help="Enable torch.cuda.nvtx markers during execution",
    )
    parser.add_argument(
        "--tensor-comm-protocol",
        type=str, default="RDMA",
        help="Tensor transfer protocol: RDMA, TCP, or SHM (shared memory)"
    )
    parser.add_argument(
        "--tcp-transfer-device",
        type=str, default="",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for caching downloaded HuggingFace model files",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Print per-request profiling stats when each request finishes",
    )
    parser.add_argument(
        "--log-stats-file", type=str, default=None,
        help="Append per-request profiling stats to this file (implies --log-stats)",
    )
    parser.add_argument(
        "--rust-frontend", action="store_true",
        help="Serve HTTP from the Rust mstar-server binary instead of "
             "uvicorn/FastAPI; the Python process keeps preprocessing "
             "and the conductor protocol",
    )
    parser.add_argument(
        "--rust-frontend-bin", type=str, default=None,
        help="Path to the mstar-server binary (default: MSTAR_SERVER_BIN, "
             "$PATH, then rust/server/target/release)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [api_server] %(name)s: %(message)s",
    )
    quiet_noisy_loggers()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", "dummy")
    # Forward yaml-level model_kwargs to the API-server-side lightweight
    # model instance too, so it sees the same Pi05Config (action_horizon, etc.)
    # as the conductor-side instance. Without this they could diverge.
    yaml_model_kwargs = config.get("model_kwargs", {}) or {}

     # Create a lightweight model instance for prompt processing
    # (tokenization only — no GPU weights needed)
    from mstar.model.registry import get_model_class
    model = get_model_class(model_name)(
        model_path_hf=HF_MODELS.get(model_name, {}).get("model_path_hf", ""),
        cache_dir=args.cache_dir,
        **yaml_model_kwargs,
    )

    global api_server
    log_stats = args.log_stats or args.log_stats_file is not None
    api_server = APIServer(
        socket_path_prefix=args.socket_path_prefix,
        upload_dir=args.upload_dir,
        timeout_seconds=args.timeout,
        tensor_comm_protocol=CommProtocol(args.tensor_comm_protocol),
        model=model,
        model_name=model_name,
        model_config=config,
        tcp_transfer_device=args.tcp_transfer_device,
        log_stats=log_stats,
        log_stats_file=args.log_stats_file,
        enable_nvtx=args.enable_nvtx,
    )

    # Spawn conductor in a separate process
    ctx = mp.get_context("spawn")
    conductor_proc = ctx.Process(
        target=_conductor_process_target,
        args=(
            model_name,
            args.config,
            args.socket_path_prefix,
            args.enable_nvtx,
            log_stats,
            args.log_level,
            args.cache_dir,
            CommProtocol(args.tensor_comm_protocol),
            args.tcp_transfer_device
        ),
    )
    conductor_proc.start()
    logger.info("Conductor process started (pid=%d, model=%s)", conductor_proc.pid, model_name)
    api_server.conductor_proc = conductor_proc

    rust_proc = None
    bridge_dir = None
    exit_code = 0
    try:
        # Block until all workers have finished setup, so the server only binds
        # (and logs "Starting…") once it can actually serve requests.
        api_server.finalize_setup()
        if args.rust_frontend:
            import tempfile

            from mstar.api_server.rust_frontend import (
                RustFrontendBridge,
                find_server_binary,
                launch_rust_server,
            )

            bridge_dir = tempfile.mkdtemp(prefix="mstar_rust_frontend_")
            # Forward --host to the Rust frontend (it binds 127.0.0.1 by
            # default; --host 0.0.0.0 for the multi-node / container case).
            os.environ.setdefault("MSTAR_SERVER_HOST", args.host)
            rust_proc = launch_rust_server(
                find_server_binary(args.rust_frontend_bin), model_name,
                args.port, bridge_dir, args.upload_dir)
            logger.info("Starting mstar API server (Rust frontend) on port %s",
                        args.port)
            bridge = RustFrontendBridge(api_server, bridge_dir)
            api_server.set_on_fatal(bridge.stop)
            bridge.run()
        else:
            logger.info("Starting mstar API server on %s:%s", args.host, args.port)
            # uvicorn.run() inlined so the server object is reachable. The
            # message thread stops it when the conductor dies.
            server = uvicorn.Server(
                uvicorn.Config(app, host=args.host, port=args.port, access_log=False)
            )

            def _stop_server():
                server.should_exit = True

            api_server.set_on_fatal(_stop_server)
            server.run()
            if not server.started:
                exit_code = 3  # uvicorn.run()'s own code for a server that never came up
    except KeyboardInterrupt:
        pass
    except DeadConductorError as e:
        logger.error("%s", e)
        exit_code = 1
    finally:
        if rust_proc is not None:
            rust_proc.terminate()
        if bridge_dir is not None:
            import shutil
            shutil.rmtree(bridge_dir, ignore_errors=True)
        if api_server is not None:
            api_server.cleanup()
        _shutdown_conductor_process(conductor_proc)
    # A worker or conductor death is a failed run. A SIGINT stop still exits 0.
    if api_server.fatal_error is not None:
        exit_code = 1
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
