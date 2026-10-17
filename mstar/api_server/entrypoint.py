"""FastAPI server entry point for multimodal inference requests."""

import asyncio
import base64
import collections
import hashlib
import json
import logging
import multiprocessing as mp
import os
import secrets
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import msgpack
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from mstar.api_server import media_io
from mstar.api_server.data_worker import PreprocessWorker
from mstar.api_server.request_types import (
    APIServerMessage,
    PreprocessInput,
    ResultChunk,
    ResultTokens,
)
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.model.multimodal import PromptPart
from mstar.model.registry import model_init_kwargs
from mstar.profile.display import pretty_print_profile
from mstar.profile.format import OutputInfo, ReadQuarantine, RequestProfile, RequestTiming
from mstar.utils import profiler
from mstar.utils.exitcode import describe_exitcode
from mstar.utils.gc_freeze import freeze_after_setup
from mstar.utils.logging_config import quiet_noisy_loggers
from mstar.utils.orphan import watch_parent

logger = logging.getLogger(__name__)

SUPPORTED_MODALITIES = frozenset({
    "text", "image", "audio", "video", "video_frame", "action", "scalar", "tensor",
})
STREAMING_ONLY_MODALITIES = frozenset({"video_frame"})


def _seed_from_client_id(client_id: str) -> int:
    """Stable 32-bit seed matching the conductor's ``_req_id_to_seed``."""
    digest = hashlib.md5(client_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")

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
        cache_dir=cache_dir,
        **{**model_init_kwargs(model_name), **yaml_model_kwargs},
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
    # Client-visible id (supplied or server-generated). Internal routing
    # uses ``internal_id``, which is unique per submit.
    client_id: str | None = None
    internal_id: str | None = None
    # A collect/iter/websocket consumer has resolved this internal id.
    consumer_attached: bool = False
    # Set by the streaming handler on its event loop; the message thread wakes
    # it through these when a chunk or the completion lands, so the handler
    # sleeps between chunks instead of polling every millisecond per request.
    wake: Any | None = None
    loop: Any | None = None

    def notify(self) -> None:
        """Wake the streaming handler, from any thread."""
        loop, wake = self.loop, self.wake
        if loop is None or wake is None:
            return
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            pass  # the loop is closed: the client is gone


def _set_events(wakes: list) -> None:
    for wake in wakes:
        wake.set()


def notify_many(reqs) -> None:
    """``PendingRequest.notify`` for a batch: one loop wake-up (a self-pipe
    write) per event loop rather than one per chunk. A step's RESULT_TOKENS
    frame lands one chunk per running request, so the message thread was
    waking the loop once per token."""
    by_loop: dict[int, tuple[Any, list]] = {}
    for req in reqs:
        loop, wake = req.loop, req.wake
        if loop is None or wake is None:
            continue
        entry = by_loop.get(id(loop))
        if entry is None:
            entry = by_loop[id(loop)] = (loop, [])
        entry[1].append(wake)
    for loop, wakes in by_loop.values():
        try:
            loop.call_soon_threadsafe(_set_events, wakes)
        except RuntimeError:
            pass  # the loop is closed: the client is gone


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

    # A streaming handler sleeps until notified; this bounds a missed wake.
    _stream_wake_backstop_s = 0.05

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
        self.enable_nvtx = profiler.nvtx_enabled(enable_nvtx)

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
            enable_nvtx=self.enable_nvtx,
        )

        # Concurrent request tracking
        self.pending_requests: dict[str, PendingRequest] = {}
        self.recently_completed: collections.OrderedDict[str, float] = (
            collections.OrderedDict()
        )
        self._recently_completed_ttl = 15.0
        # Each submit mints a process-unique internal id. The client id
        # is kept only at this edge for abort-by-client-id and response
        # labeling; everything downstream uses the internal id.
        self._next_internal_id = 0
        # Random token so a client-supplied string cannot name another
        # request's internal id (``{client}#{token}-{n}``).
        self._id_ns = secrets.token_hex(8)
        self._client_to_internal: dict[str, str] = {}
        self._internal_to_client: dict[str, str] = {}
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
        # worker id -> its latest report of buffers kept from failed reads
        self.worker_read_quarantine: dict[str, ReadQuarantine] = {}

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
                    # the set-up heap is complete: keep gen-2 collections off
                    # it (measured 200 ms per pass on this process under load)
                    freeze_after_setup("API server")
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
            for req in list(self.pending_requests.values()):
                if req.error is None:
                    req.error = message
                    req.error_status = 503
                req.event.set()
                req.notify()
            on_fatal = self.on_fatal
        if on_fatal is not None:
            on_fatal()

    def read_quarantines(self) -> list[ReadQuarantine]:
        """Buffers kept from failed reads, per process that has kept any: the
        workers' latest reports and this process's data worker."""
        quarantines = list(self.worker_read_quarantine.values())
        own = self.preprocess_worker.tensor_manager.read_quarantine()
        if own.reads:
            quarantines.append(own)
        return quarantines

    def health_problem(self) -> str | None:
        """Why ``/health`` should fail, or None. Kept read buffers are never
        freed, so a transport that keeps failing grows them until a restart;
        past the budget the deployment says so rather than serve on quietly."""
        if self.fatal_error is not None:
            return self.fatal_error
        over = [q for q in self.read_quarantines() if q.over_budget]
        if not over:
            return None
        return "; ".join(
            f"{q.entity_id} keeps {q.nbytes} bytes of buffers from {q.reads} "
            f"failed tensor reads, past its budget of {q.budget_bytes} bytes"
            for q in over
        )

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

        Returns the client-visible ``request_id``. If a ``request_id`` is
        provided by the caller it is kept for responses, abort-by-id, and
        the conductor seed (unless ``model_kwargs`` already has a non-None
        ``seed``); otherwise a fresh uuid4 is generated. Each submit also
        mints a unique internal id used by the worker, conductor, and tensor
        keys. Client-facing lookups go only through the client-id map, so a
        supplied string cannot name another request's internal id.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        client_id = request_id
        model_kwargs = dict(model_kwargs or {})
        if model_kwargs.get("seed") is None:
            model_kwargs["seed"] = _seed_from_client_id(client_id)

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

        with self.request_lock:
            if self.fatal_error is not None:
                raise HTTPException(status_code=503, detail=self.fatal_error)
            self._next_internal_id = getattr(self, "_next_internal_id", 0) + 1
            ns = getattr(self, "_id_ns", None) or secrets.token_hex(8)
            self._id_ns = ns
            internal_id = f"{client_id}#{ns}-{self._next_internal_id}"
            self.pending_requests[internal_id] = PendingRequest(
                streaming=streaming,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                profile=RequestProfile(
                    rid=client_id,
                    timing=RequestTiming(recv_time=time.perf_counter()),
                ),
                client_id=client_id,
                internal_id=internal_id,
            )
            if not hasattr(self, "_client_to_internal"):
                self._client_to_internal = {}
            if not hasattr(self, "_internal_to_client"):
                self._internal_to_client = {}
            old = self._client_to_internal.get(client_id)
            self._client_to_internal[client_id] = internal_id
            self._internal_to_client[internal_id] = client_id
            abort_old = None
            if old is not None and old != internal_id:
                old_req = self.pending_requests.get(old)
                if old_req is not None and not old_req.consumer_attached:
                    # No consumer is bound to the previous submit: abort it
                    # through the normal path (one READS_DONE, maps dropped).
                    # An already-attached consumer stays on ``old``.
                    if self._drop_live(old):
                        abort_old = old
            # The pending id (not the cancellation) is registered here,
            # under request_lock, then _admit_lock inside mark_pending.
            # Always request_lock first if both are needed.
            mark = getattr(self.preprocess_worker, "mark_pending", None)
            if mark is not None:
                mark(internal_id)
            else:
                pending = getattr(self.preprocess_worker, "_pending_ids", None)
                if pending is None:
                    self.preprocess_worker._pending_ids = set()
                    pending = self.preprocess_worker._pending_ids
                pending.add(internal_id)

        if abort_old is not None:
            logger.info(
                "Aborting request %s; a new submit reused its client id", abort_old,
            )
            self.preprocess_worker.abort_request(abort_old)

        self._after_publish(internal_id)
        self.preprocess_worker.new_request(PreprocessInput(
            request_id=internal_id,
            text=text,
            file_paths=file_paths,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            model_kwargs=model_kwargs,
            prompt_parts=prompt_parts,
        ))

        logger.info(
            "Request %s submitted as %s  in=%s  out=%s",
            client_id, internal_id, input_modalities, output_modalities,
        )
        return client_id

    def _after_publish(self, internal_id: str) -> None:
        """Test hook: API record and worker pending set are visible."""

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
            try:
                drained = (
                    req is not None
                    and not self.preprocess_worker.has_pending_tensors(rid)
                    and self.preprocess_worker.received_final_chunks(
                        rid, req.final_outputs
                    )
                )
            except Exception:
                # This check runs every pass, so a raise would repeat until
                # the TTL; fail this request now rather than stall the rest.
                logger.exception(
                    "Checking whether request %s drained raised; failing the request", rid,
                )
                stale.append((rid, "result delivery failed; response is incomplete", False))
                continue
            if drained or req is None:
                stale.append((rid, None, drained))
            elif (now - ts) >= self._recently_completed_ttl:
                # The TTL is a last resort for chunks that never arrive;
                # closing the request as a silent success would hand the
                # client a truncated (possibly empty) response.
                logger.error(
                    "Request %s finished but its result chunks were not "
                    "delivered within %.0fs; failing the request",
                    rid, self._recently_completed_ttl,
                )
                stale.append((rid, "result delivery timed out; response is incomplete", drained))
        for rid, lost_outputs, drained in stale:
            # only set the event when there are no more pending chunks
            req = self.pending_requests.get(rid)
            if req is not None:
                if lost_outputs is not None and req.error is None:
                    req.error = lost_outputs
                    req.error_status = 500
                req.event.set()
                req.notify()
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
            if rid not in self.pending_requests:
                self._forget_mapping(rid)

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
                    try:
                        self._route_message(message)
                    except Exception:
                        # One bad message must not drop the rest of this batch,
                        # which is already off the socket.
                        logger.exception(
                            "Handling %s for request %s failed; dropped it",
                            getattr(message, "message_type", type(message).__name__),
                            getattr(getattr(message, "body", None), "request_id", None),
                        )
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

                # the handlers are woken once per pass, after every chunk of
                # the pass has been appended, not once per chunk
                to_wake: dict[int, PendingRequest] = {}
                for result_chunk in self.preprocess_worker.get_result_chunks():
                    rid = result_chunk.request_id
                    try:
                        req = self._route_result_chunk(result_chunk)
                    except Exception:
                        logger.exception(
                            "Delivering the %s chunk for request %s failed; dropped it",
                            result_chunk.modality, rid,
                        )
                        continue
                    if req is not None:
                        to_wake[id(req)] = req
                if to_wake:
                    notify_many(to_wake.values())
            except Exception:
                if self.running:
                    logger.exception("Error in message processing loop")
                    time.sleep(0.01)
            time.sleep(0.001)

    def _route_message(self, message) -> None:
        """Apply one message from the conductor or a worker."""
        if not isinstance(message, APIServerMessage):
            logger.warning("Unexpected message type: %s", type(message))
            return
        if message.message_type == "read_quarantine":
            self.worker_read_quarantine[message.body.entity_id] = message.body
            return
        if message.message_type == "result_tokens":
            self._route_result_tokens(message.body)
            return

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
                        req.error = self._client_facing_error(
                            message.body.error_message, rid,
                        )
                        req.error_status = message.body.status
                    # Release the waiting client immediately: no
                    # result is coming, so the alternative is the
                    # blanket request timeout.
                    req.event.set()
                    req.notify()
                    # The conductor has already dropped this rid,
                    # so no abort is needed — but the data worker
                    # still holds its transport state. Parking the
                    # rid here makes _prune_recently_completed
                    # release it once the client lets go.
                    self._park_completed(rid)
                elif message.message_type == "request_complete":
                    logger.info("API server received %s done", rid)
                    # Capture first: parking / abort may have
                    # already dropped this rid from pending.
                    req = self.pending_requests.get(rid)
                    self._park_completed(rid)
                    if req is None:
                        return

                    if not message.body.final_outputs:
                        logger.warning(
                            "Request %s completed with no reported "
                            "outputs; holding for late results "
                            "until the cleanup TTL", rid,
                        )
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

    def _route_result_tokens(self, body: ResultTokens) -> None:
        """One step's inline values, one per request. Each request is
        handled on its own, so one bad entry does not drop the others."""
        queued = False
        try:
            with self.request_lock:
                for i, rid in enumerate(body.request_ids):
                    try:
                        if rid in self.pending_requests:
                            # one wake-up for the frame, after every
                            # request's value is queued
                            self.preprocess_worker.new_result_token(
                                rid, body.values[i], body.loop_indices[i],
                                body.signal, body.modality, wake=False,
                            )
                            queued = True
                        elif rid not in self.recently_completed:
                            # A request that is gone needs nothing acked back.
                            logger.warning("Token for unknown request %s dropped", rid)
                    except Exception:
                        logger.exception(
                            "Handling the result token for request %s failed; dropped it", rid,
                        )
        finally:
            if queued:
                self.preprocess_worker.wake()

    def _route_result_chunk(self, result_chunk: ResultChunk) -> PendingRequest | None:
        """Hand one data-worker chunk to its pending request, and return
        that request so the caller can wake its handler."""
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
                return None
            now = time.perf_counter()
            if req.profile.timing.first_chunk_time is None:
                req.profile.timing.first_chunk_time = now
            req.profile.timing.last_chunk_time = now
            if req.client_id is not None:
                result_chunk.request_id = req.client_id
            req.chunks.append(result_chunk)

            if result_chunk.modality == "error":
                # The data worker failed this request (preprocess,
                # or postprocess of a result tensor); release the
                # waiting client with the error instead of letting
                # it time out.
                text = self._client_facing_error(
                    result_chunk.data.decode("utf-8", "replace"), rid,
                )
                result_chunk.data = text.encode("utf-8")
                if req.error is None:
                    req.error = text
                    req.error_status = int(
                        (result_chunk.metadata or {}).get("status", 500)
                    )
                req.event.set()
                # Park the rid so _prune_recently_completed reclaims
                # the data worker's per-request state once the
                # client lets go of the request.
                self._park_completed(rid)
        return req

    # ----------------------------------------------------------
    # Streaming helper
    # ----------------------------------------------------------

    def _chunks_for_client(
        self, chunks: list[ResultChunk], client_id: str, internal_id: str | None,
    ) -> tuple[list[ResultChunk], bool]:
        """``chunks`` as the client sees them: under its own id, with error
        text made client-facing, and cut after the first error, which ends
        the response. The flag says whether that error was among them."""
        out = []
        for chunk in chunks:
            chunk.request_id = client_id
            out.append(chunk)
            if chunk.modality == "error":
                chunk.data = self._client_facing_error(
                    chunk.data.decode("utf-8", "replace"), internal_id,
                ).encode("utf-8")
                return out, True
        return out, False

    async def iter_result_chunks(self, request_id: str):
        """Yield raw :class:`ResultChunk` objects as they arrive, one at a time.

        Shared source for both output surfaces: ``/generate`` formats each
        chunk as NDJSON (via :meth:`async_stream_results`), while the
        OpenAI-compatible endpoints translate the same chunks into SSE. This
        is a flattening wrapper over :meth:`iter_result_chunk_batches`, which
        hands out the chunks the way they arrive (several per wake under load)
        and owns the timeout, the drain and the abort on an early exit.
        """
        batches = self.iter_result_chunk_batches(request_id)
        try:
            async for batch in batches:
                for chunk in batch:
                    yield chunk
        finally:
            await batches.aclose()

    async def iter_result_chunk_batches(self, request_id: str):
        """Yield the :class:`ResultChunk` objects as they arrive, as one list
        per drain: everything the message thread appended since the last
        wake. Under load several tokens of a request are often waiting when
        the handler runs, and sending them as one body frame costs one write
        instead of one per token; when the loop keeps up the lists are of
        length one and nothing is delayed. The final flush and an in-band
        error come as the last list.
        """
        start = time.time()
        finished = False
        # An error chunk ends the response: whatever follows it for this
        # request (a second error, a late output) is not part of the answer.
        errored = False
        # Woken by the message thread (PendingRequest.notify) when a chunk or
        # the completion lands; the timeout below is only a backstop.
        wake = asyncio.Event()
        with self.request_lock:
            internal_id = self._lookup_client(request_id)
            self._attach_consumer(internal_id)
            req = self.pending_requests.get(internal_id) if internal_id else None
            if req is not None:
                req.loop = asyncio.get_running_loop()
                req.wake = wake
        try:
            while True:
                if time.time() - start > self.timeout_seconds:
                    raise HTTPException(status_code=500, detail="Request timed out")

                # Cleared before the read: a notification that lands after
                # this point stays set and the wait below returns at once.
                wake.clear()
                new_chunks: list[ResultChunk] = []
                done = False
                client_id = request_id
                with self.request_lock:
                    req = self.pending_requests.get(internal_id)
                    if req:
                        if req.client_id is not None:
                            client_id = req.client_id
                        avail = len(req.chunks)
                        consumed = req.consumed_chunks
                        new_chunks = req.chunks[consumed:avail]
                        req.consumed_chunks = avail
                        done = req.event.is_set()
                    else:
                        done = True

                if new_chunks:
                    batch, errored = self._chunks_for_client(new_chunks, client_id, internal_id)
                    if self.enable_nvtx:
                        for _ in batch:
                            profiler.mark("apiserver.chunk_available")
                    yield batch
                    if errored:
                        done = True

                if done:
                    logger.info("Async stream results received finish for %s", request_id)
                    # flush remaining
                    remaining: list[ResultChunk] = []
                    finished_req: PendingRequest | None = None
                    with self.request_lock:
                        req = self.pending_requests.get(internal_id)
                        if req:
                            if not errored:
                                remaining = req.chunks[req.consumed_chunks:]
                            finished_req = self.pending_requests.pop(internal_id, None)
                            if (
                                finished_req is not None
                                and internal_id not in self.recently_completed
                            ):
                                self._forget_mapping(internal_id)
                    # Profiling (incl. the optional file write) runs outside the
                    # lock; the popped request is no longer shared with other threads.
                    if finished_req is not None:
                        self._finalize_profile(finished_req)
                    out_id = (
                        finished_req.client_id
                        if finished_req is not None and finished_req.client_id is not None
                        else client_id
                    )
                    tail, tail_errored = self._chunks_for_client(remaining, out_id, internal_id)
                    errored = errored or tail_errored
                    # A request can fail after the stream is already open
                    # (preprocess error, result-delivery timeout); the HTTP
                    # status is committed by then, so the error must travel
                    # in-band as the final chunk, once.
                    if (
                        not errored
                        and finished_req is not None
                        and finished_req.error is not None
                    ):
                        tail.append(ResultChunk(
                            request_id=out_id,
                            modality="error",
                            data=self._client_facing_error(
                                str(finished_req.error), internal_id,
                            ).encode("utf-8"),
                            metadata={"status": finished_req.error_status},
                        ))
                    if tail:
                        yield tail
                    finished = True
                    break

                if new_chunks:
                    continue  # more may already be waiting
                try:
                    await asyncio.wait_for(wake.wait(), timeout=self._stream_wake_backstop_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            if not finished and internal_id is not None:
                self._abort_internal(internal_id)
            elif internal_id is not None:
                with self.request_lock:
                    req = self.pending_requests.get(internal_id)
                    if req is not None:
                        req.consumer_attached = False

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
            # Same range name as the NDJSON path on purpose, so the two
            # protocols stay directly comparable in one analyzer run.
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
        with self.request_lock:
            internal_id = self._lookup_client(request_id)
            self._attach_consumer(internal_id)
        req = None
        try:
            while True:
                with self.request_lock:
                    held = self.pending_requests.get(internal_id) if internal_id else None
                    done = held.event.is_set() if held else True
                if done:
                    break
                if time.time() - start > self.timeout_seconds:
                    self._abort_internal(internal_id)
                    raise HTTPException(status_code=500, detail="Request timed out")
                if raw_request is not None and await raw_request.is_disconnected():
                    self._abort_internal(internal_id)
                    return []
                await asyncio.sleep(0.005)

            with self.request_lock:
                req = self.pending_requests.pop(internal_id, None) if internal_id else None
                if req is not None:
                    req.consumer_attached = False
                if req is not None and internal_id not in self.recently_completed:
                    self._forget_mapping(internal_id)
        finally:
            # Cancel / disconnect before consume: abort this internal id
            # (idempotent if timeout/disconnect already did). Do not park
            # or reclaim — that would re-arm transport cleanup.
            if internal_id is not None and req is None:
                self._abort_internal(internal_id)
        if req is None:
            return []
        # Profile even on failure so --log-stats still shows the quarantine
        # when every request errors (the #332 condition).
        self._finalize_profile(req)
        if req.error is not None:
            raise HTTPException(
                status_code=req.error_status,
                detail=self._client_facing_error(req.error, internal_id),
            )
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
        req.profile.read_quarantine = self.read_quarantines()

        try:
            pretty_print_profile(req.profile, self.log_stats_file)
        except Exception:
            logger.exception("Failed to emit request profile for %s", req.profile.rid)

    def _park_completed(self, request_id: str) -> None:
        """Keep the internal id until prune or abort. Caller holds request_lock."""
        self.recently_completed[request_id] = time.time()

    def _attach_consumer(self, internal_id: str | None) -> None:
        """Mark the resolved internal id as having a live collect/iter. Caller holds request_lock."""
        if internal_id is None:
            return
        req = self.pending_requests.get(internal_id)
        if req is not None:
            req.consumer_attached = True

    def _client_facing_error(self, text: str, internal_id: str | None) -> str:
        """Rewrite an internal id that leaked into client-visible error text."""
        if not text or not internal_id:
            return text
        client = getattr(self, "_internal_to_client", {}).get(internal_id)
        if client is None:
            req = self.pending_requests.get(internal_id)
            client = getattr(req, "client_id", None) if req is not None else None
        if client is None or client == internal_id:
            return text
        return text.replace(internal_id, client)

    def _lookup_client(self, client_id: str) -> str | None:
        """Client-facing id → this process's latest internal id.

        When the client-id map exists — even if it is empty — only that map
        is consulted, so a parked or leftover internal id is not a public
        handle. The pending/parked fallback is only for test fixtures that
        never initialized the maps.
        """
        if hasattr(self, "_client_to_internal"):
            return self._client_to_internal.get(client_id)
        if client_id in self.pending_requests or client_id in self.recently_completed:
            return client_id
        return None

    def _forget_mapping(self, internal_id: str) -> None:
        mapping = getattr(self, "_internal_to_client", None)
        if not mapping:
            return
        client_id = mapping.pop(internal_id, None)
        client_map = getattr(self, "_client_to_internal", None)
        if (
            client_id is not None
            and client_map is not None
            and client_map.get(client_id) == internal_id
        ):
            client_map.pop(client_id, None)

    def _drop_live(self, internal_id: str) -> bool:
        """Pop pending/parked state. Caller holds ``request_lock``."""
        active = (
            internal_id in self.pending_requests
            or internal_id in self.recently_completed
        )
        req = self.pending_requests.pop(internal_id, None)
        self.recently_completed.pop(internal_id, None)
        if req is not None:
            # A stream still attached to it (superseded or aborted elsewhere)
            # ends on its next read instead of at the wait backstop.
            req.notify()
        if active:
            self._forget_mapping(internal_id)
        return active

    def abort_request(self, request_id: str) -> None:
        """Abort the latest live submission of this client id."""
        with self.request_lock:
            internal_id = self._lookup_client(request_id)
            active = internal_id is not None and self._drop_live(internal_id)
            label = request_id if internal_id is None else internal_id
        if not active:
            return
        logger.info("Client cancelled request %s; releasing resources", label)
        self.preprocess_worker.abort_request(internal_id)

    def _abort_internal(self, internal_id: str) -> None:
        """Abort a captured internal id (this process's own submit)."""
        with self.request_lock:
            if not self._drop_live(internal_id):
                return
        self.preprocess_worker.abort_request(internal_id)

    def release_request(self, request_id: str) -> None:
        """Drop a request whose results will never be read: abort it if it is
        still running, else just forget it (an ended request needs no abort)."""
        with self.request_lock:
            internal_id = self._lookup_client(request_id)
            req = self.pending_requests.get(internal_id) if internal_id else None
            finished = req is not None and req.event.is_set()
            if finished and internal_id is not None:
                self.pending_requests.pop(internal_id)
                if internal_id not in self.recently_completed:
                    self._forget_mapping(internal_id)
        if not finished:
            self.abort_request(request_id)

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    def cleanup(self) -> None:
        self.preprocess_worker.shutdown()
        self.running = False
        if hasattr(self, "_msg_thread") and self._msg_thread.is_alive():
            self._msg_thread.join(timeout=2)


def _resolve_warmup_path(path: str) -> str:
    """A warmup file is a local path, or ``hf://<owner>/<repo>/<path>`` for a
    file shipped with a model (its example inputs), taken from the HF cache."""
    if not path.startswith("hf://"):
        return path
    parts = path[len("hf://"):].split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"hf:// path needs <owner>/<repo>/<path>, got {path!r}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=parts[2])


@dataclass
class WarmupMedia:
    """One media input of a warmup request: a local or ``hf://`` path and its
    modality (from the extension unless the yaml names it)."""

    path: str
    modality: str

    @classmethod
    def from_config(cls, entry: Any) -> "WarmupMedia":
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not entry.get("path"):
            raise ValueError(f"a warmup file is a path or a mapping with 'path', got {entry!r}")
        path = str(entry["path"])
        modality = entry.get("modality") or _detect_modality(path)
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"cannot determine modality for warmup file {path!r}; set 'modality'"
            )
        return cls(path=path, modality=modality)


@dataclass
class WarmupRequest:
    """One entry of the deployment's ``warmup_requests``: a ``/generate``
    request in yaml form. ``files`` are the media inputs in prompt order
    (any modality, any number), followed by ``text`` as on ``/generate``."""

    text: str | None = None
    files: list[WarmupMedia] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    model_kwargs: dict | None = None

    @classmethod
    def from_config(cls, spec: Any) -> "WarmupRequest":
        if not isinstance(spec, dict):
            raise ValueError("not a mapping")
        unknown = set(spec) - {"text", "files", "output_modalities", "model_kwargs"}
        if unknown:
            raise ValueError(f"unknown key(s) {sorted(unknown)}")
        out_mods = spec.get("output_modalities") or ["text"]
        if isinstance(out_mods, str):
            out_mods = [m.strip() for m in out_mods.split(",") if m.strip()]
        files = spec.get("files") or []
        if not isinstance(files, list):
            files = [files]
        model_kwargs = spec.get("model_kwargs") or None
        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise ValueError("model_kwargs must be a mapping")
        return cls(
            text=spec.get("text") or None,
            files=[WarmupMedia.from_config(f) for f in files],
            output_modalities=list(out_mods),
            model_kwargs=model_kwargs,
        )

    def submit(self, server: APIServer, request_id: str) -> str:
        """Submit to ``server`` the way ``/generate`` would, ``hf://`` files
        resolved from the cache."""
        file_paths: dict[str, list[str]] = {}
        parts: list[PromptPart] = []
        for media in self.files:
            paths = file_paths.setdefault(media.modality, [])
            parts.append(PromptPart(modality=media.modality, index=len(paths)))
            paths.append(_resolve_warmup_path(media.path))
        if self.text:
            parts.append(PromptPart(modality="text", text=self.text))
        return server.submit_request(
            text=self.text,
            file_paths=file_paths or None,
            input_modalities=[p.modality for p in parts],
            output_modalities=self.output_modalities,
            model_kwargs=self.model_kwargs,
            prompt_parts=parts or None,
            streaming=False,
            request_id=request_id,
        )


def _run_warmup_requests(server: APIServer, specs: list) -> None:
    """Run the deployment's ``warmup_requests`` once the workers are ready and
    before the server binds, so the first client does not pay what a first
    request of a shape costs (torch.compile of a denoise step, a cold
    inductor cache). Each spec is parsed into a :class:`WarmupRequest`. A
    warmup that is malformed or fails is logged and skipped; it must not keep
    the server from coming up.
    """
    if not isinstance(specs, list):
        logger.warning("warmup_requests must be a list; ignoring %r", type(specs).__name__)
        return

    async def _one(request: WarmupRequest, request_id: str) -> None:
        await server.collect_results(request.submit(server, request_id))

    for index, spec in enumerate(specs):
        try:
            request = WarmupRequest.from_config(spec)
        except ValueError as exc:
            logger.warning("warmup_requests[%d] skipped: %s", index, exc)
            continue
        t0 = time.perf_counter()
        try:
            asyncio.run(_one(request, f"warmup-{index}"))
        except Exception as exc:  # noqa: BLE001 — a cold server is better than none
            logger.warning(
                "warmup request %d/%d failed after %.1f s: %s",
                index + 1, len(specs), time.perf_counter() - t0,
                getattr(exc, "detail", exc),
            )
            continue
        logger.info(
            "warmup request %d/%d (%s) done in %.1f s",
            index + 1, len(specs), ",".join(request.output_modalities),
            time.perf_counter() - t0,
        )


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


def _decode_ws_files(
    files: list | None, upload_dir: Path,
) -> tuple[dict[str, list[str]], list[PromptPart]]:
    """Persist a websocket message's media (``{"name", "data"}`` entries;
    ``data`` raw bytes in a msgpack frame, base64 text in a JSON frame) under
    the upload dir, grouped by the modality of each file name, the way the
    multipart ``/generate`` route does. A message rejected halfway leaves no
    files behind."""
    if files is not None and not isinstance(files, list):
        raise ValueError("files must be a list of {name, data} objects")
    file_paths: dict[str, list[str]] = {}
    parts: list[PromptPart] = []
    saved: list[Path] = []
    try:
        for entry in files or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError("each file needs a string name")
            name = entry["name"]
            modality = _detect_modality(name)
            if modality == "unknown":
                raise ValueError(f"Cannot determine modality for file: {name}")
            data = entry.get("data")
            if isinstance(data, str):
                # the decoder the data-URL paths use: unpadded or wrapped
                # base64 is fine, a bad alphabet is the client's error
                data = media_io._decode_base64_payload(data)
            if not isinstance(data, (bytes, bytearray)):
                raise ValueError(f"file {name!r} carries no data")
            base = os.path.basename(name) or "upload"
            save_path = upload_dir / f"{uuid.uuid4()}_{base}"
            save_path.write_bytes(bytes(data))
            saved.append(save_path)
            paths = file_paths.setdefault(modality, [])
            parts.append(PromptPart(modality=modality, index=len(paths)))
            paths.append(str(save_path))
    except Exception:
        for path in saved:
            path.unlink(missing_ok=True)
        raise
    return file_paths, parts


def _ws_modalities(value, key: str) -> list[str] | None:
    """A modality list as a message may spell it: a comma-separated string or
    a list of strings. Anything else is rejected by name."""
    if value is None:
        return None
    if isinstance(value, str):
        return [m.strip() for m in value.split(",") if m.strip()]
    if isinstance(value, list) and all(isinstance(m, str) for m in value):
        return list(value)
    raise ValueError(f"{key} must be a comma-separated string or a list of strings")


def _schedule_upload_cleanup(file_paths: dict[str, list[str]], delay_s: float = 60.0) -> None:
    """Delete a request's uploaded files once the preprocess worker has had
    time to read them. Shared by the HTTP and WebSocket routes."""
    def _cleanup(paths: dict[str, list[str]]) -> None:
        time.sleep(delay_s)
        for ps in paths.values():
            for p in ps:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
    threading.Thread(target=_cleanup, args=(file_paths,), daemon=True).start()


def _ws_input_layout(
    text: str | None, input_modalities: list[str] | None, parts: list[PromptPart],
) -> tuple[list[str], list[PromptPart]]:
    """Resolve a websocket message's input layout the way ``/generate`` does:
    an explicit list is the layout (a text prompt keeps its slot), else the
    order the files and text arrived in."""
    if text:
        parts = [*parts, PromptPart(modality="text", text=text)]
    if input_modalities is not None:
        in_mods = list(input_modalities)
        if text and "text" not in in_mods:
            in_mods.append("text")
        if not text:
            in_mods = [m for m in in_mods if m != "text"]
        return in_mods, []
    return [p.modality for p in parts], parts


@app.websocket("/generate/ws")
async def generate_ws(websocket: WebSocket):
    """``/generate`` over one persistent WebSocket, for control loops.

    Each incoming message is one request with the ``/generate`` fields —
    ``text``, ``files`` (``[{"name", "data"}]``), ``input_modalities``,
    ``output_modalities``, ``model_kwargs``, ``request_id`` — either a JSON
    text frame (``data`` base64) or a msgpack binary frame (``data`` raw
    bytes). Every result chunk comes back as a frame of the same encoding,
    ``{"request_id", "modality", "data", "metadata"}``, followed by
    ``{"request_id", "finish": true}``. A rejected message answers
    ``{"request_id", "error": ...}``, and so does a request that fails after
    it was accepted (with the HTTP ``status`` it would have had); no finish
    follows an error. Messages may be pipelined: a client can
    send the next observation before the previous action chunk has returned,
    and the ``request_id`` tells the replies apart. Closing the socket aborts
    whatever is still in flight.
    """
    if api_server is None:
        await websocket.close(code=1013, reason="Server not ready")
        return
    await websocket.accept()
    tasks: set[asyncio.Task] = set()
    # Pipelined requests reply from separate tasks; one frame at a time.
    send_lock = asyncio.Lock()

    async def send(payload: dict, binary: bool) -> None:
        async with send_lock:
            if binary:
                await websocket.send_bytes(msgpack.packb(payload, use_bin_type=True))
            else:
                if isinstance(payload.get("data"), (bytes, bytearray)):
                    payload = {**payload, "data": base64.b64encode(payload["data"]).decode("ascii")}
                await websocket.send_text(json.dumps(payload))

    async def serve_one(message: dict, binary: bool) -> None:
        request_id = message.get("request_id")
        file_paths: dict[str, list[str]] | None = None
        try:
            if request_id is not None and not isinstance(request_id, str):
                raise ValueError("request_id must be a string")
            out_mods = _ws_modalities(message.get("output_modalities"), "output_modalities") or ["text"]
            text = message.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError("text must be a string")
            if not text and not message.get("files"):
                raise ValueError("message carries neither text nor files")
            file_paths, parts = await run_in_threadpool(
                _decode_ws_files, message.get("files"), api_server.upload_dir,
            )
            in_mods, parts = _ws_input_layout(
                text, _ws_modalities(message.get("input_modalities"), "input_modalities"), parts,
            )
            model_kwargs = message.get("model_kwargs")
            if isinstance(model_kwargs, str):
                model_kwargs = json.loads(model_kwargs)
            if model_kwargs is not None and not isinstance(model_kwargs, dict):
                raise ValueError("model_kwargs must be a JSON object")
            request_id = api_server.submit_request(
                text=text,
                file_paths=file_paths or None,
                input_modalities=in_mods,
                output_modalities=out_mods,
                model_kwargs=model_kwargs,
                prompt_parts=parts or None,
                streaming=True,
                request_id=request_id,
            )
            # Cancelling this task (socket closed mid-stream) tears the
            # iterator down, and its ``finally`` aborts the engine request.
            failed = False
            async for chunk in api_server.iter_result_chunks(request_id):
                if chunk.modality == "error":
                    # The request failed after it was accepted. This is the
                    # iterator's last chunk, so the loop ends here; the error
                    # is the reply and no finish follows.
                    failed = True
                    await send({
                        "request_id": request_id,
                        "error": bytes(chunk.data).decode("utf-8", "replace"),
                        "status": chunk.metadata.get("status", 500),
                    }, binary)
                    continue
                await send({
                    "request_id": request_id, "modality": chunk.modality,
                    "data": bytes(chunk.data), "metadata": chunk.metadata,
                }, binary)
            if not failed:
                await send({"request_id": request_id, "finish": True}, binary)
        except (WebSocketDisconnect, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 — reported in-band, the socket stays up
            logger.exception("generate/ws request failed")
            try:
                await send({"request_id": request_id, "error": str(exc)}, binary)
            except Exception:  # noqa: BLE001
                pass
        finally:
            if file_paths:
                _schedule_upload_cleanup(file_paths)

    try:
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                break
            binary = frame.get("bytes") is not None
            try:
                if binary:
                    message = msgpack.unpackb(frame["bytes"], raw=False)
                else:
                    message = json.loads(frame.get("text") or "")
            except Exception as exc:  # noqa: BLE001 — a bad frame is reported, not fatal
                await send({"request_id": None, "error": f"undecodable frame: {exc}"}, binary)
                continue
            if not isinstance(message, dict):
                await send({"request_id": None, "error": "message must be an object"}, binary)
                continue
            task = asyncio.create_task(serve_one(message, binary))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        for task in list(tasks):
            task.cancel()


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
            _schedule_upload_cleanup(file_paths)


@app.get("/health")
async def health_check():
    # Report unhealthy while shutting down, so a load balancer stops sending
    # here, and past the failed-read buffer budget, so a liveness probe
    # restarts a deployment whose transport keeps failing.
    problem = api_server.health_problem() if api_server is not None else None
    if problem is not None:
        raise HTTPException(status_code=503, detail=problem)
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    if api_server is not None:
        api_server.cleanup()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _set_preprocess_threads(model, config: dict) -> None:
    """Size torch's intra-op pool for this process's media preprocessing.

    ``config.yaml``'s ``preprocess_torch_threads`` wins, else the model's
    ``PREPROCESS_TORCH_THREADS``; None keeps torch's default. Affects only this
    process: the conductor and workers are spawned fresh.
    """
    threads = config.get("preprocess_torch_threads", model.PREPROCESS_TORCH_THREADS)
    if threads is None:
        return
    import torch

    torch.set_num_threads(int(threads))


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
        help="Enable NVTX markers during CUDA execution (disabled on XPU and CPU)",
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
        cache_dir=args.cache_dir,
        **{**model_init_kwargs(model_name), **yaml_model_kwargs},
    )
    _set_preprocess_threads(model, config)

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
        # The deployment's warmup requests run here, before the bind: a
        # client never sees the first request of a shape.
        warmups = config.get("warmup_requests") or []
        if warmups:
            logger.info("Running %d warmup request(s) before binding", len(warmups))
            _run_warmup_requests(api_server, warmups)
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
