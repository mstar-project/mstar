"""FastAPI server entry point for multimodal inference requests."""

import asyncio
import base64
import collections
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from mstar.api_server.data_worker import PreprocessWorker
from mstar.api_server.request_types import APIServerMessage, PreprocessInput, ResultChunk
from mstar.communication.communicator import CommProtocol, ZMQCommunicator
from mstar.model.registry import HF_MODELS
from mstar.profile.display import pretty_print_profile
from mstar.profile.format import OutputInfo, RequestProfile, RequestTiming
from mstar.utils.logging_config import quiet_noisy_loggers

logger = logging.getLogger(__name__)

SUPPORTED_MODALITIES = frozenset({"text", "image", "audio", "video", "action", "scalar", "tensor"})

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
    finally:
        conductor.shutdown()


def _shutdown_conductor_process(
    conductor_proc: mp.Process,
    timeout: float = 5.0,
) -> None:
    if not conductor_proc.is_alive():
        return

    try:
        conductor_proc.send_signal(signal.SIGINT)
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
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

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
            hostname=hostname,
            socket_path_prefix=socket_path_prefix,
            tensor_comm_protocol=tensor_comm_protocol,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=self.log_stats
        )

        # Concurrent request tracking
        self.pending_requests: dict[str, PendingRequest] = {}
        self.recently_completed: collections.OrderedDict[str, float] = (
            collections.OrderedDict()
        )
        self._recently_completed_ttl = 15.0
        self.request_lock = threading.Lock()
        self.running = True

        # ZMQ channel shared with conductor / workers
        self.communicator = ZMQCommunicator(
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
        serving once it can actually handle requests.
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
            time.sleep(0.01)

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

        # Register pending request
        with self.request_lock:
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
            model_kwargs=model_kwargs
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
        stale = [
            rid
            for rid, ts in self.recently_completed.items()
            if (
                (not self.preprocess_worker.has_pending_tensors(rid)) \
                    and self.preprocess_worker.received_final_chunks(
                        rid, self.pending_requests[rid].final_outputs
                    )
            ) or (now - ts) >= self._recently_completed_ttl
        ]
        for rid in stale:
            # only set the event when there are no more pending chunks
            self.pending_requests[rid].event.set()
            # Snapshot the data worker's tx/rx now: the request is done (all
            # final chunks received), so the worker thread is no longer mutating
            # this rid's transport state, and we must read it before
            # cleanup_request drops it. Extends so it combines with the
            # conductor's worker-side transfers (set in the request_complete
            # handler).
            if self.log_stats:
                profile = self.pending_requests[rid].profile
                profile.tx_info.extend(self.preprocess_worker.get_tx_info(rid))
                profile.rx_info.extend(self.preprocess_worker.get_rx_info(rid))
            self.preprocess_worker.cleanup_request(rid)
            self.recently_completed.pop(rid, None)

    def _process_messages(self) -> None:
        """Drain the ZMQ pull socket and route results to pending requests."""
        while self.running:
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
                            elif message.message_type == "request_complete":
                                logger.info("API server received %s done", rid)
                                self.recently_completed[rid] = time.time()
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
                            logger.debug("Late message for completed %s: %s", rid, message.message_type)
                            if message.message_type == "result_tensors":
                                self.preprocess_worker.discard_result_tensors(message.body)
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
                        req = self.pending_requests[rid]
                        now = time.perf_counter()
                        if req.profile.timing.first_chunk_time is None:
                            req.profile.timing.first_chunk_time = now
                        req.profile.timing.last_chunk_time = now
                        req.chunks.append(result_chunk)
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
                    finished = True
                    break

                await asyncio.sleep(0.001)
        finally:
            if not finished:
                self.abort_request(request_id)

    async def async_stream_results(self, request_id: str):
        """Yield NDJSON lines as result chunks arrive (``/generate`` format)."""
        async for chunk in self.iter_result_chunks(request_id):
            yield self._chunk_to_ndjson(chunk)

    @staticmethod
    def _chunk_to_ndjson(chunk: ResultChunk) -> str:
        return json.dumps({
            "modality": chunk.modality,
            "data": base64.b64encode(chunk.data).decode("ascii"),
            "metadata": chunk.metadata,
        }) + "\n"

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

app = FastAPI(
    title="mstar API",
    description="Multimodal Inference API",
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

    # --- save uploaded files, grouped by modality ----------------
    file_paths: dict[str, list[str]] = {}
    if files:
        for f in files:
            modality = _detect_modality(f.filename or "")
            if modality == "unknown":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot determine modality for file: {f.filename}",
                )
            save_name = f"{uuid.uuid4()}_{f.filename}"
            save_path = api_server.upload_dir / save_name
            content = await f.read()
            await run_in_threadpool(save_path.write_bytes, content)
            file_paths.setdefault(modality, []).append(str(save_path))

    # --- resolve input modalities --------------------------------
    if input_modalities is not None:
        in_mods = [m.strip() for m in input_modalities.split(",") if m.strip()]
    else:
        in_mods: list[str] = []
        in_mods.extend(file_paths.keys())
        if text:
            in_mods.append("text")

    parsed_kwargs = json.loads(model_kwargs) if model_kwargs else None

    try:
        request_id = api_server.submit_request(
            text=text,
            file_paths=file_paths or None,
            input_modalities=in_mods,
            output_modalities=out_mods,
            model_kwargs=parsed_kwargs,
            streaming=streaming,
            request_id=request_id,
        )

        if streaming:
            return StreamingResponse(
                api_server.async_stream_results(request_id),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache"},
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
        tcp_transfer_device=args.tcp_transfer_device,
        log_stats=log_stats,
        log_stats_file=args.log_stats_file,
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

    try:
        # Block until all workers have finished setup, so the server only binds
        # (and logs "Starting…") once it can actually serve requests.
        api_server.finalize_setup()
        logger.info("Starting mstar API server on %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    except KeyboardInterrupt:
        pass
    finally:
        if api_server is not None:
            api_server.cleanup()
        _shutdown_conductor_process(conductor_proc)


if __name__ == "__main__":
    main()
