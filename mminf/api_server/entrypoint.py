"""FastAPI server entry point for multimodal inference requests."""

import asyncio
import base64
import collections
import json
import logging
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from mminf.api_server.data_worker import PreprocessWorker
from mminf.api_server.request_types import APIServerMessage, PreprocessInput, ResultChunk
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.model.registry import HF_MODELS

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
    from mminf.conductor.conductor import Conductor
    from mminf.model.registry import get_model_class

    # Read yaml early to extract optional `model_kwargs:` section for the model
    # constructor. Lets a yaml override init-time model parameters (e.g.
    # Pi05's action_horizon for the DROID benchmark variant) without code
    # changes per model. Backward compatible: missing section → empty dict,
    # so existing configs see identical behavior.
    import yaml as _yaml
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
# Mooncake KV store setup
# ------------------------------------------------------------------
def start_mooncake_master(port=8080, log_file: str | None = None):
    cmd = [
        "mooncake_master",
        "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        f"--http_metadata_server_port={port}",
    ]

    stdout = open(log_file, "a") if log_file else subprocess.DEVNULL
    stderr = subprocess.STDOUT

    process = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        process_group=os.setsid,  # start new process group
    )

    wait_for_port("localhost", 50051)
    logger.info("Successfully started Mooncake metadata server")

    return process


def wait_for_port(host, port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timeout waiting for {host}:{port}")


def stop_mooncake_master(process: subprocess.Popen):
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# ------------------------------------------------------------------
# APIServer
# ------------------------------------------------------------------

class APIServer:
    """Accept multimodal requests, forward to conductor, collect results."""

    def __init__(
        self,
        socket_path_prefix: str = "/tmp/mminf",
        upload_dir: str = "/tmp/mminf_uploads",
        hostname: str="localhost",
        timeout_seconds: float = 600.0,
        tensor_comm_protocol=CommProtocol.RDMA,
        tcp_transfer_device="",
        model=None,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

        self.preprocess_worker = PreprocessWorker(
            model=model,
            hostname=hostname,
            socket_path_prefix=socket_path_prefix,
            tensor_comm_protocol=tensor_comm_protocol,
            tcp_transfer_device=tcp_transfer_device
        )

        # Concurrent request tracking
        self.pending_requests: dict[str, dict] = {}
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

        # Background thread that drains results from conductor
        self._msg_thread = threading.Thread(
            target=self._process_messages, daemon=True
        )
        self._msg_thread.start()

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
            self.pending_requests[request_id] = {
                "chunks": [],           # list[ResultChunk]
                "event": threading.Event(),
                "streaming": streaming,
                "consumed_chunks": 0,
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "final_forward_outputs": {},
            }

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
            if not self.preprocess_worker.has_pending_tensors(rid)
            or (now - ts) >= self._recently_completed_ttl
        ]
        for rid in stale:
            # only set the event when there are no more pending chunks
            self.pending_requests[rid]["event"].set()
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
                                self.pending_requests[rid]["final_forward_outputs"] = \
                                    message.body.final_forward_outputs
                        elif rid in self.recently_completed:
                            logger.debug("Late message for completed %s", rid)
                        else:
                            logger.warning(
                                "Message for unknown request %s", rid
                            )
                for result_chunk in self.preprocess_worker.get_result_chunks():
                    logger.debug(
                        "Got result chunk of %s modality for request %s",
                        result_chunk.modality, result_chunk.request_id
                    )
                    rid = result_chunk.request_id
                    with self.request_lock:
                        self.pending_requests[rid]["chunks"].append(
                            result_chunk
                        )
            except Exception:
                if self.running:
                    logger.exception("Error in message processing loop")
                    time.sleep(0.01)
            time.sleep(0.001)

    # ----------------------------------------------------------
    # Streaming helper
    # ----------------------------------------------------------

    async def async_stream_results(self, request_id: str):
        """Yield NDJSON lines as result chunks arrive."""
        start = time.time()
        while True:
            if time.time() - start > self.timeout_seconds:
                with self.request_lock:
                    self.pending_requests.pop(request_id, None)
                raise HTTPException(status_code=500, detail="Request timed out")

            new_chunks: list[ResultChunk] = []
            done = False
            with self.request_lock:
                req = self.pending_requests.get(request_id)
                if req:
                    avail = len(req["chunks"])
                    consumed = req["consumed_chunks"]
                    new_chunks = req["chunks"][consumed:avail]
                    req["consumed_chunks"] = avail
                    done = req["event"].is_set()
                else:
                    done = True

            for chunk in new_chunks:
                yield self._chunk_to_ndjson(chunk)

            if done:
                logger.info("Async stream results received finish for %s", request_id)
                # flush remaining
                remaining: list[ResultChunk] = []
                with self.request_lock:
                    req = self.pending_requests.get(request_id)
                    if req:
                        remaining = req["chunks"][req["consumed_chunks"]:]
                        self.pending_requests.pop(request_id, None)
                for chunk in remaining:
                    yield self._chunk_to_ndjson(chunk)
                break

            await asyncio.sleep(0.001)

    @staticmethod
    def _chunk_to_ndjson(chunk: ResultChunk) -> str:
        return json.dumps({
            "modality": chunk.modality,
            "data": base64.b64encode(chunk.data).decode("ascii"),
            "metadata": chunk.metadata,
        }) + "\n"

    # ----------------------------------------------------------
    # Blocking helper (non-streaming)
    # ----------------------------------------------------------

    def collect_results(self, request_id: str) -> list[ResultChunk]:
        """Block until the request completes, then return all chunks."""
        with self.request_lock:
            req = self.pending_requests.get(request_id)
            if not req:
                raise HTTPException(
                    status_code=404, detail=f"Request {request_id} not found"
                )
            event = req["event"]

        if not event.wait(timeout=self.timeout_seconds):
            with self.request_lock:
                self.pending_requests.pop(request_id, None)
            raise HTTPException(status_code=500, detail="Request timed out")

        with self.request_lock:
            chunks = self.pending_requests[request_id]["chunks"][:]
            self.pending_requests.pop(request_id, None)
        return chunks

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    def cleanup(self) -> None:
        # stop_mooncake_master(self.mooncake_pid)
        self.preprocess_worker.shutdown()
        self.running = False
        if hasattr(self, "_msg_thread") and self._msg_thread.is_alive():
            self._msg_thread.join(timeout=2)

# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="mminf API",
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


@app.post("/generate")
async def generate(
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

        chunks = await run_in_threadpool(
            api_server.collect_results, request_id
        )
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

def main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="mminf — launch API server and conductor from a config file"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mooncake-port", type=int, default=8080)
    parser.add_argument(
        "--socket-path-prefix", type=str, default="/tmp/mminf",
        help="ZMQ IPC socket path prefix (shared with conductor/workers)",
    )
    parser.add_argument(
        "--upload-dir", type=str, default="/tmp/mminf_uploads",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [api_server] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", "dummy")
    # Forward yaml-level model_kwargs to the API-server-side lightweight
    # model instance too, so it sees the same Pi05Config (action_horizon, etc.)
    # as the conductor-side instance. Without this they could diverge.
    yaml_model_kwargs = config.get("model_kwargs", {}) or {}

     # Create a lightweight model instance for prompt processing
    # (tokenization only — no GPU weights needed)
    from mminf.model.registry import get_model_class
    model = get_model_class(model_name)(
        model_path_hf=HF_MODELS.get(model_name, {}).get("model_path_hf", ""),
        cache_dir=args.cache_dir,
        **yaml_model_kwargs,
    )

    global api_server
    api_server = APIServer(
        socket_path_prefix=args.socket_path_prefix,
        upload_dir=args.upload_dir,
        timeout_seconds=args.timeout,
        tensor_comm_protocol=CommProtocol(args.tensor_comm_protocol),
        model=model,
        tcp_transfer_device=args.tcp_transfer_device
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
            args.log_level,
            args.cache_dir,
            CommProtocol(args.tensor_comm_protocol),
            args.tcp_transfer_device
        ),
    )
    conductor_proc.start()
    logger.info("Conductor process started (pid=%d, model=%s)", conductor_proc.pid, model_name)

    try:
        logger.info("Starting mminf API server on %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    except KeyboardInterrupt:
        pass
    finally:
        if api_server is not None:
            api_server.cleanup()
        _shutdown_conductor_process(conductor_proc)


if __name__ == "__main__":
    main()
