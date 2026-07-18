"""The Rust axum frontend in front of ``APIServer``.

``mstar-server`` (``rust/server/``) owns the HTTP surface — routes, OpenAI
translation, SSE/NDJSON streaming, uploads, CORS — and speaks a flattened
msgpack protocol over the Rust ZMQ transport to this bridge, which
drives the existing Python data plane (``APIServer.submit_request`` /
``iter_result_chunks`` / ``abort_request``; preprocessing and the conductor
protocol are untouched):

    frontend -> bridge:  {t:"submit", rid, text, file_paths,
                          input_modalities, output_modalities, model_kwargs,
                          streaming} | {t:"abort", rid}
    bridge -> frontend:  {t:"chunk", rid, modality, data(bin), metadata}
                       | {t:"err", rid, msg} | {t:"done", rid}

The submit shape is ``APIServer.submit_request``'s signature verbatim — the
protocol was designed by flattening it. The bridge mesh lives in its own
socket dir (private to the frontend/bridge pair), so entity names never
collide with the conductor/worker mesh.

Opt-in: ``mstar-serve --rust-frontend`` replaces ``uvicorn.run`` with the
``mstar-server`` binary + this loop. Everything else in the process
(preprocess worker, conductor spawn, tensor transport) is identical.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

from mstar.communication.communicator import CommProtocol
from mstar.communication.rust_communicator import (
    MsgpackCodec,
    RustZMQCommunicator,
)

logger = logging.getLogger(__name__)

#: Upper bound on one blocking receive, so ``stop()`` is noticed promptly and
#: the loop stays responsive to shutdown. Not a latency knob: an arriving
#: message ends the wait immediately.
BRIDGE_RECV_WAIT_S = 0.05





def find_server_binary(explicit: str | None = None) -> str:
    """Resolve the ``mstar-server`` binary: explicit arg, ``MSTAR_SERVER_BIN``,
    ``$PATH``, then the in-repo release build."""
    for candidate in (
        explicit,
        os.getenv("MSTAR_SERVER_BIN"),
        shutil.which("mstar-server"),
        str(Path(__file__).resolve().parents[2]
            / "rust" / "server" / "target" / "release" / "mstar-server"),
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "mstar-server binary not found: build it with "
        "`cargo build --release` in rust/server/, put it on $PATH, or set "
        "MSTAR_SERVER_BIN")


def launch_rust_server(binary: str, model_name: str, port: int,
                       bridge_dir: str, upload_dir: str) -> subprocess.Popen:
    proc = subprocess.Popen([binary, model_name, str(port), bridge_dir,
                             upload_dir])
    logger.info("Rust frontend started (pid=%d, port=%d)", proc.pid, port)
    return proc


class RustFrontendBridge:
    """The serve loop between the Rust frontend and ``APIServer``."""

    def __init__(self, server, bridge_dir: str):
        self.server = server
        # Rust transport with the msgpack codec: we bind as "conductor" in
        # the PRIVATE bridge dir (that is who the frontend's bridge sends to).
        self.comm = RustZMQCommunicator(
            "conductor", push_ids=["frontend"], protocol=CommProtocol.IPC,
            ipc_socket_path_prefix=bridge_dir, codec=MsgpackCodec)
        self.running = True

    # -- outbound ----------------------------------------------------------

    def _send(self, msg: dict) -> None:
        self.comm.send("frontend", msg)

    def _err(self, rid: str, detail: str) -> None:
        self._send({"t": "err", "rid": rid, "msg": detail})

    # -- request lifecycle ---------------------------------------------------

    def _submit(self, msg: dict) -> None:
        rid = msg["rid"]
        if msg.get("tokens") is not None:
            # The Python preprocess worker owns tokenization; run the
            # frontend without MSTAR_TOKENIZER so it sends text.
            self._err(rid, "pre-tokenized ingest is not supported by this "
                           "backend (unset MSTAR_TOKENIZER)")
            return
        try:
            self.server.submit_request(
                text=msg.get("text"),
                file_paths=msg.get("file_paths") or None,
                input_modalities=list(msg.get("input_modalities") or []),
                output_modalities=list(msg.get("output_modalities") or ["text"]),
                model_kwargs=dict(msg.get("model_kwargs") or {}),
                streaming=bool(msg.get("streaming", True)),
                request_id=rid,
            )
        except Exception as e:  # noqa: BLE001 — one bad request must not kill the loop
            logger.warning("submit %s failed: %r", rid, e)
            self._err(rid, repr(e))
            return
        asyncio.get_running_loop().create_task(self._relay(rid))

    async def _relay(self, rid: str) -> None:
        """Forward this request's ResultChunks; terminal err on failure."""
        try:
            async for chunk in self.server.iter_result_chunks(rid):
                self._send({
                    "t": "chunk", "rid": rid, "modality": chunk.modality,
                    "data": bytes(chunk.data), "metadata": chunk.metadata,
                })
            self._send({"t": "done", "rid": rid})
        except Exception as e:  # noqa: BLE001 — includes the per-request timeout
            detail = getattr(e, "detail", None) or repr(e)
            logger.warning("request %s failed: %s", rid, detail)
            self._err(rid, str(detail))

    # -- loop ----------------------------------------------------------------

    async def _serve(self) -> None:
        logger.info("Rust-frontend bridge serving")
        loop = asyncio.get_running_loop()
        # Blocking receive on an executor thread: the wait happens inside the
        # transport (GIL released), so an arriving submit is picked up
        # immediately instead of on the next tick of a sleep/poll loop — and
        # an idle bridge costs no wakeups. The event loop itself stays free
        # to run the per-request relay tasks; `send` is safe from those
        # concurrently (the transport serializes its push sockets internally).
        recv = lambda: self.comm.get_all_new_messages(  # noqa: E731
            blocking=True, timeout_s=BRIDGE_RECV_WAIT_S)
        while self.running:
            for msg in await loop.run_in_executor(None, recv):
                t = msg.get("t")
                if t == "submit":
                    self._submit(msg)
                elif t == "abort":
                    self.server.abort_request(msg["rid"])
                elif t == "ping":
                    # Deep-health liveness: the frontend's /health goes red
                    # unless this loop answers.
                    self._send({"t": "pong", "rid": msg["rid"]})
                else:
                    logger.warning("bridge: unknown message type %r", t)

    def run(self) -> None:
        """Blocking serve loop (the ``uvicorn.run`` replacement)."""
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        self.running = False
