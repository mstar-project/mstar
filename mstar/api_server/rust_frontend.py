"""The Rust axum frontend in front of ``APIServer``.

``mstar-server`` (``rust/server/``) owns the HTTP surface — routes, OpenAI
translation, SSE/NDJSON streaming, uploads, CORS — and speaks a flattened
msgpack protocol over the Rust ZMQ transport to this bridge, which
drives the existing Python data plane (``APIServer.submit_request`` /
``iter_result_chunks`` / ``abort_submission``; preprocessing and the conductor
protocol are untouched):

    frontend -> bridge:  {t:"submit", rid, sub, text, file_paths,
                          input_modalities, output_modalities, model_kwargs,
                          prompt_parts, streaming} | {t:"abort", rid, sub}
                       | {t:"ping", rid, sub}
                       | {t:"metrics", rid, sub, refusals:{route, status, count}}
    bridge -> frontend:  {t:"chunk", rid, sub, modality, data(bin), metadata}
                       | {t:"err", rid, sub, msg} | {t:"done", rid, sub}
                       | {t:"pong", rid, sub}
                       | {t:"metrics", rid, sub, data(bin), metadata}

A ``metrics`` request carries the frontend's cumulative refusal counts as
three parallel lists (route template, status, count), which the API server
exports as ``mstar_http_refusals_total``. The reply is what ``GET /metrics``
serves on the Python frontend: the text rendered from ``APIServer.metrics``
with its ``content_type`` in ``metadata``, or empty ``metadata`` when metrics
are off (a 404; the counts are then ignored).

``sub`` names one submission and is never reused. A client may resubmit its
``rid`` as soon as it drops a stream, and that stream's ``done`` or late
chunks can cross the retry, so everything about a submission is addressed by
``sub``. The frontend refuses a ``rid`` still in flight with a 409 itself and
sends a dropped stream's abort before it frees the ``rid``, so the abort
arrives here before the retry's submit.

The submit shape is ``APIServer.submit_request``'s signature verbatim — the
protocol was designed by flattening it. ``prompt_parts`` travels for a chat
only, one map of ``PromptPart``'s fields per part. The bridge mesh lives in
its own socket dir (private to the frontend/bridge pair), so entity names
never collide with the conductor/worker mesh.

Opt-in: ``mstar-serve --rust-frontend`` replaces ``uvicorn.run`` with the
``mstar-server`` binary + this loop. Everything else in the process
(preprocess worker, conductor spawn, tensor transport) is identical. The
frontend enforces the API key (``--api-key`` / ``MSTAR_API_KEY``) itself; the
bridge socket is private to the pair.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from mstar.api_server.auth import API_KEY_ENV
from mstar.communication.communicator import CommProtocol
from mstar.communication.rust_communicator import (
    MsgpackCodec,
    RustZMQCommunicator,
)
from mstar.model.multimodal import PromptPart
from mstar.utils.client_errors import client_error_text

logger = logging.getLogger(__name__)

#: Upper bound on one blocking receive, so ``stop()`` is noticed promptly and
#: the loop stays responsive to shutdown. Not a latency knob: an arriving
#: message ends the wait immediately.
BRIDGE_RECV_WAIT_S = 0.05

#: The model's adapter class in ``ADAPTER_REGISTRY``; ``mstar-server`` picks
#: its OpenAI translation by it (``Adapter::from_python``).
OPENAI_ADAPTER_ENV = "MSTAR_OPENAI_ADAPTER"


def openai_adapter_class(model_name: str) -> str:
    """The class name of ``model_name``'s OpenAI adapter, "" when it has none."""
    from mstar.api_server.openai.adapters import get_adapter

    adapter = get_adapter(model_name)
    return "" if adapter is None else type(adapter).__name__





def find_server_binary(explicit: str | None = None) -> str:
    """Resolve the ``mstar-server`` binary: explicit arg, ``MSTAR_SERVER_BIN``,
    the running environment's scripts directory (where a Linux wheel installs
    it, also when the venv is not activated), ``$PATH``, then the in-repo
    release build. The scripts directory goes before ``$PATH`` so another
    environment's older binary never serves this one's bridge."""
    for candidate in (
        explicit,
        os.getenv("MSTAR_SERVER_BIN"),
        str(Path(sysconfig.get_path("scripts")) / "mstar-server"),
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
                       bridge_dir: str, upload_dir: str,
                       api_key: str | None = None) -> subprocess.Popen:
    # The key goes through the environment, not argv, which any local user
    # can read from the process list. ``api_key`` is the resolved setting, so
    # an inherited MSTAR_API_KEY must not override it.
    env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV}
    if api_key:
        env[API_KEY_ENV] = api_key
    env[OPENAI_ADAPTER_ENV] = openai_adapter_class(model_name)
    proc = subprocess.Popen([binary, model_name, str(port), bridge_dir,
                             upload_dir], env=env)
    logger.info("Rust frontend started (pid=%d, port=%d)", proc.pid, port)
    return proc


class RustFrontendBridge:
    """The serve loop between the Rust frontend and ``APIServer``."""

    def __init__(self, server, bridge_dir: str):
        # Fail fast: the frontend tokenizes when MSTAR_TOKENIZER is set and
        # ships token ids, which this backend cannot ingest (it owns
        # tokenization). Reject at launch rather than 500 every request.
        if os.getenv("MSTAR_TOKENIZER"):
            raise RuntimeError(
                "MSTAR_TOKENIZER is set, but the Rust-frontend bridge does "
                "not support frontend-tokenized (pre-tokenized) ingest — the "
                "model side owns tokenization. Unset MSTAR_TOKENIZER.")
        self.server = server
        # Rust transport with the msgpack codec: we bind as "bridge" in the
        # PRIVATE bridge dir (that is who the frontend sends to). Named
        # "bridge", not "conductor", to avoid confusion with M*'s Conductor.
        self.comm = RustZMQCommunicator(
            "bridge", push_ids=["frontend"], protocol=CommProtocol.IPC,
            ipc_socket_path_prefix=bridge_dir, codec=MsgpackCodec)
        self.running = True
        # sub -> its submission's key, until its relay ends.
        self._keys: dict[int, str] = {}

    # -- outbound ----------------------------------------------------------

    def _send(self, msg: dict) -> None:
        self.comm.send("frontend", msg)

    def _err(self, rid: str, sub: int | None, detail: str) -> None:
        self._send({"t": "err", "rid": rid, "sub": sub, "msg": detail})

    # -- request lifecycle ---------------------------------------------------

    def _submit(self, msg: dict) -> None:
        rid, sub = msg["rid"], msg.get("sub")
        if msg.get("tokens") is not None:
            # The Python preprocess worker owns tokenization; run the
            # frontend without MSTAR_TOKENIZER so it sends text.
            self._err(rid, sub, "pre-tokenized ingest is not supported by this "
                                "backend (unset MSTAR_TOKENIZER)")
            return
        try:
            parts = msg.get("prompt_parts")
            self.server.submit_request(
                text=msg.get("text"),
                file_paths=msg.get("file_paths") or None,
                input_modalities=list(msg.get("input_modalities") or []),
                output_modalities=list(msg.get("output_modalities") or ["text"]),
                model_kwargs=dict(msg.get("model_kwargs") or {}),
                prompt_parts=[PromptPart(**p) for p in parts] if parts else None,
                streaming=bool(msg.get("streaming", True)),
                request_id=rid,
            )
        except (ValueError, TypeError) as e:
            # a refused input: an in-band error chunk carries its 400 to the client
            logger.info("submit %s refused: %s", rid, e)
            self._send({
                "t": "chunk", "rid": rid, "sub": sub, "modality": "error",
                "data": str(e).encode("utf-8"), "metadata": {"status": 400},
            })
            return
        except Exception as e:  # noqa: BLE001 — one bad request must not kill the loop
            status = getattr(e, "status_code", None)
            if isinstance(status, int):
                # An HTTPException (not imported: this bridge runs without
                # fastapi), e.g. the 409 for a rid still in flight. An in-band
                # error chunk is the only reply that carries a status; `t="err"`
                # always surfaces as a 500.
                detail = getattr(e, "detail", e)
                logger.warning("submit %s refused (%d): %s", rid, status, detail)
                self._send({
                    "t": "chunk", "rid": rid, "sub": sub, "modality": "error",
                    "data": str(detail).encode(), "metadata": {"status": status},
                })
                return
            text, _ = client_error_text(
                e, f"submit {rid} failed", log=logger, type_name=False,
            )
            self._err(rid, sub, text)
            return
        # Bound to this submission now, not when the task first runs: a later
        # submit in the same receive batch may reuse the rid.
        self._keys[sub] = self.server.submission_key(rid)
        chunks = self.server.iter_result_chunks(rid)
        asyncio.get_running_loop().create_task(self._relay(rid, sub, chunks))

    async def _relay(self, rid: str, sub: int | None, chunks) -> None:
        """Forward this request's ResultChunks; terminal err on failure."""
        try:
            async for chunk in chunks:
                self._send({
                    "t": "chunk", "rid": rid, "sub": sub, "modality": chunk.modality,
                    "data": bytes(chunk.data), "metadata": chunk.metadata,
                })
            self._send({"t": "done", "rid": rid, "sub": sub})
        except Exception as e:  # noqa: BLE001 — includes the per-request timeout
            detail = getattr(e, "detail", None)
            if detail:
                detail = str(detail)
                logger.warning("request %s failed: %s", rid, detail)
            else:
                detail, _ = client_error_text(
                    e, f"request {rid} failed", log=logger, type_name=False,
                )
            self._err(rid, sub, detail)
        finally:
            self._keys.pop(sub, None)

    def _abort(self, msg: dict) -> None:
        """The frontend dropped submission ``sub``; a newer one under its rid
        (the client's retry) may already be running and is left alone."""
        sub = msg.get("sub")
        if sub is None:
            self.server.abort_request(msg["rid"])
        else:
            self.server.abort_submission(self._keys.pop(sub, None))

    def _metrics(self, msg: dict) -> None:
        metrics = self.server.metrics
        reply = {"t": "metrics", "rid": msg["rid"], "sub": msg.get("sub"),
                 "data": b"", "metadata": {}}
        if metrics is not None:
            refusals = msg.get("refusals")
            if refusals is not None:
                metrics.refusals.update_rust(
                    refusals["route"], refusals["status"], refusals["count"])
            reply["data"] = metrics.render()
            reply["metadata"] = {"content_type": metrics.content_type}
        self._send(reply)

    def _handle(self, msg: dict) -> None:
        t = msg.get("t")
        if t == "submit":
            self._submit(msg)
        elif t == "abort":
            self._abort(msg)
        elif t == "ping":
            # Deep-health liveness: the frontend's /health goes red
            # unless this loop answers.
            self._send({"t": "pong", "rid": msg["rid"], "sub": msg.get("sub")})
        elif t == "metrics":
            self._metrics(msg)
        else:
            logger.warning("bridge: unknown message type %r", t)

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
                # Per-message isolation: FastAPI turns a handler exception into
                # one 500 and keeps serving. The bridge must restore that — a
                # failure in any branch (a bad `rid` lookup in abort, a codec
                # decode error surfacing as a malformed msg, a send failure)
                # would otherwise propagate out of `_serve` → `run()` →
                # `entrypoint.py`'s `finally`, which terminates the Rust process
                # and takes the whole server down. `_submit` is already wrapped;
                # this guards every other branch.
                try:
                    self._handle(msg)
                except Exception:  # noqa: BLE001 — one bad message must not kill the loop
                    logger.exception("bridge: dropping a message that raised (%r)",
                                     msg.get("t"))

    def run(self) -> None:
        """Blocking serve loop (the ``uvicorn.run`` replacement)."""
        # with the message thread: tokens pass from it to this loop. The Rust
        # process is already running, so it keeps the process mask. Test doubles
        # stand in for the server without the pin.
        pin = getattr(self.server, "pin_result_thread", None)
        if pin is not None:
            pin()
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        self.running = False
