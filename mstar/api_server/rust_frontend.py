"""The Rust axum frontend in front of ``APIServer``.

``mstar-server`` (``rust/server/``) owns the HTTP surface — routes, OpenAI
translation, SSE/NDJSON streaming, uploads, CORS — and speaks a flattened
msgpack protocol over the Rust ZMQ transport to this bridge, which
drives the existing Python data plane (``APIServer.submit_request`` /
``iter_result_chunks`` / ``abort_submission``; preprocessing and the conductor
protocol are untouched):

    frontend -> bridge:  {t:"submit", rid, sub, text, file_paths,
                          input_modalities, output_modalities, model_kwargs,
                          prompt_parts, streaming, accepts_chunks, report_usage}
                       | {t:"abort", rid, sub}
                       | {t:"ping", rid, sub}
                       | {t:"metrics", rid, sub, refusals:{route, status, count}}
                       | {t:"check_ref_audio", rid, sub, path, decoder, bare}
                       | {t:"voices", rid, sub}
    bridge -> frontend:  {t:"chunk", rid, sub, modality, data(bin), metadata}
                       | {t:"chunks", subs, modalities, payloads(bin), metadatas}
                       | {t:"err", rid, sub, msg} | {t:"done", rid, sub}
                       | {t:"pong", rid, sub}
                       | {t:"metrics", rid, sub, data(bin), metadata}
                       | {t:"checked", rid, sub, error?, status?}
                       | {t:"voices", rid, sub, metadata:{voices?}}

A ``metrics`` request carries the frontend's cumulative refusal counts as
three parallel lists (route template, status, count), which the API server
exports as ``mstar_http_refusals_total``. The reply is what ``GET /metrics``
serves on the Python frontend: the text rendered from ``APIServer.metrics``
with its ``content_type`` in ``metadata``, or empty ``metadata`` when metrics
are off (a 404; the counts are then ignored).

``chunks`` carries many submissions' chunks as parallel lists, in order:
the API server's message thread hands the bridge every chunk it drained in
one pass (``APIServer.relay_chunks_to``), so a decode step costs one message
rather than one per request. Only a submit that sets ``accepts_chunks`` gets
them, so a frontend binary older than this bridge still gets ``chunk``s. A
submission's error chunk, and anything its handler finds unrelayed at the
end, always goes out as ``chunk``.

A submit that sets ``report_usage`` (a chat) gets a last ``chunk`` of
modality ``usage`` before ``done``: its token counts and finish reason in
``metadata`` (``APIServer.submit_request``'s ``report_usage``).

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
import json
import logging
import os
import re
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
from mstar.graph.runtime.utils import CHECKOUT_RUST, is_mstar_rust_crate, require_rust_extension
from mstar.model.multimodal import PromptPart
from mstar.utils.client_errors import client_error_text, scrub_paths
from mstar.utils.threads import ThreadPool

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





def find_server_binary(explicit: str | None = None, rust_dir: Path = CHECKOUT_RUST) -> str:
    """Resolve the ``mstar-server`` binary: explicit arg, ``MSTAR_SERVER_BIN``,
    the running environment's scripts directory (where a Linux wheel installs
    it, also when the venv is not activated), ``$PATH``, then the in-repo
    release build. The scripts directory goes before ``$PATH`` so another
    environment's older binary never serves this one's bridge.

    In a checkout, a binary built from other sources than ``rust_dir`` is
    skipped, or used with a warning when named by the arg or the variable.
    Raises FileNotFoundError when there is none, RuntimeError when all are
    stale."""
    checkout = is_mstar_rust_crate(rust_dir)
    if checkout:
        require_rust_extension()  # its source_hash is what binaries are compared with
    stale = []
    for candidate, named in (
        (explicit, True),
        (os.getenv("MSTAR_SERVER_BIN"), True),
        (str(Path(sysconfig.get_path("scripts")) / "mstar-server"), False),
        (shutil.which("mstar-server"), False),
        (str(rust_dir / "server" / "target" / "release" / "mstar-server"), False),
    ):
        if not (candidate and Path(candidate).is_file()):
            continue
        problem = _stale_server(candidate, rust_dir) if checkout else None
        if problem is None:
            return candidate
        if named:
            logger.warning("Using mstar-server %s, which %s", candidate, problem)
            return candidate
        stale.append(f"{candidate} {problem}")
    if stale:
        # Both touches, for the reason in require_rust_extension.
        server = rust_dir / "server"
        raise RuntimeError(
            f"No mstar-server built from {rust_dir}: {'; '.join(stale)}. Rebuild it with "
            f"`touch {rust_dir / 'build.rs'} {server / 'build.rs'} && "
            f"cargo build --release --manifest-path {server / 'Cargo.toml'}`.")
    raise FileNotFoundError(
        "mstar-server binary not found: build it with "
        "`cargo build --release` in rust/server/, put it on $PATH, or set "
        "MSTAR_SERVER_BIN")


def _stale_server(binary: str, rust_dir: Path) -> str | None:
    """Why ``binary`` was not built from the sources in ``rust_dir``, or None."""
    import mstar_rust

    # No arguments, which every mstar-server exits on (an older one would
    # serve `--source-hash` as a model name); a newer one prints its sources.
    try:
        ran = subprocess.run([binary], capture_output=True, text=True, errors="replace", timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not be run ({e})"
    built = re.search(r"built from sources ([0-9a-f]{16}) ([0-9a-f]{16})", ran.stdout + ran.stderr)
    if built is None:
        return "predates the source check"
    current = (mstar_rust.source_hash(str(rust_dir)), mstar_rust.source_hash(str(rust_dir / "server")))
    if built.groups() == current:
        return None
    return (f"was built from sources hashing to {' '.join(built.groups())}, "
            f"but {rust_dir} and its server/ hash to {' '.join(current)}")


#: The serve settings ``mstar-server`` reads from its environment. The
#: launcher's resolved values win over inherited ones, as the API key's does.
REQUEST_TIMEOUT_ENV = "MSTAR_REQUEST_TIMEOUT_S"
SAMPLE_RATE_ENV = "MSTAR_SAMPLE_RATE"
#: The libsndfile ``soundfile`` loaded, which mstar-server opens for mp3/flac/opus.
LIBSNDFILE_ENV = "MSTAR_LIBSNDFILE"
#: ``ServedModels`` as JSON: the names that select the base model, and the LoRA adapters.
SERVED_MODELS_ENV = "MSTAR_SERVED_MODELS"


def soundfile_library() -> str | None:
    """The libsndfile that ``soundfile`` loaded (the copy its wheel bundles, else the
    system one it found), for ``mstar-server`` to encode the compressed speech formats
    with the same library; ``None`` without soundfile, which leaves only wav and pcm
    on both frontends (``media_io.audio_formats``). These are soundfile's own module
    attributes, which it sets as it loads the library."""
    try:
        import soundfile
    except (ImportError, OSError):
        return None
    # `_libname` is set only when the packaged library failed to load and soundfile
    # fell back to the system one; `_full_path` is set before that attempt
    return getattr(soundfile, "_libname", None) or getattr(soundfile, "_full_path", None)


def served_models_json(served_models) -> str:
    """``SERVED_MODELS_ENV``'s value for a ``ServedModels``."""
    return json.dumps({
        "base": sorted(served_models.base), "adapters": list(served_models.adapters),
        "lora": served_models.lora,
    })


def launch_rust_server(binary: str, model_name: str, port: int,
                       bridge_dir: str, upload_dir: str,
                       api_key: str | None = None, timeout_s: float | None = None,
                       sample_rate: int | None = None, served_models=None) -> subprocess.Popen:
    """``timeout_s`` is ``--timeout`` and ``sample_rate`` the model's audio output rate
    (``get_output_sample_rate``); ``None`` leaves the frontend's own default.
    ``served_models`` is the API server's ``ServedModels``, whose LoRA adapters a
    request's ``model`` selects; ``None`` serves the base model alone."""
    # The key goes through the environment, not argv, which any local user
    # can read from the process list. ``api_key`` is the resolved setting, so
    # an inherited MSTAR_API_KEY must not override it.
    env = {k: v for k, v in os.environ.items() if k != API_KEY_ENV}
    if api_key:
        env[API_KEY_ENV] = api_key
    if timeout_s is not None:
        # inf and nan never time out on the Python frontend, so they get a deadline past
        # any request (~31 years); a negative one times every request out there, as 0 does
        timeout_s = float(timeout_s)
        env[REQUEST_TIMEOUT_ENV] = repr(max(timeout_s, 0.0) if timeout_s < 1e9 else 1e9)
    if sample_rate is not None:
        env[SAMPLE_RATE_ENV] = str(int(sample_rate))
    env.pop(SERVED_MODELS_ENV, None)
    if served_models is not None:
        env[SERVED_MODELS_ENV] = served_models_json(served_models)
    env.pop(LIBSNDFILE_ENV, None)
    if (libsndfile := soundfile_library()) is not None:
        env[LIBSNDFILE_ENV] = libsndfile
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
        # set by _serve; the API server's message thread schedules relay() on it
        self.loop: asyncio.AbstractEventLoop | None = None

    # -- outbound ----------------------------------------------------------

    def _send(self, msg: dict) -> None:
        self.comm.send("frontend", msg)

    def _err(self, rid: str, sub: int | None, detail: str) -> None:
        self._send({"t": "err", "rid": rid, "sub": sub, "msg": detail})

    def relay(self, tags: list[int], chunks: list) -> None:
        """``ChunkSink.relay``: one ``chunks`` message for a message pass."""
        self._send({
            "t": "chunks", "subs": tags,
            "modalities": [c.modality for c in chunks],
            "payloads": [bytes(c.data) for c in chunks],
            "metadatas": [c.metadata for c in chunks],
        })

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
                report_usage=bool(msg.get("report_usage")),
            )
        except (ValueError, TypeError) as e:
            # a refused input: an in-band error chunk carries its 400 to the client.
            # A TypeError may be a bug here instead (the two sides' PromptPart
            # drifting), so it keeps its traceback, as on the router.
            if isinstance(e, ValueError):
                logger.info("submit %s refused: %s", rid, e)
                text = scrub_paths(str(e))
            else:
                text, _ = client_error_text(e, f"submit {rid} failed", log=logger, type_name=False)
            self._send({
                "t": "chunk", "rid": rid, "sub": sub, "modality": "error",
                "data": text.encode("utf-8"), "metadata": {"status": 400},
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
        key = self._keys[sub] = self.server.submission_key(rid)
        if msg.get("accepts_chunks"):
            self.server.relay_chunks_to(key, self, sub)
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

    def _check_ref_audio(self, msg: dict) -> None:
        """``mstar-server`` saved a request's reference clip; the model's own loader
        judges it (``media_io.check_saved_ref_audio``), off this loop since it
        decodes. The reply carries the refusal and its status, or no ``error``."""
        reply = {"t": "checked", "rid": msg.get("rid"), "sub": msg.get("sub")}

        async def check() -> None:
            # every outcome replies: the frontend waits on it before submitting
            try:
                from mstar.api_server import media_io

                path = Path(msg["path"])
                # it deletes what it refuses, so only a clip the frontend saved under upload_dir
                if not path.resolve().is_relative_to(Path(self.server.upload_dir).resolve()):
                    raise RuntimeError(f"ref_audio {path} is outside the upload directory")
                await asyncio.to_thread(
                    media_io.check_saved_ref_audio, str(path), msg["decoder"], bool(msg.get("bare")))
                self._send(reply)
            except ValueError as exc:
                self._send({**reply, "error": str(exc), "status": 400})
            except Exception:  # noqa: BLE001 — the request fails, the bridge goes on
                logger.exception("bridge: checking a ref_audio clip raised")
                self._send({**reply, "error": "the server could not check ref_audio", "status": 500})

        self.loop.create_task(check())

    def _voices(self, msg: dict) -> None:
        """``GET /v1/audio/voices``' body, asked per request since a model's list can
        change while it serves (CosyVoice3 lists its voices directory); no ``voices``
        in ``metadata`` when the model publishes none (a 404)."""
        reply = {"t": "voices", "rid": msg.get("rid"), "sub": msg.get("sub"), "data": b"", "metadata": {}}

        async def answer() -> None:
            try:
                from mstar.api_server.openai.serving_speech import list_voices

                voices = await asyncio.to_thread(list_voices, self.server)
                if voices is not None:
                    reply["metadata"] = {"voices": voices.model_dump()}
                self._send(reply)
            except Exception:  # noqa: BLE001 — the request fails, the bridge goes on
                logger.exception("bridge: listing the voices raised")
                self._send({"t": "chunk", "rid": msg.get("rid"), "sub": msg.get("sub"),
                            "modality": "error", "data": b"the server could not list the voices",
                            "metadata": {"status": 500}})

        self.loop.create_task(answer())

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
        elif t == "check_ref_audio":
            self._check_ref_audio(msg)
        elif t == "voices":
            self._voices(msg)
        else:
            logger.warning("bridge: unknown message type %r", t)

    # -- loop ----------------------------------------------------------------

    async def _serve(self) -> None:
        logger.info("Rust-frontend bridge serving")
        loop = self.loop = asyncio.get_running_loop()
        # asyncio would start its default pool from this thread, and so on
        # the result CPU when the bridge is pinned; handlers' to_thread work
        # (audio planning, request decoding) belongs on the process mask
        loop.set_default_executor(ThreadPool(thread_name_prefix="mstar-bridge"))
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
