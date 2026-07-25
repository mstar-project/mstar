"""MSTAR_DETOK_PROC — off-process token->text detokenization.

vLLM keeps its API/serve process free of the per-token detokenize + postprocess
cost by running detok in a separate OS process (EngineCore vs. API proc). Ours
does not: the ``PreprocessWorkerThread`` runs ``model.postprocess`` — the
tokenizer decode — inline on the uvicorn/serve process, and a py-spy profile put
``_postprocess_batch`` at 23% of that process's CPU floor, contending for the one
GIL the HTTP serving loop also needs.

This module moves that work to a dedicated child process. The serve-side data
worker keeps ALL request-context state and ALL emit ordering (the MSTAR_ORDERED_EMIT
FIFO, per-rid cleanup, the tensor-read accounting) exactly where it was; only the
leaf ``model.postprocess`` call for TEXT chunks crosses the process boundary.
Because the ordering machinery already operates on opaque ``ResultChunk`` objects,
deferring a chunk's ``data`` changes
nothing it does — the chunk is reordered as before and its bytes are filled in
later, off-process.

Design (see docs / the emit_sidecar.py prior art in mstar/worker):

- ONE detok child process per serve process (like vLLM's single detok proc, and
  like the emit sidecar). A single process consuming its input FIFO and emitting
  its output FIFO preserves per-rid order for free: the data worker submits a
  rid's chunks in the already-reordered order, so FIFO-in == FIFO-out == correct
  delivery order. We do not fan out across cores — the goal is to get the decode
  OFF the serve GIL, not to parallelize decode.

- Only ``modality == "text"`` is offloaded. Audio/image ``postprocess`` is cheap
  (a ``.numpy().tobytes()``), is not the GIL hog, and would mean shipping large
  tensors across the boundary — those stay inline.

- Byte-identity: the child loads the SAME lightweight tokenizer-only model the
  serve process builds (same ``get_model_class(model_name)(...)`` construction,
  same model_kwargs) and calls the SAME ``model.postprocess`` on a tensor
  reconstructed from the shipped ints/dtype/dims. No incremental/prefix detok
  state is introduced (the model's text postprocess is a stateless per-chunk
  ``tokenizer.decode``); introducing prefix state would change bytes, so we keep
  it per-chunk. Per-rid registration/drop records are still exchanged so the
  child can free any per-rid structure on completion/abort (and as the hook a
  future stateful detok would need).

- Boot-time only: process topology cannot follow ``MSTAR_DYNFLAGS`` (a child is
  spawned at init, exactly as the emit sidecar documents). ``MSTAR_DETOK_PROC``
  is read ONCE when the ``PreprocessWorker`` is constructed; A/B is via two
  server boots, not a runtime flip.

- Failure semantics: the child dying (or its input queue saturating) is a
  PERMANENT fall back to the inline path, never a hang and never a dropped
  request. Any chunk already handed to the child but not yet returned is
  re-postprocessed inline, in submit order, before the flag flips — so per-rid
  order is preserved and every request still completes with correct bytes. New
  chunks then postprocess inline in the serve process, as with the flag off.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
from dataclasses import dataclass

import torch

from mstar.api_server.request_types import PendingDetok, ResultChunk

logger = logging.getLogger(__name__)

# Bounded serve->child work queue. A trip means the child fell behind the serve
# process badly enough that buffering more would only add latency and memory;
# treat it as failure (permanent inline fallback), not backpressure — mirrors
# the emit sidecar's SNDHWM policy.
DETOK_IN_MAXSIZE = 4096


def _reconstruct(pd: PendingDetok) -> torch.Tensor:
    """Rebuild the exact tensor the inline path would pass to postprocess.

    Byte-identity hinges on this matching ``_build_inline_chunks`` /
    ``_process_read_tensors``: ``torch.tensor(ints, dtype).reshape(dims)``.
    """
    return torch.tensor(pd.ints, dtype=pd.dtype).reshape(pd.dims)


@dataclass
class _RidDrop:
    """Free any per-rid state the child holds (request complete/abort)."""
    request_id: str


@dataclass
class _Work:
    """One deferred text-detok item. Small and plainly picklable (ints, a
    torch.dtype, a tuple) — deliberately NOT a torch tensor, so nothing rides
    torch's multiprocessing shared-memory reducer (which would leak /dev/shm
    segments; see the workspace disk rules)."""
    work_id: int
    request_id: str
    modality: str
    ints: list
    dtype: object
    dims: tuple


class DetokClient:
    """Serve-side handle to the detok child.

    Owns the spawned process, the two queues, the outstanding-work table, and a
    receiver thread that turns returned bytes back into emitted chunks. Used
    from the data worker thread (``submit``) and its own receiver thread; the
    shared state (``_outstanding`` / ``_failed``) is guarded by ``_lock``.
    """

    def __init__(
        self,
        *,
        model,
        model_name: str,
        cache_dir: str | None,
        model_kwargs: dict | None,
        out_queue: queue.Queue,
        log_level: str = "INFO",
    ):
        # ``model`` is the serve process's own tokenizer-only instance, used
        # ONLY for the inline fallback (never sent to the child).
        self._model = model
        self._out_queue = out_queue

        self._lock = threading.Lock()
        self._outstanding: dict[int, ResultChunk] = {}
        self._next_id = 0
        self._failed = False
        self._stop = threading.Event()

        ctx = mp.get_context("spawn")
        # Bounded in-queue: a Full put is a failure signal (see submit). The
        # out-queue is unbounded — the child must never block returning results.
        self._in: mp.Queue = ctx.Queue(maxsize=DETOK_IN_MAXSIZE)
        self._out: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=run_detok_proc,
            kwargs=dict(
                model_name=model_name,
                cache_dir=cache_dir,
                model_kwargs=model_kwargs or {},
                in_queue=self._in,
                out_queue=self._out,
                log_level=log_level,
            ),
            daemon=True,
            name="mstar_detok",
        )
        self._proc.start()
        logger.info(
            "MSTAR_DETOK_PROC: detok process spawned (pid=%d, model=%s)",
            self._proc.pid, model_name,
        )

        self._receiver = threading.Thread(
            target=self._receive_loop, name="mstar_detok_receiver", daemon=True,
        )
        self._receiver.start()

    # -- serve-side (data worker thread) ------------------------------------

    def healthy(self) -> bool:
        return not self._failed and self._proc.is_alive()

    def submit(self, chunk: ResultChunk) -> bool:
        """Hand a deferred-text chunk to the child. Returns False if the child
        is unavailable (caller must postprocess inline instead). On success the
        chunk is owned by the client until the receiver emits it."""
        pd = chunk.pending_detok
        with self._lock:
            if self._failed:
                return False
            wid = self._next_id
            self._next_id += 1
            self._outstanding[wid] = chunk
            try:
                # Non-blocking under the lock: the queue only fills when the
                # child has stalled, and blocking here would stall the serve
                # worker on the very thing we moved off it.
                self._in.put_nowait(_Work(
                    work_id=wid,
                    request_id=chunk.request_id,
                    modality=chunk.modality,
                    ints=pd.ints,
                    dtype=pd.dtype,
                    dims=pd.dims,
                ))
            except queue.Full:
                del self._outstanding[wid]
                logger.critical(
                    "MSTAR_DETOK_PROC: work queue full (%d) — falling back to "
                    "inline detok permanently", DETOK_IN_MAXSIZE,
                )
                self._fail_locked()
                return False
        return True

    def drop_rid(self, request_id: str) -> None:
        """Best-effort: tell the child to free per-rid state (complete/abort).
        Outstanding items for the rid are left to return normally; the serve
        side drops their late chunks via the existing tolerant path."""
        if self._failed or not self._proc.is_alive():
            return
        try:
            self._in.put_nowait(_RidDrop(request_id=request_id))
        except queue.Full:
            # A drop is advisory (detok is stateless per chunk today); losing it
            # cannot corrupt output. Don't escalate to failure over cleanup.
            pass

    # -- receiver thread ----------------------------------------------------

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._out.get(timeout=0.1)
            except queue.Empty:
                # No results pending. If the child has died, recover every
                # outstanding item inline and go permanently inline.
                if not self._proc.is_alive() and not self._failed:
                    logger.critical(
                        "MSTAR_DETOK_PROC: detok process (pid=%s) died — "
                        "recovering %d in-flight chunk(s) inline and falling "
                        "back permanently",
                        self._proc.pid, len(self._outstanding),
                    )
                    self._fail_and_drain()
                    return
                continue
            wid, data = item
            with self._lock:
                chunk = self._outstanding.pop(wid, None)
            if chunk is None:
                # Already recovered inline (fallback) or its rid was cleaned up.
                continue
            chunk.data = data
            chunk.pending_detok = None
            self._out_queue.put(chunk)

    # -- failure handling ---------------------------------------------------

    def _fail_locked(self) -> None:
        """Mark failed and inline-drain outstanding. Caller holds ``_lock``."""
        self._failed = True
        items = sorted(self._outstanding.items())  # ascending work_id
        self._outstanding.clear()
        self._drain_inline(items)

    def _fail_and_drain(self) -> None:
        with self._lock:
            if self._failed:
                return
            self._failed = True
            items = sorted(self._outstanding.items())
            self._outstanding.clear()
        # Popped from _outstanding under the lock, so a late child return for
        # any of these wids hits the pop-None path and is ignored (no dup).
        self._drain_inline(items)

    def _drain_inline(self, items: list[tuple[int, ResultChunk]]) -> None:
        """Postprocess the given chunks inline, in submit order, and emit.

        Submit order (ascending work_id) is per-rid delivery order, so draining
        in this order before any later inline emit preserves each rid's stream.
        """
        for _wid, chunk in items:
            pd = chunk.pending_detok
            try:
                chunk.data = self._model.postprocess(_reconstruct(pd), chunk.modality)
            except Exception:
                logger.exception(
                    "MSTAR_DETOK_PROC: inline recovery postprocess failed for "
                    "rid=%s; emitting empty chunk so the request does not hang",
                    chunk.request_id,
                )
                chunk.data = b""
            chunk.pending_detok = None
            self._out_queue.put(chunk)

    # -- lifecycle ----------------------------------------------------------

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._in.put_nowait(None)  # sentinel: clean child exit
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
                if self._proc.is_alive():
                    self._proc.kill()
        if self._receiver.is_alive():
            self._receiver.join(timeout=1.0)
        # Anything still outstanding at shutdown: recover inline so no request
        # is left without its bytes.
        with self._lock:
            items = sorted(self._outstanding.items())
            self._outstanding.clear()
            self._failed = True
        if items:
            self._drain_inline(items)


def run_detok_proc(
    *,
    model_name: str,
    cache_dir: str | None,
    model_kwargs: dict,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    log_level: str = "INFO",
) -> None:
    """Detok child entry point. Module-level for spawn picklability (same
    pattern as the conductor/sidecar targets)."""
    # Pure-CPU process: make CUDA unreachable before anything can touch torch,
    # so a stray import can't initialize a context here (design parity with the
    # emit sidecar).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [mstar_detok] %(name)s: %(message)s",
        force=True,
    )
    try:
        from mstar.utils.logging_config import quiet_noisy_loggers
        quiet_noisy_loggers()
    except Exception:
        pass

    # MSTAR_BURST_CAP (default off): cap this pure-CPU detok child's thread
    # fan-out (postprocess/decode). No-op when off.
    try:
        from mstar.utils.burst_cap import apply_process_thread_cap
        apply_process_thread_cap("detok")
    except Exception:
        pass

    from mstar.model.registry import HF_MODELS, get_model_class

    # Build the SAME lightweight tokenizer-only instance the serve process uses
    # (entrypoint.main / _conductor_process_target), so postprocess is identical.
    model = get_model_class(model_name)(
        model_path_hf=HF_MODELS.get(model_name, {}).get("model_path_hf", ""),
        cache_dir=cache_dir,
        **(model_kwargs or {}),
    )
    logger.info("mstar_detok ready (model=%s, pid=%d)", model_name, os.getpid())

    # Per-rid registry. Detok is stateless per chunk today (byte-identity), so
    # this only tracks liveness and is freed on drop — the hook a future
    # incremental/prefix detok would key its state on.
    live_rids: set[str] = set()
    parent_pid = os.getppid()
    processed = 0

    while True:
        try:
            item = in_queue.get(timeout=0.2)
        except queue.Empty:
            # Parent-death watch: getppid() flips to the reaper when the serve
            # process is gone. Exit so we don't linger as an orphan.
            if os.getppid() != parent_pid:
                logger.warning("mstar_detok: parent gone; exiting")
                break
            continue
        if item is None:
            logger.info("mstar_detok: sentinel received; exiting")
            break
        if type(item) is _RidDrop:
            live_rids.discard(item.request_id)
            continue
        # _Work
        live_rids.add(item.request_id)
        tensor = torch.tensor(item.ints, dtype=item.dtype).reshape(item.dims)
        data = model.postprocess(tensor, item.modality)
        out_queue.put((item.work_id, data))
        processed += 1

    logger.info("mstar_detok exiting: processed=%d", processed)
