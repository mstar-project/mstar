"""MSTAR_PREPROC_PROC — off-process request-side multimodal preprocessing.

vLLM keeps its API/serve process free of engine host work by splitting the HTTP
process from the EngineCore process. Fix #13 (MSTAR_DETOK_PROC) moved only the
*output*-side ``tokenizer.decode`` off the serve process. The much larger cost
still on the serve process is the *input*-side multimodal preprocessing: for
i2t, HF/torchvision image decode + smart-resize + normalize + patchify
(``model.load_image`` + ``model.process_prompt``), which our own note put at
~175 ms per image and which fans out across ~21 cores via torch's intra-op
threadpool. A burst of 32 image requests runs all 32 of those preprocesses
serially on the ONE data-worker thread inside the uvicorn process, holding the
GIL the uvicorn event loop and the ZMQ result-drain thread also need — so
in-flight decodes' token emission stalls exactly while a new batch is admitted.

This module moves that CPU burst into a dedicated pool of OS child processes:

- The heavy, CUDA-free leaf work (``load_image`` / ``load_audio`` / ``load_video``
  + ``process_prompt``) runs in a child; the child returns the produced tensors
  as plain bytes. The serve-side data worker keeps ALL transport / conductor
  state (``tensor_manager.store_and_return_tensor_info``, register/persist, the
  ``NewRequestConductor`` send) exactly where it was — only the CPU preprocessing
  crosses the process boundary. See ``preprocess_tensors`` (the shared, single
  implementation both the inline path and the child call, so they cannot drift)
  and ``PreprocessWorkerThread._admit_request`` (the transport tail).

- POOL, not a single child (unlike detok): the goal here IS to parallelize the
  burst across cores. Each child has its own bounded in-queue; jobs are dispatched
  round-robin. One shared out-queue returns results. The serve worker thread polls
  completions in its existing run loop (no receiver thread needed).

- In-order admission: completions are emitted to the conductor in submit order via
  a small reorder buffer, so a concurrent burst reaches the conductor in the exact
  order the inline path would have used. The flag's ONLY effect on the happy path
  is *where* preprocessing ran — same admissions, same order, byte-identical
  tensors.

Byte-identity (bitwise-identical produced tensors vs. inline):

- The child builds the SAME lightweight tokenizer/processor model the serve
  process builds (same ``get_model_class(model_name)(...)`` construction, same
  ``model_kwargs``) and calls the SAME ``preprocess_tensors``. The env is
  inherited by spawn, so the same MSTAR_GPU_IMAGE_PREPROCESS / processor code
  path is taken. Image/text preprocessing is device-agnostic on CPU (the data
  worker's device is ``cpu``), so hiding CUDA in the child changes nothing for
  i2t: identical torchvision/HF ops on identical input bytes -> identical output.
- Tensors cross as raw bytes + dtype + shape and are rebuilt with
  ``torch.frombuffer(...).reshape(...)`` — the exact same bytes reinterpreted with
  the exact same dtype and shape (bitwise identical). Deliberately NOT torch
  tensors over the queue, so nothing rides torch's mp shared-memory reducer (which
  would leak ``/dev/shm`` segments; see the workspace disk rules). The values the
  model receives are identical; the rebuilt container is contiguous (a no-op for
  the values).
- AUDIO CAVEAT: with ``MSTAR_GPU_MEL=1`` (default) and CUDA present, the serve
  process computes the log-mel on the GPU, which is NOT bit-identical to the CPU
  mel a CUDA-hidden child would produce. So an audio-input request is NOT
  offloaded while GPU-mel is effectively on (it runs inline, byte-identical). i2t
  (image+text) is always offloaded. See ``PreprocessWorkerThread._should_offload``.

Failure semantics (mirror the detok sidecar): a child dying, an errored job, or a
saturated in-queue is a PERMANENT fall back to inline preprocessing — never a hang
and never a dropped request. Any still-outstanding job is recovered by running the
identical inline path on the serve worker thread, exactly once (a returned job is
popped from the outstanding table, so a late child result is dropped).

Boot-time only: the pool is spawned when ``PreprocessWorker`` is constructed and
cannot follow MSTAR_DYNFLAGS (process topology). A/B is via two server boots.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
from collections import OrderedDict
from typing import Any

import torch

from mstar.api_server.request_types import PreprocessInput

logger = logging.getLogger(__name__)

# Per-child bounded work queue. A full put means that child fell far enough
# behind that buffering more would only add latency; treat it as failure
# (permanent inline fallback), mirroring the detok sidecar's policy.
PREPROC_IN_MAXSIZE = 1024

# A wire tensor is (raw_bytes, torch.dtype, shape_tuple) — all plainly picklable.
WireTensor = tuple[bytes, Any, tuple]


# ---------------------------------------------------------------------------
# Shared preprocessing (single implementation for inline + child, no drift)
# ---------------------------------------------------------------------------

def preprocess_tensors(
    model, input: PreprocessInput, device: str
) -> tuple[dict, dict]:
    """Load raw modality tensors and run ``process_prompt``.

    This is the exact CPU work that used to live inline in
    ``PreprocessWorkerThread._process_input``, extracted so the inline path and
    the child process run byte-identical code. Returns ``(tensors,
    input_metadata)`` ready for ``store_and_return_tensor_info``.
    """
    tensors: dict = {}
    input_metadata: dict = {}

    # Load raw modality tensors from file_paths (images, audio, video) so they
    # can be passed to process_prompt() below.
    if input.file_paths is not None:
        for modality in input.file_paths:
            key = f"{modality}_inputs"
            tensors[key] = []
            input_metadata[key] = []
            for filepath in input.file_paths[modality]:
                if modality == "image":
                    out = model.load_image(filepath, device)
                elif modality == "audio":
                    out = model.load_audio(filepath, device)
                elif modality == "video":
                    out = model.load_video(filepath, device)
                else:
                    continue
                tensors[key].append(out.data)
                input_metadata[key].append(out.metadata)

    # Tokenize the prompt and let the model augment/transform the tensors dict
    # (Qwen3-Omni computes pixel_values, image_grid_thw, audio_features, ...).
    if model is not None:
        prompt_tensors = model.process_prompt(
            input.text,
            input.input_modalities,
            input.output_modalities,
            tensors=tensors,
            input_metadata=input_metadata,
            **(input.model_kwargs or {}),
        )
        if prompt_tensors:
            tensors.update(prompt_tensors)
    elif input.text is not None:
        # Fallback: encode as UTF-8 bytes -> uint8 tensor.
        byte_data = input.text.encode("utf-8")
        tensors["text_inputs"] = [
            torch.tensor(list(byte_data), dtype=torch.uint8, device=device)
        ]

    return tensors, input_metadata


# ---------------------------------------------------------------------------
# Byte-exact tensor <-> wire helpers (no torch mp reducer, no /dev/shm)
# ---------------------------------------------------------------------------

def _tensor_to_wire(t: torch.Tensor) -> WireTensor:
    """Serialize a CPU tensor to (raw_bytes, dtype, shape).

    The bytes are the tensor's exact memory (viewed as uint8), so rebuilding
    with the same dtype+shape is bitwise identical. Works for any dtype
    (float16/bf16/int64/bool/...) since we reinterpret the raw bytes rather than
    go through numpy (which lacks bf16)."""
    t = t.detach().to("cpu").contiguous()
    raw = t.flatten().view(torch.uint8).numpy().tobytes()
    return (raw, t.dtype, tuple(t.shape))


def _tensor_from_wire(wire: WireTensor) -> torch.Tensor:
    """Rebuild the exact tensor from (raw_bytes, dtype, shape). Contiguous."""
    raw, dtype, shape = wire
    if len(raw) == 0:
        # Zero-element tensor: frombuffer rejects an empty buffer, so build it
        # directly (still byte-identical: no elements to differ).
        return torch.empty(shape, dtype=dtype)
    # bytearray -> writable buffer (avoids frombuffer's read-only warning); the
    # copy is unavoidable anyway (the pickled bytes are transient).
    flat = torch.frombuffer(bytearray(raw), dtype=dtype)
    return flat.reshape(shape)


def _tensors_to_wire(tensors: dict) -> dict:
    return {k: [_tensor_to_wire(t) for t in v] for k, v in tensors.items()}


def _tensors_from_wire(wire: dict) -> dict:
    return {k: [_tensor_from_wire(w) for w in v] for k, v in wire.items()}


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

def run_preproc_proc(
    *,
    model_name: str,
    cache_dir: str | None,
    model_kwargs: dict,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    num_threads: int,
    log_level: str = "INFO",
) -> None:
    """Preproc child entry point. Module-level for spawn picklability (same
    pattern as the detok/conductor targets)."""
    # Pure-CPU process: make CUDA unreachable before torch can touch it, so a
    # stray import can't init a context here and the process never touches a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [mstar_preproc] %(name)s: %(message)s",
        force=True,
    )
    try:
        from mstar.utils.logging_config import quiet_noisy_loggers
        quiet_noisy_loggers()
    except Exception:
        pass

    if num_threads > 0:
        # Cap intra-op parallelism so a pool of N children does not oversubscribe
        # the box (the whole preprocess is a torch/torchvision multi-thread burst).
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass

    from mstar.model.registry import HF_MODELS, get_model_class

    # Build the SAME lightweight tokenizer/processor instance the serve process
    # uses, so preprocess_tensors is byte-identical.
    model = get_model_class(model_name)(
        model_path_hf=HF_MODELS.get(model_name, {}).get("model_path_hf", ""),
        cache_dir=cache_dir,
        **(model_kwargs or {}),
    )
    logger.info("mstar_preproc ready (model=%s, pid=%d)", model_name, os.getpid())

    parent_pid = os.getppid()
    processed = 0

    while True:
        try:
            item = in_queue.get(timeout=0.2)
        except queue.Empty:
            # Parent-death watch: getppid() flips to the reaper when the serve
            # process is gone. Exit so we don't linger as an orphan.
            if os.getppid() != parent_pid:
                logger.warning("mstar_preproc: parent gone; exiting")
                break
            continue
        if item is None:
            logger.info("mstar_preproc: sentinel received; exiting")
            break
        index, pinput = item
        try:
            tensors, input_metadata = preprocess_tensors(model, pinput, "cpu")
            out_queue.put((index, "ok", _tensors_to_wire(tensors), input_metadata))
        except Exception as e:  # noqa: BLE001 — reported back, retried inline
            logger.exception(
                "mstar_preproc: preprocessing failed for rid=%s; reporting error "
                "(serve process will retry inline)", pinput.request_id,
            )
            out_queue.put((index, "err", repr(e), None))
        processed += 1

    logger.info("mstar_preproc exiting: processed=%d", processed)


# ---------------------------------------------------------------------------
# Serve-side client (used only from the data-worker thread)
# ---------------------------------------------------------------------------

# A completion returned by get_ready():
#   ("ok", input, tensors, input_metadata)  -> worker runs _admit_request
#   ("inline", input)                       -> worker runs the full inline path
Completion = tuple


class PreprocClient:
    """Serve-side handle to the preprocessing pool.

    All methods run on the single data-worker thread (submit / get_ready /
    cancel_rid / forget_rid / shutdown); children talk only through the queues.
    No lock is needed — the serve-side state is single-threaded, and the
    children are separate processes.
    """

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str | None,
        model_kwargs: dict | None,
        num_procs: int,
        num_threads: int,
        log_level: str = "INFO",
    ):
        self._n = max(1, num_procs)
        ctx = mp.get_context("spawn")
        # One shared out-queue (unbounded: children must never block returning).
        self._out: mp.Queue = ctx.Queue()
        self._in: list[mp.Queue] = []
        self._procs: list = []
        for i in range(self._n):
            inq: mp.Queue = ctx.Queue(maxsize=PREPROC_IN_MAXSIZE)
            proc = ctx.Process(
                target=run_preproc_proc,
                kwargs=dict(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    model_kwargs=model_kwargs or {},
                    in_queue=inq,
                    out_queue=self._out,
                    num_threads=num_threads,
                    log_level=log_level,
                ),
                daemon=True,
                name=f"mstar_preproc_{i}",
            )
            proc.start()
            self._in.append(inq)
            self._procs.append(proc)

        # Submit-order emission + exactly-once fallback bookkeeping.
        self._next_submit = 0            # next index to assign
        self._next_emit = 0              # next index to hand back in order
        self._rr = 0                     # round-robin dispatch cursor
        self._outstanding: "OrderedDict[int, tuple]" = OrderedDict()  # idx -> (cid, input)
        self._done: dict[int, Completion] = {}   # completed, awaiting in-order emit
        # rid -> its live indices (outstanding OR done-but-not-yet-emitted). Lets
        # cancel_rid mark only rids that actually have a job in flight, and lets
        # the cancel marker self-clear once that rid's last job drains (no
        # unbounded set, no forget_rid call needed).
        self._rid_to_indices: dict[str, set[int]] = {}
        self._cancelled: set[str] = set()        # rids aborted before admission
        self._failed = False

        logger.info(
            "MSTAR_PREPROC_PROC: pool spawned (%d procs, pids=%s)",
            self._n, [p.pid for p in self._procs],
        )

    # -- dispatch -----------------------------------------------------------

    def _all_alive(self) -> bool:
        return all(p.is_alive() for p in self._procs)

    def submit(self, input: PreprocessInput) -> bool:
        """Dispatch a preprocessing job. Returns False if the pool is
        unavailable (the caller must preprocess inline instead)."""
        if self._failed:
            return False
        if not self._all_alive():
            self._fail()
            return False
        idx = self._next_submit
        cid = self._rr
        try:
            self._in[cid].put_nowait((idx, input))
        except queue.Full:
            logger.critical(
                "MSTAR_PREPROC_PROC: child %d work queue full (%d) — falling "
                "back to inline preprocessing permanently", cid, PREPROC_IN_MAXSIZE,
            )
            self._fail()
            return False
        self._outstanding[idx] = (cid, input)
        self._rid_to_indices.setdefault(input.request_id, set()).add(idx)
        self._next_submit += 1
        self._rr = (self._rr + 1) % self._n
        return True

    def cancel_rid(self, request_id: str) -> None:
        """Mark a rid aborted IFF it still has a job in flight in the pool. Its
        not-yet-admitted completion is then dropped in get_ready (never sent to
        the conductor), so an abort that races ahead of admission cannot leave
        the request admitted-but-un-aborted. A no-op when the request was already
        admitted (no live job) — the normal abort path handles that case, and the
        marker would otherwise leak. The marker self-clears when the job drains."""
        if request_id in self._rid_to_indices:
            self._cancelled.add(request_id)

    def _release_index(self, idx: int, request_id: str) -> None:
        """A completion has been emitted or dropped: retire its index and, once
        the rid has no live job left, clear any cancel marker for it."""
        s = self._rid_to_indices.get(request_id)
        if s is not None:
            s.discard(idx)
            if not s:
                del self._rid_to_indices[request_id]
                self._cancelled.discard(request_id)

    # -- completion (polled by the worker run loop) -------------------------

    def get_ready(self) -> list[Completion]:
        """Drain child results and return the contiguous in-submit-order run of
        completions ready to admit. Recovers outstanding jobs inline on failure.
        Cancelled rids are consumed (advance the order) but not returned."""
        # 1. Drain the shared out-queue into the reorder buffer.
        while True:
            try:
                idx, status, payload, meta = self._out.get_nowait()
            except queue.Empty:
                break
            if idx not in self._outstanding:
                # Already recovered inline (fallback) — drop, so we never admit
                # the same request twice.
                continue
            _cid, input = self._outstanding.pop(idx)
            if status == "ok":
                self._done[idx] = ("ok", input, _tensors_from_wire(payload), meta)
            else:
                # Child errored on this job: retry inline on the worker thread so
                # the outcome is identical to the flag-off path (a deterministic
                # bad input reproduces and is logged; nothing double-admits).
                self._done[idx] = ("inline", input)

        # 2. A dead child is a permanent failure.
        if not self._failed and not self._all_alive():
            self._fail()

        # 3. On failure, recover every still-outstanding job inline, in order.
        if self._failed and self._outstanding:
            for idx in sorted(self._outstanding):
                _cid, input = self._outstanding[idx]
                self._done[idx] = ("inline", input)
            self._outstanding.clear()

        # 4. Emit the contiguous ready run in submit order.
        out: list[Completion] = []
        while self._next_emit in self._done:
            idx = self._next_emit
            comp = self._done.pop(idx)
            self._next_emit += 1
            rid = comp[1].request_id
            cancelled = rid in self._cancelled
            self._release_index(idx, rid)
            if cancelled:
                # Aborted before admission: consume the slot but do not admit.
                continue
            out.append(comp)

        # 4b. After failure there is no more in-flight work to preserve order
        # against; flush whatever remains (a queue-full skip can leave a gap).
        if self._failed and self._done:
            for idx in sorted(self._done):
                comp = self._done.pop(idx)
                rid = comp[1].request_id
                cancelled = rid in self._cancelled
                self._release_index(idx, rid)
                if not cancelled:
                    out.append(comp)
            self._next_emit = self._next_submit
        return out

    # -- failure / lifecycle ------------------------------------------------

    def _fail(self) -> None:
        if self._failed:
            return
        self._failed = True
        logger.critical(
            "MSTAR_PREPROC_PROC: pool unavailable — falling back to inline "
            "preprocessing permanently (%d job(s) recovered inline)",
            len(self._outstanding),
        )

    def shutdown(self, timeout: float = 5.0) -> None:
        self._failed = True
        for inq in self._in:
            try:
                inq.put_nowait(None)  # sentinel: clean child exit
            except Exception:
                pass
        for proc in self._procs:
            if proc.is_alive():
                proc.join(timeout=timeout)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.kill()
