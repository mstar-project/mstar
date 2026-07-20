"""Tensor transport over a shared-memory arena.

``ArenaShmCommunicationManager`` replaces the file transport's per-tensor
open/write/read/unlink with a Rust segmented ``/dev/shm`` arena (persistent
mmaps + first-fit coalescing allocator, vendored in ``rust/``). Producer:
``register_for_send`` reserves a slot and D2H-copies into the segment on the
copy stream; the ``(segment, offset)`` rides the existing descriptors.
Consumer: ``start_read_tensors`` maps the named segment once, reads
zero-copy, H2D-copies on the copy stream; the producer reclaims on ACK,
gated by a CUDA-event future so an edge is never ACKed before its copies
land. Segments are mapped once and never move, so the one-time
``cudaHostRegister`` per segment (within ``MSTAR_SHM_ARENA_PIN_MAX_MB``)
holds for its lifetime and keeps the side-stream copies truly async.

Capacity degrades in layers: grow by segments up to
``MSTAR_SHM_ARENA_MAX_SEGMENTS``; at the cap, briefly backpressure for
consumer ACKs; then spill the tensor to the per-uuid file protocol
(``MSTAR_SHM_ARENA_SPILL``, default on) — slower, never fails, like the old
transport at saturation. ``stats_summary()`` exposes occupancy and the
fragmentation gauge (largest contiguous free block); ``--log-stats`` logs it
periodically.

Ceilings are PER-ENTITY and multiply across a node: with E entities
(workers + the api-server data worker), /dev/shm demand can reach
``MAX_SEGMENTS x SEGMENT_MB x E`` and pinned host RAM approx
``PIN_MAX_MB x E`` (consumers pin peer segments too, so one process can pin
more than its own arena holds). Construction fails fast when one entity's
ceiling already exceeds /dev/shm, and warns when it exceeds current free
space or when the pin budget is an outsized share of physical RAM.

Selection: ``create_tensor_communication_manager`` picks this manager for
the SHM protocol when ``MSTAR_SHM_ARENA`` is ``1`` (require) or ``AUTO``
(use if the ``mstar_rust`` extension imports); default ``0`` keeps the file
transport. See :doc:`environment_variables` for all knobs.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import weakref
from concurrent.futures import Future

import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import (
    FutureAndPointers,
    SharedMemoryCommunicationManager,
    _deserialize_tensor,
    _nullcontext,
    _serialize_tensor,
)
from mstar.graph.base import GraphEdge, TensorPointerInfo

logger = logging.getLogger(__name__)

_CUDA_HOST_ALREADY_REGISTERED = 712


class _CudaEventFuture:
    """Future-shaped CUDA event. An edge whose H2D copies ride behind this
    event reports ready only once the copies have completed on the device —
    so the ACK that lets the producer reclaim the arena slot is deferred by
    ``get_ready_tensors``'s existing future polling instead of a blocking
    host synchronize."""

    def __init__(self, stream):
        # blocking=True: `synchronize` SLEEPS until the event fires. The
        # default (False) busy-waits — the watcher thread that turns these
        # into wake futures would burn a full core per wait.
        self._event = torch.cuda.Event(blocking=True)
        self._event.record(stream)

    def done(self) -> bool:
        return self._event.query()

    def result(self) -> None:
        self._event.synchronize()


_CUDART = None


def _cudart():
    """libcudart via ctypes: unlike the torch binding, a ctypes call
    RELEASES the GIL, so registering a 256 MiB segment (tens of ms) on the
    send path cannot stall the process's other Python threads (serve loop,
    stream relays, the wake watcher)."""
    global _CUDART
    if _CUDART is None:
        import ctypes
        import ctypes.util

        name = ctypes.util.find_library("cudart") or "libcudart.so"
        lib = ctypes.CDLL(name)
        lib.cudaHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.cudaHostRegister.restype = ctypes.c_int
        _CUDART = lib
    return _CUDART


def _unlink_paths(paths: list) -> None:
    """Exit-time segment unlink (weakref.finalize target — must not
    reference the manager). Workers exit with the manager still alive, so
    the interpreter never garbage-collects it and the Rust Drop (which
    unlinks) never runs; without this, every worker run leaks its full
    segments in /dev/shm until swept by a later start."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _unpin(ptr: int) -> bool:
    """cudaHostUnregister a previously pinned mapping (GIL released via
    ctypes). Returns success."""
    if not torch.cuda.is_available():
        return False
    try:
        lib = _cudart()
        import ctypes

        lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        lib.cudaHostUnregister.restype = ctypes.c_int
        rc = lib.cudaHostUnregister(ptr)
    except OSError:
        rc = torch.cuda.cudart().cudaHostUnregister(ptr)
    return rc == 0


def _pin(ptr: int, nbytes: int) -> bool:
    """cudaHostRegister a mapped segment (idempotent). Returns success.
    GIL released for the duration (see ``_cudart``)."""
    if not torch.cuda.is_available():
        return False
    try:
        rc = _cudart().cudaHostRegister(ptr, nbytes, 0)
    except OSError:  # no loadable libcudart: fall back to the torch binding
        rc = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
    ok = rc in (0, _CUDA_HOST_ALREADY_REGISTERED)
    if not ok:
        logger.warning("cudaHostRegister(%#x, %d) failed rc=%d — copies fall "
                       "back to pageable bandwidth", ptr, nbytes, rc)
    return ok


class ArenaShmCommunicationManager(SharedMemoryCommunicationManager):
    """Tensor transport via the Rust shared-memory arena (``mstar_rust``)."""

    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        device: str,
        communicator: BaseCommunicator,
        shm_dir: str | None = None,
        enable_prof: bool = False,
    ):
        super().__init__(
            my_entity_id=my_entity_id, hostname=hostname, device=device,
            communicator=communicator, shm_dir=shm_dir,
            enable_prof=enable_prof,
        )
        from mstar_rust import SegmentedShmArena, ShmArena

        self._ShmArena = ShmArena
        segment_mb = int(os.getenv("MSTAR_SHM_ARENA_SEGMENT_MB", "256"))
        max_segments = int(os.getenv("MSTAR_SHM_ARENA_MAX_SEGMENTS", "32"))
        self._full_timeout_s = float(
            os.getenv("MSTAR_SHM_ARENA_FULL_TIMEOUT_S", "30"))
        self._pin_segments = (
            os.getenv("MSTAR_SHM_ARENA_PIN", "1") == "1"
            and torch.cuda.is_available() and str(device) != "cpu"
        )
        # Pinned host memory is a system-wide resource (pages come out of the
        # OS's pageable pool), so it gets its own budget, distinct from the
        # segment cap. Segments past the budget stay unpinned — copies still
        # work, they just lose async overlap. Oversized dedicated segments
        # (single allocations larger than a segment) are never pinned: a
        # one-shot transfer doesn't amortize the registration cost.
        self._pin_budget = int(
            os.getenv("MSTAR_SHM_ARENA_PIN_MAX_MB", "4096")) << 20
        self._pinned_bytes = 0
        self._pin_budget_warned = False
        self._segment_bytes = segment_mb << 20
        # Arena saturation spills to the per-uuid file transport (the old
        # protocol) instead of failing: bursty or oversized workloads degrade
        # to file-copy speed, matching the prior manager's "slower, never
        # fails" behavior. MSTAR_SHM_ARENA_SPILL=0 restores strict
        # backpressure + timeout.
        self._spill = os.getenv("MSTAR_SHM_ARENA_SPILL", "1") == "1"
        # Default 0: on a worker, TENSOR_RECEIVED ACKs (which free slots)
        # are processed by the SAME thread that would sit in this grace
        # wait, so waiting is pure dead time there — spill immediately.
        # Deployments where another thread drains the communicator (the
        # threaded api_server) can set a small grace to ride out bursts.
        self._spill_after_s = float(
            os.getenv("MSTAR_SHM_ARENA_SPILL_AFTER_S", "0"))
        self._frag_warned = False
        # h2d completion watcher: turns a CUDA event into a real Future so
        # the worker's eventfd wakes the moment reads finish (no 10 ms tick).
        self._wake_q: queue.Queue | None = None
        self._wake_lock = threading.Lock()
        # Serializes pin accounting and peer-map insertion: _pin releases
        # the GIL (ctypes), so unlocked read-modify-write of _pinned_bytes
        # tears, and check-then-insert on _peer_segments can map+pin the
        # same segment twice (the loser leaking forever).
        self._pin_lock = threading.Lock()
        # Concurrent start_read_tensors calls (threaded api-server): used to
        # keep eviction from unmapping a segment another thread is copying
        # from before its future lands in `pending`.
        self._reads_active = 0
        # Periodic occupancy/fragmentation logging, tied to --log-stats
        # (enable_prof) and time-gated.
        self._stats_interval_s = float(
            os.getenv("MSTAR_SHM_ARENA_STATS_INTERVAL_S", "60"))
        self._stats_last = 0.0
        # CEILINGS ARE PER-ENTITY and multiply across a node: every entity
        # (workers + the api-server data worker) creates its own arena, so
        # node /dev/shm demand can reach
        #     MAX_SEGMENTS x SEGMENT_MB x num_entities
        # and node pinned RAM approx PIN_MAX_MB x num_entities (a consumer
        # pins peer segments too, so one process's pinned bytes can exceed
        # its own arena). Check the static tmpfs ceiling NOW instead of
        # surfacing as ENOSPC on a growth mid-run.
        per_entity_max = (segment_mb << 20) * max_segments
        try:
            st = os.statvfs("/dev/shm")
            shm_total = st.f_frsize * st.f_blocks
            shm_avail = st.f_frsize * st.f_bavail
        except OSError:
            shm_total = shm_avail = None
        self._shm_total = shm_total
        if shm_total is not None:
            if per_entity_max > shm_total:
                raise RuntimeError(
                    f"SHM arena ceiling for ONE entity "
                    f"({per_entity_max >> 20} MiB = "
                    f"MSTAR_SHM_ARENA_MAX_SEGMENTS x _SEGMENT_MB) exceeds "
                    f"/dev/shm ({shm_total >> 20} MiB) — and every entity "
                    f"multiplies this. Lower the knobs or grow tmpfs.")
            if per_entity_max > shm_avail:
                logger.warning(
                    "ARENA: this entity's ceiling (%d MiB) exceeds current "
                    "/dev/shm free space (%d MiB); ceilings are per-entity "
                    "and multiply across workers — growth may hit ENOSPC "
                    "under load", per_entity_max >> 20, shm_avail >> 20)
        try:
            phys = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            phys = None
        if (self._pin_segments and phys is not None
                and self._pin_budget > phys // 4):
            logger.warning(
                "ARENA: MSTAR_SHM_ARENA_PIN_MAX_MB (%d MiB) exceeds a "
                "quarter of physical RAM (%d MiB) FOR ONE ENTITY; pinned "
                "budgets multiply across entities and come out of the OS's "
                "pageable pool", self._pin_budget >> 20, phys >> 20)
        # INSTANCE-UNIQUE base name: a fixed per-entity name in the global
        # /dev/shm namespace collides across servers (a second server's
        # create() truncates the first's live segments — silent corruption)
        # and across users (permission-denied at startup). pid + random
        # token makes each instance's names unique; wire-compatibility is
        # free because consumers open whatever segment name the descriptor
        # carries. The pid embedded in the name also enables the orphan
        # sweep below.
        import secrets

        self._sweep_orphans()
        base = (f"mstar_arena_{my_entity_id}_{os.getpid()}_"
                f"{secrets.token_hex(4)}")
        self._arena = SegmentedShmArena.create(
            base, segment_mb << 20, max_segments)
        # Guaranteed exit-time unlink: weakref.finalize runs at interpreter
        # shutdown even when the manager is still referenced (the worker
        # case — no explicit cleanup path runs there, so the Rust Drop
        # never fires). The callback captures only the mutable name list,
        # which _sync_segments extends as the arena grows.
        self._own_segment_paths: list[str] = []
        self._finalizer = weakref.finalize(
            self, _unlink_paths, self._own_segment_paths)
        # Producer-side segment views (memoryviews are stable: segments
        # never move or resize) + how many segments are already pinned.
        self._seg_views: list[memoryview] = []
        self._pinned_segments = 0
        self._sync_segments()
        # uuid -> (segment_idx, offset) for sender-side reclaim.
        # (register_for_send receives the TensorPointerInfos directly and
        # stamps them in place — no side-table needed.)
        self._arena_locs: dict[str, tuple[int, int]] = {}
        # uuid -> stage time, for the TTL backstop: a request aborted after
        # staging but before every consumer ACKs defers reclaim forever
        # (cleanup_request waits for ACKs that will never come). A slot
        # older than the REQUEST timeout cannot have a legitimate reader —
        # the request is dead by contract — so freeing past a bound safely
        # above it cannot race a real consumer. Default OFF pending review
        # discussion; enable with MSTAR_SHM_ARENA_SLOT_TTL_S (recommend
        # >= 2x the request timeout).
        self._arena_ts: dict[str, float] = {}
        self._slot_ttl_s = float(
            os.getenv("MSTAR_SHM_ARENA_SLOT_TTL_S", "0"))
        self._ttl_reclaimed_total = 0
        # Consumer-side: peer segment name -> (arena, memoryview,
        # pinned_nbytes) — pinned_nbytes 0 when the segment wasn't pinned.
        # Entries are EVICTED (unpin + unmap) once the backing file is gone
        # (producer finished or restarted): with instance-unique names a
        # restarting producer mints new names every generation, so a
        # never-evicting cache would grow mappings and pinned bytes without
        # bound on any long-lived consumer.
        self._peer_segments: dict[str, tuple[object, memoryview, int]] = {}
        self._peer_evict_last = 0.0

    # -- segments --------------------------------------------------------

    def _sweep_orphans(self) -> None:
        """A SIGKILLed server never runs Drop, orphaning up to its full
        arena in /dev/shm until reboot. Names embed the owning pid, so a
        startup sweep can reclaim any segment whose owner is gone. Files we
        cannot judge (foreign naming) or cannot remove (another user's)
        are left with a debug note."""
        try:
            names = os.listdir("/dev/shm")
        except OSError:
            return
        for name in names:
            if not name.startswith("mstar_arena_"):
                continue
            parts = name.split(".")[0].rsplit("_", 2)
            if len(parts) != 3 or not parts[1].isdigit():
                continue   # pre-uniquification or foreign naming: skip
            if os.path.exists(f"/proc/{parts[1]}"):
                continue   # owner alive
            try:
                os.unlink(f"/dev/shm/{name}")
                logger.info("ARENA: swept orphaned segment %s "
                            "(owner pid %s is gone)", name, parts[1])
            except OSError as e:
                logger.debug("ARENA: cannot sweep %s: %s", name, e)

    def _maybe_pin(self, ptr: int, nbytes: int) -> int:
        """Pin within budget; returns the bytes actually pinned (0 if
        skipped/failed). Runs under _pin_lock: _pin releases the GIL, so
        budget check + register + accounting must be one atomic unit."""
        # Oversized dedicated segments ARE pinned (within budget): freed
        # segments are reused for later large tensors, so the registration
        # amortizes over the segment's lifetime, not one transfer.
        if not self._pin_segments:
            return 0
        with self._pin_lock:
            return self._pin_locked(ptr, nbytes)

    def _pin_locked(self, ptr: int, nbytes: int) -> int:
        if self._pinned_bytes + nbytes > self._pin_budget:
            if not self._pin_budget_warned:
                logger.warning(
                    "ARENA: pinned-memory budget reached (%d MiB, "
                    "MSTAR_SHM_ARENA_PIN_MAX_MB); further segments stay "
                    "unpinned — copies work but lose async overlap",
                    self._pin_budget >> 20)
                self._pin_budget_warned = True
            return 0
        if _pin(ptr, nbytes):
            self._pinned_bytes += nbytes
            self._pinned_segments += 1
            return nbytes
        return 0

    def _sync_segments(self) -> None:
        grew = False
        while len(self._seg_views) < self._arena.num_segments:
            i = len(self._seg_views)
            seg = self._arena.segment(i)
            self._own_segment_paths.append(
                f"/dev/shm/{self._arena.segment_name(i)}")
            self._seg_views.append(memoryview(seg))
            self._maybe_pin(*seg.ptr_len())
            grew = True
        if grew:
            total, free, largest = self._arena.stats()
            if (self._shm_total is not None
                    and total > self._shm_total * 0.8):
                logger.warning(
                    "ARENA: this entity's segments now total %d MiB — over "
                    "80%% of /dev/shm (%d MiB). Oversized dedicated "
                    "segments grow past the static ceiling; other entities "
                    "multiply this further.",
                    total >> 20, self._shm_total >> 20)
            logger.info(
                "ARENA: grew to %d segments (%d MiB total, %d MiB free, "
                "largest free block %d MiB, %d MiB pinned)",
                self._arena.num_segments, total >> 20, free >> 20,
                largest >> 20, self._pinned_bytes >> 20)

    def stats_summary(self) -> dict:
        """Occupancy/fragmentation snapshot (named apart from the raw
        ``SegmentedShmArena.stats`` tuple). The fragmentation signature is
        `largest_free_block` collapsing while `free_bytes` stays high."""
        total, free, largest = self._arena.stats()
        return {
            "segments": self._arena.num_segments,
            "total_bytes": total,
            "free_bytes": free,
            "largest_free_block": largest,
            "pinned_bytes": self._pinned_bytes,
        }

    def _reclaim_expired(self) -> int:
        """TTL backstop for abort-orphaned slots (see _arena_ts). Returns
        the number of slots/files reclaimed."""
        if not self._slot_ttl_s:
            return 0
        now = time.monotonic()
        n = 0
        for uuid, ts in list(self._arena_ts.items()):
            if now - ts < self._slot_ttl_s:
                continue
            self._arena_ts.pop(uuid, None)
            if (loc := self._arena_locs.pop(uuid, None)) is not None:
                self._arena.free(*loc)
                n += 1
            if (path := self._shm_files.pop(uuid, None)) is not None:
                try:
                    os.unlink(path)
                    n += 1
                except FileNotFoundError:
                    pass
        if n:
            self._ttl_reclaimed_total += n
            logger.warning(
                "ARENA: TTL-reclaimed %d slot(s)/file(s) older than %.0fs "
                "(%d total) — requests aborted without consumer ACKs; "
                "the ACK path is leaking",
                n, self._slot_ttl_s, self._ttl_reclaimed_total)
        return n

    def _maybe_log_stats(self) -> None:
        """Under ``--log-stats`` (enable_prof), log the snapshot at most
        once per MSTAR_SHM_ARENA_STATS_INTERVAL_S so a long soak leaves an
        occupancy/fragmentation time series in the logs."""
        if not self.enable_prof:
            return
        now = time.monotonic()
        if now - self._stats_last < self._stats_interval_s:
            return
        self._stats_last = now
        self._reclaim_expired()
        st = self.stats_summary()
        # live_slots/spill_files climbing while requests finish = reclaim
        # being deferred (e.g. aborts without ACKs) — the soak's leak canary.
        st["live_slots"] = len(self._arena_locs)
        st["spill_files"] = len(self._shm_files)
        logger.info("ARENA stats: %s", st)

    def _peer_view(self, segment_name: str) -> memoryview:
        entry = self._peer_segments.get(segment_name)
        if entry is None:
            with self._pin_lock:
                entry = self._peer_segments.get(segment_name)
                if entry is None:   # lost the race: another thread mapped it
                    arena = self._ShmArena.open(segment_name)
                    pinned = (self._pin_locked(*arena.ptr_len())
                              if self._pin_segments else 0)
                    entry = self._peer_segments[segment_name] = (
                        arena, memoryview(arena), pinned)
        return entry[1]

    def _evict_dead_peers(self) -> None:
        """Drop cached peer segments whose backing file is gone. Gated on
        `self.pending` being empty: an mmap must outlive any in-flight h2d
        copy that reads from it, and pending futures are exactly those
        copies. Time-gated to keep the existence checks off the hot path."""
        now = time.monotonic()
        # _reads_active > 1: another thread may have queued an async H2D
        # copy whose future hasn't reached `pending` yet — unmapping under
        # it would be a use-after-free. Only the sole active reader evicts.
        if (self.pending or self._reads_active > 1
                or now - self._peer_evict_last < 10.0):
            return
        self._peer_evict_last = now
        for name in list(self._peer_segments):
            if os.path.exists(f"/dev/shm/{name}"):
                continue
            with self._pin_lock:
                arena, view, pinned = self._peer_segments[name]
                if pinned:
                    # Unregister BEFORE unmapping: tearing down a mapping
                    # CUDA still holds registered leaves a dangling pinned
                    # range. On failure keep the entry (retry next sweep)
                    # so accounting stays truthful.
                    ptr, _len = arena.ptr_len()
                    if not _unpin(ptr):
                        logger.warning(
                            "ARENA: cudaHostUnregister failed for dead peer "
                            "%s; keeping mapping until it succeeds", name)
                        continue
                    self._pinned_bytes -= pinned
                del self._peer_segments[name]
            view.release()
            logger.debug("ARENA: evicted dead peer segment %s "
                         "(%d MiB pinned released)", name, pinned >> 20)

    # -- producer ---------------------------------------------------------

    def _reserve(self, nbytes: int) -> tuple[int, int] | None:
        """Reserve with layered degradation. At the segment cap: wait
        briefly for consumer ACKs to free space; then (default) return None
        so the caller SPILLS this tensor to the per-uuid file transport —
        slower, never fails, exactly the old manager's saturation behavior.
        MSTAR_SHM_ARENA_SPILL=0 keeps strict backpressure for the full
        timeout, then raises."""
        grace = self._spill_after_s if self._spill else self._full_timeout_s
        deadline = time.monotonic() + grace
        warned = False
        while True:
            try:
                seg, off = self._arena.reserve(max(nbytes, 1))
            except RuntimeError:
                if self._reclaim_expired():
                    continue   # expired slots freed: retry the reserve
                total, free, largest = self._arena.stats()
                if free >= nbytes and not self._frag_warned:
                    # The fragmentation signature: enough TOTAL free space,
                    # but no contiguous block large enough.
                    logger.warning(
                        "ARENA: fragmentation — need %d bytes with %d free "
                        "but largest free block is %d (of %d total). "
                        "Consider larger MSTAR_SHM_ARENA_SEGMENT_MB.",
                        nbytes, free, largest, total)
                    self._frag_warned = True
                if time.monotonic() > deadline:
                    if self._spill:
                        return None
                    raise RuntimeError(
                        f"SHM arena full for >{self._full_timeout_s}s "
                        f"({self._arena.num_segments} segments); raise "
                        "MSTAR_SHM_ARENA_MAX_SEGMENTS / _SEGMENT_MB or check "
                        "for consumers not ACKing") from None
                if not warned:
                    logger.warning(
                        "SHM arena at capacity; backpressuring sends until "
                        "consumers ACK%s",
                        " (then spilling to file)" if self._spill else "")
                    warned = True
                time.sleep(0.002)
                continue
            self._sync_segments()
            return seg, off

    def register_for_send(
        self, request_id: str, tensor_infos: list[TensorPointerInfo],
        skip_cuda_sync: bool = False,
    ):
        if not skip_cuda_sync and torch.cuda.is_available():
            torch.cuda.default_stream().synchronize()
        ctx = (
            torch.cuda.stream(self._d2h_stream)
            if self._d2h_stream is not None
            else _nullcontext()
        )
        queued = False
        self._maybe_log_stats()
        with ctx:
            for info_arg in tensor_infos:
                uuid = info_arg.uuid
                if self.tensor_store.is_registered(request_id, uuid):
                    continue
                tensor = self.tensor_store.get_tensor(request_id, uuid)
                t0 = time.perf_counter()
                t = tensor.detach().contiguous()
                nbytes = t.numel() * t.element_size()
                loc = self._reserve(nbytes)
                if loc is not None and self.tensor_store.is_registered(
                        request_id, uuid):
                    # Lost a concurrent-duplicate race: another thread
                    # registered this uuid while our reserve released the
                    # GIL. Return our slot instead of orphaning it.
                    self._arena.free(*loc)
                    continue
                if loc is None:
                    # Arena saturated: spill THIS tensor to the per-uuid file
                    # protocol (infos keep shm_segment=None — the consumer
                    # falls back to the file read for exactly those).
                    data = _serialize_tensor(t)
                    path = self._shm_path(self.my_entity_id, uuid)
                    with open(path, "wb") as f:
                        f.write(data)
                    self._shm_files[uuid] = path
                    self._arena_ts[uuid] = time.monotonic()
                    self.tensor_store.set_metadata(
                        request_id, uuid, mem_registered=True)
                    if self.enable_prof:
                        self._record_tx(request_id, uuid, len(data),
                                        time.perf_counter() - t0)
                    logger.debug("ARENA: spilled %s to %s (%d bytes)",
                                 uuid, path, len(data))
                    continue
                seg, off = loc
                try:
                    if nbytes:
                        host = torch.frombuffer(
                            self._seg_views[seg][off:off + nbytes],
                            dtype=torch.uint8,
                        ).view(t.dtype).reshape(t.shape)
                        # Async D2H into the pinned segment when a copy
                        # stream exists (one sync below covers the batch);
                        # blocking otherwise, so the descriptor can never
                        # ship ahead of the bytes.
                        host.copy_(
                            t, non_blocking=self._d2h_stream is not None)
                        queued = True
                    self._arena_locs[uuid] = (seg, off)
                    self._arena_ts[uuid] = time.monotonic()
                except BaseException:
                    # Anything that unwinds between reserve and the
                    # _arena_locs record would orphan the slot forever
                    # (cleanup can only free what is recorded).
                    self._arena.free(seg, off)
                    raise
                seg_name = self._arena.segment_name(seg)
                info_arg.shm_segment = seg_name
                info_arg.shm_offset = off
                self.tensor_store.set_metadata(
                    request_id, uuid, mem_registered=True)
                if self.enable_prof:
                    self._record_tx(
                        request_id, uuid, nbytes, time.perf_counter() - t0)
                logger.debug("ARENA: staged %s at %s+%d (%d bytes)",
                             uuid, seg_name, off, nbytes)
        if queued and self._d2h_stream is not None:
            # The control message referencing these bytes is sent after we
            # return; the consumer must never observe a partial copy.
            self._d2h_stream.synchronize()

    # -- consumer ---------------------------------------------------------

    def start_read_tensors(
        self, request_id: str, graph_edges: list[GraphEdge],
        graph_walk: str | None = None,
    ):
        # Increment races are benign here: a torn count can only make
        # eviction OVER-cautious (skip a sweep), never unsafe.
        self._reads_active += 1
        try:
            return self._start_read_tensors(request_id, graph_edges,
                                            graph_walk)
        finally:
            self._reads_active -= 1

    def _start_read_tensors(self, request_id, graph_edges, graph_walk):
        self._evict_dead_peers()
        h2d_did_work = False
        read_edges: list[tuple[GraphEdge, float]] = []
        ctx = (
            torch.cuda.stream(self._h2d_stream)
            if self._h2d_stream is not None
            else _nullcontext()
        )
        with ctx:
            for graph_edge in graph_edges:
                if len(graph_edge.tensor_info) == 0:
                    continue
                rx_t0 = time.perf_counter()
                for info in graph_edge.tensor_info:
                    if info.source_entity == self.my_entity_id:
                        self._slice_existing_tensor(
                            request_id=request_id, name=graph_edge.name,
                            next_node=graph_edge.next_node,
                            graph_walk=graph_walk, info=info,
                        )
                        self.tensor_store.increment_ref(
                            request_id, info.uuid, 1)
                        continue
                    if self.tensor_store.check_uuid_presence(
                            request_id, info.uuid):
                        self.tensor_store.increment_ref(
                            request_id, info.uuid, 1)
                        continue
                    if info.shm_segment is None:
                        # Spilled at the producer (arena saturated): read the
                        # per-uuid file, the old protocol's path.
                        path = self._shm_path(info.source_entity, info.uuid)
                        with open(path, "rb") as f:
                            f.seek(info.offset)
                            data = f.read(info.nbytes)
                        tensor = _deserialize_tensor(
                            data, self.device, tensor_info=info)
                    else:
                        tensor = self._read_from_arena(info)
                    h2d_did_work = h2d_did_work or tensor.numel() > 0
                    self.tensor_store.put_tensor(
                        request_id, info.uuid, tensor)
                    self.tensor_store.set_metadata(
                        request_id, info.uuid, mem_registered=False)
                    # +1 transit (released by get_ready_tensors), +1 usage
                    # (released by _cleanup_consumed_inputs).
                    self.tensor_store.increment_ref(request_id, info.uuid, 2)
                read_edges.append(
                    (graph_edge, time.perf_counter() - rx_t0))
        future = None
        if h2d_did_work and self._h2d_stream is not None:
            # Downstream kernels see the data (device-side ordering only).
            torch.cuda.default_stream(self.device).wait_stream(
                self._h2d_stream)
            # The producer reclaims the slot when an edge is ACKed, and the
            # source is its live mapping — so the edge must not report ready
            # until its copies have completed (the file path's f.read() made
            # this implicit). One event covers the batch: all copies were
            # queued on the h2d stream in program order. get_ready_tensors
            # polls it — no host block here.
            future = _CudaEventFuture(self._h2d_stream)
        for graph_edge, rx_time in read_edges:
            self.pending.append(
                FutureAndPointers(
                    future=future, graph_edges=[graph_edge],
                    request_id=request_id, rx_time=rx_time,
                )
            )
        if future is None:
            return []
        # A real Future for the worker's eventfd (EventWakeup.register_
        # futures needs add_done_callback): completed by the watcher thread
        # the moment the h2d copies finish, so an otherwise-idle worker
        # re-checks get_ready_tensors immediately instead of on its next
        # poll tick.
        return [self._wake_when_done(future)]

    def _wake_when_done(self, cuda_future) -> Future:
        with self._wake_lock:   # two callers must not race the create
            if self._wake_q is None:
                self._wake_q = queue.Queue()
                # staticmethod target closing over ONLY the queue: a bound
                # method would pin the whole manager (and its arena — so
                # segments never unlink) for the daemon thread's lifetime.
                threading.Thread(
                    target=self._watch_wakes, args=(self._wake_q,),
                    daemon=True,
                    name=f"arena-h2d-wake-{self.my_entity_id}").start()
        fut: Future = Future()
        self._wake_q.put((cuda_future, fut))
        return fut

    @staticmethod
    def _watch_wakes(q: queue.Queue) -> None:
        # Events are queued in stream order, so sequential waits are exact.
        # None is the close() sentinel.
        while True:
            item = q.get()
            if item is None:
                return
            cuda_future, fut = item
            try:
                cuda_future.result()   # event.synchronize (GIL released)
                fut.set_result(None)
            except Exception:          # noqa: BLE001 — wake best-effort;
                # a cancelled/already-resolved future must not kill the
                # watcher (its death silently downgrades every later wake
                # to the poll tick and grows the queue unboundedly).
                if not fut.done():
                    try:
                        fut.set_result(None)
                    except Exception:  # noqa: BLE001
                        pass

    def _read_from_arena(self, info: TensorPointerInfo) -> torch.Tensor:
        if info.nbytes == 0:
            t = torch.empty(
                info.dims, dtype=info.dtype)
            return t.to(self.device) if self.device != "cpu" else t
        view = self._peer_view(info.shm_segment)
        start = info.shm_offset + info.offset  # offset = TP-shard read offset
        flat = torch.frombuffer(
            view[start:start + info.nbytes], dtype=torch.uint8)
        t = flat.view(info.dtype).reshape(info.dims)
        if self.device != "cpu":
            return t.to(self.device, non_blocking=True)
        return t.clone()  # CPU consumer: own the bytes past reclaim

    # -- reclaim ----------------------------------------------------------

    def close(self) -> None:
        """Stop the wake watcher (segments/pins release with the arena's
        Drop once the manager is garbage collected)."""
        with self._wake_lock:
            if self._wake_q is not None:
                self._wake_q.put(None)
                self._wake_q = None

    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        # Grandparent cleanup (refcounts): skip the file manager's unlink.
        super(SharedMemoryCommunicationManager, self)._cleanup_by_uuid(
            request_id, uuid)
        self._arena_ts.pop(uuid, None)
        if (loc := self._arena_locs.pop(uuid, None)) is not None:
            self._arena.free(*loc)
            logger.debug("ARENA: freed %s at %s", uuid, loc)
        if (path := self._shm_files.pop(uuid, None)) is not None:
            try:
                os.unlink(path)   # spilled tensor: reclaim the file
            except FileNotFoundError:
                pass
        if not self.tensor_store.check_uuid_presence(request_id, uuid):
            return
        self.tensor_store.remove_tensor(request_id, uuid)
