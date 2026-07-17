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
transport at saturation. ``stats()`` exposes occupancy and the fragmentation
gauge (largest contiguous free block).

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
        self._event = torch.cuda.Event()
        self._event.record(stream)

    def done(self) -> bool:
        return self._event.query()

    def result(self) -> None:
        self._event.synchronize()


def _pin(ptr: int, nbytes: int) -> bool:
    """cudaHostRegister a mapped segment (idempotent). Returns success."""
    if not torch.cuda.is_available():
        return False
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
        self._spill_after_s = float(
            os.getenv("MSTAR_SHM_ARENA_SPILL_AFTER_S", "0.05"))
        self._frag_warned = False
        # h2d completion watcher: turns a CUDA event into a real Future so
        # the worker's eventfd wakes the moment reads finish (no 10 ms tick).
        self._wake_q: queue.Queue | None = None
        self._arena = SegmentedShmArena.create(
            f"mstar_arena_{my_entity_id}", segment_mb << 20, max_segments)
        # Producer-side segment views (memoryviews are stable: segments
        # never move or resize) + how many segments are already pinned.
        self._seg_views: list[memoryview] = []
        self._pinned_segments = 0
        self._sync_segments()
        # uuid -> (segment_idx, offset) for sender-side reclaim, and
        # uuid -> [TensorPointerInfo] to stamp locations at register time
        # (infos are created in store_and_return_tensor_info but serialize
        # to the wire only after register_for_send).
        self._arena_locs: dict[str, tuple[int, int]] = {}
        self._infos_by_uuid: dict[str, list[TensorPointerInfo]] = {}
        # Consumer-side: peer segment name -> (arena, memoryview).
        self._peer_segments: dict[str, tuple[object, memoryview]] = {}

    # -- segments --------------------------------------------------------

    def _maybe_pin(self, ptr: int, nbytes: int) -> None:
        if not self._pin_segments:
            return
        if nbytes > self._segment_bytes:
            logger.info(
                "ARENA: leaving oversized segment unpinned (%d MiB): a "
                "one-shot transfer doesn't amortize cudaHostRegister",
                nbytes >> 20)
            return
        if self._pinned_bytes + nbytes > self._pin_budget:
            if not self._pin_budget_warned:
                logger.warning(
                    "ARENA: pinned-memory budget reached (%d MiB, "
                    "MSTAR_SHM_ARENA_PIN_MAX_MB); further segments stay "
                    "unpinned — copies work but lose async overlap",
                    self._pin_budget >> 20)
                self._pin_budget_warned = True
            return
        if _pin(ptr, nbytes):
            self._pinned_bytes += nbytes
            self._pinned_segments += 1

    def _sync_segments(self) -> None:
        grew = False
        while len(self._seg_views) < self._arena.num_segments:
            seg = self._arena.segment(len(self._seg_views))
            self._seg_views.append(memoryview(seg))
            self._maybe_pin(*seg.ptr_len())
            grew = True
        if grew:
            total, free, largest = self._arena.stats()
            logger.info(
                "ARENA: grew to %d segments (%d MiB total, %d MiB free, "
                "largest free block %d MiB, %d MiB pinned)",
                self._arena.num_segments, total >> 20, free >> 20,
                largest >> 20, self._pinned_bytes >> 20)

    def stats(self) -> dict:
        """Occupancy/fragmentation snapshot for periodic stats logging.
        The fragmentation signature is `largest_free_block` collapsing while
        `free_bytes` stays high."""
        total, free, largest = self._arena.stats()
        return {
            "segments": self._arena.num_segments,
            "total_bytes": total,
            "free_bytes": free,
            "largest_free_block": largest,
            "pinned_bytes": self._pinned_bytes,
        }

    def _peer_view(self, segment_name: str) -> memoryview:
        entry = self._peer_segments.get(segment_name)
        if entry is None:
            arena = self._ShmArena.open(segment_name)
            self._maybe_pin(*arena.ptr_len())
            entry = self._peer_segments[segment_name] = (
                arena, memoryview(arena))
        return entry[1]

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

    def store_and_return_tensor_info(self, *args, **kwargs):
        infos = super().store_and_return_tensor_info(*args, **kwargs)
        for info_list in infos.values():
            for info in info_list:
                self._infos_by_uuid.setdefault(info.uuid, []).append(info)
        return infos

    def register_for_send(
        self, request_id: str, uuids: list[str],
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
        with ctx:
            for uuid in uuids:
                if self.tensor_store.is_registered(request_id, uuid):
                    continue
                tensor = self.tensor_store.get_tensor(request_id, uuid)
                t0 = time.perf_counter()
                t = tensor.detach().contiguous()
                nbytes = t.numel() * t.element_size()
                loc = self._reserve(nbytes)
                if loc is None:
                    # Arena saturated: spill THIS tensor to the per-uuid file
                    # protocol (infos keep shm_segment=None — the consumer
                    # falls back to the file read for exactly those).
                    data = _serialize_tensor(t)
                    path = self._shm_path(self.my_entity_id, uuid)
                    with open(path, "wb") as f:
                        f.write(data)
                    self._shm_files[uuid] = path
                    self.tensor_store.set_metadata(
                        request_id, uuid, mem_registered=True)
                    if self.enable_prof:
                        self._record_tx(request_id, uuid, len(data),
                                        time.perf_counter() - t0)
                    logger.debug("ARENA: spilled %s to %s (%d bytes)",
                                 uuid, path, len(data))
                    continue
                seg, off = loc
                if nbytes:
                    host = torch.frombuffer(
                        self._seg_views[seg][off:off + nbytes],
                        dtype=torch.uint8,
                    ).view(t.dtype).reshape(t.shape)
                    # Async D2H into the pinned segment; one stream sync
                    # below covers the batch.
                    host.copy_(t, non_blocking=True)
                    queued = True
                self._arena_locs[uuid] = (seg, off)
                seg_name = self._arena.segment_name(seg)
                for info in self._infos_by_uuid.get(uuid, ()):
                    info.shm_segment = seg_name
                    info.shm_offset = off
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
        if self._wake_q is None:
            self._wake_q = queue.Queue()
            threading.Thread(
                target=self._watch_wakes, daemon=True,
                name=f"arena-h2d-wake-{self.my_entity_id}").start()
        fut: Future = Future()
        self._wake_q.put((cuda_future, fut))
        return fut

    def _watch_wakes(self) -> None:
        # Events are queued in stream order, so sequential waits are exact.
        while True:
            cuda_future, fut = self._wake_q.get()
            try:
                cuda_future.result()   # event.synchronize (GIL released)
            except Exception:          # noqa: BLE001 — wake regardless
                pass
            fut.set_result(None)

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

    def _cleanup_by_uuid(self, request_id: str, uuid: str):
        # Grandparent cleanup (refcounts): skip the file manager's unlink.
        super(SharedMemoryCommunicationManager, self)._cleanup_by_uuid(
            request_id, uuid)
        self._infos_by_uuid.pop(uuid, None)
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
