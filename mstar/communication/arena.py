"""RFC #130 Step 2: tensor transport over a shared-memory arena.

``ArenaShmCommunicationManager`` replaces ``SharedMemoryCommunicationManager``'s
per-tensor file open/write/read/unlink with the Rust segmented ``/dev/shm``
arena (persistent mmaps + first-fit allocator, vendored in ``rust/``):

* **Producer** — ``register_for_send`` reserves ``(segment, offset)`` per
  tensor and D2H-copies straight into the segment's buffer view on the
  dedicated copy stream. The location is stamped onto the tensor's
  ``TensorPointerInfo`` (``shm_segment``/``shm_offset``), which ships to the
  consumer on the existing control mesh — the wire shape is unchanged, two
  optional fields ride along.
* **Consumer** — ``start_read_tensors`` opens the named segment once (lazy,
  cached), views the bytes zero-copy (``torch.frombuffer``), and H2D-copies
  on the dedicated stream. The H2D stream is synchronized before returning:
  the producer reclaims the slot on ACK, so the bytes must be device-resident
  before the ACK can be sent (the file transport got this for free because
  ``f.read()`` copied).
* **Pinning** — each mapped segment is ``cudaHostRegister``-ed once, on both
  sides (``MSTAR_SHM_ARENA_PIN``, default on with CUDA). A registered segment
  reaches page-locked copy bandwidth, and the copies through the side streams
  stay truly asynchronous — a plain pageable mmap would silently synchronize
  the host and defeat the D2H/H2D overlap. Segments are created once and
  never move, so a registration holds for the segment's lifetime.
* **Capacity** — the arena grows segment by segment up to
  ``MSTAR_SHM_ARENA_MAX_SEGMENTS`` (existing mappings never move — that is
  what keeps registrations and open consumer views valid; oversized tensors
  get a dedicated segment). At the cap, ``register_for_send`` backpressures:
  it waits for consumers to ACK and retries, failing loudly after
  ``MSTAR_SHM_ARENA_FULL_TIMEOUT_S``.

Selection: ``create_tensor_communication_manager`` picks this manager for the
SHM protocol when ``MSTAR_SHM_ARENA`` is ``1`` (require) or ``AUTO`` (use if
the ``mstar_rust`` extension imports); default ``0`` keeps the file transport.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import (
    FutureAndPointers,
    SharedMemoryCommunicationManager,
    _nullcontext,
)
from mstar.graph.base import GraphEdge, TensorPointerInfo

logger = logging.getLogger(__name__)

_CUDA_HOST_ALREADY_REGISTERED = 712


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

    def _sync_segments(self) -> None:
        while len(self._seg_views) < self._arena.num_segments:
            seg = self._arena.segment(len(self._seg_views))
            self._seg_views.append(memoryview(seg))
            if self._pin_segments:
                _pin(*seg.ptr_len())
                self._pinned_segments += 1

    def _peer_view(self, segment_name: str) -> memoryview:
        entry = self._peer_segments.get(segment_name)
        if entry is None:
            arena = self._ShmArena.open(segment_name)
            if self._pin_segments:
                _pin(*arena.ptr_len())
            entry = self._peer_segments[segment_name] = (
                arena, memoryview(arena))
        return entry[1]

    # -- producer ---------------------------------------------------------

    def _reserve(self, nbytes: int) -> tuple[int, int]:
        """Reserve with backpressure: at the segment cap, wait for consumer
        ACKs to free space instead of failing outright."""
        deadline = time.monotonic() + self._full_timeout_s
        warned = False
        while True:
            try:
                seg, off = self._arena.reserve(max(nbytes, 1))
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"SHM arena full for >{self._full_timeout_s}s "
                        f"({self._arena.num_segments} segments); raise "
                        "MSTAR_SHM_ARENA_MAX_SEGMENTS / _SEGMENT_MB or check "
                        "for consumers not ACKing") from None
                if not warned:
                    logger.warning(
                        "SHM arena at capacity; backpressuring sends until "
                        "consumers ACK")
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
                seg, off = self._reserve(nbytes)
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
                        raise RuntimeError(
                            f"tensor {info.uuid} from {info.source_entity} "
                            "has no arena location: the producer is not "
                            "running the arena transport (MSTAR_SHM_ARENA "
                            "must match across the deployment)")
                    tensor = self._read_from_arena(info)
                    h2d_did_work = h2d_did_work or tensor.numel() > 0
                    self.tensor_store.put_tensor(
                        request_id, info.uuid, tensor)
                    self.tensor_store.set_metadata(
                        request_id, info.uuid, mem_registered=False)
                    # +1 transit (released by get_ready_tensors), +1 usage
                    # (released by _cleanup_consumed_inputs).
                    self.tensor_store.increment_ref(request_id, info.uuid, 2)
                self.pending.append(
                    FutureAndPointers(
                        future=None, graph_edges=[graph_edge],
                        request_id=request_id,
                        rx_time=time.perf_counter() - rx_t0,
                    )
                )
        if h2d_did_work and self._h2d_stream is not None:
            # The producer reclaims the slot when this edge is ACKed, and the
            # source is its live mapping — the copy must be complete before
            # any ACK can go out (the file path's f.read() made this
            # implicit). The H2D still overlapped default-stream work.
            self._h2d_stream.synchronize()
            torch.cuda.default_stream(self.device).wait_stream(
                self._h2d_stream)
        return []

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
        if not self.tensor_store.check_uuid_presence(request_id, uuid):
            return
        self.tensor_store.remove_tensor(request_id, uuid)
