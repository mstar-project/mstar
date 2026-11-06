"""Tensor transport over a shared-memory arena.

``ArenaShmCommunicationManager`` moves tensors through a Rust segmented
``/dev/shm`` arena (persistent mmaps + first-fit coalescing allocator, vendored
in ``rust/``), with no per-tensor open/write/read/unlink. Producer:
``register_for_send`` reserves a slot and D2H-copies into the segment on the
copy stream; the ``(segment, offset)`` rides the existing descriptors.
Consumer: ``start_read_tensors`` maps the named segment once, reads
zero-copy, H2D-copies on the copy stream; the producer reclaims on ACK,
gated by a CUDA-event future so an edge is never ACKed before its copies
land. Segments are mapped once and never move, so the one-time
``cudaHostRegister`` per segment (within ``MSTAR_SHM_ARENA_PIN_MAX_MB``)
holds for its lifetime and keeps the side-stream copies truly async.

Capacity degrades in layers: grow by segments up to
``MSTAR_SHM_ARENA_MAX_SEGMENTS`` (or until /dev/shm cannot back another
segment: segments commit their pages when created); at the cap, briefly
backpressure for consumer ACKs; then spill the tensor to a file of its own
(``MSTAR_SHM_ARENA_SPILL``, default on) — slower, never fails.
``stats_summary()`` exposes occupancy and the
fragmentation gauge (largest contiguous free block); ``--log-stats`` logs it
periodically.

Capacity (growth, pinning, backpressure, spill files, reclaim, peer mappings) lives
in ``rust/src/arena_capacity.rs``. The per-tensor paths run in Rust
(``rust/src/arena_copy.rs``) with no Python call per tensor, on a CPU manager or an
NVIDIA device. This file keeps startup, sizing, and the torch-call paths ROCm and XPU
devices run, which also take the rows Rust hands back.

Ceilings are PER-ENTITY and multiply across a node: with E entities
(workers + the api-server data worker), /dev/shm demand can reach
``MAX_SEGMENTS x SEGMENT_MB x E`` and pinned host RAM approx
``PIN_MAX_MB x E`` (consumers pin peer segments too, so one process can pin
more than its own arena holds). With neither ``MAX_SEGMENTS`` nor
``SEGMENT_MB`` set, the API server sizes every arena of its deployment once,
before any exists (``export_arena_plan``): /dev/shm's free space split among
the deployment's arenas plus one share kept for spill files. Each entity
measuring free space for itself would give the ones that start later less,
since a first segment's pages are committed when it is created. An entity
started without that plan falls back to its own measurement. Set by hand,
construction fails fast when one entity's ceiling already exceeds /dev/shm.
Either way it warns when the ceiling exceeds current free space or when the
pin budget is an outsized share of physical RAM.

An entity can also run without an arena of its own: when its first segment
cannot be created (or /dev/shm is too small for a 16 MiB share), and on
devices other than CPU and CUDA (``_arena_supported``), which use files until
the arena is measured there. It then sends every tensor as a per-uuid file and
still reads its peers' arenas, so it mixes with entities that have one.

Selection: ``create_tensor_communication_manager`` builds this manager for
the SHM protocol. See :doc:`environment_variables` for the knobs.
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import threading
import time
import weakref
from concurrent.futures import Future
from contextlib import nullcontext as _nullcontext
from typing import NamedTuple

import numpy as np
import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensor_store import RustTensorStore
from mstar.communication.tensors import (
    DEPLOYMENT_TOKEN_ENV,
    FutureAndPointers,
    LocalTransferEngine,
    Rid,
    TensorCommunicationManager,
    _deployment_namespace,
    _hold_namespace_lock,
    _sweep_dead_namespaces,
)
from mstar.communication.torch_abi import require_torch_abi
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.platforms import current_platform
from mstar.utils.containers import ParallelList
from mstar.utils.host_copy import HostCopier
from mstar.utils.threads import Thread

logger = logging.getLogger(__name__)

def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes: header + contiguous raw data."""
    t = tensor.detach().contiguous().cpu()
    # reshape(-1) so 0-dim (scalar) tensors can be byte-viewed: PyTorch
    # disallows viewing a 0-dim tensor to a dtype with a different element size.
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


def _deserialize_tensor(
    data: bytes | memoryview, device: str,
    tensor_info: TensorPointerInfo
) -> torch.Tensor:
    """Reconstruct a tensor from bytes produced by ``_serialize_tensor``."""
    if isinstance(data, memoryview):
        data = bytes(data)
    shape = tensor_info.dims
    dtype = tensor_info.dtype
    if len(data) == 0:
        t = torch.empty(shape, dtype=dtype)
    else:
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype).reshape(shape)
    if device != "cpu":
        t = t.to(device)
    return t


# Docker gives a container 64 MB of /dev/shm, and one video or audio tensor can outgrow that
_SHM_DIR_WARN_BYTES = 1 << 30


def _default_shm_dir() -> str:
    """Return the default shared-memory directory for the current platform."""
    if platform.system() == "Linux" and os.path.isdir("/dev/shm"):
        return "/dev/shm"
    return "/tmp/mstar_shm"


_sizing_logged = False

#: ``<segment MiB>x<max segments>`` for every arena of one deployment, ``0x0``
#: for none; set by ``export_arena_plan`` and inherited by the processes the
#: API server starts.
ARENA_PLAN_ENV = "MSTAR_SHM_ARENA_PLAN"
#: Without a plan an entity takes a quarter of the free space it finds: its
#: arena and two others, plus the spill share.
_UNPLANNED_ENTITIES = 3


def _default_sizing(budget: int, entities: int) -> tuple[int, int] | None:
    """(segment MiB, max segments) for each of ``entities`` arenas sharing
    ``budget`` bytes, with one more share left for the files sends spill to:
    256 MiB segments, 1 to 32 of them, or where a share is under 256 MiB, one
    segment of the largest power-of-two MiB that fits. None where a share is
    under 16 MiB: the entities send by files."""
    share_mb = (budget >> 20) // (entities + 1)
    if share_mb < 16:
        return None
    segment_mb = 256 if share_mb >= 256 else 1 << (share_mb.bit_length() - 1)
    return segment_mb, min(32, share_mb // segment_mb)


def _shm_space() -> tuple[int | None, int | None]:
    """/dev/shm's total and free bytes, or Nones where it cannot be read."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return None, None
    return st.f_frsize * st.f_blocks, st.f_frsize * st.f_bavail


def export_arena_plan(worker_devices: list[str] | None) -> None:
    """Size every arena of this deployment once, from /dev/shm's free space
    before any of them exists, and hand the sizes to the processes the caller
    starts (``ARENA_PLAN_ENV``). Called by the API server, which holds one
    arena itself, for its workers on ``worker_devices``; None (no deployment
    config) drops a plan an earlier server in this process left."""
    os.environ.pop(ARENA_PLAN_ENV, None)
    _, avail = _shm_space()
    if worker_devices is None or avail is None:
        return
    entities = 1 + sum(map(_arena_supported, worker_devices))
    sizes = _default_sizing(avail, entities) or (0, 0)
    os.environ[ARENA_PLAN_ENV] = f"{sizes[0]}x{sizes[1]}"
    logger.info(
        "ARENA: /dev/shm has %d MiB free for %d arenas and their spill files; "
        "each arena grows to %d segments of %d MiB (MSTAR_SHM_ARENA_MAX_SEGMENTS "
        "and _SEGMENT_MB override)", avail >> 20, entities, sizes[1], sizes[0])


def _arena_sizing() -> tuple[int, int, int | None, int | None]:
    """Segment MiB, max segments, and /dev/shm's total and free bytes. Knobs
    the user set are taken as given (the constructor fails fast if they do not
    fit); with neither set the deployment's plan applies, and without one the
    sizes come from /dev/shm's free space now. Either way a small or busy
    tmpfs (a Kubernetes pod gets 64 MiB) runs a small arena, or none (0
    segments of 0 MiB), and spills to files instead of refusing to start."""
    global _sizing_logged
    total, avail = _shm_space()
    segment_env = os.environ.get("MSTAR_SHM_ARENA_SEGMENT_MB")
    max_env = os.environ.get("MSTAR_SHM_ARENA_MAX_SEGMENTS")
    if segment_env or max_env:
        segment_mb, max_segments = int(segment_env or 256), int(max_env or 32)
        if segment_mb < 1 or max_segments < 1:
            raise ValueError("MSTAR_SHM_ARENA_SEGMENT_MB and _MAX_SEGMENTS "
                             "must be at least 1")
        return segment_mb, max_segments, total, avail
    if plan := os.environ.get(ARENA_PLAN_ENV):
        segment_mb, _, max_segments = plan.partition("x")
        return int(segment_mb), int(max_segments), total, avail
    if avail is None:
        return 256, 32, total, avail
    sizes = _default_sizing(avail, _UNPLANNED_ENTITIES)
    if sizes is None:
        return 0, 0, total, avail
    segment_mb, max_segments = sizes
    if not _sizing_logged:
        _sizing_logged = True
        logger.info(
            "ARENA: /dev/shm has %d MiB free; each entity's arena grows "
            "to %d segments of %d MiB (MSTAR_SHM_ARENA_MAX_SEGMENTS and "
            "_SEGMENT_MB override)", avail >> 20, max_segments, segment_mb)
    return segment_mb, max_segments, total, avail


def _arena_supported(device) -> bool:
    """Whether an entity on ``device`` stages its sends in an arena: CPU, and
    CUDA. ROCm (``_pin`` loads libcudart), XPU (no copy streams) and the rest
    send by per-uuid files until the arena is ported and measured there."""
    kind = torch.device(device).type
    return kind == "cpu" or (kind == "cuda" and current_platform.is_cuda())

# c10::DeviceType, as rust/src/torch_abi/mod.rs names them.
DEVICE_CPU = 0
DEVICE_CUDA = 1


class _Placements(NamedTuple):
    """Where a batch's staged tensors landed, as columns.

    Index-parallel, one entry per staged tensor, in the layout
    ``set_shm_placement`` takes -- so staging fills three lists rather than
    minting an object per tensor that ``_write_back`` then takes apart again.

    The segment is its INDEX, not its name: the store learned the names once,
    as the arena grew into them.

    Empty is ``not p.uuids``, not ``not p`` -- this is a 3-tuple, so it is
    always truthy. ``ParallelList`` does NOT share that edge: it defines
    ``__len__``, which ``bool`` falls back on, so an empty one is falsey.
    """

    uuids: list[int]
    segment_idxs: list[int]
    offsets: list[int]

    @classmethod
    def empty(cls) -> "_Placements":
        return cls([], [], [])

    def add(self, uuid: int, segment_idx: int, offset: int) -> None:
        """All three at once: they mean nothing out of lockstep."""
        self.uuids.append(uuid)
        self.segment_idxs.append(segment_idx)
        self.offsets.append(offset)


class _CudaEventFuture:
    """Future-shaped CUDA event. An edge whose H2D copies ride behind this
    event reports ready only once the copies have completed on the device —
    so the ACK that lets the producer reclaim the arena slot is deferred by
    ``get_ready_tensors``'s existing future polling instead of a blocking
    host synchronize. ``keep`` holds host buffers the copies read from, which
    no stream tag protects when they were queued by cudart directly; the wake
    watcher holds this object until the event fires, whoever else lets go."""

    def __init__(self, stream, keep: list | None = None):
        # blocking=True: `synchronize` SLEEPS until the event fires. The
        # default (False) busy-waits — the watcher thread that turns these
        # into wake futures would burn a full core per wait.
        self._event = torch.cuda.Event(blocking=True)
        self._event.record(stream)
        self._keep = keep

    def done(self) -> bool:
        return self._event.query()

    def result(self) -> None:
        self._event.synchronize()


_CUDART = None


def _cudart():
    """libcudart, loaded by the soname of torch's CUDA major, which reaches the
    copy torch already loaded even where it is not on the loader path (a pip
    install of torch)."""
    global _CUDART
    if _CUDART is None:
        import ctypes

        if not torch.version.cuda:
            raise OSError("torch is not built with CUDA")
        _CUDART = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    return _CUDART


def _cudart_addresses(*names: str) -> tuple[int, ...]:
    """Addresses of the cudart calls the Rust transport makes, in the given order."""
    import ctypes

    lib = _cudart()
    return tuple(ctypes.cast(getattr(lib, name), ctypes.c_void_p).value for name in names)


def _cudart_copy_fns() -> tuple[int, ...]:
    return _cudart_addresses(
        "cudaMemcpyAsync", "cudaGetLastError", "cudaGetErrorString", "cudaStreamSynchronize")


def _pin_fns() -> tuple[int, int, int] | None:
    """The runtime's (register, unregister, get_last_error) addresses, or None
    where they cannot be loaded and copies stay pageable."""
    try:
        return _cudart_addresses("cudaHostRegister", "cudaHostUnregister", "cudaGetLastError")
    except OSError as e:
        logger.warning("ARENA: cannot load libcudart (%s); copies fall back to pageable bandwidth", e)
        return None


def _gather(view, starts: list[int], nbytes: int, dst: np.ndarray, *,
            words_max_bytes: int) -> None:
    """``nbytes`` from each of ``starts`` in ``view``, back to back into ``dst``."""
    if (
        nbytes <= words_max_bytes and nbytes % 8 == 0 and len(view) % 8 == 0
        and all(s % 8 == 0 for s in starts)
    ):
        words = np.frombuffer(view, dtype=np.int64)
        w = nbytes // 8
        at = np.asarray(starts, dtype=np.int64) // 8
        np.take(words, (at[:, None] + np.arange(w)).ravel(), out=dst.view(np.int64))
        return
    seg = np.frombuffer(view, dtype=np.uint8)
    for i, s in enumerate(starts):
        dst[i * nbytes:(i + 1) * nbytes] = seg[s:s + nbytes]


class ArenaShmCommunicationManager(TensorCommunicationManager):
    """Tensor transport via the Rust shared-memory arena (``mstar_rust``).

    ``_arena`` is None when this entity has none of its own (see the module
    docstring): every send then spills to a per-uuid file. ``use_arena=False``
    asks for that outright.
    """

    #: Directories besides ``shm_dir`` that hold this transport's
    #: ``mstar_<namespace>_*`` files; each gets its own namespace lock.
    _namespace_file_dirs = ("/dev/shm",)

    # Staging writes the segment and offset onto the descriptor the store
    # holds, which routing may already have copied -- so every outgoing edge
    # has to be refreshed from the store before it is sent. See
    # ``TensorCommunicationManager.refresh_shm_placement``.
    stamps_shm_placement = True

    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        device: str,
        communicator: BaseCommunicator,
        shm_dir: str | None = None,
        enable_prof: bool = False,
        copy_workers: int = 1,
        use_arena: bool = True,
    ):
        engine = LocalTransferEngine(hostname=hostname)
        super().__init__(
            my_entity_id=my_entity_id,
            my_session_id=engine.get_session_id(),
            device=device,
            communicator=communicator,
            transfer_engine=engine,
            enable_prof=enable_prof,
        )

        self.shm_dir = shm_dir or _default_shm_dir()
        os.makedirs(self.shm_dir, exist_ok=True)
        # at boot, not at the first large send: that failure would be the only other sign of a small dir
        try:
            st = os.statvfs(self.shm_dir)
            shm_dir_total = st.f_frsize * st.f_blocks
        except OSError:
            shm_dir_total = None
        if shm_dir_total is not None and shm_dir_total < _SHM_DIR_WARN_BYTES:
            logger.warning(
                "SHM: %s holds only %d MiB and every tensor in flight is written there, so a large one will "
                "fail its send with ENOSPC. Give it more room (docker --shm-size), or pass "
                "--tensor-comm-protocol TCP, which is slower", self.shm_dir, shm_dir_total >> 20)

        self._shm_namespace = _deployment_namespace(communicator)
        # The launch token is in the namespace, so a later launch on the same
        # endpoint (uuid counters back at 1) never writes these names.
        self.names_unique_per_launch = bool(
            self._shm_namespace and os.environ.get(DEPLOYMENT_TOKEN_ENV))
        if self._shm_namespace:
            # A lock guards only the files in its own directory.
            dirs = [d for d in dict.fromkeys(
                (self.shm_dir, *self._namespace_file_dirs)) if os.path.isdir(d)]
            for d in dirs:
                _hold_namespace_lock(d, self._shm_namespace)
            for d in dirs:
                _sweep_dead_namespaces(d, self._shm_namespace)

        # Dedicated copy streams: D2H/H2D run here so they don't serialize
        # behind the next GPU step queued on the default stream.
        self._d2h_stream: torch.cuda.Stream | None = None
        self._h2d_stream: torch.cuda.Stream | None = None
        if torch.cuda.is_available() and str(device) != "cpu":
            self._d2h_stream = torch.cuda.Stream(device=device)
            self._h2d_stream = torch.cuda.Stream(device=device)

        from mstar_rust import ArenaTransport, SegmentedShmArena

        # At startup, not at the first tensor: an unsupported torch is an error here.
        has_torch_abi = require_torch_abi(device)

        # A CPU consumer's reads (the API server's outputs) copy on these.
        self._host_copier = HostCopier(copy_workers)

        supported = use_arena and _arena_supported(device)
        segment_mb, max_segments, shm_total, shm_avail = (
            _arena_sizing() if supported else (0, 0, None, None))
        self._pin_segments = (
            os.getenv("MSTAR_SHM_ARENA_PIN", "1") == "1"
            and torch.cuda.is_available() and str(device) != "cpu"
            and supported
        )
        # Pinned host memory is a system-wide resource (pages come out of the
        # OS's pageable pool), so it gets its own budget, distinct from the
        # segment cap. Segments past the budget stay unpinned — copies still
        # work, they just lose async overlap. Oversized dedicated segments
        # (single allocations larger than a segment) are never pinned: a
        # one-shot transfer doesn't amortize the registration cost.
        pin_budget = int(os.getenv("MSTAR_SHM_ARENA_PIN_MAX_MB", "4096")) << 20
        # _read_many_from_arena's gather limits: the crossovers depend on the host
        self._gather_item_bytes = int(
            os.getenv("MSTAR_SHM_ARENA_GATHER_MAX_BYTES", "16384"))
        self._gather_words_bytes = int(
            os.getenv("MSTAR_SHM_ARENA_GATHER_WORDS_MAX_BYTES", "512"))
        self._segment_bytes = segment_mb << 20
        # Arena saturation spills to per-uuid files instead of failing: bursty or
        # oversized workloads degrade to file-copy speed. MSTAR_SHM_ARENA_SPILL=0
        # restores strict backpressure + timeout.
        spill = os.getenv("MSTAR_SHM_ARENA_SPILL", "1") == "1"
        # Default 0: on a worker, TENSOR_RECEIVED ACKs (which free slots)
        # are processed by the SAME thread that would sit in this grace
        # wait, so waiting is pure dead time there — spill immediately.
        # Deployments where another thread drains the communicator (the
        # threaded api_server) can set a small grace to ride out bursts.
        spill_after_s = float(os.getenv("MSTAR_SHM_ARENA_SPILL_AFTER_S", "0"))
        # h2d completion watcher: turns a CUDA event into a real Future so
        # the worker's eventfd wakes the moment reads finish (no 10 ms tick).
        self._wake_q: queue.Queue | None = None
        self._wake_lock = threading.Lock()
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
        if self._pin_segments and phys is not None and pin_budget > phys // 4:
            logger.warning(
                "ARENA: MSTAR_SHM_ARENA_PIN_MAX_MB (%d MiB) exceeds a "
                "quarter of physical RAM (%d MiB) FOR ONE ENTITY; pinned "
                "budgets multiply across entities and come out of the OS's "
                "pageable pool", pin_budget >> 20, phys >> 20)
        # INSTANCE-UNIQUE base name: a fixed per-entity name in the global
        # /dev/shm namespace collides across servers (a second server's
        # create() truncates the first's live segments — silent corruption)
        # and across users (permission-denied at startup). pid + random
        # token makes each instance's names unique; wire-compatibility is
        # free because consumers open whatever segment name the descriptor
        # carries. Under a deployment namespace the segments are named like
        # its tensor files, so its lock marks them live and the dead-
        # namespace sweep reclaims them; a pid cannot, since /proc is per
        # PID namespace and containers can share /dev/shm. Without one, the
        # pid in the name drives the orphan sweep below.
        import secrets

        self._sweep_orphans()
        ns = self._shm_namespace
        base = (f"mstar_{ns}_arena_{my_entity_id}_{secrets.token_hex(4)}"
                if ns else f"mstar_arena_{my_entity_id}_{os.getpid()}_"
                f"{secrets.token_hex(4)}")
        self._arena = None
        if not supported:
            logger.info(
                "ARENA: not used on %s (%s); this entity sends by per-uuid "
                "files and reads its peers' arenas", device,
                current_platform.name)
        elif not segment_mb:
            logger.warning(
                "ARENA: /dev/shm has %s MiB free, too little for a 16 MiB "
                "arena per entity and room for spill files; this entity sends "
                "by per-uuid files",
                "?" if shm_avail is None else shm_avail >> 20)
        else:
            try:
                self._arena = SegmentedShmArena.create(
                    base, segment_mb << 20, max_segments)
            except (RuntimeError, OSError) as e:
                logger.warning(
                    "ARENA: cannot create this entity's first segment (%s); "
                    "its sends spill to per-uuid files", e)
        # Producer-side segment views (memoryviews are stable: segments never
        # move or resize), for the torch-call path. Filled as it meets them.
        self._seg_views: dict[int, memoryview] = {}
        # What the store has been told about segments, for the torch-call path:
        # the table lives in the bookkeeper, so a store swapped in after
        # construction starts empty and has to be told again.
        self._segments_named_to = None
        self._segments_named = 0
        # The torch-free half of the transport and, where the wrapper reads this
        # device's tensors, its per-tensor paths (_read_device_type is set).
        self._read_device_type = None
        device_index, cuda = -1, None
        if has_torch_abi:
            device_index, cuda = self._rust_copies()
        # Stage times feed the TTL backstop: a request aborted after staging but
        # before every consumer ACKs defers reclaim forever (cleanup_request
        # waits for ACKs that will never come). A slot older than the REQUEST
        # timeout cannot have a legitimate reader — the request is dead by
        # contract — so freeing past a bound safely above it cannot race a real
        # consumer. Default OFF pending review discussion; enable with
        # MSTAR_SHM_ARENA_SLOT_TTL_S (recommend >= 2x the request timeout).
        self._transport = ArenaTransport(
            self._arena, self.shm_dir, f"mstar_{ns}_" if ns else "mstar_", my_entity_id,
            spill, spill_after_s,
            float(os.getenv("MSTAR_SHM_ARENA_FULL_TIMEOUT_S", "30")),
            float(os.getenv("MSTAR_SHM_ARENA_SLOT_TTL_S", "0")),
            shm_total=shm_total, device_index=device_index,
            gather_item_bytes=self._gather_item_bytes, cuda=cuda,
            d2h_stream=self._d2h_stream.cuda_stream if cuda else None,
            h2d_stream=self._h2d_stream.cuda_stream if cuda else None,
        )
        if self._pin_segments and (pin_fns := _pin_fns()) is not None:
            self._transport.set_pins(pin_fns, pin_budget)
        # Guaranteed exit-time unlink: weakref.finalize runs at interpreter
        # shutdown even when the manager is still referenced (the worker
        # case — no explicit cleanup path runs there, so the Rust Drop never
        # fires).
        self._finalizer = weakref.finalize(self, self._transport.unlink_segments)
        # Unregister before the arenas unmap (close(), or when the manager is
        # collected). Not at interpreter exit: the driver drops every
        # registration with the process.
        self._unpin_finalizer = weakref.finalize(self, self._transport.release_pins)
        self._unpin_finalizer.atexit = False
        self._peer_evict_last = 0.0
        self._default_stream = (
            torch.cuda.default_stream(self.device) if self._h2d_stream is not None else None)

    def _rust_copies(self) -> tuple[int, tuple[int, ...] | None]:
        """Turn on the Rust per-tensor paths where the wrapper reads this device's
        tensors, a CPU manager or an NVIDIA device; any other device (ROCm, XPU)
        stays with the torch calls in this file. The device index and cudart's
        addresses for the transport."""
        device = torch.device(self.device)
        if device.type == "cpu":
            self._read_device_type = DEVICE_CPU
            return -1, None
        if not (self._on_cuda and current_platform.is_cuda()):
            return -1, None
        try:
            cuda = _cudart_copy_fns()
        except OSError as e:
            logger.warning("ARENA: cannot load libcudart (%s); copies stay with torch", e)
            return -1, None
        self._read_device_type = DEVICE_CUDA
        return (device.index if device.index is not None else torch.cuda.current_device()), cuda

    @property
    def needs_cpu_tensor(self) -> bool:
        # The send is a host write, so a host copy the producer already made
        # saves the device-to-host copy here.
        return True

    def _send_source(self, uuid: int) -> torch.Tensor:
        """What to serialize for ``uuid``: the stored host copy if there is
        one, else the device tensor (which the write copies down)."""
        cpu = self.tensor_store.get_cpu_tensor(uuid)
        return cpu if cpu is not None else self.tensor_store.get_tensor(uuid)

    def _shm_path(self, entity_id: str, uuid: int) -> str:
        return self._transport.spill_path(entity_id, uuid)

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

    def stats_summary(self) -> dict:
        """Occupancy/fragmentation snapshot (named apart from the raw
        ``SegmentedShmArena.stats`` tuple). The fragmentation signature is
        `largest_free_block` collapsing while `free_bytes` stays high."""
        arena = self._arena
        total, free, largest = arena.stats() if arena is not None else (0, 0, 0)
        return {
            "segments": arena.num_segments if arena is not None else 0,
            "total_bytes": total,
            "free_bytes": free,
            "largest_free_block": largest,
            "pinned_bytes": self._transport.pinned_bytes,
        }

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
        self._transport.reclaim_expired()
        st = self.stats_summary()
        # live_slots/spill_files climbing while requests finish = reclaim
        # being deferred (e.g. aborts without ACKs) — the soak's leak canary.
        st["live_slots"], st["spill_files"] = self._transport.live()
        logger.info("ARENA stats: %s", st)

    def _peer_view(self, segment_name: str) -> memoryview:
        return memoryview(self._transport.peer(segment_name))

    def _evict_dead_peers(self) -> None:
        """Drop cached peer segments whose backing file is gone. Gated on
        `self.pending` being empty: an mmap must outlive any in-flight h2d
        copy that reads from it, and pending futures are exactly those
        copies. Time-gated to keep the existence checks off the hot path."""
        now = time.monotonic()
        # _reads_active > 1: another thread may have queued an async H2D
        # copy whose future hasn't reached `pending` yet — unmapping under
        # it would be a use-after-free. Only the sole active reader evicts.
        if (self.pending or self._kept_reads or self._reads_active > 1
                or now - self._peer_evict_last < 10.0):
            return
        self._peer_evict_last = now
        self._transport.evict_dead_peers()

    # -- producer ---------------------------------------------------------

    def _d2h_ctx(self):
        return (
            torch.cuda.stream(self._d2h_stream)
            if self._d2h_stream is not None
            else _nullcontext()
        )

    def _segment_view(self, seg: int) -> memoryview:
        view = self._seg_views.get(seg)
        if view is None:
            view = self._seg_views[seg] = memoryview(self._arena.segment(seg))
        return view

    def _name_segments_to_store(self) -> None:
        """Tell the store what each new segment index is called, for placement
        written here (the Rust path names them as it publishes): once per segment
        rather than once per tensor, so placement can cross as an index."""
        store = self.tensor_store
        if store is not self._segments_named_to:
            self._segments_named_to, self._segments_named = store, 0
        for i in range(self._segments_named, self._arena.num_segments):
            store.register_shm_segment(i, self._arena.segment_name(i))
        self._segments_named = self._arena.num_segments

    def _stage_one(
        self, request_id: Rid, uuid: int,
        info_arg: TensorPointerInfo | None,
        placements: _Placements,
    ) -> bool:
        """Stage one tensor into the arena. Returns whether a D2H was queued.

        The caller owns the CUDA sync, the ``_d2h_ctx`` context and the
        trailing stream sync, so a batch pays for each of those once -- and
        the writeback of ``placements``, which is why they are collected here
        rather than stored one at a time.
        """
        if self.tensor_store.is_registered(uuid):
            return False
        tensor = self._send_source(uuid)
        t0 = time.perf_counter()
        t = tensor.detach().contiguous()
        nbytes = t.numel() * t.element_size()
        loc = self._transport.reserve(nbytes)
        if loc is not None and (
            self.tensor_store.is_registered(uuid)
            or not self._transport.claim(uuid, *loc)
        ):
            # Lost a concurrent-duplicate race: another thread registered this
            # uuid, or is staging it, while our reserve released the GIL. Return
            # our slot instead of orphaning it.
            self._transport.return_slot(*loc)
            return False
        if loc is None:
            # Arena saturated: spill THIS tensor to its own file (infos keep
            # shm_segment=None — the consumer reads the file for exactly those).
            data = _serialize_tensor(t)
            if not self._transport.spill_write(uuid, data):
                return False  # another thread has it, as above
            self.tensor_store.set_metadata(uuid, mem_registered=True)
            if self.enable_prof:
                self._record_tx(request_id, uuid, len(data), time.perf_counter() - t0)
            logger.debug("ARENA: spilled %s (%d bytes)", uuid, len(data))
            return False
        seg, off = loc
        queued = False
        try:
            if nbytes:
                flat = torch.frombuffer(
                    self._segment_view(seg)[off:off + nbytes],
                    dtype=torch.uint8,
                )
                # Async D2H into the pinned segment when a copy stream exists
                # (the caller's single sync covers the batch); blocking
                # otherwise, so the descriptor can never ship ahead of the bytes.
                if t.is_cuda:
                    flat.view(t.dtype).reshape(t.shape).copy_(
                        t, non_blocking=self._d2h_stream is not None)
                    queued = True
                elif t.device.type == "cpu":
                    # A stored host copy: one memcpy, nothing for the stream
                    # sync to wait on. numpy, not Tensor.copy_: torch splits a
                    # CPU copy of more than 32K elements over its intra-op
                    # OpenMP pool, whose threads then spin on the cores the
                    # worker's own threads need.
                    np.copyto(flat.numpy(),
                              t.reshape(-1).view(torch.uint8).numpy())
                else:
                    flat.view(t.dtype).reshape(t.shape).copy_(t)
        except BaseException:
            self._transport.free([uuid])
            raise
        # Still stamped in place when the caller handed one in: that contract
        # is relied on (test_descriptor_survives_register_for_send). The
        # uuid-driven path has no caller object and needs none.
        if info_arg is not None:
            info_arg.shm_segment = self._arena.segment_name(seg)
            info_arg.shm_offset = off
        # The store must learn the segment, not just the caller's object. With
        # the Python bookkeeper those are the same object and this is a
        # re-assign; with the Rust one the descriptor was COPIED in, so without
        # this the stamp dies on a throwaway and the tensor ships with
        # shm_segment=None -- which the consumer reads as "spilled to a file"
        # and goes looking for a file that was never written.
        #
        # Only the two fields travel, not the descriptor: rebuilding one for
        # Python and marshalling it back costs ~3.5us per tensor to write two
        # integers the store already holds. The segment goes as its INDEX --
        # the store learned the name once, at the grow.
        placements.add(uuid, seg, off)
        self.tensor_store.set_metadata(uuid, mem_registered=True)
        if self.enable_prof:
            self._record_tx(request_id, uuid, nbytes, time.perf_counter() - t0)
        logger.debug("ARENA: staged %s at %d+%d (%d bytes)", uuid, seg, off, nbytes)
        return queued

    def _write_back(self, placements: _Placements):
        """One crossing for the batch, not one per tensor."""
        if not placements.uuids:
            return
        self._name_segments_to_store()
        # The columns go straight across, already in the callee's layout.
        self.tensor_store.set_shm_placement(
            placements.uuids, placements.segment_idxs, placements.offsets,
        )

    def _stage_batch(self, uuids: list[int], request_of, infos=None) -> None:
        """Stage ``uuids`` in Rust as far as it can go and the rest here.

        ``request_of(i)`` is the request of ``uuids[i]``, asked only of a row
        that needs it. ``infos``, parallel to ``uuids``, are stamped in place.
        """
        self._maybe_log_stats()
        placements = _Placements.empty()
        store = self.tensor_store
        rest = range(len(uuids))
        queued = False
        try:
            with self._d2h_ctx():
                if self._read_device_type is not None and isinstance(store, RustTensorStore):
                    t0 = time.perf_counter()
                    done = self._transport.stage(
                        store._rust, uuids, infos is not None or self.enable_prof)
                    queued = done.queued
                    if done.error is not None:
                        if done.errno is not None:
                            raise OSError(done.errno, done.error)
                        raise RuntimeError(done.error)
                    rest = done.deferred
                    if done.segments is not None:
                        self._after_staged(uuids, request_of, infos, done,
                                           time.perf_counter() - t0)
                for i in rest:
                    queued |= self._stage_one(
                        request_of(i), uuids[i],
                        infos[i] if infos is not None else None, placements)
        finally:
            self._write_back(placements)
            if queued and self._d2h_stream is not None:
                # The control message referencing these bytes is sent after we
                # return; the consumer must never observe a partial copy. On
                # the way out of an error too: a queued copy would otherwise
                # land in a slot its cleanup has freed for someone else.
                self._d2h_stream.synchronize()

    def _after_staged(self, uuids, request_of, infos, done, elapsed: float) -> None:
        """What a staged row still owes Python: the caller's descriptor, and the
        profile, which attributes the call's time to its rows evenly."""
        n = max(sum(1 for s in done.segments if s != -1), 1)
        for i, (segment, offset, nbytes) in enumerate(
                zip(done.segments, done.offsets, done.nbytes, strict=True)):
            if segment == -1:
                continue
            if infos is not None and segment >= 0:
                infos[i].shm_segment = self._arena.segment_name(segment)
                infos[i].shm_offset = offset
            if self.enable_prof:
                self._record_tx(request_of(i), uuids[i], nbytes, elapsed / n)

    def register_for_send(
        self, request_id: Rid, tensor_infos: list[TensorPointerInfo],
        skip_cuda_sync: bool = False,
    ):
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()
        self._stage_batch(
            [info.uuid for info in tensor_infos], lambda _: request_id, tensor_infos)

    def register_for_send_uuids(
        self,
        per_request: ParallelList[Rid, list[int]],
        skip_cuda_sync: bool = False,
    ):
        """Uuid-driven staging: no descriptor is rebuilt to be read for its
        uuid, and placement goes back as two columns rather than a whole
        descriptor. The whole batch stages inside one stream context and syncs
        once at the end, rather than one host-blocking sync per request."""
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()

        owners: list[Rid] = []

        def request_of(i: int) -> Rid:
            if not owners:
                owners.extend(rid for rid, uuids in per_request for _ in uuids)
            return owners[i]

        self._stage_batch([u for us in per_request.values for u in us], request_of)

    # -- consumer ---------------------------------------------------------

    def start_read_tensors(
        self, request_id: Rid, graph_edges: list[GraphEdge],
        graph_walk: str | None = None,
    ):
        return self._start_reads([(request_id, graph_edges)], graph_walk)

    def start_read_tensors_multi(self, reads):
        """One stream context, peer sweep, event and wake for every request's
        reads, not one each."""
        return self._start_reads(reads, None)

    def _start_reads(self, requests, graph_walk):
        # Increment races are benign here: a torn count can only make
        # eviction OVER-cautious (skip a sweep), never unsafe.
        self._reads_active += 1
        try:
            if not any(e.tensor_info for _rid, edges in requests for e in edges):
                # signal-only edges: no stream context to pay for, but the
                # (time-gated) sweep of dead peers' segments still runs here
                self._evict_dead_peers()
                return []
            return self._start_read_tensors(requests, graph_walk)
        finally:
            self._reads_active -= 1

    def _start_read_tensors(self, requests, graph_walk):
        self._evict_dead_peers()
        h2d_did_work = False
        stats = self.transfer_stats
        # one per edge read; the future is filled in once every copy is queued
        reads: list[FutureAndPointers] = []
        ctx = (
            torch.cuda.stream(self._h2d_stream)
            if self._h2d_stream is not None
            else _nullcontext()
        )
        # read and stored together once the rest are settled; one met again before
        # then takes the extra reference after the put
        staged: list[tuple] = []
        taken: dict[int, TensorPointerInfo] = {}
        again: list[int] = []
        keep: list = []

        def settle(request_id, edge, info) -> bool:
            """Whether this tensor still has to be read."""
            if info.source_entity == self.my_entity_id:
                self._slice_existing_tensor(
                    request_id=request_id, name=edge.name, next_node=edge.next_node,
                    graph_walk=graph_walk, info=info,
                )
                self.tensor_store.increment_ref(info.uuid, 1)
                return False
            if self._take_cached_read(request_id, info, taken, again):
                return False
            staged.append((request_id, info))
            taken[info.uuid] = info
            return True

        t0 = time.perf_counter()
        store = self.tensor_store
        with ctx:
            try:
                if (self._read_device_type is not None and stats is None
                        and isinstance(store, RustTensorStore)):
                    # No Python per tensor: Rust reads what it can and names the rest.
                    done = self._transport.read_edges(
                        store._rust, requests, 2, self._read_device_type, self.slice_origins,
                        self._host_copy_from())
                    h2d_did_work = done.did_work
                    keep.extend(done.keep)
                    reads = [
                        FutureAndPointers(future=None, graph_edges=[edge], request_id=request_id)
                        for request_id, edges in requests for edge in edges if edge.tensor_info
                    ]
                    for r, e, i in zip(done.other_requests, done.other_edges, done.other_infos, strict=True):
                        request_id, edges = requests[r]
                        settle(request_id, edges[e], edges[e].tensor_info[i])
                else:
                    for request_id, edges in requests:
                        for edge in edges:
                            if not edge.tensor_info:
                                continue
                            fp = FutureAndPointers(future=None, graph_edges=[edge], request_id=request_id)
                            for info in edge.tensor_info:
                                if settle(request_id, edge, info) and stats is not None:
                                    fp.count_read(info)
                            reads.append(fp)
                if staged:
                    h2d_did_work |= self._read_staged(staged, keep)
                if again:
                    self.tensor_store.increment_ref_batch_uniform(again, 1)
            except BaseException:
                # Queued copies read ``keep``, which goes with this frame.
                if self._h2d_stream is not None:
                    self._h2d_stream.synchronize()
                raise
        # profiling attributes the call's time to its reads evenly
        rx = (time.perf_counter() - t0) / max(len(reads), 1)
        for fp in reads:
            fp.rx_time = rx
        future = None
        if h2d_did_work and self._h2d_stream is not None:
            # Downstream kernels see the data (device-side ordering only).
            self._default_stream.wait_stream(self._h2d_stream)
            # The producer reclaims the slot when an edge is ACKed, and the
            # source is its live mapping — so the edge must not report ready
            # until its copies have completed (the file path's f.read() made
            # this implicit). One event covers the batch: all copies were
            # queued on the h2d stream in program order. get_ready_tensors
            # polls it — no host block here.
            future = _CudaEventFuture(self._h2d_stream, keep)
        for fp in reads:
            fp.future = future
        self.pending.extend(reads)
        if future is None:
            return []
        # A real Future for the worker's eventfd (EventWakeup.register_
        # futures needs add_done_callback): completed by the watcher thread
        # the moment the h2d copies finish, so an otherwise-idle worker
        # re-checks get_ready_tensors immediately instead of on its next
        # poll tick.
        return [self._wake_when_done(future)]

    def _host_copy_from(self) -> int | None:
        """The size from which a CPU consumer leaves a read to its copier's own threads."""
        copier = self._host_copier
        return 2 * copier.min_slice_bytes if copier.workers > 1 else None

    def _read_staged(self, staged: list[tuple], keep: list) -> bool:
        """Read the rows into the store, each with its two references: +1 transit
        (released by get_ready_tensors), +1 usage (released by
        _cleanup_consumed_inputs). Whether any bytes moved. ``keep`` gets the host
        buffers the queued copies read from."""
        rids = [rid for rid, _ in staged]
        infos = [info for _, info in staged]
        store = self.tensor_store
        did_work = False
        if self._read_device_type is not None and isinstance(store, RustTensorStore):
            done = self._transport.read(
                store._rust, rids, infos, 2, self._read_device_type, self.slice_origins,
                self._host_copy_from())
            did_work = done.did_work
            keep.extend(done.keep)
            if not done.deferred:
                return did_work
            rids = [rids[i] for i in done.deferred]
            infos = [infos[i] for i in done.deferred]
        uuids = [info.uuid for info in infos]
        tensors = self._read_many_from_arena(infos)
        store.put_tensor_batch_multi(rids, uuids, tensors, infos)
        store.increment_ref_batch_uniform(uuids, 2)
        return did_work or any(t.numel() > 0 for t in tensors)

    def _wake_when_done(self, cuda_future) -> Future:
        with self._wake_lock:   # two callers must not race the create
            if self._wake_q is None:
                self._wake_q = queue.Queue()
                # staticmethod target closing over ONLY the queue: a bound
                # method would pin the whole manager (and its arena — so
                # segments never unlink) for the daemon thread's lifetime.
                Thread(
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

    # Small items of one shape, dtype and segment (a step's streamed tokens)
    # are gathered on the host into one pinned buffer, copied once and split
    # on the device: one copy call instead of a copy per item (~7 us each).
    # The default limits were measured on an H200 host, reading k items of n
    # bytes: the gather wins from k=2 up to n=16 KB and breaks even at 64 KB.
    # Up to 512 B an index of 8-byte words is fastest; above it, one slice
    # copy per item.
    def _read_many_from_arena(self, infos: list[TensorPointerInfo]) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(infos)
        if self.device != "cpu":
            groups: dict[tuple, list[int]] = {}
            for k, info in enumerate(infos):
                if 0 < info.nbytes <= self._gather_item_bytes and info.shm_segment is not None:
                    key = (info.shm_segment, info.nbytes, info.dtype, tuple(info.dims))
                    groups.setdefault(key, []).append(k)
            for (segment, nbytes, dtype, dims), ks in groups.items():
                if len(ks) < 2:
                    continue
                starts = [infos[k].shm_offset + infos[k].offset for k in ks]
                # pinned, so the copy is a true async DMA; the caching host
                # allocator holds the block until the copy is done
                host = torch.empty(len(ks) * nbytes, dtype=torch.uint8, pin_memory=True)
                _gather(self._peer_view(segment), starts, nbytes, host.numpy(),
                        words_max_bytes=self._gather_words_bytes)
                rows = host.to(self.device, non_blocking=True).view(dtype).reshape(len(ks), *dims)
                for k, t in zip(ks, rows.unbind(0), strict=True):
                    out[k] = t
        return [t if t is not None else self._read_from_arena(i) for t, i in zip(out, infos, strict=True)]

    def _read_from_arena(self, info: TensorPointerInfo) -> torch.Tensor:
        if info.shm_segment is None:
            # Spilled at the producer: its file is under the producer's uuid,
            # which a slice re-pointed here no longer carries.
            uuid = self.slice_origins.get(info.uuid, info.uuid)
            with open(self._shm_path(info.source_entity, uuid), "rb") as f:
                f.seek(info.offset)
                data = f.read(info.nbytes)
            return _deserialize_tensor(data, self.device, tensor_info=info)
        if info.nbytes == 0:
            t = torch.empty(
                info.dims, dtype=info.dtype)
            return t.to(self.device) if self.device != "cpu" else t
        view = self._peer_view(info.shm_segment)
        start = info.shm_offset + info.offset  # offset = TP-shard read offset
        if self.device == "cpu":
            # A CPU consumer owns a copy past reclaim. clone() would split it
            # over torch's intra-op pool, sized for small ops; the copier keeps
            # an image one memcpy and slices a video over threads of its own.
            flat = self._host_copier.copy(
                np.frombuffer(view, dtype=np.uint8, count=info.nbytes, offset=start))
            return flat.view(info.dtype).reshape(info.dims)
        flat = torch.frombuffer(
            view[start:start + info.nbytes], dtype=torch.uint8)
        t = flat.view(info.dtype).reshape(info.dims)
        # Async only when the h2d stream's event gates the ACK: with no copy
        # stream (XPU) nothing would wait for the copy before the slot is freed.
        return t.to(self.device, non_blocking=self._h2d_stream is not None)

    # -- reclaim ----------------------------------------------------------

    def unlink_owned_files(self) -> None:
        """The spill files, then this arena's own segments, for a process that
        is stopping: its exit-time finalizer never runs when a hung teardown
        ends in SIGKILL or the watchdog's hard exit. Peers keep the mappings
        they have; the finalizer still runs at a normal exit and finds them
        gone, and unlinks any segment grown after this call."""
        self._transport.unlink_owned()

    def close(self) -> None:
        """Stop the wake watcher and unregister every pinned segment (the
        segments themselves unmap with the arena's Drop once the manager is
        garbage collected). Segments mapped after this stay unpinned."""
        with self._wake_lock:
            if self._wake_q is not None:
                self._wake_q.put(None)
                self._wake_q = None
        self._unpin_finalizer()

    def cleanup_collectable(
        self, collectable: list[int], registered: list[bool],
    ):
        """``_cleanup_by_uuid`` for a batch, in one crossing per table: a decode
        step frees one tensor per request here."""
        if not collectable:
            return
        del registered  # no transfer engine to unregister from
        store = self.tensor_store
        store.mark_forgotten(collectable)
        self._transport.free(collectable)
        for side in (self.uuid_to_shard_dim, self.uuid_to_edge_name):
            if side:
                for uuid in collectable:
                    side.pop(uuid, None)
        for uuid in store.remove_tensors(collectable):
            logger.warning("Trying to cleanup tensor %s, but uuid not found", uuid)

    def _cleanup_by_uuid(self, uuid: int, registered: bool | None = None):
        super()._cleanup_by_uuid(uuid, registered)
        self._transport.free([uuid])
        if not self.tensor_store.check_uuid_presence(uuid):
            return
        self.tensor_store.remove_tensor(uuid)
