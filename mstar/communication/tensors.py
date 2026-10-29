import errno
import fcntl
import hashlib
import logging
import math
import os
import platform
import re
import secrets
import shutil
import stat
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from typing import NamedTuple

from mstar.communication.tensor_store import (  # NameToTensorList, Rid re-exported
    ColumnarTensorInfo,
    NameToTensorList,
    Rid,
    TensorStore,
)
from mstar.distributed.base import ShardingConfig
from mstar.observability.catalog import TRANSFER_API_SERVER
from mstar.observability.recorders import EdgeTransfers
from mstar.profile.format import RxInfo, TxInfo
from mstar.utils.containers import ParallelList

try:
    from mooncake.engine import TransferEngine
except Exception as _err:
    MOONCAKE_IMPORT_ERROR = _err
    TransferEngine = None
else:
    MOONCAKE_IMPORT_ERROR = None
import torch

from mstar.communication.communicator import (
    DEPLOYMENT_ANCHOR_ENTITY,
    BaseCommunicator,
    CommProtocol,
)
from mstar.communication.tensor_uuid import _DATA_WORKER as _DATA_WORKER_ENTITY
from mstar.communication.tensor_uuid import TensorUuidMinter
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo
from mstar.utils.ipc_format import TensorReceived, WorkerMessage, WorkerMessageType
from mstar.utils.profiler import PHASE_PERIOD, phase_record

logger = logging.getLogger(__name__)


class StoredOutputs(NamedTuple):
    """One batch's stored outputs, in the layout routing consumes.

    ``flat_uuids``/``flat_rids``/``signal_idxs`` are index-parallel, one entry
    per stored tensor. ``num_tensors`` has one entry per (request, signal)
    pair, request-major then signal order. Returned as columns rather than a
    dict keyed by request and signal because the only caller flattens it
    immediately -- building the dict just to take it apart again cost more
    than the store itself.
    """

    flat_uuids: list[int]
    flat_rids: list[int]
    signal_idxs: list[int]
    num_tensors: list[int]


@dataclass
class FutureAndPointers:
    future: Future | None
    graph_edges: list[GraphEdge]
    request_id: Rid = ""
    rx_time: float | None = None # seconds
    # What the read moved over the transport (see count_read), set only while
    # transfer metrics are on and cleared once counted. Unannotated, so class
    # attributes rather than fields: a read with them off stores nothing more.
    read_src = None
    read_bytes = 0
    read_tensors = 0
    # src -> [bytes, tensors] for producers after the first: the conductor
    # joins one name produced in several walks or nodes into one edge
    read_more = None

    def count_read(self, info: TensorPointerInfo) -> None:
        """Count one tensor this read moved over the transport."""
        src = info._source_node_name
        if src is None:
            # The data worker mints the request's inputs outside any node.
            # Every worker output carries its node; the entity id is a
            # bounded fallback should one ever arrive without.
            src = (TRANSFER_API_SERVER if info.source_entity == _DATA_WORKER_ENTITY
                   else info.source_entity)
        if src == self.read_src:
            self.read_bytes += info.nbytes
            self.read_tensors += 1
        elif self.read_src is None:
            self.read_src, self.read_bytes, self.read_tensors = src, info.nbytes, 1
        else:
            more = self.read_more
            if more is None:
                more = self.read_more = {}
            row = more.setdefault(src, [0, 0])
            row[0] += info.nbytes
            row[1] += 1


#: Earlier Mooncake releases let any host that reaches the TCP data port read
#: or write arbitrary process memory (CVE-2026-103764). NVD lists "before
#: 0.3.13" because it points at a later refactor; the fix (Mooncake #2314)
#: shipped in 0.3.12.post1, see https://github.com/kvcache-ai/Mooncake/issues/4626.
MOONCAKE_PATCHED_VERSION = "0.3.12.post1"
_MOONCAKE_DISTRIBUTIONS = (
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine-non-cuda",
)
_mooncake_version_checked = False


def _mooncake_dists_owning(engine_file: str | None, dists: list[str]) -> set[str]:
    """Which of *dists* installed ``engine_file``, the Mooncake engine this
    process imported: a RECORD entry that locates to it with its hash.

    One replaced in the same directory keeps a hash that no longer matches,
    and one installed elsewhere locates to another file. Empty when it can't
    be told; a distribution whose metadata can't be read is skipped, since
    this only refines a warning."""
    import base64
    from importlib.metadata import distribution
    if engine_file is None:
        return set()
    try:
        with open(engine_file, "rb") as fh:
            digest = base64.urlsafe_b64encode(hashlib.sha256(fh.read()).digest()).rstrip(b"=").decode()
    except OSError:
        return set()
    engine_path = os.path.realpath(engine_file)
    found = set()
    for dist in dists:
        try:
            for f in distribution(dist).files or []:
                if (f.name == os.path.basename(engine_file) and f.hash is not None
                        and f.hash.mode == "sha256" and f.hash.value == digest
                        and os.path.realpath(f.locate()) == engine_path):
                    found.add(dist)
        except Exception:  # noqa: BLE001 - e.g. a corrupt RECORD
            continue
    return found


def _warn_if_mooncake_unpatched() -> None:
    """Warn once per process if the installed Mooncake predates that fix.

    The extras require a patched release, so this only fires where the
    environment was built without resolving them (``--no-deps``, a prebuilt
    image). The distributions install the same ``mooncake`` package, and a
    replaced one can leave its metadata behind, so with several installed
    only the one that installed the imported engine counts; if that can't be
    told, any unpatched one warns, since the warning blocks nothing."""
    global _mooncake_version_checked
    if _mooncake_version_checked:
        return
    _mooncake_version_checked = True
    try:
        from packaging.version import Version
    except ImportError:
        return
    from importlib.metadata import PackageNotFoundError, version
    installed: list[tuple[str, Version]] = []
    for dist in _MOONCAKE_DISTRIBUTIONS:
        try:
            installed.append((dist, Version(version(dist))))
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - a version that can't be read or parsed (corrupt or stale metadata)
            return
    owner_unknown = False
    if len(installed) > 1:
        engine = sys.modules.get("mooncake.engine")
        owners = _mooncake_dists_owning(getattr(engine, "__file__", None), [d for d, _ in installed])
        if owners:
            installed = [(d, v) for d, v in installed if d in owners]
        else:
            owner_unknown = True
    patched = Version(MOONCAKE_PATCHED_VERSION)
    unpatched = [(d, v) for d, v in installed if v < patched]
    if unpatched and owner_unknown:
        logger.warning(
            "Several Mooncake distributions are installed (%s) and none can be "
            "matched to the imported engine; if it comes from one that predates "
            "%s, any host that can reach the transfer engine's TCP data port can "
            "read or write this process's memory (CVE-2026-103764). Uninstall the "
            "old ones, and keep the TCP and RDMA transports on a trusted network.",
            ", ".join(f"{d} {v}" for d, v in installed), MOONCAKE_PATCHED_VERSION,
        )
    elif installed and len(unpatched) == len(installed):
        logger.warning(
            "Installed Mooncake (%s) predates %s: any host that can reach the "
            "transfer engine's TCP data port can read or write this process's "
            "memory (CVE-2026-103764). Upgrade it, and keep the TCP and RDMA "
            "transports on a trusted network.",
            ", ".join(f"{d} {v}" for d, v in installed), MOONCAKE_PATCHED_VERSION,
        )


# ---------------------------------------------------------------------------
# TensorTransferEngine abstraction
# ---------------------------------------------------------------------------

class TensorTransferEngine(ABC):
    """Abstract interface for low-level memory registration and async reads.

    Wraps the transport-specific engine (Mooncake RDMA, local no-op, etc.)
    so that higher-level code (PagedAllocationManager, TensorCommunicationManager)
    never imports or depends on a specific transport library.
    """

    @abstractmethod
    def register_memory(self, ptr: int, nbytes: int) -> int:
        """Register a memory region for remote access. Returns 0 on success."""
        ...

    @abstractmethod
    def unregister_memory(self, ptr: int) -> int:
        """Unregister a previously registered memory region. Returns 0 on success."""
        ...

    @abstractmethod
    def get_async_reader(self, device) -> "AsyncMooncakeReader | None":
        """Return an async reader for background transfers, or None if not needed."""
        ...

    @abstractmethod
    def get_session_id(self) -> str:
        """Return the session ID for this engine (e.g., 'hostname:port')."""
        ...


# ---------------------------------------------------------------------------
# Mooncake implementation
# ---------------------------------------------------------------------------

@dataclass
class TransferReadInfo:
    source_session_id: str
    local_ptr: int
    remote_ptr: int
    nbytes: int
    fp: FutureAndPointers | None = None


class AsyncMooncakeReader:
    """Background thread for non-blocking mooncake READ operations.

    The caller records a CUDA event on the default stream and
    submits the write task to a thread pool. Worker thread waits on event via
    dedicated CUDA stream, then does blocking mooncake PUTs.
    The default stream is never blocked by store writes.
    """

    def __init__(
        self, engine, device,
        max_workers: int = 3,
        max_batch_size=500,
    ):
        self._engine = engine
        self.max_batch_size = max_batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: list[Future] = []
        # none on the host: host buffers have no GPU work to wait for, and a stream or event would
        # create a CUDA context on a GPU that other processes may have filled
        self._copy_stream = torch.cuda.Stream(device=device) if torch.device(device).type == "cuda" else None

    def submit(self, read_info: list[TransferReadInfo]) -> Future:
        """Non-blocking: enqueue a batch of READs.

        On a device, records a CUDA event on the current stream to ensure GPU data
        is ready before the background thread reads it.
        """
        if not read_info:
            return
        event = torch.cuda.current_stream().record_event() if self._copy_stream is not None else None
        future = self._executor.submit(self._do_read, read_info, event)
        self._pending.append(future)
        # Prune completed futures to avoid unbounded growth
        self._pending = [f for f in self._pending if not f.done()]
        return future

    def _do_read(self, read_info: list["TransferReadInfo"], event: torch.cuda.Event | None):
        """Worker thread: wait for GPU data via CUDA event, then PUT."""
        if event is not None:
            self._copy_stream.wait_event(event)
            self._copy_stream.synchronize()

        start_time = time.perf_counter()

        # group read_info by session id for batch read
        grouped_read = {}
        for info in read_info:
            grouped_read.setdefault(info.source_session_id, []).append(info)

        for (session_id, infos) in grouped_read.items():
            for start in range(0, len(infos), self.max_batch_size):
                end = min(start + self.max_batch_size, len(infos))

                status = self._engine.batch_transfer_sync_read(
                    session_id,
                    [infos[i].local_ptr for i in range(start, end)],
                    [infos[i].remote_ptr for i in range(start, end)],
                    [infos[i].nbytes for i in range(start, end)],
                )
                if status < 0:
                    raise RuntimeError(f"Mooncake read failed. Status: {status}")
        end_time = time.perf_counter()
        # The caller points every read of one edge at its FutureAndPointers
        # only when it wants the time (--log-stats or transfer metrics).
        if read_info[0].fp is not None:
            for info in read_info:
                info.fp.rx_time = (end_time - start_time)

    def wait_all(self):
        """Block until all pending writes complete. Re-raises exceptions."""
        for f in self._pending:
            f.result()
        self._pending.clear()

    def shutdown(self):
        """Wait for pending writes and shut down the thread pool."""
        self.wait_all()
        self._executor.shutdown(wait=True)


#: Mooncake's wheel per CUDA major. All of them install the same ``mooncake``
#: package; the default one is built for CUDA 12.
_MOONCAKE_WHEELS = {
    "12": "mooncake-transfer-engine",
    "13": "mooncake-transfer-engine-cuda13",
}


def _mooncake_install_hint(err: BaseException | None, torch_cuda: str | None) -> str:
    """Say which Mooncake wheel loads next to this torch build."""
    major = torch_cuda.split(".")[0] if torch_cuda else None
    want = _MOONCAKE_WHEELS.get(major) if major else "mooncake-transfer-engine-non-cuda"
    missing = re.search(r"libcudart\.so\.(\d+)", str(err))
    if want is None or missing is None:
        if want and isinstance(err, ModuleNotFoundError) and (err.name or "").startswith("mooncake"):
            return f"Install {want} or use SHM protocol."
        return "Install mooncake-transfer-engine or use SHM protocol."
    needed = missing.group(1)
    if needed == major:
        fix = (
            "`uv pip install nvidia-cuda-runtime-cu12`"
            if needed == "12"
            else f"put libcudart.so.{needed} on the library path"
        )
        return f"torch is built for CUDA {torch_cuda} too, so {fix}, or use SHM protocol."
    torch_desc = f"built for CUDA {torch_cuda}" if major else "built without CUDA"
    have = _MOONCAKE_WHEELS.get(needed)
    uninstall = f"uv pip uninstall {have} && " if have else ""
    return (
        f"This Mooncake build needs the CUDA {needed} runtime, but torch is {torch_desc}. "
        f"Switch to {want}: `{uninstall}uv pip install --reinstall --no-deps {want}`, "
        "or use SHM protocol."
    )


class MooncakeTransferEngine(TensorTransferEngine):
    """Wraps mooncake.engine.TransferEngine for RDMA or TCP transport."""

    def __init__(
        self,
        hostname: str,
        protocol: CommProtocol,
        metadata_server: str = "P2PHANDSHAKE",
        tcp_transfer_device: str = "",
    ):
        if TransferEngine is None:
            detail = (
                f"{type(MOONCAKE_IMPORT_ERROR).__name__}: "
                f"{MOONCAKE_IMPORT_ERROR}"
                if MOONCAKE_IMPORT_ERROR is not None
                else "unknown import failure"
            )
            raise RuntimeError(
                "Mooncake TransferEngine is required for RDMA/TCP protocol. "
                f"Failed to load mooncake: {detail}. "
                + _mooncake_install_hint(MOONCAKE_IMPORT_ERROR, torch.version.cuda)
            )
        _warn_if_mooncake_unpatched()

        if protocol == CommProtocol.RDMA:
            transfer_device = ""
        elif protocol == CommProtocol.TCP:
            transfer_device = tcp_transfer_device
        else:
            raise NotImplementedError(f"Unknown protocol {protocol} for mooncake")

        self._engine = TransferEngine()
        self._engine.initialize(
            hostname,
            metadata_server,
            protocol.value.lower(),
            transfer_device,
        )
        self._session_id = f"{hostname}:{self._engine.get_rpc_port()}"

    def register_memory(self, ptr: int, nbytes: int) -> int:
        return self._engine.register_memory(ptr, nbytes)

    def unregister_memory(self, ptr: int) -> int:
        return self._engine.unregister_memory(ptr)

    def get_async_reader(self, device) -> AsyncMooncakeReader:
        return AsyncMooncakeReader(self._engine, device=device)

    def get_session_id(self) -> str:
        return self._session_id


class LocalTransferEngine(TensorTransferEngine):
    """No-op engine for SHM / single-node — data is already in local GPU memory."""

    def __init__(self, hostname: str):
        self._session_id = hostname

    def register_memory(self, ptr: int, nbytes: int) -> int:
        return 0  # no-op

    def unregister_memory(self, ptr: int) -> int:
        return 0  # no-op

    def get_async_reader(self, device) -> None:
        return None  # no remote reads needed

    def get_session_id(self) -> str:
        return self._session_id


# ---------------------------------------------------------------------------
# TensorCommunicationManager base class (Comment 1: shared methods)
# ---------------------------------------------------------------------------

@dataclass
class BufferedShards:
    total_fanin: int
    shard_dim: int
    # source rank -> tensor
    shards: dict[int, list[torch.Tensor]]

    def is_done(self):
        return len(self.shards) >= self.total_fanin

    def consolidate(self) -> list[torch.Tensor]:
        assert self.is_done(), \
            "Can't consolidate shards until all fanin contributions are received"
        keys = sorted(self.shards.keys())
        n = len(self.shards[keys[0]])
        assert all(len(self.shards[k]) == n for k in keys), (
            f"BufferedShards: source ranks contributed unequal tensor counts: "
            f"{ {k: len(self.shards[k]) for k in keys} }"
        )
        return [
            torch.cat([self.shards[k][i] for k in keys], dim=self.shard_dim)
            for i in range(n)
        ]


EdgeNameToTxInfo = dict[str, TxInfo]
SourceAndEdgeToRxInfo = dict[tuple[str, str], RxInfo]


def _is_dense(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` fills exactly its ``nbytes`` from ``data_ptr()``,
    with no gaps or overlaps: contiguous up to a permutation of its dims."""
    if tensor.numel() == 0:
        return True
    expected = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


class TensorCommunicationManager(ABC):
    """Base class for inter-worker tensor transport.

    Holds common attributes and shared method implementations. Subclasses
    only need to override ``__init__``, ``register_for_send``,
    ``register_for_send_uuids``, ``start_read_tensors``, and
    ``_cleanup_by_uuid``.

    ``request_id`` is the worker's integer rid handle. The api-server data
    worker drives this same class with string request ids, which behaves
    identically because nothing here does more than use the id as an opaque
    key -- only ``rid_to_str`` cares, and its default (``str``) is the
    identity on strings.
    """

    # Transfer metrics, off until enable_transfer_metrics(); class-level so a
    # manager built without __init__ reads them as off too.
    transfer_stats: EdgeTransfers | None = None
    _transfer_dst: str | None = None
    #: The ``transport`` label: the protocol the bytes move over.
    transport_label: str = CommProtocol.SHM.value

    def __init__(
        self,
        my_entity_id: str,
        my_session_id: str,
        device: str,
        communicator: BaseCommunicator,
        transfer_engine: TensorTransferEngine,
        enable_prof: bool=False
    ):
        self.my_entity_id = my_entity_id
        self.my_session_id = my_session_id
        self.enable_prof = enable_prof
        self.device = device
        # by the manager's device, not torch.cuda.is_available() alone: a host manager holds only host tensors,
        # and its first sync would create a CUDA context on a GPU that other processes may have filled
        self._on_cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.communicator = communicator
        self.transfer_engine = transfer_engine
        self.tensor_store = TensorStore()
        # Everything here keys off the integer rid handle; the TENSOR_RECEIVED
        # ack is the only thing that leaves the process, so it is the only
        # place the string is needed. The worker installs the real de-interner.
        self.rid_to_str: Callable[[Rid], str] = str
        self._uuid_minter = TensorUuidMinter(my_entity_id)
        self.pending: list[FutureAndPointers] = []
        self.read_finished: dict[Rid, set[int]] = {}

        self.req_rx_info: dict[Rid, SourceAndEdgeToRxInfo] = {}
        self.req_tx_info: dict[Rid, EdgeNameToTxInfo] = {}
        # uuid -> edge/signal name, so register_for_send (which doesn't see
        # edge names) can attribute TX bytes/time to the right edge. Only tracked under
        # ``enable_prof``.
        self.uuid_to_edge_name: dict[int, str] = {}

        # req_id -> cfg
        self.sharding_configs: dict[Rid, ShardingConfig] = {}
        # req_id -> {name -> shards}
        self.buffered_shards: dict[Rid, dict[str, BufferedShards]] = {}
        self.uuid_to_shard_dim: dict[int, int | None] = {}
        self._flat_copy_stream: torch.cuda.Stream | None = None
        self._flat_copy_pending = False

    @property
    def needs_cpu_tensor(self) -> bool:
        """Whether this transport sends from host memory, so the store keeps
        the producer's host copies (``cpu_tensors``) for the send to reuse.
        False for transports that read device memory (TCP / RDMA)."""
        return False

    #: Whether a consumer reads a stored tensor as ``nbytes`` contiguous bytes
    #: from its ``address`` (Mooncake RDMA / TCP), so the store must hold a
    #: layout that one flat read reproduces. False for transports that
    #: serialize a contiguous copy at send time (SHM). An attribute, not a
    #: property: every store call reads it.
    needs_flat_layout: bool = False

    # ---- shared: store ----
    def _flat_layout(
        self, tensor: torch.Tensor, sharded: bool,
        ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """``tensor`` if one flat read of its ``nbytes`` from ``data_ptr()``
        reproduces it, else a contiguous copy. Only for a non-contiguous
        tensor on a ``needs_flat_layout`` transport.

        A dense permutation (a transpose) is kept as is: the consumer
        rebuilds it from the strides. A sharded tensor must be contiguous,
        since its slices are byte ranges of rows. A CUDA copy runs on a side
        stream so it does not queue behind the next step; the store waits for
        it before returning the descriptors.

        The side stream is ordered after the producer only by ``ready_event``
        (the step's completion event). Without one, the producer's writes must
        already be complete when the store is called, not just queued ahead on
        some stream: the default-stream sync of ``skip_cuda_sync=False``, or
        the caller's own wait.
        """
        if not sharded and _is_dense(tensor):
            return tensor
        if not tensor.is_cuda:
            return tensor.contiguous()
        stream = self._flat_copy_stream
        if stream is None:
            stream = self._flat_copy_stream = torch.cuda.Stream(device=tensor.device)
        if ready_event is not None:
            stream.wait_event(ready_event)
        with torch.cuda.stream(stream):
            copy = tensor.contiguous()
        self._flat_copy_pending = True
        return copy

    def _wait_flat_copies(self) -> None:
        self._flat_copy_pending = False
        self._flat_copy_stream.synchronize()

    @staticmethod
    def _host_copies_by_tensor(
        tensors: NameToTensorList, cpu_tensors: NameToTensorList | None,
    ) -> dict[int, torch.Tensor]:
        """One request's host copies, keyed by ``id`` of the device tensor
        each mirrors (matched by name and position in the producer's output).

        Keyed by tensor, not name, because outputs get renamed on the way to
        their signal (Qwen3.5 stores ``new_token`` as ``text_inputs``).
        """
        if not cpu_tensors:
            return {}
        by_tensor: dict[int, torch.Tensor] = {}
        for name, cpu_list in cpu_tensors.items():
            device_list = tensors.get(name)
            if not device_list or not isinstance(cpu_list, list):
                continue
            for device_tensor, cpu in zip(device_list, cpu_list, strict=False):
                by_tensor[id(device_tensor)] = cpu
        return by_tensor

    def _cpu_counterpart(
        self, by_tensor: dict[int, torch.Tensor], tensor: torch.Tensor,
        canonical: torch.Tensor, shard_dim: int | None,
    ) -> torch.Tensor | None:
        """The producer's host copy of ``tensor``, if it gave a usable one: a
        host tensor of the same shape and dtype, else None (the send then
        copies from the device tensor)."""
        cpu = by_tensor.get(id(tensor))
        if not torch.is_tensor(cpu) or cpu.device.type != "cpu":
            return None
        cpu = self._ensure_leading_shard_dim(shard_dim, cpu)
        if cpu.shape != canonical.shape or cpu.dtype != canonical.dtype:
            return None
        return cpu

    def _ensure_leading_shard_dim(self, shard_dim: int | None, tensor: torch.Tensor):
        """Move ``shard_dim`` to dim 0, preserving the relative order of the
        remaining dims. The inverse is ``_undo_leading_shard_dim``."""
        if shard_dim is None or shard_dim == 0:
            return tensor
        perm = [shard_dim] + [i for i in range(tensor.ndim) if i != shard_dim]
        return tensor.permute(*perm).contiguous()

    def _undo_leading_shard_dim(self, shard_dim: int | None, tensor: torch.Tensor):
        """Inverse of ``_ensure_leading_shard_dim``: move dim 0 back to position
        ``shard_dim``, preserving the relative order of the remaining dims."""
        if shard_dim is None or shard_dim == 0:
            return tensor
        inverse_perm = (
            list(range(1, shard_dim + 1)) + [0] + list(range(shard_dim + 1, tensor.ndim))
        )
        return tensor.permute(*inverse_perm).contiguous()

    def _source_tp(
        self, cfg: ShardingConfig | None,
        node_name: str | None, graph_walk: str | None,
    ) -> tuple[int, int]:
        """This rank's ``(tp_size, tp_rank)`` for the node being stored.

        ``register_request`` is a hard prereq on the worker side, but the API
        server calls the store without registering, so a missing config means
        "no group" / TP=1 rather than an error.
        """
        source_group = (
            cfg.get_sharding_group(node_name, graph_walk)
            if cfg is not None else None
        )
        if source_group is None:
            return 1, 0
        return source_group.tp_size, source_group._tp_rank or 0

    def _mint_tensor_infos(
        self, request_id: Rid, tensors: NameToTensorList,
        node_name: str | None, graph_walk: str | None,
        out_rids: list[int], out_uuids: list[int],
        out_tensors: list[torch.Tensor],
        out_infos: list[TensorPointerInfo] | None = None,
        cpu_tensors: NameToTensorList | None = None,
        out_cpu: list[torch.Tensor | None] | None = None,
    ) -> dict[str, list[TensorPointerInfo]]:
        """Mint uuids and descriptors for one request's outputs.

        Touches no store: the flat ``(request_id, uuid, tensor, info)`` columns are
        appended to the caller's lists, so the caller decides how wide a batch
        to hand ``put_tensor_batch_multi``. Minting order is name then tensor,
        which is what makes the caller's flat columns index-parallel.
        """
        cfg = self.sharding_configs.get(request_id)
        source_tp_size, source_tp_rank = self._source_tp(
            cfg, node_name, graph_walk,
        )

        by_tensor = (
            self._host_copies_by_tensor(tensors, cpu_tensors)
            if out_cpu is not None else {}
        )
        flat = self.needs_flat_layout
        tensor_info: dict[str, list[TensorPointerInfo]] = {}
        for name, tensor_list in tensors.items():
            tensor_info[name] = []

            shard_dim = cfg.shard_dim.get(name) if cfg is not None else None
            for tensor in tensor_list:
                tensor_uuid = self._uuid_minter.mint()
                # TODO: only rearrange when (1) the tensor will be sent and
                # (2) it may be split along the shard dim in transport. Doing
                # it here unconditionally so TensorPointerInfo dims/strides
                # match what receivers will read.
                canonical = self._ensure_leading_shard_dim(shard_dim, tensor)
                if flat and not canonical.is_contiguous():
                    canonical = self._flat_layout(canonical, shard_dim is not None)
                # Built before the put: the store keeps the descriptor so an
                # edge can be rebuilt from its uuid alone. This is the same
                # object the edge carries, so later in-place edits (the
                # shm_segment register_for_send stamps on) are visible to both.
                info = TensorPointerInfo(
                    dims=canonical.shape,
                    dtype=canonical.dtype,
                    stride=canonical.stride(),
                    nbytes=canonical.nbytes,
                    address=canonical.data_ptr(),
                    uuid=tensor_uuid,
                    source_session_id=self.my_session_id,
                    source_entity=self.my_entity_id,
                    source_tp_size=source_tp_size,
                    source_tp_rank=source_tp_rank,
                    _source_node_name=node_name,
                    _source_graph_walk=graph_walk,
                )
                out_rids.append(request_id)
                out_uuids.append(tensor_uuid)
                out_tensors.append(canonical)
                out_infos.append(info)
                if out_cpu is not None:
                    out_cpu.append(self._cpu_counterpart(
                        by_tensor, tensor, canonical, shard_dim,
                    ))
                if cfg is not None:
                    self.uuid_to_shard_dim[tensor_uuid] = shard_dim
                if self.enable_prof:
                    self.uuid_to_edge_name[tensor_uuid] = name

                logger.debug("Storing tensor name %s uuid %s", name, tensor_uuid)
                tensor_info[name].append(info)
        return tensor_info

    def store_and_return_tensor_info(
        self, request_id: Rid, tensors: NameToTensorList,
        node_name: str | None=None,
        graph_walk: str | None=None,
        skip_cuda_sync: bool = False,
        cpu_tensors: NameToTensorList | None = None,
    ) -> dict[str, list[TensorPointerInfo]]:
        """Store one request's outputs and return their descriptors.

        ``cpu_tensors`` mirrors ``tensors`` with host copies the caller already
        made; they are kept only if ``needs_cpu_tensor``.
        """
        # CUDA sync ensures GPU writes to ``tensors`` are visible before
        # callers hand out ``tensor.data_ptr()`` to peers (RDMA register,
        # SHM serialize). With same-thread async scheduling the caller
        # has typically already synced on ``output.completion_event``
        # (which waits for *only* the producing step, not the queued next
        # step), so the unconditional default-stream sync here would
        # uselessly drain GPU(N+1). Pass ``skip_cuda_sync=True`` from
        # those call sites, and only once the producer's writes are complete:
        # a flat-layout copy (``_flat_layout``) reads them during this call.
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()
        request_ids: list[Rid] = []
        uuids: list[int] = []
        canonicals: list[torch.Tensor] = []
        infos: list[TensorPointerInfo] = []
        want_cpu = cpu_tensors is not None and self.needs_cpu_tensor
        cpus: list[torch.Tensor | None] | None = [] if want_cpu else None
        tensor_info = self._mint_tensor_infos(
            request_id, tensors, node_name, graph_walk,
            request_ids, uuids, canonicals, infos,
            cpu_tensors=cpu_tensors if want_cpu else None, out_cpu=cpus,
        )
        self.tensor_store.put_tensor_batch_multi(
            request_ids, uuids, canonicals, infos, cpu_tensors=cpus,
        )
        if self._flat_copy_pending:
            self._wait_flat_copies()
        return tensor_info

    def store_row_outputs_batch(
        self,
        request_ids: list[Rid],
        outputs,
        signals: list[str],
        row_views: dict[str, tuple[torch.Tensor, ...]],
        row_request_ids: tuple[Rid, ...],
        node_name: str | None = None,
        graph_walk: str | None = None,
        host_rows=None,
    ) -> StoredOutputs | None:
        """``store_and_return_tensor_info_batch`` for a step whose per-request
        outputs are views of row-addressed batch tensors (a decode step: row i
        of ``new_token`` is request i).

        The rows of one batch tensor differ only in address, so their
        descriptors are built from one row's shape, stride and dtype instead
        of being read off every view, and the host copies come from the stop
        check's row buffers (``host_rows``) by index rather than being matched
        tensor by tensor. Every (request, signal) entry must be exactly one
        tensor that *is* a row view, or the result is None and the caller
        takes the general path; nothing is recorded before that is settled.
        What is stored is what the general path would store: one uuid per
        (request, signal), request-major, the same row views.
        """
        if not row_views or not self.tensor_store.has_put_tensor_batch_columns:
            return None
        row_of = {rid: i for i, rid in enumerate(row_request_ids)}
        sharding = self.sharding_configs
        first_cfg = sharding.get(request_ids[0]) if request_ids else None
        tp_size, tp_rank = self._source_tp(first_cfg, node_name, graph_walk)
        columns = ColumnarTensorInfo(
            source_session_id=self.my_session_id,
            source_entity=self.my_entity_id,
            source_node_name=node_name,
            source_graph_walk=graph_walk,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        want_cpu = host_rows is not None and self.needs_cpu_tensor
        flat = self.needs_flat_layout
        # signal -> the row tensor it views, learned from the first request
        # that has it and checked on every other
        alias: dict[str, str] = {}
        # row tensor -> (base address, row step, nbytes, ndims, dims, stride,
        # dtype index, host rows or None)
        template: dict[str, tuple] = {}
        flat_uuids: list[int] = []
        flat_rids: list[int] = []
        signal_idxs: list[int] = []
        num_tensors: list[int] = []
        canonicals: list[torch.Tensor] = []
        cpus: list[torch.Tensor | None] | None = [] if want_cpu else None
        sharded_uuids: list[int] = []
        mint = self._uuid_minter.mint
        add_row = columns.add_row
        try:
            for rid in request_ids:
                i = row_of.get(rid)
                if i is None:
                    return None
                rid_out = outputs.get(rid)
                cfg = sharding.get(rid)
                if cfg is not first_cfg and self._source_tp(
                    cfg, node_name, graph_walk,
                ) != (tp_size, tp_rank):
                    raise ValueError(
                        f"request {rid} has sharding "
                        f"{self._source_tp(cfg, node_name, graph_walk)} for "
                        f"node {node_name!r} walk {graph_walk!r}, but the batch "
                        f"was built for {(tp_size, tp_rank)}; one batch is one "
                        "node on one worker and cannot mix sharding groups"
                    )
                for s, signal in enumerate(signals):
                    lst = rid_out.get(signal) if rid_out else None
                    if not lst:
                        num_tensors.append(0)
                        continue
                    if type(lst) is not list or len(lst) != 1:
                        return None
                    tensor = lst[0]
                    name = alias.get(signal)
                    if name is None:
                        for cand, views in row_views.items():
                            if views[i] is tensor:
                                name = cand
                                break
                        if name is None:
                            return None
                        alias[signal] = name
                        if name not in template:
                            # rows of one tensor share a layout; one that a
                            # flat read cannot reproduce is copied per tensor
                            # by the general path
                            if flat and not row_views[name][0].is_contiguous():
                                return None
                            template[name] = self._row_template(
                                views, columns, host_rows if want_cpu else None, name,
                            )
                    elif row_views[name][i] is not tensor:
                        return None
                    # a sharded output is permuted on store; not a plain row
                    if cfg is not None and cfg.shard_dim.get(signal):
                        return None
                    base, step, nbytes, ndims, dims, stride, dtype_idx, host = template[name]
                    uuid = mint()
                    flat_uuids.append(uuid)
                    flat_rids.append(rid)
                    signal_idxs.append(s)
                    num_tensors.append(1)
                    canonicals.append(tensor)
                    add_row(uuid, base + i * step, nbytes, ndims, dims, stride, dtype_idx)
                    if want_cpu:
                        cpus.append(host[i] if host is not None else None)
                    if cfg is not None:
                        sharded_uuids.append(uuid)
        except IndexError:
            # a row index past the views or the host rows: not this batch's rows
            return None
        # nothing above touched shared state; from here the general path's
        # guarantees apply
        shard_dims = self.uuid_to_shard_dim
        for uuid in sharded_uuids:
            shard_dims[uuid] = None
        edge_names = self.uuid_to_edge_name
        if self.enable_prof:
            for uuid, s in zip(flat_uuids, signal_idxs, strict=True):
                edge_names[uuid] = signals[s]
        try:
            self.tensor_store.put_tensor_batch_columns(
                flat_rids, canonicals, columns, cpu_tensors=cpus,
            )
        except Exception:
            for uuid in flat_uuids:
                shard_dims.pop(uuid, None)
                edge_names.pop(uuid, None)
            raise
        return StoredOutputs(flat_uuids, flat_rids, signal_idxs, num_tensors)

    @staticmethod
    def _row_template(
        views: tuple[torch.Tensor, ...], columns: ColumnarTensorInfo,
        host_rows, name: str,
    ) -> tuple:
        """What every row of one batch tensor shares, for the row store: the
        first row's address and the step to the next, byte size, shape,
        stride and dtype index, plus the matching host rows (one view per row)
        when ``host_rows`` carries a buffer of this name with the rows' shape
        and dtype."""
        first = views[0]
        base = first.data_ptr()
        step = views[1].data_ptr() - base if len(views) > 1 else 0
        dims = tuple(first.shape)
        host = None
        if host_rows is not None:
            buf = host_rows.buffers.get(name)
            if (
                torch.is_tensor(buf) and buf.device.type == "cpu"
                and buf.dim() == len(dims) and buf.dtype == first.dtype
                and tuple(buf.shape[1:]) == dims[1:] and buf.shape[0] >= len(views)
            ):
                host = buf.split(1)
        return (
            base, step, first.nbytes, len(dims), dims, tuple(first.stride()),
            columns.dtype_index(first.dtype), host,
        )

    def store_and_return_tensor_info_batch(
        self,
        request_ids: list[Rid],
        outputs: dict[Rid, NameToTensorList],
        signals: list[str],
        node_name: str | None=None,
        graph_walk: str | None=None,
        skip_cuda_sync: bool = False,
        cpu_tensors: dict[Rid, NameToTensorList] | None = None,
        ready_event: torch.cuda.Event | None = None,
    ) -> StoredOutputs:
        """Store a whole output batch and return the columns routing needs.

        One pass: the flat columns are filled as the uuids are minted, so no
        per-request or per-signal dict is built to be taken apart again. The
        mint order is therefore request-major then signal order, which is the
        order the columns are in -- that is what keeps them index-parallel.

        Which form is marshalled depends on the backend. A bookkeeper that
        pays per tensor to marshal (the Rust one) gets columns of primitives;
        one that stores the descriptor by reference (the Python one) gets
        descriptors, since building columns for it would only add work.

        Outputs under a name no signal carries are not stored at all: no edge
        will ever read them, and a stored tensor nothing references is only
        freed when its request is torn down.

        ``cpu_tensors`` mirrors ``outputs`` with host copies the caller already
        made (e.g. the stop check's), matched as in ``_host_copies_by_tensor``;
        kept only if ``needs_cpu_tensor``.

        ``skip_cuda_sync=True`` as in ``store_and_return_tensor_info``: the
        producer's writes must be complete, unless ``ready_event`` (the
        producing step's completion event) is given, which a flat-layout copy
        waits on before it reads them.
        """
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()

        flat_uuids: list[int] = []
        flat_rids: list[int] = []
        signal_idxs: list[int] = []
        num_tensors: list[int] = []
        canonicals: list[torch.Tensor] = []
        want_cpu = cpu_tensors is not None and self.needs_cpu_tensor
        cpus: list[torch.Tensor | None] | None = [] if want_cpu else None
        flat = self.needs_flat_layout

        columnar = self.tensor_store.has_put_tensor_batch_columns
        infos: list[TensorPointerInfo] = []
        columns: ColumnarTensorInfo | None = None

        if columnar:
            # Invariant across one node's output batch, so carried once for
            # the whole batch and interned once on the far side.
            first_request_id = next(iter(request_ids), None)
            tp_size, tp_rank = self._source_tp(
                self.sharding_configs.get(first_request_id), node_name, graph_walk,
            )
            columns = ColumnarTensorInfo(
                source_session_id=self.my_session_id,
                source_entity=self.my_entity_id,
                source_node_name=node_name,
                source_graph_walk=graph_walk,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )

        try:
            for request_id in request_ids:
                tensors = outputs.get(request_id) or {}
                by_tensor = (
                    self._host_copies_by_tensor(
                        tensors, cpu_tensors.get(request_id),
                    ) if want_cpu else {}
                )
                cfg = self.sharding_configs.get(request_id)
                source_tp_size, source_tp_rank = self._source_tp(
                    cfg, node_name, graph_walk,
                )
                if columnar and (source_tp_size, source_tp_rank) != (
                    columns.tp_size, columns.tp_rank
                ):
                    raise ValueError(
                        f"request {request_id} has sharding "
                        f"{(source_tp_size, source_tp_rank)} for node "
                        f"{node_name!r} walk {graph_walk!r}, but the batch was "
                        f"built for {(columns.tp_size, columns.tp_rank)}; one "
                        "batch is one node on one worker and cannot mix "
                        "sharding groups"
                    )

                for i, signal in enumerate(signals):
                    tensor_list = tensors.get(signal)
                    if not tensor_list:
                        num_tensors.append(0)
                        continue
                    shard_dim = (
                        cfg.shard_dim.get(signal) if cfg is not None else None
                    )
                    for tensor in tensor_list:
                        tensor_uuid = self._uuid_minter.mint()
                        canonical = self._ensure_leading_shard_dim(
                            shard_dim, tensor,
                        )
                        if flat and not canonical.is_contiguous():
                            canonical = self._flat_layout(
                                canonical, shard_dim is not None, ready_event,
                            )
                        flat_uuids.append(tensor_uuid)
                        flat_rids.append(request_id)
                        signal_idxs.append(i)
                        canonicals.append(canonical)
                        if want_cpu:
                            cpus.append(self._cpu_counterpart(
                                by_tensor, tensor, canonical, shard_dim,
                            ))
                        if not columnar:
                            infos.append(self._tensor_info(
                                canonical, tensor_uuid, source_tp_size,
                                source_tp_rank, node_name, graph_walk,
                            ))
                        if cfg is not None:
                            self.uuid_to_shard_dim[tensor_uuid] = shard_dim
                        if self.enable_prof:
                            self.uuid_to_edge_name[tensor_uuid] = signal
                    num_tensors.append(len(tensor_list))

            if columnar:
                columns.add_tensors_canonical(flat_uuids, canonicals)
                self.tensor_store.put_tensor_batch_columns(
                    flat_rids, canonicals, columns, cpu_tensors=cpus,
                )
            else:
                self.tensor_store.put_tensor_batch_multi(
                    flat_rids, flat_uuids, canonicals, infos, cpu_tensors=cpus,
                )
        except Exception:
            # The per-tensor bookkeeping above is written before the store
            # put lands, so on any raise those uuids would sit in the maps
            # with nothing in the store -- and both cleanup paths iterate
            # ``get_all_uuids(rid)``, so nothing could ever reach them.
            # ``flat_uuids`` is appended before either map, so it covers them.
            for uuid in flat_uuids:
                self.uuid_to_shard_dim.pop(uuid, None)
                self.uuid_to_edge_name.pop(uuid, None)
            raise
        if self._flat_copy_pending:
            self._wait_flat_copies()
        return StoredOutputs(flat_uuids, flat_rids, signal_idxs, num_tensors)

    def _tensor_info(
        self, canonical: torch.Tensor, tensor_uuid: int,
        source_tp_size: int, source_tp_rank: int,
        node_name: str | None, graph_walk: str | None,
    ) -> TensorPointerInfo:
        """One output's descriptor. Only the non-columnar backends build these
        -- see ``store_and_return_tensor_info_batch``."""
        return TensorPointerInfo(
            dims=canonical.shape,
            dtype=canonical.dtype,
            stride=canonical.stride(),
            nbytes=canonical.nbytes,
            address=canonical.data_ptr(),
            uuid=tensor_uuid,
            source_session_id=self.my_session_id,
            source_entity=self.my_entity_id,
            source_tp_size=source_tp_size,
            source_tp_rank=source_tp_rank,
            _source_node_name=node_name,
            _source_graph_walk=graph_walk,
        )

    def set_output_ref_counts(
        self,
        safety_hold_uuids: set[int],
        routed_edges: list[GraphEdge],
    ):
        """Adjust ref counts from the safety hold (1) to the actual fanout.

        Called after ``process_node_outputs`` determines the real routing.
        ``safety_hold_uuids`` is the set of UUIDs that were given ref=1 by
        ``increment_ref_batch_uniform`` after
        ``store_and_return_tensor_info_batch``.
        ``routed_edges`` is the flat list of all edges that will actually be
        consumed (local ingestion, remote send, persist, emit, streaming).
        """
        actual_counts: dict[int, int] = {uuid: 0 for uuid in safety_hold_uuids}
        for edge in routed_edges:
            for info in edge.tensor_info:
                if info.uuid in actual_counts:
                    actual_counts[info.uuid] += 1

        up_uuids: list[int] = []
        up_counts: list[int] = []
        down_uuids: list[int] = []
        down_counts: list[int] = []
        for uuid, count in actual_counts.items():
            delta = count - 1  # subtract the safety hold of 1
            if delta > 0:
                up_uuids.append(uuid)
                up_counts.append(delta)
            elif delta < 0:
                down_uuids.append(uuid)
                down_counts.append(-delta)
        # Two crossings for a node's whole output batch rather than one per
        # tensor: this settles every output of every request in the batch.
        if up_uuids:
            self.tensor_store.increment_ref_batch(up_uuids, up_counts)
        if down_uuids:
            self.dereference_batch(down_uuids, down_counts)

    # ---- shared: descriptors about to go on the wire ----

    #: Whether this transport writes placement onto a descriptor AFTER routing
    #: has already copied it. Only the SHM arena does (``shm_segment`` /
    #: ``shm_offset``); the per-uuid file and Mooncake paths derive everything
    #: a consumer needs from ``uuid`` and ``address``, so they leave this off
    #: and ``refresh_shm_placement`` stays free for them.
    stamps_shm_placement: bool = False

    def refresh_shm_placement(self, edges: list[GraphEdge]) -> None:
        """Re-read arena placement from the store onto descriptors about to be
        sent.

        Registration stamps ``shm_segment`` / ``shm_offset`` onto the
        descriptor the STORE holds, and that is the only copy it can reach --
        ``register_for_send_uuids`` is handed uuids, not objects. A SHARDED
        edge does not carry the store's copy: ``fanout_graph_edges`` clones it
        per destination to rewrite dims/nbytes/offset, and it clones during
        routing, which runs BEFORE staging. So the clone never sees the stamp
        and ships ``shm_segment=None``, which a consumer reads as "the producer
        spilled this one to a per-uuid file" -- and then opens a file the arena
        producer never wrote.

        Called at each point an edge leaves this process, after
        ``register_for_send*`` has run. Only the two placement fields are
        copied: everything else on the clone is this destination's slice and
        must not be overwritten.
        """
        if not self.stamps_shm_placement:
            return
        for edge in edges:
            for info in edge.tensor_info:
                stored = self.tensor_store.get_info(info.uuid)
                # `is`: the replicated fanout hands on the store's own object
                # under the Python bookkeeper, so there is nothing to copy.
                if stored is None or stored is info:
                    continue
                info.shm_segment = stored.shm_segment
                info.shm_offset = stored.shm_offset

    # ---- abstract: transport-specific ----

    @abstractmethod
    def register_for_send_uuids(
        self,
        per_request: ParallelList[Rid, list[int]],
        skip_cuda_sync: bool = False,
    ):
        """Mark every request's tensors ready for remote consumers, given
        uuids rather than descriptors.

        Registration reads exactly one field off a descriptor -- ``uuid`` --
        and takes the tensor itself from the store. Under the Rust bookkeeper,
        rebuilding a descriptor so the callee can read its uuid costs ~2.2us
        each (15 attribute crossings plus the dataclass), and the caller
        already has the uuids. ``skip_cuda_sync`` as for ``register_for_send``.
        """
        ...

    @abstractmethod
    def register_for_send(
        self, request_id: Rid, tensor_infos: list[TensorPointerInfo],
        skip_cuda_sync: bool = False,
    ):
        """Mark these tensors ready for remote consumers to RDMA-read.

        ``skip_cuda_sync=True`` skips the default-stream sync this call normally
        issues to ensure the source tensors' writes are visible before their
        addresses are shared with peers. Callers must have already synced on
        their own (e.g. before a batched loop) — meant to cut N serialized
        syncs to 1 when registering many tensors in a row.

        If self.enable_prof is set, this should also update self.req_tx_info
        """
        ...

    def _slice_existing_tensor(
        self, request_id: Rid, name: str, next_node: str,
        graph_walk: str | None, info: TensorPointerInfo
    ):
        shard_dim = self.sharding_configs[request_id].shard_dim.get(name)
        dest_group = self.sharding_configs[request_id].group_mapping.get(
            NodeAndGraphWalk(next_node, graph_walk)
        )
        dest_tp_size = dest_group.tp_size if dest_group is not None else 1
        if info.source_tp_size != dest_tp_size and shard_dim is not None:
            canonical_tensor = self.tensor_store.get_tensor(info.uuid)
            # Canonical layout has shard_dim leading and is contiguous, so a
            # contiguous byte range maps to a contiguous range of rows along
            # dim 0. Slicing along dim 0 keeps the view contiguous.
            bytes_per_row = (
                canonical_tensor.element_size()
                * math.prod(canonical_tensor.shape[1:])
            )
            if bytes_per_row == 0:
                # Rows of zero bytes (a [4, 0] tensor) carry no offset to
                # recover the start from, and hold nothing to read: only the
                # row count matters.
                start, end = 0, info.dims[0]
            else:
                assert info.offset % bytes_per_row == 0 and info.nbytes % bytes_per_row == 0, (
                    f"slice offset {info.offset} / nbytes {info.nbytes} not aligned "
                    f"to canonical row size {bytes_per_row}"
                )
                start = info.offset // bytes_per_row
                end = start + info.nbytes // bytes_per_row
            slice_view = canonical_tensor[start:end]
            new_uuid = self._uuid_minter.mint()
            self.uuid_to_shard_dim[new_uuid] = shard_dim
            # Release this edge's stake on the producer's UUID — the slice now
            # owns it. The slice view keeps the underlying storage alive even
            # if the producer's UUID GCs.
            self.dereference(info.uuid, 1)
            info.uuid = new_uuid
            # After the re-point, so the descriptor the store keeps already
            # names the slice rather than the producer's tensor.
            self.tensor_store.put_tensor(request_id, new_uuid, slice_view, info)

    @abstractmethod
    def start_read_tensors(
        self, request_id: Rid, graph_edges: list[GraphEdge],
        graph_walk: str | None = None
    ) -> list[Future]:
        ...

    def begin_read_batch(self) -> None:  # noqa: B027
        """Open a read batch: until ``end_read_batch``, ``start_read_tensors``
        may defer copies so that the batch's reads share them, and returns no
        futures for what it deferred. A transport with nothing to share
        ignores this."""

    def end_read_batch(self) -> list[Future]:
        """Issue what the batch deferred and close it. Returns futures that
        complete when the copies land."""
        return []

    def _cleanup_by_uuid(self, uuid: int, registered: bool | None = None):
        """``registered`` is the transport's ``mem_registered`` flag, passed in
        by a batched dereference that has already dropped the record it lives
        on. ``None`` means the record is still there -- ask the store."""
        self.uuid_to_shard_dim.pop(uuid, None)
        self.uuid_to_edge_name.pop(uuid, None)

    # ---- shared: polling & ACKs ----

    def _collect_and_send_acks(
        self, request_id: Rid | None, graph_edges: list[GraphEdge],
        wire_request_id: str | None = None,
    ):
        """``wire_request_id`` overrides ``rid_to_str(request_id)``, for a
        request that no longer has a handle."""
        acks: dict[str, dict[int, int]] = {}
        for edge in graph_edges:
            for info in edge.tensor_info:
                if info.source_entity not in acks:
                    acks[info.source_entity] = {}
                acks[info.source_entity][info.uuid] = acks[info.source_entity].get(
                    info.uuid, 0) + 1
        for source_entity, tensors in acks.items():
            if source_entity == self.my_entity_id:
                continue
            if wire_request_id is None:
                wire_request_id = self.rid_to_str(request_id)
            self.communicator.send(
                source_entity,
                WorkerMessage(
                    message_type=WorkerMessageType.TENSOR_RECEIVED,
                    body=TensorReceived(
                        request_id=wire_request_id,
                        successful_tensors=tensors,
                        failed_tensor_ids=[],
                    ),
                ),
            )

    def ack_unread_tensors(
        self, request_id: str, graph_edges: list[GraphEdge],
    ):
        """Ack tensors for an already-removed request without reading them.

        The producer holds these buffers until it gets the TENSOR_RECEIVED ack;
        emit it here so it can reclaim them. ``request_id`` is the wire string:
        a removed request has no handle on a worker.
        """
        self._collect_and_send_acks(None, graph_edges, wire_request_id=request_id)

    def enable_transfer_metrics(self) -> EdgeTransfers:
        """Count what this manager reads from now on (see ``EdgeTransfers``).
        Called once, before or while reads run on the reading thread."""
        if self.my_entity_id == _DATA_WORKER_ENTITY:
            self._transfer_dst = TRANSFER_API_SERVER
        self.transfer_stats = EdgeTransfers(self.transport_label)
        return self.transfer_stats

    def _record_transfer(self, stats: EdgeTransfers, ep: FutureAndPointers) -> None:
        """One delivered edge read: one transfer per producing node. The edge
        is read as a whole, so an edge joined from several producers splits
        its read time between them by bytes (by tensors if it moved none)."""
        src = ep.read_src
        # Counted once: a later future that raises leaves this entry pending
        # for the next poll.
        ep.read_src = None
        dst = self._transfer_dst or ep.graph_edges[0].next_node
        seconds = ep.rx_time or 0.0
        if ep.read_more is None:
            stats.record(src, dst, ep.read_bytes, ep.read_tensors, seconds)
            return
        rows = {src: [ep.read_bytes, ep.read_tensors], **ep.read_more}
        total = sum(r[0] for r in rows.values())
        col = 0 if total else 1
        total = total or sum(r[1] for r in rows.values())
        for src, row in rows.items():
            stats.record(src, dst, row[0], row[1], seconds * row[col] / total)

    def get_ready_tensors(self, graph_walk: str | None=None) -> dict[Rid, list[GraphEdge]]:
        ready: dict[Rid, list[GraphEdge]] = {}
        still_pending = []
        uuid_to_time = {}
        stats = self.transfer_stats
        # the async reader finishes a request's reads in any order, so a read
        # that is done waits for the ones started before it: a streamed token
        # must not reach the client ahead of the one before it
        waiting: set[Rid] = set()
        for ep in self.pending:
            done = ep.future is None or ep.future.done()
            if done and ep.request_id not in waiting:
                if ep.future is not None:
                    ep.future.result()
                if stats is not None and ep.read_src is not None:
                    self._record_transfer(stats, ep)
                for edge in ep.graph_edges:
                    ready.setdefault(ep.request_id, []).append(edge)
                    logger.debug(
                        "Finished reading in %d tensors %s for graph node %s",
                        len(edge.tensor_info), edge.name, edge.next_node
                    )
                    if self.enable_prof:
                        for info in edge.tensor_info:
                            uuid_to_time[info.uuid] = ep.rx_time
            else:
                waiting.add(ep.request_id)
                still_pending.append(ep)
        self.pending = still_pending

        final_ready: dict[Rid, list[GraphEdge]] = {}
        for req_id, edges in ready.items():
            self._collect_and_send_acks(req_id, edges)
            seen_uuids = self.read_finished.setdefault(req_id, set())
            rx_infos = self.req_rx_info.setdefault(req_id, {})
            for edge in edges:
                if not edge.tensor_info:
                    final_ready.setdefault(req_id, []).append(edge)
                    continue
                # Already-emitted edge re-surfacing (re-delivery / retry): skip
                # the buffering path and pass through unchanged.
                if any(info.uuid in seen_uuids for info in edge.tensor_info):
                    final_ready.setdefault(req_id, []).append(edge)
                    continue

                if self.enable_prof:
                    # Locally-produced (same-entity) tensors aren't received over
                    # the wire — skip them. ``rx_time`` is the whole edge's read
                    # batch, so attribute it once per source (not per tensor).
                    per_source: dict[str, list[int]] = {}  # source -> [bytes, count]
                    edge_rx_time = 0.0
                    for info in edge.tensor_info:
                        if info.source_entity == self.my_entity_id:
                            continue
                        agg = per_source.setdefault(info.source_entity, [0, 0])
                        agg[0] += info.nbytes
                        agg[1] += 1
                        edge_rx_time = uuid_to_time.get(info.uuid) or edge_rx_time
                    for source_entity, (nbytes, cnt) in per_source.items():
                        rx_key = (source_entity, edge.name)
                        if rx_key not in rx_infos:
                            rx_infos[rx_key] = RxInfo(
                                edge_name=edge.name,
                                source_entity=source_entity,
                                dest_entity=self.my_entity_id,
                            )
                        rx_infos[rx_key].update(
                            num_bytes=nbytes, time=edge_rx_time, count_increment=cnt,
                        )

                shard_dim = edge._shard_dim
                if edge.name not in self.buffered_shards.get(req_id, {}):
                    total_fanin = edge._total_fanin
                    if total_fanin > 1:
                        self.buffered_shards[req_id][edge.name] = BufferedShards(
                            total_fanin=total_fanin, shard_dim=shard_dim,
                            shards={},
                        )

                tensors: list[torch.Tensor] = []
                for info in edge.tensor_info:
                    self.tensor_store.dereference(info.uuid, 1)
                    seen_uuids.add(info.uuid)
                    self.uuid_to_shard_dim[info.uuid] = shard_dim
                    tensors.append(self.get_tensor(info.uuid))

                if edge.name in self.buffered_shards.get(req_id, {}):
                    buf = self.buffered_shards[req_id][edge.name]
                    buf.shards.setdefault(
                        edge.tensor_info[0].source_tp_rank, []
                    ).extend(tensors)
                    # Release the "+1 for graph-node usage" ref now that the
                    # tensor data has been copied into the buffer; the standard
                    # GC path drops the registered memory when refcount hits 0.
                    self.dereference_batch_uniform(
                        [info.uuid for info in edge.tensor_info]
                    )

                    if buf.is_done():
                        consolidated = buf.consolidate()
                        new_infos: list[TensorPointerInfo] = []
                        for uuid_, tensor in (
                            (self._uuid_minter.mint(), t) for t in consolidated
                        ):
                            info = TensorPointerInfo(
                                dims=tensor.shape, dtype=tensor.dtype,
                                nbytes=tensor.nbytes, address=tensor.data_ptr(),
                                stride=tensor.stride(), uuid=uuid_,
                                source_session_id=self.my_session_id,
                                source_entity=self.my_entity_id,
                            )
                            self.tensor_store.put_tensor(
                                req_id, uuid_, tensor, info,
                            )
                            # +1 for graph-node usage (released by the
                            # downstream consumer via _cleanup_consumed_inputs)
                            self.tensor_store.increment_ref(uuid_, 1)
                            # consolidated tensor is in its original layout
                            # (cat happened along the real shard_dim); record
                            # None so get_tensor doesn't try to un-rearrange.
                            self.uuid_to_shard_dim[uuid_] = None
                            new_infos.append(info)
                        edge.tensor_info = new_infos
                        del self.buffered_shards[req_id][edge.name]
                        final_ready.setdefault(req_id, []).append(edge)
                    # else: still waiting on more shards; don't emit yet
                else:
                    final_ready.setdefault(req_id, []).append(edge)
        return final_ready

    def register_request(
        self, request_id: Rid, sharding_config: ShardingConfig
    ):
        self.sharding_configs[request_id] = sharding_config
        # once per partition on a worker serving several: keep shards that
        # already arrived for the first
        self.buffered_shards.setdefault(request_id, {})

    # ---- shared: profiling (tx/rx) ----

    def _record_tx(
        self, request_id: Rid, uuid: int, num_bytes: int, elapsed: float
    ):
        """Record bytes/time spent making ``uuid`` available to remote consumers.

        Keyed by edge name (looked up from ``uuid_to_edge_name``); the sender
        doesn't know the recipient at register time — and may even register data
        that is never read — so TX carries no dest and is not per-connection.
        """
        edge_name = self.uuid_to_edge_name.get(uuid, "?")
        tx = self.req_tx_info.setdefault(request_id, {})
        if edge_name not in tx:
            tx[edge_name] = TxInfo(
                edge_name=edge_name, source_entity=self.my_entity_id
            )
        tx[edge_name].update(num_bytes=num_bytes, time=elapsed)

    def get_tx_info(self, request_id: Rid) -> list[TxInfo]:
        return list(self.req_tx_info.get(request_id, {}).values())

    def get_rx_info(self, request_id: Rid) -> list[RxInfo]:
        return list(self.req_rx_info.get(request_id, {}).values())

    # ---- shared: TensorStore delegation ----

    def get_tensor(self, uuid: int) -> torch.Tensor:
        tensor = self.tensor_store.get_tensor(uuid)
        shard_dim = self.uuid_to_shard_dim.get(uuid)
        if shard_dim is None:
            return tensor
        return self._undo_leading_shard_dim(shard_dim, tensor)

    def drop_host_copies(self) -> None:
        """Forget the host copies stored with a step's outputs once that step's
        sends are done. They are views of buffers the producer reuses on its
        next step, and a tensor can be sent in a later step than the one that
        stored it (a loop's accumulated output is sent when the loop ends)."""
        self.tensor_store.drop_cpu_tensors()

    def set_persist(self, uuid: int, persist: bool):
        self.tensor_store.set_metadata(uuid, persist=persist)
        if self.tensor_store.can_gc(uuid):
            self._cleanup_by_uuid(uuid)

    def dereference(self, uuid: int, n: int = 1):
        self.tensor_store.dereference(uuid, n=n)
        if self.tensor_store.can_gc(uuid):
            self._cleanup_by_uuid(uuid)

    def dereference_batch_uniform(self, uuids: list[int], n: int = 1):
        """``dereference`` for a batch, in one crossing into the bookkeeper.

        Per uuid the loop above pays one crossing for the decrement and one
        for ``can_gc``, plus ``is_registered`` and ``forget_tensor`` on
        anything it frees. Here the bookkeeper does all four and hands back
        only what became collectable, leaving this side the part it cannot
        do: the shm files, the arena slots and the memory unregistration.
        """
        self.cleanup_collectable(
            *self.tensor_store.dereference_batch_uniform(uuids, n=n, cleanup=True)
        )

    def dereference_batch(self, uuids: list[int], counts: list[int]):
        """``dereference_batch_uniform`` where the count differs per uuid: a
        TENSOR_RECEIVED ack covers as many reads as the edge fanned out to."""
        self.cleanup_collectable(
            *self.tensor_store.dereference_batch(uuids, counts, cleanup=True)
        )

    def cleanup_collectable(
        self, collectable: list[int], registered: list[bool],
    ):
        """Transport-side teardown for tensors the bookkeeper has already
        dereferenced to zero and forgotten -- what a batched dereference, or
        the graph runtime releasing a node's consumed inputs, hands back."""
        if not collectable:
            return
        self.tensor_store.mark_forgotten(collectable)
        for uuid, was_registered in zip(collectable, registered, strict=True):
            try:
                self._cleanup_by_uuid(uuid, registered=was_registered)
            except Exception as e:
                # The store already popped the record, so this uuid cannot be
                # retried -- log and keep going so one bad tensor does not
                # strand the rest of the batch the same way.
                logger.error(
                    "Error cleaning up tensor uuid %d: %s, skipping.", uuid, e,
                )

    def increment_ref(self, uuid: int, n: int = 1):
        self.tensor_store.increment_ref(uuid, n=n)

    def increment_ref_batch_uniform(self, uuids: list[int], n: int = 1):
        """``increment_ref_batch`` for a uniform count, which is what the
        safety hold on a stored output batch always is."""
        self.tensor_store.increment_ref_batch_uniform(uuids, n)

    def has_inflight_reads(self, request_id: Rid) -> bool:
        """Whether any async read for this request is still touching a remote
        segment. Synchronous (SHM) reads complete inside ``start_read_tensors``
        with ``future=None``, so they never count here; only outstanding
        async (RDMA) futures do. Gates the READS_DONE ACK during a drain.
        """
        return any(
            ep.request_id == request_id
            and ep.future is not None and not ep.future.done()
            for ep in self.pending
        )

    def cleanup_request(self, request_id: Rid, force: bool = False):
        """Teardown for a request's tensor state.

        Default (soft): drop tensors that are safe to GC, defer any still
        referenced or persisted. Does NOT force-drop persisted signals — that
        is the conductor-coordinated hard cleanup, see
        :meth:`force_cleanup_request`.
        """
        self.read_finished.pop(request_id, None)
        self.buffered_shards.pop(request_id, None)
        self.sharding_configs.pop(request_id, None)
        self.req_rx_info.pop(request_id, None)
        self.req_tx_info.pop(request_id, None)
        for uuid in self.tensor_store.get_all_uuids(request_id):
            self.uuid_to_shard_dim.pop(uuid, None)
            if not self.tensor_store.can_gc(uuid) and not force:
                logger.warning(
                    "Deferring cleanup of tensor uuid %s "
                    "(awaiting TENSOR_RECEIVED ACK or unpersist)", uuid
                )
                continue
            self._cleanup_by_uuid(uuid)

        self._collect_and_send_acks(
            request_id,
            sum([ep.graph_edges for ep in self.pending if ep.request_id == request_id], start=[]),
        )
        self.pending = [ep for ep in self.pending if ep.request_id != request_id]

    def force_cleanup_request(self, request_id: Rid):
        """Unconditional teardown: drop every tensor for the request and unlink
        its SHM, ignoring ref counts and persist markers. Safe only after every
        reader has confirmed (READS_DONE) it has no in-flight reads for the
        request. Also reclaims non-persisted buffers whose readers drained
        before ACKing (which the soft path would otherwise defer forever).
        """
        self.cleanup_request(request_id, force=True)

    #: True when no other launch can write a file under this manager's names
    #: (see ``SharedMemoryCommunicationManager``); irrelevant to transports
    #: that write no files.
    names_unique_per_launch = False

    def unlink_owned_files(self) -> None:  # noqa: B027
        """Unlink the shared-memory files this process wrote for its peers and
        still owns. A no-op here: only the SHM file transport (and the arena's
        spills) writes any."""


# ---------------------------------------------------------------------------
# MooncakeCommunicationManager
# ---------------------------------------------------------------------------

class MooncakeCommunicationManager(TensorCommunicationManager):
    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        device: str,
        communicator: BaseCommunicator,
        protocol: CommProtocol = CommProtocol.RDMA,
        metadata_server: str = "P2PHANDSHAKE",
        tcp_transfer_device: str = "",
        enable_prof: bool=False
    ):
        engine = MooncakeTransferEngine(
            hostname=hostname,
            protocol=protocol,
            metadata_server=metadata_server,
            tcp_transfer_device=tcp_transfer_device,
        )
        super().__init__(
            my_entity_id=my_entity_id,
            my_session_id=engine.get_session_id(),
            device=device,
            communicator=communicator,
            transfer_engine=engine,
            enable_prof=enable_prof
        )
        self.protocol = protocol
        # the consumer has the engine copy ``nbytes`` from ``address + offset``
        # into a buffer it views with the descriptor's strides
        self.needs_flat_layout = protocol in (CommProtocol.RDMA, CommProtocol.TCP)
        self._async_reader = AsyncMooncakeReader(engine._engine, device=device)

    @property
    def transport_label(self) -> str:
        return self.protocol.value

    def register_for_send(self, request_id, tensor_infos, skip_cuda_sync=False):
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()
        for info in tensor_infos:
            self._register_one(request_id, info.uuid)

    def register_for_send_uuids(
        self,
        per_request: ParallelList[Rid, list[int]],
        skip_cuda_sync: bool = False,
    ):
        """Uuid-driven form. Registration here reads nothing off a descriptor
        but the uuid -- the tensor itself comes from the store -- so taking
        uuids avoids rebuilding one per tensor."""
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()
        for request_id, uuids in per_request:
            for uuid in uuids:
                self._register_one(request_id, uuid)

    def _register_one(self, request_id: Rid, uuid: int) -> None:
        """Pin one tensor for remote reads. The caller owns the CUDA sync."""
        already_registered = self.tensor_store.is_registered(uuid)
        if self.protocol in (CommProtocol.RDMA, CommProtocol.TCP):
            if already_registered:
                return
            logger.debug("Registering %s for send", uuid)
            tensor = self.tensor_store.get_tensor(uuid)
            t0 = time.perf_counter()
            # Mooncake rejects a 0-byte region, and a reader of an empty
            # tensor copies nothing (see start_read_tensors), so there is
            # nothing to pin.
            ret_value = self.transfer_engine.register_memory(
                tensor.data_ptr(), tensor.nbytes
            ) if tensor.nbytes else 0
            if PHASE_PERIOD:
                phase_record(
                    "tensors.register_memory", time.perf_counter() - t0,
                )
            if ret_value != 0:
                raise RuntimeError(
                    f"Mooncake memory registration failed for request id "
                    f"{request_id}, uuid {uuid}."
                )
            if self.enable_prof:
                # RDMA: the receiver does the actual read, so the sender-side
                # cost is the memory-pinning (register) time.
                self._record_tx(
                    request_id, uuid, tensor.nbytes, time.perf_counter() - t0
                )
        elif self.enable_prof and not already_registered:
            tensor = self.tensor_store.get_tensor(uuid)
            self._record_tx(request_id, uuid, tensor.nbytes, 0.0)
        self.tensor_store.set_metadata(uuid, mem_registered=True)

    def _cleanup_by_uuid(self, uuid: int, registered: bool | None = None):
        super()._cleanup_by_uuid(uuid, registered)
        logger.debug("Deleting tensor uuid %s", uuid)
        if not self.tensor_store.check_uuid_presence(uuid):
            logger.warning("Trying to cleanup tensor %s, but uuid not found", uuid)
            return
        if registered is None:
            registered = self.tensor_store.is_registered(uuid)
        if self.protocol in (CommProtocol.RDMA, CommProtocol.TCP) and registered:
            tensor = self.tensor_store.get_tensor(uuid)
            # an empty tensor is marked ready without being registered
            if tensor.nbytes:
                ret_value = self.transfer_engine.unregister_memory(tensor.data_ptr())
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory unregistration failed.")
        self.tensor_store.remove_tensor(uuid)

    def start_read_tensors(
        self, request_id: Rid, graph_edges: list[GraphEdge],
        graph_walk: str | None=None
    ) -> list[Future]:
        futures = []
        stats = self.transfer_stats
        for graph_edge in graph_edges:
            if len(graph_edge.tensor_info) == 0:
                continue

            logger.debug(
                "Starting to read in %d tensors %s for graph node %s",
                len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node
            )

            read_info = []
            # Created before the reads and pointed at by each of them, so the
            # async reader can stamp ``rx_time`` on it when the read completes
            # (the reader runs on another thread; the object identity is
            # stable, only ``.future`` is filled in after submit).
            fp = FutureAndPointers(
                future=None, graph_edges=[graph_edge], request_id=request_id
            )
            for info in graph_edge.tensor_info:
                if info.source_entity == self.my_entity_id:
                    self._slice_existing_tensor(
                        request_id=request_id, name=graph_edge.name,
                        next_node=graph_edge.next_node,
                        graph_walk=graph_walk, info=info
                    )
                    self.tensor_store.increment_ref(info.uuid, 1)
                    continue
                if self.tensor_store.check_uuid_presence(info.uuid):
                    self.tensor_store.increment_ref(info.uuid, 1)
                    continue
                buffer = torch.empty(
                    info.dims, dtype=info.dtype, device=self.device
                ).as_strided(info.dims, stride=info.stride)
                self.tensor_store.put_tensor(
                    request_id=request_id, uuid=info.uuid, tensor=buffer, info=info,
                )
                self.tensor_store.set_metadata(info.uuid, mem_registered=True
                )
                # +1 for transit (released by get_ready_tensors)
                # +1 for graph-node usage (released by _cleanup_consumed_inputs)
                self.tensor_store.increment_ref(info.uuid, 2
                )
                if stats is not None:
                    fp.count_read(info)
                # Nothing to read, and Mooncake rejects a 0-byte region: the
                # empty buffer is the whole tensor. With no reads, the edge
                # gets no future and is ready once earlier reads are.
                if info.nbytes == 0:
                    continue

                if self.protocol in (CommProtocol.RDMA, CommProtocol.TCP):
                    self.transfer_engine.register_memory(buffer.data_ptr(), info.nbytes)

                read_info.append(TransferReadInfo(
                    source_session_id=info.source_session_id,
                    local_ptr=buffer.data_ptr(),
                    remote_ptr=info.address + info.offset,
                    nbytes=info.nbytes,
                ))
                logger.debug("Started transfer read for uuid %s", info.uuid)
            if self.enable_prof or fp.read_src is not None:
                for ri in read_info:
                    ri.fp = fp
            fut = self._async_reader.submit(read_info)
            fp.future = fut
            if fut is not None:
                futures.append(fut)
            self.pending.append(fp)
        return futures


# ---------------------------------------------------------------------------
# Shared-memory tensor serialization helpers
# ---------------------------------------------------------------------------

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


DEPLOYMENT_TOKEN_ENV = "MSTAR_DEPLOYMENT_TOKEN"
#: The token the last API server in this environment minted.
_MINTED_TOKEN_ENV = "MSTAR_DEPLOYMENT_TOKEN_MINTED"


def ensure_deployment_token() -> str:
    """Mint this launch's token unless a launcher set one. Called by the API
    server, which every other process of the deployment descends from
    (``spawn`` children inherit the environment).

    A token an API server minted is never kept by the next launch that finds
    it, a server built again in this process or started by one that holds
    it: on the same endpoint that launch would write the names a stopping
    worker of the old one may still unlink."""
    token = os.environ.get(DEPLOYMENT_TOKEN_ENV)
    if not token or token == os.environ.get(_MINTED_TOKEN_ENV):
        token = secrets.token_hex(8)
        os.environ[DEPLOYMENT_TOKEN_ENV] = os.environ[_MINTED_TOKEN_ENV] = token
    return token


def _deployment_namespace(communicator) -> str:
    """A token every process of ONE deployment derives identically, and two
    deployments that share a tmpfs never share.

    Tensor uuids are per-entity counters, so ``worker_0``'s uuid 5 exists in
    every deployment at once; a file named by entity and uuid alone is written
    and unlinked by every server on the host. The conductor's endpoint
    separates deployments on one host: over TCP a second deployment on the
    same port fails to bind, and over IPC -- where libzmq would happily
    unlink and replace the socket file -- the conductor's ``_lock_deployment``
    refuses to start. Containers that share ``/dev/shm`` but not the network
    namespace or ``/tmp`` can both run the same endpoint, so the launch token
    (``ensure_deployment_token``) is mixed in when set.
    """
    if getattr(communicator, "protocol", None) not in (
        CommProtocol.IPC, CommProtocol.TCP
    ):
        return ""
    endpoint = communicator._endpoint(DEPLOYMENT_ANCHOR_ENTITY)
    if endpoint.startswith("ipc://"):
        # realpath: "/tmp/mstar_x/" and "/tmp/mstar_x" are one deployment.
        endpoint = "ipc://" + os.path.realpath(endpoint.removeprefix("ipc://"))
    token = os.environ.get(DEPLOYMENT_TOKEN_ENV)
    key = f"{token}\0{endpoint}" if token else endpoint
    return hashlib.sha1(key.encode()).hexdigest()[:12]


_NAMESPACE_LOCK_RE = re.compile(r"mstar_([0-9a-f]{12})\.lock")

#: Lock path -> fd, held (never closed) for the life of the process.
_held_namespace_locks: dict[str, int] = {}


def _hold_namespace_lock(shm_dir: str, namespace: str) -> None:
    """Hold a shared ``flock`` on ``mstar_<namespace>.lock`` before this
    process writes any file under the namespace, so a namespace whose lock
    nobody holds belongs to a deployment that is gone."""
    path = os.path.join(shm_dir, f"mstar_{namespace}.lock")
    if path in _held_namespace_locks:
        return
    while True:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        # Waits only while a sweeper deletes the namespace; it unlinks the
        # lock file before letting go, so recheck that we hold the named one.
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            named = os.lstat(path)
        except FileNotFoundError:
            named = None
        held = os.fstat(fd)
        if named is not None and (held.st_dev, held.st_ino) == (
            named.st_dev, named.st_ino
        ):
            _held_namespace_locks[path] = fd
            return
        os.close(fd)


def _sweep_dead_namespaces(shm_dir: str, own_namespace: str) -> None:
    """Delete every ``mstar_<ns>_*`` file in ``shm_dir`` of each of this
    uid's namespaces whose lock there this process can take exclusively,
    then the lock file. A per-launch namespace never collides with a dead
    run's files, so nothing else reclaims them."""
    try:
        names = os.listdir(shm_dir)
    except OSError:
        return
    uid = os.getuid()
    for name in names:
        m = _NAMESPACE_LOCK_RE.fullmatch(name)
        if m is None or m[1] == own_namespace:
            continue
        path = os.path.join(shm_dir, name)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            held = os.fstat(fd)
            if held.st_uid != uid:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue   # a process of that deployment is alive
            named = os.lstat(path)
            if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                continue
            prefix = f"mstar_{m[1]}_"
            swept = 0
            for stale in os.listdir(shm_dir):
                if stale.startswith(prefix):
                    try:
                        os.unlink(os.path.join(shm_dir, stale))
                        swept += 1
                    except OSError:
                        pass
            os.unlink(path)
            logger.info("SHM: swept %d file(s) of dead deployment namespace "
                        "%s", swept, m[1])
        except OSError as e:
            logger.debug("SHM: cannot sweep %s: %s", name, e)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# SharedMemoryCommunicationManager
# ---------------------------------------------------------------------------

class SharedMemoryCommunicationManager(TensorCommunicationManager):
    """Tensor transport via file I/O to a tmpfs directory (``/dev/shm``)."""

    #: Directories besides ``shm_dir`` that hold this transport's
    #: ``mstar_<namespace>_*`` files; each gets its own namespace lock.
    _namespace_file_dirs: tuple[str, ...] = ()

    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        device: str,
        communicator: BaseCommunicator,
        shm_dir: str | None = None,
        enable_prof: bool=False
    ):
        engine = LocalTransferEngine(hostname=hostname)
        super().__init__(
            my_entity_id=my_entity_id,
            my_session_id=engine.get_session_id(),
            device=device,
            communicator=communicator,
            transfer_engine=engine,
            enable_prof=enable_prof
        )

        self.shm_dir = shm_dir or _default_shm_dir()
        os.makedirs(self.shm_dir, exist_ok=True)
        # at boot, not at the first large send: that failure would be the only other sign of a small dir
        try:
            st = os.statvfs(self.shm_dir)
            shm_total = st.f_frsize * st.f_blocks
        except OSError:
            shm_total = None
        if shm_total is not None and shm_total < _SHM_DIR_WARN_BYTES:
            logger.warning(
                "SHM: %s holds only %d MiB and every tensor in flight is written there, so a large one will "
                "fail its send with ENOSPC. Give it more room (docker --shm-size), or pass "
                "--tensor-comm-protocol TCP, which is slower", self.shm_dir, shm_total >> 20)

        # uuid → file path for sender-side cleanup
        self._shm_files: dict[int, str] = {}
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
        ns = f"{self._shm_namespace}_" if self._shm_namespace else ""
        return os.path.join(self.shm_dir, f"mstar_{ns}{entity_id}_{uuid}")

    def _create_shm_file(self, path: str):
        """Open ``path`` for writing, having created it ourselves.

        O_EXCL because the name is fully predictable -- namespace, entity and
        a per-entity counter -- in a world-writable directory. A plain
        ``open(path, "wb")`` onto a file some other uid got there first either
        writes through into a file we do not own or fails the batch late on
        PermissionError; creating it is what makes the uuid a claim.

        Residue we DO own is a different case and is reclaimed rather than
        raised on: without a launch token the namespace is derived from the
        conductor endpoint alone, so a restart on the same ports collides with
        every file the dead run left (uuid counters restart at 1 too).
        Concurrent same-namespace deployments cannot reach here -- the launch
        token separates deployments that share a tmpfs across containers, and
        on one host the ports differ over TCP and the conductor's deployment
        lock refuses a second one over IPC.
        """
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            return os.fdopen(os.open(path, flags, 0o600), "wb")
        except FileExistsError:
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode) or st.st_uid != os.getuid():
                raise RuntimeError(
                    f"refusing to write tensor file {path}: it exists and is "
                    f"not ours (owner uid {st.st_uid}, ours {os.getuid()}, "
                    f"symlink {stat.S_ISLNK(st.st_mode)}). Another user or an "
                    "unrelated deployment holds this path; give this "
                    "deployment its own ZMQ endpoint so it derives a "
                    "different namespace."
                ) from None
            os.unlink(path)
            try:
                return os.fdopen(os.open(path, flags, 0o600), "wb")
            except FileExistsError:
                raise RuntimeError(
                    f"tensor file {path} was recreated while reclaiming our "
                    "own stale copy of it -- another process is writing this "
                    "deployment's namespace"
                ) from None

    def _write_shm_file(self, path: str, data: bytes) -> None:
        """Write one tensor's bytes to ``path``.

        A full shm dir fails the write part way, and no cleanup knows the partial file: left behind, it
        would keep the space it took, and the next request would find the dir fuller still.
        """
        try:
            with self._create_shm_file(path) as f:
                f.write(data)
        except OSError as exc:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            if exc.errno != errno.ENOSPC:
                raise
            free = shutil.disk_usage(self.shm_dir).free
            if free < len(data):
                reason = f"{self.shm_dir} has {free} bytes free, too few for a {len(data)}-byte tensor"
            else:
                reason = (
                    f"{self.shm_dir} reported no space for a {len(data)}-byte tensor, with {free} bytes free after "
                    "the failed write: it may be out of inodes, or other writers held the space meanwhile"
                )
            raise OSError(
                errno.ENOSPC,
                f"{reason}. Give it more room (docker --shm-size), or pass --tensor-comm-protocol TCP, which is slower",
            ) from exc

    def register_for_send(
        self, request_id: Rid, tensor_infos: list[TensorPointerInfo],
        skip_cuda_sync: bool = False,
    ):
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()
        # Producer's completion event is already waited on upstream, so the
        # source tensors are device-visible on entry. Running the D2H on a
        # dedicated stream keeps .cpu()'s host-wait bounded by the copy
        # itself instead of stalling behind GPU(N+1) on the default stream.
        ctx = (
            torch.cuda.stream(self._d2h_stream)
            if self._d2h_stream is not None
            else _nullcontext()
        )
        with ctx:
            for info in tensor_infos:
                self._write_one(request_id, info.uuid)

    def register_for_send_uuids(
        self,
        per_request: ParallelList[Rid, list[int]],
        skip_cuda_sync: bool = False,
    ):
        """Uuid-driven form. This path reads nothing off a descriptor but the
        uuid, so taking uuids avoids rebuilding one per tensor."""
        if not skip_cuda_sync and self._on_cuda:
            torch.cuda.default_stream().synchronize()
        ctx = (
            torch.cuda.stream(self._d2h_stream)
            if self._d2h_stream is not None
            else _nullcontext()
        )
        with ctx:
            for request_id, uuids in per_request:
                for uuid in uuids:
                    self._write_one(request_id, uuid)

    def _write_one(self, request_id: Rid, uuid: int) -> None:
        """Serialize one tensor to its per-uuid file. The caller owns the
        CUDA sync and the copy-stream context."""
        if self.tensor_store.is_registered(uuid):
            return
        tensor = self._send_source(uuid)
        t0 = time.perf_counter()
        data = _serialize_tensor(tensor)
        path = self._shm_path(self.my_entity_id, uuid)
        self._write_shm_file(path, data)
        self._shm_files[uuid] = path
        self.tensor_store.set_metadata(uuid, mem_registered=True)
        if self.enable_prof:
            # SHM: the serialize + file write IS the send work.
            self._record_tx(request_id, uuid, len(data), time.perf_counter() - t0)
        logger.debug("SHM: wrote tensor %s to %s (%d bytes)", uuid, path, len(data))

    def start_read_tensors(
        self, request_id: Rid, graph_edges: list[GraphEdge],
        graph_walk: str | None = None
    ):
        # Run H2D copies on a dedicated stream so they overlap with the
        # consumer's in-flight default-stream work. We make default wait
        # for the H2D stream at the end so downstream kernels see the data.
        h2d_did_work = False
        stats = self.transfer_stats
        ctx = (
            torch.cuda.stream(self._h2d_stream)
            if self._h2d_stream is not None
            else _nullcontext()
        )
        with ctx:
            for graph_edge in graph_edges:
                if len(graph_edge.tensor_info) == 0:
                    continue
                logger.debug(
                    "SHM: starting read of %d tensors %s for graph node %s",
                    len(graph_edge.tensor_info), graph_edge.name, graph_edge.next_node,
                )
                rx_t0 = time.perf_counter()
                fp = FutureAndPointers(
                    future=None, graph_edges=[graph_edge], request_id=request_id,
                )
                for info in graph_edge.tensor_info:
                    if info.source_entity == self.my_entity_id:
                        self._slice_existing_tensor(
                            request_id=request_id, name=graph_edge.name,
                            next_node=graph_edge.next_node,
                            graph_walk=graph_walk, info=info
                        )
                        self.tensor_store.increment_ref(info.uuid, 1)
                        continue
                    if self.tensor_store.check_uuid_presence(info.uuid):
                        self.tensor_store.increment_ref(info.uuid, 1)
                        continue
                    if info.shm_segment is not None:
                        raise RuntimeError(
                            f"tensor {info.uuid} from {info.source_entity} "
                            "was staged in an SHM arena but this consumer "
                            "runs the file transport — MSTAR_SHM_ARENA must "
                            "match across the deployment")
                    path = self._shm_path(info.source_entity, info.uuid)
                    with open(path, "rb") as f:
                        f.seek(info.offset)
                        data = f.read(info.nbytes)
                    tensor = _deserialize_tensor(data, self.device, tensor_info=info)
                    h2d_did_work = True
                    self.tensor_store.put_tensor(request_id, info.uuid, tensor, info)
                    self.tensor_store.set_metadata(info.uuid, mem_registered=False)
                    # +1 for transit (released by get_ready_tensors)
                    # +1 for graph-node usage (released by _cleanup_consumed_inputs)
                    self.tensor_store.increment_ref(info.uuid, 2)
                    if stats is not None:
                        fp.count_read(info)
                    logger.debug("SHM: read tensor %s from %s", info.uuid, path)
                fp.rx_time = time.perf_counter() - rx_t0
                self.pending.append(fp)
        if h2d_did_work and self._h2d_stream is not None:
            torch.cuda.default_stream(self.device).wait_stream(self._h2d_stream)
        return []

    def _cleanup_by_uuid(self, uuid: int, registered: bool | None = None):
        super()._cleanup_by_uuid(uuid, registered)
        logger.debug("SHM: cleaning up tensor uuid %s", uuid)
        if not self.tensor_store.check_uuid_presence(uuid):
            logger.warning("SHM: cleanup tensor %s, uuid not found", uuid)
            return
        if uuid in self._shm_files:
            path = self._shm_files.pop(uuid)
            try:
                os.unlink(path)
                logger.debug("SHM: unlinked %s", path)
            except FileNotFoundError:
                pass
        self.tensor_store.remove_tensor(uuid)

    def unlink_owned_files(self) -> None:
        """Unlink every tensor file this manager wrote that is still waiting
        for its TENSOR_RECEIVED or REMOVE_REQUEST, for a process that is
        stopping. A second call finds nothing left."""
        files = self._shm_files
        # The orphan watchdog's thread can run this at the same time, so each
        # path is taken by one caller, and put back for the retry if a stop
        # cuts its unlink.
        while True:
            try:
                uuid, path = files.popitem()
            except KeyError:
                break
            try:
                os.unlink(path)
            except OSError:
                # Already gone (FileNotFoundError) or not removable: either
                # way, keep going so the rest are still reclaimed.
                pass
            except BaseException:
                files[uuid] = path
                raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tensor_communication_manager(
    protocol: CommProtocol,
    my_entity_id: str,
    hostname: str,
    device: str,
    communicator: BaseCommunicator,
    metadata_server: str = "P2PHANDSHAKE",
    tcp_transfer_device: str = "",
    shm_dir: str | None = None,
    enable_prof: bool=False
) -> TensorCommunicationManager:
    """Select tensor transport backend based on protocol.

    For the SHM protocol, ``MSTAR_SHM_ARENA`` selects the implementation
    (see ``docs/environment_variables.rst``): ``0`` (default) — per-uuid
    files; ``1`` — the Rust shared-memory arena (raises if the
    ``mstar_rust`` extension is missing); ``AUTO`` — the arena when the
    extension imports, files otherwise. The flag must match across the
    deployment: the arena location rides in the tensor descriptors, so a
    file-transport consumer cannot read an arena producer.
    """
    if protocol == CommProtocol.SHM:
        choice = os.getenv("MSTAR_SHM_ARENA", "0").upper()
        if choice not in ("0", "1", "AUTO"):
            raise ValueError(
                f"MSTAR_SHM_ARENA must be 0, 1, or AUTO; got {choice!r}")
        if choice != "0":
            try:
                from mstar.communication.arena import (
                    ArenaShmCommunicationManager,
                )
            except ImportError:
                if choice == "1":
                    raise
                logger.debug("MSTAR_SHM_ARENA=AUTO: mstar_rust not "
                             "installed, using the file transport")
            else:
                return ArenaShmCommunicationManager(
                    my_entity_id=my_entity_id,
                    hostname=hostname,
                    device=device,
                    communicator=communicator,
                    shm_dir=shm_dir,
                    enable_prof=enable_prof
                )
        return SharedMemoryCommunicationManager(
            my_entity_id=my_entity_id,
            hostname=hostname,
            device=device,
            communicator=communicator,
            shm_dir=shm_dir,
            enable_prof=enable_prof
        )
    return MooncakeCommunicationManager(
        my_entity_id=my_entity_id,
        hostname=hostname,
        device=device,
        communicator=communicator,
        protocol=protocol,
        metadata_server=metadata_server,
        tcp_transfer_device=tcp_transfer_device,
        enable_prof=enable_prof
    )
