import gc
import logging
import math
import os
import sys
import threading
import time
import time as _time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import cached_property, partial
from time import sleep
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from mstar.communication.codec import WireCodec
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import (
    LocalTransferEngine,
    NameToTensorList,
    StoredOutputs,
    create_tensor_communication_manager,
)
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine import apply_dynamo_config
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AllocationFailed, ResourceReqConfig
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.transfer import (
    TransferEngineInfo,
    make_deployment_kv_shm_dir,
)
from mstar.engine.resources.sampler.config import SAMPLED_LOGPROB
from mstar.graph.runtime.base import (
    GraphRuntime,
    InlineColumn,
    PostprocessNeed,
    PostprocessRequest,
    PostprocessStep,
    RouteOutput,
    RouteTensors,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
)
from mstar.graph.runtime.utils import require_rust_extension
from mstar.model.base import Model, WorkerGraph
from mstar.model.submodule_base import (
    END_PARTITION,
    BatchedModelOutput,
    HostRows,
    chunk_loop_name,
)
from mstar.profile.worker import WorkerProfileInfo
from mstar.streaming.stream_buffer import StreamingEdge
from mstar.utils import cpus, knobs
from mstar.utils.client_errors import client_error_text
from mstar.utils.containers import ParallelList
from mstar.utils.device import DeviceCapability
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    FailRequests,
    MessageSource,
    ReadsDone,
    RemoveRequest,
    SetupDone,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.profiler import (
    PhaseLaps,
    nvtx_enabled,
    range_pop,
    range_push,
    set_step_key,
)
from mstar.utils.threads import ThreadPool, pin_thread
from mstar.worker.admission import Admission
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.engine_manager import EngineManager
from mstar.worker.executing_batch import input_metadata_and_fwd_info, make_executing_batch
from mstar.worker.host_pool import HostPool
from mstar.worker.input_ingest import InputIngest
from mstar.worker.lockstep import Lockstep, tp_async_sched_setting
from mstar.worker.micro_scheduler import (
    CHAIN_LANE,
    MicroScheduler,
    ScheduledBatch,
    tp_chunk_ranges,
    unfinished_chunk_rids,
)
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.phase_timer import PhaseTimer, step_key
from mstar.worker.preemption import Preemption
from mstar.worker.rebuild import DEFAULT_REBUILD_HOST_MB, Rebuilds, node_writes, policies_run_here
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.side_steps import SIDE_LANE, SideStepLane
from mstar.worker.spec_rows import DeferredRowInputs
from mstar.worker.speculation import (
    Speculation,
    SpeculationBuilder,
    SpeculationContext,
    SpeculationPreparer,
)
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker_metrics import WorkerMetrics

logger = logging.getLogger(__name__)

# seconds between "no offload possible" lines for one node and walk: a hold is
# retried every backoff, and a line per retry buries the rest of the log

# How often ``_await_admit_settled`` warns while a step has yet to admit. It
# never gives up; see there.
_ADMIT_FENCE_WARN_S = 10.0

# a ScheduleTPNode row the leader did not chunk
_UNCHUNKED = ChunkRange(-1, -1, -1, -1)




def _make_graph_runtime(
    communicator,
    tensor_manager,
    **kwargs,
) -> GraphRuntime:
    """The Rust graph runtime.

    It requires the Rust communicator and the Rust bookkeeper: it sends
    WORKER_GRAPHS_DONE and INPUT_SIGNALS itself on an Arc to the same
    transport, and settles refcounts on a share of the same bookkeeper, rather
    than hopping back into Python for either. Anything else has no such object
    to share, so this raises on it.
    """
    require_rust_extension()

    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.graph.runtime.rust import RustGraphRuntime

    if not isinstance(communicator, RustZMQCommunicator):
        raise ValueError(
            "The graph runtime needs the Rust communicator, but this worker "
            f"built a {type(communicator).__name__}."
        )
    if not issubclass(communicator.codec, WireCodec):
        raise ValueError(
            "The graph runtime needs the msgpack codec, but this worker "
            f"built a {communicator.codec.__name__}."
        )
    bookkeeping = tensor_manager.tensor_store.bookkeeping
    if not hasattr(bookkeeping, "_rust"):
        raise ValueError(
            "The graph runtime needs the Rust TensorBookkeeping: it "
            "holds a share of it, so a Python one cannot be handed over. "
            f"Got {type(bookkeeping).__name__}."
        )
    return RustGraphRuntime(
        bookkeeping=bookkeeping,
        communicator=communicator,
        **kwargs,
    )


def _make_rebuilds(
    model: Model | None, model_config: dict | None, my_walks: set[tuple[str, str]], device: torch.device,
    followed: frozenset[str] = frozenset(),
) -> Rebuilds:
    """What a dropped request is rebuilt from, on each node whose model declares how and whose rebuild this worker
    runs: a disaggregated decode node can't run the prefill walk a rebuild needs."""
    graphs = model.get_graph_walk_graphs() if model is not None else {}
    policies = policies_run_here(model.rebuild_policies(), graphs, my_walks) if model is not None else {}
    host_mb = (model_config or {}).get("rebuild_host_mb", DEFAULT_REBUILD_HOST_MB)
    if isinstance(host_mb, bool) or not isinstance(host_mb, int) or host_mb < 0:
        raise ValueError(f"rebuild_host_mb: {host_mb!r} is not a non-negative int")
    return Rebuilds(policies, graphs, HostPool(host_mb << 20, device) if host_mb and policies else None, followed)


def _wait_ms(raw: str) -> float:
    """Milliseconds for ``Event.wait``, which overflows past TIMEOUT_MAX."""
    value = float(raw)
    if not math.isfinite(value) or value / 1000.0 > threading.TIMEOUT_MAX:
        raise ValueError(f"{raw!r} is not a usable wait")
    return value


def _env_number(name: str, default, parse, minimum):
    """A numeric tuning knob from the environment.

    A malformed value falls back to *default* and one below *minimum* is
    raised to it, each with a warning, as ``MSTAR_RECOMPILE_LIMIT`` already
    does. The worker reads these when it is built: raising from ``run()``
    would come after weight load, warmup and capture, and after the conductor
    has counted the worker ready.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %s; using %s", name, raw, minimum, minimum)
        return minimum
    return value


@dataclass
class _SpecFairness:
    """The decode loop's fairness settings for ``_build_speculation``: whether
    to peek for other ready work, and the consecutive-step cap."""
    peek: bool
    max_consecutive: int


@dataclass
class PendingBatch:
    batch: ScheduledBatch
    node_batch: ExecutingBatch
    node_name: str
    partition: str
    graph_walk: str
    future: Future
    speculative_new_iter: bool = False
    loop_name: str = None
    # The leader's broadcast seq for this batch (``ScheduleTPNode.spec_seq``).
    tp_seq: int = -1
    # perf_counter when the main thread handed it to the GPU thread, stamped
    # only while metrics are on and only off the speculative path: a
    # speculated step is handed off before the step it follows finishes, so
    # its step time starts at that finish anyway (see WorkerMetrics.step_done)
    submitted_at: float = 0.0
    # the chain's step, or a side lane's (``mstar.worker.side_steps``)
    lane: int = CHAIN_LANE


class ModelStops(NamedTuple):
    """``Worker._check_stops``: the model's stops by rid, the rows it failed
    (which leave the batch), and the rows whose partition it ended."""
    stops: dict[int, set[str]]
    dropped: list[int]
    ended: list[int]


class SpeculationTarget(NamedTuple):
    info: SpeculationOutput
    walk_rids: list[int]  # the step's rids on ``info.graph_walk``


# pinned host-row buffers in flight at once (see Worker._stage_host_rows)
_HOST_ROWS_RING = 4
_HOST_SPANS_PURPOSE = tuple(f"output_spans{slot}" for slot in range(_HOST_ROWS_RING))


def _host_column(host_rows: HostRows | None, name: str) -> torch.Tensor | None:
    """Host buffer ``name``'s first value per row, a view; None without one."""
    buf = None if host_rows is None else host_rows.buffers.get(name)
    if not torch.is_tensor(buf) or buf.dim() == 0 or buf.numel() == 0:
        return None
    return buf.reshape(buf.shape[0], -1)[:, 0]


class _LazyPerRid(Mapping):
    """``Worker._rows_to_per_rid`` on demand: the per-rid host dicts are built
    the first time a consumer reads them. The batched stop check, the row store
    and the inline emit read the rows directly, so in the common decode step
    nobody does, and the per-request dicts are never built."""

    __slots__ = ("_host", "_request_ids", "_built")

    def __init__(self, host: dict, request_ids: list[int]):
        self._host = host
        self._request_ids = request_ids
        self._built: dict | None = None

    def _dict(self) -> dict:
        if self._built is None:
            self._built = Worker._rows_to_per_rid(self._host, self._request_ids)
        return self._built

    def __getitem__(self, key):
        return self._dict()[key]

    def __iter__(self):
        return iter(self._dict())

    def __len__(self) -> int:
        return len(self._request_ids)

    # the built dict's own views, rather than Mapping's per-key lookups
    def items(self):
        return self._dict().items()

    def keys(self):
        return self._dict().keys()

    def values(self):
        return self._dict().values()


class _LazyHostOutputs(Mapping):
    """Per-rid host copies of a step's outputs, copied by output name on first
    read (``copy_names(names) -> {name: {rid: host value}}``, one sync per call).

    The stop check reads at most the sampled token, and a loop that stops on
    its iteration count reads nothing, so copying every output down each step
    paid for transfers nobody read. Outputs that leave by a host transport are
    copied by the send instead; ``landed`` hands it what was copied here.
    """

    __slots__ = ("_source", "_copy_names", "_landed")

    def __init__(self, source: dict, copy_names: Callable[[list[str]], dict]):
        self._source = source
        self._copy_names = copy_names
        self._landed: dict[str, dict] = {}

    def _copy(self, names) -> None:
        missing = [n for n in dict.fromkeys(names) if n not in self._landed]
        if missing:
            self._landed.update(self._copy_names(missing))

    def _host(self, name: str) -> dict:
        self._copy((name,))
        return self._landed[name]

    def landed(self, names=()) -> dict[int, NameToTensorList]:
        """What was copied, ``names`` first copied too, in one batched copy
        and one sync for a host transport, which would otherwise copy each
        tensor on its own."""
        self._copy(names)
        out: dict[int, NameToTensorList] = {}
        for name, by_rid in self._landed.items():
            for rid, value in by_rid.items():
                out.setdefault(rid, {})[name] = value
        return out

    def get(self, rid, default=None):
        row = self._source.get(rid)
        if row is None:
            return default
        return _LazyHostRow(self, rid, row) if isinstance(row, dict) else row

    def __getitem__(self, rid):
        row = self._source[rid]
        return _LazyHostRow(self, rid, row) if isinstance(row, dict) else row

    def __iter__(self):
        return iter(self._source)

    def __len__(self) -> int:
        return len(self._source)


class _LazyHostRow(Mapping):
    """One request's view of ``_LazyHostOutputs``: reading a name copies it."""

    __slots__ = ("_parent", "_rid", "_row")

    def __init__(self, parent: _LazyHostOutputs, rid, row: dict):
        self._parent, self._rid, self._row = parent, rid, row

    def __getitem__(self, name):
        value = self._row[name]
        # only a list of tensors has anything to copy
        return self._parent._host(name)[self._rid] if isinstance(value, list) else value

    def __contains__(self, name) -> bool:
        return name in self._row

    def __iter__(self):
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)


class Worker:
    """
    Real worker that integrates RequestStateManager, EngineManager,
    MicroScheduler, and MooncakeCommunicationManager to execute
    computation via engines.
    """

    # This worker's chunk Loops, set at load. Empty here so a worker built
    # without one -- any test fake -- chunks nowhere rather than raising from
    # postprocess.
    chunk_loops: ChunkLoops = ChunkLoops()

    # Metrics are off unless __init__ gets a status interval; class-level so a
    # worker built without __init__ reads them as off too.
    _metrics: WorkerMetrics | None = None

    # The CPUs the main, GPU and plan threads share (None: not pinned) and
    # the process mask everything else keeps; see ``mstar.utils.cpus``.
    _step_cpus: cpus.StepCpus | None = None
    # Walks that run beside the main chain, when this worker has any
    _side_steps: SideStepLane | None = None
    # lane -> the rows its step in flight holds (``_hold_rows``); replaced,
    # never updated in place
    _rows_by_lane: Mapping[int, frozenset[int]] = MappingProxyType({})
    _process_cpus: frozenset[int] = frozenset()

    @cached_property
    def device_capability(self) -> DeviceCapability:
        return DeviceCapability.of(self.device)

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        model: Model,
        my_worker_graphs: list[WorkerGraph],
        model_config: dict,
        all_worker_graph_ids_to_graph_walks: dict[int, set[str]],
        all_worker_graph_ids_to_nodes: dict[int, set[str]],
        all_worker_graph_ids_to_dyn_loops: dict[int, set[str]],
        sharding_config: ShardingConfig,
        parallel_groups: WorkerParallelGroups,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        enable_prof: bool = False,
        tcp_transfer_device="",
        dist_init_method=None,
        status_interval_s: float | None = None,
        local_worker_devices: list[str] | None = None,
        all_worker_graph_ids_to_instances: dict[int, list[list[str]]] | None = None,
    ):
        # before anything below builds a resource or a submodule that reads one
        knobs.configure((model_config or {}).get("knobs"))
        self.worker_id = worker_id
        self.device = device
        self.enable_nvtx = nvtx_enabled(enable_nvtx, device)
        self._init_step_state()

        # How long the submitter holds off the GIL waiting for the GPU thread
        # to reach the launch. submit_spec often sits on this cap, but raising
        # it to 8ms did not help; tunable for another look.
        self._launch_wait_s = _env_number(
            "MSTAR_LAUNCH_WAIT_MS", 5.0, _wait_ms, 0.0,
        ) / 1000.0
        self._max_consecutive_spec = _env_number(
            "MSTAR_MAX_CONSECUTIVE_SPEC_STEPS", 1024, int, 0,
        )
        if self._max_consecutive_spec == 0:
            logger.warning(
                "Worker %s: MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0 turns speculation "
                "off: every step yields before the next one is built",
                worker_id,
            )
        self.timer = PhaseTimer(worker_id=worker_id, enable_nvtx=self.enable_nvtx)

        self.enable_prof = enable_prof
        self.profile_info = WorkerProfileInfo()

        if self.device_capability.is_accelerator and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)
        self._plan_step_cpus(worker_ids, local_worker_devices)

        # ``dist_init_method`` is normally provided by the conductor — it
        # picks a free TCP port at startup so multiple ``mstar`` runs on
        # the same host don't collide. The ``tcp://{hostname}:29500``
        # fallback is for standalone Worker construction (e.g. tests);
        # production paths always pass a value.
        if dist_init_method is None:
            dist_init_method = f"tcp://{hostname}:29500"

        self.parallel_groups = parallel_groups
        self.parallel_groups.init_dist(
            init_method=dist_init_method,
            device=self.device,
        )

        layout = WorkerLayout.derive(
            model, my_worker_graphs, all_worker_graph_ids_to_nodes,
            sharding_config, parallel_groups, model_config,
        )
        self.parallel_nodes = layout.parallel_nodes
        self.parallel_leader_nodes = layout.parallel_leader_nodes
        self.is_tp_follower = layout.is_tp_follower
        self.chunk_loops = layout.chunk_loops
        self._all_wg_nodes = layout.all_wg_nodes
        self._my_consumer_connections = list(layout.consumer_connections)
        self._streaming_edge_names = layout.streaming_edge_names
        self._innermost_loop_by_node = layout.innermost_loop_by_node
        node_to_partition = dict(layout.node_to_partition)
        my_walks = layout.my_walks
        node_names = set(layout.node_names)

        self.communicator = make_communicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        self.wakeup_event = EventWakeup()
        self.communicator.register_event_for_poll(self.wakeup_event)

        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id=worker_id,
            hostname=hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof
        )
        kv_shm_dir_factory = None
        if (
            isinstance(
                self.tensor_manager.transfer_engine, LocalTransferEngine,
            )
            and self.device.type != "cuda"
        ):
            kv_shm_dir_factory = partial(
                make_deployment_kv_shm_dir,
                self.tensor_manager._shm_namespace,
                socket_path_prefix=socket_path_prefix,
                dist_init_method=dist_init_method,
            )

        topology = model.get_partition_topology() if model else None
        # The graph runtime owns the per-request queues and the graph state.
        self._graph_runtime = _make_graph_runtime(
            my_worker_id=self.worker_id,
            my_worker_graphs=my_worker_graphs,
            all_wg_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_wg_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
            all_wg_ids_to_nodes=all_worker_graph_ids_to_nodes,
            node_to_partition=node_to_partition,
            sharding_config=sharding_config,
            tensor_manager=self.tensor_manager,
            communicator=self.communicator,
            combined_walk_of=dict(layout.combined_walk_of),
            walk_driven_signals=frozenset(
                conn.edge_name for conn in topology.connections
                if conn.consumer_walk is not None
            ) if topology is not None else frozenset(),
        )

        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            model_config=model_config,
            parallel_groups=self.parallel_groups,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.transfer_engine,
                shm_dir_factory=kv_shm_dir_factory,
            ),
            graph_runtime=self._graph_runtime,
            model=model,
            enable_nvtx=self.enable_nvtx,
            enable_prof=self.enable_prof,
            device_capability=self.device_capability,
        )
        self._never_publish_unshared(all_worker_graph_ids_to_instances)
        self._declare_stop_policies(my_walks)

        self.request_state = RequestStateManager(
            node_to_partition=node_to_partition,
        )

        self.streams = StreamBufferManager(
            self.request_state, self.tensor_manager, self._my_consumer_connections,
            model.get_graph_walk_graphs() if model else None,
            graph_runtime=self._graph_runtime, topology=topology,
        )

        layout = replace(layout, side_keys=frozenset(self.engine_manager.side_keys))
        self.layout = layout
        self._rebuilds = _make_rebuilds(model, model_config, my_walks, self.device, layout.followed_nodes)
        services = WorkerServices(
            worker_id=worker_id,
            device=device,
            graph_runtime=self._graph_runtime,
            engine_manager=self.engine_manager,
            tensor_manager=self.tensor_manager,
            request_state=self.request_state,
            communicator=self.communicator,
            parallel_groups=self.parallel_groups,
            rebuilds=self._rebuilds,
            streams=self.streams,
        )
        self.lifecycle = RequestLifecycle(layout=layout)
        self.admission = Admission(services=services, layout=layout)
        # TP async scheduling: the leader speculates N+1 during forward N and
        # broadcasts it at once; followers rebuild it during their own N.
        self.lockstep = Lockstep(
            services=services, layout=layout, admission=self.admission,
            async_sched=tp_async_sched_setting(
                os.environ.get("MSTAR_TP_ASYNC_SCHED"), model_config.get("tp_async_sched"),
            ),
            wg_instances=all_worker_graph_ids_to_instances,
        )
        # without these the runtime sees no parallel node, and its speculation
        # filter never holds one to the leader
        self._graph_runtime.set_node_metadata(
            parallel_nodes=layout.parallel_nodes,
            parallel_leader_nodes=layout.parallel_leader_nodes,
            tp_async_nodes={n for n in node_names if self.lockstep.async_for(n)},
        )
        self.scheduler = MicroScheduler(
            services=services, layout=layout, lifecycle=self.lifecycle, lockstep=self.lockstep,
            admission=self.admission, model_config=model_config,
        )
        self._preemption = Preemption(
            self.engine_manager, self._rebuilds, self.scheduler, self.request_state, self.admission.log_throttled,
            self._rid_str,
        )
        self._set_kv_writes(model, my_walks)
        if self.engine_manager.side_keys:
            self._side_steps = SideStepLane(self, SIDE_LANE)
            logger.info(
                "Worker %s: %s run beside the chain on a side stream",
                worker_id, list(self.engine_manager.side_keys),
            )

        self.tensor_manager.rid_to_str = self._rid_str

        # The scheduler offers no rebuild for a row a step in flight holds
        # (shared by reference).
        self.scheduler.in_flight_rids = self._in_flight_rids
        self.partition_topology = topology

        if status_interval_s is not None:
            self._metrics = self._start_metrics(status_interval_s)
        self.ingest = InputIngest(
            services=services, layout=layout, lifecycle=self.lifecycle, lockstep=self.lockstep,
            wakeup_event=self.wakeup_event, metrics=self._metrics, enable_nvtx=self.enable_nvtx,
            set_rebuildable=self._set_rebuildable,
        )

    def _init_step_state(self) -> None:
        """The state a worker carries from step to step, empty; also what
        ``mstar.testing_utils.worker.make_worker`` starts from, so a field added here
        reaches every test that builds a worker."""
        # (node, walk) -> the loop-back signals the node keeps on the device
        self._device_loopback: dict[tuple[str, str], frozenset[str]] = {}
        self._host_rows_slot = 0

        # rows an in-flight GPU step or its speculation references, on any lane
        # (see ``_hold_rows``); removes and drains for these are deferred
        self._in_flight_rids: set[int] = set()

        # Side stream for D→H copies in postprocess (check_stop pre-materialize).
        # The default stream has GPU(N+1) queued behind GPU(N)'s outputs after
        # speculation, so syncing on default would also drain GPU(N+1) and
        # erase the overlap. The side stream waits on
        # ``output.completion_event`` (recorded after GPU(N)) and then runs
        # an isolated D→H, so the main thread only blocks on the copy.
        # Lazy-initialized — workers without CUDA never touch it.
        self._d2h_stream: torch.Stream | None = None
        self._pinned_d2h_buffers: dict[
            tuple[str, torch.dtype, int], list[torch.Tensor]
        ] = defaultdict(list)


    def _start_metrics(self, interval_s: float) -> WorkerMetrics:
        """Metrics on: send the conductor a WorkerStatus every ``interval_s``,
        and from ``ENABLE_METRICS`` on record the steps of the nodes this rank
        leads and the tensors every rank reads."""
        return WorkerMetrics(
            self.worker_id, interval_s, frozenset(self.parallel_leader_nodes),
            self.engine_manager.engine, self.scheduler, self._rebuilds,
            send=lambda status: self.communicator.send("conductor", ConductorMessage(
                message_type=ConductorMessageType.WORKER_STATUS, body=status,
            )),
            enable_transfers=self.tensor_manager.enable_transfer_metrics,
            preemptions=self._preemption.counts,
        )

    def _rid_str(self, request_id: int) -> str:
        """The wire identity, for a message about to leave this process."""
        return self._graph_runtime.get_rid_string(request_id)

    def _never_publish_unshared(
        self, wg_instances: dict[int, list[list[str]]] | None,
    ) -> None:
        """Turn off publication for good of each resource whose declaring
        nodes have one lockstep instance between them in this deployment:
        ``_skip_unshared_publish`` would skip it for every request."""
        if wg_instances is None:
            return
        engine = self.engine_manager.engine
        keys = []
        for key, nodes in engine.publisher_nodes.items():
            instances = {
                frozenset(instance)
                for wg_id, wg_nodes in self._all_wg_nodes.items()
                if not nodes.isdisjoint(wg_nodes)
                for instance in wg_instances.get(wg_id, ())
            }
            if len(instances) == 1 and self.worker_id in next(iter(instances)):
                keys.append(key)
        if keys:
            engine.never_publish(keys)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _fail_requests(self, errors: dict[int, str]) -> None:
        """Report requests this worker can no longer serve to the conductor.
        The request stays until the conductor's REMOVE_REQUEST comes back."""
        self.lifecycle.fail(errors)
        self._report_failures()

    def _report_failures(self) -> None:
        """Tell the conductor of the failures ``RequestLifecycle`` has recorded,
        and stop what the scheduler has queued for them."""
        errors = self.lifecycle.take_failures()
        if not errors:
            return
        wire_errors = {self.lifecycle.wire_of(rid): msg for rid, msg in errors.items()}
        for wire_rid, msg in wire_errors.items():
            logger.error("Worker %s failing request %s: %s", self.worker_id, wire_rid, msg)
        self.scheduler.drop_backlogged(errors)
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.FAIL_REQUESTS,
                body=FailRequests(errors=wire_errors),
            ),
        )

    def _teardown(self, in_flight: Collection[int]) -> None:
        """The teardown sequence: every remove and drain that no step in
        ``in_flight`` holds back."""
        self._apply_removes(in_flight)
        self._apply_drains(in_flight)

    def _apply_removes(self, in_flight: Collection[int]) -> None:
        # Rank 0 tears down nothing while its journal holds moves: each follower replays them with the next step and
        # tears down after the step a removal is stamped with, so the pages leave in the same order on every rank.
        if self.lockstep.owes_moves():
            return
        for rid in self.lifecycle.removes_ready(in_flight, self.lockstep.last_consumed_seq):
            self._tear_down_request(rid)

    def _tear_down_request(self, rid: int) -> None:
        # Rank 0 fans its removes out so that a follower tears down in the same
        # gap between steps, and can't tear down state rank 0 is still reading.
        wire_rid = self.lifecycle.wire_of(rid)
        self.lockstep.forward_teardown(rid, WorkerMessage(
            message_type=WorkerMessageType.REMOVE_REQUEST,
            body=RemoveRequest(
                request_id=wire_rid, source=MessageSource.TP_RANK_0,
                after_tp_seq=self.lockstep.last_broadcast_seq,
            ),
        ))
        # Hard cleanup: force-drop every tensor for the rid (unlink SHM),
        # ignoring ref counts / persist. Safe because the conductor only sends
        # REMOVE_REQUEST once every reader has confirmed drained (READS_DONE):
        # on the abort/fail path via a prior DrainRequest, on the happy path
        # once the api server finished reading the outputs.
        laps = PhaseLaps("worker.msg.remove_request.")
        for message in self.lifecycle.removed(rid, wire_rid):
            self.ingest.drop_for_removed(message)
        try:
            self._release_request(rid, wire_rid, laps)
        except BaseException:
            self.lifecycle.teardown_failed(rid)
            raise

    def _release_request(self, rid: int, wire_rid: str, laps: PhaseLaps) -> None:
        freed = self.engine_manager.remove_request(rid)
        if freed:
            self.admission.lift_caps(freed)
        laps.lap("engine")
        self.streams.remove_request(rid)
        self.request_state.remove_request(rid)
        self.tensor_manager.force_cleanup_request(rid)
        laps.lap("worker_state_tensors")
        self.profile_info.pop_request(rid)
        self.scheduler.clear_rid(rid)
        self.admission.clear(rid)
        self.lockstep.clear(rid, wire_rid)
        self._rebuilds.remove_request(rid)
        self._preemption.remove_request(rid)
        laps.lap("misc")
        # DEBUG: once per request on the main thread, it cost as much as the
        # engine's own removal (test/waypoint/serve_rollout.py, which waits on
        # it, runs the server at DEBUG)
        logger.debug("Request cleanup complete: %s", wire_rid)
        laps.lap("log")
        # Last: frees the handle for reuse, so nothing above may run after it.
        self._graph_runtime.remove_request(rid)
        laps.lap("runtime")

    def _set_kv_writes(self, model: Model | None, my_walks: set[tuple[str, str]]) -> None:
        """Record what each node of each paged KV cache writes here, and give each cache the loops that grow it
        in a walk this worker runs on one of its nodes."""
        labels = model.kv_labels() if model is not None else {}
        # key -> node -> its `NodeWrites`
        self._kv_writes = {}
        for kv in self.engine_manager.engine.paged_kv_caches():
            self._kv_writes[kv.key] = node_writes(labels.get(kv.key, {}), kv.nodes, my_walks)
            kv.cache.local_loops = {loop for writes in self._kv_writes[kv.key].values() for loop in writes.loops}

    def _set_rebuildable(self, request_id: int, configs: dict[str, ResourceReqConfig]) -> None:
        kv_configs = {key: config for key, config in configs.items() if isinstance(config, KVReqConfig)}
        layouts = {key: (config.kv_layout or {}).values() for key, config in kv_configs.items()}
        unpausable = any(config.unpausable for config in kv_configs.values())
        for key, rebuildable in self._rebuilds.ingest_request(request_id, self._kv_writes, layouts, unpausable).items():
            if key in configs:
                configs[key].rebuildable = rebuildable

    def _apply_drains(self, in_flight: Collection[int]) -> None:
        """Phase-1 teardown (abort/fail): stop scheduling a rid, then ACK
        READS_DONE once its in-flight reads finish. The hard cleanup waits for
        the conductor's REMOVE_REQUEST, sent only after every reader has ACKed."""
        lifecycle = self.lifecycle
        for wire_rid in lifecycle.drains_ready(in_flight):
            rid = lifecycle.begin_drain(wire_rid)
            if rid is not None:
                # each rank drains and ACKs its own READS_DONE (the conductor
                # waits on every rank)
                self.lockstep.forward_teardown(rid, WorkerMessage(
                    message_type=WorkerMessageType.DRAIN_REQUEST,
                    body=DrainRequest(request_id=wire_rid, source=MessageSource.TP_RANK_0),
                ))
                self.scheduler.drop_backlogged((rid,))
        for wire_rid, rid in lifecycle.open_drains():
            if rid is not None and self.tensor_manager.has_inflight_reads(rid):
                continue  # let get_ready_tensors resolve the futures; retry next pass
            # Not under the handle guard: a ScheduleTPNode can land before the
            # NEW_REQUEST that mints the handle, so the queued batch is counted
            # under the wire string while the handle is still None. ACKing here
            # would let the REMOVE tear the worker-graph queues out from under a
            # batch still sitting at the head of the TP FIFO.
            if self.lockstep.pending_follow_count(wire_rid) > 0:
                continue
            lifecycle.reads_done(wire_rid)
            self.communicator.send(
                "conductor",
                ConductorMessage(
                    message_type=ConductorMessageType.READS_DONE,
                    body=ReadsDone(request_id=wire_rid, entity_id=self.worker_id),
                ),
            )


    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _next_batch(self, yield_away_from_target: tuple[str, str] | None) -> ScheduledBatch | None:
        """The non-speculative path's step: one away from ``yield_away_from_target`` when given, else any.

        A follower's first call can take a head of rank 0's with no rows, and rank 0 applied the removals
        stamped with that head before it sent the next step, so this rank applies them before building it."""
        if yield_away_from_target is not None:
            batch = self.scheduler.next_batch(exclude_target=yield_away_from_target)
            if batch is not None:
                return batch
            if self.is_tp_follower:
                self._apply_removes(self._in_flight_rids)
        return self.scheduler.next_batch()

    def _build_executing_batch(self, batch: ScheduledBatch) -> ExecutingBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        batch_partition = self.request_state.get_partition_for_node(batch.node_name)

        # One walk of the columns for both: each rid's inputs as tensors, and
        # per rid, the edges that carried a stream's final chunk. Only this side can
        # turn a uuid back into a tensor, which is why the runtime reports uuids
        # and the resolution happens here.
        #
        # Seeded with the batch's rids rather than just the ones that have
        # edges: a rid with nothing ready still needs an entry, because
        # per_request_info is keyed off these and the engine indexes it by rid.
        # skip_missing: a backstop, not a fix. A live rid holding a freed uuid
        # is a bug (an abandoned speculation handing back a still-ingested
        # chunk is one), but resolving the batch in one call means one such rid
        # would otherwise fail every other request in the step -- 64 duplex
        # sessions for one bad chunk. Take out just that rid and report it, so
        # the failure is attributed and loud.
        per_request_inputs, final_edges, unresolved = (
            self.streams.input_tensors(
                batch.input_edges, batch.node_name, batch.request_to_worker_graph,
                skip_missing=True,
            )
        )
        for rid in unresolved:
            # Before the get_fwd_info loop below, so a rid that is going to be
            # failed anyway is not asked for forward-pass state as well.
            per_request_inputs.pop(rid, None)
            final_edges.pop(rid, None)
        per_request_input_metadata, per_request_info = input_metadata_and_fwd_info(
            self.request_state, self.streams,
            node_name=batch.node_name, batch_partition=batch_partition,
            per_request_inputs=per_request_inputs,
        )

        node_batch = make_executing_batch(
            self.streams,
            node_name=batch.node_name,
            graph_walk=batch.graph_walk,
            request_ids=[
                rid for rid in batch.request_to_worker_graph
                if rid not in unresolved
            ],
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
            final_edges=final_edges,
            per_request_input_metadata=per_request_input_metadata,
            request_walks=batch.request_walks,
            chunk_ranges=batch.chunk_ranges,
            engine=self.engine_manager.get_engine(batch.node_name) if batch.request_walks else None,
            rebuild_rids=batch.rebuild_rids, rebuilds=self._rebuilds,
        )
        node_batch.read_check_rids = batch.read_check_rids
        for rid in unresolved:
            # Reported the way a per-rid stage reports: left in
            # ``failed_requests`` for the run loop to drop and fail, so the
            # client gets an error instead of a session that silently stops.
            logger.error(
                "Worker %s: request %s has an input tensor the store no longer "
                "holds; failing it and running %s without it",
                self.worker_id, rid, batch.node_name,
            )
            node_batch.register_failure(
                rid, KeyError(f"input tensor missing for {batch.node_name}"),
            )
        return node_batch

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def _release_finished_prompts(self, batch: ScheduledBatch) -> None:
        """Let go of the prepared input of each prompt whose last chunk just
        landed; earlier chunks are cut from it (``Engine._prepare_chunk``)."""
        engine = self.engine_manager.get_engine(batch.node_name)
        for rid, chunk in batch.chunk_ranges.items():
            if chunk.is_final and rid in batch.request_to_worker_graph:
                engine.release_chunk_inputs(rid, batch.node_name)

    def _count_new_tokens(
        self,
        new_token_idxs: list[int],
        flat_rids: list[int],
        flat_uuids: list[int],
        signals: list[str],
        signal_idxs: list[int],
    ) -> dict[int, dict[str, int]]:
        """Token counts for the conductor, per rid. Needs numel(), hence the
        tensors -- which is why this stays here and not behind the contract.

        Indices into this batch's columns, which also carry the rid and the
        signal, so the runtime hands over no strings for this.

        First edge of a signal wins, per rid. The runtime does NOT drop repeat
        edges (``new_token_outputs`` is a bare filter), and one output routed
        to two destinations is two edges carrying the same tensors, so summing
        them would double every token.
        """
        counts: dict[int, dict[str, int]] = {}
        for idx in new_token_idxs:
            per_rid = counts.setdefault(flat_rids[idx], {})
            signal = signals[signal_idxs[idx]]
            if signal in per_rid:
                continue
            per_rid[signal] = self.tensor_manager.get_tensor(
                flat_uuids[idx]
            ).numel()
        return counts

    def _profiling_payloads(self, rids: list[int]) -> ParallelList:
        """rx/tx/timings for each rid's WORKER_GRAPHS_DONE."""
        return ParallelList(rids, [
            (
                self.tensor_manager.get_rx_info(rid),
                self.tensor_manager.get_tx_info(rid),
                self.profile_info.per_rid_graph_timings.get(rid, {}),
            ) for rid in rids
        ])

    def _register_outputs(
        self,
        route_output: RouteOutput,
    ) -> frozenset[int]:
        """Register the tensors the runtime says remote consumers (other
        workers, the api server, the conductor via persist) will read. It has
        already deduped them by uuid. A request whose tensors the transfer
        engine refuses fails alone; returns those rids, so that nothing of
        theirs is sent.
        """
        # Uuids, not descriptors: registration only ever reads ``uuid`` off
        # them and takes the tensor from the store.
        per_rid: dict[int, list[int]] = {}
        for uuid, request_id in zip(
            route_output.register_uuids,
            route_output.register_rids,
            strict=True,
        ):
            per_rid.setdefault(request_id, []).append(uuid)
        if not per_rid:
            return frozenset()
        # One staging pass for the batch: the arena collapses what would be one
        # host-blocking D2H stream sync per request into one.
        failed = self.tensor_manager.register_for_send_uuids(
            ParallelList(list(per_rid), list(per_rid.values())),
            skip_cuda_sync=True,
        )
        if not failed:
            return frozenset()
        # A follower still runs a failed rid while rank 0 sends it; report once.
        self._fail_requests({
            rid: f"Error in worker: {msg}" for rid, msg in failed.items()
            if rid not in self.lifecycle.failed_rids
        })
        return frozenset(failed)


    # ------------------------------------------------------------------
    # Main loop — async scheduling
    #
    # Pipeline shape:
    #   iter K (main thread):                          GPU thread
    #     CPU preamble  ───────────────► overlaps with execute_batch(N)
    #     speculate + build N+1
    #     await GPU(N).future Python return
    #     thread N's outputs → N+1's loop-back inputs
    #     submit GPU(N+1) ───────────────► execute_batch(N+1)
    #     _postprocess_batch(N) ─────────► overlap with GPU(N+1)
    #
    # Speculation scope (currently): AR engine only, intra-worker, 1-deep,
    # for rids whose loop is still continuing.
    # ------------------------------------------------------------------

    def _preplan_spec(
        self,
        pending: PendingBatch | None,
        speculation: Speculation,
    ) -> bool:
        """Pre-plan the speculative batch on the plan thread.

        Waits on batch N's ``commit_done`` — its resource state has to be
        committed before N+1 can admit and plan against it — and nothing else.
        The batch is not prepared yet, so the plan runs against the capture
        config's shape for the batch size; only a batched capture can serve
        that. See ``Engine.pre_plan_for_batch`` for what moving
        ``prepare_inputs`` ahead of the forward would take (and unlock).

        Returns True when the batch was pre-planned; False means ``exec``
        plans it inline, which is always correct, just slower.
        """
        spec_batch = speculation.node_batch
        engine = self.engine_manager.get_engine(spec_batch.node_name)

        if not engine.can_pre_plan(spec_batch.node_name):
            return False
        if self.timer.period:
            sb = speculation.scheduled_batch
            set_step_key(step_key(sb.node_name, sb.graph_walk))
        try:
            if pending is not None:
                # A step N this slow (a long admit, say) skips the pre-plan
                # rather than holding plan_executor; exec plans N+1 inline.
                with self.timer.span("worker.plan_thread.await_commit"):
                    committed = pending.node_batch.commit_done.wait(timeout=10.0)
                if not committed:
                    logger.warning(
                        "Worker %s: plan_executor timed out waiting for "
                        "batch N commit; skipping pre-plan", self.worker_id,
                    )
                    return False
            with self.timer.span("worker.plan_thread.reserve_slot"):
                leased = engine.reserve_replay_slot(spec_batch) is not None
            if not leased:
                return False  # eager, or no batched capture for this shape
            with self.timer.span(
                "worker.plan_thread.pre_plan",
                cpu="worker.plan_thread.pre_plan.cpu",
            ):
                return engine.pre_plan_for_batch(spec_batch)
        except Exception:
            logger.exception("Worker %s: plan_executor pre-plan failed", self.worker_id)
            self._reset_skip_plan_flags(spec_batch)
            return False
        finally:
            if self.timer.period:
                set_step_key(None)

    def _continuing_seq_lens(
        self, pending: PendingBatch, node_name: str, graph_walk: str,
        continuing: Collection[int], same_node: bool,
    ) -> list[int]:
        """The spans the continuing rows will run, for the merge's token
        budget: the submodule's estimate, -1 where it has none."""
        engine = self.engine_manager.get_engine(node_name)
        prev, info = pending.node_batch, pending.node_batch.per_request_info
        lens = []
        for rid in continuing:
            span = engine.speculative_input_seq_len(
                node_name, graph_walk, info.get(rid),
                prev.seq_len_of(rid) if same_node else -1,
            )
            lens.append(-1 if span is None else span)
        return lens

    def _reset_skip_plan_flags(self, spec_node_batch: ExecutingBatch) -> None:
        """Drop the pre-plan staged for ``spec_node_batch``.

        Used when the pre-plan was dispatched but the batch never reached the
        GPU thread: the resources would otherwise promote that stale plan into
        the next step that leases the same slot. Targeted at this batch's own
        lease, so a different slot's valid in-flight pre-plan isn't stomped.
        """
        engine = self.engine_manager.get_engine(spec_node_batch.node_name)
        engine.reset_pre_plan_for_batch(spec_node_batch)

    def _engine_thread_pool(self, name: str) -> ThreadPool:
        """A one-thread pool for the GPU or plan thread, started here, before the loop: before torch
        2.12 dynamo's config is process-wide, and a plan thread whose initializer ran at its first
        pre-plan could read and write the recompile limit while the GPU thread's compiled call patches it
        (``with_recompile_limit``). ``apply_dynamo_config`` never lowering the limit covers a limit
        raised before the pools start (a DiT's, at compile time), not that interleaving. A failed
        initializer also stops the worker here rather than failing every step."""
        pool = ThreadPool(
            max_workers=1, thread_name_prefix=f"{name}-{self.worker_id}",
            initializer=self._init_engine_thread,
        )
        pool.submit(int).result()
        return pool

    def _init_engine_thread(self) -> None:
        """Pin this executor thread to the worker's accelerator device.

        The CUDA current device is per-thread and defaults to 0. PyTorch
        ops carry per-tensor device guards, but raw Triton launches and
        bare ``torch.cuda.current_stream()`` / ``synchronize()`` calls
        resolve against the THREAD's device — on a worker whose model
        lives on a non-zero device, work issued from an unpinned thread
        lands on device 0's stream, unordered with the real compute.

        It also applies mstar's dynamo config, because dynamo config is
        per-thread since torch 2.12 and this thread compiles, and moves the
        thread to the step CPUs.
        """
        if self.device_capability.is_accelerator and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)
        apply_dynamo_config()
        self._pin_step_thread()

    def _plan_step_cpus(
        self, worker_ids: list[str], local_worker_devices: list[str] | None,
    ) -> None:
        """Build this process's CPU plan, keep this worker's step CPUs and log
        them. A ``step_thread_core`` number the mask can't honour raises.

        ``local_worker_devices`` is the device of every worker the conductor
        started, in ``worker_ids`` order; standalone, this worker is the only
        one.
        """
        devices = local_worker_devices or [str(self.device)]
        slot = worker_ids.index(self.worker_id) if local_worker_devices else 0
        plan = cpus.devices_plan(knobs.step_thread_core(), devices, knobs.idle_hold())
        cpus.install(plan)
        self._process_cpus = plan.mask
        self._step_cpus = plan.workers[slot] if plan.workers else None
        step = self._step_cpus
        if step is None:
            # a host that doesn't call for pinning is the API server's one line
            if not plan.declined:
                logger.info("Worker %s: step threads not pinned (%s)", self.worker_id, plan.why)
        else:
            logger.info(
                "Worker %s: step threads (main, GPU, plan) on CPUs %s (%s; %s, worker %d of %d, mask %s)",
                self.worker_id, cpus.format_cpu_list(step.cpus), step.why,
                self.device, slot + 1, len(devices), cpus.format_cpu_list(plan.mask),
            )

    def _pin_step_thread(self) -> None:
        if self._step_cpus is not None:
            pin_thread(self._step_cpus.cpus, self._process_cpus)

    def _execute_on_gpu_thread(
        self,
        batch: ScheduledBatch,
        node_batch: ExecutingBatch,
        plan_future: Future | None = None,
    ) -> dict[int, NameToTensorList]:
        """Run the engine on the GPU executor thread.

        The NVTX range bracketing this call is ``synchronize=False`` —
        adding a ``cudaDeviceSynchronize`` at the marker boundary would
        drain the GPU on every iter and hide the overlap between
        post-processing and the next step's kernel execution.

        Once the step's work is submitted we record a CUDA event on the
        default stream; anything that reads the output VALUES waits on it.
        """
        from mstar.utils.profiler import range_pop, range_push

        timed = self.timer.period
        t_start = _time.perf_counter() if timed else 0.0
        c_start = _time.thread_time() if timed else 0.0
        if timed:
            set_step_key(step_key(batch.node_name, batch.graph_walk))
        laps = PhaseLaps("worker.gpu_thread.")
        engine = self.engine_manager.get_engine(batch.node_name)
        if self.enable_nvtx:
            range_push("worker.gpu_thread_start", synchronize=False)
            range_pop(synchronize=False)
            range_push(
                f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                synchronize=False,
            )
        try:
            # Every output of this step stays on the device: the engine keeps
            # the rows and skips the per-request dicts and tail (cached per node
            # and walk, a dict read here).
            node_batch.rows_only = self._all_on_device(
                engine, batch.node_name, batch.graph_walk, batch.output_signals,
            )
            if batch.request_walks:
                node_batch.loopback_rids = self._loopback_rids(engine, batch)
            if logger.isEnabledFor(logging.DEBUG):
                # test/waypoint/serve_rollout.py parses this line (wire ids,
                # logged before prepare_inputs runs) to recover the DiT schedule.
                # Wire ids come from the infos; the runtime may be mid-call on
                # the main thread.
                infos = node_batch.per_request_info
                logger.debug(
                    "Executing: %s graph_walk=%s %s", node_batch.node_name,
                    batch.graph_walk, [infos[r].request_id for r in node_batch.request_ids],
                )
            laps.lap("setup")
            # The plan thread wrote this batch's plan into the resources; wait
            # for it before the forward reads them. Waiting releases the GIL —
            # which is the point: running prepare_inputs here instead just puts
            # the two threads in contention for it, and measured worse.
            if plan_future is not None:
                with self.timer.span("worker.gpu_thread.await_plan"):
                    plan_future.result()
            laps.skip()
            try:
                if node_batch.read_check_rids is not None:
                    with self.timer.span("worker.gpu_thread.read_check"):
                        self.lockstep.drop_rows_whose_read_failed(node_batch)
                with self.timer.span(
                    "worker.gpu_thread.prepare_inputs",
                    cpu="worker.gpu_thread.prepare_inputs.cpu",
                ):
                    engine.prepare_inputs(node_batch)
            except Exception:
                if plan_future is not None and node_batch.preplanned_rids is not None:
                    # The stage was for this batch, which now never runs.
                    engine.reset_pre_plan_for_batch(node_batch)
                raise
            # call is_stale after prepare_inputs because prepare_inputs may drop rids
            laps.skip()
            if plan_future is not None and engine.preplan_is_stale(node_batch):
                engine.reset_pre_plan_for_batch(node_batch)
            execution_stream = (
                torch.accelerator.current_stream(self.device)
                if self.device_capability.is_accelerator
                else None
            )
            # Device time of the step; the phase timers only see host submit
            # time, and both are needed to tell a starved iter from a busy one.
            gpu_start = None
            if self.timer.period and execution_stream is not None:
                gpu_start = torch.Event(device=self.device, enable_timing=True)
                gpu_start.record(execution_stream)
            if self.timer.period:
                # after prepare_inputs, which can drop rids — this is the row
                # count the forward really pays for
                self.timer.record(
                    f"rows.{batch.graph_walk}#", len(node_batch.request_ids),
                )
            laps.lap("pre_exec")
            with self.timer.span(
                "worker.gpu_thread.exec", cpu="worker.gpu_thread.exec.cpu",
            ):
                outputs = engine.exec_and_postprocess(node_batch)
            laps.skip()
            if execution_stream is not None:
                self._stage_host_rows(outputs)
                laps.lap("stage_host_rows")
                event = torch.Event(device=self.device)
                event.record(execution_stream)
                node_batch.completion_event = event
                laps.lap("completion_event")
                if gpu_start is not None:
                    gpu_end = torch.Event(device=self.device, enable_timing=True)
                    gpu_end.record(execution_stream)
                    self.timer.record_gpu_span(gpu_start, gpu_end, step_key(batch.node_name, batch.graph_walk))
                    laps.lap("gpu_exec_events")
            return outputs
        finally:
            # Safety net: a step that raised before publishing outputs or
            # committing would otherwise strand the plan thread preparing the
            # next one. The engine does this too on its own paths; here covers
            # a raise outside them.
            node_batch.release_waiters()
            # Publish each resource's durable state onto per_request_info so
            # the next iter's prep and the conductor see it. Runs regardless
            # of success, allocation failure, or an uncaught raise —
            # finalize_batch reads whatever state the engine actually reached.
            laps.skip()
            publish_rids = self.lifecycle.publishable(node_batch)
            node_batch.resource_publish_info = engine.finalize_batch(
                node_batch, publish_request_ids=publish_rids,
            )
            laps.lap("finalize_batch")
            if timed:
                self.timer.record(
                    "worker.gpu_thread.total", _time.perf_counter() - t_start,
                )
                self.timer.record(
                    "worker.gpu_thread.total.cpu", _time.thread_time() - c_start,
                )
                set_step_key(None)
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _await_admit_settled(self, pending: "PendingBatch | None") -> None:
        """Block until the in-flight step has admitted.

        Admit is on the GPU thread; every offload and reload is on this one. A
        page move in the window between a step's broadcast and its admit lands
        in the NEXT step's resident delta, so the follower applies it after
        admitting this step while the leader admitted with it already applied.
        The event is set at the step's first launch, after every row has
        admitted, or when the step ends without launching
        (``_submit_gpu_step``). On the per-request path later rows plan after
        it lifts; their plans read only their own in-flight rows.

        It waits however long the admit takes: giving up lets the ranks split,
        and this thread blocks on the step's result later in the pass anyway.
        Separate from the submitter's ``MSTAR_LAUNCH_WAIT_MS`` wait on the same
        event, a GIL throttle that is meant to expire.
        """
        if pending is None:
            return
        event = pending.node_batch.launch_started_event
        if event is None or event.is_set():
            return
        waited = 0.0
        with self.timer.span("worker.await_admit_settled"):
            while not event.wait(timeout=_ADMIT_FENCE_WARN_S):
                waited += _ADMIT_FENCE_WARN_S
                logger.warning(
                    "Worker %s: still waiting for node=%s walk=%s to admit after "
                    "%.0f s; scheduling and teardowns wait for it",
                    self.worker_id, pending.node_name, pending.graph_walk, waited,
                )

    def _submit_gpu_step(
        self,
        gpu_executor: ThreadPool,
        batch: ScheduledBatch,
        node_batch: ExecutingBatch,
        plan_future: Future | None,
        execute: Callable | None = None,
    ) -> Future:
        """Submit a step to the GPU thread with a fresh launch event.

        The engine sets the event at launch. A step that ends without one (a
        refused admit, a raise, an executor that never ran it) sets it on
        completion, so the admit fence and the submitter's wait always lift.
        """
        batch.chunk_loop_walks = self.chunk_loops.walks_by_node.get(batch.node_name, frozenset())
        event = node_batch.launch_started_event = threading.Event()
        future = gpu_executor.submit(
            execute or self._execute_on_gpu_thread, batch, node_batch, plan_future,
        )
        future.add_done_callback(lambda _f: event.set())
        return future

    def _submit_step(
        self,
        gpu_executor: ThreadPool,
        batch: ScheduledBatch,
        node_batch: ExecutingBatch,
        *,
        lane: int = CHAIN_LANE,
        plan_future: Future | None = None,
        speculated: bool = False,
        execute: Callable | None = None,
        **step: Any,
    ) -> PendingBatch:
        """Hand a step to the GPU thread as ``lane``'s, through ``execute`` if
        the lane runs it its own way. A ``speculated`` step was flagged in flight
        when its speculation settled, and is not timed from here (see
        ``PendingBatch.submitted_at``). Which rows the lane holds is the
        caller's to record (``_hold_rows``): the chain's step N+1 takes over
        only once N is postprocessed."""
        if not speculated:
            self._set_in_flight_flag(batch, True)
        submitted_at = _time.perf_counter() if self._metrics is not None and not speculated else 0.0
        future = self._submit_gpu_step(gpu_executor, batch, node_batch, plan_future, execute=execute)
        self.wakeup_event.register_future(future)
        step.setdefault("partition", self.request_state.get_partition_for_node(batch.node_name))
        return PendingBatch(
            batch=batch, node_batch=node_batch, node_name=batch.node_name,
            graph_walk=batch.graph_walk, future=future, submitted_at=submitted_at,
            lane=lane, **step,
        )

    def _hold_rows(self, lane: int, rids: Iterable[int]) -> None:
        """The rows ``lane``'s step in flight holds are now ``rids``. Teardown
        defers for every lane's (``_in_flight_rids``); a main-loop error fails
        the chain's."""
        rows = self._rows_by_lane = {**self._rows_by_lane, lane: frozenset(rids)}
        # in place: the scheduler holds the set too, and starts no rebuild for a row in it
        self._in_flight_rids.clear()
        self._in_flight_rids.update(*rows.values())

    def _held_beside(self, lane: int) -> set[int]:
        """The rows the other lanes' steps in flight hold."""
        return set().union(*(rows for other, rows in self._rows_by_lane.items() if other != lane))

    def _fail_step_rows(self, step: PendingBatch, outputs: BatchedModelOutput) -> None:
        """Fail the rows a per-rid stage (prepare_inputs, a per-request
        forward, postprocess) blamed; the rest of the step still postprocesses
        and routes."""
        failed = dict(step.node_batch.failed_requests)
        self._drop_failed_rids(step, outputs, failed)
        self._fail_requests(failed)

    def _finish_step(self, step: PendingBatch, outputs: BatchedModelOutput) -> None:
        """A step that ran, on any lane: its batch limit lifts, its outputs
        are postprocessed and routed, and metrics count it."""
        self.admission.clear_batch_limit(
            step.node_name, step.graph_walk, len(step.batch.request_to_worker_graph),
            walks=step.batch.rids_by_walk(),
        )
        chain = step.lane == CHAIN_LANE
        span = "worker.postprocess_batch" if chain else "worker.postprocess_side"
        keyed = self.timer.period > 0
        if keyed:
            set_step_key(step_key(step.node_name, step.graph_walk))
        try:
            with self.timer.span(span, cpu=f"{span}.cpu"):
                self._postprocess_batch(step, outputs, clear_loop_stops=chain)
        finally:
            if keyed:
                set_step_key(None)
        if self._metrics is not None:
            self._metrics.step_done(step.node_name, step.graph_walk, step.node_batch.inputs, step.submitted_at)

    def _count_turn(self, pending: "PendingBatch | None") -> None:
        """Once ``pending``'s admit has settled (``_await_admit_settled``): if
        it took the step, the step's rows have had their turn at its node. A
        refused step is retried, and its rows keep their place in line."""
        if pending is not None and pending.node_batch.admit_error is None:
            self.scheduler.note_ran(pending.batch)

    def _drop_rows_whose_loop_stopped(self, pending: PendingBatch) -> None:
        """Take a refused speculative step's rows whose loop stopped at N out
        of its batch. Postprocess drops them on a step that runs; pushed back,
        their emptied node would pop with no inputs."""
        if not pending.speculative_new_iter:
            return
        for rid in self._graph_runtime.pending_loop_stop_rids(
            list(pending.batch.rids_by_walk()), pending.loop_name,
        ):
            pending.batch.discard_rid(rid)

    def _handle_admit_failure(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch,
        referenced_rids: frozenset[str] = frozenset(),
    ) -> None:
        """Re-queue a batch whose admit refused it, and for a lack of pages make
        room for the retry. Under TP this runs on every rank: page allocation is
        driven by rank 0's scheduler and replicated by the broadcast, so the
        ranks refuse the same batches (``KVCacheManager.post_warmup_validate``
        fails fast at startup if that ever breaks).

        A teardown this rank has been holding releases pages, and spending one
        can mean no eviction is needed at all. After the push-back, so a rid torn
        down here cannot be pushed onto queues that have just been dismantled.
        ``referenced_rids``: what a speculation still touches; empty in practice,
        since nothing is speculated on a refusal. This step's own rids stay in
        ``_in_flight_rids`` until the loop clears the pending step, and their
        removals wait for that.
        """
        self.admission.requeue(batch, node_batch)
        if not isinstance(node_batch.admit_error, AllocationFailed):
            return
        self._teardown(self._in_flight_rids.union(referenced_rids))
        node, walk = batch.node_name, batch.graph_walk
        batch_ids = set(batch.request_to_worker_graph)
        if not self.lockstep.follows(node) and self._preemption.preempt(node_batch):
            return
        self.admission.relieve(
            batch, node_batch, self.lifecycle.arrivals,
            last_in_line=partial(self.scheduler.last_in_line, node, walk),
            could_refill=partial(self.scheduler.could_refill, node, walk, batch_ids),
        )

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

    def _async_schedulable(self, batch: ScheduledBatch) -> bool:
        """Any walk of the batch may speculate; the runtime skips the others."""
        return any(
            self._graph_runtime.is_async_schedulable(batch.node_name, walk)
            for walk in batch.rids_by_walk()
        )

    def _can_speculate(self, batch: ScheduledBatch) -> bool:
        if not self._async_schedulable(batch):
            return False
        if batch.node_name in self.parallel_nodes:
            # Only the leader initiates, only under TP async scheduling.
            return (
                self.lockstep.leads_async(batch.node_name)
            )
        return True

    def _plans_ahead(self, spec: Speculation) -> bool:
        """With the prefix cache open, a TP leader admits a yield-away step at exec, as its followers
        do. A pre-plan refused on rank 0 alone puts its pages back at the end of rank 0's free list,
        and the ranks then hand out different page ids, which only the cache reads."""
        node = spec.node_batch.node_name
        if spec.node_batch.read_check_rids is not None:
            # every rank admits a read-checked step inline, after the check: a follower
            # that could not read a row pre-plans without it, rank 0 with it
            return False
        return not (spec.is_yield_away and node in self.parallel_nodes and self._prefix_cache_open(node))

    def _prefix_cache_open(self, node_name: str) -> bool:
        return self.engine_manager.get_engine(node_name).prefix_cache_open(node_name)

    def _is_tp_follow_pending(self, pending: PendingBatch) -> bool:
        """The leader will send a head or a marker for ``pending``. Mirrors ``_can_speculate``."""
        return self.lockstep.follows_async(pending.node_name) and self._async_schedulable(pending.batch)

    def _settle_speculation(
        self, speculation: Speculation, success: bool,
        dropped_rids: set[int] = frozenset(),
    ) -> None:
        """Settle the runtime's staged streaming ingests, then give back the
        chunks of every rid that will not run.

        The order matters: the undo takes the chunk out of the node's input
        slot, and only then is it handed back to its StreamBuffer. Returning it
        while it is still in the slot is what tracked one chunk twice (a freed
        tensor behind a stale buffered edge) and, because a same-node prep
        ingests into the next-iter slot, let the FOLLOWING chunk take the
        current slot and be consumed first.

        Settles once per speculation: ``spec_id`` and the consumed edges are
        cleared here, so a second call is a no-op on both sides. Without that,
        a failed settle after a successful one would return the kept chunks to
        their buffers while they are still in the node's slot.

        On ``success`` the same call marks the batch speculatively scheduled, so
        "this speculation is now real" is ONE crossing into the runtime rather
        than two. The two halves address DIFFERENT rids: the flag goes on
        ``scheduled_batch``, already pruned of the dropped rids, while the undo
        targets exactly those dropped rids. The flag is cleared elsewhere
        (``_clear_in_flight_flag``), on paths with no stage to settle.
        """
        batch = speculation.scheduled_batch
        rid_to_wg = batch.graph_rows() if success else {}
        self._graph_runtime.commit_speculation(
            speculation.spec_id, success, list(dropped_rids),
            # Keeps these nodes off the ready queue while the step is in flight.
            node=batch.node_name if rid_to_wg else None,
            wg_ids=Worker._wg_ids_of(batch) if rid_to_wg else [],
            scheduled_rids=list(rid_to_wg),
        )
        give_back = (
            speculation.consumed_streaming_edges.items() if not success
            else [
                (rid, speculation.consumed_streaming_edges.get(rid, []))
                for rid in dropped_rids
            ]
        )
        for rid, edges in list(give_back):
            for se in edges:
                self.streams.return_edge(rid, se)
            speculation.consumed_streaming_edges.pop(rid, None)
        # The kept rids' chunks now belong to the step; nothing is left to undo.
        speculation.spec_id = 0
        speculation.consumed_streaming_edges.clear()
        if not success:
            self.engine_manager.get_engine(batch.node_name).release_cached_prefix(speculation.node_batch)

    @staticmethod
    def _speculated_rids(speculation: Speculation | None) -> frozenset[int]:
        """The rids a built speculation will run, which a teardown or eviction
        on the current step's behalf must leave alone."""
        if speculation is None:
            return frozenset()
        return frozenset(speculation.scheduled_batch.request_to_worker_graph)

    def _choose_spec_target(self, batch: ScheduledBatch) -> SpeculationTarget | None:
        """The (node, walk) the step after ``batch`` speculates into, with the
        rids of ``batch`` on that walk; None when nothing can be speculated."""
        # Every row, grouped by real walk: the runtime filters eligibility per
        # walk and drops a loop-back no row can take; prep_spec_rids the loop
        # completion per rid.
        groups = batch.rids_by_walk()
        if batch.rebuild_rids:
            rebuild = batch.rebuild_rids
            groups = {walk: kept for walk, rids in groups.items() if (kept := [r for r in rids if r not in rebuild])}
            if not groups:
                return None
        if len(groups) == 1:
            rows = next(iter(groups.values()))
            counts = None if len(rows) == 1 else [len(rows)]
        else:
            rows = [rid for walk_rids in groups.values() for rid in walk_rids]
            counts = [len(walk_rids) for walk_rids in groups.values()]
        ready_for_spec = self._graph_runtime.speculate_node(
            batch.node_name, list(groups), rows, counts,
            batch.loop_strides(rows),
        )
        if not ready_for_spec:
            return None
        # TODO: use the microscheduler to break ties when ready_for_spec
        # contains multiple ready nodes or walks
        info = ready_for_spec[0]
        if len(ready_for_spec) > 1 and (info.node_name, info.graph_walk) in self.chunk_loops.walks:
            # After a mixed step, prefer another walk's loop-back (decode) to
            # a chunk Loop's tick: the chain continues only its target walk's
            # rows, a mixing chain takes the rows mid-prefill along too, and
            # a prompt otherwise lets other ready work go first after each step.
            info = next(
                (i for i in ready_for_spec if (i.node_name, i.graph_walk) not in self.chunk_loops.walks),
                info,
            )
        return SpeculationTarget(info, groups[info.graph_walk])

    def _try_speculate_next(
        self,
        pending: PendingBatch,
        target: SpeculationTarget,
    ) -> Speculation | None:
        """Build a speculative N+1 batch + node_batch, by checking which nodes
        will become ready after the current batch's outputs are ingested.

        The speculated batch is a merge of:
          * **continuing** rids (subset of batch_N still alive, not
            pending-stop / pending-remove) — placeholder inputs are gathered
            from the registry now (``_get_input_tensors``) and the entries
            tied to ``consumed_edges`` are overwritten with batch_N's outputs
            after await by ``_thread_outputs_to_speculative``.
          * **fresh** rids — newly-arrived requests whose spec-target node
            is ready in the queue right now. Their inputs come from the
            usual tensor_manager path (same as ``_build_executing_batch``).
            Without this merge, new rids have to wait for the entire
            current speculation chain to drain before they can be scheduled.

        Which of them run is the scheduler's (``pick_chain_step``). When the
        rows waiting there take the whole step, it comes back as a yield-away
        of those rows (``Speculation.gave_way``) instead.
        """
        laps = PhaseLaps("speculate.")
        batch_N = pending.batch
        spec_target_info, walk_rids = target
        graph_walk = spec_target_info.graph_walk
        rid = walk_rids[0]
        spec_node_name = spec_target_info.node_name
        spec_target = (spec_node_name, graph_walk)
        # The merge below takes only this chain's capture group; another
        # group's backlog runs only once the chain yields to it. The chain's
        # rows run this step one loop iteration on, and their group is the
        # one they have there.
        next_iter_of = spec_target_info.loop_name if spec_target_info.is_new_loop_iter else None
        if self.scheduler.backlog_splits_from(spec_target, rid, next_iter_of):
            return None
        # Removes, aborts and failures are filtered here; prep_spec_rids
        # assumes that. A rid is also filtered when the target node isn't ready
        # to run it, since the target's admit would otherwise skip the readiness
        # check. The teardown filter comes first, since the readiness check would
        # queue a torn-down rid or reload its pages.
        # N's rows mid-prefill take their next chunk in this step, not the one
        # after; their nodes stay off the ready queue until N's chunk lands.
        candidates, mid_prefill = batch_N.get_continuing_rows(
            spec_node_name, walk_rids,
            self.lifecycle.is_tearing_down if self.lifecycle.teardowns_pending() else None,
            self.chunk_loops.speculated,
        )
        if spec_node_name != batch_N.node_name:
            partition = self.request_state.get_partition_for_node(spec_node_name)
            candidates = [
                r for r in candidates
                if self.admission.check_ready(
                    spec_node_name, r, self.request_state.get_fwd_info(r, partition),
                )
            ]
        # Who runs the step: the chain only ever continues its own rows, so
        # rows waiting at its target (backlogged, or on its queues) get the
        # step here or not at all while it runs. Nothing is popped yet.
        chain = self.scheduler.pick_chain_step(
            spec_target, candidates, mid_prefill, next_iter_of,
        )
        laps.lap("pick_step")
        mid_prefill = chain.mid_prefill
        if chain.pick.room == 0 and not mid_prefill:
            # the waiting rows take the whole step, as a step of their own
            batch = self.scheduler.take_chain_step(chain)
            return None if batch is None else self._yield_away_speculation(batch, gave_way=True)
        # prep keeps the first `room` of the chain's rows, in the pick's order
        prep = self._spec_preparer.prepare(
            batch_N, spec_node_name, graph_walk, chain.pick.continuing, chain.pick.room, laps,
            batch_N.loop_strides(chain.pick.continuing),
        )
        laps.lap("continuing_inputs")
        continuing = prep.ready_rids

        if not continuing and not mid_prefill:
            return None
        # fresh rows of another walk (a mixing chain): rid -> walk
        other_walks: dict[int, str] = {}
        fresh_final_edges: dict[int, set[str]] = {}

        # Merge in the waiting rows the pick gave the rest of the step.
        # Speculation only consumes work compatible with the spec target. In
        # partitioned models, unrelated ready work stays queued for
        # the normal scheduler path.
        fresh_batch = self.scheduler.take_chain_step(
            chain, len(continuing),
            # estimated from their spans in this step, if a token budget needs them
            lambda: self._continuing_seq_lens(
                pending, spec_node_name, graph_walk, continuing,
                spec_node_name == batch_N.node_name,
            ),
        )
        laps.lap("fresh_schedule")

        if fresh_batch is not None:
            # The merge below relabels these node objects with the spec
            # target's name/walk, so a batch for any other node must not be
            # merged in.
            assert fresh_batch.node_name == spec_node_name, (
                f"Speculation asked for {spec_node_name!r} but the "
                f"scheduler returned {fresh_batch.node_name!r}"
            )
            fresh_inputs, fresh_final, _ = self.streams.input_tensors(
                fresh_batch.input_edges, spec_node_name,
                fresh_batch.request_to_worker_graph,
            )
            in_flight = set(continuing)
            for rid in fresh_batch.request_to_worker_graph:
                if rid in in_flight:
                    # Shouldn't happen — continuing rids are held by the
                    # in-flight step and shouldn't be in ready queues —
                    # but if it does, the in-flight rid wins.
                    #
                    # Never a TP-follow batch: targeted calls don't pop the
                    # TP-follow FIFO (a rejected ScheduleTPNode can't re-queue).
                    self._graph_runtime.push_back_node(
                        fresh_batch.node_name, [rid],
                        [fresh_batch.request_to_worker_graph[rid]],
                    )
                    continue

                # a row mid-prefill has no popped inputs: its chunk is cut from what the engine holds
                prep.per_request_inputs[rid] = fresh_inputs.get(rid, {})
                prep.request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )
                if rid in fresh_final:
                    fresh_final_edges[rid] = fresh_final[rid]
                walk = fresh_batch.walk_of(rid)
                if walk != graph_walk:
                    other_walks[rid] = walk

        laps.lap("fresh_merge")
        logger.debug("Speculating: %s %s", spec_node_name, continuing)
        speculation = self._spec_builder.build(
            pending, spec_target_info, prep, other_walks, fresh_final_edges,
            chunk_ranges=None if fresh_batch is None else fresh_batch.chunk_ranges,
            incomplete_node_rids=None if fresh_batch is None else fresh_batch.incomplete_node_rids,
            rebuild_rids=frozenset() if fresh_batch is None else fresh_batch.rebuild_rids,
        )
        # N's rows, not fresh ones: a cleared speculation must not ready their
        # nodes while N's chunk is in flight; N's settle does that
        spec_rows = speculation.scheduled_batch.request_to_worker_graph
        speculation.continuing_rids.update(
            row.request_id for row in mid_prefill if row.request_id in spec_rows
        )
        if chain.pick.ordered:
            speculation.scheduled_batch.turn_order = chain.pick.taken
        laps.lap("assemble")
        return speculation

    @cached_property
    def _spec_context(self) -> SpeculationContext:
        return SpeculationContext(
            graph_runtime=self._graph_runtime, tensor_manager=self.tensor_manager,
            request_state=self.request_state, scheduler=self.scheduler,
            streams=self.streams, engine_manager=self.engine_manager,
        )

    @cached_property
    def _spec_preparer(self) -> SpeculationPreparer:
        return SpeculationPreparer(self._spec_context)

    @cached_property
    def _spec_builder(self) -> SpeculationBuilder:
        return SpeculationBuilder(self._spec_context)

    def _thread_outputs_to_speculative(
        self, speculation: Speculation,
        outputs_N: dict[int, NameToTensorList],
    ):
        """Splice batch N's outputs into the spec batch's inputs.

        Runs on the main thread right after N's future resolves, which is
        when the GPU thread has enqueued N, so it must not read a tensor
        VALUE: it only moves tensor lists and tests which keys are present.
        N's kernels may still be running.
        """
        threaded_continuing: set[int] = set()
        dropped: set[int] = set()
        # A rows-only step kept every output on the device: a rid that ran is
        # present, and its loop-back signal arrives empty, the way every
        # fallback batch of such a walk does (``device_loopback_signals``).
        rows_only = getattr(outputs_N, "rows_only", False)
        row_inputs = speculation.node_batch.per_request_input_tensors
        if isinstance(row_inputs, DeferredRowInputs):
            if rows_only and speculation.continuing_rids.issubset(
                outputs_N.row_request_ids or (),
            ):
                # Every continuing row ran: what the loop below would do row by
                # row, for all of them at once.
                row_inputs.set_signals(name for name, _ in speculation.consumed_edges)
                speculation.dropped = set()
                return
            # Anything else edits rows one by one.
            speculation.node_batch.per_request_input_tensors = row_inputs.materialize()
        ran = frozenset(outputs_N.row_request_ids or ()) if rows_only else None
        # a mixed N's rows that looped back on the device arrive empty too
        loopback = getattr(outputs_N, "loopback_rids", frozenset())
        chunked = speculation.scheduled_batch.chunk_ranges
        for rid in list(speculation.node_batch.request_ids):
            if rid not in speculation.continuing_rids:
                continue  # fresh rid — inputs already gathered.
            if rid in chunked:
                # mid-prefill: its chunk is cut from the input the engine holds, not N's outputs
                threaded_continuing.add(rid)
                continue
            if rid in loopback:
                inputs = speculation.node_batch.per_request_input_tensors[rid]
                for input_name, _ in speculation.consumed_edges:
                    inputs[input_name] = []
                threaded_continuing.add(rid)
                continue
            if ran is not None:
                if rid in ran:
                    inputs = speculation.node_batch.per_request_input_tensors[rid]
                    for input_name, _ in speculation.consumed_edges:
                        inputs[input_name] = []
                    threaded_continuing.add(rid)
                else:
                    dropped.add(rid)
                continue
            rid_outputs = outputs_N.get(rid, {})
            ok = True
            for input_name, _ in speculation.consumed_edges:
                tensors = rid_outputs.get(input_name, [])
                if not tensors:
                    ok = False
                    break
                speculation.node_batch.per_request_input_tensors[rid][input_name] \
                    = list(tensors)
            if ok:
                threaded_continuing.add(rid)
            else:
                dropped.add(rid)

        if dropped:
            logger.warning(
                "Speculation: dropped rids %s (no loop-back output from N)",
                sorted(dropped),
            )
            # Its final chunks go back below; it must not flush or report done.
            speculation.node_batch.drop_rids(dropped)
            for r in dropped:
                speculation.node_batch.per_request_input_tensors.pop(r, None)
                speculation.scheduled_batch.discard_rid(r)
                speculation.node_batch.per_request_input_metadata.pop(r, None)
        speculation.continuing_rids = threaded_continuing
        speculation.dropped = dropped

    # ------------------------------------------------------------------
    # TP async scheduling — the follower side
    # ------------------------------------------------------------------
    # A follower whose in-flight batch N is a head's spec_from_seq rebuilds
    # that head from replicated state during its own N. No commit / cancel:
    # every post-N verdict is derived per rank. The leader always sends a head
    # or a TPNoSpeculation marker per step, settled before N is post-processed.

    def _try_follow_speculation(self, pending: PendingBatch) -> Speculation | None:
        head = self.lockstep.follow_head()
        if head is None or not head.speculative:
            return None
        if head.spec_from_seq != pending.tp_seq:
            # From some other step: the serial path gets it in FIFO order.
            return None
        key = self.scheduler._key
        if head.node_name != pending.node_name or \
                key(head.node_name, head.graph_walk) != key(pending.node_name, pending.graph_walk):
            # Leaders only speculate same-node loop-backs; leave it serial.
            logger.warning(
                "Worker %s: speculative head %s/%s (seq %d) does not match its "
                "parent batch %s/%s; leaving it for the serial path",
                self.worker_id, head.node_name, head.graph_walk, head.spec_seq,
                pending.node_name, pending.graph_walk,
            )
            return None

        # Same gate as the serial path: a step must not be built against a page
        # state that is still mid-replay. Every poll retries the owed moves, so
        # returning here makes progress where reading the half-applied state as
        # "not ready" never could.
        # Of the steps before the head, only N may still index a page of the head's hit, or hold pages back
        committed = pending.node_batch.commit_done
        if not self.lockstep.settle_delta(caught_up=committed.is_set()):
            if committed.is_set() or not self.lockstep.awaits_commit():
                return None
            # A reload in the head waits for N's commit, which every exit of N's step releases.
            committed.wait()
            if not self.lockstep.settle_delta(caught_up=True):
                return None

        batch_N = pending.batch
        # The leader names its rids by wire string; these are this rank's handles.
        head_rids = self.lockstep.tp_rids(head)
        # before anything is popped: lists that do not match the rows raise
        chunk_ranges = tp_chunk_ranges(head, head_rids)
        # rows whose input read failed here: built without, as on the serial path
        unread = self.lockstep.unread_rows(head, head_rids)
        head_walk_idx = dict(zip(head_rids, head.walk_idx, strict=True)) if head.walks else None
        in_batch, held = batch_N.request_to_worker_graph, batch_N.incomplete_node_rids
        # rebuild rows, which the graph runtime holds no node of, so they are never popped. A rebuild whose chunk ran
        # in N is one of N's rows mid-prefill, which rank 0 stamps a rebuild row from its log too.
        rebuild = self.lockstep.follow_rebuilds(
            head.node_name, [r for r in head_rids if r not in in_batch or r in held],
        )
        # rows read here: prepped (continuing) or popped (fresh); every row,
        # unread ones included, runs the leader's continuation of N
        in_n = [r for r in head_rids if r in in_batch]
        # N's rows mid-prefill loop back on nothing of N's: as on the leader,
        # their next chunk is cut from the input held since their first. N's
        # chunks came from the leader, so every rank splits these out alike.
        mid_prefill = {r for r in in_n if r in held} if held else set()
        continuing = [r for r in in_n if r not in unread and r not in mid_prefill]
        fresh = [r for r in head_rids if r not in in_batch and r not in unread and r not in rebuild]
        if not in_n:
            return None
        spec_walk = head.walks[head_walk_idx[in_n[0]]] if head.walks else head.graph_walk
        # The leader already chose the target; this just reports its loop
        # context. speculate_node's eligibility filter cannot run here -- it
        # requires the node be a parallel LEADER node.
        spec_target_info = self._graph_runtime.get_spec_target(
            batch_N.node_name, head.node_name, spec_walk, in_n[0],
        )
        if spec_target_info is None:
            return None

        # Polling the StreamBuffers stays on this side: they hold tensors.
        polled: list[tuple[int, StreamingEdge]] = []
        per_rid_counts: list[int] = []
        for r in continuing:
            edges = self.streams.poll_for(r, head.node_name)
            polled.extend((r, e) for e in edges)
            per_rid_counts.append(len(edges))

        # Fresh rids are popped BEFORE the continuing prep, so the only undo
        # a failure needs is push_back_node. The reverse order leaves a
        # successful all-or-nothing prep to unwind, with no API for it.
        popped = self.lockstep.pop_ready_rids(
            head.node_name, head.graph_walk, fresh,
            walks=head.walks or None,
            walk_idx=[head_walk_idx[r] for r in fresh] if head.walks else None,
        )

        def _return_all_polled() -> None:
            for r, se in polled:
                self.streams.return_edge(r, se)

        if popped is None:
            # ``pop_ready_rids`` scans with ``allow_reload=False``: an offloaded
            # rid reads not-ready and waits for a replayed delta. The delta is
            # settled above, so a rid still off-device is a divergence, not a lag.
            _return_all_polled()
            self.lockstep.warn_unchecked_read_failure(head, fresh)
            return None

        # output_signals are handled by SpeculationBuilder.assemble
        fresh_wg, fresh_edges = popped.wg_ids, popped.edge_specs

        # The leader's list is the composition every rank runs: all-or-nothing,
        # no local finished-rid skipping, no room_for_continuing cap.
        prep = self._graph_runtime.prep_follow_spec_rids(SpeculationPrepInput(
            spec_node_name=head.node_name,
            curr_node_name=batch_N.node_name,
            graph_walk=spec_walk,
            rids=continuing,
            room_for_continuing=None,
            streaming_edges=self.streams.edge_specs(polled),
            streaming_edges_per_rid=per_rid_counts,
        ))
        if prep is None:
            # prep rolled its own ingests back; hand the fresh rids their ready
            # slots back too, so the serial path can still pick them up.
            self._graph_runtime.push_back_node(
                head.node_name, list(fresh_wg), list(fresh_wg.values()),
            )
            _return_all_polled()
            return None

        consumed = set(prep.consumed_streaming_edge_idxs)
        consumed_streaming_edges: dict[int, list[StreamingEdge]] = {}
        for i, (r, se) in enumerate(polled):
            if i in consumed:
                self.streams.mark_ingested(r, se)
                consumed_streaming_edges.setdefault(r, []).append(se)
            else:
                self.streams.return_edge(r, se)

        # Both seeded with their rid lists, so the `in prep_inputs` test below
        # still separates prepped rids from fresh ones -- a ready rid with no
        # prepped edge gets an empty mapping, as a zero-length slice used to.
        prep_inputs = self.streams.input_tensors(
            prep.input_edges, head.node_name, prep.ready_rids,
        ).by_rid
        fresh_inputs, fresh_final_edges, _ = self.streams.input_tensors(
            fresh_edges, head.node_name, list(fresh_wg),
        )
        prep_wg = dict(zip(prep.ready_rids, prep.wg_ids, strict=True))

        new_request_to_worker_graph: dict[int, int] = {}
        per_request_inputs: dict[int, NameToTensorList] = {}
        for rid in head_rids:  # wire order == the leader's step order, split rows ordered
            if rid in prep_inputs:
                new_request_to_worker_graph[rid] = prep_wg[rid]
                per_request_inputs[rid] = prep_inputs[rid]
            elif rid in unread or rid in rebuild:
                per_request_inputs[rid] = {}
            elif rid in mid_prefill:
                new_request_to_worker_graph[rid] = in_batch[rid]
                per_request_inputs[rid] = {}
            else:
                new_request_to_worker_graph[rid] = fresh_wg[rid]
                per_request_inputs[rid] = fresh_inputs[rid]
        request_walks = {r: head.walks[w] for r, w in head_walk_idx.items()} if head.walks else None
        if unread or rebuild:
            # In with no inputs, as on the serial path: the step has rank 0's
            # rows until its read check takes these out. An unread row not in
            # N is no continuing row, so voiding the speculation
            # (``_maybe_clear_spec``) pushes it back though it was never popped.
            # Harmless only while every later head naming it is read-checked,
            # so ``unread_rows`` keeps it out of the pop: rank 0 checks each
            # step taking the input until one runs (``Lockstep.consume_reads``),
            # and that step's check takes the row out on every rank.
            new_request_to_worker_graph, _ = self.lockstep.with_unpopped_rows(
                head, head_rids, unread | rebuild, new_request_to_worker_graph, request_walks or {},
            )

        # Committed: the serial path must not see the head any more.
        self.lockstep.pop_follow_head()
        if fresh:
            self.scheduler.note_follow_pop(key(head.node_name, head.graph_walk))
        # Its rids count as in flight now, so a remove landing before submit is
        # deferred like any other; ``_set_pending`` re-derives the set on submit.
        self._hold_rows(CHAIN_LANE, self._rows_by_lane.get(CHAIN_LANE, frozenset()) | set(head_rids))
        logger.debug(
            "Follow-speculating: %s %s (seq %d from %d)",
            head.node_name, head_rids, head.spec_seq, head.spec_from_seq,
        )
        speculation = self._spec_builder.assemble(
            pending, spec_target_info,
            new_request_to_worker_graph, per_request_inputs,
            consumed_streaming_edges,
            [r for r in in_n if r not in mid_prefill] if mid_prefill else in_n,
            fresh_final_edges,
            is_same_node=True, spec_id=prep.spec_id, tp_seq=head.spec_seq,
            request_walks=request_walks,
            chunk_ranges=chunk_ranges,
            incomplete_node_rids=unfinished_chunk_rids(chunk_ranges),
            rebuild_rids=rebuild,
        )
        # N's rows, as on the leader: N's settle readies them, not a cleared speculation
        speculation.continuing_rids.update(mid_prefill)
        if chunk_ranges:
            # a fresh row's first chunk caps its Loop, as the leader's merge did
            self.lockstep.cap_chunk_loops(speculation.scheduled_batch)
        if head.read_check:
            speculation.node_batch.read_check_rids = tuple(head_rids)
        return speculation

    @staticmethod
    def _step_voids_head(pending: PendingBatch) -> str | None:
        """Why N's verdict voids a head built on it, or ``None``. Both fields are
        final once N's result is taken (``_step_result``); the leader clears on exactly these two."""
        if pending.node_batch.admit_error is not None:
            return "admit_error"
        if pending.node_batch.failed_requests:
            return "failed rids"
        return None

    def _step_result(self, pending: PendingBatch) -> BatchedModelOutput:
        """N's outputs. A forward that raised becomes N's verdict instead: no
        outputs and every row failed (``failed_requests``), which voids a head
        built on N and lets a step independent of N run. When the raise is on
        every rank, every rank does so alike. A raise on one rank alone leaves
        the others in a step it won't run, and the group wedges (#1235)."""
        try:
            return pending.future.result()
        except Exception as exc:
            err, _ = client_error_text(
                exc, f"Worker {self.worker_id} step {pending.node_name}/{pending.graph_walk} failed",
                log=logger,
            )
            failed = pending.node_batch.failed_requests
            for rid in pending.batch.request_to_worker_graph:
                failed.setdefault(rid, f"Error in worker: {err}")
            return BatchedModelOutput()

    def _await_tp_follow_step(
        self, pending: PendingBatch, arm: Callable[[Speculation], None],
    ) -> tuple[BatchedModelOutput | None, Speculation | None]:
        """Wait for step N and settle the leader's decision about N+1. Returns
        ``(outputs or None, armed head or None)``. Once N is done this never
        returns without a decision: going serial early strands the leader."""
        s = pending.tp_seq
        outputs: BatchedModelOutput | None = None
        t_done = last_warn = 0.0
        while True:
            if outputs is None and pending.future.done():
                outputs = self._step_result(pending)
                t_done = last_warn = _time.perf_counter()
            if self.lockstep.is_no_spec(s):
                return outputs, None
            head = self.lockstep.follow_head()
            if head is not None and head.speculative and head.spec_from_seq == s:
                void = self._step_voids_head(pending) if outputs is not None else None
                if void is not None and not self._prefix_cache_open(head.node_name):
                    # The leader clears its speculation on this verdict too.
                    self.lockstep.void_follow_head()
                    logger.debug(
                        "Worker %s: dropped speculative head seq %d (parent %d %s)",
                        self.worker_id, head.spec_seq, s, void,
                    )
                    return outputs, None
                # With the prefix cache open, build the head even when N's verdict voids it, since the
                # leader pre-plans every head it sends and this rank's pages must move as the leader's do.
                spec = self._try_follow_speculation(pending)
                if spec is not None:
                    arm(spec)
                    return outputs, spec
                # Not buildable yet (fresh rid / stream chunk): poll and retry.
            elif head is not None and head.spec_seq > s:
                self.lockstep.leader_passed(s, head)
                return outputs, None
            self.communicator.wait_for_work(20)
            self.ingest.poll()
            self._report_failures()
            # Rank 0 released these before it sends the head this loop waits for.
            self._teardown(self._in_flight_rids)
            if outputs is not None:
                now = _time.perf_counter()
                if now - last_warn > 2.0:
                    last_warn = now
                    logger.warning(
                        "Worker %s: still waiting for the leader's decision on "
                        "step seq %d, %.1fs after it finished (head queued: %s)",
                        self.worker_id, s, now - t_done,
                        head is not None and head.spec_from_seq == s,
                    )

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _wg_ids_of(batch: ScheduledBatch) -> list[int]:
        """``set_in_flight``'s worker graphs: one id for a step of one walk
        (a node in a walk lives in one worker graph), else one per rid."""
        rid_to_wg = batch.graph_rows()
        if batch.request_walks or not rid_to_wg:
            return list(rid_to_wg.values())
        return [next(iter(rid_to_wg.values()))]

    def _set_in_flight_flag(self, batch: ScheduledBatch, value: bool) -> None:
        self._graph_runtime.set_in_flight(
            batch.node_name, list(batch.graph_rows()),
            Worker._wg_ids_of(batch), value,
        )

    def _clear_in_flight_flag(self, batch: ScheduledBatch) -> None:
        self._set_in_flight_flag(batch, False)


    def _wait_for_step(self, batch_N: PendingBatch) -> None:
        """Wait for batch N's GPU work: its completion event, or the whole
        device when it has none."""
        if self.device.type == "cpu":
            return
        if batch_N.node_batch.completion_event is not None:
            with self.timer.span("worker.postprocess.event_sync"):
                batch_N.node_batch.completion_event.synchronize()
        else:
            with self.timer.span("worker.postprocess.device_sync"):
                torch.accelerator.synchronize(self.device)

    def _postprocess_batch(
        self, batch_N: PendingBatch,
        outputs: BatchedModelOutput,
        clear_loop_stops: bool = True,
    ):
        # Stage stopwatch named like the NVTX ranges below; a clock stamp, not
        # a nested `timer.span`, because of the two early returns.
        _pp_t = _time.perf_counter() if self.timer.period else 0.0
        def _pp_stage(name: str) -> None:
            nonlocal _pp_t
            if not self.timer.period:
                return
            now = _time.perf_counter()
            self.timer.buf[f"worker.postprocess.{name}"].append(now - _pp_t)
            _pp_t = now

        self._finish_rebuild_rows(batch_N, outputs)
        if self.enable_nvtx:
            range_push("worker.postprocess.cleanup_inputs", synchronize=False)

        # a row mid-prefill keeps its inputs: its next chunk reads them again
        held = batch_N.batch.incomplete_node_rids
        rids = [
            r for r in batch_N.batch.request_to_worker_graph if r not in held
        ] if held else list(batch_N.batch.request_to_worker_graph)
        self.lockstep.consume_reads(batch_N.node_batch)
        # What the runtime dereferenced to zero and cannot reclaim itself: a
        # runtime behind the contract has the bookkeeper, not the shm files or
        # the registered memory.
        self.tensor_manager.cleanup_collectable(
            *self._graph_runtime.cleanup_consumed_inputs(
                batch_N.batch.node_name, rids,
                [batch_N.batch.request_to_worker_graph[r] for r in rids],
            )
        )
        _pp_stage("cleanup_inputs")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.pending_loop_stops", synchronize=False)
        # If any nodes in the batch have "overstayed" their loop stop, then make
        # sure to not route their outputs
        valid_rids = set(batch_N.node_batch.request_ids) | batch_N.node_batch.cached_rids
        if batch_N.speculative_new_iter:
            stopped_rids = self._graph_runtime.pending_loop_stop_rids(
                list(batch_N.batch.rids_by_walk()), batch_N.loop_name,
            )
            stopped_rids &= set(batch_N.node_batch.request_ids)
            for stopped_rid in stopped_rids:
                outputs.pop(stopped_rid, None)
                valid_rids.discard(stopped_rid)
                batch_N.batch.discard_rid(stopped_rid)
                batch_N.node_batch.per_request_info.pop(stopped_rid, None)
                # its loop already stopped; see ExecutingBatch.drop_rids
                batch_N.node_batch.final_stream_rids.discard(stopped_rid)
                batch_N.node_batch.partition_done_rids.discard(stopped_rid)
        # keep the forward's order: other code walks this list positionally
        batch_N.node_batch.request_ids = [
            rid for rid in batch_N.node_batch.request_ids if rid in valid_rids
        ]
        if not valid_rids:
            # no outputs leave, but the step's kernels may still be running
            # in the slot a later step reuses
            self._wait_for_step(batch_N)
            if self.enable_nvtx:
                range_pop(synchronize=False)
            return

        # pending stops are only needed for one iteration of the chain, so can
        # be cleared now; not by a side lane's step (``_finish_step``), which
        # lands between two chain steps: the chain's next one has yet to read them.
        if clear_loop_stops:
            self._graph_runtime.clear_pending_loop_stops()

        # A rid failed or drained after this step was built is torn down, so
        # none of its outputs may leave: the step can already be in flight
        # when the rid fails (the stop check of the step before it, say).
        # Not on a follower: rank 0 decides what runs there, and may still
        # send a step that needs this one's loop-back.
        if (
            self.lifecycle.failed_rids
            and not self.lockstep.follows(batch_N.node_name)
        ):
            torn_down = self.lifecycle.failed_rids & valid_rids
            if torn_down:
                self._drop_failed_rids(batch_N, outputs, torn_down)
                valid_rids -= torn_down
                if not valid_rids:
                    # no outputs leave, but the step's kernels may still be
                    # running in the capture slot a later step reuses
                    self._wait_for_step(batch_N)
                    if self.enable_nvtx:
                        range_pop(synchronize=False)
                    return

        # An engine can drop rids that were skipped during execution (a
        # submodule's prepare_inputs returned None) from node_batch.request_ids,
        # but it cannot reach the worker-side ScheduledBatch. Reconcile it here so
        # the routing/output loops below only touch rids that ran or that the
        # cache served.
        for rid in list(batch_N.batch.request_to_worker_graph):
            if rid not in valid_rids:
                batch_N.batch.discard_rid(rid)

        _pp_stage("pending_loop_stops")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.synchronize_completion_event", synchronize=False)

        # Wait for batch N's completion event before proceeding
        # TODO: may need to refine this based on how it affects performance?
        if batch_N.batch.request_to_worker_graph:
            self._wait_for_step(batch_N)

        if self.enable_prof:
            batch_N.node_batch.exec_timings.fwd_end = time.perf_counter()

        _pp_stage("completion_event_sync")
        if batch_N.batch.chunk_ranges:
            self._release_finished_prompts(batch_N.batch)
        # Waited on above, so the spans have landed; cut before anything reads
        # the outputs: the stop check, the store, a streaming edge.
        empty_spans: dict[int, list[str]] = {}
        if outputs.output_spans and batch_N.batch.request_to_worker_graph:
            empty_spans = self._crop_to_spans(outputs)
            _pp_stage("crop_spans")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.prematerialize", synchronize=False)

        # Prematerialising pulls the stop check's rows to the host, so this can
        # carry a device transfer.
        engine = self.engine_manager.get_engine(batch_N.node_name)
        cpu_outputs, host_rows = self._prematerialize_for_check_stop(
            outputs, batch_N.node_batch.completion_event,
            request_ids=batch_N.node_batch.request_ids,
        )
        for rid in self._rebuilds.keep_prompts(batch_N.node_batch, batch_N.batch.incomplete_node_rids):
            engine.keeps_no_ids(batch_N.node_name, rid)
        # Until its last chunk a row's sampled token is not a real sample, so it
        # keys no prefix and decides no stop (`_check_stops`).
        engine.step_finished(
            batch_N.node_batch, cpu_outputs, host_rows=host_rows,
            skip_rids=batch_N.batch.incomplete_node_rids,
        )
        self._rebuilds.keep_samples(
            batch_N.node_batch, cpu_outputs, host_rows, skip_rids=batch_N.batch.incomplete_node_rids,
        )
        _pp_stage("prematerialize")
        runtime_stops, tokens = self._runtime_stop_column(batch_N, host_rows)
        checked = None
        if not runtime_stops:
            # the model's check_stop decides, before the runtime, which then
            # routes without stopping for it
            checked = self._check_stops(batch_N, outputs, cpu_outputs, host_rows, engine).stops
            _pp_stage("check_stop")
        request = self._get_postprocess_request(
            batch_N, host_rows, engine, runtime_stops, tokens, checked,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.runtime", synchronize=False)

        # The runtime checks the step's stops, routes its outputs and sends
        # them, stopping for what only the worker has (PostprocessNeed).
        # Main loop only: the runtime holds `&mut self` across its GIL
        # release, so a concurrent caller gets "Already mutably borrowed"
        # rather than blocking.
        rt = self._graph_runtime
        _t_rt = _time.perf_counter() if self.timer.period else 0.0
        step = rt.begin_postprocess(request)
        rt_time = _time.perf_counter() - _t_rt if self.timer.period else 0.0
        stored = None
        sent_here = False
        while step.need is not PostprocessNeed.DONE:
            _t = _time.perf_counter() if self.timer.period else 0.0
            if step.need is PostprocessNeed.STOPS:
                model_stops = self._check_stops(batch_N, outputs, cpu_outputs, host_rows, engine)
                # after the request was built, which carries the step's other
                # partition ends; the runtime reports them on the same send
                for rid in model_stops.ended:
                    rt.mark_partition_done(rid, batch_N.partition)
                stage = "check_stop"
                _t_rt = _time.perf_counter() if self.timer.period else 0.0
                step = rt.resume_postprocess(stops=model_stops.stops, dropped=model_stops.dropped)
            elif step.need is PostprocessNeed.STORE:
                # What the stop check withheld (``StopCheck.withheld``), and an
                # output cut to an empty span, is not stored, so not routed: a
                # streaming edge then carries no item for the row, and the
                # producer's item count leaves it out as well.
                for withheld in (batch_N.node_batch.withheld_outputs, empty_spans):
                    for rid, names in withheld.items():
                        rid_outputs = outputs.get(rid)
                        if rid_outputs:
                            for name in names:
                                rid_outputs.pop(name, None)
                stored, route_tensors = self._store_outputs(
                    batch_N, outputs, request, step, cpu_outputs, host_rows, engine,
                )
                stage = "store_tensors"
                _t_rt = _time.perf_counter() if self.timer.period else 0.0
                step = rt.resume_postprocess(stored=route_tensors)
            else:
                send = self._send_input(batch_N, request, step, stored, engine)
                sent_here = True
                stage = "send_prep"
                _t_rt = _time.perf_counter() if self.timer.period else 0.0
                step = rt.resume_postprocess(send=send)
            if self.timer.period:
                now = _time.perf_counter()
                self.timer.record(f"worker.postprocess.{stage}", _t_rt - _t)
                rt_time += now - _t_rt
        if self.timer.period:
            self.timer.record("worker.postprocess.runtime", rt_time)
            _pp_t = _time.perf_counter()

        if self.enable_nvtx:
            range_pop(synchronize=False)
        if not sent_here and batch_N.batch.request_to_worker_graph:
            # sent without the worker (SEND does these before its send)
            self._buffer_publish(batch_N)
            rt.update_dynamic_loop_iters(
                batch_N.node_batch.per_request_info, batch_N.partition,
            )
        for rid in step.completed_rids:
            self.request_state.flush_publish_info(rid, batch_N.partition)
        # Those sends were the last readers of this step's host copies.
        self.tensor_manager.drop_host_copies()
        _pp_stage("tail")
        engine.after_output_copy(batch_N.node_batch)
        _pp_stage("after_output_copy")

    def _finish_rebuild_rows(self, batch_N: PendingBatch, outputs: BatchedModelOutput) -> None:
        """Take the step's rebuild rows off it and advance their rebuilds. A rebuild's last chunk lands it and
        lets its request's decode row run."""
        batch, node_batch = batch_N.batch, batch_N.node_batch
        rebuild = batch.rebuild_rids
        if not rebuild:
            return
        node = batch_N.node_name
        # the engine also skips a row the prefix cache served whole, whose rebuild is done (below)
        skipped = (rebuild & node_batch.skipped_rids) - node_batch.cached_rids
        if skipped:
            self._fail_requests({
                rid: f"Error in worker: {node} skipped the rebuild row of {self._rid_str(rid)}, whose inputs come "
                "from its kept ids" for rid in skipped
            })
        # a row the prefix cache served whole ran no forward, and its rebuild is done all the same
        ran = set(node_batch.request_ids) | node_batch.cached_rids
        landed = {
            rid: cursor for rid in rebuild
            if rid in ran and (cursor := self._rebuilds.land_chunk(rid, node, batch.chunk_ranges.get(rid)))
        }
        if landed:
            # the step's kernels read the prepared input released here
            self._wait_for_step(batch_N)
            engine = self.engine_manager.get_engine(node)
            for rid, cursor in landed.items():
                if not cursor.resumes_prefill:
                    engine.release_chunk_inputs(rid, node)
                engine.land_rebuild(node, rid)
                self._preemption.rebuilt(rid, node, cursor.total)
        for rid in rebuild:
            outputs.pop(rid, None)
            batch.discard_rid(rid)
            batch.chunk_ranges.pop(rid, None)
            node_batch.per_request_info.pop(rid, None)
        batch.incomplete_node_rids -= rebuild
        node_batch.cached_rids -= rebuild
        node_batch.request_ids = [rid for rid in node_batch.request_ids if rid not in rebuild]

    def _runtime_stop_column(
        self, batch_N: PendingBatch, host_rows: "HostRows | None",
    ) -> tuple[bool, "torch.Tensor | None"]:
        """Whether the runtime checks the step's declared stop policies, and
        the token column they read (None when none reads one). A step that
        mixes walks, has no declared policies, or has no column keeps
        check_stop."""
        key = (batch_N.node_name, batch_N.graph_walk)
        if (
            key not in self._stop_token_signal or len(batch_N.batch.rids_by_walk()) > 1
            # a row mid-prefill decides no stop, so its step keeps check_stop
            or batch_N.batch.incomplete_node_rids
        ):
            return False, None
        token_signal = self._stop_token_signal[key]
        if token_signal is None:
            return True, None
        tokens = _host_column(host_rows, token_signal)
        return tokens is not None, tokens

    def _get_postprocess_request(
        self, batch_N: PendingBatch, host_rows: "HostRows | None", engine,
        runtime_stops: bool, tokens: "torch.Tensor | None",
        checked: dict[int, set[str]] | None,
    ) -> PostprocessRequest:
        """The step's ``PostprocessRequest``: its rows, its stops known before
        the runtime's check (``checked``: the model's, when the runtime does
        not check), and the host columns the runtime reads in place."""
        batch = batch_N.batch
        by_walk = batch.rids_by_walk()
        walks = list(by_walk) or [batch_N.graph_walk]
        if len(by_walk) <= 1:
            rids, rid_walk_idx = list(by_walk.get(walks[0], ())), None
            # a combined batch keeps its first walk's names, which may not be this walk's
            signals = (
                list(self._walk_signals(batch, walks[0])) if batch.request_walks
                else batch.output_signals
            )
        else:
            rids, rid_walk_idx = [], []
            for i, walk_rids in enumerate(by_walk.values()):
                rids.extend(walk_rids)
                rid_walk_idx.extend([i] * len(walk_rids))
            # several walks route under the union of their names; a walk's
            # node ignores the ones it lacks
            signals = list(dict.fromkeys(
                signal for walk in walks for signal in self._walk_signals(batch, walk)
            ))
        # Stream-terminated loop: a stream-consuming node inside a loop has no
        # internal stop signal (unlike a self-EOS loop), so when it consumes
        # the terminal chunk of its stream (``final_stream_rids``: the
        # StreamBuffer popped ``is_final``, which a ``continue_after_done``
        # policy never sets) end its innermost enclosing loop. Without this
        # the loop would spin to ``max_iters`` and the partition would never
        # report done.
        stops: dict[int, set[str]] = {}
        held = batch.incomplete_node_rids  # mid-prefill: decides no stop
        for rid in batch_N.node_batch.final_stream_rids:
            if rid in held:
                continue
            loop_name = self._innermost_loop_by_node.get((batch.walk_of(rid), batch_N.node_name))
            if loop_name is not None:
                stops.setdefault(rid, set()).add(loop_name)
        for rid, loops in (checked or {}).items():
            stops.setdefault(rid, set()).update(loops)
        # Device loop-back: every routed signal stays on the device and the
        # client's copy rides inline, so the route stores nothing.
        # A step with log-probs stores instead: STORE passes them to the route.
        inline = None
        if (
            host_rows is not None and rid_walk_idx is None
            and SAMPLED_LOGPROB not in host_rows.buffers
            # an inline route carries every output, a withheld one too
            and not batch_N.node_batch.withheld_outputs
            and self._all_on_device(engine, batch_N.node_name, batch_N.graph_walk, signals)
        ):
            client = engine.inline_client_signals(batch_N.node_name, batch_N.graph_walk)
            signal = next((s for s in client if s in signals), None)
            column = None if signal is None else _host_column(host_rows, client[signal])
            if column is not None:
                inline = InlineColumn(signal, column)
        return PostprocessRequest(
            partition=batch_N.partition, node_name=batch_N.node_name, graph_walk=walks[0],
            output_signals=signals,
            wg_ids=ParallelList(rids, [batch.request_to_worker_graph[r] for r in rids]),
            walks=walks if rid_walk_idx is not None else None, rid_walk_idx=rid_walk_idx,
            stops=stops, runtime_stops=runtime_stops, tokens=tokens,
            row_rids=host_rows.request_ids if host_rows is not None else (),
            inline=inline,
            # the consuming pass (not the earlier ingest) reports the partition
            # done, so it rides the WGD that also counts this pass's last chunks
            partition_done=list(batch_N.node_batch.partition_done_rids),
            python_send=self.enable_prof,
            loop_strides=batch.loop_strides(rids),
        )

    def _check_stops(
        self, batch_N: PendingBatch, outputs: BatchedModelOutput, cpu_outputs,
        host_rows: "HostRows | None", engine,
    ) -> ModelStops:
        """The model's ``check_stop``, less the ``END_PARTITION`` stops, which
        end their rows' partitions instead (``partition_done_rids``)."""
        stops = engine.check_stop_for_batch(batch_N.node_batch, cpu_outputs, host_rows=host_rows)
        # Until its last chunk a row's sampled token is not a real sample: a
        # stop would cut the prompt short. Its chunk Loop ends by its own count.
        for rid in batch_N.batch.incomplete_node_rids:
            stops.pop(rid, None)
        ended = []
        # Only a row outside any loop can end its pass here: inside one it
        # would stop only the innermost loop, and a speculated next iteration
        # would carry no report of it. Returning it from a loop is a model
        # bug, so the client sees an internal error. A row whose innermost
        # loop is its own chunk Loop is on its last chunk (above), where that
        # Loop ends by its count, so it counts as outside one.
        for rid in [rid for rid, s in stops.items() if END_PARTITION in s]:
            walk = batch_N.batch.walk_of(rid)
            loop = self._innermost_loop_by_node.get((walk, batch_N.node_name))
            if loop is not None and loop != chunk_loop_name(batch_N.node_name, walk):
                batch_N.node_batch.register_failure(rid, RuntimeError(
                    f"check_stop of node {batch_N.node_name!r} returned END_PARTITION "
                    f"inside a loop of walk {walk!r}; it may end a partition only "
                    "from a row outside any loop"
                ), walk)
            else:
                del stops[rid]
                batch_N.node_batch.partition_done_rids.add(rid)
                ended.append(rid)
        if not batch_N.node_batch.failed_requests:
            return ModelStops(stops, [], ended)
        # A rid whose stop check raised (or was refused above) has no
        # trustworthy stop decision: routing it would either run its loop
        # forever or end it early. Only ever the ones this stop stage just
        # added: the caller already reported (and cleared) the rids that
        # failed in prepare_inputs / their forward / postprocess.
        failed = dict(batch_N.node_batch.failed_requests)
        self._drop_failed_rids(batch_N, outputs, failed)
        self._fail_requests(failed)
        for rid in failed:
            stops.pop(rid, None)
        return ModelStops(stops, list(failed), ended)

    def _declare_stop_policies(self, walks: set[tuple[str, str]]) -> None:
        """Give the graph runtime each local (node, walk)'s declared stop
        policies (``NodeSubmodule.get_stop_policies``), and note the token
        signal each one reads. A walk whose policies read two token signals
        keeps ``check_stop``."""
        # (node, walk) -> the token signal its policies read, or None
        self._stop_token_signal: dict[tuple[str, str], str | None] = {}
        for node, walk in sorted(walks):
            policies = self.engine_manager.get_engine(node).get_stop_policies(node, walk)
            if policies is None:
                continue
            signals = {p.token_signal for p in policies.values() if p.reads_token}
            if len(signals) > 1:
                logger.warning(
                    "%s/%s: stop policies read several token signals %s; check_stop decides",
                    node, walk, sorted(signals),
                )
                continue
            self._graph_runtime.set_stop_policies(node, walk, policies)
            self._stop_token_signal[(node, walk)] = next(iter(signals), None)

    def _buffer_publish(self, batch_N: PendingBatch) -> set[int]:
        """Buffer this step's publication for the conductor; the rids that
        published. CurrentForwardPassInfo also contains publication inherited
        from peer ranks, so only what this worker produced goes."""
        buffered: set[int] = set()
        for rid, published in batch_N.node_batch.resource_publish_info.items():
            if published:
                self.request_state.buffer_publish_info(rid, batch_N.partition, published)
                buffered.add(rid)
        return buffered

    def _store_outputs(
        self, batch_N: PendingBatch, outputs: BatchedModelOutput,
        request: PostprocessRequest, step: PostprocessStep, cpu_outputs,
        host_rows: "HostRows | None" = None, engine=None,
    ) -> tuple[StoredOutputs, RouteTensors]:
        """STORE: store the step's routed rows' outputs, held for the route;
        what the route takes of them."""
        rids = step.wg_ids.keys
        signals = request.output_signals
        graph_walk = request.graph_walk
        # One store call for the batch rather than one per request, and the
        # flat columns come back already built: the manager fills them as it
        # mints, so nothing is keyed by request and signal only to be taken
        # apart again here.
        # Any of the walks names the source's TP group: a combined walk's walks
        # share one (refuse_combined_walks_across_tp_groups).
        # Pass the stop check's host copies so a host-memory transport needn't
        # copy the rows again. They are views of pinned buffers the next step
        # reuses, so they are dropped once this step's sends are done.
        # Row views of one batch clone (a decode step) are described from one
        # row; anything else goes through the general per-tensor path. Both
        # take the real walk, not the combined label: the runtime keys the
        # store and the route on it.
        # The client copy first: whether it rides inline decides whether the
        # tensors are needed at all.
        inline_signal, inline_values = self._inline_emit_values(
            engine, batch_N, host_rows, rids, signals,
        )
        logprobs = self._row_logprobs(host_rows, rids)

        def store(rids):
            stored = None
            if outputs.row_views and outputs.row_request_ids is not None:
                stored = self.tensor_manager.store_row_outputs_batch(
                    rids, outputs, signals, outputs.row_views, outputs.row_request_ids,
                    node_name=batch_N.node_name,
                    graph_walk=graph_walk,
                    host_rows=host_rows,
                )
            if stored is None:
                stored = self.tensor_manager.store_and_return_tensor_info_batch(
                    rids, outputs, signals,
                    node_name=batch_N.node_name,
                    graph_walk=graph_walk,
                    skip_cuda_sync=True,
                    cpu_tensors=(
                        cpu_outputs.landed(signals if self.tensor_manager.needs_cpu_tensor else ())
                        if isinstance(cpu_outputs, _LazyHostOutputs) else cpu_outputs
                    ),
                    ready_event=batch_N.node_batch.completion_event,
                )
            return stored

        loopback = outputs.loopback_rids if inline_signal is not None else ()
        if inline_signal is not None and self._all_on_device(
            engine, batch_N.node_name, batch_N.graph_walk, signals,
        ):
            # Device loop-back: the node keeps every routed signal's value on
            # the device (the sampler's slot master) and the client copy rides
            # inline, so nothing is stored, held or freed. The route still
            # runs per request for the loop counters and the fallbacks; it
            # carries zero tensors per (request, signal).
            stored = StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))
        elif loopback:
            # A mixed step: the rows whose walk loops back on the device carry
            # zero tensors, as they would in a step of their walk alone; only
            # the others are stored.
            routed = [rid for rid in rids if rid not in loopback]
            stored = store(routed) if routed else StoredOutputs([], [], [], [])
            n_sig = len(signals)
            counts, j = [], 0
            for rid in rids:
                if rid in loopback:
                    counts.extend([0] * n_sig)
                else:
                    counts.extend(stored.num_tensors[j:j + n_sig])
                    j += n_sig
            stored = stored._replace(num_tensors=counts)
        else:
            stored = store(rids)
        self.lockstep.note_local_inputs(
            stored, signals, batch_N.node_name, rids,
            request.walks or [graph_walk], step.rid_walk_idx,
        )
        flat_uuids = stored.flat_uuids
        num_tensors = stored.num_tensors
        # Safety hold: ref=1 until the real fanout is known, which
        # complete_and_route_batch settles. One call for the batch rather than
        # one per tensor -- with a Rust bookkeeper each is a boundary crossing,
        # and a 128-request batch has hundreds of them. The count is uniform,
        # so it crosses as a scalar rather than a list built per batch.
        if flat_uuids:
            self.tensor_manager.increment_ref_batch_uniform(flat_uuids, 1)

        return stored, RouteTensors(flat_uuids, num_tensors, inline_signal, inline_values, logprobs)

    def _send_input(
        self, batch_N: PendingBatch, request: PostprocessRequest, step: PostprocessStep,
        stored: StoredOutputs | None, engine,
    ) -> SendInput:
        """SEND: what the send needs of the worker, after the stopped rows'
        finalize, the step's publication and the route's tensor work."""
        rids = step.wg_ids.keys
        signals = request.output_signals
        route_output = step.route_output
        if stored is None:  # routed inline: nothing stored
            stored = StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))
        # Ordinary publication precedes stop detection on the GPU thread.
        # Export final-only state now, after the last iteration committed and
        # before the send reports the completed loop to the conductor. The
        # ordinary publication of a stopping request is finished first, so the
        # completion carries both, as it did when every step published.
        if step.stopped_rids:
            engine.materialize_publish(batch_N.node_batch, step.stopped_rids)
            engine.finalize_stopped_requests(batch_N.node_batch, step.stopped_rids)
        buffered_publish = self._buffer_publish(batch_N)
        # One runtime call rather than a Python loop over the rows (#368).
        self._graph_runtime.update_dynamic_loop_iters(
            batch_N.node_batch.per_request_info, batch_N.partition,
        )
        # Normally empty: cleanup_consumed_inputs took them before the route.
        # Not empty if a completion ever precedes it, and then nobody else will.
        if route_output.freed_inputs.uuids:
            self.tensor_manager.cleanup_collectable(*route_output.freed_inputs)
        # a request whose outputs the transfer engine refused failed alone: none of it is sent
        failed_rids = self._register_outputs(route_output)
        if self.enable_prof:
            self.profile_info.register_end(
                batch_N.node_batch.node_name,
                batch_N.node_batch.graph_walk,
                batch_N.node_batch.request_ids,
                batch_N.node_batch.exec_timings,
            )
        laps = PhaseLaps("worker.postprocess.stream_send.")
        new_token_counts = self._count_new_tokens(
            route_output.new_token_output_idxs,
            stored.flat_rids, stored.flat_uuids, signals, stored.signal_idxs,
        )
        laps.lap("count_tokens")

        # Before the sends below: remote consumers read the walks off the
        # fwd_info that rides with the streamed edges.
        self.streams.assign_consumer_walks(batch_N)

        # The buffer holds each item by its uuid, taking over the routing
        # reference the runtime counted for the local stream destination.
        for signal, per_signal in route_output.local_streaming_by_signal.items():
            consumer = self.streams.walk_driven_edges.get(signal)
            by_rid: dict[int, list[int]] = {}
            for rid, uuid in per_signal:
                by_rid.setdefault(rid, []).append(uuid)
            for rid, uuids in by_rid.items():
                req_info = self.request_state.per_request_info[rid]
                stream_buf = req_info.stream_buffers[signal]
                walks = req_info.per_partition_info[batch_N.partition].current_fwd_info.stream_consumer_walks
                stream_buf.pre_read_register_joined(
                    uuids, walks.get(consumer) if consumer else None,
                )
                for uuid in uuids:
                    stream_buf.put(uuid, uuid)
        laps.lap("local_streams")

        send_rids = [rid for rid in rids if rid not in failed_rids] if failed_rids else list(rids)
        needs_info = route_output.rids_needing_request_info
        info_rids = (
            send_rids if needs_info is None
            else [rid for rid in send_rids if rid in needs_info]
        )
        # A frame to a peer or the conductor carries this step's publication:
        # finish the snapshot for those rids (the stopped ones already are).
        # The snapshot pins the lengths and page counts of step N, so the
        # commit of N+1 that may have landed since does not leak in; only a
        # reset or fork in between moves a stream past its snapshot, and the
        # KV manager publishes such a stream as it stands (its generation
        # says so).
        if info_rids:
            engine.materialize_publish(batch_N.node_batch, info_rids)
            for rid in info_rids:
                if rid in buffered_publish:
                    continue
                published = batch_N.node_batch.resource_publish_info.get(rid)
                if published:
                    self.request_state.buffer_publish_info(
                        rid, batch_N.partition, published,
                    )
                    buffered_publish.add(rid)
        laps.lap("publish")
        # Only the rids with something to report cross: a decode step has no
        # new-token counts (the inline token is not counted) and no stream
        # buffers, and an empty entry per rid cost a dict each to build and to
        # extract on the runtime side. A rid left out gets nothing added,
        # which is what an empty entry did.
        ntc_rids = [rid for rid in send_rids if new_token_counts.get(rid)]
        consumed = self.streams.consumption_batch(send_rids)
        laps.lap("consumption")
        return SendInput(
            completion_id=route_output.completion_id,
            per_request_info=ParallelList(
                info_rids,
                [
                    self.request_state.get_fwd_info(rid, batch_N.partition)
                    for rid in info_rids
                ],
            ),
            new_token_counts=ParallelList(
                ntc_rids, [new_token_counts[rid] for rid in ntc_rids],
            ),
            stream_tokens_consumed=ParallelList(
                [rid for rid, _ in consumed], [c for _, c in consumed],
            ),
            profiling=self._profiling_payloads(send_rids) if self.enable_prof
            else None,
            resource_publish_info=ParallelList(
                info_rids,
                [
                    self.request_state.get_pending_publish_info(
                        rid, batch_N.partition,
                    )
                    for rid in info_rids
                ],
            ),
            failed_rids=failed_rids,
        )

    def _walk_signals(self, batch: ScheduledBatch, walk: str):
        signals = batch.walk_output_signals.get(walk)
        if signals is not None:
            return signals
        return self._graph_runtime.get_output_signals(batch.node_name, walk)

    def _loopback_rids(self, engine, batch: ScheduledBatch) -> frozenset:
        """A mixed step's rows whose own walk keeps every output signal on
        the device. Asked per real walk: the combined label is no walk of the
        submodule's, and would keep none."""
        on_device = {
            walk for walk in set(batch.request_walks.values())
            if self._all_on_device(
                engine, batch.node_name, walk, self._walk_signals(batch, walk),
            )
        }
        if not on_device:
            return frozenset()
        return frozenset(
            rid for rid, walk in batch.request_walks.items() if walk in on_device
        )

    def _all_on_device(self, engine, node_name: str, graph_walk: str, signals) -> bool:
        """Whether every output signal of this step is one the node keeps on
        the device (cached per node and walk: it depends on the node's
        resources, not on the step)."""
        # without inline emit, a rows-only step has no path for the client's tokens
        if not signals or not self._graph_runtime.supports_inline_emit:
            return False
        key = (node_name, graph_walk)
        on_device = self._device_loopback.get(key)
        if on_device is None:
            on_device = self._device_loopback[key] = frozenset(
                engine.device_loopback_signals(node_name, graph_walk)
            )
        return all(s in on_device for s in signals)

    def _inline_emit_values(
        self, engine, batch_N: PendingBatch, host_rows: "HostRows | None",
        rids: list[int], signals,
    ) -> tuple[str | None, list[int]]:
        """The values of a client signal that can ride inline this step.

        Taken off the stop check's host copy, one ``tolist`` for the batch.
        Every rid must have a row, or the step keeps the tensor path for the
        whole batch: the runtime applies the values positionally.
        """
        if host_rows is None or not self._graph_runtime.supports_inline_emit:
            return None, []
        if not batch_N.batch.request_walks:
            return self._inline_emit_one_walk(engine, batch_N, host_rows, rids, signals)
        # Per real walk: a combined walk is no walk of the submodule's.
        walk_of = batch_N.batch.walk_of
        walks = list(batch_N.batch.rids_by_walk()) or [batch_N.graph_walk]
        candidates: dict[str, tuple[str, set[str]]] = {}
        for walk in walks:
            for signal, buffer_name in engine.inline_client_signals(
                batch_N.node_name, walk,
            ).items():
                candidates.setdefault(signal, (buffer_name, set()))[1].add(walk)
        if not candidates:
            return None, []
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        for signal, (buffer_name, inline_walks) in candidates.items():
            if signal not in signals:
                continue
            buf = host_rows.buffers.get(buffer_name)
            if not torch.is_tensor(buf) or buf.dim() == 0 or buf.numel() == 0:
                continue
            values = buf.reshape(buf.shape[0], -1)[:, 0].tolist()
            out = []
            for rid in rids:
                i = row_of.get(rid)
                if i is not None and i < len(values):
                    out.append(values[i])
                elif walk_of(rid) in inline_walks:
                    return None, []  # an inline row with no value: tensor path
                else:
                    out.append(0)  # its walk sends this signal no inline value
            return signal, out
        return None, []

    def _inline_emit_one_walk(
        self, engine, batch_N: PendingBatch, host_rows: "HostRows",
        rids: list[int], signals,
    ) -> tuple[str | None, list[int]]:
        """`_inline_emit_values` for a step of one walk: every row takes the
        value, so one comprehension, and a slice when the host rows are in
        the batch's order (the usual case)."""
        candidates = engine.inline_client_signals(batch_N.node_name, batch_N.graph_walk)
        for signal, buffer_name in candidates.items():
            if signal not in signals:
                continue
            buf = host_rows.buffers.get(buffer_name)
            if not torch.is_tensor(buf) or buf.dim() == 0 or buf.numel() == 0:
                continue
            values = buf.reshape(buf.shape[0], -1)[:, 0].tolist()
            n = len(rids)
            if len(values) >= n and list(host_rows.request_ids[:n]) == rids:
                out = values[:n]
            else:
                row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
                try:
                    out = [values[row_of[rid]] for rid in rids]
                except (KeyError, IndexError):
                    return None, []
            return signal, out
        return None, []

    @staticmethod
    def _row_logprobs(host_rows: "HostRows | None", rids: list[int]) -> list[float | None]:
        """Each rid's sampled-token log-prob off the stop check's host copy,
        None where the sampler left NaN (the row did not ask). Empty when no
        rid has one."""
        buf = None if host_rows is None else host_rows.buffers.get(SAMPLED_LOGPROB)
        if buf is None:
            return []
        values = dict(zip(host_rows.request_ids, buf.tolist(), strict=False))
        out = [None if math.isnan(v := values.get(rid, math.nan)) else v for rid in rids]
        return out if any(v is not None for v in out) else []

    def _stage_host_rows(self, outputs: "BatchedModelOutput") -> None:
        """Start the device-to-host copy of the step's row-addressed stop
        buffers here on the gpu thread, on the execution stream right behind
        the clone that made them, so the completion event recorded next
        covers it. The main thread's stop check then finds the rows landed
        once it has waited on that event: no side stream, no second sync, and
        one fewer GIL release on its critical path.

        A small ring of pinned buffers: step N's rows are read by the main
        thread while this thread already stages N+1's (the pipeline is one
        step deep, the ring leaves slack).

        The step's output spans (``BatchedModelOutput.output_spans``) take the
        same path, so cutting the outputs to them costs no wait of its own.
        """
        buffers = outputs.check_stop_buffers
        rows = bool(buffers) and outputs.row_request_ids is not None
        if not rows and not outputs.output_spans:
            return
        slot = self._host_rows_slot = (self._host_rows_slot + 1) % _HOST_ROWS_RING
        if rows:
            outputs.host_rows = HostRows(
                tuple(outputs.row_request_ids),
                self._copy_to_pinned(buffers, f"host_rows{slot}"),
            )
        if outputs.output_spans:
            outputs.host_spans = self._copy_to_pinned(
                outputs.output_spans, _HOST_SPANS_PURPOSE[slot],
            )

    def _copy_to_pinned(self, buffers: dict, purpose: str) -> dict:
        """Queue a non-blocking copy of each device tensor in ``buffers`` into
        a pinned buffer of ``purpose``; anything else passes through."""
        host: dict[str, torch.Tensor] = {}
        # a tensor under two names is copied once
        landed: dict[int, torch.Tensor] = {}
        for index, (name, tensor) in enumerate(buffers.items()):
            if not (
                torch.is_tensor(tensor)
                and self.device_capability.is_accelerator
                and tensor.device.type == self.device.type
            ):
                host[name] = tensor
                continue
            if id(tensor) in landed:
                host[name] = landed[id(tensor)]
                continue
            buf = self._get_pinned_d2h_buffer(
                purpose, tensor.shape, tensor.dtype, index,
            )
            buf.copy_(tensor, non_blocking=True)
            host[name] = landed[id(tensor)] = buf
        return host

    @staticmethod
    def _crop_to_spans(outputs: "BatchedModelOutput") -> dict[int, list[str]]:
        """Cut each row of a spanned output to its span along the last dim
        (``NodeSubmodule.spanned_outputs``). Called once the step's completion
        event has fired, so the spans staged behind the step are on the host.
        The cut is a view, and it replaces the tensor in its list in place, so
        a second name that ``postprocess`` or ``unpack_packed_outputs`` gave
        the same list is cut with it. Two names the forward returned on its own
        get their own lists from the engine and keep their own widths.

        Returns rid -> the outputs whose span is empty, which the caller leaves
        out of the store as it does a withheld output.
        """
        spans = outputs.host_spans or outputs.output_spans
        per_rid = outputs.per_rid_outputs
        cut = False
        empty: dict[int, list[str]] = {}
        for name, span in spans.items():
            # columns, not a [start, end] list per row
            starts, ends = span.t().tolist()
            for rid, start, end in zip(outputs.span_request_ids, starts, ends, strict=True):
                row = per_rid.get(rid)
                tensors = row.get(name) if row is not None else None
                if not tensors:
                    continue
                if end <= start:
                    empty.setdefault(rid, []).append(name)
                for i, tensor in enumerate(tensors):
                    if start > 0 or end < tensor.shape[-1]:
                        tensors[i] = tensor[..., max(start, 0):max(end, 0)]
                        cut = True
        if cut:
            # the views no longer describe the stored rows
            outputs.row_views = None
        return empty

    def _get_pinned_d2h_buffer(
        self,
        purpose: str,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        index: int = 0,
    ) -> torch.Tensor:
        numel = torch.Size(shape).numel()
        key = (purpose, dtype, self._pinned_size(numel))
        buffers = self._pinned_d2h_buffers[key]
        while len(buffers) <= index:
            buffers.append(
                torch.empty(key[2], dtype=dtype, device="cpu", pin_memory=True)
            )
        return buffers[index][:numel].view(shape)

    @staticmethod
    def _pinned_size(numel: int) -> int:
        return 1 << max(numel - 1, 0).bit_length()


    def _d2h_batched(
        self, buffers: dict, side: torch.Stream,
    ) -> dict[str, torch.Tensor]:
        """One device-to-host copy per named buffer, landed before this returns.

        Replaces a copy per tensor per request (48 at decode batch 16). Row i
        belongs to the i-th request in forward order
        (``BatchedModelOutput.row_request_ids``), not the batch's current
        request list, which has already dropped requests stopped a step ago.
        Padded rows past the real ones are not read. The host buffers are
        reused by the next step.
        """
        host: dict[str, torch.Tensor] = {}
        # a tensor under two names is copied once
        landed: dict[int, torch.Tensor] = {}
        with side:
            for index, (name, tensor) in enumerate(buffers.items()):
                if not (
                    torch.is_tensor(tensor)
                    and tensor.device.type == self.device.type
                ):
                    host[name] = tensor
                    continue
                if id(tensor) in landed:
                    host[name] = landed[id(tensor)]
                    continue
                buf = self._get_pinned_d2h_buffer(
                    "check_stop_batched", tensor.shape, tensor.dtype, index,
                )
                buf.copy_(tensor, non_blocking=True)
                host[name] = landed[id(tensor)] = buf
        side.synchronize()
        return host

    @staticmethod
    def _rows_to_per_rid(
        host: dict, request_ids: list[int],
    ) -> dict[int, NameToTensorList]:
        """Slice row-addressed buffers into the per-rid form ``check_stop``
        reads: row i goes to ``request_ids[i]``."""
        # One ``split`` per buffer makes every row view in a single call.
        rows = {
            name: buf.split(1)
            for name, buf in host.items()
            if torch.is_tensor(buf) and buf.shape
        }
        out: dict[int, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            out[rid] = {
                name: [views[i]] for name, views in rows.items() if i < len(views)
            }
        return out

    def _prematerialize_for_check_stop(
        self,
        outputs: "BatchedModelOutput",
        completion_event: torch.Event | None,
        request_ids: list[int] | None = None,
    ) -> tuple[Mapping[int, Mapping[str, list[torch.Tensor]]], HostRows | None]:
        """Side-stream D→H of the CUDA tensors in ``outputs`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns per-rid host outputs, plus ``HostRows`` when the submodule
        handed over row-addressed buffers, for a batched stop check. Without
        those, the outputs are a ``_LazyHostOutputs``, which copies an output
        name for every rid the first time it is read. Skipped (returns
        ``outputs`` unchanged) when there's no completion event (CPU execution)
        or when CUDA is unavailable.

        The host tensors are views of pinned buffers the next step reuses. Their
        one other reader, a ``needs_cpu_tensor`` transport, gets them through
        the store and sends in ``_register_outputs`` later in the same
        ``_postprocess_batch``, which then drops them from the store.
        """
        source = outputs.get_check_stop_input()
        # Rows are in forward order, which the engine stamped on the output.
        row_rids = (
            list(outputs.row_request_ids)
            if outputs.row_request_ids is not None else request_ids
        )
        landed = outputs.host_rows
        if landed is not None and completion_event is not None:
            # staged by the gpu thread behind the step; the caller has waited
            # on the completion event, so the rows are here
            return _LazyPerRid(landed.buffers, list(landed.request_ids)), landed
        if (
            completion_event is None
            or not self.device_capability.is_available
        ):
            if outputs.check_stop_buffers is not None and row_rids is not None:
                # host tensors already (a CPU device): only the re-keying
                host = outputs.check_stop_buffers
                return (
                    Worker._rows_to_per_rid(host, row_rids),
                    HostRows(tuple(row_rids), host),
                )
            return source, None
        if not source:
            return source, None

        if self._d2h_stream is None:
            self._d2h_stream = torch.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(completion_event)

        if outputs.check_stop_buffers is not None and row_rids is not None:
            host = self._d2h_batched(outputs.check_stop_buffers, side)
            return (
                _LazyPerRid(host, row_rids),
                HostRows(tuple(row_rids), host),
            )

        # Shared by every name copied this step, so two names never land in
        # one pinned buffer.
        buffer_indices: dict[tuple[str, torch.dtype, int], int] = defaultdict(int)

        def copy_names(names: list[str]) -> dict[str, dict]:
            host: dict[str, dict] = {name: {} for name in names}
            with side:
                for name in names:
                    for rid, name_to_list in source.items():
                        if not isinstance(name_to_list, dict) or name not in name_to_list:
                            continue
                        tensors = name_to_list[name]
                        if not isinstance(tensors, list):
                            host[name][rid] = tensors
                            continue
                        new_list = []
                        for t in tensors:
                            if torch.is_tensor(t) and t.device.type == self.device.type:
                                # by size class, not shape: a per-shape count gives
                                # two shapes in one class the same buffer
                                key = ("check_stop", t.dtype, self._pinned_size(t.numel()))
                                idx = buffer_indices[key]
                                buffer_indices[key] += 1
                                cpu_t = self._get_pinned_d2h_buffer(
                                    "check_stop", t.shape, t.dtype, idx,
                                )
                                cpu_t.copy_(t, non_blocking=True)
                                new_list.append(cpu_t)
                            else:
                                new_list.append(t)
                        host[name][rid] = new_list
            side.synchronize()
            return host

        return _LazyHostOutputs(source, copy_names), None

    def _drop_failed_rids(
        self, pending: PendingBatch,
        outputs: BatchedModelOutput,
        failed_requests: Collection[int],
    ) -> None:
        """Excise ``failed_requests`` (rids, or a rid -> error map) from a finished batch.

        A rid that raised in its per-request forward or in ``postprocess`` is
        still carried in the batch (the engine only recorded the error), and one
        that raised in ``prepare_inputs`` is already out of
        ``node_batch.request_ids`` but not out of the worker-side
        ``ScheduledBatch``. Either way we must not route its outputs or mark its
        node complete — that's how a request that blew up mid-walk ends up
        reported to the client as a successful empty response.
        ``_postprocess_batch`` reconciles the remaining structures from
        ``node_batch.request_ids``.
        """
        for rid in failed_requests:
            outputs.pop(rid, None)
            pending.batch.discard_rid(rid)
            pending.node_batch.per_request_info.pop(rid, None)
            pending.node_batch.cached_rids.discard(rid)
            # Clear what we just reported, so a later stage that fails more
            # rids (check_stop, below the forward) can tell its own from these
            # and doesn't report them to the conductor twice.
            pending.node_batch.failed_requests.pop(rid, None)
            # its streams do not end here; see ExecutingBatch.drop_rids
            pending.node_batch.final_stream_rids.discard(rid)
            pending.node_batch.partition_done_rids.discard(rid)
        pending.node_batch.request_ids = [
            rid for rid in pending.node_batch.request_ids
            if rid not in failed_requests
        ]

    def _handle_main_loop_error(
        self,
        exc: Exception,
        in_flight: "tuple[PendingBatch | None, ...]",
        batch: ScheduledBatch | None,
        speculation: "Speculation | None" = None,
    ) -> None:
        """Fail everything the crashed iteration touched and drain its futures.

        Attribution is batch-granular here: a raise out of the batch build or
        output routing can't be pinned on one request (the
        stages that *can* attribute report through
        ``ExecutingBatch.failed_requests`` instead), so every rid this iteration
        touched fails together. Sequential retry of the batch — see the design
        discussion on #123 — would go here; today a batch-level crash is
        terminal for its rids.

        The caller still has to clear its own in-flight state; see the main
        loop's handler.
        """
        # Report the leaf exception to the client and keep the full chain in
        # the log. A forward's own raise doesn't come here (``_step_result``).
        err, _ = client_error_text(
            exc, f"Worker {self.worker_id} error in main loop", log=logger,
        )

        # the chain's rows: a side lane's step fails on its own (``SideStepLane``)
        failed_rids: set[int] = set(self._rows_by_lane.get(CHAIN_LANE, ()))
        if speculation is not None and not speculation.submitted:
            # a follower's build holds the head's fresh rows with the chain's, and they go back as on rank 0
            failed_rids -= speculation.scheduled_batch.request_to_worker_graph.keys() - speculation.continuing_rids
        for stale in in_flight:
            if stale is None:
                continue
            failed_rids.update(stale.batch.request_to_worker_graph)
            self._clear_in_flight_flag(stale.batch)
            # Drain before dropping the reference: the future owns engine state
            # on the GPU thread, and an abandoned one leaves that thread writing
            # into a batch nobody will collect. Already-finished futures return
            # at once.
            if stale.future is None:
                continue
            try:
                stale.future.result()
            except Exception:
                logger.debug(
                    "Worker %s discarding failed batch for node %s",
                    self.worker_id, stale.node_name,
                )
        if batch is not None:
            failed_rids.update(batch.request_to_worker_graph)
            self._clear_in_flight_flag(batch)
        if speculation is not None and not speculation.submitted:
            # Built but never submitted. Its fresh rids came off their queues
            # (a yield-away step is all fresh rids), so they go back or nothing
            # would schedule them again; the continuing ones belong to the
            # pending batch and fail with it. Keyed on submission rather than
            # on an armed pre-plan: the plan future is None without pre-plan
            # and after a refusal reset it, and the rids are stranded either way.
            if speculation.plan_future is not None:
                # The plan thread may still be staging a pre-plan for a step
                # that will now never run. Drain it, then drop the stage, or
                # the next step to lease that slot would find it.
                try:
                    speculation.plan_future.result()
                except Exception:
                    logger.debug(
                        "Worker %s discarding the pre-plan of the failed iteration",
                        self.worker_id,
                    )
                speculation.plan_future = None
                self._reset_skip_plan_flags(speculation.node_batch)
            sb = speculation.scheduled_batch
            # This step never ran, so its staged chunks go back to their
            # buffers -- in particular for the fresh rids pushed back below,
            # which survive this error and would otherwise hold a chunk in a
            # slot the next ingest can overtake.
            self._settle_speculation(speculation, success=False)
            self._clear_in_flight_flag(sb)
            fresh = {
                rid: wg_id
                for rid, wg_id in sb.graph_rows().items()
                if rid not in speculation.continuing_rids
                and rid not in failed_rids
            }
            if fresh:
                self._graph_runtime.push_back_node(
                    sb.node_name, list(fresh), list(fresh.values()),
                )

        self._fail_requests({rid: f"Error in worker: {err}" for rid in failed_rids})

    def _build_speculation(
        self, pending: PendingBatch | None, consecutive: int,
        fair: "_SpecFairness",
    ) -> tuple[Speculation | None, tuple[str, str] | None]:
        """Speculate the step after ``pending`` (N+1 from N), or a
        yield-away batch when fairness says so. Returns the speculation
        and the (node, walk) to exclude on the non-speculative path."""
        speculation = None
        yield_away_from_target = None
        batch = None
        # Build nothing on a refusal; the caller's admit fence is what
        # makes it knowable here. Ordinary speculation would be dropped anyway,
        # but a yield-away batch survives ``_maybe_clear_spec`` and is
        # already broadcast — so the eviction below would land after its
        # head went out, and the follower would admit it a delta behind.
        # Skipping keeps that eviction in the gap between steps.
        #
        # The marker below stays outside this: a follower settles N on
        # exactly one of {head, marker}, and it moves no pages.
        admit_refused = (
            pending is not None
            and pending.node_batch.admit_error is not None
        )

        if pending is not None and self._can_speculate(pending.batch):
            # Fairness check (peek-based, replaces the old iter-
            # counter cap): only break the spec chain when there's
            # another (node, walk) actually ready to schedule on
            # this worker. On single-walk workers (Orpheus LLM,
            # Orpheus SNAC) this returns False and we always speculate.
            # Keyed on the walk the chain speculates into, not the step's:
            # after a mixed step that is the combined walk, which no row has,
            # so the chain's own parked rows would count as other work.
            target = self._choose_spec_target(pending.batch)
            chain_key = (
                (target.info.node_name, target.info.graph_walk) if target is not None
                else (pending.node_name, pending.graph_walk)
            )
            # A prompt chain yields from its first step: speculating its next
            # chunk must not keep decode rows waiting a second prompt step. A
            # walk with no token budget runs its prompts whole and keeps
            # main's order.
            other_ready = (
                fair.peek
                and (
                    consecutive >= 1
                    or (chain_key in self.chunk_loops.speculated and self.scheduler.chunks(*chain_key))
                )
                and self.scheduler.has_ready_excluding(chain_key)
            )
            must_yield_away = consecutive >= fair.max_consecutive or other_ready
            if target is not None and not must_yield_away and not admit_refused:
                if self.enable_nvtx:
                    range_push("worker.speculate", synchronize=False)
                _t0 = _time.perf_counter() if self.timer.period else 0.0
                speculation = self._try_speculate_next(pending, target)
                if self.timer.period:
                    self.timer.record("speculate", _time.perf_counter() - _t0)
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                if speculation is not None and not speculation.is_yield_away:
                    # Broadcast the head now, during forward N, so
                    # followers build it during theirs (-1: not parallel).
                    speculation.tp_seq = self.lockstep.broadcast(
                        speculation.node_batch,
                        speculative=True, spec_from_seq=pending.tp_seq,
                    )
            self.lockstep.maybe_broadcast_nospec(pending, speculation)
            # ``yield_away_from_target`` stays None when the admit
            # refused, so the non-speculative path below schedules
            # without the fairness exclusion — it should be free to pick
            # up the batch ``_handle_admit_failure`` just pushed back.
            if speculation is None and not admit_refused:
                if (
                    not must_yield_away and fair.peek and consecutive < 1
                    and self.scheduler.chunks(pending.node_name, pending.graph_walk)
                ):
                    # The chain could not continue, so there is no chain to
                    # keep going: other ready work goes first, as it would
                    # after the chain's next step. Without this, the step after
                    # a prompt step whose prompts all finished (or whose rows
                    # mid-prefill are not speculated, `speculate_mid_prefill`)
                    # starts another prompt while decode rows wait out both.
                    # Only after a step under a token budget: elsewhere a
                    # prompt runs whole, and main's order stands.
                    must_yield_away = self.scheduler.has_ready_excluding(chain_key)
                yield_away_from_target = chain_key if must_yield_away else None
                with self.timer.span("worker.schedule_yield_away"):
                    batch = self.scheduler.next_batch(exclude_target=yield_away_from_target)
                if batch is not None:
                    speculation = self._yield_away_speculation(batch)
            if speculation is not None and speculation.is_yield_away:
                # A leader stamps the seq it sends; a follower's
                # batch keeps the one it came off the FIFO with.
                ya_seq = self.lockstep.broadcast(speculation.node_batch)
                speculation.tp_seq = ya_seq if ya_seq >= 0 else speculation.scheduled_batch.tp_seq

        return speculation, yield_away_from_target

    def _yield_away_speculation(
        self, batch: ScheduledBatch, gave_way: bool = False,
    ) -> Speculation:
        """A step independent of the one in flight, built now so it overlaps
        it; broadcast by the caller once N's marker is out."""
        node_batch = self._build_executing_batch(batch)
        logger.debug("Yield away: %s %s", batch.node_name, node_batch.request_ids)
        return Speculation(
            scheduled_batch=batch,
            node_batch=node_batch,
            consumed_edges=set(),
            continuing_rids=set(),  # n/a
            partition=self.request_state.get_partition_for_node(batch.node_name),
            is_new_iter=False,
            is_same_node=False,
            is_yield_away=True,
            gave_way=gave_way,
        )

    def run(self) -> None:
        switch_interval = os.environ.get("MSTAR_PY_SWITCH_INTERVAL_SEC", "")
        if switch_interval:
            try:
                sys.setswitchinterval(float(switch_interval))
                logger.info(
                    "Worker %s: Python thread switch interval set to %ss",
                    self.worker_id,
                    switch_interval,
                )
            except ValueError:
                logger.warning(
                    "Worker %s: ignoring invalid MSTAR_PY_SWITCH_INTERVAL_SEC=%r",
                    self.worker_id,
                    switch_interval,
                )

        # Bound the load-time asymmetry between workers before any
        # subgroup NCCL collective fires inside the per-bs CUDA-graph
        # capture loop. Without this fence, a worker with a small model
        # (e.g. an 8B Talker) can finish loading, enter warmup, and hit
        # its first subgroup barrier while a worker with a 30B Thinker
        # is still streaming safetensors shards. The subgroup NCCL comm
        # is created lazily on that first collective; its connect-retry
        # budget is ~33 s, which is shorter than the load-time delta on
        # large multi-tower models. Syncing here means every worker
        # reaches warmup at the same wall-clock instant, so subgroup
        # bootstrap completes within the retry budget.
        self.parallel_groups.barrier_all()
        self.lockstep.verify_async_sched_agrees()

        # CUDA graph capture before entering the main loop
        self.engine_manager.warmup_all()

        # Sync every worker before the main loop opens. Per-batch-size
        # captures inside CudaGraphRunner are already barriered on the
        # node-local TP group, but that doesn't bound the time between
        # ``warmup_and_capture`` returning and ``run()`` starting to
        # schedule. Without this fence, a TP leader can finish warmup
        # quickly, schedule its first batch, and ZMQ-send
        # ``ScheduleTPNode`` to a follower that's still inside another
        # engine's ``warmup``. The follower can't service the message
        # yet, but the leader will sit on the first NCCL collective.
        self.parallel_groups.barrier_all()
        self.parallel_groups.restore_default_timeout()

        # Everything tracked right now — weights, capture buffers, wrapper
        # state — lives for the process, so gen2 gains nothing by walking it
        # every cycle. Collect first so nothing garbage gets made permanent,
        # then move the rest out of GC's reach. Refcounting still frees these,
        # and objects created after this stay fully collected; only a cycle
        # alive at this instant would now be retained.
        gc.collect()
        gc.freeze()
        logger.info(
            "Worker %s: gc.freeze() after warmup — %d objects moved to the "
            "permanent generation", self.worker_id, gc.get_freeze_count(),
        )
        if gen0 := knobs.gc_gen0_threshold():
            gc.set_threshold(gen0, *gc.get_threshold()[1:])
            logger.info("Worker %s: gc thresholds %s", self.worker_id, gc.get_threshold())
        self.timer.track_gc()

        # Setup (weight load + warmup + CUDA-graph capture) is complete. Tell
        # the conductor this worker is ready. The conductor blocks its main
        # loop until every worker reports in, so the API server only advertises
        # readiness once all workers can actually serve.
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.SETUP_DONE,
                body=SetupDone(worker_id=self.worker_id),
            ),
        )

        # The main thread hands every step to the GPU and plan threads and
        # back, so all three move to the step CPUs (each executor thread in
        # its initializer).
        self._pin_step_thread()

        # The async worker path needs decode submission to return quickly so
        # the main loop can overlap queue/tensor polling and post-processing
        # with GPU execution. Run the engine unconditionally on a dedicated
        # 1-worker GPU thread.
        gpu_executor = self._engine_thread_pool("mstar-gpu")
        logger.info(
            "Worker %s: engine runs on dedicated GPU thread",
            self.worker_id,
        )
        # Dedicated thread that pre-plans FlashInfer attention for the
        # speculatively-built next batch. Runs concurrent with main thread's
        # await_gpu (which releases the GIL), so plan()'s Python work isn't
        # contended by main thread's fast/slow postprocess
        #
        # With double-buffered wrappers (MSTAR_NUM_SLOTS=2) and
        # advance_event signaling, plan(N+1) runs concurrent with replay(N)
        # on the disjoint slot — the actual GPU overlap. plan_executor waits
        # on prev_advance_event (signaled right after advance_seq_lens(N) on
        # the GPU thread, ~tens of µs into replay)
        #
        # Default ON; `knobs: {pre_plan: false}` or MSTAR_PRE_PLAN_SPEC=0
        # falls back to the double-buffer-without-pre-plan baseline.
        pre_plan_spec = knobs.pre_plan_enabled()
        launch_wait_s = self._launch_wait_s
        plan_executor = None
        if pre_plan_spec:
            plan_executor = self._engine_thread_pool("mstar-plan")
            logger.info(
                "Worker %s: plan_executor enabled — speculative plan() "
                "pre-runs on a dedicated thread",
                self.worker_id,
            )
        # In-flight: (batch, node_batch, batch_partition, future) | None.
        pending: PendingBatch | None = None

        # MSTAR_SPEC_PEEK_FOR_FAIRNESS=1: only break the spec chain when
        # MicroScheduler.has_ready_excluding finds another (node, walk)
        # ready RIGHT NOW. Single-walk workers always speculate; multi-walk
        # workers yield only when there's actual contention.
        spec_peek_for_fairness = (
            os.environ.get("MSTAR_SPEC_PEEK_FOR_FAIRNESS", "1") == "1"
        )
        fair = _SpecFairness(
            peek=spec_peek_for_fairness, max_consecutive=self._max_consecutive_spec,
        )
        consecutive_spec_steps = 0
        yield_away_from_target: tuple[str, str] | None = None

        side_steps = self._side_steps

        def _arm_speculation(spec: Speculation) -> None:
            # Hand N+1 to the plan thread NOW. It waits on N's
            # commit event, then reserves a slot and pre-plans —
            # while the main thread sits in await_gpu with the GIL
            # released and N's kernels are still running.
            if plan_executor is not None and self._plans_ahead(spec):
                spec.plan_future = plan_executor.submit(
                    self._preplan_spec, pending, spec,
                )

        def _set_pending(p: PendingBatch):
            nonlocal pending
            pending = p
            self._hold_rows(CHAIN_LANE, p.batch.request_to_worker_graph if p else ())

        timer = self.timer
        phase_period = timer.period

        # None with metrics off: then nothing below records or sends anything
        metrics = self._metrics

        # Reset per iteration (not just where they're first used) so the
        # error handler below sees only this iteration's work — a stale
        # ``batch`` from a previous pass would otherwise be failed twice, and
        # a raise before the first assignment would hit UnboundLocalError
        # inside the handler itself.
        batch: ScheduledBatch | None = None
        spec_pending: PendingBatch | None = None
        speculation: Speculation | None = None

        while True:
            from mstar.utils.profiler import range_pop, range_push
            # The step whose decision on the next one (a head from it, or its
            # marker) ``_build_speculation`` settled, for a TP-async lead step.
            # The error handler owes the followers a marker for any other lead
            # step it drops.
            decided: PendingBatch | None = None
            # The seq of a lockstep step this leader broadcast and has not
            # submitted: the followers run it unless N's verdict voids it on
            # every rank, so a raise that drops it must stop this rank (see the
            # handler below). A follower's own raises are not covered.
            committed: int | None = None
            try:
                batch = None
                spec_pending = None
                _iter_start = _time.perf_counter() if phase_period else 0.0
                _iter_cpu = _time.thread_time() if phase_period else 0.0
                laps = PhaseLaps("worker.main.")
                # Everything below moves pages — the removes and drains directly,
                # ``_process_messages`` through a follower's delta, the readiness
                # scans through ``check_ready``'s reload — and none of it may run
                # while N's admit is outstanding. See ``_await_admit_settled``.
                self._await_admit_settled(pending)
                self._count_turn(pending)
                laps.skip()
                # removals held for the journal would otherwise wait out a chain of another node's steps
                self.lockstep.flush_moves(pending)
                self._apply_removes(self._in_flight_rids)
                # Requests a resource refused for good since last pass, in a
                # readiness scan or in a step's admit (``Admission.requeue``).
                # They fail here, one pass after the refusal.
                self._fail_requests(self.admission.take_admit_errors())
                self._apply_drains(self._in_flight_rids)
                self._preemption.make_room()
                laps.lap("teardown")

                # 1. CPU preamble — overlaps with GPU(N).
                # synchronize=False on every range so torch.cuda.synchronize()
                # doesn't drain the in-flight GPU work and undo the overlap.
                if self.enable_nvtx:
                    range_push("worker.process_messages", synchronize=False)
                self.ingest.process_messages()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                laps.lap("process_messages")

                if self.enable_nvtx:
                    range_push("worker.check_ready_tensors", synchronize=False)
                self.ingest.check_ready_tensors()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                laps.lap("check_ready_tensors")

                if self.enable_nvtx:
                    range_push("worker.poll_stream_buffers", synchronize=False)
                self.ingest.poll_stream_buffers()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                laps.lap("poll_stream_buffers")

                # What the preamble recorded, for requests nothing is using.
                self._report_failures()
                self._teardown(self._in_flight_rids)
                laps.lap("ingest_teardown")

                if side_steps is not None:
                    if pending is None:
                        side_steps.finish()
                    side_steps.launch(gpu_executor)
                    laps.lap("side_steps")

                # Reads only: whether a rank sends this changes nothing it runs.
                if metrics is not None:
                    metrics.maybe_report()
                    laps.lap("metrics")

                # 2. Speculatively schedule + build N+1 — overlaps with GPU(N).
                # Only when (a) there's a pending step and (b) it's AR-engine.
                # For non-AR or non-loop-body steps, falls through to the
                # non-speculative path below (drain, then schedule).
                speculation, yield_away_from_target = self._build_speculation(
                    pending, consecutive_spec_steps, fair,
                )
                decided = pending
                if speculation is not None and speculation.tp_seq >= 0 and not self.is_tp_follower:
                    committed = speculation.tp_seq
                # inclusive of `speculate` and `worker.schedule_yield_away`
                laps.lap("build_speculation")
                if speculation is not None:
                    _arm_speculation(speculation)
                    laps.lap("arm_speculation")

                # 3. If pending: await GPU(N), submit speculated GPU(N+1)
                # asap, then post-process N (fast then slow) overlapping
                # with GPU(N+1).
                spec_pending = None
                if pending is not None:
                    outputs: BatchedModelOutput | None = None
                    if speculation is None and self._is_tp_follow_pending(pending):
                        # Follower: watch for the head while N runs and settle
                        # the leader's decision before N is post-processed.
                        if self.enable_nvtx:
                            range_push("worker.follow_await", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        outputs, speculation = self._await_tp_follow_step(
                            pending, _arm_speculation,
                        )
                        if phase_period:
                            timer.record("follow_await", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)

                    if outputs is None:
                        if self.enable_nvtx:
                            range_push("worker.await_gpu", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        outputs = self._step_result(pending)
                        if phase_period:
                            timer.record("await_gpu", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)

                    if speculation is not None and not speculation.is_yield_away and self._step_voids_head(pending):
                        committed = None  # when the verdict (a raise, say) is on every rank, every rank voids the head
                    laps.skip()
                    # set node._in_flight to false, since
                    # the node has just completed
                    self._clear_in_flight_flag(pending.batch)
                    laps.lap("clear_in_flight")

                    def _maybe_clear_spec():
                        nonlocal speculation
                        # Speculation cleanup splits by kind:
                        #
                        # * Non-yield-away spec depended on pending's outputs
                        #   (the plan thread already threaded them in). Pending's
                        #   output is invalid, so the spec batch can't run.
                        #
                        # * Yield-away spec is independent of pending.
                        #   But ``_handle_admit_failure`` may have shifted
                        #   the engine's KV-cache state (paused/offloaded rids),
                        #   so reset pre-plan.
                        if speculation is not None:
                            if speculation.plan_future is not None:
                                speculation.plan_future.result()
                                self._reset_skip_plan_flags(
                                    speculation.node_batch
                                )
                                speculation.plan_future = None
                            if not speculation.is_yield_away:
                                # The step will not run at all: un-ingest every
                                # staged chunk and hand them all back.
                                sb = speculation.scheduled_batch
                                self._settle_speculation(speculation, success=False)
                                # Fresh rids are not re-readied by N's routing
                                # the way continuing rids are: give them back.
                                rows = sb.graph_rows()
                                fresh = [
                                    rid for rid in rows
                                    if rid not in speculation.continuing_rids
                                    and rows.get(rid) is not None
                                ]
                                self._graph_runtime.push_back_node(
                                    sb.node_name, fresh, [rows[r] for r in fresh],
                                )
                                speculation = None

                    if pending.node_batch.admit_error is not None:
                        # Admit refused pending, so no forward ran.
                        # ``_handle_admit_failure`` pushes the nodes back and on
                        # KV-cache OOM offloads or holds. Clearing the speculation
                        # first drops its pre-plan, releasing pages the eviction
                        # would otherwise go looking for. Nothing is speculated on
                        # a refusal, so in practice there is nothing to clear.
                        self._clear_in_flight_flag(pending.batch)
                        _maybe_clear_spec()
                        self._drop_rows_whose_loop_stopped(pending)
                        self._handle_admit_failure(
                            pending.batch, pending.node_batch,
                            referenced_rids=self._speculated_rids(speculation),
                        )

                    if pending.node_batch.failed_requests:
                        # Drop the speculation: it was built from pending's
                        # rids and may thread outputs that the failed rids
                        # never produced.
                        _maybe_clear_spec()
                        self._fail_step_rows(pending, outputs)

                    if speculation is not None:
                        spec_batch = speculation.scheduled_batch
                        spec_node_batch = speculation.node_batch
                        laps.lap("failure_checks")
                        if not speculation.is_yield_away:
                            self._thread_outputs_to_speculative(speculation, outputs)
                            laps.lap("thread_outputs")
                        # The one settle for a step that runs: marks the batch
                        # speculatively scheduled (so its nodes are not put back
                        # on the ready queue while it executes -- not the dropped
                        # rids, which the threading already pruned), and keeps
                        # the staged ingests except for those dropped rids,
                        # whose chunks are un-ingested and go back to their
                        # buffers. Outside the submit guard below, so an
                        # all-dropped batch is still settled.
                        self._settle_speculation(
                            speculation, success=True,
                            dropped_rids=speculation.dropped,
                        )
                        laps.lap("settle_speculation")

                        if spec_batch.request_to_worker_graph:
                            if self.enable_nvtx:
                                range_push("worker.submit_spec", synchronize=False)
                            _t0 = _time.perf_counter() if phase_period else 0.0
                            # Staleness is checked on the GPU thread, after
                            # the plan future resolves and after prepare; see
                            # _execute_on_gpu_thread.

                            # Hold the main thread off the GIL until the GPU
                            # thread reaches the forward launch; see exec.
                            spec_pending = self._submit_step(
                                gpu_executor, spec_batch, spec_node_batch,
                                plan_future=speculation.plan_future, speculated=True,
                                partition=speculation.partition,
                                speculative_new_iter=speculation.is_new_iter,
                                loop_name=speculation.loop_name,
                                tp_seq=speculation.tp_seq,
                            )
                            # the GPU thread owns the pre-plan and the rids
                            # from here; see _handle_main_loop_error
                            speculation.plan_future = None
                            speculation.submitted = True
                            committed = None
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                                range_push("worker.gpu_submit_queued", synchronize=False)
                            with self.timer.span("worker.submit_spec.launch_wait"):
                                spec_node_batch.launch_started_event.wait(
                                    timeout=launch_wait_s,
                                )
                            if phase_period:
                                timer.record("submit_spec", _time.perf_counter() - _t0)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                        else:
                            # Every rank drops the same rows, so none runs it.
                            committed = None
                            if speculation.plan_future is not None:
                                # All continuing rids were dropped, so no spec
                                # batch was submitted. Drop the orphaned pre-plan
                                # so the next step to lease that slot doesn't
                                # promote it. Await first: this batch never reaches
                                # the GPU thread, so nothing else joins the future,
                                # and resetting under a running plan races it.
                                speculation.plan_future.result()
                                self._reset_skip_plan_flags(speculation.node_batch)
                                speculation.plan_future = None

                    # Post-process N (routing stage) — runs concurrently with
                    # GPU(N+1) if we submitted one above. Skipped on any admit
                    # failure since the output tensors aren't valid;
                    # ``_handle_admit_failure`` already rehabilitated the
                    # failed rids upstream.
                    laps.skip()
                    if pending.node_batch.admit_error is None:
                        self._finish_step(pending, outputs)
                        laps.skip()

                    # After N, while N+1 runs: the same thread, but not ahead of
                    # the step the chain is about to speculate.
                    if side_steps is not None:
                        side_steps.finish()
                    laps.lap("step_done")
                    # Before anything schedules N's rows again: otherwise a serial
                    # chain would hold a drained rid at every teardown point.
                    in_flight = self._held_beside(CHAIN_LANE)
                    if spec_pending is not None:
                        in_flight.update(spec_pending.batch.request_to_worker_graph)
                    self._apply_drains(in_flight)
                    laps.lap("post_removes")
                    _set_pending(None)
                    if spec_pending is None:
                        # N is done and nothing runs, so tear down what waited behind
                        # N before the next step is built: its admit may need the
                        # pages. On a follower these include what rank 0 stamped with
                        # N, or with a head this rank dropped, which rank 0 tore down
                        # before it sent any later step.
                        self._teardown(self._in_flight_rids)
                    laps.lap("set_pending")

                if spec_pending is not None:
                    if speculation.is_yield_away and not speculation.gave_way:
                        consecutive_spec_steps = 0
                    else:
                        consecutive_spec_steps += 1
                    if phase_period:
                        timer.pass_done(len(spec_pending.batch.request_to_worker_graph), _iter_start, _iter_cpu)
                    _set_pending(spec_pending)
                    continue
                consecutive_spec_steps = 0

                # 4. Non-speculative path: no pending or speculation skipped
                # (e.g., non-AR engine, or loop ended). Run MicroScheduler.
                with self.timer.span("worker.schedule"):
                    batch = self._next_batch(yield_away_from_target)
                if batch is None or batch.node_name not in self.lockstep.led:
                    self.lockstep.flush_moves()
                if batch is None:
                    if side_steps is not None and side_steps.pending is not None:
                        side_steps.finish(block=True)
                        continue
                    self.communicator.wait_for_work(10)
                    continue

                if self.enable_nvtx:
                    range_push("worker.build_node_batch", synchronize=False)
                node_batch = self._build_executing_batch(batch)
                batch_partition = self.request_state.get_partition_for_node(batch.node_name)

                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # Nothing speculated this batch, so prepare it here. The GPU
                # thread then admits, plans and runs it inline; the slot is
                # leased inside exec, once the token count is known.
                # A leader stamps the seq it sends; a follower's batch keeps
                # the one it came off the FIFO with; everyone else -1.
                broadcast_seq = self.lockstep.broadcast(node_batch)
                if broadcast_seq >= 0:
                    committed = broadcast_seq
                fallthrough_tp_seq = broadcast_seq if broadcast_seq >= 0 else batch.tp_seq

                # Unlike the spec submit, this path doesn't hold off the GIL for
                # the launch — but it still has to be fenceable, or the next
                # iteration schedules against an admit that hasn't run.
                step = self._submit_step(
                    gpu_executor, batch, node_batch,
                    partition=batch_partition, tp_seq=fallthrough_tp_seq,
                )
                committed = None
                logger.debug("Scheduling: %s %s", batch.node_name, node_batch.request_ids)
                _set_pending(step)
                # Same accounting as the speculative path above
                if phase_period:
                    timer.pass_done(len(batch.request_to_worker_graph), _iter_start, _iter_cpu)
            except Exception as e:
                self._handle_main_loop_error(e, (pending, spec_pending), batch, speculation)
                if committed is not None:
                    # Failing the step's rows is all this rank can do: the followers
                    # are in its collectives, or will be, and no marker releases
                    # them. Stop, and the conductor's dead-worker check fails the
                    # rest instead of the group hanging.
                    raise SystemExit(
                        f"Worker {self.worker_id}: dropped lockstep step seq {committed} "
                        f"that the followers run ({e!r})"
                    ) from e
                # Follower: a head from a step that raised must not sit at the
                # FIFO front with failed rids.
                if pending is not None and self._is_tp_follow_pending(pending):
                    self.lockstep.close_follow_step(pending)
                # Leader: a dropped step whose decision never went out (the
                # raise came before or in the speculation build, or N+1 was
                # already submitted) leaves the followers awaiting it, and a
                # drain deferred behind it with them.
                else:
                    for dropped in (pending, spec_pending if spec_pending is not pending else None):
                        if dropped is not None and dropped is not decided and self._can_speculate(dropped.batch):
                            self.lockstep.maybe_broadcast_nospec(dropped, None)
                # Clear the in-flight step: its rows are failed and its future
                # drained, and it must not be post-processed again.
                _set_pending(None)
                consecutive_spec_steps = 0
                sleep(0.01)
