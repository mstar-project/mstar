import atexit
import hashlib
import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
from collections import defaultdict, deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
import yaml

from mstar.api_server.request_types import (
    APIServerMessage,
    ConductorStatus,
    RequestComplete,
    RequestFailed,
)
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.conductor.mps import MpsDaemon
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionDefinition,
    PartitionState,
    StreamingConnectionState,
    merge_publish_info,
)
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import GlobalParallelConfig, WorkerParallelGroups
from mstar.engine.resources import KVReqConfig, ResourceReqConfig
from mstar.engine.resources.kv.config import refuse_undeclared_labels, without_layouts
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo
from mstar.model.base import ForwardPassArgs, Model, WorkerGraph
from mstar.observability import catalog
from mstar.observability.recorders import Gauge, MetricSnapshot, merge
from mstar.platforms import apply_process_env_defaults, current_platform
from mstar.profile.format import RxInfo, TxInfo
from mstar.profile.worker import GraphTimings
from mstar.streaming.topology import CLIENT
from mstar.utils import knobs
from mstar.utils.client_errors import client_error_text, scrub_paths
from mstar.utils.exitcode import describe_exitcode
from mstar.utils.gc_freeze import freeze_after_setup
from mstar.utils.generation_kwargs import check_seed
from mstar.utils.ipc_format import (
    ClientStreamInput,
    ConductorMessageType,
    DrainRequest,
    FailRequests,
    InputSignals,
    NewRequest,
    NewRequestConductor,
    ReadsDone,
    RemoveRequest,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.logging_config import quiet_noisy_loggers
from mstar.utils.orphan import exit_when_orphaned
from mstar.utils.profiler import nvtx_enabled, range_pop, range_push
from mstar.validation import validate_deployment
from mstar.validation.resources import streamed_partitions

logger = logging.getLogger(__name__)


class DeadWorkerError(RuntimeError):
    """A worker process exited. Raised out of ``Conductor.run`` so the
    conductor tears the deployment down and exits non-zero."""

    def __init__(self, worker_id: str, pid: int | None, exitcode: int | None):
        super().__init__(
            f"worker {worker_id} (pid {pid}) exited with {describe_exitcode(exitcode)}"
        )
        self.worker_id = worker_id
        self.pid = pid
        self.exitcode = exitcode


def _req_id_to_seed(req_id: str):
    """Map a request id to a 32-bit seed.

    Uses ``hashlib.md5`` rather than Python's builtin ``hash`` so the result
    is **stable across processes**: Python salts ``hash`` per-interpreter via
    ``PYTHONHASHSEED``, which would otherwise make the conductor's per-request
    seed unpredictable from a client process. A deterministic mapping lets a
    client pin ``request_id`` and reproduce the exact noise the server's
    sampler will use, which is essential for noise-controlled debugging
    (e.g. comparing Pi0.5 server output against a reference implementation).
    """
    digest = hashlib.md5(req_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _request_seed(model_kwargs: dict, request_id: str) -> int:
    """The request's seed: its explicit ``seed`` kwarg, else one from its id.

    Raises ``ValueError`` on a bad explicit seed (see ``check_seed``).
    """
    seed = model_kwargs.get("seed")
    if seed is None:
        return _req_id_to_seed(request_id)
    check_seed(seed)
    return seed


def _seed_id(body: NewRequestConductor) -> str:
    """The id a derived seed comes from: the client's, so a pinned request_id
    reproduces the same noise whatever key the API server ran it under."""
    if body.client_request_id is None:
        return body.request_id
    return body.client_request_id


def _pick_free_tcp_port() -> int:
    """Ask the OS for an unused ephemeral TCP port.

    Binds to port 0, reads back the assignment, releases. There is a tiny
    race window between this release and the NCCL TCPStore bind in the
    worker process — small enough in practice for single-host use. The
    point of picking dynamically is to avoid colliding with another
    ``mstar`` instance hard-coded to 29500 on the same host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def keep_bootstrap_sockets_on_loopback() -> None:
    """Have NCCL and Gloo open their sockets on loopback, unless the user
    picked an interface (or, for NCCL, an address family).

    Unset, NCCL listens on a routable interface and Gloo on whatever address
    the hostname resolves to (a pod or container IP), for as long as the
    deployment runs. Every rank is spawned by this conductor on this host, so
    loopback reaches all of them; a launcher that places ranks on other hosts
    must not call this. Data between GPUs on one host goes over NVLink, PCIe
    P2P or SHM either way.
    """
    if "NCCL_SOCKET_IFNAME" not in os.environ and "NCCL_SOCKET_FAMILY" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")


def _exit_when_orphaned(
    worker_id: str, parent=None, poll_s: float = 0.5, before_hard_exit=None,
) -> None:
    """Worker-side watchdog that leaves once the conductor is gone.

    So no worker outlives it holding GPU memory. That includes a worker still
    in setup, which would otherwise wedge in a startup collective waiting for a
    peer that already left. SIGTERM is the graceful path in
    ``_worker_process_target``. The conductor watches the API server the same
    way (``_conductor_process_target``).
    """
    exit_when_orphaned(
        f"Worker {worker_id}", "conductor", signal.SIGTERM,
        parent=parent, poll_s=poll_s, before_hard_exit=before_hard_exit,
    )


def _unlink_owned_files(tensor_manager) -> None:
    """Unlink the files ``tensor_manager`` still owns, if no later launch can
    be writing the same names: under a launch token it names its own, so this
    holds even once the conductor is gone. Without one (a Conductor built
    directly, not under an API server) the names are the endpoint's alone, a
    restart on the same prefix reuses them, and no check of the conductor
    covers the unlinks that follow it (#767); the files stay for a later
    start to sweep or reclaim instead."""
    if tensor_manager.names_unique_per_launch:
        tensor_manager.unlink_owned_files()


def _run_worker(worker) -> None:
    """``worker.run()``, then unlink the tensor files (and arena segments) the
    worker still owns.

    A stop (the conductor's SIGTERM, a Ctrl-C) leaves the files of requests in
    flight waiting for a TENSOR_RECEIVED or REMOVE_REQUEST that will not come,
    in a RAM-backed dir. Then leave with the first stop or error ``run()``
    ended on.
    """
    stop = None
    try:
        worker.run()
    except BaseException as e:
        stop = e
    handlers = None
    while True:
        try:
            # A Ctrl-C reaches the conductor too, and its SIGTERM lands here
            # within milliseconds, often already pending. One that fires before
            # these are ignored is caught below, and the unlinks start over
            # (each file is untracked only once it is gone).
            if handlers is None:
                handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            _unlink_owned_files(worker.tensor_manager)
            break
        except (KeyboardInterrupt, SystemExit) as e:
            stop = stop or e
    # A signal that arrives while ignored is dropped, not queued. Putting the
    # handlers back lets a second stop break a teardown that hangs (a thread
    # join, an atexit hook). One that fires right away must not hide the
    # error run() ended on.
    try:
        for s, h in handlers.items():
            if h is not None:  # None: set outside Python, nothing to put back
                signal.signal(s, h)
    except (KeyboardInterrupt, SystemExit) as e:
        stop = stop or e
    if stop is not None:
        raise stop


def worker_device(device_type: str, rank: int, rank_devices: dict[int, int] | None = None) -> str:
    """The device string a worker rank runs on: its own number unless the
    deployment's ``rank_devices`` maps it elsewhere (several ranks may share a
    GPU); CPU deployments ignore the mapping."""
    if device_type == "cpu":
        return "cpu"
    index = (rank_devices or {}).get(rank, rank)
    return f"{device_type}:{index}"


def _worker_process_target(
    worker_id: str,
    worker_ids: list[str],
    my_worker_graphs: list[WorkerGraph],
    model_config: dict,
    all_worker_graph_ids_to_graph_walks: dict[int, set[str]],
    all_worker_graph_ids_to_nodes: dict[int, set[str]],
    all_worker_graph_ids_to_dyn_loops: dict[int, set[str]],
    sharding_config: ShardingConfig,
    parallel_groups: WorkerParallelGroups,
    hostname: str,
    socket_path_prefix: str,
    dist_init_method: str,
    enable_nvtx: bool = False,
    enable_prof: bool = False,
    model: Model | None = None,
    device: str = "cuda",
    log_level: str = "INFO",
    tensor_comm_protocol=CommProtocol.RDMA,
    tcp_transfer_device="",
    status_interval_s: float | None = None,
    env: dict[str, str] | None = None,
):
    """Top-level target for spawned worker processes. Must be module-level for
    picklability. ``status_interval_s`` turns on the worker's metrics."""
    # before anything in this process touches the device runtime; the MPS pipe
    # directory in ``env`` decides whether this process becomes an MPS client
    apply_process_env_defaults()
    os.environ.update(env or {})
    # SIGTERM (the conductor's p.terminate()) defaults to immediate death:
    # no unwinding, no atexit, no finalizers — so a worker's shared-memory
    # segments were never unlinked and leaked into /dev/shm for the life of
    # the box. Turning it into SystemExit unwinds the interpreter normally,
    # which runs the transport's cleanup (``_run_worker`` for per-tensor
    # files, the arena's Drop for its segments). The main process gets this for
    # free from SIGINT -> KeyboardInterrupt, which is why only the workers
    # leaked.
    def _graceful_exit(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=f"%(asctime)s %(levelname)s [{worker_id}] %(name)s: %(message)s",
        force=True,
    )
    quiet_noisy_loggers()
    built: list = []  # the Worker, once it exists
    threading.Thread(
        target=_exit_when_orphaned, args=(worker_id,), daemon=True, name="parent-watch",
        kwargs={"before_hard_exit": lambda: [_unlink_owned_files(w.tensor_manager) for w in built]},
    ).start()

    from mstar.worker.worker import Worker
    logger.debug("Launching worker %s with graph nodes %s", worker_id, str(
        [set(wg.section.get_nodes()) for wg in my_worker_graphs]
    ))
    try:
        worker = Worker(
            worker_id=worker_id,
            worker_ids=worker_ids,
            model=model,
            my_worker_graphs=my_worker_graphs,
            model_config=model_config,
            all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
            all_worker_graph_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
            sharding_config=sharding_config,
            parallel_groups=parallel_groups,
            hostname=hostname,
            socket_path_prefix=socket_path_prefix,
            dist_init_method=dist_init_method,
            enable_nvtx=enable_nvtx,
            enable_prof=enable_prof,
            device=torch.device(device),
            tensor_comm_protocol=tensor_comm_protocol,
            tcp_transfer_device=tcp_transfer_device,
            status_interval_s=status_interval_s,
        )
    except SystemExit:
        raise  # the graceful exit above (SIGTERM, or the watchdog), not a failure
    except BaseException as e:
        logger.exception("Worker %s failed to initialize: %s", worker_id, str(e))
        raise e
    built.append(worker)
    _run_worker(worker)


@dataclass
class RequestData:
    # Request-level shared state
    persist_signals: dict[str, list[TensorPointerInfo]]  # signals passed back to conductor
    persist_signal_ref_cnt: dict[int, int]  # uuid -> number of times it was passed to workers
    worker_graph_to_workers: dict[int, list[str]]
    all_worker_graph_ids: set[int]
    max_output_tokens: int
    random_seed: int
    # resource label -> the config this request's resources were opened with
    resource_configs: dict[str, ResourceReqConfig]
    sharding_config: ShardingConfig | None = None
    # set once a NewRequest goes out; before that a failed ingest has nothing to drain
    dispatched: bool = False
    # set once request_complete goes out; after that the client must never get
    # request_failed
    completion_sent: bool = False

    # Partition state (always populated — single-partition models use a "default" partition)
    partition_states: dict[str, PartitionState] = field(default_factory=dict)
    partition_definitions: dict[str, PartitionDefinition] = field(default_factory=dict)

    # Per-streaming-connection state, keyed by (from_partition, to_partition, edge_name)
    streaming_connections: dict[tuple[str, str, str], StreamingConnectionState] = field(default_factory=dict)

    # output name -> chunks the workers emitted to the client; the API server
    # holds the request open until it has received this many of each
    output_chunk_counts: dict[str, int] = field(default_factory=dict)

    # Conductor-side profiling timestamps, in ``time.perf_counter()`` seconds.
    # Comparable with the api-server stamps because perf_counter is
    # CLOCK_MONOTONIC (boot-relative) and these processes share a host. 0 until
    # stamped.
    conductor_ingest_time: float = 0.0
    conductor_finish_time: float = 0.0
    # when the conductor took it in (queued, or admitted on arrival); None
    # unless metrics are on
    conductor_arrival_time: float | None = None
    graph_timings: GraphTimings = field(default_factory=dict)
    # Tensor-transport profiling merged across workers. Keyed so repeated
    # WorkerGraphsDone updates (cumulative per worker) replace rather than
    # double-count: rx by (source, dest, edge), tx by (source, edge).
    rx_info: dict[tuple[str, str, str], RxInfo] = field(default_factory=dict)
    tx_info: dict[tuple[str, str], TxInfo] = field(default_factory=dict)

    def remove_persist_signal_uuids(self, uuids: list[int]):
        uuids = set(uuids)
        for name in self.persist_signals:
            self.persist_signals[name] = [
                info for info in self.persist_signals[name] if info.uuid not in uuids
            ]

        for uuid in uuids:
            del self.persist_signal_ref_cnt[uuid]

    def get_incoming_connections(self, partition_name: str) -> list[StreamingConnectionState]:
        """Return all streaming connections where the given partition is the consumer."""
        return [
            conn for conn in self.streaming_connections.values()
            if conn.to_partition == partition_name
        ]


class ClientStream(NamedTuple):
    """Where a client stream's chunks go: the node that consumes the edge, a
    walk it runs in (for routing to its workers) and its partition."""
    node: str
    graph_walk: str
    partition: str


@dataclass
class DrainingRequest:
    """Teardown barrier state: a request whose participants are draining their
    reads. Once every entity in ``expected_acks`` has sent READS_DONE it is safe
    to send the hard RemoveRequest to all ``participants``."""
    expected_acks: set[str]
    participants: set[str]
    # Set for the fail path: the client notification is deferred until the
    # barrier completes (so the API server doesn't tear down mid-read).
    failure_error: str | None = None
    failure_status: int = 500


def _message_name(message) -> str:
    """A message's type as a log or error text shows it: ``FAIL_REQUESTS``,
    not ``ConductorMessageType.FAIL_REQUESTS``."""
    message_type = getattr(message, "message_type", None)
    if message_type is None:
        return type(message).__name__
    return getattr(message_type, "name", str(message_type))


def _message_request_ids(message) -> list[str | None]:
    """The requests a message is about: its ``request_id``, or every request a
    FAIL_REQUESTS body names (``errors`` has no ``request_id``). ``[None]``
    when it names none."""
    body = getattr(message, "body", None)
    errors = getattr(body, "errors", None)
    if isinstance(errors, dict) and errors:
        return list(errors)
    return [getattr(body, "request_id", None)]


class Conductor:
    # Metrics are off unless __init__ gets an interval; class-level so an
    # object built without __init__ reads them as off too.
    _status_interval_s: float | None = None

    def __init__(
        self,
        model: Model,
        model_config_file: str,
        socket_path_prefix: str = "/tmp/mstar",
        hostname: str = "localhost",
        enable_nvtx: bool = False,
        enable_prof: bool = False,
        log_level: str = "INFO",
        tensor_comm_protocol=CommProtocol.RDMA,
        tcp_transfer_device="",
        status_interval_s: float | None = None,
    ):
        self.requests: dict[str, RequestData] = {}
        # Requests in teardown: kept in self.requests (so they still count toward
        # concurrency until their GPU state is freed) with barrier state here.
        self.draining: dict[str, DrainingRequest] = {}
        # Backstop for a participant that never ACKs (crashed, hung, OOMed): the
        # barrier is force-finalized so one faulty worker can't hold a
        # concurrency slot — or the client's failure notification — forever.
        self._drain_ttl_s = float(os.environ.get("MSTAR_DRAIN_TTL_S", "120"))

        # READS_DONE that arrived before the barrier was registered (the
        # preprocess worker self-drains on abort before we process ABORT_REQUEST).
        self._early_reads_done: dict[str, set[str]] = {}
        # Aborts for requests we haven't ingested yet: the preprocess worker
        # forwards ABORT_REQUEST from its abort queue before it finishes
        # preprocessing, so it can outrun the NEW_REQUEST it aborts.
        self._early_abort_requests: set[str] = set()
        # (deadline, rid) FIFOs — expiry sweeps for the three above, so an entry
        # whose awaited message never arrives can't pile up. Deadlines are all
        # the same TTL from insertion, so each deque stays sorted.
        self._draining_deadlines: deque[tuple[float, str]] = deque()
        self._early_reads_done_deadlines: deque[tuple[float, str]] = deque()
        self._early_abort_deadlines: deque[tuple[float, str]] = deque()

        self.model = model
        self.hostname = hostname
        self.socket_path_prefix = socket_path_prefix
        self.log_level = log_level
        self.enable_prof = enable_prof
        self.tensor_comm_protocol = tensor_comm_protocol
        self.tcp_transfer_device = tcp_transfer_device

        self._worker_processes: list[mp.Process] = []
        # A worker that dies sends nothing, so its process handle is the only
        # signal. The startup wait and the main loop poll it on this cadence
        # (see _poll_worker_liveness).
        self._liveness_interval_s = 0.5
        self._next_liveness_check = 0.0
        self.waiting_queue: list[NewRequestConductor] = []

        with open(model_config_file, "r") as f:
            self.model_config = yaml.safe_load(f)
        # fail here, once, rather than in every worker
        knobs.check(self.model_config.get("knobs"))
        # Every check that needs only the model, this YAML and the environment,
        # before any worker exists; raises listing every problem found.
        deployment = validate_deployment(model, self.model_config, model_config_file)
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        self.device_type = accelerator.type if accelerator is not None else "cpu"
        self.enable_nvtx = nvtx_enabled(enable_nvtx, torch.device(self.device_type))
        logger.info("Detected worker device type: %s", self.device_type)
        logger.info("Inductor compile cache: %s", os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
        self.max_concurrent_requests: int = self.model_config.get(
            "max_concurrent_requests", None
        )
        # Optional ``rank_devices: {rank: device_index}`` places a worker rank
        # on a device other than the one its number implies, e.g. two workers
        # sharing one GPU so a light node's steps stop interleaving with a
        # heavy node's on the same worker loop.
        self.rank_devices = {
            int(rank): int(index)
            for rank, index in (self.model_config.get("rank_devices") or {}).items()
        }
        # Optional ``mps: true``: workers run as clients of a CUDA MPS daemon the
        # conductor owns, so workers sharing a GPU overlap their kernels instead
        # of time-slicing. validate_deployment checked the value.
        self._mps_requested = self.model_config.get("mps", False)
        self._mps: MpsDaemon | None = None

        self.default_sharding_config = deployment.sharding_config
        # The conductor is the only process that sees every worker graph, so it
        # owns the numbering. Workers receive their graphs (and these ids) at
        # spawn, so everyone agrees without the ids having to be negotiated.
        worker_graphs = deployment.worker_graphs
        # edge name -> where its chunks go, for each stream the client feeds. After
        # the worker graphs are built: a model may shape them from the deployment (bagel).
        self._client_streams = self._find_client_streams(model)
        for index, worker_graph in enumerate(worker_graphs):
            worker_graph.worker_graph_id = index
        self.worker_graphs = {
            worker_graph.worker_graph_id: worker_graph
            for worker_graph in worker_graphs
        }

        # Set up worker graph TP ranks; validate_deployment checked each worker
        # graph belongs to one sharding group of its tp_size.
        self.streaming_consumers = set()
        self.node_walk_to_wg: dict[tuple[str, str], WorkerGraph] = {}

        # (worker idx) -> {sharding group key: rank within the lockstep instance}
        self.worker_group_to_instance_rank: dict[int, dict[str, int]] = {}

        graph_walks = set()
        streamed_inputs: dict[str, set[str]] = defaultdict(set)
        for wg in self.worker_graphs.values():
            for walk in wg.graph_walks:
                graph_walks.add(walk)
            for name, node in wg.section.get_nodes().items():
                for walk in wg.graph_walks:
                    self.node_walk_to_wg[(name, walk)] = wg
                if node.consumes_stream:
                    self.streaming_consumers.add(name)
                    streamed_inputs[name] |= node._streaming_inputs
        stream_edges = {
            (edge.name, name, walk, edge.next_node)
            for wg in self.worker_graphs.values()
            for name, node in wg.section.get_nodes().items()
            for edge in node.outputs
            if edge.name in streamed_inputs.get(edge.next_node, ())
            for walk in wg.graph_walks
        }
        # a Loop streams its outputs on completion; the producer is the inner
        # node that makes the signal
        stream_edges |= {
            (edge.name, name, walk, edge.next_node)
            for wg in self.worker_graphs.values()
            for loop in wg.section.get_loops().values()
            for edge in loop.outputs + loop.accumulated_outputs
            if edge.name in streamed_inputs.get(edge.next_node, ())
            for name, node in loop.get_nodes().items()
            if any(out.name == edge.name for out in node.outputs)
            for walk in wg.graph_walks
        }
        self.default_sharding_config.assert_streams_keep_tp_size(sorted(stream_edges))

        for group in self.default_sharding_config.groups:
            group_graph_walks = group.graph_walks or graph_walks
            group_key = group.key_str()
            for walk in group_graph_walks:
                for node in group.nodes:
                    if (node, walk) not in self.node_walk_to_wg:
                        continue
                    wg = self.node_walk_to_wg[(node, walk)]
                    # The instance rank is the worker's index within the
                    # lockstep instance (0..tp*sp-1), used by the
                    # replicated-signal fanout (so exactly one rank — index 0
                    # — forwards a replicated tensor to a downstream node).
                    for ranks in wg._instance_ranks:
                        for i, r in enumerate(ranks):
                            self.worker_group_to_instance_rank.setdefault(r, {})[group_key] = i

        # Pick a free TCP port for the NCCL init store. Done once on the
        # conductor and shared with every spawned worker via
        # ``dist_init_method`` so two ``mstar`` instances on the same host
        # don't collide on a hard-coded port.
        self._dist_init_port = _pick_free_tcp_port()
        self._dist_init_method = f"tcp://{hostname}:{self._dist_init_port}"

        # Metrics: with an interval, the run loop sends the API server a
        # ConductorStatus that often (see _maybe_send_status).
        self._status_interval_s = status_interval_s
        self._next_status = 0.0
        # request id -> when it was queued at capacity
        self._queued_since: dict[str, float] = {}
        self._wg_nodes = {
            wg_id: tuple(wg.section.get_nodes()) for wg_id, wg in self.worker_graphs.items()
        }
        self._components = sorted({n for nodes in self._wg_nodes.values() for n in nodes})
        self._requests_gauge = Gauge(catalog.CONDUCTOR_REQUESTS)
        self._in_flight_gauge = Gauge(catalog.COMPONENT_IN_FLIGHT)
        # worker id -> its latest WorkerStatus snapshots (cumulative)
        self._worker_metrics: dict[str, list[MetricSnapshot]] = {}

        os.makedirs(socket_path_prefix, exist_ok=True)
        self._derive_worker_info()
        self._launch_workers()

        self.communicator = make_communicator(
            my_id="conductor",
            push_ids=self.worker_ids + ["api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )

    def _max_output_tokens(self, model_kwargs: dict) -> int:
        """The request's output cap: an int >= 1, else ``ValueError`` (a 400).
        Checked here because the cap is compared inside the run loop, where a
        non-int would raise and the request would hang."""
        requested = model_kwargs.get("max_output_tokens")
        if requested is not None and (
            isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
        ):
            raise ValueError(f"max_output_tokens must be an integer >= 1; got {requested!r}")
        cap = self.model.get_max_output_tokens(**model_kwargs)
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"max_output_tokens must be an integer >= 1; got {cap!r}")
        return cap

    def _loop_caps(self, model_kwargs: dict) -> dict[str, int]:
        """Each named loop's cap for the request. A cap past its loop's ``max_iters``
        would end the request short of what it asked for, so it is a ``ValueError`` (a 400)."""
        caps = self.model.get_loop_caps(**model_kwargs)
        for name, cap in caps.items():
            max_iters = self._loop_max_iters.get(name)
            if max_iters is not None and cap > max_iters:
                raise ValueError(
                    f"this request needs {cap} iterations of {name!r}, and this model runs at most {max_iters}"
                )
        return {name: caps.get(name, n) for name, n in self._loop_max_iters.items()}

    def _resource_specs(self) -> dict:
        """The model's resource specs by key, after this deployment's YAML, for admission-time checks."""
        specs = getattr(self, "_resource_specs_cache", None)
        if specs is None:
            built = self.model.get_node_resources() or []
            apply_yaml_overrides(built, self.model_config)
            specs = {s.resource_key: s for s in built}
            self._resource_specs_cache = specs
        return specs

    def _streamed_partitions(self) -> frozenset[str]:
        streamed = getattr(self, "_streamed_partitions_cache", None)
        if streamed is None:
            streamed = streamed_partitions(list(self._resource_specs().values()), self.model)
            self._streamed_partitions_cache = streamed
        return streamed

    def _fit_loop_caps(self, model_kwargs: dict, configs: dict, cap: int) -> tuple[int, dict[str, int]]:
        """The largest ``max_output_tokens`` up to ``cap``, and its loop caps, that keeps every loop
        within what the request's resources hold. A loop that ``max_output_tokens`` doesn't drive is
        left as it is, and ``validate`` then refuses the request if it doesn't fit."""
        specs = self._resource_specs()
        room: dict[str, int] = {}
        for key, cfg in configs.items():
            for name, n in cfg.loop_room(specs.get(key)).items():
                room[name] = min(n, room.get(name, n))

        def overrun(caps: dict[str, int]) -> int:
            return max((caps[name] - n for name, n in room.items() if name in caps), default=0)

        caps = self._loop_caps({**model_kwargs, "max_output_tokens": cap})
        over = overrun(caps)
        while over > 0 and cap > 1:
            cap = max(1, cap - over)
            caps = self._loop_caps({**model_kwargs, "max_output_tokens": cap})
            over, last = overrun(caps), over
            if over >= last:
                # the loop that overruns is one max_output_tokens doesn't drive
                break
        return cap, caps

    def _get_resource_configs(
        self, model_kwargs: dict,
        partition_fwd_args: dict[str, ForwardPassArgs]
    ) -> dict[str, ResourceReqConfig]:
        """The per-resource config each new request is opened with.

        Resolved once, here, and carried on the request: the worker hands each
        config to its resource at ingest. KV shape is not part of this — that
        is a deployment-wide property the model declares in its resource specs.

        A KV resource with declared labels gets a config even when the model
        returned none, or its keys and layout would have nowhere to go.
        """
        configs = self.model.get_request_resource_configs(
            partition_fwd_args=partition_fwd_args, model_kwargs=model_kwargs
        )
        for key, labels in self._kv_labels.items():
            configs.setdefault(key, KVReqConfig(needed_labels=list(labels)))
        return configs

    def _derive_worker_info(self):
        """Derive per-rank worker info from the worker graphs."""
        # Collect unique ranks and per-rank worker graphs
        rank_to_worker_graphs: dict[int, list[WorkerGraph]] = defaultdict(list)
        for worker_graph in self.worker_graphs.values():
            for rank in worker_graph.ranks:
                rank_to_worker_graphs[rank].append(worker_graph)

        self._sorted_ranks = sorted(rank_to_worker_graphs.keys())
        self.worker_ids = [f"worker_{rank}" for rank in self._sorted_ranks]

        # Messages that arrive before all workers report SETUP_DONE; replayed
        # on the first main-loop iteration. See _wait_for_workers_ready.
        self._startup_message_backlog: list = []

        # Per-worker graph units, engine configs
        self._per_worker_graphs: dict[str, list[WorkerGraph]] = {}

        for rank in self._sorted_ranks:
            worker_id = f"worker_{rank}"
            worker_graphs = rank_to_worker_graphs[rank]
            self._per_worker_graphs[worker_id] = worker_graphs

        # Global maps needed by all workers
        self._all_worker_graph_ids_to_graph_walks: dict[int, set[str]] = {
            worker_graph_id: worker_graph.graph_walks for worker_graph_id, worker_graph in self.worker_graphs.items()
        }
        self._all_worker_graph_ids_to_nodes: dict[int, set[str]] = {
            worker_graph_id: set(worker_graph.section.get_nodes())
            for worker_graph_id, worker_graph in self.worker_graphs.items()
        }
        self._all_worker_graph_ids_to_dyn_loops: dict[int, set[str]] = {
            worker_graph_id: set(worker_graph.section.get_loops())
            for worker_graph_id, worker_graph in self.worker_graphs.items()
        }
        self._loop_max_iters: dict[str, int] = {}
        for worker_graph in self.worker_graphs.values():
            for name, loop in worker_graph.section.get_loops().items():
                self._loop_max_iters[name] = max(loop.max_iters, self._loop_max_iters.get(name, 0))
        self._kv_labels: dict[str, tuple[str, ...]] = {
            key: tuple(labels) for key, labels in self.model.kv_labels().items()
        }

        # set each group's _tp_rank (the worker's instance rank) per worker
        self.per_worker_sharding_config: dict[str, ShardingConfig] = {}
        for i, worker_id in enumerate(self.worker_ids):
            sharding_cfg = self.default_sharding_config.clone_empty()
            for group in sharding_cfg.groups:
                group_key = group.key_str()
                if group_key in self.worker_group_to_instance_rank.get(i, {}):
                    group._tp_rank = self.worker_group_to_instance_rank[i][group_key]
            self.per_worker_sharding_config[worker_id] = sharding_cfg

        self.parallel_config = GlobalParallelConfig(
            worker_graphs=self.worker_graphs,
            worker_ids=self.worker_ids,
            dist_timeout_s=self.model_config.get("dist_timeout_s"),
            tp_allreduce=self.model_config.get("tp_allreduce"),
            tp_allreduce_max_kb=self.model_config.get("tp_allreduce_max_kb"),
        )

    def _launch_workers(self):
        """Spawn one process per worker rank using spawn context."""
        # Set here too. A spawned child imports the model's modules (and their
        # kernel libraries) while unpickling its arguments, before the worker
        # target runs, and it inherits os.environ.
        apply_process_env_defaults()
        keep_bootstrap_sockets_on_loopback()
        # JIT-build the fused-MoE CUDA op once here, for a model that asks: its
        # workers then find it cached instead of racing on the same build lock
        if getattr(self.model, "prebuild_fused_moe", False):
            from mstar.utils.fused_moe.align import _cuda_op_available

            _cuda_op_available()
        ctx = mp.get_context("spawn")
        atexit.register(self.shutdown)
        if self._mps_requested:
            # a host check, not a config one: shipped configs validate on any machine
            if not current_platform.is_cuda():
                raise ValueError(f"mps: true needs an NVIDIA GPU, not {current_platform!r}")
            self._mps = MpsDaemon()
            self._mps.start()
        for rank, worker_id in zip(self._sorted_ranks, self.worker_ids, strict=True):
            p = ctx.Process(
                target=_worker_process_target,
                kwargs={
                    "worker_id": worker_id,
                    "worker_ids": self.worker_ids,
                    "my_worker_graphs": self._per_worker_graphs[worker_id],
                    "model_config": self.model_config,
                    "all_worker_graph_ids_to_graph_walks": self._all_worker_graph_ids_to_graph_walks,
                    "all_worker_graph_ids_to_nodes": self._all_worker_graph_ids_to_nodes,
                    "all_worker_graph_ids_to_dyn_loops": self._all_worker_graph_ids_to_dyn_loops,
                    "sharding_config": self.per_worker_sharding_config[worker_id],
                    "parallel_groups": self.parallel_config.per_worker_config[worker_id],
                    "hostname": self.hostname,
                    "socket_path_prefix": self.socket_path_prefix,
                    "dist_init_method": self._dist_init_method,
                    "model": self.model,
                    "enable_nvtx": self.enable_nvtx,
                    "enable_prof": self.enable_prof,
                    "device": worker_device(self.device_type, rank, self.rank_devices),
                    "log_level": self.log_level,
                    "tensor_comm_protocol": self.tensor_comm_protocol,
                    "tcp_transfer_device": self.tcp_transfer_device,
                    "status_interval_s": self._status_interval_s,
                    "env": self._mps.client_env if self._mps is not None else None,
                },
                daemon=False,
            )
            p.start()
            self._worker_processes.append(p)

    def shutdown(self):
        """Terminate and join all worker processes, then stop the MPS daemon if any."""
        try:
            self._stop_workers()
        finally:
            if self._mps is not None:
                self._mps.stop()

    def _stop_workers(self):
        if not self._worker_processes:
            return  # already done (run()'s caller and the atexit hook both call this)
        logger.info("Shutting down conductor...")
        # SIGTERM is handled in _worker_process_target as a graceful exit so
        # the transports' cleanup runs (shm segments unlinked). A worker
        # blocked in a C call cannot service it in time, so escalate to
        # SIGKILL after the join window rather than hang the shutdown —
        # the leftover segments are then reclaimed by the next arena
        # start's orphan sweep.
        for p in self._worker_processes:
            if p.is_alive():
                p.terminate()
        for p in self._worker_processes:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(
                    "Worker pid %s did not exit on SIGTERM; killing", p.pid)
                p.kill()
                p.join(timeout=5)
        self._worker_processes.clear()

    def _dead_workers(self) -> list[tuple[str, mp.Process]]:
        """(worker_id, handle) for every worker process that has exited.
        Handles are appended in ``worker_ids`` order by ``_launch_workers``.
        ``is_alive`` reaps the child, so ``exitcode`` is set afterwards."""
        return [
            (worker_id, p)
            for worker_id, p in zip(self.worker_ids, self._worker_processes, strict=True)
            if not p.is_alive()
        ]

    def _worker_nodes(self, worker_id: str) -> set[str]:
        return {
            node
            for wg in self._per_worker_graphs.get(worker_id, [])
            for node in wg.section.get_nodes()
        }

    def _poll_worker_liveness(self) -> None:
        """Raise ``DeadWorkerError`` if a worker process has exited.

        A worker that dies during setup (init exception, OOM kill) or while
        serving (SIGKILL, segfault) sends nothing, so without this check the
        startup wait and the main loop would wait for it forever and its
        requests would sit until the API server's timeout. A dead worker is
        fatal for the deployment. Every waiting client gets a 503 naming it,
        then the exception propagates out of ``run`` so the conductor
        terminates the remaining workers and exits non-zero.
        """
        now = time.perf_counter()
        if now < self._next_liveness_check:
            return
        self._next_liveness_check = now + self._liveness_interval_s
        dead = self._dead_workers()
        if not dead:
            return
        for worker_id, p in dead:
            logger.error(
                "Worker %s (pid %s) exited with %s. It hosted nodes %s. "
                "Shutting the deployment down.",
                worker_id, p.pid, describe_exitcode(p.exitcode),
                sorted(self._worker_nodes(worker_id)),
            )
        worker_id, p = dead[0]
        self._fail_all_requests(
            f"worker {worker_id} (pid {p.pid}) exited with "
            f"{describe_exitcode(p.exitcode)}, so the server is shutting down"
        )
        raise DeadWorkerError(worker_id, p.pid, p.exitcode)

    def _fail_all_requests(self, error_message: str, status: int = 503) -> None:
        """Notify the client of every request still awaiting a result, bypassing
        the drain barrier. Only for a deployment that is going down. The workers
        are about to be terminated anyway, and waiting for a dead participant's
        READS_DONE would just hold the client until the request timeout."""
        pending = [body.request_id for body in self.waiting_queue]
        for request_id in self.requests:
            dr = self.draining.get(request_id)
            if dr is not None and dr.failure_error is None:
                continue  # completed or aborted, the client has already heard
            pending.append(request_id)
        for request_id in pending:
            self.communicator.send(
                "api_server",
                APIServerMessage(
                    message_type="request_failed",
                    body=RequestFailed(
                        request_id=request_id,
                        error_message=error_message,
                        status=status,
                    ),
                ),
            )

    def _assign_worker_graphs_to_workers(self) -> dict[int, list[str]]:
        """
        For a request, assign worker graphs to workers. DP picks are
        coordinated by ``_group_id`` so all wgs derived from the same
        node_group land on the same (replica's) workers — without this,
        two wgs sharing a TP group could end up on different DP replicas
        and break model topology.

        TODO: smarter assignment that minimizes cross-graph-walk tensor
        transfer (e.g., bias toward keeping prefill→decode handoff local
        for the same request).
        """
        # _group_id -> chosen DP-replica index within that group's ranks
        group_id_to_replica_idx: dict[int, int] = {}
        result = {}
        for wg_id, wg in self.worker_graphs.items():
            if wg._instance_ranks:
                # Route to a whole instance (the lockstep unit): every rank of a
                # tp*sp instance runs the request together, so its TP all-reduce
                # and SP all-to-all collectives stay in sync. Picking a single TP
                # row here would desync the SP all-to-all across rows.
                replica_idx = group_id_to_replica_idx.setdefault(
                    wg._group_id, np.random.randint(len(wg._instance_ranks)),
                )
                ranks = wg._instance_ranks[replica_idx]
                result[wg_id] = [f"worker_{r}" for r in ranks]
            else:
                replica_idx = group_id_to_replica_idx.setdefault(
                    wg._group_id, np.random.randint(len(wg.ranks)),
                )
                result[wg_id] = [f"worker_{wg.ranks[replica_idx]}"]
        return result

    def _build_request_sharding_config(
        self, worker_graph_to_workers: dict[int, list[str]],
    ) -> ShardingConfig:
        """Per-request ShardingConfig: clone default + setup with this
        request's worker assignments.

        TODO: each worker also builds its own ShardingConfig from
        ``worker_graph_to_workers`` (with the worker's own ``_tp_rank``
        set). The duplication keeps conductor↔worker chatter down, but if
        request setup ever becomes a hotspot, consider sending the built
        config over instead.
        """
        cfg = self.default_sharding_config.clone_empty()
        node_to_workers: dict[NodeAndGraphWalk, list[str]] = {}
        for wg_id, worker_ids in worker_graph_to_workers.items():
            wg = self.worker_graphs[wg_id]
            for walk in wg.graph_walks:
                for node_name in wg.section.get_nodes():
                    node_to_workers[NodeAndGraphWalk(node_name, walk)] = worker_ids
        cfg.setup(node_to_workers)
        cfg.assert_stream_consumer_compatibility(self.streaming_consumers)
        return cfg

    @staticmethod
    def _find_client_streams(model: Model) -> dict[str, ClientStream]:
        """Where each client stream's chunks go: to the one node that consumes
        it, in the partition it flows into, as validate_deployment checked."""
        connections = model.get_partition_topology().client_connections()
        if not connections:
            return {}
        walks = model.get_graph_walk_graphs()
        partitions = {p.name: p for p in model.get_partitions()}
        streams: dict[str, ClientStream] = {}
        for conn in connections:
            name, partition = conn.edge_name, partitions[conn.to_partition]
            (consumer,) = {
                node.name for walk in walks.values() for node in walk.get_nodes().values()
                if name in node.input_names
            }
            walk = max(walk for walk in partition.graph_walks if consumer in walks[walk].get_nodes())
            streams[name] = ClientStream(consumer, walk, partition.name)
        return streams

    def _split_inputs_to_workers(
        self,
        sharding_config: ShardingConfig,
        inputs: list[GraphEdge],
        graph_walk: str,
    ) -> dict[str, list[GraphEdge]]:
        """Route inputs to consumer workers using per-source-rank fanout.

        tensor_info is grouped by (source_tp_rank, _source_node_name,
        _source_graph_walk); each group fans out via the request's
        ShardingConfig. Multi-rank sources produce one edge per source rank
        per dest, which the consumer's fan-in path consolidates.
        """
        inputs_per_worker: dict[str, list[GraphEdge]] = defaultdict(list)
        # fanned-out edge -> (input, index) of each descriptor, for the rejoin
        positions: dict[int, list[tuple[int, int]]] = {}
        for e_idx, edge in enumerate(inputs):
            if not edge.tensor_info:
                # Signal-only — broadcast to every dest worker.
                dest_workers = sharding_config.node_to_worker.get(
                    NodeAndGraphWalk(edge.next_node, graph_walk), [],
                )
                for dest_worker in dest_workers:
                    inputs_per_worker[dest_worker].append(edge.clone())
                continue

            groups: dict[tuple, list[int]] = defaultdict(list)
            for i, info in enumerate(edge.tensor_info):
                key = (
                    info.source_tp_rank,
                    info._source_node_name,
                    info._source_graph_walk,
                )
                groups[key].append(i)

            for (src_rank, src_node, src_walk), idxs in groups.items():
                sub_edge = edge.clone()
                sub_edge.tensor_info = [edge.tensor_info[i] for i in idxs]
                fanout = sharding_config.fanout_graph_edges(
                    sub_edge,
                    source_node=src_node,
                    source_graph_walk=src_walk,
                    dest_graph_walk=graph_walk,
                    source_tp_rank=src_rank,
                )
                for dest_worker, sliced_edge in fanout.items():
                    # fan-out keeps one descriptor per input descriptor, in order
                    positions[id(sliced_edge)] = [(e_idx, i) for i in idxs]
                    inputs_per_worker[dest_worker].append(sliced_edge)
        # The split above yields one edge per (rank, node, walk), so a name
        # produced across walks arrives as several. The consumer would take the
        # second as the next iteration's input and drop the third, so rejoin
        # them in persist order. Keyed by source rank: a TP fan-in edge stays
        # one per rank, which is what the consumer consolidates on.
        for dest_worker, edges in inputs_per_worker.items():
            merged: list[GraphEdge] = []
            first_by_key: dict[tuple[str, str, int], GraphEdge] = {}
            joined: dict[int, list[tuple[int, int]]] = {}
            for edge in edges:
                if not edge.tensor_info:
                    merged.append(edge)  # signal-only, nothing to join
                    continue
                key = (
                    edge.name, edge.next_node,
                    edge.tensor_info[0].source_tp_rank,
                )
                first = first_by_key.get(key)
                if first is not None:
                    # Rank numbering only lines up if the contributors share a
                    # source TP size, which is what fan-in counts.
                    assert first._total_fanin == edge._total_fanin, (
                        f"{edge.name}: contributors to one name disagree on "
                        f"fan-in ({first._total_fanin} vs {edge._total_fanin})"
                    )
                    first.tensor_info = first.tensor_info + edge.tensor_info
                    joined.setdefault(id(first), list(positions[id(first)])).extend(
                        positions[id(edge)]
                    )
                    continue
                first_by_key[key] = edge
                merged.append(edge)
            for edge in merged:
                order = joined.get(id(edge))
                if order is not None:
                    edge.tensor_info = [
                        info for _, info in sorted(
                            zip(order, edge.tensor_info, strict=True), key=lambda p: p[0],
                        )
                    ]
            inputs_per_worker[dest_worker] = merged
        return inputs_per_worker

    def _update_persist_ref_counts(
        self, request_id: str, inputs: list[GraphEdge]
    ):
        """Update reference counts for persist signals in inputs."""
        ref_cnts = self.requests[request_id].persist_signal_ref_cnt
        for edge in inputs:
            for info in edge.tensor_info:
                if info.uuid not in ref_cnts:
                    ref_cnts[info.uuid] = 0
                ref_cnts[info.uuid] += 1

    def _un_persist_tensors(
        self, request_id: str, tensor_info: list[TensorPointerInfo]
    ):
        entity_id_to_msg = {}
        uuids = []
        for info in tensor_info:
            uuid_to_ref_count = entity_id_to_msg.setdefault(
                info.source_entity, UnpersistTensors(
                    request_id=request_id, uuid_to_ref_count={}
                )
            ).uuid_to_ref_count

            if info.uuid in uuid_to_ref_count:
                # duplicate; skip
                continue
            ref_cnt = self.requests[request_id].persist_signal_ref_cnt.get(info.uuid)
            if ref_cnt is None:
                continue  # tensor not tracked (e.g., from a different partition)
            uuid_to_ref_count[info.uuid] = ref_cnt
            uuids.append(info.uuid)
        self.requests[request_id].remove_persist_signal_uuids(uuids)

        for (entity, body) in entity_id_to_msg.items():
            self.communicator.send(
                entity, WorkerMessage(
                    message_type=WorkerMessageType.UNPERSIST_TENSORS,
                    body=body
                )
            )

    def _try_admit_waiting(self):
        """Drain the waiting queue up to the concurrency cap."""
        while self.waiting_queue:
            if (self.max_concurrent_requests is not None
                    and len(self.requests) >= self.max_concurrent_requests):
                break
            body = self.waiting_queue.pop(0)
            logger.info(
                "Admitting queued request %s (%d/%s in-flight)",
                body.request_id, len(self.requests),
                str(self.max_concurrent_requests),
            )
            self._admit_request(body)

    def _ingest_request(
        self, body: NewRequestConductor
    ):
        """
        When a new request comes in from the API server, assign workers,
        initialize partition states, and kick off all partitions.
        """

        if body.request_id in self._early_abort_requests:
            # The abort outran this NEW_REQUEST. The preprocess worker's input
            # signals exist only now, so this is the first moment we can safely
            # tell it to drop them; it has already stopped reading the rid.
            self._early_abort_requests.discard(body.request_id)
            self._early_reads_done.pop(body.request_id, None)
            self._send_remove_to_preprocess_worker(body.request_id)
            logger.info(
                "Request %s was aborted before ingest; dropping", body.request_id
            )
            return
        try:
            _request_seed(body.model_kwargs or {}, _seed_id(body))
        except ValueError as exc:
            self._reject_request(body.request_id, str(exc), status=400)
            return
        if (self.max_concurrent_requests is not None
                and len(self.requests) >= self.max_concurrent_requests):
            ended = body.client_input.ended if body.client_input is not None else ()
            if self._client_streams.keys() - set(ended):
                # A live call that waited for a slot would start, and stay, as
                # far behind its client as it waited.
                self._reject_request(
                    body.request_id, "server at capacity: a call with open input can't wait for a slot",
                    status=503,
                )
                return
            logger.info(
                "Request %s queued (at capacity: %d/%d)",
                body.request_id, len(self.requests),
                self.max_concurrent_requests,
            )
            self.waiting_queue.append(body)
            if self._status_interval_s is not None:
                self._queued_since[body.request_id] = time.perf_counter()
            return
        if self.enable_nvtx:
            range_push("conductor._do_ingest_request")
        self._admit_request(body)
        if self.enable_nvtx:
            range_pop()

    def _admit_request(self, body: NewRequestConductor):
        """Dispatch ``body``; if that raises, fail the request instead of dropping it.

        An exception here would otherwise reach the run loop, which only logs
        it, and the client would wait for its timeout. A ``ValueError`` raised
        before any worker heard of the request is the client's fault: a 400.
        """
        try:
            self._do_ingest_request(body)
        # Any failure must reach the client.
        except Exception as exc:  # noqa: BLE001
            request_data = self.requests.get(body.request_id)
            if request_data is None or not request_data.dispatched:
                self.requests.pop(body.request_id, None)
                if isinstance(exc, ValueError):
                    self._reject_request(body.request_id, scrub_paths(str(exc)), status=400)
                    return
                text, _ = client_error_text(
                    exc, f"Ingest failed for request {body.request_id}", log=logger,
                )
                self._reject_request(body.request_id, f"Ingest failed: {text}", status=500)
            elif body.request_id not in self.draining:
                text, _ = client_error_text(
                    exc, f"Ingest failed for request {body.request_id}", log=logger,
                )
                # Some workers may already hold it: drain them like a worker failure.
                self._remove_request(
                    body.request_id, request_data, failure_error=f"Ingest failed: {text}",
                )
        else:
            if body.client_input is not None:
                self._stream_input(body.client_input)

    def _reject_request(self, request_id: str, error: str, status: int):
        """Fail a request that was never dispatched to any worker.

        Only the preprocess worker holds its input signals, so hard-remove
        them there, as an abort before admission does, then tell the client.
        """
        logger.info("Rejecting request %s (%d): %s", request_id, status, error)
        self._early_reads_done.pop(request_id, None)
        self._send_remove_to_preprocess_worker(request_id)
        self._send_for_teardown(
            request_id, "api_server",
            APIServerMessage(
                message_type="request_failed",
                body=RequestFailed(request_id=request_id, error_message=error, status=status),
            ),
        )

    def _do_ingest_request(
        self, body: NewRequestConductor
    ):
        """Actually dispatch a request to workers (no admission check)."""
        logger.debug("Conductor ingesting request %s", body.request_id)
        ingest_time = time.perf_counter()
        # A request admitted on arrival never waited.
        arrival_time = (
            None if self._status_interval_s is None
            else self._queued_since.pop(body.request_id, ingest_time)
        )
        worker_graph_to_workers = self._assign_worker_graphs_to_workers()

        model_kwargs = body.model_kwargs or {}
        max_output_tokens = self._max_output_tokens(model_kwargs)
        loop_caps = self._loop_caps(model_kwargs)
        # Honor an explicit per-request seed (e.g. OpenAI ``seed``) when given;
        # otherwise derive a stable seed from the request id. ``_ingest_request``
        # already rejected a bad one.
        seed = _request_seed(model_kwargs, _seed_id(body))

        partitions = self.model.get_partitions()
        topology = self.model.get_partition_topology()

        # Build partition states and definitions
        partition_states: dict[str, PartitionState] = {}
        partition_definitions: dict[str, PartitionDefinition] = {}
        for p in partitions:
            partition_definitions[p.name] = p
            partition_states[p.name] = PartitionState(
                partition_name=p.name,
                metadata=CurrentForwardConductorMetadata(
                    input_modalities=body.initial_input_modalities,
                    output_modalities=body.initial_output_modalities,
                    graph_walk="",
                    is_prefill=True,
                ),
                random_seed=seed,
            )

        # Build per-connection streaming state
        streaming_connections: dict[tuple[str, str, str], StreamingConnectionState] = {}
        for conn in topology.connections:
            # a tuple: names may hold ":" or "->" ("C:x" + "y" vs "C" + "x:y")
            key = (conn.from_partition, conn.to_partition, conn.edge_name)
            streaming_connections[key] = StreamingConnectionState(
                from_partition=conn.from_partition,
                to_partition=conn.to_partition,
                edge_name=conn.edge_name,
            )

        # Collect all worker_graph_ids per worker for the NewRequest
        worker_to_worker_graph_ids: dict[str, list[int]] = defaultdict(list)
        for wg_id, worker_ids in worker_graph_to_workers.items():
            for worker_id in worker_ids:
                worker_to_worker_graph_ids[worker_id].append(wg_id)

        request_data = RequestData(
            persist_signals=body.initial_signals,
            persist_signal_ref_cnt={},
            worker_graph_to_workers=worker_graph_to_workers,
            all_worker_graph_ids=set(worker_graph_to_workers.keys()),
            max_output_tokens=max_output_tokens,
            random_seed=seed,
            partition_states=partition_states,
            partition_definitions=partition_definitions,
            streaming_connections=streaming_connections,
            resource_configs={},
            sharding_config=self._build_request_sharding_config(worker_graph_to_workers),
            conductor_ingest_time=ingest_time,
            conductor_arrival_time=arrival_time,
        )
        self.requests[body.request_id] = request_data

        # Kick off all partitions by calling get_initial_forward_pass_args per partition
        partition_fwd_args: dict[str, ForwardPassArgs] = {}
        for p in partitions:
            fwd_args = self.model.get_initial_forward_pass_args(
                partition_name=p.name,
                input_modalities=body.initial_input_modalities,
                output_modalities=body.initial_output_modalities,
                input_signals=body.initial_signals,
                model_kwargs=body.model_kwargs,
            )
            pstate = partition_states[p.name]
            # if a partition is not active at all in the request, register that here
            pstate.is_done = fwd_args.request_done

            pstate.metadata = fwd_args.full_metadata
            pstate.metadata.kwargs.update(fwd_args.step_metadata)
            self._set_partition_worker_graph_ids(
                body.request_id, p.name, fwd_args.full_metadata.graph_walk,
            )
            partition_fwd_args[p.name] = fwd_args

        # after the initial fwd args: the configs are derived from them (BAGEL
        # reads `requires_cfg` off the metadata the model just settled)
        request_data.resource_configs = self._get_resource_configs(
            model_kwargs, partition_fwd_args
        )
        # keyed by resource, so each config is handed only its own chain
        kwargs = model_kwargs or {}
        prefix_keys = kwargs.get("prefix_keys") or {}
        prefix_tail = kwargs.get("prefix_tail") or {}
        prefix_decode = kwargs.get("prefix_decode") or {}
        prefix_cache = kwargs.get("prefix_cache")
        if not all(partition_states[name].is_done for name in self._streamed_partitions()):
            prefix_cache = False
        kv_layout = kwargs.get("kv_layout") or {}
        refuse_undeclared_labels(type(self.model).__name__, self._kv_labels, kv_layout, request_data.resource_configs)
        for key, cfg in request_data.resource_configs.items():
            cfg.apply_conductor_config(
                seed=seed,
                resource_key=key,
                prefix_keys=prefix_keys.get(key),
                prefix_tail=prefix_tail.get(key),
                prefix_decode=prefix_decode.get(key),
                prefix_cache=prefix_cache,
                kv_layout=kv_layout.get(key),
                loop_caps=loop_caps,
            )
        if not any(model_kwargs.get(key) is not None for key in self.model.LENGTH_KWARGS):
            request_data.max_output_tokens, loop_caps = self._fit_loop_caps(
                model_kwargs, request_data.resource_configs, max_output_tokens,
            )
            for key, cfg in request_data.resource_configs.items():
                cfg.apply_conductor_config(seed=seed, resource_key=key, loop_caps=loop_caps)
        specs = self._resource_specs()
        for key, cfg in request_data.resource_configs.items():
            cfg.validate(specs.get(key))

        # Send NewRequest to each worker with the appropriate partition's inputs
        request_data.dispatched = True
        for worker_id, worker_graph_ids in worker_to_worker_graph_ids.items():
            # Determine which partition this worker serves
            for partition_name, partition_wg_ids in self._resolve_worker_partition(
                worker_graph_ids, partitions,
            ).items():
                fwd_args = partition_fwd_args[partition_name]
                pstate = partition_states[partition_name]
                inputs_per_worker = self._split_inputs_to_workers(
                    sharding_config=request_data.sharding_config,
                    inputs=fwd_args.inputs,
                    graph_walk=fwd_args.full_metadata.graph_walk,
                )

                self._update_persist_ref_counts(
                    body.request_id,
                    inputs_per_worker.get(worker_id, [])
                )

                message = NewRequest(
                    request_id=body.request_id,
                    partition_worker_graph_ids=partition_wg_ids,
                    worker_graph_to_workers=worker_graph_to_workers,
                    initial_inputs=inputs_per_worker.get(worker_id, []),
                    loop_caps=loop_caps,
                    request_info=CurrentForwardPassInfo(
                        request_id=body.request_id,
                        graph_walk=fwd_args.full_metadata.graph_walk,
                        step_metadata=fwd_args.step_metadata,
                        stream_lead_items=fwd_args.stream_lead_items,
                        fwd_index=pstate.fwd_pass_number,
                        random_seed=pstate.random_seed,
                        partition_name=partition_name,
                        max_tokens=request_data.max_output_tokens,
                        resource_configs=request_data.resource_configs
                    ),
                )
                self.communicator.send(
                    worker_id, WorkerMessage(
                        message_type=WorkerMessageType.NEW_REQUEST,
                        body=message,
                    ),
                )

        request_data.resource_configs = without_layouts(request_data.resource_configs)

        # After the loop: a tensor's reader count must cover every worker and
        # partition it went to before the producer may free it.
        self._un_persist_tensors(body.request_id, [
            info for fwd_args in partition_fwd_args.values()
            for info in fwd_args.unpersist_tensors
        ])

    def _resolve_worker_partition(
        self, worker_graph_ids: list[int],
        partitions: list[PartitionDefinition],
    ) -> dict[str, set[int]]:
        """Find which partition(s) a set of worker graphs belongs to."""
        partition_wg_ids = {}
        for wg_id in worker_graph_ids:
            wg_walks = self._all_worker_graph_ids_to_graph_walks.get(wg_id, set())
            for p in partitions:
                if wg_walks & p.graph_walks:
                    partition_wg_ids.setdefault(p.name, set()).add(wg_id)
        return partition_wg_ids

    def _set_partition_worker_graph_ids(
        self, request_id: str, partition_name: str, graph_walk: str,
    ):
        """Update the set of active worker graph IDs for a partition's walk."""
        pstate = self.requests[request_id].partition_states[partition_name]
        pstate.current_worker_graph_ids = {
            wg_id for wg_id in self.requests[request_id].all_worker_graph_ids
            if graph_walk in self.worker_graphs[wg_id].graph_walks
        }

    PREPROCESS_WORKER = "api_server_preprocess_worker"

    def _request_workers(self, request_data: RequestData) -> set[str]:
        return {
            worker_id
            for worker_ids in request_data.worker_graph_to_workers.values()
            for worker_id in worker_ids
        }

    def _register_draining(
        self, request_id: str, expected_acks: set[str], participants: set[str],
        failure_error: str | None = None, failure_status: int = 500,
    ):
        """Start the teardown barrier for a request. Applies any READS_DONE that
        raced ahead of registration, and finalizes immediately if already
        satisfied (e.g. a happy path with no outstanding reader)."""
        expected_acks = set(expected_acks) - self._early_reads_done.pop(request_id, set())
        self.draining[request_id] = DrainingRequest(
            expected_acks=expected_acks,
            participants=participants,
            failure_error=failure_error,
            failure_status=failure_status,
        )
        self._draining_deadlines.append(
            (time.perf_counter() + self._drain_ttl_s, request_id)
        )
        if not expected_acks:
            self._finalize_draining(request_id)

    def _send_drain(self, request_id: str, entities: set[str]):
        for entity in entities:
            self._send_for_teardown(
                request_id, entity,
                WorkerMessage(
                    message_type=WorkerMessageType.DRAIN_REQUEST,
                    body=DrainRequest(request_id),
                ),
            )

    def _send_remove_to_preprocess_worker(self, request_id: str):
        self._send_for_teardown(
            request_id, self.PREPROCESS_WORKER,
            WorkerMessage(
                message_type=WorkerMessageType.REMOVE_REQUEST,
                body=RemoveRequest(request_id),
            ),
        )

    def _send_for_teardown(self, request_id: str, entity: str, message):
        """Send one teardown message; a send that raises is logged so it can't
        skip the sends, drain record or free that follow it. A dead peer does
        not raise on either transport (the send queues, then blocks at the
        high-water mark), so this is only a backstop."""
        try:
            self.communicator.send(entity, message)
        except Exception:
            logger.exception(
                "Could not send %s for request %s to %s",
                _message_name(message), request_id, entity,
            )

    def _finalize_draining(self, request_id: str):
        """Every reader has drained: send the hard RemoveRequest to all
        participants, notify the client on the fail path, then free state."""
        dr = self.draining.pop(request_id, None)
        if dr is None:
            return
        # The drain record is gone, so nothing would ever finish this teardown
        # later: a send that raises must not skip the other sends or the free.
        for entity in dr.participants:
            self._send_for_teardown(
                request_id, entity,
                WorkerMessage(
                    message_type=WorkerMessageType.REMOVE_REQUEST,
                    body=RemoveRequest(request_id),
                ),
            )
        if dr.failure_error is not None:
            self._send_for_teardown(
                request_id, "api_server",
                APIServerMessage(
                    message_type="request_failed",
                    body=RequestFailed(
                        request_id=request_id,
                        error_message=dr.failure_error,
                        status=dr.failure_status,
                    ),
                ),
            )
        self.requests.pop(request_id, None)
        logger.info("Tore down request %s; freed worker resources", request_id)
        self._try_admit_waiting()

    def _handle_reads_done(self, body: ReadsDone):
        dr = self.draining.get(body.request_id)
        if dr is None:
            # Raced ahead of registration (preprocess-worker self-drain on abort).
            if body.request_id not in self._early_reads_done:
                self._early_reads_done_deadlines.append(
                    (time.perf_counter() + self._drain_ttl_s, body.request_id)
                )
            self._early_reads_done.setdefault(body.request_id, set()).add(body.entity_id)
            return
        dr.expected_acks.discard(body.entity_id)
        if not dr.expected_acks:
            self._finalize_draining(body.request_id)

    @staticmethod
    def _pop_expired(deadlines: deque[tuple[float, str]], now: float) -> list[str]:
        expired = []
        while deadlines and deadlines[0][0] <= now:
            expired.append(deadlines.popleft()[1])
        return expired

    def _sweep_expiry(self):
        """Expire teardown bookkeeping whose awaited message never arrived: push
        a stalled drain barrier through, and drop stale early-message entries so
        they can't accumulate for requests that never come back."""
        now = time.perf_counter()
        for request_id in self._pop_expired(self._draining_deadlines, now):
            dr = self.draining.get(request_id)
            if dr is None:
                continue  # finalized normally
            logger.error(
                "Drain barrier for request %s timed out after %.0fs with no "
                "READS_DONE from %s; forcing teardown. A stalled reader may "
                "still hold one of its segments.",
                request_id, self._drain_ttl_s, sorted(dr.expected_acks),
            )
            self._finalize_draining(request_id)
        for request_id in self._pop_expired(self._early_abort_deadlines, now):
            if request_id in self._early_abort_requests:
                self._early_abort_requests.discard(request_id)
                logger.debug(
                    "Dropping stale abort tombstone for request %s", request_id
                )
        for request_id in self._pop_expired(self._early_reads_done_deadlines, now):
            if self._early_reads_done.pop(request_id, None) is not None:
                logger.debug(
                    "Dropping stale early READS_DONE for request %s", request_id
                )

    def _fail_requests(self, body: FailRequests):
        """Tear down requests a worker reported as unservable. Routes through the
        drain barrier; the client is notified (request_failed) only once every
        reader has drained, so no output is unlinked under an in-flight read."""
        for rid, error_message in body.errors.items():
            request_data = self.requests.get(rid)
            if request_data is None or rid in self.draining:
                # Expected under TP: every rank raises symmetrically and each
                # reports the failure, so only the first report finds the
                # request. Also covers a client abort racing the failure.
                logger.info(
                    "Failure for request %s ignored; already finished, aborted, "
                    "or failed by another worker (%s)", rid, error_message,
                )
                continue
            logger.error("Request %s failed on a worker: %s", rid, error_message)
            try:
                self._remove_request(rid, request_data, failure_error=error_message)
            except Exception as exc:
                # the other requests in the batch keep their own error
                self._fail_after_error(rid, "failing it", exc)

    def _abort_request(self, request_id: str):
        """Tear down a request the client abandoned, freeing its worker GPU state."""
        for i, body in enumerate(self.waiting_queue):
            if body.request_id == request_id:
                # Queued, never dispatched: the preprocess worker is the only
                # holder of its (persisted) input signals, so hard-remove there.
                self.waiting_queue.pop(i)
                if self._status_interval_s is not None:
                    self._queued_since.pop(request_id, None)
                self._early_reads_done.pop(request_id, None)
                self._send_remove_to_preprocess_worker(request_id)
                logger.info("Aborted request %s before admission", request_id)
                return

        request_data = self.requests.get(request_id)
        if request_data is None:
            # Either already torn down (nothing to do) or the abort outran its
            # NEW_REQUEST. Tombstone it: _ingest_request drops the request and
            # removes the preprocess worker's signals once they exist. Sending
            # the RemoveRequest here instead would land before they do and leak
            # the segment. The tombstone expires on its own (see _sweep_expiry).
            self._early_abort_requests.add(request_id)
            self._early_abort_deadlines.append(
                (time.perf_counter() + self._drain_ttl_s, request_id)
            )
            logger.info(
                "Abort for request %s: unknown; already finished, or racing its "
                "own ingest", request_id,
            )
            return
        if request_id in self.draining:
            logger.info("Abort for request %s ignored; already draining", request_id)
            return
        self._remove_request(request_id, request_data)

    def _remove_request(
        self, request_id: str, request_data: RequestData,
        failure_error: str | None = None,
    ):
        """Begin teardown for an abort/fail: drain every participant's reads,
        then (on the barrier's completion) hard-remove. Shared by the abort
        (client went away) and fail (worker can't serve it) paths.
        """
        workers = self._request_workers(request_data)
        participants = workers | {self.PREPROCESS_WORKER}
        # The drain record below is what finishes this teardown (or its TTL
        # does), so a send that raises must not skip it. The sends go first: a
        # barrier whose READS_DONE all came early finalizes on registration,
        # and a DRAIN_REQUEST must not reach a participant after its
        # REMOVE_REQUEST.
        self._send_drain(request_id, participants)
        logger.info("Draining request %s (%d participants) before teardown", request_id, len(participants))
        self._register_draining(
            request_id,
            expected_acks=participants,
            participants=participants,
            failure_error=failure_error,
        )

    def _process_request_done(
        self, request_id: str
    ):
        """Called when all partitions are done."""
        logger.info("Request %s done", request_id)
        request_data = self.requests[request_id]
        request_data.conductor_finish_time = time.perf_counter()

        # Tell the client first, so nothing below adds to completion latency.
        self.communicator.send(
            "api_server",
            APIServerMessage(
                message_type="request_complete",
                body=RequestComplete(
                    request_id=request_id,
                    output_chunk_counts=request_data.output_chunk_counts,
                    conductor_ingest_time=request_data.conductor_ingest_time,
                    conductor_finish_time=request_data.conductor_finish_time,
                    graph_timings=request_data.graph_timings,
                    rx_info=list(request_data.rx_info.values()),
                    tx_info=list(request_data.tx_info.values()),
                    conductor_arrival_time=request_data.conductor_arrival_time,
                )
            )
        )
        request_data.completion_sent = True

        # Unpersist anything still held, with correct ref counts, so producers
        # reclaim it as the final reads ACK — before the hard teardown.
        still_persisted = [
            info
            for infos in request_data.persist_signals.values()
            for info in infos
        ]
        if still_persisted:
            self._un_persist_tensors(request_id, still_persisted)

        # Workers are done reading (the graph finished); the only remaining
        # reader is the preprocess worker still delivering outputs. Defer the
        # hard RemoveRequest until it signals READS_DONE, so a worker output
        # isn't unlinked under an in-flight read.
        # NOTE: "workers done reading" assumes a well-behaved graph. A model bug
        # that emits extra/erroneous tensors could leave a worker read scheduled
        # past completion, which this path won't wait for. Gating the happy path
        # on per-worker READS_DONE too (like abort/fail) would handle that
        # robustly — TODO once the barrier has proven out. A model with a client
        # stream already does: the client can append after the model ended the
        # request, so a chunk can be in a worker's read at any point.
        workers = self._request_workers(request_data)
        expected_acks = {self.PREPROCESS_WORKER}
        if self._client_streams:
            self._send_drain(request_id, workers)
            expected_acks |= workers
        self._register_draining(
            request_id,
            expected_acks=expected_acks,
            participants=workers | {self.PREPROCESS_WORKER},
        )

    def _process_worker_graphs_done(
        self, body: WorkerGraphsDone
    ) -> list[str]:
        """Process a WorkerGraphsDone message.

        Uses the partition_name from the message directly.
        Returns list of partition names whose full forward pass has completed.
        """
        if body.request_id not in self.requests:
            logger.debug(
                "Ignoring late WORKER_GRAPHS_DONE for completed request %s",
                body.request_id
            )
            return []

        request_data = self.requests[body.request_id]
        partition_name = body.partition_name
        if self.enable_prof:
            request_data.graph_timings.update(body.graph_timings)
            for rx in body.rx_info:
                request_data.rx_info[(rx.source_entity, rx.dest_entity, rx.edge_name)] = rx
            for tx in body.tx_info:
                request_data.tx_info[(tx.source_entity, tx.edge_name)] = tx

        # From every rank: a rank reports only the chunks it sent itself, and
        # each of those reaches the API server as its own message.
        counts = request_data.output_chunk_counts
        for name in body.output_signal_names:
            counts[name] = counts.get(name, 0) + 1

        # Matched by edge, not by this message's partition: a worker reports
        # what it sent on whichever completion goes out next. Every rank's
        # items are its own, so none is skipped.
        if body.stream_items_sent:
            for conn in request_data.streaming_connections.values():
                for worker_id, n in body.stream_items_sent.get(conn.edge_name, {}).items():
                    conn.items_sent[worker_id] = conn.items_sent.get(worker_id, 0) + n

        pstate = request_data.partition_states.get(partition_name)
        if pstate is None:
            logger.warning(
                "WorkerGraphsDone for unknown partition %s (request %s)",
                partition_name, body.request_id,
            )
            return []

        # Persist signals: every rank contributes its shard (different uuid +
        # source_tp_rank); accumulate across ranks, do not dedup.
        if body.persist_signals:
            for name, infos in body.persist_signals.items():
                request_data.persist_signals.setdefault(name, []).extend(infos)

        # Resource publish info is rank-sharded. Every TP rank contributes its
        # own KV pages and transfer descriptor, and PublishedKVInfo.update()
        # folds those entries by rank.
        merge_publish_info(
            pstate.resource_publish_info, body.resource_publish_info
        )

        # Absorb-only fields are replicated across TP ranks; only the rank-0
        # message contributes.
        if body.is_first_tp_rank:
            if body.new_token_counts:
                for name, count in body.new_token_counts.items():
                    pstate.num_output_tokens += count
                    for conn in request_data.streaming_connections.values():
                        if conn.from_partition == partition_name and conn.edge_name == name:
                            conn.token_count += count

            if body.stream_tokens_consumed:
                for conn in request_data.streaming_connections.values():
                    if conn.from_partition == partition_name:
                        continue  # skip producer connections
                    consumed = body.stream_tokens_consumed.get(conn.edge_name, 0)
                    conn.consumed_count = max(conn.consumed_count, consumed)

        # Each wg is only marked complete when all its TP ranks have reported.
        for wg_id in body.worker_graph_ids:
            count = pstate.wg_rank_completions.get(wg_id, 0) + 1
            pstate.wg_rank_completions[wg_id] = count
            expected = len(request_data.worker_graph_to_workers[wg_id])
            if count >= expected:
                pstate.completed_worker_graph_ids.add(wg_id)

        # Check if this partition's forward pass is fully done
        done_partitions = []
        if pstate.current_worker_graph_ids.issubset(pstate.completed_worker_graph_ids):
            done_partitions.append(partition_name)

        return done_partitions

    def _process_done_forward(
        self, request_id: str, partition_name: str,
        partition_done_from_worker: bool = False,
    ) -> bool:
        """Process a completed forward pass for a specific partition.

        Calls get_partition_forward_pass_args for all partitions uniformly.
        If the result has inputs, sends them. If not, the partition
        self-triggers (e.g., via StreamBuffer on the worker).

        Returns True if the **entire** request is done (all partitions finished).
        """
        request_data = self.requests[request_id]
        pstate = request_data.partition_states[partition_name]

        incoming_connections = request_data.get_incoming_connections(partition_name)

        # For partitions that self-trigger via StreamBuffer (have incoming
        # connections with topology), worker signals partition_done directly.
        if incoming_connections and partition_done_from_worker:
            pstate.is_done = True

        prev_walk =  pstate.metadata.graph_walk
        fwd_args = self.model.get_partition_forward_pass_args(
            partition_name=partition_name,
            partition_metadata=pstate.metadata,
            persist_signals=request_data.persist_signals,
            incoming_connections=incoming_connections,
        )
        pstate.metadata = fwd_args.full_metadata
        pstate.metadata.kwargs.update(fwd_args.step_metadata)

        # The worker's signal is authoritative for a stream-terminated
        # partition: the pass that consumed the stream's final chunk has run.
        # Treat it as the partition's own `request_done` so its downstream
        # connections get `producer_done` and the chain closes. A model that
        # inferred the end from the connection counters instead
        # (`consumed_count >= token_count`) could race: `consumed_count` counts
        # chunks popped for execution, and every worker-graphs-done report from
        # a worker carries every buffer's count, so a colocated consumer's report
        # could finish this partition while its last step was still running,
        # and that step's output arrived for a request already gone.
        if incoming_connections and partition_done_from_worker:
            fwd_args.request_done = True

        # Check max output tokens for partitions that produce tokens
        if pstate.num_output_tokens >= request_data.max_output_tokens:
            logger.info(
                "Partition %s reached max output tokens %d. Ending.",
                partition_name, request_data.max_output_tokens,
            )
            fwd_args.request_done = True

        logger.debug(
            "Partition %s of request %s: %s -> %s (request_done=%s, tokens=%d)",
            partition_name, request_id, prev_walk,
            fwd_args.full_metadata.graph_walk, fwd_args.request_done,
            pstate.num_output_tokens,
        )

        if fwd_args.request_done:
            pstate.is_done = True
            # Signal producer_done to all outgoing connections, once per
            # consumer partition however many edges lead there
            consumers: set[str] = set()
            for conn in request_data.streaming_connections.values():
                if conn.from_partition == partition_name:
                    conn.producer_done = True
                    if conn.to_partition not in consumers:
                        consumers.add(conn.to_partition)
                        self._send_producer_done(request_id, conn.from_partition, conn.to_partition)
        elif fwd_args.inputs:
            # Partition has inputs to send — conductor-driven
            self._send_partition_inputs(request_id, partition_name, fwd_args)
        # else: no inputs — partition self-triggers via StreamBuffer

        self._un_persist_tensors(request_id, fwd_args.unpersist_tensors)

        # Reset partition forward pass state
        pstate.completed_worker_graph_ids = set()
        pstate.current_worker_graph_ids = set()
        pstate.wg_rank_completions = {}
        pstate.fwd_pass_number += 1

        self._set_partition_worker_graph_ids(
            request_id, partition_name, fwd_args.full_metadata.graph_walk,
        )

        # Request done when ALL partitions are done
        return all(ps.is_done for ps in request_data.partition_states.values())

    def _send_partition_inputs(
        self, request_id: str, partition_name: str, fwd_args: ForwardPassArgs,
    ):
        """Send InputSignals for a specific partition's next forward pass."""
        self._send_inputs(
            request_id, partition_name, fwd_args.inputs,
            fwd_args.full_metadata.graph_walk, fwd_args.step_metadata,
        )

    def _send_inputs(
        self, request_id: str, partition_name: str, inputs: list[GraphEdge],
        graph_walk: str, step_metadata: dict,
    ) -> dict[str, list[GraphEdge]]:
        """Split ``inputs`` over the ranks that consume them, count each rank as
        a reader of the tensors, and send each its INPUT_SIGNALS. Returns what
        each rank was sent."""
        request_data = self.requests[request_id]
        pstate = request_data.partition_states[partition_name]

        inputs_per_worker = self._split_inputs_to_workers(
            sharding_config=request_data.sharding_config,
            inputs=inputs,
            graph_walk=graph_walk,
        )

        for worker, worker_inputs in inputs_per_worker.items():
            self._update_persist_ref_counts(request_id, worker_inputs)
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=worker_inputs,
                    request_info=CurrentForwardPassInfo(
                        request_id=request_id,
                        graph_walk=graph_walk,
                        step_metadata=step_metadata,
                        fwd_index=pstate.fwd_pass_number,
                        random_seed=pstate.random_seed,
                        resource_publish_info=pstate.resource_publish_info,
                        partition_name=partition_name,
                        max_tokens=request_data.max_output_tokens,
                        resource_configs=request_data.resource_configs,
                    ),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker, message)
        return inputs_per_worker

    def _send_producer_done(
        self, request_id: str, producer_partition: str,
        consumer_partition_name: str, edges: Collection[str] | None = None,
    ):
        """Send producer_done signal to the consumer partition's worker(s): for
        every stream between the two partitions, or only ``edges``."""
        request_data = self.requests[request_id]
        pstate = request_data.partition_states[consumer_partition_name]

        # Find which workers handle this consumer partition
        consumer_workers = set()
        pdef = request_data.partition_definitions[consumer_partition_name]
        for wg_id, worker_ids in request_data.worker_graph_to_workers.items():
            walks = self._all_worker_graph_ids_to_graph_walks.get(wg_id, set())
            if walks & pdef.graph_walks:
                consumer_workers.update(worker_ids)

        # The worker ends the streams the signal counts, so it names each edge
        # it ends: every one between the two partitions, or ``edges``.
        conns = [
            conn for conn in request_data.streaming_connections.values()
            if conn.from_partition == producer_partition
            and conn.to_partition == consumer_partition_name
            and (edges is None or conn.edge_name in edges)
        ]
        for worker_id in consumer_workers:
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=[],
                    request_info=CurrentForwardPassInfo(
                        request_id=request_id,
                        graph_walk=pstate.metadata.graph_walk or "",
                        fwd_index=pstate.fwd_pass_number,
                        random_seed=pstate.random_seed,
                        partition_name=consumer_partition_name,
                        max_tokens=request_data.max_output_tokens,
                        resource_configs=request_data.resource_configs
                    ),
                    partition_name=consumer_partition_name,
                    producer_done=set([producer_partition]),
                    producer_done_items={
                        conn.edge_name: conn.items_sent.get(worker_id, 0) for conn in conns
                    },
                ),
            )
            self.communicator.send(worker_id, message)

    def _stream_input(self, body: ClientStreamInput):
        """Forward the next chunk of a request's client streams, or end them.
        A chunk this request can't take fails the request alone."""
        request_id = body.request_id
        request_data = self.requests.get(request_id)
        if request_data is None or request_id in self.draining:
            # It ended, failed or was aborted (a request still waiting for a
            # slot has no open input); the preprocess worker frees the chunk
            # with the request's other tensors.
            return
        try:
            for name, infos in body.signals.items():
                self._forward_client_chunk(request_id, request_data, name, infos)
            # each stream ends on its own; the call's input has ended once all have
            ended: dict[str, list[str]] = {}
            for name in body.ended:
                stream = self._client_streams.get(name)
                if stream is None:
                    logger.warning("Request %s: %r is not a client stream; not ending it", request_id, name)
                    continue
                conn = request_data.streaming_connections[CLIENT, stream.partition, name]
                if not conn.producer_done:
                    conn.producer_done = True
                    ended.setdefault(stream.partition, []).append(name)
            for partition in sorted(ended):
                self._send_producer_done(request_id, CLIENT, partition, edges=ended[partition])
        except Exception as exc:  # noqa: BLE001 — this request's failure, not the batch's
            text, _ = client_error_text(exc, f"Client input for request {request_id} failed", log=logger)
            self._remove_request(request_id, request_data, failure_error=f"client input failed: {text}")

    def _forward_client_chunk(
        self, request_id: str, request_data: RequestData, name: str,
        infos: list[TensorPointerInfo],
    ):
        """Send one client chunk to the workers of the node that consumes it,
        the way a partition's inputs are sent, and release its persist hold."""
        stream = self._client_streams.get(name)
        if stream is None:
            logger.warning("Request %s: %r is not a client stream; dropping it", request_id, name)
            return
        conn = request_data.streaming_connections[CLIENT, stream.partition, name]
        pstate = request_data.partition_states[stream.partition]
        if conn.producer_done or pstate.is_done:
            return  # nothing reads it any more; freed with the request
        if not infos:
            return
        edge = GraphEdge(next_node=stream.node, name=conn.edge_name, tensor_info=list(infos), is_streaming=True)
        sent = self._send_inputs(request_id, stream.partition, [edge], stream.graph_walk, {})
        conn.token_count += len(infos)
        # each tensor is one stream item; the producer-done signal reports them
        for worker, edges in sent.items():
            n = sum(len(e.tensor_info) for e in edges)
            conn.items_sent[worker] = conn.items_sent.get(worker, 0) + n
        # After the send: the hold is released with every reader counted.
        self._un_persist_tensors(request_id, infos)

    def _wait_for_workers_ready(self) -> None:
        """Block until every worker reports ``SETUP_DONE`` (weight load + warmup
        + CUDA-graph capture), so the main loop opens only once all workers can
        serve. Any non-``SETUP_DONE`` message that races in is stashed and
        replayed on the first main-loop iteration so it isn't lost. Raises
        ``DeadWorkerError`` if a worker exits before reporting.
        """
        pending = set(self.worker_ids)
        logger.info(
            "Conductor waiting for %d worker(s) to finish setup", len(pending)
        )
        while pending:
            for message in self.communicator.get_all_new_messages():
                if message.message_type == ConductorMessageType.SETUP_DONE:
                    pending.discard(message.body.worker_id)
                else:
                    self._startup_message_backlog.append(message)
            if pending:
                self._poll_worker_liveness()
                time.sleep(0.01)
        logger.info("Conductor: all %d worker(s) ready", len(self.worker_ids))

    def _route_message(
        self, message, done_partition_forwards: list[tuple[str, str, bool]],
    ) -> None:
        """Apply one message from the api server or a worker. A WORKER_GRAPHS_DONE
        that finishes a partition's pass is queued on ``done_partition_forwards``."""
        if message.message_type == ConductorMessageType.NEW_REQUEST:
            self._ingest_request(message.body)
        elif message.message_type == ConductorMessageType.ABORT_REQUEST:
            self._abort_request(message.body.request_id)
        elif message.message_type == ConductorMessageType.FAIL_REQUESTS:
            self._fail_requests(message.body)
        elif message.message_type == ConductorMessageType.READS_DONE:
            self._handle_reads_done(message.body)
        elif message.message_type == ConductorMessageType.CLIENT_STREAM_INPUT:
            self._stream_input(message.body)
        elif message.message_type == ConductorMessageType.WORKER_STATUS:
            self._worker_metrics[message.body.worker_id] = message.body.metrics
        elif message.message_type == ConductorMessageType.ENABLE_METRICS:
            # Every rank: each counts the tensors it reads. Sent before any
            # message of a request the API server submits after its warmups.
            for worker_id in self.worker_ids:
                self.communicator.send(worker_id, WorkerMessage(
                    message_type=WorkerMessageType.ENABLE_METRICS, body=message.body,
                ))
        elif message.message_type == ConductorMessageType.WORKER_GRAPHS_DONE:
            rid = message.body.request_id
            # Draining requests linger in self.requests (concurrency
            # accounting) but are being torn down — ignore late dones.
            if rid not in self.requests or rid in self.draining:
                logger.debug(
                    "WORKER_GRAPHS_DONE for unknown request %s (already completed?)", rid
                )
                return

            if self.enable_nvtx:
                range_push("conductor._process_worker_graphs_done")
            done_parts = self._process_worker_graphs_done(message.body)
            for pname in done_parts:
                done_partition_forwards.append(
                    (rid, pname, message.body.partition_done)
                )
            if self.enable_nvtx:
                range_pop()
        else:
            raise ValueError(f"Unknown message type: {message.message_type}")

    def _fail_after_error(
        self, request_id: str | None, step: str, exc: Exception,
    ) -> None:
        """Fail the one request whose handling raised, through the drain barrier,
        so its client gets the error and its concurrency slot is freed.

        A request whose ``request_complete`` already went out is only torn
        down: its client has the result."""
        text, _ = client_error_text(
            exc, f"Conductor error {step} for request {request_id}", log=logger,
        )
        request_data = self.requests.get(request_id)
        if request_data is None or request_id in self.draining:
            return  # unknown, never admitted, or already being torn down
        try:
            self._remove_request(
                request_id, request_data,
                failure_error=(
                    None if request_data.completion_sent
                    else f"Conductor error {step}: {text}"
                ),
            )
        except Exception:
            logger.exception("Could not fail request %s", request_id)

    def _maybe_send_status(self) -> None:
        """Send the API server a ConductorStatus once per status interval.
        The gauges are sampled here rather than kept up to date per message."""
        now = time.perf_counter()
        if now < self._next_status:
            return
        self._next_status = now + self._status_interval_s
        self.communicator.send(
            "api_server",
            APIServerMessage(message_type="conductor_status", body=self._status()),
        )

    def _status(self) -> ConductorStatus:
        running = 0
        in_flight = dict.fromkeys(self._components, 0)
        for request_id, request_data in self.requests.items():
            if request_id in self.draining:
                continue
            running += 1
            # wgs of this request's current passes that have not reported back
            nodes = set()
            for pstate in request_data.partition_states.values():
                if pstate.is_done:
                    continue
                for wg_id in pstate.current_worker_graph_ids - pstate.completed_worker_graph_ids:
                    nodes.update(self._wg_nodes[wg_id])
            for node in nodes:
                in_flight[node] += 1
        states = self._requests_gauge
        states.set(("waiting",), len(self.waiting_queue))
        states.set(("running",), running)
        states.set(("draining",), len(self.draining))
        for node, count in in_flight.items():
            self._in_flight_gauge.set((node,), count)
        # Summed across workers: a node's replicas each report their own; for
        # the engine families the other ranks of a lockstep instance report
        # nothing, while every rank counts the tensor transfers it reads.
        by_family: dict[str, list[MetricSnapshot]] = {}
        for snapshots in self._worker_metrics.values():
            for snap in snapshots:
                by_family.setdefault(snap.name, []).append(snap)
        return ConductorStatus(metrics=[
            states.snapshot(), self._in_flight_gauge.snapshot(),
            *(merge(family, by_family.get(name, ()))
              for name, family in catalog.WORKER_FAMILIES.items()),
        ])

    def run(self):
        from mstar.utils.profiler import range_pop, range_push

        self._wait_for_workers_ready()
        # the set-up heap is complete: keep gen-2 collections off it
        freeze_after_setup("Conductor")

        self.communicator.send(
            "api_server",
            APIServerMessage(message_type="setup_done")
        )

        while True:
            if self.enable_nvtx:
                range_push("conductor.run_loop")

            try:
                done_partition_forwards: list[tuple[str, str, bool]] = []

                startup_backlog = self._startup_message_backlog
                self._startup_message_backlog = []
                # Every message, pass and completion is isolated: the batch is
                # already off the socket, so one that raises would otherwise
                # take the rest of it down and they would never be resent.
                for message in startup_backlog + self.communicator.get_all_new_messages():
                    try:
                        self._route_message(message, done_partition_forwards)
                    except Exception as exc:
                        step = f"handling {_message_name(message)}"
                        for request_id in _message_request_ids(message):
                            self._fail_after_error(request_id, step, exc)

                completed_requests = []

                for request_id, partition_name, p_done in done_partition_forwards:
                    if request_id not in self.requests or request_id in self.draining:
                        continue  # already completed by another partition in this cycle
                    try:
                        all_done = self._process_done_forward(
                            request_id, partition_name,
                            partition_done_from_worker=p_done,
                        )
                    except Exception as exc:
                        self._fail_after_error(
                            request_id, f"advancing partition {partition_name}", exc,
                        )
                        continue
                    if all_done:
                        completed_requests.append(request_id)

                for request_id in dict.fromkeys(completed_requests):
                    if request_id in self.requests and request_id not in self.draining:
                        try:
                            self._process_request_done(request_id)
                        except Exception as exc:
                            self._fail_after_error(request_id, "completing it", exc)

                self._sweep_expiry()
                if self._status_interval_s is not None:
                    self._maybe_send_status()

            except Exception:
                logger.exception("Conductor error in main loop")
            finally:
                if self.enable_nvtx:
                    range_pop()

            # Outside the try above so DeadWorkerError leaves the loop instead of
            # being logged as a main-loop error and retried.
            self._poll_worker_liveness()
            time.sleep(0.001)
