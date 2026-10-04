import gc
import logging
import os
import sys
import threading
import time
import time as _time
from collections import defaultdict, deque
from collections.abc import Callable, Container, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from time import sleep

import torch

from mstar.communication.codec import WireCodec
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import (
    LocalTransferEngine,
    NameToTensorList,
    create_tensor_communication_manager,
)
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine import apply_torch_config
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AllocationFailed, StepContext
from mstar.engine.resources.kv.transfer import (
    TransferEngineInfo,
    make_deployment_kv_shm_dir,
)
from mstar.graph.base import GraphEdge, GraphSection
from mstar.graph.graph_io import format_graph_edge_list
from mstar.graph.runtime.base import (
    ColumnarEdgeSpecs,
    EdgeSpec,
    GraphRuntime,
    RouteInput,
    RouteOutput,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
)
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.runtime.utils import GraphRuntimeType, resolve_graph_runtime_type
from mstar.model.base import Model, WorkerGraph
from mstar.model.submodule_base import BatchedModelOutput, HostRows, InputMetadata
from mstar.profile.worker import WorkerProfileInfo
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo, StreamingEdge
from mstar.utils.containers import ParallelList, RecentSet
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    FailRequests,
    InputSignals,
    MessageSource,
    NewRequest,
    ReadsDone,
    RemoveRequest,
    ScheduleTPNode,
    SetupDone,
    StopLoops,
    TensorReceived,
    TPNoSpeculation,
    UnpersistTensors,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.profiler import PHASE_PERIOD, phase_buffer, range_pop, range_push
from mstar.worker.engine_manager import EngineManager
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.node_manager_utils import RequestStateManager

logger = logging.getLogger(__name__)

# seconds between "no offload possible" lines for one node and walk: a hold is
# retried every backoff, and a line per retry buries the rest of the log
_HOLD_LOG_INTERVAL = 5.0

# ``_await_admit_settled``'s safety net, not a tuning knob: the GPU thread
# releases the event on every exit, so this only fires if that thread died.
_ADMIT_FENCE_TIMEOUT_S = 10.0


def _make_graph_runtime(
    communicator,
    tensor_manager,
    **kwargs,
) -> GraphRuntime:
    """The runtime ``MSTAR_RUST_GRAPH`` selects (see
    ``resolve_graph_runtime_type``).

    Rust requires the Rust communicator and the Rust bookkeeper: it sends
    WORKER_GRAPHS_DONE and INPUT_SIGNALS itself on an Arc to the same
    transport, and settles refcounts on a share of the same bookkeeper, rather
    than hopping back into Python for either. A Python one of either has no
    such object to share, so asking for Rust with one raises.
    """
    if resolve_graph_runtime_type(log=True) == GraphRuntimeType.PYTHON:
        return PythonGraphRuntime(
            communicator=communicator,
            tensor_manager=tensor_manager,
            **kwargs,
        )

    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.graph.runtime.rust import RustGraphRuntime

    if not isinstance(communicator, RustZMQCommunicator):
        raise ValueError(
            "MSTAR_RUST_GRAPH=1 needs the Rust communicator, but this worker "
            f"built a {type(communicator).__name__}. Set MSTAR_RUST_ZMQ=1."
        )
    if not issubclass(communicator.codec, WireCodec):
        raise ValueError(
            "MSTAR_RUST_GRAPH=1 needs the msgpack codec, but this worker "
            f"built a {communicator.codec.__name__}. Set "
            "MSTAR_WIRE_CODEC=msgpack."
        )
    bookkeeping = tensor_manager.tensor_store.bookkeeping
    if not hasattr(bookkeeping, "_rust"):
        raise ValueError(
            "MSTAR_RUST_GRAPH=1 needs the Rust TensorBookkeeping: the runtime "
            "holds a share of it, so a Python one cannot be handed over. "
            f"Got {type(bookkeeping).__name__}."
        )
    return RustGraphRuntime(
        bookkeeping=bookkeeping,
        communicator=communicator,
        **kwargs,
    )


def _parse_tp_async_sched(raw: str) -> tuple[bool, frozenset[str] | None]:
    """``MSTAR_TP_ASYNC_SCHED``: ``0``/empty off, ``1`` every parallel node,
    or a comma-separated node list. Returns ``(enabled, nodes_or_None)``."""
    raw = raw.strip()
    if raw in ("", "0"):
        return False, None
    if raw == "1":
        return True, None
    return True, frozenset(n.strip() for n in raw.split(",") if n.strip())


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


@dataclass
class Speculation:
    scheduled_batch: ScheduledBatch
    node_batch: ExecutingBatch
    # ``(name, next_node)`` pairs the spec batch consumed from batch_N's
    # outputs. Two cases:
    #   * Same-node loop-back (AR decode iter K → iter K+1): pairs are
    #     ``{(name, batch_N.node_name) for name in loop_back_outputs}``.
    #   * Forward node A -> node B transition: pairs are
    #     ``{(edge.name, edge.next_node) for edge in batch_N.outputs if
    #     edge.next_node == spec_target.node_name}``.
    # Consumed in ``_thread_outputs_to_speculative`` to splice batch_N's
    # outputs into the spec batch's per-rid input tensors.
    consumed_edges: set[tuple[str, str]]
    continuing_rids: set[int]
    partition: str
    is_new_iter: bool
    is_same_node: bool
    # rid -> edges
    consumed_streaming_edges: dict[int, list[StreamingEdge]] = field(default_factory=dict)
    # Handle for the runtime's staged streaming ingests. Settled exactly once,
    # via ``_settle_speculation``: on submit (keeping them), or on abandon /
    # per-rid drop (undoing those rids', so the chunk leaves the node's slot
    # before it is handed back to its StreamBuffer).
    spec_id: int = 0
    is_yield_away: bool = False
    loop_name: str | None = None
    dropped: set[int] = field(default_factory=set)

    plan_future: Future | None = None
    # TP async scheduling: the broadcast seq of this speculation (leader:
    # assigned at broadcast; follower: the head's). PendingBatch.tp_seq on submit.
    tp_seq: int = -1


class EvictionPolicy(Enum):
    """Strategy for choosing which request to offload to CPU on OOM."""
    LRU = "lru"  # least-recently-used (by execution time)
    # TODO: PRIORITY — ask a named resource how much it wants each candidate
    # gone (``Engine.offload_priority``), for cases where "least recently used"
    # is the wrong question. Needs eviction-policy metadata naming which
    # resource to consult.


class Worker:
    """
    Real worker that integrates RequestStateManager, EngineManager,
    MicroScheduler, and MooncakeCommunicationManager to execute
    computation via engines.
    """

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
        dist_init_method=None
    ):
        self.worker_id = worker_id
        self.device = device
        self.enable_nvtx = enable_nvtx

        # Per-phase wall-clock timing (MSTAR_PHASE_TIMING). On the worker
        # rather than in run()'s scope because the GPU and plan threads
        # record into it too; run() owns the periodic flush.
        self._phase_period = PHASE_PERIOD
        self._phase_buf = phase_buffer()
        # (start, end) CUDA events per step, drained once they land
        self._gpu_spans: deque = deque(maxlen=64)

        self.enable_prof = enable_prof
        self.profile_info = WorkerProfileInfo()

        if self.device.type != "cpu" and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)

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

        # Build node_to_partition mapping from model's partitions and graph walks
        node_to_partition: dict[str, str] = {}
        if model is not None:
            partitions = model.get_partitions()
            walks = model.get_graph_walk_graphs()
            for pdef in partitions:
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section:
                        for node_name in section.get_nodes():
                            node_to_partition[node_name] = pdef.name

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
                socket_path_prefix=socket_path_prefix,
                dist_init_method=dist_init_method,
            )

        node_names = set()
        for wg in my_worker_graphs:
            node_names.update(wg.section.get_nodes())


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
        )

        self.request_state = RequestStateManager(
            node_to_partition=node_to_partition,
        )

        # The lockstep unit for a node is its whole instance: the tensor-parallel
        # row composed with the sequence-parallel column. Exactly one rank per
        # instance — instance rank 0, i.e. rank 0 in BOTH its TP and SP comm
        # groups — leads scheduling and broadcasts ScheduleTPNode to the rest;
        # every other instance rank follows. Keying the leader off the TP rank
        # alone would elect one leader per TP row (e.g. ranks 0 and 2 of a
        # tp2*sp2 instance), racing the followers and desyncing the per-step
        # graph walk.
        self.parallel_leader_nodes = set([
            node for node in node_names
            if self.parallel_groups.get_instance_rank_for_node(node) == 0
        ])

        # v1: disallow multiple lockstep-scheduled nodes in the same worker.
        # A node is lockstep-scheduled when its instance spans more than one rank,
        # i.e. tp_size * sp_size > 1. Pure sequence-parallel nodes (tp_size 1,
        # sp_size > 1) need this too: their attention all-to-all requires the
        # whole instance to step together. Without SP this is just tp_size > 1.
        self.parallel_nodes = set([
            node for node in node_names
            if self.parallel_groups.get_instance_world_size_for_node(node) > 1
        ])
        if len(self.parallel_nodes) > 1:
            raise NotImplementedError(
                f"Multiple parallel nodes {self.parallel_nodes} found in worker "
                f"{worker_id}; current implementation requires at most one "
                "lockstep-parallel node per worker."
            )

        self.is_tp_follower = len(self.parallel_nodes - self.parallel_leader_nodes) > 0

        # TP async scheduling: the leader speculates N+1 during forward N and
        # broadcasts it at once; followers rebuild it during their own N.
        self.tp_async_sched, self.tp_async_nodes = _parse_tp_async_sched(
            os.environ.get("MSTAR_TP_ASYNC_SCHED", "0")
        )
        # Leader: monotonic seq stamped on every ScheduleTPNode it sends.
        self._tp_broadcast_seq = 0
        # Follower: leader steps that will NOT be followed by a head; the
        # leader said so (TPNoSpeculation) or this rank closed the step.
        self._tp_nospec: RecentSet[int] = RecentSet(self._TP_NOSPEC_KEEP)
        self._tp_leader_gap_warned = False
        tp_async_on = sorted(n for n in self.parallel_nodes if self._tp_async_for(n))
        if tp_async_on:
            logger.info(
                "Worker %s: TP async scheduling ON for %s (%s)",
                worker_id, tp_async_on,
                "follower" if self.is_tp_follower else "leader",
            )
        elif self.tp_async_sched and self.parallel_nodes:
            logger.warning(
                "Worker %s: MSTAR_TP_ASYNC_SCHED=%r names none of this worker's "
                "parallel nodes %s; running the serial protocol",
                worker_id, os.environ.get("MSTAR_TP_ASYNC_SCHED"),
                sorted(self.parallel_nodes),
            )

        self.scheduler = MicroScheduler(
            self.engine_manager,
            parallel_leader_nodes=self.parallel_leader_nodes
        )

        # Request ids are strings on the wire and ints (handles) inside this
        # worker; the runtime holds the table where the two meet.
        self.scheduler.runtime = self._graph_runtime
        self.scheduler.rid_of = self._rid
        self.tensor_manager.rid_to_str = self._rid_str

        # wire request id -> messages for requests that are not in the queue.
        # Keyed by the string: these arrive before the NEW_REQUEST that mints
        # the handle.
        self._unprocessed_messages: dict[str, list[WorkerMessage]] = {}

        # CPU offloading: LRU tracking and eviction policy
        self._last_active: dict[tuple[int, str], float] = {}  # (request_id, node_name) -> monotonic timestamp
        self.eviction_policy = EvictionPolicy.LRU
        # (node, walk) -> when its last hold was logged, and the holds since
        self._hold_logged: dict[tuple[str, str], tuple[float, int]] = {}

        # Async-scheduling cross-iter state. Initialized here (rather than in
        # run()) because _remove_request — which can be invoked indirectly
        # from _process_messages on any iter — reads/writes them.
        # _in_flight_rids: rids referenced by an in-flight GPU step or its
        #   speculation; REMOVE_REQUEST for these is deferred.
        # _pending_removes: deferred REMOVE_REQUESTs.
        self._in_flight_rids: set[int] = set()
        self._pending_removes: set[int] = set()

        # step seq -> rids rank 0 released after that step, held until this rank
        # has consumed it. See ``_removal_step_reached``.
        self._removes_awaiting_step: dict[int, list[str]] = {}

        # Teardown drain (abort/fail): _pending_drains hold DrainRequests deferred
        # behind an in-flight GPU step; _draining_rids have stopped reading and
        # persist until REMOVE_REQUEST (so no read can restart after READS_DONE);
        # _reads_done_sent tracks which have already ACKed. These three are keyed
        # by the wire string: a DRAIN can precede the NEW_REQUEST that mints the
        # handle, and a draining rid has to refuse that late NEW_REQUEST.
        self._pending_drains: set[str] = set()
        self._draining_rids: set[str] = set()
        self._reads_done_sent: set[str] = set()

        # Let the scheduler see deferred removes so it stops initiating new work
        # for those rids (shared by reference — mutations are visible to both).
        self.scheduler.pending_removes = self._pending_removes

        # Side stream for D→H copies in postprocess (check_stop pre-materialize).
        # The default stream has GPU(N+1) queued behind GPU(N)'s outputs after
        # speculation, so syncing on default would also drain GPU(N+1) and
        # erase the overlap. The side stream waits on
        # ``output.completion_event`` (recorded after GPU(N)) and then runs
        # an isolated D→H, so the main thread only blocks on the copy.
        # Lazy-initialized — workers without CUDA never touch it.
        self._d2h_stream: "torch.cuda.Stream | None" = None
        self._pinned_d2h_buffers: dict[
            tuple[str, torch.dtype, int], list[torch.Tensor]
        ] = defaultdict(list)

        # Streaming buffers: request_id -> edge_name -> list of tensors
        # (Legacy path — kept for models without PartitionTopology)
        self.streaming_buffers: dict[int, dict[str, list[torch.Tensor]]] = {}

        # New streaming path: PartitionTopology + StreamBuffer on consumer worker
        self.partition_topology = model.get_partition_topology() if model else None

        # Determine which partition this worker serves (by checking which node names
        # appear in my_worker_graphs vs the topology connections)
        self._my_consumer_connections = []
        if self.partition_topology:
            my_node_names = set()
            for wg in my_worker_graphs:
                my_node_names.update(wg.section.get_nodes())
            for conn in self.partition_topology.connections:
                # Check if any graph walk graph node for the consumer partition is on this worker
                # by checking if the streaming edge's next_node is in my nodes
                if any(n in my_node_names for n in self._get_node_names_for_partition(conn.to_partition, model)):
                    self._my_consumer_connections.append(conn)

        # Set of edge names that arrive via streaming (used to distinguish
        # streaming inputs from conductor-triggered non-streaming inputs
        # when checking whether a target node is ready for ingestion).
        self._streaming_edge_names: set[str] = {
            conn.edge_name for conn in self._my_consumer_connections
        }

        # Build consumer node cache: edge_name -> consuming node name.
        self._consumer_node_cache: dict[str, str] = {}
        if self._my_consumer_connections and model:
            self._consumer_node_cache = self._build_consumer_node_cache(
                self._my_consumer_connections, model.get_graph_walk_graphs(),
            )

        # edge_name -> partition the stream feeds
        self._stream_partition: dict[str, str] = {
            conn.edge_name: conn.to_partition for conn in self._my_consumer_connections
        }

        # (graph_walk, node) -> innermost enclosing loop, for ending a
        # stream-terminated loop in _postprocess_batch. Structural, so it is
        # derived from the model here rather than asked of the graph runtime
        # per step -- which also keeps it identical under both runtimes.
        self._innermost_loop_by_node: dict[tuple[str, str], str] = {}
        if self._my_consumer_connections and model:
            self._innermost_loop_by_node = self._build_innermost_loop_cache(
                model.get_graph_walk_graphs(),
            )

    @staticmethod
    def _build_innermost_loop_cache(
        walks: dict[str, GraphSection],
    ) -> dict[tuple[str, str], str]:
        """Map each ``(graph_walk, node)`` to its innermost enclosing loop.

        The innermost one is the candidate with no other candidate strictly
        inside it. Replaces reading ``loop_name_order[-1]`` off the per-request
        ``NestedLoopIndices``, which the worker no longer sees now that loop
        state sits behind the ``GraphRuntime`` contract.
        """
        cache: dict[tuple[str, str], str] = {}
        for walk_name, section in walks.items():
            loops = section.get_loops()
            for node_name in section.get_nodes():
                enclosing = {
                    name: loop for name, loop in loops.items()
                    if node_name in loop.get_nodes()
                }
                for name, loop in enclosing.items():
                    if not set(loop.section.get_loops()) & enclosing.keys():
                        cache[(walk_name, node_name)] = name
                        break
        return cache

    @staticmethod
    def _build_consumer_node_cache(
        connections, walks: dict[str, GraphSection],
    ) -> dict[str, str]:
        """Map each incoming streaming edge to the node that consumes it.

        Recurses through ``get_nodes()`` so a consumer nested inside a
        ``Loop`` / ``Sequential`` / ``Parallel`` is found too. Matching the
        walk's top-level ``input_names`` only finds a walk that is itself a
        single ``GraphNode``; a decode ``Loop`` whose inner node consumes the
        streamed edge would be missed, and its chunks routed to ``next_node=""``.
        """
        cache: dict[str, str] = {}
        for conn in connections:
            for section in walks.values():
                for node in section.get_nodes().values():
                    if conn.edge_name in node.input_names:
                        cache[conn.edge_name] = node.name
        return cache

    def _get_node_names_for_partition(self, partition_name: str, model: Model) -> list[str]:
        """Get the node names that belong to a partition.

        Recurses into each walk so nodes nested in a ``Loop`` / ``Sequential``
        / ``Parallel`` are included. Taking the section's own ``name`` would
        return the Loop's name (e.g. ``"talker_decode_loop"``) instead of the
        consuming node's, so its ``StreamBuffer`` would never be created and
        routing the streamed edge would fail with a ``KeyError``.
        """
        walks = model.get_graph_walk_graphs()
        partitions = model.get_partitions()
        for pdef in partitions:
            if pdef.name == partition_name:
                nodes = set()
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section is not None:
                        nodes.update(section.get_nodes().keys())
                return list(nodes)
        return []

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    # -- request id string <-> handle -------------------------------------
    # A handle is minted on NEW_REQUEST and looked up on every other inbound
    # message; the string goes back on the wire at each send site.

    def _rid(self, request_id: str) -> int | None:
        """Handle for an inbound message's request id, or None if this worker
        does not know it (removed already -- a benign race, not an error)."""
        return self._graph_runtime.get_rid_handle(request_id)

    def _rid_str(self, request_id: int) -> str:
        """The wire identity, for a message about to leave this process."""
        return self._graph_runtime.get_rid_string(request_id)

    def _add_new_request(self, body: NewRequest) -> None:
        if body.request_id in self._draining_rids:
            # Being torn down (out-of-order NEW after DRAIN); don't start reads.
            return
        logger.debug("Worker %s received request %s", self.worker_id, body.request_id)
        # The one place a handle is minted; a request with several partitions
        # on this worker gets the same one for each. Everything below keys on it.
        request_id = self._graph_runtime.add_request(
            request_id=body.request_id,
            partition=body.request_info.partition_name,
            graph_walk=body.request_info.graph_walk,
            partition_worker_graph_ids=body.partition_worker_graph_ids,
            worker_graph_to_workers=ParallelList.from_dict(body.worker_graph_to_workers),
        )
        body.request_info.rid_handle = request_id

        now = _time.monotonic()
        for node_name in self.engine_manager.evictable_nodes():
            self._last_active[(request_id, node_name)] = now

        self.request_state.add_request(request_id, body.request_info)
        self.engine_manager.add_request(
            request_id, body.request_info.resource_configs,
        )
        self.tensor_manager.register_request(
            request_id, self._graph_runtime.get_sharding_config(request_id),
        )

        # Create StreamBuffers for consumer connections on this worker. A request
        # with several partitions here arrives once per partition: keep the
        # buffer the first arrival made (it may already hold items).
        req_info = self.request_state.per_request_info[request_id]
        for conn in self._my_consumer_connections:
            sbuf = req_info.stream_buffers.get(conn.edge_name)
            if sbuf is None:
                sbuf = StreamBuffer(
                    request_id=request_id,
                    edge_name=conn.edge_name,
                    from_partition=conn.from_partition,
                    policy=conn.chunk_policy_factory(),
                )
                req_info.stream_buffers[conn.edge_name] = sbuf
                consumer = self._consumer_node_cache.get(conn.edge_name, "")
                req_info.stream_buffers_by_consumer.setdefault(consumer, {})[conn.edge_name] = sbuf
            if conn.to_partition == body.request_info.partition_name:
                lead = body.request_info.stream_lead_items.get(conn.edge_name)
                if lead:
                    sbuf.prime_context(lead)

        # Start RDMA reads for tensors that have tensor_info
        futures = self.tensor_manager.start_read_tensors(
            request_id, body.initial_inputs,
            graph_walk=body.request_info.graph_walk
        )
        self.wakeup_event.register_futures(futures)

        # Signal-only edges (tensor_info is None) can be processed immediately
        signal_only = [
            edge for edge in body.initial_inputs if len(edge.tensor_info) == 0
        ]
        if signal_only:
            self._graph_runtime.ingest_inputs_batch(
                self._one_rids_edges(request_id, signal_only), can_buffer=True,
            )
        # process messages that may have came in out-of-order
        if body.request_id in self._unprocessed_messages:
            self._process_message_list(self._unprocessed_messages[body.request_id])
            del self._unprocessed_messages[body.request_id]


    def _remove_request(self, body: RemoveRequest) -> None:
        if self.is_tp_follower:
            if body.source not in (MessageSource.TP_RANK_0, MessageSource.SELF):
                return # wait for rank 0's forward, to avoid race conditions
            if not self._removal_step_reached(body):
                # Rank 0 released these pages between two steps; tearing down
                # before this rank finishes the earlier one would have it admit
                # that step with pages rank 0 no longer held. Follower-only:
                # ``last_consumed_tp_seq`` never advances on a leader, which
                # would park a stamped teardown for good.
                self._removes_awaiting_step.setdefault(
                    body.after_tp_seq, []
                ).append(body.request_id)
                return

        # Async-scheduling deferral: if this rid is currently held by an
        # in-flight GPU step (or its speculation), tearing down engine /
        # tensor state now would race the GPU thread reading those tensors
        # / KV pages. Queue the remove and apply it once no in-flight step
        # references the rid (see _apply_pending_removes_safe_to_drop in
        # the run loop).
        request_id = self._rid(body.request_id)
        if request_id is None:
            # Never admitted here, or already removed: no handle-keyed state to
            # clear. The wire-string-keyed TP-follow count is the exception --
            # it can be non-empty for a rid that never got a handle, and no
            # later message would ever pop it.
            self.scheduler.clear_wire_rid(body.request_id)
            return
        if request_id in self._in_flight_rids:
            self._pending_removes.add(request_id)
            return

        # If we are the TP leader for this request, signal the followers to
        # remove it too. Followers defer removal until they get this message
        # (see the guard at the top of this method) so they can't tear down
        # state we're still reading from an in-flight step/speculation.
        sharding = self._graph_runtime.get_sharding_config(request_id)
        if sharding is not None:
            followers: set[str] = set()
            for group in sharding.groups:
                # _workers is rank-ordered; index 0 is this worker when we are
                # rank 0. Only real TP groups (tp_size > 1) have followers.
                if group.tp_size > 1 and group._tp_rank == 0:
                    followers.update(group._workers[1:])
            for worker in followers:
                self.communicator.send(
                    worker, msg=WorkerMessage(
                        message_type=WorkerMessageType.REMOVE_REQUEST,
                        body=RemoveRequest(
                            request_id=body.request_id,
                            source=MessageSource.TP_RANK_0,
                            # this rank is about to release the pages, having
                            # broadcast up to here; followers must do it in the
                            # same gap between steps
                            after_tp_seq=self._tp_broadcast_seq - 1,
                        )
                    )
                )

        # Hard cleanup: force-drop every tensor for the rid (unlink SHM),
        # ignoring ref counts / persist. Safe because the conductor only sends
        # REMOVE_REQUEST once every reader has confirmed drained (READS_DONE) —
        # on the abort/fail path via a prior DrainRequest, on the happy path
        # once the api server finished reading the outputs.
        self._draining_rids.discard(body.request_id)
        self._pending_drains.discard(body.request_id)
        self._reads_done_sent.discard(body.request_id)
        self.engine_manager.remove_request(request_id)
        self.request_state.remove_request(request_id)
        self.tensor_manager.force_cleanup_request(request_id)
        self.profile_info.pop_request(request_id)
        self.streaming_buffers.pop(request_id, None)
        self.scheduler.clear_rid(request_id, body.request_id)
        self._pending_removes.discard(request_id)

        for node_name in self.engine_manager.evictable_nodes():
            self._last_active.pop((request_id, node_name), None)
        logger.info("Request cleanup complete: %s", body.request_id)

        # Last: frees the handle for reuse, so nothing above may run after it.
        self._graph_runtime.remove_request(request_id)

    def _drain_request(self, body: DrainRequest) -> None:
        """Phase-1 teardown (abort/fail): stop scheduling and reading this rid,
        then ACK READS_DONE once its in-flight reads finish. The hard cleanup
        (force_cleanup_request) waits for the conductor's REMOVE_REQUEST, sent
        only after every reader has ACKed."""
        if self.is_tp_follower and body.source not in (
            MessageSource.TP_RANK_0, MessageSource.SELF
        ):
            return  # honor only the leader's forwarded drain, like REMOVE_REQUEST

        # Defer behind an in-flight GPU step, same as REMOVE_REQUEST: clearing
        # the scheduler while a step references the rid would race the GPU thread.
        handle = self._rid(body.request_id)
        if handle is not None and handle in self._in_flight_rids:
            self._pending_drains.add(body.request_id)
            return

        self._begin_drain(body.request_id)

    def _begin_drain(self, request_id: str) -> None:
        """Takes the wire string: a drain can arrive for a request this worker
        never admitted, and it still has to go into ``_draining_rids`` so a late
        NEW_REQUEST is refused."""
        handle = self._rid(request_id)
        # Fan the drain to TP followers so each rank drains and ACKs its own
        # READS_DONE (the conductor waits on every rank).
        sharding = (
            None if handle is None else self._graph_runtime.get_sharding_config(handle)
        )
        if sharding is not None:
            followers: set[str] = set()
            for group in sharding.groups:
                if group.tp_size > 1 and group._tp_rank == 0:
                    followers.update(group._workers[1:])
            for worker in followers:
                self.communicator.send(
                    worker, msg=WorkerMessage(
                        message_type=WorkerMessageType.DRAIN_REQUEST,
                        body=DrainRequest(
                            request_id=request_id,
                            source=MessageSource.TP_RANK_0,
                        )
                    )
                )

        # Stop new leader work but keep draining committed TP batches (failed_rids
        # is exactly this gate); keep engine/tensor/queue state until REMOVE.
        if handle is not None:
            self.scheduler.fail_rids({handle})
        self._draining_rids.add(request_id)
        self._complete_drain_if_ready(request_id)

    def _complete_drain_if_ready(self, request_id: str) -> None:
        """ACK READS_DONE once no async read for the rid is still in flight.
        The rid stays in _draining_rids (reads gated) until REMOVE_REQUEST."""
        if request_id not in self._draining_rids:
            return
        if request_id in self._reads_done_sent:
            return
        handle = self._rid(request_id)
        # A request this worker never admitted holds no handle-keyed state, so
        # there is nothing in flight to wait on.
        if handle is not None and self.tensor_manager.has_inflight_reads(handle):
            return  # let get_ready_tensors resolve the futures; retry next iter
        # Not under the handle guard: a ScheduleTPNode can land before the
        # NEW_REQUEST that mints the handle, so the queued batch is counted
        # under the wire string while ``handle`` is still None. ACKing here
        # would let the REMOVE tear the worker-graph queues out from under a
        # batch still sitting at the head of the TP FIFO.
        if self.scheduler.pending_tp_follow_count.get(request_id, 0) > 0:
            return  # wait for committed TP-follow batches to drain first
        self._reads_done_sent.add(request_id)
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.READS_DONE,
                body=ReadsDone(
                    request_id=request_id, entity_id=self.worker_id
                ),
            ),
        )

    def _apply_pending_drains(self, in_flight_rids: set[int]) -> None:
        """Begin drains that were deferred behind an in-flight GPU step, and
        re-check draining rids whose reads may now have finished.

        ``_pending_drains`` holds wire strings and ``in_flight_rids`` holds
        handles, so the comparison goes through the rid table.
        """
        for rid in [
            r for r in self._pending_drains if self._rid(r) not in in_flight_rids
        ]:
            self._pending_drains.discard(rid)
            self._begin_drain(rid)
        for rid in list(self._draining_rids):
            self._complete_drain_if_ready(rid)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        # Uuids are global, so a late ack needs no rid lookup and works even
        # after the request is gone.
        # One crossing for the ack: a reader confirms a whole edge at a time,
        # and the count differs per uuid (an edge read by several consumers).
        self.tensor_manager.dereference_batch(
            list(body.successful_tensors),
            list(body.successful_tensors.values()),
        )

    def _process_new_inputs(self, body: InputSignals) -> None:
        # Draining for teardown: don't start new reads for this rid. The
        # producer's segment may be unlinked once every reader ACKs READS_DONE.
        if body.request_id in self._draining_rids:
            return
        logger.debug(
            "Received new signals %s at worker %s for request %s",
            format_graph_edge_list(body.inputs), self.worker_id, body.request_id
        )
        request_id = self._rid(body.request_id)
        if request_id is None:
            # _process_message_list parks signals for an unknown request, so
            # this only happens if that changes; there is nothing to route into.
            logger.warning(
                "Worker %s: dropping input signals for unknown request %s",
                self.worker_id, body.request_id,
            )
            return
        req_info = self.request_state.per_request_info.get(request_id)

        if self.enable_nvtx:
            range_push("process_new_inputs.routing_update")
        # Handle producer_done signal: mark all StreamBuffers for this request as done
        if body.producer_done:
            if req_info:
                for sbuf in req_info.stream_buffers.values():
                    if sbuf.from_partition in body.producer_done:
                        # If we have multiple consumer partitions colocated, we need to signal
                        # the right one
                        sbuf.signal_done()

        # Separate streaming edges — they'll be handled when tensors are ready
        # (streaming edges with tensor_info go through RDMA, handled in _check_ready_tensors)
        non_streaming = [edge for edge in body.inputs if not edge.is_streaming]
        streaming_with_tensors = [edge for edge in body.inputs if edge.is_streaming and edge.tensor_info]

        # Only update fwd_info when there are non-streaming edges (i.e., this is
        # a conductor-triggered forward pass, not just streaming data from another
        # partition). Streaming-only InputSignals must not overwrite the current
        # partition's fwd_info.
        if non_streaming:
            # A fwd_info off the wire carries only the string (or the sender's
            # handle); stamp this worker's so submodules keying per-request
            # state can use it directly.
            body.request_info.rid_handle = request_id
            self._graph_runtime.set_walk(
                request_id, body.partition_name, body.request_info.graph_walk,
            )
            self.request_state.update_request_info(
                request_id, current_fwd_info=body.request_info,
                partition_name=body.partition_name
            )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.start_read")
        # Start RDMA reads for non-streaming edges with tensor_info
        futures = self.tensor_manager.start_read_tensors(
            request_id, non_streaming,
            graph_walk=body.request_info.graph_walk
        )
        self.wakeup_event.register_futures(futures)
        # Start RDMA reads for streaming edges with tensor_info (will be routed to buffer in _check_ready_tensors)
        if streaming_with_tensors:
            futures = self.tensor_manager.start_read_tensors(
                request_id, streaming_with_tensors,
            )
            self.wakeup_event.register_futures(futures)
            for edge in streaming_with_tensors:
                stream_buf = req_info.stream_buffers[edge.name]
                for info in edge.tensor_info:
                    stream_buf.pre_read_register(info.uuid)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.process_inputs")

        # Streaming signal-only edges: nothing to buffer (no tensor data)
        # This shouldn't normally happen for streaming edges

        # Signal-only non-streaming edges can be processed immediately
        signal_only = [edge for edge in non_streaming if len(edge.tensor_info) == 0]
        if signal_only:
            self._graph_runtime.ingest_inputs_batch(
                self._one_rids_edges(request_id, signal_only), can_buffer=True,
            )
        if self.enable_nvtx:
            range_pop()

    def _unpersist_tensors(self, body: UnpersistTensors):
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            self.tensor_manager.increment_ref(uuid, n=ref_cnt)
            self.tensor_manager.set_persist(uuid, persist=False)

    def _stop_loops(self, body: StopLoops):
        request_id = self._rid(body.request_id)
        if request_id is None:
            return
        self._graph_runtime.apply_peer_loop_stops(
            request_id, body.partition_name, body.loop_stop_times,
        )

    def _process_message_list(self, messages: list[WorkerMessage]):
        msg_types_needing_active_request = [
            WorkerMessageType.REMOVE_REQUEST,
            WorkerMessageType.INPUT_SIGNALS,
            WorkerMessageType.STOP_LOOPS
        ]
        # Snapshot: a REMOVE handled mid-iteration can re-buffer trailing
        # signals onto this same list, and mutating it while iterating it would
        # never terminate.
        for message in list(messages):
            # per_request_info is handle-keyed, so resolve the wire string
            # first; an unknown one has no handle and is parked the same way.
            if (
                message.message_type in msg_types_needing_active_request and \
                self._rid(message.body.request_id)
                not in self.request_state.per_request_info
            ):
                # got an out-of-order request
                self._unprocessed_messages.setdefault(
                    message.body.request_id, []
                ).append(message)
                continue
            if message.message_type == WorkerMessageType.NEW_REQUEST:
                self._add_new_request(message.body)
            elif message.message_type == WorkerMessageType.DRAIN_REQUEST:
                self._drain_request(message.body)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                self._remove_request(message.body)
            elif message.message_type == WorkerMessageType.INPUT_SIGNALS:
                self._process_new_inputs(message.body)
            elif message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                self._handle_tensor_received(message.body)
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                self._unpersist_tensors(message.body)
            elif message.message_type == WorkerMessageType.STOP_LOOPS:
                self._stop_loops(message.body)
            elif message.message_type == WorkerMessageType.SCHEDULE_TP:
                self._register_tp_follow(message.body)
            elif message.message_type == WorkerMessageType.TP_NO_SPEC:
                self._register_tp_nospec(message.body)

    def _process_messages(self) -> None:
        self._process_message_list(self.communicator.get_all_new_messages())

    # ------------------------------------------------------------------
    # Tensor readiness
    # ------------------------------------------------------------------

    def _route_streaming_tensor(self, request_id: int, edge: GraphEdge) -> None:
        """Route a streaming tensor to its request's StreamBuffer for this edge."""
        req_info = self.request_state.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]

        for info in edge.tensor_info:
            tensor = self.tensor_manager.get_tensor(info.uuid)

            stream_buf.put(info.uuid, tensor.clone())
        # After the loop: one crossing for the edge rather than one per tensor.
        self.tensor_manager.dereference_batch_uniform(
            [info.uuid for info in edge.tensor_info]
        )

    def _pop_streaming_edge(
        self, sbuf: StreamBuffer, edge_name: str, request_id: int
    ) -> StreamingEdge | None:
        consumer_node = self._consumer_node_cache.get(edge_name, "")
        waiting = sbuf.pop_waiting_edge()
        if waiting is not None:
            return waiting
        if sbuf.has_chunk_ready():
            chunk = sbuf.pop_chunk()
            chunk_tensor = chunk.data.get("data")
            if chunk_tensor is None:
                # Empty chunk — producer done, no more data.
                # Create edge with empty tensor_info.
                synthetic_edge = GraphEdge(
                    next_node=consumer_node,
                    name=edge_name,
                    tensor_info=[],
                    _final_stream_chunk=chunk.is_final,
                )
            else:
                # Normal chunk — store tensor and create edge with tensor_info.
                # Local streaming tensors are routed from outputs that were
                # already gated on the producer completion event before being
                # stored, so avoid a default-stream sync here. If future
                # streaming producers bypass that path, StreamChunk should
                # carry producer events and this call site should wait on
                # those events before storing with skip_cuda_sync=True.
                tensor_infos = self.tensor_manager.store_and_return_tensor_info(
                    request_id, {edge_name: [chunk_tensor]},
                    skip_cuda_sync=True,
                )
                synthetic_edge = GraphEdge(
                    next_node=consumer_node,
                    name=edge_name,
                    tensor_info=tensor_infos.get(edge_name, []),
                    _final_stream_chunk=chunk.is_final,
                )
            return StreamingEdge(synthetic_edge, chunk.info)
        return None

    def _poll_stream_buffers_for_speculation(
        self, request_id: int, node_name: str
    ) -> list[StreamingEdge]:
        result = []
        req_info = self.request_state.per_request_info.get(request_id)
        if req_info is None:
            return []
        for edge_name, sbuf in req_info.stream_buffers_by_consumer.get(node_name, {}).items():
            edge = self._pop_streaming_edge(sbuf, edge_name, request_id)
            if edge is not None:
                result.append(edge)
        return result

    def _return_streaming_edge(
        self, request_id: int, streaming_edge: StreamingEdge
    ):
        """Hand a chunk back to its StreamBuffer after it was refused.

        Both polling paths use it: the plain one above and the speculative
        prep, which rolls its ingests back per rid.
        """
        req_info = self.request_state.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(streaming_edge.edge.name)
        if sbuf is not None:
            sbuf.store_uningested_edge(streaming_edge)
        # The step that took the final chunk will not run; the stream is live again.
        edge = streaming_edge.edge
        if edge._final_stream_chunk:
            req_info.ended_streams.discard(edge.name)

    def _mark_stream_ingested(
        self, request_id: int, streaming_edge: StreamingEdge
    ) -> None:
        """
        Remember the chunk the graph accepted, for the batch that consumes it.
        NOTE: a single ingested_chunk field works because streaming edges are
        ingested with can_buffer=False -- except when speculating a loop-back,
        where the current value is already in-flight in the forward, so the
        buffered chunk is the only one a later build can still be describing.
        """
        req_info = self.request_state.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(streaming_edge.edge.name)
        if sbuf is not None:
            sbuf.ingested_chunk = streaming_edge.chunk

    def _stream_chunks_for(
        self, request_id: int, node_name: str, inputs: NameToTensorList
    ) -> dict[str, StreamChunkInfo] | None:
        """Chunk info for the streamed inputs this step consumes, or None."""
        req_info = self.request_state.per_request_info.get(request_id)
        if req_info is None:
            return None
        sbufs = req_info.stream_buffers_by_consumer.get(node_name)
        if not sbufs:
            return None
        chunks = {
            name: sbuf.ingested_chunk for name, sbuf in sbufs.items()
            if name in inputs and sbuf.ingested_chunk is not None
        }
        return chunks or None

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        # Every ready chunk is popped first and ingested in one call (and one
        # Python <> Rust roundtrip)
        polled: list[tuple[int, StreamingEdge]] = []
        for request_id, req_info in list(self.request_state.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                streaming_edge = self._pop_streaming_edge(sbuf, edge_name, request_id)
                if streaming_edge is not None:
                    polled.append((request_id, streaming_edge))
        if not polled:
            return

        # Streaming edges go through the same path as regular ones —
        # ReadySignals.is_ready_for_streaming flips on as soon as the streaming
        # inputs are the only ones missing. The final-chunk signal rides the
        # synthetic edge to the consuming pass, which reports the partition done
        # in _postprocess_batch — NOT here, where an earlier in-flight pass's
        # WGD could read it before the final output chunk is emitted.
        refused = set(self._graph_runtime.ingest_inputs_batch(
            self._edge_block((rid, [se.edge]) for rid, se in polled),
            # important: only ingest for this loop iter!
            can_buffer=False,
            is_streaming=True,
        ))
        # Indices into the block, which is `polled` order.
        for i, (rid, se) in enumerate(polled):
            if i in refused:
                self._return_streaming_edge(rid, se)
            else:
                self._mark_stream_ingested(rid, se)


    def _check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready graph edges to worker graph queues."""
        self.wakeup_event.drain()
        ready = self.tensor_manager.get_ready_tensors()
        for request_id, edges in ready.items():
            # Separate streaming edges from normal edges
            streaming = [e for e in edges if e.is_streaming]
            normal = [e for e in edges if not e.is_streaming]

            if self.enable_nvtx:
                range_push("check_ready-tensors.route_streaming")
            for edge in streaming:
                self._route_streaming_tensor(request_id, edge)

            if self.enable_nvtx:
                range_pop(synchronize=False)
                range_push("process_new_inputs.process_inputs")

            if normal:
                self._graph_runtime.ingest_inputs_batch(
                    self._one_rids_edges(request_id, normal), can_buffer=True,
                )
            if self.enable_nvtx:
                range_pop(synchronize=False)

    # ------------------------------------------------------------------
    # CPU offloading
    # ------------------------------------------------------------------

    def _try_offload_cold_request(
        self, node_name: str, batch_ids: set[int],
        affected_resources: set[str] | None= None
    ) -> int | None:
        """Offload one request's state for ``node_name``, freeing room to retry.

        Prefers a victim outside *batch_ids*; falls back to one inside it (the
        caller then excludes it from execution). Returns the victim, or None
        when nothing could be reclaimed.
        """
        engine = self.engine_manager.get_engine(node_name)
        if not engine.evictable(node_name):
            return None

        candidates = [
            rid for (rid, node) in self._last_active
            if node == node_name and not engine.is_offloaded(node_name, rid)
        ]

        # a request admitted but not yet run holds no pages: offloading it
        # frees nothing and the retry would re-pick it
        candidates = [
            rid for rid in candidates if engine.reclaimable(node_name, rid, affected_resources)
        ]

        if not candidates:
            # Nothing holds pages of the resource that ran out. Eviction is the
            # wrong tool here — the working set does not fit — and saying so
            # separates it from the host pool being full, which looks identical
            # from the caller and wants the opposite response.
            logger.warning(
                "No eviction candidate on %s for %s: nothing holds reclaimable "
                "pages of it. The working set does not fit; eviction cannot "
                "help.", node_name,
                "any resource" if affected_resources is None
                else ", ".join(sorted(affected_resources)),
            )
            return None

        # prefer evicting requests that aren't currently executing
        external = [rid for rid in candidates if rid not in batch_ids]
        victim_id = self._select_eviction_victim(node_name, external or candidates)
        freed = engine.offload_request(node_name, victim_id)
        if freed <= 0:
            # A victim was found and would not move. The usual cause is the host
            # pool being full (see ``CPUPagePool.offload_stream``) — different
            # from having no candidate, and fixed by a larger
            # ``cpu_offload_pages`` rather than by scheduling less.
            logger.warning(
                "Eviction victim %s on %s freed nothing — the host pool is "
                "most likely full. Raise cpu_offload_pages for the resource "
                "that ran out.", victim_id, node_name,
            )
            return None
        logger.info(
            "Offloaded request %s from %s (%d reclaimed, policy=%s, in_batch=%s)",
            victim_id, node_name, freed, self.eviction_policy.value,
            victim_id in batch_ids,
        )
        return victim_id

    def _select_eviction_victim(
        self, node_name: str, candidates: list[int]
    ) -> int:
        """Pick a victim from *candidates*.

        LRU only today; ``EvictionPolicy`` has no other member yet. Oldest
        last_active first — a candidate the worker has never run sorts oldest,
        which is what we want once the caller has filtered out the ones
        holding nothing.
        """
        return min(
            candidates,
            key=lambda rid: self._last_active.get((rid, node_name), 0.0),
        )

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_executing_batch(self, batch: ScheduledBatch) -> ExecutingBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_info: dict[int, CurrentForwardPassInfo] = {}
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
            batch.input_edges.to_input_tensors(
                self.tensor_manager.get_tensor, batch.request_to_worker_graph,
                skip_missing=True,
            )
        )
        for rid in unresolved:
            # Before the get_fwd_info loop below, so a rid that is going to be
            # failed anyway is not asked for forward-pass state as well.
            per_request_inputs.pop(rid, None)
            final_edges.pop(rid, None)
        per_request_input_metadata: dict[int, InputMetadata] = {}
        for request_id, inputs in per_request_inputs.items():
            per_request_info[request_id] = self.request_state.get_fwd_info(
                request_id, batch_partition
            )
            if chunks := self._stream_chunks_for(request_id, batch.node_name, inputs):
                per_request_input_metadata[request_id] = InputMetadata(stream_chunks=chunks)

        node_batch = self._make_executing_batch(
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
        )
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

    def _settle_final_streams(
        self, node_name: str, final_edges: dict[int, set[str]],
    ) -> tuple[set[int], set[int]]:
        """Record the streams whose final chunk this step consumes, per rid.

        Returns two sets, ``(node_done, partition_done)``: the rids whose
        streams into ``node_name`` have all ended once this step runs, and
        the rids whose streams into the partition have. The second is a
        subset of the first. Streams end in any order and over several
        steps, so one final chunk alone means only that its own stream ended. A
        ``continue_after_producer_done`` stream never sends a final chunk;
        it holds neither back.
        """
        node_done: set[int] = set()
        partition_done: set[int] = set()
        for rid, edges in final_edges.items():
            req_info = self.request_state.per_request_info.get(rid)
            if req_info is None:
                node_done.add(rid)
                partition_done.add(rid)
                continue
            req_info.ended_streams |= edges
            partition = self._stream_partition.get(next(iter(edges)))
            live = [
                edge_name for edge_name, sbuf in req_info.stream_buffers.items()
                if edge_name not in req_info.ended_streams
                and not sbuf.policy.continue_after_producer_done()
            ]
            if all(self._consumer_node_cache.get(e) != node_name for e in live):
                node_done.add(rid)
            if all(self._stream_partition.get(e) != partition for e in live):
                partition_done.add(rid)
        return node_done, partition_done

    def _make_executing_batch(
        self,
        node_name: str,
        graph_walk: str,
        request_ids: list[int],
        per_request_input_tensors: dict[int, NameToTensorList],
        per_request_info: dict[int, CurrentForwardPassInfo],
        final_edges: dict[int, set[str]] | None = None,
        per_request_input_metadata: dict[int, InputMetadata] | None = None,
    ) -> ExecutingBatch:
        """One step's batch, with the step context the engine drives it through.

        ``final_edges`` maps a rid to the streams whose final chunk the step
        consumes. The context starts unleased and eager; a slot is reserved
        later, once the real token count is known.
        """
        final_stream_rids, stream_partition_done_rids = self._settle_final_streams(
            node_name, final_edges or {},
        )
        return ExecutingBatch(
            node_name=node_name,
            per_request_info=per_request_info,
            per_request_input_tensors=per_request_input_tensors,
            final_stream_rids=final_stream_rids,
            stream_partition_done_rids=stream_partition_done_rids,
            per_request_input_metadata=per_request_input_metadata or {},
            step_context=StepContext(
                request_ids=tuple(request_ids),
                graph_walk=graph_walk,
                slot=0,
                capture=False,
            ),
        )

    def maybe_send_zmq_to_tp_followers(
        self, node_batch: ExecutingBatch,
        *, speculative: bool = False, spec_from_seq: int = -1,
    ) -> int:
        """Broadcast this batch as a ``ScheduleTPNode``. Returns its seq, or
        ``-1`` when this worker does not lead the node (nothing sent)."""
        if node_batch.node_name not in self.parallel_nodes or \
                node_batch.node_name not in self.parallel_leader_nodes:
            return -1
        seq = self._tp_broadcast_seq
        self._tp_broadcast_seq += 1
        # Drained once, outside the loop: every rank needs the same moves, so
        # draining per follower would give the first one everything and the rest
        # nothing. Each gets its own copy because each pops as it replays.
        engine = self.engine_manager.get_engine(node_batch.node_name)
        resident_delta = engine.take_resident_delta(node_batch.node_name)
        # this worker is only a part of one TP group for this node,
        # so, we can just look at the sharding_config for the first
        # request to get the relevant workers
        sample_rid = node_batch.request_ids[0]
        workers = self._graph_runtime.get_sharding_config(sample_rid).get_sharding_group(
            node_batch.node_name, node_batch.graph_walk
        )._workers[1:]
        for worker in workers:
            self.communicator.send(
                worker, msg=WorkerMessage(
                    message_type=WorkerMessageType.SCHEDULE_TP,
                    body=ScheduleTPNode(
                        node_name=node_batch.node_name,
                        graph_walk=node_batch.graph_walk,
                        request_ids=[self._rid_str(r) for r in node_batch.request_ids],
                        speculative=speculative,
                        spec_seq=seq,
                        spec_from_seq=spec_from_seq,
                        resident_delta=resident_delta.copy(),
                    )
                )
            )
        return seq

    def _broadcast_tp_nospec(self, pending: PendingBatch) -> None:
        """Tell followers no speculative head will follow step ``pending.tp_seq``,
        so each step they await settles on exactly one of {head, marker}."""
        # Not ``pending.node_batch.request_ids``: the GPU thread may be
        # ``drop_rids``-ing that list right now (empty at B=1 on a veto).
        sample_rid = next(iter(pending.batch.request_to_worker_graph))
        workers = self._graph_runtime.get_sharding_config(sample_rid).get_sharding_group(
            pending.node_name, pending.graph_walk
        )._workers[1:]
        for worker in workers:
            self.communicator.send(
                worker, msg=WorkerMessage(
                    message_type=WorkerMessageType.TP_NO_SPEC,
                    body=TPNoSpeculation(
                        node_name=pending.node_name,
                        graph_walk=pending.graph_walk,
                        spec_from_seq=pending.tp_seq,
                    )
                )
            )

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def _push_back_batch(self, batch: ScheduledBatch) -> None:
        """Return a whole batch's nodes to the ready set (admit refusal, OOM)."""
        rids = list(batch.request_to_worker_graph)
        self._graph_runtime.push_back_node(
            batch.node_name, rids,
            [batch.request_to_worker_graph[rid] for rid in rids],
        )

    @staticmethod
    def _edge_block(
        per_rid: Iterable[tuple[int, list[GraphEdge]]],
    ) -> ColumnarEdgeSpecs:
        """Arriving edges as columns, for the runtime's ingest.

        Only uuids cross: the runtime rebuilds the descriptors from the tensor
        store, which is why they are kept there. Filled straight from the edges
        -- an EdgeSpec per edge in between would cost as much as the columns
        save.
        """
        block = ColumnarEdgeSpecs.empty()
        for rid, edges in per_rid:
            for edge in edges:
                block.add_edge(rid, edge)
        return block

    @classmethod
    def _one_rids_edges(
        cls, rid: int, edges: list[GraphEdge],
    ) -> ColumnarEdgeSpecs:
        """``_edge_block`` for the common single-request case."""
        return cls._edge_block(((rid, edges),))

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

    def _stream_consumption(self, rid: int) -> dict[str, int]:
        req_info = self.request_state.per_request_info.get(rid)
        if req_info is None:
            return {}
        return {
            edge_name: sbuf._consumed
            for edge_name, sbuf in req_info.stream_buffers.items()
        }

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
    ):
        """Register the tensors the runtime says remote consumers (other
        workers, the api server, the conductor via persist) will read. It has
        already deduped them by uuid.
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
            return
        # One staging pass for the batch: the arena collapses what would be one
        # host-blocking D2H stream sync per request into one.
        self.tensor_manager.register_for_send_uuids(
            ParallelList(list(per_rid), list(per_rid.values())),
            skip_cuda_sync=True,
        )


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
        try:
            if pending is not None:
                # Safety timeout — the engine releases this event even on its
                # failure paths, so it should only fire if the GPU thread died.
                # Bail out rather than block plan_executor forever.
                with self._span("worker.plan_thread.await_commit"):
                    committed = pending.node_batch.commit_done.wait(timeout=10.0)
                if not committed:
                    logger.warning(
                        "Worker %s: plan_executor timed out waiting for "
                        "batch N commit; skipping pre-plan", self.worker_id,
                    )
                    return False
            with self._span("worker.plan_thread.reserve_slot"):
                leased = engine.reserve_replay_slot(spec_batch) is not None
            if not leased:
                return False  # eager, or no batched capture for this shape
            with self._span("worker.plan_thread.pre_plan"):
                return engine.pre_plan_for_batch(spec_batch)
        except Exception:
            logger.exception("Worker %s: plan_executor pre-plan failed", self.worker_id)
            self._reset_skip_plan_flags(spec_batch)
            return False

    def _reset_skip_plan_flags(self, spec_node_batch: ExecutingBatch) -> None:
        """Drop the pre-plan staged for ``spec_node_batch``.

        Used when the pre-plan was dispatched but the batch never reached the
        GPU thread: the resources would otherwise promote that stale plan into
        the next step that leases the same slot. Targeted at this batch's own
        lease, so a different slot's valid in-flight pre-plan isn't stomped.
        """
        engine = self.engine_manager.get_engine(spec_node_batch.node_name)
        engine.reset_pre_plan_for_batch(spec_node_batch)

    def _init_engine_thread(self) -> None:
        """Pin this executor thread to the worker's accelerator device.

        The CUDA current device is per-thread and defaults to 0. PyTorch
        ops carry per-tensor device guards, but raw Triton launches and
        bare ``torch.cuda.current_stream()`` / ``synchronize()`` calls
        resolve against the THREAD's device — on a worker whose model
        lives on a non-zero device, work issued from an unpinned thread
        lands on device 0's stream, unordered with the real compute.

        It also applies mstar's torch config, because dynamo config is
        per-thread since torch 2.12 and this thread compiles.
        """
        if self.device.type != "cpu" and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)
        apply_torch_config()

    @contextmanager
    def _span(self, name: str):
        """One NVTX range plus one MSTAR_PHASE_TIMING sample, same name.

        Safe off the main thread: the append is the only shared mutation and
        run()'s flush snapshots before it clears.
        """
        nvtx = self.enable_nvtx
        if nvtx:
            range_push(name, synchronize=False)
        t0 = _time.perf_counter() if self._phase_period else 0.0
        try:
            yield
        finally:
            if self._phase_period:
                self._phase_buf[name].append(_time.perf_counter() - t0)
            if nvtx:
                range_pop(synchronize=False)

    def _phase_record(self, name: str, dt: float) -> None:
        if self._phase_period > 0:
            self._phase_buf[name].append(dt)

    def _drain_gpu_spans(self) -> None:
        """Record the GPU times whose events have landed, leaving the rest:
        reading a pair still in flight would block the host on the GPU."""
        while self._gpu_spans:
            start, end = self._gpu_spans[0]
            if not end.query():
                return
            self._gpu_spans.popleft()
            # elapsed_time is ms; the phase buffer holds seconds
            self._phase_record("gpu_exec", start.elapsed_time(end) / 1000)

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

        engine = self.engine_manager.get_engine(batch.node_name)
        if logger.isEnabledFor(logging.DEBUG):
            # test/waypoint/serve_rollout.py parses this line (wire ids, logged
            # before prepare_inputs runs) to recover the DiT schedule.
            logger.debug(
                "Executing: %s graph_walk=%s %s", node_batch.node_name,
                batch.graph_walk, [self._rid_str(r) for r in node_batch.request_ids],
            )
        if self.enable_nvtx:
            range_push("worker.gpu_thread_start", synchronize=False)
            range_pop(synchronize=False)
        # The plan thread wrote this batch's plan into the resources; wait for
        # it before the forward reads them. Waiting releases the GIL — which is
        # the point: running prepare_inputs here instead just puts the two
        # threads in contention for it, and measured worse.
        if plan_future is not None:
            with self._span("worker.gpu_thread.await_plan"):
                plan_future.result()
        if self.enable_nvtx:
            range_push(
                f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                synchronize=False,
            )
        try:
            try:
                with self._span("worker.gpu_thread.prepare_inputs"):
                    engine.prepare_inputs(node_batch)
            except Exception:
                if plan_future is not None and node_batch.preplanned_rids is not None:
                    # The stage was for this batch, which now never runs.
                    engine.reset_pre_plan_for_batch(node_batch)
                raise
            # call is_stale after prepare_inputs because prepare_inputs may drop rids
            if plan_future is not None and engine.preplan_is_stale(node_batch):
                engine.reset_pre_plan_for_batch(node_batch)
            execution_stream = (
                torch.accelerator.current_stream(self.device)
                if self.device.type != "cpu"
                else None
            )
            # Device time of the step; the phase timers only see host submit
            # time, and both are needed to tell a starved iter from a busy one.
            gpu_start = None
            if self._phase_period and execution_stream is not None:
                gpu_start = torch.Event(enable_timing=True)
                gpu_start.record(execution_stream)
            if self._phase_period:
                # after prepare_inputs, which can drop rids — this is the row
                # count the forward really pays for
                self._phase_record(
                    f"rows.{batch.graph_walk}#", len(node_batch.request_ids),
                )
            with self._span("worker.gpu_thread.exec"):
                outputs = engine.exec_and_postprocess(node_batch)
            if execution_stream is not None:
                event = torch.Event()
                event.record(execution_stream)
                node_batch.completion_event = event
                if gpu_start is not None:
                    gpu_end = torch.Event(enable_timing=True)
                    gpu_end.record(execution_stream)
                    self._gpu_spans.append((gpu_start, gpu_end))
                    self._drain_gpu_spans()
            return outputs
        finally:
            # Safety net: a step that raised before the forward would otherwise
            # leave the submitter blocked on this for the full wait timeout.
            if node_batch.launch_started_event is not None:
                node_batch.launch_started_event.set()
            # Safety net: a step that raised before publishing outputs or
            # committing would otherwise strand the plan thread preparing the
            # next one. The engine does this too on its own paths; here covers
            # a raise outside them.
            node_batch.release_waiters()
            # Publish each resource's durable state onto per_request_info so
            # the next iter's prep and the conductor see it. Runs regardless
            # of success, allocation failure, or an uncaught raise —
            # finalize_batch reads whatever state the engine actually reached.
            publish_rids = self._publishable_request_ids(node_batch)
            node_batch.resource_publish_info = engine.finalize_batch(
                node_batch, publish_request_ids=publish_rids,
            )
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _await_admit_settled(self, pending: "PendingBatch | None") -> None:
        """Block until the in-flight step's admit has run (for TP leaders).

        Admit is on the GPU thread; every offload and reload is on this one. A
        page move in the window between a step's broadcast and its admit lands
        in the NEXT step's resident delta, so the follower applies it after
        admitting this step while the leader admitted with it already applied.

        Separate from the submitter's ``MSTAR_LAUNCH_WAIT_MS`` wait on the same
        event: that one is a GIL throttle and is meant to expire early. This
        one is for correctness, so it waits out the step.
        """
        if pending is None:
            return
        event = pending.node_batch.launch_started_event
        if event is None or event.is_set():
            return
        with self._span("worker.await_admit_settled"):
            settled = event.wait(timeout=_ADMIT_FENCE_TIMEOUT_S)
        if not settled:
            logger.warning(
                "Worker %s: timed out waiting for node=%s walk=%s to admit "
                "before scheduling; page moves may diverge across TP ranks",
                self.worker_id, pending.node_name, pending.graph_walk,
            )

    def _handle_admit_failure(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch,
        referenced_rids: frozenset[str] = frozenset(),
    ) -> None:
        """Re-queue a batch whose admit refused it, so the step can be retried.

        Every admit failure needs the push-back; only an ``AllocationFailed``
        also needs an eviction. ``RequestOffloading`` means the rid is already
        on its way to the host, so evicting anything else is wasted work — the
        retry is gated on ``check_ready`` reloading it.
        """
        reason = node_batch.admit_error
        if isinstance(reason, AllocationFailed):
            self._handle_allocation_failure(batch, node_batch, referenced_rids)
            return

        self._push_back_batch(batch)
        logger.info(
            "Admit refused node=%s walk=%s (%s): re-queued %d requests",
            batch.node_name, batch.graph_walk,
            type(reason).__name__, len(batch),
        )

    def _handle_allocation_failure(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch,
        referenced_rids: frozenset[str] = frozenset(),
    ) -> None:
        """Push back nodes and hold the rids for backoff after KV OOM.

        Under TP, this runs on every rank of the TP group independently:
        admission decisions (``add_request`` / ``alloc`` / ``free``) are
        all driven by rank 0's scheduler and replicated via the
        ``ScheduleTPNode`` ZMQ broadcast, so the page allocator state is
        symmetric across ranks. Both rank 0 and followers raise
        ``AllocationFailedError`` on the same batch and both reach this
        function with the same ``batch_ids``; their local actions
        (push-back, hold) produce identical follower state.

        ``KVCacheManager.post_warmup_validate`` fails fast at startup if
        that invariant ever breaks. TP async scheduling leans on the same
        symmetry: a follower voids a speculative head from its own verdict.

        Eviction is the one decision that is NOT derivable from the broadcast,
        since LRU orders on wall clock. So only rank 0 picks a victim; its choice
        rides out on the next ``ScheduleTPNode`` as a resident-set delta, and a
        follow rank returns below without evicting or holding anything. Holding
        in particular would be wrong rather than merely useless: a follow rank
        has no scheduling decision a backoff could improve, and rank 0 hit the
        same OOM on the same batch and is already driving the retry.
        """
        batch_ids = set(batch.request_to_worker_graph)

        # Push all batch nodes back to their queues
        self._push_back_batch(batch)

        # A teardown this rank has been holding releases pages, and spending one
        # can mean no eviction is needed at all. After the push-back above, so a
        # rid torn down here cannot be pushed onto queues that have just been dismantled.
        self._apply_removes_whose_step_landed()

        # ``referenced_rids``, not ``_in_flight_rids``: this step's admit refused,
        # so no forward ran and its own pages are fair game. Empty in practice —
        # nothing is speculated on a refusal — so it only guards a future caller
        # that broadcasts something here.
        self._apply_pending_removes_safe_to_drop(referenced_rids)

        if self._is_tp_follower_node(batch.node_name):
            return

        # scope the eviction to whichever resource actually ran out, when the
        # admit named one
        failed = node_batch.failed_resource
        victim_id = self._try_offload_cold_request(
            node_batch.node_name, batch_ids,
            affected_resources=None if failed is None else {failed},
        )

        if victim_id is not None:
            self.scheduler.hold_requests([victim_id])
            logger.warning(
                "OOM on node=%s walk=%s: offloaded victim=%s, "
                "retrying %d remaining requests",
                batch.node_name, batch.graph_walk, victim_id,
                len(batch_ids) - (1 if victim_id in batch_ids else 0),
            )
        else:
            # Nothing could be evicted, so the batch has to get smaller or it will
            # refuse identically for ever. Hold the one request the admit named,
            # not the whole batch, so the retry is immediate. Symmetric without
            # coordinating: both ranks refuse the same batch for the same request.
            blamed = getattr(node_batch.admit_error, "request_id", None)
            shed = [blamed] if blamed in batch_ids else list(batch_ids)
            self.scheduler.hold_requests(shed)
            key = (batch.node_name, batch.graph_walk)
            now = _time.monotonic()
            last, unlogged = self._hold_logged.get(key, (None, 0))
            if last is not None and now - last < _HOLD_LOG_INTERVAL:
                self._hold_logged[key] = (last, unlogged + 1)
                return
            self._hold_logged[key] = (now, 0)
            logger.warning(
                "OOM on node=%s walk=%s: no offload possible, holding %s of %d "
                "requests so the retry is smaller (%d earlier holds not logged)",
                batch.node_name, batch.graph_walk,
                blamed if len(shed) == 1 else f"all {len(shed)}",
                len(batch_ids), unlogged,
            )

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

    def _publishable_request_ids(self, batch: ExecutingBatch) -> list[int]:
        """Do not create a late KV snapshot for an aborting request.

        REMOVE_REQUEST is deferred while a GPU step is in flight. The
        publication on that step's completion must still exclude the rid.
        """
        return [
            rid for rid in batch.request_ids
            if (info := batch.per_request_info.get(rid)) is not None
            if rid not in self._pending_removes
            and rid not in self.scheduler.failed_rids
            and info.request_id not in self._pending_drains
            and info.request_id not in self._draining_rids
        ]

    def _can_speculate(self, batch: ScheduledBatch) -> bool:
        if not self._graph_runtime.is_async_schedulable(
            batch.node_name, batch.graph_walk
        ):
            return False
        if batch.node_name in self.parallel_nodes:
            # Only the leader initiates, only under TP async scheduling.
            return (
                self._tp_async_for(batch.node_name)
                and batch.node_name in self.parallel_leader_nodes
            )
        return True

    def _is_tp_follower_node(self, node_name: str) -> bool:
        """This rank follows ``node_name``: rank 0 of its instance decides both
        what runs on it and what gets evicted from it. Mirrors the engine's
        ``_tp_follower_nodes``, which refuses the eviction itself."""
        return (
            node_name in self.parallel_nodes
            and node_name not in self.parallel_leader_nodes
        )

    def _tp_async_for(self, node_name: str) -> bool:
        """TP async scheduling applies to this node (flag, narrowed by node list)."""
        return self.tp_async_sched and (
            self.tp_async_nodes is None or node_name in self.tp_async_nodes
        )

    def _verify_tp_async_sched_agrees(self) -> None:
        """Refuse a per-rank flag mismatch at startup: a follower would wait for
        a decision the leader never sends, or get a head it cannot build."""
        for node in sorted(self.parallel_nodes):
            local = torch.tensor(
                [int(self._tp_async_for(node))], dtype=torch.int64, device=self.device,
            )
            for group in (
                self.parallel_groups.get_tp_config_for_node(node),
                self.parallel_groups.get_sp_config_for_node(node),
            ):
                if group.world_size == 1:
                    continue
                values = group.all_gather(local, dim=0).cpu().tolist()
                if any(v != values[0] for v in values):
                    raise RuntimeError(
                        f"MSTAR_TP_ASYNC_SCHED disagrees across the ranks of {node!r} "
                        f"(ranks {group.group_members}: async={values}); set it "
                        "identically on every rank of the instance."
                    )

    def _is_tp_lead_pending(self, pending: PendingBatch) -> bool:
        """``pending`` is a parallel batch this worker leads under TP async."""
        return (
            self._tp_async_for(pending.node_name)
            and pending.node_name in self.parallel_nodes
            and pending.node_name in self.parallel_leader_nodes
        )

    def _tp_lead_needs_marker(
        self, pending: PendingBatch, speculation: Speculation | None,
    ) -> bool:
        """Leader owes followers a marker: no head went out for ``pending``
        (nothing speculated, or a non-parallel target, never broadcast)."""
        return self._is_tp_lead_pending(pending) and (
            speculation is None or speculation.tp_seq < 0
        )

    def _is_tp_follow_pending(self, pending: PendingBatch) -> bool:
        """``pending`` is a parallel batch this worker follows under TP async:
        the leader will send a head or a marker for it. Mirrors ``_can_speculate``."""
        return (
            self._tp_async_for(pending.node_name)
            and self.is_tp_follower
            and pending.node_name in self.parallel_nodes
            and pending.node_name not in self.parallel_leader_nodes
            and self._graph_runtime.is_async_schedulable(
                pending.node_name, pending.graph_walk
            )
        )

    def _speculative_fwd_info(
        self,
        request_ids: list[int],
        partition: str,
        spec_target: SpeculationOutput,
        continuing: Container[int],
    ) -> dict[int, CurrentForwardPassInfo]:
        """The step context a speculative batch runs each rid under.

        A request's shared info is refreshed as each batch is built, so while
        iteration k of a loop is in flight it reads k, and stays there until the
        step lands. A speculative next iteration that ran under it would take k
        again: a denoise step would run the same sigma twice, and the overshoot
        past the last step would never be vetoed. So the speculative batch gets a
        view of its own, with the counters it will run at: one past the in-flight
        iteration for a continuing rid on a new iteration of its loop, and the
        loops' current indices otherwise (a fresh rid's loop already advanced when
        its last step routed). The mutable per-request tables are shared, so what
        an engine records on the view is seen by later steps; the in-flight batch
        keeps the shared info, so its stop check still reads k.
        """
        per_request_info = {}
        for rid, new_iters in self._graph_runtime.get_dynamic_loop_iters(
            request_ids, partition=partition,
        ):
            info = self.request_state.get_fwd_info(rid, partition)
            counts = dict(info.dynamic_loop_iter_counts)
            counts.update(new_iters)
            if spec_target.is_new_loop_iter and rid in continuing:
                loop = spec_target.loop_name
                counts[loop] = counts.get(loop, 0) + 1
            per_request_info[rid] = replace(info, dynamic_loop_iter_counts=counts)
        return per_request_info

    def _assemble_speculation(
        self,
        pending: PendingBatch,
        spec_target: SpeculationOutput,
        request_to_worker_graph: dict[int, int],
        per_request_inputs: dict[int, NameToTensorList],
        consumed_streaming_edges: dict[int, list[StreamingEdge]],
        continuing: list[int],
        *,
        is_same_node: bool,
        spec_id: int = 0,
        tp_seq: int = -1,
    ) -> Speculation:
        """Package prepared rids (batch order) into the ``Speculation`` the main
        loop runs. Leader and follower differ only in how they pick the rids."""
        spec_node = spec_target.node_name
        request_ids = list(request_to_worker_graph)
        per_request_input_metadata = {
            rid: InputMetadata(stream_chunks=chunks) for rid in request_ids
            if (chunks := self._stream_chunks_for(rid, spec_node, per_request_inputs[rid]))
        }
        spec_batch = ScheduledBatch(
            node_name=spec_node,
            graph_walk=pending.graph_walk,
            request_to_worker_graph=request_to_worker_graph,
            # Reported by whichever step chose the target -- speculate_node on
            # the leader, get_spec_target on a follower -- so the hot path does
            # not cross back into the runtime once per forward pass.
            output_signals=spec_target.output_signals,
            tp_seq=tp_seq,
        )
        spec_node_batch = self._make_executing_batch(
            node_name=spec_node,
            graph_walk=pending.graph_walk,
            request_ids=request_ids,
            per_request_input_tensors=per_request_inputs,
            per_request_info=self._speculative_fwd_info(
                request_ids, pending.partition, spec_target, set(continuing),
            ),
            final_edges={
                rid: {se.edge.name for se in edges if se.chunk.is_final}
                for rid, edges in consumed_streaming_edges.items()
                if any(se.chunk.is_final for se in edges)
            },
            per_request_input_metadata=per_request_input_metadata,
        )
        return Speculation(
            scheduled_batch=spec_batch,
            node_batch=spec_node_batch,
            # Outputs of batch_N the spec batch consumes: every edge of the
            # in-flight node whose destination is the spec node.
            consumed_edges=self._graph_runtime.get_consumed_edges(
                pending.node_name, spec_node, pending.graph_walk,
            ),
            continuing_rids=set(continuing),
            partition=pending.partition,
            is_new_iter=spec_target.is_new_loop_iter,
            is_same_node=is_same_node,
            loop_name=spec_target.loop_name,
            consumed_streaming_edges=consumed_streaming_edges,
            spec_id=spec_id,
            tp_seq=tp_seq,
        )

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
        (``_clear_speculative_flag``), on paths with no stage to settle.
        """
        batch = speculation.scheduled_batch
        rids = list(batch.request_to_worker_graph) if success else []
        self._graph_runtime.commit_speculation(
            speculation.spec_id, success, list(dropped_rids),
            # Keeps these nodes off the ready queue while the step is in flight.
            node=batch.node_name if rids else None,
            wg_id=batch.request_to_worker_graph[rids[0]] if rids else None,
            scheduled_rids=rids,
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
                self._return_streaming_edge(rid, se)
            speculation.consumed_streaming_edges.pop(rid, None)
        # The kept rids' chunks now belong to the step; nothing is left to undo.
        speculation.spec_id = 0
        speculation.consumed_streaming_edges.clear()

    def _is_tearing_down(self, rid: int) -> bool:
        """Removed, aborted or failed: no further speculative work for ``rid``.

        A deferred drain only fires once the rid leaves ``_in_flight_rids``,
        and a same-node speculation chain keeps it there every step, so a
        drain the chain does not see would never fire and the rollout would
        run to ``max_iters`` for a client that has gone.

        ``rid`` is the worker handle; ``_pending_drains``/``_draining_rids``
        hold wire strings.
        """
        if rid in self._pending_removes or rid in self.scheduler.failed_rids:
            return True
        wire = self._rid_str(rid)
        return wire in self._pending_drains or wire in self._draining_rids

    def _try_speculate_next(
        self,
        pending: PendingBatch
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
        """
        batch_N = pending.batch
        graph_walk = pending.graph_walk

        # sample node and RID to see which node we will be speculating
        # (TODO: refine this to be, e.g., a majority vote)
        rid = next(iter(batch_N.request_to_worker_graph))

        # The runtime applies the async/TP-async eligibility filter; the
        # loop-completion filter is per rid and lives in prep_spec_rids.
        ready_for_spec = self._graph_runtime.speculate_node(
            batch_N.node_name, graph_walk, rid,
        )
        if not ready_for_spec:
            return # no nodes can be speculated

        # TODO: use the microscheduler to break ties when ready_for_spec
        # contains multiple ready nodes
        spec_target_info = ready_for_spec[0]
        spec_node_name = spec_target_info.node_name
        speculating_same_node = spec_node_name == batch_N.node_name

        new_request_to_worker_graph: dict[int, int] = {}
        per_request_inputs: dict[int, NameToTensorList] = {}
        consumed_streaming_edges: dict[int, list[StreamingEdge]] = {}
        # Backlogged rids for this target get first claim on the batch: they
        # have already waited a step, and the chain only ever continues its own
        # rids, so at the cap they would never be reached. None => uncapped.
        spec_target = (spec_node_name, batch_N.graph_walk)
        # The merge below takes only this chain's capture group; another
        # group's backlog runs only once the chain yields to it.
        if self.scheduler.backlog_splits_from(self.request_state, spec_target, rid):
            return None
        max_continuing = self.scheduler.room_for_continuing(spec_target)

        # Removes, aborts and failures are filtered here; prep_spec_rids
        # assumes that.
        candidates = [
            r for r in batch_N.request_to_worker_graph if not self._is_tearing_down(r)
        ]
        # Polling the StreamBuffers stays on this side: they hold real tensors.
        polled: list[tuple[int, StreamingEdge]] = []
        per_rid_counts: list[int] = []
        for r in candidates:
            edges = self._poll_stream_buffers_for_speculation(r, spec_node_name)
            polled.extend((r, e) for e in edges)
            per_rid_counts.append(len(edges))

        prep = self._graph_runtime.prep_spec_rids(SpeculationPrepInput(
            spec_node_name=spec_node_name,
            curr_node_name=batch_N.node_name,
            graph_walk=graph_walk,
            rids=candidates,
            room_for_continuing=max_continuing,
            streaming_edges=[
                EdgeSpec(
                    signal=e.name, next_node=e.next_node,
                    uuids=[i.uuid for i in e.tensor_info],
                    is_final_streaming_chunk=e._final_stream_chunk,
                ) for _r, (e, _chunk) in polled
            ],
            streaming_edges_per_rid=per_rid_counts,
        ))

        # Anything not consumed goes back to its StreamBuffer, so a later
        # scheduling of this node picks it up normally.
        consumed = set(prep.consumed_streaming_edge_idxs)
        for i, (r, se) in enumerate(polled):
            if i not in consumed:
                self._return_streaming_edge(r, se)
            else:
                self._mark_stream_ingested(r, se)
                consumed_streaming_edges.setdefault(r, []).append(se)

        continuing = set(prep.ready_rids)
        # `ready_rids` seeds it, so a ready rid with no prepped edge still gets
        # an empty mapping -- what slicing a zero-length run used to give.
        prepped_inputs = prep.input_edges.to_input_tensors(
            self.tensor_manager.get_tensor, prep.ready_rids,
        ).by_rid
        for i, r in enumerate(prep.ready_rids):
            per_request_inputs[r] = prepped_inputs[r]
            new_request_to_worker_graph[r] = prep.wg_ids[i]

        if not continuing:
            return None

        # Merge in fresh rids whose spec-target node is ready right now
        # Speculation only consumes work compatible with the spec target. In
        # partitioned models, unrelated ready work stays queued for
        # the normal scheduler path.
        fresh_batch = self.scheduler.get_next_batch(
            self.request_state,
            target=spec_target,
            pre_existing_batch_size=len(continuing),
            capture_group_of=prep.ready_rids[0],
        )

        if fresh_batch is not None:
            # The merge below relabels these node objects with the spec
            # target's name/walk, so a batch for any other node must not be
            # merged in.
            assert fresh_batch.node_name == spec_node_name, (
                f"Speculation asked for {spec_node_name!r} but the "
                f"scheduler returned {fresh_batch.node_name!r}"
            )
            fresh_inputs = fresh_batch.input_edges.to_input_tensors(
                self.tensor_manager.get_tensor,
                fresh_batch.request_to_worker_graph,
            ).by_rid
            for rid in fresh_batch.request_to_worker_graph:
                if rid in continuing:
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

                per_request_inputs[rid] = fresh_inputs[rid]
                new_request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )

        logger.debug("Speculating: %s %s", spec_node_name, continuing)
        return self._assemble_speculation(
            pending, spec_target_info,
            new_request_to_worker_graph, per_request_inputs,
            consumed_streaming_edges, continuing,
            is_same_node=speculating_same_node, spec_id=prep.spec_id,
        )

    def _thread_outputs_to_speculative(
        self, speculation: Speculation,
        outputs_N: dict[int, NameToTensorList],
    ):
        """Splice batch N's outputs into the spec batch's inputs.

        Runs on the plan thread as soon as N's outputs are published, so it
        must not read a tensor VALUE: it only moves tensor lists and tests
        which keys are present. N's kernels may still be running.
        """
        threaded_continuing: set[int] = set()
        dropped: set[int] = set()
        for rid in list(speculation.node_batch.request_ids):
            if rid not in speculation.continuing_rids:
                continue  # fresh rid — inputs already gathered.
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
            speculation.node_batch.request_ids = [
                r for r in speculation.node_batch.request_ids if r not in dropped
            ]
            for r in dropped:
                speculation.node_batch.per_request_input_tensors.pop(r, None)
                speculation.node_batch.per_request_info.pop(r, None)
                speculation.scheduled_batch.request_to_worker_graph.pop(r, None)
                speculation.node_batch.per_request_input_metadata.pop(r, None)
                # Its final chunks go back below; it must not flush or report done.
                speculation.node_batch.final_stream_rids.discard(r)
                speculation.node_batch.stream_partition_done_rids.discard(r)
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
        head = self.scheduler.peek_tp_follow()
        if head is None or not head.speculative:
            return None
        if head.spec_from_seq != pending.tp_seq:
            # From some other step: the serial path gets it in FIFO order.
            return None
        if head.node_name != pending.node_name or head.graph_walk != pending.graph_walk:
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
        if not self.scheduler.settle_tp_follow_delta():
            return None

        batch_N = pending.batch
        rid0 = next(iter(batch_N.request_to_worker_graph))
        # The leader already chose the target; this just reports its loop
        # context. speculate_node's eligibility filter cannot run here -- it
        # requires the node be a parallel LEADER node.
        spec_target_info = self._graph_runtime.get_spec_target(
            batch_N.node_name, head.node_name, head.graph_walk, rid0,
        )
        if spec_target_info is None:
            return None

        # The leader names its rids by wire string; these are this rank's handles.
        head_rids = self.scheduler.tp_rids(head)
        continuing = [r for r in head_rids if r in batch_N.request_to_worker_graph]
        fresh = [r for r in head_rids if r not in batch_N.request_to_worker_graph]

        # Polling the StreamBuffers stays on this side: they hold tensors.
        polled: list[tuple[int, StreamingEdge]] = []
        per_rid_counts: list[int] = []
        for r in continuing:
            edges = self._poll_stream_buffers_for_speculation(r, head.node_name)
            polled.extend((r, e) for e in edges)
            per_rid_counts.append(len(edges))

        # Fresh rids are popped BEFORE the continuing prep, so the only undo
        # a failure needs is push_back_node. The reverse order leaves a
        # successful all-or-nothing prep to unwind, with no API for it.
        popped = self.scheduler.pop_ready_rids(
            self.request_state, head.node_name, head.graph_walk, fresh,
        )

        def _return_all_polled() -> None:
            for r, se in polled:
                self._return_streaming_edge(r, se)

        if popped is None:
            # ``pop_ready_rids`` scans with ``allow_reload=False``: an offloaded
            # rid reads not-ready and waits for a replayed delta. The delta is
            # settled above, so a rid still off-device is a divergence, not a lag.
            _return_all_polled()
            return None

        # output_signals are handled by _assemble_speculation
        fresh_wg, fresh_edges, _ = popped

        # The leader's list is the composition every rank runs: all-or-nothing,
        # no local finished-rid skipping, no room_for_continuing cap.
        prep = self._graph_runtime.prep_follow_spec_rids(SpeculationPrepInput(
            spec_node_name=head.node_name,
            curr_node_name=batch_N.node_name,
            graph_walk=head.graph_walk,
            rids=continuing,
            room_for_continuing=None,
            streaming_edges=[
                EdgeSpec(
                    signal=e.name, next_node=e.next_node,
                    uuids=[i.uuid for i in e.tensor_info],
                    is_final_streaming_chunk=e._final_stream_chunk,
                ) for _r, (e, _chunk) in polled
            ],
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
                self._mark_stream_ingested(r, se)
                consumed_streaming_edges.setdefault(r, []).append(se)
            else:
                self._return_streaming_edge(r, se)

        # Both seeded with their rid lists, so the `in prep_inputs` test below
        # still separates prepped rids from fresh ones -- a ready rid with no
        # prepped edge gets an empty mapping, as a zero-length slice used to.
        get_tensor = self.tensor_manager.get_tensor
        prep_inputs = prep.input_edges.to_input_tensors(
            get_tensor, prep.ready_rids,
        ).by_rid
        fresh_inputs = fresh_edges.to_input_tensors(
            get_tensor, list(fresh_wg),
        ).by_rid
        prep_wg = dict(zip(prep.ready_rids, prep.wg_ids, strict=True))

        new_request_to_worker_graph: dict[int, int] = {}
        per_request_inputs: dict[int, NameToTensorList] = {}
        for rid in head_rids:  # wire order == the leader's batch order
            if rid in prep_inputs:
                new_request_to_worker_graph[rid] = prep_wg[rid]
                per_request_inputs[rid] = prep_inputs[rid]
            else:
                new_request_to_worker_graph[rid] = fresh_wg[rid]
                per_request_inputs[rid] = fresh_inputs[rid]

        # Committed: the serial path must not see the head any more.
        self.scheduler.pop_tp_follow_head()
        # Its rids count as in flight now, so a remove landing before submit is
        # deferred like any other; ``_set_pending`` re-derives the set on submit.
        self._in_flight_rids |= set(head_rids)
        logger.debug(
            "Follow-speculating: %s %s (seq %d from %d)",
            head.node_name, head_rids, head.spec_seq, head.spec_from_seq,
        )
        return self._assemble_speculation(
            pending, spec_target_info,
            new_request_to_worker_graph, per_request_inputs,
            consumed_streaming_edges, continuing,
            is_same_node=True, spec_id=prep.spec_id, tp_seq=head.spec_seq,
        )

    # How many "no speculative head from step s" seqs a follower remembers.
    _TP_NOSPEC_KEEP = 1024

    def _register_tp_nospec(self, message: TPNoSpeculation) -> None:
        """The leader sends no head from its step ``spec_from_seq``."""
        if self._tp_async_for(message.node_name):
            self._tp_nospec.add(message.spec_from_seq)

    def _register_tp_follow(self, message: ScheduleTPNode) -> None:
        """Queue a ScheduleTPNode; drop a head for a step this rank closed."""
        if not message.request_ids:
            logger.warning(
                "Worker %s: dropped empty ScheduleTPNode for %s/%s (seq %d)",
                self.worker_id, message.node_name, message.graph_walk,
                message.spec_seq,
            )
            return
        if (
            self._tp_async_for(message.node_name) and message.speculative
            and message.spec_from_seq in self._tp_nospec
        ):
            logger.debug(
                "Worker %s: dropped speculative head seq %d (parent %d closed)",
                self.worker_id, message.spec_seq, message.spec_from_seq,
            )
            return
        self.scheduler.register_tp_follow(message)

    def _close_tp_follow_step(self, pending: PendingBatch) -> None:
        """This step's forward raised (symmetrically on the leader, which never
        submitted its head): drop a queued head from it, and any arriving later."""
        self._tp_nospec.add(pending.tp_seq)
        head = self.scheduler.peek_tp_follow()
        if head is not None and head.speculative and head.spec_from_seq == pending.tp_seq:
            self.scheduler.pop_tp_follow_head()

    @staticmethod
    def _step_voids_head(pending: PendingBatch) -> str | None:
        """Why N's verdict voids a head built on it, or ``None``. Both fields are
        final once N's future is done; the leader clears on exactly these two."""
        if pending.node_batch.admit_error is not None:
            return "admit_error"
        if pending.node_batch.failed_requests:
            return "failed rids"
        return None

    def _await_tp_follow_step(
        self, pending: PendingBatch, arm: Callable[[Speculation], None],
    ) -> tuple[dict[int, NameToTensorList] | None, Speculation | None]:
        """Wait for step N and settle the leader's decision about N+1. Returns
        ``(outputs or None, armed head or None)``. Once N is done this never
        returns without a decision: going serial early strands the leader."""
        s = pending.tp_seq
        outputs: dict[int, NameToTensorList] | None = None
        t_done = last_warn = 0.0
        while True:
            if outputs is None and pending.future.done():
                outputs = pending.future.result()
                t_done = last_warn = _time.perf_counter()
            if s in self._tp_nospec:
                return outputs, None
            head = self.scheduler.peek_tp_follow()
            if head is not None and head.speculative and head.spec_from_seq == s:
                void = self._step_voids_head(pending) if outputs is not None else None
                if void is not None:
                    # The leader clears its speculation on this verdict too.
                    self.scheduler.pop_tp_follow_head()
                    logger.debug(
                        "Worker %s: dropped speculative head seq %d (parent %d %s)",
                        self.worker_id, head.spec_seq, s, void,
                    )
                    return outputs, None
                spec = self._try_follow_speculation(pending)
                if spec is not None:
                    arm(spec)
                    return outputs, spec
                # Not buildable yet (fresh rid / stream chunk): poll and retry.
            elif head is not None and head.spec_seq > s:
                # Per-peer FIFO: a later message with no decision for s means
                # none is coming (flag off on the leader). Go serial.
                if not self._tp_leader_gap_warned:
                    self._tp_leader_gap_warned = True
                    logger.warning(
                        "Worker %s: leader moved past step seq %d without a "
                        "speculation decision (front: seq %d); treating as "
                        "no-spec. Is MSTAR_TP_ASYNC_SCHED set on every rank?",
                        self.worker_id, s, head.spec_seq,
                    )
                self._tp_nospec.add(s)
                return outputs, None
            self.communicator.wait_for_work(20)
            self._process_messages()
            self._check_ready_tensors()
            self._poll_stream_buffers()
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
    def _set_speculative_flag(self, batch: ScheduledBatch, value: bool) -> None:
        rids = list(batch.request_to_worker_graph)
        if not rids:
            return
        self._graph_runtime.set_speculatively_scheduled(
            batch.node_name, batch.request_to_worker_graph[rids[0]],
            rids, value,
        )

    def _clear_speculative_flag(self, batch: ScheduledBatch) -> None:
        self._set_speculative_flag(batch, False)


    def _postprocess_batch(
        self, batch_N: PendingBatch,
        outputs: BatchedModelOutput,
    ):
        # Stage stopwatch named like the NVTX ranges below; a clock stamp, not
        # a nested `_span`, because of the two early returns.
        _pp_t = _time.perf_counter() if self._phase_period else 0.0
        def _pp_stage(name: str) -> None:
            nonlocal _pp_t
            if not self._phase_period:
                return
            now = _time.perf_counter()
            self._phase_buf[f"worker.postprocess.{name}"].append(now - _pp_t)
            _pp_t = now

        if self.enable_nvtx:
            range_push("worker.postprocess.cleanup_inputs", synchronize=False)

        rids = list(batch_N.batch.request_to_worker_graph)
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
        valid_rids = set(batch_N.node_batch.request_ids)
        if batch_N.speculative_new_iter:
            stopped_rids = self._graph_runtime.pending_loop_stop_rids(
                batch_N.graph_walk, batch_N.loop_name,
            ) & set(batch_N.node_batch.request_ids)
            for stopped_rid in stopped_rids:
                outputs.pop(stopped_rid, None)
                valid_rids.discard(stopped_rid)
                batch_N.batch.request_to_worker_graph.pop(stopped_rid, None)
                batch_N.node_batch.per_request_info.pop(stopped_rid, None)
        # keep the forward's order: other code walks this list positionally
        batch_N.node_batch.request_ids = [
            rid for rid in batch_N.node_batch.request_ids if rid in valid_rids
        ]
        if not valid_rids:
            range_pop(synchronize=False)
            return

        # pending stops are only needed for one iteration, so can be cleared now
        self._graph_runtime.clear_pending_loop_stops()

        # An engine can drop rids that were skipped during execution (a
        # submodule's prepare_inputs returned None) from node_batch.request_ids,
        # but it cannot reach the worker-side ScheduledBatch. Reconcile it here so
        # the routing/output loops below only touch rids that produced outputs.
        for rid in list(batch_N.batch.request_to_worker_graph):
            if rid not in valid_rids:
                batch_N.batch.request_to_worker_graph.pop(rid, None)

        _pp_stage("pending_loop_stops")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.update_lru", synchronize=False)

        # Update LRU
        t = _time.monotonic()
        for rid in batch_N.node_batch.request_ids:
            self._last_active[(rid, batch_N.node_name)] = t

        _pp_stage("update_lru")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.synchronize_completion_event", synchronize=False)

        # Wait for batch N's completion event before proceeding
        # TODO: may need to refine this based on how it affects performance?
        if self.device.type != "cpu" and batch_N.batch.request_to_worker_graph:
            if batch_N.node_batch.completion_event is not None:
                with self._span("worker.postprocess.event_sync"):
                    batch_N.node_batch.completion_event.synchronize()
            else:
                with self._span("worker.postprocess.device_sync"):
                    torch.accelerator.synchronize(self.device)

        if self.enable_prof:
            batch_N.node_batch.exec_timings.fwd_end = time.perf_counter()

        _pp_stage("completion_event_sync")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.check_stop", synchronize=False)

        per_request_info = batch_N.node_batch.per_request_info
        for rid, new_iters in self._graph_runtime.get_dynamic_loop_iters(
            list(per_request_info), partition=batch_N.partition,
        ):
            per_request_info[rid].dynamic_loop_iter_counts.update(new_iters)

        # Check for stops. Prematerialising pulls sampled tokens to the host,
        # so this can carry a device transfer as well as the stop logic.
        _t_stop = _time.perf_counter() if self._phase_period else 0.0
        engine = self.engine_manager.get_engine(batch_N.node_name)
        cpu_outputs, host_rows = self._prematerialize_for_check_stop(
            outputs, batch_N.node_batch.completion_event,
            request_ids=batch_N.node_batch.request_ids,
        )
        _pp_stage("prematerialize")
        stops = engine.check_stop_for_batch(
            batch_N.node_batch, cpu_outputs, host_rows=host_rows,
        )
        if self._phase_period:
            self._phase_record(
                "worker.postprocess.check_stop", _time.perf_counter() - _t_stop,
            )
        # the same host copy, before stops, so a request ending here still indexes its pages
        engine.extend_prefix_chains(batch_N.node_batch, cpu_outputs)

        # Stream-terminated loop: a stream-consuming node inside a loop has no
        # internal stop signal (unlike a self-EOS loop), so when it consumes the
        # terminal chunk of its stream (``final_stream_rids``: the StreamBuffer
        # popped ``is_final``, which a ``continue_after_done`` policy never
        # sets) end its innermost enclosing loop. Without this the loop would
        # spin to ``max_iters`` and the partition would never report done.
        loop_name = self._innermost_loop_by_node.get(
            (batch_N.graph_walk, batch_N.node_name)
        )
        if loop_name is not None:
            for rid in batch_N.node_batch.final_stream_rids:
                stops.setdefault(rid, set()).add(loop_name)

        if batch_N.node_batch.failed_requests:
            # A rid whose stop check raised has no trustworthy stop decision:
            # routing it would either run its loop forever or end it early.
            # Fail it here and take it out of the batch before the routing
            # loops below touch it. Only ever the ones `check_stop_for_batch`
            # just added: the caller already reported (and cleared) the rids
            # that failed in prepare_inputs / postprocess.
            failed = dict(batch_N.node_batch.failed_requests)
            self._drop_failed_rids(batch_N, outputs, failed)
            self._fail_requests(failed)
            if not batch_N.node_batch.request_ids:
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                return

        _pp_stage("check_stop")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.stop_loops", synchronize=False)


        # Stop loops, if applicable. The runtime filters rids whose walk does
        # not contain the loop, snapshots the stop times, records the pending
        # stops and fans out to peers.
        # Main loop only: the runtime holds `&mut self` across its GIL
        # release, so a concurrent caller gets "Already mutably
        # borrowed" rather than blocking.
        stopped_rids = []
        if stops:
            stopped_rids = self._graph_runtime.stop_loops_batched(
                partition=batch_N.partition,
                graph_walk=batch_N.graph_walk,
                last_node_run=batch_N.node_name,
                loop_names=ParallelList(
                    list(stops), [list(v) for v in stops.values()],
                ),
            )

        # Ordinary publication precedes stop detection on the GPU thread.
        # Export final-only state now, after the last iteration committed and
        # before routing reports the completed loop to the conductor.
        if stopped_rids:
            engine.finalize_stopped_requests(batch_N.node_batch, stopped_rids)

        # CurrentForwardPassInfo also contains publication inherited from peer
        # ranks. Buffer only what this worker produced for the conductor.
        for rid in batch_N.node_batch.request_ids:
            self.request_state.buffer_publish_info(
                rid,
                batch_N.partition,
                batch_N.node_batch.resource_publish_info.get(rid, {}),
            )

        _pp_stage("stop_loops")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.route_outputs", synchronize=False)
        # Store this batch's output tensors, then hand the runtime their
        # uuids. The store keeps the descriptors, so routing needs no
        # TensorPointerInfo objects.
        rids = list(batch_N.batch.request_to_worker_graph)
        # Recorded by the pop that scheduled this batch, not asked for again --
        # and taken from the GRAPH, so a model returning a tensor under a name
        # no edge carries cannot change what gets routed. Stale outputs are
        # dropped by complete_and_route_batch itself.
        signals = batch_N.batch.output_signals
        # One store call for the batch rather than one per request, and the
        # flat columns come back already built: the manager fills them as it
        # mints, so nothing is keyed by request and signal only to be taken
        # apart again here.
        _t_store = _time.perf_counter() if self._phase_period else 0.0
        # Pass the stop check's host copies so a host-memory transport needn't
        # copy the rows again. They are views of pinned buffers the next step
        # reuses; safe because the sends below are their last reader.
        stored = self.tensor_manager.store_and_return_tensor_info_batch(
            rids, outputs, signals,
            node_name=batch_N.node_name,
            graph_walk=batch_N.graph_walk,
            skip_cuda_sync=True,
            cpu_tensors=cpu_outputs,
        )
        flat_uuids = stored.flat_uuids
        flat_rids = stored.flat_rids
        signal_idxs = stored.signal_idxs
        num_tensors = stored.num_tensors
        # Safety hold: ref=1 until the real fanout is known, which
        # complete_and_route_batch settles. One call for the batch rather than
        # one per tensor -- with a Rust bookkeeper each is a boundary crossing,
        # and a 128-request batch has hundreds of them. The count is uniform,
        # so it crosses as a scalar rather than a list built per batch.
        self.tensor_manager.increment_ref_batch_uniform(flat_uuids, 1)
        if self._phase_period:
            self._phase_record(
                "worker.postprocess.store_tensors",
                _time.perf_counter() - _t_store,
            )

        # The graph runtime's own share of postprocess: the routing call.
        _t_route = _time.perf_counter() if self._phase_period else 0.0
        route_output = self._graph_runtime.complete_and_route_batch(
            RouteInput(
                partition=batch_N.partition,
                graph_walk=batch_N.graph_walk,
                node_name=batch_N.node_name,
                output_signals=signals,
                wg_ids=ParallelList(
                    rids,
                    [batch_N.batch.request_to_worker_graph[r] for r in rids],
                ),
                tensors=flat_uuids,
                num_tensors=num_tensors,
            ),
        )
        if self._phase_period:
            self._phase_record(
                "worker.postprocess.route", _time.perf_counter() - _t_route,
            )

        # Normally empty: cleanup_consumed_inputs ran above and took them. Not
        # empty if a completion ever precedes it, and then nobody else will.
        if route_output.freed_inputs.uuids:
            self.tensor_manager.cleanup_collectable(*route_output.freed_inputs)

        _pp_stage("route_outputs")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.register_outputs", synchronize=False)
        with self._span("worker.postprocess.register_outputs"):
            self._register_outputs(route_output)
        _pp_stage("register_outputs")

        # send outputs
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.send_outputs", synchronize=False)

        # The consuming pass (not the earlier ingest) reports the partition
        # done, so it rides this pass's WGD with the final output loop index.
        for rid in batch_N.node_batch.stream_partition_done_rids:
            self._graph_runtime.mark_stream_partition_done(rid, batch_N.partition)

        # set this before send_outputs so that we can send updated profiling info to the conductor
        if self.enable_prof:
            self.profile_info.register_end(
                batch_N.node_batch.node_name,
                batch_N.node_batch.graph_walk,
                batch_N.node_batch.request_ids,
                batch_N.node_batch.exec_timings,
            )

        # Before the local-streaming dereference below: numel() needs the
        # tensors, and a new_token whose only other destination is a local
        # stream holds a single reference, which that dereference drops to zero
        # -- collecting the tensor this count is about to read.
        new_token_counts = self._count_new_tokens(
            route_output.new_token_output_idxs,
            flat_rids, flat_uuids, signals, signal_idxs,
        )

        # Local streaming stays here: a StreamBuffer holds real tensors, so it
        # cannot move behind the runtime's contract.
        streamed: list[int] = []
        for signal, per_signal in route_output.local_streaming_by_signal.items():
            for rid, uuid in per_signal:
                req_info = self.request_state.per_request_info[rid]
                stream_buf = req_info.stream_buffers[signal]
                stream_buf.pre_read_register(uuid)
                tensor = self.tensor_manager.get_tensor(uuid)
                stream_buf.put(uuid, tensor.clone())
                streamed.append(uuid)
        # After the loop: one crossing for the batch rather than one per
        # streamed tensor.
        self.tensor_manager.dereference_batch_uniform(streamed)

        send_rids = list(rids)
        # Timed apart from the send: these comprehensions are per rid, and
        # building them is the worker's cost, not the runtime's.
        _t_prep = _time.perf_counter() if self._phase_period else 0.0
        needs_info = route_output.rids_needing_request_info
        info_rids = (
            send_rids if needs_info is None
            else [rid for rid in send_rids if rid in needs_info]
        )
        send_input = SendInput(
            completion_id=route_output.completion_id,
            per_request_info=ParallelList(
                info_rids,
                [
                    self.request_state.get_fwd_info(rid, batch_N.partition)
                    for rid in info_rids
                ],
            ),
            new_token_counts=ParallelList(
                send_rids,
                [new_token_counts.get(rid, {}) for rid in send_rids],
            ),
            stream_tokens_consumed=ParallelList(
                send_rids,
                [self._stream_consumption(rid) for rid in send_rids],
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
        )
        _t_send = _time.perf_counter() if self._phase_period else 0.0
        if self._phase_period:
            self._phase_record("worker.postprocess.send_prep", _t_send - _t_prep)
        # Main loop only: the runtime holds `&mut self` across its GIL
        # release, so a concurrent caller gets "Already mutably
        # borrowed" rather than blocking.
        completed_rids = self._graph_runtime.send_outputs(send_input)
        for rid in completed_rids:
            self.request_state.flush_publish_info(rid, batch_N.partition)
        if self._phase_period:
            self._phase_record(
                "worker.postprocess.send", _time.perf_counter() - _t_send,
            )

        _pp_stage("send_outputs")
        if self.enable_nvtx:
            range_pop(synchronize=False)

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
        self, buffers: dict, side: torch.cuda.Stream,
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
        with torch.cuda.stream(side):
            for index, (name, tensor) in enumerate(buffers.items()):
                if not (torch.is_tensor(tensor) and tensor.is_cuda):
                    host[name] = tensor
                    continue
                buf = self._get_pinned_d2h_buffer(
                    "check_stop_batched", tensor.shape, tensor.dtype, index,
                )
                buf.copy_(tensor, non_blocking=True)
                host[name] = buf
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
        completion_event: torch.cuda.Event | None,
        request_ids: list[int] | None = None,
    ) -> tuple[dict[int, NameToTensorList], HostRows | None]:
        """Side-stream D→H of every CUDA tensor in ``outputs`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns per-rid outputs with the CUDA tensors replaced by CPU
        copies, plus ``HostRows`` when the submodule handed over row-addressed
        buffers, for a batched stop check. Skipped (returns ``outputs`` unchanged) when there's no completion
        event (CPU execution) or when CUDA is unavailable.

        AR engines emit small per-rid output dicts (sampled token + maybe
        a code) so the cost is negligible. If a future engine emits large
        tensors here (e.g. activations), revisit.

        The host tensors are views of pinned buffers the next step reuses. Their
        one other reader, a ``needs_cpu_tensor`` transport, gets them through
        the store and sends in ``_register_outputs`` later in the same
        ``_postprocess_batch``, never reading them after.
        """
        source = outputs.get_check_stop_input()
        # Rows are in forward order, which the engine stamped on the output.
        row_rids = (
            list(outputs.row_request_ids)
            if outputs.row_request_ids is not None else request_ids
        )
        if not torch.cuda.is_available() or completion_event is None:
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
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(completion_event)

        if outputs.check_stop_buffers is not None and row_rids is not None:
            host = self._d2h_batched(outputs.check_stop_buffers, side)
            return (
                Worker._rows_to_per_rid(host, row_rids),
                HostRows(tuple(row_rids), host),
            )

        cpu_per_rid: dict = {}
        buffer_indices: dict[tuple[str, torch.dtype, int], int] = defaultdict(int)
        with torch.cuda.stream(side):
            for rid, name_to_list in source.items():
                if not isinstance(name_to_list, dict):
                    cpu_per_rid[rid] = name_to_list
                    continue
                cpu_per_rid[rid] = {}
                for name, tensors in name_to_list.items():
                    if not isinstance(tensors, list):
                        cpu_per_rid[rid][name] = tensors
                        continue
                    new_list = []
                    for t in tensors:
                        if torch.is_tensor(t) and t.is_cuda:
                            # by size class, not shape: a per-shape count gives two shapes in one class the same buffer
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
                    cpu_per_rid[rid][name] = new_list
        side.synchronize()

        return cpu_per_rid, None

    def _removal_step_reached(self, body: RemoveRequest) -> bool:
        """Whether this rank has consumed the step rank 0 released these pages
        after. Unstamped removals (``-1``) are not ordered against anything."""
        return (
            body.after_tp_seq < 0
            or self.scheduler.last_consumed_tp_seq >= body.after_tp_seq
        )

    def _apply_removes_whose_step_landed(self) -> None:
        """Tear down the requests rank 0 released once this rank reaches the step
        it released them after. Cannot strand them: the step was broadcast before
        the removal, so this rank gets there."""
        if not self._removes_awaiting_step:
            return
        reached = self.scheduler.last_consumed_tp_seq
        for seq in sorted(s for s in self._removes_awaiting_step if s <= reached):
            for rid in self._removes_awaiting_step.pop(seq):
                self._remove_request(RemoveRequest(
                    request_id=rid, source=MessageSource.SELF,
                ))

    def _apply_pending_removes_safe_to_drop(
        self, in_flight_rids: set[int]
    ) -> None:
        """Apply ``REMOVE_REQUEST`` for any rid that is not currently held by
        an in-flight GPU step. Removes for in-flight rids stay deferred and
        are reattempted next iter."""
        to_apply = [r for r in self._pending_removes if r not in in_flight_rids]
        for rid in to_apply:
            self._pending_removes.discard(rid)
            self._remove_request(RemoveRequest(
                request_id=self._rid_str(rid), source=MessageSource.SELF,
            ))

    def _drop_failed_rids(
        self, pending: PendingBatch,
        outputs: dict[int, NameToTensorList],
        failed_requests: dict[str, str],
    ) -> None:
        """Excise ``failed_requests`` from a finished batch.

        A rid that raised in ``postprocess`` is still carried in the batch (the
        engine only recorded the error), and one that raised in
        ``prepare_inputs`` is already out of ``node_batch.request_ids`` but not
        out of the worker-side ``ScheduledBatch``. Either way we must not route
        its outputs or mark its node complete — that's how a request that blew
        up mid-walk ends up reported to the client as a successful empty
        response. ``_postprocess_batch`` reconciles the remaining structures
        from ``node_batch.request_ids``.
        """
        for rid in failed_requests:
            outputs.pop(rid, None)
            pending.batch.request_to_worker_graph.pop(rid, None)
            pending.node_batch.per_request_info.pop(rid, None)
            # Clear what we just reported, so a later stage that fails more
            # rids (check_stop, below the forward) can tell its own from these
            # and doesn't report them to the conductor twice.
            pending.node_batch.failed_requests.pop(rid, None)
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

        Attribution is batch-granular here: a raise out of the forward, the
        batch build, or output routing can't be pinned on one request (the
        stages that *can* attribute report through
        ``ExecutingBatch.failed_requests`` instead), so every rid this iteration
        touched fails together. Sequential retry of the batch — see the design
        discussion on #123 — would go here; today a batch-level crash is
        terminal for its rids.

        The caller still has to clear its own in-flight state; see the main
        loop's handler.
        """
        # The exception is usually surfacing out of ``pending.future.result()``,
        # where the concurrent.futures machinery has already wrapped the
        # engine's traceback. Report the leaf exception to the client and keep
        # the full chain in the log.
        err = f"{type(exc).__name__}: {exc}"
        logger.exception("Worker %s error in main loop: %s", self.worker_id, err)

        failed_rids: set[int] = set(self._in_flight_rids)
        for stale in in_flight:
            if stale is None:
                continue
            failed_rids.update(stale.batch.request_to_worker_graph)
            self._clear_speculative_flag(stale.batch)
            # Drain before dropping the reference: the future owns engine state
            # on the GPU thread, and an abandoned one leaves that thread writing
            # into a batch nobody will collect. Already-finished futures (the
            # common case — one of them is what just raised) return at once.
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
            self._clear_speculative_flag(batch)
        if speculation is not None and speculation.plan_future is not None:
            # Armed but never submitted: the plan thread may still be staging
            # a pre-plan for a step that will now never run. Drain it, then
            # drop the stage, or the next step to lease that slot would find
            # it. Its fresh rids go back to their queues; the continuing ones
            # belong to the pending batch and fail with it.
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
            self._clear_speculative_flag(sb)
            fresh = {
                rid: wg_id
                for rid, wg_id in sb.request_to_worker_graph.items()
                if rid not in speculation.continuing_rids
                and rid not in failed_rids
            }
            if fresh:
                self._graph_runtime.push_back_node(
                    sb.node_name, list(fresh), list(fresh.values()),
                )

        self._fail_requests({rid: f"Error in worker: {err}" for rid in failed_rids})

    def _fail_requests(self, errors: dict[int, str]) -> None:
        """Report requests this worker can no longer serve to the conductor.

        ``errors`` maps request_id -> message. Rids the worker has already
        torn down are dropped: reporting them would leave a permanent entry
        in ``scheduler.failed_rids`` (the conductor answers a failure with a
        REMOVE_REQUEST, and it won't send one for a request it no longer
        knows about).
        """
        errors = {
            rid: msg for rid, msg in errors.items()
            if rid in self.request_state.per_request_info
        }
        if not errors:
            return
        wire_errors = {self._rid_str(rid): msg for rid, msg in errors.items()}
        for wire_rid, msg in wire_errors.items():
            logger.error("Worker %s failing request %s: %s", self.worker_id, wire_rid, msg)
        # Stop scheduling new work for these rids while the teardown is in
        # flight; the conductor's REMOVE_REQUEST clears the entry.
        self.scheduler.fail_rids(set(errors))
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.FAIL_REQUESTS,
                body=FailRequests(errors=wire_errors),
            ),
        )
        # Note: we do not cleanup the request right now; we wait for the conductor
        # to officially send a removal message

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
        self._verify_tp_async_sched_agrees()

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

        # The async worker path needs decode submission to return quickly so
        # the main loop can overlap queue/tensor polling and post-processing
        # with GPU execution. Run the engine unconditionally on a dedicated
        # 1-worker GPU thread.
        gpu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mstar-gpu-{self.worker_id}",
            initializer=self._init_engine_thread,
        )
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
        # Default ON. Set MSTAR_PRE_PLAN_SPEC=0 to fall back to the
        # double-buffer-without-pre-plan baseline.
        pre_plan_spec = os.environ.get("MSTAR_PRE_PLAN_SPEC", "1") == "1"
        # How long the submitter holds off the GIL waiting for the GPU thread
        # to reach the launch. submit_spec often sits on this cap, but raising
        # it to 8ms did not help; tunable for another look.
        launch_wait_s = float(os.environ.get("MSTAR_LAUNCH_WAIT_MS", "5")) / 1000.0
        plan_executor = None
        if pre_plan_spec:
            plan_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mstar-plan-{self.worker_id}",
                initializer=self._init_engine_thread,
            )
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
        max_consecutive_spec = int(os.environ.get("MSTAR_MAX_CONSECUTIVE_SPEC_STEPS", "1024"))
        spec_peek_for_fairness = (
            os.environ.get("MSTAR_SPEC_PEEK_FOR_FAIRNESS", "1") == "1"
        )
        consecutive_spec_steps = 0
        yield_away_from_target: tuple[str, str] | None = None

        def _set_pending(p: PendingBatch):
            nonlocal pending
            pending = p
            self._in_flight_rids = set(p.batch.request_to_worker_graph) if p else set()

        # Per-phase wall-clock instrumentation, gated by MSTAR_PHASE_TIMING.
        # When enabled, every Nth speculative iter logs a histogram so we can
        # see whether await_gpu time = "GPU still running" (overlap working)
        # vs "GPU done, idle" (overlap not paying off). Set the env var to a
        # positive integer = the dump period in iters (e.g. 200).
        phase_period = self._phase_period
        phase_buf = self._phase_buf
        phase_iter = [0]
        _phase_record = self._phase_record

        # Requests in flight per iteration over the window. Reported beside
        # the timings because it is what separates ramp-up, steady state and
        # drain: a phase mean averaged across those is not a measurement of
        # anything. ``benchmark/worker_phases`` segments on it.
        phase_bs: list[int] = []

        def _phase_flush() -> None:
            if phase_period <= 0 or phase_iter[0] % phase_period != 0:
                return
            # list() first: the GPU and plan threads append to this while we
            # read it, and iterating the live dict would raise on resize.
            samples = sorted((k, v) for k, v in list(phase_buf.items()) if v)
            parts = []
            for name, vs in samples:
                vs = sorted(vs)
                n = len(vs)
                # a trailing '#' marks a count, not a duration
                scale, unit = (1, "") if name.endswith("#") else (1000, "ms")
                p50 = vs[n // 2] * scale
                p95 = vs[min(n - 1, int(n * 0.95))] * scale
                mean = (sum(vs) / n) * scale
                parts.append(
                    f"{name}: p50={p50:.2f}{unit} p95={p95:.2f}{unit} "
                    f"mean={mean:.2f}{unit} n={n}"
                )
            bs = (sum(phase_bs) / len(phase_bs)) if phase_bs else 0.0
            logger.info(
                "Worker %s phase-timing iter=%d bs=%.2f: %s",
                self.worker_id, phase_iter[0], bs, " | ".join(parts),
            )
            phase_buf.clear()
            phase_bs.clear()

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
            try:
                batch = None
                spec_pending = None
                _iter_start = _time.perf_counter() if phase_period else 0.0
                # Everything below moves pages — the removes and drains directly,
                # ``_process_messages`` through a follower's delta, the readiness
                # scans through ``check_ready``'s reload — and none of it may run
                # while N's admit is outstanding. See ``_await_admit_settled``.
                self._await_admit_settled(pending)
                self._apply_pending_removes_safe_to_drop(
                    self._in_flight_rids
                )
                # Requests a resource declared unservable during last pass's
                # readiness scans. They are in no batch, so nothing else would
                # ever fail them.
                self._fail_requests(self.scheduler.take_admit_errors())
                self._apply_pending_drains(self._in_flight_rids)

                # 1. CPU preamble — overlaps with GPU(N).
                # synchronize=False on every range so torch.cuda.synchronize()
                # doesn't drain the in-flight GPU work and undo the overlap.
                if self.enable_nvtx:
                    range_push("worker.process_messages", synchronize=False)
                self._process_messages()
                # Removals rank 0 stamped with a step this rank has now reached.
                self._apply_removes_whose_step_landed()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("worker.check_ready_tensors", synchronize=False)
                self._check_ready_tensors()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("worker.poll_stream_buffers", synchronize=False)
                self._poll_stream_buffers()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # 2. Speculatively schedule + build N+1 — overlaps with GPU(N).
                # Only when (a) there's a pending step and (b) it's AR-engine.
                # For non-AR or non-loop-body steps, falls through to the
                # non-speculative path below (drain, then schedule).
                speculation = None
                yield_away_from_target = None

                # Build nothing on a refusal; the fence above is what makes it
                # knowable here. Ordinary speculation would be dropped anyway,
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
                    must_yield_for_fairness = (
                        spec_peek_for_fairness
                        and consecutive_spec_steps >= 1
                        and self.scheduler.has_ready_excluding(
                            self.request_state,
                            (pending.node_name, pending.graph_walk),
                        )
                    )
                    must_yield_away = (
                        consecutive_spec_steps >= max_consecutive_spec
                        or must_yield_for_fairness
                    )
                    if not must_yield_away and not admit_refused:
                        if self.enable_nvtx:
                            range_push("worker.speculate", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        speculation = self._try_speculate_next(pending)
                        if phase_period:
                            _phase_record("speculate", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)
                        if speculation is not None:
                            # Broadcast the head now, during forward N, so
                            # followers build it during theirs (-1: not parallel).
                            speculation.tp_seq = self.maybe_send_zmq_to_tp_followers(
                                speculation.node_batch,
                                speculative=True, spec_from_seq=pending.tp_seq,
                            )
                    if self._tp_lead_needs_marker(pending, speculation):
                        self._broadcast_tp_nospec(pending)
                    # ``yield_away_from_target`` stays None when the admit
                    # refused, so the non-speculative path below schedules
                    # without the fairness exclusion — it should be free to pick
                    # up the batch ``_handle_admit_failure`` just pushed back.
                    if speculation is None and not admit_refused:
                        yield_away_from_target = (
                            pending.node_name,
                            pending.graph_walk,
                        ) if must_yield_away else None
                        with self._span("worker.schedule_yield_away"):
                            batch = self.scheduler.get_next_batch(
                                self.request_state,
                                exclude_target=yield_away_from_target,
                            )
                        if batch is not None:
                            node_batch = self._build_executing_batch(batch)
                            batch_partition = self.request_state.get_partition_for_node(batch.node_name)
                            logger.debug(f"Yield away: {batch.node_name} {node_batch.request_ids}")
                            speculation = Speculation(
                                scheduled_batch=batch,
                                node_batch=node_batch,
                                consumed_edges=set(),
                                continuing_rids=set(), # n/a
                                partition=batch_partition,
                                is_new_iter=False,
                                is_same_node=False,
                                is_yield_away=True
                            )

                            # A leader stamps the seq it sends; a follower's
                            # batch keeps the one it came off the FIFO with.
                            ya_seq = self.maybe_send_zmq_to_tp_followers(node_batch)
                            speculation.tp_seq = ya_seq if ya_seq >= 0 else batch.tp_seq

                def _arm_speculation(spec: Speculation) -> None:
                    # Hand N+1 to the plan thread NOW. It waits on N's
                    # commit event, then reserves a slot and pre-plans —
                    # while the main thread sits in await_gpu with the GIL
                    # released and N's kernels are still running.
                    if plan_executor is not None:
                        spec.plan_future = plan_executor.submit(
                            self._preplan_spec, pending, spec,
                        )

                if speculation is not None:
                    _arm_speculation(speculation)

                # 3. If pending: await GPU(N), submit speculated GPU(N+1)
                # asap, then post-process N (fast then slow) overlapping
                # with GPU(N+1).
                spec_pending = None
                if pending is not None:
                    outputs: dict[int, NameToTensorList] | None = None
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
                            _phase_record("follow_await", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)

                    if outputs is None:
                        if self.enable_nvtx:
                            range_push("worker.await_gpu", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        outputs = pending.future.result()
                        if phase_period:
                            _phase_record("await_gpu", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)

                    # set node._speculatively_scheduled to false, since
                    # the node has just completed
                    self._clear_speculative_flag(pending.batch)

                    def _maybe_clear_spec():
                        nonlocal speculation
                        # Speculation cleanup splits by kind:
                        #
                        # * Non-yield-away spec depended on pending's outputs
                        #   (the plan thread already threaded them in). Pending's
                        #   output is invalid, so the spec batch can't run.
                        #
                        # * Yield-away spec is independent of pending.
                        #   But ``_handle_allocation_failure`` may have shifted
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
                                self._settle_speculation(speculation, success=False)
                                # Fresh rids are not re-readied by N's routing
                                # the way continuing rids are: give them back.
                                sb = speculation.scheduled_batch
                                fresh = [
                                    rid for rid in sb.request_to_worker_graph
                                    if rid not in speculation.continuing_rids
                                    and sb.request_to_worker_graph.get(rid)
                                    is not None
                                ]
                                self._graph_runtime.push_back_node(
                                    sb.node_name, fresh,
                                    [sb.request_to_worker_graph[r] for r in fresh],
                                )
                                speculation = None

                    if pending.node_batch.admit_error is not None:
                        # Admit refused pending, so no forward ran.
                        # ``_handle_admit_failure`` pushes the nodes back and on
                        # KV-cache OOM offloads or holds. Clearing the speculation
                        # first drops its pre-plan, releasing pages the eviction
                        # would otherwise go looking for. Nothing is speculated on
                        # a refusal, so in practice there is nothing to clear.
                        self._clear_speculative_flag(pending.batch)
                        _maybe_clear_spec()
                        self._handle_admit_failure(
                            pending.batch, pending.node_batch,
                            referenced_rids=frozenset(
                                speculation.scheduled_batch.node_objects
                            ) if speculation is not None else frozenset(),
                        )

                    if pending.node_batch.failed_requests:
                        # A per-rid stage (prepare_inputs / postprocess) blamed
                        # specific requests. Drop the speculation: it was built
                        # from pending's rids and may thread outputs that the
                        # failed rids never produced. The rest of the batch
                        # still post-processes and routes normally below.
                        _maybe_clear_spec()
                        failed = dict(pending.node_batch.failed_requests)
                        self._drop_failed_rids(pending, outputs, failed)
                        self._fail_requests(failed)

                    if speculation is not None:
                        spec_batch = speculation.scheduled_batch
                        spec_node_batch = speculation.node_batch
                        if not speculation.is_yield_away:
                            self._thread_outputs_to_speculative(speculation, outputs)
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

                        if spec_batch.request_to_worker_graph:
                            if self.enable_nvtx:
                                range_push("worker.submit_spec", synchronize=False)
                            _t0 = _time.perf_counter() if phase_period else 0.0
                            # Staleness is checked on the GPU thread, after
                            # the plan future resolves and after prepare; see
                            # _execute_on_gpu_thread.

                            # Hold the main thread off the GIL until the GPU
                            # thread reaches the forward launch; see exec.
                            spec_launch_started = threading.Event()
                            spec_node_batch.launch_started_event = spec_launch_started
                            spec_future = gpu_executor.submit(
                                self._execute_on_gpu_thread,
                                spec_batch, spec_node_batch,
                                speculation.plan_future,
                            )
                            # the GPU thread owns the pre-plan from here; see
                            # _handle_main_loop_error
                            speculation.plan_future = None
                            self.wakeup_event.register_future(spec_future)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                                range_push("worker.gpu_submit_queued", synchronize=False)
                            with self._span("worker.submit_spec.launch_wait"):
                                spec_launch_started.wait(timeout=launch_wait_s)
                            if phase_period:
                                _phase_record("submit_spec", _time.perf_counter() - _t0)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                            spec_pending = PendingBatch(
                                batch=spec_batch,
                                node_batch=spec_node_batch,
                                node_name=spec_batch.node_name,
                                partition=speculation.partition,
                                graph_walk=spec_batch.graph_walk,
                                future=spec_future,
                                speculative_new_iter=speculation.is_new_iter,
                                loop_name=speculation.loop_name,
                                tp_seq=speculation.tp_seq,
                            )
                        elif speculation.plan_future is not None:
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
                    if pending.node_batch.admit_error is None:
                        with self._span("worker.postprocess_batch"):
                            self._postprocess_batch(pending, outputs)

                    # Removes for any rid not in the in-flight spec step
                    # are safe to apply now.
                    in_flight = set(spec_pending.batch.request_to_worker_graph) if spec_pending else set()
                    self._apply_pending_removes_safe_to_drop(in_flight)
                    self._apply_pending_drains(in_flight)
                    _set_pending(None)

                if spec_pending is not None:
                    if speculation.is_yield_away:
                        consecutive_spec_steps = 0
                    else:
                        consecutive_spec_steps += 1
                    if phase_period:
                        _phase_record("iter_total", _time.perf_counter() - _iter_start)
                        phase_bs.append(len(spec_pending.batch.request_to_worker_graph))
                        phase_iter[0] += 1
                        _phase_flush()
                    _set_pending(spec_pending)
                    continue
                consecutive_spec_steps = 0

                # 4. Non-speculative path: no pending or speculation skipped
                # (e.g., non-AR engine, or loop ended). Run MicroScheduler.
                with self._span("worker.schedule"):
                    batch = None
                    if yield_away_from_target is not None:
                        batch = self.scheduler.get_next_batch(
                            self.request_state,
                            exclude_target=yield_away_from_target,
                        )
                    if batch is None:
                        batch = self.scheduler.get_next_batch(self.request_state)
                if batch is None:
                    self.communicator.wait_for_work(10)
                    continue

                if self.enable_nvtx:
                    range_push("worker.build_node_batch", synchronize=False)
                node_batch = self._build_executing_batch(batch)
                batch_partition = self.request_state.get_partition_for_node(batch.node_name)

                for request_id, new_iters in self._graph_runtime.get_dynamic_loop_iters(
                    list(node_batch.per_request_info), partition=batch_partition,
                ):
                    node_batch.per_request_info[request_id] \
                        .dynamic_loop_iter_counts.update(new_iters)
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # Nothing speculated this batch, so prepare it here. The GPU
                # thread then admits, plans and runs it inline; the slot is
                # leased inside exec, once the token count is known.
                # A leader stamps the seq it sends; a follower's batch keeps
                # the one it came off the FIFO with; everyone else -1.
                broadcast_seq = self.maybe_send_zmq_to_tp_followers(node_batch)
                fallthrough_tp_seq = broadcast_seq if broadcast_seq >= 0 else batch.tp_seq

                # Unlike the spec submit, this path doesn't hold off the GIL for
                # the launch — but it still has to be fenceable, or the next
                # iteration schedules against an admit that hasn't run.
                node_batch.launch_started_event = threading.Event()
                future = gpu_executor.submit(
                    self._execute_on_gpu_thread, batch, node_batch, None,
                )
                self.wakeup_event.register_future(future)
                logger.debug(f"Scheduling: {batch.node_name} {node_batch.request_ids}")
                _set_pending(PendingBatch(
                    batch=batch,
                    node_batch=node_batch,
                    node_name=batch.node_name,
                    partition=batch_partition,
                    graph_walk=batch.graph_walk,
                    future=future,
                    tp_seq=fallthrough_tp_seq,
                ))
                # Same accounting as the speculative path above
                if phase_period:
                    _phase_record("iter_total", _time.perf_counter() - _iter_start)
                    phase_bs.append(len(batch.request_to_worker_graph))
                    phase_iter[0] += 1
                    _phase_flush()
            except Exception as e:
                self._handle_main_loop_error(e, (pending, spec_pending), batch, speculation)
                # Follower: a head from a step that raised must not sit at the
                # FIFO front with failed rids.
                if pending is not None and self._is_tp_follow_pending(pending):
                    self._close_tp_follow_step(pending)
                # Clear the in-flight step. Without this the next iteration
                # calls .result() on the same completed-with-exception future
                # and re-raises forever, wedging the worker on one bad batch.
                _set_pending(None)
                consecutive_spec_steps = 0
                sleep(0.01)
