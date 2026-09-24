import gc
import logging
import os
import sys
import threading
import time
import time as _time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from time import sleep

import torch

from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AllocationFailed, StepContext
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.graph.base import GraphEdge
from mstar.graph.graph_io import format_graph_edge_list
from mstar.graph.runtime.base import (
    EdgeSpec,
    RouteInput,
    RouteOutput,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
)
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import Model, WorkerGraph
from mstar.profile.worker import WorkerProfileInfo
from mstar.streaming.stream_buffer import StreamBuffer
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
    consumed_streaming_edges: dict[int, list[GraphEdge]] = field(default_factory=dict)
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

        node_names = set()
        for wg in my_worker_graphs:
            node_names.update(wg.section.get_nodes())

        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            model_config=model_config,
            parallel_groups=self.parallel_groups,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.transfer_engine
            ),
            model=model,
            enable_nvtx=self.enable_nvtx,
            enable_prof=self.enable_prof
        )

        # The graph runtime owns the per-request queues and the graph state.
        self._graph_runtime = PythonGraphRuntime(
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
        # The runtime takes an explicit set, so resolve the two special cases
        # here: tp_async_nodes=None means "every node", and the feature being
        # off means "none". An empty set on the runtime side is just "none".
        self._graph_runtime.set_node_metadata(
            parallel_nodes=self.parallel_nodes,
            parallel_leader_nodes=self.parallel_leader_nodes,
            tp_async_nodes={
                n for n in node_names if self._tp_async_for(n)
            },
        )

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
            tuple[str, torch.dtype, tuple[int, ...]], list[torch.Tensor]
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

        # Build consumer node cache: edge_name -> next_node name
        self._consumer_node_cache: dict[str, str] = {}
        if self._my_consumer_connections and model:
            walks = model.get_graph_walk_graphs()
            for conn in self._my_consumer_connections:
                for section in walks.values():
                    if hasattr(section, 'input_names') and conn.edge_name in section.input_names:
                        self._consumer_node_cache[conn.edge_name] = section.name

    def _get_node_names_for_partition(self, partition_name: str, model: Model) -> list[str]:
        """Get the node names that belong to a partition."""
        walks = model.get_graph_walk_graphs()
        partitions = model.get_partitions()
        for pdef in partitions:
            if pdef.name == partition_name:
                nodes = set()
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section and hasattr(section, 'name'):
                        nodes.add(section.name)
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

        # Create StreamBuffers for consumer connections on this worker
        for conn in self._my_consumer_connections:
            req_info = self.request_state.per_request_info[request_id]
            req_info.stream_buffers[conn.edge_name] = StreamBuffer(
                request_id=request_id,
                edge_name=conn.edge_name,
                from_partition=conn.from_partition,
                policy=conn.chunk_policy_factory(),
            )

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
                self._edge_specs(request_id, signal_only), can_buffer=True,
            )
        # process messages that may have came in out-of-order
        if body.request_id in self._unprocessed_messages:
            self._process_message_list(self._unprocessed_messages[body.request_id])
            del self._unprocessed_messages[body.request_id]


    def _remove_request(self, body: RemoveRequest) -> None:
        if self.is_tp_follower and body.source not in (MessageSource.TP_RANK_0, MessageSource.SELF):
            return # wait for removal message from TP rank 0 to avoid race conditions

        # Async-scheduling deferral: if this rid is currently held by an
        # in-flight GPU step (or its speculation), tearing down engine /
        # tensor state now would race the GPU thread reading those tensors
        # / KV pages. Queue the remove and apply it once no in-flight step
        # references the rid (see _apply_pending_removes_safe_to_drop in
        # the run loop).
        request_id = self._rid(body.request_id)
        if request_id is None:
            return  # never admitted here, or already removed
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
        self.scheduler.clear_rid(request_id)
        self._pending_removes.discard(request_id)

        for node_name in self.engine_manager.evictable_nodes():
            self._last_active.pop((request_id, node_name), None)

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
        # A request this worker never admitted has nothing in flight, so it
        # can ACK straight away.
        if handle is not None:
            if self.tensor_manager.has_inflight_reads(handle):
                return  # let get_ready_tensors resolve the futures; retry next iter
            if self.scheduler.pending_tp_follow_count.get(handle, 0) > 0:
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
                self._edge_specs(request_id, signal_only), can_buffer=True,
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
    ) -> GraphEdge | None:
        consumer_node = self._consumer_node_cache.get(edge_name, "")
        synthetic_edge = sbuf.pop_waiting_edge()
        if synthetic_edge is None and sbuf.has_chunk_ready():
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
        return synthetic_edge

    def _poll_stream_buffers_for_speculation(
        self, request_id: int, node_name: str
    ) -> list[GraphEdge]:
        result = []
        req_info = self.request_state.per_request_info.get(request_id)
        if req_info is None:
            return []
        for edge_name, sbuf in req_info.stream_buffers.items():
            consumer_node = self._consumer_node_cache.get(edge_name, "")
            if consumer_node != node_name:
                continue
            edge = self._pop_streaming_edge(sbuf, edge_name, request_id)
            if edge is not None:
                result.append(edge)
        return result

    def _return_speculative_streaming_edge(
        self, request_id: int, edge: GraphEdge
    ):
        req_info = self.request_state.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(edge.name)
        if sbuf is not None:
            sbuf.store_uningested_edge(edge)

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        for request_id, req_info in list(self.request_state.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                synthetic_edge = self._pop_streaming_edge(sbuf, edge_name, request_id)

                if synthetic_edge is not None:
                    # Streaming edges go through the same path as regular ones —
                    # ReadySignals.is_ready_for_streaming flips on as soon as
                    # the streaming inputs are the only ones missing. Empty
                    # leftover list means the edge was claimed. The final-chunk
                    # signal rides the synthetic edge to the consuming pass,
                    # which reports the partition done in _postprocess_batch —
                    # NOT here, where an earlier in-flight pass's WGD could read
                    # it before the final output chunk is emitted.
                    uningested = self._graph_runtime.ingest_inputs_batch(
                        self._edge_specs(request_id, [synthetic_edge]),
                        # important: only ingest for this loop iter!
                        can_buffer=False,
                        is_streaming=True,
                    )
                    if uningested:
                        sbuf.store_uningested_edge(synthetic_edge)


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
                    self._edge_specs(request_id, normal), can_buffer=True,
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
            return None

        # prefer evicting requests that aren't currently executing
        external = [rid for rid in candidates if rid not in batch_ids]
        victim_id = self._select_eviction_victim(node_name, external or candidates)
        freed = engine.offload_request(node_name, victim_id)
        if freed <= 0:
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

    def _tensors_for(self, edges: list[EdgeSpec]) -> NameToTensorList:
        """Resolve a node's ready inputs to tensors. The runtime reports them
        as uuids; only this side can turn those back into tensors."""
        return {
            spec.signal: [
                self.tensor_manager.get_tensor(uuid) for uuid in spec.uuids
            ] for spec in edges
        }

    def _build_executing_batch(self, batch: ScheduledBatch) -> ExecutingBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_inputs: dict[int, NameToTensorList] = {}
        per_request_info: dict[int, CurrentForwardPassInfo] = {}
        final_stream_rids: set[int] = set()
        batch_partition = self.request_state.get_partition_for_node(batch.node_name)

        for request_id, edges in batch.input_edges.items():
            per_request_inputs[request_id] = self._tensors_for(edges)
            if any(spec.is_final_streaming_chunk for spec in edges):
                final_stream_rids.add(request_id)
            per_request_info[request_id] = self.request_state.get_fwd_info(request_id, batch_partition)

        return self._make_executing_batch(
            node_name=batch.node_name,
            graph_walk=batch.graph_walk,
            request_ids=list(batch.request_to_worker_graph),
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
            final_stream_rids=final_stream_rids,
        )

    def _make_executing_batch(
        self,
        node_name: str,
        graph_walk: str,
        request_ids: list[int],
        per_request_input_tensors: dict[int, NameToTensorList],
        per_request_info: dict[int, CurrentForwardPassInfo],
        final_stream_rids: set[int] | None = None,
    ) -> ExecutingBatch:
        """One step's batch, with the step context the engine drives it through.

        The context starts unleased and eager; a slot is reserved later, once
        the real token count is known.
        """
        return ExecutingBatch(
            node_name=node_name,
            per_request_info=per_request_info,
            per_request_input_tensors=per_request_input_tensors,
            final_stream_rids=final_stream_rids or set(),
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
    def _edge_specs(rid: int, edges: list[GraphEdge]) -> ParallelList:
        """Flatten edges into (rid, EdgeSpec) pairs for the runtime.

        Only uuids cross: the runtime rebuilds the descriptors from the tensor
        store, which is why they are kept there.
        """
        return ParallelList(
            [rid] * len(edges),
            [
                EdgeSpec(
                    signal=edge.name,
                    next_node=edge.next_node,
                    uuids=[info.uuid for info in edge.tensor_info],
                    is_final_streaming_chunk=edge._final_stream_chunk,
                ) for edge in edges
            ],
        )

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

        The runtime already dropped repeat edges of a signal, so summing is
        safe: one output routed to two destinations is two edges carrying the
        same tensors, and counting both would double every token.
        """
        counts: dict[int, dict[str, int]] = {}
        for idx in new_token_idxs:
            per_rid = counts.setdefault(flat_rids[idx], {})
            signal = signals[signal_idxs[idx]]
            per_rid[signal] = per_rid.get(signal, 0) + (
                self.tensor_manager.get_tensor(flat_uuids[idx]).numel()
            )
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

    def _init_cuda_executor_thread(self) -> None:
        """Pin this executor thread to the worker's accelerator device.

        The CUDA current device is per-thread and defaults to 0. PyTorch
        ops carry per-tensor device guards, but raw Triton launches and
        bare ``torch.cuda.current_stream()`` / ``synchronize()`` calls
        resolve against the THREAD's device — on a worker whose model
        lives on a non-zero device, work issued from an unpinned thread
        lands on device 0's stream, unordered with the real compute.
        """
        if self.device.type != "cpu" and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)

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
        logger.debug("Executing batch for node %s", node_batch.node_name)
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
            with self._span("worker.gpu_thread.prepare_inputs"):
                engine.prepare_inputs(node_batch)
            # call is_stale after prepare_inputs because prepare_inputs may drop rids
            if plan_future is not None and engine.preplan_is_stale(node_batch):
                engine.reset_pre_plan_for_batch(node_batch)
            with self._span("worker.gpu_thread.exec"):
                outputs = engine.exec_and_postprocess(node_batch)
            execution_stream = (
                torch.accelerator.current_stream(self.device)
                if self.device.type != "cpu"
                else None
            )
            if execution_stream is not None:
                event = torch.Event()
                event.record(execution_stream)
                node_batch.completion_event = event
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
            engine.finalize_batch(node_batch)
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _handle_admit_failure(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch
    ) -> None:
        """Re-queue a batch whose admit refused it, so the step can be retried.

        Every admit failure needs the push-back; only an ``AllocationFailed``
        also needs an eviction. ``RequestOffloading`` means the rid is already
        on its way to the host, so evicting anything else is wasted work — the
        retry is gated on ``check_ready`` reloading it.
        """
        reason = node_batch.admit_error
        if isinstance(reason, AllocationFailed):
            self._handle_allocation_failure(batch, node_batch)
            return

        self._push_back_batch(batch)
        logger.info(
            "Admit refused node=%s walk=%s (%s): re-queued %d requests",
            batch.node_name, batch.graph_walk,
            type(reason).__name__, len(batch),
        )

    def _handle_allocation_failure(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch
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

        v2 caveat: this function does not yet coordinate ``_last_active``
        / eviction-victim selection across TP ranks. Wall-clock LRU can
        pick different victims per rank under contention, leading to
        request-id ↔ page-index drift and (eventually) asymmetric OOM on
        future reloads. Today's TP configs don't enable CPU offload, so
        the path isn't exercised; revisit when we light up offload + TP.
        """
        batch_ids = set(batch.request_to_worker_graph)
        # scope the eviction to whichever resource actually ran out, when the
        # admit named one
        failed = node_batch.failed_resource
        victim_id = self._try_offload_cold_request(
            node_batch.node_name, batch_ids,
            affected_resources=None if failed is None else {failed},
        )

        # Push all batch nodes back to their queues
        self._push_back_batch(batch)

        if victim_id is not None:
            self.scheduler.hold_requests([victim_id])
            logger.warning(
                "OOM on node=%s walk=%s: offloaded victim=%s, "
                "retrying %d remaining requests",
                batch.node_name, batch.graph_walk, victim_id,
                len(batch_ids) - (1 if victim_id in batch_ids else 0),
            )
        else:
            self.scheduler.hold_requests(list(batch_ids))
            key = (batch.node_name, batch.graph_walk)
            now = _time.monotonic()
            last, unlogged = self._hold_logged.get(key, (None, 0))
            if last is not None and now - last < _HOLD_LOG_INTERVAL:
                self._hold_logged[key] = (last, unlogged + 1)
                return
            self._hold_logged[key] = (now, 0)
            logger.warning(
                "OOM on node=%s walk=%s: no offload possible, "
                "holding %d requests (%d earlier holds not logged)",
                batch.node_name, batch.graph_walk, len(batch_ids), unlogged,
            )

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

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

    def _assemble_speculation(
        self,
        pending: PendingBatch,
        spec_target: SpeculationOutput,
        request_to_worker_graph: dict[int, int],
        per_request_inputs: dict[int, NameToTensorList],
        consumed_streaming_edges: dict[int, list[GraphEdge]],
        continuing: list[int],
        *,
        is_same_node: bool,
        tp_seq: int = -1,
    ) -> Speculation:
        """Package prepared rids (batch order) into the ``Speculation`` the main
        loop runs. Leader and follower differ only in how they pick the rids."""
        spec_node = spec_target.node_name
        request_ids = list(request_to_worker_graph)
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
            per_request_info={
                rid: self.request_state.get_fwd_info(rid, pending.partition)
                for rid in request_ids
            },
            final_stream_rids={
                rid for rid, edges in consumed_streaming_edges.items()
                if any(e._final_stream_chunk for e in edges)
            },
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
            tp_seq=tp_seq,
        )

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
        consumed_streaming_edges: dict[int, list[GraphEdge]] = {}
        # Backlogged rids for this target get first claim on the batch: they
        # have already waited a step, and the chain only ever continues its own
        # rids, so at the cap they would never be reached. None => uncapped.
        spec_target = (spec_node_name, batch_N.graph_walk)
        max_continuing = self.scheduler.room_for_continuing(spec_target)

        # Removes are filtered here; prep_spec_rids assumes that.
        candidates = [
            r for r in batch_N.request_to_worker_graph if r not in self._pending_removes
        ]
        # Polling the StreamBuffers stays on this side: they hold real tensors.
        polled: list[tuple[int, GraphEdge]] = []
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
                ) for _r, e in polled
            ],
            streaming_edges_per_rid=per_rid_counts,
        ))

        # Anything not consumed goes back to its StreamBuffer, so a later
        # scheduling of this node picks it up normally.
        consumed = set(prep.consumed_streaming_edge_idxs)
        for i, (r, edge) in enumerate(polled):
            if i not in consumed:
                self._return_speculative_streaming_edge(r, edge)
            else:
                consumed_streaming_edges.setdefault(r, []).append(edge)

        continuing = set(prep.ready_rids)
        cursor = 0
        for i, r in enumerate(prep.ready_rids):
            count = prep.input_edges_per_rid[i]
            per_request_inputs[r] = self._tensors_for(
                prep.input_edges[cursor:cursor + count]
            )
            cursor += count
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
            pre_existing_batch_size=len(continuing)
        )

        if fresh_batch is not None:
            # The merge below relabels these node objects with the spec
            # target's name/walk, so a batch for any other node must not be
            # merged in.
            assert fresh_batch.node_name == spec_node_name, (
                f"Speculation asked for {spec_node_name!r} but the "
                f"scheduler returned {fresh_batch.node_name!r}"
            )
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

                per_request_inputs[rid] = self._tensors_for(
                    fresh_batch.input_edges[rid]
                )
                new_request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )

        logger.debug("Speculating: %s %s", spec_node_name, continuing)
        return self._assemble_speculation(
            pending, spec_target_info,
            new_request_to_worker_graph, per_request_inputs,
            consumed_streaming_edges, continuing,
            is_same_node=speculating_same_node,
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
                for edge in speculation.consumed_streaming_edges.get(r, []):
                    self._return_speculative_streaming_edge(r, edge)
                speculation.consumed_streaming_edges.pop(r, None)
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
        polled: list[tuple[int, GraphEdge]] = []
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
            for r, edge in polled:
                self._return_speculative_streaming_edge(r, edge)

        if popped is None:
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
                ) for _r, e in polled
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
        consumed_streaming_edges: dict[int, list[GraphEdge]] = {}
        for i, (r, edge) in enumerate(polled):
            if i in consumed:
                consumed_streaming_edges.setdefault(r, []).append(edge)
            else:
                self._return_speculative_streaming_edge(r, edge)

        prep_inputs: dict[int, list[EdgeSpec]] = {}
        cursor = 0
        for i, r in enumerate(prep.ready_rids):
            count = prep.input_edges_per_rid[i]
            prep_inputs[r] = prep.input_edges[cursor:cursor + count]
            cursor += count
        prep_wg = dict(zip(prep.ready_rids, prep.wg_ids, strict=True))

        new_request_to_worker_graph: dict[int, int] = {}
        per_request_inputs: dict[int, NameToTensorList] = {}
        for rid in head_rids:  # wire order == the leader's batch order
            if rid in prep_inputs:
                new_request_to_worker_graph[rid] = prep_wg[rid]
                per_request_inputs[rid] = self._tensors_for(prep_inputs[rid])
            else:
                new_request_to_worker_graph[rid] = fresh_wg[rid]
                per_request_inputs[rid] = self._tensors_for(fresh_edges[rid])

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
            is_same_node=True, tp_seq=head.spec_seq,
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
        outputs: dict[int, NameToTensorList],
    ):
        if self.enable_nvtx:
            range_push("worker.postprocess.cleanup_inputs", synchronize=False)

        rids = list(batch_N.batch.request_to_worker_graph)
        self._graph_runtime.cleanup_consumed_inputs(
            batch_N.batch.node_name, rids,
            [batch_N.batch.request_to_worker_graph[r] for r in rids],
        )
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
        batch_N.node_batch.request_ids = list(valid_rids)
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

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.update_lru", synchronize=False)

        # Update LRU
        t = _time.monotonic()
        for rid in batch_N.node_batch.request_ids:
            self._last_active[(rid, batch_N.node_name)] = t

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.synchronize_completion_event", synchronize=False)

        # Wait for batch N's completion event before proceeding
        # TODO: may need to refine this based on how it affects performance?
        if self.device.type != "cpu" and batch_N.batch.request_to_worker_graph:
            if batch_N.node_batch.completion_event is not None:
                if self.enable_nvtx:
                    range_push("worker.postprocess.completion_event_sync", synchronize=False)
                batch_N.node_batch.completion_event.synchronize()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
            else:
                torch.accelerator.synchronize(self.device)

        if self.enable_prof:
            batch_N.node_batch.exec_timings.fwd_end = time.perf_counter()

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.check_stop", synchronize=False)

        per_request_info = batch_N.node_batch.per_request_info
        for rid, new_iters in self._graph_runtime.get_dynamic_loop_iters(
            list(per_request_info), partition=batch_N.partition,
        ):
            per_request_info[rid].dynamic_loop_iter_counts.update(new_iters)

        # Check for stops
        engine = self.engine_manager.get_engine(batch_N.node_name)
        cpu_outputs = self._prematerialize_for_check_stop(
            outputs, batch_N.node_batch.completion_event,
        )
        stops = engine.check_stop_for_batch(batch_N.node_batch, cpu_outputs)
        # the same host copy, before stops, so a request ending here still indexes its pages
        engine.extend_prefix_chains(batch_N.node_batch, cpu_outputs)
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

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.stop_loops", synchronize=False)

        # Stop loops, if applicable. The runtime filters rids whose walk does
        # not contain the loop, snapshots the stop times, records the pending
        # stops and fans out to peers.
        if stops:
            self._graph_runtime.stop_loops_batched(
                partition=batch_N.partition,
                graph_walk=batch_N.graph_walk,
                last_node_run=batch_N.node_name,
                loop_names=ParallelList(
                    list(stops), [list(v) for v in stops.values()],
                ),
            )

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
        stored = self.tensor_manager.store_and_return_tensor_info_batch(
            rids, outputs, signals,
            node_name=batch_N.node_name,
            graph_walk=batch_N.graph_walk,
            skip_cuda_sync=True,
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
            self.tensor_manager.tensor_store,
        )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.register_outputs", synchronize=False)
        self._register_outputs(route_output)

        # send outputs
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.send_outputs", synchronize=False)

        # The consuming pass (not the earlier ingest) reports the partition
        # done, so it rides this pass's WGD with the final output loop index.
        for rid in batch_N.node_batch.final_stream_rids:
            self._graph_runtime.mark_stream_partition_done(rid, batch_N.partition)

        # set this before send_outputs so that we can send updated profiling info to the conductor
        if self.enable_prof:
            self.profile_info.register_end(
                batch_N.node_batch.node_name,
                batch_N.node_batch.graph_walk,
                batch_N.node_batch.request_ids,
                batch_N.node_batch.exec_timings,
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
        # numel() needs the tensors, so the counting stays on this side.
        new_token_counts = self._count_new_tokens(
            route_output.new_token_output_idxs,
            flat_rids, flat_uuids, signals, signal_idxs,
        )
        self._graph_runtime.send_outputs(SendInput(
            completion_id=route_output.completion_id,
            per_request_info=ParallelList(
                send_rids,
                [
                    self.request_state.get_fwd_info(rid, batch_N.partition)
                    for rid in send_rids
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
        ))

        if self.enable_nvtx:
            range_pop(synchronize=False)

    def _get_pinned_d2h_buffer(
        self,
        purpose: str,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        index: int = 0,
    ) -> torch.Tensor:
        key = (purpose, dtype, tuple(shape))
        buffers = self._pinned_d2h_buffers[key]
        while len(buffers) <= index:
            buffers.append(
                torch.empty(key[2], dtype=dtype, device="cpu", pin_memory=True)
            )
        return buffers[index]

    def _prematerialize_for_check_stop(
        self,
        outputs: dict[int, NameToTensorList],
        completion_event: torch.cuda.Event | None,
    ) -> dict[int, NameToTensorList]:
        """Side-stream D→H of every CUDA tensor in ``outputs`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns per-rid outputs with the CUDA tensors replaced by CPU
        copies. Skipped (returns ``outputs`` unchanged) when there's no
        completion event (CPU execution) or when CUDA is unavailable.

        AR engines emit small per-rid output dicts (sampled token + maybe
        a code) so the cost is negligible. If a future engine emits large
        tensors here (e.g. activations), revisit.
        """
        if not torch.cuda.is_available() or completion_event is None:
            return outputs
        if not outputs:
            return outputs

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(completion_event)

        cpu_per_rid: dict = {}
        buffer_indices: dict[tuple[str, torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        with torch.cuda.stream(side):
            for rid, name_to_list in outputs.items():
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
                            key = ("check_stop", t.dtype, tuple(t.shape))
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

        return cpu_per_rid

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
            initializer=self._init_cuda_executor_thread,
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
                initializer=self._init_cuda_executor_thread,
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
                p50 = vs[n // 2] * 1000
                p95 = vs[min(n - 1, int(n * 0.95))] * 1000
                mean = (sum(vs) / n) * 1000
                parts.append(f"{name}: p50={p50:.2f}ms p95={p95:.2f}ms mean={mean:.2f}ms n={n}")
            logger.info(
                "Worker %s phase-timing iter=%d: %s",
                self.worker_id, phase_iter[0], " | ".join(parts),
            )
            phase_buf.clear()

        # Reset per iteration (not just where they're first used) so the
        # error handler below sees only this iteration's work — a stale
        # ``batch`` from a previous pass would otherwise be failed twice, and
        # a raise before the first assignment would hit UnboundLocalError
        # inside the handler itself.
        batch: ScheduledBatch | None = None
        spec_pending: PendingBatch | None = None

        while True:
            from mstar.utils.profiler import range_pop, range_push
            try:
                batch = None
                spec_pending = None
                _iter_start = _time.perf_counter() if phase_period else 0.0
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
                    if not must_yield_away:
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
                    if speculation is None:
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
                                for rid, edges in speculation.consumed_streaming_edges.items():
                                    for edge in edges:
                                        self._return_speculative_streaming_edge(rid, edge)
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
                        # ``_handle_admit_failure`` pushes the GraphNodes back
                        # to the scheduler queue, and on KV-cache OOM also
                        # offloads or holds the failed rids.
                        self._handle_admit_failure(
                            pending.batch, pending.node_batch
                        )
                        self._clear_speculative_flag(pending.batch)
                        _maybe_clear_spec()

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
                        # set node._speculatively_scheduled to true, so that it doesn't
                        # accidentally get put on the ready queue while already executing
                        # this does not include the dropped rids
                        self._set_speculative_flag(spec_batch, True)

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
                            self.wakeup_event.register_future(spec_future)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                                range_push("worker.gpu_submit_queued", synchronize=False)
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
                # Same accounting as the speculative path above. Without it a
                # workload that never speculates -- image generation is one --
                # flushes nothing, so MSTAR_PHASE_TIMING silently reports no
                # breakdown at all, and the shared sample buffer (cleared only
                # by _phase_flush) grows for the life of the run. The idle
                # `batch is None` spin above is deliberately NOT counted:
                # it is a wait, and counting it would both skew iter_total and
                # make the flush period mean something other than iterations.
                if phase_period:
                    _phase_record("iter_total", _time.perf_counter() - _iter_start)
                    phase_iter[0] += 1
                    _phase_flush()
            except Exception as e:
                self._handle_main_loop_error(e, (pending, spec_pending), batch)
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
