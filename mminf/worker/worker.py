import logging
import time as _time
from enum import Enum
from time import sleep

import torch
import torch.distributed as dist

from mminf.api_server.request_types import APIServerMessage, ResultTensors
from mminf.communication.communicator import (
    CommProtocol,
    MOONCAKE_PROTOCOLS,
    ZMQCommunicator,
)
from mminf.communication.tensors import (
    EdgeSpec,
    MooncakeCommunicationManager,
    NameToTensorList,
    NVSHMEMCommunicationManager,
    TensorCommunicationManager,
    create_tensor_communication_manager,
)
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import EngineType, NodeBatch, NodeOutput
from mminf.engine.kv_store import KVCacheConfig, StoreWritePolicy, TransferEngineInfo
from mminf.graph.base import FilteredEdges, GraphEdge, TensorPointerInfo
from mminf.graph.request_queues import format_graph_edge_list
from mminf.graph.special_destinations import SPECIAL_DESTINATIONS
from mminf.model.base import Model, WorkerGraph
from mminf.streaming.stream_buffer import StreamBuffer
from mminf.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    NewRequest,
    RemoveRequest,
    StopLoops,
    TensorReceived,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)
from mminf.worker.engine_manager import EngineManager
from mminf.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mminf.worker.node_manager_utils import (
    NodeOutputRouting,
    WorkerGraphQueues,
    WorkerGraphsManager,
)

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Strategy for choosing which request to offload to CPU on OOM."""
    LRU = "lru"              # least-recently-used (by execution time)
    MOST_PAGES = "most_pages"  # request holding the most GPU pages


class Worker:
    """Worker that integrates WorkerGraphsManager, EngineManager, MicroScheduler,
    and one-or-two TensorCommunicationManagers (Mooncake + optional NVSHMEM)
    to execute computation via engines.

    Dual-manager design (NVSHMEM mode):
    - ``self.managers: dict[CommProtocol, TensorCommunicationManager]`` —
      Mooncake manager always installed (handles api_server↔worker hops).
      NVSHMEM manager installed when this worker joins the NVSHMEM PG.
    - ``self._uuid_to_manager: dict[str, TensorCommunicationManager]`` —
      populated at produce-time (store) and consume-time (start_read_tensors).
      Used for ACK routing, tensor retrieval, and Loop output refcount.
    - ``resolve_transport(edge)`` — picks NVSHMEM or Mooncake per
      ``GraphEdge.transport`` and AUTO-resolution rules.
    """

    # ------------------------------------------------------------------
    # Captured-transport watchdog (per §7.1 of the captured-NVSHMEM design doc)
    # ------------------------------------------------------------------
    REPLAY_TIMEOUT_S: float = 30.0
    REPLAY_POLL_INTERVAL_S: float = 0.001

    class WorkerAbort(RuntimeError):
        """Raised by replay_with_watchdog when a captured replay exceeds its
        deadline (indicating a dead peer or hung NVSHMEM operation). The main
        loop propagates this to process exit; the Conductor surfaces the
        failure and restarts the worker pool."""

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        model: Model,
        my_worker_graphs: list[WorkerGraph],
        kv_config: dict[str, KVCacheConfig],
        model_config: dict,
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, set[str]],
        all_worker_graph_ids_to_dyn_loops: dict[str, set[str]],
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        mooncake_port: int = 8080,
        tcp_transfer_device: str = "",
        # NVSHMEM additions (default-safe so existing callers don't break)
        master_service: str = "localhost:50051",
        node_to_rank: dict[str, int] | None = None,
        edge_specs: list[EdgeSpec] | None = None,
    ):
        self.worker_id = worker_id
        self.device = device
        self.enable_nvtx = enable_nvtx

        # Build node_to_partition mapping from model's partitions and graph walks
        node_to_partition: dict[str, str] = {}
        if model is not None:
            partitions = model.get_partitions()
            walks = model.get_graph_walk_graphs()
            for pdef in partitions:
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section:
                        for node_name in section.get_node_names():
                            node_to_partition[node_name] = pdef.name

        self.communicator = ZMQCommunicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )

        # ------------------------------------------------------------------
        # Dual-manager setup
        # ------------------------------------------------------------------
        self.managers: dict[CommProtocol, TensorCommunicationManager] = {}
        # Maps uuid → manager that produced/is consuming it. Used for:
        #   (a) producer side: ACK routing in _handle_tensor_received
        #   (b) consumer side: tensor retrieval in _build_node_batch /
        #       _cleanup_consumed_inputs / _send_outputs
        #   (c) Loop output refcount (via the resolver passed to
        #       WorkerGraphQueues.manager_resolver)
        self._uuid_to_manager: dict[str, TensorCommunicationManager] = {}

        if tensor_comm_protocol == CommProtocol.NVSHMEM:
            # Derive rank from worker_id ("worker_0" → rank 0).
            nvshmem_rank = int(worker_id.removeprefix("worker_"))
            nvshmem_world = len(worker_ids)
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_service}",
                rank=nvshmem_rank,
                world_size=nvshmem_world,
            )
            nvshmem_group = dist.group.WORLD

            entity_id_to_rank = {
                wid: int(wid.removeprefix("worker_")) for wid in worker_ids
            }
            # Merge node name → rank so store_and_populate_graph_edges can
            # look up node names (e.g. "LLM", "vit_encoder") directly.
            if node_to_rank:
                entity_id_to_rank.update(node_to_rank)

            nvshmem_mgr = NVSHMEMCommunicationManager(
                my_entity_id=worker_id,
                rank=nvshmem_rank,
                world_size=nvshmem_world,
                device=self.device,
                communicator=self.communicator,
                group=nvshmem_group,
                entity_id_to_rank=entity_id_to_rank,
            )
            self.managers[CommProtocol.NVSHMEM] = nvshmem_mgr
            self.rank = nvshmem_rank
            # Mooncake uses RDMA for api_server↔NVSHMEM-worker hops.
            _mooncake_protocol = CommProtocol.RDMA
        else:
            self.rank = -1
            _mooncake_protocol = tensor_comm_protocol

        # Mooncake (or SHM) — always installed; handles api_server↔worker hops
        # and any edges that resolve to a non-NVSHMEM protocol. We use the
        # factory so SHM workers get SharedMemoryCommunicationManager and
        # RDMA/TCP workers get MooncakeCommunicationManager. ``hostname`` may
        # arrive as "{host}:{rendezvous_port}" because Conductor overloads it
        # to carry the NVSHMEM rendezvous addr; Mooncake's P2PHANDSHAKE needs
        # "{host}:{rpc_port}" (2 fields), so strip the rendezvous part here.
        # NVSHMEM init uses master_service, not hostname, so stripping is safe.
        _mooncake_hostname = hostname.split(":", 1)[0] if ":" in hostname else hostname
        mooncake_mgr = create_tensor_communication_manager(
            protocol=_mooncake_protocol,
            my_entity_id=worker_id,
            hostname=_mooncake_hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
        )
        for p in MOONCAKE_PROTOCOLS:
            self.managers[p] = mooncake_mgr
        self._mooncake_mgr = mooncake_mgr
        self._mooncake_protocol = _mooncake_protocol

        node_names = set()
        for wg in my_worker_graphs:
            node_names.update(wg.section.get_node_names())

        # KV transfers always use the Mooncake/SHM manager's transfer_engine
        # (NVSHMEM does not carry KV cache pages cross-rank).
        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            kv_config=kv_config,
            model_config=model_config,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=mooncake_mgr.my_session_id,
                transfer_engine=mooncake_mgr.transfer_engine,
            ),
            model=model,
            enable_nvtx=self.enable_nvtx,
        )

        # Resolver passed to WorkerGraphQueues so Loop sections can refcount
        # cached outputs against whichever manager produced each UUID.
        # Falls back to Mooncake if the UUID hasn't been registered yet
        # (e.g. UUIDs from EMPTY_DESTINATION edges or pre-registry sites).
        def _resolve_manager(uuid: str) -> TensorCommunicationManager:
            return self._uuid_to_manager.get(uuid, self._mooncake_mgr)

        self.worker_graphs_manager = WorkerGraphsManager(
            queues={
                worker_graph.worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph.worker_graph_id,
                    graph_walks=worker_graph.graph_walks,
                    worker_graph=worker_graph,
                    per_request_queues={},
                    manager_resolver=_resolve_manager,
                )
                for worker_graph in my_worker_graphs
            },
            per_request_info={},
            all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
            all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
            node_to_partition=node_to_partition,
        )

        self.scheduler = MicroScheduler(self.engine_manager)

        # Determine store write policy based on worker graph topology
        node_engine_types = model.get_node_engine_types() if model is not None else {}
        write_policy = self._compute_store_write_policy(
            my_worker_graphs, all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_nodes,
            node_engine_types=node_engine_types,
        )
        self.engine_manager.set_alloc_write_policies(write_policy)
        logger.info(
            "Worker %s: store write policy = %s", worker_id, write_policy.value
        )

        self._unprocessed_messages = {} # req_id -> messages for requests that are not in the queue

        # CPU offloading: LRU tracking and eviction policy
        self._last_active: dict[tuple[str, str], float] = {}  # (request_id, node_name) -> monotonic timestamp
        self.eviction_policy = EvictionPolicy.LRU

        # Streaming buffers: request_id -> edge_name -> list of tensors
        # (Legacy path — kept for models without PartitionTopology)
        self.streaming_buffers: dict[str, dict[str, list[torch.Tensor]]] = {}

        # New streaming path: PartitionTopology + StreamBuffer on consumer worker
        self.partition_topology = model.get_partition_topology() if model else None

        # Determine which partition this worker serves (by checking which node names
        # appear in my_worker_graphs vs the topology connections)
        self._my_consumer_connections = []
        if self.partition_topology:
            my_node_names = set()
            for wg in my_worker_graphs:
                my_node_names.update(wg.section.get_node_names())
            for conn in self.partition_topology.connections:
                if any(n in my_node_names for n in self._get_node_names_for_partition(conn.to_partition, model)):
                    self._my_consumer_connections.append(conn)

        # Set of edge names that arrive via streaming (used to distinguish
        # streaming inputs from conductor-triggered non-streaming inputs).
        self._streaming_edge_names: set[str] = {
            conn.edge_name for conn in self._my_consumer_connections
        }

        # Build consumer node cache: edge_name -> next_node name
        self._consumer_node_cache: dict[str, str] = {}
        if self._my_consumer_connections and model:
            walks = model.get_graph_walk_graphs()
            for conn in self._my_consumer_connections:
                for section in walks.values():
                    if hasattr(section, 'input_ids') and conn.edge_name in section.input_ids:
                        self._consumer_node_cache[conn.edge_name] = section.name

        # EdgeSpec list assigned by the Conductor for this worker's rank.
        # Consumed by run() before engine warmup to bring up the captured
        # NVSHMEM-transport symmetric heap.
        self._captured_edge_specs: list[EdgeSpec] = list(edge_specs or [])

    # ------------------------------------------------------------------
    # Transport resolution
    # ------------------------------------------------------------------

    def resolve_transport(self, edge: GraphEdge) -> CommProtocol:
        """Determine the CommProtocol to use for a given GraphEdge.

        Resolution rules (in priority order):
        1. Explicit override: edge.transport != AUTO → use it directly (validates
           that the manager is installed).
        2. EMIT_TO_CLIENT / non-worker destinations → Mooncake (api_server is not
           an NVSHMEM peer).
        3. Both producer and consumer are in the NVSHMEM process group AND on
           different ranks → NVSHMEM.
        4. Fallback → Mooncake.
        """
        if edge.transport != CommProtocol.AUTO:
            # Rule 1: explicit override wins.
            if edge.transport in MOONCAKE_PROTOCOLS:
                return edge.transport
            if edge.transport == CommProtocol.NVSHMEM:
                if edge.next_node in SPECIAL_DESTINATIONS:
                    raise ValueError(
                        f"Transport NVSHMEM explicitly requested for edge "
                        f"'{edge.name}' → '{edge.next_node}' which is a special "
                        "destination (e.g. EMIT_TO_CLIENT); api_server is not "
                        "an NVSHMEM peer."
                    )
                if CommProtocol.NVSHMEM not in self.managers:
                    raise ValueError(
                        f"Transport NVSHMEM explicitly requested for edge "
                        f"'{edge.name}' → '{edge.next_node}' but this worker "
                        "has no NVSHMEM manager installed."
                    )
                return CommProtocol.NVSHMEM
            raise ValueError(f"Unsupported CommProtocol override: {edge.transport}")

        # Rule 2: special destinations (EMIT_TO_CLIENT, EMPTY_DESTINATION) → Mooncake.
        if edge.next_node in SPECIAL_DESTINATIONS:
            return self._mooncake_protocol

        # Rule 3: next_node is in the NVSHMEM process group AND on a different
        # rank → NVSHMEM. Same-rank self-edges (e.g. LLM→latents→LLM in the
        # image_gen denoising loop) fall through to Rule 4: the staging-slot
        # ACK would never arrive in a single-threaded worker loop, causing
        # deadlock. Mooncake uses the local tensor store with no ACK needed.
        if CommProtocol.NVSHMEM in self.managers:
            nvshmem_mgr = self.managers[CommProtocol.NVSHMEM]
            if edge.next_node in nvshmem_mgr.entity_id_to_rank:
                consumer_rank = nvshmem_mgr.entity_id_to_rank[edge.next_node]
                if consumer_rank != nvshmem_mgr.rank:
                    return CommProtocol.NVSHMEM

        # Rule 4: fallback → Mooncake.
        return self._mooncake_protocol

    def _get_manager(self, protocol: CommProtocol) -> TensorCommunicationManager:
        """Return the manager for a resolved (non-AUTO) CommProtocol."""
        if protocol in self.managers:
            return self.managers[protocol]
        raise RuntimeError(
            f"No manager installed for protocol {protocol} on worker {self.worker_id}"
        )

    def _mgr(self, uuid: str) -> TensorCommunicationManager:
        """Look up the manager that owns a given UUID, defaulting to Mooncake."""
        return self._uuid_to_manager.get(uuid, self._mooncake_mgr)

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

    def _compute_store_write_policy(
        self,
        my_worker_graphs: list[WorkerGraph],
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, set[str]],
        node_engine_types: dict[str, EngineType] | None = None,
    ) -> StoreWritePolicy:
        """Determine whether this worker needs to write KV to the mooncake store.

        If this worker handles ALL AR engine graph walks, no other worker
        needs its KV cache — return NEVER. Otherwise return ALWAYS.
        """
        my_ar_walks_nodes: set[str] = set()
        all_ar_walks_nodes: set[str] = set()

        def _is_ar(node_name: str) -> bool:
            engine = self.engine_manager.node_to_engine.get(node_name)
            if engine is not None:
                return engine.engine_type() == EngineType.AR
            if node_engine_types and node_name in node_engine_types:
                return node_engine_types[node_name] == EngineType.AR
            return False

        for wg in my_worker_graphs:
            for node_name in wg.section.get_node_names():
                if _is_ar(node_name):
                    my_ar_walks_nodes.update([(walk, node_name) for walk in wg.graph_walks])

        for wg_id, walks in all_worker_graph_ids_to_graph_walks.items():
            nodes = all_worker_graph_ids_to_nodes.get(wg_id, set())
            for node_name in nodes:
                if _is_ar(node_name):
                    all_ar_walks_nodes.update([(walk, node_name) for walk in walks])

        if not all_ar_walks_nodes:
            return StoreWritePolicy.NEVER

        if my_ar_walks_nodes == all_ar_walks_nodes:
            logger.info(
                "No LLM disaggregation detected; my_ar_walks_nodes == all_ar_walks_nodes: %s",
                str(my_ar_walks_nodes)
            )
            return StoreWritePolicy.NEVER

        return StoreWritePolicy.ALWAYS

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _add_new_request(self, body: NewRequest) -> None:
        logger.debug("Worker %s received request %s", self.worker_id, body.request_id)
        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is not None:
            for node_name in ar_engine.submodule_management.keys():
                self._last_active[(body.request_id, node_name)] = _time.monotonic()

        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            partition_worker_graph_ids=body.partition_worker_graph_ids,
            worker_graph_to_worker=body.worker_graph_to_worker,
            current_fwd_info=body.request_info
        )
        self.engine_manager.add_request(body.request_id)

        # Create StreamBuffers for consumer connections on this worker
        for conn in self._my_consumer_connections:
            req_info = self.worker_graphs_manager.per_request_info[body.request_id]
            req_info.stream_buffers[conn.edge_name] = StreamBuffer(
                request_id=body.request_id,
                edge_name=conn.edge_name,
                from_partition=conn.from_partition,
                policy=conn.chunk_policy_factory(),
            )

        # Consumer dispatch: route each edge to the appropriate manager based
        # on TPI.source_rank. source_rank >= 0 → NVSHMEM; -1 → Mooncake.
        self._dispatch_start_read(body.request_id, body.initial_inputs)

        # Signal-only edges (tensor_info is None) can be processed immediately
        signal_only = [
            edge for edge in body.initial_inputs if len(edge.tensor_info) == 0
        ]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only
            )
        # process messages that may have come in out-of-order
        if body.request_id in self._unprocessed_messages:
            self._process_message_list(self._unprocessed_messages[body.request_id])
            del self._unprocessed_messages[body.request_id]

    def _dispatch_start_read(
        self, request_id: str, graph_edges: list[GraphEdge]
    ) -> None:
        """Route graph edges to the appropriate manager's start_read_tensors.

        Consumer dispatch rule: determined by TPI.source_rank.
          source_rank >= 0  → NVSHMEM (produced by an NVSHMEM peer rank)
          source_rank == -1 → Mooncake (produced by api_server or Mooncake worker)

        Populates _uuid_to_manager for downstream get_tensor / dereference calls.
        """
        nvshmem_edges: list[GraphEdge] = []
        mooncake_edges: list[GraphEdge] = []

        for edge in graph_edges:
            if not edge.tensor_info:
                continue  # signal-only
            first = edge.tensor_info[0]
            if first.source_rank >= 0:
                nvshmem_edges.append(edge)
                for info in edge.tensor_info:
                    self._uuid_to_manager[info.uuid] = self.managers.get(
                        CommProtocol.NVSHMEM, self._mooncake_mgr
                    )
            else:
                mooncake_edges.append(edge)
                for info in edge.tensor_info:
                    self._uuid_to_manager[info.uuid] = self._mooncake_mgr

        if nvshmem_edges and CommProtocol.NVSHMEM in self.managers:
            self.managers[CommProtocol.NVSHMEM].start_read_tensors(
                request_id, nvshmem_edges
            )
            logger.debug(
                "manager=nvshmem op=read request_id=%s edges=%s",
                request_id, [e.name for e in nvshmem_edges],
            )
        if mooncake_edges:
            self._mooncake_mgr.start_read_tensors(request_id, mooncake_edges)
            logger.debug(
                "manager=mooncake op=read request_id=%s edges=%s",
                request_id, [e.name for e in mooncake_edges],
            )

    def _remove_request(self, body: RemoveRequest) -> None:
        self.engine_manager.remove_request(body.request_id)
        self.worker_graphs_manager.remove_request(body.request_id)
        # Cleanup all unique managers (set() because Mooncake protocols all
        # alias to one manager).
        for mgr in set(self.managers.values()):
            mgr.cleanup_request(body.request_id)
        self.streaming_buffers.pop(body.request_id, None)

        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is not None:
            for node_name in ar_engine.submodule_management.keys():
                self._last_active.pop((body.request_id, node_name), None)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed transfer, free source buffers.

        Routes each ACK'd UUID to the manager that produced it via
        _uuid_to_manager. NVSHMEM UUIDs are batched into one handle_ack call
        (freeing one staging slot); Mooncake UUIDs are deref'd individually.
        """
        nvshmem_mgr = self.managers.get(CommProtocol.NVSHMEM)
        nvshmem_uuids: dict[str, int] = {}
        mooncake_uuids: dict[str, int] = {}

        for uuid, ref_cnt in body.successful_tensors.items():
            mgr = self._uuid_to_manager.get(uuid)
            if mgr is not None and mgr is nvshmem_mgr:
                nvshmem_uuids[uuid] = ref_cnt
            else:
                mooncake_uuids[uuid] = ref_cnt

        if nvshmem_uuids and nvshmem_mgr is not None:
            nvshmem_mgr.handle_ack(body.request_id, nvshmem_uuids)
        for uuid, ref_cnt in mooncake_uuids.items():
            self._mooncake_mgr.dereference(body.request_id, uuid, n=ref_cnt)

    def _process_new_inputs(self, body: InputSignals) -> None:
        logger.debug(
            "Received new signals %s at worker %s for request %s",
            format_graph_edge_list(body.inputs), self.worker_id, body.request_id
        )
        req_info = self.worker_graphs_manager.per_request_info.get(body.request_id)

        # Handle producer_done signal: mark all StreamBuffers for this request as done
        if body.producer_done:
            if req_info:
                for sbuf in req_info.stream_buffers.values():
                    if sbuf.from_partition in body.producer_done:
                        sbuf.signal_done()

        # Separate streaming edges — they'll be handled when tensors are ready
        non_streaming = [edge for edge in body.inputs if not edge.is_streaming]
        streaming_with_tensors = [edge for edge in body.inputs if edge.is_streaming and edge.tensor_info]

        # Only update fwd_info when there are non-streaming edges (i.e., this is
        # a conductor-triggered forward pass, not just streaming data from another
        # partition). Streaming-only InputSignals must not overwrite the current
        # partition's fwd_info.
        if non_streaming:
            self.worker_graphs_manager.update_request_info(
                body.request_id, current_fwd_info=body.request_info,
                partition_name=body.partition_name
            )

        # Consumer dispatch: route by source_rank.
        self._dispatch_start_read(body.request_id, non_streaming)
        if streaming_with_tensors:
            self._dispatch_start_read(body.request_id, streaming_with_tensors)
            for edge in streaming_with_tensors:
                stream_buf = req_info.stream_buffers[edge.name]
                for info in edge.tensor_info:
                    stream_buf.pre_read_register(info.uuid)

        # Signal-only non-streaming edges can be processed immediately
        signal_only = [edge for edge in non_streaming if len(edge.tensor_info) == 0]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id,
                inputs=signal_only,
            )

    def _unpersist_tensors(self, body: UnpersistTensors):
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            mgr = self._mgr(uuid)
            mgr.increment_ref(body.request_id, uuid, n=ref_cnt)
            mgr.set_persist(body.request_id, uuid, persist=False)

    def _stop_loops(self, body: StopLoops):
        if not self.worker_graphs_manager.has_partition(
            body.request_id, body.partition_name
        ):
            return
        fwd_info = self.worker_graphs_manager.get_fwd_info(
            body.request_id, body.partition_name
        )
        loop_names = set()
        for name, stop_time in body.loop_stop_times.items():
            if name not in fwd_info.loop_stop_times or stop_time.label_context_gt(
                fwd_info.loop_stop_times[name], name
            ):
                loop_names.add(name)
            fwd_info.loop_stop_times[name] = stop_time
        if loop_names:
            self.worker_graphs_manager.stop_loops(
                body.request_id, body.partition_name, loop_names
            )

    def _process_message_list(self, messages: list[WorkerMessage]):
        msg_types_needing_active_request = [
            WorkerMessageType.REMOVE_REQUEST,
            WorkerMessageType.INPUT_SIGNALS,
            WorkerMessageType.STOP_LOOPS
        ]
        for message in messages:
            if (
                message.message_type in msg_types_needing_active_request and \
                message.body.request_id not in self.worker_graphs_manager.per_request_info
            ):
                # got an out-of-order request
                self._unprocessed_messages.setdefault(
                    message.body.request_id, []
                ).append(message)
                continue
            if message.message_type == WorkerMessageType.NEW_REQUEST:
                self._add_new_request(message.body)
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

    def _process_messages(self) -> None:
        self._process_message_list(self.communicator.get_all_new_messages())

    # ------------------------------------------------------------------
    # Tensor readiness
    # ------------------------------------------------------------------

    def _route_streaming_tensor(self, request_id: str, edge: GraphEdge) -> None:
        """Route a streaming tensor to a StreamBuffer."""
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]

        for info in edge.tensor_info:
            mgr = self._mgr(info.uuid)
            tensor = mgr.get_tensor(request_id=request_id, uuid=info.uuid)
            stream_buf.put(info.uuid, tensor.clone())
            mgr.dereference(request_id, info.uuid)

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        for request_id, req_info in list(self.worker_graphs_manager.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                consumer_node = self._consumer_node_cache.get(edge_name, "")
                partition_name = self.worker_graphs_manager.get_partition_for_node(consumer_node)

                synthetic_edge = sbuf.pop_waiting_edge()

                if synthetic_edge is None and sbuf.has_chunk_ready():
                    chunk = sbuf.pop_chunk()
                    chunk_tensor = chunk.data.get("data")
                    if chunk_tensor is None:
                        # Empty chunk — producer done, no more data.
                        synthetic_edge = GraphEdge(
                            next_node=consumer_node,
                            name=edge_name,
                            tensor_info=[],
                        )
                    else:
                        # Streaming chunks always go through Mooncake; populate
                        # the registry so downstream get_tensor calls find them.
                        tensor_infos = self._mooncake_mgr.store_and_return_tensor_info(
                            request_id, {edge_name: [chunk_tensor]},
                        )
                        for info_list in tensor_infos.values():
                            for info in info_list:
                                self._uuid_to_manager[info.uuid] = self._mooncake_mgr
                        synthetic_edge = GraphEdge(
                            next_node=consumer_node,
                            name=edge_name,
                            tensor_info=tensor_infos.get(edge_name, []),
                        )

                if synthetic_edge is not None:
                    ingested = len(self.worker_graphs_manager.process_new_streaming_inputs(
                        request_id=request_id, inputs=[synthetic_edge],
                    )) == 0
                    if not ingested:
                        sbuf.store_uningested_edge(synthetic_edge)
                    elif sbuf.reached_final_chunk:
                        req_info.per_partition_info[partition_name].stream_partition_done = True

    def _check_ready_tensors(self) -> None:
        """Poll for completed transfers across all managers, feed ready edges
        to the worker graph queues."""
        for mgr in set(self.managers.values()):
            ready = mgr.get_ready_tensors()
            for request_id, edges in ready.items():
                streaming = [e for e in edges if e.is_streaming]
                normal = [e for e in edges if not e.is_streaming]

                for edge in streaming:
                    self._route_streaming_tensor(request_id, edge)

                if normal:
                    self.worker_graphs_manager.process_new_inputs(
                        request_id=request_id, inputs=normal,
                    )

    # ------------------------------------------------------------------
    # CPU offloading
    # ------------------------------------------------------------------

    def _try_offload_cold_request(
        self, node_name: str, batch_ids: set[str]
    ) -> str | None:
        """Offload one request's KV pages to CPU using the configured eviction policy.

        Prefers requests outside *batch_ids*. If none exist, falls back to
        picking a victim *within* the batch (the caller should then exclude
        it from execution).

        Returns the victim request_id, or None if offloading wasn't possible.
        """
        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is None:
            return None

        submod_mgmt = ar_engine.submodule_management[node_name]
        cache_mgmt = submod_mgmt.kv_management
        if cache_mgmt.cpu_page_pool is None:
            return None

        alloc = cache_mgmt.alloc_manager

        external: list[tuple[str, int]] = []
        in_batch: list[tuple[str, int]] = []
        for rid, labels in alloc.request_states.items():
            total_pages = sum(len(s.page_indices) for s in labels.values())
            if total_pages == 0:
                continue
            if rid in batch_ids:
                in_batch.append((rid, total_pages))
            else:
                external.append((rid, total_pages))

        candidates = external or in_batch
        if not candidates:
            return None

        victim_id = self._select_eviction_victim(node_name, candidates)
        freed = alloc.offload_request(victim_id, cache_mgmt.cpu_page_pool)
        logger.info(
            "Offloaded request %s to CPU (%d GPU pages freed, "
            "policy=%s, in_batch=%s)",
            victim_id, freed, self.eviction_policy.value,
            victim_id in batch_ids,
        )
        return victim_id if freed > 0 else None

    def _select_eviction_victim(
        self, node_name: str, candidates: list[tuple[str, int]]
    ) -> str:
        """Pick a victim from *candidates* based on ``self.eviction_policy``.

        Each candidate is ``(request_id, total_gpu_pages)``.
        """
        if self.eviction_policy == EvictionPolicy.MOST_PAGES:
            return max(candidates, key=lambda x: x[1])[0]

        return min(
            candidates,
            key=lambda x: (
                self._last_active.get((x[0], node_name), 0.0),
                -x[1],
            ),
        )[0]

    def _try_reload_request(self, node_name: str, request_id: str) -> bool:
        """Reload an offloaded request back to GPU. Returns True if reloaded."""
        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is None:
            return False

        submod_mgmt = ar_engine.submodule_management[node_name]
        cache_mgmt = submod_mgmt.kv_management
        if cache_mgmt.cpu_page_pool is None:
            return False

        if not cache_mgmt.cpu_page_pool.is_offloaded(request_id):
            return False

        try:
            cache_mgmt.alloc_manager.reload_request(
                request_id, cache_mgmt.cpu_page_pool
            )
            logger.info("Reloaded request %s from CPU to GPU", request_id)
            return True
        except RuntimeError:
            logger.debug("Cannot reload request %s yet (insufficient GPU pages)", request_id)
            return False

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_node_batch(self, batch: ScheduledBatch) -> NodeBatch:
        """Gather input tensors from the appropriate manager for each request."""
        per_request_inputs: dict[str, NameToTensorList] = {}
        per_request_info: dict[str, CurrentForwardPassInfo] = {}
        batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

        for request_id, node in batch.node_objects.items():
            tensors = {}
            for input_name in node.ready_inputs:
                tensors[input_name] = [
                    self._mgr(info.uuid).get_tensor(
                        request_id=request_id, uuid=info.uuid
                    ) for info in node.ready_inputs[input_name].tensor_info
                ]
            per_request_inputs[request_id] = tensors
            per_request_info[request_id] = self.worker_graphs_manager.get_fwd_info(request_id, batch_partition)

        return NodeBatch(
            node_name=batch.node_name,
            graph_walk=batch.graph_walk,
            request_ids=list(batch.node_objects.keys()),
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info
        )

    # ------------------------------------------------------------------
    # Input cleanup
    # ------------------------------------------------------------------

    def _cleanup_consumed_inputs(self, batch: ScheduledBatch) -> None:
        """Free input tensors that were consumed by the just-executed node."""
        for request_id, node in batch.node_objects.items():
            for graph_edge in node.ready_inputs.values():
                if graph_edge._persist_for_loop:
                    continue
                for info in graph_edge.tensor_info:
                    self._mgr(info.uuid).dereference(request_id, info.uuid)

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------

    def _store_outputs_and_finish_loops(
        self,
        batch: ScheduledBatch,
        output: "NodeOutput",
        filtered_outputs_per_request: dict[str, list[GraphEdge]],
    ) -> dict[str, FilteredEdges]:
        """Store output tensors via the appropriate manager (resolved per-edge)
        and finalize Loop bookkeeping.

        ``filtered_outputs_per_request`` contains, for each request, only the
        GraphNode output edges whose names are actually present in the
        submodule's returned output dict. Edges absent from the output dict
        (e.g., Talker non-last prefill which returns {}, or Thinker with
        audio_output=False which omits thinker_states) are excluded so that
        empty-tensor_info edges are not routed downstream.

        Each kept edge is grouped by ``resolve_transport(edge)`` and routed to
        the NVSHMEM manager (cross-rank worker→worker) or the Mooncake manager
        (everything else, including api_server hops). Both managers' outputs
        are merged into a single ``output_tensor_info`` dict for Loop caching.
        """
        output_edges: dict[str, FilteredEdges] = {}

        # Single sync before the rid loop avoids N serialized syncs across
        # register_for_send calls (each manager respects skip_cuda_sync).
        if torch.cuda.is_available() and batch.node_objects:
            torch.cuda.default_stream().synchronize()

        for request_id, node in batch.node_objects.items():
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            )  # name -> list of tensors
            filtered_outputs = filtered_outputs_per_request.get(request_id, [])
            output_edges[request_id] = FilteredEdges(
                kept=filtered_outputs,
                filtered_out=[]
            )

            if not request_output_tensors:
                continue  # Node produced no outputs (e.g., KV-cache-only prefill step)

            # Group filtered outputs by resolved transport.
            nvshmem_edges: list[GraphEdge] = []
            mooncake_edges: list[GraphEdge] = []
            for edge in filtered_outputs:
                if self.resolve_transport(edge) == CommProtocol.NVSHMEM:
                    nvshmem_edges.append(edge)
                else:
                    mooncake_edges.append(edge)

            output_tensor_info: dict[str, list[TensorPointerInfo]] = {}

            if nvshmem_edges and CommProtocol.NVSHMEM in self.managers:
                nvshmem_names = {e.name for e in nvshmem_edges}
                nvshmem_tensors = {
                    n: t for n, t in request_output_tensors.items()
                    if n in nvshmem_names
                }
                nvshmem_mgr = self.managers[CommProtocol.NVSHMEM]
                nv_info = nvshmem_mgr.store_and_populate_graph_edges(
                    request_id=request_id,
                    tensors=nvshmem_tensors,
                    graph_edges=nvshmem_edges,
                )
                if nv_info:
                    output_tensor_info.update(nv_info)
                for edge in nvshmem_edges:
                    for info in edge.tensor_info:
                        self._uuid_to_manager[info.uuid] = nvshmem_mgr
                logger.debug(
                    "manager=nvshmem op=store request_id=%s edges=%s",
                    request_id, [e.name for e in nvshmem_edges],
                )

            if mooncake_edges:
                mooncake_names = {e.name for e in mooncake_edges}
                mooncake_tensors = {
                    n: t for n, t in request_output_tensors.items()
                    if n in mooncake_names
                }
                mc_info = self._mooncake_mgr.store_and_populate_graph_edges(
                    request_id=request_id,
                    tensors=mooncake_tensors,
                    graph_edges=mooncake_edges,
                )
                if mc_info:
                    output_tensor_info.update(mc_info)
                for edge in mooncake_edges:
                    for info in edge.tensor_info:
                        self._uuid_to_manager[info.uuid] = self._mooncake_mgr
                logger.debug(
                    "manager=mooncake op=store request_id=%s edges=%s",
                    request_id, [e.name for e in mooncake_edges],
                )

            worker_graph_id = self.worker_graphs_manager.get_worker_graph_id_for_node(
                request_id, node_name=node.name
            )
            waiting_node = self.worker_graphs_manager.get_waiting_node(request_id, worker_graph_id)
            if waiting_node is not None:
                waiting_node.cache_outputs(output_tensor_info)
            output_edges[request_id] = self.worker_graphs_manager.complete_loops(
                request_id, worker_graph_id, output_edges[request_id].kept,
                done_node=batch.node_name
            )

            # Filtered-out edges still need their refcounts dropped, routed
            # via the per-uuid manager registry.
            for edge in output_edges[request_id].filtered_out:
                for info in edge.tensor_info:
                    self._mgr(info.uuid).dereference(request_id, info.uuid)

        return output_edges

    def _register_outputs(
        self,
        batch: ScheduledBatch,
        routing_per_request: dict[str, NodeOutputRouting],
    ):
        """For outputs going to other workers, register tensors for RDMA send
        on the Mooncake manager (NVSHMEM has no explicit registration step).
        """
        for request_id, _node in batch.node_objects.items():
            routing = routing_per_request[request_id]
            mooncake_uuids: set[str] = set()
            for edge in (
                routing.persist +
                sum(routing.to_workers.values(), start=[]) +
                routing.emit_to_client +
                sum(routing.streaming_to_workers.values(), start=[])
            ):
                for info in edge.tensor_info:
                    if self._uuid_to_manager.get(info.uuid) is self._mooncake_mgr:
                        mooncake_uuids.add(info.uuid)
            if mooncake_uuids:
                self._mooncake_mgr.register_for_send(
                    request_id=request_id, uuids=mooncake_uuids,
                    skip_cuda_sync=True,
                )

            for edge in routing.persist:
                for info in edge.tensor_info:
                    self._mgr(info.uuid).set_persist(
                        request_id=request_id, uuid=info.uuid, persist=True
                    )

    def _send_outputs(
        self, request_id: str, outputs: NodeOutputRouting,
        graph_walk: str | None = None,
        partition_name: str | None = None,
        prematerialized_new_tokens: dict[str, list[int]] | None = None,
    ) -> None:
        """Send outputs to other workers and to the conductor.

        Persist signals are buffered and sent together with the
        WORKER_GRAPHS_DONE message to avoid race conditions.

        ``prematerialized_new_tokens`` (optional): `{signal_name: [int, ...]}`
        for this request, where the caller has already done the D→H copy
        for the new-token tensors. When provided, this function skips the
        per-tensor ``.cpu()`` call — meaningful when the caller batched
        multiple requests' new-token transfers into a single D→H to avoid
        N serialized ``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` per step.
        """
        if graph_walk is None:
            graph_walk = self.worker_graphs_manager.get_graph_walk(request_id, partition_name)
        for worker_id, edges in outputs.to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)

        if outputs.persist:
            self.worker_graphs_manager.buffer_persist_signals(
                request_id, outputs.persist
            )

        if outputs.new_token_outputs:
            name_to_new_token: dict = {}
            for signal in outputs.new_token_outputs:
                if signal.name in name_to_new_token:
                    continue
                if (
                    prematerialized_new_tokens is not None
                    and signal.name in prematerialized_new_tokens
                ):
                    new_tokens = prematerialized_new_tokens[signal.name]
                else:
                    new_tokens = []  # list[int]
                    for tensor_info in signal.tensor_info:
                        tensor = self._mgr(tensor_info.uuid).get_tensor(
                            request_id=request_id, uuid=tensor_info.uuid,
                        )
                        new_tokens.extend(tensor.cpu().numpy().tolist())
                name_to_new_token[signal.name] = new_tokens

                self.worker_graphs_manager.buffer_new_tokens(
                    request_id, name_to_new_token
                )

        if outputs.emit_to_client:
            self.worker_graphs_manager.buffer_output_signals(
                request_id, outputs.emit_to_client
            )
            for graph_edge in outputs.emit_to_client:
                message = APIServerMessage(
                    message_type="result_tensors",
                    body=ResultTensors(
                        request_id=request_id,
                        modality=graph_edge.output_modality,
                        graph_edge=graph_edge,
                        fwd_pass_number=self.worker_graphs_manager.get_fwd_number(request_id, partition_name),
                        metadata={}
                    )
                )
                self.communicator.send("api_server", message)

        # Local streaming: route to StreamBuffer
        req_info = self.worker_graphs_manager.per_request_info[request_id]
        for edge in outputs.streaming_local:
            stream_buf = req_info.stream_buffers[edge.name]
            for info in edge.tensor_info:
                stream_buf.pre_read_register(info.uuid)
            self._route_streaming_tensor(request_id, edge)

        # Remote streaming: send to destination workers
        for worker_id, edges in outputs.streaming_to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)
        if outputs.completed_worker_graph_ids:
            fwd_info = self.worker_graphs_manager.get_fwd_info(request_id, partition_name)
            if partition_name is None:
                partition_name = getattr(fwd_info, 'partition_name', 'default')
            req_info = self.worker_graphs_manager.per_request_info.get(request_id)
            p_done = req_info.per_partition_info[partition_name].stream_partition_done \
                if req_info else False

            stream_consumed = {}
            if req_info:
                for edge_name, sbuf in req_info.stream_buffers.items():
                    stream_consumed[edge_name] = sbuf._consumed

            message = ConductorMessage(
                message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
                body=WorkerGraphsDone(
                    request_id=request_id,
                    worker_graph_ids=outputs.completed_worker_graph_ids,
                    persist_signals=self.worker_graphs_manager.flush_persist_signals(request_id),
                    new_tokens=self.worker_graphs_manager.flush_new_tokens(request_id),
                    output_signal_names=self.worker_graphs_manager.flush_output_signals(request_id),
                    per_label_seq_info=self.worker_graphs_manager.get_seq_info(request_id, partition_name),
                    partition_name=partition_name,
                    partition_done=p_done,
                    stream_tokens_consumed=stream_consumed,
                ),
            )
            self.communicator.send("conductor", message)

    # ------------------------------------------------------------------
    # Captured-NVSHMEM transport bootstrap + watchdog
    # ------------------------------------------------------------------

    def _init_captured_transport_if_enabled(self) -> None:
        """Bring up the captured NVSHMEM-transport symmetric heap + pad
        bootstrap on this worker's NVSHMEM manager, if any. No-op when NVSHMEM
        is not installed or when no EdgeSpecs were assigned to this rank."""
        nvshmem_mgr = self.managers.get(CommProtocol.NVSHMEM)
        if nvshmem_mgr is None:
            if self._captured_edge_specs:
                logger.warning(
                    "Worker %s received %d captured-transport EdgeSpecs but "
                    "has no NVSHMEM manager installed; ignoring",
                    self.worker_id, len(self._captured_edge_specs),
                )
            return
        if not self._captured_edge_specs:
            logger.info(
                "Worker %s: no captured-transport EdgeSpecs assigned; "
                "skipping init_edges/warmup", self.worker_id,
            )
            return
        nvshmem_mgr.init_edges(self._captured_edge_specs)
        nvshmem_mgr.warmup()
        logger.info(
            "Worker %s: captured NVSHMEM transport ready (n_edges=%d)",
            self.worker_id, len(self._captured_edge_specs),
        )

    def replay_with_watchdog(
        self,
        graph: "torch.cuda.CUDAGraph",
        stream: torch.cuda.Stream,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> float:
        """Replay a captured graph on ``stream`` with a deadline-polling
        watchdog. Returns the observed wall-clock replay latency in
        microseconds. Raises :class:`Worker.WorkerAbort` if the replay fails
        to complete within ``timeout_s`` (default :data:`REPLAY_TIMEOUT_S`)."""
        deadline_s = self.REPLAY_TIMEOUT_S if timeout_s is None else timeout_s
        poll_s = (
            self.REPLAY_POLL_INTERVAL_S if poll_interval_s is None else poll_interval_s
        )
        event = torch.cuda.Event()
        t0 = _time.perf_counter()
        graph.replay()
        event.record(stream)
        deadline = t0 + deadline_s
        while not event.query():
            if _time.perf_counter() > deadline:
                logger.error(
                    "Worker %s: captured replay exceeded deadline (%.1fs); aborting",
                    self.worker_id, deadline_s,
                )
                raise Worker.WorkerAbort(
                    f"captured replay exceeded {deadline_s:.1f}s deadline"
                )
            _time.sleep(poll_s)
        latency_us = (_time.perf_counter() - t0) * 1e6
        nvshmem_mgr = self.managers.get(CommProtocol.NVSHMEM)
        if nvshmem_mgr is not None:
            nvshmem_mgr.mark_replay(latency_us)
        return latency_us

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Bring up captured NVSHMEM-transport BEFORE engine warmup. Engine
        # capture paths may call record_send/record_recv which require a
        # warmed manager.
        self._init_captured_transport_if_enabled()

        # CUDA graph capture before entering the main loop
        self.engine_manager.warmup_all()

        while True:
            from mminf.utils.profiler import range_pop, range_push
            try:
                # 1. Process ZMQ messages (new requests, input signals, removals)
                if self.enable_nvtx:
                    range_push("worker.process_messages", synchronize=True)
                self._process_messages()
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 2. Check for ready transfers from all managers
                if self.enable_nvtx:
                    range_push("worker.check_ready_tensors", synchronize=True)
                self._check_ready_tensors()
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 2b. Poll StreamBuffers — pop chunks when ready, feed as normal inputs
                if self.enable_nvtx:
                    range_push("worker.poll_stream_buffers", synchronize=True)
                self._poll_stream_buffers()
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 3. Pick next batch via MicroScheduler
                if self.enable_nvtx:
                    range_push("worker.schedule", synchronize=True)
                batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
                if self.enable_nvtx:
                    range_pop(synchronize=True)
                if batch is None:
                    sleep(0.001)
                    continue

                # 4. Gather input tensors for the batch
                if self.enable_nvtx:
                    range_push("worker.build_node_batch", synchronize=True)
                node_batch = self._build_node_batch(batch)
                batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

                for request_id, req_info in node_batch.per_request_info.items():
                    req_info.dynamic_loop_iter_counts.update(
                        self.worker_graphs_manager.get_dynamic_loop_iters(
                            request_id, partition=batch_partition,
                        )
                    )
                    batch.node_objects[request_id].clear_outputs()
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 5. Execute via engine
                engine = self.engine_manager.get_engine(batch.node_name)
                logger.debug("Executing batch for node %s on engine %s", node_batch.node_name, str(type(engine)))
                if self.enable_nvtx:
                    range_push(
                        f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                        synchronize=True,
                    )
                try:
                    output = engine.execute_batch(node_batch)
                finally:
                    if self.enable_nvtx:
                        range_pop(synchronize=True)

                # 5a. Handle allocation failure: offload a victim, retry the rest
                if output.allocation_failed:
                    batch_ids = set(batch.node_objects.keys())
                    victim_id = self._try_offload_cold_request(node_batch.node_name, batch_ids)

                    for request_id, node in batch.node_objects.items():
                        wg_id = batch.request_to_worker_graph[request_id]
                        self.worker_graphs_manager.queues[wg_id].push_back_node(
                            request_id, node
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
                        self.scheduler.hold_requests(list(batch_ids))
                        logger.warning(
                            "OOM on node=%s walk=%s: no offload possible, "
                            "holding %d requests",
                            batch.node_name, batch.graph_walk, len(batch_ids),
                        )
                    continue

                # Update LRU timestamps for successfully executed requests
                if self.enable_nvtx:
                    range_push("worker.update_request_info", synchronize=True)
                now = _time.monotonic()
                for rid in batch.node_objects:
                    self._last_active[(rid, batch.node_name)] = now

                for rid, req_info in node_batch.per_request_info.items():
                    if req_info.dynamic_loop_stop_signals:
                        self.worker_graphs_manager.stop_loops(
                            rid, partition=batch_partition,
                            loop_names=req_info.dynamic_loop_stop_signals,
                            req_info=req_info, last_node_run=batch.node_name
                        )

                    self.worker_graphs_manager.update_request_info(
                        rid, current_fwd_info=req_info,
                        per_label_seq_info=req_info.per_label_seq_info,
                        partition_name=batch_partition,
                    )
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 5b. Free consumed input tensors
                if self.enable_nvtx:
                    range_push("worker.cleanup_inputs", synchronize=True)
                self._cleanup_consumed_inputs(batch)
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 6. Route outputs through WorkerGraphsManager first to determine routing.
                # Filter each node's output edges to only those the submodule actually
                # produced.
                if self.enable_nvtx:
                    range_push("worker.route_outputs", synchronize=True)
                filtered_outputs_per_request: dict[str, list[GraphEdge]] = {}
                for request_id, node in batch.node_objects.items():
                    request_output_tensors = output.per_request_output_tensors.get(
                        request_id, {}
                    )
                    filtered_outputs = [
                        e for e in node.outputs if e.name in request_output_tensors
                    ]
                    filtered_outputs_per_request[request_id] = filtered_outputs

                node_outputs = self._store_outputs_and_finish_loops(
                    batch, output=output,
                    filtered_outputs_per_request=filtered_outputs_per_request
                )

                routing_per_request: dict[str, NodeOutputRouting] = {}
                for request_id in batch.node_objects:
                    routing = self.worker_graphs_manager.process_node_outputs(
                        request_id, node_outputs[request_id].kept, graph_walk=batch.graph_walk
                    )
                    routing_per_request[request_id] = routing
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                    # 6b. send "loop done" messages to the corresponding workers
                    stop_loop_workers = {}
                    for loop_name in node_batch.per_request_info[request_id].dynamic_loop_stop_signals:
                        for worker in self.worker_graphs_manager.get_dyn_loop_workers(
                            request_id, batch_partition, loop_name
                        ):
                            stop_loop_workers.setdefault(worker, set()).add(loop_name)
                    for worker, loop_names in stop_loop_workers.items():
                        if worker == self.worker_id:
                            continue
                        self.communicator.send(
                            entity_id=worker,
                            msg=WorkerMessage(
                                message_type=WorkerMessageType.STOP_LOOPS,
                                body=StopLoops(
                                    request_id=request_id,
                                    loop_names=loop_names,
                                    loop_stop_times=node_batch.per_request_info[request_id].loop_stop_times,
                                    partition_name=batch_partition
                                )
                            )
                        )

                # 7. Store output tensors, register RDMA if needed
                if self.enable_nvtx:
                    range_push("worker.store_outputs", synchronize=True)
                self._register_outputs(batch, routing_per_request)
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                # 8. Send outputs to other workers / conductor.
                # Pre-materialize new-token tensors across all rids in a single
                # batched D→H to avoid N serialized cudaMemcpyAsync syncs.
                if self.enable_nvtx:
                    range_push("worker.send_outputs", synchronize=True)

                prematerialized_per_rid: dict[str, dict[str, list[int]]] = {}
                collected: list[tuple[str, str, torch.Tensor]] = []
                for rid in batch.node_objects.keys():
                    routing = routing_per_request[rid]
                    if not routing.new_token_outputs:
                        continue
                    seen_names: set[str] = set()
                    for signal in routing.new_token_outputs:
                        if signal.name in seen_names:
                            continue
                        seen_names.add(signal.name)
                        for tinfo in signal.tensor_info:
                            tensor = self._mgr(tinfo.uuid).get_tensor(
                                request_id=rid, uuid=tinfo.uuid,
                            )
                            collected.append((rid, signal.name, tensor))

                if collected:
                    lengths = [t.numel() for t in (tr for _, _, tr in collected)]
                    flat = torch.cat(
                        [t.flatten() for _, _, t in collected]
                    ).cpu().tolist()
                    off = 0
                    for (rid, sig_name, _), n in zip(collected, lengths, strict=True):
                        rid_map = prematerialized_per_rid.setdefault(rid, {})
                        rid_map.setdefault(sig_name, []).extend(flat[off:off + n])
                        off += n

                for request_id in batch.node_objects.keys():
                    self._send_outputs(
                        request_id, routing_per_request[request_id],
                        graph_walk=batch.graph_walk,
                        partition_name=batch_partition,
                        prematerialized_new_tokens=prematerialized_per_rid.get(
                            request_id
                        ),
                    )

                for _rid, req_info in node_batch.per_request_info.items():
                    req_info.dynamic_loop_stop_signals.clear()
                if self.enable_nvtx:
                    range_pop(synchronize=True)
            except Exception:
                logger.exception("Worker %s error in main loop", self.worker_id)
                sleep(0.01)
