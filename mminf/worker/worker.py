import logging
import time as _time
from enum import Enum
from time import sleep

import torch

from mminf.api_server.request_types import APIServerMessage, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager, NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import EngineType, NodeBatch, NodeOutput
from mminf.engine.kv_store import StoreWritePolicy, TransferEngineInfo
from mminf.graph.base import GraphEdge
from mminf.graph.request_queues import format_graph_edge_list
from mminf.model.base import Model, WorkerGraph
from mminf.streaming.stream_buffer import StreamBuffer
from mminf.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    NewRequest,
    RemoveRequest,
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
    """
    Real worker that integrates WorkerGraphsManager, EngineManager,
    MicroScheduler, and MooncakeCommunicationManager to execute
    computation via engines.
    """

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        my_worker_graphs: list[WorkerGraph],
        engine_configs: list[dict],
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, list[str]],
        hostname: str = "localhost",
        master_service: str = "localhost:50051",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        model: Model | None = None,
        mooncake_port: int=8080,
        tcp_transfer_device=""
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

        self.worker_graphs_manager = WorkerGraphsManager(
            queues={
                worker_graph.worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph.worker_graph_id,
                    graph_walks=worker_graph.graph_walks,
                    worker_graph=worker_graph,
                    per_request_queues={},
                )
                for worker_graph in my_worker_graphs
            },
            per_request_info={},
            all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
            node_to_partition=node_to_partition,
        )


        self.communicator = ZMQCommunicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        self.tensor_manager = MooncakeCommunicationManager(
            my_entity_id=worker_id,
            hostname=hostname,
            communicator=self.communicator,
            protocol=tensor_comm_protocol,
            tcp_transfer_device=tcp_transfer_device,
            device=self.device
        )

        self.engine_manager = EngineManager.from_config(
            engine_configs=engine_configs, device=device,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.engine
            ),
            model=model,
            enable_nvtx=self.enable_nvtx
        )
        self.scheduler = MicroScheduler(self.engine_manager)

        # Determine store write policy based on worker graph topology
        node_engine_types = model.get_node_engine_types() if model is not None else {}
        write_policy = self._compute_store_write_policy(
            my_worker_graphs, all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_nodes,
            node_engine_types=node_engine_types,
        )
        alloc_mgr = self.engine_manager.get_ar_alloc_manager()
        if alloc_mgr is not None:
            alloc_mgr.write_policy = write_policy
        logger.info(
            "Worker %s: store write policy = %s", worker_id, write_policy.value
        )

        self._unprocessed_messages = {} # req_id -> messages for requests that are not in the queue

        # CPU offloading: LRU tracking and eviction policy
        self._last_active: dict[str, float] = {}  # request_id -> monotonic timestamp
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
                for _walk_name, section in walks.items():
                    if hasattr(section, 'input_ids') and conn.edge_name in section.input_ids:
                        self._consumer_node_cache[conn.edge_name] = section.name

        # Give all engines a reference to streaming buffers (for legacy path)
        if not self._my_consumer_connections:
            for engine in self.engine_manager._unique_engines():
                engine.set_streaming_buffers(self.streaming_buffers)

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
        all_worker_graph_ids_to_nodes: dict[str, list[str]],
        node_engine_types: dict[str, EngineType] | None = None,
    ) -> StoreWritePolicy:
        """Determine whether this worker needs to write KV to the mooncake store.

        If this worker handles ALL AR engine graph walks, no other worker
        needs its KV cache — return NEVER. Otherwise return ALWAYS.
        """
        my_ar_walks_nodes: set[str] = set()
        all_ar_walks_nodes: set[str] = set()

        def _is_ar(node_name: str) -> bool:
            # Check local engine first, then fall back to model's type map
            engine = self.engine_manager.node_to_engine.get(node_name)
            if engine is not None:
                return engine.engine_type() == EngineType.AR
            if node_engine_types and node_name in node_engine_types:
                return node_engine_types[node_name] == EngineType.AR
            return False

        # Collect this worker's AR graph walks
        for wg in my_worker_graphs:
            for node_name in wg.section.get_node_names():
                if _is_ar(node_name):
                    my_ar_walks_nodes.update([(walk, node_name) for walk in wg.graph_walks])

        # Collect all workers' AR graph walks
        for wg_id, walks in all_worker_graph_ids_to_graph_walks.items():
            nodes = all_worker_graph_ids_to_nodes.get(wg_id, [])
            for node_name in nodes:
                if _is_ar(node_name):
                    all_ar_walks_nodes.update([(walk, node_name) for walk in walks])

        if not all_ar_walks_nodes:
            return StoreWritePolicy.NEVER  # no AR engines at all

        if my_ar_walks_nodes == all_ar_walks_nodes:
            logger.info(
                "No LLM disaggregation detected; my_ar_walks_nodes == all_ar_walks_nodes: %s",
                str(my_ar_walks_nodes)
            )
            return StoreWritePolicy.NEVER  # all AR walks on this worker

        return StoreWritePolicy.ALWAYS

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _add_new_request(self, body: NewRequest) -> None:
        logger.debug("Worker %s received request %s", self.worker_id, body.request_id)
        self._last_active[body.request_id] = _time.monotonic()
        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            worker_graph_ids=body.worker_graph_ids,
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

        # Start RDMA reads for tensors that have tensor_info
        self.tensor_manager.start_read_tensors(
            body.request_id, body.initial_inputs,
        )

        # Signal-only edges (tensor_info is None) can be processed immediately
        signal_only = [
            edge for edge in body.initial_inputs if len(edge.tensor_info) == 0
        ]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only
            )
        # process messages that may have came in out-of-order
        if body.request_id in self._unprocessed_messages:
            self._process_message_list(self._unprocessed_messages[body.request_id])
            del self._unprocessed_messages[body.request_id]


    def _remove_request(self, body: RemoveRequest) -> None:
        self.engine_manager.remove_request(body.request_id)
        self.worker_graphs_manager.remove_request(body.request_id)
        self.tensor_manager.cleanup_request(body.request_id)
        self._last_active.pop(body.request_id, None)
        self.streaming_buffers.pop(body.request_id, None)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for (uuid, ref_cnt) in body.successful_tensors.items():
            self.tensor_manager.dereference(
                body.request_id, uuid, n=ref_cnt
            )

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
            self.worker_graphs_manager.update_request_info(
                body.request_id, current_fwd_info=body.request_info,
                partition_name=body.partition_name
            )

        # Start RDMA reads for non-streaming edges with tensor_info
        self.tensor_manager.start_read_tensors(
            body.request_id, non_streaming,
        )
        # Start RDMA reads for streaming edges with tensor_info (will be routed to buffer in _check_ready_tensors)
        if streaming_with_tensors:
            self.tensor_manager.start_read_tensors(
                body.request_id, streaming_with_tensors,
            )
            for edge in streaming_with_tensors:
                stream_buf = req_info.stream_buffers[edge.name]
                for info in edge.tensor_info:
                    stream_buf.pre_read_register(info.uuid)

        # Streaming signal-only edges: nothing to buffer (no tensor data)
        # This shouldn't normally happen for streaming edges

        # Signal-only non-streaming edges can be processed immediately
        signal_only = [edge for edge in non_streaming if len(edge.tensor_info) == 0]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id,
                inputs=signal_only,
            )

    def _unpersist_tensors(self, body: UnpersistTensors):
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            self.tensor_manager.increment_ref(
                body.request_id, uuid, n=ref_cnt
            )
            self.tensor_manager.set_persist(
                body.request_id, uuid, persist=False
            )

    def _process_message_list(self, messages: list[WorkerMessage]):
        msg_types_needing_active_request = [
            WorkerMessageType.REMOVE_REQUEST,
            WorkerMessageType.INPUT_SIGNALS,
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

    def _process_messages(self) -> None:
        self._process_message_list(self.communicator.get_all_new_messages())

    # ------------------------------------------------------------------
    # Tensor readiness
    # ------------------------------------------------------------------

    def _route_streaming_tensor(self, request_id: str, edge: GraphEdge) -> None:
        """Route a streaming tensor to either a StreamBuffer"""
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]

        for info in edge.tensor_info:
            tensor = self.tensor_manager.get_tensor(
                request_id=request_id, uuid=info.uuid,
            )
            stream_buf.put(info.uuid, tensor.clone())
            self.tensor_manager.dereference(request_id, info.uuid)

    def _find_waiting_node(
        self,
        request_id: str,
        node_name: str,
        partition_name: str | None = None,
    ) -> "GraphNode | None":
        """Find the GraphNode with *node_name* in this request's ACTIVE worker graph queues.

        If ``partition_name`` is given, only worker graphs that are active
        for that partition's currently-running walk are searched.  This is
        critical for partitions with multiple walks sharing a node name
        (e.g., the Talker has ``talker_prefill`` AND ``talker_decode``, both
        with ``name="Talker"`` but different ``input_ids``).  Without this
        filter, an inactive walk's stale queue could be returned, and the
        non-streaming gating logic would check the wrong node's input_ids.

        Searches both the ``waiting`` section and the ``ready`` list so that we
        can inspect ``input_ids`` and ``ready_inputs`` before deciding whether
        to ingest streaming data.
        """
        from mminf.graph.base import GraphNode as _GN
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        if req_info is None:
            return None

        # Narrow the search to only the active worker graphs for the
        # consumer partition if one was provided.  Otherwise fall back to
        # all worker graphs for the request (backwards-compatible).
        if partition_name is not None:
            part_info = req_info.per_partition_info.get(partition_name)
            if part_info is None:
                return None
            candidate_wg_ids = part_info.graph_walk_worker_graph_ids
        else:
            candidate_wg_ids = req_info.worker_graph_ids

        for wg_id in candidate_wg_ids:
            queue = self.worker_graphs_manager.queues.get(wg_id)
            if queue is None:
                continue
            per_req = queue.per_request_queues.get(request_id)
            if per_req is None:
                continue
            # Check ready list first
            for node in per_req.ready:
                if node.name == node_name:
                    return node
            # Check waiting section
            if per_req.waiting is not None:
                for name in per_req.waiting.get_node_names():
                    if name == node_name:
                        # Retrieve the actual GraphNode from the waiting section
                        # by walking its structure
                        found = self._extract_node_from_section(per_req.waiting, node_name)
                        if found is not None:
                            return found
        return None

    def _extract_node_from_section(self, section, node_name: str):
        """Recursively extract a GraphNode by name from a GraphSection tree."""
        from mminf.graph.base import GraphNode as _GN
        if isinstance(section, _GN):
            return section if section.name == node_name else None
        # Sequential, Parallel, Loop — they all have a `sections` or `section` attr
        if hasattr(section, 'sections'):
            for child in section.sections:
                result = self._extract_node_from_section(child, node_name)
                if result is not None:
                    return result
        if hasattr(section, 'section') and section.section is not None:
            result = self._extract_node_from_section(section.section, node_name)
            if result is not None:
                return result
        # Loop has curr_iter_section and nxt_iter_section
        for attr in ('curr_iter_section', 'nxt_iter_section'):
            child = getattr(section, attr, None)
            if child is not None:
                result = self._extract_node_from_section(child, node_name)
                if result is not None:
                    return result
        return None

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        for request_id, req_info in list(self.worker_graphs_manager.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                consumer_node = self._consumer_node_cache.get(edge_name, "")
                partition_name = self.worker_graphs_manager.get_partition_for_node(consumer_node)

                # Only ingest streaming data when the target node's non-streaming
                # inputs are already satisfied. This prevents the race condition
                # where streaming data lands in the wrong graph walk during
                # transitions.  We must pass ``partition_name`` so that
                # _find_waiting_node only searches the CURRENTLY ACTIVE walk's
                # worker graphs -- otherwise, for partitions with multiple walks
                # that share a node name (e.g., Talker: talker_prefill AND
                # talker_decode both have name="Talker" but different input_ids),
                # we might inspect the stale inactive walk's node and get
                # incorrect non-streaming gate decisions.
                target_graph_node = self._find_waiting_node(
                    request_id, consumer_node, partition_name=partition_name,
                )
                if target_graph_node is not None:
                    non_streaming_inputs = target_graph_node.input_ids - self._streaming_edge_names
                    if non_streaming_inputs and not non_streaming_inputs.issubset(
                        set(target_graph_node.ready_inputs.keys())
                    ):
                        continue  # conductor hasn't activated this node yet

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
                        )
                    else:
                        # Normal chunk — store tensor and create edge with tensor_info
                        tensor_infos = self.tensor_manager.store_and_return_tensor_info(
                            request_id, {edge_name: [chunk_tensor]},
                        )
                        synthetic_edge = GraphEdge(
                            next_node=consumer_node,
                            name=edge_name,
                            tensor_info=tensor_infos.get(edge_name, []),
                        )

                if synthetic_edge is not None:
                    ingested = len(self.worker_graphs_manager.process_new_inputs(
                        request_id=request_id, inputs=[synthetic_edge],
                    )) == 0
                    if not ingested:
                        sbuf.store_uningested_edge(synthetic_edge)
                    elif sbuf.reached_final_chunk:
                        req_info.per_partition_info[partition_name].stream_partition_done = True


    def _check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready graph edges to worker graph queues."""
        ready = self.tensor_manager.get_ready_tensors()
        for request_id, edges in ready.items():
            # Separate streaming edges from normal edges
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
        self, batch_ids: set[str]
    ) -> str | None:
        """Offload one request's KV pages to CPU using the configured eviction policy.

        Prefers requests outside *batch_ids*. If none exist, falls back to
        picking a victim *within* the batch (the caller should then exclude
        it from execution).

        Returns the victim request_id, or None if offloading wasn't possible.
        """
        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is None or ar_engine.cpu_page_pool is None:
            return None
        alloc = ar_engine.alloc_manager

        # Gather all candidates with (rid, total_pages), split by location
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

        # Prefer external victims; fall back to in-batch
        candidates = external or in_batch
        if not candidates:
            return None

        victim_id = self._select_eviction_victim(candidates)
        freed = alloc.offload_request(victim_id, ar_engine.cpu_page_pool)
        logger.info(
            "Offloaded request %s to CPU (%d GPU pages freed, "
            "policy=%s, in_batch=%s)",
            victim_id, freed, self.eviction_policy.value,
            victim_id in batch_ids,
        )
        return victim_id if freed > 0 else None

    def _select_eviction_victim(
        self, candidates: list[tuple[str, int]]
    ) -> str:
        """Pick a victim from *candidates* based on ``self.eviction_policy``.

        Each candidate is ``(request_id, total_gpu_pages)``.
        """
        if self.eviction_policy == EvictionPolicy.MOST_PAGES:
            return max(candidates, key=lambda x: x[1])[0]

        # LRU: pick the request with the oldest last_active timestamp.
        # Ties (or missing entries) broken by most pages.
        return min(
            candidates,
            key=lambda x: (
                self._last_active.get(x[0], 0.0),  # oldest first
                -x[1],                               # then most pages
            ),
        )[0]

    def _try_reload_request(self, request_id: str) -> bool:
        """Reload an offloaded request back to GPU. Returns True if reloaded."""
        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is None or ar_engine.cpu_page_pool is None:
            return False
        if not ar_engine.cpu_page_pool.is_offloaded(request_id):
            return False
        try:
            ar_engine.alloc_manager.reload_request(
                request_id, ar_engine.cpu_page_pool
            )
            logger.info("Reloaded request %s from CPU to GPU", request_id)
            return True
        except RuntimeError:
            # Not enough GPU pages to reload; will retry later
            logger.debug("Cannot reload request %s yet (insufficient GPU pages)", request_id)
            return False

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_node_batch(self, batch: ScheduledBatch) -> NodeBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_inputs: dict[str, NameToTensorList] = {}
        per_request_info: dict[CurrentForwardPassInfo] = {}
        batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

        for request_id, node in batch.node_objects.items():
            tensors = {}
            for input_name in node.ready_inputs:
                tensors[input_name] = [
                    self.tensor_manager.get_tensor(
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
                    self.tensor_manager.dereference(
                        request_id, info.uuid
                    )

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------

    def _store_outputs(
        self,
        batch: ScheduledBatch,
        output: "NodeOutput",
        routing_per_request: dict[str, NodeOutputRouting],
        filtered_outputs_per_request: dict[str, list[GraphEdge]],
    ) -> dict[str, list[GraphEdge]]:
        """
        For outputs going to other workers: register tensors for RDMA send
        and populate tensor_info on the GraphEdges.
        For outputs staying local: store tensors in tensor_manager.
        Returns the output edges per request (with tensor_info filled in).

        ``filtered_outputs_per_request`` contains, for each request, only the
        GraphNode output edges whose names are actually present in the
        submodule's returned output dict. Edges absent from the output dict
        (e.g., Talker non-last prefill which returns {}, or Thinker with
        audio_output=False which omits thinker_states) are excluded so that
        empty-tensor_info edges are not routed downstream.
        """
        output_edges: dict[str, list[GraphEdge]] = {}

        for request_id, node in batch.node_objects.items():
            # output name to list of tensors
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            ) # name -> list of tensors
            filtered_outputs = filtered_outputs_per_request.get(request_id, [])
            output_edges[request_id] = filtered_outputs

            if not request_output_tensors:
                continue  # Node produced no outputs (e.g., KV-cache-only prefill step)

            self.tensor_manager.store_and_populate_graph_edges(
                request_id=request_id,
                tensors=request_output_tensors,
                graph_edges=filtered_outputs
            )

            routing = routing_per_request[request_id]
            uuids = set()
            for edge in (
                routing.persist +
                sum(routing.to_workers.values(), start=[]) +
                routing.emit_to_client +
                sum(routing.streaming_to_workers.values(), start=[])
            ):
                uuids.update([
                    info.uuid for info in edge.tensor_info
                ])
            self.tensor_manager.register_for_send(
                request_id=request_id, uuids=uuids
            )

            for edge in routing.persist:
                for info in edge.tensor_info:
                    self.tensor_manager.set_persist(
                        request_id=request_id, uuid=info.uuid, persist=True
                    )

        return output_edges

    def _send_outputs(
        self, request_id: str, outputs: NodeOutputRouting,
        graph_walk: str | None = None,
        partition_name: str | None = None,
    ) -> None:
        """
        Send outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        WORKER_GRAPHS_DONE message to avoid race conditions.
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

        # Buffer persist signals for this request
        if outputs.persist:
            self.worker_graphs_manager.buffer_persist_signals(
                request_id, outputs.persist
            )

        if outputs.new_token_outputs:
            name_to_new_token: dict = {}
            for signal in outputs.new_token_outputs:
                if signal.name in name_to_new_token:
                    continue # don't double-count new tokens
                new_tokens = [] # list[int]
                for tensor_info in signal.tensor_info:
                    tensor = self.tensor_manager.get_tensor(
                        request_id=request_id,
                        uuid=tensor_info.uuid
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

        # Handle streaming edges
        # Local streaming: route to StreamBuffer or legacy buffer
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

            # Collect stream consumption info
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
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        # CUDA graph capture before entering the main loop
        self.engine_manager.warmup_all()

        while True:
            from mminf.utils.profiler import range_pop, range_push
            try:
                # 1. Process ZMQ messages (new requests, input signals, removals)
                self._process_messages()

                # 2. Check for ready RDMA tensors, feed to worker graph queues
                self._check_ready_tensors()

                # 2b. Poll StreamBuffers — pop chunks when ready, feed as normal inputs
                self._poll_stream_buffers()

                # 3. Pick next batch via MicroScheduler
                batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
                if batch is None:
                    sleep(0.01)
                    continue

                # 4. Gather input tensors for the batch
                node_batch = self._build_node_batch(batch)

                # 5. Execute via engine
                engine = self.engine_manager.get_engine(batch.node_name)
                logger.debug("Executing batch for node %s on engine %s", node_batch.node_name, str(type(engine)))
                if self.enable_nvtx:
                    range_push(
                        f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                        synchronize=False,
                    )
                try:
                    output = engine.execute_batch(node_batch)
                finally:
                    if self.enable_nvtx:
                        range_pop(synchronize=False)

                # 5a. Handle allocation failure: offload a victim, retry the rest
                if output.allocation_failed:
                    batch_ids = set(batch.node_objects.keys())
                    victim_id = self._try_offload_cold_request(batch_ids)

                    # Push all batch nodes back to their queues
                    for request_id, node in batch.node_objects.items():
                        wg_id = batch.request_to_worker_graph[request_id]
                        self.worker_graphs_manager.queues[wg_id].push_back_node(
                            request_id, node
                        )

                    if victim_id is not None:
                        # Only hold the offloaded victim (needs CPU→GPU reload)
                        self.scheduler.hold_requests([victim_id])
                        logger.warning(
                            "OOM on node=%s walk=%s: offloaded victim=%s, "
                            "retrying %d remaining requests",
                            batch.node_name, batch.graph_walk, victim_id,
                            len(batch_ids) - (1 if victim_id in batch_ids else 0),
                        )
                    else:
                        # No offloading possible; hold all requests briefly
                        self.scheduler.hold_requests(list(batch_ids))
                        logger.warning(
                            "OOM on node=%s walk=%s: no offload possible, "
                            "holding %d requests",
                            batch.node_name, batch.graph_walk, len(batch_ids),
                        )
                    continue

                # Update LRU timestamps for successfully executed requests
                now = _time.monotonic()
                for rid in batch.node_objects:
                    self._last_active[rid] = now

                batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

                for rid, req_info in node_batch.per_request_info.items():
                    self.worker_graphs_manager.update_request_info(
                        rid, per_label_seq_info=req_info.per_label_seq_info,
                        partition_name=batch_partition,
                    )

                # 5b. Free consumed input tensors
                self._cleanup_consumed_inputs(batch)

                # 6. Route outputs through WorkerGraphsManager first to determine routing.
                # Filter each node's output edges to only those the submodule actually
                # produced. This matters for cases like Talker non-last prefill (which
                # returns {} -> no edges routed) or Thinker with audio_output=False
                # (which omits thinker_states). Without filtering, edges whose names are
                # absent from the output dict would be routed with empty tensor_info.
                filtered_outputs_per_request: dict[str, list[GraphEdge]] = {}
                routing_per_request: dict[str, NodeOutputRouting] = {}
                for request_id, node in batch.node_objects.items():
                    request_output_tensors = output.per_request_output_tensors.get(
                        request_id, {}
                    )
                    filtered_outputs = [
                        e for e in node.outputs if e.name in request_output_tensors
                    ]
                    filtered_outputs_per_request[request_id] = filtered_outputs
                    routing = self.worker_graphs_manager.process_node_outputs(
                        request_id, filtered_outputs, graph_walk=batch.graph_walk
                    )
                    routing_per_request[request_id] = routing

                # 7. Store output tensors, register RDMA if needed
                self._store_outputs(
                    batch, output, routing_per_request, filtered_outputs_per_request
                )

                # 8. Send outputs to other workers / conductor
                for request_id in batch.node_objects.keys():
                    self._send_outputs(
                        request_id, routing_per_request[request_id],
                        graph_walk=batch.graph_walk,
                        partition_name=batch_partition,
                    )
            except Exception:
                logger.exception("Worker %s error in main loop", self.worker_id)
                sleep(0.01)
