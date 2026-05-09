import logging
import os
import sys
import threading
import time as _time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from time import sleep

import torch

from mminf.api_server.request_types import APIServerMessage, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.event import EventWakeup
from mminf.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import EngineType, NodeBatch, NodeOutput
from mminf.engine.kv_store import KVCacheConfig, StoreWritePolicy, TransferEngineInfo
from mminf.graph.base import FilteredEdges, GraphEdge
from mminf.graph.request_queues import format_graph_edge_list
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
from mminf.utils.profiler import range_pop, range_push
from mminf.worker.engine_manager import EngineManager
from mminf.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mminf.worker.node_manager_utils import (
    NodeOutputRouting,
    WorkerGraphQueues,
    WorkerGraphsManager,
)

logger = logging.getLogger(__name__)


@dataclass
class SlowPostprocessResult:
    prematerialized_new_tokens: dict[str, dict[str, list[int]]]
    new_stops: dict[str, set[str]]


@dataclass
class PendingPostproc:
    """In-flight slow-postprocess task awaiting finalization on the main thread.

    ``advanced_loops`` records, per request, the dynamic-loop names whose
    iter counter advanced during this batch's fast postprocess. Combined
    with ``new_stops`` from the slow postprocess, it lets
    ``_finalize_slow_postprocess`` clear ``_curr_iter_section`` on loops
    that both stopped and advanced — so the eventual ``register_finished``
    (via ``_pending_stops``) actually marks the loop done.
    """
    batch: "ScheduledBatch"
    node_batch: NodeBatch
    partition: str | None
    routing: dict[str, NodeOutputRouting]
    future: Future
    advanced_loops: dict[str, set[str]]


@dataclass
class Speculation:
    scheduled_batch: ScheduledBatch
    node_batch: NodeBatch
    loop_back_inputs: set[str]
    continuing_rids: set[str]
    streaming_edges: dict[str, list[GraphEdge]] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)

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

        self.communicator = ZMQCommunicator(
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
        )

        node_names = set()
        for wg in my_worker_graphs:
            node_names.update(wg.section.get_node_names())

        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            kv_config=kv_config,
            model_config=model_config,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.transfer_engine
            ),
            model=model,
            enable_nvtx=self.enable_nvtx
        )

        self.worker_graphs_manager = WorkerGraphsManager(
            queues={
                worker_graph.worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph.worker_graph_id,
                    graph_walks=worker_graph.graph_walks,
                    worker_graph=worker_graph,
                    per_request_queues={},
                    tensor_manager=self.tensor_manager
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

        # Async-scheduling cross-iter state. Initialized here (rather than in
        # run()) because _remove_request — which can be invoked indirectly
        # from _process_messages on any iter — reads/writes them.
        # _in_flight_rids: rids referenced by an in-flight GPU step or its
        #   speculation; REMOVE_REQUEST for these is deferred.
        # _pending_removes: deferred REMOVE_REQUESTs.
        # _pending_stops: deferred loop-stop signals from check_stop in the
        #   prior iter's slow_postprocess. Keyed by (rid, partition_name) —
        #   stops are partition-scoped because loop names live on a single
        #   partition's queues. Without partitioning, a stop generated by
        #   Thinker (e.g., ``thinker_decode_loop``) could be popped by a
        #   later Talker batch with the same rid; ``stop_loops`` would
        #   silently no-op on Talker partition (no such loop) and the stop
        #   would be lost — Thinker's loop would keep running. See the
        #   Qwen3-Omni audio-mode bug.
        self._in_flight_rids: set[str] = set()
        self._postproc_inflight_rids: set[str] = set()
        self._pending_removes: set[str] = set()
        self._pending_stops: dict[tuple[str, str | None], set[str]] = {}

        # Side stream for D→H copies of new tokens in slow_postprocess. The
        # default stream has GPU(N+1) queued behind GPU(N)'s tokens after
        # speculation, so syncing on default would also drain GPU(N+1) and
        # erase the streaming-latency win. The side stream waits on
        # ``output.completion_event`` (recorded after GPU(N)) and then runs
        # an isolated D→H, so the main thread only blocks on the copy.
        # Lazy-initialized — workers without CUDA never touch it.
        self._d2h_stream: "torch.cuda.Stream | None" = None
        self._pinned_d2h_buffers: dict[
            tuple[str, torch.dtype, tuple[int, ...]], list[torch.Tensor]
        ] = defaultdict(list)

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
                for section in walks.values():
                    if hasattr(section, 'input_ids') and conn.edge_name in section.input_ids:
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
            nodes = all_worker_graph_ids_to_nodes.get(wg_id, set())
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

        # Start RDMA reads for tensors that have tensor_info
        futures = self.tensor_manager.start_read_tensors(
            body.request_id, body.initial_inputs,
        )
        self.wakeup_event.register_futures(futures)

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
        # Async-scheduling deferral: if this rid is currently held by an
        # in-flight GPU step (or its speculation), tearing down engine /
        # tensor state now would race the GPU thread reading those tensors
        # / KV pages. Queue the remove and apply it once no in-flight step
        # references the rid (see _apply_pending_removes_safe_to_drop in
        # the run loop).
        if body.request_id in getattr(self, "_in_flight_rids", set()) or \
                body.request_id in getattr(self, "_postproc_inflight_rids", set()):
            self._pending_removes.add(body.request_id)
            return
        self.engine_manager.remove_request(body.request_id)
        self.worker_graphs_manager.remove_request(body.request_id)
        self.tensor_manager.cleanup_request(body.request_id)
        self.streaming_buffers.pop(body.request_id, None)

        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is not None:
            for node_name in ar_engine.submodule_management.keys():
                self._last_active.pop((body.request_id, node_name), None)

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
            self.worker_graphs_manager.update_request_info(
                body.request_id, current_fwd_info=body.request_info,
                partition_name=body.partition_name
            )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.start_read")
        # Start RDMA reads for non-streaming edges with tensor_info
        futures = self.tensor_manager.start_read_tensors(
            body.request_id, non_streaming,
        )
        self.wakeup_event.register_futures(futures)
        # Start RDMA reads for streaming edges with tensor_info (will be routed to buffer in _check_ready_tensors)
        if streaming_with_tensors:
            futures = self.tensor_manager.start_read_tensors(
                body.request_id, streaming_with_tensors,
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
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id,
                inputs=signal_only,
            )
        if self.enable_nvtx:
            range_pop()

    def _unpersist_tensors(self, body: UnpersistTensors):
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            self.tensor_manager.increment_ref(
                body.request_id, uuid, n=ref_cnt
            )
            self.tensor_manager.set_persist(
                body.request_id, uuid, persist=False
            )

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
        """Route a streaming tensor to either a StreamBuffer"""
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]

        for info in edge.tensor_info:
            tensor = self.tensor_manager.get_tensor(
                request_id=request_id, uuid=info.uuid,
            )

            stream_buf.put(info.uuid, tensor.clone())
            self.tensor_manager.dereference(request_id, info.uuid)
    
    def _pop_streaming_edge(
        self, sbuf: StreamBuffer, edge_name: str, request_id: str
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
                )
        return synthetic_edge

    def _poll_stream_buffers_for_speculation(
        self, request_id: str, node_name: str
    ) -> list[GraphEdge]:
        result = []
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
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
        self, request_id: str, edge: GraphEdge
    ):
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(edge.name)
        if sbuf is not None:
            sbuf.store_uningested_edge(edge)

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        for request_id, req_info in list(self.worker_graphs_manager.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                consumer_node = self._consumer_node_cache.get(edge_name, "")
                partition_name = self.worker_graphs_manager.get_partition_for_node(consumer_node)
                synthetic_edge = self._pop_streaming_edge(sbuf, edge_name, request_id)

                if synthetic_edge is not None:
                    ingested = len(self.worker_graphs_manager.process_new_streaming_inputs(
                        request_id=request_id, inputs=[synthetic_edge],
                    )) == 0
                    if not ingested:
                        sbuf.store_uningested_edge(synthetic_edge)
                    elif sbuf.reached_final_chunk:
                        req_info.per_partition_info[partition_name].stream_partition_done = True


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
                self.worker_graphs_manager.process_new_inputs(
                    request_id=request_id, inputs=normal,
                )
            if self.enable_nvtx:
                range_pop(synchronize=False)

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

        # LRU: pick the request with the oldest last_active timestamp.
        # Ties (or missing entries) broken by most pages.
        return min(
            candidates,
            key=lambda x: (
                self._last_active.get((x[0], node_name), 0.0),  # oldest first
                -x[1],                               # then most pages
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

    def _store_outputs_and_finish_loops(
        self,
        batch: ScheduledBatch,
        output: "NodeOutput",
        filtered_outputs_per_request: dict[str, list[GraphEdge]],
    ) -> dict[str, FilteredEdges]:
        """
        ``filtered_outputs_per_request`` contains, for each request, only the
        GraphNode output edges whose names are actually present in the
        submodule's returned output dict. Edges absent from the output dict
        (e.g., Talker non-last prefill which returns {}, or Thinker with
        audio_output=False which omits thinker_states) are excluded so that
        empty-tensor_info edges are not routed downstream.
        """
        output_edges: dict[str, FilteredEdges] = {}

        # tensor_manager.register_for_send would issue
        # `torch.cuda.default_stream().synchronize()` per rid — at bs=8 that
        # was 8 serialized syncs (+ their implicit API overhead). One sync
        # before the rid loop is enough: we only need the preceding forward's
        # writes to be visible on the source stream before we hand tensor
        # addresses to peers.
        #
        # Prefer ``output.completion_event.synchronize()`` over
        # ``default_stream().synchronize()`` here. With speculative
        # scheduling, GPU(N+1) has already been queued on the default
        # stream behind GPU(N)'s tokens; a plain default-stream sync would
        # block until GPU(N+1) drains too, undoing the overlap. The event
        # was recorded right after GPU(N), so syncing on it waits only for
        # GPU(N).
        if self.enable_nvtx:
            from mminf.utils.profiler import range_pop, range_push
            range_push("worker.route_outputs.store.cuda_sync", synchronize=False)
        if torch.cuda.is_available() and batch.node_objects:
            if output.completion_event is not None:
                if self.enable_nvtx:
                    range_push("worker.route_outputs.store.completion_event_sync", synchronize=False)
                output.completion_event.synchronize()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
            else:
                torch.cuda.default_stream().synchronize()
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.route_outputs.store.register", synchronize=False)

        for request_id, node in batch.node_objects.items():
            # output name to list of tensors
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            ) # name -> list of tensors
            filtered_outputs = filtered_outputs_per_request.get(request_id, [])
            output_edges[request_id] = FilteredEdges(
                kept=filtered_outputs,
                filtered_out=[]
            )

            if not request_output_tensors:
                continue  # Node produced no outputs (e.g., KV-cache-only prefill step)

            output_tensor_info = self.tensor_manager.store_and_populate_graph_edges(
                request_id=request_id,
                tensors=request_output_tensors,
                graph_edges=filtered_outputs,
                # We already synced on output.completion_event above,
                # which waits only for GPU(N) — the unconditional
                # default-stream sync inside store_and_return_tensor_info
                # would also drain the speculatively-queued GPU(N+1).
                skip_cuda_sync=True,
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

            # if any outputs were filtered out, we must dereference them
            for edge in output_edges[request_id].filtered_out:
                for info in edge.tensor_info:
                    self.tensor_manager.dereference(request_id, info.uuid)

        if self.enable_nvtx:
            range_pop(synchronize=False)  # worker.route_outputs.store.register
        return output_edges


    def _register_outputs(
        self,
        batch: ScheduledBatch,
        routing_per_request: dict[str, NodeOutputRouting],
    ):
        """
        For outputs going to other workers: register tensors for RDMA send
        and populate tensor_info on the GraphEdges.
        For outputs staying local: store tensors in tensor_manager.
        Returns the output edges per request (with tensor_info filled in).
        """
        for request_id, _node in batch.node_objects.items():
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
                request_id=request_id, uuids=uuids,
                skip_cuda_sync=True,
            )

            for edge in routing.persist:
                for info in edge.tensor_info:
                    self.tensor_manager.set_persist(
                        request_id=request_id, uuid=info.uuid, persist=True
                    )


    def _send_outputs(
        self, request_id: str, outputs: NodeOutputRouting,
        graph_walk: str | None = None,
        partition_name: str | None = None,
        prematerialized_new_tokens: dict[str, list[int]] | None = None,
    ) -> None:
        """
        Send outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        WORKER_GRAPHS_DONE message to avoid race conditions.

        ``prematerialized_new_tokens`` (optional): `{signal_name: [int, ...]}`
        for this request, where the caller has already done the D→H copy
        for the new-token tensors. When provided, this function skips the
        per-tensor ``.cpu()`` call — meaningful when the caller batched
        multiple requests' new-token transfers into a single D→H to avoid
        N serialized ``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` per
        step.
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
                if (
                    prematerialized_new_tokens is not None
                    and signal.name in prematerialized_new_tokens
                ):
                    new_tokens = prematerialized_new_tokens[signal.name]
                else:
                    new_tokens = []  # list[int]
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
    # Main loop — async scheduling (Option A')
    #
    # Pipeline shape:
    #   iter K (main thread):                          GPU thread
    #     CPU preamble  ───────────────► overlaps with execute_batch(N)
    #     speculate + build N+1
    #     await GPU(N).future Python return
    #     thread N's outputs → N+1's loop-back inputs
    #     submit GPU(N+1) ───────────────► execute_batch(N+1)
    #     fast_postprocess(N) ───────────► overlap with GPU(N+1)
    #     slow_postprocess(N) ───────────► overlap with GPU(N+1)
    #
    # Speculation scope (currently): AR engine only, intra-worker, 1-deep,
    # for rids whose loop is still continuing. Notes for extending to other
    # engines (b) and across partitions / cross-worker (c) live in
    # ASYNC_REDESIGN.md.
    # ------------------------------------------------------------------

    def _pre_plan_for_speculative_batch(
        self,
        engine,
        spec_node_batch: NodeBatch,
        prev_advance_event: "threading.Event | None",
    ) -> bool:
        """Phase 3 double-buffer: dispatch entry point on plan_executor.

        Waits on ``prev_advance_event`` — set by the GPU thread RIGHT AFTER
        ``advance_seq_lens(prev)`` runs (~tens of µs into prev replay) —
        rather than on the full prev_future. This is the key to overlap:
        plan(N+1) starts as soon as alloc_manager state is post-(N), which
        is well before replay(N)'s GPU work finishes. plan(N+1) runs
        concurrent with the rest of replay(N)'s GPU kernels on the disjoint
        slot's wrapper buffers. await_plan on the GPU thread should drop
        to ~0 because plan(N+1) has finished long before replay(N+1)
        begins.

        Returns True if pre-planning was applied; False otherwise — the
        caller submits the spec batch with plan_future regardless, so a
        False return means the GPU thread will plan inline (no skip).
        """
        try:
            if prev_advance_event is not None:
                # Safety timeout — should fire well within 100ms in normal
                # operation. If it doesn't (e.g., GPU thread crashed),
                # bail out rather than block plan_executor forever.
                if not prev_advance_event.wait(timeout=10.0):
                    logger.warning(
                        "Worker %s: plan_executor timed out waiting for "
                        "prev advance_event; skipping pre-plan",
                        self.worker_id,
                    )
                    self._reset_skip_plan_flags(spec_node_batch)
                    return False
            return engine.pre_plan_for_batch(
                spec_node_batch,
                prev_completion_event=None,
            )
        except Exception:
            logger.exception("Worker %s: plan_executor pre-plan failed", self.worker_id)
            self._reset_skip_plan_flags(spec_node_batch)
            return False

    def _reset_skip_plan_flags(self, spec_node_batch: NodeBatch) -> None:
        """Clear pre-plan state on the SPECIFIC slot that
        ``pre_plan_for_batch`` targeted for ``spec_node_batch``.

        Used to recover from speculation drops / failures where the pre-
        plan was dispatched but the spec batch never reached the GPU
        thread — leaving entries in the slot's ``_pre_planned_labels``
        would cause the next real plan_attention call on that slot to
        short-circuit incorrectly.

        Slot-targeted (not worker-global) so that any other slot's
        valid in-flight pre-plan whose flags have not yet been consumed
        by the matching replay isn't stomped. The engine's
        ``reset_pre_plan_for_batch`` looks up the same (key, slot) the
        pre-plan path used; absent that method (non-AR engine), this is
        a no-op (those engines don't pre-plan).
        """
        engine = self.engine_manager.get_engine(spec_node_batch.node_name)
        reset = getattr(engine, "reset_pre_plan_for_batch", None)
        if reset is not None:
            reset(spec_node_batch)

    def _execute_on_gpu_thread(
        self,
        batch: ScheduledBatch,
        node_batch: NodeBatch,
        plan_future: Future | None = None,
        advance_event: "threading.Event | None" = None,
    ) -> NodeOutput:
        """Run the engine on the GPU executor thread.

        The NVTX range bracketing this call is ``synchronize=False`` —
        adding a ``cudaDeviceSynchronize`` at the marker boundary would
        drain the GPU on every iter and hide the overlap between
        post-processing and the next step's kernel execution.

        After ``execute_with_max_batch_size`` returns we record a CUDA event
        on the default stream and stash it on the output. Downstream sync
        points on the main thread (`register_for_send` sync,
        side-stream-gated D→H of new tokens in ``_slow_postprocess``) wait
        on this event instead of `default_stream().synchronize()`. With
        speculation, the next GPU step has typically already been queued on
        the default stream by the time the main thread tries to sync, so a
        plain `synchronize()` would block on GPU(N+1)'s drain. The event
        was recorded *before* GPU(N+1) was submitted, so waiting on it
        returns as soon as GPU(N) is done.
        """
        from mminf.utils.profiler import range_pop, range_push

        engine = self.engine_manager.get_engine(batch.node_name)
        logger.debug(
            "Executing batch for node %s on engine %s",
            node_batch.node_name, str(type(engine))
        )
        if self.enable_nvtx:
            range_push("worker.gpu_thread_start", synchronize=False)
            range_pop(synchronize=False)
        # Phase 3: wait for the plan_executor's pre-planned wrapper.plan()
        # call to finish before running this batch — its results land on the
        # captured graph's persistent wrappers, and the next plan_attention
        # call(s) will see the matching label in _pre_planned_labels only
        # because plan_executor populated it. Wait releases the GIL.
        if plan_future is not None:
            if self.enable_nvtx:
                range_push("worker.gpu_thread.await_plan", synchronize=False)
            try:
                plan_future.result()
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)
        if self.enable_nvtx:
            range_push(
                f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                synchronize=False,
            )
        try:
            output = engine.execute_with_max_batch_size(node_batch)
            if torch.cuda.is_available():
                event = torch.cuda.Event()
                event.record(torch.cuda.default_stream(self.device))
                output.completion_event = event
            return output
        finally:
            # Phase 3 safety net: ensure advance_event fires even if the
            # engine raised before reaching ``advance_seq_lens`` inside
            # ``_run_basic_batched``. Without this, a plan_executor waiting
            # on prev_advance_event would block forever on the failure
            # path.
            if advance_event is not None:
                advance_event.set()
            # Same idea for launch_started_event: if the engine raised
            # before reaching the deep set site, release the main-thread
            # waiter early instead of making it eat the full timeout.
            launch_started_event = node_batch.metadata.get("launch_started_event")
            if launch_started_event is not None:
                launch_started_event.set()
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _handle_allocation_failure(
        self, batch: ScheduledBatch, node_batch: NodeBatch
    ) -> None:
        batch_ids = set(batch.node_objects.keys())
        victim_id = self._try_offload_cold_request(node_batch.node_name, batch_ids)

        # Push all batch nodes back to their queues
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

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

    def _can_speculate(self, batch: ScheduledBatch) -> bool:
        """True iff we can speculatively schedule the next step from this batch.

        Currently restricted to AR engine: AR decode loops have a stable next
        node (the same node looping back), and the next-step input tensor is
        produced by the current step's submodule.postprocess (rebound output
        name). FlowEngine / EncoderDecoderEngine / AudioCodecEngine don't have
        this property today, so we fall back to the non-speculative path for
        them — i.e. drain the in-flight step before scheduling the next.

        TODO(extension): generalize to (b) any same-engine walks (prefill →
        decode transitions, flow loop bodies) and (c) cross-engine /
        cross-worker (e.g. LLM → flow). See ASYNC_REDESIGN.md.
        """
        if any(
            not getattr(node, "enable_async_scheduling", True)
            for node in batch.node_objects.values()
        ):
            return False
        engine = self.engine_manager.get_engine(batch.node_name)
        return engine.engine_type() == EngineType.AR

    def _loop_back_input_names(self, node) -> set[str]:
        """For an AR loop body, the set of input names that come from this
        node's own loop-back outputs (edges where ``next_node == node.name``).

        For Orpheus/BAGEL/Qwen3-Omni decode loops this is ``{"text_inputs"}``;
        the submodule.postprocess rebinds ``new_token`` → ``text_inputs`` so
        the same name appears on both sides of the loop.
        """
        return {edge.name for edge in node.outputs if edge.next_node == node.name}

    def _try_speculate_next(
        self,
        batch_N: ScheduledBatch,
        partition_N: str | None,
    ) -> Speculation | None:
        """Build a speculative N+1 batch + node_batch.

        Returns ``(spec_batch, spec_node_batch, loop_back_inputs,
        continuing_rids)`` where ``continuing_rids`` are the subset of
        spec_batch's rids whose inputs need to be threaded from GPU(N)'s
        outputs (the rest are fresh rids whose inputs were already
        gathered from tensor_manager). Returns None when no continuing rids
        survive (the loop chain has fully drained / been stopped).

        The speculated batch is a merge of:
          * **continuing** rids (subset of batch_N still in the loop, not
            pending-stop / pending-remove) — placeholder inputs to be
            overwritten by GPU(N)'s outputs after we await.
          * **fresh** rids — newly-arrived requests whose decode-loop node
            is ready in the queue right now. Their inputs come from the
            usual tensor_manager path (same as ``_build_node_batch``).
            Without this merge, new rids have to wait for the entire
            current speculation chain to drain before they can be
            scheduled — a major regression for concurrent throughput.

        Speculation requires the loop body's required inputs to be a subset
        of its loop-back outputs (i.e. every input name has a same-name
        ``next_node == node.name`` output edge), so the fresh-rid input
        gathering only has to handle those names.
        """
        # Find loop-back inputs from a sample node. Speculation requires that
        # ALL of the node's required inputs are loop-back or streaming
        # (otherwise we'd need to gather other inputs from tensor_manager,
        #  and the queue state for those isn't necessarily ready in this iter).
        sample_node = next(iter(batch_N.node_objects.values()))
        loop_back_inputs = self._loop_back_input_names(sample_node)
        if not loop_back_inputs:
            return None
        for input_name in sample_node.input_ids:
            if input_name not in loop_back_inputs \
                    and input_name not in sample_node._streaming_inputs:
                # Has a non-loop-back / streaming required input — speculation skipped.
                # (E.g. a node that takes both a loop-back tensor and a fresh
                # external input on each iter.)
                return None

        continuing = []
        for rid in batch_N.node_objects:
            if rid in self._pending_removes:
                continue
            if (rid, partition_N) in self._pending_stops:
                # Pending stop targets THIS batch's partition — its loop
                # is about to be finished, don't speculate further work
                # on it. Stops on other partitions don't gate speculation
                # of this partition's loop.
                continue
            continuing.append(rid)
        if not continuing:
            return None

        # Clone GraphNode + ScheduledBatch metadata for the speculated step.
        new_node_objects = {}
        new_request_to_worker_graph = {}
        per_request_inputs = {}
        per_request_info = {}
        for rid in continuing:
            new_node_objects[rid] = batch_N.node_objects[rid].clone_for_next_iter()
            new_request_to_worker_graph[rid] = batch_N.request_to_worker_graph[rid]
            # Placeholder inputs for continuing rids — filled in by
            # _thread_outputs_to_speculative once GPU(N) returns.
            per_request_inputs[rid] = {name: [] for name in loop_back_inputs}
            per_request_info[rid] = self.worker_graphs_manager.get_fwd_info(
                rid, partition_N
            )
        
        streaming_edges: dict[str, list[GraphEdge]] = {}
        if sample_node._streaming_inputs:
            has_streaming_ready = []
            for rid in continuing:
                streaming_edges[rid] = self._poll_stream_buffers_for_speculation(
                    rid, batch_N.node_name
                )
                if len(streaming_edges[rid]) < len(sample_node._streaming_inputs):
                    for edge in streaming_edges[rid]:
                        self._return_speculative_streaming_edge(rid, edge)
                    
                    new_node_objects.pop(rid, None)
                    per_request_inputs.pop(rid, None)
                    per_request_info.pop(rid, None)
                    new_request_to_worker_graph.pop(rid, None)
                else:
                    has_streaming_ready.append(rid)
                    for edge in streaming_edges[rid]:
                        per_request_inputs[rid][edge.name] = [
                            self.tensor_manager.get_tensor(
                                request_id=rid, uuid=info.uuid,
                            )
                            for info in edge.tensor_info
                        ]
            continuing = has_streaming_ready
            if not continuing:
                return None

        # ── merge in fresh rids whose decode-loop node is ready right now ──
        # Speculation should only consume work compatible with the in-flight
        # AR loop. In partitioned models, unrelated ready work (e.g. SNAC or a
        # fresh prefill) must not cancel LLM decode speculation; it stays queued
        # for the normal scheduler path.
        fresh_batch = self.scheduler.get_next_batch(
            self.worker_graphs_manager,
            target_node_name=batch_N.node_name,
            target_graph_walk=batch_N.graph_walk,
        )
        if fresh_batch is not None:
            for rid, node in fresh_batch.node_objects.items():
                if rid in new_node_objects:
                    # Shouldn't happen — continuing rids are held by the
                    # in-flight step and shouldn't be in ready queues —
                    # but if it does, the in-flight rid wins.
                    wg_id = fresh_batch.request_to_worker_graph[rid]
                    self.worker_graphs_manager.queues[wg_id].push_back_node(rid, node)
                    continue
                tensors = {}
                for input_name in node.ready_inputs:
                    tensors[input_name] = [
                        self.tensor_manager.get_tensor(
                            request_id=rid, uuid=info.uuid,
                        )
                        for info in node.ready_inputs[input_name].tensor_info
                    ]
                per_request_inputs[rid] = tensors
                per_request_info[rid] = self.worker_graphs_manager.get_fwd_info(
                    rid, partition_N
                )
                new_node_objects[rid] = node
                new_request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )

        spec_batch = ScheduledBatch(
            node_name=batch_N.node_name,
            graph_walk=batch_N.graph_walk,
            node_objects=new_node_objects,
            request_to_worker_graph=new_request_to_worker_graph,
        )

        spec_node_batch = NodeBatch(
            node_name=batch_N.node_name,
            graph_walk=batch_N.graph_walk,
            request_ids=list(new_node_objects.keys()),
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
        )

        # Update dynamic_loop_iter_counts (same bookkeeping as the regular
        # build path). Must happen before submit so the engine sees the
        # right count for the upcoming step.
        for rid, req_info in spec_node_batch.per_request_info.items():
            req_info.dynamic_loop_iter_counts.update(
                self.worker_graphs_manager.get_dynamic_loop_iters(
                    rid, partition=partition_N,
                )
            )
            spec_batch.node_objects[rid].clear_outputs()

        return Speculation(
            scheduled_batch=spec_batch,
            node_batch=spec_node_batch,
            loop_back_inputs=loop_back_inputs,
            continuing_rids=set(continuing),
            streaming_edges=streaming_edges
        )

    def _thread_outputs_to_speculative(
        self, speculation: Speculation, output_N: NodeOutput
    ):
        """Replace placeholder inputs in ``spec_node_batch`` with N's actual
        output tensors, for the subset of rids that came from batch_N
        (``continuing_rids``). Fresh rids merged into the speculative batch
        already had their inputs gathered from tensor_manager; we leave
        those alone.

        Sets ``(threaded_continuing, dropped)``:
        - ``threaded_continuing``: continuing rids whose loop-back outputs
          were successfully threaded.
        - ``dropped``: continuing rids whose required loop-back output was
          missing — these get removed from the spec batch (rare; would be
          wasted GPU work).
        """
        threaded_continuing: set[str] = set()
        dropped: set[str] = set()
        for rid in list(speculation.node_batch.request_ids):
            if rid not in speculation.continuing_rids:
                continue  # fresh rid — inputs already gathered.
            rid_outputs = output_N.per_request_output_tensors.get(rid, {})
            ok = True
            for input_name in speculation.loop_back_inputs:
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

                # also give back its streaming inputs, if any
                for edge in speculation.streaming_edges.get(rid, []):
                    self._return_speculative_streaming_edge(rid, edge)
                speculation.streaming_edges.pop(rid, None)

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
        speculation.continuing_rids = threaded_continuing
        speculation.dropped = dropped

    # ------------------------------------------------------------------
    # Post-processing — split into fast (intra-worker routing, no value
    # reads) and slow (D→H of new tokens, ZMQ to conductor, check_stop).
    # Slow runs after submit GPU(N+1), so its .cpu() sync on default stream
    # waits for GPU(N+1) to drain. That's a streaming-token-latency cost
    # we accept for the throughput win; see ASYNC_REDESIGN.md C-phase note
    # for the side-stream D→H follow-up that recovers it.
    # ------------------------------------------------------------------

    def _fast_postprocess(
        self,
        batch: ScheduledBatch,
        node_batch: NodeBatch,
        batch_partition: str | None,
        output: NodeOutput,
        speculation_consumed_loop_back: dict[str, set[str]] | None = None,
        spec_node_name: str | None = None
    ) -> dict[str, NodeOutputRouting]:
        """Pure-Python routing / queue updates / register_for_send. No tensor
        value reads — safe to run while GPU(N+1) is in flight. Returns the
        per-rid routing decisions for slow_postprocess to consume.

        ``speculation_consumed_loop_back``: ``{rid: {edge_name, ...}}`` —
        edges that the speculation already threaded into N+1's input. We
        keep these in ``filtered_outputs`` so ``complete_loops`` still sees
        the loop-back signal (loop continues), but we *exclude* them from
        ``process_node_outputs`` so the queue doesn't get stale loop-back
        entries that would never be consumed (speculation chain handles
        them outside the queue). We then dereference the UUIDs that
        ``store_outputs_and_finish_loops`` allocated for those edges.
        """
        from mminf.utils.profiler import range_pop, range_push

        speculation_consumed_loop_back = speculation_consumed_loop_back or {}

        # Some engines can skip requests after prepare_inputs() decides the
        # current inputs are not executable yet (for example, SNAC needs enough
        # streamed tokens to form a frame). They remove those rids from
        # NodeBatch, but ScheduledBatch still owns the popped graph nodes. Push
        # skipped nodes back and shrink this postprocess pass to the rids that
        # actually ran.
        active_rids = {
            rid for rid in node_batch.request_ids
            if rid in node_batch.per_request_info and rid in batch.node_objects
        }
        skipped_rids = set(batch.node_objects) - active_rids
        for rid in skipped_rids:
            node = batch.node_objects[rid]
            wg_id = batch.request_to_worker_graph.get(rid) \
                if batch.request_to_worker_graph else None
            if wg_id is not None:
                self.worker_graphs_manager.queues[wg_id].push_back_node(rid, node)
        if skipped_rids:
            logger.debug(
                "Worker %s: skipped %d/%d rids in %s.%s after engine filtering",
                self.worker_id, len(skipped_rids),
                len(active_rids) + len(skipped_rids),
                batch.node_name, batch.graph_walk,
            )
            batch.node_objects = {
                rid: node for rid, node in batch.node_objects.items()
                if rid in active_rids
            }
            if batch.request_to_worker_graph is not None:
                batch.request_to_worker_graph = {
                    rid: wg_id for rid, wg_id in batch.request_to_worker_graph.items()
                    if rid in active_rids
                }

        # Update LRU + worker_graphs_manager fwd info + apply stop_loops
        if self.enable_nvtx:
            range_push("worker.update_request_info", synchronize=False)
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
            range_pop(synchronize=False)

        # Apply pending stops/removes deferred from prior iter.
        # Stops apply to loops on rids in this batch (1 wasted step per stop).
        # Removes apply only to rids no longer referenced by the in-flight
        # GPU step (handled by caller — we just consume our snapshot here).
        # Returned mapping marks loop-back input names that must NOT be
        # re-ingested into the freshly-reset queue (see method docstring).
        stopped_loop_backs = self._apply_pending_stops_to_batch(batch, batch_partition)
        if stopped_loop_backs:
            # Merge stopped-loop loop-back exclusions into the speculation-
            # consumed map; ``process_node_outputs`` filters both alike when
            # building ``kept_for_routing`` below.
            for rid, names in stopped_loop_backs.items():
                speculation_consumed_loop_back.setdefault(rid, set()).update(names)

        if self.enable_nvtx:
            range_push("worker.cleanup_inputs", synchronize=False)
        self._cleanup_consumed_inputs(batch)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if self.enable_nvtx:
            range_push("worker.route_outputs", synchronize=False)
            range_push("worker.filter_outputs", synchronize=False)
        filtered_outputs_per_request: dict[str, list[GraphEdge]] = {}
        for request_id, node in batch.node_objects.items():
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            )
            filtered_outputs = [
                e for e in node.outputs if e.name in request_output_tensors
            ]
            filtered_outputs_per_request[request_id] = filtered_outputs
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.route_outputs.store", synchronize=False)

        # Apply spec consumption BEFORE complete_loops. ``update_for_spec``
        # is what marks the Loop's pre-emptively-allocated next-iter body as
        # consumed (sets ``_curr_iter_section = None``). When complete_loops
        # runs after this, it sees the post-spec Loop state and can correctly
        # detect natural termination at ``max_iters`` — emitting ``Loop.outputs``
        # (e.g. BAGEL image_gen Loop's ``latents → vae_decoder``) and
        # short-circuiting the queue. Without this re-ordering, complete_loops
        # saw a stale ``curr_iter_section=GraphNode`` (the spec-cloned body's
        # template), ``_iter_done()`` returned False, the Loop never
        # terminated through complete_loops, downstream nodes never got the
        # final outputs, and the loop ran forever past max_iters.
        if spec_node_name is not None:
            for request_id in batch.node_objects:
                if speculation_consumed_loop_back.get(request_id):
                    self.worker_graphs_manager.apply_spec_consumption(
                        request_id, spec_node_name=spec_node_name,
                    )

        # Stops applied in this batch consume the loop body the same way a
        # spec replay does: the next-iter body that ``process_node_outputs``
        # queued in the prior iter (``_curr_iter_section = GraphNode(...)``)
        # is no longer wanted, because the loop is terminating. If we don't
        # clear it now, the upcoming ``complete_loops`` call recurses into
        # ``GraphNode.complete_loops``, which returns ``new_waiting=self``,
        # restoring ``_curr_iter_section`` to the GraphNode. Then
        # ``_iter_done()`` is False (curr_iter_section is non-None even
        # though _finished=True and _waiting_for_execution is empty), the
        # Loop never reports done, the worker graph never emits
        # WORKER_GRAPHS_DONE with completed_worker_graph_ids, the partition
        # never reaches partition_done=True at the conductor, and the
        # request hangs forever.
        #
        # The no-spec scenarios that hit this on Q3-Omni:
        #   1. Fairness yield at the spec gate above — Thinker shares
        #      worker_1 with Talker; after each Thinker iter, Talker has
        #      ready work, so ``has_ready_excluding(Thinker, thinker_decode)``
        #      returns True, ``must_yield_for_fairness`` is True, and the
        #      iter applying the stop never even calls
        #      ``_try_speculate_next``. spec_node_name stays None and
        #      apply_spec_consumption is skipped from the spec branch.
        #   2. Non-AR engines (e.g. ``code_predictor`` with custom
        #      enable_async_scheduling) where ``_can_speculate`` returns
        #      False outright.
        # In both, ``_apply_pending_stops_to_batch`` correctly registers the
        # stop, but the next-iter body is left dangling on the Loop without
        # this call.
        for request_id in stopped_loop_backs:
            if request_id in batch.node_objects:
                self.worker_graphs_manager.apply_spec_consumption(
                    request_id, spec_node_name=batch.node_name,
                )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.store_outputs_and_finish_loops", synchronize=False)
        node_outputs = self._store_outputs_and_finish_loops(
            batch, output=output,
            filtered_outputs_per_request=filtered_outputs_per_request
        )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.process_node_outputs", synchronize=False)
        routing_per_request: dict[str, NodeOutputRouting] = {}
        for request_id, node in batch.node_objects.items():
            kept = node_outputs[request_id].kept
            consumed_names = speculation_consumed_loop_back.get(request_id, set())
            if request_id in stopped_loop_backs:
                # Wasted speculative iter: this body ran AFTER EOS was
                # detected (stop sat in _pending_stops while the
                # speculatively-built batch was already on the GPU), so
                # every output it produced is post-termination garbage.
                # Drop them all — including streaming edges to downstream
                # partitions (e.g. codec_tokens → Code2Wav, which would
                # otherwise tack a final junk frame onto the audio) — and
                # dereference the tensors that ``store_outputs_and_finish_loops``
                # already allocated UUIDs for. We still call
                # ``process_node_outputs`` with empty kept so the queue's
                # is_done check fires and ``WORKER_GRAPHS_DONE`` gets
                # emitted.
                for edge in kept:
                    for info in edge.tensor_info:
                        self.tensor_manager.dereference(request_id, info.uuid)
                kept_for_routing = []
            elif consumed_names:
                # Dereference the loop-back UUIDs that store_outputs_and_finish_loops
                # allocated; speculation already holds the tensor via Python ref.
                for edge in kept:
                    if edge.next_node == node.name and edge.name in consumed_names:
                        for info in edge.tensor_info:
                            self.tensor_manager.dereference(request_id, info.uuid)
                kept_for_routing = [
                    e for e in kept
                    if not (e.next_node == node.name and e.name in consumed_names)
                ]
            else:
                kept_for_routing = kept
            # Pass spec_node_name=None now; we already applied spec consumption
            # above so process_node_outputs shouldn't re-apply it.
            routing = self.worker_graphs_manager.process_node_outputs(
                request_id, kept_for_routing, graph_walk=batch.graph_walk,
                spec_node_name=None,
            )
            routing_per_request[request_id] = routing
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)

        # Send "loop done" messages to peer workers (small ZMQ msgs, no
        # tensor data).
        for request_id in batch.node_objects:
            stop_loop_workers: dict[str, set[str]] = {}
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

        if self.enable_nvtx:
            range_push("worker.store_outputs", synchronize=False)
        self._register_outputs(batch, routing_per_request)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        return routing_per_request

    def _d2h_new_tokens(
        self,
        tensors: list[torch.Tensor],
        completion_event: "torch.cuda.Event | None",
    ) -> list[int]:
        """Batched D→H copy of new-token tensors, gated on GPU(N)'s
        completion event so it does not block on GPU(N+1) (which is queued
        on default stream behind GPU(N)).

        Falls back to the simple ``torch.cat([...]).cpu()`` when CUDA is
        unavailable, the tensors are already on CPU, or no completion event
        was recorded (non-CUDA execution).

        Safety: assumes ``tensors`` are fresh allocations (not views into
        CUDA-graph static buffers that GPU(N+1) will overwrite). Sampler
        outputs from FlashInfer's ``top_p_sampling_from_probs`` qualify;
        if a future change makes the new-token tensor a static-buffer
        view, this needs an extra clone-on-default-stream-before-event-
        record step on the GPU thread.
        """
        if not tensors:
            return []
        first = tensors[0]
        on_cuda = first.is_cuda and torch.cuda.is_available()
        if not on_cuda or completion_event is None:
            return torch.cat([t.flatten() for t in tensors]).cpu().tolist()

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(completion_event)
        with torch.cuda.stream(side):
            flat_gpu = torch.cat([t.flatten() for t in tensors])
            flat_cpu = self._get_pinned_d2h_buffer(
                "new_tokens", flat_gpu.shape, flat_gpu.dtype,
            )
            flat_cpu.copy_(flat_gpu, non_blocking=True)
        side.synchronize()
        return flat_cpu.tolist()

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
        output: NodeOutput,
    ) -> NodeOutput:
        """Side-stream D→H of every CUDA tensor in
        ``output.per_request_output_tensors`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns a fresh ``NodeOutput`` with CPU tensors for the per-rid
        outputs, sharing the original's allocation_failed / event fields.
        Skipped (returns ``output`` unchanged) when there's no completion
        event (CPU execution) or when CUDA is unavailable.

        AR engines emit small per-rid output dicts (sampled token + maybe
        a code) so the cost is negligible. If a future engine emits large
        tensors here (e.g. activations), revisit.
        """
        if not torch.cuda.is_available() or output.completion_event is None:
            return output
        if not output.per_request_output_tensors:
            return output

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(output.completion_event)

        cpu_per_rid: dict = {}
        buffer_indices: dict[tuple[str, torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        with torch.cuda.stream(side):
            for rid, name_to_list in output.per_request_output_tensors.items():
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

        return NodeOutput(
            per_request_output_tensors=cpu_per_rid,
            allocation_failed=output.allocation_failed,
            alloc_pages_short=output.alloc_pages_short,
            alloc_failed_request_id=output.alloc_failed_request_id,
            completion_event=output.completion_event,
        )

    def _compute_slow_postprocess(
        self,
        batch: ScheduledBatch,
        node_batch: NodeBatch,
        output: NodeOutput,
        routing_per_request: dict[str, NodeOutputRouting],
    ) -> SlowPostprocessResult:
        """Background half of slow postprocessing.

        Runs the event-gated D→H work and EOS / stop detection on a
        dedicated postproc thread so the main loop can keep polling queues
        while GPU(N+1) executes. It intentionally does *not* mutate
        worker_graphs_manager or send messages; those finalization steps stay
        on the main thread to avoid broad cross-thread state races.
        """
        from mminf.utils.profiler import range_pop, range_push

        if self.enable_nvtx:
            range_push("worker.postproc_compute", synchronize=False)

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
                    tensor = self.tensor_manager.get_tensor(
                        request_id=rid, uuid=tinfo.uuid,
                    )
                    collected.append((rid, signal.name, tensor))

        if collected:
            lengths = [t.numel() for t in (tr for _, _, tr in collected)]
            flat = self._d2h_new_tokens(
                [t for _, _, t in collected],
                completion_event=output.completion_event,
            )
            off = 0
            for (rid, sig_name, _), n in zip(collected, lengths, strict=True):
                rid_map = prematerialized_per_rid.setdefault(rid, {})
                rid_map.setdefault(sig_name, []).extend(flat[off:off + n])
                off += n

        # Deferred-EOS check: run submodule.check_stop on the actual output
        # tensors. Stops returned here apply to the *next* iter's fast
        # postprocess (the in-flight GPU step has already been submitted
        # under the assumption the rid continues — that's the 1-wasted-
        # step cost per stop). Sampler seen-mask staleness for rep-penalty
        # is accepted: step N+1's sampling sees the mask state from before
        # N's token was added. See ASYNC_REDESIGN.md.
        engine = self.engine_manager.get_engine(batch.node_name)
        # Pre-materialize tensors to CPU on a side stream gated on
        # event(N) so check_stop's .item() doesn't full-stream-sync (which
        # would block on the in-flight GPU(N+1) submission and erase the
        # overlap). check_stop_for_batch expects NodeBatch (it iterates
        # request_ids and reads per_request_info), not ScheduledBatch.
        cpu_output = self._prematerialize_for_check_stop(output)
        new_stops = engine.check_stop_for_batch(node_batch, cpu_output)

        if self.enable_nvtx:
            range_pop(synchronize=False)

        return SlowPostprocessResult(
            prematerialized_new_tokens=prematerialized_per_rid,
            new_stops=new_stops,
        )

    def _finalize_slow_postprocess(
        self,
        batch: ScheduledBatch,
        node_batch: NodeBatch,
        batch_partition: str | None,
        routing_per_request: dict[str, NodeOutputRouting],
        result: SlowPostprocessResult,
        advanced_loops: dict[str, set[str]] | None = None,
    ) -> dict[str, set[str]]:
        """Main-thread half of slow postprocessing.

        Called once the background postproc task finishes. Emits the delayed
        worker/conductor/api_server messages using the prematerialized token
        payloads, then returns the deferred stop signals to apply to the next
        speculative iter.

        ``advanced_loops`` (rid -> {loop_name, ...}) names the dynamic loops
        whose iter counter advanced during this batch's fast postprocess.
        For any loop that BOTH advanced this iter AND was just stopped (in
        ``result.new_stops``), the next iter ingested a loop-back into a
        body it should never run. We clear that loop's ``_curr_iter_section``
        so when ``register_finished`` is later called via ``_pending_stops``,
        ``_iter_done()`` returns True and the worker graph is marked done
        — otherwise no ``WORKER_GRAPHS_DONE`` is ever emitted and the
        partition stalls (e.g. Qwen Talker waits forever for Thinker's
        hidden states).
        """
        from mminf.utils.profiler import range_pop, range_push

        if self.enable_nvtx:
            range_push("worker.send_outputs", synchronize=False)

        for request_id in batch.node_objects.keys():
            self._send_outputs(
                request_id, routing_per_request[request_id],
                graph_walk=batch.graph_walk,
                partition_name=batch_partition,
                prematerialized_new_tokens=result.prematerialized_new_tokens.get(
                    request_id
                ),
            )

        for _rid, req_info in node_batch.per_request_info.items():
            req_info.dynamic_loop_stop_signals.clear()

        if advanced_loops:
            for rid, stops in result.new_stops.items():
                advanced = advanced_loops.get(rid)
                if not advanced:
                    continue
                to_clear = stops & advanced
                if to_clear:
                    self.worker_graphs_manager.clear_dyn_loop_curr_iter_section(
                        rid, partition=batch_partition, loop_names=to_clear,
                    )

        if self.enable_nvtx:
            range_pop(synchronize=False)

        return result.new_stops

    def _apply_pending_stops_to_batch(
        self,
        batch: ScheduledBatch,
        batch_partition: str | None,
    ) -> dict[str, set[str]]:
        """Apply any deferred stops from the previous iter that target rids
        in this batch. Called from fast_postprocess so the stop_loops side
        effects (queue updates, complete_loops) are visible to the next
        iter's speculation.

        Stops for rids NOT in this batch are handled by
        ``_drain_orphan_pending_stops`` at the top of the worker loop —
        without that, a pending stop whose loop has no further body to
        run (e.g. Talker after EOS, no spec) sits forever and the
        partition never reports done.

        Returns a ``{rid: {loop_back_input_name, ...}}`` map of loop-back
        input names that should be EXCLUDED from this iter's output routing
        — when a loop is stopped, its loop-back outputs MUST NOT be re-
        ingested into the just-reset queue (otherwise the queue picks them
        up, schedules another loop iter, and the model keeps generating
        until it produces another `<|im_end|>` — which fires another stop
        signal, queue resets again, infinite cycle producing duplicated
        tokens). The caller merges this into ``speculation_consumed_loop_back``
        so ``_fast_postprocess`` filters these edges out of the routing
        kept-list, matching what speculation does for spec-consumed
        loop-backs.
        """
        stopped_loop_backs: dict[str, set[str]] = {}
        for key in list(self._pending_stops.keys()):
            rid, stop_partition = key
            if stop_partition != batch_partition:
                # Stop targets a different partition (e.g. Thinker stop
                # popped by a Talker batch with the same rid); leave it
                # pending until a batch on the matching partition runs.
                continue
            if rid not in batch.node_objects:
                continue
            loop_names = self._pending_stops.pop(key)
            if rid not in self.worker_graphs_manager.per_request_info:
                continue
            fwd_info = self.worker_graphs_manager.get_fwd_info(rid, batch_partition)
            loop_back_signals = self.worker_graphs_manager.stop_loops(
                rid, partition=batch_partition, loop_names=loop_names,
                req_info=fwd_info, last_node_run=batch.node_name,
            )
            # Drop the stopped loop's loop-back inputs from this iter's output
            # routing. ``batch.node_objects[rid]`` is the running node; its
            # self-loop outputs (next_node == node.name) that match a stopped
            # loop's loop-back signal were just sampled by this iter, but the
            # loop they'd feed has been finished — keeping them in the routing
            # kept-list re-ingests them into the post-reset queue and starts
            # an infinite cycle. The intersection with ``loop_back_signals``
            # is what scopes the drop to the *stopped* loops only: a node
            # that participates in two distinct loops keeps the surviving
            # loop's loop-back tensor.
            node = batch.node_objects[rid]
            stopped_loop_backs[rid] = {
                edge.name for edge in node.outputs
                if edge.next_node == node.name
                and (edge.name, edge.next_node) in loop_back_signals
            }
        return stopped_loop_backs

    def _drain_orphan_pending_stops(
        self,
        pending: tuple[ScheduledBatch, NodeBatch, str | None, Future] | None,
        pending_postproc: list[PendingPostproc],
    ) -> None:
        """Apply stops for ``(rid, partition)`` keys that have no in-flight
        batch, no scheduled batch, and no body sitting in ``ready`` waiting
        to be popped — those stops would otherwise sit forever.

        Stops covered by an in-flight or scheduled batch are left alone;
        the standard ``_apply_pending_stops_to_batch`` call inside that
        batch's fast_postprocess will consume them.
        """
        if not self._pending_stops:
            return

        in_flight_keys: set[tuple[str, str | None]] = set()
        if pending is not None:
            p_batch, _, p_partition, _ = pending
            for rid in p_batch.node_objects:
                in_flight_keys.add((rid, p_partition))
        for pp in pending_postproc:
            for rid in pp.batch.node_objects:
                in_flight_keys.add((rid, pp.partition))

        for key in list(self._pending_stops.keys()):
            if key in in_flight_keys:
                continue
            rid, partition = key
            per_request_info = self.worker_graphs_manager.per_request_info.get(rid)
            if per_request_info is None:
                self._pending_stops.pop(key)
                continue
            part_info = per_request_info.per_partition_info.get(partition)
            if part_info is None:
                self._pending_stops.pop(key)
                continue
            # If a body is in `ready` for this rid+partition, the
            # scheduler will pop it next and a fast_postprocess will
            # fire — let the standard path handle the stop there.
            has_ready_body = any(
                rid in self.worker_graphs_manager.queues[wg].per_request_queues
                and len(self.worker_graphs_manager.queues[wg].per_request_queues[rid].ready) > 0
                for wg in part_info.graph_walk_worker_graph_ids
            )
            if has_ready_body:
                continue
            loop_names = self._pending_stops.pop(key)
            fwd_info = self.worker_graphs_manager.get_fwd_info(rid, partition)
            self.worker_graphs_manager.stop_loops(
                rid, partition=partition, loop_names=loop_names,
                req_info=fwd_info, last_node_run=None,
            )
            self._finalize_orphan_stop(rid, partition)

    def _finalize_orphan_stop(self, rid: str, partition: str | None) -> None:
        """Force-complete loops for an orphan stop and emit
        WORKER_GRAPHS_DONE for any worker graph that became done.
        Called from ``_apply_pending_stops_to_batch`` for rids whose
        stop isn't covered by the current batch. ``stop_loops`` has
        already been called at this point (so ``_finished=True``); this
        method drives complete_loops for the bodies sitting in `ready`
        — calling it with each ready body's name removes that name from
        the loop's ``_waiting_for_execution``, after which ``_is_done``
        returns True and the loop emits its accumulated outputs.
        """
        per_request_info = self.worker_graphs_manager.per_request_info.get(rid)
        if per_request_info is None:
            return
        part_info = per_request_info.per_partition_info.get(partition)
        if part_info is None:
            return
        completed_wg_ids: list[str] = []
        for wg_id in part_info.graph_walk_worker_graph_ids:
            queue = self.worker_graphs_manager.queues[wg_id]
            per_req_q = queue.per_request_queues.get(rid)
            if per_req_q is None:
                continue
            per_req_q.ready.clear()

            # TODO this is hacky, just trying to get something working
            for name in per_req_q.waiting.get_node_names():
                out = per_req_q.waiting.complete_loops(name)
                per_req_q.waiting = out.new_waiting
            if queue.is_done(rid):
                completed_wg_ids.append(wg_id)
                queue.reset(rid)

        if not completed_wg_ids:
            return
        graph_walk = self.worker_graphs_manager.get_graph_walk(rid, partition)
        routing = NodeOutputRouting(
            routed_to_this_worker_graph=[],
            persist=[],
            to_workers={},
            completed_worker_graph_ids=completed_wg_ids,
        )
        self._send_outputs(
            rid, routing,
            graph_walk=graph_walk,
            partition_name=partition,
        )

    def _apply_pending_removes_safe_to_drop(
        self, in_flight_rids: set[str]
    ) -> None:
        """Apply ``REMOVE_REQUEST`` for any rid that is not currently held by
        an in-flight GPU step. Removes for in-flight rids stay deferred and
        are reattempted next iter."""
        to_apply = [r for r in self._pending_removes if r not in in_flight_rids]
        for rid in to_apply:
            self._pending_removes.discard(rid)
            self._remove_request(RemoveRequest(request_id=rid))

    def _drain_completed_postprocess(
        self,
        pending_postproc: list[PendingPostproc],
    ) -> None:
        """Finalize any completed background slow-postprocess tasks in FIFO order."""
        while pending_postproc and pending_postproc[0].future.done():
            pp = pending_postproc.pop(0)
            self._postproc_inflight_rids.difference_update(pp.batch.node_objects.keys())
            result: SlowPostprocessResult = pp.future.result()
            new_stops = self._finalize_slow_postprocess(
                pp.batch, pp.node_batch, pp.partition, pp.routing, result,
                advanced_loops=pp.advanced_loops,
            )
            if new_stops:
                for rid, stops in new_stops.items():
                    self._pending_stops.setdefault(
                        (rid, pp.partition), set()
                    ).update(stops)

    def run(self) -> None:
        switch_interval = os.environ.get("MMINF_PY_SWITCH_INTERVAL_SEC", "")
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
                    "Worker %s: ignoring invalid MMINF_PY_SWITCH_INTERVAL_SEC=%r",
                    self.worker_id,
                    switch_interval,
                )

        # CUDA graph capture before entering the main loop
        self.engine_manager.warmup_all()

        # The async worker path needs decode submission to return quickly so
        # the main loop can overlap queue/tensor polling and post-processing
        # with GPU execution. Run the engine unconditionally on a dedicated
        # 1-worker GPU thread.
        gpu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mminf-gpu-{self.worker_id}"
        )
        logger.info(
            "Worker %s: engine runs on dedicated GPU thread",
            self.worker_id,
        )
        # Phase 3 (single-buffer): dedicated thread that pre-plans FlashInfer
        # attention for the speculatively-built next batch. Runs concurrent
        # with main thread's await_gpu (which releases the GIL), so plan()'s
        # Python work isn't contended by main thread's fast/slow post — that
        # contention is what made the spec-path plan_attention 2.3× slower
        # than the fall-through path.
        #
        # With double-buffered wrappers (CudaGraphRunner.NUM_SLOTS=2) and
        # advance_event signaling, plan(N+1) runs concurrent with replay(N)
        # on the disjoint slot — the actual GPU overlap that single-buffer
        # Phase 3 couldn't deliver. plan_executor waits on
        # prev_advance_event (signaled right after advance_seq_lens(N) on
        # the GPU thread, ~tens of µs into replay) instead of prev_future
        # (which only resolves after replay completes), so plan() starts
        # early. See ASYNC_REDESIGN.md for the design.
        #
        # Default ON. Set MMINF_PRE_PLAN_SPEC=0 to fall back to the
        # double-buffer-without-pre-plan baseline (slightly slower than
        # single-buffer Phase 1'' due to alternation overhead with no
        # offsetting plan-overlap win).
        pre_plan_spec = os.environ.get("MMINF_PRE_PLAN_SPEC", "1") == "1"
        plan_executor = None
        if pre_plan_spec:
            plan_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mminf-plan-{self.worker_id}"
            )
            logger.info(
                "Worker %s: plan_executor enabled — speculative plan() "
                "pre-runs on a dedicated thread",
                self.worker_id,
            )
        # Background postprocessing is useful for overlap, but it touches CUDA
        # tensors and tensor-manager state from a second thread. Keep it opt-in
        # while the dedicated GPU thread path is being stabilized.
        use_postproc_thread = os.environ.get("MMINF_USE_POSTPROC_THREAD", "") == "1"
        postproc_executor = None
        if use_postproc_thread:
            postproc_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mminf-postproc-{self.worker_id}"
            )
            logger.info(
                "Worker %s: slow postprocess runs on dedicated postproc thread",
                self.worker_id,
            )

        # Cross-iter async-scheduling state lives on self (initialized in
        # __init__) so _remove_request can see it from message processing.
        # In-flight: (batch, node_batch, batch_partition, future) | None.
        pending: tuple[ScheduledBatch, NodeBatch, str | None, Future] | None = None
        pending_postproc: list[PendingPostproc] = []
        # consecutive-spec cap: the original Phase 1' design caps consecutive
        # speculative steps at 1 to give other (node, walk) pairs a turn at
        # the scheduler — important on multi-walk workers (Qwen-Omni's
        # Thinker+Talker on the same worker). On single-walk workers
        # (Orpheus LLM, Orpheus SNAC) the cap forces every other iter
        # through MicroScheduler+build for no fairness gain — see the trace
        # analysis: spec/fall-through alternation is the source of the
        # plan_attention variance flagged earlier.
        #
        # MMINF_SPEC_PEEK_FOR_FAIRNESS=1 (default) replaces the iter-counter
        # heuristic with a peek-based check: only break the spec chain when
        # MicroScheduler.has_ready_excluding finds another (node, walk)
        # ready RIGHT NOW. Single-walk workers always speculate; multi-walk
        # workers yield only when there's actual contention. The
        # MMINF_MAX_CONSECUTIVE_SPEC_STEPS cap is still respected as a
        # safety ceiling for pathological cases (default 1024 ≈ unbounded
        # for any reasonable workload).
        max_consecutive_spec = int(os.environ.get("MMINF_MAX_CONSECUTIVE_SPEC_STEPS", "1024"))
        spec_peek_for_fairness = (
            os.environ.get("MMINF_SPEC_PEEK_FOR_FAIRNESS", "1") == "1"
        )
        consecutive_spec_steps = 0
        yield_away_from_target: tuple[str, str] | None = None

        def _set_pending(p):
            nonlocal pending
            pending = p
            self._in_flight_rids = set(p[0].node_objects.keys()) if p else set()

        # Per-phase wall-clock instrumentation, gated by MMINF_PHASE_TIMING.
        # When enabled, every Nth speculative iter logs a histogram so we can
        # see whether await_gpu time = "GPU still running" (overlap working)
        # vs "GPU done, idle" (overlap not paying off). Set the env var to a
        # positive integer = the dump period in iters (e.g. 200).
        phase_period = int(os.environ.get("MMINF_PHASE_TIMING", "0") or "0")
        phase_buf: dict[str, list[float]] = defaultdict(list)
        phase_iter = [0]

        def _phase_record(name: str, dt: float) -> None:
            if phase_period > 0:
                phase_buf[name].append(dt)

        def _phase_flush() -> None:
            if phase_period <= 0 or phase_iter[0] % phase_period != 0:
                return
            samples = sorted((k, v) for k, v in phase_buf.items() if v)
            parts = []
            for name, vs in samples:
                vs.sort()
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

        while True:
            from mminf.utils.profiler import range_pop, range_push
            try:
                _iter_start = _time.perf_counter() if phase_period else 0.0
                if postproc_executor is not None:
                    self._drain_completed_postprocess(pending_postproc)
                self._apply_pending_removes_safe_to_drop(
                    self._in_flight_rids | self._postproc_inflight_rids
                )

                # Drain pending stops with no in-flight batch, scheduled
                # batch, or ready body to consume them. Required for non-
                # spec loops where iter N's split_off_ready put body N+1
                # in `ready`, but EOS was detected in slow_postprocess(N)
                # and the scheduler later popped body N+1 (running it as
                # the wasted step) — once body N+1 enters the pending
                # batch, fast_postprocess(N+1) will consume the stop, but
                # for stops detected after the queue is fully drained
                # (loop ended without a wasted step), no batch fires and
                # the stop sits forever. This drain catches that case.
                self._drain_orphan_pending_stops(pending, pending_postproc)

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
                spec_plan_future: Future | None = None
                # Tracks the NodeBatch the in-flight pre-plan targeted, so
                # cleanup paths (alloc-fail, dropped rids, no-spec-submit)
                # can call ``_reset_skip_plan_flags`` on the right slot
                # rather than wiping every captured graph's slots.
                spec_plan_target: NodeBatch | None = None
                if (
                    pending is not None
                    and self._can_speculate(pending[0])
                ):
                    # Fairness check (peek-based, replaces the old iter-
                    # counter cap): only break the spec chain when there's
                    # another (node, walk) actually ready to schedule on
                    # this worker. On single-walk workers (Orpheus LLM,
                    # Orpheus SNAC) this returns False and we always speculate.
                    must_yield_for_fairness = (
                        spec_peek_for_fairness
                        and consecutive_spec_steps >= 1
                        and self.scheduler.has_ready_excluding(
                            self.worker_graphs_manager,
                            (pending[0].node_name, pending[0].graph_walk),
                        )
                    )
                    if (
                        consecutive_spec_steps < max_consecutive_spec
                        and not must_yield_for_fairness
                    ):
                        if self.enable_nvtx:
                            range_push("worker.speculate", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        speculation = self._try_speculate_next(
                            pending[0], pending[2]
                        )
                        if phase_period:
                            _phase_record("speculate", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)
                        # Phase 3 double-buffer: reserve the slot for
                        # batch_(N+1) NOW so both pre-plan and replay (queued
                        # below) target the SAME slot — and the OPPOSITE
                        # slot from batch_N's in-flight replay. The
                        # reservation lives on spec_node_batch.metadata
                        # ['cuda_graph_slot']; the engine forwards it to
                        # the runner.
                        if speculation is not None:
                            spec_node_batch_for_plan = speculation.node_batch
                            engine = self.engine_manager.get_engine(
                                speculation.node_batch.node_name
                            )
                            if hasattr(engine, "reserve_replay_slot"):
                                engine.reserve_replay_slot(speculation.node_batch)
                        # Kick off pre-planning on the plan_executor NOW —
                        # its Python work runs while the main thread is in
                        # await_gpu (releases GIL). plan_executor waits on
                        # prev's advance_event (signaled ~tens of µs into
                        # replay(N), right after advance_seq_lens(N)) so
                        # plan(N+1) starts WAY BEFORE replay(N) finishes —
                        # the actual GPU overlap that single-buffer Phase 3
                        # couldn't deliver. plan(N+1) writes the inactive
                        # slot's wrapper buffers; replay(N) keeps running
                        # uncontested on the active slot.
                        if (
                            speculation is not None
                            and plan_executor is not None
                        ):
                            engine = self.engine_manager.get_engine(
                                speculation.node_batch.node_name
                            )
                            if hasattr(engine, "pre_plan_for_batch"):
                                prev_advance_event_for_plan: threading.Event | None = None
                                if pending is not None:
                                    prev_advance_event_for_plan = (
                                        pending[1].metadata.get("advance_event")
                                    )
                                spec_plan_future = plan_executor.submit(
                                    self._pre_plan_for_speculative_batch,
                                    engine,
                                    speculation.node_batch,
                                    prev_advance_event_for_plan,
                                )
                                spec_plan_target = spec_node_batch_for_plan
                    else:
                        yield_away_from_target = (
                            pending[0].node_name,
                            pending[0].graph_walk,
                        )

                # 3. If pending: await GPU(N), submit speculated GPU(N+1)
                # asap, then post-process N (fast then slow) overlapping
                # with GPU(N+1).
                spec_pending = None
                if pending is not None:
                    p_batch, p_node_batch, p_partition, p_future = pending
                    _set_pending(None)
                    if self.enable_nvtx:
                        range_push("worker.await_gpu", synchronize=False)
                    _t0 = _time.perf_counter() if phase_period else 0.0
                    output: NodeOutput = p_future.result()
                    if phase_period:
                        _phase_record("await_gpu", _time.perf_counter() - _t0)
                    if self.enable_nvtx:
                        range_pop(synchronize=False)

                    if output.allocation_failed:
                        # Drain any speculation: if we already speculated
                        # N+1, discard it. The speculated step would have
                        # depended on N's outputs which are unusable now.
                        # Per design: discard speculation, retry batch_N.
                        self._handle_allocation_failure(p_batch, p_node_batch)
                        if speculation is not None:
                            for rid, streaming_edges in speculation.streaming_edges.items():
                                for edge in streaming_edges:
                                    self._return_speculative_streaming_edge(rid, edge)
                        speculation = None
                        # If a pre-plan was dispatched, wait for it to finish
                        # so its wrapper.plan() side effects don't leak into
                        # the next iter's batch (different rids/seq_lens).
                        # Then clear the skip flag explicitly so the next real
                        # plan_attention call recomputes from scratch.
                        if spec_plan_future is not None:
                            spec_plan_future.result()
                            self._reset_skip_plan_flags(spec_plan_target)
                            spec_plan_future = None
                            spec_plan_target = None
                        # No in-flight GPU work now; safe to apply all
                        # pending removes.
                        self._apply_pending_removes_safe_to_drop(
                            set(self._postproc_inflight_rids)
                        )
                        continue

                    spec_consumed: dict[str, set[str]] = {}
                    spec_node_name = None
                    if speculation is not None:
                        self._thread_outputs_to_speculative(
                            speculation, output
                        )
                        # Drop continuing rids whose thread-through failed
                        # (engine produced no output for them). Fresh rids
                        # in spec_batch are unaffected — they had their
                        # inputs gathered from tensor_manager.
                        for rid in speculation.dropped:
                            speculation.scheduled_batch.node_objects.pop(rid, None)
                            speculation.scheduled_batch.request_to_worker_graph.pop(rid, None)
                        spec_batch = speculation.scheduled_batch
                        spec_node_batch = speculation.node_batch
                        if spec_batch.node_objects:
                            spec_node_name = spec_batch.node_name
                            # Only continuing rids consumed loop-back from
                            # batch_N — fresh rids' loop-back doesn't exist
                            # in batch_N's output dict, so fast_postprocess
                            # has nothing to deref/skip for them.
                            for rid in speculation.continuing_rids:
                                spec_consumed[rid] = set(speculation.loop_back_inputs)
                            if self.enable_nvtx:
                                range_push("worker.submit_spec", synchronize=False)
                            _t0 = _time.perf_counter() if phase_period else 0.0
                            # If pre-plan was dispatched but the spec_batch
                            # composition changed (rids dropped post thread-
                            # through), the pre-planned wrapper buffers no
                            # longer match — fall back to inline planning by
                            # waiting on the future and clearing the flag.
                            plan_future_for_submit = spec_plan_future
                            if (
                                spec_plan_future is not None
                                and speculation.dropped
                            ):
                                spec_plan_future.result()
                                self._reset_skip_plan_flags(spec_plan_target)
                                plan_future_for_submit = None
                            spec_plan_future = None  # ownership transferred
                            spec_plan_target = None
                            # Phase 3: attach a fresh advance_event to this
                            # batch so the NEXT iter's plan_executor can
                            # gate on advance_seq_lens(THIS batch).
                            spec_advance_event = threading.Event()
                            spec_node_batch.metadata["advance_event"] = spec_advance_event

                            # Block the main thread until the GPU executor
                            # thread is about to launch CUDA kernels (set
                            # deep in the engine: before graph.replay() in
                            # CudaGraphRunner, or before forward/forward_batched
                            # in the eager AR path).
                            spec_launch_started_event = threading.Event()
                            spec_node_batch.metadata["launch_started_event"] = spec_launch_started_event
                            spec_future = gpu_executor.submit(
                                self._execute_on_gpu_thread,
                                spec_batch, spec_node_batch,
                                plan_future_for_submit,
                                spec_advance_event,
                            )
                            self.wakeup_event.register_future(spec_future)
                            if self.enable_nvtx:
                                range_push("worker.gpu_submit_queued", synchronize=False)
                                range_pop(synchronize=False)
                            spec_launch_started_event.wait(timeout=0.005)
                            if phase_period:
                                _phase_record("submit_spec", _time.perf_counter() - _t0)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                            spec_pending = (
                                spec_batch, spec_node_batch, p_partition,
                                spec_future,
                            )

                    # Cleanup: if pre-plan was dispatched but no spec was
                    # submitted (spec_batch fully drained, or speculation
                    # became invalid), drain the plan future and clear the
                    # _pre_planned_labels set so the next non-spec path's
                    # plan_attention runs from scratch.
                    if spec_plan_future is not None:
                        spec_plan_future.result()
                        self._reset_skip_plan_flags(spec_plan_target)
                        spec_plan_future = None
                        spec_plan_target = None

                    # Post-process N — runs concurrently with GPU(N+1)
                    # if we submitted one above.
                    _t0 = _time.perf_counter() if phase_period else 0.0
                    routing = self._fast_postprocess(
                        p_batch, p_node_batch, p_partition, output,
                        speculation_consumed_loop_back=spec_consumed,
                        spec_node_name=spec_node_name
                    )
                    advanced_loops: dict[str, set[str]] = {}
                    for rid, req_info in p_node_batch.per_request_info.items():
                        prev_iters = dict(req_info.dynamic_loop_iter_counts)
                        new_iters = self.worker_graphs_manager.get_dynamic_loop_iters(
                            rid, partition=p_partition,
                        )
                        req_info.dynamic_loop_iter_counts.update(new_iters)
                        advanced = {
                            loop_name for loop_name, count in new_iters.items()
                            if loop_name in prev_iters and count > prev_iters[loop_name]
                        }
                        if advanced:
                            advanced_loops[rid] = advanced
                    if phase_period:
                        _phase_record("fast_post", _time.perf_counter() - _t0)
                    _t0 = _time.perf_counter() if phase_period else 0.0
                    if postproc_executor is not None:
                        postproc_future = postproc_executor.submit(
                            self._compute_slow_postprocess,
                            p_batch, p_node_batch, output, routing,
                        )
                        self.wakeup_event.register_future(postproc_future)
                        self._postproc_inflight_rids.update(p_batch.node_objects.keys())
                        pending_postproc.append(
                            PendingPostproc(
                                batch=p_batch,
                                node_batch=p_node_batch,
                                partition=p_partition,
                                routing=routing,
                                future=postproc_future,
                                advanced_loops=advanced_loops,
                            )
                        )
                    else:
                        result = self._compute_slow_postprocess(
                            p_batch, p_node_batch, output, routing,
                        )
                        new_stops = self._finalize_slow_postprocess(
                            p_batch, p_node_batch, p_partition, routing, result,
                            advanced_loops=advanced_loops,
                        )
                        if new_stops:
                            for rid, stops in new_stops.items():
                                self._pending_stops.setdefault(
                                    (rid, p_partition), set()
                                ).update(stops)
                    if phase_period:
                        _phase_record("slow_post", _time.perf_counter() - _t0)

                    # Removes for any rid not in the in-flight spec step
                    # are safe to apply now.
                    in_flight = set(spec_pending[0].node_objects.keys()) if spec_pending else set()
                    in_flight |= self._postproc_inflight_rids
                    self._apply_pending_removes_safe_to_drop(in_flight)

                if spec_pending is not None:
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
                if self.enable_nvtx:
                    range_push("worker.schedule", synchronize=False)
                batch = None
                if yield_away_from_target is not None:
                    batch = self.scheduler.get_next_batch(
                        self.worker_graphs_manager,
                        exclude_target=yield_away_from_target,
                    )
                if batch is None:
                    batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                if batch is None:
                    self.communicator.wait_for_work(10)
                    continue

                if self.enable_nvtx:
                    range_push("worker.build_node_batch", synchronize=False)
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
                    range_pop(synchronize=False)

                # Phase 3 double-buffer: reserve the slot on the main
                # thread before submission so the per-key counter advances
                # in main-thread order. Without this, the GPU thread would
                # advance the counter at run time and races with main-
                # thread reservations from later iters.
                fallthrough_engine = self.engine_manager.get_engine(batch.node_name)
                if hasattr(fallthrough_engine, "reserve_replay_slot"):
                    fallthrough_engine.reserve_replay_slot(node_batch)

                # Phase 3: attach a fresh advance_event so the next iter's
                # plan_executor (if it speculates) can wait on this batch's
                # advance_seq_lens.
                fallthrough_advance_event = threading.Event()
                node_batch.metadata["advance_event"] = fallthrough_advance_event
                future = gpu_executor.submit(
                    self._execute_on_gpu_thread, batch, node_batch,
                    None, fallthrough_advance_event,
                )
                self.wakeup_event.register_future(future)
                _set_pending((batch, node_batch, batch_partition, future))
            except Exception:
                logger.exception("Worker %s error in main loop", self.worker_id)
                sleep(0.01)
