import logging
from time import sleep

import torch

from mminf.api_server.request_types import APIServerMessage, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager, NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import EngineType, NodeBatch, NodeOutput
from mminf.engine.kv_store import MooncakeStoreConfig, StoreWritePolicy, TransferEngineInfo
from mminf.graph.base import GraphEdge
from mminf.graph.request_queues import format_graph_edge_list
from mminf.model.base import Model, WorkerGraph
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
        )

        self.engine_manager = EngineManager.from_config(
            engine_configs=engine_configs, device=device,
            mooncake_cfg=MooncakeStoreConfig(
                hostname=hostname,
                metadata_server=f"http://localhost:{mooncake_port}/metadata",
                protocol=tensor_comm_protocol,
                master_service=master_service
            ),
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
        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            worker_graph_ids=body.worker_graph_ids,
            worker_graph_to_worker=body.worker_graph_to_worker,
            current_fwd_info=body.request_info
        )
        self.engine_manager.add_request(body.request_id)

        # Start RDMA reads for tensors that have tensor_info
        self.tensor_manager.start_read_tensors(
            body.request_id, body.initial_inputs,
            device=self.device
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

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for (uuid, ref_cnt) in body.successful_tensors.items():
            self.tensor_manager.dereference(
                body.request_id, uuid, n=ref_cnt
            )

    def _process_new_inputs(self, body: InputSignals) -> None:
        self.worker_graphs_manager.update_request_info(
            body.request_id, current_fwd_info=body.request_info
        )

        logger.debug(
            "Received new signals %s at worker %s for request %s",
            format_graph_edge_list(body.inputs), self.worker_id, body.request_id
        )

        # Start RDMA reads for tensors with tensor_info
        self.tensor_manager.start_read_tensors(
            body.request_id, body.inputs,
            device=self.device
        )

        # Signal-only edges can be processed immediately
        signal_only = [edge for edge in body.inputs if len(edge.tensor_info) == 0]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only
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

    def _check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready graph edges to worker graph queues."""
        ready = self.tensor_manager.get_ready_tensors()
        for request_id, edges in ready.items():
            self.worker_graphs_manager.process_new_inputs(
                request_id=request_id, inputs=edges
            )

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_node_batch(self, batch: ScheduledBatch) -> NodeBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_inputs: dict[str, NameToTensorList] = {}
        per_request_info: dict[CurrentForwardPassInfo] = {}

        for request_id, node in batch.node_objects.items():
            tensors = {}
            for input_name in node.ready_inputs:
                tensors[input_name] = [
                    self.tensor_manager.get_tensor(
                        request_id=request_id, uuid=info.uuid
                    ) for info in node.ready_inputs[input_name].tensor_info
                ]
            per_request_inputs[request_id] = tensors
            per_request_info[request_id] = self.worker_graphs_manager.get_fwd_info(request_id)

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
    ) -> dict[str, list[GraphEdge]]:
        """
        For outputs going to other workers: register tensors for RDMA send
        and populate tensor_info on the GraphEdges.
        For outputs staying local: store tensors in tensor_manager.
        Returns the output edges per request (with tensor_info filled in).
        """
        output_edges: dict[str, list[GraphEdge]] = {}

        for request_id, node in batch.node_objects.items():
            # output name to list of tensors
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            ) # name -> list of tensors
            output_edges[request_id] = node.outputs

            if not request_output_tensors:
                # TODO (error handling?): this should not happen
                continue

            self.tensor_manager.store_and_populate_graph_edges(
                request_id=request_id,
                tensors=request_output_tensors,
                graph_edges=node.outputs
            )

            routing = routing_per_request[request_id]
            uuids = set()
            for edge in (
                routing.persist + \
                sum(routing.to_workers.values(), start=[]) + \
                routing.emit_to_client
            ):
                uuids.update([
                    info.uuid for info in edge.tensor_info \
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
        self, request_id: str, outputs: NodeOutputRouting
    ) -> None:
        """
        Send outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        WORKER_GRAPHS_DONE message to avoid race conditions.
        """
        for worker_id, edges in outputs.to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id)
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
                        fwd_pass_number=self.worker_graphs_manager.get_fwd_number(request_id),
                        metadata={}
                    )
                )
                self.communicator.send("api_server", message)

        if outputs.completed_worker_graph_ids:
            message = ConductorMessage(
                message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
                body=WorkerGraphsDone(
                    request_id=request_id,
                    worker_graph_ids=outputs.completed_worker_graph_ids,
                    persist_signals=self.worker_graphs_manager.flush_persist_signals(request_id),
                    new_tokens=self.worker_graphs_manager.flush_new_tokens(request_id),
                    output_signal_names=self.worker_graphs_manager.flush_output_signals(request_id),
                    per_label_seq_info=self.worker_graphs_manager.get_seq_info(request_id),
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

                # 3. Pick next batch via MicroScheduler
                batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
                if batch is None:
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
                
                for rid, req_info in node_batch.per_request_info.items():
                    self.worker_graphs_manager.update_request_info(
                        rid, per_label_seq_info=req_info.per_label_seq_info
                    )

                # 5b. Free consumed input tensors
                self._cleanup_consumed_inputs(batch)

                # 6. Route outputs through WorkerGraphsManager first to determine routing
                routing_per_request: dict[str, NodeOutputRouting] = {}
                for request_id,node in batch.node_objects.items():
                    routing = self.worker_graphs_manager.process_node_outputs(
                        request_id, node.outputs
                    )
                    routing_per_request[request_id] = routing

                # 7. Store output tensors, register RDMA if needed
                self._store_outputs(batch, output, routing_per_request)

                # 8. Send outputs to other workers / conductor
                for request_id in batch.node_objects.keys():
                    self._send_outputs(request_id, routing_per_request[request_id])
            except Exception:
                logger.exception("Worker %s error in main loop", self.worker_id)
                sleep(0.01)
