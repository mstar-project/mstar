import logging

import torch

from mminf.api_server.request_types import APIServerMessage, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager, NameToTensorList
from mminf.engine.base import NodeBatch, NodeOutput
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
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        model: Model | None = None,
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

        self.engine_manager = EngineManager.from_config(
            engine_configs=engine_configs, device=device, model=model,
            enable_nvtx=self.enable_nvtx
        )
        self.scheduler = MicroScheduler(self.engine_manager)

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
        )

        # Per-request metadata from conductor (e.g., cache_labels, snapshot_after)
        self._per_request_metadata: dict[str, dict] = {}
        self._unprocessed_messages = {} # req_id -> messages for requests that are not in the queue

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _add_new_request(self, body: NewRequest) -> None:
        logger.debug("Worker %s received request %s", self.worker_id, body.request_id)
        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            worker_graph_ids=body.worker_graph_ids,
            worker_graph_to_worker=body.worker_graph_to_worker,
        )
        self.engine_manager.add_request(body.request_id)

        self.worker_graphs_manager.update_graph_walk_and_fwd_number(
            body.request_id, body.initial_graph_walk, fwd_number=0
        )
        logger.debug(
            "Request %s set to graph walk %s on worker %s",
            body.request_id, body.initial_graph_walk, self.worker_id
        )

        # Store per-request metadata from conductor
        if body.per_request_metadata:
            self._per_request_metadata[body.request_id] = body.per_request_metadata

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
        self._per_request_metadata.pop(body.request_id, None)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for (uuid, ref_cnt) in body.successful_tensors.items():
            self.tensor_manager.dereference(
                body.request_id, uuid, n=ref_cnt
            )

    def _process_new_inputs(self, body: InputSignals) -> None:
        self.worker_graphs_manager.update_graph_walk_and_fwd_number(
            body.request_id, body.graph_walk,body.fwd_pass_number
        )
        logger.debug(
            "Request %s set to graph walk %s on worker %s",
            body.request_id, body.graph_walk, self.worker_id
        )

        logger.debug(
            "Received new signals %s at worker %s for request %s",
            format_graph_edge_list(body.inputs), self.worker_id, body.request_id
        )

        # Update per-request metadata from conductor
        if body.per_request_metadata:
            self._per_request_metadata[body.request_id] = body.per_request_metadata

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
                message.body.request_id not in self._per_request_metadata
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
        per_request_metadata: dict[str, dict] = {}

        for request_id, node in batch.node_objects.items():
            tensors = {}
            for input_name in node.ready_inputs:
                tensors[input_name] = [
                    self.tensor_manager.get_tensor(
                        request_id=request_id, uuid=info.uuid
                    ) for info in node.ready_inputs[input_name].tensor_info
                ]
            per_request_inputs[request_id] = tensors

            # Include per-request metadata (e.g., cache_labels, snapshot_after)
            if request_id in self._per_request_metadata:
                per_request_metadata[request_id] = self._per_request_metadata[request_id]

        return NodeBatch(
            node_name=batch.node_name,
            graph_walk=batch.graph_walk,
            request_ids=list(batch.node_objects.keys()),
            per_request_input_tensors=per_request_inputs,
            per_request_metadata=per_request_metadata,
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
                    graph_walk=self.worker_graphs_manager.get_graph_walk(request_id),
                    fwd_pass_number=self.worker_graphs_manager.get_fwd_number(request_id),
                    inputs=edges,
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
                    output_signal_names=self.worker_graphs_manager.flush_output_signals(request_id)
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
