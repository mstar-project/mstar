import torch

from mminf.api_server.entrypoint import APIServerMessage, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager, NameToTensorList
from mminf.engine.base import StageBatch, StageOutput
from mminf.graph.base import GraphPointer
from mminf.ipc_formats import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    NewRequest,
    RemoveRequest,
    SubgraphsDone,
    TensorReceived,
    WorkerMessage,
    WorkerMessageType,
)
from mminf.model.base import Model, Subgraph
from mminf.worker.engine_manager import EngineManager
from mminf.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mminf.worker.stage_manager_utils import (
    StageOutputRouting,
    SubgraphQueues,
    SubgraphsManager,
)


class Worker:
    """
    Real worker that integrates SubgraphsManager, EngineManager,
    MicroScheduler, and MooncakeCommunicationManager to execute
    computation via engines.
    """

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        my_subgraphs: list[Subgraph],
        engine_configs: list[dict],
        all_subgraph_ids_to_phases: dict[str, set[str]],
        all_subgraph_ids_to_stages: dict[str, list[str]],
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        model: Model | None = None,
    ):
        self.worker_id = worker_id
        self.device = device

        self.subgraphs_manager = SubgraphsManager(
            queues={
                subgraph.subgraph_id: SubgraphQueues(
                    subgraph_id=subgraph.subgraph_id,
                    phases=subgraph.phases,
                    subgraph=subgraph,
                    per_request_queues={},
                )
                for subgraph in my_subgraphs
            },
            per_request_info={},
            all_subgraph_ids_to_phases=all_subgraph_ids_to_phases,
            all_subgraph_ids_to_stages=all_subgraph_ids_to_stages,
        )

        self.engine_manager = EngineManager.from_config(
            engine_configs, device, model=model
        )
        self.scheduler = MicroScheduler(self.engine_manager)

        self.communicator = ZMQCommunicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        self.tensor_manager = MooncakeCommunicationManager(
            my_entity_id=worker_id,
            hostname=hostname,
            communicator=self.communicator,
            protocol=tensor_comm_protocol,
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _add_new_request(self, body: NewRequest) -> None:
        self.subgraphs_manager.add_request(
            request_id=body.request_id,
            subgraph_ids=body.subgraph_ids,
            subgraph_to_worker=body.subgraph_to_worker,
        )
        self.engine_manager.add_request(body.request_id)

        self.subgraphs_manager.update_phase(
            body.request_id, body.initial_phase
        )

        # Start RDMA reads for tensors that have tensor_info
        self.tensor_manager.start_read_tensors(
            body.request_id, body.initial_inputs,
            device=self.device
        )

        # Signal-only pointers (tensor_info is None) can be processed immediately
        signal_only = [
            ptr for ptr in body.initial_inputs if len(ptr.tensor_info) == 0
        ]
        if signal_only:
            self.subgraphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only
            )

    def _remove_request(self, body: RemoveRequest) -> None:
        self.engine_manager.remove_request(body.request_id)
        self.subgraphs_manager.remove_request(body.request_id)
        self.tensor_manager.cleanup_request(body.request_id)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for name_uuid in body.successful_tensors:
            self.tensor_manager.cleanup(
                body.request_id, name_uuid.tensor_id,
                [name_uuid.uuid]
            )

    def _process_new_inputs(self, body: InputSignals) -> None:
        self.subgraphs_manager.update_phase(body.request_id, body.phase)

        # Start RDMA reads for tensors with tensor_info
        self.tensor_manager.start_read_tensors(
            body.request_id, body.inputs,
            device=self.device
        )

        # Signal-only pointers can be processed immediately
        signal_only = [ptr for ptr in body.inputs if len(ptr.tensor_info) == 0]
        if signal_only:
            self.subgraphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only
            )

    def _process_messages(self) -> None:
        for message in self.communicator.get_all_new_messages():
            if message.message_type == WorkerMessageType.NEW_REQUEST:
                self._add_new_request(message.body)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                self._remove_request(message.body)
            elif message.message_type == WorkerMessageType.INPUT_SIGNALS:
                self._process_new_inputs(message.body)
            elif message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                self._handle_tensor_received(message.body)

    # ------------------------------------------------------------------
    # Tensor readiness
    # ------------------------------------------------------------------

    def _check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready pointers to subgraph queues."""
        ready = self.tensor_manager.get_ready_tensors()
        for request_id, pointers in ready.items():
            self.subgraphs_manager.process_new_inputs(
                request_id=request_id, inputs=pointers
            )

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_stage_batch(self, batch: ScheduledBatch) -> StageBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_inputs: dict[str, NameToTensorList] = {}

        for request_id, stage in batch.stage_objects.items():
            tensors = {}
            for input_name in stage.ready_inputs:
                tensors[input_name] = [
                    self.tensor_manager.get_tensor(
                        request_id=request_id, tensor_name=input_name,
                        uuid=info.uuid
                    ) for info in stage.ready_inputs[input_name].tensor_info
                ]
            per_request_inputs[request_id] = tensors

        return StageBatch(
            stage_name=batch.stage_name,
            phase=batch.phase,
            request_ids=list(batch.stage_objects.keys()),
            per_request_input_tensors=per_request_inputs,
        )

    # ------------------------------------------------------------------
    # Input cleanup
    # ------------------------------------------------------------------

    def _cleanup_consumed_inputs(self, batch: ScheduledBatch) -> None:
        """Free input tensors that were consumed by the just-executed stage."""
        for request_id, stage in batch.stage_objects.items():
            for input in stage.ready_inputs.values():
                if not (input._persist_for_loop or input.back_to_conductor):
                    self.tensor_manager.cleanup(
                        request_id, input.name,
                        [info.uuid for info in input.tensor_info]
                    )

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------

    def _store_outputs(
        self,
        batch: ScheduledBatch,
        output: "StageOutput",
        routing_per_request: dict[str, StageOutputRouting],
    ) -> dict[str, list[GraphPointer]]:
        """
        For outputs going to other workers: register tensors for RDMA send
        and populate tensor_info on the GraphPointers.
        For outputs staying local: store tensors in tensor_manager.
        Returns the output pointers per request (with tensor_info filled in).
        """
        output_pointers: dict[str, list[GraphPointer]] = {}

        for request_id, stage in batch.stage_objects.items():
            # output name to list of tensors
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            ) # name -> list of tensors
            output_pointers[request_id] = stage.outputs

            if not request_output_tensors:
                # TODO (error handling?): this should not happen
                continue

            self.tensor_manager.store_and_populate_graph_edges(
                request_id=request_id,
                tensors=request_output_tensors,
                graph_pointers=stage.outputs
            )

            routing = routing_per_request[request_id]
            seen_uuids = set()
            for ptr in (
                routing.to_conductor + \
                sum(routing.to_workers.values(), start=[]) + \
                routing.stream_out
            ):
                uuids = [
                    info.uuid for info in ptr.tensor_info \
                        if info.uuid not in seen_uuids
                ] # all uuids that have not been seen yet
                if not uuids:
                    continue
                seen_uuids.update(uuids)
                self.tensor_manager.register_for_send(
                    request_id=request_id, name=ptr.name, uuids=uuids
                )

        return output_pointers

    def _send_outputs(
        self, request_id: str, outputs: StageOutputRouting
    ) -> None:
        """
        Send outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        SUBGRAPHS_DONE message to avoid race conditions.
        """
        for worker_id, pointers in outputs.to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    phase=self.subgraphs_manager.get_phase(request_id),
                    inputs=pointers,
                ),
            )
            self.communicator.send(worker_id, message)

        # Buffer persist signals for this request
        if outputs.to_conductor:
            self.subgraphs_manager.buffer_persist_signals(
                request_id, outputs.to_conductor
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
                        tensor_name=signal.name,
                        uuid=tensor_info.uuid
                    )
                    new_tokens.extend(tensor.cpu().numpy().tolist())
                name_to_new_token[signal.name] = new_tokens

            self.subgraphs_manager.buffer_new_tokens(
                request_id, name_to_new_token
            )

        if outputs.stream_out:
            for graph_edge in outputs.stream_out:
                message = APIServerMessage(
                    message_type="result_chunk",
                    body=ResultTensors(
                        request_id=request_id,
                        modality=graph_edge.output_modality,
                        graph_edge=graph_edge,
                        metadata={}
                    )
                )
                self.communicator.send("api_server", message)

        if outputs.completed_subgraphs:
            message = ConductorMessage(
                message_type=ConductorMessageType.SUBGRAPHS_DONE,
                body=SubgraphsDone(
                    request_id=request_id,
                    subgraph_ids=outputs.completed_subgraphs,
                    persist_signals=self.subgraphs_manager.flush_persist_signals(request_id),
                    new_tokens=self.subgraphs_manager.flush_new_tokens(request_id),
                ),
            )
            self.communicator.send("conductor", message)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while True:
            # 1. Process ZMQ messages (new requests, input signals, removals)
            self._process_messages()

            # 2. Check for ready RDMA tensors, feed to subgraph queues
            self._check_ready_tensors()

            # 3. Pick next batch via MicroScheduler
            batch = self.scheduler.get_next_batch(self.subgraphs_manager)
            if batch is None:
                continue

            # 4. Gather input tensors for the batch
            stage_batch = self._build_stage_batch(batch)

            # 5. Execute via engine
            engine = self.engine_manager.get_engine(batch.stage_name)
            output = engine.execute_batch(stage_batch)

            # 5b. Free consumed input tensors
            self._cleanup_consumed_inputs(batch)

            # 6. Route outputs through SubgraphsManager first to determine routing
            routing_per_request: dict[str, StageOutputRouting] = {}
            for request_id,stage in batch.stage_objects.items():
                routing = self.subgraphs_manager.process_stage_outputs(
                    request_id, stage.outputs
                )
                routing_per_request[request_id] = routing

            # 7. Store output tensors, register RDMA if needed
            self._store_outputs(batch, output, routing_per_request)

            # 8. Send outputs to other workers / conductor
            for request_id in batch.stage_objects.keys():
                self._send_outputs(request_id, routing_per_request[request_id])
