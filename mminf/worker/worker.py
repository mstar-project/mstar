import torch

from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager, NameAndRequestId
from mminf.engine.base import StageBatch
from mminf.graph.base import GraphPointer, TensorPointerInfo
from mminf.ipc_formats import (
    ConductorMessage, ConductorMessageType, InputSignals,
    NewRequest, RemoveRequest, SubgraphsDone, WorkerMessage, WorkerMessageType,
)
from mminf.model.base import Subgraph
from mminf.worker.dummy_worker import (
    StageOutputRouting, SubgraphQueues, SubgraphsManager,
)
from mminf.worker.engine_manager import EngineManager
from mminf.worker.micro_scheduler import MicroScheduler, ScheduledBatch


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

        self.engine_manager = EngineManager.from_config(engine_configs, device)
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
            body.request_id, body.initial_inputs
        )

        # Signal-only pointers (tensor_info is None) can be processed immediately
        signal_only = [
            ptr for ptr in body.initial_inputs if ptr.tensor_info is None
        ]
        if signal_only:
            self.subgraphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only
            )

    def _remove_request(self, body: RemoveRequest) -> None:
        self.engine_manager.remove_request(body.request_id)
        self.subgraphs_manager.remove_request(body.request_id)

    def _process_new_inputs(self, body: InputSignals) -> None:
        self.subgraphs_manager.update_phase(body.request_id, body.phase)

        # Start RDMA reads for tensors with tensor_info
        self.tensor_manager.start_read_tensors(body.request_id, body.inputs)

        # Signal-only pointers can be processed immediately
        signal_only = [ptr for ptr in body.inputs if ptr.tensor_info is None]
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
        per_request_inputs: dict[str, dict[str, torch.Tensor]] = {}

        for i, request_id in enumerate(batch.request_ids):
            stage = batch.stages[i] if i < len(batch.stages) else batch.stages[0]
            tensors = {}
            for input_name in stage.input_ids:
                key = NameAndRequestId(input_name, request_id)
                if key in self.tensor_manager.tensors:
                    tensors[input_name] = self.tensor_manager.tensors[key]
            per_request_inputs[request_id] = tensors

        return StageBatch(
            stage_name=batch.stage_name,
            phase=batch.phase,
            request_ids=batch.request_ids,
            per_request_input_tensors=per_request_inputs,
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

        for i, request_id in enumerate(batch.request_ids):
            stage = batch.stages[i] if i < len(batch.stages) else batch.stages[0]
            request_output_tensors = output.per_request_output_tensors.get(
                request_id, {}
            )

            # Store all output tensors locally
            for tensor_name, tensor in request_output_tensors.items():
                self.tensor_manager.tensors[
                    NameAndRequestId(tensor_name, request_id)
                ] = tensor

            routing = routing_per_request[request_id]

            # For tensors going to other workers, register for RDMA send
            external_pointers = []
            for worker_id, pointers in routing.to_workers.items():
                external_pointers.extend(pointers)

            if external_pointers and request_output_tensors:
                # Filter to only tensors that actually go external
                external_names = {ptr.name for ptr in external_pointers}
                external_tensors = {
                    name: t for name, t in request_output_tensors.items()
                    if name in external_names
                }
                if external_tensors:
                    self.tensor_manager.register_and_prepare_to_send(
                        request_id, external_tensors, external_pointers
                    )

            output_pointers[request_id] = stage.outputs

        return output_pointers

    def _send_outputs(
        self, request_id: str, outputs: StageOutputRouting
    ) -> None:
        """Send outputs to other workers and to the conductor."""
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

        if outputs.to_conductor:
            message = ConductorMessage(
                message_type=ConductorMessageType.PERSIST_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=outputs.to_conductor,
                    phase=self.subgraphs_manager.get_phase(request_id),
                ),
            )
            self.communicator.send("conductor", message)

        if outputs.completed_subgraphs:
            message = ConductorMessage(
                message_type=ConductorMessageType.SUBGRAPHS_DONE,
                body=SubgraphsDone(
                    request_id=request_id,
                    subgraph_ids=outputs.completed_subgraphs,
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

            # 6. Route outputs through SubgraphsManager first to determine routing
            routing_per_request: dict[str, StageOutputRouting] = {}
            for i, request_id in enumerate(batch.request_ids):
                stage = batch.stages[i] if i < len(batch.stages) else batch.stages[0]
                routing = self.subgraphs_manager.process_stage_outputs(
                    request_id, stage.outputs
                )
                routing_per_request[request_id] = routing

            # 7. Store output tensors, register RDMA if needed
            self._store_outputs(batch, output, routing_per_request)

            # 8. Send outputs to other workers / conductor
            for request_id in batch.request_ids:
                self._send_outputs(request_id, routing_per_request[request_id])
