import time

from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager
from mminf.ipc_formats import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    NewRequest,
    RemoveRequest,
    WorkerGraphsDone,
    TensorReceived,
    WorkerMessage,
    WorkerMessageType,
)
from mminf.model.base import WorkerGraph
from mminf.worker.stage_manager_utils import (
    StageOutputRouting,
    WorkerGraphQueues,
    WorkerGraphsManager,
)


class DummyWorker:
    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        my_worker_graphs: list[WorkerGraph],
        all_worker_graph_ids_to_phases: dict[str, set[str]], # for all worker graphs
        all_worker_graph_ids_to_stages: dict[str, list[str]], # for all worker graphs
        hostname: str="localhost", # TODO: figure this out
        socket_path_prefix: str="/tmp/mminf",
        tensor_comm_protocol=CommProtocol.RDMA,
    ):
        """
        Initial in-progress worker implementation. This worker cannnot actually
        do work, but it provides a sense of the data movement between workers
        and the worker graph queue structure.
        """
        self.worker_id = worker_id
        self.worker_graphs_manager = WorkerGraphsManager(
            queues={
                worker_graph.worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph.worker_graph_id,
                    phases=worker_graph.phases,
                    worker_graph=worker_graph,
                    per_request_queues={}
                ) for worker_graph in my_worker_graphs
            },
            per_request_info={},
            all_worker_graph_ids_to_phases=all_worker_graph_ids_to_phases,
            all_worker_graph_ids_to_stages=all_worker_graph_ids_to_stages
        )

        self.communicator = ZMQCommunicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix
        )
        self.tensor_manager = MooncakeCommunicationManager(
            my_entity_id=worker_id,
            hostname=hostname,
            communicator=self.communicator,
            protocol=tensor_comm_protocol,
        )

    def _add_new_request(
        self, body: NewRequest
    ):
        """
        Add a request to the worker graph queues
        """
        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            worker_graph_ids=body.worker_graph_ids,
            worker_graph_to_worker=body.worker_graph_to_worker
        )

        # TODO Atindra: start reading in tensors from body.initial_inputs

        self.worker_graphs_manager.update_phase(
            body.request_id, body.initial_phase
        )
        self.worker_graphs_manager.process_new_inputs(
            request_id=body.request_id,
            inputs=body.initial_inputs
        )

    def _remove_request(self, body: RemoveRequest):
        """
        Upon seeing EOS, we want to remove the queues for the request that has
        just completed
        """
        self.worker_graphs_manager.remove_request(body.request_id)
        self.tensor_manager.cleanup_request(body.request_id)

    def _handle_tensor_received(self, body: TensorReceived):
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for name_addr in body.successful_tensors:
            self.tensor_manager.cleanup(
                body.request_id, name_addr.tensor_id,
                name_addr.address
            )

    def _process_new_inputs(
        self, body: InputSignals
    ):
        """
        When either the conductor or other workers send tensors to this worker,
        process those inputs (update the ready/waiting queues for the proper
        worker graphs on this worker, e.g.)
        """
        self.worker_graphs_manager.update_phase(
            body.request_id, body.phase
        )

        # TODO Atindra: start reading in tensors from body.initial_inputs
        # Also, somewhere else, we will have to call self.worker_graphs_manager.process_new_inputs
        # when the tensors are actually ready

    def _process_messages(self):
        """
        Processes all pending messages (communication from conductor and other
        workers to this worker)
        """
        for message in self.communicator.get_all_new_messages():
            if message.message_type == WorkerMessageType.NEW_REQUEST:
                self._add_new_request(message.body)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                self._remove_request(message.body)
            elif message.message_type == WorkerMessageType.INPUT_SIGNALS:
                self._process_new_inputs(message.body)
            elif message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                self._handle_tensor_received(message.body)

    def _send_outputs(
        self,
        request_id: str,
        outputs: StageOutputRouting
    ):
        """
        Sends outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        WORKER_GRAPHS_DONE message to avoid race conditions.
        """
        for worker in outputs.to_workers:
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    phase=self.worker_graphs_manager.get_phase(request_id),
                    inputs=outputs.to_workers[worker]
                )
            )
            self.communicator.send(worker, message)

        # Buffer persist signals for this request
        if outputs.persist:
            self.worker_graphs_manager.buffer_persist_signals(
                request_id, outputs.persist
            )

        if outputs.completed_worker_graph_ids:
            message = ConductorMessage(
                message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
                body=WorkerGraphsDone(
                    request_id=request_id,
                    worker_graph_ids=outputs.completed_worker_graph_ids,
                    persist_signals=self.worker_graphs_manager.flush_persist_signals(request_id),
                )
            )
            self.communicator.send("conductor", message)

    def run(self):
        # TODO: this is just a dummy version
        while True:
            self._process_messages()
            for queue in self.worker_graphs_manager.queues.values():
                ready_stage_names = queue.get_ready_stage_names()

                for request_id, names in ready_stage_names.items():
                    stages = queue.pop_ready_stages(request_id, names)
                    for s in stages:
                        outputs = self.worker_graphs_manager.process_stage_outputs(
                            request_id, s.outputs
                        )
                        # TODO: in the real worker, we have to update
                        # self.worker_graphs_manager.per_request_info[request_id].tensors
                        # with the tensors from the stage output, for all tensor IDs
                        # in outputs.routed_to_this_worker_graph

                        self._send_outputs(request_id, outputs)
            time.sleep(0.1) # just for dummy worker to simulate work being done
