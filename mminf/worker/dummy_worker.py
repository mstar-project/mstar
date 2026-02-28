import time

from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager
from mminf.model.base import Subgraph
from mminf.ipc_formats import (
    ConductorMessage, ConductorMessageType, InputSignals,
    NewRequest, RemoveRequest, SubgraphsDone, WorkerMessage, WorkerMessageType
)
from mminf.worker.stage_manager_utils import (
    StageOutputRouting, SubgraphQueues, SubgraphsManager,
)


class DummyWorker:
    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        my_subgraphs: list[Subgraph],
        all_subgraph_ids_to_phases: dict[str, set[str]], # for all subgraphs
        all_subgraph_ids_to_stages: dict[str, list[str]], # for all subgraphs
        hostname: str="localhost", # TODO: figure this out
        socket_path_prefix: str="/tmp/mminf",
        tensor_comm_protocol=CommProtocol.RDMA,
    ):
        """
        Initial in-progress worker implementation. This worker cannnot actually
        do work, but it provides a sense of the data movement between workers
        and the subgraph queue structure.
        """
        self.worker_id = worker_id
        self.subgraphs_manager = SubgraphsManager(
            queues={
                subgraph.subgraph_id: SubgraphQueues(
                    subgraph_id=subgraph.subgraph_id,
                    phases=subgraph.phases,
                    subgraph=subgraph,
                    per_request_queues={}
                ) for subgraph in my_subgraphs
            },
            per_request_info={},
            all_subgraph_ids_to_phases=all_subgraph_ids_to_phases,
            all_subgraph_ids_to_stages=all_subgraph_ids_to_stages
        )

        self.communicator = ZMQCommunicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server"],
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
        Add a request to the subgraph queues
        """
        self.subgraphs_manager.add_request(
            request_id=body.request_id,
            subgraph_ids=body.subgraph_ids,
            subgraph_to_worker_id=body.subgraph_to_worker
        )

        # TODO Atindra: start reading in tensors from body.initial_inputs

        self.subgraphs_manager.update_phase(
            body.request_id, body.initial_phase
        )
        self.subgraphs_manager.process_new_inputs(
            request_id=body.request_id,
            inputs=body.initial_inputs
        )

    def _remove_request(self, body: RemoveRequest):
        """
        Upon seeing EOS, we want to remove the queues for the request that has
        just completed
        """
        self.subgraphs_manager.remove_request(body.request_id)

    def _process_new_inputs(
        self, body: InputSignals
    ):
        """
        When either the conductor or other workers send tensors to this worker,
        process those inputs (update the ready/waiting queues for the proper
        subgraphs on this worker, e.g.)
        """
        self.subgraphs_manager.update_phase(
            body.request_id, body.phase
        )

        # TODO Atindra: start reading in tensors from body.initial_inputs
        # Also, somewhere else, we will have to call self.subgraphs_manager.process_new_inputs
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
            # TODO: handle the tensor_received message

    def _send_outputs(
        self,
        request_id: str,
        outputs: StageOutputRouting
    ):
        """
        Sends outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        SUBGRAPHS_DONE message to avoid race conditions.
        """
        for worker in outputs.to_workers:
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    phase=self.subgraphs_manager.get_phase(request_id),
                    inputs=outputs.to_workers[worker]
                )
            )
            self.communicator.send(worker, message)

        # Buffer persist signals for this request
        if outputs.to_conductor:
            self.subgraphs_manager.buffer_persist_signals(
                request_id, outputs.to_conductor
            )

        if outputs.completed_subgraphs:
            message = ConductorMessage(
                message_type=ConductorMessageType.SUBGRAPHS_DONE,
                body=SubgraphsDone(
                    request_id=request_id,
                    subgraph_ids=outputs.completed_subgraphs,
                    persist_signals=self.subgraphs_manager.flush_persist_signals(request_id),
                )
            )
            self.communicator.send("conductor", message)

    def run(self):
        # TODO: this is just a dummy version
        while True:
            self._process_messages()
            for queue in self.subgraphs_manager.queues.values():
                ready_stage_names = queue.get_ready_stage_names()

                for request_id, names in ready_stage_names.items():
                    stages = queue.pop_ready_stages(request_id, names)
                    for s in stages:
                        outputs = self.subgraphs_manager.process_stage_outputs(
                            request_id, s.outputs
                        )
                        # TODO: in the real worker, we have to update
                        # self.subgraphs_manager.per_request_info[request_id].tensors
                        # with the tensors from the stage output, for all tensor IDs
                        # in outputs.routed_to_this_subgraph

                        self._send_outputs(request_id, outputs)
            time.sleep(0.1) # just for dummy worker to simulate work being done
