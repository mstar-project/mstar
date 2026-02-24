
from dataclasses import dataclass, field
import time

import numpy as np
import zmq

from mminf.ipc_formats import (
    ConductorMessage, ConductorMessageType, ConductorTensors, InputTensors,
    NewRequest, NewRequestConductor, SubgraphsDone, WorkerMessage,
    WorkerMessageType
)
from mminf.model.base import CurrentForwardMetadata, ForwardPassInputs, Model, TensorData


@dataclass
class RequestData:
    current_forward_metadata: CurrentForwardMetadata
    tensors: dict[str, TensorData]
    new_outputs: dict[str, TensorData] # tensors passed back to conductor
    subgraph_to_worker: dict[str, str]

    # for tracking progress
    all_subgraph_ids: set[str]
    completed_subgraph_ids: set[str] = field(default_factory=set)

    # TODO: will need to add to this as we build things out



class DummyConductor:
    """
    Initial in-progress conductor implementation. TODO: this is extremely
    un-optimized, but it provides a sense of the data movement between the
    conductor and the workers
    """
    def __init__(
        self,
        worker_ids: list[str],
        model: Model,
        model_config_file: str,
        worker_socket_path_prefix: str="/tmp/mminf/workers/",
        conductor_socket_path: str="/tmp/mminf/conductor.ipc"
    ):
        self.requests: dict[str, RequestData] = {}
        self.worker_ids = worker_ids
        self.model = model

        self.subgraphs = {
            subgraph.subgraph_id: subgraph \
                for subgraph in model.get_subgraphs(model_config_file)
        }
        # TODO: properly launch workers via Ray

        self.context = zmq.Context()
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.connect(f"ipc://{conductor_socket_path}")
        self.result_socket.setsockopt(zmq.LINGER, 0)

        self.worker_sockets: dict[str, zmq.SyncSocket] = {}
        for id in worker_ids:
            self.worker_sockets[id] = self.context.socket(zmq.PUSH)
            self.worker_sockets[id].connect(
                f"ipc://{worker_socket_path_prefix}/{id}.ipc"
            )
            self.worker_sockets[id].setsockopt(zmq.LINGER, 0)

    def _assign_subgraphs_to_workers(self) -> dict[str, str]:
        """
        For a request, assign subgraphs to workers. This is relevant in the
        data parallel case, where there may be a subgraph that is replicated
        across many workers.
        """
        # Do a random policy for now. TODO: refine this
        return {
            subgraph_id: self.worker_ids[np.random.choice(subgraph.ranks)] \
                for subgraph_id, subgraph in self.subgraphs.items()
        }
    
    def _split_inputs_to_workers(
        self, subgraph_to_worker: dict[str, str],
        inputs: ForwardPassInputs
    ) -> dict[str, ForwardPassInputs]:
        """
        Given the full ForwardPassInputs for kicking off a new forward pass,
        return a mapping of worker_id to the ForwardPassInputs that are routed
        to that worker. ForwardPassInputs consists of graph pointers and tensors.
        """
        inputs_per_worker: dict[str, ForwardPassInputs] = {}
        for subgraph_id, worker_id in subgraph_to_worker.items():
            stages = set(self.subgraphs[subgraph_id].section.get_stage_names())
            pointers = {}
            tensors = {}

            for signal, dests in inputs.pointers.items():
                dests = list(filter(lambda dest: dest.next_stage in stages, dests))
                if len(dests) == 0:
                    continue
                pointers[signal] = dests
                tensors[signal] = inputs.tensors[signal]
            inputs_per_worker[worker_id] = ForwardPassInputs(
                tensors=tensors,
                pointers=pointers
            )
        
        return inputs_per_worker


    def _ingest_request(
        self, body: NewRequestConductor
    ):
        """
        When a new request comes in from the API server, assign workers for each
        subgraph (for all possible execution phases, e.g., prefill, decode, image_gen),
        and notify the workers that the request has arrived + provide the appropriate
        workers with the appropriate initial inputs for the forward pass 
        """
        subgraph_to_worker = self._assign_subgraphs_to_workers()
        request_data = RequestData(
            current_forward_metadata=self.model.get_initial_forward_metadata(
                body.initial_input_modalities,
                body.initial_output_modalities
            ),
            tensors=body.initial_inputs,
            subgraph_to_worker=subgraph_to_worker,
            all_subgraph_ids=set(subgraph_to_worker.keys())
        )
        self.requests[body.request_id] = request_data

        first_forward_inputs = self.model.get_forward_pass_inputs(
            request_data.tensors,
            request_data.current_forward_metadata
        )

        # send data to appropriate workers
        worker_to_subgraph_ids: dict[str, list[str]] = {}
        inputs_per_worker = self._split_inputs_to_workers(
            subgraph_to_worker=subgraph_to_worker,
            inputs=first_forward_inputs
        )
        for subgraph_id, worker_id in subgraph_to_worker.items():
            if worker_id not in worker_to_subgraph_ids:
                worker_to_subgraph_ids[worker_id] = []
            worker_to_subgraph_ids[worker_id].append(subgraph_id)

        for worker, subgraph_ids in worker_to_subgraph_ids:
            message = NewRequest(
                request_id=body.request_id,
                subgraph_ids=subgraph_ids,
                subgraph_to_worker=subgraph_to_worker,
                initial_phase=request_data.current_forward_metadata.phase,
                initial_inputs=inputs_per_worker[worker].pointers,
                initial_tensors=inputs_per_worker[worker].tensors
            )
            self.worker_sockets[worker].send_pyobj(WorkerMessage(
                message_type=WorkerMessageType.NEW_REQUEST,
                body=message
            ))
    
    def _process_subgraphs_done(
        self, body: SubgraphsDone
    ):
        """
        When some subgraphs have completed (the worker notifies the conductor that
        the subgraphs have completed), update the metadata for this request. If this
        is the end of a forward pass (i.e., all subgraphs for the current computation
        phase have completed), then start a new forward pass (determine the input and
        output modalities for the new forward pass, wrangle input tensors and send
        them to the appropriate workers)
        """
        request_data = self.requests[body.request_id]
        request_data.completed_subgraph_ids.update(
            body.subgraph_ids
        )
        
        done_with_forward = request_data.all_subgraph_ids.issubset(
            request_data.completed_subgraph_ids
        )

        if done_with_forward:
            # start a new forward pass
            # TODO: look for EOS
            self.model.update_for_next_forward(
                metadata=request_data.current_forward_metadata,
                input_tensors=request_data.tensors,
                new_outputs=request_data.new_outputs
            )

            fwd_inputs = self.model.get_forward_pass_inputs(
                input_tensors=request_data.tensors,
                metadata=request_data.current_forward_metadata
            )

            inputs_per_worker = self._split_inputs_to_workers(
                subgraph_to_worker=request_data.subgraph_to_worker,
                inputs=fwd_inputs
            )

            for worker, inputs in inputs_per_worker.items():
                message = WorkerMessage(
                    message_type=WorkerMessageType.INPUT_TENSORS,
                    body=InputTensors(
                        request_id=body.request_id,
                        phase=request_data.current_forward_metadata.phase,
                        inputs=inputs.pointers,
                        tensors=inputs.tensors
                    )
                )
                self.worker_sockets[worker].send_pyobj(message)

    def _process_new_tensors(self, body: ConductorTensors):
        """
        If worker has sent tensors back to the conductor, process those.
        """
        self.requests[body.request_id].tensors.update(body.tensors)

    def run(self):
        while True:
            message: ConductorMessage = self.result_socket.recv_pyobj()
            if message.message_type == ConductorMessageType.NEW_REQUEST:
                self._ingest_request(message.body)
            elif message.message_type == ConductorMessageType.SUBGRAPHS_DONE:
                self._process_subgraphs_done(message.body)
            elif message.message_type == ConductorMessageType.TENSORS:
                self._process_new_tensors(message.body)
            else:
                raise ValueError(f"Unknown message type: {message.message_type}")
            time.sleep(0.1) # just for dummy conductor!
            