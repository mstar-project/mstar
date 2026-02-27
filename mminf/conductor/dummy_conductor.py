
from copy import deepcopy
from dataclasses import dataclass, field
import time

import numpy as np

from mminf.communication.communicator import ZMQCommunicator
from mminf.graph.base import GraphPointer, TensorPointerInfo
from mminf.ipc_formats import (
    ConductorMessageType, InputSignals,
    NewRequest, NewRequestConductor, PersistSignals, SubgraphsDone, WorkerMessage,
    WorkerMessageType
)
from mminf.model.base import CurrentForwardMetadata, Model


@dataclass
class RequestData:
    current_forward_metadata: CurrentForwardMetadata
    fwd_inputs: list[GraphPointer]
    persist_signals: dict[str, TensorPointerInfo] # signals passed back to conductor
    subgraph_to_worker: dict[str, str]
    new_tokens: list[int]

    # for tracking progress
    all_subgraph_ids: set[str]
    current_subgraph_ids: set[str]
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
        socket_path_prefix: str="/tmp/mminf"
    ):
        self.requests: dict[str, RequestData] = {}
        self.worker_ids = worker_ids
        self.model = model

        self.subgraphs = {
            subgraph.subgraph_id: subgraph \
                for subgraph in model.get_subgraphs(model_config_file)
        }
        # TODO: properly launch workers

        self.communicator = ZMQCommunicator(
            my_id="conductor",
            push_ids=worker_ids + ["api_server"],
            ipc_socket_path_prefix=socket_path_prefix
        )

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
        inputs: list[GraphPointer]
    ) -> dict[str, list[GraphPointer]]:
        """
        Given the full ForwardPassInputs for kicking off a new forward pass,
        return a mapping of worker_id to the ForwardPassInputs that are routed
        to that worker. ForwardPassInputs consists of graph pointers and tensors.
        """
        inputs_per_worker: dict[str, list[GraphPointer]] = {}
        for subgraph_id, worker_id in subgraph_to_worker.items():
            stages = set(self.subgraphs[subgraph_id].section.get_stage_names())

            inputs_per_worker[worker_id] = [
                ptr for ptr in inputs if ptr.next_stage in stages
            ]
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
            fwd_inputs=[],
            persist_signals=body.initial_signals,
            subgraph_to_worker=subgraph_to_worker,
            all_subgraph_ids=set(subgraph_to_worker.keys()),
            current_subgraph_ids=set(),
            new_tokens=[],
        )
        self.requests[body.request_id] = request_data

        first_forward_inputs = self.model.get_forward_pass_inputs(
            request_data.current_forward_metadata,
            persist_signals=body.initial_signals
        )
        self._set_current_subgraph_ids(
            body.request_id,
            request_data.current_forward_metadata.phase
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
                initial_inputs=inputs_per_worker[worker],
            )
            self.communicator.send(
                worker, WorkerMessage(
                    message_type=WorkerMessageType.NEW_REQUEST,
                    body=message
                )
            )
    
    def _set_current_subgraph_ids(
        self, request_id: str, phase: str
    ):
        self.requests[request_id].current_subgraph_ids = set([
            sg for sg in self.requests[request_id].all_subgraph_ids \
                if phase in self.subgraphs[sg].phases
        ])

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
        
        done_with_forward = request_data.current_subgraph_ids.issubset(
            request_data.completed_subgraph_ids
        )

        if done_with_forward:
            # start a new forward pass
            # TODO: look for EOS
            prev_forward_meta = deepcopy(request_data.current_forward_metadata)
            request_data.current_forward_metadata = \
                self.model.update_for_next_forward(
                    metadata=request_data.current_forward_metadata,
                    new_tokens=request_data.new_tokens,
                )
            self._set_current_subgraph_ids(
                body.request_id,
                request_data.current_forward_metadata.phase
            )

            fwd_inputs = self.model.get_forward_pass_inputs(
                metadata=request_data.current_forward_metadata,
                persist_signals=request_data.persist_signals,
                prev_forward_metadata=prev_forward_meta
            )

            inputs_per_worker = self._split_inputs_to_workers(
                subgraph_to_worker=request_data.subgraph_to_worker,
                inputs=fwd_inputs
            )

            for worker, inputs in inputs_per_worker.items():
                message = WorkerMessage(
                    message_type=WorkerMessageType.INPUT_SIGNALS,
                    body=InputSignals(
                        request_id=body.request_id,
                        phase=request_data.current_forward_metadata.phase,
                        inputs=inputs,
                    )
                )
                self.communicator.send(worker, message)
            
            request_data.fwd_inputs = fwd_inputs
            request_data.new_tokens = []
            request_data.completed_subgraph_ids = set()

    def _process_new_tensors(self, body: PersistSignals):
        """
        If worker has sent tensors back to the conductor, process those.
        """
        self.requests[body.request_id].persist_signals.update(
            body.signals
        )
        self.requests[body.request_id].new_tokens.extend(body.new_tokens)
        

    def run(self):
        while True:
            for message in self.communicator.get_all_new_messages():
                if message.message_type == ConductorMessageType.NEW_REQUEST:
                    self._ingest_request(message.body)
                elif message.message_type == ConductorMessageType.SUBGRAPHS_DONE:
                    self._process_subgraphs_done(message.body)
                elif message.message_type == ConductorMessageType.PERSIST_SIGNALS:
                    self._process_new_tensors(message.body)
                else:
                    raise ValueError(f"Unknown message type: {message.message_type}")
            time.sleep(0.1) # just for dummy conductor!
            