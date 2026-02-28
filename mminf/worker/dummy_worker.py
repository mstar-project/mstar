from copy import deepcopy
from dataclasses import dataclass, field
import time

from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager
from mminf.model.base import Subgraph
from mminf.graph.base import GraphPointer, GraphStage
from mminf.graph.request_queues import PerRequestStageQueues, ProcessedInputs
from mminf.ipc_formats import (
    ConductorMessage, ConductorMessageType, InputSignals,
    NewRequest, RemoveRequest, SubgraphsDone, WorkerMessage, WorkerMessageType
)


@dataclass(frozen=True)
class StageAndPhase:
    """
    Tuple of stage name and phase, e.g., (LLM, decode) or (flow, image_gen)
    """
    stage: str
    phase: str


@dataclass
class StageOutputRouting:
    routed_to_this_subgraph: set[str] # set of tensor ids
    to_conductor: list[GraphPointer] # outputs that are going back to the conductor
    to_workers: dict[str, list[GraphPointer]] # worker id to signals
    completed_subgraphs: list[str] = field(default_factory=[])  # list of subgraph IDs


@dataclass
class SubgraphQueues:
    """
    For a single subgraph, keeps track of which stages are waiting on which
    inputs for each request, and which stages are ready to run per request.
    """
    subgraph_id: str
    phases: set[str] # e.g., this subgraph is active during decode and image_gen
                     # but not the prefill phase
    subgraph: Subgraph
    per_request_queues: dict[str, PerRequestStageQueues] # request_id -> queue

    def process_new_inputs(self, request_id: str, inputs: list[GraphPointer]) -> ProcessedInputs:
        """
        Add new inputs for a request, and update waiting/ready stages accordingly.
        Returns any signals that should be sent to other subgraphs.
        """
        return self.per_request_queues[request_id].process_new_inputs(inputs)
    
    def is_done(self, request_id) -> bool:
        q = self.per_request_queues[request_id]
        return q.waiting is None and len(q.ready) == 0
    
    def add_request(self, request_id: str):
        """
        Initialize queues for a new request
        """
        self.per_request_queues[request_id] = PerRequestStageQueues(
            waiting=deepcopy(self.subgraph.section),
            subgraph_id=self.subgraph_id
        )

    def remove_request(self, request_id: str):
        """
        Delete queues for a completed/removed request (saw EOS)
        """
        if request_id in self.per_request_queues:
            del self.per_request_queues[request_id]
    
    def get_ready_stage_names(self) -> dict[str, str]:
        """
        Returns mapping of request id to ready stage names for that request
        """
        return {
            request_id: [s.name for s in q.ready] \
                for (request_id, q) in self.per_request_queues.items()
        }

    def pop_ready_stages(
        self, request_id: str, stage_names: list[str]
    ) -> list[GraphStage]:
        """
        Remove the given stage names from the ready queue for the request an
        return the corresponding GraphStage objects
        """
        stages = []
        if request_id in self.per_request_queues:
            q = self.per_request_queues[request_id]
            pop_idxs = set(
                [i for i, stage in enumerate(q.ready) if stage.name in set(stage_names)]
            )
            stages = [q.ready[i] for i in pop_idxs]
            q.ready = [stage for i, stage in enumerate(q.ready) if i not in pop_idxs]
        return stages
    
    def reset(self, request_id):
        """
        At the end of a subgraph, reset the queues for that request so that it
        can be used for the next full model forward pass
        """
        self.per_request_queues[request_id].waiting = deepcopy(self.subgraph.section)
        self.per_request_queues[request_id].ready = []


@dataclass
class PerRequestInfo:
    """
    Information about a request that the worker needs to keep track of:
    - stage_to_worker: for all stages. This is, e.g., how we say that if
        an output goes to (LLM, decode phase), what worker that points to.
    - subgraph_ids: mainly redundant information / syntactic sugar. This is
        the list of subgraph IDs that are on this worker and used by this request
        (across all possible phases)
    - current_phase: which computation path we are currently on, e.g., prefill,
        decode, image_gen, etc.
    - phase_subgraph_ids: subgraph IDs used in the current phase (e.g., if there
        is a prefill LLM subgraph and decode LLM subgraph and we are in decode,
        this list only includes the decode subgraph)
    - tensors: TBD
    """
    stage_to_worker: dict[StageAndPhase, str]  # for all stages
    subgraph_ids: list[str] # for this worker
    current_phase: str = field(default=None)

    # phase_subgraph_ids = subgraphs for the current phase
    phase_subgraph_ids: list[str] = field(default_factory=list) # for this worker


@dataclass
class SubgraphsManager:
    """
    Manages the subgraphs that this worker is responsible for, and the queues
    for each subgraph and request. Also keeps track of which stages belong
    to which subgraphs, and which subgraphs belong to which phases, for
    routing external outputs to the correct worker.
    """
    queues: dict[str, SubgraphQueues] # subgraph_id to queues
    per_request_info: dict[str, PerRequestInfo] # request id to info

    # The following two are for routing purposes:
    all_subgraph_ids_to_phases: dict[str, set[str]] # for subgraphs on different workers too
    all_subgraph_ids_to_stages: dict[str, str] # for subgraphs on different workers too

    def update_phase(self, request_id: str, phase: str):
        if self.per_request_info[request_id].current_phase != phase:
            self.per_request_info[request_id].current_phase = phase
            self.per_request_info[request_id].phase_subgraph_ids = [
                id for id in self.per_request_info[request_id].subgraph_ids \
                    if phase in self.all_subgraph_ids_to_phases[id]
            ]
    
    def get_phase(self, request_id: str):
        return self.per_request_info[request_id].current_phase

    def process_new_inputs(
        self,
        request_id: str,
        inputs: list[GraphPointer]
    ):
        """
        Updates queues with new inputs for a request
        """
        subgraph_ids = self.per_request_info[request_id].phase_subgraph_ids
        for subgraph_id in subgraph_ids:
            self.queues[subgraph_id].process_new_inputs(request_id, inputs)
        

    def process_stage_outputs(
        self, request_id: str,
        outputs: list[GraphPointer]
    ) -> StageOutputRouting:
        """
        After a stage has finished processing, use its outputs to update
        subgraph queues, and return any outputs that should be sent to other
        subgraphs or the conductor.

        I.e., it updates ready/waiting queues for subgraphs on this current
        worker, and directs external outputs to subgraphs on the appropriate
        (different) worker.
        """
        # (1) find back_to_conductor flags
        to_conductor = [ptr for ptr in outputs if ptr.back_to_conductor]

        # (2) process all internal-facing outputs
        subgraph_ids = self.per_request_info[request_id].phase_subgraph_ids

        completed_subgraphs = []
        routed_to_this_subgraph = set()
        for subgraph_id in subgraph_ids:
            queue = self.queues[subgraph_id]
            # process_new_inputs consumes outputs that are used as
            # stage inputs within `queue`, and returns the graph pointers that
            # were not consumed
            processed_inputs = queue.process_new_inputs(request_id, outputs)
            outputs = processed_inputs.for_other_subgraphs
            routed_to_this_subgraph.update(processed_inputs.routed_to_this_subgraph)
            if queue.is_done(request_id):
                completed_subgraphs.append(subgraph_id)
                queue.reset(request_id)
        # all outputs left over at this point are external outputs (to stages
        # in different workers)

        # (3) get mapping of worker to external outputs
        # Filter out back_to_conductor pointers (e.g., stream_out) — they
        # don't route to any worker
        to_workers: dict[str, list[GraphPointer]] = {}
        for ptr in outputs:
            if ptr.back_to_conductor:
                continue
            stage_phase = StageAndPhase(
                stage=ptr.next_stage, phase=self.get_phase(request_id)
            )
            if stage_phase not in self.per_request_info[request_id].stage_to_worker:
                raise ValueError(
                    f"Output pointer targets unknown stage/phase: {stage_phase}. "
                    f"Check graph construction."
                )
            worker_id = self.per_request_info[request_id].stage_to_worker[stage_phase]
            if worker_id not in to_workers:
                to_workers[worker_id] = []
            to_workers[worker_id].append(ptr)

        return StageOutputRouting(
            routed_to_this_subgraph=routed_to_this_subgraph,
            to_conductor=to_conductor,
            to_workers=to_workers,
            completed_subgraphs=completed_subgraphs
        )
    
    def add_request(
        self, request_id: str,
        subgraph_ids: list[str], # for this worker's subgraphs
        subgraph_to_worker: dict[str, str] # for other / all subgraphs
    ):
        """
        Set up queues and info for a new request. This includes adding the request
        to the relevant subgraph queues, and updating the mapping of which worker
        is responsible for which stages for this request (for output routing).
        """
        stage_to_worker = {}
        for id in subgraph_ids:
            self.queues[id].add_request(request_id)
        
        for subgraph_id, worker_id in subgraph_to_worker.items():
            for phase in self.all_subgraph_ids_to_phases[subgraph_id]:
                stage_to_worker.update({
                    StageAndPhase(
                        stage=name,
                        phase=phase
                    ): worker_id for name in self.all_subgraph_ids_to_stages[subgraph_id]
                })
        self.per_request_info[request_id] = PerRequestInfo(
            stage_to_worker=stage_to_worker,
            subgraph_ids=subgraph_ids
        )
    
    def remove_request(self, request_id: str):
        if request_id in self.per_request_info:
            for queue_id in self.per_request_info[request_id].phase_subgraph_ids:
                self.queues[queue_id].remove_request(request_id)
            del self.per_request_info[request_id]


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
        Sends outputs to other workers and to the conductor
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
        
        # to conductor
        if outputs.to_conductor:
            message = ConductorMessage(
                message_type=ConductorMessageType.PERSIST_SIGNAL,
                body=InputSignals(
                    request_id=request_id,
                    inputs=outputs.to_conductor,
                    phase=self.subgraphs_manager.get_phase(request_id)
                )
            )
            self.communicator.send("conductor", message)
        
        if outputs.completed_subgraphs:
            message = ConductorMessage(
                message_type=ConductorMessageType.SUBGRAPHS_DONE,
                body=SubgraphsDone(
                    request_id=request_id,
                    subgraph_ids=outputs.completed_subgraphs
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