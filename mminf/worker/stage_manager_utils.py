from copy import deepcopy
from dataclasses import dataclass, field

from mminf.graph.base import GraphPointer, GraphStage
from mminf.graph.request_queues import PerRequestStageQueues, ProcessedInputs
from mminf.model.base import SPECIAL_DESTINATIONS, STREAM_OUT, Subgraph


@dataclass(frozen=True)
class StageAndPhase:
    """
    Tuple of stage name and phase, e.g., (LLM, decode) or (flow, image_gen)
    """
    stage: str
    phase: str


@dataclass
class StageOutputRouting:
    routed_to_this_subgraph:list[GraphPointer]
    to_conductor: list[GraphPointer] # outputs that are going back to the conductor
    to_workers: dict[str, list[GraphPointer]] # worker id to signals
    completed_subgraphs: list[str] = field(default_factory=list)  # list of subgraph IDs
    stream_out: list[GraphPointer] = field(default_factory=list)
    new_token_outputs: list[GraphPointer] = field(default_factory=list)
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
    - pending_persist_signals: buffered persist signals awaiting flush on
        SUBGRAPHS_DONE
    - tensors: TBD
    """
    stage_to_worker: dict[StageAndPhase, str]  # for all stages
    subgraph_ids: list[str] # for this worker
    current_phase: str = field(default=None)

    # phase_subgraph_ids = subgraphs for the current phase
    phase_subgraph_ids: list[str] = field(default_factory=list) # for this worker

    pending_persist_signals: list[GraphPointer] = field(default_factory=list)
    pending_new_tokens: dict[str, list[int]] = field(default_factory=dict)


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
        new_token_outputs = [ptr for ptr in outputs if ptr.is_new_token]

        # (2) process all internal-facing outputs
        subgraph_ids = self.per_request_info[request_id].phase_subgraph_ids

        completed_subgraphs = []
        routed_to_this_worker = [] # list of graph edges
        external_outputs = outputs
        for subgraph_id in subgraph_ids:
            queue = self.queues[subgraph_id] # ready / waiting graph node queue
            # process_new_inputs consumes outputs that are used as
            # stage inputs within `queue`
            processed_inputs = queue.process_new_inputs(request_id, external_outputs)

            # keep updating outputs to be the edges that have not yet been utilized
            external_outputs = processed_inputs.for_other_subgraphs
            routed_to_this_worker += processed_inputs.routed_to_this_subgraph
            if queue.is_done(request_id):
                completed_subgraphs.append(subgraph_id)
                queue.reset(request_id)
        # all outputs left over at this point are external outputs (to stages
        # in different workers)

        # (3) get mapping of worker to external outputs
        # Skip pointers whose next_stage is a special destination (e.g.,
        # stream_out is a virtual destination, not a real stage on any worker).
        # Note: back_to_conductor pointers may ALSO route to a worker
        # (e.g., concat_text outputs text_emb -> LLM with back_to_conductor=True),
        # so we do NOT filter on back_to_conductor here.
        to_workers: dict[str, list[GraphPointer]] = {}
        stream_out: list[GraphPointer] = []
        for ptr in external_outputs:
            stage_phase = StageAndPhase(
                stage=ptr.next_stage, phase=self.get_phase(request_id)
            )
            if stage_phase not in self.per_request_info[request_id].stage_to_worker:
                if ptr.next_stage in SPECIAL_DESTINATIONS:
                    if ptr.next_stage == STREAM_OUT:
                        stream_out.append(ptr)
                    continue  # e.g., stream_out — already captured in to_conductor
                raise ValueError(
                    f"Output pointer targets unknown stage/phase: {stage_phase}. "
                    f"Check graph construction."
                )
            worker_id = self.per_request_info[request_id].stage_to_worker[stage_phase]
            if worker_id not in to_workers:
                to_workers[worker_id] = []
            to_workers[worker_id].append(ptr)

        return StageOutputRouting(
            routed_to_this_subgraph=routed_to_this_worker,
            to_conductor=to_conductor,
            to_workers=to_workers,
            stream_out=stream_out,
            new_token_outputs=new_token_outputs,
            completed_subgraphs=completed_subgraphs,
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

    def buffer_persist_signals(
            self, request_id: str,
            signals: list[GraphPointer]
        ):
        """Extend the pending persist signals for a request."""
        self.per_request_info[request_id].pending_persist_signals.extend(signals)

    def buffer_new_tokens(
        self, request_id: str,
        new_tokens: dict[str, list[int]]
    ):
        """Update the pending new tokens for a request."""
        for name, tokens in new_tokens.items():
            if name not in self.per_request_info[request_id].pending_new_tokens:
                self.per_request_info[request_id].pending_new_tokens[name] = []
            self.per_request_info[request_id].pending_new_tokens[name].extend(tokens)

    def flush_persist_signals(self, request_id: str) -> list[GraphPointer]:
        """Pop and return all buffered persist signals for a request."""
        info = self.per_request_info[request_id]
        signals = info.pending_persist_signals
        info.pending_persist_signals = []
        return signals

    def flush_new_tokens(self, request_id: str) -> dict[str, list[int]]:
        """Pop and return all buffered new tokens for a request."""
        info = self.per_request_info[request_id]
        new_tokens = info.pending_new_tokens
        info.pending_new_tokens = {}
        return new_tokens
