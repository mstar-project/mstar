import logging
from copy import deepcopy
from dataclasses import dataclass, field

from mminf.graph.base import GraphEdge, GraphStage, TensorPointerInfo
from mminf.graph.request_queues import PerRequestStageQueues, ProcessedInputs, format_graph_edge_list
from mminf.graph.special_destinations import SPECIAL_DESTINATIONS, STREAM_OUT
from mminf.model.base import WorkerGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageAndPhase:
    """
    Tuple of stage name and phase, e.g., (LLM, decode) or (flow, image_gen)
    """
    stage: str
    phase: str


@dataclass
class StageOutputRouting:
    routed_to_this_worker_graph:list[GraphEdge]
    persist: list[GraphEdge] # outputs that are going back to the conductor
    to_workers: dict[str, list[GraphEdge]] # worker id to signals
    stream_out: list[GraphEdge] = field(default_factory=list)
    new_token_outputs: list[GraphEdge] = field(default_factory=list)
    completed_worker_graph_ids: list[str] = field(default_factory=list)


@dataclass
class WorkerGraphQueues:
    """
    For a single worker graph, keeps track of which stages are waiting on which
    inputs for each request, and which stages are ready to run per request.
    """
    worker_graph_id: str
    phases: set[str] # e.g., this worker graph is active during decode and image_gen
                     # but not the prefill phase
    worker_graph: WorkerGraph
    per_request_queues: dict[str, PerRequestStageQueues] # request_id -> queue

    def process_new_inputs(self, request_id: str, inputs: list[GraphEdge]) -> ProcessedInputs:
        """
        Add new inputs for a request, and update waiting/ready stages accordingly.
        Returns any signals that should be sent to other worker graphs.
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
            waiting=deepcopy(self.worker_graph.section),
            worker_graph_id=self.worker_graph_id
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
        Remove the given stage names from the ready queue for the request and
        return the corresponding GraphStage objects.
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
        At the end of a worker graph, reset the queues for a request so it can
        be used for the next full model forward pass.
        """
        self.per_request_queues[request_id].waiting = deepcopy(self.worker_graph.section)
        self.per_request_queues[request_id].ready = []


@dataclass
class PerRequestInfo:
    """
    Information about a request that the worker needs to keep track of:
    - stage_to_worker: for all stages. This is, e.g., how we say that if
        an output goes to (LLM, decode phase), what worker that points to.
    - worker_graph_ids: mainly redundant information / syntactic sugar. This is
        the list of worker graph IDs that are on this worker and used by this request
        (across all possible phases)
    - current_phase: which computation path we’re currently on, e.g., prefill,
        decode, image_gen, etc.
    - phase_worker_graph_ids: worker graph IDs used in the current phase (e.g., if there
        is a prefill LLM worker graph and decode LLM worker graph and we are in decode,
        this list only includes the decode worker graph)
    - pending_persist_signals: buffered persist signals awaiting flush on
        WORKER_GRAPHS_DONE
    - tensors: TBD
    """
    stage_to_worker: dict[StageAndPhase, str]  # for all stages
    worker_graph_ids: list[str] # for this worker
    current_phase: str = field(default=None)

    # phase_worker_graph_ids = worker graphs for current phase
    phase_worker_graph_ids: list[str] = field(default_factory=list) # for this worker

    pending_persist_signals: list[GraphEdge] = field(default_factory=list)
    pending_new_tokens: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class WorkerGraphsManager:
    """
    Manages the worker graphs that this worker is responsible for, and the queues
    for each graph and request. Also keeps track of which stages belong
    to which worker graphs, and which worker graphs belong to which phases, for
    routing external outputs to the correct worker.
    """
    queues: dict[str, WorkerGraphQueues] # worker graph id to queues
    per_request_info: dict[str, PerRequestInfo] # request id to info

    # The following two are for routing purposes:
    all_worker_graph_ids_to_phases: dict[str, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_stages: dict[str, str] # for worker graphs on different workers too

    def update_phase(self, request_id: str, phase: str):
        if self.per_request_info[request_id].current_phase != phase:
            self.per_request_info[request_id].current_phase = phase
            self.per_request_info[request_id].phase_worker_graph_ids = [
                graph_id for graph_id in self.per_request_info[request_id].worker_graph_ids \
                    if phase in self.all_worker_graph_ids_to_phases[graph_id]
            ]

    def get_phase(self, request_id: str):
        return self.per_request_info[request_id].current_phase

    def process_new_inputs(
        self,
        request_id: str,
        inputs: list[GraphEdge]
    ):
        """
        Updates queues with new inputs for a request.
        """
        worker_graph_ids = self.per_request_info[request_id].phase_worker_graph_ids
        for worker_graph_id in worker_graph_ids:
            self.queues[worker_graph_id].process_new_inputs(request_id, inputs)


    def process_stage_outputs(
        self, request_id: str,
        outputs: list[GraphEdge]
    ) -> StageOutputRouting:
        """
        After a stage has finished processing, use its outputs to update
        worker graph queues, and return any outputs that should be sent to other
        worker graphs or the conductor.

        I.e., it updates ready/waiting queues for worker graphs on this current
        worker, and directs external outputs to worker graphs on the appropriate
        (different) worker.
        """
        # (1) find back_to_conductor flags
        to_conductor = [edge for edge in outputs if edge.persist]
        new_token_outputs = [edge for edge in outputs if edge.is_new_token]

        # (2) process all internal-facing outputs
        worker_graph_ids = self.per_request_info[request_id].phase_worker_graph_ids

        completed_worker_graph_ids = []
        routed_to_this_worker: list[GraphEdge] = [] # list of graph edges
        external_outputs: list[GraphEdge] = outputs
        for worker_graph_id in worker_graph_ids:
            queue = self.queues[worker_graph_id] # ready / waiting graph node queue
            # process_new_inputs consumes outputs that are used as
            # stage inputs within `queue`
            processed_inputs = queue.process_new_inputs(request_id, external_outputs)

            # keep updating outputs to be the edges that have not yet been utilized
            external_outputs = processed_inputs.for_other_worker_graphs
            routed_to_this_worker += processed_inputs.routed_to_this_worker_graph
            if queue.is_done(request_id):
                completed_worker_graph_ids.append(worker_graph_id)
                queue.reset(request_id)
        # all outputs left over at this point are external outputs (to stages
        # in different workers)

        # (3) get mapping of worker to external outputs
        # Skip edges whose next_stage is a special destination (e.g.,
        # stream_out is a virtual destination, not a real stage on any worker).
        # Note: back_to_conductor edges may ALSO route to a worker
        # (e.g., concat_text outputs text_emb -> LLM with back_to_conductor=True),
        # so we do NOT filter on back_to_conductor here.
        to_workers: dict[str, list[GraphEdge]] = {}
        stream_out: list[GraphEdge] = []
        for edge in external_outputs:
            stage_phase = StageAndPhase(
                stage=edge.next_stage, phase=self.get_phase(request_id)
            )
            if stage_phase not in self.per_request_info[request_id].stage_to_worker:
                if edge.next_stage in SPECIAL_DESTINATIONS or edge.persist:
                    if edge.next_stage == STREAM_OUT:
                        stream_out.append(edge)
                    continue  # e.g., stream_out — already captured in to_conductor
                raise ValueError(
                    f"Output edge targets unknown stage/phase: {stage_phase}. "
                    f"Check graph construction."
                )
            worker_id = self.per_request_info[request_id].stage_to_worker[stage_phase]
            if worker_id not in to_workers:
                to_workers[worker_id] = []
            to_workers[worker_id].append(edge)

        logger.debug(
            ("Finished processing outputs from rid %s. \n"
             "Routed to this worker: %s; sent to others: %s; persist signals: %s"),
            request_id, format_graph_edge_list(routed_to_this_worker),
            format_graph_edge_list(external_outputs), format_graph_edge_list(to_conductor),
        )
        if completed_worker_graph_ids:
            logger.debug("Completed %d worker graphs", len(completed_worker_graph_ids))

        return StageOutputRouting(
            routed_to_this_worker_graph=routed_to_this_worker,
            persist=to_conductor,
            to_workers=to_workers,
            stream_out=stream_out,
            new_token_outputs=new_token_outputs,
            completed_worker_graph_ids=completed_worker_graph_ids,
        )

    def add_request(
        self, request_id: str,
        worker_graph_ids: list[str], # for this worker's worker graphs
        worker_graph_to_worker: dict[str, str] # for other / all worker graphs
    ):
        """
        Set up queues and info for a new request. This includes adding the request
        to the relevant worker graph queues, and updating the mapping of which worker
        is responsible for which stages for this request (for output routing).
        """
        stage_to_worker = {}
        for graph_id in worker_graph_ids:
            self.queues[graph_id].add_request(request_id)

        for worker_graph_id, worker_id in worker_graph_to_worker.items():
            for phase in self.all_worker_graph_ids_to_phases[worker_graph_id]:
                stage_to_worker.update({
                    StageAndPhase(
                        stage=name,
                        phase=phase
                    ): worker_id for name in self.all_worker_graph_ids_to_stages[worker_graph_id]
                })
        self.per_request_info[request_id] = PerRequestInfo(
            stage_to_worker=stage_to_worker,
            worker_graph_ids=worker_graph_ids
        )

    def remove_request(self, request_id: str):
        if request_id in self.per_request_info:
            for queue_id in self.per_request_info[request_id].worker_graph_ids:
                self.queues[queue_id].remove_request(request_id)
            del self.per_request_info[request_id]

    def buffer_persist_signals(
            self, request_id: str,
            signals: list[GraphEdge]
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

    def flush_persist_signals(self, request_id: str) -> dict[str, list[TensorPointerInfo]]:
        """Pop and return all buffered persist signals for a request.

        Converts from internal list[GraphEdge] to the dict format
        expected by the conductor (name -> list[TensorPointerInfo]).
        """
        info = self.per_request_info[request_id]
        signals = info.pending_persist_signals
        info.pending_persist_signals = []
        result: dict[str, list[TensorPointerInfo]] = {}
        for edge in signals:
            result[edge.name] = edge.tensor_info
        return result

    def flush_new_tokens(self, request_id: str) -> dict[str, list[int]]:
        """Pop and return all buffered new tokens for a request."""
        info = self.per_request_info[request_id]
        new_tokens = info.pending_new_tokens
        info.pending_new_tokens = {}
        return new_tokens
