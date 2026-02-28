

from dataclasses import dataclass, field

from mminf.graph.base import (
    DestToGraphPointers, GraphPointer, GraphSection, GraphStage,
    get_stage_to_inputs_mapping
)

@dataclass
class ProcessedInputs:
    routed_to_this_subgraph: set[str]
    for_other_subgraphs: list[GraphPointer]


@dataclass
class PerRequestStageQueues:
    """
    The worker has a list of subgraphs; each subgraph has a list of requests
    using that subgraph. For every (subgraph, request) pair, we instantiate
    one of these queues.
    """
    waiting: GraphSection | None
    ready: list[GraphStage] = field(default_factory=list)
    subgraph_id: str = field(default="")
    
    def _update_ready_waiting(self):
        """
        Moves sections from the waiting section to the ready queue,
        replaces self.waiting with whatever's left
        """
        if self.waiting is None:
            return
        new_ready, new_waiting = self.waiting.split_off_ready()
        self.ready += new_ready
        self.waiting = new_waiting
    
    def process_new_inputs(
        self,
        new_inputs: list[GraphPointer]
    ) -> ProcessedInputs:
        """
        Processes all outputs that feed into the waiting graph section, and
        return a dictionary of external output pointers (ones that are feeding
        to different subgraphs)
        """
        if self.waiting is None:
            return ProcessedInputs(
                routed_to_this_subgraph=set(),
                for_other_subgraphs=new_inputs,
            )

        new_inputs: DestToGraphPointers = get_stage_to_inputs_mapping(new_inputs)
        ingested = self.waiting.ingest_inputs(new_inputs)
        external_outputs = sum(
            new_inputs.values(), start=[]
        )
        
        self._update_ready_waiting()
        return ProcessedInputs(
            for_other_subgraphs=external_outputs,
            routed_to_this_subgraph=ingested
        )

