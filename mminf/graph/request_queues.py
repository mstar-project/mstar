

import logging
from dataclasses import dataclass, field

from mminf.graph.base import DestToGraphPointers, GraphPointer, GraphSection, GraphStage, get_stage_to_inputs_mapping

logger = logging.getLogger(__name__)


def format_graph_edge_list(
    lst: list[GraphPointer]
):
    return ", ".join([f"{edge.name} -> {edge.next_stage}" for edge in lst])


@dataclass
class ProcessedInputs:
    routed_to_this_subgraph: list[GraphPointer]
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
        # for input in new_inputs:
        #     input._persist_for_loop = False

        if self.waiting is None:
            return ProcessedInputs(
                routed_to_this_subgraph=[],
                for_other_subgraphs=new_inputs,
            )

        logger.debug(
            "Processed new graph inputs: %s.",
            format_graph_edge_list(new_inputs)
        )

        new_inputs: DestToGraphPointers = get_stage_to_inputs_mapping(new_inputs)
        ingested = self.waiting.ingest_inputs(new_inputs)
        external_outputs = sum(
            new_inputs.values(), start=[]
        )

        self._update_ready_waiting()
        logger.debug(
            ("Finished processing new graph inputs. Ready stages: %s, waiting: %s.\n"
             "Ingested inputs %s, didn't ingest %s"),
            str([node.name for node in self.ready]),
            str(list(self.waiting.get_stage_names())) if self.waiting else "[]",
            str([i.name for i in ingested]),
            str([e.name for e in external_outputs])
        )
        return ProcessedInputs(
            for_other_subgraphs=external_outputs, # inputs **not** utilized for self.waiting
            routed_to_this_subgraph=ingested, # inputs utilized for self.waiting
        )

