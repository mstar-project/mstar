

import logging
from dataclasses import dataclass, field

from mminf.graph.base import DestToGraphEdges, GraphEdge, GraphNode, GraphSection, get_node_to_inputs_mapping

logger = logging.getLogger(__name__)


def format_graph_edge_list(
    lst: list[GraphEdge]
):
    return ", ".join([f"{edge.name} -> {edge.next_node}" for edge in lst])


@dataclass
class ProcessedInputs:
    routed_to_this_worker_graph: list[GraphEdge]
    for_other_worker_graphs: list[GraphEdge]


@dataclass
class PerRequestNodeQueues:
    """
    The worker has a list of worker graphs; each worker graph has a list of requests
    using that graph. For every (worker graph, request) pair, we instantiate
    one of these queues.
    """
    waiting: GraphSection | None
    ready: list[GraphNode] = field(default_factory=list)
    worker_graph_id: str = field(default="")

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
        new_inputs: list[GraphEdge]
    ) -> ProcessedInputs:
        """
        Processes all outputs that feed into the waiting graph section, and
        return a dictionary of external output graph edges (ones that are feeding
        to different worker graphs)
        """
        # for input in new_inputs:
        #     input._persist_for_loop = False

        if self.waiting is None:
            return ProcessedInputs(
                routed_to_this_worker_graph=[],
                for_other_worker_graphs=new_inputs,
            )

        logger.debug(
            "Processed new graph inputs: %s.",
            format_graph_edge_list(new_inputs)
        )

        new_inputs: DestToGraphEdges = get_node_to_inputs_mapping(new_inputs)
        ingested = self.waiting.ingest_inputs(new_inputs)
        external_outputs = sum(
            new_inputs.values(), start=[]
        )

        self._update_ready_waiting()
        logger.debug(
            ("Finished processing new graph inputs. Ready nodes: %s, waiting: %s.\n"
             "Ingested inputs %s, didn't ingest %s"),
            str([node.name for node in self.ready]),
            str(list(self.waiting.get_node_names())) if self.waiting else "[]",
            str([i.name for i in ingested]),
            str([e.name for e in external_outputs])
        )
        return ProcessedInputs(
            for_other_worker_graphs=external_outputs, # inputs **not** utilized for self.waiting
            routed_to_this_worker_graph=ingested, # inputs utilized for self.waiting
        )
