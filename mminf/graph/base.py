import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def update_list_dicts(signals: dict[str, list], new_signals: dict[str, list]):
    for key, new_signals_elem in new_signals.items():
        if key in signals:
            signals[key].extend(new_signals_elem)
        else:
            signals[key] = new_signals_elem


@dataclass
class TensorPointerInfo:
    dims: list[int]
    dtype: str
    nbytes: int
    address: int
    stride: list[int]
    uuid: str # for indexing storage
    source_session_id: str # "{HOSTNAME}:{client_engine.get_rpc_port()}"
    source_entity: str # which {worker, api_server} the tensor is on

# class ConnectionType(Enum):
#     STREAM = "stream" # thinker-talker
#     BLOCKING = "blocking" # we need to wait for all tensors in the list to finish

# assume blocking case for all nodes for now.

@dataclass
class GraphEdge:
    next_node: str
    # connection_type: ConnectionType
    # wait_for_next_tensor : bool = True
    name: str
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)

    # Flags
    persist: bool = field(default=False) # previously back_to_conductor
    is_new_token: bool = field(default=False)
    # only for EMIT_TO_CLIENT
    output_modality: str = field(default="") # text | image | video | audio
    _persist_for_loop: bool = field(default=False)

# Two different ways of defining graph edges
DestToGraphEdges = dict[str, list[GraphEdge]]

def get_node_to_inputs_mapping(
    new_inputs: list[GraphEdge]
)-> DestToGraphEdges:
    result: DestToGraphEdges = {}
    for edge in new_inputs:
        if edge.next_node not in result:
            result[edge.next_node] = []
        result[edge.next_node].append(edge)
    return result


@dataclass
class GraphSection(ABC):
    @abstractmethod
    def get_node_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_inputs(self) -> list[GraphEdge]:
        """
        All external or "loop-back" inputs into a worker graph
        """
        pass

    @abstractmethod
    def get_outputs(self) -> list[GraphEdge]:
        """
        All external or "loop-back" outputs from a worker graph
        """
        pass

    @abstractmethod
    def ingest_inputs(self, node_to_inputs: DestToGraphEdges) -> list[GraphEdge]:
        """
        Adds inputs to the appropriate "ready_input_ids", and **mutates
        node_to_inputs** to remove the inputs that were able to be added
        to the "ready_input_ids".
        """
        pass

    @abstractmethod
    def split_off_ready(self) -> tuple[list["GraphNode"], "GraphSection"]:
        """
        Returns a list of nodes that are ready to be run, and a graph section
        of what is still waiting.
        """
        pass


@dataclass
class GraphNode(GraphSection):
    name: str
    input_ids: set[str]
    outputs: list[GraphEdge]
    consumes_stream: bool = field(default=False)

    # Populated as previous nodes complete
    # This will also include, e.g., tensor UUIDs associated with these inputs
    ready_inputs: dict[str, GraphEdge] = field(default_factory=dict) # name -> graph edge

    def __post_init__(self):
        # if the user inputs, e.g., a list, turn it into a set
        self.input_ids = set(self.input_ids)

    def is_ready(self):
        return self.input_ids.issubset(set(self.ready_inputs.keys()))

    def get_node_names(self) -> list[str]:
        return [self.name]

    def get_inputs(self) -> list[GraphEdge]:
        return [
            GraphEdge(next_node=self.name, name=id) for id in self.input_ids
        ]

    def get_outputs(self) -> list[GraphEdge]:
        return self.outputs

    def ingest_inputs(self, node_to_inputs: DestToGraphEdges):
        if self.name not in node_to_inputs:
            return []

        ingested = {
            edge.name: edge for edge in node_to_inputs[self.name] \
                if edge.name in self.input_ids and edge.name not in self.ready_inputs
        } # ingest ids that this node takes in, and are not already ready
        self.ready_inputs.update(ingested)

        # remove the ingested ids from the node_to_inputs
        node_to_inputs[self.name] = [
            edge for edge in node_to_inputs[self.name] \
                if edge.name not in ingested
        ]
        if len(node_to_inputs[self.name]) == 0:
            del node_to_inputs[self.name]
        logger.debug(
            "Node %s ingesting inputs %s", self.name, list(ingested.values())
        )
        return list(ingested.values())

    def split_off_ready(self):
        if self.is_ready():
            return [self], None
        return [], self


@dataclass
class Sequential(GraphSection):
    sections: list[GraphSection]

    def get_node_names(self) -> list[str]:
        return sum([
            s.get_node_names() for s in self.sections
        ], start=[])

    def _get_inputs_outputs(self):
        # In the case that this section is part of a loop, "loop-back"
        # variables are included in the list of inputs and outputs
        inputs: list[GraphEdge] = []
        outputs: list[GraphEdge] = []
        output_names = set()
        for s in self.sections:
            node_names = s.get_node_names()
            inputs_of_s = s.get_inputs()

            # filter out internal signals from the new inputs
            inputs += ([inp for inp in inputs_of_s if inp.name not in output_names])
            # filter internal output signals from the output list
            outputs = [edge for edge in outputs if not (
                edge.name not in inputs_of_s and edge.next_node in node_names
            )]

            # add new outputs to the output list
            outputs_of_s = s.get_outputs()
            outputs += outputs_of_s
            output_names.update([edge.name for edge in outputs_of_s])
        return inputs, outputs

    def get_inputs(self) -> list[GraphEdge]:
        return self._get_inputs_outputs()[0]

    def get_outputs(self) -> list[GraphEdge]:
        return self._get_inputs_outputs()[1]

    def ingest_inputs(self, node_to_inputs: DestToGraphEdges):
        ingested = []
        for s in self.sections:
            ingested += s.ingest_inputs(node_to_inputs)
        return ingested

    def split_off_ready(self):
        first_ready, first_waiting = self.sections[0].split_off_ready()

        if first_waiting:
            waiting = [first_waiting] + self.sections[1:]
        else:
            waiting = self.sections[1:]

        if len(waiting) == 0:
            return first_ready, None
        return first_ready, Sequential(sections=waiting)


@dataclass
class Parallel(GraphSection):
    sections: list[GraphSection]

    def get_node_names(self) -> list[str]:
        return sum([
            s.get_node_names() for s in self.sections
        ], start=[])

    def get_inputs(self):
        return sum(
            [s.get_inputs() for s in self.sections], start=[]
        )

    def get_outputs(self):
        return sum(
            [s.get_outputs() for s in self.sections], start=[]
        )

    def ingest_inputs(self, node_to_inputs: DestToGraphEdges):
        ingested = []
        for s in self.sections:
            ingested += s.ingest_inputs(node_to_inputs)
        return ingested

    def split_off_ready(self):
        ready = []
        waiting = []
        for node in self.sections:
            node_ready, node_waiting = node.split_off_ready()
            ready += node_ready
            if node_waiting is not None:
                waiting.append(node_waiting)

        if len(waiting) == 0:
            return ready, None
        return ready, Parallel(sections=waiting)


@dataclass
class Loop(GraphSection):
    section: GraphSection # this is used to populate next_section and
                          # in-progress section; it remains "clean"
    n_iters: int
    outputs: list[GraphEdge]
    curr_iter: int = field(default=0)
    external_inputs: list[GraphEdge] = field(default=None)
    loop_back_signals: list[GraphEdge] = field(default=None)
    curr_iter_section: GraphSection = field(default=None)
    nxt_iter_section: GraphSection = field(default=None)

    def get_outputs(self):
        return self.outputs

    def get_inputs(self):
        return self.section.get_inputs()

    def get_node_names(self):
        return self.section.get_node_names()

    def ingest_inputs(self, node_to_inputs: DestToGraphEdges):
        # Populate the current iteration first, then populate the next iteration
        # if there are any leftover inputs (which would signal either inputs that
        # are not for this section, or loop-back inputs)
        ingested: list[GraphEdge] = []
        if self.curr_iter_section is not None:
            ingested += self.curr_iter_section.ingest_inputs(node_to_inputs)

        # we should only be populating the nxt_iter_section with loop-back inputs,
        # so exclude external inputs from populating nxt_iter_section. This logic
        # is required to make nested loops work.
        my_external_inputs = {
            (edge.name, edge.next_node): edge for edge in self.external_inputs
        }
        external_inputs = {
            dest: [
                edge for edge in inputs if (edge.name, dest) in my_external_inputs
            ] for dest, inputs in node_to_inputs.items()
        }
        for dest in node_to_inputs:
            node_to_inputs[dest] = [
                i for i in node_to_inputs[dest] \
                if i not in external_inputs[dest]
            ]

        ingested += self.nxt_iter_section.ingest_inputs(node_to_inputs)
        update_list_dicts(node_to_inputs, external_inputs)

        if self.curr_iter != self.n_iters - 1:
            for graph_edge in ingested:
                if (graph_edge.name, graph_edge.next_node) in my_external_inputs:
                    my_external_inputs[(
                        graph_edge.name, graph_edge.next_node
                    )].tensor_info = graph_edge.tensor_info
                    graph_edge._persist_for_loop = True
        return ingested

    def _get_external_inputs(self):
        inputs = self.section.get_inputs()
        internal_outputs = self.section.get_outputs()
        output_names_dests = set([(edge.name, edge.next_node) for edge in internal_outputs])

        # compute "external inputs", i.e., ones that don't come from looping
        # back, and make sure those are populated for the next loop iter
        return [
            inp for inp in inputs if (inp.name, inp.next_node) not in output_names_dests
        ]

    def _get_loop_back_signals(self) -> list[GraphEdge]:
        # these inputs and outputs only include external and loop-back signals;
        # they do not include signals that are purely internal to the section
        inputs = self.section.get_inputs()
        input_names_dests = [
            (edge.name, edge.next_node) for edge in inputs
        ]
        outputs = self.section.get_outputs()

        return [
            edge for edge in outputs if (edge.name, edge.next_node) in input_names_dests
        ]

    def _replace_outputs_for_final_iter(
        self, section: GraphSection,
    ):
        """
        For the final iteration, we want to: (1) remove all loop-back signals
        from the graph, and (2) add in self.outputs where appropriate
        """
        loop_back_signals = self.loop_back_signals
        loop_back_name_dests = set([
            (edge.name, edge.next_node) for edge in loop_back_signals
        ])
        full_outputs = self.outputs

        def replace_outputs(node_outputs: list[GraphEdge]):
            node_output_names = set([
                edge.name for edge in node_outputs
            ])
            outputs_to_add = [
                edge for edge in full_outputs if edge.name in node_output_names
            ]

            return outputs_to_add + [
                edge for edge in node_outputs \
                    if (edge.name, edge.next_node) not in loop_back_name_dests
            ]
        if isinstance(section, GraphNode) or isinstance(section, Loop):
            section.outputs = replace_outputs(section.outputs)
        elif isinstance(section, Sequential) or isinstance(section, Parallel):
            for sec in section.sections:
                self._replace_outputs_for_final_iter(sec)

    def __post_init__(self):
        if self.curr_iter_section is None:
            self.curr_iter_section = deepcopy(self.section)
        if self.nxt_iter_section is None:
            self.nxt_iter_section = deepcopy(self.section)
        if self.external_inputs is None:
            self.external_inputs = self._get_external_inputs()
        if self.loop_back_signals is None:
            self.loop_back_signals = self._get_loop_back_signals()

        if self.n_iters == self.curr_iter + 1:
            self._replace_outputs_for_final_iter(self.curr_iter_section)

    def _advance_one_iter(self) -> "Loop":
        curr_iter_section = self.nxt_iter_section
        nxt_iter_section = deepcopy(self.section)

        logger.info(
            "Advancing loop with nodes %s from iter %d -> %d (out of %d)",
            str(self.section.get_node_names()), self.curr_iter,
            self.curr_iter + 1, self.n_iters
        )

        loop = Loop(
            section=self.section,
            curr_iter_section=curr_iter_section,
            nxt_iter_section=nxt_iter_section,
            curr_iter=self.curr_iter + 1,
            n_iters=self.n_iters,
            outputs=self.outputs,
            external_inputs=self.external_inputs,
            loop_back_signals=self.loop_back_signals
        )
        loop.ingest_inputs(get_node_to_inputs_mapping(
            self.external_inputs
        ))
        return loop

    def split_off_ready(self):
        loop = self if self.curr_iter_section is not None \
            else self._advance_one_iter()
        first_ready, first_waiting = loop.curr_iter_section.split_off_ready()
        loop.curr_iter_section = first_waiting

        if loop.n_iters == loop.curr_iter + 1: # last iteration
            return first_ready, first_waiting

        loop.curr_iter_section = first_waiting
        return first_ready, loop
