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
    uuid: str # for all the cleanups, list[tensor] indexing
    source_session_id: str # e.g., f"{HOSTNAME}:{client_engine.get_rpc_port()}"
    source_entity: str # which {worker, api_server} the tensor is on

# class ConnectionType(Enum):
#     RELAY = "relay" # thinker-talker
#     BLOCKING = "blocking" # we need to wait for all tensors in the list to finish

# assume blocking case for all stages for now.

@dataclass
class GraphPointer:
    next_stage: str
    # connection_type: ConnectionType
    # wait_for_next_tensor : bool = True
    name: str
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)

    # Flags
    back_to_conductor: bool = field(default=False)
    is_new_token: bool = field(default=False)
    # only for STREAM_OUT
    output_modality: str = field(default="") # text | image | video | audio
    _persist_for_loop: bool = field(default=False)

# Two different ways of defining graph edges
DestToGraphPointers = dict[str, list[GraphPointer]]

def get_stage_to_inputs_mapping(
    new_inputs: list[GraphPointer]
)-> DestToGraphPointers:
    result: DestToGraphPointers = {}
    for ptr in new_inputs:
        if ptr.next_stage not in result:
            result[ptr.next_stage] = []
        result[ptr.next_stage].append(ptr)
    return result


@dataclass
class GraphSection(ABC):
    @abstractmethod
    def get_stage_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_inputs(self) -> list[GraphPointer]:
        """
        All external or "loop-back" inputs into a subgraph
        """
        pass

    @abstractmethod
    def get_outputs(self) -> list[GraphPointer]:
        """
        All external or "loop-back" outputs from a subgraph
        """
        pass

    @abstractmethod
    def ingest_inputs(self, stage_to_inputs: DestToGraphPointers) -> list[GraphPointer]:
        """
        Adds inputs to the appropriate "ready_input_ids", and **mutates
        stage_to_inputs** to remove the inputs that were able to be added
        to the "ready_input_ids".
        """
        pass

    @abstractmethod
    def split_off_ready(self) -> tuple[list["GraphStage"], "GraphSection"]:
        """
        Returns a list of stages that are ready to be run, and a graph section
        of what is still waiting.
        """
        pass


@dataclass
class GraphStage(GraphSection):
    name: str
    input_ids: set[str]
    outputs: list[GraphPointer]
    consumes_stream: bool = field(default=False)

    # Populated as previous stages complete
    # This will also include, e.g., tensor UUIDs associated with these inputs
    ready_inputs: dict[str, GraphPointer] = field(default_factory=dict) # name -> graph edge

    def __post_init__(self):
        # if the user inputs, e.g., a list, turn it into a set
        self.input_ids = set(self.input_ids)

    def is_ready(self):
        return self.input_ids.issubset(set(self.ready_inputs.keys()))

    def get_stage_names(self) -> list[str]:
        return [self.name]

    def get_inputs(self) -> list[GraphPointer]:
        return [
            GraphPointer(next_stage=self.name, name=id) for id in self.input_ids
        ]

    def get_outputs(self) -> list[GraphPointer]:
        return self.outputs

    def ingest_inputs(self, stage_to_inputs: DestToGraphPointers):
        if self.name not in stage_to_inputs:
            return []

        ingested = {
            ptr.name: ptr for ptr in stage_to_inputs[self.name] \
                if ptr.name in self.input_ids and ptr.name not in self.ready_inputs
        } # ingest ids that this stage takes in, and are not already ready
        self.ready_inputs.update(ingested)

        # remove the ingested ids from the stage_to_inputs
        stage_to_inputs[self.name] = [
            ptr for ptr in stage_to_inputs[self.name] \
                if ptr.name not in ingested
        ]
        if len(stage_to_inputs[self.name]) == 0:
            del stage_to_inputs[self.name]
        logger.debug(
            "Stage %s ingesting inputs %s", self.name, list(ingested.values())
        )
        return list(ingested.values())

    def split_off_ready(self):
        if self.is_ready():
            return [self], None
        return [], self


@dataclass
class Sequential(GraphSection):
    sections: list[GraphSection]

    def get_stage_names(self) -> list[str]:
        return sum([
            s.get_stage_names() for s in self.sections
        ], start=[])

    def _get_inputs_outputs(self):
        # In the case that this section is part of a loop, "loop-back"
        # variables are included in the list of inputs and outputs
        inputs: list[GraphPointer] = []
        outputs: list[GraphPointer] = []
        output_names = set()
        for s in self.sections:
            stage_names = s.get_stage_names()
            inputs_of_s = s.get_inputs()

            # filter out internal signals from the new inputs
            inputs += ([inp for inp in inputs_of_s if inp.name not in output_names])
            # filter internal output signals from the output list
            outputs = [ptr for ptr in outputs if not (
                ptr.name not in inputs_of_s and ptr.next_stage in stage_names
            )]

            # add new outputs to the output list
            outputs_of_s = s.get_outputs()
            outputs += outputs_of_s
            output_names.update([ptr.name for ptr in outputs_of_s])
        return inputs, outputs

    def get_inputs(self) -> list[GraphPointer]:
        return self._get_inputs_outputs()[0]

    def get_outputs(self) -> list[GraphPointer]:
        return self._get_inputs_outputs()[1]

    def ingest_inputs(self, stage_to_inputs: DestToGraphPointers):
        ingested = []
        for s in self.sections:
            ingested += s.ingest_inputs(stage_to_inputs)
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

    def get_stage_names(self) -> list[str]:
        return sum([
            s.get_stage_names() for s in self.sections
        ], start=[])

    def get_inputs(self):
        return sum(
            [s.get_inputs() for s in self.sections], start=[]
        )

    def get_outputs(self):
        return sum(
            [s.get_outputs() for s in self.sections], start=[]
        )

    def ingest_inputs(self, stage_to_inputs: DestToGraphPointers):
        ingested = []
        for s in self.sections:
            ingested += s.ingest_inputs(stage_to_inputs)
        return ingested

    def split_off_ready(self):
        ready = []
        waiting = []
        for stage in self.sections:
            stage_ready, stage_waiting = stage.split_off_ready()
            ready += stage_ready
            if stage_waiting is not None:
                waiting.append(stage_waiting)

        if len(waiting) == 0:
            return ready, None
        return ready, Parallel(sections=waiting)


@dataclass
class Loop(GraphSection):
    section: GraphSection # this is used to populate next_section and
                          # in-progress section; it remains "clean"
    n_iters: int
    outputs: list[GraphPointer]
    curr_iter: int = field(default=0)
    external_inputs: list[GraphPointer] = field(default=None)
    loop_back_signals: list[GraphPointer] = field(default=None)
    curr_iter_section: GraphSection = field(default=None)
    nxt_iter_section: GraphSection = field(default=None)

    def get_outputs(self):
        return self.outputs

    def get_inputs(self):
        return self.section.get_inputs()

    def get_stage_names(self):
        return self.section.get_stage_names()

    def ingest_inputs(self, stage_to_inputs: DestToGraphPointers):
        # Populate the current iteration first, then populate the next iteration
        # if there are any leftover inputs (which would signal either inputs that
        # are not for this section, or loop-back inputs)
        ingested: list[GraphPointer] = []
        if self.curr_iter_section is not None:
            ingested += self.curr_iter_section.ingest_inputs(stage_to_inputs)

        # we should only be populating the nxt_iter_section with loop-back inputs,
        # so exclude external inputs from populating nxt_iter_section. This logic
        # is required to make nested loops work.
        my_external_inputs = {
            (ptr.name, ptr.next_stage): ptr for ptr in self.external_inputs
        }
        external_inputs = {
            dest: [
                ptr for ptr in inputs if (ptr.name, dest) in my_external_inputs
            ] for dest, inputs in stage_to_inputs.items()
        }
        for dest in stage_to_inputs:
            stage_to_inputs[dest] = [
                i for i in stage_to_inputs[dest] \
                    if i not in external_inputs[dest]
            ]

        ingested += self.nxt_iter_section.ingest_inputs(stage_to_inputs)
        update_list_dicts(stage_to_inputs, external_inputs)

        if self.curr_iter != self.n_iters - 1:
            for input in ingested:
                if (input.name, input.next_stage) in my_external_inputs:
                    my_external_inputs[(
                        input.name, input.next_stage
                    )].tensor_info = input.tensor_info
                    input._persist_for_loop = True
        return ingested

    def _get_external_inputs(self):
        inputs = self.section.get_inputs()
        internal_outputs = self.section.get_outputs()
        output_names_dests = set([(ptr.name, ptr.next_stage) for ptr in internal_outputs])

        # compute "external inputs", i.e., ones that don't come from looping
        # back, and make sure those are populated for the next loop iter
        return [
            inp for inp in inputs if (inp.name, inp.next_stage) not in output_names_dests
        ]

    def _get_loop_back_signals(self) -> list[GraphPointer]:
        # these inputs and outputs only include external and loop-back signals;
        # they do not include signals that are purely internal to the section
        inputs = self.section.get_inputs()
        input_names_dests = [
            (ptr.name, ptr.next_stage) for ptr in inputs
        ]
        outputs = self.section.get_outputs()

        return [
            ptr for ptr in outputs if (ptr.name, ptr.next_stage) in input_names_dests
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
            (ptr.name, ptr.next_stage) for ptr in loop_back_signals
        ])
        full_outputs = self.outputs

        def replace_outputs(stage_outputs: list[GraphPointer]):
            stage_output_names = set([
                ptr.name for ptr in stage_outputs
            ])
            outputs_to_add = [
                ptr for ptr in full_outputs if ptr.name in stage_output_names
            ]

            return outputs_to_add + [
                ptr for ptr in stage_outputs \
                    if (ptr.name, ptr.next_stage) not in loop_back_name_dests
            ]
        if isinstance(section, GraphStage) or isinstance(section, Loop):
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

        logger.debug(
            "Advancing loop with stages %s from iter %d -> %d (out of %d)",
            str(self.section.get_stage_names()), self.curr_iter,
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
        loop.ingest_inputs(get_stage_to_inputs_mapping(
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
