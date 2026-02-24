from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field


def update_list_dicts(signals: dict[str, list], new_signals: dict[str, list]):
    for key in new_signals:
        if key in signals:
            signals[key].extend(new_signals[key])
        else:
            signals[key] = new_signals[key]


@dataclass
class GraphPointer:
    next_stage: str
    back_to_conductor: bool = field(default=False)


# Three different ways of defining graph edges
SignalToDests = dict[str, list[str]]
SignalToDestsAndFlags = dict[str, list[GraphPointer]]
DestToInputs = dict[str, list[str]]


def remove_flags(
    new_inputs: SignalToDestsAndFlags
) -> SignalToDests:
    return {
        id: [ptr.next_stage for ptr in new_inputs[id]] \
                for id in new_inputs
    }

def get_stage_to_inputs_mapping(
    new_inputs: SignalToDests
)-> DestToInputs:
    result: DestToInputs = {}
    for input_id, ptrs in new_inputs.items():
        for ptr in ptrs:
            if ptr not in result:
                result[ptr] = []
            result[ptr].append(input_id)
    return result


def get_signal_to_dest_mapping(
    dest_to_inputs: DestToInputs
) -> SignalToDests:
    result: SignalToDests = {}
    for dest, inputs in dest_to_inputs.items():
        for input in inputs:
            if input not in result:
                result[input] = []
            result[input].append(dest)
    return result

@dataclass
class GraphSection(ABC):
    @abstractmethod
    def get_stage_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_inputs(self) -> SignalToDests:
        """
        All external or "loop-back" inputs into a subgraph
        """
        pass

    @abstractmethod
    def get_outputs(self) -> SignalToDests:
        """
        All external or "loop-back" outputs from a subgraph
        """
        pass

    @abstractmethod
    def ingest_inputs(self, stage_to_inputs: DestToInputs) -> set[str]:
        """
        Adds inputs to the appropriate "ready_input_ids", and **mutates
        stage_to_inputs** to remove the inputs that were able to be added
        to the "ready_input_ids".

        Returns a set of the input tensors that were added to ready_input_ids.
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
    outputs: SignalToDestsAndFlags
    consumes_stream: bool = field(default=False)

    # populated as previous stages complete
    ready_input_ids: set[str] = field(default_factory=set)

    def __post_init__(self):
        # if the user inputs, e.g., a list, turn it into a set
        self.input_ids = set(self.input_ids)
    
    def is_ready(self):
        return self.input_ids.issubset(self.ready_input_ids)
    
    def get_stage_names(self) -> list[str]:
        return [self.name]

    def get_inputs(self) -> SignalToDests:
        return {
            inp: [self.name] for inp in self.input_ids
        }
    
    def get_outputs(self) -> SignalToDests:
        return remove_flags(self.outputs)
    
    def ingest_inputs(self, stage_to_inputs: DestToInputs):
        if self.name not in stage_to_inputs:
            return

        ingested_ids = set([
            id for id in stage_to_inputs[self.name] \
                if id in self.input_ids and id not in self.ready_input_ids
        ]) # ingest ids that this stage takes in, and are not already ready
        self.ready_input_ids.update(ingested_ids)

        if ingested_ids:
            print(f"Stage {self.name} ingesting inputs {ingested_ids}")

        # remove the ingested ids from the stage_to_inputs
        stage_to_inputs[self.name] = [
            id for id in stage_to_inputs[self.name] if id not in ingested_ids
        ]
        if len(stage_to_inputs[self.name]) == 0:
            del stage_to_inputs[self.name]
        return ingested_ids

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
        inputs: SignalToDests = {}
        outputs: SignalToDests = {}
        for s in self.sections:
            stage_names = s.get_stage_names()
            new_inputs = s.get_inputs()
            for i in new_inputs:
                # filters out the internal signals from the input list
                if i not in outputs:
                    inputs[i] = new_inputs[i]
                else:
                    outputs[i] = [ptr for ptr in outputs[i] if ptr not in stage_names]
            outputs.update(s.get_outputs())
        return inputs, outputs
    
    def get_inputs(self):
        return self._get_inputs_outputs()[0]
    
    def get_outputs(self):
        return self._get_inputs_outputs()[1]
    
    def ingest_inputs(self, stage_to_inputs: DestToInputs):
        ingested = set()
        for s in self.sections:
            ingested.update(s.ingest_inputs(stage_to_inputs))
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
        inputs = {}
        for s in self.sections:
            update_list_dicts(inputs, s.get_inputs())
        return inputs
    
    def get_outputs(self):
        outputs = {}
        for s in self.sections:
            update_list_dicts(outputs, s.get_outputs())
        return outputs
    
    def ingest_inputs(self, stage_to_inputs: DestToInputs):
        ingested = set()
        for s in self.sections:
            ingested.update(s.ingest_inputs(stage_to_inputs))
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
    outputs: SignalToDestsAndFlags
    curr_iter: int = field(default=0)
    external_inputs: SignalToDests = field(default=None)
    loop_back_signals: SignalToDests = field(default=None)
    curr_iter_section: GraphSection = field(default=None)
    nxt_iter_section: GraphSection = field(default=None)
    
    def get_outputs(self) -> SignalToDests:
        return remove_flags(self.outputs)
    
    def get_inputs(self) -> SignalToDests:
        return self.section.get_inputs()
    
    def get_stage_names(self):
        return self.section.get_stage_names()
    
    def ingest_inputs(self, stage_to_inputs: DestToInputs):
        # Populate the current iteration first, then populate the next iteration
        # if there are any leftover inputs (which would signal either inputs that
        # are not for this section, or loop-back inputs)
        ingested = set()
        if self.curr_iter_section is not None:
            ingested.update(
                self.curr_iter_section.ingest_inputs(stage_to_inputs)
            )

        # we should only be populating the nxt_iter_section with loop-back inputs,
        # so exclude external inputs from populating nxt_iter_section. This logic
        # is required to make nested loops work.
        external_inputs = {
            dest: [i for i in inputs if dest in self.external_inputs.get(i, [])] \
                for dest, inputs in stage_to_inputs.items()
        }
        for dest in stage_to_inputs:
            stage_to_inputs[dest] = [
                i for i in stage_to_inputs[dest] \
                    if i not in external_inputs[dest]
            ]

        ingested.update(
            self.nxt_iter_section.ingest_inputs(stage_to_inputs)
        )
        update_list_dicts(stage_to_inputs, external_inputs)
        return ingested
    
    def _get_external_inputs(self):
        inputs = self.section.get_inputs()
        internal_outputs = self.section.get_outputs()

        # compute "external inputs", i.e., ones that don't come from looping
        # back, and make sure those are populated for the next loop iter
        return {
            i: inputs[i] for i in inputs if i not in internal_outputs
        }

    def _get_loop_back_signals(self) -> SignalToDests:
        # these inputs and outputs only include external and loop-back signals;
        # they do not include signals that are purely internal to the section
        inputs = self.section.get_inputs()
        outputs = self.section.get_outputs()

        return {
            i: inputs[i] for i in inputs if i in outputs
        }

    def _replace_outputs_for_final_iter(
        self, section: GraphSection,
    ):
        """
        For the final iteration, we want to: (1) remove all loop-back signals
        from the graph, and (2) add in self.outputs where appropriate
        """
        loop_back_signals = self.loop_back_signals
        def replace_outputs(stage_outputs: SignalToDestsAndFlags):
            for output in stage_outputs:
                # remove loop backs
                stage_outputs[output] = [
                    pointer for pointer in stage_outputs[output] \
                        if pointer.next_stage not in loop_back_signals.get(output, [])
                ]
                if output in self.outputs:
                    # add in final output
                    stage_outputs[output].extend(self.outputs[output])
        if isinstance(section, GraphStage) or isinstance(section, Loop):
            replace_outputs(section.outputs)
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
