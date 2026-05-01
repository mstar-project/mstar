import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class FilteredEdges:
    kept: list["GraphEdge"]
    filtered_out: list["GraphEdge"]


@dataclass
class LoopCompletionOutput:
    new_waiting: "GraphSection"
    outputs: list["GraphEdge"] = field(default_factory=list)
    loop_back_name_dests_to_remove: set[tuple[str, str]] = field(default_factory=set)

    def filter_out_loop_back(self, edges: list["GraphEdge"]) -> FilteredEdges:
        return FilteredEdges(
            kept=[edge for edge in edges if (edge.name, edge.next_node) not in self.loop_back_name_dests_to_remove],
            filtered_out=[edge for edge in edges if (edge.name, edge.next_node) in self.loop_back_name_dests_to_remove],
        )


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
    uuid: str  # for indexing storage
    source_session_id: str  # "{HOSTNAME}:{client_engine.get_rpc_port()}"
    source_entity: str  # which {worker, api_server} the tensor is on


@dataclass
class GraphEdge:
    next_node: str
    name: str
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)

    # Flags
    persist: bool = field(default=False)  # previously back_to_conductor
    conductor_new_token: bool = field(default=False)
    is_streaming: bool = field(default=False)  # streaming edge: tokens accumulate at destination buffer
    # only for EMIT_TO_CLIENT
    output_modality: str = field(default="")  # text | image | video | audio
    _persist_for_loop: bool = field(default=False)

    def clone_for_next_iter(self) -> "GraphEdge":
        """Fresh copy with empty tensor_info; same routing/flags. Preserves
        subclass type (e.g. StreamingGraphEdge). Used by the worker's async-
        scheduling path to build the next loop-iter's GraphNode without
        sharing output edge state with the in-flight step."""
        import dataclasses
        return dataclasses.replace(self, tensor_info=[])


# Two different ways of defining graph edges
DestToGraphEdges = dict[str, list[GraphEdge]]


def get_node_to_inputs_mapping(new_inputs: list[GraphEdge]) -> DestToGraphEdges:
    result: DestToGraphEdges = {}
    for edge in new_inputs:
        if edge.next_node not in result:
            result[edge.next_node] = []
        result[edge.next_node].append(edge)
    return result


@dataclass
class GraphSection(ABC):
    @abstractmethod
    def get_node_names(self) -> set[str]:
        pass

    @abstractmethod
    def get_dyn_loop_names(self) -> set[str]:
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

    @abstractmethod
    def split_off_ready_for_streaming(self) -> list["GraphNode"]:
        pass
    
    @abstractmethod
    def split_off_for_spec(self, spec_node_name: str) -> tuple[bool, "GraphSection | None"]:
        """
        Returns whether a node was split off, and the new waiting
        """
        pass

    @abstractmethod
    def cache_outputs(
        self,
        tensor_info: dict[str, list[TensorPointerInfo]],
    ):
        pass

    @abstractmethod
    def complete_loops(self, done_node: str) -> LoopCompletionOutput:
        """
        Checks if any loops have completed, and returns their cached output edges,
        if any
        """
        pass

    @abstractmethod
    def register_communication_info(self, communication_manager, request_id: str):
        pass

    @abstractmethod
    def reset(self):
        pass


@dataclass
class GraphNode(GraphSection):
    name: str
    input_ids: set[str]
    outputs: list[GraphEdge]
    # Names of inputs the node will accept if provided, but that do NOT
    # block readiness when absent.  Use for optional kwargs to the
    # underlying submodule (e.g., V-JEPA 2's ``context_mask`` /
    # ``target_mask``, which have sensible defaults built inside the
    # submodule).  Must be disjoint from ``input_ids``.
    optional_input_ids: set[str] = field(default_factory=set)
    consumes_stream: bool = field(default=False)
    enable_async_scheduling: bool = True
    _streaming_inputs: set[str] = field(default_factory=set)
    _split_off_for_streaming: bool = field(default=False)

    # Populated as previous nodes complete
    # This will also include, e.g., tensor UUIDs associated with these inputs
    ready_inputs: dict[str, GraphEdge] = field(default_factory=dict)  # name -> graph edge

    def __post_init__(self):
        # if the user inputs, e.g., a list, turn it into a set
        self.input_ids = set(self.input_ids)
        self.optional_input_ids = set(self.optional_input_ids)
        _overlap = self.input_ids & self.optional_input_ids
        if _overlap:
            raise ValueError(
                f"GraphNode.input_ids and GraphNode.optional_input_ids must be disjoint; overlap: {sorted(_overlap)}"
            )

    def is_ready(self):
        # Only required inputs gate readiness — optional ones are accepted
        # opportunistically in ingest_inputs and ignored when absent.
        return self.input_ids.issubset(set(self.ready_inputs.keys()))

    def is_ready_except_streaming(self):
        return self.input_ids.issubset(set(self.ready_inputs.keys()).union(self._streaming_inputs))

    def get_node_names(self) -> set[str]:
        return {self.name}

    def get_dyn_loop_names(self) -> set[str]:
        return set()

    def get_inputs(self) -> list[GraphEdge]:
        # Required + optional — the graph dispatcher routes any matching
        # edge here; ``is_ready`` still gates only on required names.
        all_ids = self.input_ids | self.optional_input_ids
        return [GraphEdge(next_node=self.name, name=id) for id in all_ids]

    def get_outputs(self) -> list[GraphEdge]:
        return self.outputs

    def ingest_inputs(self, node_to_inputs: DestToGraphEdges):
        if self.name not in node_to_inputs:
            return []

        accept = self.input_ids | self.optional_input_ids
        ingested = {
            edge.name: edge
            for edge in node_to_inputs[self.name]
            if edge.name in accept and edge.name not in self.ready_inputs
        }  # ingest required + optional inputs this node accepts
        self.ready_inputs.update(ingested)

        # remove the ingested ids from the node_to_inputs
        node_to_inputs[self.name] = [edge for edge in node_to_inputs[self.name] if edge.name not in ingested]
        if len(node_to_inputs[self.name]) == 0:
            del node_to_inputs[self.name]
        logger.debug("Node %s ingesting inputs %s", self.name, list(ingested.values()))
        return list(ingested.values())

    def split_off_ready(self):
        if self.is_ready():
            return [self], None
        return [], self

    def split_off_for_spec(self, spec_node_name: str) -> tuple[bool, "GraphSection | None"]:
        if self.name == spec_node_name:
            return True, None
        return False, self

    def split_off_ready_for_streaming(self):
        if self._split_off_for_streaming:
            return []
        if self.is_ready_except_streaming():
            self._split_off_for_streaming = True
            return [self]
        return []

    def cache_outputs(
        self,
        tensor_info: dict[str, list[TensorPointerInfo]],
    ):
        return

    def complete_loops(self, done_node) -> LoopCompletionOutput:
        return LoopCompletionOutput(self)

    def register_communication_info(self, communication_manager, request_id: str):
        return

    def reset(self):
        self.ready_inputs.clear()
        self._split_off_for_streaming = False

    def clear_outputs(self):
        for edge in self.outputs:
            edge.tensor_info.clear()

    def clone_for_next_iter(self) -> "GraphNode":
        """Fresh GraphNode for the next loop iter — same shape, no
        ready_inputs, fresh output edges (with empty tensor_info). Used by
        the worker's async-scheduling path to speculatively build batch_N+1
        while batch_N is still in flight on the GPU thread."""
        clone = GraphNode(
            name=self.name,
            input_ids=set(self.input_ids),
            outputs=[edge.clone_for_next_iter() for edge in self.outputs],
            optional_input_ids=set(self.optional_input_ids),
            consumes_stream=self.consumes_stream,
            enable_async_scheduling=self.enable_async_scheduling,
        )
        clone._streaming_inputs = set(self._streaming_inputs)
        return clone


@dataclass
class Sequential(GraphSection):
    sections: list[GraphSection]

    def get_node_names(self) -> set[str]:
        res = set()
        for s in self.sections:
            res.update(s.get_node_names())
        return res

    def get_dyn_loop_names(self) -> set[str]:
        res = set()
        for s in self.sections:
            res.update(s.get_dyn_loop_names())
        return res

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
            inputs += [inp for inp in inputs_of_s if inp.name not in output_names]
            # filter internal output signals from the output list
            outputs = [edge for edge in outputs if not (edge.name not in inputs_of_s and edge.next_node in node_names)]

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

    def split_off_ready_for_streaming(self):
        return self.sections[0].split_off_ready_for_streaming()
    
    def split_off_for_spec(self, spec_node_name: str):
        split, first_waiting = self.sections[0].split_off_for_spec(spec_node_name)
        if first_waiting:
            waiting = [first_waiting] + self.sections[1:]
        else:
            waiting = self.sections[1:]

        if len(waiting) == 0:
            return split, None
        return split, Sequential(sections=waiting)

    def cache_outputs(
        self,
        tensor_info: dict[str, list[TensorPointerInfo]],
    ):
        for sec in self.sections:
            sec.cache_outputs(tensor_info)

    def complete_loops(self, done_node) -> LoopCompletionOutput:
        output = self.sections[0].complete_loops(done_node)
        if output.new_waiting is None:
            waiting = self.sections[1:]
        else:
            waiting = [output.new_waiting] + self.sections[1:]

        if len(waiting) > 0:
            output.new_waiting = Sequential(sections=waiting)
        return output

    def register_communication_info(self, communication_manager, request_id: str):
        for sec in self.sections:
            sec.register_communication_info(communication_manager, request_id)

    def reset(self):
        for sec in self.sections:
            sec.reset()


@dataclass
class Parallel(GraphSection):
    sections: list[GraphSection]

    def get_node_names(self) -> set[str]:
        res = set()
        for s in self.sections:
            res.update(s.get_node_names())
        return res

    def get_dyn_loop_names(self) -> set[str]:
        res = set()
        for s in self.sections:
            res.update(s.get_dyn_loop_names())
        return res

    def get_inputs(self):
        return sum([s.get_inputs() for s in self.sections], start=[])

    def get_outputs(self):
        return sum([s.get_outputs() for s in self.sections], start=[])

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

    def split_off_ready_for_streaming(self):
        return sum([sec.split_off_ready_for_streaming() for sec in self.sections], start=[])
    
    def split_off_for_spec(self, spec_node_name: str):
        sections = []
        any_split = split
        for section in self.sections:
            split, new_waiting = section.split_off_for_spec(spec_node_name)
            any_split &= split
            sections.append(new_waiting)
        if not sections:
            return any_split, None
        return any_split, Parallel(sections)

    def cache_outputs(
        self,
        tensor_info: dict[str, list[TensorPointerInfo]],
    ):
        for sec in self.sections:
            sec.cache_outputs(tensor_info)

    def complete_loops(self, done_node) -> LoopCompletionOutput:
        outputs = []
        loop_back_name_dests = set()
        waiting = []
        for sec in self.sections:
            out = sec.complete_loops(done_node)
            outputs += out.outputs
            loop_back_name_dests.update(out.loop_back_name_dests_to_remove)
            if out.new_waiting is not None:
                waiting.append(out.new_waiting)

        if len(waiting) > 0:
            return LoopCompletionOutput(
                new_waiting=Parallel(sections=waiting),
                outputs=outputs,
                loop_back_name_dests_to_remove=loop_back_name_dests,
            )
        return LoopCompletionOutput(
            new_waiting=None, outputs=outputs, loop_back_name_dests_to_remove=loop_back_name_dests
        )

    def register_communication_info(self, communication_manager, request_id: str):
        for sec in self.sections:
            sec.register_communication_info(communication_manager, request_id)

    def reset(self):
        for sec in self.sections:
            sec.reset()


@dataclass
class Loop(GraphSection):
    # "section" is used by users to specify the loop iterior.It is also
    # used in the current copy-free "advance iter" logic; see "_advance_one_iter"
    # for more information
    section: GraphSection
    max_iters: int
    outputs: list[GraphEdge]
    # Per-iteration outputs: every iter's tensor_info is appended to an
    # internal accumulator (``_accumulated_cache``) and the full list is
    # emitted on loop completion.  Unlike ``outputs`` — whose cache is
    # wiped by ``_advance_one_iter`` so only the last iter's tensor_info
    # survives — this cache is preserved across iterations.  Used for
    # walks that want to stream or gather every iteration's output
    # (e.g., autoregressive rollout that emits H predictions).
    # Must be disjoint from ``outputs`` by name.
    accumulated_outputs: list[GraphEdge] = field(default_factory=list)
    curr_iter: int = field(default=0, repr=False)
    _external_inputs: list[GraphEdge] = field(default=None, repr=False)
    _loop_back_signals: list[GraphEdge] = field(default=None, repr=False)
    _curr_iter_section: GraphSection = field(default=None, repr=False)
    _nxt_iter_section: GraphSection = field(default=None, repr=False)
    _cached_outputs: dict[str, list[TensorPointerInfo]] = field(default_factory=dict, repr=False)
    _accumulated_cache: dict[str, list[TensorPointerInfo]] = field(default_factory=dict, repr=False)
    _output_names: set[str] = field(default_factory=set, repr=False)
    _accumulated_output_names: set[str] = field(default_factory=set, repr=False)
    _uuid_label: str = field(default_factory=lambda: str(uuid4()), repr=False)

    # For handling tensor reference counting of loop outputs
    _tensor_manager: Any | None = field(default=None, repr=False)  # no type annotation because
    # of circular imports
    _request_id: str | None = field(default=None, repr=False)
    _waiting_for_execution: set[str] = field(default_factory=set, repr=False)

    def get_outputs(self):
        return self.outputs

    def get_inputs(self):
        return self.section.get_inputs()

    def get_node_names(self):
        return self.section.get_node_names()

    def get_dyn_loop_names(self) -> set[str]:
        return self.section.get_dyn_loop_names()

    def register_communication_info(self, communication_manager, request_id: str):
        self._tensor_manager = communication_manager
        self._request_id = request_id
        self.section.register_communication_info(communication_manager, request_id)

    def cache_outputs(
        self,
        tensor_info: dict[str, list[TensorPointerInfo]],
    ):
        # Terminal outputs: cache gets wiped at each ``_advance_one_iter`` so
        # only the last iter's tensor_info survives to completion.
        names = [key for key in tensor_info if key in self._output_names]
        for out_name in names:
            tensor_infos = tensor_info[out_name]
            for info in tensor_infos:
                self._tensor_manager.increment_ref(self._request_id, info.uuid)
            self._cached_outputs.setdefault(out_name, []).extend(tensor_infos)

        # Per-iteration accumulated outputs: cache survives across iterations
        # (cleared only on loop completion via ``reset``), so every iter's
        # tensor_info is preserved and emitted as a single list at the end.
        acc_names = [key for key in tensor_info if key in self._accumulated_output_names]
        for out_name in acc_names:
            tensor_infos = tensor_info[out_name]
            for info in tensor_infos:
                self._tensor_manager.increment_ref(self._request_id, info.uuid)
            self._accumulated_cache.setdefault(out_name, []).extend(tensor_infos)

        if self._curr_iter_section is not None:
            self._curr_iter_section.cache_outputs(tensor_info)

    def _uncache_outputs(self):
        for tensor_infos in self._cached_outputs.values():
            for info in tensor_infos:
                self._tensor_manager.dereference(self._request_id, info.uuid)
        self._cached_outputs.clear()

    def _uncache_accumulated_outputs(self):
        """Dereference and clear the per-iteration accumulated cache.

        Called from ``reset`` only — NOT from ``_advance_one_iter`` — because
        the whole point of ``accumulated_outputs`` is to survive the inter-
        iteration cleanup that ``_uncache_outputs`` performs.  Guarded against
        a missing ``_tensor_manager`` so ``reset`` works on freshly built
        Loop copies before ``register_communication_info`` has run.
        """
        if self._tensor_manager is not None and self._request_id is not None:
            for tensor_infos in self._accumulated_cache.values():
                for info in tensor_infos:
                    self._tensor_manager.dereference(self._request_id, info.uuid)
        self._accumulated_cache.clear()

    def ingest_inputs(self, node_to_inputs: DestToGraphEdges):
        # Populate the current iteration first, then populate the next iteration
        # if there are any leftover inputs (which would signal either inputs that
        # are not for this section, or loop-back inputs)
        ingested: list[GraphEdge] = []
        if self._curr_iter_section is not None:
            ingested += self._curr_iter_section.ingest_inputs(node_to_inputs)

        # we should only be populating the nxt_iter_section with loop-back inputs,
        # so exclude external inputs from populating nxt_iter_section. This logic
        # is required to make nested loops work.
        my_external_inputs = {(edge.name, edge.next_node): edge for edge in self._external_inputs}
        external_inputs = {
            dest: [edge for edge in inputs if (edge.name, dest) in my_external_inputs]
            for dest, inputs in node_to_inputs.items()
        }
        for dest in node_to_inputs:
            node_to_inputs[dest] = [i for i in node_to_inputs[dest] if i not in external_inputs[dest]]

        ingested += self._nxt_iter_section.ingest_inputs(node_to_inputs)
        update_list_dicts(node_to_inputs, external_inputs)

        if self.curr_iter != self.max_iters - 1:
            for graph_edge in ingested:
                if (graph_edge.name, graph_edge.next_node) in my_external_inputs:
                    my_external_inputs[(graph_edge.name, graph_edge.next_node)].tensor_info = graph_edge.tensor_info
                    graph_edge._persist_for_loop = True
        return ingested

    def _get_external_inputs(self):
        inputs = self.section.get_inputs()
        internal_outputs = self.section.get_outputs()
        output_names_dests = set([(edge.name, edge.next_node) for edge in internal_outputs])

        # compute "external inputs", i.e., ones that don't come from looping
        # back, and make sure those are populated for the next loop iter
        return [inp for inp in inputs if (inp.name, inp.next_node) not in output_names_dests]

    def _get_loop_back_signals(self) -> list[GraphEdge]:
        # these inputs and outputs only include external and loop-back signals;
        # they do not include signals that are purely internal to the section
        inputs = self.section.get_inputs()
        input_names_dests = [(edge.name, edge.next_node) for edge in inputs]
        outputs = self.section.get_outputs()

        return [edge for edge in outputs if (edge.name, edge.next_node) in input_names_dests]

    def __post_init__(self):
        # In the disaggregated case, we need filter self.outputs for outputs
        # that this subgraph actually produces
        outputs_we_produce = set([edge.name for edge in self.section.get_outputs()])
        self.outputs = [edge for edge in self.outputs if edge.name in outputs_we_produce]
        self.accumulated_outputs = [edge for edge in self.accumulated_outputs if edge.name in outputs_we_produce]

        self._output_names = set([edge.name for edge in self.outputs])
        self._accumulated_output_names = set([edge.name for edge in self.accumulated_outputs])
        _overlap = self._output_names & self._accumulated_output_names
        if _overlap:
            raise ValueError(
                f"Loop.outputs and Loop.accumulated_outputs must be disjoint by name; overlap: {sorted(_overlap)}"
            )

        if self._curr_iter_section is None:
            self._curr_iter_section = self.section
        if self._nxt_iter_section is None:
            self._nxt_iter_section = deepcopy(self.section)
        if self._external_inputs is None:
            self._external_inputs = self._get_external_inputs()
        if self._loop_back_signals is None:
            self._loop_back_signals = self._get_loop_back_signals()

    def _advance_one_iter(self) -> "Loop":
        """
        Advance the iteration by essentially resetting the curr_iter_section
        and swapping it with the next_iter_section. This allows loop iteration while
        remaining copy-free. The overall logic is as follows:

        1. Loop instantiation: _curr_iter_section is set to section ("obj A"), and
        _nxt_iter_section is set to a copy ("obj B")

        2. Advance iter 0 -> 1: reset self.section (obj A). Set _curr_iter_section
        to obj B (which has possibly accumulated inputs in its tenure as
        _nxt_iter_section), and _curr_iter_section is set to the newly-reset obj B.
        We have to use self.section here because _curr_iter_section will be None before
        calling _advance_one_iter.

        3. Advance iter 1 -> 2: Same as (2), but sets _curr_iter_section to obj A,
        and sets _nxt_iter_section to a freshly-reset obj B.
        """
        self._uncache_outputs()

        # TODO: maybe if we call node.reset right after running each node in the worker,
        # so we know that all nodes in self.section have already been reset if
        # self._iter_done() returns True, we don't have to call this here. But we
        # will have to make sure to carefully reset the loop curr_iter in complete_loops(),
        # so leaving this as a comment to think about later.
        self.section.reset()

        new_curr, new_next = self._nxt_iter_section, self.section
        self._curr_iter_section = new_curr
        self._nxt_iter_section = new_next
        self.section = new_curr
        self.curr_iter += 1

        logger.info(
            "Advancing loop with nodes %s from iter %d -> %d (out of %d)",
            str(self.section.get_node_names()),
            self.curr_iter,
            self.curr_iter + 1,
            self.max_iters,
        )

        self.ingest_inputs(get_node_to_inputs_mapping(self._external_inputs))

    def _is_done(self):
        return (self.max_iters == self.curr_iter + 1) and self._iter_done()

    def _iter_done(self):
        return (self._curr_iter_section is None) and len(self._waiting_for_execution) == 0

    def split_off_ready(self):
        if self._iter_done():
            if self._is_done():
                return [], None
            self._advance_one_iter()
        elif self._curr_iter_section is None:
            return [], self
        first_ready, first_waiting = self._curr_iter_section.split_off_ready()
        self._waiting_for_execution.update([node.name for node in first_ready])
        self._curr_iter_section = first_waiting
        return first_ready, self
    
    def split_off_for_spec(self, spec_node_name):
        if self._iter_done():
            if self._is_done():
                return [], None
            self._advance_one_iter()
        elif self._curr_iter_section is None:
            return [], self
        split, new_waiting = self._curr_iter_section.split_off_for_spec(spec_node_name)
        self._curr_iter_section = new_waiting
        return split, self

    def split_off_ready_for_streaming(self):
        return (
            self._curr_iter_section.split_off_ready_for_streaming() if self._curr_iter_section is not None else []
        ) + self._nxt_iter_section.split_off_ready_for_streaming()

    def complete_loops(self, done_node: str) -> LoopCompletionOutput:
        if done_node in self._waiting_for_execution:
            self._waiting_for_execution.remove(done_node)
        output_signals = []
        loop_back_name_dests = set()

        # recursive call
        if self._curr_iter_section is not None:
            recursive_output = self._curr_iter_section.complete_loops(done_node)
            output_signals = recursive_output.outputs
            loop_back_name_dests = recursive_output.loop_back_name_dests_to_remove
            self._curr_iter_section = recursive_output.new_waiting

        # check if the loop is done after the recursive call updates _curr_iter_section
        done = self._is_done()
        if not done:
            return LoopCompletionOutput(
                new_waiting=self, outputs=output_signals, loop_back_name_dests_to_remove=loop_back_name_dests
            )

        # if done, new_waiting is None and also need to collect our outputs
        for output in self.outputs:
            if output.name in self._cached_outputs:
                output.tensor_info = self._cached_outputs[output.name]
                output_signals.append(output)

        # Emit per-iteration accumulated outputs (disjoint from self.outputs
        # by __post_init__ validation).  tensor_info here is the concatenated
        # list of every iter's contribution that was appended by cache_outputs.
        for output in self.accumulated_outputs:
            if output.name in self._accumulated_cache:
                output.tensor_info = self._accumulated_cache[output.name]
                output_signals.append(output)

        loop_back_name_dests.update([(edge.name, edge.next_node) for edge in self._loop_back_signals])

        return LoopCompletionOutput(
            new_waiting=None, outputs=output_signals, loop_back_name_dests_to_remove=loop_back_name_dests
        )        

    def reset(self):
        self.section.reset()
        self._curr_iter_section = self.section
        self._nxt_iter_section.reset()
        self.curr_iter = 0
        # ``_cached_outputs`` is wiped by ``_advance_one_iter`` between iters
        # and is expected to be empty at reset; the accumulated cache is NOT
        # wiped between iters, so we must clean it up here or a fresh run of
        # the same Loop would emit stale tensor_infos from the previous run.
        self._uncache_accumulated_outputs()


@dataclass
class DynamicLoop(Loop):
    name: str = field(default="loop")
    _finished: bool = field(default=False)

    def register_finished(self):
        self._finished = True

    def get_dyn_loop_names(self) -> set[str]:
        res = super().get_dyn_loop_names()
        res.add(self.name)
        return res

    def _is_done(self):
        return ((self.max_iters == self.curr_iter + 1) or self._finished) and self._iter_done()

    def reset(self):
        super().reset()
        self._finished = False
