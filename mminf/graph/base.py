from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class NodeAndGraphWalk:
    """
    Pair of node name and graph walk, e.g., (LLM, decode) or (flow, image_gen).
    graph_walk may be None for streaming-consumer lookups where the graph walk
    is not known a priori.
    """
    node: str
    graph_walk: str | None


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
    offset: int = 0 # offset, in bytes, of the read (e.g., for in-transport sharding
                    # in tensor parallel configurations)
    source_tp_size: int = 1
    source_tp_rank: int = 0

    _source_node_name: str | None = None
    _source_graph_walk: str | None = None

    def clone(self):
        return TensorPointerInfo(
            dims=self.dims[:],
            dtype=self.dtype,
            nbytes=self.nbytes,
            address=self.address,
            stride=self.stride[:],
            uuid=self.uuid,
            source_session_id=self.source_session_id,
            source_entity=self.source_entity,
            offset=self.offset,
            source_tp_size=self.source_tp_size,
            source_tp_rank=self.source_tp_rank,
            _source_node_name=self._source_node_name,
            _source_graph_walk=self._source_graph_walk
        )


@dataclass
class GraphEdge:
    next_node: str
    name: str
    tensor_info: list[TensorPointerInfo] = field(default_factory=list)

    # Flags
    persist: bool = field(default=False)  # previously back_to_conductor
    conductor_new_token: bool = field(default=False)  # legacy; not yet wired in new system
    is_streaming: bool = field(default=False)  # streaming edge: tokens accumulate at destination buffer
    # only for EMIT_TO_CLIENT
    output_modality: str = field(default="")  # text | image | video | audio
    _persist_for_loop: bool = field(default=False)

    # Set for sharded configurations
    _total_fanin: int = 1
    _shard_dim: int | None = None

    def clone(self):
        return GraphEdge(
            next_node=self.next_node,
            name=self.name,
            tensor_info=self.tensor_info[:],
            persist=self.persist,
            conductor_new_token=self.conductor_new_token,
            is_streaming=self.is_streaming,
            output_modality=self.output_modality,
            _persist_for_loop=self._persist_for_loop
        )


NameAndDest = tuple[str, str]
@dataclass
class NodeInputsOutputs:
    ext_inputs: set[NameAndDest]
    ext_outputs: list[GraphEdge]
    loop_back: set[NameAndDest]


@dataclass
class NodeCompletionOutput:
    output_edges: list[GraphEdge] = field(default_factory=list)
    # loop-back (name, dest) pairs the caller must NOT route; the loop has finished so
    # we do not want the loop-back signals to propagate
    filtered_signals: set[NameAndDest] = field(default_factory=set)


@dataclass
class SpeculativeNodeInfo:
    node_name: str
    # True if this node received loop-back inputs speculatively, meaning it will re-run
    # in the next iteration of the same loop rather than advancing to a new section.
    is_new_loop_iter: bool
    # Name of the enclosing Loop, when the node lives inside one. ``None`` for
    # top-level nodes. Lets callers look up ``wgio.loops[loop_name]`` to inspect
    # ``curr_iter`` / ``max_iters`` / ``_finish_signal`` before deciding whether
    # to speculate.
    loop_name: str | None = None


class GraphSection(ABC):
    @abstractmethod
    def get_inputs_outputs(self) -> NodeInputsOutputs:
        """Return the I/O signature of this section.

        Construction-time only — used to derive Loop._external_inputs /
        _loop_back_inputs. Not for runtime introspection.
        """

    @abstractmethod
    def get_nodes(self) -> dict[str, "GraphNode"]:
        """Flat map of all GraphNodes in this section, keyed by name."""

    @abstractmethod
    def get_loops(self) -> dict[str, "Loop"]:
        """Flat map of all Loops in this section, keyed by name."""


@dataclass
class ReadySignals:
    """Readiness state for one 'slot' of a GraphNode.

    Each node holds two instances — ready_signals (current iteration) and
    ready_next_iter — so that loop-back inputs arriving during execution can
    be buffered without overwriting the current slot.
    """
    node_name: str
    input_names: set[str]
    streaming_inputs: set[str]

    # Populated as previous nodes complete
    # This will also include, e.g., tensor UUIDs associated with these inputs
    ready_inputs: dict[str, GraphEdge] = field(default_factory=dict)  # name -> graph edge
    ready_names: set[str] = field(default_factory=set)

    is_ready: bool = False
    is_ready_for_streaming: bool = False

    def __post_init__(self):
        self._tensor_manager = None
        self._request_id = None
    
    def register_communication_info(self, communication_manager, request_id: str):
        self._tensor_manager = communication_manager
        self._request_id =  request_id
    
    def update(self, edge: GraphEdge):
        assert edge.name in self.input_names, \
            f"Node {self.node_name} does not take input named {edge.name}"
        assert edge.name not in self.ready_inputs, \
            f"Node {self.node_name} already has ready input named {edge.name}"
        self.ready_inputs[edge.name] = edge
        self.ready_names.add(edge.name)

        if self.is_ready:
            return
        self.is_ready = self.input_names.issubset(self.ready_names)
        # ready once the only missing inputs are streaming ones (which arrive incrementally)
        self.is_ready_for_streaming = self.is_ready or \
            self.is_ready_for_streaming or (
            self.input_names.issuperset(self.ready_names.union(self.streaming_inputs))
        )
    
    def clear(self):
        if self._tensor_manager is not None:
            for edge in self.ready_inputs.values():
                if edge._persist_for_loop:
                    continue
                for info in edge.tensor_info:
                    self._tensor_manager.dereference(self._request_id, info.uuid)
        self.ready_inputs.clear()
        self.ready_names.clear()
        self.is_ready = False
        self.is_ready_for_streaming = False

    def remove(self, edge_name: str) -> "GraphEdge | None":
        """Reverse ``update(edge)`` without dereferencing tensor_info.

        Used by the worker's speculation rollback path when a streaming chunk
        was ingested for a readiness check and the rid was subsequently
        dropped. The chunk is handed back to its StreamBuffer's uningested
        cache; the StreamBuffer remains responsible for the tensor lifecycle,
        so we must not dereference here.

        Returns the removed edge, or None if no such name was present.
        """
        edge = self.ready_inputs.pop(edge_name, None)
        if edge is None:
            return None
        self.ready_names.discard(edge_name)
        self.is_ready = self.input_names.issubset(self.ready_names)
        self.is_ready_for_streaming = self.is_ready or self.input_names.issuperset(
            self.ready_names | self.streaming_inputs
        )
        return edge


@dataclass
class GraphNode(GraphSection):
    name: str
    input_names: set[str]
    outputs: list[GraphEdge]
    consumes_stream: bool = False
    # When False, the worker's speculation gate (``_can_speculate``) refuses to
    # build a speculative N+1 batch for this node. Models opt out of spec for
    # nodes whose forward pass mutates global state in ways the spec path
    # doesn't account for — e.g. BAGEL's flow-loop LLM / combine_cfg, where
    # the CFG-parallel branches need their KV caches snapshotted in lockstep.
    enable_async_scheduling: bool = True

    # Whether this node is currently speculatively being executed: if so, we
    # don't want to queue it in a "ready" queue because it's already being
    # executed, but we do want to update everything else as normal.
    _speculatively_scheduled: bool = False

    _streaming_inputs: set[str] = field(default_factory=set)

    _managing_registry: "GraphStateRegistry | None" = None

    def __post_init__(self):
        # if the user inputs, e.g., a list, turn it into a set
        self.input_names = set(self.input_names)
        self.ready_signals = ReadySignals(
            node_name=self.name,
            input_names=self.input_names,
            streaming_inputs=self._streaming_inputs
        )
        self.ready_next_iter = ReadySignals(
            node_name=self.name,
            input_names=self.input_names,
            streaming_inputs=self._streaming_inputs
        )
        self.speculative_signals = ReadySignals(
            node_name=self.name,
            input_names=self.input_names,
            streaming_inputs=self._streaming_inputs
        )
        self._tensor_manager = None
        self._request_id = None
    
    def register_communication_info(self, communication_manager, request_id: str):
        self.ready_signals.register_communication_info(
            communication_manager, request_id
        )
        self.ready_next_iter.register_communication_info(
            communication_manager, request_id
        )

    def _register_streaming(self, streaming_inputs: set[str]):
        # Mutate the existing set in place so that the ReadySignals instances
        # built in __post_init__ (which capture self._streaming_inputs by
        # reference) stay in sync. Rebinding the attribute would leave them
        # pointing at the original empty set.
        if not streaming_inputs:
            return
        self.consumes_stream = True
        self._streaming_inputs.clear()
        self._streaming_inputs.update(streaming_inputs)

    def ingest_input(self, edge: GraphEdge, can_buffer: bool=True) -> bool:
        """Try to ingest an arriving edge.

        Returns True on success, False when the edge is rejected; e.g., this
        is a streaming edge that we are not redy for and needs to get re-queued 
        """
        if edge.next_node != self.name:
            return False
        if edge.name not in self.input_names:
            return False
        if edge.name not in self.ready_signals.ready_names:
            self.ready_signals.update(edge)
        elif can_buffer and (
            edge.name not in self.ready_next_iter.ready_names
        ):
            # already have this input for the current iteration — buffer for the next loop iter
            # we do not buffer streaming signals, or else we may lose streaming signals if
            # there is actually no next iter
            self.ready_next_iter.update(edge)
        else:
            return False
        self._managing_registry.register_ingested_input(edge)
        return True

    def get_inputs_outputs(self):
        loop_back = {
            (edge.name, self.name) for edge in self.outputs if edge.next_node == self.name
        }
        return NodeInputsOutputs(
            ext_inputs={
                (inp, self.name) for inp in self.input_names if (inp, self.name) not in loop_back
            },
            ext_outputs=[out for out in self.outputs if out.next_node != self.name],
            loop_back=loop_back
        )
    
    def is_ready_for_speculation(
        self, check_next_iter: bool=False,
        allow_streaming: bool=True
    ):
        needed_inputs = self.input_names
        if allow_streaming:
            needed_inputs = needed_inputs - self._streaming_inputs
        if check_next_iter:
            return needed_inputs.issubset(
                self.ready_next_iter.ready_names | self.speculative_signals.ready_names
            )
        return needed_inputs.issubset(
            self.ready_signals.ready_names | self.speculative_signals.ready_names
        )
    
    def get_nodes(self):
        return {self.name: self}
    
    def get_loops(self):
        return {}
    
    def reset_for_outer_iter(self):
        self.ready_signals.clear()
        # promote next-iter inputs to current; reuse the now-empty object as the new next-iter buffer
        self.ready_signals, self.ready_next_iter = (
            self.ready_next_iter, self.ready_signals
        )
    
    def clear(self):
        self.ready_signals.clear()
        self.ready_next_iter.clear()
        self.speculative_signals.clear()

    def reset_outputs(self):
        for out in self.outputs:
            out.tensor_info.clear()

    def complete(self) -> NodeCompletionOutput:
        return self._managing_registry.mark_entity_complete(self.name)


@dataclass
class Sequential(GraphSection):
    sections: list[GraphSection]

    def get_nodes(self):
        nodes = {}
        for sec in self.sections:
            nodes.update(sec.get_nodes())
        return nodes
    
    def get_loops(self):
        loops = {}
        for sec in self.sections:
            loops.update(sec.get_loops())
        return loops

    def get_inputs_outputs(self):
        nodes_so_far = set()

        ext_outputs: list[GraphEdge] = []
        ext_inputs: set[NameAndDest] = set()
        loop_back: set[NameAndDest] = set()
        internal_signals: set[NameAndDest] = set()

        for sec in self.sections:
            io = sec.get_inputs_outputs()

            loop_back |= io.loop_back
            ext_inputs |= io.ext_inputs
            ext_outputs.extend(io.ext_outputs)

            internal_signals |= {
                (edge.name, edge.next_node) for edge in ext_outputs \
                    if (edge.name, edge.next_node) in ext_inputs
            }
            loop_back |= {
                (edge.name, edge.next_node) for edge in io.ext_outputs \
                    if edge.next_node in nodes_so_far
            }

            ext_outputs = [
                edge for edge in ext_outputs \
                    if (edge.name, edge.next_node) not in (internal_signals | loop_back)
            ]
            ext_inputs -= (internal_signals | loop_back)

            nodes_so_far.update(sec.get_nodes().keys())
        return NodeInputsOutputs(
            ext_inputs=ext_inputs,
            ext_outputs=ext_outputs,
            loop_back=loop_back
        )


@dataclass
class Parallel(GraphSection):
    sections: list[GraphSection]

    def get_nodes(self):
        nodes = {}
        for sec in self.sections:
            nodes.update(sec.get_nodes())
        return nodes
    
    def get_loops(self):
        loops = {}
        for sec in self.sections:
            loops.update(sec.get_loops())
        return loops

    def get_inputs_outputs(self):
        ext_inputs: set[NameAndDest] = set()
        loop_back: set[NameAndDest] = set()
        ext_outputs: list[GraphEdge] = []

        nodes = set(self.get_nodes().keys())
        for sec in self.sections:
            sec_io = sec.get_inputs_outputs()

            sec_io.ext_inputs = set({(name, dest) for name, dest in sec_io.ext_inputs if dest not in nodes})
            sec_io.loop_back |= set({(name, dest) for name, dest in sec_io.ext_inputs if dest in nodes})
            sec_io.ext_outputs = [edge for edge in sec_io.ext_outputs  if edge.next_node not in nodes]

            ext_inputs.update(sec_io.ext_inputs)
            loop_back.update(sec_io.loop_back)
            ext_outputs.extend(sec_io.ext_outputs)
        return NodeInputsOutputs(ext_inputs=ext_inputs, ext_outputs=ext_outputs, loop_back=loop_back)


@dataclass
class Loop(GraphSection):
    section: GraphSection
    max_iters: int
    outputs: list[GraphEdge]

    # MUST manually set the loop name if you want to be able to dynamically register
    # loop stop signals, since stop signals are registered by loop name
    name: str = field(default_factory=lambda: str(uuid4()))

    # Per-iteration outputs whose tensor_info is appended across iterations and emitted
    # all at once on loop completion. Disjoint from outputs by name.
    accumulated_outputs: list[GraphEdge] = field(default_factory=list)

    curr_iter: int = field(default=0)
    is_done: bool = False
    _finish_signal: bool = False

    _managing_registry: "GraphStateRegistry | None" = None
    # external inputs are saved so they can be re-injected at the start of each new iteration
    _ingested_external_inputs: list[GraphEdge] = field(default_factory=list)
    _ingested_external_input_names: set[str] = field(default_factory=set)

    _cached_outputs: dict[str, list[TensorPointerInfo]] = field(default_factory=dict)
    _accumulated_cache: dict[str, list[TensorPointerInfo]] = field(default_factory=dict)
    _accumulated_output_names: set[str] = field(default_factory=set)

    # These fields are usually set in __post_init__, except in the case of disaggregated
    # loops, where they are set manually
    _external_inputs: set[NameAndDest] | None = None
    _loop_back_inputs: set[NameAndDest] | None = None


    def get_nodes(self):
        return self.section.get_nodes()
    
    def register_finished(self):
        self._finish_signal = True
    
    def get_loops(self):
        loops = self.section.get_loops()
        loops[self.name] = self
        return loops
    
    def get_inputs_outputs(self):
        inp_out = self.section.get_inputs_outputs()
        inp_out.ext_outputs = self.outputs + self.accumulated_outputs
        return inp_out
    
    def _advance_one_iter(self):
        self.curr_iter += 1
        self.inner_registry.reset_for_iter()
        self._uncache_outputs()
    
    def ingest_external_input(self, graph_edge: GraphEdge):
        # track one copy of each external input for re-injection on the next iteration
        if (graph_edge.name, graph_edge.next_node) in self._external_inputs \
                and graph_edge.name not in self._ingested_external_input_names:
            self._ingested_external_inputs.append(graph_edge)
            self._ingested_external_input_names.add(graph_edge.name)
            graph_edge._persist_for_loop = True
        self._managing_registry.register_ingested_input(graph_edge)
    
    def complete_iter(self):
        """Called when every entity in the loop's section has finished for this iteration.

        If the loop is done (last iter or finish signal received): populates output
        tensor_info from the cache, marks itself complete in the outer registry, and
        returns the loop's declared outputs with loop-back signals in filtered_signals.
        Otherwise: advances curr_iter, resets the inner registry, and returns the
        saved external inputs so they can be re-routed into the next iteration.
        """
        if self.max_iters == self.curr_iter + 1 or self._finish_signal:
            self.is_done = True
            # Clear buffered loop-back inputs so no stale ready state lingers after termination.
            # Note: wg_state_registry.ready_next_iter may still hold node names added via
            # register_ingested_input before termination — cleaning those up is deferred
            # (they're harmless since reset_for_iter won't be called after the loop is done).
            self.inner_registry.clear()
            self._managing_registry.mark_entity_complete(self.name)
            for edge in self.outputs:
                edge.tensor_info = self._cached_outputs.get(edge.name, [])
            for edge in self.accumulated_outputs:
                edge.tensor_info = self._accumulated_cache.get(edge.name, [])
            
            # Don't dereference, becaue the tensors will be used downstream
            self._cached_outputs.clear()
            self._accumulated_cache.clear()
            return NodeCompletionOutput(
                output_edges=self.outputs + self.accumulated_outputs,
                filtered_signals=self._loop_back_inputs
            )
        else:
            self._advance_one_iter()
            return NodeCompletionOutput(
                output_edges=self._ingested_external_inputs
            )
    
    def register_communication_info(self, communication_manager, request_id: str):
        self._tensor_manager = communication_manager
        self._request_id = request_id
    
    def maybe_cache_output(self, edges: list[GraphEdge]):
        """Snapshot tensor_info for any edge that matches a declared loop output.

        Called after every entity completion so the most recent iteration's
        tensor_info is available when complete_iter populates self.outputs.
        Accumulated output tensor_info is appended across iterations.
        Deduplicates by name so multi-destination edges don't double-count.
        """
        seen_names: set[str] = set()
        for edge in edges:
            if edge.name in seen_names:
                continue
            seen_names.add(edge.name)
            if edge.name in self._output_names:
                self._cached_outputs.setdefault(
                    edge.name,[]
                ).extend(edge.tensor_info)
                if self._tensor_manager is not None:
                    for info in edge.tensor_info:
                        self._tensor_manager.increment_ref(self._request_id, info.uuid)

            elif edge.name in self._accumulated_output_names:
                self._accumulated_cache.setdefault(edge.name, []).extend(edge.tensor_info)
                if self._tensor_manager is not None:
                    for info in edge.tensor_info:
                        self._tensor_manager.increment_ref(self._request_id, info.uuid)

    
    def __post_init__(self):
        # Will be set later
        self._tensor_manager = None
        self._request_id = None

        io = self.section.get_inputs_outputs()
        if self._external_inputs is None:
            self._external_inputs = io.ext_inputs
        if self._loop_back_inputs is None:
            self._loop_back_inputs = io.loop_back

        # In the disaggregated case, we need filter self.outputs for outputs
        # that this subgraph actually produces
        outputs_we_produce = set([edge.name for edge in io.ext_outputs]) | set([name for name, _ in io.loop_back])
        self.outputs = [edge for edge in self.outputs if edge.name in outputs_we_produce]
        self.accumulated_outputs = [edge for edge in self.accumulated_outputs if edge.name in outputs_we_produce]

        self._output_names = {out.name for out in self.outputs}
        self._accumulated_output_names = {out.name for out in self.accumulated_outputs}
        _overlap = self._output_names & self._accumulated_output_names
        if _overlap:
            raise ValueError(
                f"Loop.outputs and Loop.accumulated_outputs must be disjoint by name; "
                f"overlap: {sorted(_overlap)}"
            )
        self.inner_registry = LoopStateRegistry(self)
    
    def _reset_metadata(self):
        self._ingested_external_inputs.clear()
        self._ingested_external_input_names.clear()
        self.curr_iter = 0
        self._finish_signal = False
        self.is_done = False
    
    def _uncache_outputs(self):
        if self._tensor_manager is not None and self._request_id is not None:
            for tensor_infos in self._cached_outputs.values():
                for info in tensor_infos:
                    self._tensor_manager.dereference(self._request_id, info.uuid)
        self._cached_outputs.clear()
    
    def _uncache_accumulated_outputs(self):
        if self._tensor_manager is not None and self._request_id is not None:
            for tensor_infos in self._accumulated_cache.values():
                for info in tensor_infos:
                    self._tensor_manager.dereference(self._request_id, info.uuid)
        self._accumulated_cache.clear()

    def reset_for_outer_iter(self):
        self.inner_registry.reset_for_iter()
        self._reset_metadata()
    
    def clear(self):
        self.inner_registry.clear()
        self._reset_metadata()
        self._uncache_accumulated_outputs()


class GraphStateRegistry(ABC):
    def __init__(
        self, graph_section: GraphSection,
    ):
        # "Outer-level" nodes and loops
        self.managed_entities: dict[str, GraphNode|Loop] = {}

        def _set_managed_entities(section: GraphSection):
            if isinstance(section, GraphNode) or isinstance(section, Loop):
                # stop here — Loop's inner entities are owned by its own LoopStateRegistry
                self.managed_entities[section.name] = section
                return
            assert isinstance(section, Sequential) or isinstance(section, Parallel)
            for sec in section.sections:
                _set_managed_entities(sec)
        _set_managed_entities(graph_section)

        self.is_done = False
        self._num_completed_entities = 0
        self._num_managed_entities = len(self.managed_entities)
    
    def mark_entity_complete(self, entity_name: str) -> NodeCompletionOutput:
        """Record that an entity has finished; no-ops if already done (safeguard only)."""
        if not self.is_done and entity_name in self.managed_entities:
            self._num_completed_entities += 1
            self.is_done = self._num_completed_entities == self._num_managed_entities
        return NodeCompletionOutput(
            output_edges=self.managed_entities[entity_name].outputs
        )
    
    @abstractmethod
    def register_ingested_input(self, graph_edge: GraphEdge):
        pass
    
    def reset_for_iter(self):
        self.is_done = False
        self._num_completed_entities = 0
        for entity in self.managed_entities.values():
            entity.reset_for_outer_iter()

    def clear(self):
        self.is_done = False
        self._num_completed_entities = 0
        for entity in self.managed_entities.values():
            entity.clear()


class LoopStateRegistry(GraphStateRegistry):
    def __init__(
        self, loop: Loop
    ):
        super().__init__(loop.section)
        self.loop = loop
    
    def mark_entity_complete(self, entity_name: str) -> NodeCompletionOutput:
        output = super().mark_entity_complete(entity_name)
        self.loop.maybe_cache_output(self.managed_entities[entity_name].outputs)
        if self.is_done:
            loop_out = self.loop.complete_iter()
            output.filtered_signals.update(loop_out.filtered_signals)
            # strip loop-back edges from this entity's raw outputs, then append the
            # loop's declared outputs (either the external outputs if done, or the
            # re-injected external inputs if advancing to the next iteration)
            output.output_edges = [
                edge for edge in output.output_edges \
                    if (edge.name, edge.next_node) not in loop_out.filtered_signals
            ]
            output.output_edges.extend(loop_out.output_edges)
        return output
    
    def register_ingested_input(self, graph_edge: GraphEdge):
        self.loop.ingest_external_input(graph_edge)


class WorkerGraphStateRegistry(GraphStateRegistry):
    def __init__(self, graph_section: GraphSection):
        super().__init__(graph_section)

        self.ready_names = set()
        self.ready_next_iter = set()
        self.ready_streaming_next_iter = set()

        self.nodes = graph_section.get_nodes()
        self.only_streaming_inputs = {
            name for name, node in self.nodes.items() \
                if node.input_names.issubset(node._streaming_inputs)
        }

        self.ready_for_streaming = set(self.only_streaming_inputs)
        self.ready_streaming_next_iter = set(self.only_streaming_inputs)

    def register_ingested_input(self, graph_edge: GraphEdge):
        node = self.nodes[graph_edge.next_node]
        # If node._speculatively_scheduled, the node is already executing as
        # a spec batch. We don't want to double-queue it (either for the
        # current iter via ``ready_names`` or for the next iter via
        # ``ready_next_iter`` — both eventually feed the scheduler) so we
        # gate every queue add on the flag.
        if node.ready_signals.is_ready:
            if not node._speculatively_scheduled:
                self.ready_names.add(node.name)
            self.ready_for_streaming.discard(node.name)
        elif node.ready_signals.is_ready_for_streaming:
            if not node._speculatively_scheduled:
                self.ready_for_streaming.add(node.name)

        if node.ready_next_iter.is_ready:
            if not node._speculatively_scheduled:
                self.ready_next_iter.add(node.name)
            self.ready_streaming_next_iter.discard(node.name)
        elif node.ready_next_iter.is_ready_for_streaming:
            if not node._speculatively_scheduled:
                self.ready_streaming_next_iter.add(node.name)

    def mark_entity_complete(self, entity_name: str) -> NodeCompletionOutput:
        # Top-level entities have no outer loop to drive a reset_for_iter, so
        # we clear the entity's ready_signals here so the same node can be
        # ingested for a future forward pass without ingest_input falling
        # through to ready_next_iter
        output = super().mark_entity_complete(entity_name)
        entity = self.managed_entities.get(entity_name)
        if isinstance(entity, GraphNode):
            entity.ready_signals.clear()
        return output

    def reset_for_iter(self):
        super().reset_for_iter()
        self.ready_names.clear()
        self.ready_for_streaming = set(self.only_streaming_inputs)
        # promote next-iter ready sets to current; nodes that already received their
        # loop-back inputs are immediately ready for the new iteration
        self.ready_names, self.ready_next_iter = (
            self.ready_next_iter, self.ready_names
        )
        self.ready_for_streaming, self.ready_streaming_next_iter = (
            self.ready_streaming_next_iter, self.ready_for_streaming
        )

    def clear(self):
        super().clear()
        self.ready_names.clear()
        self.ready_for_streaming = set(self.only_streaming_inputs)
        self.ready_next_iter.clear()
        self.ready_for_streaming = set(self.only_streaming_inputs)
