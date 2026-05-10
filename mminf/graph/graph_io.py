from mminf.graph.base import *


def format_graph_edge_list(lst: list[GraphEdge]) -> str:
    return ", ".join([f"{edge.name} -> {edge.next_node}" for edge in lst])


@dataclass
class NestedLoopIndices:
    loop_name_order: list[str] # from outer to inner
    loop_indices: dict[str]
    fwd_pass_idx: int

    def label_context_gt(self, other: "NestedLoopIndices", target_loop_name: str) -> bool:
        """
        Whether the iter indices of "self" are greater than the indices of
        "other", specifically in the path leading up to (but not including)
        "target_loop_name".

        For instance, if we are stopping the loop "target_label" but don't
        want to double-stop it, we can keep track of the last time it was
        stopped and only stop it again `new_time.label_context_gt(prev, label)`.
        """
        if self.fwd_pass_idx > other.fwd_pass_idx:
            return True
        if self.fwd_pass_idx < other.fwd_pass_idx:
            return False
        for name in self.loop_name_order:
            if name == target_loop_name:
                break

            our_idx, their_idx = self.loop_indices.get(name, 0), other.loop_indices.get(name, 0)
            if our_idx > their_idx:
                return True
            if our_idx < their_idx:
                return False
        return False
            

class WorkerGraphIO:
    """Primary interface between the worker execution loop and a computation graph.

    The worker calls ingest_input with arriving edges, reads ready_node_names to pick
    a node to run, calls mark_node_complete, and routes the returned output edges.
    register_loop_finish_signal handles externally-signalled loop termination (e.g. EOS).
    """
    def __init__(
        self, graph: GraphSection
    ):
        self.nodes = graph.get_nodes()
        self.loops = graph.get_loops()
        self.graph = graph

        # set up managing registries
        self.wg_state_registry = WorkerGraphStateRegistry(graph)

        def _setup_registries(
            sec: GraphSection, registry: GraphStateRegistry
        ):
            if isinstance(sec, GraphNode):
                sec._managing_registry = registry
                return
            if isinstance(sec, Loop):
                sec._managing_registry = registry
                return _setup_registries(sec.section, sec.inner_registry)
            # either sequential or parallel
            assert isinstance(sec, Sequential) or isinstance(sec, Parallel)
            for s in sec.sections:
                _setup_registries(s, registry)
        _setup_registries(graph, self.wg_state_registry)

        self._nodes_with_speculative_inputs: set[str] = set()
        # node_name → True if any speculatively ingested input was a loop-back signal
        self._speculative_node_has_loop_back: dict[str, bool] = {}
        self._speculative_ready: dict[str, SpeculativeNodeInfo] = {}

    @property
    def ready_node_names(self):
        return self.wg_state_registry.ready_names
    
    @property
    def ready_for_streaming(self):
        return self.wg_state_registry.ready_for_streaming
    
    def get_node(self, name: str):
        assert name in self.nodes
        return self.nodes[name]
    
    def ingest_input(
        self, graph_edge: GraphEdge
    ) -> bool:
        """Route an arriving edge to its destination node.

        Returns False if next_node is not in this graph (e.g. destined for another worker).
        """
        if graph_edge.next_node not in self.nodes:
            return False
        node = self.nodes[graph_edge.next_node]
        node.ingest_input(graph_edge)
        return True
    
    def mark_node_complete(
        self, node_name: str
    ) -> NodeCompletionOutput:
        """Signal that a node has finished executing.

        The caller must route each edge in output_edges, skipping any in filtered_signals.
        """
        node = self.nodes[node_name]
        return node.complete()

    def ingest_for_speculation(self, edge: GraphEdge) -> None:
        """Ingest an anticipated output edge into the speculative buffer of its destination.

        Called with the edges the currently-executing node is expected to produce, before
        it has actually completed. The edge's tensor_info will be empty at this point;
        callers must not copy the edge object — actual tensor_info is updated in-place
        when the producing node completes (no-copy semantics).

        Readiness gate is strict: the destination node is reported as
        speculatively ready only when speculative_signals alone covers
        input_names. Callers (e.g. worker `_can_speculate`) are responsible
        for ensuring all required inputs are loop-back / streaming and will
        be ingested speculatively before checking ready_for_speculation.
        """
        if edge.next_node not in self.nodes:
            return
        node = self.nodes[edge.next_node]
        if edge.name in node.speculative_signals.ready_names:
            return
        node.speculative_signals.update(edge)
        is_loop_back = (
            isinstance(node._managing_registry, LoopStateRegistry)
            and (edge.name, edge.next_node) in node._managing_registry.loop._loop_back_inputs
        )
        if is_loop_back:
            self._speculative_node_has_loop_back[node.name] = True
        self._nodes_with_speculative_inputs.add(node.name)
        if node.input_names.issubset(node.speculative_signals.ready_names) \
                and node.name not in self._speculative_ready:
            self._speculative_ready[node.name] = SpeculativeNodeInfo(
                node_name=node.name,
                is_new_loop_iter=self._speculative_node_has_loop_back.get(node.name, False),
            )

    @property
    def ready_for_speculation(self) -> list[SpeculativeNodeInfo]:
        """Nodes that are speculatively ready based on anticipated outputs of the current node."""
        return list(self._speculative_ready.values())

    def clear_speculative_inputs(self) -> None:
        """Clear all speculative buffers — call when discarding a speculative schedule."""
        for node_name in self._nodes_with_speculative_inputs:
            self.nodes[node_name].speculative_signals.clear()
        self._nodes_with_speculative_inputs.clear()
        self._speculative_node_has_loop_back.clear()
        self._speculative_ready.clear()

    def register_loop_finish_signal(
        self, loop_name: str
    ):
        """
        Registers an external loop finish signal (e.g., saw an EOS)
        """
        if loop_name in self.loops:
            self.loops[loop_name].register_finished()
    
    def clear(self):
        self.wg_state_registry.clear()

    def register_communication_info(self, communication_manager, request_id: str):
        for loop in self.loops.values():
            loop.register_communication_info(
                communication_manager, request_id
            )
    
    def get_loop_indices(self):
        return {
            name: loop.curr_iter for name, loop in self.loops.items()
        }
    
    def get_nested_loop_idxs(
        self, fwd_pass_idx: int,
        target_loop_name: str
    ):
        def _get_loop_order(inner_loop_name: str):
            assert inner_loop_name in self.loops
            loop = self.loops[inner_loop_name]
            if isinstance(loop._managing_registry, WorkerGraphStateRegistry):
                return [inner_loop_name]
            assert isinstance(loop._managing_registry, LoopStateRegistry)
            return _get_loop_order(
                loop._managing_registry.loop.name
            ) + [inner_loop_name]
        return NestedLoopIndices(
            loop_name_order=_get_loop_order(target_loop_name),
            loop_indices=self.get_loop_indices(),
            fwd_pass_idx=fwd_pass_idx
        )