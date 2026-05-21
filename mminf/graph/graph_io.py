from mminf.graph.base import *
from mminf.graph.loop_indices import NestedLoopIndices  # re-exported for backward compat


def format_graph_edge_list(lst: list[GraphEdge]) -> str:
    return ", ".join([f"{edge.name} -> {edge.next_node}" for edge in lst])



class WorkerGraphIO:
    """Primary interface between the worker execution loop and a computation graph.

    The worker calls ingest_input with arriving edges, reads ready_node_names to pick
    a node to run, calls mark_node_complete, and routes the returned output edges.
    register_loop_finish_signal handles externally-signalled loop termination (e.g. EOS).
    """
    def __init__(
        self, graph: GraphSection,
        wg_id: str | None=None
    ):
        self.nodes = graph.get_nodes()
        self.loops = graph.get_loops()
        self.graph = graph
        self.wg_id = wg_id
        self.num_times_run = 0

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
        self, graph_edge: GraphEdge, can_buffer: bool=True
    ) -> bool:
        """Route an arriving edge to its destination node.

        Returns False when the edge isn't claimed here: ``next_node`` not in
        this graph, OR the node exists but rejected the edge (name mismatch
        / both ready slots full — see GraphNode.ingest_input). Callers should
        treat the False return as "try the next destination" (cross-worker
        routing or StreamBuffer re-queue).
        """
        if graph_edge.next_node not in self.nodes:
            return False
        return self.nodes[graph_edge.next_node].ingest_input(
            graph_edge, can_buffer
        )
    
    def mark_node_complete(
        self, node_name: str
    ) -> NodeCompletionOutput:
        """Signal that a node has finished executing.

        The caller must route each edge in output_edges, skipping any in filtered_signals.
        """
        prev_done_val = self.wg_state_registry.is_done
        node = self.nodes[node_name]
        completion = node.complete()
        # apply filtering, if needed
        completion.output_edges = [
            edge for edge in completion.output_edges \
                if (edge.name, edge.next_node) not in completion.filtered_signals
        ]

        if self.wg_state_registry.is_done and not prev_done_val:
            self.num_times_run += 1
        return completion
    
    def ingest_for_speculation(
        self, edges: list[GraphEdge], source_node: str
    ) -> list[SpeculativeNodeInfo]:
        """Ingest an set of anticipated output edges into speculative buffers.

        Returns a list of nodes that are ready to be specilatively executed:
        (A) if the same as source_node, that speculative & next_iter & streaming cover
        all of the inputs.

        (B) if not the same, that speculative & inputs & streaming cover the inputs.
        """
        dest_nodes = set()
        next_iter_nodes = set()

        source_reg = self.nodes[source_node]._managing_registry
        source_inside_loop = isinstance(source_reg, LoopStateRegistry)
        loop_back_inputs = (
            set() if not source_inside_loop else source_reg.loop._loop_back_inputs
        )

        for edge in edges:
            if edge.next_node not in self.nodes:
                continue # TODO: cross-worker-graph speculation
            node = self.nodes[edge.next_node]
            node.speculative_signals.update(edge)
            self._nodes_with_speculative_inputs.add(node.name)
            dest_nodes.add(node.name)

            if (edge.name, edge.next_node) in loop_back_inputs:
                next_iter_nodes.add(edge.next_node)

        ready_for_spec = []
        for dest in dest_nodes:
            node = self.nodes[dest]
            if node.is_ready_for_speculation(
                check_next_iter=dest == source_node,
                allow_streaming=True
            ):
                # ``loop_name`` is the destination's enclosing loop — that's
                # what callers need to inspect ``curr_iter`` / ``max_iters`` /
                # ``_finish_signal`` for the loop-completion filter. Source's
                # loop is irrelevant once we know which node is the spec target.
                dest_reg = node._managing_registry
                dest_loop_name = (
                    dest_reg.loop.name
                    if isinstance(dest_reg, LoopStateRegistry)
                    else None
                )
                ready_for_spec.append(SpeculativeNodeInfo(
                    node_name=dest,
                    is_new_loop_iter=dest in next_iter_nodes,
                    loop_name=dest_loop_name,
                ))
        return ready_for_spec

    def clear_speculative_inputs(self) -> None:
        """Clear all speculative buffers — call when discarding a speculative schedule."""
        for node_name in self._nodes_with_speculative_inputs:
            self.nodes[node_name].speculative_signals.clear()
        self._nodes_with_speculative_inputs.clear()

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
        for node in self.nodes.values():
            node.register_communication_info(
                communication_manager, request_id
            )
    
    def get_loop_indices(self):
        return {
            name: loop.curr_iter for name, loop in self.loops.items()
        }
    
    def get_nested_loop_idxs(
        self, target_loop_name: str
    ):
        def _get_loop_order(inner_loop_name: str):
            assert inner_loop_name in self.loops
            loop = self.loops[inner_loop_name]
            if not isinstance(loop._managing_registry, LoopStateRegistry):
                return [inner_loop_name]
            return _get_loop_order(
                loop._managing_registry.loop.name
            ) + [inner_loop_name]
        return NestedLoopIndices(
            loop_name_order=_get_loop_order(target_loop_name),
            loop_indices=self.get_loop_indices(),
            wg_fwd_pass_idx=self.num_times_run
        )
    
    def get_nested_loop_idxs_for_node(
        self, node_name: str
    ):
        assert node_name in self.nodes
        node = self.nodes[node_name]
        if not isinstance(node._managing_registry, LoopStateRegistry):
            return NestedLoopIndices(
                loop_name_order=[],
                loop_indices={},
                wg_fwd_pass_idx=self.num_times_run
            )
        return self.get_nested_loop_idxs(
            target_loop_name=node._managing_registry.loop.name
        )