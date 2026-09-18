"""One request, its own copy of the graph, and the real graph layer.

This module repeats the part of the worker that runs after a forward pass.
``Worker._postprocess_batch`` does these steps, in this order:

1. read the inputs of the step out of the ready slot of the node
2. compute the outputs
3. ask each submodule if a loop must stop, and register the finish signal
4. ``reset_outputs``, then write the new values into the output edges of the
   node, one value for each name
5. ``mark_node_complete``, then clone each output edge
6. route each clone: into this graph, to the client, or to another worker

This module makes the same calls in the same order. It changes two of the
steps. Step 2 computes a checksum. Step 4 puts that checksum into
``TensorPointerInfo.uuid`` instead of moving a tensor.

Step 2 belongs to the caller, in two parts:

* ``begin_node`` gives the inputs of the step
* the caller computes one value for each output name
* ``finish_node`` writes those values and routes them

``model_run`` computes the value from the inputs alone. ``kv_run`` computes it
inside a real step of the resource layer. The value then also covers the cache
pages that the step reads.

``WorkerGraphQueues.add_request`` gives one copy of the graph to each request,
and this module does the same. Thus two requests share no state here. The
machine that drives the resources makes them share a batch and a cache.
"""

from __future__ import annotations

from typing import NamedTuple

from fuzzer.tier1.spec import GraphPlan, build_section
from fuzzer.tier1.values import carrier, seed_value, values_of
from mstar.graph.base import GraphEdge
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT, SPECIAL_DESTINATIONS

__all__ = ["NodeInputs", "RequestRun"]


class NodeInputs(NamedTuple):
    """What one step of one node reads, and where it sits in the pass."""

    request_id: str
    node_name: str
    # The number of times this node ran before, inside this pass.
    run_index: int
    # The name of each input, with the values that arrived under it.
    inputs: list[tuple[str, tuple[str, ...]]]
    # The names this step must give a value to.
    output_names: list[str]


class RequestRun:
    """One request: its graph, and the forward passes that it runs."""

    def __init__(self, graph_plan: GraphPlan, request_id: str, stops: list[dict]):
        self.plan = graph_plan
        self.request_id = request_id
        self.stops = stops
        self.io = WorkerGraphIO(build_section(graph_plan), wg_id="wg0")

        self.in_flight = False
        self.pass_index = 0
        # The values that reached the client during the pass, by edge name.
        self.emitted: dict[str, tuple[str, ...]] = {}
        # The number of times each node ran during the pass.
        self.runs: dict[str, int] = {}
        # Edges that no node of this graph took, and edges for another graph.
        self.rejected: list[GraphEdge] = []
        self.external: list[GraphEdge] = []

    # -- the lifecycle of a pass --------------------------------------------

    def start(self) -> None:
        """Start a forward pass. The conductor sends the first edges."""
        self.in_flight = True
        # The system under test counts the passes. A wrong count changes every
        # checksum of the pass, so the oracle sees it.
        self.pass_index = self.io.num_times_run
        self.emitted = {}
        self.runs = {}
        self.rejected = []
        self.external = []
        for name, dest in self.plan.seeds:
            edge = GraphEdge(name=name, next_node=dest)
            edge.tensor_info = [
                carrier(seed_value(self.request_id, self.pass_index, name))
            ]
            self._route(edge)

    def ready(self) -> list[str]:
        """The nodes that may run now, in a stable order.

        ``ready_node_names`` is a set of strings, and the order of a set of
        strings changes between processes. The driver sorts it, so a case
        replays.
        """
        return sorted(self.io.ready_node_names)

    def is_done(self) -> bool:
        return self.io.wg_state_registry.is_done

    def finish(self) -> dict[str, tuple[str, ...]]:
        """End the pass, as ``WorkerGraphQueues.reset`` does, and give the
        values that reached the client."""
        emitted = dict(self.emitted)
        self.io.clear()
        self.in_flight = False
        return emitted

    def abort(self) -> None:
        """Drop the request, as the worker does for a cleanup."""
        self.io.clear()
        self.in_flight = False

    # -- one step ------------------------------------------------------------

    def begin_node(self, node_name: str) -> NodeInputs:
        """Take a node out of the ready queue and read what its step sees.

        The caller computes one value for each output name, and hands them
        back to ``finish_node``.
        """
        node = self.io.nodes[node_name]
        self.io.ready_node_names.discard(node_name)

        run_index = self.runs.get(node_name, 0)
        self.runs[node_name] = run_index + 1

        output_names: list[str] = []
        for edge in node.outputs:
            if edge.name not in output_names:
                output_names.append(edge.name)

        return NodeInputs(
            request_id=self.request_id,
            node_name=node_name,
            run_index=run_index,
            inputs=[
                (edge.name, values_of(edge))
                for edge in node.ready_signals.ready_inputs.values()
            ],
            output_names=output_names,
        )

    def finish_node(self, node_name: str, values: dict[str, str]) -> None:
        """Write the values of a step into the graph, and route what it sends."""
        node = self.io.nodes[node_name]
        run_index = self.runs[node_name] - 1

        # One list for each name. Two edges of one name carry the same list,
        # as ``store_and_populate_graph_edges`` gives them.
        by_name = {name: [carrier(value)] for name, value in values.items()}

        # The worker registers a stop before it marks the node complete, so
        # the loop reads the signal in the same iteration.
        for stop in self.stops:
            if stop["node"] == node_name and stop["run"] == run_index:
                self.io.register_loop_finish_signal(stop["loop"])

        node.reset_outputs()
        for edge in node.outputs:
            edge.tensor_info = by_name[edge.name]

        completion = self.io.mark_node_complete(node_name)
        for edge in [edge.clone() for edge in completion.output_edges]:
            self._route(edge)

    def _route(self, edge: GraphEdge) -> None:
        """Send one edge to its destination."""
        if edge.next_node == EMIT_TO_CLIENT:
            self.emitted[edge.name] = self.emitted.get(edge.name, ()) + values_of(edge)
            return
        if edge.next_node in SPECIAL_DESTINATIONS:
            return
        if edge.next_node not in self.io.nodes:
            self.external.append(edge)
            return
        if not self.io.ingest_input(edge, can_buffer=True):
            self.rejected.append(edge)
