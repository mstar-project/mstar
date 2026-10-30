"""How a node recomputes a dropped request's KV, as the model declares it."""

from collections.abc import Mapping
from dataclasses import dataclass

from mstar.graph.base import GraphSection


@dataclass(frozen=True)
class RebuildPolicy:
    """One row of ``walk`` prefills the request's prompt and every id it has sampled since, but the
    last, which the next decode step reads.

    ``ids_input`` is the input of ``walk`` that carries the prompt's ids, and the row's ids go under it.
    ``sample_output`` is the output that carries the one id a row samples, and ``loop`` the loop each of
    whose iterations samples one.
    """
    walk: str
    ids_input: str
    sample_output: str
    loop: str

    def loop_walks(self, node: str, graphs: Mapping[str, GraphSection]) -> frozenset[str]:
        """The walks in which ``loop`` runs ``node``: those a request can be dropped in."""
        return frozenset(
            walk for walk, graph in graphs.items()
            if (loop := graph.get_loops().get(self.loop)) is not None and node in loop.get_nodes()
        )

    def check(self, node: str, graphs: Mapping[str, GraphSection]) -> None:
        """Raise unless ``walk`` runs ``node``, which reads only ``ids_input``, the one input a rebuild row
        carries, and outputs ``sample_output`` there, and some walk runs ``node`` inside ``loop``."""
        graph = graphs.get(self.walk)
        graph_node = None if graph is None else graph.get_nodes().get(node)
        if graph_node is None:
            raise ValueError(f"{node}'s rebuild policy names walk {self.walk!r}, which doesn't run {node}")
        if set(graph_node.input_names) != {self.ids_input}:
            raise ValueError(
                f"{node}'s rebuild row carries only {self.ids_input!r}, and {node} reads "
                f"{sorted(graph_node.input_names)} in {self.walk!r}"
            )
        if self.sample_output not in {edge.name for edge in graph_node.outputs}:
            raise ValueError(
                f"{node}'s rebuild policy keeps samples from {self.sample_output!r}, which {node} doesn't output "
                f"in {self.walk!r}"
            )
        if not self.loop_walks(node, graphs):
            raise ValueError(f"{node}'s rebuild policy names loop {self.loop!r}, which runs {node} in no walk")
