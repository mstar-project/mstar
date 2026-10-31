"""How a node recomputes a dropped request's KV, as the model declares it."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from mstar.graph.base import GraphSection


@dataclass(frozen=True)
class RebuildPolicy:
    """One row of the walk a request prefilled in recomputes its prompt and every id it has sampled
    since, but the last, which the next decode step reads.

    ``walk`` prefills a prompt of ids. ``copied_inputs`` maps each other walk that prefills a prompt,
    such as one with images, to the inputs besides the ids that the node reads there. The worker copies
    those inputs to the host when the walk prefills a request, and the request's rebuild row reads them
    again.

    ``ids_input`` is the input that carries the prompt's ids, and the row's ids go under it.
    ``sample_output`` is the output that carries the one id a row samples, and ``loop`` the loop each of
    whose iterations samples one.
    """
    walk: str
    ids_input: str
    sample_output: str
    loop: str
    copied_inputs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def prefill_walks(self) -> frozenset[str]:
        """The walks a request can prefill in and be rebuilt from."""
        return frozenset({self.walk, *self.copied_inputs})

    def loop_walks(self, node: str, graphs: Mapping[str, GraphSection]) -> frozenset[str]:
        """The walks in which ``loop`` runs ``node``: those a request can be dropped in."""
        return frozenset(
            walk for walk, graph in graphs.items()
            if (loop := graph.get_loops().get(self.loop)) is not None and node in loop.get_nodes()
        )

    def check(self, node: str, graphs: Mapping[str, GraphSection]) -> None:
        """Raise unless each prefill walk runs ``node``, which reads there only the ids and the inputs the
        worker copies, and outputs ``sample_output``, and some walk runs ``node`` inside ``loop``."""
        for walk in sorted(self.prefill_walks):
            self._check_walk(node, graphs, walk, {self.ids_input, *self.copied_inputs.get(walk, ())})
        if not self.loop_walks(node, graphs):
            raise ValueError(f"{node}'s rebuild policy names loop {self.loop!r}, which runs {node} in no walk")

    def _check_walk(self, node: str, graphs: Mapping[str, GraphSection], walk: str, carried: set[str]) -> None:
        graph = graphs.get(walk)
        graph_node = None if graph is None else graph.get_nodes().get(node)
        if graph_node is None:
            raise ValueError(f"{node}'s rebuild policy names walk {walk!r}, which doesn't run {node}")
        if set(graph_node.input_names) != carried:
            raise ValueError(
                f"{node}'s rebuild row carries {sorted(carried)} in {walk!r}, and {node} reads "
                f"{sorted(graph_node.input_names)} there"
            )
        if self.sample_output not in {edge.name for edge in graph_node.outputs}:
            raise ValueError(
                f"{node}'s rebuild policy keeps samples from {self.sample_output!r}, which {node} doesn't output "
                f"in {walk!r}"
            )
