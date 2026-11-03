"""How a node recomputes a dropped request's KV, as the model declares it."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from mstar.graph.base import GraphSection
from mstar.model.submodule_base import chunk_tick_name


@dataclass(frozen=True)
class RebuildPolicy:
    """A request that prefilled in ``walks`` is rebuilt by one row per prefill step, each run in its step's walk.
    The rows recompute the request's prompt and every id it has sampled since, except the last, which the next
    decode step reads.

    ``ids_input`` is the input that carries the prompt's ids, and a row's ids go under it. Whatever else the node
    reads in a walk, such as the encoder's output for a prompt with images, the worker copies to the host when the
    walk prefills a request, and the rebuild row reads it again. ``copy_width`` gives, for each such walk, how many
    values its copies hold per KV slot, each in the dtype the node runs in.
    ``sample_output`` is the output that carries the one id a row samples, and ``loop`` the loop each of
    whose iterations samples one.
    """
    walks: frozenset[str]
    ids_input: str
    sample_output: str
    loop: str
    copy_width: Mapping[str, int] = field(default_factory=dict)

    def loop_walks(self, node: str, graphs: Mapping[str, GraphSection]) -> frozenset[str]:
        """The walks in which ``loop`` runs ``node``: those a request can be dropped in."""
        return frozenset(
            walk for walk, graph in graphs.items()
            if (loop := graph.get_loops().get(self.loop)) is not None and node in loop.get_nodes()
        )

    def ids_walks(self, node: str, graphs: Mapping[str, GraphSection]) -> frozenset[str]:
        """The walks of ``walks`` in which ``node`` reads ``ids_input``: those an ids span can be written in."""
        return frozenset(walk for walk in self.walks if self.ids_input in self._reads(node, graphs, walk))

    def check(self, node: str, graphs: Mapping[str, GraphSection]) -> dict[str, frozenset[str]]:
        """What ``node`` reads in each of ``walks`` besides ``ids_input``, which the worker copies. Raise unless
        every walk runs ``node``, some walk reads ``ids_input``, ``copy_width`` names each walk where ``node`` reads
        more than ``ids_input``, and some walk runs ``node`` inside ``loop``. Also raise when ``node`` reads
        ``ids_input`` in a walk without outputting ``sample_output`` there, since such a walk can end a prompt."""
        if not self.ids_walks(node, graphs):
            raise ValueError(
                f"{node}'s rebuild row carries its ids as {self.ids_input!r}, which {node} reads in none of its walks"
            )
        copied = {}
        for walk in sorted(self.walks):
            reads = self._reads(node, graphs, walk)
            if self.ids_input in reads and self.sample_output not in {
                edge.name for edge in graphs[walk].get_nodes()[node].outputs
            }:
                raise ValueError(
                    f"{node}'s rebuild policy keeps samples from {self.sample_output!r}, which {node} doesn't output "
                    f"in {walk!r}, where it reads {self.ids_input!r}"
                )
            copied[walk] = frozenset(reads - {self.ids_input})
            if copied[walk] and walk not in self.copy_width:
                raise ValueError(
                    f"{node} reads {sorted(copied[walk])} besides its ids in {walk!r}, and its rebuild policy doesn't "
                    "say how many values their host copies hold per slot"
                )
        if not self.loop_walks(node, graphs):
            raise ValueError(f"{node}'s rebuild policy names loop {self.loop!r}, which runs {node} in no walk")
        return copied

    @staticmethod
    def _reads(node: str, graphs: Mapping[str, GraphSection], walk: str) -> set[str]:
        """What ``node`` reads in ``walk``, less its chunk Loop's tick, which carries no tensor."""
        graph = graphs.get(walk)
        graph_node = None if graph is None else graph.get_nodes().get(node)
        if graph_node is None:
            raise ValueError(f"{node}'s rebuild policy names walk {walk!r}, which doesn't run {node}")
        return set(graph_node.input_names) - {chunk_tick_name(node, walk)}
