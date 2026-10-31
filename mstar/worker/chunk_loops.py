"""The worker's side of chunked prefill: which (node, walk) pairs run as a
chunk Loop, and how far each step moves them. The Loop itself does the rest
(``Model.apply_chunking``): it counts the tokens its chunks ran and ends at
its prompt's length."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from mstar.model.submodule_base import chunk_loop_name
from mstar.worker.micro_scheduler import ScheduledBatch, chunk_loop_stride

if TYPE_CHECKING:
    from mstar.model.base import WorkerGraph


class ChunkLoops:
    """This worker's chunk Loops, read off its graphs at load."""

    def __init__(
        self, walks: Iterable[tuple[str, str]] = (),
        speculated: Iterable[tuple[str, str]] | None = None,
    ):
        # (node, walk) pairs whose prefill runs as a chunk Loop
        self.walks: frozenset[tuple[str, str]] = frozenset(walks)
        # the ones whose rows mid-prefill take their next chunk in the speculative step
        self.speculated: frozenset[tuple[str, str]] = (
            self.walks if speculated is None else self.walks & frozenset(speculated)
        )
        # node -> those walks, which the scheduler marks final chunks by
        self.walks_by_node: dict[str, frozenset[str]] = {
            node: frozenset(w for n, w in self.walks if n == node)
            for node, _ in self.walks
        }

    @classmethod
    def from_graphs(
        cls, worker_graphs: "list[WorkerGraph]",
        speculate_mid_prefill: Mapping[str, object] | None = None,
        chunked: Iterable[tuple[str, str]] = (),
    ) -> "ChunkLoops":
        """``speculate_mid_prefill`` is the serving config's, checked against
        ``chunked``, every (node, walk) the model chunks."""
        walks = cls.walks_in(worker_graphs)
        return cls(walks, walks - mid_prefill_speculation_off(speculate_mid_prefill, chunked))

    @staticmethod
    def walks_in(worker_graphs: "list[WorkerGraph]") -> frozenset[tuple[str, str]]:
        """``(node, walk)`` pairs these graphs run inside a chunk Loop.

        Read off the graphs rather than off the model's chunking policy: the
        policy says where a Loop may be built, and a deployment that excludes
        a node from chunking builds none there. The graph cannot disagree with
        itself, and a Loop nothing ends spins to ``max_iters`` holding the
        request's state.
        """
        found: set[tuple[str, str]] = set()
        for wg in worker_graphs:
            loops = wg.section.get_loops()
            found.update(
                (node, walk)
                for node in wg.section.get_nodes() for walk in wg.graph_walks
                if chunk_loop_name(node, walk) in loops
            )
        return frozenset(found)

    def loop_strides(
        self, batch: ScheduledBatch, rids: Iterable[int],
    ) -> list[int] | None:
        """``RouteInput.loop_strides`` for ``rids`` of ``batch``, in that
        order: a row of a chunk-Loop walk moves its Loop by its chunk
        (``chunk_loop_stride``), and any other row its loop by one. None for a
        step with no row in a chunk Loop, which is every decode step.

        Keyed on the walk having a Loop rather than on the row having a chunk
        range: a row with no range ran whole, which also ends the Loop -- a
        walk with no token budget is never measured or cut.
        """
        walks = self.walks_by_node.get(batch.node_name)
        # one lookup for a single-walk step outside a chunk Loop (decode)
        if not walks or (not batch.request_walks and batch.graph_walk not in walks):
            return None
        ranges = batch.chunk_ranges
        walk_of = batch.walk_of
        return [
            chunk_loop_stride(ranges.get(rid)) if walk_of(rid) in walks else 1
            for rid in rids
        ]


def mid_prefill_speculation_off(
    setting: Mapping[str, object] | None, chunked: Iterable[tuple[str, str]],
) -> frozenset[tuple[str, str]]:
    """The (node, walk) pairs the serving config's ``speculate_mid_prefill``
    turns off: per node, one bool for every walk it chunks, or a mapping of
    walk -> bool. On (the default), a row mid-prefill takes its next chunk in
    the speculative step, as a decode row takes its next token."""
    walks_of_node: dict[str, set[str]] = defaultdict(set)
    for node, walk in chunked:
        walks_of_node[node].add(walk)
    off = set()
    for node, value in (setting or {}).items():
        if node not in walks_of_node:
            raise ValueError(f"speculate_mid_prefill names {node!r}, which chunks no walk")
        per_walk = value if isinstance(value, Mapping) else dict.fromkeys(walks_of_node[node], value)
        for walk, on in per_walk.items():
            if walk not in walks_of_node[node]:
                raise ValueError(f"speculate_mid_prefill names walk {walk!r} of {node!r}, which it does not chunk")
            if not isinstance(on, bool):
                raise ValueError(f"speculate_mid_prefill for {node}/{walk}: {on!r} is not a bool")
            if not on:
                off.add((node, walk))
    return frozenset(off)
