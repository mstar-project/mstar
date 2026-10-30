"""The worker's side of chunked prefill: which (node, walk) pairs run as a
chunk Loop, ending a Loop on its final chunk, and recording what each chunk
ran. The Loop itself does the rest (``Model.apply_chunking``)."""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from mstar.model.submodule_base import chunk_loop_name
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch, runs_final_chunk

if TYPE_CHECKING:
    from mstar.engine.engine import Engine
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

    def end_finished(self, batch: ScheduledBatch, stops: dict) -> set[int]:
        """End the chunk Loop of every row that just ran its last chunk, and
        name the rows still mid-prefill.

        Keyed on the walk having a Loop rather than on the row having a chunk
        range: a row with no range ran whole, which is also a final chunk -- a
        walk with no token budget is never measured or cut, and without this
        its Loop would run to ``max_iters`` and the request would never release
        its state.

        Until its last chunk a row's sampled token is not a real sample, so it
        decides no stop -- a spurious one would cut the prompt short
        mid-prefill -- and keys no prefix. Its own output the Loop drops.
        """
        node = batch.node_name
        mid_prefill: set[int] = set()
        walks = self.walks_by_node.get(node)
        # one lookup for a single-walk step outside a chunk Loop (decode)
        if not walks or (not batch.request_walks and batch.graph_walk not in walks):
            return mid_prefill
        for rid in batch.request_to_worker_graph:
            walk = batch.walk_of(rid)
            if walk not in walks:
                continue
            loop_name = chunk_loop_name(node, walk)
            if runs_final_chunk(batch.chunk_ranges.get(rid)):
                stops.setdefault(rid, set()).add(loop_name)
            else:
                stops.pop(rid, None)
                mid_prefill.add(rid)
        return mid_prefill

    @staticmethod
    def advance(
        batch: ScheduledBatch, scheduler: MicroScheduler, engine: "Engine",
    ) -> None:
        """Record what each chunked row ran, and let go of a finished prompt's
        prepared input.

        The chunk Loop does the rest: the re-injected prompt re-readies the
        node, and the Loop's edges drop or gather each chunk's output.
        """
        node = batch.node_name
        for rid, (_, end, total) in batch.chunk_ranges.items():
            if rid not in batch.request_to_worker_graph:
                continue
            scheduler.advance_chunk(rid, node, end)
            if end >= total:
                engine.release_chunk_inputs(rid, node)


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
