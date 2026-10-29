"""The worker's side of chunked prefill: which (node, walk) pairs run as a
chunk Loop, ending a Loop on its final chunk, and recording what each chunk
ran. The Loop itself does the rest (``Model.apply_chunking``)."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from mstar.model.submodule_base import chunk_loop_name
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch, runs_final_chunk

if TYPE_CHECKING:
    from mstar.engine.engine import Engine
    from mstar.model.base import WorkerGraph


class ChunkLoops:
    """This worker's chunk Loops, read off its graphs at load."""

    def __init__(self, walks: Iterable[tuple[str, str]] = ()):
        # (node, walk) pairs whose prefill runs as a chunk Loop
        self.walks: frozenset[tuple[str, str]] = frozenset(walks)
        # node -> those walks, which the scheduler marks final chunks by
        self.walks_by_node: dict[str, frozenset[str]] = {
            node: frozenset(w for n, w in self.walks if n == node)
            for node, _ in self.walks
        }

    @classmethod
    def from_graphs(cls, worker_graphs: "list[WorkerGraph]") -> "ChunkLoops":
        return cls(cls.walks_in(worker_graphs))

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
        if not self.walks:
            return mid_prefill
        for rid in batch.request_to_worker_graph:
            walk = batch.walk_of(rid)
            if (node, walk) not in self.walks:
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
