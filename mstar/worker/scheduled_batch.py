"""A step's rows as the scheduler hands them on, and the chunk arithmetic that
goes with them. Below the scheduler and ``Lockstep``, which both build batches."""
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from mstar.engine.chunking import ChunkRange
from mstar.graph.runtime.base import ColumnarEdgeSpecs, GraphRuntime
from mstar.model.submodule_base import CHUNK_LOOP_END, chunk_loop_name
from mstar.utils.ipc_format import ScheduleTPNode

# A rid a TP follower is told about but has already removed. Wire messages
# carry string rids, so a head from the leader can name a request this rank
# tore down; ``tp_rids`` maps those to this sentinel and ``pop_ready_rids``
# reads it as not-ready. No request ever owns it, and it is deliberately NOT
# tolerated anywhere else -- an unknown rid off the wire is expected, an
# unknown rid from local state is a bug and should still raise.
REMOVED_RID = -1


def runs_final_chunk(chunk: ChunkRange | None) -> bool:
    """Whether a row with this range ends its prompt. No range means the row
    runs whole, which is also its last chunk."""
    return chunk is None or chunk.is_final


def chunk_loop_stride(chunk: ChunkRange | None) -> int:
    """How far a row of a chunk-Loop walk moves its Loop: its chunk's stride,
    or to the end for a row that runs whole."""
    return CHUNK_LOOP_END if chunk is None else chunk.stride


def tp_chunk_ranges(message: ScheduleTPNode, rids: list[int]) -> dict[int, ChunkRange]:
    """The chunks the leader cut for ``message``'s rows, keyed by this rank's
    handles for them (``rids``, from ``tp_rids``). Lists that do not match the
    rows mean the ranks disagree on the step, so they raise."""
    if not message.chunk_starts:
        return {}
    return {
        rid: ChunkRange(start, end, total, stride)
        for rid, start, end, total, stride in zip(
            rids, message.chunk_starts, message.chunk_ends, message.chunk_totals,
            message.chunk_strides, strict=True,
        )
        if start >= 0 and rid != REMOVED_RID
    }


def unfinished_chunk_rids(chunk_ranges: Mapping[int, ChunkRange]) -> set[int]:
    """The rows of ``chunk_ranges`` whose prompt needs another chunk."""
    return {rid for rid, chunk in chunk_ranges.items() if not runs_final_chunk(chunk)}


class MidPrefillRow(NamedTuple):
    """A row mid-prefill that nothing popped: a row of the caller's in-flight
    step whose node needs another chunk, or a rebuild the log holds. ``start``
    is where its next chunk starts, 0 for a rebuild that has run none, and
    ``total`` its prompt's length."""
    request_id: int
    worker_graph_id: int
    graph_walk: str
    start: int
    total: int


class ContinuingRows(NamedTuple):
    """The rows of an in-flight step that run again in the step speculated
    after it, split by where their next step's inputs come from."""
    # from the step's loop-back outputs, as decode's do
    loop_back: list[int]
    # from the inputs held since their first chunk; the chunk Loop's tick
    # carries nothing, and the scheduler cuts their next chunk
    mid_prefill: list[MidPrefillRow]


@dataclass
class ScheduledBatch:
    """
    A batch of nodes ready to be executed.

    The term "ScheduledBatch" is a slight misnomer: this might exceed the batch
    size cap, so it may end up getting split into multiple batches at
    scheduling time.
    """
    node_name: str
    graph_walk: str
    # request_id -> worker_graph_id (for push-back on OOM)
    request_to_worker_graph: dict[int, int] = field(default_factory=dict)
    # The node's ready inputs for every rid in the batch, as the pop reported
    # them -- carried so the batch build never has to walk node.ready_signals
    # again. Columns, with the rid on each edge, so nothing here builds an
    # object per edge; the batch build walks them once
    # (``to_input_tensors``). The three methods below have to keep this and
    # ``request_to_worker_graph`` describing the same rids.
    input_edges: ColumnarEdgeSpecs = field(
        default_factory=ColumnarEdgeSpecs.empty
    )
    # The node's output edge names, recorded by the POP, or asked of the
    # runtime for rows no pop took (``_mid_prefill_rows``). Structural, so it is
    # the same for every rid and survives a split. Carried here so completion
    # never has to ask the runtime again -- and so the names come from the
    # GRAPH rather than from whatever tensors the model happened to return.
    output_signals: list[str] = field(default_factory=list)
    # ``ScheduleTPNode.spec_seq`` this batch came off the TP-follow FIFO with,
    # -1 otherwise. ``split_off_first`` / ``merge`` only ever see -1.
    tp_seq: int = -1
    # rid -> real walk, set iff ``graph_walk`` is a combined walk
    request_walks: dict[int, str] = field(default_factory=dict)
    # real walk -> its output edge names, under a combined walk
    walk_output_signals: dict[str, list[str]] = field(default_factory=dict)
    # rid -> its measured InputSeqLenInfo, or None for no length; a waiting row's inputs do not change
    row_lens: dict[int, Any] = field(default_factory=dict)
    # A TP follower's copy of ``ScheduleTPNode.read_check``: the leader's rows,
    # in its order, including those this rank could not read; None for no check
    read_check_rids: tuple[int, ...] | None = None
    # The order the step took its rows in when they didn't all fit and that
    # isn't the batch's own (a speculation chain lists its rows first);
    # `MicroScheduler.note_ran` gives out their turns in it.
    turn_order: list[int] | None = None
    # `rids_by_walk`, computed once; `merge` and `discard_rid` reset it
    _by_walk: dict[str, list[int]] | None = field(default=None, repr=False, compare=False)
    # Set once scheduled: rid -> the part of a chunked row's input this step
    # runs; and the rows whose node needs at least one more chunk.
    chunk_ranges: dict[int, ChunkRange] = field(default_factory=dict)
    incomplete_node_rids: set[int] = field(default_factory=set)
    # Set once scheduled too: the rows that rebuild their request from its
    # kept ids, which the graph runtime holds no node of.
    rebuild_rids: frozenset[int] = frozenset()
    # This node's walks that run as a chunk Loop on this worker, stamped when
    # the step is submitted (``StepExecutor._submit_gpu_step``): the graphs say,
    # not the scheduler, and every step that runs passes through there.
    chunk_loop_walks: frozenset[str] = frozenset()
    # this step is the run ready rows waited for (``ready_waits``)
    owed_run: bool = False


    def walk_of(self, rid: int) -> str:
        return self.request_walks.get(rid, self.graph_walk)

    def graph_rows(self) -> dict[int, int]:
        """``request_to_worker_graph`` without the rebuild rows: the rows the graph runtime holds."""
        if not self.rebuild_rids:
            return self.request_to_worker_graph
        return {rid: wg for rid, wg in self.request_to_worker_graph.items() if rid not in self.rebuild_rids}

    def loop_strides(self, rids: Iterable[int], repeats: int) -> list[int] | None:
        """``RouteInput.loop_strides`` for ``rids``, in that order: a step that
        stands for ``repeats`` iterations (``StepContext.repeats``) moves every
        row's loop by that many, a row of a chunk-Loop walk moves its Loop by its
        chunk (``chunk_loop_stride``), and any other row its loop by one. None for
        a step with no row in a chunk Loop, which is every other decode step.

        Keyed on the walk having a Loop rather than on the row having a chunk
        range: a row with no range ran whole, which also ends the Loop -- a
        walk with no token budget is never measured or cut.
        """
        if repeats > 1:
            return [repeats] * len(rids)
        walks = self.chunk_loop_walks
        # one lookup for a single-walk step outside a chunk Loop (decode)
        if not walks or (not self.request_walks and self.graph_walk not in walks):
            return None
        ranges = self.chunk_ranges
        walk_of = self.walk_of
        return [
            chunk_loop_stride(ranges.get(rid)) if walk_of(rid) in walks else 1
            for rid in rids
        ]

    def get_continuing_rows(
        self, node_name: str, walk_rids: list[int],
        tearing_down: Callable[[int], bool] | None,
        speculated: Collection[tuple[str, str]],
    ) -> ContinuingRows:
        """This step's rows that continue into a speculated step on
        ``node_name``: ``walk_rids`` (the target walk's rows) that loop back,
        and the rows mid-prefill whose (node, walk) speculates its next chunk
        (``speculate_mid_prefill``). ``tearing_down``: rows to leave out, or
        None when none can be."""
        held = self.incomplete_node_rids
        if not held and tearing_down is None:
            return ContinuingRows(walk_rids, [])
        loop_back = [
            r for r in walk_rids
            if r not in held and (tearing_down is None or not tearing_down(r))
        ]
        if not held or node_name != self.node_name:
            return ContinuingRows(loop_back, [])
        mid_prefill = [
            MidPrefillRow(
                r, self.request_to_worker_graph[r], self.walk_of(r),
                self.chunk_ranges[r].end, self.chunk_ranges[r].total,
            )
            for r in held
            if r in self.request_to_worker_graph
            and (tearing_down is None or not tearing_down(r))
            and (node_name, self.walk_of(r)) in speculated
        ]
        return ContinuingRows(loop_back, mid_prefill)


    def rids_by_walk(self) -> dict[str, list[int]]:
        """The batch's rids grouped by real walk, in batch order. Shared:
        callers must not modify it."""
        if self._by_walk is None:
            if not self.request_walks:
                self._by_walk = {self.graph_walk: list(self.request_to_worker_graph)}
            else:
                groups: dict[str, list[int]] = {}
                for rid in self.request_to_worker_graph:
                    groups.setdefault(self.request_walks[rid], []).append(rid)
                self._by_walk = groups
        return self._by_walk

    def wg_ids(self) -> list[int]:
        """``set_in_flight``'s worker graphs: one id for a step of one walk
        (a node in a walk lives in one worker graph), else one per rid."""
        rid_to_wg = self.graph_rows()
        if self.request_walks or not rid_to_wg:
            return list(rid_to_wg.values())
        return [next(iter(rid_to_wg.values()))]

    def discard_rid(self, rid: int) -> None:
        """Drop ``rid``'s row. Every removal goes through here, so the walk
        grouping never goes stale; ``input_edges`` is the caller's."""
        self.request_to_worker_graph.pop(rid, None)
        self.request_walks.pop(rid, None)
        self.row_lens.pop(rid, None)
        self._by_walk = None

    def relabel_if_uniform(self) -> None:
        """Label a combined batch whose rows share one real walk with that walk."""
        if not self.request_walks:
            return
        walks = set(self.request_walks.values())  # rows and request_walks share keys
        if len(walks) == 1:
            self.graph_walk = walks.pop()
            # the merged batch kept its first walk's names, which may not be this walk's
            self.output_signals = self.walk_output_signals.get(
                self.graph_walk, self.output_signals,
            )
            self.request_walks = {}
            self.walk_output_signals = {}

    def merge(self, other: "ScheduledBatch") -> None:
        """Fold ``other``'s requests in, ours first — they have waited longer."""
        assert (self.node_name, self.graph_walk) == (
            other.node_name, other.graph_walk
        ), "only batches for the same (node, walk) share a backlog entry"
        assert self.request_to_worker_graph.keys().isdisjoint(
            other.request_to_worker_graph
        ), "a rid is in both halves of a merge; its node was popped twice"
        self.request_to_worker_graph.update(other.request_to_worker_graph)
        self.input_edges.extend(other.input_edges)
        self.request_walks.update(other.request_walks)
        self.walk_output_signals.update(other.walk_output_signals)
        self.row_lens.update(other.row_lens)
        self._by_walk = None

    def split_off_first(
        self, bs: int | None, exclude_rids: set[int] | None = None
    ) -> "tuple[ScheduledBatch | None, ScheduledBatch | None]":
        """
        Return the first ScheduledBatch, as well as the remainder (None if
        all requests have been scheduled)
        """
        exclude_rids = exclude_rids or {}
        rids = self.request_to_worker_graph.keys() # as of py3.7, preserves ordering

        if bs is None:
            bs = len(rids)
        keep_rids = [rid for rid in rids if rid not in exclude_rids]
        exclude_rids = [rid for rid in rids if rid in exclude_rids]

        if len(keep_rids) <= bs and not exclude_rids:
            # The whole point of the columnar block: the common case hands the
            # batch back untouched, so nothing is re-sliced per edge.
            return self, None
        if not keep_rids:
            return None, self

        taken = keep_rids[:bs]
        left = keep_rids[bs:] + exclude_rids
        return ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            request_to_worker_graph={
                rid: self.request_to_worker_graph[rid] for rid in taken
            },
            input_edges=self.input_edges.select_rids(set(taken)),
            output_signals=self.output_signals,
            request_walks=self._walks_for(taken),
            row_lens=self._row_lens_for(taken),
            # shared, not trimmed to the walks each half keeps: an extra entry
            # costs nothing, since outputs are only looked up for walks present
            walk_output_signals=self.walk_output_signals,
        ), ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            request_to_worker_graph={
                rid: self.request_to_worker_graph[rid] for rid in left
            },
            input_edges=self.input_edges.select_rids(set(left)),
            output_signals=self.output_signals,
            request_walks=self._walks_for(left),
            row_lens=self._row_lens_for(left),
            walk_output_signals=self.walk_output_signals,
        )

    def split_off_rids(
        self, rids: Sequence[int],
    ) -> "tuple[ScheduledBatch, ScheduledBatch | None]":
        """``rids`` (all of them in this batch), in that order, and the
        remainder in batch order (None if nothing is left)."""
        if len(rids) == len(self.request_to_worker_graph):
            if list(rids) == list(self.request_to_worker_graph):
                return self, None
        taken = set(rids)
        left = [rid for rid in self.request_to_worker_graph if rid not in taken]
        return self._sub_batch(list(rids)), self._sub_batch(left) if left else None

    def _sub_batch(self, rids: list[int]) -> "ScheduledBatch":
        return ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            request_to_worker_graph={rid: self.request_to_worker_graph[rid] for rid in rids},
            input_edges=self.input_edges.select_rids(set(rids)),
            output_signals=self.output_signals,
            request_walks=self._walks_for(rids),
            row_lens=self._row_lens_for(rids),
            walk_output_signals=self.walk_output_signals,
        )

    def _walks_for(self, rids: list[int]) -> dict[int, str]:
        if not self.request_walks:
            return {}
        return {rid: self.request_walks[rid] for rid in rids}

    def _row_lens_for(self, rids: list[int]) -> dict[int, Any]:
        if not self.row_lens:
            return {}
        return {rid: self.row_lens[rid] for rid in rids if rid in self.row_lens}

    def __len__(self):
        return len(self.request_to_worker_graph)


class PopReadyResult(NamedTuple):
    wg_ids: dict[int, int]
    edge_specs: ColumnarEdgeSpecs
    # Flat, not per-rid: the node's output edge names are structural, so every
    # rid in the batch shares them.
    output_signals: tuple[str, ...]
    # rid -> real walk, and real walk -> its output edge names, under a combined walk
    request_walks: dict[int, str] = {}
    walk_output_signals: dict[str, list[str]] = {}


def cap_chunk_loops(
    batch: ScheduledBatch, runtime: GraphRuntime, loop_walks_by_node: Mapping[str, frozenset[str]],
) -> None:
    """Cap the chunk Loop of each row of ``batch`` on its first chunk at its
    whole input's length, which is what ends the Loop. A TP follower caps from
    the leader's chunks: the cap decides on which step the Loop ends, so every
    rank of the instance has to set the same one."""
    node = batch.node_name
    loop_walks = loop_walks_by_node.get(node, ())
    capped = [
        rid for rid, chunk in batch.chunk_ranges.items()
        if chunk.is_first and rid not in batch.rebuild_rids and batch.walk_of(rid) in loop_walks
    ]
    if capped:
        walks = [batch.walk_of(rid) for rid in capped]
        runtime.set_loop_caps(
            node, capped, walks, [chunk_loop_name(node, walk) for walk in walks],
            [batch.chunk_ranges[rid].total for rid in capped],
        )
