import functools
import itertools
import logging
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.graph.runtime.base import ColumnarEdgeSpecs, GraphRuntime, ReadyNodeSpec, RequestWalks
from mstar.model.submodule_base import CHUNK_LOOP_END, InputMetadata, StepOrder, chunk_loop_name
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode
from mstar.worker.batch_builder import (
    BatchBuilderType,
    BatchBuildRequest,
    BatchBuildResult,
    LineOrder,
    StepCandidates,
    StepPick,
    make_batch_builder,
)
from mstar.worker.engine_manager import EngineManager
from mstar.worker.node_manager_utils import RequestStateManager

logger = logging.getLogger(__name__)

# A rid a TP follower is told about but has already removed. Wire messages
# carry string rids, so a head from the leader can name a request this rank
# tore down; ``tp_rids`` maps those to this sentinel and ``pop_ready_rids``
# reads it as not-ready. No request ever owns it, and it is deliberately NOT
# tolerated anywhere else -- an unknown rid off the wire is expected, an
# unknown rid from local state is a bug and should still raise.
REMOVED_RID = -1

# Below every turn and arrival: a sort key offset that puts a row ahead of them all
_AHEAD = 1 << 62


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: int
    worker_graph_id: int
    graph_walk: str


class MidPrefillRow(NamedTuple):
    """A row of the caller's in-flight step whose node needs another chunk:
    the end of the chunk in flight, where its next one starts, and its
    prompt's length."""
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
    # The node's output edge names, recorded by the POP. Structural, so it is
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
    # This node's walks that run as a chunk Loop on this worker, stamped when
    # the step is submitted (``Worker._submit_gpu_step``): the graphs say,
    # not the scheduler, and every step that runs passes through there.
    chunk_loop_walks: frozenset[str] = frozenset()


    def walk_of(self, rid: int) -> str:
        return self.request_walks.get(rid, self.graph_walk)

    def loop_strides(self, rids: Iterable[int]) -> list[int] | None:
        """``RouteInput.loop_strides`` for ``rids``, in that order: a row of a
        chunk-Loop walk moves its Loop by its chunk (``chunk_loop_stride``),
        and any other row its loop by one. None for a step with no row in a
        chunk Loop, which is every decode step.

        Keyed on the walk having a Loop rather than on the row having a chunk
        range: a row with no range ran whole, which also ends the Loop -- a
        walk with no token budget is never measured or cut.
        """
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


class SchedulingType(Enum):
    ROUND_ROBIN = "round_robin"
    # TODO: priority. It used to key off a per-engine-type table, which no
    # longer exists — every node runs on the same engine now. The replacement
    # is for the model to declare a (node, graph_walk) priority, since only it
    # knows which walk is latency-sensitive. Worth weighing against
    # head-of-line blocking: a busy high-priority walk starves the rest.

class PopReadyResult(NamedTuple):
    wg_ids: dict[int, int]
    edge_specs: ColumnarEdgeSpecs
    # Flat, not per-rid: the node's output edge names are structural, so every
    # rid in the batch shares them.
    output_signals: tuple[str, ...]
    # rid -> real walk, and real walk -> its output edge names, under a combined walk
    request_walks: dict[int, str] = {}
    walk_output_signals: dict[str, list[str]] = {}


class ChainPick(NamedTuple):
    """A speculation chain's next step at its target, as ``pick_chain_step``
    chose it; ``take_chain_step`` pops the rows it took off the queues."""
    key: tuple[str, str]
    pick: StepPick
    # the target's ready-queue scan the pick ran over
    specs: list
    # the caller's rows mid-prefill that the step took (``pick_chain_step``)
    mid_prefill: Sequence[MidPrefillRow] = ()


def runs_final_chunk(chunk: ChunkRange | None) -> bool:
    """Whether a row with this range ends its prompt. No range means the row
    runs whole, which is also its last chunk."""
    return chunk is None or chunk.is_final


def chunk_loop_stride(chunk: ChunkRange | None) -> int:
    """How far a row of a chunk-Loop walk moves its Loop: its chunk's stride,
    or to the end for a row that runs whole."""
    return CHUNK_LOOP_END if chunk is None else chunk.stride


def runs_first_chunk(chunk: ChunkRange) -> bool:
    """Whether ``chunk`` is its prompt's first, which caps its chunk Loop. A
    first chunk's stride runs from 0 even past a cached prefix; a later one's
    is its own span, which starts past 0."""
    return chunk.stride == chunk.end


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


def _pre_existing_tokens(seq_lens: Collection[int], n: int) -> int:
    """The tokens a speculation chain's own ``n`` rows will run, as the
    submodule estimates them (``speculative_input_seq_len``): their inputs do
    not exist yet."""
    # A row with no estimate (-1) guesses one token, a decode row.
    return sum(s if s >= 0 else 1 for s in seq_lens) + n - len(seq_lens)


class MicroScheduler:
    """
    Simple MVP scheduler: scans all worker graph queues for ready nodes,
    groups by node name, returns the highest-priority group.
    """

    # Seconds to wait before retrying a held request after OOM
    HOLD_BACKOFF_SECONDS = 0.05

    def __init__(
        self, engine_manager: EngineManager,
        sched_type=SchedulingType.ROUND_ROBIN,
        parallel_leader_nodes: set[str] | None = None,
        max_consec_tp_follower_batches: int = 1,
        batch_builder_type: BatchBuilderType = BatchBuilderType.FIFO,
        combined_walk_of: dict[tuple[str, str], str] | None = None,
        speculation_mixes_walks: Mapping[str, Any] | None = None,
    ):
        self.engine_manager = engine_manager
        self._orders: dict[tuple[str, str], StepOrder] = {}
        # (node, real walk) -> combined walk; scheduling keys use the combined one
        self.combined_walk_of = combined_walk_of or {}
        self._walks_of_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (node, walk), combined in self.combined_walk_of.items():
            self._walks_of_key[(node, combined)].append(walk)
        # (node, walk) -> whether its speculation chain merges the other walks
        # of its combined walk: the config's settings, then each default as
        # `speculation_mixes` first decides it
        self._mixing_chains = self._parse_speculation_mixes_walks(
            speculation_mixes_walks or {},
        )
        self._max_bs: dict[tuple[str, str], int | None] = {}
        self._walk_caps_of_key: dict[tuple[str, str], dict[str, int | None]] = {}
        self._row_caps_of_key: dict[tuple[str, str], dict[str, int]] = {}
        self.batch_builder = make_batch_builder(batch_builder_type)
        self.batch_number = 0
        # The graph runtime, installed by the worker. Interning and the
        # (walk, node) -> worker graph index both live there.
        self.runtime: "GraphRuntime | None" = None
        # node -> the walks whose prefill runs as a chunk Loop there; the
        # worker installs the ones its graphs build
        self.chunk_loop_walks: dict[str, frozenset[str]] = {}
        # Looks up the handle for a wire request id; the worker installs the
        # runtime's. Only the TP-follow messages need it -- everything else
        # here already holds handles. Defaults to the identity so a scheduler
        # driven directly with its own keys (tests) needs no table.
        self.rid_of: Callable[[str], int | None] = lambda r: r
        # Resolves a tensor uuid, for row lengths read from popped inputs; the
        # worker installs the tensor manager's. None skips the room check.
        self.get_tensor: Callable[[int], Any] | None = None
        # (rid, node) -> when its prompt's first chunk was cut, for serving
        # prompts in start order. Read only for a row whose chunk Loop has
        # started, so the next prompt's first chunk simply overwrites it.
        self.chunk_started: dict[tuple[int, str], int] = {}
        self._chunk_starts = itertools.count()
        # (rid, node) -> cached prefix a measured, not yet chunked row starts past
        self.cached_prefix: dict[tuple[int, str], int] = {}
        self._warned_removed_rid = False
        self.sched_type = sched_type

        # RIDs that have failed but have not gone through the cleanup procedure;
        # these cannot be scheduled (unless this is a TP follower node)
        self.failed_rids: set[int] = set()
        # rid -> message, for requests a resource declared unservable during a
        # readiness check. Drained by the worker, which reports them onward.
        self.admit_errors: dict[int, str] = {}
        # Rids an input read of which failed on this rank. A TP follower never
        # gets those inputs, so it builds the leader's step without the row and
        # the step's read check takes it out on every rank (see
        # ``Worker._drop_rows_whose_read_failed``).
        self.read_failed_rids: set[int] = set()
        self._warned_read_check = False

        # lockstep-parallel (TP / SP instance) scheduling
        self.parallel_leader_nodes = parallel_leader_nodes
        self.tp_batches_pending_schedule = deque()
        self.num_consec_tp_follower_batches = 0
        self.max_consec_tp_follower_batches = max_consec_tp_follower_batches

        # Batches already assembled and waiting their turn: what is left of a
        # ready set too big for one step. Taken before scanning the queues, so
        # a split batch finishes before anything new starts.

        # (node, graph walk) -> ScheduledBatch
        self.backlog: dict[tuple[str, str], ScheduledBatch] = {}

        self.node_and_walk_to_last_batch_num = {}
        # scheduling key -> rid -> its turn there: a counter over every row of
        # every admitted step, given out in the order the step took its rows
        # (`note_ran`), so sorting by it is first come, first served. Kept per
        # key: a row that runs another node meanwhile has still waited here.
        self.last_ran: dict[tuple[str, str], dict[int, int]] = {}
        self._turns = 0
        # scheduling key -> the first turn of its latest admitted step: the
        # rows at or past it hold the key's step
        self._latest_step: dict[tuple[str, str], int] = {}
        # rid -> the order requests arrived at this worker in (`note_arrival`)
        self.arrival: dict[int, int] = {}
        self._arrivals = 0
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[int, float] = {}
        # node -> rids waiting for capacity there, which readiness answers without asking the
        # engine until `Engine.admit_waiting` admits them
        self.parked: dict[str, set[int]] = {}
        # (node, graph walk) -> most rows a batch may take while its admits are
        # refused with nothing to evict; see `limit_batch`
        self.batch_limits: dict[tuple[str, str], int] = {}
        # (node, graph walk) -> most rows the KV pool can hold at once even
        # with nothing else resident; see `cap_batch`
        self.batch_ceilings: dict[tuple[str, str], int] = {}
        # (node, graph walk) -> the resource whose refusal set its cap, if named
        self.ceiling_resources: dict[tuple[str, str], str | None] = {}
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: dict[int, None] = {}

        # WIRE STRING -> number of committed (ZMQ received) tp follow batches
        # still queued. On the fail/abort path we drain these before ACKing
        # READS_DONE so the worker graph queues aren't torn down while a follow
        # still needs to pop from them.
        #
        # Keyed by the wire string, not the handle, unlike everything else
        # here: a ScheduleTPNode can arrive before the NEW_REQUEST that mints
        # the rid's handle, and a batch that went uncounted in that window is
        # exactly the one the drain must wait for. Purged by ``clear_wire_rid``.
        self.pending_tp_follow_count: dict[str, int] = defaultdict(int)

        # pending resident deltas for TP follow nodes
        self._pending_resident_deltas: dict[str, OffloadDelta] = {}
        self._last_resident_delta: int = -1
        # popped, not settled: a settled head can still fail to build, and its removals wait for its admit
        self._last_popped_tp_seq: int = -1

    def _parse_speculation_mixes_walks(
        self, setting: Mapping[str, Any],
    ) -> dict[tuple[str, str], bool]:
        """The serving config's ``speculation_mixes_walks``: per node, one bool
        for every walk of its combined walks, or a mapping of walk -> bool."""
        walks_of_node: dict[str, set[str]] = defaultdict(set)
        for node, walk in self.combined_walk_of:
            walks_of_node[node].add(walk)
        out: dict[tuple[str, str], bool] = {}
        for node, value in setting.items():
            if node not in walks_of_node:
                raise ValueError(
                    f"speculation_mixes_walks names {node!r}, which has no combined walk"
                )
            per_walk = value if isinstance(value, Mapping) else dict.fromkeys(walks_of_node[node], value)
            for walk, mixes in per_walk.items():
                if walk not in walks_of_node[node]:
                    raise ValueError(
                        f"speculation_mixes_walks names walk {walk!r} of {node!r}, "
                        "which is in none of its combined walks"
                    )
                if not isinstance(mixes, bool):
                    raise ValueError(
                        f"speculation_mixes_walks for {node}/{walk}: {mixes!r} is not a bool"
                    )
                out[(node, walk)] = mixes
        return out

    def speculation_mixes(self, node_name: str, graph_walk: str) -> bool:
        """Whether a speculation chain on ``(node_name, graph_walk)`` merges
        ready rows of the other walks of its combined walk. Off, the chain
        takes only its own walk's rows, and another walk's ready rows make it
        yield; steps then mix walks only when the scheduler builds one fresh,
        which plans inline.

        Unless the config says otherwise, on wherever the combined walk has a
        token budget: chunked prompts then ride in decode steps rather than
        decode sitting out every chunk step. Without one a merged prompt runs
        whole and stretches the decode step it joins."""
        key = (node_name, graph_walk)
        mixes = self._mixing_chains.get(key)
        if mixes is None:
            combined = self.combined_walk_of.get(key)
            mixes = combined is not None and self.engine_manager.get_engine(
                node_name,
            ).get_max_batch_tokens(node_name, combined) is not None
            self._mixing_chains[key] = mixes
        return mixes

    @property
    def last_consumed_tp_seq(self) -> int:
        """Highest leader step this rank has taken off the FIFO. What orders a
        forwarded removal against the step stream — see
        ``Worker._removal_step_reached``."""
        return self._last_popped_tp_seq

    def _node_rr_order(
        self, ready: dict[tuple[str, str], list[ReadyNodeEntry]],
    ) -> list[tuple[str, str]]:
        """The (node, walk) keys with a ready row, least recently scheduled
        first; ties go to the first scanned."""
        return sorted(
            ready,
            key=lambda key: self.node_and_walk_to_last_batch_num.get(key, 0),
        )

    def note_ran(self, batch: ScheduledBatch) -> None:
        """Give ``batch``'s rows their turn under its scheduling key, in the
        order its step took them. Called once the step's admit has taken it,
        so the rows of a refused step keep their place in line."""
        key = self._key(batch.node_name, batch.graph_walk)
        turns = self.last_ran.get(key)
        if turns is None:
            turns = self.last_ran[key] = {}
        rows = batch.request_to_worker_graph
        if batch.turn_order is not None:
            listed = [rid for rid in batch.turn_order if rid in rows]
            if len(listed) < len(rows):
                seen = set(listed)
                listed += [rid for rid in rows if rid not in seen]
            rows = listed
        first = self._turns + 1
        self._turns += len(rows)
        turns.update(zip(rows, range(first, self._turns + 1), strict=True))
        self._latest_step[key] = first

    def last_in_line(self, node_name: str, graph_walk: str, rids: Iterable[int]) -> int:
        """Of ``rids``, the one a step of ``(node_name, graph_walk)`` would
        reach last: the line a capped step's pick reads."""
        return max(rids, key=self._line_order(self._key(node_name, graph_walk)).key)

    def note_arrival(self, rid: int) -> None:
        """A request reached this worker; a request with several partitions
        here keeps its first arrival."""
        if rid not in self.arrival:
            self._arrivals += 1
            self.arrival[rid] = self._arrivals

    def _order_of(self, key: tuple[str, str]) -> StepOrder:
        """``key``'s `StepOrder`, as its submodule declares it. A combined
        walk's step holds its walks' rows, so it takes turns if any of its
        walks does."""
        order = self._orders.get(key)
        if order is None:
            node, walk = key
            engine = self.engine_manager.get_engine(node)
            orders = [engine.get_step_order(node, real) for real in self._walks_of_key.get(key) or (walk,)]
            order = StepOrder.TURNS if StepOrder.TURNS in orders else StepOrder.ARRIVAL
            self._orders[key] = order
        return order

    def _line_order(self, key: tuple[str, str]) -> LineOrder:
        """``StepCandidates.line_order`` for ``key``."""
        turns = self.last_ran.get(key, {})
        arrived = self.arrival.get
        if self._order_of(key) is StepOrder.TURNS:
            # longest-waiting first; a row new to the key ahead of every row
            # that ran here, the earliest request first
            def waited(rid: int) -> int:
                turn = turns.get(rid)
                return arrived(rid, 0) - _AHEAD if turn is None else turn
            return LineOrder(waited, strict=False)
        holding = self._latest_step.get(key, _AHEAD)

        # the rows holding the step, then the earliest request
        def arrival(rid: int) -> int:
            return arrived(rid, 0) - (_AHEAD if turns.get(rid, 0) >= holding else 0)
        return LineOrder(arrival, strict=True)

    def hold_requests(self, request_ids: list[int]) -> None:
        """Put requests on hold for a brief backoff period after OOM."""
        deadline = time.monotonic() + self.HOLD_BACKOFF_SECONDS
        for rid in request_ids:
            self.held_until[rid] = deadline

    def limit_batch(self, node_name: str, graph_walk: str, rows: int) -> None:
        """Keep ``(node_name, graph_walk)``'s batches to ``rows`` (at least
        one) after an admit refused with nothing to evict, until a step of it
        admits. ``rows`` is what the free pages held, so one refusal sizes it.

        Holding the blamed rows alone does not shrink the retry: the ready
        queue refills the batch to the engine's cap, so a cap's worth of rows
        that cannot fit together is refused on every pass.
        """
        key = (node_name, graph_walk)
        limit = max(rows, 1)
        self.batch_limits[key] = min(limit, self.batch_limits.get(key, limit))

    def could_refill(self, node_name: str, graph_walk: str, batch_rids: set[int]) -> bool:
        """Whether rows of ``(node_name, graph_walk)``'s scheduling key other
        than ``batch_rids`` wait to run, so they could refill a retry of that
        batch to its size. When none do, the batch took every row it could,
        and holding the blamed one is what makes the retry smaller.

        Asked of the graph, not the engine: an engine readiness check would
        start reloading an offloaded row. Offloaded rows are left out; they
        come back only once their pages fit. Refusal path only.
        """
        key = self._key(node_name, graph_walk)
        if self._live_backlog(key) - batch_rids:
            return True
        exclude = self.failed_rids.union(self.pending_removes, self.held_until, batch_rids)
        engine = self.engine_manager.get_engine(node_name)
        return any(
            not engine.is_offloaded(spec.node_name, rid)
            for spec in self._ready_specs(exclude, key, None)
            if spec.node_name == node_name
            for rid in spec.rids
        )

    def cap_batch(
        self, node_name: str, graph_walk: str, rows: int, resource: str | None = None,
    ) -> None:
        """Keep ``(node_name, graph_walk)``'s batches to ``rows`` (at least
        one), the most the KV pool ``resource`` holds even with nothing else
        resident.

        Unlike `limit_batch`, a step that admits at this size does not lift it:
        the next full batch could never fit either, so it would be refused,
        and with a victim to evict, evict one for nothing, on every other
        step. Lifted by a step that admits fewer rows than the cap, where the
        queue no longer fills it and the next batch may be of smaller rows,
        or by a teardown that hands pages of ``resource`` back (see
        `lift_batch_caps`).
        """
        key = (node_name, graph_walk)
        cap = max(rows, 1)
        if cap <= self.batch_ceilings.get(key, cap):
            self.batch_ceilings[key] = cap
            self.ceiling_resources[key] = resource

    def lift_batch_caps(self, resources: Collection[str]) -> None:
        """A teardown handed pages of ``resources`` back: lift the `cap_batch`
        caps their refusals set, and a cap whose refusal named no resource.

        A cap is sized from the rows refused then. On a busy walk every step
        is at the cap, so a smaller step never comes to lift it, and it would
        outlast the rows that set it.
        """
        for key, resource in list(self.ceiling_resources.items()):
            if resource is None or resource in resources:
                del self.batch_ceilings[key], self.ceiling_resources[key]

    def clear_batch_limit(
        self, node_name: str, graph_walk: str, rows: int, walks: Iterable[str] = (),
    ) -> None:
        """A step of ``rows`` rows of ``(node_name, graph_walk)`` admitted:
        lift its `limit_batch` limit, and its `cap_batch` cap if the step was
        smaller than both. ``walks`` are the real walks the step ran rows of.

        A step of a real walk lifts its combined walk's limit too: a mixed
        batch refused at two rows leaves a limit of one, which no mixed step
        can meet, so no step under the combined label would ever lift it. A
        mixed step lifts the limits of the walks it ran, since the combined
        walk's own cap sized it; their caps stay, as the combined cap, not
        their queues, set how many of their rows it took.
        """
        keys = [(node_name, graph_walk)]
        combined = self.combined_walk_of.get((node_name, graph_walk))
        if combined is not None:
            keys.append((node_name, combined))
        for key in keys:
            limit = self.batch_limits.pop(key, None)
            ceiling = self.batch_ceilings.get(key)
            if ceiling is not None and rows < min(ceiling, limit or ceiling):
                del self.batch_ceilings[key], self.ceiling_resources[key]
        for walk in walks:
            if walk != graph_walk:
                self.batch_limits.pop((node_name, walk), None)

    def tp_rids(self, message: ScheduleTPNode) -> list[int]:
        """``ScheduleTPNode`` crosses the wire, so its rids are strings; these
        are this rank's handles for them, in the leader's order. A rid this rank
        does not know maps to ``REMOVED_RID``."""
        return [
            REMOVED_RID if (handle := self.rid_of(r)) is None else handle
            for r in message.request_ids
        ]

    def register_tp_follow(
        self, message: ScheduleTPNode
    ):
        self.tp_batches_pending_schedule.append(message)
        # Count every rid the leader named, including ones this rank cannot
        # resolve yet -- ``tp_rids`` would flatten those to REMOVED_RID and
        # lose the very batch the drain has to wait for.
        for rid in message.request_ids:
            self.pending_tp_follow_count[rid] += 1

    # TP-follow FIFO accessors for the follower's async path (head only).

    def peek_tp_follow(self) -> ScheduleTPNode | None:
        if not self.tp_batches_pending_schedule:
            return None
        return self.tp_batches_pending_schedule[0]

    def _apply_resident_delta(self, node_name: str, new_delta: OffloadDelta | None=None) -> bool:
        if node_name not in self._pending_resident_deltas:
            self._pending_resident_deltas[node_name] = OffloadDelta.new()
        delta = self._pending_resident_deltas[node_name]
        if new_delta is not None:
            delta.extend(new_delta)

        if not len(delta):
            return True

        engine = self.engine_manager.get_engine(node_name)
        return engine.apply_resident_delta(node_name, delta)

    def _apply_delta_from_message(self, message: ScheduleTPNode) -> bool:
        if self._last_resident_delta < message.spec_seq:
            self._last_resident_delta = message.spec_seq
            return self._apply_resident_delta(message.node_name, message.resident_delta)
        return self._apply_resident_delta(message.node_name)

    def settle_tp_follow_delta(self, caught_up: bool) -> bool:
        """Replay the front head's resident delta; True once every move landed.

        The one seam both consumers of this FIFO go through — the serial path
        below and the async follower's ``_try_follow_speculation`` — so neither
        can build a step against a half-replayed page state. The async path used
        to skip this entirely and read the difference as "rid not ready", which
        it then polled on for ever.

        ``caught_up`` means every step this rank took before the head has committed.
        """
        head = self.peek_tp_follow()
        if head is None:
            return True
        if not self._apply_delta_from_message(head):
            return False
        if not head.prefix_hits and not head.page_trails:
            return True
        engine = self.engine_manager.get_engine(head.node_name)
        try:
            engine.check_page_trails(head.node_name, head.graph_walk, head.spec_seq, head.page_trails)
            return engine.adopt_cached_prefix(head, caught_up)
        except Exception as exc:
            # The error would repeat on every pass over this head, and failing its requests would
            # leave the other ranks stuck in its collective.
            raise SystemExit(f"TP ranks diverged: {exc}") from exc

    def pop_tp_follow_head(self) -> ScheduleTPNode:
        # Sole exit for a queued follow batch: every consumer (the serial path
        # and the async follower's build / drop / void paths) pops here, so the
        # drain refcount is discharged in one place.
        message = self.tp_batches_pending_schedule.popleft()
        self._apply_delta_from_message(message)
        self._last_popped_tp_seq = max(self._last_popped_tp_seq, message.spec_seq)
        for rid in message.request_ids:
            if rid not in self.pending_tp_follow_count:
                continue
            self.pending_tp_follow_count[rid] -= 1
            if self.pending_tp_follow_count[rid] <= 0:
                self.pending_tp_follow_count.pop(rid, None)
        return message

    def pop_ready_rids(
        self, request_state: RequestStateManager,
        node_name: str, graph_walk: str, request_ids: list[int],
        walks: list[str] | None = None, walk_idx: list[int] | None = None,
    ) -> PopReadyResult | None:
        """Pop ``node_name`` for exactly ``request_ids``, all or none.
        Checked for every rid before anything is popped, so the caller
        retries later for a partially ready set. Under a combined
        ``graph_walk``, ``request_ids[i]`` runs ``walks[walk_idx[i]]``."""
        if not request_ids:
            return PopReadyResult({}, ColumnarEdgeSpecs.empty(), ())
        # Engine readiness first: pop_rids treats it as a prerequisite, and it
        # is all-or-nothing too, so one not-ready rid leaves the set intact.
        node_partition = request_state.get_partition_for_node(node_name)
        for rid in request_ids:
            # ``tp_rids`` hands us REMOVED_RID for a request this rank has
            # already torn down. There is no forward-pass state to ask about,
            # and popping is all-or-none, so the whole set waits -- the same
            # answer an unresolvable rid has always been meant to give.
            # ``get_fwd_info`` indexes rather than gets, so asking it would
            # raise KeyError and kill the follower's main loop.
            if rid == REMOVED_RID:
                # Warned once: deferring is correct for a rid that is merely
                # gone, but if the head can NEVER be satisfied the follower
                # waits here forever, and a silent stall is far harder to
                # diagnose than the KeyError this replaced.
                if not self._warned_removed_rid:
                    self._warned_removed_rid = True
                    logger.warning(
                        "Node %s walk %s: a TP head names a request this rank "
                        "has already removed; deferring the batch. If the "
                        "follower stops making progress, this is why.",
                        node_name, graph_walk,
                    )
                return None
            fwd_info = request_state.get_fwd_info(rid, node_partition)
            if not self._check_ready(node_name, rid, fwd_info,  allow_reload=False):
                return None

        # one walk: the plain pop, with no per-rid walk index to build
        popped = (
            self.runtime.pop_walk_rids(
                node_name, RequestWalks(request_ids, walks, walk_idx), check_ready=True,
            ) if walks
            else self.runtime.pop_rids(node_name, graph_walk, request_ids, check_ready=True)
        )
        if popped is None:
            return None
        rids = popped.wg_ids.keys
        result = PopReadyResult(
            dict(zip(rids, popped.wg_ids.values, strict=True)),
            popped.input_edges, popped.output_signals,
        )
        if walks:
            result = result._replace(
                request_walks={rid: walks[w] for rid, w in zip(rids, popped.rid_walk_idx, strict=True)},
                walk_output_signals={
                    walk: list(signals)
                    for walk, signals in zip(walks, popped.walk_output_signals, strict=True)
                },
            )

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[self._key(node_name, graph_walk)] = self.batch_number
        return result

    def _try_schedule_tp_follow(
        self, request_state: RequestStateManager,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        if len(self.tp_batches_pending_schedule) == 0:
            return
        first_tp_node: ScheduleTPNode = self.tp_batches_pending_schedule[0]
        head_key = self._key(first_tp_node.node_name, first_tp_node.graph_walk)
        if exclude_target is not None and head_key == self._key(*exclude_target):
            return
        if self.num_consec_tp_follower_batches >= self.max_consec_tp_follower_batches and \
                self.has_ready_excluding(request_state, head_key):
            return
        # Check readiness for every rid to pop all-or-none. Use the
        # leader's graph walk. No step of the node is in flight on this path.
        if not self.settle_tp_follow_delta(caught_up=True):
            return # every pending move has to land before the step is built
        rids = self.tp_rids(first_tp_node)
        # before anything is popped: lists that do not match the rows raise
        chunk_ranges = tp_chunk_ranges(first_tp_node, rids)
        unread = self.unread_rows(first_tp_node, rids)
        popped = self.pop_ready_rids(
            request_state, first_tp_node.node_name, first_tp_node.graph_walk,
            [r for r in rids if r not in unread],
            walks=first_tp_node.walks,
            walk_idx=[
                w for r, w in zip(rids, first_tp_node.walk_idx, strict=True) if r not in unread
            ] if first_tp_node.walks else None,
        )
        if popped is None:
            self.warn_unchecked_read_failure(first_tp_node, rids)
            return
        request_to_worker_graph, input_edges, output_signals, request_walks, walk_signals = popped
        if unread:
            request_to_worker_graph, request_walks = self.with_unread_rows(
                first_tp_node, rids, unread, request_to_worker_graph, request_walks,
            )

        self.pop_tp_follow_head()

        batch = ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            request_to_worker_graph=request_to_worker_graph,
            input_edges=input_edges,
            output_signals=output_signals,
            tp_seq=first_tp_node.spec_seq,
            request_walks=request_walks,
            walk_output_signals=walk_signals,
            read_check_rids=tuple(rids) if first_tp_node.read_check else None,
            # the leader's chunks, so the chunk Loop ends on the same step here
            chunk_ranges=chunk_ranges,
            incomplete_node_rids=unfinished_chunk_rids(chunk_ranges),
        )
        self.cap_chunk_loops(batch)
        return batch

    def unread_rows(self, head: ScheduleTPNode, rids: list[int]) -> set[int]:
        """The rows of ``head`` whose input read failed on this rank, which a
        follower puts in the step it builds with no inputs, for the step's read
        check to take out on every rank. Only under a read check: without one no
        rank takes the row out, so the follower pops it like any other (the
        failed read may have been for another step)."""
        failed = self.read_failed_rids
        if not head.read_check or not failed:
            return set()
        return {r for r in rids if r in failed}

    def with_unread_rows(
        self, head: ScheduleTPNode, rids: list[int], unread: set[int],
        request_to_worker_graph: dict[int, int], request_walks: dict[int, str],
    ) -> tuple[dict[int, int], dict[int, str]]:
        """``request_to_worker_graph`` and ``request_walks`` with ``unread``'s
        rows put back, in the leader's order: the step has rank 0's rows on
        every rank until its read check, so it is built, submitted, voided and
        settled alike. The unread rows were never popped."""
        walk_of = dict(zip(rids, (head.walks[w] for w in head.walk_idx), strict=True)) if head.walks else {}
        request_walks = dict(request_walks)
        rows = {}
        for rid in rids:
            if rid not in unread:
                rows[rid] = request_to_worker_graph[rid]
                continue
            walk = walk_of.get(rid, head.graph_walk)
            rows[rid] = self.runtime.get_worker_graph_id_for_node(head.node_name, walk)
            if walk_of:
                request_walks[rid] = walk
        return rows, request_walks

    def warn_unchecked_read_failure(self, head: ScheduleTPNode, rids: list[int]) -> None:
        """Once: a head with no read check waits on a row whose read failed
        here, which it never gets. Only a read rank 0 did not mark gets here."""
        if self._warned_read_check or head.read_check or not self.read_failed_rids:
            return
        if self.read_failed_rids.isdisjoint(rids):
            return
        self._warned_read_check = True
        logger.error(
            "Node %s walk %s: a TP head with no read check names a request whose "
            "input read failed on this rank; the instance waits on it",
            head.node_name, head.graph_walk,
        )


    def get_next_batch(
        self,
        request_state: RequestStateManager,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        """
        One step's worth: a pending TP follow batch if there is one, else
        the batch builder's pick from one (node, walk)'s backlog and ready rows.

        A (node, walk) with a backlog — the remainder of a ready set too big
        for one forward — goes first, so a split set drains before anything
        else starts; its fresh ready rows join the same step.

        A batch holds one capture group (``Engine.capture_group``); rids of
        another group wait in the backlog for a batch of their own.

        A speculation chain's step at its own target is ``pick_chain_step``'s.

        Args:
            exclude_target: If set, skip this (node_name, graph_walk) pair:
                a yield-away from the speculation chain running it.
        """
        # done before any early returns
        self._let_rows_back(request_state)

        # Note: a TP follow batch has to be scheduled irrespective of failure.
        # Rank 0 already committed to this batch and will sit on the collective
        # inside the forward until every follower joins it, so a follower that
        # skipped the batch because one of its rids failed locally would hang
        # the whole TP group. A popped ScheduleTPNode has no re-queue path
        # and must be submitted unconditionally.
        tp_follow_batch = self._try_schedule_tp_follow(
            request_state, exclude_target=exclude_target,
        )
        if tp_follow_batch is None:
            self.num_consec_tp_follower_batches = 0
        else:
            self.num_consec_tp_follower_batches += 1
            return tp_follow_batch

        if self.sched_type != SchedulingType.ROUND_ROBIN:
            raise NotImplementedError(f"Unknown scheduling type {self.sched_type}")
        # A chain merges and yields away within its own real walk, or its whole
        # combined walk under `speculation_mixes`.
        skip_walk = None
        if exclude_target is not None and not self.speculation_mixes(*exclude_target):
            skip_walk = exclude_target[1]
        scan_exclude = exclude_target if skip_walk else None
        exclude_target = None if exclude_target is None else self._key(*exclude_target)
        scan_exclude = scan_exclude or exclude_target
        # A queued row can be first in line even where a backlog fills the
        # step on its own (a row holding the step, back on its queue).
        ready = self._scan_ready(request_state, None, scan_exclude)

        def schedule(key, entries):
            # a yield-away runs the other walks: the chain's own rows parked
            # under the combined walk wait for its next step
            return self._schedule_key(request_state, key, entries, skip_walk=skip_walk)

        # A backlogged key goes first, oldest first, so a split set drains
        # before anything else starts; its fresh rows ride along. A key whose
        # step comes out empty (every backlogged row blocked) is passed over
        # rather than leaving the worker idle behind it. `exclude_target` is
        # only a fairness hint, and finishing a split set beats fairness --
        # except under ARRIVAL, where the rows holding a key's step keep it: a
        # yield-away from the chain running it leaves that backlog to it.
        keys = [k for k in self.backlog if not self._left_to_its_chain(k, exclude_target, skip_walk)]
        for key in keys:
            scheduled = schedule(key, ready.get(key, []))
            if scheduled is not None:
                return scheduled
        # A walk with no room (a cap of 0) takes nothing, so it stays the least
        # recent; picked anyway, it would starve the walk that frees the room.
        ready = {
            key: rows for key, rows in ready.items() if self._walk_caps(key)[key[1]] != 0
        }
        # Keep going down the round-robin order rather than idling behind the
        # first node whose step comes out empty. That node is not marked
        # scheduled, so round robin picks it again next pass anyway -- and
        # under a colocated model the node behind it is the one that frees
        # its pages (snac_decoder behind Orpheus's LLM).
        for rr_key in self._node_rr_order(ready):
            if rr_key in keys:
                continue  # already tried above, off its backlog
            scheduled = schedule(rr_key, ready.get(rr_key, []))
            if scheduled is not None:
                return scheduled
        return None

    def _left_to_its_chain(
        self, key: tuple[str, str], exclude_key: tuple[str, str] | None,
        skip_walk: str | None,
    ) -> bool:
        """Whether a yield-away from ``exclude_key`` leaves ``key``'s backlog
        alone: under ARRIVAL, when the chain it yields from holds that step.
        A combined walk's other walks still run (their rows are the
        yield-away's to take; ``skip_walk`` parks the chain's own)."""
        if key != exclude_key or self._order_of(key) is not StepOrder.ARRIVAL:
            return False
        return skip_walk is None or key not in self._walks_of_key

    def _scan_ready(
        self, request_state: RequestStateManager,
        target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
    ) -> dict[tuple[str, str], list[ReadyNodeEntry]]:
        """Engine-ready rows on the queues, by scheduling key, in scan order."""
        ready: dict[tuple[str, str], list[ReadyNodeEntry]] = {}
        # Do not schedule a request that was removed between scheduling
        # cycles, has its remove deferred for in-flight safety, is in OOM
        # backoff, or recently failed.
        exclude = self.failed_rids.union(self.pending_removes, self.held_until)
        # only rank 0 can initiate scheduling: a followed node's head is
        # served by _try_schedule_tp_follow
        for spec in self._rows_here(self._ready_specs(exclude, target, exclude_target)):
            node_partition = request_state.get_partition_for_node(
                spec.node_name
            )
            wg_id = self.runtime.get_worker_graph_id_for_node(
                spec.node_name, spec.graph_walk,
            )
            entries = None
            for rid in spec.rids:
                fwd_info = request_state.get_fwd_info(rid, node_partition)
                # check if the node is ready on the engine level
                # (e.g., for AR, whether the kv cache is read in)
                if not self._check_ready(spec.node_name, rid, fwd_info):
                    continue
                if entries is None:
                    entries = ready.setdefault(self._key(spec.node_name, spec.graph_walk), [])
                entries.append(ReadyNodeEntry(rid, wg_id, spec.graph_walk))
        return ready

    def _key(self, node_name: str, graph_walk: str) -> tuple[str, str]:
        """The scheduling key: the walk's combined walk if it has one."""
        return node_name, self.combined_walk_of.get((node_name, graph_walk), graph_walk)

    def label_for(
        self, node_name: str, request_walks: dict[int, str],
    ) -> tuple[str, dict[int, str]]:
        """A batch's walk label and ``request_walks`` for rows with these real
        walks. The batch takes ``request_walks`` itself, not a copy."""
        walks = set(request_walks.values())
        if len(walks) == 1:
            return walks.pop(), {}
        return self._key(node_name, next(iter(walks)))[1], request_walks

    def _ready_specs(
        self, exclude: set[int],
        target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
    ):
        """``runtime.get_ready_nodes`` for scheduling keys, which the runtime
        matches against both real and combined walks."""
        return self.runtime.get_ready_nodes(
            exclude, target=target, exclude_target=exclude_target,
        )

    def _rows_here(
        self, specs: Iterable[ReadyNodeSpec],
        head_node: str | None = None, head_rids: Collection[int] = (),
    ) -> Iterator[ReadyNodeSpec]:
        """Of graph-ready ``specs``, the rows that are work this rank can take:
        every row of a node it leads, and of a node rank 0 leads only the rows
        of rank 0's FIFO head (``head_node``'s ``head_rids``), since such a node
        runs here only as rank 0 sends it."""
        leads = self.parallel_leader_nodes
        for spec in specs:
            if leads is None or spec.node_name in leads:
                yield spec
            elif spec.node_name == head_node:
                rids = [rid for rid in spec.rids if rid in head_rids]
                if rids:
                    yield spec._replace(rids=rids)

    def _schedule_key(
        self, request_state: RequestStateManager,
        node_walk: tuple[str, str], entries: list[ReadyNodeEntry],
        skip_walk: str | None = None,
    ) -> ScheduledBatch | None:
        """One step for ``node_walk`` out of its backlog and its ready
        ``entries`` (already found ready), as the batch builder picks it.
        Under a combined walk, backlogged rows of ``skip_walk`` (a yield-away
        from that walk) stay parked."""
        backlogged = self.backlog.pop(node_walk, None)
        live = [] if backlogged is None else self._backlog_candidates(
            backlogged, skip_walk=skip_walk,
        )
        found = {entry.request_id for entry in entries}
        walk_of = None
        if node_walk in self._walks_of_key:
            walks = {entry.request_id: entry.graph_walk for entry in entries}
            walk_of = lambda rid: walks.get(rid) or backlogged.walk_of(rid)  # noqa: E731
        check = self._readiness(request_state, node_walk[0])
        pick = self.batch_builder.pick(self._step_candidates(
            request_state, node_walk, backlog=live,
            queued=[entry.request_id for entry in entries],
            walk_of=walk_of,
            ready=lambda rid: rid in found or check(rid),
        ))
        if backlogged is not None:
            self._drop_failed_rows(backlogged)
            if not backlogged:
                backlogged = None  # every row failed
        fresh = self._assemble_batch(request_state, *node_walk, entries) \
            if entries else None
        return self._build_step(request_state, node_walk, backlogged, fresh or None, pick)

    def _build_step(
        self, request_state: RequestStateManager, node_walk: tuple[str, str],
        backlogged: ScheduledBatch | None, fresh: ScheduledBatch | None,
        pick: StepPick, mid_prefill: Collection[MidPrefillRow] = (),
        pre_existing_batch_size: int = 0,
        pre_existing_seq_lens: Callable[[], Collection[int]] | None = None,
    ) -> ScheduledBatch | None:
        """The step ``pick`` chose out of ``backlogged``, ``fresh`` and the
        caller's ``mid_prefill`` rows, under the node's token budget if it has
        one. ``pre_existing_batch_size`` rows of a speculation chain run in it
        too; ``pre_existing_seq_lens`` estimates their spans, asked for only
        when the budget needs them (estimating walks every chain row, and a
        steady decode step usually brings no other row)."""
        engine = self.engine_manager.get_engine(node_walk[0])
        max_tokens = engine.get_max_batch_tokens(*node_walk)
        row_tokens: dict[int, int] = {}
        chunkable: set[int] = set()
        started: dict[int, int] = {}
        # rid -> (start, total) of a popped or backlogged row mid-prefill
        resumed: dict[int, tuple[int, int]] = {}
        continuing, starts = self._mid_prefill_rows(
            node_walk, mid_prefill, row_tokens, chunkable, started,
        )
        if backlogged is None and not fresh and continuing is None:
            return None
        # only for the chunk sizing below, never to gate admission, so a walk
        # with no budget skips it: measuring a row costs an input-tensor build
        # and a cached-prefix probe (memoized for a backlogged row, `_row_lens`)
        if max_tokens is not None:
            taken = set(pick.taken)
            for batch in (backlogged, fresh):
                if batch:
                    tokens, chunks = self._measure_rows(request_state, batch, started, resumed, taken)
                    row_tokens.update(tokens)
                    chunkable |= chunks
        min_chunk = 1
        if max_tokens is not None and row_tokens:
            min_chunk = engine.min_chunk_tokens(max_tokens)
            seq_lens = pre_existing_seq_lens() if pre_existing_seq_lens is not None else ()
            max_tokens -= _pre_existing_tokens(seq_lens, pre_existing_batch_size)
        budgeted = max_tokens is not None and bool(row_tokens)
        return self._build_and_schedule(BatchBuildRequest(
            node_name=node_walk[0], graph_walk=node_walk[1],
            backlog=backlogged, fresh=fresh, pick=pick, continuing=continuing,
            max_batch_tokens=max_tokens if budgeted else None,
            walk_max_tokens=engine.walk_token_caps(*node_walk) if budgeted else None,
            row_tokens=row_tokens,
            chunkable=chunkable,
            min_chunk_tokens=min_chunk,
            start_order=started,
        ), starts, resumed)

    def _mid_prefill_rows(
        self, node_walk: tuple[str, str], rows: Collection[MidPrefillRow],
        row_tokens: dict[int, int], chunkable: set[int], started: dict[int, int],
    ) -> tuple[ScheduledBatch | None, dict[int, tuple[int, int]]]:
        """``rows`` as a batch to build from, measured from where each next
        chunk starts; not popped, since their nodes are in flight. Returns the
        batch, or None for no row, and rid -> (its next chunk's start, its
        prompt's length)."""
        node = node_walk[0]
        rows = [row for row in rows if row.start < row.total]
        if not rows:
            return None, {}
        batch = ScheduledBatch(
            node_name=node, graph_walk=node_walk[1],
            request_to_worker_graph={row.request_id: row.worker_graph_id for row in rows},
        )
        if node_walk in self._walks_of_key:
            batch.request_walks = {row.request_id: row.graph_walk for row in rows}
        for row in rows:
            rid = row.request_id
            row_tokens[rid] = row.total - row.start
            started[rid] = self.chunk_started.get((rid, node), 0)
            chunkable.add(rid)
        return batch, {row.request_id: (row.start, row.total) for row in rows}


    def pick_chain_step(
        self, request_state: RequestStateManager,
        target: tuple[str, str], continuing: Sequence[int],
        mid_prefill: Collection[MidPrefillRow] = (),
    ) -> "ChainPick":
        """Who runs the step a speculation chain speculates into at
        ``target``: its ``continuing`` rows (ready, in step order), the rows
        backlogged there and the ones on ``target``'s queues, as the batch
        builder picks them. Nothing is popped; ``take_chain_step`` takes the
        backlogged and queued rows it chose once the chain's rows are prepped.

        The chain only ever continues its own rows, so this is the one place
        a row waiting at the chain's target can get the step while the chain
        runs. Unless the chain mixes walks (`speculation_mixes`), only rows of
        its own walk are in it; another walk's rows make it yield instead.

        ``mid_prefill``: the chain's step's rows whose chunk is in flight.
        Their nodes are off the ready queue until it lands, so without them
        the next chunk would wait a step. They wait first in line, ahead of
        the backlog, and are never popped.
        """
        self._let_rows_back(request_state)
        key = self._key(*target)
        own_walk = None if self.speculation_mixes(*target) else target[1]
        backlogged = self.backlog.get(key)
        live = [] if backlogged is None else self._backlog_candidates(
            backlogged, only_walk=own_walk,
        )
        mid = [
            row for row in mid_prefill
            if (own_walk is None or row.graph_walk == own_walk)
            and self._key(target[0], row.graph_walk) == key
            and row.start < row.total
        ] if mid_prefill else []
        if mid:
            live = [row.request_id for row in mid] + live
        specs = []
        # a chain holding every request here has nothing waiting on the queues
        if len(request_state.per_request_info) > len(continuing) + len(mid):
            exclude = self.failed_rids.union(self.pending_removes, self.held_until)
            specs = [
                spec for spec in self._rows_here(self._ready_specs(
                    exclude, target if own_walk is not None else key, None,
                ))
                if spec.rids
            ]
        queued = specs[0].rids if len(specs) == 1 else [rid for spec in specs for rid in spec.rids]
        walk_of = None
        if key in self._walks_of_key:
            walks = {rid: spec.graph_walk for spec in specs for rid in spec.rids}
            walks.update((row.request_id, row.graph_walk) for row in mid)
            walk_of = lambda rid: walks.get(rid) or backlogged.walk_of(rid)  # noqa: E731
        ready = self._readiness(request_state, target[0])
        if mid:
            check, in_flight = ready, {row.request_id for row in mid}
            ready = lambda rid: rid in in_flight or check(rid)  # noqa: E731
        pick = self.batch_builder.pick(self._step_candidates(
            request_state, key, backlog=live, queued=queued, walk_of=walk_of,
            ready=ready,
            continuing=continuing, own_walk=target[1],
            one_walk=own_walk,
        ))
        if mid:
            took = set(pick.taken)
            mid = [row for row in mid if row.request_id in took]
        return ChainPick(key, pick, specs, mid)

    def take_chain_step(
        self, request_state: RequestStateManager, chain: "ChainPick",
        pre_existing_batch_size: int = 0,
        pre_existing_seq_lens: Callable[[], Collection[int]] | None = None,
    ) -> ScheduledBatch | None:
        """The backlogged and queued rows ``chain`` chose, popped as one batch:
        the rows that join the chain's step, or the step itself when the
        chain gave all of it away. Queued rows it held back (another capture
        group, a walk the step can't mix) are popped and parked, so the chain
        yields to them (``backlog_splits_from``) rather than run past them.
        The chain's rows mid-prefill that it took join from their held inputs;
        the chain's own ``pre_existing_batch_size`` rows count against a token
        budget (``_build_step``)."""
        key = chain.key
        own = set(chain.pick.continuing)
        taken = [rid for rid in chain.pick.taken if rid not in own]
        if not taken and not chain.pick.held_back:
            return None
        backlogged = self.backlog.get(key)
        parked = set() if backlogged is None else backlogged.request_to_worker_graph.keys()
        wanted = {rid for rid in (*taken, *chain.pick.held_back) if rid not in parked}
        entries = []
        if wanted:
            for spec in chain.specs:
                wg_id = self.runtime.get_worker_graph_id_for_node(spec.node_name, spec.graph_walk)
                entries.extend(
                    ReadyNodeEntry(rid, wg_id, spec.graph_walk)
                    for rid in spec.rids if rid in wanted
                )
        if backlogged is not None and not parked.isdisjoint(taken):
            self.backlog.pop(key)
        else:
            backlogged = None
        fresh = self._assemble_batch(request_state, *key, entries) if entries else None
        return self._build_step(
            request_state, key, backlogged, fresh or None, chain.pick,
            chain.mid_prefill, pre_existing_batch_size, pre_existing_seq_lens,
        )

    def _step_candidates(
        self, request_state: RequestStateManager, key: tuple[str, str], *,
        backlog: Sequence[int], queued: Sequence[int],
        walk_of: Callable[[int], str] | None, ready: Callable[[int], bool],
        continuing: Sequence[int] = (), own_walk: str | None = None,
        one_walk: str | None = None,
    ) -> StepCandidates:
        node, walk = key
        if key not in self._walks_of_key:
            one_walk = walk
        engine = self.engine_manager.get_engine(node)
        return StepCandidates(
            graph_walk=walk,
            continuing=continuing, own_walk=own_walk,
            backlog=backlog, queued=queued,
            walk_of=walk_of or (lambda _: walk),
            one_walk=one_walk,
            ready=ready,
            capture_group=(
                functools.partial(self._capture_group, request_state, node, walk)
                if engine.splits_by_capture_key(node, walk) else None
            ),
            walk_caps=self._walk_caps(key),
            walk_row_caps=self._walk_row_caps(key),
            line_order=functools.partial(self._line_order, key),
        )

    def _readiness(
        self, request_state: RequestStateManager, node_name: str,
    ) -> Callable[[int], bool]:
        """`_check_ready` for one waiting row of ``node_name``."""
        partition = request_state.get_partition_for_node(node_name)
        return lambda rid: self._check_ready(
            node_name, rid, request_state.get_fwd_info(rid, partition),
        )

    def _capture_group(
        self, request_state: RequestStateManager,
        node_name: str, graph_walk: str, rid: int,
    ) -> Any | None:
        engine = self.engine_manager.get_engine(node_name)
        if not engine.splits_by_capture_key(node_name, graph_walk):
            return None
        fwd_info = request_state.get_fwd_info(
            rid, request_state.get_partition_for_node(node_name),
        )
        # a streamed input's chunk can decide the capture (a codec's window size)
        chunks = request_state.ingested_stream_chunks(rid, node_name)
        return engine.capture_group(
            node_name, graph_walk, rid, fwd_info,
            input_metadata=InputMetadata(stream_chunks=chunks) if chunks else None,
        )

    def backlog_splits_from(
        self, request_state: RequestStateManager,
        target: tuple[str, str], rid: int,
    ) -> bool:
        """Whether ``target``'s backlog holds a live rid a speculation chain
        anchored at ``rid`` would never merge (``chain_must_yield``), so the
        chain has to yield for it to run.
        """
        target = self._key(*target)
        live = self._live_backlog(target)
        if not live:
            return False
        return self.batch_builder.chain_must_yield(
            self.backlog[target], live, rid,
            functools.partial(self._capture_group, request_state, *target),
        )

    def _live_backlog(self, node_walk: tuple[str, str]) -> set[int]:
        """``node_walk``'s backlogged rids that are not failed, removed or held."""
        waiting = self.backlog.get(node_walk)
        if waiting is None:
            return set()
        now = time.monotonic()
        return {
            r for r in waiting.request_to_worker_graph
            if r not in self.failed_rids and r not in self.pending_removes
            and self.held_until.get(r, 0.0) <= now
        }

    def _row_lens(
        self, request_state: RequestStateManager, batch: ScheduledBatch,
        rids: list[int],
    ) -> dict[int, Any] | None:
        """``rids``' lengths, for submodules that report them before prepare;
        measured once per row and kept on ``batch``. None if not reported."""
        engine = self.engine_manager.get_engine(batch.node_name)
        if self.get_tensor is None or not engine.declares_input_sequence_len(batch.node_name):
            return None
        unmeasured = [rid for rid in rids if rid not in batch.row_lens]
        if unmeasured:
            partition = request_state.get_partition_for_node(batch.node_name)
            inputs = batch.input_edges.to_input_tensors(
                self.get_tensor, unmeasured, skip_missing=True,
            ).by_rid
            for rid in unmeasured:
                batch.row_lens[rid] = engine.input_sequence_len(
                    batch.node_name, batch.walk_of(rid),
                    request_state.get_fwd_info(rid, partition), inputs.get(rid, {}),
                    rid=rid,
                )
        return batch.row_lens

    def _measure_rows(
        self, request_state: RequestStateManager, batch: ScheduledBatch,
        started: dict[int, int], resumed: dict[int, tuple[int, int]],
        only: Collection[int],
    ) -> tuple[dict[int, int], set[int]]:
        """For a node that reports row lengths: the remaining tokens of each
        of ``batch``'s rows in ``only`` (the step's), and the ones that may be
        chunked. A row mid-prefill goes in ``started`` with its start order and
        in ``resumed`` with where its chunk Loop stands
        (``_chunk_loop_progress``)."""
        node = batch.node_name
        engine = self.engine_manager.get_engine(node)
        rids = [rid for rid in batch.request_to_worker_graph if rid in only]
        progress = self._chunk_loop_progress(batch, rids)
        lens = self._row_lens(
            request_state, batch, [rid for rid in rids if rid not in progress],
        )
        if lens is None:
            return {}, set()
        tokens: dict[int, int] = {}
        chunkable: set[int] = set()
        for rid in rids:
            walk = batch.walk_of(rid)
            at = progress.get(rid)
            if at is not None:
                tokens[rid] = at[1] - at[0]
                started[rid] = self.chunk_started.get((rid, node), 0)
                resumed[rid] = at
            else:
                info = lens[rid]
                if info is None:
                    continue
                if info.cached_prefix:
                    self.cached_prefix[(rid, node)] = info.cached_prefix
                tokens[rid] = info.seq_len
            if engine.supports_chunked_prefill(node, walk):
                chunkable.add(rid)
        return tokens, chunkable

    def _chunk_loop_progress(
        self, batch: ScheduledBatch, rids: list[int],
    ) -> dict[int, tuple[int, int]]:
        """rid -> (tokens run, the prompt's length) for the rows of ``rids``
        mid-prefill, read off their chunk Loops. Nothing of theirs is in
        flight: a row whose chunk runs is off the ready queue and out of the
        backlog until the chunk lands, when the Loop has counted it."""
        node = batch.node_name
        walks = self.chunk_loop_walks.get(node)
        if not walks:
            return {}
        rows = [rid for rid in rids if batch.walk_of(rid) in walks]
        if not rows:
            return {}
        row_walks = [batch.walk_of(rid) for rid in rows]
        counts, caps = self.runtime.loop_progress(
            node, rows, row_walks, [chunk_loop_name(node, walk) for walk in row_walks],
        )
        return {
            rid: (count, cap)
            for rid, count, cap in zip(rows, counts, caps, strict=True) if count > 0
        }

    def _backlog_candidates(
        self, batch: ScheduledBatch,
        only_walk: str | None = None, skip_walk: str | None = None,
    ) -> list[int]:
        """The rows of a backlogged ``batch`` a step could take, in backlog
        order; the rest stay parked without asking the engine. Rows of a walk
        other than ``only_walk``, or of ``skip_walk``, are parked."""
        # Held (OOM backoff) and pending-remove rows wait without asking the
        # engine, as on the fresh path: asking reloads an offloaded row, which
        # would OOM it again straight away.
        now = time.monotonic()
        held, removing, failed = self.held_until, self.pending_removes, self.failed_rids
        rows = [
            rid for rid in batch.request_to_worker_graph
            if rid not in removing and rid not in failed and held.get(rid, 0.0) <= now
        ]
        if batch.request_walks and (only_walk or skip_walk) is not None:
            rows = [
                rid for rid in rows
                if (only_walk is None or batch.walk_of(rid) == only_walk)
                and batch.walk_of(rid) != skip_walk
            ]
        return rows

    def _drop_failed_rows(self, batch: ScheduledBatch) -> None:
        """Take the rows a readiness check failed out of a backlogged chunk
        that is out of `self.backlog` right now, where `_drop_backlogged_rid`
        cannot reach it. Parked again, a failed row would never leave."""
        dropped = [rid for rid in batch.request_to_worker_graph if rid in self.failed_rids]
        if dropped:
            for rid in dropped:
                batch.discard_rid(rid)
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )

    def _build_and_schedule(
        self, request: BatchBuildRequest,
        mid_prefill_starts: dict[int, tuple[int, int]] | None = None,
        resumed: dict[int, tuple[int, int]] | None = None,
    ) -> ScheduledBatch | None:
        """Have the batch builder split what was popped for ``request``'s
        (node, walk) by its pick, and carry out its verdict.

        The single place a batch becomes scheduled, so the round-robin
        bookkeeping lives here — the backlog path has to count as scheduling
        its (node, walk) too, or a walk being served out of the backlog would
        look perpetually least-recent once it drains.
        """
        node_walk = (request.node_name, request.graph_walk)
        # Read before the build: `_merged` folds `fresh` into `backlog` in place.
        expected = set()
        if request.backlog is not None:
            expected |= request.backlog.request_to_worker_graph.keys()
        if request.fresh is not None:
            expected |= request.fresh.request_to_worker_graph.keys()
        if request.continuing is not None:
            expected |= request.continuing.request_to_worker_graph.keys()
        result: BatchBuildResult = self.batch_builder.build_batch(request)
        self._refuse_lost_rows(node_walk, expected, result)
        backlog, returned = result.backlog, result.returned
        if mid_prefill_starts:
            # left out, a row mid-prefill goes back to its queue when its chunk lands
            if backlog is not None:
                backlog, _ = backlog.split_off_first(None, exclude_rids=set(mid_prefill_starts))
            if returned:
                returned = {rid: wg for rid, wg in returned.items() if rid not in mid_prefill_starts}
        # Whatever was popped past this step has to be remembered here, or it
        # would never run.
        if backlog is not None:
            self._backlog(node_walk, backlog)
        if returned:
            self._set_in_flight(request.node_name, returned, False)
            self.runtime.push_back_node(
                request.node_name, list(returned), list(returned.values()),
            )
        if result.scheduled is not None:
            self._mark_scheduled(*node_walk, len(result.scheduled))
            self._assign_chunks(
                result.scheduled, result.chunk_tokens or {}, request.row_tokens,
                {**(resumed or {}), **(mid_prefill_starts or {})},
            )
            result.scheduled.relabel_if_uniform()
        return result.scheduled

    @staticmethod
    def _refuse_lost_rows(
        node_walk: tuple[str, str], expected: set[int], result: BatchBuildResult,
    ) -> None:
        """Every row handed to a builder must come back out of it.

        A row missing from all three of ``scheduled``, ``backlog`` and
        ``returned`` has already come off its ready queue and is marked in
        flight, so nothing will ever schedule it again and the request hangs
        for good. Raising here costs one set per step and names the rows; the
        alternative is a silent hang with no evidence of where it started.
        """
        got = set()
        for part in (result.scheduled, result.backlog):
            if part is not None:
                got |= part.request_to_worker_graph.keys()
        if result.returned:
            got |= result.returned.keys()
        if got != expected:
            raise AssertionError(
                f"batch builder dropped rows for {node_walk}: "
                f"lost {sorted(expected - got)}, invented {sorted(got - expected)}"
            )

    def _assign_chunks(
        self, batch: ScheduledBatch, chunk_tokens: dict[int, int],
        row_tokens: dict[int, int], resumed: dict[int, tuple[int, int]],
    ) -> None:
        """Stamp the ``ChunkRange`` each chunked row runs this step.

        A row mid-prefill resumes at ``resumed``'s (start, total): where its
        chunk Loop stands, or for a speculation the end of the chunk in
        flight. A row first cut here starts past its cached prefix and caps
        its chunk Loop at its whole input's length, which is what ends it.

        A row whose walk has no token budget is never measured, so it gets no
        range at all: it runs whole, and its stride ends its chunk Loop
        (``chunk_loop_stride``).
        """
        if not (chunk_tokens or resumed or self.cached_prefix):
            return
        node = batch.node_name
        for rid in batch.request_to_worker_graph:
            at = resumed.get(rid)
            if at is not None:
                start, total = at
                end = start + chunk_tokens.get(rid, total - start)
                batch.chunk_ranges[rid] = ChunkRange(start, end, total, end - start)
            else:
                key = (rid, node)
                cached = self.cached_prefix.pop(key, 0)
                if rid not in chunk_tokens:
                    continue
                total = cached + row_tokens[rid]
                end = cached + chunk_tokens[rid]
                batch.chunk_ranges[rid] = ChunkRange(cached, end, total, end)
                self.chunk_started[key] = next(self._chunk_starts)
            if end < total:
                batch.incomplete_node_rids.add(rid)
        self.cap_chunk_loops(batch)

    def cap_chunk_loops(self, batch: ScheduledBatch) -> None:
        """Cap the chunk Loop of each row of ``batch`` on its first chunk at
        its whole input's length, which is what ends the Loop. A TP follower
        caps from the leader's chunks: the cap decides on which step the Loop
        ends, so every rank of the instance has to set the same one."""
        node = batch.node_name
        loop_walks = self.chunk_loop_walks.get(node, ())
        capped = [
            rid for rid, chunk in batch.chunk_ranges.items()
            if runs_first_chunk(chunk) and batch.walk_of(rid) in loop_walks
        ]
        if capped:
            walks = [batch.walk_of(rid) for rid in capped]
            self.runtime.set_loop_caps(
                node, capped, walks, [chunk_loop_name(node, walk) for walk in walks],
                [batch.chunk_ranges[rid].total for rid in capped],
            )

    def _backlog(self, node_walk: tuple[str, str], batch: ScheduledBatch) -> None:
        """Park ``batch``, folding it into whatever already waits under this key.

        Replacing would drop the rids already there — their graph nodes came
        off the ready queues when they were first assembled, so nothing would
        ever schedule them again and the requests hang.
        """
        existing = self.backlog.get(node_walk)
        if existing is None:
            self.backlog[node_walk] = batch
        else:
            existing.merge(batch)
        # Claimed like a running batch: an input arriving while it waits must
        # not put its node back on the ready queue, where it would be popped
        # a second time.
        self._set_in_flight(batch.node_name, batch.request_to_worker_graph, True)

    def _set_in_flight(
        self, node_name: str, rid_to_wg: dict[int, int], in_flight: bool,
    ) -> None:
        self.runtime.set_in_flight(
            node_name, list(rid_to_wg), list(rid_to_wg.values()), in_flight,
        )

    def _mark_scheduled(
        self, node_name: str, graph_walk: str, num_requests: int,
    ) -> None:
        """Record that this (node, walk) just ran, for round-robin ordering."""
        logger.debug(
            "MicroScheduler scheduling node %s with graph walk %s for %d requests",
            node_name, graph_walk, num_requests,
        )
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(node_name, graph_walk)] = self.batch_number

    def _drop_backlogged_rid(self, rid: int) -> None:
        """Take a request out of anything still queued for it.

        Its node was popped off the ready queue when the batch was assembled,
        so this is the only place holding it.
        """
        for batch in self.backlog.values():
            # Membership, not a falsy pop: a worker graph id of 0 is real.
            if rid not in batch.request_to_worker_graph:
                continue
            batch.discard_rid(rid)
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        self.backlog = {
            k: v for k, v in self.backlog.items() if len(v) > 0
        }

    def _walk_caps(self, key: tuple[str, str]) -> dict[str, int | None]:
        """``BatchBuildRequest.walk_caps`` for ``key``: its walk's cap, or a
        combined key's walks' caps and its own under its label, each lowered
        by any limit from a refused admit. The engine's caps are cached; the
        limits come and go, so they are applied on every call."""
        caps = self._walk_caps_of_key.get(key)
        if caps is None:
            walks = self._walks_of_key.get(key, ())
            caps = {walk: self._engine_cap(key[0], walk) for walk in walks}
            caps[key[1]] = self._engine_cap(*key)
            self._walk_caps_of_key[key] = caps
        if not self.batch_limits and not self.batch_ceilings:
            return caps
        return {walk: self._limited(key[0], walk, cap) for walk, cap in caps.items()}

    def _walk_row_caps(self, key: tuple[str, str]) -> dict[str, int]:
        """``BatchBuildRequest.walk_row_caps`` for ``key``: empty unless it is
        a combined walk. Fixed by the model, so asked once."""
        caps = self._row_caps_of_key.get(key)
        if caps is None:
            engine = self.engine_manager.get_engine(key[0])
            caps = {}
            for walk in self._walks_of_key.get(key, ()):
                cap = engine.get_max_rows_in_combined(key[0], walk)
                if cap is not None:
                    caps[walk] = cap
            self._row_caps_of_key[key] = caps
        return caps

    def _max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one, lowered by
        any limit from a refused admit (see `limit_batch`)."""
        return self._limited(node_name, graph_walk, self._engine_cap(node_name, graph_walk))

    def _engine_cap(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one. A combined
        walk's cap is the model's to declare, via its captures or max_batch_size.
        Fixed once the graphs are captured, so asked once."""
        key = (node_name, graph_walk)
        if key not in self._max_bs:
            self._max_bs[key] = self.engine_manager.get_engine(node_name).get_max_batch_size(
                node_name, graph_walk
            )
        return self._max_bs[key]

    def _limited(self, node_name: str, graph_walk: str, cap: int | None) -> int | None:
        """``cap`` lowered by any limit or cap from a refused admit (see
        `limit_batch` and `cap_batch`)."""
        key = (node_name, graph_walk)
        for limit in (self.batch_limits.get(key), self.batch_ceilings.get(key)):
            if limit is not None:
                cap = limit if cap is None else min(cap, limit)
        return cap

    def _assemble_batch(
        self,
        request_state: RequestStateManager,
        node_name: str,
        graph_walk: str,
        entries: list[ReadyNodeEntry],
    ) -> ScheduledBatch | None:
        del request_state  # readiness was already established upstream
        combined = (node_name, graph_walk) in self._walks_of_key
        rids = [entry.request_id for entry in entries]
        walks = {entry.graph_walk for entry in entries} if combined else {graph_walk}
        rows = None
        if len(walks) == 1:  # the common case: one walk, the plain pop
            walk = walks.pop()
            popped = self.runtime.pop_rids(node_name, walk, rids)
        else:
            rows = RequestWalks.from_walks(rids, [entry.graph_walk for entry in entries])
            popped = self.runtime.pop_walk_rids(node_name, rows)
        if popped is None or not popped.wg_ids.keys:
            return None
        batch_rids = popped.wg_ids.keys
        batch = ScheduledBatch(
            node_name=node_name,
            graph_walk=graph_walk,
            request_to_worker_graph=dict(zip(batch_rids, popped.wg_ids.values, strict=True)),
            input_edges=popped.input_edges,
            output_signals=list(popped.output_signals),
        )
        if not combined:
            return batch
        if rows is None:
            batch.request_walks = dict.fromkeys(batch_rids, walk)
            batch.walk_output_signals = {walk: list(popped.output_signals)}
        else:
            batch.request_walks = {
                rid: rows.walks[w] for rid, w in zip(batch_rids, popped.rid_walk_idx, strict=True)
            }
            batch.walk_output_signals = {
                walk: list(signals)
                for walk, signals in zip(rows.walks, popped.walk_output_signals, strict=True)
            }
        return batch


    def _backlog_has_schedulable(
        self, exclude_target: tuple[str, str] | None,
        exclude_walk: str | None = None,
    ) -> bool:
        """Whether a backlogged chunk outside ``exclude_target`` holds a rid
        still worth running; with ``exclude_walk``, only that walk's rows
        under ``exclude_target`` are outside it."""
        now = time.monotonic()
        for node_walk, batch in self.backlog.items():
            only_other = node_walk == exclude_target and exclude_walk is not None
            if node_walk == exclude_target and not only_other:
                continue
            if only_other and (
                set(batch.request_walks.values()) if batch.request_walks else {batch.graph_walk}
            ) == {exclude_walk}:
                continue  # every row is of the excluded walk
            for rid in batch.request_to_worker_graph:
                if only_other and batch.walk_of(rid) == exclude_walk:
                    continue
                if rid in self.failed_rids or rid in self.pending_removes:
                    continue
                if self.held_until.get(rid, 0.0) > now:
                    continue
                return True
        return False

    def has_ready_excluding(
        self,
        request_state: RequestStateManager,
        exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Cheap peek: any worker-graph queue ready with a (node, walk) other
        than `exclude_target`? Used by the speculation path to decide whether
        breaking the spec chain for fairness is actually warranted on this
        worker — on single-walk workers (e.g. Orpheus LLM) the answer is
        always False, so speculation can run every iter.

        Does NOT pop or modify queue state. Mirrors the ready-scan in
        get_next_batch but stops at the first match.

        Backlogged chunks count too: their graph nodes came off the queues
        when they were assembled, so the scan below cannot see them. A chunk
        under `exclude_target` is left out like any other — the speculative
        merge takes that one before its own rids, so it needs no yield.

        Only `exclude_target`'s own real walk is excluded: a ready row of
        another walk under the same combined walk is another target, which a
        speculation chain on that walk never merges, so it counts. A chain that
        mixes (`speculation_mixes`) merges those, so its whole combined walk is
        excluded.
        """
        mixes = exclude_target is not None and self.speculation_mixes(*exclude_target)
        exclude_walk = exclude_target[1] if exclude_target is not None and not mixes else None
        # the runtime scan below takes the real walk as given, or the combined one
        scan_exclude = exclude_target
        exclude_target = None if exclude_target is None else self._key(*exclude_target)
        if mixes:
            scan_exclude = exclude_target
        if self._backlog_has_schedulable(exclude_target, exclude_walk):
            return True

        # A failed rid is normally invisible here — get_next_batch refuses to
        # schedule it, so reporting it as ready would break the spec chain for
        # work that never materializes. The exception is a rid sitting in the
        # head TP follow batch: get_next_batch *will* schedule that one (rank 0
        # is waiting on it), so it counts as real ready work.
        tp_pend_rids: set[int] = set()
        head_node: str | None = None
        if self.tp_batches_pending_schedule:
            pend: ScheduleTPNode = self.tp_batches_pending_schedule[0]
            if self._key(pend.node_name, pend.graph_walk) != exclude_target:
                tp_pend_rids = set(self.tp_rids(pend))
                head_node = pend.node_name
        # Don't bother expiring held_until here — we only read it; the next
        # get_next_batch call will refresh.
        now = time.monotonic()
        exclude = {
            rid for rid in self.failed_rids if rid not in tp_pend_rids
        } | {
            rid for rid, t in self.held_until.items() if t > now
        }

        # Graph readiness is necessary but not sufficient, so a False here is
        # final and skips the engine pass.
        if not self.runtime.has_ready_excluding(exclude, scan_exclude):
            return False
        for spec in self._rows_here(self._ready_specs(exclude, None, scan_exclude), head_node, tp_pend_rids):
            node_partition = request_state.get_partition_for_node(
                spec.node_name
            )
            for rid in spec.rids:
                fwd_info = request_state.get_fwd_info(
                    rid, node_partition
                )
                if self._check_ready(spec.node_name, rid, fwd_info):
                    return True
        return False

    def _let_rows_back(self, request_state: RequestStateManager) -> None:
        """Before a scan: drop the holds that have run out, and unpark the
        rows whose capacity has freed."""
        now = time.monotonic()
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }
        self._unpark(request_state)

    def _unpark(self, request_state: RequestStateManager) -> None:
        """Let readiness ask the engine again about each parked request it has since admitted."""
        if not any(self.parked.values()):
            return

        def published_of(rid: int, node_name: str):
            fwd_info = request_state.get_fwd_info(rid, request_state.get_partition_for_node(node_name))
            return fwd_info.resource_publish_info

        for engine in {*map(self.engine_manager.get_engine, self.parked)}:
            for node_name, rid in engine.admit_waiting(published_of):
                self.parked[node_name].discard(rid)

    def _check_ready(
        self, node_name: str, rid: int, fwd_info: CurrentForwardPassInfo,
        allow_reload: bool=True
    ) -> bool:
        """Engine-level readiness, with a terminal failure taken out of the
        scan. Retryable not-ready (an in-flight KV read, a reload that doesn't
        fit) just comes back False; a ``terminal`` reason never will, so the
        rid goes to the worker to fail instead of being rescanned forever. A
        rid waiting for capacity is parked on its node until `_unpark`."""
        if rid in self.parked.get(node_name, ()):
            return False
        engine = self.engine_manager.get_engine(node_name)
        outcome = engine.check_ready(node_name, rid, fwd_info, allow_reload=allow_reload)
        if outcome.queued:
            self.parked.setdefault(node_name, set()).add(rid)
            return False
        if outcome.reason is not None and outcome.reason.terminal:
            logger.error(
                "Request %s cannot be served on node %s by resource %s: %s",
                rid, node_name, outcome.failed_resource, outcome.reason.message,
            )
            self.admit_errors[rid] = (
                f"resource {outcome.failed_resource} rejected the request: "
                f"{outcome.reason.message}"
            )
            self.fail_rids({rid})
            return False
        return outcome.ok and outcome.ready

    def pending_rows(self) -> Counter[str]:
        """Per node this rank schedules, rows ready for a step but not in one:
        its backlog, and graph-ready rows still on the queues. Read-only and
        graph-level (no engine readiness check, which can reload), so a held
        or not yet reloaded row counts; a failed or removed one does not."""
        exclude = self.failed_rids.union(self.pending_removes)
        rows: Counter[str] = Counter()
        for (node, _), waiting in self.backlog.items():
            rows[node] += sum(1 for rid in waiting.request_to_worker_graph if rid not in exclude)
        for spec in self._rows_here(self.runtime.get_ready_nodes(exclude)):
            rows[spec.node_name] += len(spec.rids)
        return rows

    def take_admit_errors(self) -> dict[int, str]:
        """Hand the accumulated terminal admit failures to the caller, once."""
        errors, self.admit_errors = self.admit_errors, {}
        return errors

    def fail_rids(self, rids: set[int]) -> None:
        """Stop scheduling new work for requests reported to the conductor as
        failed. Cleared by ``clear_rid`` when the removal comes back."""
        self.failed_rids.update(rids)
        for rid in rids:
            self._drop_backlogged_rid(rid)

    def clear_rid(self, rid: int, rid_str: str) -> None:
        """Forget all per-request scheduler state; called on REMOVE_REQUEST.

        Takes both identities because ``pending_tp_follow_count`` is keyed by
        the wire string and everything else by the handle."""
        self.failed_rids.discard(rid)
        self.read_failed_rids.discard(rid)
        self.admit_errors.pop(rid, None)
        for key in [key for key in self.chunk_started if key[0] == rid]:
            del self.chunk_started[key]
        for key in [key for key in self.cached_prefix if key[0] == rid]:
            del self.cached_prefix[key]
        self.held_until.pop(rid, None)
        for turns in self.last_ran.values():
            turns.pop(rid, None)
        self.arrival.pop(rid, None)
        for rids in self.parked.values():
            rids.discard(rid)
        self._drop_backlogged_rid(rid)
        self.clear_wire_rid(rid_str)

    def clear_wire_rid(self, rid_str: str) -> None:
        """Forget the wire-string-keyed state for a rid.

        Split out of ``clear_rid`` for the REMOVE of a rid this rank never
        admitted: there is no handle to clear anything else with, but a
        TP-follow batch may still be counted against the string."""
        self.pending_tp_follow_count.pop(rid_str, None)

