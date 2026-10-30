import logging
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, GraphRuntime, RequestWalks
from mstar.model.submodule_base import InputMetadata
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode
from mstar.worker.batch_builder import (
    BATCH_BUILDERS,
    BatchBuilderType,
)
from mstar.worker.engine_manager import EngineManager
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.sched_core import (
    NO_GROUP,
    CheckRequest,
    CheckResult,
    PassRequest,
    RoundRobin,
    RustSchedCore,
    SchedCore,
    SchedKeys,
    Verdict,
)

logger = logging.getLogger(__name__)

# A rid a TP follower is told about but has already removed. Wire messages
# carry string rids, so a head from the leader can name a request this rank
# tore down; ``tp_rids`` maps those to this sentinel and ``pop_ready_rids``
# reads it as not-ready. No request ever owns it, and it is deliberately NOT
# tolerated anywhere else -- an unknown rid off the wire is expected, an
# unknown rid from local state is a bug and should still raise.
REMOVED_RID = -1


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: int
    worker_graph_id: int
    graph_walk: str
    # its capture group's id, as the check reported it (``NO_GROUP``: none)
    group: int = NO_GROUP


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
    # A TP follower's copy of ``ScheduleTPNode.read_check``: the leader's rows,
    # in its order, including those this rank could not read; None for no check
    read_check_rids: tuple[int, ...] | None = None
    # `rids_by_walk`, computed once; `merge` and `discard_rid` reset it
    _by_walk: dict[str, list[int]] | None = field(default=None, repr=False, compare=False)

    def walk_of(self, rid: int) -> str:
        return self.request_walks.get(rid, self.graph_walk)

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
            walk_output_signals=self.walk_output_signals,
        )

    def _walks_for(self, rids: list[int]) -> dict[int, str]:
        if not self.request_walks:
            return {}
        return {rid: self.request_walks[rid] for rid in rids}

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


def _is_rust_runtime(runtime: GraphRuntime | None) -> bool:
    try:
        from mstar.graph.runtime.rust import RustGraphRuntime
    except ImportError:  # mstar_rust is not built, so no runtime is one
        return False
    return isinstance(runtime, RustGraphRuntime)


class MicroScheduler:
    """
    Simple MVP scheduler: scans all worker graph queues for ready nodes,
    groups by node name, returns the highest-priority group.
    """

    # Seconds to wait before retrying a held request after OOM
    HOLD_BACKOFF_SECONDS = 0.05
    # class-level too, for a scheduler built without __init__ (tests)
    _core: SchedCore | None = None

    def __init__(
        self, engine_manager: EngineManager,
        runtime: GraphRuntime | None = None,
        sched_type=SchedulingType.ROUND_ROBIN,
        parallel_leader_nodes: set[str] | None = None,
        max_consec_tp_follower_batches: int = 1,
        batch_builder_type: BatchBuilderType = BatchBuilderType.FIFO,
        combined_walk_of: dict[tuple[str, str], str] | None = None,
        speculation_mixes_walks: Mapping[str, Any] | None = None,
    ):
        self.engine_manager = engine_manager
        self.keys = SchedKeys(engine_manager, combined_walk_of, speculation_mixes_walks)
        self.batch_builder = BATCH_BUILDERS[batch_builder_type]
        if self.batch_builder.python is None and not _is_rust_runtime(runtime):
            raise ValueError(
                f"batch builder {batch_builder_type.value!r} has only a Rust backend, "
                "which needs the Rust graph runtime (MSTAR_RUST_GRAPH)"
            )
        self.rr = RoundRobin()
        # The graph runtime. Interning and the (walk, node) -> worker graph
        # index both live there.
        self.runtime = runtime
        # Looks up the handle for a wire request id; the worker installs the
        # runtime's. Only the TP-follow messages need it -- everything else
        # here already holds handles. Defaults to the identity so a scheduler
        # driven directly with its own keys (tests) needs no table.
        self.rid_of: Callable[[str], int | None] = lambda r: r
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

        # The pass and the backlog, chosen at first use (see ``core``)
        self._core: SchedCore | None = None
        # capture group -> id (see `_group`)
        self._group_ids: dict[Any, int] = {}

        # pending resident deltas for TP follow nodes
        self._pending_resident_deltas: dict[str, OffloadDelta] = {}
        self._last_resident_delta: int = -1
        # popped, not settled: a settled head can still fail to build, and its removals wait for its admit
        self._last_popped_tp_seq: int = -1

    def speculation_mixes(self, node_name: str, graph_walk: str) -> bool:
        return self.keys.speculation_mixes(node_name, graph_walk)

    def label_for(
        self, node_name: str, request_walks: dict[int, str],
    ) -> tuple[str, dict[int, str]]:
        return self.keys.label_for(node_name, request_walks)

    def _key(self, node_name: str, graph_walk: str) -> tuple[str, str]:
        return self.keys.key(node_name, graph_walk)

    @property
    def combined_walk_of(self) -> dict[tuple[str, str], str]:
        return self.keys.combined_walk_of

    @property
    def backlog(self) -> dict[tuple[str, str], ScheduledBatch]:
        """The Python core's backlog (tests); the Rust core keeps its own."""
        return self.core.backlog

    @property
    def batch_number(self) -> int:
        return self.rr.batch_number

    @property
    def node_and_walk_to_last_batch_num(self) -> dict[tuple[str, str], int]:
        return self.rr.last_batch_num

    @property
    def last_consumed_tp_seq(self) -> int:
        """Highest leader step this rank has taken off the FIFO. What orders a
        forwarded removal against the step stream — see
        ``Worker._removal_step_reached``."""
        return self._last_popped_tp_seq

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
        self._sync_limit(key)

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
        now = time.monotonic()
        waiting = self.pending_removes.keys() | {
            rid for rid, until in self.held_until.items() if until > now
        }
        if set(self.core.live_backlog(key, waiting, self.failed_rids)) - batch_rids:
            return True
        exclude = self.failed_rids.union(self.pending_removes, self.held_until, batch_rids)
        engine = self.engine_manager.get_engine(node_name)
        return any(
            not engine.is_offloaded(spec.node_name, rid)
            for spec in self.runtime.get_ready_nodes(exclude, target=key)
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
            self._sync_limit(key)

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
                self._sync_limit(key)

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
            self._sync_limit(key)
        for walk in walks:
            if walk != graph_walk and self.batch_limits.pop((node_name, walk), None) is not None:
                self._sync_limit((node_name, walk))

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

        self.rr.note(self._key(node_name, graph_walk))
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

        return ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            request_to_worker_graph=request_to_worker_graph,
            input_edges=input_edges,
            output_signals=output_signals,
            tp_seq=first_tp_node.spec_seq,
            request_walks=request_walks,
            walk_output_signals=walk_signals,
            read_check_rids=tuple(rids) if first_tp_node.read_check else None,
        )

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
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
        # e.g., when adding to a speculative batch, we want to the requests that
        # are being speculated to be included in the batch size cap
        pre_existing_batch_size: int=0,
        capture_group_of: int | None = None,
    ) -> ScheduledBatch | None:
        """
        One step's worth: a pending TP follow batch if there is one, else
        the batch builder's pick from one (node, walk)'s backlog and ready rows.

        A (node, walk) with a backlog — the remainder of a ready set too big
        for one forward — goes first, so a split set drains before anything
        else starts; its fresh ready rows join the same step.

        A batch holds one capture group (``Engine.capture_group``); rids of
        another group wait in the backlog for a batch of their own.

        Args:
            target: If set, only schedule this (node name, graph walk).
            exclude_target: If set, skip this (node_name, graph_walk) pair.
            capture_group_of: If set, the rid whose capture group the batch
                must share; the speculation merge passes a continuing rid.
                Otherwise the batch's first rid sets the group.
        """
        # Expire stale hold entries; done before any early returns
        now = time.monotonic()
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }
        self._unpark(request_state)

        # Note: a TP follow batch has to be scheduled irrespective of failure.
        # Rank 0 already committed to this batch and will sit on the collective
        # inside the forward until every follower joins it, so a follower that
        # skipped the batch because one of its rids failed locally would hang
        # the whole TP group. A popped ScheduleTPNode ha sno re-queue path
        # and must be submitted unconditionally. So that a targeted call,
        # (the speculation fresh-rid merge, which may reject what it is
        # handed) is never served from the FIFO.
        tp_follow_batch = None if target is not None else self._try_schedule_tp_follow(
            request_state, exclude_target=exclude_target,
        )
        if tp_follow_batch is None:
            self.num_consec_tp_follower_batches = 0
        else:
            self.num_consec_tp_follower_batches += 1
            return tp_follow_batch

        if self.sched_type != SchedulingType.ROUND_ROBIN:
            raise NotImplementedError(f"Unknown scheduling type {self.sched_type}")
        waiting = frozenset(self.pending_removes.keys() | self.held_until.keys())
        core = self.core
        step = core.begin(PassRequest(
            target, exclude_target, pre_existing_batch_size, capture_group_of,
            waiting=waiting, failed=self.failed_rids,
        ))
        while step.check is not None:
            step = core.resume(self._run_checks(request_state, step.check))
        return step.batch

    def _run_checks(
        self, request_state: RequestStateManager, request: CheckRequest,
    ) -> CheckResult:
        """Engine readiness for a ``CheckRequest``'s rows, in order; the
        engine half of a pass, the same for both cores. A terminal failure is
        reported and the rid failed (``_check_ready``), and comes back FAILED;
        a rid waiting for capacity is parked on its node, as there.

        A node that splits batches by capture key also reports each ready
        row's group, and the anchor's: the anchor row's when the request names
        one, else the first ready row's. Under ``by_anchor`` a ready row of
        another group is OFF_GROUP and does not count toward the limit."""
        rids = request.rids
        verdicts = bytearray(len(rids))
        nodes, node_idx, walks = request.nodes, request.node_idx, request.walks
        engines = [self.engine_manager.get_engine(node) for node in nodes]
        # `_check_ready` per row, with the per-node lookups hoisted
        partitions = [request_state.get_partition_for_node(node) for node in nodes]
        checks = [engine.check_ready for engine in engines]
        parked = self.parked
        parked_at = [parked.get(node, ()) for node in nodes]
        split = [
            walks is not None and engine.splits_by_capture_key(node, walks[n])
            for n, (node, engine) in enumerate(zip(nodes, engines, strict=True))
        ]
        groups = [NO_GROUP] * len(rids) if any(split) else None
        anchor_group = None
        if groups is not None and request.anchor is not None:
            anchor_group = self._group(request_state, nodes[0], walks[0], request.anchor)
        get_fwd_info = request_state.get_fwd_info
        limit, taken = request.limit, 0
        for i, rid in enumerate(rids):
            if 0 <= limit <= taken:
                verdicts[i:] = bytes((Verdict.UNCHECKED,)) * (len(rids) - i)
                break
            n = 0 if node_idx is None else node_idx[i]
            if rid in parked_at[n]:
                continue
            full = checks[n](nodes[n], rid, get_fwd_info(rid, partitions[n]), allow_reload=True)
            if full.queued:
                parked.setdefault(nodes[n], set()).add(rid)
                continue
            outcome = full.outcome
            if outcome.reason is not None and outcome.reason.terminal:
                self._fail_unservable(nodes[n], rid, full)
                verdicts[i] = Verdict.FAILED
                continue
            if not (outcome.ok and outcome.ready):
                continue
            verdicts[i] = Verdict.READY
            if split[n]:
                group = groups[i] = self._group(request_state, nodes[n], walks[n], rid)
                if anchor_group is None:
                    anchor_group = group
                if request.by_anchor and group != anchor_group:
                    verdicts[i] = Verdict.OFF_GROUP
                    continue
            taken += 1
        return CheckResult(bytes(verdicts), groups, anchor_group)

    def _group(
        self, request_state: RequestStateManager, node_name: str, graph_walk: str, rid: int,
    ) -> int:
        """``Engine.capture_group`` for ``rid``, as an id stable for this
        scheduler's life (cores compare ids, not the engine's values)."""
        fwd_info = request_state.get_fwd_info(
            rid, request_state.get_partition_for_node(node_name),
        )
        # a streamed input's chunk can decide the capture (a codec's window size)
        chunks = request_state.ingested_stream_chunks(rid, node_name)
        group = self.engine_manager.get_engine(node_name).capture_group(
            node_name, graph_walk, rid, fwd_info,
            input_metadata=InputMetadata(stream_chunks=chunks) if chunks else None,
        )
        return self._group_ids.setdefault(group, len(self._group_ids))

    @property
    def core(self) -> SchedCore:
        """The pass and the backlog: the Rust core with the Rust graph
        runtime, whose graph state it works on, where the batch builder has a
        Rust backend; the Python core otherwise. Built at first use, once the
        engines' caps are known. A lockstep (TP/SP) node needs nothing of
        either: only its leader schedules it, and a follower's TP-follow FIFO
        is taken before the core runs."""
        if self._core is None:
            rust = self.batch_builder.rust is not None and _is_rust_runtime(self.runtime)
            self._core = self._rust_core() if rust else self._python_core()
            for (node, walk), limit in self.keys.limits.items():
                self._core.set_limit(node, walk, limit)
        return self._core

    def _sync_limit(self, key: tuple[str, str]) -> None:
        """``key``'s batch limit for the core: the smaller of its `limit_batch`
        limit and its `cap_batch` ceiling, or none."""
        limits = [v for v in (self.batch_limits.get(key), self.batch_ceilings.get(key)) if v is not None]
        limit = min(limits) if limits else None
        if limit == self.keys.limits.get(key):
            return
        if limit is None:
            del self.keys.limits[key]
        else:
            self.keys.limits[key] = limit
        if self._core is not None:
            self._core.set_limit(*key, limit)

    def _rust_core(self) -> RustSchedCore:
        keys = {self._key(node, walk) for node, walk in self.runtime.local_node_walks()}
        return RustSchedCore(self.runtime, [
            (node, label, self.keys.walks_of_key.get((node, label), []),
             self.keys.engine_walk_caps((node, label)), self.keys.walk_row_caps((node, label)))
            for node, label in sorted(keys)
        ], self.parallel_leader_nodes or (), self.keys.mixing_chains, self.batch_builder.rust)

    def _python_core(self) -> SchedCore:
        # imported here: it builds this module's batches
        from mstar.worker.sched_core.python import PythonSchedCore

        return PythonSchedCore(
            self.runtime, self.keys, self.batch_builder.python(), self.rr,
            self.parallel_leader_nodes or (),
        )

    def _waiting(self) -> frozenset[int]:
        """Rows that wait without a check: deferred removes and held rows."""
        now = time.monotonic()
        return frozenset(self.pending_removes.keys() | {
            rid for rid, t in self.held_until.items() if t > now
        })

    def backlog_splits_from(
        self, request_state: RequestStateManager,
        target: tuple[str, str], rid: int,
    ) -> bool:
        """Whether ``target``'s backlog holds a live rid of another capture
        group than ``rid``'s: a speculation chain anchored at ``rid`` would
        never merge it, so the chain has to yield for it to run."""
        node, label = key = self._key(*target)
        if not self.engine_manager.get_engine(node).splits_by_capture_key(node, label):
            return False
        live = self.core.live_backlog(key, self._waiting(), self.failed_rids)
        if not live:
            return False
        anchor = self._group(request_state, node, label, rid)
        return any(self._group(request_state, node, label, r) != anchor for r in live)

    def _drop_backlogged_rid(self, rid: int) -> None:
        """Take a request out of anything still queued for it: its node was
        popped off the ready queue when the batch was assembled, so the
        backlog is the only place holding it."""
        self.core.drop_rids((rid,))

    def room_for_continuing(self, target: tuple[str, str]) -> int | None:
        """How many of a speculative batch's own rids fit in ``target``'s step
        beside its backlog; see ``BaseBatchBuilder.room_for_continuing``."""
        return self.core.room_for_continuing(target)

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
        if self.core.backlog_has_schedulable(
            exclude_target, exclude_walk, self._waiting(), self.failed_rids,
        ):
            return True

        # A failed rid is normally invisible here — get_next_batch refuses to
        # schedule it, so reporting it as ready would break the spec chain for
        # work that never materializes. The exception is a rid sitting in the
        # head TP follow batch: get_next_batch *will* schedule that one (rank 0
        # is waiting on it), so it counts as real ready work.
        tp_pend_rids: set[int] = set()
        if self.tp_batches_pending_schedule:
            pend: ScheduleTPNode = self.tp_batches_pending_schedule[0]
            if self._key(pend.node_name, pend.graph_walk) != exclude_target:
                tp_pend_rids = set(self.tp_rids(pend))
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
        for spec in self.runtime.get_ready_nodes(exclude, exclude_target=scan_exclude):
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
            self._fail_unservable(node_name, rid, outcome)
            return False
        return outcome.ok and outcome.ready

    def _fail_unservable(self, node_name: str, rid: int, outcome) -> None:
        """Park a rid a resource declared unservable for the worker to fail."""
        logger.error(
            "Request %s cannot be served on node %s by resource %s: %s",
            rid, node_name, outcome.failed_resource, outcome.reason.message,
        )
        self.admit_errors[rid] = (
            f"resource {outcome.failed_resource} rejected the request: "
            f"{outcome.reason.message}"
        )
        self.fail_rids({rid})

    def pending_rows(self) -> Counter[str]:
        """Per node this rank schedules, rows ready for a step but not in one:
        its backlog, and graph-ready rows still on the queues. Read-only and
        graph-level (no engine readiness check, which can reload), so a held
        or not yet reloaded row counts; a failed or removed one does not."""
        exclude = self.failed_rids.union(self.pending_removes)
        rows: Counter[str] = Counter()
        for node, count in self.core.backlog_rows(exclude):
            rows[node] += count
        for spec in self.runtime.get_ready_nodes(exclude):
            if spec.node_name in self.parallel_leader_nodes:
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
        self.held_until.pop(rid, None)
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

