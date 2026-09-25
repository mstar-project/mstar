import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AdmitRuntimeError
from mstar.graph.runtime.base import ColumnarEdgeSpecs, GraphRuntime
from mstar.utils.ipc_format import ScheduleTPNode
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


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: int
    worker_graph_id: int
    graph_walk: str


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

    def merge(self, other: "ScheduledBatch") -> None:
        """Fold ``other``'s requests in, ours first — they have waited longer."""
        assert (self.node_name, self.graph_walk) == (
            other.node_name, other.graph_walk
        ), "only batches for the same (node, walk) share a backlog entry"
        self.request_to_worker_graph.update(other.request_to_worker_graph)
        self.input_edges.extend(other.input_edges)

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
        ), ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            request_to_worker_graph={
                rid: self.request_to_worker_graph[rid] for rid in left
            },
            input_edges=self.input_edges.select_rids(set(left)),
            output_signals=self.output_signals,
        )

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
    ):
        self.engine_manager = engine_manager
        self.batch_number = 0
        # The graph runtime, installed by the worker. Interning and the
        # (walk, node) -> worker graph index both live there.
        self.runtime: "GraphRuntime | None" = None
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
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[int, float] = {}
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: set[int] = set()

        # rid -> number of committed (ZMQ received) tp follow batches still
        # queued. On the fail/abort path we drain these before ACKing READS_DONE
        # so the worker graph queues aren't torn down while a follow still needs
        # to pop from them.
        self.pending_tp_follow_count: dict[int, int] = defaultdict(int)

    def _select_node_rr(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        best_node_name = None
        best_graph_walk = None
        least_recent_step = float('inf')

        for node_name, reqs in node_name_to_requests.items():
            for req in reqs:
                step = self.node_and_walk_to_last_batch_num.get((
                    node_name, req.graph_walk
                ), 0)
                if step < least_recent_step:
                    least_recent_step = step
                    best_node_name = node_name
                    best_graph_walk = req.graph_walk
        return best_node_name, best_graph_walk

    def hold_requests(self, request_ids: list[int]) -> None:
        """Put requests on hold for a brief backoff period after OOM."""
        deadline = time.monotonic() + self.HOLD_BACKOFF_SECONDS
        for rid in request_ids:
            self.held_until[rid] = deadline

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
        for rid in self.tp_rids(message):
            if rid != REMOVED_RID:
                self.pending_tp_follow_count[rid] += 1

    # TP-follow FIFO accessors for the follower's async path (head only).

    def peek_tp_follow(self) -> ScheduleTPNode | None:
        if not self.tp_batches_pending_schedule:
            return None
        return self.tp_batches_pending_schedule[0]

    def pop_tp_follow_head(self) -> ScheduleTPNode:
        # Sole exit for a queued follow batch: every consumer (the serial path
        # and the async follower's build / drop / void paths) pops here, so the
        # drain refcount is discharged in one place.
        message = self.tp_batches_pending_schedule.popleft()
        for rid in self.tp_rids(message):
            if rid not in self.pending_tp_follow_count:
                continue
            self.pending_tp_follow_count[rid] -= 1
            if self.pending_tp_follow_count[rid] <= 0:
                self.pending_tp_follow_count.pop(rid, None)
        return message

    def pop_ready_rids(
        self, request_state: RequestStateManager,
        node_name: str, graph_walk: str, request_ids: list[int],
    ) -> PopReadyResult | None:
        """Pop ``node_name`` for exactly ``request_ids``, all or none.
        Checked for every rid before anything is popped, so the caller
        retries later for a partially ready set."""
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
            if not self._check_ready(node_name, rid, fwd_info):
                return None

        popped = self.runtime.pop_rids(
            node_name, graph_walk, request_ids, check_ready=True,
        )
        if popped is None:
            return None
        batch_rids, wg_ids = popped.wg_ids.keys, popped.wg_ids.values

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(node_name, graph_walk)] = self.batch_number
        return PopReadyResult(
            dict(zip(batch_rids, wg_ids, strict=True)),
            popped.input_edges,
            popped.output_signals,
        )

    def _try_schedule_tp_follow(
        self, request_state: RequestStateManager,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        if len(self.tp_batches_pending_schedule) == 0:
            return
        first_tp_node: ScheduleTPNode = self.tp_batches_pending_schedule[0]
        if exclude_target is not None and \
                (first_tp_node.node_name, first_tp_node.graph_walk) == exclude_target:
            return
        if self.num_consec_tp_follower_batches >= self.max_consec_tp_follower_batches and \
                self.has_ready_excluding(
                    request_state,
                    (first_tp_node.node_name, first_tp_node.graph_walk)
                ):
            return
        # Check readiness for every rid to pop all-or-none. Use the
        # leader's graph walk.
        popped = self.pop_ready_rids(
            request_state, first_tp_node.node_name,
            first_tp_node.graph_walk, self.tp_rids(first_tp_node),
        )
        if popped is None:
            return
        request_to_worker_graph, input_edges, output_signals = popped

        self.pop_tp_follow_head()

        return ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            request_to_worker_graph=request_to_worker_graph,
            input_edges=input_edges,
            output_signals=output_signals,
            tp_seq=first_tp_node.spec_seq,
        )


    def get_next_batch(
        self,
        request_state: RequestStateManager,
        max_batch_size: int | None = None,
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
        # e.g., when adding to a speculative batch, we want to the requests that
        # are being speculated to be included in the batch size cap
        pre_existing_batch_size: int=0,
    ) -> ScheduledBatch | None:
        """
        Scans all worker graph queues for ready nodes, groups by node name,
        and returns one step's worth.

        A backlogged step — the remainder of a ready set too big for one
        forward — takes precedence over a fresh scan, so a split set drains
        before anything else starts.

        Args:
            max_batch_size: If set, limit the number of requests in the batch.
                Defaults to the engine's cap for the (node, walk) it picks.
            target: If set, only schedule this (node name, graph walk).
            exclude_target: If set, skip this (node_name, graph_walk) pair.
        """
        # Expire stale hold entries; done before any early returns
        now = time.monotonic()
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }

        sched_from_backlog = self._schedule_from_backlogged(
            request_state, target=target,
            max_batch_size=max_batch_size,
            pre_existing_batch_size=pre_existing_batch_size
        )
        if sched_from_backlog is not None:
            return sched_from_backlog

        # Collect all ready (node_name, request_id, graph_walk) tuples
        # grouped by node name
        node_name_to_requests: dict[str, list[ReadyNodeEntry]] = {}

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

        # Do not schedule a request that was removed between scheduling
        # cycles, has its remove deferred for in-flight safety, is in OOM
        # backoff, or recently failed.
        exclude = self.pending_removes | set(self.held_until) | self.failed_rids
        for spec in self.runtime.get_ready_nodes(
            exclude, target=target, exclude_target=exclude_target,
        ):
            if spec.node_name not in self.parallel_leader_nodes:
                continue  # only rank 0 can initiate scheduling!
            node_partition = request_state.get_partition_for_node(
                spec.node_name
            )
            wg_id = self.runtime.get_worker_graph_id_for_node(
                spec.node_name, spec.graph_walk,
            )
            for rid in spec.rids:
                fwd_info = request_state.get_fwd_info(rid, node_partition)
                # check if the node is ready on the engine level
                # (e.g., for AR, whether the kv cache is read in)
                if not self._check_ready(spec.node_name, rid, fwd_info):
                    continue
                node_name_to_requests.setdefault(spec.node_name, []).append(
                    ReadyNodeEntry(rid, wg_id, spec.graph_walk)
                )

        if not node_name_to_requests:
            return None

        if self.sched_type != SchedulingType.ROUND_ROBIN:
            raise NotImplementedError(f"Unknown scheduling type {self.sched_type}")
        best_node_name, graph_walk = self._select_node_rr(node_name_to_requests)

        if best_node_name is None:
            return None

        # Pop ready nodes for all requests of this node name
        entries = [e for e in node_name_to_requests[best_node_name] \
                   if e.graph_walk == graph_walk]

        if max_batch_size is None:
            max_batch_size = self._max_batch_size(best_node_name, graph_walk)
        remaining = self._remaining_capacity(max_batch_size, pre_existing_batch_size)
        if remaining is not None and remaining <= 0:
            # The caller already holds a full batch. Assembling would pop these
            # nodes off their ready queues with nowhere to run them, so leave
            # them queued for the next pass.
            return None

        full_batch = self._assemble_batch(
            request_state, best_node_name, graph_walk, entries
        )
        if not full_batch:
            return None

        # Everything past the first step is already popped off the queues, so
        # it has to be remembered here or it would never run.
        return self._cap_batch_and_schedule(batch=full_batch, max_bs=remaining)

    @staticmethod
    def _remaining_capacity(
        max_batch_size: int | None, pre_existing: int,
    ) -> int | None:
        """What is left of the cap once the caller's own rows are counted.
        None stays None: an uncapped node takes the whole ready set."""
        return None if max_batch_size is None else max_batch_size - pre_existing

    def _filter_cap_and_schedule(
        self, batch: ScheduledBatch, max_bs: int,
        request_state: RequestStateManager,
    ):
        node_partition = request_state.get_partition_for_node(batch.node_name)
        not_ready_rids = {
            rid for rid in batch.request_to_worker_graph if not self._check_ready(
                batch.node_name, rid,
                request_state.get_fwd_info(rid, node_partition),
            )
        }
        # A failed rid is not "not ready yet": excluding it would put it
        # straight back in the backlog. This chunk is out of `self.backlog`
        # right now, so `_drop_backlogged_rid` cannot reach it.
        dropped = not_ready_rids & self.failed_rids
        if dropped:
            for rid in dropped:
                batch.request_to_worker_graph.pop(rid, None)
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        not_ready_rids -= self.failed_rids
        return self._cap_batch_and_schedule(batch, max_bs, not_ready_rids)


    def _cap_batch_and_schedule(
        self, batch: ScheduledBatch, max_bs: int | None,
        exclude_rids: set[int] | None=None
    ) -> ScheduledBatch | None:
        """One step's worth off ``batch``; whatever is left goes to the backlog.

        The single place a batch becomes scheduled, so the round-robin
        bookkeeping lives here — the backlog path has to count as scheduling
        its (node, walk) too, or a walk being served out of the backlog would
        look perpetually least-recent once it drains.
        """
        node_walk = (batch.node_name, batch.graph_walk)
        if max_bs is not None and max_bs <= 0:
            self._backlog(node_walk, batch)
            return None
        capped_batch, remainder = batch.split_off_first(
            max_bs, exclude_rids=exclude_rids
        )
        # only ever store a real remainder: `_drop_backlogged_rid` walks these
        if remainder is not None:
            self._backlog(node_walk, remainder)
        if capped_batch is not None:
            self._mark_scheduled(*node_walk, len(capped_batch))
        return capped_batch

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

    def _schedule_from_backlogged(
        self, request_state: RequestStateManager,
        target: tuple[str, str] | None = None,
        max_batch_size: int | None=None,
        pre_existing_batch_size: int = 0
    ) -> ScheduledBatch | None:
        """The oldest backlogged step with a ready request in it.

        A caller targeting a specific (node, walk) — the TP-follow and
        speculation paths — must get that one or nothing, so a backlog entry
        for something else stays put. ``exclude_target`` is only a fairness
        hint, and finishing a split set beats fairness.

        A chunk whose requests are all blocked (its pages went to an eviction
        while it waited) is put back and the next one tried, rather than
        returning None and leaving the worker idle behind it.
        """
        if not self.backlog or target is not None and target not in self.backlog:
            return None

        # snapshot: `_cap_batch_and_schedule` re-inserts what it doesn't take,
        # which moves that entry to the back
        node_walks = list(self.backlog.keys())
        for node_walk in node_walks:
            if target is not None and node_walk != target:
                continue
            backlogged = self.backlog.pop(node_walk)
            curr_max_bs = self._max_batch_size(backlogged.node_name, backlogged.graph_walk) \
                if max_batch_size is None else max_batch_size
            scheduled = self._filter_cap_and_schedule(
                batch=backlogged,
                max_bs=self._remaining_capacity(curr_max_bs, pre_existing_batch_size),
                request_state=request_state
            )
            if scheduled is not None:
                return scheduled
        return None

    def _drop_backlogged_rid(self, rid: int) -> None:
        """Take a request out of anything still queued for it.

        Its node was popped off the ready queue when the batch was assembled,
        so this is the only place holding it.
        """
        for batch in self.backlog.values():
            # Membership, not a falsy pop: a worker graph id of 0 is real.
            if rid not in batch.request_to_worker_graph:
                continue
            del batch.request_to_worker_graph[rid]
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        self.backlog = {
            k: v for k, v in self.backlog.items() if len(v) > 0
        }

    def _max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one."""
        return self.engine_manager.get_engine(node_name).get_max_batch_size(
            node_name, graph_walk
        )

    def _assemble_batch(
        self,
        request_state: RequestStateManager,
        node_name: str,
        graph_walk: str,
        entries: list[ReadyNodeEntry],
    ) -> ScheduledBatch | None:
        del request_state  # readiness was already established upstream
        popped = self.runtime.pop_rids(
            node_name, graph_walk, [entry.request_id for entry in entries],
        )
        if popped is None:
            return
        batch_rids, wg_ids = popped.wg_ids.keys, popped.wg_ids.values
        if not batch_rids:
            return
        return ScheduledBatch(
            node_name=node_name,
            graph_walk=graph_walk,
            request_to_worker_graph=dict(zip(batch_rids, wg_ids, strict=True)),
            input_edges=popped.input_edges,
            output_signals=popped.output_signals,
        )


    def _backlog_has_schedulable(
        self, exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Whether a backlogged chunk outside ``exclude_target`` holds a rid
        still worth running."""
        now = time.monotonic()
        for node_walk, batch in self.backlog.items():
            if node_walk == exclude_target:
                continue
            for rid in batch.request_to_worker_graph:
                if rid in self.failed_rids or rid in self.pending_removes:
                    continue
                if self.held_until.get(rid, 0.0) > now:
                    continue
                return True
        return False

    def room_for_continuing(self, target: tuple[str, str]) -> int | None:
        """How many of a speculative batch's own rids fit once this target's
        backlog is served first.

        The chain only ever continues its own rids, so at the cap a backlogged
        chunk under the same (node, walk) would never be reached. Giving the
        backlog first claim costs the displaced rids one step — their nodes go
        ready again when the in-flight batch lands.

        None for an uncapped node: everything fits, so nothing is displaced.
        """
        cap = self._max_batch_size(*target)
        if cap is None:
            return None
        waiting = self.backlog.get(target)
        return max(0, cap - (0 if waiting is None else len(waiting)))

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
        """
        if self._backlog_has_schedulable(exclude_target):
            return True

        # A failed rid is normally invisible here — get_next_batch refuses to
        # schedule it, so reporting it as ready would break the spec chain for
        # work that never materializes. The exception is a rid sitting in the
        # head TP follow batch: get_next_batch *will* schedule that one (rank 0
        # is waiting on it), so it counts as real ready work.
        tp_pend_rids: set[int] = set()
        if self.tp_batches_pending_schedule:
            pend: ScheduleTPNode = self.tp_batches_pending_schedule[0]
            if (pend.node_name, pend.graph_walk) != exclude_target:
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
        if not self.runtime.has_ready_excluding(exclude, exclude_target):
            return False
        for spec in self.runtime.get_ready_nodes(
            exclude, exclude_target=exclude_target,
        ):
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

    def _check_ready(
        self, node_name: str, rid: int, fwd_info: CurrentForwardPassInfo,
    ) -> bool:
        """Engine-level readiness, with a terminal failure taken out of the
        scan. Retryable not-ready (an in-flight KV read, a reload that doesn't
        fit) just comes back False; an ``AdmitRuntimeError`` never will, so the
        rid is parked for the worker to fail instead of rescanned forever."""
        engine = self.engine_manager.get_engine(node_name)
        outcome = engine.check_ready(node_name, rid, fwd_info)
        if isinstance(outcome.reason, AdmitRuntimeError):
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

    def clear_rid(self, rid: int) -> None:
        """Forget all per-request scheduler state; called on REMOVE_REQUEST."""
        self.failed_rids.discard(rid)
        self.admit_errors.pop(rid, None)
        self.held_until.pop(rid, None)
        self._drop_backlogged_rid(rid)
        self.pending_tp_follow_count.pop(rid, None)

