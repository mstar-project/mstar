import functools
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AdmitRuntimeError
from mstar.graph.runtime.base import ColumnarEdgeSpecs, GraphRuntime
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode
from mstar.worker.batch_builder import (
    BatchBuilderType,
    BatchBuildRequest,
    BatchBuildResult,
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
    # rid -> real walk, set iff ``graph_walk`` is a combined walk
    request_walks: dict[int, str] = field(default_factory=dict)

    def walk_of(self, rid: int) -> str:
        return self.request_walks.get(rid, self.graph_walk)

    def rids_by_walk(self) -> dict[str, list[int]]:
        """The batch's rids grouped by real walk, in batch order."""
        if not self.request_walks:
            return {self.graph_walk: list(self.request_to_worker_graph)}
        groups: dict[str, list[int]] = {}
        for rid in self.request_to_worker_graph:
            groups.setdefault(self.request_walks[rid], []).append(rid)
        return groups

    def relabel_if_uniform(self) -> None:
        """Label a combined batch whose rows share one real walk with that walk."""
        if not self.request_walks:
            return
        walks = {self.request_walks[rid] for rid in self.request_to_worker_graph}
        if len(walks) == 1:
            self.graph_walk = walks.pop()
            self.request_walks = {}

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
        ), ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            request_to_worker_graph={
                rid: self.request_to_worker_graph[rid] for rid in left
            },
            input_edges=self.input_edges.select_rids(set(left)),
            output_signals=self.output_signals,
            request_walks=self._walks_for(left),
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
    # rid -> real walk when the pop spanned several walks
    request_walks: dict[int, str] = {}


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
    ):
        self.engine_manager = engine_manager
        # (node, real walk) -> combined walk; scheduling keys use the combined one
        self.combined_walk_of = combined_walk_of or {}
        self._walks_of_key: dict[tuple[str, str], list[str]] = defaultdict(list)
        for (node, walk), combined in self.combined_walk_of.items():
            self._walks_of_key[(node, combined)].append(walk)
        self.batch_builder = make_batch_builder(batch_builder_type)
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

    @property
    def last_consumed_tp_seq(self) -> int:
        """Highest leader step this rank has taken off the FIFO. What orders a
        forwarded removal against the step stream — see
        ``Worker._removal_step_reached``."""
        return self._last_popped_tp_seq

    def _select_node_rr(
        self, ready: dict[tuple[str, str], list[ReadyNodeEntry]],
    ) -> tuple[str, str] | None:
        """The least recently scheduled (node, walk) with a ready row; ties go
        to the first scanned."""
        return min(
            ready, default=None,
            key=lambda key: self.node_and_walk_to_last_batch_num.get(key, 0),
        )

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

    def settle_tp_follow_delta(self) -> bool:
        """Replay the front head's resident delta; True once every move landed.

        The one seam both consumers of this FIFO go through — the serial path
        below and the async follower's ``_try_follow_speculation`` — so neither
        can build a step against a half-replayed page state. The async path used
        to skip this entirely and read the difference as "rid not ready", which
        it then polled on for ever.

        """
        head = self.peek_tp_follow()
        if head is None:
            return True
        return self._apply_delta_from_message(head)

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
        request_walks: list[str] | None = None,
    ) -> PopReadyResult | None:
        """Pop ``node_name`` for exactly ``request_ids``, all or none.
        Checked for every rid before anything is popped, so the caller
        retries later for a partially ready set. ``request_walks`` gives each
        rid's real walk under a combined ``graph_walk``."""
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

        groups: dict[str, list[int]] = {}
        for i, rid in enumerate(request_ids):
            walk = request_walks[i] if request_walks else graph_walk
            groups.setdefault(walk, []).append(rid)
        result: PopReadyResult | None = None
        for walk, rids in groups.items():
            popped = self.runtime.pop_rids(node_name, walk, rids, check_ready=True)
            if popped is None:
                if result is not None:  # undo the walks already popped
                    self.runtime.push_back_node(
                        node_name, list(result.wg_ids), list(result.wg_ids.values()),
                    )
                return None
            wg_ids = dict(zip(popped.wg_ids.keys, popped.wg_ids.values, strict=True))
            if result is None:
                result = PopReadyResult(wg_ids, popped.input_edges, popped.output_signals)
            else:
                result.wg_ids.update(wg_ids)
                result.edge_specs.extend(popped.input_edges)
        if request_walks:
            result = result._replace(request_walks=dict(zip(request_ids, request_walks, strict=True)))

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
        # leader's graph walk.
        if not self.settle_tp_follow_delta():
            return # every pending move has to land before the step is built
        popped = self.pop_ready_rids(
            request_state, first_tp_node.node_name,
            first_tp_node.graph_walk, self.tp_rids(first_tp_node),
            request_walks=first_tp_node.request_walks,
        )
        if popped is None:
            return
        request_to_worker_graph, input_edges, output_signals, request_walks = popped

        self.pop_tp_follow_head()

        return ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            request_to_worker_graph=request_to_worker_graph,
            input_edges=input_edges,
            output_signals=output_signals,
            tp_seq=first_tp_node.spec_seq,
            request_walks=request_walks,
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
            max_batch_size: If set, limit the number of requests in the batch.
                Defaults to the engine's cap for the (node, walk) it picks.
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
        target = None if target is None else self._key(*target)
        exclude_target = None if exclude_target is None else self._key(*exclude_target)
        scan = True
        if target is not None:
            room = self._room_left(target, max_batch_size, pre_existing_batch_size)
            if room == 0:
                return None  # the caller is full: nothing to scan for
            # the caller and its backlog fill the step, so no fresh row could join
            scan = room is None or len(self._live_backlog(target)) < room
        ready = self._scan_ready(request_state, target, exclude_target) if scan else {}

        # A backlogged key goes first, oldest first, so a split set drains
        # before anything else starts; its fresh rows ride along. A key whose
        # step comes out empty (every backlogged row blocked) is passed over
        # rather than leaving the worker idle behind it. `exclude_target` is
        # only a fairness hint, and finishing a split set beats fairness.
        keys = [k for k in self.backlog if target is None or k == target]
        rr_key = self._select_node_rr(ready)
        if rr_key is not None and rr_key not in keys:
            keys.append(rr_key)
        for key in keys:
            scheduled = self._schedule_key(
                request_state, key, ready.get(key, []),
                max_batch_size=max_batch_size,
                pre_existing_batch_size=pre_existing_batch_size,
                capture_group_of=capture_group_of,
            )
            if scheduled is not None:
                return scheduled
        return None

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
        exclude = self.pending_removes | set(self.held_until) | self.failed_rids
        for spec in self._ready_specs(exclude, target, exclude_target):
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
                ready.setdefault(self._key(spec.node_name, spec.graph_walk), []).append(
                    ReadyNodeEntry(rid, wg_id, spec.graph_walk)
                )
        return ready

    def _key(self, node_name: str, graph_walk: str) -> tuple[str, str]:
        """The scheduling key: the walk's combined walk if it has one."""
        return node_name, self.combined_walk_of.get((node_name, graph_walk), graph_walk)

    def label_for(
        self, node_name: str, request_walks: dict[int, str],
    ) -> tuple[str, dict[int, str]]:
        """A batch's walk label and ``request_walks`` for rows with these real walks."""
        walks = set(request_walks.values())
        if len(walks) == 1:
            return walks.pop(), {}
        return self._key(node_name, next(iter(walks)))[1], dict(request_walks)

    def _ready_specs(
        self, exclude: set[int],
        target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
    ):
        """``runtime.get_ready_nodes`` filtered by scheduling keys, which the
        runtime knows only as real walks."""
        targets = [None] if target is None else [
            (target[0], walk) for walk in self._walks_of_key.get(target, [target[1]])
        ]
        runtime_exclude = None if exclude_target in self._walks_of_key else exclude_target
        for t in targets:
            for spec in self.runtime.get_ready_nodes(
                exclude, target=t, exclude_target=runtime_exclude,
            ):
                if exclude_target is not None \
                        and self._key(spec.node_name, spec.graph_walk) == exclude_target:
                    continue
                yield spec

    def _schedule_key(
        self, request_state: RequestStateManager,
        node_walk: tuple[str, str], entries: list[ReadyNodeEntry],
        max_batch_size: int | None,
        pre_existing_batch_size: int,
        capture_group_of: int | None,
    ) -> ScheduledBatch | None:
        """One step for ``node_walk`` out of its backlog and its ready
        ``entries``, composed by the batch builder."""
        remaining = self._room_left(node_walk, max_batch_size, pre_existing_batch_size)
        if remaining == 0:
            # The caller already holds a full batch. Assembling would pop these
            # nodes off their ready queues with nowhere to run them, so leave
            # them and the backlog as they are for the next pass.
            return None
        backlogged = self.backlog.pop(node_walk, None)
        blocked = set()
        if backlogged is not None:
            blocked = self._unready_backlog_rids(backlogged, request_state)
            if not backlogged:
                backlogged = None  # every row failed

        fresh = self._assemble_batch(request_state, *node_walk, entries) \
            if entries else None
        if backlogged is None and not fresh:
            return None
        walk_caps = {
            walk: self._remaining_capacity(
                self._max_batch_size(node_walk[0], walk), pre_existing_batch_size,
            )
            for walk in self._walks_of_key.get(node_walk, ())
        }
        return self._build_and_schedule(request_state, BatchBuildRequest(
            node_name=node_walk[0], graph_walk=node_walk[1],
            backlog=backlogged, fresh=fresh or None, blocked_rids=blocked,
            max_batch_size=remaining,
            capture_group_of=capture_group_of,
            walk_caps=walk_caps,
        ))

    def _room_left(
        self, node_walk: tuple[str, str], max_batch_size: int | None,
        pre_existing_batch_size: int,
    ) -> int | None:
        """What is left of ``node_walk``'s cap once the caller's own rows are
        counted. None stays None: an uncapped node takes the whole ready set."""
        if max_batch_size is None:
            max_batch_size = self._max_batch_size(*node_walk)
        if max_batch_size is None:
            return None
        return max(max_batch_size - pre_existing_batch_size, 0)

    def _capture_group(
        self, request_state: RequestStateManager,
        node_name: str, graph_walk: str, rid: int,
    ) -> Any | None:
        fwd_info = request_state.get_fwd_info(
            rid, request_state.get_partition_for_node(node_name),
        )
        return self.engine_manager.get_engine(node_name).capture_group(
            node_name, graph_walk, rid, fwd_info,
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

    def _unready_backlog_rids(
        self, batch: ScheduledBatch, request_state: RequestStateManager,
    ) -> set[int]:
        """The rows of a backlogged ``batch`` that cannot run yet; failed rows
        are dropped from it instead."""
        node_partition = request_state.get_partition_for_node(batch.node_name)
        # Held (OOM backoff) and pending-remove rows wait without asking the
        # engine, as on the fresh path: asking reloads an offloaded row, which
        # would OOM it again straight away.
        now = time.monotonic()
        waiting = {
            rid for rid in batch.request_to_worker_graph
            if rid in self.pending_removes or self.held_until.get(rid, 0.0) > now
        }
        not_ready_rids = waiting | {
            rid for rid in batch.request_to_worker_graph
            if rid not in waiting and not self._check_ready(
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
        return not_ready_rids - self.failed_rids

    def _build_and_schedule(
        self, request_state: RequestStateManager, request: BatchBuildRequest,
    ) -> ScheduledBatch | None:
        """Have the batch builder compose one step for ``request``'s (node,
        walk), and carry out its verdict.

        The single place a batch becomes scheduled, so the round-robin
        bookkeeping lives here — the backlog path has to count as scheduling
        its (node, walk) too, or a walk being served out of the backlog would
        look perpetually least-recent once it drains.
        """
        node_walk = (request.node_name, request.graph_walk)
        request.capture_group = functools.partial(
            self._capture_group, request_state, *node_walk,
        )
        result: BatchBuildResult = self.batch_builder.build_batch(request)
        # Whatever was popped past this step has to be remembered here, or it
        # would never run.
        if result.backlog is not None:
            self._backlog(node_walk, result.backlog)
        if result.returned:
            self._set_in_flight(request.node_name, result.returned, False)
            self.runtime.push_back_node(
                request.node_name, list(result.returned),
                list(result.returned.values()),
            )
        if result.scheduled is not None:
            self._mark_scheduled(*node_walk, len(result.scheduled))
            result.scheduled.relabel_if_uniform()
        return result.scheduled

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
        by_wg: dict[int, list[int]] = defaultdict(list)
        for rid, wg_id in rid_to_wg.items():
            by_wg[wg_id].append(rid)
        for wg_id, rids in by_wg.items():
            self.runtime.set_in_flight(node_name, wg_id, rids, in_flight)

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
            del batch.request_to_worker_graph[rid]
            batch.input_edges = batch.input_edges.select_rids(
                batch.request_to_worker_graph.keys()
            )
        self.backlog = {
            k: v for k, v in self.backlog.items() if len(v) > 0
        }

    def _max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one. A combined
        walk's cap is the model's to declare, via its captures or max_batch_size."""
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
        by_walk: dict[str, list[int]] = {}
        for entry in entries:
            by_walk.setdefault(entry.graph_walk, []).append(entry.request_id)
        batch = None
        for walk, rids in by_walk.items():
            popped = self.runtime.pop_rids(node_name, walk, rids)
            if popped is None or not popped.wg_ids.keys:
                continue
            batch_rids, wg_ids = popped.wg_ids.keys, popped.wg_ids.values
            part = ScheduledBatch(
                node_name=node_name,
                graph_walk=graph_walk,
                request_to_worker_graph=dict(zip(batch_rids, wg_ids, strict=True)),
                input_edges=popped.input_edges,
                output_signals=popped.output_signals,
                request_walks={} if walk == graph_walk else dict.fromkeys(batch_rids, walk),
            )
            if batch is None:
                batch = part
            else:
                batch.merge(part)
        return batch


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
        """How many of a speculative batch's own rids fit in ``target``'s step
        beside its backlog; see ``BaseBatchBuilder.room_for_continuing``."""
        target = self._key(*target)
        return self.batch_builder.room_for_continuing(
            self._max_batch_size(*target), self.backlog.get(target),
        )

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
        exclude_target = None if exclude_target is None else self._key(*exclude_target)
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
        runtime_exclude = None if exclude_target in self._walks_of_key else exclude_target
        if not self.runtime.has_ready_excluding(exclude, runtime_exclude):
            return False
        for spec in self._ready_specs(exclude, None, exclude_target):
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
        allow_reload: bool=True
    ) -> bool:
        """Engine-level readiness, with a terminal failure taken out of the
        scan. Retryable not-ready (an in-flight KV read, a reload that doesn't
        fit) just comes back False; an ``AdmitRuntimeError`` never will, so the
        rid is parked for the worker to fail instead of rescanned forever."""
        engine = self.engine_manager.get_engine(node_name)
        outcome = engine.check_ready(node_name, rid, fwd_info, allow_reload=allow_reload)
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

    def clear_rid(self, rid: int, rid_str: str) -> None:
        """Forget all per-request scheduler state; called on REMOVE_REQUEST.

        Takes both identities because ``pending_tp_follow_count`` is keyed by
        the wire string and everything else by the handle."""
        self.failed_rids.discard(rid)
        self.admit_errors.pop(rid, None)
        self.held_until.pop(rid, None)
        self._drop_backlogged_rid(rid)
        self.clear_wire_rid(rid_str)

    def clear_wire_rid(self, rid_str: str) -> None:
        """Forget the wire-string-keyed state for a rid.

        Split out of ``clear_rid`` for the REMOVE of a rid this rank never
        admitted: there is no handle to clear anything else with, but a
        TP-follow batch may still be counted against the string."""
        self.pending_tp_follow_count.pop(rid_str, None)

