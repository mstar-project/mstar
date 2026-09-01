import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AdmitRuntimeError
from mstar.graph.base import GraphNode
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.engine_manager import EngineManager
from mstar.worker.node_manager_utils import WorkerGraphsManager

logger = logging.getLogger(__name__)


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: str
    worker_graph_id: str
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
    node_objects: dict[str,GraphNode]
    # request_id -> worker_graph_id (for push-back on OOM)
    request_to_worker_graph: dict[str, str] = None

    def split_off_first(
        self, bs: int | None, exclude_rids: set[str] | None = None
    ) -> "tuple[ScheduledBatch | None, ScheduledBatch | None]":
        """
        Return the first ScheduledBatch, as well as the remainder (None if
        all requests have been scheduled)
        """
        exclude_rids = exclude_rids or {}
        rids = self.node_objects.keys() # as of py3.7, preserves ordering

        if bs is None:
            bs = len(rids)
        keep_rids = [rid for rid in rids if rid not in exclude_rids]
        exclude_rids = [rid for rid in rids if rid in exclude_rids]

        if len(keep_rids) <= bs and not exclude_rids:
            return self, None
        if not keep_rids:
            return None, self

        rids = list(rids)
        return ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            node_objects={
                rid: self.node_objects[rid] for rid in keep_rids[:bs]
            },
            request_to_worker_graph={
                rid: self.request_to_worker_graph[rid] for rid in keep_rids[:bs]
            }
        ), ScheduledBatch(
            node_name=self.node_name,
            graph_walk=self.graph_walk,
            node_objects={
                rid: self.node_objects[rid] for rid in keep_rids[bs:] + exclude_rids
            },
            request_to_worker_graph={
                rid: self.request_to_worker_graph[rid] for rid in keep_rids[bs:] + exclude_rids
            }
        )


class SchedulingType(Enum):
    ROUND_ROBIN = "round_robin"
    # TODO: priority. It used to key off a per-engine-type table, which no
    # longer exists — every node runs on the same engine now. The replacement
    # is for the model to declare a (node, graph_walk) priority, since only it
    # knows which walk is latency-sensitive. Worth weighing against
    # head-of-line blocking: a busy high-priority walk starves the rest.


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
        self.sched_type = sched_type

        # RIDs that have failed but have not gone through the cleanup procedure;
        # these cannot be scheduled (unless this is a TP follower node)
        self.failed_rids: set[str] = set()
        # rid -> message, for requests a resource declared unservable during a
        # readiness check. Drained by the worker, which reports them onward.
        self.admit_errors: dict[str, str] = {}

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
        self.held_until: dict[str, float] = {}
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: set[str] = set()

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

    def hold_requests(self, request_ids: list[str]) -> None:
        """Put requests on hold for a brief backoff period after OOM."""
        deadline = time.monotonic() + self.HOLD_BACKOFF_SECONDS
        for rid in request_ids:
            self.held_until[rid] = deadline

    def register_tp_follow(
        self, message: ScheduleTPNode
    ):
        self.tp_batches_pending_schedule.append(message)

    def _try_schedule_tp_follow(
        self, worker_graphs_manager: WorkerGraphsManager,
        target_node_name: str | None = None,
        target_graph_walk: str | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        if len(self.tp_batches_pending_schedule) == 0:
            return
        first_tp_node: ScheduleTPNode = self.tp_batches_pending_schedule[0]
        # Respect the caller's filters: a targeted call (e.g. the speculation
        # path asking for one specific node/walk) must not be handed a TP
        # follower batch for some other node, or the caller will merge those
        # node objects into a batch labeled with the target's name.
        if target_node_name is not None and first_tp_node.node_name != target_node_name:
            return
        if target_graph_walk is not None and first_tp_node.graph_walk != target_graph_walk:
            return
        if exclude_target is not None and \
                (first_tp_node.node_name, first_tp_node.graph_walk) == exclude_target:
            return
        if self.num_consec_tp_follower_batches >= self.max_consec_tp_follower_batches and \
                self.has_ready_excluding(
                    worker_graphs_manager,
                    (first_tp_node.node_name, first_tp_node.graph_walk)
                ):
            return
        # check if batch is ready
        node_partition = worker_graphs_manager.get_partition_for_node(first_tp_node.node_name)
        # Use the leader's graph walk, not this worker's current one: the
        # follower may lag or lead the leader's partition state.
        wgid = worker_graphs_manager.get_worker_graph_id_for_node(
            first_tp_node.request_ids[0], first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
        )
        queue = worker_graphs_manager.queues[wgid]
        for rid in first_tp_node.request_ids:
            wg = queue.per_request_queues[rid]
            if first_tp_node.node_name not in wg.ready_node_names:
                return
            fwd_info = worker_graphs_manager.get_fwd_info(rid, node_partition)
            # check if the node is ready on the engine level
            # (e.g., for AR, whether the kv cache is read in)
            if not self._check_ready(first_tp_node.node_name, rid, fwd_info):
                return

        node_objects = {}
        request_to_worker_graph = {}

        # TODO: this code is also repeated below, should pull into a helper fn
        for rid in first_tp_node.request_ids:
            popped = queue.pop_ready_nodes(rid, [first_tp_node.node_name])
            if popped:
                assert len(popped) == 1
                node_objects[rid] = popped[0]
                request_to_worker_graph[rid] = wgid

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            first_tp_node.node_name, first_tp_node.graph_walk
        )] = self.batch_number

        self.tp_batches_pending_schedule.popleft()

        return ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )


    def get_next_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
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
            worker_graphs_manager, target=target,
            max_batch_size=max_batch_size,
            pre_existing_batch_size=pre_existing_batch_size
        )
        if sched_from_backlog is not None:
            return sched_from_backlog

        if target is not None:
            target_node_name, target_graph_walk = target
        else:
            target_node_name, target_graph_walk = None, None

        # Collect all ready (node_name, request_id, graph_walk) tuples
        # grouped by node name
        node_name_to_requests: dict[str, list[ReadyNodeEntry]] = {}

        # Note: a TP follow batch has to be scheduled irrespective of failure.
        # Rank 0 already committed to this batch and will sit on the collective
        # inside the forward until every follower joins it, so a follower that
        # skipped the batch because one of its rids failed locally would hang
        # the whole TP group.
        tp_follow_batch = self._try_schedule_tp_follow(
            worker_graphs_manager,
            target_node_name=target_node_name,
            target_graph_walk=target_graph_walk,
            exclude_target=exclude_target,
        )
        if tp_follow_batch is None:
            self.num_consec_tp_follower_batches = 0
        else:
            self.num_consec_tp_follower_batches += 1
            return tp_follow_batch

        for worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if (
                    request_id not in worker_graphs_manager.per_request_info
                ) or request_id in self.pending_removes or (
                    request_id in self.held_until
                ) or request_id in self.failed_rids:
                    # Do not want to schedule if: request was removed between
                    # scheduling cycles, remove deferred for in-flight safety,
                    # request in OOM backoff, or request recently failed
                    continue
                for sname in node_names:
                    if sname not in self.parallel_leader_nodes:
                        continue # only rank 0 can initiate scheduling!
                    if target_node_name is not None and sname != target_node_name:
                        continue
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(request_id, node_partition)
                    if target_graph_walk is not None and graph_walk != target_graph_walk:
                        continue
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    # check if the node is ready on the engine level
                    # (e.g., for AR, whether the kv cache is read in)
                    if not self._check_ready(sname, request_id, fwd_info):
                        continue
                    node_name_to_requests.setdefault(sname, []).append(
                        ReadyNodeEntry(request_id, worker_graph_id, graph_walk)
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

        full_batch = self._assemble_batch(
            worker_graphs_manager, best_node_name, graph_walk, entries
        )
        if not full_batch:
            return None

        if max_batch_size is None:
            max_batch_size = self._max_batch_size(best_node_name, graph_walk)

        # Everything past the first step is already popped off the queues, so
        # it has to be remembered here or it would never run.
        return self._cap_batch_and_schedule(
            batch=full_batch,
            max_bs=self._remaining_capacity(max_batch_size, pre_existing_batch_size),
        )

    @staticmethod
    def _remaining_capacity(
        max_batch_size: int | None, pre_existing: int,
    ) -> int | None:
        """What is left of the cap once the caller's own rows are counted.
        None stays None: an uncapped node takes the whole ready set."""
        return None if max_batch_size is None else max_batch_size - pre_existing

    def _filter_cap_and_schedule(
        self, batch: ScheduledBatch, max_bs: int,
        worker_graphs_manager: WorkerGraphsManager,
    ):
        node_partition = worker_graphs_manager.get_partition_for_node(batch.node_name)
        not_ready_rids = {
            rid for rid in batch.node_objects if not self._check_ready(
                batch.node_name, rid,
                worker_graphs_manager.get_fwd_info(rid, node_partition),
            )
        }
        # A failed rid is not "not ready yet": excluding it would put it
        # straight back in the backlog. This chunk is out of `self.backlog`
        # right now, so `_drop_backlogged_rid` cannot reach it.
        for rid in not_ready_rids & self.failed_rids:
            batch.node_objects.pop(rid, None)
            batch.request_to_worker_graph.pop(rid, None)
        not_ready_rids -= self.failed_rids
        return self._cap_batch_and_schedule(batch, max_bs, not_ready_rids)


    def _cap_batch_and_schedule(
        self, batch: ScheduledBatch, max_bs: int | None,
        exclude_rids: set[str] | None=None
    ) -> ScheduledBatch | None:
        """One step's worth off ``batch``; whatever is left goes to the backlog.

        The single place a batch becomes scheduled, so the round-robin
        bookkeeping lives here — the backlog path has to count as scheduling
        its (node, walk) too, or a walk being served out of the backlog would
        look perpetually least-recent once it drains.
        """
        node_walk = (batch.node_name, batch.graph_walk)
        if max_bs is not None and max_bs <= 0:
            self.backlog[node_walk] = batch
            return None
        capped_batch, remainder = batch.split_off_first(
            max_bs, exclude_rids=exclude_rids
        )
        # never store None: `_drop_backlogged_rid` walks these
        if remainder is None:
            self.backlog.pop(node_walk, None)
        else:
            self.backlog[node_walk] = remainder
        if capped_batch is not None:
            self._mark_scheduled(*node_walk, len(capped_batch.node_objects))
        return capped_batch

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
        self, worker_graphs_manager: WorkerGraphsManager,
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
                worker_graphs_manager=worker_graphs_manager
            )
            if scheduled is not None:
                return scheduled
        return None

    def _drop_backlogged_rid(self, rid: str) -> None:
        """Take a request out of anything still queued for it.

        Its node was popped off the ready queue when the batch was assembled,
        so this is the only place holding it.
        """
        for batch in self.backlog.values():
            batch.node_objects.pop(rid, None)
            batch.request_to_worker_graph.pop(rid, None)
        self.backlog = {
            k: v for k, v in self.backlog.items() if v.node_objects
        }

    def _max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """The engine's cap for this (node, walk), if it has one."""
        return self.engine_manager.get_engine(node_name).get_max_batch_size(
            node_name, graph_walk
        )

    def _assemble_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name: str,
        graph_walk: str,
        entries: list[ReadyNodeEntry],
    ) -> ScheduledBatch | None:
        node_objects = {}
        request_to_worker_graph = {}
        for entry in entries:
            queue = worker_graphs_manager.queues[entry.worker_graph_id]
            popped = queue.pop_ready_nodes(entry.request_id, [node_name])
            if popped:
                assert len(popped) == 1
                node_objects[entry.request_id] = popped[0]
                request_to_worker_graph[entry.request_id] = entry.worker_graph_id

        if not node_objects:
            return

        return ScheduledBatch(
            node_name=node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )


    def has_ready_excluding(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Cheap peek: any worker-graph queue ready with a (node, walk) other
        than `exclude_target`? Used by the speculation path to decide whether
        breaking the spec chain for fairness is actually warranted on this
        worker — on single-walk workers (e.g. Orpheus LLM) the answer is
        always False, so speculation can run every iter.

        Does NOT pop or modify queue state. Mirrors the ready-scan in
        get_next_batch but stops at the first match.
        """

        # A failed rid is normally invisible here — get_next_batch refuses to
        # schedule it, so reporting it as ready would break the spec chain for
        # work that never materializes. The exception is a rid sitting in the
        # head TP follow batch: get_next_batch *will* schedule that one (rank 0
        # is waiting on it), so it counts as real ready work.
        tp_pend_rids: set[str] = set()
        if self.tp_batches_pending_schedule:
            pend: ScheduleTPNode = self.tp_batches_pending_schedule[0]
            if (pend.node_name, pend.graph_walk) != exclude_target:
                tp_pend_rids = set(pend.request_ids)
        now = time.monotonic()
        # Don't bother expiring held_until here — we only read it; the next
        # get_next_batch call will refresh.
        for _worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                if request_id in self.failed_rids and request_id not in tp_pend_rids:
                    continue

                for sname in node_names:
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(
                        request_id, node_partition,
                    )
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    if not self._check_ready(sname, request_id, fwd_info):
                        continue
                    return True
        return False

    def _check_ready(
        self, node_name: str, rid: str, fwd_info: CurrentForwardPassInfo,
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

    def take_admit_errors(self) -> dict[str, str]:
        """Hand the accumulated terminal admit failures to the caller, once."""
        errors, self.admit_errors = self.admit_errors, {}
        return errors

    def fail_rids(self, rids: set[str]) -> None:
        """Stop scheduling new work for requests reported to the conductor as
        failed. Cleared by ``clear_rid`` when the removal comes back."""
        self.failed_rids.update(rids)
        for rid in rids:
            self._drop_backlogged_rid(rid)

    def clear_rid(self, rid: str) -> None:
        """Forget all per-request scheduler state; called on REMOVE_REQUEST."""
        self.failed_rids.discard(rid)
        self.admit_errors.pop(rid, None)
        self.held_until.pop(rid, None)
        self._drop_backlogged_rid(rid)

