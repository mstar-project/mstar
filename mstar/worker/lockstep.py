"""Both sides of a lockstep instance's step stream, the leader's and a follower's.

Rank 0 of an instance decides what runs and broadcasts each step as a
``ScheduleTPNode``; every other rank keeps those in a FIFO and builds the same step
from its own copy of the inputs (AGENTS.md invariant 11). This holds what that
takes on each side, except the choice between the follow head and a
colocated node, which is ``MicroScheduler``'s. ``build_follow_step`` opens the
head on the serial path; ``Worker._try_follow_speculation`` repeats that
sequence by hand on the async path, popping only the fresh rows.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Collection, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from torch.distributed import ReduceOp

from mstar.engine.resources import Residency
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RequestWalks
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import (
    MoveKind,
    OffloadDelta,
    ScheduleTPNode,
    TPNoSpeculation,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.admission import Admission
from mstar.worker.executing_batch import make_executing_batch
from mstar.worker.scheduled_batch import (
    REMOVED_RID,
    PopReadyResult,
    ScheduledBatch,
    cap_chunk_loops,
    tp_chunk_ranges,
    unfinished_chunk_rids,
)
from mstar.worker.services import WorkerLayout, WorkerServices

if TYPE_CHECKING:
    from mstar.communication.tensors import StoredOutputs
    from mstar.engine.engine import ExecutingBatch
    from mstar.graph.base import GraphEdge
    from mstar.worker.speculation import Speculation
    from mstar.worker.step_executor import PendingBatch

logger = logging.getLogger(__name__)

# The ScheduleTPNode columns of a row with no chunk range
_UNCHUNKED = (-1, -1, -1, -1)
# How many closed steps a follower remembers: a head for a step this old cannot arrive
_TP_NOSPEC_KEEP = 1024


def parse_tp_async_sched(raw: str) -> tuple[bool, frozenset[str] | None]:
    """``MSTAR_TP_ASYNC_SCHED``: ``0``/``false``/empty off, ``1``/``true`` every
    parallel node, or a comma-separated node list. Returns ``(enabled, nodes_or_None)``."""
    raw = raw.strip()
    # a quoted yaml boolean arrives as a string; it is the switch, not a node name
    word = raw.lower()
    if word in ("", "0", "false", "no", "off"):
        return False, None
    if word in ("1", "true", "yes", "on"):
        return True, None
    return True, frozenset(n.strip() for n in raw.split(",") if n.strip())


def tp_async_sched_setting(
    env: str | None, config: bool | str | list[str] | None,
) -> tuple[bool, frozenset[str] | None]:
    """The deployment config's ``tp_async_sched`` (true/false or a node list),
    which a non-empty ``MSTAR_TP_ASYNC_SCHED`` overrides."""
    if env:
        return parse_tp_async_sched(env)
    if isinstance(config, (list, tuple)):
        return parse_tp_async_sched(",".join(config))
    if isinstance(config, bool):
        return config, None
    return parse_tp_async_sched(str(config if config is not None else "1"))


def moves_alone(message: ScheduleTPNode) -> ScheduleTPNode:
    """Return ``message`` without its rows, prefix hits and page trails, so it carries only rank 0's moves."""
    return replace(message, request_ids=[], speculative=False, prefix_hits={}, page_trails={})


class Lockstep:
    def __init__(
        self, services: WorkerServices, layout: WorkerLayout, admission: Admission,
        async_sched: tuple[bool, frozenset[str] | None] = (False, None),
        wg_instances: dict[int, list[list[str]]] | None = None,
    ):
        self._worker_id = services.worker_id
        self._runtime = services.graph_runtime
        self._engine_manager = services.engine_manager
        self._tensor_manager = services.tensor_manager
        self._request_state = services.request_state
        self._streams = services.streams
        self._rebuilds = services.rebuilds
        self._communicator = services.communicator
        self._parallel_groups = services.parallel_groups
        self._admission = admission
        self._chunk_loop_walks = layout.chunk_loops.walks_by_node
        self._parallel_nodes = layout.parallel_nodes
        self._followed_nodes = layout.followed_nodes
        self._is_follower = layout.is_tp_follower
        self._streaming_edge_names = layout.streaming_edge_names
        self._async_sched, self._async_nodes = async_sched

        # -- leader --
        # monotonic seq stamped on every ScheduleTPNode it sends
        self._broadcast_seq = 0
        # the lockstep nodes this worker leads, and per request and node the
        # inputs of it that its followers read from elsewhere, not yet consumed
        # by a step of that node (see `step_reads`)
        self._led = layout.parallel_nodes & layout.parallel_leader_nodes
        self._unread: dict[int, dict[str, set[str]]] = {}
        # (producer, walk) -> {signal: the led nodes it is routed to}; structural
        self._consumers: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
        # the followers it sends its steps to
        self._followers = self._instance_followers(wg_instances, layout)

        # -- follower --
        self._follow_fifo: deque[ScheduleTPNode] = deque()
        # WIRE STRING -> number of committed (ZMQ received) follow steps still
        # queued. On the fail/abort path these drain before ACKing READS_DONE, so
        # the worker graph queues aren't torn down while a follow still needs to
        # pop from them.
        #
        # Keyed by the wire string, not the handle, unlike everything else here:
        # a ScheduleTPNode can arrive before the NEW_REQUEST that mints the rid's
        # handle, and a step that went uncounted in that window is exactly the
        # one the drain must wait for. Purged by `clear_wire_rid`.
        self._follow_count: dict[str, int] = defaultdict(int)
        # leader steps that will NOT be followed by a head; the leader said so
        # (TPNoSpeculation) or this rank closed the step
        self._no_spec: RecentSet[int] = RecentSet(_TP_NOSPEC_KEEP)
        self._leader_gap_warned = False
        # rank 0's moves this rank has yet to replay, on any of the nodes it follows
        self._pending_delta = OffloadDelta.new()
        self._last_delta_seq = -1
        # popped, not settled: a settled head can still fail to build, and its
        # removals wait for its admit
        self._last_popped_seq = -1
        self._warned_removed_rid = False

        # -- both --
        # Rids an input read of which failed on this rank. A follower never gets
        # those inputs, so it builds the leader's step without the row and the
        # step's read check takes it out on every rank
        # (`drop_rows_whose_read_failed`).
        self.read_failed_rids: set[int] = set()
        self._warned_read_check = False

        on = sorted(n for n in self._parallel_nodes if self.async_for(n))
        if on:
            logger.info(
                "Worker %s: TP async scheduling ON for %s (%s)", self._worker_id, on,
                "follower" if self._is_follower else "leader",
            )
        elif self._async_sched and self._parallel_nodes:
            logger.warning(
                "Worker %s: TP async scheduling names %s, none of this worker's "
                "parallel nodes %s; running the serial protocol",
                self._worker_id, sorted(self._async_nodes or ()), sorted(self._parallel_nodes),
            )

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    def follows(self, node_name: str) -> bool:
        """This rank follows ``node_name``: rank 0 of its instance decides both
        what runs on it and what gets evicted from it. Mirrors the engine's
        ``_tp_follower_nodes``, which refuses the eviction itself."""
        return node_name in self._followed_nodes

    def async_for(self, node_name: str) -> bool:
        """TP async scheduling applies to this node (flag, narrowed by node list)."""
        return self._async_sched and (self._async_nodes is None or node_name in self._async_nodes)

    def leads_async(self, node_name: str) -> bool:
        """``node_name`` is a parallel node this worker leads under TP async."""
        return self.async_for(node_name) and node_name in self._led

    def follows_async(self, node_name: str) -> bool:
        """``node_name`` is a parallel node this worker follows under TP async:
        the leader sends a head or a marker for each step of it."""
        return self.async_for(node_name) and self._is_follower and node_name in self._followed_nodes

    def verify_async_sched_agrees(self) -> None:
        """Refuse a per-rank flag mismatch at startup: a follower would wait for
        a decision the leader never sends, or get a head it cannot build."""
        for node in sorted(self._parallel_nodes):
            on = int(self.async_for(node))
            joint = self._parallel_groups.get_joint_group_for_node(node)
            lo, neg_hi = joint.agree([on, -on])
            if lo != -neg_hi:
                raise RuntimeError(
                    f"MSTAR_TP_ASYNC_SCHED disagrees across the ranks of {node!r} "
                    f"(async min {lo}, max {-neg_hi}); set it identically on every "
                    "rank of the instance."
                )

    # ------------------------------------------------------------------
    # Leader
    # ------------------------------------------------------------------

    def _instance_followers(
        self, wg_instances: dict[int, list[list[str]]] | None, layout: WorkerLayout,
    ) -> dict[str, list[str]]:
        """The other ranks of the instance of each lockstep node this rank leads, which rank 0 sends its steps to.
        Known when the worker starts, so rank 0's moves can go out in a step with no rows before it sends any step."""
        followers = {}
        for wg_id, instances in (wg_instances or {}).items():
            for node in layout.all_wg_nodes.get(wg_id, set()) & self._led:
                for instance in instances:
                    if self._worker_id in instance:
                        followers[node] = [worker for worker in instance if worker != self._worker_id]
        return followers

    def owes_moves(self) -> bool:
        """Whether the journal of the lockstep instance this rank leads holds moves no step has carried."""
        return any(self._engine_manager.get_engine(node).owes_moves() for node in self._led)

    def flush_moves(self, pending: PendingBatch | None = None) -> None:
        """Send the instance's followers a step with no rows while the journal holds moves, unless ``pending``
        is a step of the instance's own. That step still waits for its speculation decision, which a step with no
        rows sent first would read as none. A step of any of the instance's nodes carries every move."""
        if pending is not None and pending.node_name in self._led:
            return
        if self.owes_moves():
            self.broadcast(make_executing_batch(
                self._streams, node_name=min(self._led), graph_walk="", request_ids=[],
                per_request_input_tensors={}, per_request_info={},
            ))

    @property
    def led(self) -> frozenset[str]:
        """The lockstep nodes this worker leads."""
        return self._led

    @property
    def last_broadcast_seq(self) -> int:
        """The seq of the last step sent, which orders a forwarded teardown."""
        return self._broadcast_seq - 1

    def broadcast(
        self, node_batch: ExecutingBatch, *, speculative: bool = False, spec_from_seq: int = -1,
    ) -> int:
        """Send this batch to the followers as a ``ScheduleTPNode``. Returns its
        seq, or ``-1`` when this worker does not lead the node (nothing sent)."""
        if node_batch.node_name not in self._led:
            return -1
        seq = self._broadcast_seq
        self._broadcast_seq += 1
        # Drained once, outside the loop: every rank needs the same moves, so
        # draining per follower would give the first one everything and the rest
        # nothing. Each gets its own copy because each pops as it replays.
        engine = self._engine_manager.get_engine(node_batch.node_name)
        resident_delta = engine.take_resident_delta()
        # taken before the leases, since a follower checks it before it adopts them
        page_trails = engine.page_trails(node_batch.node_name)
        prefix_hits = engine.decide_cached_prefix(node_batch)
        read_check = self.step_reads(node_batch)
        if read_check:
            node_batch.read_check_rids = tuple(node_batch.request_ids)
        ctx = node_batch.step_context
        workers = self._followers[node_batch.node_name]
        request_ids = node_batch.request_ids
        walks, walk_idx = [], []
        if ctx.request_walks:
            rows = RequestWalks.from_walks(request_ids, [ctx.walk_of(r) for r in request_ids])
            walks, walk_idx = rows.walks, rows.walk_idx
        # the leader alone cuts chunks: a follower's own cut could differ
        chunk_starts: list[int] = []
        chunk_ends: list[int] = []
        chunk_totals: list[int] = []
        chunk_strides: list[int] = []
        if node_batch.chunk_ranges:
            for rid in request_ids:
                start, end, total, stride = node_batch.chunk_ranges.get(rid, _UNCHUNKED)
                chunk_starts.append(start)
                chunk_ends.append(end)
                chunk_totals.append(total)
                chunk_strides.append(stride)
        for worker in workers:
            self._communicator.send(
                worker, msg=WorkerMessage(
                    message_type=WorkerMessageType.SCHEDULE_TP,
                    body=ScheduleTPNode(
                        node_name=node_batch.node_name,
                        graph_walk=node_batch.graph_walk,
                        request_ids=[self._runtime.get_rid_string(r) for r in request_ids],
                        speculative=speculative,
                        spec_seq=seq,
                        spec_from_seq=spec_from_seq,
                        resident_delta=resident_delta.copy(),
                        walks=walks,
                        walk_idx=walk_idx,
                        prefix_hits=prefix_hits,
                        page_trails=page_trails,
                        read_check=read_check,
                        chunk_starts=chunk_starts,
                        chunk_ends=chunk_ends,
                        chunk_totals=chunk_totals,
                        chunk_strides=chunk_strides,
                    )
                )
            )
        return seq

    def maybe_broadcast_nospec(self, pending: PendingBatch, speculation: Speculation | None) -> None:
        """If this worker leads ``pending`` and no head went out for it (nothing
        speculated, or a non-parallel target, never broadcast), tell the
        followers none will, so each step they await settles on exactly one of
        {head, marker}."""
        if not self.leads_async(pending.node_name) or (speculation is not None and speculation.tp_seq >= 0):
            return
        # Not ``pending.node_batch.request_ids``: the GPU thread may be
        # ``drop_rids``-ing that list right now (empty at B=1 on a veto).
        sample_rid = next(iter(pending.batch.request_to_worker_graph))
        workers = self._runtime.get_sharding_config(sample_rid).get_sharding_group(
            pending.node_name, pending.batch.walk_of(sample_rid)
        )._workers[1:]
        for worker in workers:
            self._communicator.send(
                worker, msg=WorkerMessage(
                    message_type=WorkerMessageType.TP_NO_SPEC,
                    body=TPNoSpeculation(
                        node_name=pending.node_name,
                        graph_walk=pending.graph_walk,
                        spec_from_seq=pending.tp_seq,
                    )
                )
            )

    def note_reads(self, ready: dict[int, list[GraphEdge]]) -> None:
        """Rank 0: the inputs of a lockstep node it leads that landed here by a
        read, which each follower reads too. This rank's own output of a led node
        coming back (a persisted token as the decode walk's input) is every
        rank's own copy, with no read."""
        led, me = self._led, self._tensor_manager.my_entity_id
        if not led:
            return
        for rid, edges in ready.items():
            for edge in edges:
                if edge.next_node not in led or not edge.tensor_info:
                    continue
                if all(
                    i._source_node_name in led and i.source_entity == me
                    for i in edge.tensor_info
                ):
                    continue
                self._unread.setdefault(rid, {}).setdefault(edge.next_node, set()).add(edge.name)

    def note_local_inputs(
        self, stored: StoredOutputs, signals: list[str], producer: str,
        rids: list[int], walks: list[str], rid_walk_idx: list[int] | None,
    ) -> None:
        """Rank 0: outputs of a node outside the instance (an encoder beside
        rank 0), which the lockstep nodes they are routed to take without a
        read and each follower reads. Noted under each of those nodes, since a
        step of one spends only its own. ``rids[i]`` ran
        ``walks[rid_walk_idx[i]]``, or ``walks[0]`` when ``rid_walk_idx`` is None."""
        if not self._led or producer in self._parallel_nodes:
            return
        unread, consumers = self._unread, self._consumers
        per_walk = []
        for walk in walks:
            routes = consumers.get((producer, walk))
            if routes is None:
                routes = consumers[(producer, walk)] = self._led_consumers(producer, walk)
            per_walk.append(routes)
        routes_of = None if rid_walk_idx is None else {
            rid: per_walk[w] for rid, w in zip(rids, rid_walk_idx, strict=True)
        }
        for rid, sig in zip(stored.flat_rids, stored.signal_idxs, strict=True):
            routes = per_walk[0] if routes_of is None else routes_of[rid]
            signal = signals[sig]
            for node in routes.get(signal, ()):
                unread.setdefault(rid, {}).setdefault(node, set()).add(signal)

    def _led_consumers(self, producer: str, walk: str) -> dict[str, tuple[str, ...]]:
        """Which led nodes each of ``producer``'s signals goes to in ``walk``."""
        routes: dict[str, list[str]] = {}
        for node in sorted(self._led):
            for signal, _ in self._runtime.get_consumed_edges(producer, node, walk):
                routes.setdefault(signal, []).append(node)
        return {signal: tuple(nodes) for signal, nodes in routes.items()}

    def step_reads(self, node_batch: ExecutingBatch) -> bool:
        """Rank 0: whether a row of this step takes an input a follower reads,
        so a follower's read of it can have failed."""
        unread, streamed = self._unread, self._streaming_edge_names
        if not unread and not streamed:
            return False
        node, tensors = node_batch.node_name, node_batch.per_request_input_tensors
        for rid in node_batch.request_ids:
            names = tensors.get(rid)
            if not names:
                continue
            # a stream chunk lands with no step to mark, so a chunk is always a read
            if streamed and not streamed.isdisjoint(names):
                return True
            by_node = unread.get(rid)
            pending = by_node.get(node) if by_node else None
            if pending and not pending.isdisjoint(names):
                return True
        return False

    def consume_reads(self, node_batch: ExecutingBatch) -> None:
        """Rank 0, once a step of a node it leads ran: the marked inputs its rows
        took for that node are spent. Not at the broadcast: a step that does not
        run (a voided speculation, a refused admit) leaves its rows to a later one."""
        unread, node = self._unread, node_batch.node_name
        if not unread or node not in self._led:
            return
        tensors = node_batch.per_request_input_tensors
        for rid in node_batch.request_ids:
            by_node = unread.get(rid)
            pending = by_node.get(node) if by_node else None
            if pending:
                pending.difference_update(tensors.get(rid, ()))
                if not pending:
                    del by_node[node]
                    if not by_node:
                        del unread[rid]

    # ------------------------------------------------------------------
    # Follower
    # ------------------------------------------------------------------

    def enqueue_no_spec(self, message: TPNoSpeculation) -> None:
        """The leader sends no head from its step ``spec_from_seq``."""
        if self.async_for(message.node_name):
            self._no_spec.add(message.spec_from_seq)

    def forward_teardown(self, rid: int, message: WorkerMessage) -> None:
        """Fan a DRAIN or a REMOVE out to the followers of each TP group this
        rank leads for the request."""
        sharding = self._runtime.get_sharding_config(rid)
        if sharding is None:
            return
        followers: set[str] = set()
        for group in sharding.groups:
            # rank-ordered; index 0 is this worker when it is rank 0
            if group.tp_size > 1 and group._tp_rank == 0:
                followers.update(group._workers[1:])
        for worker in followers:
            self._communicator.send(worker, msg=message)

    def enqueue_follow(self, message: ScheduleTPNode) -> None:
        """Queue a ScheduleTPNode. A head for a step this rank closed is queued as a step of moves alone."""
        if (
            self.async_for(message.node_name) and message.speculative
            and message.spec_from_seq in self._no_spec
        ):
            logger.debug(
                "Worker %s: dropped speculative head seq %d (parent %d closed)",
                self._worker_id, message.spec_seq, message.spec_from_seq,
            )
            message = moves_alone(message)
        self._follow_fifo.append(message)
        # Count every rid the leader named, including ones this rank cannot
        # resolve yet -- ``tp_rids`` would flatten those to REMOVED_RID and
        # lose the very step the drain has to wait for.
        for rid in message.request_ids:
            self._follow_count[rid] += 1

    def close_follow_step(self, pending: PendingBatch) -> None:
        """This step's forward raised (symmetrically on the leader, which never
        submitted its head): a head from it, queued or arriving later, becomes a step of moves alone."""
        self._no_spec.add(pending.tp_seq)
        head = self.follow_head()
        if head is not None and head.speculative and head.spec_from_seq == pending.tp_seq:
            # settle may have adopted some of the head's hit, and the leader releases the whole lease
            self._engine_manager.get_engine(head.node_name).drop_cached_prefix(head)
            self.void_follow_head()

    def is_no_spec(self, seq: int) -> bool:
        """Whether no head will follow step ``seq``."""
        return seq in self._no_spec

    def leader_passed(self, seq: int, head: ScheduleTPNode) -> None:
        """A later message with no decision for step ``seq`` means none is coming
        (per-peer FIFO; the flag is off on the leader). Treated as a no-spec."""
        if not self._leader_gap_warned:
            self._leader_gap_warned = True
            logger.warning(
                "Worker %s: leader moved past step seq %d without a "
                "speculation decision (front: seq %d); treating as "
                "no-spec. Is MSTAR_TP_ASYNC_SCHED set on every rank?",
                self._worker_id, seq, head.spec_seq,
            )
        self._no_spec.add(seq)

    def follow_head(self) -> ScheduleTPNode | None:
        """The leader's oldest step not yet taken, if any."""
        return self._follow_fifo[0] if self._follow_fifo else None

    def _apply_resident_delta(
        self, node_name: str, new_delta: OffloadDelta | None = None, caught_up: bool = True,
    ) -> bool:
        delta = self._pending_delta
        if new_delta is not None:
            delta.extend(new_delta)

        if not len(delta):
            return True

        return self._engine_manager.get_engine(node_name).apply_resident_delta(delta, self._rebuilds, caught_up)

    def awaits_commit(self, step_rids: Collection[int]) -> bool:
        """Whether the replay stopped on what the commit of the step holding ``step_rids`` lets land: a reload, a drop
        or an offload of one of ``step_rids``, or rank 0's leases once every move has landed."""
        move = self._pending_delta.peek_left()
        if move is None:
            return True
        rid = self._runtime.get_rid_handle(move.rid)
        return rid is not None and (
            move.kind == MoveKind.RELOAD or move.kind in (MoveKind.DROP, MoveKind.OFFLOAD) and rid in step_rids
        )

    def _apply_delta_from_message(self, message: ScheduleTPNode, caught_up: bool = True) -> bool:
        new_delta = None
        if self._last_delta_seq < message.spec_seq:
            self._last_delta_seq = message.spec_seq
            new_delta = message.resident_delta
        try:
            return self._apply_resident_delta(message.node_name, new_delta, caught_up)
        except Exception as exc:
            raise SystemExit(f"TP ranks diverged: {exc}") from exc

    def settle_delta(self, caught_up: bool) -> bool:
        """Replay the front head's resident delta; True once every move landed.

        The one seam both consumers of the FIFO go through, the serial path and
        the async follower's rebuild, so neither can build a step against a
        half-replayed page state.

        ``caught_up`` means every step this rank took before the head has committed.
        """
        head = self.follow_head()
        if head is None:
            return True
        if not self._apply_delta_from_message(head, caught_up):
            return False
        engine = self._engine_manager.get_engine(head.node_name)
        try:
            self._start_follow_rebuilds(head)
            if not head.prefix_hits and not head.page_trails:
                return True
            engine.check_page_trails(head.node_name, head.graph_walk, head.spec_seq, head.page_trails)
            return engine.adopt_cached_prefix(head, caught_up)
        except Exception as exc:
            # The error would repeat on every pass over this head, and failing its requests would
            # leave the other ranks stuck in its collective.
            raise SystemExit(f"TP ranks diverged: {exc}") from exc

    def pop_follow_head(self) -> ScheduleTPNode:
        """Sole exit for a queued follow step: every consumer (the serial path
        and the async follower's build path) pops here. `void_follow_head`
        discharges the drain refcount of a head this rank won't build."""
        message = self._follow_fifo.popleft()
        self._apply_delta_from_message(message)
        self._last_popped_seq = max(self._last_popped_seq, message.spec_seq)
        self._discharge(message.request_ids)
        return message

    def void_follow_head(self) -> None:
        """Turn the front head, which this rank won't build, into a step of moves alone. Rank 0 made the head's
        moves and stamps the removals it makes next with the head's seq, so the seq counts as consumed only when
        the step is popped, after its moves land."""
        head = self._follow_fifo[0]
        self._discharge(head.request_ids)
        self._follow_fifo[0] = moves_alone(head)

    def _discharge(self, rids: list[str]) -> None:
        for rid in rids:
            if rid not in self._follow_count:
                continue
            self._follow_count[rid] -= 1
            if self._follow_count[rid] <= 0:
                self._follow_count.pop(rid, None)

    @property
    def last_consumed_seq(self) -> int:
        """Highest leader step this rank has taken off the FIFO: what orders a
        forwarded removal against the step stream."""
        return self._last_popped_seq

    def pending_follow_count(self, wire_rid: str) -> int:
        """How many queued follow steps name this request."""
        return self._follow_count.get(wire_rid, 0)

    def tp_rids(self, message: ScheduleTPNode) -> list[int]:
        """``ScheduleTPNode`` crosses the wire, so its rids are strings; these
        are this rank's handles for them, in the leader's order. A rid this rank
        does not know maps to ``REMOVED_RID``."""
        rid_of = self._runtime.get_rid_handle
        return [
            REMOVED_RID if (handle := rid_of(r)) is None else handle
            for r in message.request_ids
        ]

    def pop_ready_rids(
        self, node_name: str, graph_walk: str, request_ids: list[int],
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
        request_state = self._request_state
        node_partition = request_state.get_partition_for_node(node_name)
        check_ready = self._admission.check_ready
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
            if not check_ready(node_name, rid, fwd_info, allow_reload=False):
                return None

        # one walk: the plain pop, with no per-rid walk index to build
        popped = (
            self._runtime.pop_walk_rids(
                node_name, RequestWalks(request_ids, walks, walk_idx), check_ready=True,
            ) if walks
            else self._runtime.pop_rids(node_name, graph_walk, request_ids, check_ready=True)
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
        return result

    def build_follow_step(self) -> ScheduledBatch | None:
        """The step at the head of the FIFO, popped, on the serial path: readiness
        for every rid so it pops all or none, under the leader's graph walk. No
        step of the node is in flight here. None to wait: every pending move has
        to land before the step is built."""
        if not self.settle_delta(caught_up=True):
            return None
        head = self.follow_head()
        if not head.request_ids:
            # a step with no rows, so pop it rather than build an empty batch
            self.pop_follow_head()
            return None
        rids = self.tp_rids(head)
        # before anything is popped: lists that do not match the rows raise
        chunk_ranges = tp_chunk_ranges(head, rids)
        unread = self.unread_rows(head, rids)
        rebuild = self.follow_rebuilds(head.node_name, rids)
        unpopped = unread | rebuild
        popped = self.pop_ready_rids(
            head.node_name, head.graph_walk,
            [r for r in rids if r not in unpopped],
            walks=head.walks,
            walk_idx=[
                w for r, w in zip(rids, head.walk_idx, strict=True) if r not in unpopped
            ] if head.walks else None,
        )
        if popped is None:
            self.warn_unchecked_read_failure(head, rids)
            return None
        request_to_worker_graph, input_edges, output_signals, request_walks, walk_signals = popped
        if unpopped:
            request_to_worker_graph, request_walks = self.with_unpopped_rows(
                head, rids, unpopped, request_to_worker_graph, request_walks,
            )

        self.pop_follow_head()

        batch = ScheduledBatch(
            node_name=head.node_name,
            graph_walk=head.graph_walk,
            request_to_worker_graph=request_to_worker_graph,
            input_edges=input_edges,
            output_signals=output_signals,
            tp_seq=head.spec_seq,
            request_walks=request_walks,
            walk_output_signals=walk_signals,
            rebuild_rids=rebuild,
            read_check_rids=tuple(rids) if head.read_check else None,
            # the leader's chunks, so the chunk Loop ends on the same step here
            chunk_ranges=chunk_ranges,
            incomplete_node_rids=unfinished_chunk_rids(chunk_ranges),
        )
        self.cap_chunk_loops(batch)
        return batch

    def follow_rebuilds(self, node_name: str, rids: Iterable[int]) -> frozenset[int]:
        """The rows of rank 0's step that rebuild a request dropped on ``node_name``, which
        `_start_follow_rebuilds` started when the step settled."""
        return frozenset(rid for rid in rids if self._rebuilds.rebuilding(rid, node_name))

    def _start_follow_rebuilds(self, head: ScheduleTPNode) -> None:
        """Start the rebuild of each request of ``head`` dropped on its node, from the same source as rank 0's,
        since the request's chunk Loop is lockstep. Rank 0 started it before it sent the step, so rank 0 leased the
        request's cached prefix under the keys the rebuild lays out, and this rank starts it before it takes that
        lease."""
        node_name = head.node_name
        if node_name not in self._rebuilds.nodes:
            return
        engine = self._engine_manager.get_engine(node_name)
        request_state = self._request_state
        partition = request_state.get_partition_for_node(node_name)
        for rid in self.tp_rids(head):
            # A row a step on the node holds is resident or rebuilding already, since a drop there waits only for that
            # step.
            if (
                rid != REMOVED_RID and not self._rebuilds.rebuilding(rid, node_name)
                and engine.residency(node_name, rid) is Residency.DROPPED and not engine.in_step(node_name, rid)
            ):
                self._admission.start_rebuild(node_name, rid, request_state.get_fwd_info(rid, partition).graph_walk)

    def cap_chunk_loops(self, batch: ScheduledBatch) -> None:
        """``cap_chunk_loops`` for a step this rank took from the leader."""
        cap_chunk_loops(batch, self._runtime, self._chunk_loop_walks)

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

    def with_unpopped_rows(
        self, head: ScheduleTPNode, rids: list[int], unpopped: Collection[int],
        request_to_worker_graph: dict[int, int], request_walks: dict[int, str],
    ) -> tuple[dict[int, int], dict[int, str]]:
        """``request_to_worker_graph`` and ``request_walks`` with the rows
        nothing popped put back, in the leader's order: rows whose read failed
        here, which the step carries on every rank until its read check takes
        them out, so it is built, submitted, voided and settled alike, and
        rebuild rows, which the graph runtime holds no node of."""
        walk_of = dict(zip(rids, (head.walks[w] for w in head.walk_idx), strict=True)) if head.walks else {}
        request_walks = dict(request_walks)
        rows = {}
        for rid in rids:
            if rid not in unpopped:
                rows[rid] = request_to_worker_graph[rid]
                continue
            walk = walk_of.get(rid, head.graph_walk)
            rows[rid] = self._runtime.get_worker_graph_id_for_node(head.node_name, walk)
            if walk_of:
                request_walks[rid] = walk
        return rows, request_walks

    def warn_unchecked_read_failure(self, head: ScheduleTPNode, rids: Iterable[int]) -> None:
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

    # ------------------------------------------------------------------
    # Both
    # ------------------------------------------------------------------

    def record_read_failures(self, rids: Iterable[int]) -> None:
        """Every failed read is recorded for a step's read check, a request this
        rank already failed for another reason included: until the conductor's
        removal lands, rank 0 can still send a read-checked step naming it."""
        self.read_failed_rids.update(rids)

    def drop_rows_whose_read_failed(self, node_batch: ExecutingBatch) -> None:
        """Before preparing a step rank 0 sent with ``read_check``: take out the
        rows whose input read failed on any rank of the instance, on every rank.
        A follower already left such a row out of what it built; the rank that
        failed the read reported it, and the others fail it here, so its outputs
        and its pages never exist on any rank.

        One vote of a flag per row over the instance (``JointGroups.agree``
        with MAX), in the leader's row order, on the GPU thread where every rank
        prepares the step."""
        rids = node_batch.read_check_rids
        failed = self.read_failed_rids
        votes = self._parallel_groups.get_joint_group_for_node(node_batch.node_name).agree(
            [int(r in failed) for r in rids], ReduceOp.MAX,
        )
        flagged = {r for r, f in zip(rids, votes, strict=True) if f}
        if not flagged:
            return
        for rid in flagged:
            node_batch.failed_requests.setdefault(
                rid, "Error in worker: input tensor read failed on a rank of its TP group",
            )
        node_batch.drop_rids(flagged)

    def clear(self, rid: int, wire_rid: str) -> None:
        """Forget a request; called from its teardown."""
        self._unread.pop(rid, None)
        self.read_failed_rids.discard(rid)
        self.clear_wire_rid(wire_rid)

    def clear_wire_rid(self, wire_rid: str) -> None:
        """Forget the wire-string-keyed state for a rid. Separate for the REMOVE
        of a rid this rank never admitted: there is no handle to clear anything
        else with, but a follow step may still be counted against the string."""
        self._follow_count.pop(wire_rid, None)
