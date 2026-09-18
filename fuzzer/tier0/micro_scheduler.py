"""MicroScheduler: which ready nodes become the next forward pass.

This machine tests `MicroScheduler`. This module supplies the queues, the
worker graph manager, the engine and the clock as test doubles.

Ops: make_ready, next_batch, hold, advance_time, fail, clear, set_not_ready,
set_unservable, pending_remove, forget_request, tp_follow.

Invariants
----------
sched.batch_is_not_empty                      a returned batch has rows
sched.batch_matches_target                    a targeted call returns that pair
sched.exclude_only_bypassed_by_backlog_or_tp  an excluded pair comes back only
                                              from the backlog or the TP queue
sched.batch_respects_cap                      a batch plus pre-existing rows
                                              stays within the cap
sched.scheduled_rid_not_also_backlogged       a scheduled rid is not backlogged
sched.does_not_schedule_dropped_rid           a failed or removing rid is not
                                              given new work
sched.clear_rid_leaves_no_trace               clear_rid removes every record
sched.no_ready_work_is_lost                   ready work is on a queue or in
                                              the backlog unless dropped
sched.backlog_entries_are_never_empty         no backlog entry is empty
sched.backlog_key_matches_batch               a backlog key matches its batch
sched.backlog_batch_is_consistent             node_objects and worker graphs
                                              cover the same rids
sched.tp_follow_count_is_positive             a pending TP count is above zero
sched.tp_follow_count_empty_with_empty_fifo   an empty TP queue counts nothing
sched.quiesce_empties_backlog                 clearing every rid empties it

Not covered:

* `WorkerGraphsManager`
* the execution of a batch: the machine records it and then drops it
* how long a hold lasts: the clock is a test double
* fairness and starvation
* the lockstep properties of the tensor-parallel follow path"""

from __future__ import annotations

import random
from collections.abc import Iterator

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401
from mstar.engine.resources.step import (
    FULL_ADMIT_NOT_READY,
    FULL_ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    FullAdmitOutcome,
)
from mstar.graph.base import GraphNode
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker import micro_scheduler as micro_scheduler_module
from mstar.worker.micro_scheduler import MicroScheduler

UNSERVABLE = FullAdmitOutcome(
    AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError("fuzz: unservable")),
    "fuzz_resource",
)


class _Clock:
    """A deterministic replacement for the ``time`` module of the scheduler."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class _PerRequestQueue:
    """The ready nodes of one request, inside one worker graph."""

    def __init__(self, nodes: dict[str, GraphNode]) -> None:
        self.nodes = nodes
        self.ready_node_names: set[str] = set()

    def clear(self) -> None:
        self.ready_node_names.clear()


class _Queue:
    """A replacement for the queues of the requests of one worker graph."""

    def __init__(self, nodes: dict[str, GraphNode]) -> None:
        self._nodes = nodes
        self.per_request_queues: dict[str, _PerRequestQueue] = {}

    def ensure(self, rid: str) -> _PerRequestQueue:
        if rid not in self.per_request_queues:
            self.per_request_queues[rid] = _PerRequestQueue(self._nodes)
        return self.per_request_queues[rid]

    def get_ready_node_names(self) -> dict[str, set[str]]:
        return {
            rid: set(queue.ready_node_names)
            for rid, queue in self.per_request_queues.items()
            if queue.ready_node_names
        }

    def pop_ready_nodes(self, request_id: str, node_names: list[str]) -> list[GraphNode]:
        popped = []
        queue = self.per_request_queues.get(request_id)
        if queue is None:
            return popped
        for name in node_names:
            queue.ready_node_names.discard(name)
            popped.append(queue.nodes[name])
        return popped


class _Engine:
    """A replacement engine. It gives the batch limits and the admit results."""

    def __init__(self, caps: dict[tuple[str, str], int | None]) -> None:
        self.caps = caps
        self.not_ready: set[str] = set()
        self.unservable: set[str] = set()

    def get_max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        return self.caps.get((node_name, graph_walk))

    def check_ready(self, node_name, rid, fwd_info) -> FullAdmitOutcome:
        del node_name, fwd_info
        if rid in self.unservable:
            return UNSERVABLE
        if rid in self.not_ready:
            return FULL_ADMIT_NOT_READY
        return FULL_ADMIT_OK


class _EngineManager:
    """A replacement manager. Every node uses the same engine."""

    def __init__(self, engine: _Engine) -> None:
        self.engine = engine

    def get_engine(self, node_name: str) -> _Engine:
        del node_name
        return self.engine


class _Manager:
    """A replacement for ``WorkerGraphsManager``."""

    def __init__(self, queues: dict[str, _Queue], walks: dict[str, str]) -> None:
        self.queues = queues
        self.per_request_info: dict[str, object] = {}
        self._walks = walks  # Maps a request ID onto its graph walk.
        self._fwd_info: dict[str, object] = {}

    def get_partition_for_node(self, node_name: str) -> str:
        del node_name
        return "default"

    def get_graph_walk(self, rid: str, partition: str) -> str:
        del partition
        return self._walks[rid]

    def get_fwd_info(self, rid: str, partition: str) -> object:
        del partition
        return self._fwd_info.setdefault(rid, object())

    def get_worker_graph_id_for_node(self, rid: str, node_name: str, graph_walk=None):
        del node_name, graph_walk
        for wgid, queue in self.queues.items():
            if rid in queue.per_request_queues:
                return wgid
        return next(iter(self.queues))


class MicroSchedulerMachine(StateMachine):
    name = "micro_scheduler"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        num_nodes = rng.randint(1, 3)
        num_walks = rng.randint(1, 2)
        return {
            "num_requests": rng.randint(1, 4),
            "num_nodes": num_nodes,
            "num_walks": num_walks,
            "num_worker_graphs": rng.randint(1, 2),
            # None means no limit, which is a different code path.
            "caps": [
                rng.choice([None, 1, 2, 3])
                for _ in range(num_nodes * num_walks)
            ],
            "max_consec_tp_follower_batches": rng.randint(1, 2),
        }

    def __init__(self, config: dict) -> None:
        self.config = config
        self.rids = [f"r{i}" for i in range(config["num_requests"])]
        self.nodes = [f"N{i}" for i in range(config["num_nodes"])]
        self.walks = [f"w{i}" for i in range(config["num_walks"])]

        caps_list = list(config["caps"])
        self.caps = {
            (node, walk): caps_list[index]
            for index, (node, walk) in enumerate(
                (n, w) for n in self.nodes for w in self.walks
            )
        }

        graph_nodes = {
            name: GraphNode(name=name, input_names=set(), outputs=[])
            for name in self.nodes
        }
        self.queues = {
            f"wg{i}": _Queue(graph_nodes)
            for i in range(config["num_worker_graphs"])
        }
        # One request belongs to exactly one worker graph and one walk.
        self.rid_walk = {
            rid: self.walks[index % len(self.walks)]
            for index, rid in enumerate(self.rids)
        }
        self.rid_wg = {
            rid: f"wg{index % config['num_worker_graphs']}"
            for index, rid in enumerate(self.rids)
        }

        self.manager = _Manager(self.queues, self.rid_walk)
        self.engine = _Engine(self.caps)
        for rid in self.rids:
            self.manager.per_request_info[rid] = object()
            self.queues[self.rid_wg[rid]].ensure(rid)

        self.clock = _Clock()
        micro_scheduler_module.time = self.clock

        self.scheduler = MicroScheduler(
            engine_manager=_EngineManager(self.engine),
            parallel_leader_nodes=set(self.nodes),
            max_consec_tp_follower_batches=config["max_consec_tp_follower_batches"],
        )

        # (request, node) pairs made ready and not yet used or dropped.
        self.outstanding: set[tuple[str, str]] = set()
        self._cleared: set[str] = set()

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        choice = rng.random()
        if choice < 0.34:
            return Op("make_ready", (rng.randrange(8), rng.randrange(4)))
        if choice < 0.66:
            return Op(
                "next_batch",
                (
                    rng.choice([-1, 0, 1, 2, 3]),       # -1 gives no limit
                    rng.choice([-1] + list(range(4))),  # target, -1 gives none
                    rng.choice([-1] + list(range(4))),  # exclude, -1 gives none
                    rng.randint(0, 2),                  # pre_existing_batch_size
                ),
            )
        if choice < 0.72:
            return Op("hold", (rng.randrange(8),))
        if choice < 0.77:
            return Op("advance_time", (rng.randint(0, 200),))
        if choice < 0.82:
            return Op("fail", (rng.randrange(8),))
        if choice < 0.88:
            return Op("clear", (rng.randrange(8),))
        if choice < 0.91:
            return Op("set_not_ready", (rng.randrange(8), rng.randint(0, 1)))
        if choice < 0.93:
            return Op("set_unservable", (rng.randrange(8),))
        if choice < 0.96:
            return Op("pending_remove", (rng.randrange(8), rng.randint(0, 1)))
        if choice < 0.98:
            return Op("forget_request", (rng.randrange(8),))
        return Op("tp_follow", (rng.randrange(4), rng.randrange(8)))

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        if config["num_requests"] > 1:
            smaller = dict(config)
            smaller["num_requests"] = config["num_requests"] - 1
            yield smaller
        if config["num_worker_graphs"] > 1:
            smaller = dict(config)
            smaller["num_worker_graphs"] = 1
            yield smaller

    # -- helpers -------------------------------------------------------------

    def _rid(self, index: int) -> str:
        """Map an index from an op onto a request ID."""
        return self.rids[index % len(self.rids)]

    def _node(self, index: int) -> str:
        """Map an index from an op onto a node name."""
        return self.nodes[index % len(self.nodes)]

    def _pair(self, index: int) -> tuple[str, str]:
        """Map an index from an op onto a (node, walk) pair."""
        pairs = [(n, w) for n in self.nodes for w in self.walks]
        return pairs[index % len(pairs)]

    def _queue_pairs(self) -> set[tuple[str, str]]:
        """Collect every (request, node) pair that is on a ready queue."""
        return {
            (rid, name)
            for queue in self.queues.values()
            for rid, per_request in queue.per_request_queues.items()
            for name in per_request.ready_node_names
        }

    def _backlog_pairs(self) -> set[tuple[str, str]]:
        """Collect every (request, node) pair that is in the backlog."""
        return {
            (rid, batch.node_name)
            for batch in self.scheduler.backlog.values()
            for rid in batch.node_objects
        }

    # -- execution -----------------------------------------------------------

    def execute(self, op: Op) -> None:
        """Apply the op, then account for the drops that it permits.

        A fail op and a clear op both discard the backlogged work of their
        request. This method removes exactly those pairs. It does not exclude
        every failed request from the conservation check. The oracle can
        therefore still see work disappear for a live request.
        """
        backlog_before = self._backlog_pairs()
        failed_before = set(self.scheduler.failed_rids)
        self._cleared: set[str] = set()

        self._execute(op)

        may_drop = (
            failed_before
            | set(self.scheduler.failed_rids)
            | set(self.scheduler.admit_errors)
            | self._cleared
        )
        vanished = backlog_before - self._backlog_pairs()
        self.outstanding -= {pair for pair in vanished if pair[0] in may_drop}

    def _execute(self, op: Op) -> None:
        if op.kind == "make_ready":
            rid, node = self._rid(op.args[0]), self._node(op.args[1])
            if (rid, node) in self._backlog_pairs():
                # Unreachable in the real worker: the batch already popped
                # this node, and nothing returns it before it runs.
                return
            self.queues[self.rid_wg[rid]].ensure(rid).ready_node_names.add(node)
            self.outstanding.add((rid, node))

        elif op.kind == "next_batch":
            self._next_batch(*op.args)

        elif op.kind == "hold":
            self.scheduler.hold_requests([self._rid(op.args[0])])

        elif op.kind == "advance_time":
            self.clock.now += op.args[0] / 1000.0

        elif op.kind == "fail":
            # `execute` already reconciles what a failed request may drop.
            self.scheduler.fail_rids({self._rid(op.args[0])})

        elif op.kind == "clear":
            rid = self._rid(op.args[0])
            self._cleared.add(rid)
            self.scheduler.clear_rid(rid)
            self._check_cleared(rid)

        elif op.kind == "set_not_ready":
            rid = self._rid(op.args[0])
            if op.args[1]:
                self.engine.not_ready.add(rid)
            else:
                self.engine.not_ready.discard(rid)

        elif op.kind == "set_unservable":
            self.engine.unservable.add(self._rid(op.args[0]))

        elif op.kind == "pending_remove":
            rid = self._rid(op.args[0])
            if op.args[1]:
                self.scheduler.pending_removes.add(rid)
            else:
                self.scheduler.pending_removes.discard(rid)

        elif op.kind == "forget_request":
            # Models a REMOVE_REQUEST arriving between scheduling cycles.
            self.manager.per_request_info.pop(self._rid(op.args[0]), None)

        elif op.kind == "tp_follow":
            node, walk = self._pair(op.args[0])
            rid = self._rid(op.args[1])
            self.scheduler.register_tp_follow(
                ScheduleTPNode(
                    node_name=node, graph_walk=walk, request_ids=[rid],
                )
            )

        else:
            raise AssertionError(f"unknown op {op.kind}")

    def _next_batch(
        self, max_bs: int, target_index: int, exclude_index: int, pre_existing: int,
    ) -> None:
        """Ask for one batch, then check the batch against the contract."""
        cap_arg = None if max_bs < 0 else max_bs
        target = None if target_index < 0 else self._pair(target_index)
        exclude = None if exclude_index < 0 else self._pair(exclude_index)
        backlog_before = set(self.scheduler.backlog)
        fifo_before = len(self.scheduler.tp_batches_pending_schedule)

        batch = self.scheduler.get_next_batch(
            self.manager,
            max_batch_size=cap_arg,
            target=target,
            exclude_target=exclude,
            pre_existing_batch_size=pre_existing,
        )
        if batch is None:
            return

        pair = (batch.node_name, batch.graph_walk)
        # A TP-follow batch is exempt from the cap: rank 0 already committed
        # to it and is waiting in the collective.
        from_tp_fifo = len(self.scheduler.tp_batches_pending_schedule) < fifo_before
        size = len(batch.node_objects)

        require(
            "sched.batch_is_not_empty",
            size > 0,
            f"get_next_batch returned an empty batch for {pair}",
        )
        require(
            "sched.batch_matches_target",
            target is None or pair == target,
            f"asked for {target} but got {pair}; the caller's own rows belong "
            "to a different node and the batch cannot run",
        )
        if exclude is not None and pair == exclude:
            require(
                "sched.exclude_only_bypassed_by_backlog_or_tp",
                from_tp_fifo or pair in backlog_before,
                f"{pair} was excluded but a freshly scanned batch came back for it",
            )

        effective_cap = cap_arg if cap_arg is not None else self.caps.get(pair)
        if effective_cap is not None and not from_tp_fifo:
            require(
                "sched.batch_respects_cap",
                size + pre_existing <= effective_cap,
                f"batch of {size} plus {pre_existing} pre-existing row(s) "
                f"exceeds the cap {effective_cap} for {pair}",
            )

        backlogged_now = self._backlog_pairs()
        for rid in batch.node_objects:
            require(
                "sched.scheduled_rid_not_also_backlogged",
                (rid, batch.node_name) not in backlogged_now,
                f"{rid} is in the returned {pair} batch and in the backlog at "
                "once; it would run twice",
            )
            if not from_tp_fifo:
                require(
                    "sched.does_not_schedule_dropped_rid",
                    rid not in self.scheduler.failed_rids
                    and rid not in self.scheduler.pending_removes,
                    f"{rid} was scheduled for {pair} after being failed or "
                    "marked for removal",
                )
            self.outstanding.discard((rid, batch.node_name))

    # -- invariants ----------------------------------------------------------

    def _check_cleared(self, rid: str) -> None:
        """Check that ``clear_rid`` removed every record of one request."""
        traces = []
        if rid in self.scheduler.failed_rids:
            traces.append("failed_rids")
        if rid in self.scheduler.admit_errors:
            traces.append("admit_errors")
        if rid in self.scheduler.held_until:
            traces.append("held_until")
        if rid in self.scheduler.pending_tp_follow_count:
            traces.append("pending_tp_follow_count")
        if any(rid in batch.node_objects for batch in self.scheduler.backlog.values()):
            traces.append("backlog")
        require(
            "sched.clear_rid_leaves_no_trace",
            not traces,
            f"clear_rid({rid}) left the request behind in {traces}",
        )

    def check(self) -> None:
        reachable = self._queue_pairs() | self._backlog_pairs()
        lost = self.outstanding - reachable
        require(
            "sched.no_ready_work_is_lost",
            not lost,
            f"{sorted(lost)} were made ready but are on no queue and in no "
            "backlog entry; those requests can never be scheduled again",
        )

        for node_walk, batch in self.scheduler.backlog.items():
            require(
                "sched.backlog_entries_are_never_empty",
                bool(batch.node_objects),
                f"backlog holds an empty batch under {node_walk}, which blocks "
                "that key from being re-created",
            )
            require(
                "sched.backlog_key_matches_batch",
                node_walk == (batch.node_name, batch.graph_walk),
                f"backlog key {node_walk} holds a batch for "
                f"{(batch.node_name, batch.graph_walk)}",
            )
            require(
                "sched.backlog_batch_is_consistent",
                set(batch.node_objects) == set(batch.request_to_worker_graph),
                f"backlog batch under {node_walk} has node_objects for "
                f"{sorted(batch.node_objects)} but worker graphs for "
                f"{sorted(batch.request_to_worker_graph)}",
            )

        for rid, count in self.scheduler.pending_tp_follow_count.items():
            require(
                "sched.tp_follow_count_is_positive",
                count > 0,
                f"pending_tp_follow_count[{rid}]={count}; a drain barrier keyed "
                "on this would never clear",
            )
        if not self.scheduler.tp_batches_pending_schedule:
            require(
                "sched.tp_follow_count_empty_with_empty_fifo",
                not self.scheduler.pending_tp_follow_count,
                "the TP-follow FIFO is empty but "
                f"{dict(self.scheduler.pending_tp_follow_count)} is still counted "
                "as pending",
            )

    def final_check(self) -> None:
        for rid in self.rids:
            self.scheduler.clear_rid(rid)
            self._check_cleared(rid)
        require(
            "sched.quiesce_empties_backlog",
            not self.scheduler.backlog,
            f"after clearing every request the backlog still holds "
            f"{sorted(self.scheduler.backlog)}",
        )
