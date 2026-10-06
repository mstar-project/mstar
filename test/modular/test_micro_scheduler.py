"""MicroScheduler batching, backlog and fairness.

The scheduler cuts a ready set down to one step and keeps the rest in a
backlog. Three things have to hold across that split:

* the cap covers the whole batch the caller will run, including rows it
  already has (the speculation path merges continuing rids with fresh ones);
* a backlogged chunk is re-checked before it goes out again — after a KV OOM
  the pages it needs may be gone, and it must not skip the hold backoff;
* every batch handed out counts as scheduling its (node, walk), or round-robin
  stops rotating.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources.step import (
    FULL_ADMIT_NOT_READY,
    FULL_ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    FullAdmitOutcome,
)
from mstar.graph.runtime.base import (
    ColumnarEdgeSpecs,
    PopRidsOutput,
    ReadyNodeSpec,
)
from mstar.utils.containers import ParallelList
from mstar.worker.batch_builder import (
    BaseBatchBuilder,
    BatchBuildRequest,
    BatchBuildResult,
    FIFOBatchBuilder,
)
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch


def _edge_block(rids) -> ColumnarEdgeSpecs:
    """One ready input per rid, so the split / drop paths that re-slice the
    columns have something to re-slice."""
    block = ColumnarEdgeSpecs.empty()
    for i, rid in enumerate(rids):
        block.add(rid, "token", [i + 1], False)
    return block

NODE = "LLM"
WALK = "decode"


class _Queue:
    def __init__(self, rids, node=NODE):
        self._ready = {rid: {node} for rid in rids}

    def get_ready_node_names(self):
        return {rid: set(names) for rid, names in self._ready.items()}

    def pop_ready_nodes(self, rid, node_names):
        if rid not in self._ready:
            return []
        self._ready.pop(rid)
        return [SimpleNamespace(name=node_names[0])]


class _Runtime:
    """Graph-level ready scan, over the manager's queues.

    The real PythonGraphRuntime owns the queues and RequestStateManager shares
    the same dict; this mirrors that by reading the manager's.
    """

    def __init__(self, manager: _Manager):
        self._manager = manager

    def _scan(self, exclude_rids, target, exclude_target):
        target_node, target_walk = target if target is not None else (None, None)
        for queue in self._manager.queues.values():
            for rid, node_names in queue.get_ready_node_names().items():
                if rid in exclude_rids \
                        or rid not in self._manager.per_request_info:
                    continue
                walk = self._manager.walk_of(rid)
                for node_name in node_names:
                    if target_node is not None and node_name != target_node:
                        continue
                    if target_walk is not None and walk != target_walk:
                        continue
                    if exclude_target is not None \
                            and (node_name, walk) == exclude_target:
                        continue
                    yield node_name, walk, rid

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        grouped: dict[tuple[str, str], list] = {}
        for node_name, walk, rid in self._scan(
            exclude_rids, target, exclude_target,
        ):
            grouped.setdefault((node_name, walk), []).append(rid)
        return [
            ReadyNodeSpec(node_name, walk, rids)
            for (node_name, walk), rids in grouped.items()
        ]

    def has_ready_excluding(self, exclude_rids, exclude_target=None):
        for _ in self._scan(exclude_rids, None, exclude_target):
            return True
        return False

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        del node_name, graph_walk
        return "wg0"

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        self._manager.pops.append((graph_walk, list(request_ids)))
        assert all(self._manager.walk_of(r) == graph_walk for r in request_ids), (
            "pop_rids resolves one worker graph per walk"
        )
        queue = self._manager.queues["wg0"]
        if check_ready:
            ready = queue.get_ready_node_names()
            for rid in request_ids:
                if node_name not in ready.get(rid, ()):
                    return None
        rids, wg_ids = [], []
        for rid in request_ids:
            if queue.pop_ready_nodes(rid, [node_name]):
                rids.append(rid)
                wg_ids.append("wg0")
        return PopRidsOutput(
            wg_ids=ParallelList(rids, wg_ids),
            input_edges=_edge_block(rids),
            output_signals=(f"out_{graph_walk}",),
        )

    def get_nodes(self, node_name, rids, wg_ids):
        del wg_ids
        return [SimpleNamespace(name=node_name) for _ in rids]

    def push_back_node(self, node_name, rids, wg_ids):
        del wg_ids
        queue = self._manager.queues["wg0"]
        for rid in rids:
            queue._ready.setdefault(rid, set()).add(node_name)

    def set_in_flight(self, node_name, wg_id, rids, in_flight):
        del wg_id
        for rid in rids:
            if in_flight:
                self._manager.in_flight.add((rid, node_name))
            else:
                self._manager.in_flight.discard((rid, node_name))


class _Manager:
    """Stands in for RequestStateManager: one worker graph, one node."""

    def __init__(self, rids, node=NODE, walk=WALK, walks=None):
        self.queues = {"wg0": _Queue(rids, node)}
        self.per_request_info = dict.fromkeys(rids, object())
        self._walk = walk
        # rid -> real walk, overriding ``walk``
        self.walks = dict(walks or {})
        self.pops: list = []
        self.runtime = _Runtime(self)
        # (rid, node) pairs marked in flight
        self.in_flight: set = set()

    def ingest(self, rid, node=NODE):
        """An input landing for a node whose inputs are already satisfied:
        the graph re-readies it unless it is in flight."""
        if (rid, node) not in self.in_flight:
            self.queues["wg0"]._ready.setdefault(rid, set()).add(node)

    def get_partition_for_node(self, node_name):
        del node_name
        return "default"

    def get_graph_walk(self, rid, partition):
        del partition
        return self.walk_of(rid)

    def walk_of(self, rid):
        return self.walks.get(rid, self._walk)

    def get_fwd_info(self, rid, partition):
        del rid, partition
        return object()


class _Engine:
    def __init__(
        self, max_bs=None, not_ready=frozenset(), unservable=frozenset(),
        groups=None,
    ):
        self._max_bs = max_bs
        self.not_ready = set(not_ready)
        # rids a resource rejects outright, as opposed to "not yet"
        self.unservable = set(unservable)
        # rid -> capture group; a rid not named here has none
        self.groups = dict(groups or {})

    def get_max_batch_size(self, node_name, graph_walk):
        del node_name
        if isinstance(self._max_bs, dict):
            return self._max_bs.get(graph_walk)
        return self._max_bs

    def capture_group(self, node_name, graph_walk, rid, fwd_info):
        del node_name, graph_walk, fwd_info
        return self.groups.get(rid)

    def check_ready(self, node_name, rid, fwd_info, allow_reload=True):
        del allow_reload, node_name, fwd_info
        if rid in self.unservable:
            return FullAdmitOutcome(
                AdmitOutcome(
                    ok=False, ready=False,
                    reason=AdmitRuntimeError(f"{rid} is doomed"),
                ),
                "kv",
            )
        if rid in self.not_ready:
            return FULL_ADMIT_NOT_READY
        return FULL_ADMIT_OK


def _scheduler(engine: _Engine) -> MicroScheduler:
    return MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine),
        parallel_leader_nodes={NODE},
    )


def _next_batch(sched: MicroScheduler, manager: _Manager, **kwargs):
    """Bind the runtime that scans this manager's queues, then schedule.

    The worker installs the runtime once at startup; these tests build a fresh
    manager per call, so the binding happens here instead.
    """
    sched.runtime = manager.runtime
    return sched.get_next_batch(manager, **kwargs)


def _has_ready(sched: MicroScheduler, manager: _Manager, exclude_target=None):
    sched.runtime = manager.runtime
    return sched.has_ready_excluding(manager, exclude_target)


def _batch(rids, node=NODE, walk=WALK) -> ScheduledBatch:
    return ScheduledBatch(
        node_name=node, graph_walk=walk,
        input_edges=_edge_block(rids),
        request_to_worker_graph=dict.fromkeys(rids, "wg0"),
    )


# ── capping ─────────────────────────────────────────────────────────────


def test_an_uncapped_node_takes_the_whole_ready_set():
    """`get_max_batch_size` returns None for a node with no cap; that must
    stay None all the way down rather than becoming an arithmetic operand."""
    sched = _scheduler(_Engine(max_bs=None))

    batch = _next_batch(sched, _Manager([f"r{i}" for i in range(5)]))

    assert len(batch) == 5
    assert not sched.backlog


def test_an_uncapped_node_survives_a_pre_existing_batch_size():
    sched = _scheduler(_Engine(max_bs=None))

    batch = _next_batch(sched,
        _Manager([f"r{i}" for i in range(3)]), pre_existing_batch_size=2,
    )

    assert len(batch) == 3


def test_the_cap_counts_rows_the_caller_already_has():
    """The regression: continuing and fresh rids were each capped, their
    union was not, so a merged batch could pass the node's max."""
    sched = _scheduler(_Engine(max_bs=4))

    batch = _next_batch(sched,
        _Manager([f"r{i}" for i in range(4)]), pre_existing_batch_size=3,
    )

    assert len(batch) == 1, "4 cap - 3 already held = 1"
    assert len(sched.backlog[(NODE, WALK)].request_to_worker_graph) == 3


def test_a_full_caller_batch_schedules_nothing_and_leaves_the_queue_alone():
    """With no capacity left there is nothing to assemble. These rids stay
    queued rather than being parked in a backlog the caller cannot drain."""
    sched = _scheduler(_Engine(max_bs=4))
    manager = _Manager([f"r{i}" for i in range(2)])

    batch = _next_batch(sched, manager, pre_existing_batch_size=4)

    assert batch is None
    assert sched.backlog == {}
    assert set(manager.queues["wg0"]._ready) == {"r0", "r1"}


def test_a_fully_drained_ready_set_leaves_no_backlog_entry():
    """A None remainder must not be stored: `_drop_backlogged_rid` walks these."""
    sched = _scheduler(_Engine(max_bs=8))

    _next_batch(sched, _Manager(["r0", "r1"]))

    assert sched.backlog == {}
    sched._drop_backlogged_rid("r0")  # would raise on a stored None


# ── backlog ─────────────────────────────────────────────────────────────


def test_a_backlogged_chunk_is_rechecked_before_going_back_out():
    """After a KV OOM the rest of a split ready set must not go straight back
    against the same exhausted pages."""
    engine = _Engine(max_bs=2, not_ready={"r2"})
    sched = _scheduler(engine)
    sched.backlog[(NODE, WALK)] = _batch(["r2", "r3"])

    batch = _next_batch(sched, _Manager([]))

    assert list(batch.request_to_worker_graph) == ["r3"]
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["r2"]


def test_an_uncapped_node_can_be_served_from_the_backlog():
    """The backlog path resolves the cap itself, and `None` has to survive
    that resolution rather than being subtracted from."""
    sched = _scheduler(_Engine(max_bs=None))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1"])

    batch = _next_batch(sched, _Manager([]), pre_existing_batch_size=1)

    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert sched.backlog == {}


def test_a_backlogged_chunk_that_is_wholly_unready_stays_put():
    engine = _Engine(max_bs=2, not_ready={"r2", "r3"})
    sched = _scheduler(engine)
    sched.backlog[(NODE, WALK)] = _batch(["r2", "r3"])

    assert _next_batch(sched, _Manager([])) is None
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["r2", "r3"]


def test_a_blocked_chunk_is_skipped_for_the_next_one():
    """A walk whose pages went to an eviction must not hold the worker idle
    behind it when another backlogged walk is runnable."""
    sched = _scheduler(_Engine(max_bs=8, not_ready={"r0"}))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    batch = _next_batch(sched, _Manager([]))

    assert batch.node_name == "B"
    # the blocked one is kept, for a later pass
    assert list(sched.backlog[("A", WALK)].request_to_worker_graph) == ["r0"]


def test_skipping_does_not_lose_a_blocked_chunk_when_nothing_else_runs():
    sched = _scheduler(_Engine(max_bs=8, not_ready={"r0", "r1"}))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    assert _next_batch(sched, _Manager([])) is None
    assert set(sched.backlog) == {("A", WALK), ("B", WALK)}


def test_a_targeted_call_does_not_skip_to_another_walk():
    """Targeting is a hard filter: the speculation path merges what it gets
    into a batch labelled with its own node, so a different walk's chunk
    would be mislabelled."""
    sched = _scheduler(_Engine(max_bs=8, not_ready={"r0"}))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    assert _next_batch(sched, _Manager([]), target=("A", WALK)) is None
    assert set(sched.backlog) == {("A", WALK), ("B", WALK)}


def test_a_backlogged_chunk_still_respects_the_nodes_cap():
    """The backlog path resolves the cap itself; dropping it on the floor
    sends an oversized batch out, which fails `can_batch` and degrades to one
    eager forward per request."""
    sched = _scheduler(_Engine(max_bs=2))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1", "r2", "r3"])

    batch = _next_batch(sched, _Manager([]))

    assert list(batch.request_to_worker_graph) == ["r0", "r1"]
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["r2", "r3"]


def test_the_hold_backoff_expires_before_the_backlog_is_taken():
    """The expiry used to sit after the backlog's early return, so a
    backlogged chunk skipped it entirely."""
    sched = _scheduler(_Engine(max_bs=8))
    sched.hold_requests(["r0"])
    sched.held_until["r0"] = 0.0  # already elapsed
    sched.backlog[(NODE, WALK)] = _batch(["r0"])

    _next_batch(sched, _Manager([]))

    assert "r0" not in sched.held_until


def test_the_oldest_backlog_entry_goes_first():
    """FIFO, as the deque was. `popitem` takes the newest, which starves an
    entry that keeps being overtaken."""
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    assert _next_batch(sched, _Manager([])).node_name == "A"
    assert _next_batch(sched, _Manager([])).node_name == "B"


def test_a_targeted_call_takes_only_its_own_backlog_entry():
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    batch = _next_batch(sched, _Manager([]), target=("B", WALK))

    assert batch.node_name == "B"
    assert ("A", WALK) in sched.backlog


def test_a_full_caller_batch_does_not_clobber_the_backlog():
    """The regression: above the node's cap the caller already holds a full
    batch, so nothing can be scheduled — but the scan still popped fresh nodes
    off the ready queues and parked them under the backlog's own key, dropping
    the chunk that was already there. Those rids are off the queues, so they
    never run again."""
    sched = _scheduler(_Engine(max_bs=16))
    backlogged = [f"b{i}" for i in range(8)]
    sched.backlog[(NODE, WALK)] = _batch(backlogged)

    # the speculative batch already holds the full cap
    batch = _next_batch(sched,
        _Manager(["f0", "f1"]), target=(NODE, WALK), pre_existing_batch_size=16,
    )

    assert batch is None, "no capacity left, so nothing should be scheduled"
    still_queued = set(sched.backlog[(NODE, WALK)].request_to_worker_graph)
    assert set(backlogged) <= still_queued, (
        f"lost {sorted(set(backlogged) - still_queued)} from the backlog"
    )


def test_fresh_work_does_not_evict_a_blocked_backlog_chunk():
    """Capacity remains, but a wholly-blocked chunk is already parked under
    this key. The scan's leftovers must fold in beside it — replacing drops
    rids whose graph nodes are already off the ready queues."""
    sched = _scheduler(_Engine(max_bs=2, not_ready={"b0", "b1"}))
    sched.backlog[(NODE, WALK)] = _batch(["b0", "b1"])

    batch = _next_batch(sched, _Manager(["f0", "f1", "f2"]))

    assert set(batch.request_to_worker_graph) == {"f0", "f1"}
    held = set(sched.backlog[(NODE, WALK)].request_to_worker_graph)
    assert held == {"b0", "b1", "f2"}, f"lost the blocked chunk: {held}"


def test_fresh_work_merges_into_an_existing_backlog_entry():
    sched = _scheduler(_Engine(max_bs=2))
    sched.backlog[(NODE, WALK)] = _batch(["b0", "b1"])

    _next_batch(sched, _Manager(["f0", "f1", "f2"]), pre_existing_batch_size=2)

    held = set(sched.backlog[(NODE, WALK)].request_to_worker_graph)
    assert {"b0", "b1"} <= held, "the older chunk must survive"


def test_no_capacity_leaves_the_ready_queue_alone():
    """Popping nodes we cannot schedule is what makes the loss possible."""
    sched = _scheduler(_Engine(max_bs=16))
    manager = _Manager(["f0", "f1"])

    _next_batch(sched, manager, pre_existing_batch_size=16)

    assert set(manager.queues["wg0"]._ready) == {"f0", "f1"}, (
        "fresh nodes were popped off the queue with nowhere to put them"
    )


# ── backlog vs speculation ──────────────────────────────────────────────


def test_room_for_continuing_reserves_the_backlog_first():
    """32 decodes at a cap of 16: the 16 waiting take the whole next batch, so
    the spec chain keeps none of its own. Without this the chain re-speculates
    the same 16 forever and the other half never runs."""
    sched = _scheduler(_Engine(max_bs=16))
    sched.backlog[(NODE, WALK)] = _batch([f"b{i}" for i in range(16)])

    assert sched.room_for_continuing((NODE, WALK)) == 0


def test_room_for_continuing_splits_a_partial_backlog():
    sched = _scheduler(_Engine(max_bs=16))
    sched.backlog[(NODE, WALK)] = _batch([f"b{i}" for i in range(6)])

    assert sched.room_for_continuing((NODE, WALK)) == 10


def test_room_for_continuing_is_unbounded_without_a_cap():
    sched = _scheduler(_Engine(max_bs=None))
    sched.backlog[(NODE, WALK)] = _batch(["b0"])

    assert sched.room_for_continuing((NODE, WALK)) is None


def test_the_reserved_room_is_exactly_what_the_backlog_then_fills():
    """The two halves have to agree: whatever `room_for_continuing` holds
    back, the follow-up call must actually hand over."""
    sched = _scheduler(_Engine(max_bs=16))
    backlogged = [f"b{i}" for i in range(16)]
    sched.backlog[(NODE, WALK)] = _batch(backlogged)

    keep = sched.room_for_continuing((NODE, WALK))
    batch = _next_batch(sched,
        _Manager([]), target=(NODE, WALK), pre_existing_batch_size=keep,
    )

    assert set(batch.request_to_worker_graph) == set(backlogged)
    assert sched.backlog == {}


# ── fairness peek ───────────────────────────────────────────────────────


def test_the_peek_sees_a_backlogged_walk_the_queues_cannot_show():
    """A backlogged chunk's nodes are off the ready queues, so the scan alone
    reports no contention and the spec chain never yields to it."""
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("other", "prefill")] = _batch(["b0"], node="other", walk="prefill")

    assert _has_ready(sched, _Manager([]), (NODE, WALK)) is True


def test_the_peek_ignores_the_speculated_walks_own_backlog():
    """That chunk is absorbed by the merge (`room_for_continuing`), so it is
    not a reason to break the chain."""
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[(NODE, WALK)] = _batch(["b0"])

    assert _has_ready(sched, _Manager([]), (NODE, WALK)) is False


def test_the_peek_ignores_a_backlog_of_failed_rids():
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("other", "prefill")] = _batch(["b0"], node="other", walk="prefill")
    sched.failed_rids.add("b0")

    assert _has_ready(sched, _Manager([]), (NODE, WALK)) is False


# ── capture groups ──────────────────────────────────────────────────────

# chatterbox T3: guided (cfg_weight > 0) and unguided requests replay
# different decode captures, so a batch of both runs eager
_MIXED = {"g0": True, "u0": False, "g1": True, "u1": False}


def test_a_mixed_ready_set_schedules_one_capture_group():
    """The regression: a mixed batch ran T3 eager one request at a time."""
    sched = _scheduler(_Engine(max_bs=16, groups=_MIXED))

    batch = _next_batch(sched, _Manager(list(_MIXED)))

    assert list(batch.request_to_worker_graph) == ["g0", "g1"]
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["u0", "u1"]


def test_the_other_group_runs_next_from_the_backlog():
    sched = _scheduler(_Engine(max_bs=16, groups=_MIXED))
    _next_batch(sched, _Manager(list(_MIXED)))

    batch = _next_batch(sched, _Manager([]))

    assert list(batch.request_to_worker_graph) == ["u0", "u1"]
    assert sched.backlog == {}


def test_a_mixed_backlog_is_split_too():
    sched = _scheduler(_Engine(max_bs=16, groups=_MIXED))
    sched.backlog[(NODE, WALK)] = _batch(["u0", "g0", "u1", "g1"])

    batch = _next_batch(sched, _Manager([]))

    assert list(batch.request_to_worker_graph) == ["u0", "u1"]
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["g0", "g1"]


def test_an_unready_head_does_not_set_the_group():
    """The group comes from the first rid that can run, not one that waits."""
    sched = _scheduler(_Engine(max_bs=16, groups=_MIXED, not_ready={"u0"}))
    sched.backlog[(NODE, WALK)] = _batch(["u0", "g0", "u1", "g1"])

    batch = _next_batch(sched, _Manager([]))

    assert list(batch.request_to_worker_graph) == ["g0", "g1"]


def test_without_capture_groups_nothing_is_split():
    sched = _scheduler(_Engine(max_bs=16))

    batch = _next_batch(sched, _Manager(list(_MIXED)))

    assert len(batch) == 4
    assert sched.backlog == {}


def test_the_speculation_merge_takes_only_the_chains_group():
    """Fresh rids merged into a guided chain must be guided; the rest wait."""
    sched = _scheduler(_Engine(max_bs=16, groups={**_MIXED, "c0": True}))

    batch = _next_batch(sched,
        _Manager(["u0", "g0", "u1", "g1"]), target=(NODE, WALK),
        pre_existing_batch_size=1, capture_group_of="c0",
    )

    assert list(batch.request_to_worker_graph) == ["g0", "g1"]
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["u0", "u1"]


def test_a_merge_with_nothing_in_its_group_backlogs_the_rest():
    sched = _scheduler(_Engine(max_bs=16, groups={**_MIXED, "c0": True}))

    batch = _next_batch(sched,
        _Manager(["u0", "u1"]), target=(NODE, WALK),
        pre_existing_batch_size=1, capture_group_of="c0",
    )

    assert batch is None
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["u0", "u1"]


def test_a_chain_yields_to_another_groups_backlog():
    """The merge never takes those rids, so without a yield they starve."""
    sched = _scheduler(_Engine(max_bs=16, groups={**_MIXED, "c0": True}))
    sched.backlog[(NODE, WALK)] = _batch(["u0", "g0"])

    assert sched.backlog_splits_from(_Manager([]), (NODE, WALK), "c0") is True


def test_a_chain_keeps_going_over_its_own_groups_backlog():
    sched = _scheduler(_Engine(max_bs=16, groups={**_MIXED, "c0": True}))
    sched.backlog[(NODE, WALK)] = _batch(["g0", "g1"])

    assert sched.backlog_splits_from(_Manager([]), (NODE, WALK), "c0") is False


def test_a_chain_does_not_yield_to_a_failed_rid():
    sched = _scheduler(_Engine(max_bs=16, groups={**_MIXED, "c0": True}))
    sched.backlog[(NODE, WALK)] = _batch(["u0"])
    sched.fail_rids({"u0"})

    assert sched.backlog_splits_from(_Manager([]), (NODE, WALK), "c0") is False


# ── round robin ─────────────────────────────────────────────────────────


def test_scheduling_a_batch_advances_the_round_robin_cursor():
    """The regression: the bookkeeping moved out of batch assembly, so every
    (node, walk) stayed at 0 and `_select_node_rr` kept picking the same one."""
    sched = _scheduler(_Engine(max_bs=2))

    _next_batch(sched, _Manager([f"r{i}" for i in range(4)]))

    assert sched.node_and_walk_to_last_batch_num[(NODE, WALK)] > 0


def test_serving_from_the_backlog_also_advances_the_cursor():
    sched = _scheduler(_Engine(max_bs=1))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1"])

    _next_batch(sched, _Manager([]))
    first = sched.node_and_walk_to_last_batch_num[(NODE, WALK)]
    _next_batch(sched, _Manager([]))

    assert sched.node_and_walk_to_last_batch_num[(NODE, WALK)] > first


def test_the_cursor_does_not_advance_when_nothing_is_scheduled():
    sched = _scheduler(_Engine(max_bs=4))

    _next_batch(sched, _Manager(["r0"]), pre_existing_batch_size=4)

    assert (NODE, WALK) not in sched.node_and_walk_to_last_batch_num


@pytest.mark.parametrize("exclude", [None, set()])
def test_split_off_first_accepts_an_empty_exclusion(exclude):
    first, rest = _batch(["r0", "r1", "r2"]).split_off_first(2, exclude)

    assert list(first.request_to_worker_graph) == ["r0", "r1"]
    assert list(rest.request_to_worker_graph) == ["r2"]


# ── terminal admit failures ─────────────────────────────────────────────


def test_an_unservable_rid_is_parked_for_the_worker_to_fail():
    """A resource that rejects a request outright (a failed KV transfer) must
    not leave it rescanned forever: the rid is dropped from scheduling and
    handed to the worker, which reports it to the conductor."""
    sched = _scheduler(_Engine(max_bs=4, unservable={"r0"}))

    batch = _next_batch(sched, _Manager(["r0", "r1"]))

    assert list(batch.request_to_worker_graph) == ["r1"]
    assert sched.failed_rids == {"r0"}
    errors = sched.take_admit_errors()
    assert set(errors) == {"r0"}
    assert "r0 is doomed" in errors["r0"]
    # drained exactly once, so the worker does not re-report it every iteration
    assert sched.take_admit_errors() == {}


def test_an_unservable_rid_is_dropped_from_the_backlog():
    sched = _scheduler(_Engine(max_bs=4, unservable={"r0"}))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1"])

    batch = _next_batch(sched, _Manager([]))

    assert list(batch.request_to_worker_graph) == ["r1"]
    assert set(sched.take_admit_errors()) == {"r0"}
    assert sched.backlog == {}


def test_clearing_a_rid_forgets_its_undelivered_admit_error():
    sched = _scheduler(_Engine(max_bs=4, unservable={"r0"}))
    _next_batch(sched, _Manager(["r0"]))

    sched.clear_rid("r0", "r0")

    assert sched.take_admit_errors() == {}
    assert sched.failed_rids == set()


# ── engine capture-group gate ───────────────────────────────────────────


class _Submodule:
    def __init__(self, split):
        self._split = split

    def split_batches_by_capture_key(self, graph_walk):
        del graph_walk
        return self._split

    def cg_key_info(self, graph_walk, per_request_info):
        del graph_walk
        (info,) = per_request_info.values()
        return info.guided


def _capture_group(split=True, captured=(WALK,)):
    from mstar.engine.engine import Engine

    runner = SimpleNamespace(captures_walk=lambda walk: walk in captured)
    engine = SimpleNamespace(_submodules={
        NODE: SimpleNamespace(submodule=_Submodule(split), cuda_graph_runner=runner),
    })
    return Engine.capture_group(engine, NODE, WALK, 0, SimpleNamespace(guided=True))


def test_an_opted_in_submodule_groups_by_its_capture_key():
    assert _capture_group() is True


def test_a_submodule_that_does_not_opt_in_is_never_split():
    """The default: a mixed batch keeps whatever path it ran before."""
    assert _capture_group(split=False) is None


def test_a_walk_without_a_capture_is_never_split():
    assert _capture_group(captured=()) is None


# ── pluggable batch builder ─────────────────────────────────────────────


class _ReturnOddBuilder(BaseBatchBuilder):
    """Runs the even-numbered rids and hands the odd ones back to the queue."""

    def __init__(self):
        self.requests: list[BatchBuildRequest] = []

    def build_batch(self, request):
        self.requests.append(request)
        batch = request.fresh
        odd = {rid for rid in batch.request_to_worker_graph if int(rid[1:]) % 2}
        scheduled, returned = batch.split_off_first(None, exclude_rids=odd)
        return BatchBuildResult(scheduled, returned=returned.request_to_worker_graph)


def test_the_scheduler_runs_the_configured_builder():
    sched = _scheduler(_Engine(max_bs=8))
    builder = sched.batch_builder = _ReturnOddBuilder()
    manager = _Manager(["r0", "r1", "r2", "r3"])

    batch = _next_batch(sched, manager, capture_group_of="r2")

    assert list(batch.request_to_worker_graph) == ["r0", "r2"]
    (request,) = builder.requests
    assert (request.node_name, request.graph_walk) == (NODE, WALK)
    assert request.max_batch_size == 8
    assert request.capture_group_of == "r2"


def test_returned_rows_go_back_to_their_ready_queue_not_the_backlog():
    sched = _scheduler(_Engine(max_bs=8))
    sched.batch_builder = _ReturnOddBuilder()
    manager = _Manager(["r0", "r1", "r2", "r3"])

    _next_batch(sched, manager)

    assert not sched.backlog
    assert set(manager.queues["wg0"].get_ready_node_names()) == {"r1", "r3"}


def test_fifo_puts_the_backlog_ahead_of_fresh_rows():
    """Not reachable through `get_next_batch` yet, which serves a backlog on
    its own; a builder handed both must still keep arrival order."""
    builder = FIFOBatchBuilder()
    result = builder.build_batch(BatchBuildRequest(
        node_name=NODE, graph_walk=WALK,
        backlog=_batch(["b0", "b1"]),
        fresh=_batch(["f0", "f1"]),
        max_batch_size=3,
    ))

    assert list(result.scheduled.request_to_worker_graph) == ["b0", "b1", "f0"]
    assert list(result.backlog.request_to_worker_graph) == ["f1"]
    assert result.returned is None


# ── one builder call per step ───────────────────────────────────────────


def test_fresh_rows_join_a_short_backlog_in_one_step():
    """A one-row remainder must not run alone while its walk has fresh rows
    queued: that step would run at a fraction of the batch."""
    sched = _scheduler(_Engine(max_bs=3))
    sched.backlog[(NODE, WALK)] = _batch(["b0"])

    batch = _next_batch(sched, _Manager(["f0", "f1", "f2"]))

    assert list(batch.request_to_worker_graph) == ["b0", "f0", "f1"]
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["f2"]


def test_a_backlogged_walk_goes_before_a_less_recent_fresh_one():
    sched = _scheduler(_Engine(max_bs=8))
    sched.node_and_walk_to_last_batch_num[("B", WALK)] = 5
    sched.backlog[("B", WALK)] = _batch(["b0"], node="B")

    batch = _next_batch(sched, _Manager(["f0"]))

    assert (batch.node_name, list(batch.request_to_worker_graph)) == ("B", ["b0"])


def test_a_blocked_backlog_falls_through_to_the_ready_scan():
    sched = _scheduler(_Engine(max_bs=8, not_ready={"b0"}))
    sched.backlog[("B", WALK)] = _batch(["b0"], node="B")

    batch = _next_batch(sched, _Manager(["f0"]))

    assert (batch.node_name, list(batch.request_to_worker_graph)) == (NODE, ["f0"])
    assert list(sched.backlog[("B", WALK)].request_to_worker_graph) == ["b0"]


def test_a_full_caller_pops_neither_the_backlog_nor_the_queue():
    sched = _scheduler(_Engine(max_bs=2))
    sched.backlog[(NODE, WALK)] = _batch(["b0"])
    manager = _Manager(["f0"])

    assert _next_batch(
        sched, manager, target=(NODE, WALK), pre_existing_batch_size=2,
    ) is None
    assert list(sched.backlog[(NODE, WALK)].request_to_worker_graph) == ["b0"]
    assert set(manager.queues["wg0"].get_ready_node_names()) == {"f0"}


def test_a_tp_follow_batch_goes_before_the_backlog():
    """The leader already sits in the collective for it."""
    from mstar.utils.ipc_format import ScheduleTPNode

    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("B", WALK)] = _batch(["b0"], node="B")
    sched.register_tp_follow(
        ScheduleTPNode(node_name=NODE, graph_walk=WALK, request_ids=["t0"]),
    )

    batch = _next_batch(sched, _Manager(["t0"]))

    assert (batch.node_name, list(batch.request_to_worker_graph)) == (NODE, ["t0"])
    assert ("B", WALK) in sched.backlog


def test_an_input_landing_on_a_backlogged_row_does_not_ready_it_again():
    """A backlogged row's node is off its ready queue but not yet running. A
    streamed input arriving for it then re-readied it, the next scan popped it
    again, and the rid ran twice."""
    sched = _scheduler(_Engine(max_bs=1))
    manager = _Manager(["r0", "r1"])

    assert list(_next_batch(sched, manager).request_to_worker_graph) == ["r0"]
    assert ("r1", NODE) in manager.in_flight
    manager.ingest("r1")

    assert list(_next_batch(sched, manager).request_to_worker_graph) == ["r1"]
    assert _next_batch(sched, manager) is None


def test_returned_rows_are_no_longer_in_flight():
    sched = _scheduler(_Engine(max_bs=8))
    sched.batch_builder = _ReturnOddBuilder()
    manager = _Manager(["r0", "r1"])
    manager.in_flight.add(("r1", NODE))

    _next_batch(sched, manager)

    assert ("r1", NODE) not in manager.in_flight


# ── combined walks ──────────────────────────────────────────────────────

MIXED = "mixed"


def _combined_scheduler(engine: _Engine) -> MicroScheduler:
    return MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine),
        parallel_leader_nodes={NODE},
        combined_walk_of={(NODE, "prefill"): MIXED, (NODE, WALK): MIXED},
    )


def test_a_combined_walk_batches_its_walks_together():
    sched = _combined_scheduler(_Engine(max_bs=8))
    manager = _Manager(["p0", "d0", "d1"], walks={"p0": "prefill"})

    batch = _next_batch(sched, manager)

    assert batch.graph_walk == MIXED
    assert batch.request_walks == {"p0": "prefill", "d0": WALK, "d1": WALK}
    assert sorted(walk for walk, _ in manager.pops) == ["decode", "prefill"]


def test_a_combined_batch_of_one_walk_keeps_its_real_label():
    """A pure-decode step still replays the decode captures."""
    sched = _combined_scheduler(_Engine(max_bs=8))

    batch = _next_batch(sched, _Manager(["d0", "d1"]))

    assert batch.graph_walk == WALK
    assert batch.request_walks == {}


def test_the_backlog_and_round_robin_key_on_the_combined_walk():
    sched = _combined_scheduler(_Engine(max_bs=2))
    manager = _Manager(["p0", "d0", "d1"], walks={"p0": "prefill"})

    first = _next_batch(sched, manager)
    assert set(sched.backlog) == {(NODE, MIXED)}
    assert (NODE, MIXED) in sched.node_and_walk_to_last_batch_num
    second = _next_batch(sched, manager)

    assert len(first) + len(second) == 3
    assert second.walk_of("d1") == WALK


def test_a_real_walk_target_takes_its_combined_walks_rows():
    """The speculation merge names the walk it continues; fresh rows of the
    other constituent walk ride along."""
    sched = _combined_scheduler(_Engine(max_bs=8))
    manager = _Manager(["p0", "d0"], walks={"p0": "prefill"})

    batch = _next_batch(sched, manager, target=(NODE, WALK))

    assert set(batch.request_to_worker_graph) == {"p0", "d0"}


def test_excluding_a_real_walk_excludes_its_combined_walk():
    sched = _combined_scheduler(_Engine(max_bs=8))
    manager = _Manager(["p0", "d0"], walks={"p0": "prefill"})

    assert not _has_ready(sched, manager, exclude_target=(NODE, "prefill"))
    assert _next_batch(sched, manager, exclude_target=(NODE, WALK)) is None


def test_a_split_keeps_each_halfs_walks():
    batch = _batch(["p0", "d0", "d1"], walk=MIXED)
    batch.request_walks = {"p0": "prefill", "d0": WALK, "d1": WALK}

    first, rest = batch.split_off_first(2)

    assert first.request_walks == {"p0": "prefill", "d0": WALK}
    assert rest.request_walks == {"d1": WALK}


def test_a_follower_pops_each_walk_of_a_combined_head():
    sched = _combined_scheduler(_Engine(max_bs=8))
    manager = _Manager(["p0", "d0"], walks={"p0": "prefill"})
    sched.runtime = manager.runtime

    popped = sched.pop_ready_rids(
        manager, NODE, MIXED, ["p0", "d0"], request_walks=["prefill", WALK],
    )

    assert set(popped.wg_ids) == {"p0", "d0"}
    assert popped.request_walks == {"p0": "prefill", "d0": WALK}


def test_a_follower_pop_is_all_or_none_across_walks():
    sched = _combined_scheduler(_Engine(max_bs=8))
    manager = _Manager(["p0"], walks={"p0": "prefill", "d0": WALK})
    manager.per_request_info["d0"] = object()  # known, but its node is not ready
    sched.runtime = manager.runtime

    assert sched.pop_ready_rids(
        manager, NODE, MIXED, ["p0", "d0"], request_walks=["prefill", WALK],
    ) is None
    assert "p0" in manager.queues["wg0"].get_ready_node_names()


def test_a_walk_that_would_shrink_the_step_waits_its_turn():
    """Six decode rows fit a decode step of 8; taking the prompt would cap the
    step at its walk's 2. It is left out, and runs first next step."""
    sched = _combined_scheduler(_Engine(max_bs={MIXED: 8, WALK: 8, "prefill": 2}))
    rids = [f"d{i}" for i in range(3)] + ["p0"] + [f"d{i}" for i in range(3, 6)]
    manager = _Manager(rids, walks={"p0": "prefill"})

    first = _next_batch(sched, manager)
    assert first.graph_walk == WALK and len(first) == 6
    assert list(sched.backlog[(NODE, MIXED)].request_to_worker_graph) == ["p0"]

    manager.queues["wg0"]._ready.update({f"d{i}": {NODE} for i in range(6)})
    second = _next_batch(sched, manager)
    assert second.graph_walk == MIXED and len(second) == 2
    assert next(iter(second.request_to_worker_graph)) == "p0"


def test_a_walk_joins_when_it_costs_nothing():
    sched = _combined_scheduler(_Engine(max_bs={MIXED: 8, WALK: 8, "prefill": 4}))
    manager = _Manager(["d0", "p0", "d1"], walks={"p0": "prefill"})

    batch = _next_batch(sched, manager)

    assert batch.graph_walk == MIXED and len(batch) == 3



def test_a_relabelled_step_routes_with_its_own_walks_outputs():
    """The merged batch keeps its first walk's output names; a step that turns
    out to be one walk must route with that walk's, or a decode row loses its
    loop-back edge."""
    sched = _combined_scheduler(_Engine(max_bs=8, not_ready={"p0"}))
    backlogged = _batch(["p0"], walk=MIXED)
    backlogged.request_walks = {"p0": "prefill"}
    backlogged.output_signals = ["out_prefill"]
    backlogged.walk_output_signals = {"prefill": ["out_prefill"]}
    sched.backlog[(NODE, MIXED)] = backlogged

    batch = _next_batch(sched, _Manager(["d0", "d1"]))

    assert batch.graph_walk == WALK
    assert list(batch.output_signals) == [f"out_{WALK}"]
