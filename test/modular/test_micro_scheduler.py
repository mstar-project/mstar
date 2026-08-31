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

from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch

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


class _Manager:
    """Stands in for WorkerGraphsManager: one worker graph, one node."""

    def __init__(self, rids, node=NODE, walk=WALK):
        self.queues = {"wg0": _Queue(rids, node)}
        self.per_request_info = dict.fromkeys(rids, object())
        self._walk = walk

    def get_partition_for_node(self, node_name):
        del node_name
        return "default"

    def get_graph_walk(self, rid, partition):
        del rid, partition
        return self._walk

    def get_fwd_info(self, rid, partition):
        del rid, partition
        return object()


class _Engine:
    def __init__(self, max_bs=None, not_ready=frozenset()):
        self._max_bs = max_bs
        self.not_ready = set(not_ready)

    def get_max_batch_size(self, node_name, graph_walk):
        del node_name, graph_walk
        return self._max_bs

    def check_ready(self, node_name, rid, fwd_info):
        del node_name, fwd_info
        return rid not in self.not_ready


def _scheduler(engine: _Engine) -> MicroScheduler:
    return MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: engine),
        parallel_leader_nodes={NODE},
    )


def _batch(rids, node=NODE, walk=WALK) -> ScheduledBatch:
    return ScheduledBatch(
        node_name=node, graph_walk=walk,
        node_objects={rid: object() for rid in rids},
        request_to_worker_graph=dict.fromkeys(rids, "wg0"),
    )


# ── capping ─────────────────────────────────────────────────────────────


def test_an_uncapped_node_takes_the_whole_ready_set():
    """`get_max_batch_size` returns None for a node with no cap; that must
    stay None all the way down rather than becoming an arithmetic operand."""
    sched = _scheduler(_Engine(max_bs=None))

    batch = sched.get_next_batch(_Manager([f"r{i}" for i in range(5)]))

    assert len(batch.node_objects) == 5
    assert not sched.backlog


def test_an_uncapped_node_survives_a_pre_existing_batch_size():
    sched = _scheduler(_Engine(max_bs=None))

    batch = sched.get_next_batch(
        _Manager([f"r{i}" for i in range(3)]), pre_existing_batch_size=2,
    )

    assert len(batch.node_objects) == 3


def test_the_cap_counts_rows_the_caller_already_has():
    """The regression: continuing and fresh rids were each capped, their
    union was not, so a merged batch could pass the node's max."""
    sched = _scheduler(_Engine(max_bs=4))

    batch = sched.get_next_batch(
        _Manager([f"r{i}" for i in range(4)]), pre_existing_batch_size=3,
    )

    assert len(batch.node_objects) == 1, "4 cap - 3 already held = 1"
    assert len(sched.backlog[(NODE, WALK)].node_objects) == 3


def test_a_full_caller_batch_schedules_nothing_and_backlogs_it_all():
    sched = _scheduler(_Engine(max_bs=4))

    batch = sched.get_next_batch(
        _Manager([f"r{i}" for i in range(2)]), pre_existing_batch_size=4,
    )

    assert batch is None
    assert len(sched.backlog[(NODE, WALK)].node_objects) == 2


def test_a_fully_drained_ready_set_leaves_no_backlog_entry():
    """A None remainder must not be stored: `_drop_backlogged_rid` walks these."""
    sched = _scheduler(_Engine(max_bs=8))

    sched.get_next_batch(_Manager(["r0", "r1"]))

    assert sched.backlog == {}
    sched._drop_backlogged_rid("r0")  # would raise on a stored None


# ── backlog ─────────────────────────────────────────────────────────────


def test_a_backlogged_chunk_is_rechecked_before_going_back_out():
    """After a KV OOM the rest of a split ready set must not go straight back
    against the same exhausted pages."""
    engine = _Engine(max_bs=2, not_ready={"r2"})
    sched = _scheduler(engine)
    sched.backlog[(NODE, WALK)] = _batch(["r2", "r3"])

    batch = sched.get_next_batch(_Manager([]))

    assert list(batch.node_objects) == ["r3"]
    assert list(sched.backlog[(NODE, WALK)].node_objects) == ["r2"]


def test_an_uncapped_node_can_be_served_from_the_backlog():
    """The backlog path resolves the cap itself, and `None` has to survive
    that resolution rather than being subtracted from."""
    sched = _scheduler(_Engine(max_bs=None))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1"])

    batch = sched.get_next_batch(_Manager([]), pre_existing_batch_size=1)

    assert list(batch.node_objects) == ["r0", "r1"]
    assert sched.backlog == {}


def test_a_backlogged_chunk_that_is_wholly_unready_stays_put():
    engine = _Engine(max_bs=2, not_ready={"r2", "r3"})
    sched = _scheduler(engine)
    sched.backlog[(NODE, WALK)] = _batch(["r2", "r3"])

    assert sched.get_next_batch(_Manager([])) is None
    assert list(sched.backlog[(NODE, WALK)].node_objects) == ["r2", "r3"]


def test_a_blocked_chunk_is_skipped_for_the_next_one():
    """A walk whose pages went to an eviction must not hold the worker idle
    behind it when another backlogged walk is runnable."""
    sched = _scheduler(_Engine(max_bs=8, not_ready={"r0"}))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    batch = sched.get_next_batch(_Manager([]))

    assert batch.node_name == "B"
    # the blocked one is kept, for a later pass
    assert list(sched.backlog[("A", WALK)].node_objects) == ["r0"]


def test_skipping_does_not_lose_a_blocked_chunk_when_nothing_else_runs():
    sched = _scheduler(_Engine(max_bs=8, not_ready={"r0", "r1"}))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    assert sched.get_next_batch(_Manager([])) is None
    assert set(sched.backlog) == {("A", WALK), ("B", WALK)}


def test_a_targeted_call_does_not_skip_to_another_walk():
    """Targeting is a hard filter: the speculation path merges what it gets
    into a batch labelled with its own node, so a different walk's chunk
    would be mislabelled."""
    sched = _scheduler(_Engine(max_bs=8, not_ready={"r0"}))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    assert sched.get_next_batch(_Manager([]), target=("A", WALK)) is None
    assert set(sched.backlog) == {("A", WALK), ("B", WALK)}


def test_a_backlogged_chunk_still_respects_the_nodes_cap():
    """The backlog path resolves the cap itself; dropping it on the floor
    sends an oversized batch out, which fails `can_batch` and degrades to one
    eager forward per request."""
    sched = _scheduler(_Engine(max_bs=2))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1", "r2", "r3"])

    batch = sched.get_next_batch(_Manager([]))

    assert list(batch.node_objects) == ["r0", "r1"]
    assert list(sched.backlog[(NODE, WALK)].node_objects) == ["r2", "r3"]


def test_the_hold_backoff_expires_before_the_backlog_is_taken():
    """The expiry used to sit after the backlog's early return, so a
    backlogged chunk skipped it entirely."""
    sched = _scheduler(_Engine(max_bs=8))
    sched.hold_requests(["r0"])
    sched.held_until["r0"] = 0.0  # already elapsed
    sched.backlog[(NODE, WALK)] = _batch(["r0"])

    sched.get_next_batch(_Manager([]))

    assert "r0" not in sched.held_until


def test_the_oldest_backlog_entry_goes_first():
    """FIFO, as the deque was. `popitem` takes the newest, which starves an
    entry that keeps being overtaken."""
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    assert sched.get_next_batch(_Manager([])).node_name == "A"
    assert sched.get_next_batch(_Manager([])).node_name == "B"


def test_a_targeted_call_takes_only_its_own_backlog_entry():
    sched = _scheduler(_Engine(max_bs=8))
    sched.backlog[("A", WALK)] = _batch(["r0"], node="A")
    sched.backlog[("B", WALK)] = _batch(["r1"], node="B")

    batch = sched.get_next_batch(_Manager([]), target=("B", WALK))

    assert batch.node_name == "B"
    assert ("A", WALK) in sched.backlog


# ── round robin ─────────────────────────────────────────────────────────


def test_scheduling_a_batch_advances_the_round_robin_cursor():
    """The regression: the bookkeeping moved out of batch assembly, so every
    (node, walk) stayed at 0 and `_select_node_rr` kept picking the same one."""
    sched = _scheduler(_Engine(max_bs=2))

    sched.get_next_batch(_Manager([f"r{i}" for i in range(4)]))

    assert sched.node_and_walk_to_last_batch_num[(NODE, WALK)] > 0


def test_serving_from_the_backlog_also_advances_the_cursor():
    sched = _scheduler(_Engine(max_bs=1))
    sched.backlog[(NODE, WALK)] = _batch(["r0", "r1"])

    sched.get_next_batch(_Manager([]))
    first = sched.node_and_walk_to_last_batch_num[(NODE, WALK)]
    sched.get_next_batch(_Manager([]))

    assert sched.node_and_walk_to_last_batch_num[(NODE, WALK)] > first


def test_the_cursor_does_not_advance_when_nothing_is_scheduled():
    sched = _scheduler(_Engine(max_bs=4))

    sched.get_next_batch(_Manager(["r0"]), pre_existing_batch_size=4)

    assert (NODE, WALK) not in sched.node_and_walk_to_last_batch_num


@pytest.mark.parametrize("exclude", [None, set()])
def test_split_off_first_accepts_an_empty_exclusion(exclude):
    first, rest = _batch(["r0", "r1", "r2"]).split_off_first(2, exclude)

    assert list(first.node_objects) == ["r0", "r1"]
    assert list(rest.node_objects) == ["r2"]
