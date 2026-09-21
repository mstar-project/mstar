"""A follower tells rank 0 what its own prefix index matched.

The ranks index independently: a generated page is keyed on each rank's own
postprocess path, whose timing against the next commit is that rank's, so at
any instant one rank can hold a page another has not keyed yet. Rank 0 cannot
see that gap from where it stands, and this is the only message in the worker
that travels upward — every other one fans out from the leader.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Import mstar from THIS worktree, not the venv's editable install, as the
# other TP tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.distributed.base import ShardingGroup  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    TPPrefixMatch,
    WorkerMessageType,
)
from mstar.worker.worker import Worker  # noqa: E402

NODE = "LLM"
WALK = "prefill"


def _group(tp_rank: int, tp_size: int) -> ShardingGroup:
    group = ShardingGroup(nodes={NODE}, tp_size=tp_size, _tp_rank=tp_rank)
    group.register_workers([f"worker_{rank}" for rank in range(tp_size)])
    return group


def _worker(
    tp_rank: int, matched: list[dict[str, int]], tp_size: int = 2,
) -> Worker:
    """A rank of a ``tp_size`` group, with its engine's answer scripted."""
    w = Worker.__new__(Worker)
    w.worker_id = f"worker_{tp_rank}"
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda entity, msg: w.sent.append((entity, msg)),
    )
    w.parallel_nodes = {NODE}
    w.parallel_leader_nodes = {NODE} if tp_rank == 0 else set()
    w._tp_prefix_replies = {}
    w._tp_prefix_waiting = {}
    group = _group(tp_rank, tp_size)
    w.worker_graphs_manager = SimpleNamespace(
        per_request_info={"r1": SimpleNamespace(
            sharding_config=SimpleNamespace(
                get_sharding_group=lambda node, walk: group,
            ),
        )},
        get_partition_for_node=lambda node: "default",
        get_graph_walk=lambda rid, partition: WALK,
    )
    w.engine_manager = SimpleNamespace(
        get_engine=lambda node: SimpleNamespace(
            matched_prefixes=lambda node_name, request_id: matched,
        ),
    )
    return w


# ── the reply ───────────────────────────────────────────────────────────


def test_a_follower_answers_rank_zero_with_what_it_matched():
    w = _worker(1, [{"main": 96}])

    Worker._report_prefix_match(w, "r1")

    assert len(w.sent) == 1, f"one keyed resource sent {len(w.sent)} replies"
    entity, message = w.sent[0]
    assert entity == "worker_0", f"the reply went to {entity}, not to rank 0"
    assert message.message_type is WorkerMessageType.TP_PREFIX_MATCH
    assert message.body.matched == {"main": 96}, (
        "the reply does not carry the length this rank matched"
    )


def test_a_reply_names_the_rank_that_sent_it():
    w = _worker(1, [{"main": 96}])

    Worker._report_prefix_match(w, "r1")

    assert w.sent[0][1].body.tp_rank == 1, (
        "a reply rank 0 cannot attribute to a rank counts twice when it is "
        "sent twice, and the minimum is then taken over the wrong set"
    )


def test_a_node_with_two_caches_answers_for_each():
    w = _worker(1, [{"main": 96}, {"cross": 32}])

    Worker._report_prefix_match(w, "r1")

    assert [m.body.matched for _, m in w.sent] == [{"main": 96}, {"cross": 32}], (
        "two caches on one node hold two prefixes, and each has its own "
        "length to agree on"
    )


def test_a_node_that_keys_nothing_answers_nothing():
    w = _worker(1, [])

    Worker._report_prefix_match(w, "r1")

    assert w.sent == [], "an undeclared node still sent a reply"


def test_the_leader_does_not_answer_itself():
    w = _worker(0, [{"main": 96}])

    Worker._report_prefix_match(w, "r1")

    assert w.sent == [], "rank 0 sent its own match up to itself"


def test_a_request_this_rank_never_took_answers_nothing():
    w = _worker(1, [{"main": 96}])

    Worker._report_prefix_match(w, "gone")

    assert w.sent == [], (
        "a request removed between its NewRequest and this reply was still "
        "answered, from a sharding config that is no longer there"
    )


# ── what the leader keeps ───────────────────────────────────────────────


def test_the_leader_keeps_one_answer_a_rank():
    w = _worker(0, [{"main": 96}])

    Worker._record_prefix_match(w, TPPrefixMatch("r1", NODE, 1, {"main": 64}))
    Worker._record_prefix_match(w, TPPrefixMatch("r1", NODE, 2, {"main": 48}))

    assert w._tp_prefix_replies == {"r1": {1: {"main": 64}, 2: {"main": 48}}}, (
        "the ranks' answers were merged, so the group cannot tell whether "
        "every rank has answered"
    )


def test_a_rank_that_answers_twice_is_counted_once():
    w = _worker(0, [{"main": 96}])

    Worker._record_prefix_match(w, TPPrefixMatch("r1", NODE, 1, {"main": 64}))
    Worker._record_prefix_match(w, TPPrefixMatch("r1", NODE, 1, {"main": 64}))

    assert w._tp_prefix_replies["r1"] == {1: {"main": 64}}, (
        "a resent reply was counted as a second rank"
    )


# ── the wait ────────────────────────────────────────────────────────────


def test_the_leader_holds_the_first_step_until_every_rank_answers():
    w = _worker(0, [{"main": 96}])

    Worker._await_prefix_match(w, "r1")

    assert ("r1", NODE) in w._tp_prefix_waiting, (
        "the first step could run on the length this rank alone matched"
    )


def test_the_last_reply_lets_the_step_go():
    w = _worker(0, [{"main": 96}])
    Worker._await_prefix_match(w, "r1")

    Worker._record_prefix_match(w, TPPrefixMatch("r1", NODE, 1, {"main": 64}))

    assert w._tp_prefix_waiting == {}, (
        "the whole group had answered and the step was still held"
    )


def test_one_rank_short_keeps_the_step_held():
    w = _worker(0, [{"main": 96}], tp_size=3)
    Worker._await_prefix_match(w, "r1")

    Worker._record_prefix_match(w, TPPrefixMatch("r1", NODE, 1, {"main": 64}))

    assert ("r1", NODE) in w._tp_prefix_waiting, (
        "two of three ranks settled a length the third never agreed to"
    )


def test_a_group_of_one_waits_for_nobody():
    w = _worker(0, [{"main": 96}], tp_size=1)

    Worker._await_prefix_match(w, "r1")

    assert w._tp_prefix_waiting == {}, (
        "a single-rank deployment held its first step for a reply nobody sends"
    )


def test_a_node_that_keys_nothing_waits_for_nobody():
    w = _worker(0, [])

    Worker._await_prefix_match(w, "r1")

    assert w._tp_prefix_waiting == {}, (
        "an undeclared node held its first step over a length that has no "
        "meaning for it"
    )
