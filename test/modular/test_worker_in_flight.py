"""The worker's in-flight flag reaches every worker graph a batch spans."""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker


def test_a_mixed_batch_flags_each_worker_graph():
    """Prefill and decode rows of a combined walk live in different worker
    graphs; clearing only the first one's left the rest flagged for good,
    and their node never went ready again."""
    calls = []
    worker = Worker.__new__(Worker)
    worker._graph_runtime = SimpleNamespace(
        set_in_flight=lambda node, rids, wg_ids, value: calls.append(
            (dict(zip(rids, wg_ids, strict=True)), value)),
    )
    batch = ScheduledBatch(
        node_name="LLM", graph_walk="mixed",
        request_to_worker_graph={0: 7, 1: 9, 2: 7},
    )

    worker._clear_in_flight_flag(batch)

    assert calls == [({0: 7, 1: 9, 2: 7}, False)]  # one call, each row its own graph


def test_a_mixed_speculative_batch_is_flagged_in_each_worker_graph():
    """Flagging every row under the first row's worker graph marked another
    walk's node of the rest; the per-graph clear never reached it, so that
    node never went ready again."""
    flagged = []
    worker = Worker.__new__(Worker)
    worker._graph_runtime = SimpleNamespace(
        commit_speculation=lambda spec_id, success, dropped, node=None, wg_ids=None,
        scheduled_rids=(): flagged.append(dict(zip(scheduled_rids, wg_ids, strict=True)))
        if node else None,
    )
    batch = ScheduledBatch(
        node_name="LLM", graph_walk="mixed",
        request_to_worker_graph={0: 5, 1: 0, 2: 5},
    )
    speculation = SimpleNamespace(
        scheduled_batch=batch, spec_id=1, consumed_streaming_edges={},
    )

    worker._settle_speculation(speculation, success=True)

    assert flagged == [{0: 5, 1: 0, 2: 5}]  # one call, each row its own graph
