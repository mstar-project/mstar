"""PRIORITY scheduling must return the highest-priority node, not the last
dict key.

``_select_node_priority`` used to compute ``best_node_name`` and then return
the loop variable ``node_name`` — the last key in ``node_name_to_requests``.
When a high-priority node and a low-priority node were both ready,
``get_next_batch`` paired the *low-priority* node name with the *winning*
walk, found no members, and returned None. The worker idled forever despite
ready work.

These tests pin the failure mode on the resource-pool scheduler: insertion
order is LLM then vision so the last-iterated key is the lower-priority
node. A correct implementation still selects LLM.
"""

from __future__ import annotations

from types import SimpleNamespace

from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.worker.micro_scheduler import (
    MicroScheduler,
    ReadyNodeEntry,
    SchedulingType,
)

LLM_DECODE = ("LLM", "decode")
VISION_PREFILL = ("vision_encoder", "prefill_vision")
# Lower value = higher priority. LLM decode is the latency-sensitive walk.
NODE_WALK_PRIORITY = {
    LLM_DECODE: 0,
    VISION_PREFILL: 2,
}


class _Engine:
    def get_max_batch_size(self, node_name, graph_walk):
        del node_name, graph_walk
        return None

    def check_ready(self, node_name, rid, fwd_info):
        del node_name, rid, fwd_info
        return FULL_ADMIT_OK


class _Queue:
    def __init__(self, ready: dict[str, set[str]]):
        self.ready = {rid: set(nodes) for rid, nodes in ready.items()}

    def get_ready_node_names(self):
        return {rid: set(nodes) for rid, nodes in self.ready.items()}

    def pop_ready_nodes(self, request_id, node_names):
        popped = []
        for name in node_names:
            if name in self.ready.get(request_id, ()):
                self.ready[request_id].discard(name)
                popped.append(SimpleNamespace(name=name))
        return popped


class _GraphsManager:
    def __init__(self, ready: dict[str, set[str]], walks: dict[str, str]):
        self.queues = {"wg0": _Queue(ready)}
        self.per_request_info = dict.fromkeys(ready, object())
        self._walks = walks

    def get_partition_for_node(self, node_name):
        return node_name

    def get_graph_walk(self, rid, partition):
        del rid
        return self._walks[partition]

    def get_fwd_info(self, rid, partition):
        del rid, partition
        return object()


def _scheduler() -> MicroScheduler:
    return MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda name: _Engine()),
        sched_type=SchedulingType.PRIORITY,
        parallel_leader_nodes={"LLM", "vision_encoder"},
        node_walk_priority=NODE_WALK_PRIORITY,
    )


def test_select_node_priority_returns_ranked_node_not_last_iterated_key():
    scheduler = _scheduler()
    ready = {
        "LLM": [ReadyNodeEntry("r0", "wg0", "decode")],
        "vision_encoder": [ReadyNodeEntry("r1", "wg0", "prefill_vision")],
    }
    node, walk = scheduler._select_node_priority(ready)
    assert (node, walk) == LLM_DECODE
    assert list(ready)[-1] == "vision_encoder"


def test_get_next_batch_under_priority_schedules_ranked_node_instead_of_idling():
    """Both nodes ready on different walks: PRIORITY must emit the LLM batch.

    The buggy return of the last-iterated node name pairs vision with the
    decode walk, finds no members, and hands the worker None.
    """
    graphs = _GraphsManager(
        ready={"r0": {"LLM"}, "r1": {"vision_encoder"}},
        walks={"LLM": "decode", "vision_encoder": "prefill_vision"},
    )
    batch = _scheduler().get_next_batch(graphs)
    assert batch is not None, (
        "PRIORITY must not idle when a higher-ranked node is ready: returning "
        "the last iterated node name yields an empty (vision, decode) pair"
    )
    assert (batch.node_name, batch.graph_walk) == LLM_DECODE
    assert list(batch.node_objects) == ["r0"]
