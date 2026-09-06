"""PRIORITY scheduling must return the highest-priority node, not the last
dict key.

``_select_node_priority`` used to compute ``best_node_name`` and then return
the loop variable ``node_name`` — the last key in ``node_name_to_requests``.
When a KV node and a stateless node were both ready, ``get_next_batch``
paired the *stateless* node name with the *KV* walk, found no members, and
returned None. The worker idled forever despite ready work.

These tests pin the failure mode: insertion order is LLM then vision so the
last-iterated key is the lower-priority node. A correct implementation still
selects LLM.
"""

from __future__ import annotations

from types import SimpleNamespace

from mstar.engine.base import EngineType
from mstar.worker.micro_scheduler import (
    MicroScheduler,
    ReadyNodeEntry,
    SchedulingType,
)


class _Engine:
    def __init__(self, engine_type: EngineType):
        self._engine_type = engine_type

    def engine_type(self):
        return self._engine_type

    def check_ready(self, node_name, request_id, request_info):
        del node_name, request_id, request_info
        return True


class _EngineManager:
    def __init__(self, node_to_engine: dict[str, _Engine]):
        self.node_to_engine = node_to_engine

    def get_engine(self, node_name: str) -> _Engine:
        return self.node_to_engine[node_name]


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


def _manager() -> _EngineManager:
    # Insertion order matters: LLM first, vision last. The bug returns the
    # last key (vision) even though KV_CACHE has higher priority.
    return _EngineManager(
        {
            "LLM": _Engine(EngineType.KV_CACHE),
            "vision_encoder": _Engine(EngineType.STATELESS),
        }
    )


def test_select_node_priority_returns_kv_node_not_last_iterated_key():
    scheduler = MicroScheduler(_manager(), sched_type=SchedulingType.PRIORITY)
    ready = {
        "LLM": [ReadyNodeEntry("r0", "wg0", "decode")],
        "vision_encoder": [ReadyNodeEntry("r1", "wg0", "prefill_vision")],
    }
    node, walk = scheduler._select_node_priority(ready)
    assert (node, walk) == ("LLM", "decode")
    assert list(ready)[-1] == "vision_encoder"


def test_get_next_batch_under_priority_schedules_kv_instead_of_idling():
    """Both nodes ready on different walks: PRIORITY must emit the KV batch.

    The buggy return of the last-iterated node name pairs vision with the
    decode walk, finds no members, and hands the worker None.
    """
    scheduler = MicroScheduler(
        _manager(),
        sched_type=SchedulingType.PRIORITY,
        parallel_leader_nodes={"LLM", "vision_encoder"},
    )
    graphs = _GraphsManager(
        ready={"r0": {"LLM"}, "r1": {"vision_encoder"}},
        walks={"LLM": "decode", "vision_encoder": "prefill_vision"},
    )
    batch = scheduler.get_next_batch(graphs)
    assert batch is not None, (
        "PRIORITY must not idle when a KV node is ready: returning the last "
        "iterated node name yields an empty (vision, decode) pair"
    )
    assert batch.node_name == "LLM"
    assert batch.graph_walk == "decode"
    assert list(batch.node_objects) == ["r0"]
