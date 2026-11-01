"""The runner snapshots a step's publication per resource for the whole batch,
and the result matches the per-request form."""
from typing import Any

from mstar.engine.resources.base import Resource
from mstar.engine.resources.runner import StepRunner


class _Pub:
    """A publisher standing in for a resource: snapshots are per-rid tuples."""

    def __init__(self, name, skip=()):
        self.name = name
        self.skip = set(skip)
        self.batch_calls = 0

    def publish_snapshot_for_step(self, request_id, node_name, graph_walk):
        if request_id in self.skip:
            return None
        return (self.name, request_id, node_name, graph_walk)

    def publish_snapshot_batch(self, request_ids, node_name, graph_walk):
        self.batch_calls += 1
        return Resource.publish_snapshot_batch(self, request_ids, node_name, graph_walk)


def _runner(pubs: dict[str, Any]) -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner._resources = pubs
    runner._publish_order = list(pubs)
    runner._node_publish_order = {}
    runner._sweep = lambda node_order, order, node_name: order
    return runner


def test_batch_snapshot_matches_per_request_form():
    pubs = {"kv": _Pub("kv", skip={3}), "pos": _Pub("pos", skip={2, 3})}
    runner = _runner(pubs)
    out = runner.publish_snapshot([1, 2, 3], node_name="LLM", graph_walk="decode")
    assert out == {
        1: {"kv": ("kv", 1, "LLM", "decode"), "pos": ("pos", 1, "LLM", "decode")},
        2: {"kv": ("kv", 2, "LLM", "decode")},
    }
    assert pubs["kv"].batch_calls == 1 and pubs["pos"].batch_calls == 1
    assert runner.publish_snapshot([], node_name="LLM", graph_walk="decode") == {}
