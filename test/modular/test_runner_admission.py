"""A node's requests are admitted by the runner, in one order across the resources they wait for.

A request takes every resource's capacity or none of it, and nothing is retrieved for it before
then. A waiting request is asked again only once a resource it waits for has grown, so a deep
queue costs the scheduler nothing on a pass that freed nothing. The scheduler used to ask every
waiting request on every pass, through the whole readiness sweep.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")

from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import Resource
from mstar.engine.resources.step import ADMIT_OK, ASKS

NODE, WALK = "llm", "prefill"


class _StubPool(Resource):
    """Hands out one unit of capacity per request, and counts the calls it gets."""

    def __init__(self, units: int):
        self.free, self.held, self.calls, self.retrieved = units, set(), Counter(), []

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def ask(self, rid, node_name, graph_walk):
        self.calls["ask"] += 1
        return None if rid in self.held else ASKS

    def reserve(self, rid, node_name, graph_walk, published):
        self.calls["reserve"] += 1
        if not self.free:
            return self.free
        self.free -= 1
        self.held.add(rid)
        return None

    def unreserve(self, rid):
        self.free += 1
        self.held.discard(rid)

    def grew_since(self, mark):
        self.calls["grew_since"] += 1
        return self.free > mark

    def admit_retrieve(self, rid, node_name, graph_walk, published):
        self.retrieved.append(rid)
        return ADMIT_OK

    def remove_request(self, rid):
        if rid in self.held:
            self.unreserve(rid)


def _runner(nodes: dict[str, list[str]] | None = None, **pools: _StubPool) -> StepRunner:
    return StepRunner(pools, node_resources=nodes or {NODE: list(pools)})


def _nothing(rid, node_name):
    return None


def test_a_waiting_request_is_not_asked_again_until_its_pool_grows():
    pool = _StubPool(1)
    runner = _runner(kv=pool)
    assert runner.admit_retrieve("a", NODE, WALK).ready
    for rid in ("b", "c"):
        assert runner.admit_retrieve(rid, NODE, WALK).queued
    pool.calls.clear()

    for _ in range(100):
        assert runner.admit_waiting(_nothing) == []

    assert pool.calls == {"grew_since": 100}, "a pass that freed nothing asked the waiting requests again"
    runner.remove_request("a")
    assert runner.admit_waiting(_nothing) == [(NODE, "b")], "room came back and the first waiting request stayed out"


def test_a_request_takes_every_pool_on_its_node_or_none():
    # Whisper's decoder asks a self-attention pool and a cross-attention pool for each request
    own, cross = _StubPool(2), _StubPool(1)
    runner = _runner(own=own, cross=cross)
    assert runner.admit_retrieve("r", NODE, WALK).ready

    assert runner.admit_retrieve("c", NODE, WALK).queued
    assert own.held == {"r"}, "a request the cross pool can't take kept the self pool's capacity"


def test_pools_admit_their_waiting_requests_in_the_order_they_first_asked():
    own, cross = _StubPool(1), _StubPool(1)
    runner = _runner(own=own, cross=cross)
    assert runner.admit_retrieve("r", NODE, WALK).ready
    for rid in ("c", "d"):
        assert runner.admit_retrieve(rid, NODE, WALK).queued

    runner.remove_request("r")

    assert runner.admit_waiting(_nothing) == [(NODE, "c")]
    assert own.held == cross.held == {"c"}, "the two pools admitted different requests"


def test_nothing_is_retrieved_for_a_waiting_request():
    pool = _StubPool(0)
    runner = _runner(kv=pool)

    assert runner.admit_retrieve("a", NODE, WALK).queued
    assert pool.retrieved == [], "a retrieve started for a request still waiting for capacity"


def test_a_request_waiting_on_one_node_runs_on_another():
    pool, encoder = _StubPool(0), _StubPool(0)
    runner = _runner({NODE: ["kv"], "encoder": []}, kv=pool, enc=encoder)

    assert runner.admit_retrieve("a", NODE, WALK).queued
    assert runner.admit_retrieve("a", "encoder", WALK).ready, "waiting on the LLM's pool held back the encoder"


def test_every_node_waiting_on_a_shared_pool_resumes_when_it_admits():
    # Bagel's guidance nodes share the leader's pool, and each one's check waits on the same request
    pool = _StubPool(0)
    runner = _runner({NODE: ["kv"], "cfg": ["kv"]}, kv=pool)
    for node in (NODE, "cfg"):
        assert runner.admit_retrieve("a", node, WALK).queued

    pool.free = 1

    assert sorted(runner.admit_waiting(_nothing)) == [("cfg", "a"), (NODE, "a")], "a node stayed parked"
