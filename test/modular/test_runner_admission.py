"""A node's requests are admitted by the runner, in one order across the resources they wait for.

A request takes every resource's capacity or none of it, and nothing is retrieved for it before
then. A waiting request is asked again only once a resource it waits for has grown, so a deep
queue costs the scheduler nothing on a pass that freed nothing. The scheduler used to ask every
waiting request on every pass, through the whole readiness sweep.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Collection

sys.path.insert(0, ".")

from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import Resource
from mstar.engine.resources.step import ADMIT_OK, ASKS, AdmitOutcome, AdmitRuntimeError

NODE, WALK = "llm", "prefill"


class _StubPool(Resource):
    """Hands out one unit of capacity per request unless ``needs`` says more, refuses the requests in
    ``refuses``, and counts the calls it gets."""

    def __init__(self, units: int, needs: dict[str, int] | None = None, refuses: Collection[str] = ()):
        self.free, self.held, self.calls, self.retrieved = units, set(), Counter(), []
        self.needs, self.refuses, self.published = needs or {}, refuses, []

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def ask(self, rid, node_name, graph_walk):
        self.calls["ask"] += 1
        if rid in self.refuses:
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(f"{rid} can never fit"))
        return None if rid in self.held else ASKS

    def reserve(self, rid, node_name, graph_walk, published):
        self.calls["reserve"] += 1
        self.published.append(published)
        if self.needs.get(rid, 1) > self.free:
            return self.free
        self.free -= self.needs.get(rid, 1)
        self.held.add(rid)
        return None

    def unreserve(self, rid):
        self.free += self.needs.get(rid, 1)
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


class _PassingPool(_StubPool):
    """Lets a request pass only while a unit is left for the first request."""

    def reserve_past(self, rid, node_name, graph_walk, published):
        if self.needs.get(rid, 1) + 1 > self.free:
            return self.free
        return self.reserve(rid, node_name, graph_walk, published)


class _LendingPool(_StubPool):
    """Lets a request pass whenever it fits, as the KV cache does past a first request it refused."""

    reserve_past = _StubPool.reserve


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


def test_a_queue_whose_first_request_leaves_tries_the_next_while_another_node_waits():
    encoder_pool, llm_pool = _StubPool(0), _StubPool(1, needs={"long": 2})
    runner = _runner({"encoder": ["enc"], NODE: ["kv"]}, enc=encoder_pool, kv=llm_pool)
    assert runner.admit_retrieve("a", "encoder", WALK).queued
    for rid in ("long", "short"):
        assert runner.admit_retrieve(rid, NODE, WALK).queued

    runner.remove_request("long")

    assert runner.admit_waiting(_nothing) == [(NODE, "short")], "the encoder's pool kept the LLM's queue waiting"


def test_an_earlier_request_asking_a_second_node_goes_ahead_of_a_later_one():
    # an encoder-decoder request asks the decoder's pool only once its encoder walk is done
    encoder_pool, decoder_pool = _StubPool(2), _StubPool(1, needs={"late": 2})
    runner = _runner({"encoder": ["enc"], "decoder": ["dec"]}, enc=encoder_pool, dec=decoder_pool)
    for rid in ("early", "late"):
        assert runner.admit_retrieve(rid, "encoder", WALK).ready
    assert runner.admit_retrieve("late", "decoder", WALK).queued

    assert runner.admit_retrieve("early", "decoder", WALK).ready, "an earlier request that fits waited on a later one"


def test_a_shared_pool_reserves_with_what_the_request_published_for_the_node_that_asked():
    pool = _StubPool(0)
    runner = _runner({NODE: ["kv"], "cfg": ["kv"]}, kv=pool)
    assert runner.admit_retrieve("a", NODE, WALK).queued
    assert runner.admit_retrieve("b", NODE, WALK, {"kv": "b for the LLM"}).queued
    pool.free = 1
    runner.remove_request("a")
    pool.published.clear()

    assert runner.admit_retrieve("b", "cfg", WALK, {"kv": "b for guidance"}).ready
    assert pool.published == ["b for the LLM"]


def test_a_request_one_pool_refuses_keeps_no_place_in_another_pools_queue():
    # Qwen3.5 asks its recurrent pool before its KV cache, which refuses a prompt that can never fit
    slots, kv = _StubPool(0), _StubPool(10, refuses={"huge"})
    runner = _runner(gdn=slots, kv=kv)
    assert not runner.admit_retrieve("huge", NODE, WALK).ok
    assert runner.admit_retrieve("next", NODE, WALK).queued

    slots.free = 1

    assert runner.admit_waiting(_nothing) == [(NODE, "next")], "a refused request kept the next one waiting"
    assert slots.held == {"next"}


def test_a_first_request_whose_pool_grew_passes_where_it_waits_behind_another():
    kv, slots = _PassingPool(0), _PassingPool(2, needs={"g": 3})
    runner = _runner({"one": ["kv", "slots"], "two": ["slots"]}, kv=kv, slots=slots)
    assert runner.admit_retrieve("g", "two", WALK).queued
    assert runner.admit_retrieve("h", "one", WALK).queued
    kv.free = 1

    started = runner.admit_waiting(_nothing)

    assert ("one", "h") in started, "h waits behind g, which can't start, though its own pool grew"


def test_a_request_turned_away_by_an_out_of_date_mark_can_leave_before_it_is_tried_again():
    kv = _PassingPool(0, needs={"head": 2})
    runner = _runner(kv=kv)
    assert runner.admit_retrieve("head", NODE, WALK).queued
    kv.free = 1  # the head's mark is out of date, and the head still doesn't fit
    assert runner.admit_retrieve("p", NODE, WALK).queued
    runner.remove_request("p")

    assert runner.admit_waiting(_nothing) == [], "a request that left was tried again"


def test_a_request_turned_away_by_an_out_of_date_mark_passes_once_that_head_is_measured_again():
    kv, slots = _PassingPool(1, needs={"x": 3}), _PassingPool(2, needs={"g": 3})
    runner = _runner({"one": ["kv", "slots"], "two": ["slots"]}, kv=kv, slots=slots)
    for rid, node in (("g", "two"), ("x", "one")):
        assert runner.admit_retrieve(rid, node, WALK).queued
    kv.free = 2  # x's mark is out of date, and x still doesn't fit
    assert runner.admit_retrieve("p", "one", WALK).queued

    assert ("one", "p") in runner.admit_waiting(_nothing), "p waits a pass more, though x was just refused again"


def test_requests_pass_one_refused_while_passing_where_it_is_first():
    # "head" asks only the KV pool, so "big" is first in the rows pool's queue
    kv, rows = _PassingPool(8, needs={"head": 10, "big": 8}), _PassingPool(20)
    runner = _runner({"solo": ["kv"], NODE: ["kv", "rows"]}, kv=kv, rows=rows)
    assert runner.admit_retrieve("head", "solo", WALK).queued
    assert runner.admit_retrieve("big", NODE, WALK).queued
    small = [f"s{n}" for n in range(6)]
    for rid in small:
        runner.admit_retrieve(rid, NODE, WALK)

    runner.admit_waiting(_nothing)

    assert kv.held == set(small), "six small requests waited behind one refused while it passed, with room for them"


def test_a_request_refused_while_passing_isnt_passed_once_it_is_first():
    """b was refused while it passed the head, with a unit left for the head. Once the head left, b fit as the first
    request, but c passed it on that refusal and took 5 of the 6 units b fit in."""
    kv = _PassingPool(6, needs={"head": 10, "b": 6, "c": 5})
    runner = _runner(kv=kv)
    for rid in ("head", "b"):
        assert runner.admit_retrieve(rid, NODE, WALK).queued
    runner.remove_request("head")

    assert runner.admit_retrieve("c", NODE, WALK).queued, "c passed b, which fits now that it is first"
    assert runner.admit_waiting(_nothing) == [(NODE, "b")]


def test_no_request_passes_a_first_request_where_it_fits_while_another_resource_holds_it_back():
    """The head waits only for a LoRA slot. A request that asks only the KV cache passed it there, took 5 of the 6
    pages the head fits in, and the head then waited until that request finished."""
    kv, lora = _LendingPool(6, needs={"head": 6, "p": 5}), _StubPool(0)
    runner = _runner({NODE: ["kv", "lora"], "solo": ["kv"]}, kv=kv, lora=lora)
    assert runner.admit_retrieve("head", NODE, WALK).queued

    assert runner.admit_retrieve("p", "solo", WALK).queued, "p took pages the head fits in"
    lora.free = 1
    assert runner.admit_waiting(_nothing) == [(NODE, "head")]


def test_no_request_passes_where_the_oldest_request_waits_to_return():
    """v was dropped and waits to return, and h, refused, is first in the queue. While o, older than v, waited, c
    passed h. Once o left, v was the oldest request, and d passed h too, into the units v waits for."""
    kv, lora = _LendingPool(8, needs={"v": 4, "w": 3, "h": 7}), _StubPool(0)
    runner = _runner({NODE: ["kv"], "other": ["lora"]}, kv=kv, lora=lora)
    for arrival, rid in enumerate(("o", "v", "w", "h", "c", "d")):
        runner.ingest_request(rid, None, arrival)
    assert runner.admit_retrieve("o", "other", WALK).queued
    for rid in ("v", "w"):
        assert runner.admit_retrieve(rid, NODE, WALK).ready
    # v returns with its history and a unit more
    kv.unreserve("v")
    kv.needs["v"] = 6
    runner.readmit("v", NODE)
    assert runner.admit_retrieve("h", NODE, WALK).queued
    assert runner.admit_retrieve("c", NODE, WALK).ready, "c waited behind h, though o is older than v"
    runner.remove_request("o")

    assert runner.admit_retrieve("d", NODE, WALK).queued, "d passed h into the units v, the oldest request, waits for"
    runner.remove_request("w")
    assert (NODE, "v") in runner.admit_waiting(_nothing)
