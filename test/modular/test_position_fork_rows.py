"""Positions fork only the rows KV forked.

A step's forks can cover some of its rows (``KVStep.fork_rids``): a chunked
guided prompt forks ``cfg_text`` off ``main`` on its first chunk and again on
its last. KV scoped the forks; positions forked every row, so a guided row on
a later chunk, sharing a step with another prompt's first chunk, had its
``cfg_text`` counter jump to ``main``'s while its ``cfg_text`` KV stayed put
(PR #365 review: images shifted by 5-14/255).
"""

from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import StepContext

FORK = (("main", "cfg_text"),)


def _rope(counters) -> RopeManager:
    rope = RopeManager.__new__(RopeManager)
    rope._kv_cache_name = "kv"
    rope._counters = counters
    return rope


def _ctx(forks: KVPlanOutputs) -> StepContext:
    ctx = StepContext(request_ids=("first", "later"), graph_walk="prefill_text", slot=0, capture=False)
    ctx.plan_results = {"kv": forks}
    return ctx


def _counters():
    return {
        "first": {"main": 50},
        "later": {"main": 90, "cfg_text": 40},  # forked on its own first chunk
    }


def test_a_post_fork_moves_only_the_rows_kv_forked():
    rope = _rope(_counters())

    rope.commit(
        PositionStep(segments=(), advance=()),
        _ctx(KVPlanOutputs(post_forks=FORK, fork_rids=frozenset({"first"}))),
    )

    assert rope._counters["first"]["cfg_text"] == 50
    assert rope._counters["later"]["cfg_text"] == 40, "re-forked mid-prompt"


def test_a_step_without_fork_rows_forks_every_row():
    rope = _rope(_counters())

    rope.commit(PositionStep(segments=(), advance=()), _ctx(KVPlanOutputs(post_forks=FORK)))

    assert {rid: c["cfg_text"] for rid, c in rope._counters.items()} == {"first": 50, "later": 90}


def test_a_pre_fork_takes_the_same_rows():
    ctx = _ctx(KVPlanOutputs(pre_forks=FORK, fork_rids=frozenset({"later"})))

    assert RopeManager._forking_rids(ctx.plan_results["kv"], ctx) == ["later"]
