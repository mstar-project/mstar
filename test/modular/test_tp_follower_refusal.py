"""A TP follower refusing a step rank 0 cannot have refused fails loudly.

Each rank admits from its own pools, and rank 0 schedules within
``get_max_batch_size``. A follower that refuses a batch past its own cap has
capacity rank 0 does not: rank 0 admitted the step and is running its
collectives, and a follower that re-queued it would leave rank 0 in them
alone. Refusals within the cap are the ones both ranks share, and are retried.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

from mstar.engine.engine import Engine
from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    FullAdmitOutcome,
    RequestOffloading,
)


def _engine(cap, follower=True) -> Engine:
    engine = object.__new__(Engine)
    engine._tp_follower_nodes = {"LLM"} if follower else set()
    engine.get_max_batch_size = lambda node, walk: cap
    return engine


def _batch(rows: int):
    return SimpleNamespace(
        node_name="LLM", step_context=SimpleNamespace(graph_walk="prefill"),
        request_ids=[f"r{i}" for i in range(rows)],
    )


def _refused(reason=None) -> FullAdmitOutcome:
    reason = reason or AllocationFailed(
        message="recurrent state pool is full", pages_short=1, label="main", request_id="r2",
    )
    return FullAdmitOutcome(AdmitOutcome(ok=False, reason=reason), "kda_state")


def test_a_follower_refusing_past_its_cap_raises():
    with pytest.raises(RuntimeError, match=r"3 requests on LLM/prefill.*cap of 2.*kda_state.*diverged"):
        Engine._check_follower_refusal(_engine(cap=2), _batch(3), _refused())


@pytest.mark.parametrize("engine, rows, outcome", [
    (_engine(cap=3), 3, _refused()),  # within the cap: rank 0 refused it too
    (_engine(cap=2, follower=False), 3, _refused()),  # rank 0 retries its own refusals
    (_engine(cap=None), 3, _refused()),  # no cap to have scheduled within
    (_engine(cap=2), 3, _refused(RequestOffloading(
        message="being offloaded", label="main", request_id="r0"))),  # not capacity
], ids=["within-cap", "leader", "uncapped", "offloading"])
def test_refusals_both_ranks_share_are_left_to_retry(engine, rows, outcome):
    Engine._check_follower_refusal(engine, _batch(rows), outcome)
