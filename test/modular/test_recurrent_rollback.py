"""A refused admit gives back the slots it took, and only those.

The runner unwinds every resource a refused step reached, the refusing one
included. The pool has to find what that step's live reservation took, even
when the reservation was made on the pre-plan pass, and must leave alone a
slot the request already held. Its free list comes back in its old order, so
TP ranks that refuse the same step keep handing out the same slots.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import torch

from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    Resource,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext

POOL, AFTER = "state", "after"


def build_pool(max_slots: int) -> RecurrentStatePool:
    spec = RecurrentStateSpec(
        POOL, {"llm"},
        RecurrentStateConfig(
            num_layers=2,
            blocks={"state": RecurrentBlockConfig(shape=(2, 4, 4), dtype=torch.float32)},
            max_slots=max_slots,
        ),
    )
    return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))


class _Refuser(Resource):
    """Planned after the pool; refuses whenever told to."""

    def __init__(self):
        self.refuse = False
        self.preplans = False

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    @property
    def supports_preplan(self):
        return self.preplans

    def depends_on(self) -> set[str]:
        return {POOL}

    def admit(self, step, ctx):
        if self.refuse:
            return AdmitOutcome(ok=False, reason=AllocationFailed("full", pages_short=1, label="main", request_id="b"))
        return AdmitOutcome(ok=True)


def runner(pool: RecurrentStatePool, after: _Refuser | None = None) -> StepRunner:
    resources = {POOL: pool}
    if after is not None:
        resources[AFTER] = after
    return StepRunner(resources, node_resources={"llm": list(resources)})


def step(rids: list[str], span: int = 3, keys=(POOL,)) -> SubmoduleStep:
    segments = [Segment(rid, "main", span) for rid in rids]
    st = SubmoduleStep(
        segments=segments,
        steps={key: RecurrentStep(segments=tuple(segments)) for key in keys},
    )
    st.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False))
    return st


def slot_of(pool: RecurrentStatePool, rid: str) -> int | None:
    slot = pool._slots.get(rid, {}).get("main")
    return None if slot is None else slot.index


def test_refusal_frees_the_slots_the_pool_took_before_it():
    """Three free slots, four new requests: the pool takes three, refuses the
    fourth, and hands the three back in the order it had them."""
    pool = build_pool(max_slots=4)  # 3 usable
    run = runner(pool)
    for rid in "abcd":
        run.ingest_request(rid, {})
    free_before = list(pool._free)

    outcome = run.admit(step(list("abcd")))

    assert not outcome.ok and outcome.failed_resource == POOL
    assert pool._free == free_before
    assert all(slot_of(pool, rid) is None for rid in "abcd")
    # and the same step, one request shorter, gets the same slots it would have
    assert run.admit(step(list("abc"))).ok
    assert [slot_of(pool, rid) for rid in "abc"] == [1, 2, 3]


def test_refusal_downstream_frees_the_pools_slots():
    pool, after = build_pool(max_slots=4), _Refuser()
    run = runner(pool, after)
    for rid in "ab":
        run.ingest_request(rid, {})
    after.refuse = True

    outcome = run.admit(step(list("ab"), keys=(POOL, AFTER)))

    assert not outcome.ok and outcome.failed_resource == AFTER
    assert pool.num_free_slots == 3


def test_a_slot_already_held_is_kept():
    """A decode row holds its slot from an earlier step: a refusal of a step
    that also brings a newcomer frees the newcomer's slot, not the old one."""
    pool, after = build_pool(max_slots=4), _Refuser()
    run = runner(pool, after)
    for rid in "ab":
        run.ingest_request(rid, {})
    first = step(["a"], keys=(POOL, AFTER))
    assert run.admit(first).ok
    run.plan(first)
    run.commit(first)
    held = slot_of(pool, "a")

    after.refuse = True
    assert not run.admit(step(["a", "b"], keys=(POOL, AFTER))).ok

    assert slot_of(pool, "a") == held and pool._slots["a"]["main"].has_state
    assert slot_of(pool, "b") is None
    assert pool.num_free_slots == 2


def test_committed_steps_leave_nothing_to_unwind():
    pool = build_pool(max_slots=4)
    run = runner(pool)
    run.ingest_request("a", {})
    st = step(["a"])
    assert run.admit(st).ok
    run.plan(st)
    run.commit(st)
    pool.rollback_admit(st.get(POOL), st.ctx)
    assert slot_of(pool, "a") is not None and pool.num_free_slots == 2


def test_refusal_after_a_preplan_frees_the_preplans_slots():
    """The pre-plan pass reserved the slots; the full admit that follows
    no-ops over them, and a refusal after it must still give them back."""
    pool, after = build_pool(max_slots=4), _Refuser()
    after.preplans = True
    run = runner(pool, after)
    for rid in "ab":
        run.ingest_request(rid, {})
    st = step(list("ab"), keys=(POOL, AFTER))
    st.ctx.is_preplan = True
    assert run.pre_admit(st).ok
    run.pre_plan(st)
    st.ctx.is_preplan = False
    assert pool.num_free_slots == 1

    after.refuse = True
    assert not run.admit(st).ok

    assert pool.num_free_slots == 3
    assert slot_of(pool, "a") is None and slot_of(pool, "b") is None
    pool.clear_preplan()


def test_a_request_removed_before_the_unwind_is_not_freed_twice():
    pool = build_pool(max_slots=4)
    pool.ingest_request("a")
    st = step(["a"])
    assert pool.admit(st.get(POOL), st.ctx).ok
    pool.remove_request("a")
    pool.rollback_admit(st.get(POOL), st.ctx)
    assert sorted(pool._free) == [1, 2, 3]


def test_a_discarded_preplan_gives_back_its_slots():
    """A staged step that never runs (its batch yielded, or another step reached
    the GPU first) leaves the pool as it found it."""
    pool = build_pool(max_slots=4)
    run = runner(pool)
    for rid in "ab":
        run.ingest_request(rid, {})
    free_before = list(pool._free)
    st = step(list("ab"))
    st.ctx.is_preplan = True
    assert run.pre_admit(st).ok
    run.pre_plan(st)

    run.clear_preplan()

    assert pool._free == free_before
    assert slot_of(pool, "a") is None and slot_of(pool, "b") is None


def test_a_refusal_after_a_discarded_preplan_frees_its_slots():
    pool, after = build_pool(max_slots=4), _Refuser()
    run = runner(pool, after)
    for rid in "ab":
        run.ingest_request(rid, {})
    pre = step(list("ab"), keys=(POOL, AFTER))
    pre.ctx.is_preplan = True
    assert run.pre_admit(pre).ok
    run.pre_plan(pre)
    run.clear_preplan()

    after.refuse = True
    assert not run.admit(step(list("ab"), keys=(POOL, AFTER))).ok

    assert pool.num_free_slots == 3


def test_clearing_a_stage_leaves_a_live_steps_slots_alone():
    """A live step that raised after its forward wrote its slots is the runner's to unwind
    (its failed rids are removed, which zeroes them); clearing a stage must not free them."""
    pool = build_pool(max_slots=4)
    run = runner(pool)
    for rid in "ab":
        run.ingest_request(rid, {})
    live = step(list("ab"))
    assert run.admit(live).ok
    run.plan(live)
    for rid in "ab":
        pool.block("state", 0)[slot_of(pool, rid)].fill_(7.0)

    run.clear_preplan()
    assert slot_of(pool, "a") is not None and slot_of(pool, "b") is not None
    for rid in "ab":
        run.remove_request(rid)
    run.ingest_request("c", {})
    assert run.admit(step(["c"])).ok
    assert not pool.block("state", 0)[slot_of(pool, "c")].any()


def test_a_rolled_back_slot_goes_back_zeroed():
    # a step can raise after its forward wrote the slots it took
    pool = build_pool(max_slots=4)
    pool.ingest_request("a")
    st = step(["a"])
    assert pool.admit(st.get(POOL), st.ctx).ok
    index = slot_of(pool, "a")
    pool.block("state", 0)[index].fill_(7.0)
    pool.rollback_admit(st.get(POOL), st.ctx)
    assert index in pool._free and not pool.block("state", 0)[index].any()
