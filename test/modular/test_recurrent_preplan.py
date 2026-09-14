"""Planning the recurrent pool and the delta-net backend a step ahead.

These two were the only resources of Qwen3.5's six that planned inline, so
their work — a host loop, two pinned allocations and two H2D copies per label —
landed on the critical path between the GPU thread taking a batch and reaching
the forward launch.

What has to hold for staging to be safe: a promoted plan must be the plan the
step would have made, and an *abandoned* one must leave no trace. The second is
the sharp edge, because ``plan`` applies pre-forks, which mutate live state.

CPU-only: nothing calls a kernel, so no device is needed.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext

POOL, BACKEND = "gdn_state", "linear_attn"
NUM_LAYERS, NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 2, 2, 4, 16
# conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
CONV_DIM = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM


def pool_spec() -> RecurrentStateSpec:
    return RecurrentStateSpec(
        POOL, {"llm"},
        RecurrentStateConfig(
            num_layers=NUM_LAYERS,
            blocks={
                "state": RecurrentBlockConfig(
                    shape=(NUM_V_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32,
                ),
                "conv": RecurrentBlockConfig(
                    shape=(CONV_DIM, 3), dtype=torch.float32,
                ),
            },
            max_slots=16,
        ),
    )


def build_pool() -> RecurrentStatePool:
    return RecurrentStatePool.build(
        pool_spec(), EngineResourceInfo(device=torch.device("cpu")),
    )


def build_backend() -> GDNManager:
    return GDNManager.build(
        LinearAttnSpec(
            BACKEND, {"llm"},
            LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN),
        ),
        EngineResourceInfo(
            device=torch.device("cpu"), dependencies={POOL: pool_spec()},
        ),
    )


def ctx(rids: list[str], *, is_preplan: bool = False) -> StepContext:
    return StepContext(
        request_ids=rids, graph_walk="decode", slot=0, capture=False,
        is_preplan=is_preplan,
    )


def decode_step(rids: list[str], **kwargs) -> RecurrentStep:
    return RecurrentStep(
        segments=tuple(Segment(rid, "main", 1) for rid in rids), **kwargs
    )


def test_both_resources_declare_preplan():
    """The point of the change: neither used to, so both planned inline."""
    assert build_pool().supports_preplan
    assert build_backend().supports_preplan


def test_promoted_plan_is_the_staged_one():
    pool = build_pool()
    rids = ["a", "b", "c"]
    step = decode_step(rids)
    pool.admit(step, ctx(rids))

    staged = pool.plan(step, ctx(rids, is_preplan=True))
    staged_indices = staged["main"].slot_indices.clone()

    promoted = pool.plan(step, ctx(rids))
    assert promoted is staged, "promotion should hand back the staged object"
    torch.testing.assert_close(promoted["main"].slot_indices, staged_indices)
    # and the stage is spent, so the next step plans for itself
    assert not pool._preplanned


def test_staging_matches_planning_inline():
    """Same step, staged vs not, must address identically."""
    rids = ["a", "b"]
    inline = build_pool()
    staged = build_pool()
    for pool in (inline, staged):
        pool.admit(decode_step(rids), ctx(rids))

    got_inline = inline.plan(decode_step(rids), ctx(rids))
    staged.plan(decode_step(rids), ctx(rids, is_preplan=True))
    got_staged = staged.plan(decode_step(rids), ctx(rids))

    torch.testing.assert_close(
        got_staged["main"].slot_indices, got_inline["main"].slot_indices,
    )
    torch.testing.assert_close(
        got_staged["main"].has_state, got_inline["main"].has_state,
    )


def test_abandoned_stage_rewinds_the_fork():
    """The sharp edge: pre-forks mutate live state during plan.

    The copy itself is idempotent — it reads a source the staged step never
    wrote — but ``generation`` is not, and a slot left marked ``has_state``
    would make the next step resume from a fork that never ran.
    """
    pool = build_pool()
    rids = ["a"]
    step = decode_step(rids, pre_forks=(("main", "draft"),))
    pool.admit(step, ctx(rids))

    # give "main" state to fork from, so the fork actually changes the target
    main = pool._slots["a"]["main"]
    main.has_state = True
    before = pool._slots["a"]["draft"]
    was = (before.has_state, before.generation)

    pool.plan(step, ctx(rids, is_preplan=True))
    assert pool._slots["a"]["draft"].has_state is True, "fork should have applied"

    pool.clear_preplan()
    after = pool._slots["a"]["draft"]
    assert (after.has_state, after.generation) == was
    assert not pool._preplanned


def test_promoted_fork_is_not_rewound():
    """Promotion keeps the staged fork — only abandonment undoes it."""
    pool = build_pool()
    rids = ["a"]
    step = decode_step(rids, pre_forks=(("main", "draft"),))
    pool.admit(step, ctx(rids))
    pool._slots["a"]["main"].has_state = True

    pool.plan(step, ctx(rids, is_preplan=True))
    generation = pool._slots["a"]["draft"].generation
    pool.plan(step, ctx(rids))

    draft = pool._slots["a"]["draft"]
    assert draft.has_state is True
    assert draft.generation == generation, "promotion must not rewind"


def test_staging_twice_is_refused():
    pool = build_pool()
    rids = ["a"]
    pool.admit(decode_step(rids), ctx(rids))
    pool.plan(decode_step(rids), ctx(rids, is_preplan=True))
    with pytest.raises(AssertionError, match="already pending"):
        pool.plan(decode_step(rids), ctx(rids, is_preplan=True))


def test_runner_accepts_the_pair():
    """``StepRunner`` refuses a resource that pre-plans while a dependency
    does not — the backend reads the pool's output, so opting one in without
    the other would stage against a plan that had not been made."""
    from mstar.engine.resources.runner import StepRunner

    runner = StepRunner({POOL: build_pool(), BACKEND: build_backend()})
    assert set(runner._preplan_order) == {POOL, BACKEND}
    assert runner._preplan_order.index(POOL) < runner._preplan_order.index(BACKEND)


def test_backend_promotes_against_the_pools_addressing():
    """The backend reads the pool's output off ``ctx.plan_results``, so the
    two must stage and promote together — hence ``_check_preplan_deps``."""
    pool, backend = build_pool(), build_backend()
    rids = ["a", "b"]
    step = decode_step(rids)

    pre = ctx(rids, is_preplan=True)
    pool.admit(step, pre)
    pre.plan_results[POOL] = pool.plan(step, pre)
    staged = backend.plan(step, pre)

    live = ctx(rids)
    live.plan_results[POOL] = pool.plan(step, live)
    assert backend.plan(step, live) is staged
    assert not backend._preplanned


def test_state_dtype_override():
    """Precision of the recurrent state is a deployment call.

    bf16 halves the bandwidth on a tensor read and written every step and is
    what reaches FlashInfer's fused decode kernel, but it accumulates over a
    whole generation — so the model's default has to be overridable per
    deployment without touching block *shapes*, which are not negotiable.
    """
    spec = pool_spec()
    before = spec.config.blocks["state"]

    spec.apply_yaml_overrides(max_slots=97)
    assert spec.config.blocks["state"] is before, "max_slots must not touch dtype"
    assert spec.config.max_slots == 97

    spec.apply_yaml_overrides(state_dtype="float32")
    after = spec.config.blocks["state"]
    assert after.dtype is torch.float32
    # shape and sharding survive the swap, and the conv block is untouched
    assert after.shape == before.shape and after.shard_dims == before.shard_dims
    assert spec.config.blocks["conv"].dtype is before.dtype

    for bad in ("float33", "nn", "Tensor"):
        with pytest.raises(ValueError, match="not a torch dtype"):
            pool_spec().apply_yaml_overrides(state_dtype=bad)
