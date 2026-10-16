"""Blocks held per slot rather than per layer: one tensor for the whole stack.

They live and die with the slot like the per-layer blocks: handed out as zeros,
zeroed on release, copied by a fork. A speculating delta net declares some.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext

LAYERS, SLOTS = 3, 4


def build_pool() -> RecurrentStatePool:
    config = RecurrentStateConfig(
        num_layers=LAYERS,
        blocks={
            "state": RecurrentBlockConfig(shape=(2, 4), dtype=torch.float32),
            "count": RecurrentBlockConfig(shape=(1,), dtype=torch.int32, per_layer=False),
        },
        max_slots=SLOTS,
    )
    spec = RecurrentStateSpec("pool", {"llm"}, config)
    return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))


def run(pool: RecurrentStatePool, step: RecurrentStep, rids) -> None:
    ctx = StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False)
    for rid in rids:
        pool.ingest_request(rid)
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    pool.commit(step, ctx)


def slot_of(pool: RecurrentStatePool, rid: str, label: str = "main") -> int:
    return pool._slots[rid][label].index


def test_a_per_slot_block_has_no_layer_axis():
    pool = build_pool()
    assert pool.block("count").shape == (SLOTS, 1)
    assert pool.block("state", 1).shape == (SLOTS, 2, 4)
    with pytest.raises(ValueError, match="per slot"):
        pool.block("count", 0)
    with pytest.raises(ValueError, match="per layer"):
        pool.block("state")
    assert pool.config.slot_bytes == LAYERS * 2 * 4 * 4 + 4


def test_released_slots_are_zeroed_and_forks_copy():
    pool = build_pool()
    run(pool, RecurrentStep(segments=[Segment("a", "main", 3)]), ["a"])
    a = slot_of(pool, "a")
    pool.block("count")[a] = 7
    pool.block("state", 2)[a] = 1.5

    fork = RecurrentStep(segments=[Segment("a", "main", 1)], pre_forks=(("main", "copy"),))
    run(pool, fork, ["a"])
    copy = slot_of(pool, "a", "copy")
    assert int(pool.block("count")[copy]) == 7
    assert torch.equal(pool.block("state", 2)[copy], pool.block("state", 2)[a])

    pool.remove_request("a")
    assert pool.block("count").abs().sum() == 0
    assert pool.block("state", 2).abs().sum() == 0


def test_speculative_delta_net_blocks_shard_with_the_heads():
    geometry = DeltaNetGeometry(
        num_k_heads=4, num_v_heads=4, head_k_dim=8, head_v_dim=8, conv_kernel_size=4,
    )
    assert set(geometry.to_blocks()) == {"state", "conv"}
    config = RecurrentStateConfig(num_layers=2, blocks=geometry.to_blocks(speculative_tokens=3))
    config.shard(2)
    shapes = {name: block.shape for name, block in config.blocks.items()}
    assert shapes["spec_prefix"] == (2, 4, 48) and shapes["spec_conv"] == (2, 48, 3)
    assert shapes["spec_g"] == (2, 4, 2, 8) and shapes["spec_beta"] == (2, 4, 2)
    assert shapes["spec_len"] == shapes["spec_side"] == (1,)
    assert not config.blocks["spec_len"].per_layer
    assert DeltaNetGeometry.speculative_tokens_of(config.blocks) == 3
    assert DeltaNetGeometry.from_blocks(config.blocks).num_v_heads == 2
