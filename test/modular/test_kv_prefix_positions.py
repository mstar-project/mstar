"""A matched prefix has to carry its positions with it.

Rotation is applied before a key is written, so a cached page is only valid at
the positions its tokens were at when it was stored. The position counter is
owned by a different resource than the cache, and it has never seen those
tokens: left at zero it would place this step's first token at position 0, on
top of a prefix that is already 96 tokens long, and the step would attend its
own keys at the wrong rotation. Nothing raises when that happens — the output is
just wrong — so these tests read the position ids the plan actually produces.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import KVConfig, PositionConfig, StepContext, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionSpec, PositionStep, PosScheme
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.worker.engine_manager import _refuse_uncacheable_positions

KV = "kv"
ROPE = "rope"
RID = "r0"
NODE = "LLM"
WALK = "prefill"


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


class _Node:
    """The KV and position resources of one node, under a real runner."""

    def __init__(self):
        device = torch.device("cpu")
        self.kv = KVManager(
            cfg=KVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
                max_num_pages=64, page_size=16,
            ),
            name=KV, joint_comm_group=None, transfer_engine_info=None,
            device=device, dtype=torch.float32,
        )
        self.rope = RopeManager(
            config=PositionConfig(kv_cache=KV), device=device,
            dtype=torch.float32,
        )
        self.runner = StepRunner(
            {KV: self.kv, ROPE: self.rope},
            node_resources={NODE: [KV, ROPE]},
        )
        self.runner.ingest_request(RID)

    def step(self, span: int, label: str = "main") -> list[int]:
        """Admit and plan one step, and give back its position ids."""
        step = SubmoduleStep(
            steps={KV: KVStep(), ROPE: PositionStep()},
            segments=[Segment(RID, label, span)],
        )
        ctx = StepContext(
            request_ids=(RID,), graph_walk=WALK, slot=0, capture=False,
        )
        step.set_ctx(ctx)
        assert self.runner.admit(step).outcome.ok
        pos_ids = self.runner.plan(step)[ROPE][label].tolist()
        self.runner.commit(step)
        return pos_ids

    def reset(self) -> None:
        for resource in (self.kv, self.rope):
            resource.reset_request(RID)


# ── where the step lands ────────────────────────────────────────────────


def test_the_step_after_a_hit_is_placed_past_the_matched_prefix():
    node = _Node()

    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)

    assert node.step(4) == [96, 97, 98, 99], (
        "the step was placed on top of the prefix the cache matched"
    )


def test_a_request_that_matched_nothing_still_starts_at_zero():
    node = _Node()

    assert node.step(4) == [0, 1, 2, 3]


def test_the_counter_is_seeded_once_and_then_advances_on_its_own():
    node = _Node()
    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)

    first = node.step(4)
    second = node.step(2)

    assert first == [96, 97, 98, 99]
    assert second == [100, 101], "the seed was re-applied on a later step"


def test_a_shorter_match_never_rewinds_the_counter():
    node = _Node()
    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)
    node.step(4)

    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 32)

    assert node.step(1) == [100], "a stale match pulled the counter backwards"


def test_a_reset_forgets_what_the_cache_matched():
    node = _Node()
    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)

    node.reset()

    assert node.step(2) == [0, 1], "a reset request kept a matched length"


# ── the load-time check ─────────────────────────────────────────────────


def _specs(scheme: PosScheme) -> list:
    return [
        KVSpec(
            resource_key=KV, nodes={NODE},
            config=KVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64,
            ),
        ),
        PositionSpec(
            resource_key=ROPE, nodes={NODE},
            config=PositionConfig(kv_cache=KV, scheme=scheme),
        ),
    ]


class _Model:
    """Declares whichever resources the test wants keyed."""

    def __init__(self, *keyed: str):
        self._keyed = keyed

    def prefix_key_streams(self):
        return {key: {} for key in self._keyed}


def test_a_declared_node_over_a_block_scheme_is_refused_at_load():
    # no shipped model declares BLOCK, so the spec is built here
    with pytest.raises(ValueError, match="sequential"):
        _refuse_uncacheable_positions(_specs(PosScheme.BLOCK), _Model(KV))


def test_a_declared_node_over_a_sequential_scheme_loads():
    _refuse_uncacheable_positions(_specs(PosScheme.SEQUENTIAL), _Model(KV))


def test_an_undeclared_node_is_not_checked():
    _refuse_uncacheable_positions(_specs(PosScheme.BLOCK), _Model())


def test_only_the_cache_that_was_declared_is_checked():
    _refuse_uncacheable_positions(_specs(PosScheme.BLOCK), _Model("some_other_kv"))
