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
from dataclasses import dataclass

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    KVConfig,
    PositionConfig,
    Resource,
    StepContext,
    StepRunner,
)
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionSpec, PositionStep, PosScheme
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.base import PrefixStream
from mstar.worker.engine_manager import (
    _refuse_uncacheable_positions,
    _refuse_unknown_walks,
    _refuse_unskippable_resources,
)

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

    assert node.step(4) == [0, 1, 2, 3], (
        "a request that matched nothing was placed past a prefix it never had"
    )


def test_the_counter_is_seeded_once_and_then_advances_on_its_own():
    node = _Node()
    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)

    first = node.step(4)
    second = node.step(2)

    assert first == [96, 97, 98, 99], "the first step did not start at the match"
    assert second == [100, 101], "the seed was re-applied on a later step"


def test_a_shorter_match_never_rewinds_the_counter():
    node = _Node()
    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)
    node.step(4)

    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 32)

    assert node.step(1) == [100], "a stale match pulled the counter backwards"


def test_a_label_the_request_writes_after_its_hit_starts_at_zero():
    node = _Node()
    node.rope.apply_cached_prefix(RID, NODE, WALK, None, 96)
    node.step(4)

    assert node.step(2, label="cfg_text") == [0, 1], (
        "a label written after the hit started past a prefix only another "
        "label holds"
    )


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


def _built_rope(specs: list) -> RopeManager:
    """The position resource the load builds once the check has passed it."""
    spec = next(spec for spec in specs if isinstance(spec, PositionSpec))
    return RopeManager(
        config=spec.config, device=torch.device("cpu"), dtype=torch.float32,
    )


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
    specs = _specs(PosScheme.SEQUENTIAL)

    _refuse_uncacheable_positions(specs, _Model(KV))

    assert _built_rope(specs)._config.scheme is PosScheme.SEQUENTIAL, (
        "the node the model keyed did not load under the scheme it asked for"
    )


def test_an_undeclared_node_is_not_checked():
    specs = _specs(PosScheme.BLOCK)

    _refuse_uncacheable_positions(specs, _Model())

    assert _built_rope(specs)._config.scheme is PosScheme.BLOCK, (
        "a node nobody keyed lost the scheme the deployment asked for"
    )


def test_only_the_cache_that_was_declared_is_checked():
    specs = _specs(PosScheme.BLOCK)

    _refuse_uncacheable_positions(specs, _Model("some_other_kv"))

    assert _built_rope(specs)._config.scheme is PosScheme.BLOCK, (
        "a position resource over a cache nobody keyed was held to the "
        "sequential rule"
    )


# ── what else the node carries ──────────────────────────────────────────


class _Unskippable(Resource):
    """A resource that keeps state of its own and never learned the hooks."""

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError


@dataclass
class _StubSpec(NodeResourceSpec):
    """Declares `_Unskippable` on whichever nodes the test puts it on."""

    @property
    def resource_class(self):
        return _Unskippable


def _with_stub(*nodes: str) -> list:
    return _specs(PosScheme.SEQUENTIAL) + [
        _StubSpec(resource_key="recurrent_state", nodes=set(nodes)),
    ]


def test_a_declared_node_carrying_a_resource_that_cannot_skip_is_refused():
    with pytest.raises(ValueError, match="recurrent_state") as refusal:
        _refuse_unskippable_resources(_with_stub(NODE), _Model(KV))

    assert NODE in str(refusal.value), (
        "the refusal does not say which node the resource sits on"
    )


def test_the_same_node_without_the_declaration_loads():
    _refuse_unskippable_resources(_with_stub(NODE), _Model())


def test_a_resource_that_cannot_skip_loads_beside_a_declared_node():
    _refuse_unskippable_resources(_with_stub("Talker"), _Model(KV))


def test_a_declared_node_of_caches_positions_attention_and_sampling_loads():
    _refuse_unskippable_resources(_specs(PosScheme.SEQUENTIAL), _Model(KV))


# ── the walks a declaration names ───────────────────────────────────────


class _Walking:
    """Declares one stream on ``walk`` and runs the walks it is given."""

    def __init__(self, walk: str, runs: tuple[str, ...] = ("prefill", "decode")):
        self._walk = walk
        self._runs = runs

    def prefix_key_streams(self):
        return {KV: {"main": PrefixStream("text_inputs", "ids", self._walk, "decode")}}

    def get_graph_walk_graphs(self):
        return {walk: None for walk in self._runs}


def test_a_stream_naming_a_walk_the_model_never_runs_is_refused_at_load():
    with pytest.raises(ValueError, match="prefill_text") as refusal:
        _refuse_unknown_walks(_Walking("prefill_text"))

    assert "prefill" in str(refusal.value) and "decode" in str(refusal.value), (
        "the refusal does not say which walks the model does run"
    )


def test_a_stream_naming_walks_the_model_runs_loads():
    _refuse_unknown_walks(_Walking("prefill"))
