"""The proposed expert mask geometry is read-keyed, checked and wire-safe."""

from __future__ import annotations

import msgpack
import pytest

from mstar.conductor.conductor import _with_expert_layout
from mstar.conductor.request_info import WalkStep
from mstar.engine.state import (
    Array,
    Attend,
    ExpertCross,
    ExpertCrossLayout,
    ExpertSelf,
    ExpertSelfLayout,
    KVEntry,
    Layout,
    Read,
)
from mstar.engine.state.adapters import EXPERT_LAYOUT_KWARG, expert_layouts, kv_layouts


def _arrays() -> list[Array]:
    return [
        Array(
            "expert_queries", nodes={"expert"}, layers={0}, entry=KVEntry(heads=2, head_dim=32),
            max_len=0, reads={"self_attn": Read(mask=ExpertSelf(2, 1), op=Attend(q_heads=2))},
        ),
        Array(
            "trunk_keys", nodes={"expert"}, layers={0}, entry=KVEntry(heads=2, head_dim=32),
            max_len=0, reads={"cross_attn": Read(
                mask=ExpertCross(2, 1, 1, 2), op=Attend(q_heads=2), queries="expert_queries",
            )},
        ),
    ]


def _layouts() -> dict:
    return {
        "expert_queries": {"main": Layout(expert_self={"self_attn": ExpertSelfLayout(2)})},
        "trunk_keys": {"main": Layout(expert_cross={
            "cross_attn": ExpertCrossLayout(2, 2, 2, 4),
        })},
    }


def test_expert_layout_transport_is_read_keyed_and_small() -> None:
    arrays = _arrays()
    layout = expert_layouts(arrays, _layouts())
    assert layout == {
        "expert_queries": {"main": {"self_attn": [2]}},
        "trunk_keys": {"main": {"cross_attn": [2, 2, 2, 4]}},
    }
    assert msgpack.unpackb(msgpack.packb(layout, use_bin_type=True), raw=False) == layout
    (walk,) = _with_expert_layout([WalkStep("chunk")], layout)
    assert walk.step_metadata[EXPERT_LAYOUT_KWARG] == layout
    assert kv_layouts(arrays, _layouts(), {}) == {}


def test_episode_plan_expert_lengths_replace_only_their_read() -> None:
    arrays = _arrays()
    cold = {
        "expert_queries": {"main": {"self_attn": [1]}},
        "trunk_keys": {"main": {"cross_attn": [1, 2, 0, 2]}},
    }
    warm = {"trunk_keys": {"main": {"cross_attn": [2, 2, 2, 4]}}}
    (walk,) = _with_expert_layout(
        [WalkStep("chunk", step_metadata={EXPERT_LAYOUT_KWARG: warm})], cold, arrays,
    )
    assert walk.step_metadata[EXPERT_LAYOUT_KWARG] == {
        "expert_queries": {"main": {"self_attn": [1]}},
        "trunk_keys": {"main": {"cross_attn": [2, 2, 2, 4]}},
    }
    wire = walk.step_metadata[EXPERT_LAYOUT_KWARG]
    assert msgpack.unpackb(msgpack.packb(wire, use_bin_type=True), raw=False) == wire
    assert cold["trunk_keys"]["main"]["cross_attn"] == [1, 2, 0, 2]


def test_episode_plan_expert_lengths_work_without_data_worker_lengths() -> None:
    planned = {"trunk_keys": {"main": {"cross_attn": [2, 2, 2, 4]}}}
    (walk,) = _with_expert_layout(
        [WalkStep("chunk", step_metadata={EXPERT_LAYOUT_KWARG: planned})], None, _arrays(),
    )
    assert walk.step_metadata[EXPERT_LAYOUT_KWARG] == planned


@pytest.mark.parametrize("planned,match", [
    ({"missing": {"main": {"cross_attn": [2, 2, 2, 4]}}}, "undeclared array"),
    ({"trunk_keys": {"other": {"cross_attn": [2, 2, 2, 4]}}}, "undeclared label"),
    ({"trunk_keys": {"main": {"other": [2, 2, 2, 4]}}}, "no matching read"),
    ({"trunk_keys": {"main": {"cross_attn": [2, 2, 1, 4]}}}, "unsupported expert frame"),
])
def test_episode_plan_expert_lengths_are_checked(planned, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        _with_expert_layout(
            [WalkStep("chunk", step_metadata={EXPERT_LAYOUT_KWARG: planned})], None, _arrays(),
        )


@pytest.mark.parametrize("bad,pattern", [
    (ExpertCrossLayout(2, 2, 2, 2), "noised frames"),
    (ExpertCrossLayout(2, 2, 1, 4), "unsupported expert frame layout"),
])
def test_bad_cross_geometry_fails_before_transport(bad, pattern) -> None:
    layouts = _layouts()
    layouts["trunk_keys"]["main"] = Layout(expert_cross={"cross_attn": bad})
    with pytest.raises(RuntimeError, match=pattern):
        kv_layouts(_arrays(), layouts, {})


def test_wrong_read_key_or_kind_fails_before_transport() -> None:
    layouts = _layouts()
    layouts["trunk_keys"]["main"] = Layout(expert_cross={"other": ExpertCrossLayout(2, 2, 2, 4)})
    with pytest.raises(RuntimeError, match="names no ExpertCross read"):
        expert_layouts(_arrays(), layouts)
    layouts["trunk_keys"]["main"] = Layout(expert_self={"cross_attn": ExpertSelfLayout(2)})
    with pytest.raises(RuntimeError, match="names no ExpertSelf read"):
        expert_layouts(_arrays(), layouts)
