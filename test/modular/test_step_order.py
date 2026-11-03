"""Who goes first at a capped step: ARRIVAL unless a submodule declares otherwise,
and the two kinds of session that have to keep moving while others run do."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mstar.engine.engine import Engine
from mstar.model.submodule_base import NodeSubmodule, StepOrder

pytest.importorskip("diffusers")

from mstar.model.cosmos3.constants import VIDEO_DECODE_AR_WALK, VIDEO_GEN_AR_WALK, VIDEO_GEN_WALK  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    Cosmos3DiTSubmodule,
    Cosmos3VAEDecoderARSubmodule,
    Cosmos3VAEDecoderSubmodule,
)
from mstar.model.lingbot_world.submodules import ROLLOUT_WALK, LingBotDitSubmodule  # noqa: E402
from mstar.model.wan22.submodules import Wan22DitSubmodule  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402


def _order_of(submodule_class: type[NodeSubmodule], walk: str) -> StepOrder:
    """The order the scheduler reads for a node of ``submodule_class``, through
    the real ``Engine.get_step_order``. No weights: the hook reads no state."""
    submodule = submodule_class.__new__(submodule_class)
    engine = SimpleNamespace(
        _submodules={"node": SimpleNamespace(submodule=submodule)},
        get_step_order=lambda node, graph_walk: Engine.get_step_order(engine, node, graph_walk),
    )
    sched = MicroScheduler(engine_manager=SimpleNamespace(get_engine=lambda name: engine))
    return sched._order_of(("node", walk))


def test_a_node_that_declares_nothing_goes_by_arrival():
    assert NodeSubmodule.step_order(object(), "decode") is StepOrder.ARRIVAL
    assert _order_of(Wan22DitSubmodule, "t2v") is StepOrder.ARRIVAL


def test_lingbots_rollout_takes_turns():
    assert _order_of(LingBotDitSubmodule, ROLLOUT_WALK) is StepOrder.TURNS


def test_cosmos3s_chained_windows_take_turns():
    assert _order_of(Cosmos3DiTSubmodule, VIDEO_GEN_AR_WALK) is StepOrder.TURNS
    assert _order_of(Cosmos3VAEDecoderARSubmodule, VIDEO_DECODE_AR_WALK) is StepOrder.TURNS


def test_cosmos3s_other_walks_go_by_arrival():
    assert _order_of(Cosmos3DiTSubmodule, VIDEO_GEN_WALK) is StepOrder.ARRIVAL
    assert _order_of(Cosmos3VAEDecoderSubmodule, VIDEO_GEN_WALK) is StepOrder.ARRIVAL
