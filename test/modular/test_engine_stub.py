"""``make_engine`` builds a real ``Engine`` without loading a model."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine, SubmoduleManagement
from mstar.engine.resources.runner import StepRunner
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.resource import make_resource


def test_an_engine_starts_empty_on_the_cpu():
    engine = make_engine()
    assert isinstance(engine, Engine)
    assert engine._device == torch.device("cpu")
    assert engine._resources == {} and engine._submodules == {}
    assert engine._tp_follower_nodes == set() and not engine._enable_nvtx


def test_it_has_the_fields_only_load_model_sets():
    engine = make_engine()
    assert engine._loopback_steps == {} and engine._keyed_walks == {} and engine._item_encoders == set()


def test_an_override_replaces_a_field_or_a_method():
    runner = StepRunner({})
    engine = make_engine(_runner=runner, get_max_batch_size=lambda node, walk: 7)
    assert engine._runner is runner
    assert engine.get_max_batch_size("LLM", "decode") == 7


def test_a_name_engine_does_not_have_is_refused():
    with pytest.raises(TypeError, match="_submodule"):
        make_engine(_submodule={})


def test_chunking_sees_the_overridden_submodules():
    """``ChunkingManagement`` holds the engine's dict, so replacing it would leave chunking reading an empty one."""
    submodules = {"LLM": make_mgmt()}
    engine = make_engine(_submodules=submodules)
    assert engine._submodules == submodules
    assert engine._chunking._submodules is engine._submodules


def test_engines_share_no_state():
    make_engine(_submodules={"LLM": make_mgmt()})._resources["kv"] = object()
    engine = make_engine()
    assert engine._submodules == {} and engine._resources == {}


def test_a_mgmt_is_real_and_empty():
    mgmt = make_mgmt()
    assert isinstance(mgmt, SubmoduleManagement)
    assert mgmt.resources == {} and mgmt.piecewise_runners == {} and mgmt.resident is None


def test_a_mgmt_forwards_through_its_submodule_as_load_model_sets_it():
    submodule = SimpleNamespace(forward="one", forward_batched="many")
    mgmt = make_mgmt(submodule)
    assert mgmt.submodule is submodule and mgmt.forward == "one" and mgmt.forward_batched == "many"
    assert make_mgmt(submodule, forward="other").forward == "other"


def test_a_mgmt_takes_a_stand_in_with_no_forward():
    assert make_mgmt(object()).forward is None


def test_a_mgmt_refuses_a_field_it_does_not_have():
    with pytest.raises(TypeError, match="piecewise_runner"):
        make_mgmt(piecewise_runner={})


def test_a_mgmt_double_buffers_for_a_resource_that_asks():
    """The real ``__post_init__`` decides, which a namespace would skip."""
    assert make_mgmt().num_slots == 1
    assert make_mgmt(resources={"kv": make_resource(force_double_buffer=True)}).num_slots == 2
