"""``make_engine`` builds a real ``Engine`` without loading a model."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.runner import StepRunner
from mstar.testing_utils.engine import make_engine


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
    submodules = {"LLM": SimpleNamespace(submodule=SimpleNamespace())}
    engine = make_engine(_submodules=submodules)
    assert engine._submodules == submodules
    assert engine._chunking._submodules is engine._submodules


def test_engines_share_no_state():
    make_engine(_submodules={"LLM": SimpleNamespace()})._resources["kv"] = object()
    engine = make_engine()
    assert engine._submodules == {} and engine._resources == {}
