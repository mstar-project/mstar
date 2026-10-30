"""``compile_mode`` on a capture config: checked when the config is built, and
turned into ``torch.compile``'s arguments by ``accelerator_graph_runner.compile_kwargs``."""

import sys

import pytest

sys.path.insert(0, ".")

from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
)
from mstar.engine.accelerator_graph_runner import compile_kwargs
from mstar.model.submodule_base import NodeInputs


def _batched(mode):
    return BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode", single_request_inputs=NodeInputs(input_seq_len=1),
        compile_mode=mode,
    )


def _piecewise(mode):
    return PiecewiseBatchedConfig(
        capture_fn=lambda call: {}, make_static_inputs=lambda shape: {}, seq_len=1,
        compile_mode=mode,
    )


def test_config_names_its_own_mode():
    assert compile_kwargs("default") == {"fullgraph": False, "dynamic": False}
    assert compile_kwargs("max-autotune-no-cudagraphs")["mode"] == "max-autotune-no-cudagraphs"


@pytest.mark.parametrize("build", [_batched, _piecewise])
@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_a_mode_with_inductor_cudagraphs_is_refused_at_build(build, mode):
    # inside the runner's own capture they fail it, and the bucket serves eager
    with pytest.raises(ValueError, match="no-cudagraphs"):
        build(mode)


@pytest.mark.parametrize("build", [_batched, _piecewise])
@pytest.mark.parametrize("mode, match", [
    ("no-such-mode", "compile_mode='no-such-mode'"), ("", "compile_mode is empty"),
])
def test_an_invalid_mode_is_refused_at_build(build, mode, match):
    with pytest.raises(ValueError, match=match):
        build(mode)


def test_configs_carry_the_field():
    assert _batched("default").compile_mode == "default"
    assert _batched(None).compile_mode is None
    packed = PackedAcceleratorGraphConfig(
        capture_graph_walk="prefill", capture_token_lengths=[8],
        make_node_input=lambda n: NodeInputs(input_seq_len=n), compile_mode="default",
    )
    assert packed.compile_mode == "default"
    assert _piecewise("default").compile_mode == "default"
    assert _piecewise(None).compile_mode is None


@pytest.fixture
def fresh_deterministic_check():
    from mstar.engine import accelerator_graph_runner

    accelerator_graph_runner._has_deterministic_mode.cache_clear()
    yield
    accelerator_graph_runner._has_deterministic_mode.cache_clear()


@pytest.mark.parametrize("build", [_batched, _piecewise])
def test_the_deterministic_mode_is_the_engines_own(build, fresh_deterministic_check):
    """Not a torch.compile mode: it turns into Inductor's deterministic option."""
    import torch._inductor.config as inductor_config

    assert build("deterministic").compile_mode == "deterministic"
    kwargs = compile_kwargs("deterministic")
    assert kwargs["fullgraph"] is False and kwargs["dynamic"] is False and "mode" not in kwargs
    if hasattr(inductor_config, "deterministic"):
        assert kwargs["options"] == {"deterministic": True}


def test_a_torch_without_the_deterministic_mode_says_so(monkeypatch, caplog, fresh_deterministic_check):
    import logging

    import torch._inductor.config as inductor_config

    from mstar.engine.accelerator_graph_runner import deterministic_inductor_options

    if hasattr(inductor_config, "deterministic"):
        monkeypatch.delattr(inductor_config, "deterministic")
    with caplog.at_level(logging.WARNING, logger="mstar.engine.accelerator_graph_runner"):
        assert deterministic_inductor_options() == {}
        assert compile_kwargs("deterministic") == {"fullgraph": False, "dynamic": False}
    assert caplog.text.count("has no Inductor deterministic mode") == 1
