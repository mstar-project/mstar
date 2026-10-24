"""``compile_mode`` on a capture config: checked when the config is built, and
turned into ``torch.compile``'s arguments by ``accelerator_graph_runner.compile_kwargs``,
with the deployment's compile level (``MSTAR_COMPILE_LEVEL``) filling in a None mode."""

import sys

import pytest

sys.path.insert(0, ".")

from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
)
from mstar.engine.accelerator_graph_runner import compile_kwargs, compile_level
from mstar.model.submodule_base import NodeInputs


@pytest.fixture
def level(monkeypatch):
    """Set ``MSTAR_COMPILE_LEVEL`` (None unsets it) for ``compile_level``, which caches it."""

    def use(value):
        if value is None:
            monkeypatch.delenv("MSTAR_COMPILE_LEVEL", raising=False)
        else:
            monkeypatch.setenv("MSTAR_COMPILE_LEVEL", value)
        compile_level.cache_clear()

    yield use
    compile_level.cache_clear()


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


CD = {"coordinate_descent_tuning": True}


@pytest.mark.parametrize("value, expected", [
    (None, {"options": CD}),  # level 1 by default
    ("0", {}),
    ("1", {"options": CD}),
    ("2", {"mode": "max-autotune-no-cudagraphs"}),
    ("3", {"options": CD}),
    ("fast", {"options": CD}),
])
def test_no_config_mode_takes_the_compile_level(level, value, expected):
    level(value)
    assert compile_kwargs(None) == {"fullgraph": False, "dynamic": False, **expected}


def test_config_names_its_own_mode(level):
    level("0")  # a named mode is outside the level
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

