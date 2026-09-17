"""``compile_mode`` on a capture config, and the MSTAR_GRAPH_COMPILE_MODE
override — resolved once in ``cuda_graph_runner.compile_kwargs``."""

import importlib
import sys

import pytest

sys.path.insert(0, ".")

import mstar.engine.cuda_graph_runner as runner_mod
from mstar.engine.cuda_graph_config import (
    BatchedCudaGraphConfig,
    PackedCudaGraphConfig,
    PiecewiseBatchedConfig,
)
from mstar.engine.cuda_graph_runner import compile_kwargs, resolve_compile_mode
from mstar.model.submodule_base import NodeInputs


def test_config_mode_defaults_to_the_historical_autotune():
    assert runner_mod._COMPILE_MODE_OVERRIDE is None, "run without the env var set"
    assert resolve_compile_mode(None) == "max-autotune-no-cudagraphs"
    assert compile_kwargs(None) == {
        "fullgraph": False, "dynamic": False, "mode": "max-autotune-no-cudagraphs",
    }


def test_config_names_its_own_mode():
    assert compile_kwargs("default") == {"fullgraph": False, "dynamic": False}
    assert compile_kwargs("reduce-overhead")["mode"] == "reduce-overhead"


def test_invalid_config_mode_is_refused():
    with pytest.raises(ValueError, match="compile_mode='no-such-mode'"):
        compile_kwargs("no-such-mode")


def test_configs_carry_the_field():
    inputs = NodeInputs(input_seq_len=1)
    cfg = BatchedCudaGraphConfig(
        capture_graph_walk="decode", single_request_inputs=inputs, compile_mode="default",
    )
    assert cfg.compile_mode == "default"
    assert BatchedCudaGraphConfig(
        capture_graph_walk="decode", single_request_inputs=inputs,
    ).compile_mode is None
    packed = PackedCudaGraphConfig(
        capture_graph_walk="prefill", capture_token_lengths=[8],
        make_node_input=lambda n: NodeInputs(input_seq_len=n), compile_mode="default",
    )
    assert packed.compile_mode == "default"
    pw = PiecewiseBatchedConfig(
        capture_fn=lambda call: {}, make_static_inputs=lambda shape: {}, seq_len=1,
        compile_mode="default",
    )
    assert pw.compile_mode == "default"


def test_env_override_wins_and_invalid_env_fails_at_import(monkeypatch):
    monkeypatch.setenv("MSTAR_GRAPH_COMPILE_MODE", "default")
    mod = importlib.reload(runner_mod)
    try:
        assert mod.resolve_compile_mode("max-autotune-no-cudagraphs") == "default"
        assert mod.compile_kwargs("max-autotune-no-cudagraphs") == {
            "fullgraph": False, "dynamic": False,
        }
        monkeypatch.setenv("MSTAR_GRAPH_COMPILE_MODE", "no-such-mode")
        with pytest.raises(ValueError, match="MSTAR_GRAPH_COMPILE_MODE='no-such-mode'"):
            importlib.reload(runner_mod)
    finally:
        monkeypatch.delenv("MSTAR_GRAPH_COMPILE_MODE")
        importlib.reload(runner_mod)
