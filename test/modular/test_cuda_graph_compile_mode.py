"""``compile_mode`` on a capture config, and the MSTAR_GRAPH_COMPILE_MODE
override — resolved once in ``cuda_graph_runner.compile_kwargs``."""

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

ENV = "MSTAR_GRAPH_COMPILE_MODE"


@pytest.fixture
def no_env_override(monkeypatch):
    # The import-time read lands in a module attribute: patch that, never
    # importlib.reload the module — a reload re-creates every class in it
    # for the rest of the session, so isinstance against a runner imported
    # elsewhere fails. Also keeps these tests green in a shell that exports
    # the variable.
    monkeypatch.setattr(runner_mod, "_COMPILE_MODE_OVERRIDE", None)


def test_config_mode_defaults_to_the_historical_autotune(no_env_override):
    assert resolve_compile_mode(None) == "max-autotune-no-cudagraphs"
    assert compile_kwargs(None) == {
        "fullgraph": False, "dynamic": False, "mode": "max-autotune-no-cudagraphs",
    }


def test_config_names_its_own_mode(no_env_override):
    assert compile_kwargs("default") == {"fullgraph": False, "dynamic": False}
    assert compile_kwargs("reduce-overhead")["mode"] == "reduce-overhead"


def test_invalid_config_mode_is_refused(no_env_override):
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


def test_env_override_wins_over_the_config(monkeypatch):
    monkeypatch.setattr(runner_mod, "_COMPILE_MODE_OVERRIDE", "default")
    assert resolve_compile_mode("max-autotune-no-cudagraphs") == "default"
    assert compile_kwargs("max-autotune-no-cudagraphs") == {"fullgraph": False, "dynamic": False}


def test_env_override_is_read_and_validated_at_import():
    # the import-time statement is `_read_compile_mode_override()` over
    # os.environ; drive the same function over an explicit mapping
    read = runner_mod._read_compile_mode_override
    assert read({}) is None
    assert read({ENV: "default"}) == "default"
    assert read({ENV: "reduce-overhead"}) == "reduce-overhead"
    with pytest.raises(ValueError, match=f"{ENV}='no-such-mode'"):
        read({ENV: "no-such-mode"})


def test_module_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv(ENV, "default")
    assert runner_mod._read_compile_mode_override() == "default"
    monkeypatch.delenv(ENV)
    assert runner_mod._read_compile_mode_override() is None
