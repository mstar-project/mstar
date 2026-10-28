"""The denoise step's compile switches: the mode's torch.compile arguments,
which forwards get compiled, and the yaml validation."""

import pytest
import torch

from mstar.engine import accelerator_graph_runner
from mstar.model.cosmos3.config import DENOISE_COMPILE_MODES, Cosmos3Config
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule, denoise_compile_kwargs


def test_deterministic_mode_turns_on_inductors_deterministic_option():
    import torch._inductor.config as inductor_config

    kwargs = denoise_compile_kwargs("deterministic")
    assert kwargs["fullgraph"] is False and kwargs["dynamic"] is False
    if hasattr(inductor_config, "deterministic"):
        assert kwargs["options"] == {"deterministic": True}
    assert "mode" not in kwargs


def test_default_mode_is_torch_compiles_defaults():
    assert denoise_compile_kwargs("default") == {"fullgraph": False, "dynamic": False}


@pytest.mark.parametrize("level", ["0", "1", "2"])
def test_level_mode_follows_the_serve_compile_level(monkeypatch, level):
    monkeypatch.setenv("MSTAR_COMPILE_LEVEL", level)
    accelerator_graph_runner.compile_level.cache_clear()
    try:
        assert denoise_compile_kwargs("level") == accelerator_graph_runner.compile_kwargs(None)
        want = accelerator_graph_runner.COMPILE_LEVELS[int(level)]
        assert {k: v for k, v in denoise_compile_kwargs("level").items() if k in want} == want
    finally:
        accelerator_graph_runner.compile_level.cache_clear()


class _Transformer(torch.nn.Module):
    """The three denoise entry points the submodule wraps, and nothing else."""

    def denoise_step(self, *a):
        return a

    def denoise_step_batched_cfg(self, *a):
        return a

    def denoise_step_batched(self, *a):
        return a


@pytest.mark.parametrize("batched", [True, False])
def test_batched_denoise_is_compiled_only_when_asked(batched):
    # a compiled forward replaces the bound method with an instance attribute
    t = _Transformer()
    Cosmos3DiTSubmodule(t, Cosmos3Config(compile_denoise_batched=batched))
    assert "denoise_step" in vars(t) and "denoise_step_batched_cfg" in vars(t)
    assert ("denoise_step_batched" in vars(t)) is batched


def test_no_compile_leaves_every_forward_eager():
    t = _Transformer()
    Cosmos3DiTSubmodule(t, Cosmos3Config(compile_denoise=False))
    assert not {"denoise_step", "denoise_step_batched_cfg", "denoise_step_batched"} & set(vars(t))


def test_defaults_pass_validation():
    cfg = Cosmos3Config()
    assert cfg.denoise_compile_mode in DENOISE_COMPILE_MODES
    cfg.check_compile()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("denoise_compile_mode", "max-autotune", "denoise_compile_mode must be one of"),
        ("compile_denoise_batched", "false", "compile_denoise_batched must be true or false"),
        ("compile_denoise", 1, "compile_denoise must be true or false"),
    ],
)
def test_compile_switches_are_validated(field, value, match):
    cfg = Cosmos3Config(**{field: value})
    with pytest.raises(ValueError, match=match):
        cfg.check_compile()
