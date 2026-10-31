"""The denoise step's compile switches: the mode's torch.compile arguments,
which forwards get compiled, and the yaml validation."""

import pytest
import torch

from mstar.engine import accelerator_graph_runner
from mstar.model.cosmos3.config import DENOISE_COMPILE_MODES, Cosmos3Config
from mstar.model.cosmos3.submodules import (
    CFG_BATCHED_LABEL,
    COND_LABEL,
    UNCOND_LABEL,
    Cosmos3DiTSubmodule,
    denoise_compile_kwargs,
)
from mstar.model.cosmos3.tests.test_rotary_table import _tiny


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
    """The denoise entry points the submodule wraps, and nothing else."""

    def denoise_step(self, *a):
        return a

    def denoise_step_batched_cfg(self, *a):
        return a

    def gen_layers_batched(self, *a):
        return a


@pytest.mark.parametrize("batched", [True, False])
def test_batched_denoise_is_compiled_only_when_asked(batched):
    # a compiled forward replaces the bound method with an instance attribute
    t = _Transformer()
    Cosmos3DiTSubmodule(t, Cosmos3Config(compile_denoise_batched=batched))
    assert "denoise_step" in vars(t) and "denoise_step_batched_cfg" in vars(t)
    assert ("gen_layers_batched" in vars(t)) is batched


def test_only_the_batched_layer_stack_compiles_and_its_length_is_dynamic(monkeypatch):
    """The cross-request step compiles nothing a batch's composition shapes:
    the per-request parts stay eager, and the layer stack's pack length is
    dynamic, so one compile serves every batch size and mix of sizes."""
    calls = {}

    def fake_compile(fn, **kwargs):
        calls[fn.__name__] = kwargs
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    Cosmos3DiTSubmodule(_Transformer(), Cosmos3Config())
    want = denoise_compile_kwargs("deterministic")
    assert calls["denoise_step"] == want and calls["denoise_step_batched_cfg"] == want
    assert calls["gen_layers_batched"] == {**want, "dynamic": True}
    assert "denoise_step_batched" not in calls


def test_no_compile_leaves_every_forward_eager():
    t = _Transformer()
    Cosmos3DiTSubmodule(t, Cosmos3Config(compile_denoise=False))
    assert not {"denoise_step", "denoise_step_batched_cfg", "gen_layers_batched"} & set(vars(t))


def test_warmup_compiles_runs_the_real_step_and_a_served_row_reuses_its_compile(monkeypatch):
    """The warmup runs one two-request step through the real step at the
    default size with the layer stack compiled, so no request pays for it: a
    row of another size, against the attention a served step binds, runs what
    the warmup built."""
    sub, model, _, res = _tiny(compile_denoise=True, video_size_default=(64, 64), num_frames_video=9)
    step, calls = model.denoise_step_batched, []

    def recording_step(requests, label, attn):
        results = step(requests, label, attn)
        calls.append((requests, label, results))
        return results

    monkeypatch.setattr(model, "denoise_step_batched", recording_step)
    with torch.no_grad():  # as the engine calls it
        sub.warmup_compiles()
    [(requests, label, results)] = calls
    latent_size = sub._latent_shape(64, 64, 9)
    assert len(requests) == 2 and label == CFG_BATCHED_LABEL and requests[0]["latents"].shape == latent_size
    assert [v.shape for branches in results for v in branches] == [latent_size] * 4

    static = sub._build_static(list(range(5)), 32, 64, 5, 24.0, False, "cpu")
    latents = torch.randn(sub._latent_shape(32, 64, 5))
    timesteps = torch.full((static["num_noisy_vision_tokens"],), 500.0)
    row = model.denoise_step_row(static, latents, timesteps, static["rotary"], static["rotary"])
    n = static["num_vision_tokens"]
    res.plan(CFG_BATCHED_LABEL, [(COND_LABEL, n), (UNCOND_LABEL, n)], causal=False)
    with torch.no_grad(), torch.compiler.set_stance("fail_on_recompile"):
        ((cond_v, uncond_v),) = step([row], CFG_BATCHED_LABEL, res.attn)
    assert cond_v.shape == uncond_v.shape == latents.shape


def test_warmup_compiles_is_a_no_op_without_the_compiled_layer_stack(monkeypatch):
    sub, model, _, _ = _tiny(compile_denoise=True, compile_denoise_batched=False)
    monkeypatch.setattr(model, "denoise_step_batched", lambda *a: pytest.fail("no compile to warm"))
    sub.warmup_compiles()


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
