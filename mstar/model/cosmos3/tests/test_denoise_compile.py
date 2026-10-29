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


class _RecordingTransformer(_Transformer):
    """Records what the batched step is handed; a real layer stack would see
    the attention through ``_gen_attend``."""

    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Linear(2, 2)
        self.batched_calls = []

    # what the real step indexes per request; a warmup row missing one would raise here as it did on a server
    REQ_KEYS = ("latents", "vision_timesteps", "rotary_cond", "rotary_uncond", "vision_token_shapes",
                "vision_noisy_frame_indexes", "vision_mse_loss_indexes")

    def denoise_step_batched(self, requests, label, attn):
        for req in requests:
            for key in self.REQ_KEYS:
                req[key]
        self.batched_calls.append((requests, label, attn))
        return [(None, None)] * len(requests)


@pytest.mark.parametrize("batched", [True, False])
def test_warmup_compiles_runs_one_two_request_step_at_the_default_size(monkeypatch, batched):
    from mstar.model.cosmos3 import submodules

    cfg = Cosmos3Config(compile_denoise_batched=batched, video_size_default=(832, 480), num_frames_video=121)
    t = _RecordingTransformer()
    sub = Cosmos3DiTSubmodule(t, cfg)
    built = []
    rotary = (torch.zeros(4, 2), torch.zeros(4, 2))

    def fake_static(ids, height, width, num_frames, fps, has_image_condition, device):
        built.append((height, width, num_frames, has_image_condition))
        return {
            "num_noisy_vision_tokens": 4, "rotary": rotary,
            "vision_token_shapes": [(1, 2, 2)], "vision_noisy_frame_indexes": [torch.zeros(1)],
            "mse_gen_indexes": torch.arange(4),
        }

    monkeypatch.setattr(sub, "_build_static", fake_static)
    sub.warmup_compiles()
    if not batched:
        assert t.batched_calls == [] and built == []
        return
    assert built == [(480, 832, 121, False)]
    [(requests, label, attn)] = t.batched_calls
    assert len(requests) == 2 and label == submodules.CFG_BATCHED_LABEL
    assert requests[0]["latents"].shape == sub._latent_shape(480, 832, 121)
    # both branches take the statics' table, as a served window's rows do
    assert requests[0]["rotary_cond"] is rotary and requests[1]["rotary_uncond"] is rotary
    # the stand-in reads no plan and answers with the query's shape
    q = torch.randn(5, 2, 3)
    assert not attn.requires_kv_write and torch.equal(attn.run(q, k=q, v=q), torch.zeros_like(q))


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
