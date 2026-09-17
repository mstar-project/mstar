"""CPU checks for the 4-step distilled Super checkpoints
(``nvidia/Cosmos3-Super-{Text2Image,Image2Video}-4Step``): the distilled
sampler config is read from ``modular_model_index.json``, requests are pinned
to the fixed schedule with guidance baked in, and the denoise loop runs the
FlowMatchEuler SDE step (seedable re-noising) with the i2v anchor re-pinned
after every step. No weights: the transformer is faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import IMAGE_GEN_LOOP, Cosmos3DiTSubmodule

SIGMAS = [1.0, 0.9375, 0.8333333333333334, 0.625]

# The shipped distilled scheduler config (Cosmos3-Super-Text2Image-4Step).
SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.39.0",
    "base_image_seq_len": 256, "base_shift": 0.5,
    "fixed_step_requires_explicit_sigmas": True,
    "fixed_step_sampler_config": {"sample_type": "sde", "t_list": SIGMAS},
    "invert_sigmas": False, "max_image_seq_len": 4096, "max_shift": 1.15,
    "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "stochastic_sampling": True, "time_shift_type": "exponential",
    "use_beta_sigmas": False, "use_dynamic_shifting": False,
    "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


def _fake_distilled_dir(tmp_path: Path) -> Path:
    root = tmp_path / "super4"
    (root / "transformer").mkdir(parents=True)
    (root / "scheduler").mkdir()
    # A Nano-shaped transformer config is enough: the sampler is what is under test.
    (root / "transformer" / "config.json").write_text(json.dumps({"_class_name": "Cosmos3OmniTransformer"}))
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(SCHEDULER))
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "Cosmos3OmniPipeline", "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }))
    (root / "modular_model_index.json").write_text(json.dumps({
        "_class_name": "Cosmos3DistilledModularPipeline", "is_distilled": True, "distilled_sigmas": SIGMAS,
    }))
    return root


def test_config_reads_distilled_sampler(tmp_path) -> None:
    cfg = Cosmos3Config.from_pretrained(_fake_distilled_dir(tmp_path))
    assert cfg.is_distilled and cfg.distilled_sigmas == tuple(SIGMAS)
    assert cfg.scheduler.scheduler_class == "FlowMatchEulerDiscreteScheduler"
    assert cfg.scheduler.stochastic_sampling and cfg.scheduler.num_train_timesteps == 1000
    # Base checkpoints keep the UniPC defaults.
    base = Cosmos3Config()
    assert not base.is_distilled and base.distilled_sigmas is None
    assert base.scheduler.scheduler_class == "UniPCMultistepScheduler"


def test_distilled_request_resolution(tmp_path) -> None:
    """Steps and guidance are fixed by the checkpoint; the other modes are
    rejected; plain t2i / i2v resolve."""
    import diffusers

    model = Cosmos3Model(model_path_hf=str(_fake_distilled_dir(tmp_path)), skip_weight_loading=True)
    assert model._scheduler_class() is diffusers.FlowMatchEulerDiscreteScheduler
    p = model._resolve_gen_params({"size": "512x512"}, ["text"], ["image"])
    assert p["num_inference_steps"] == 4 and p["guidance_scale"] == 1.0
    v = model._resolve_gen_params({"num_frames": 33}, ["image", "text"], ["video"])
    assert v["num_inference_steps"] == 4 and v["guidance_scale"] == 1.0 and v["has_image_condition"]
    # Explicit matching values pass; anything else is a request error.
    ok = model._resolve_gen_params({"num_inference_steps": 4, "guidance_scale": 1.0}, ["text"], ["image"])
    assert ok["num_inference_steps"] == 4
    for bad, mods in (
        ({"num_inference_steps": 8}, ["text"]),
        ({"guidance_scale": 6.0}, ["text"]),
        ({"generate_sound": True, "num_frames": 33}, ["text"]),
        ({"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10}, ["image", "text"]),
    ):
        out_mods = ["video"] if "num_frames" in bad or "action_mode" in bad else ["image"]
        with pytest.raises(ValueError, match="distilled"):
            model._resolve_gen_params(bad, mods, out_mods)
    with pytest.raises(ValueError, match="distilled"):
        model._resolve_gen_params({"num_frames": 33}, ["video", "text"], ["video"])
    model.config.enable_windowed_video = True
    with pytest.raises(ValueError, match="distilled"):
        model._resolve_gen_params({"num_frames": 61, "window_mode": "kv"}, ["text"], ["video"])
    # The base model is untouched by the distilled rules.
    base = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    assert base._resolve_gen_params({"num_inference_steps": 8}, ["text"], ["image"])["num_inference_steps"] == 8


def _distilled_dit(tmp_path):
    cfg = Cosmos3Config.from_pretrained(_fake_distilled_dir(tmp_path))
    sub = Cosmos3DiTSubmodule(transformer=None, config=cfg, scheduler=None)
    sub.transformer = SimpleNamespace(proj_in=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)))
    return sub


def test_distilled_scheduler_and_sde_step(tmp_path) -> None:
    """The per-request scheduler is FlowMatchEuler over the fixed sigmas
    (timesteps = sigma x 1000, a trailing zero sigma), and one step is the
    reference SDE update drawn from the request's generator."""
    sub = _distilled_dit(tmp_path)
    sched = sub._new_scheduler(4, torch.device("cpu"), flow_shift=12.0, use_karras_sigma=True)
    assert type(sched).__name__ == "FlowMatchEulerDiscreteScheduler"
    assert torch.allclose(sched.timesteps.float(), torch.tensor(SIGMAS) * 1000)
    assert torch.allclose(sched.sigmas.float(), torch.tensor(SIGMAS + [0.0]))
    assert sched.config.stochastic_sampling

    gen = torch.Generator().manual_seed(7)
    st = sub.request_state("r")
    st.add_all(scheduler=sched, sde_generator=gen)
    x = torch.randn(16, 3, 4, 4)
    v = torch.randn_like(x)
    out = sub._scheduler_step(st, v, sched.timesteps[0], x)
    eps = torch.randn((1, *x.shape), generator=torch.Generator().manual_seed(7))[0]
    ref = (1.0 - SIGMAS[1]) * (x - SIGMAS[0] * v) + SIGMAS[1] * eps
    assert torch.allclose(out, ref, atol=1e-6)
    # The last step lands on x0 exactly (sigma' = 0).
    for _ in range(2):
        sched.step(v.unsqueeze(0), sched.timesteps[sched.step_index], out.unsqueeze(0), generator=gen)
    last = sub._scheduler_step(st, v, sched.timesteps[3], x)
    assert torch.allclose(last, x - SIGMAS[3] * v, atol=1e-6)


def test_distilled_i2v_loop_repins_anchor(tmp_path, monkeypatch) -> None:
    """Driven through prepare_inputs / forward for the four steps with a fake
    velocity: the SDE re-noises every frame, and frame 0 is re-pinned to the
    conditioning anchor after each step; the loop stops after step 4."""
    sub = _distilled_dit(tmp_path)
    monkeypatch.setattr(sub, "get_device", lambda: torch.device("cpu"))
    latent_shape = (1, 16, 3, 4, 4)
    anchor = torch.full(latent_shape, 0.25)
    n_tokens = 3 * 4  # 3 latent frames x (4/2 x 4/2) patches
    st = sub.request_state("r")
    st.add_all(
        cond={"num_vision_tokens": n_tokens, "num_noisy_vision_tokens": n_tokens - 4},
        uncond=None, gs=1.0, guidance_interval=None,
        scheduler=sub._new_scheduler(4, torch.device("cpu")), latent_shape=latent_shape, num_sound=None,
    )
    calls = []

    def fake_denoise(attn, static, latents, vision_timesteps, label):
        calls.append(float(vision_timesteps[0]))
        vel = torch.full_like(latents, 0.5)
        vel[:, :, 0] = 0.0  # zero velocity on the clean anchor, as the transformer's unpatchify yields
        return vel

    monkeypatch.setattr(sub, "_denoise", fake_denoise)
    fwd = SimpleNamespace(request_id="r", random_seed=3, graph_walk="video_gen")
    ei = SimpleNamespace(request_ids=["r"], per_request_states=None, resources={"attn": "paged"}, step=None)
    inputs = {"cond_latents": [anchor]}
    for step in range(4):
        ni = sub.prepare_inputs("video_gen", fwd, inputs)
        assert ni is not None and not ni.resource_step_info.cfg
        out = sub.forward("video_gen", ei, **sub.preprocess("video_gen", ei, [ni]))
        lat = out["latents"][0]
        assert torch.equal(lat[:, :, 0], anchor[:, :, 0].to(lat.dtype)), f"anchor drifted at step {step}"
        assert not torch.equal(lat[:, :, 1], anchor[:, :, 1])
        inputs = {"latents": [lat], "time_index": [out["time_index"][0]]}
    assert calls == [1000.0, 937.5, pytest.approx(833.333, abs=0.01), 625.0]
    assert sub.prepare_inputs("video_gen", fwd, inputs) is None  # the loop's extra dispatch is vetoed
    info = SimpleNamespace(graph_walk="video_gen", dynamic_loop_iter_counts={"video_gen_loop": 3})
    assert sub.check_stop("r", info, {}) == {"video_gen_loop"}
    assert IMAGE_GEN_LOOP  # keep the import honest for the t2i loop name
    # Deterministic: the same seed reproduces the rollout.
    st2 = sub.request_state("r2")
    st2.add_all(**{k: st[k] for k in ("cond", "uncond", "gs", "guidance_interval", "latent_shape", "num_sound")},
                scheduler=sub._new_scheduler(4, torch.device("cpu")))
    fwd2 = SimpleNamespace(request_id="r2", random_seed=3, graph_walk="video_gen")
    ni = sub.prepare_inputs("video_gen", fwd2, {"cond_latents": [anchor]})
    ei2 = SimpleNamespace(request_ids=["r2"], per_request_states=None, resources={"attn": "paged"}, step=None)
    first2 = sub.forward("video_gen", ei2, **sub.preprocess("video_gen", ei2, [ni]))["latents"][0]
    st3 = sub.request_state("r3")
    st3.add_all(**{k: st[k] for k in ("cond", "uncond", "gs", "guidance_interval", "latent_shape", "num_sound")},
                scheduler=sub._new_scheduler(4, torch.device("cpu")))
    fwd3 = SimpleNamespace(request_id="r3", random_seed=3, graph_walk="video_gen")
    ni = sub.prepare_inputs("video_gen", fwd3, {"cond_latents": [anchor]})
    ei3 = SimpleNamespace(request_ids=["r3"], per_request_states=None, resources={"attn": "paged"}, step=None)
    first3 = sub.forward("video_gen", ei3, **sub.preprocess("video_gen", ei3, [ni]))["latents"][0]
    assert torch.equal(first2, first3)
