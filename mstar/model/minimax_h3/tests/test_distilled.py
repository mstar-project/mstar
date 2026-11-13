"""A few-step distilled H3 checkpoint (FastVideo's FastH3): its rung schedule, the step-0
timestep table it needs, the release's contract and the requests it serves. CPU, no weights."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mstar.model.minimax_h3.components.layout import H3Layout
from mstar.model.minimax_h3.config import DENOISE_LOOP, H3Config, H3Distilled
from mstar.model.minimax_h3.minimax_h3_model import GENERATE_AV_WALK, H3Model
from mstar.model.minimax_h3.schedule import ScheduleArgs, h3_schedule, rung_sigmas
from mstar.model.minimax_h3.submodules import ONE_TIMESTEP, TEXT_EMBEDS, H3DenoiseSubmodule
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG, write_distilled_release

# FastVideo-FastH3-8-Step-V2's fastvideo_inference.json
RUNGS = (999, 874, 749, 624, 500, 375, 250, 125)


def test_rung_grid_is_fastvideos_fp32_grid():
    """FastVideo's ``_set_shifted_schedule`` (and SGLang's ``minimax_h3_time_shift_sigmas``) shift
    ``rung / 1000`` in fp32; shifting in fp64 first (diffusers' ``shift_provided_sigmas``) lands
    an ulp away on some rungs, so the values are pinned."""
    video = torch.tensor([0.9998998641967773, 0.9857883453369141, 0.9675752520561218, 0.9431681036949158,
                          0.9090909361839294, 0.8571428656578064, 0.7692307829856873, 0.5882353186607361, 0.0])
    audio = torch.tensor([0.9996663928031921, 0.9541484117507935, 0.8995195627212524, 0.8327401280403137, 0.75,
                          0.6428571343421936, 0.5, 0.30000001192092896, 0.0])
    torch.testing.assert_close(rung_sigmas(RUNGS, 10.0), video, rtol=0, atol=0)
    torch.testing.assert_close(rung_sigmas(RUNGS, 3.0), audio, rtol=0, atol=0)


def test_h3_schedule_on_rungs():
    s = h3_schedule(len(RUNGS) + 1, 10.0, 3.0, RUNGS)
    assert s.num_steps == 8 and s.sigmas.shape[0] == s.audio_sigmas.shape[0] == 9
    assert torch.equal(s.timesteps, 1.0 - s.sigmas[:-1]) and torch.equal(s.audio_timesteps, 1.0 - s.audio_sigmas[:-1])
    # the two modalities start apart: t = 1.0e-4 and 3.3e-4
    assert float(s.timesteps[0]) != float(s.audio_timesteps[0])
    with pytest.raises(ValueError, match="must be 9"):
        h3_schedule(50, 10.0, 3.0, RUNGS)
    for bad in ((), (999, 999), (1001,), (500, 600), (999.0,), (True,)):
        with pytest.raises(ValueError, match="rungs"):
            rung_sigmas(bad, 10.0)


class _StubDiT:
    dtype = torch.float32
    blocks = ()

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, layout, **_):
        self.calls.append(inputs)
        return torch.zeros_like(inputs.video), torch.zeros_like(inputs.audio)


def test_step_0_embeds_one_timestep_only_where_both_modalities_start_on_it():
    """Eager step 0 embeds one timestep when the rows share it (the linspace grid, t = 0), and the
    (video, audio, keyframe) table where a rung grid starts the two apart."""
    cfg = H3Config().transformer
    layout = H3Layout(text_len=3, latent_frames=2, latent_height=4, latent_width=4, audio_latents=2)
    stub = _StubDiT()
    node = H3DenoiseSubmodule(stub, H3Config(), loop_name=DENOISE_LOOP)
    video = torch.zeros(1, layout.num_video_rows, cfg.patch_dim)
    audio = torch.zeros(1, layout.num_audio_rows, cfg.audio_latents_dim)
    for one in (True, False):
        node.denoise(
            None, layout, video, torch.tensor([1e-4]), torch.tensor([0.9]), torch.tensor([0.9]), audio,
            torch.tensor([3e-4]), torch.tensor([0.9]), torch.tensor([0.9]), torch.tensor([0.999]),
            torch.zeros(1, 3, cfg.hidden_size), torch.tensor(True), torch.tensor(one),
        )
    assert [tuple(call.timesteps.shape) for call in stub.calls] == [(1, 1), (1, 3)]
    torch.testing.assert_close(stub.calls[1].timesteps, torch.tensor([[1e-4, 3e-4, 0.999]]), rtol=0, atol=0)


def test_the_release_contract_is_read_and_checked(tmp_path):
    write_distilled_release(tmp_path, CFG)
    distilled = H3Distilled.from_dir(tmp_path, CFG)
    assert distilled == H3Distilled(tmp_path / "transformer", RUNGS, 10.0, 3.0)
    config = H3Config(transformer=CFG).with_distilled(tmp_path)
    assert (config.video_shift, config.audio_shift, config.dit_source) == (10.0, 3.0, tmp_path / "transformer")
    for contract, match in (({"task": "fl2va"}, "t2av"), ({"guidance_scale": 3.5}, "CFG"),
                            ({"transformer_forwards": 4}, "forwards"), ({"video_scheduler_shift": 12.0}, "shift")):
        write_distilled_release(tmp_path, CFG, **contract)
        with pytest.raises(ValueError, match=match):
            H3Distilled.from_dir(tmp_path, CFG)
    write_distilled_release(tmp_path, CFG)
    with pytest.raises(ValueError, match="architecture"):
        H3Distilled.from_dir(tmp_path, replace(CFG, num_layers=3))


def _distilled_model() -> H3Model:
    model = H3Model(model_path_hf="test/minimax_h3", skip_weight_loading=True)
    model.set_config(H3Config(video_shift=10.0, audio_shift=3.0, distilled=H3Distilled(None, RUNGS, 10.0, 3.0)))
    return model


def test_requests_run_the_distilled_recipe_or_are_refused():
    model = _distilled_model()
    assert model.resolve_schedule({}) == ScheduleArgs(9, 10.0, 3.0, RUNGS)
    assert model.get_loop_caps() == {DENOISE_LOOP: 8}
    assert model.resolve_request({})["num_inference_steps"] == 9
    for kwargs in ({"num_inference_steps": 50}, {"num_inference_steps": 8}, {"flow_shift": 12.0},
                   {"audio_flow_shift": 2.0}):
        with pytest.raises(ValueError, match="distilled for 8 steps"):
            model.resolve_schedule(kwargs)
    assert model.max_input_files()["image"] == 0  # no keyframes: only t2av was distilled
    with pytest.raises(ValueError, match="step_cache"):
        H3Model(model_path_hf="test/minimax_h3", skip_weight_loading=True, distilled_checkpoint="x",
                step_cache={"threshold": 0.01})


@pytest.mark.parametrize("distilled", [False, True])
def test_step_0_is_one_timestep_on_the_linspace_grid_and_two_on_the_rungs(distilled):
    model = _distilled_model() if distilled else H3Model(model_path_hf="test/minimax_h3", skip_weight_loading=True)
    if not distilled:
        model.set_config(H3Config())
    node = H3DenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP)
    request = model.resolve_request({"height": 256, "width": 448})
    layout = model.request_layout(request, 5)
    meta = {**request, "text_len": 5, "latent_frames": layout.latent_frames, "latent_height": layout.latent_height,
            "latent_width": layout.latent_width, "audio_latents": layout.audio_latents}
    inputs = {TEXT_EMBEDS: [torch.zeros(5, 8)]}
    for k, want in ((0, not distilled), (1, False)):
        info = SimpleNamespace(graph_walk=GENERATE_AV_WALK, step_metadata=meta,
                               dynamic_loop_iter_counts={DENOISE_LOOP: k})
        assert bool(node.request_inputs(info, inputs, node.bucket_key_for(info))[ONE_TIMESTEP]) is want
