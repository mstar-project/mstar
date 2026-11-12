"""Wan2.2's DMD sampler (``sampler: dmd``, FastWan2.2-TI2V-5B) against FastVideo's recipe, on the CPU.

The oracle is FastVideo's ``DmdDenoisingStage`` loop (fastvideo/pipelines/basic/wan/stages/dmd.py),
with its ``pred_noise_to_pred_video`` and ``FlowMatchEulerDiscreteScheduler.add_noise`` copied below
and its sigma table taken from diffusers' scheduler of the same construction. A stand-in transformer
replaces the DiT, so the sampler, the seeding and the loop wiring are what is compared, bit for bit.
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.dmd import dmd_schedule, dmd_step
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.config import SAMPLER_DMD, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, DMD_NOISE, Wan22DmdDitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model

FASTWAN_TIMESTEPS = (1000, 757, 522)
# 64x64 pixels, 5 frames: latent [48, 2, 4, 4]
STEP_METADATA = {"num_inference_steps": 3, "guidance_scale": 1.0, "height": 64, "width": 64, "num_frames": 5}
LATENT_TCHW = (2, 48, 4, 4)


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


# --- FastVideo's step math, verbatim (fastvideo/models/utils.py, scheduling_flow_match_euler_discrete.py) ---

def _pred_noise_to_pred_video(pred_noise, noise_input_latent, timestep, scheduler):
    timestep = timestep.flatten(0, 1) if timestep.ndim == 2 else timestep.expand(noise_input_latent.shape[0])
    dtype = pred_noise.dtype
    pred_noise = pred_noise.double()
    noise_input_latent = noise_input_latent.double()
    sigmas, timesteps = scheduler.sigmas.double(), scheduler.timesteps.double()
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    return (noise_input_latent - sigma_t * pred_noise).to(dtype)


def _add_noise(scheduler, clean_latent, noise, timestep):
    timestep = timestep.expand(clean_latent.shape[0])
    timestep_id = torch.argmin((scheduler.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma = scheduler.sigmas[timestep_id].reshape(-1, 1, 1, 1)
    return ((1 - sigma) * clean_latent + sigma * noise).type_as(noise)


def _scheduler():
    diffusers = pytest.importorskip("diffusers")
    # FastVideo's DMD table: FlowMatchEulerDiscreteScheduler(shift=DMD_TRAINING_NOISE_SHIFT=8.0)
    return diffusers.FlowMatchEulerDiscreteScheduler(shift=8.0)


class _StandIn(torch.nn.Module):
    """A DiT stand-in whose output depends on its input and timestep, in the DiT's dtype."""

    dtype = torch.bfloat16

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        per_row = timestep.float().reshape(hidden_states.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1, 1)
        return (hidden_states.float() * 0.3 - 0.1 + per_row / 1000).to(torch.bfloat16)


def _fastvideo_dmd(seed: int) -> list[torch.Tensor]:
    """FastVideo's loop over the stand-in: its latents ``[1, T, C, H, W]`` after each step."""
    scheduler = _scheduler()
    model = _StandIn()
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, *LATENT_TCHW), generator=generator, dtype=torch.float32)
    timesteps = torch.tensor(FASTWAN_TIMESTEPS, dtype=torch.long)
    out = []
    for i, t in enumerate(timesteps):
        noise_latents = latents.clone()
        t_expand = t.repeat(1)
        pred_noise = model(latents.to(torch.bfloat16).permute(0, 2, 1, 3, 4), t_expand, None).permute(0, 2, 1, 3, 4)
        pred_video = _pred_noise_to_pred_video(
            pred_noise.flatten(0, 1), noise_latents.flatten(0, 1), t_expand[:, None].expand(-1, pred_noise.shape[1]),
            scheduler,
        ).unflatten(0, pred_noise.shape[:2])
        if i < len(timesteps) - 1:
            noise = torch.randn(latents.shape, dtype=pred_video.dtype, generator=generator)
            next_timestep = timesteps[i + 1] * torch.ones([1], dtype=torch.long)
            latents = _add_noise(
                scheduler, pred_video.flatten(0, 1), noise.flatten(0, 1), next_timestep,
            ).unflatten(0, pred_video.shape[:2])
        else:
            latents = pred_video
        out.append(latents)
    return out


# --- the schedule and the step ---

def test_dmd_schedule_is_fastvideos_table_lookup():
    scheduler = _scheduler()
    schedule = dmd_schedule(FASTWAN_TIMESTEPS)
    expected = [
        scheduler.sigmas[torch.argmin((scheduler.timesteps.double() - t).abs())] for t in FASTWAN_TIMESTEPS
    ]
    assert torch.equal(schedule.sigmas, torch.stack([*expected, torch.tensor(0.0)]))
    # the model is conditioned on the raw timesteps, not the table's
    assert torch.equal(schedule.timesteps, torch.tensor([1000.0, 757.0, 522.0]))
    assert schedule.sigmas[1].item() == pytest.approx(0.756757, abs=1e-6)


def test_dmd_step_matches_fastvideo_bitwise():
    scheduler = _scheduler()
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn((1, *LATENT_TCHW), generator=generator)
    flow = torch.randn(sample.shape, generator=generator).to(torch.bfloat16)
    noise = torch.randn(sample.shape, generator=generator).to(torch.bfloat16)
    schedule = dmd_schedule(FASTWAN_TIMESTEPS)
    t, t_next = torch.tensor([757]), torch.tensor([522])

    x0 = _pred_noise_to_pred_video(flow.flatten(0, 1), sample.flatten(0, 1), t, scheduler)
    expected = _add_noise(scheduler, x0, noise.flatten(0, 1), t_next).unflatten(0, sample.shape[:2])
    sigma, sigma_next = schedule.sigmas[1:2].view(1, 1), schedule.sigmas[2:3].view(1, 1)
    got = dmd_step(sample, flow, sigma, sigma_next, noise)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)
    # the last step returns the clean prediction
    assert torch.equal(dmd_step(sample, flow, sigma, sigma_next, None), x0.unflatten(0, sample.shape[:2]))


# --- the dit submodule, driven as the engine does ---

def _dit() -> Wan22DmdDitSubmodule:
    config = Wan22Config(sampler=SAMPLER_DMD, compile_dit=False)
    submodule = Wan22DmdDitSubmodule(_StandIn(), config)
    submodule.get_device = lambda: torch.device("cpu")
    return submodule


def _info(rid: str, seed: int, step: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=Wan22Model.VIDEO_GEN_WALK, fwd_index=0, random_seed=seed, max_tokens=0,
        step_metadata=dict(STEP_METADATA), dynamic_loop_iter_counts={DENOISE_LOOP_NAME: step},
    )


_TEXT = {"text_embeds_pos": [torch.zeros(1, 8, 16)], "text_embeds_neg": [torch.zeros(1)]}


def _run(dit, rows: list[tuple[CurrentForwardPassInfo, dict]]) -> dict:
    """One batched step over ``rows`` of (request info, carried edges): prepare_inputs -> preprocess ->
    forward_batched."""
    infos = [info for info, _ in rows]
    node_inputs = [
        dit.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, info, {**_TEXT, **carried}) for info, carried in rows
    ]
    engine_inputs = ModelInputsFromEngine(
        request_ids=[i.request_id for i in infos], per_request_info={i.request_id: i for i in infos},
    )
    stacked = dit.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, node_inputs)
    return dit.forward_batched(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **stacked)


def test_seeding_is_fastvideos_draw_order_and_layout():
    dit = _dit()
    seeds = dit.seed_loop_back(dit.bucket_key_for(_info("r", 7, 0)), torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn((1, *LATENT_TCHW), generator=generator, dtype=torch.float32)
    noise = [torch.randn((1, *LATENT_TCHW), generator=generator, dtype=torch.bfloat16) for _ in range(2)]
    assert torch.equal(seeds["latents"], latents.transpose(1, 2))
    assert seeds[DMD_NOISE].dtype == torch.bfloat16
    assert torch.equal(seeds[DMD_NOISE], torch.cat(noise).transpose(1, 2))


def test_three_step_loop_matches_fastvideo_bitwise():
    dit = _dit()
    reference = _fastvideo_dmd(seed=11)
    carried = {}
    for step in range(3):
        out = _run(dit, [(_info("r", 11, step), carried)])["r"]
        carried = {name: out[name] for name in ("latents", DMD_NOISE)}
        assert torch.equal(out["latents"][0], reference[step].transpose(1, 2).float()), f"step {step}"


def test_rows_at_different_steps_batch_like_alone():
    dit = _dit()
    # request a at step 1 (carrying its step-0 output), request b at step 0
    a0 = _run(dit, [(_info("a", 1, 0), {})])["a"]
    carried_a = {name: a0[name] for name in ("latents", DMD_NOISE)}
    alone_a = _run(dit, [(_info("a", 1, 1), carried_a)])["a"]["latents"][0]
    alone_b = _run(dit, [(_info("b", 2, 0), {})])["b"]["latents"][0]
    both = _run(dit, [(_info("a", 1, 1), carried_a), (_info("b", 2, 0), {})])
    assert torch.equal(both["a"]["latents"][0], alone_a)
    assert torch.equal(both["b"]["latents"][0], alone_b)


# --- the model's request seam and graph ---

def _model(**kwargs) -> Wan22Model:
    return Wan22Model(model_path_hf="test/wan22", skip_weight_loading=True, sampler=SAMPLER_DMD, **kwargs)


def _plan(model, model_kwargs, input_modalities=("text",), signals=None):
    return model.walk_plan(
        "default", list(input_modalities), ["video"], signals or {"text_inputs": [object()]}, model_kwargs,
    )


def test_dmd_recipe_fixes_steps_and_guidance():
    model = _model()
    assert model.get_loop_caps() == {DENOISE_LOOP_NAME: 3}
    (_, video) = _plan(model, {})
    assert video.step_metadata["num_inference_steps"] == 3
    assert video.step_metadata["guidance_scale"] == 1.0
    _plan(model, {"num_inference_steps": 3, "guidance_scale": 1.0})


@pytest.mark.parametrize("kwargs,fragment", [
    ({"num_inference_steps": 50}, "in 3 steps"),
    ({"guidance_scale": 5.0}, "without CFG"),
])
def test_dmd_rejects_other_steps_and_cfg(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        _plan(_model(), kwargs)


def test_dmd_rejects_image_input():
    model = _model()
    with pytest.raises(ValueError, match="text-to-video only"):
        image = torch.zeros(3, 704, 1280)
        model.process_prompt("a cat", ["text", "image"], ["video"], tensors={"image_inputs": [image]})
    with pytest.raises(ValueError, match="text-to-video only"):
        _plan(model, {}, ("text", "image"), {"text_inputs": [object()], "image_inputs": [object()]})


@pytest.mark.parametrize("sampler,loop_back", [
    ("unipc", {"latents", "unipc_model_outputs", "unipc_last_sample"}),
    (SAMPLER_DMD, {"latents", DMD_NOISE}),
])
def test_loop_back_edges_follow_the_sampler(sampler, loop_back):
    model = Wan22Model(model_path_hf="test/wan22", skip_weight_loading=True, sampler=sampler)
    loop = model.get_graph_walk_graphs()[Wan22Model.VIDEO_GEN_WALK].sections[0]
    assert {edge.name for edge in loop.section.outputs} == loop_back
    assert loop.section.input_names == {"text_embeds_pos", "text_embeds_neg"} | loop_back


def test_fastwan_yaml_builds_its_graphs():
    config = str(Path(__file__).resolve().parents[2] / "configs" / "wan22_fastwan.yaml")
    graphs = _model().get_worker_graphs(config)
    assert {walk for graph in graphs for walk in graph.graph_walks} >= {Wan22Model.VIDEO_GEN_WALK}


@pytest.mark.parametrize("kwargs", [
    {"sampler": "euler"},
    {"sampler": SAMPLER_DMD, "dmd_timesteps": (757, 1000)},
    {"sampler": SAMPLER_DMD, "dmd_timesteps": ()},
    {"sampler": SAMPLER_DMD, "dmd_timesteps": (1000, 0)},
    {"sampler": SAMPLER_DMD, "dmd_shift": 0},
])
def test_bad_sampler_settings_fail_at_startup(kwargs):
    with pytest.raises(ValueError):
        Wan22Config(**kwargs)


def test_dmd_checkpoint_refuses_unipc(tmp_path):
    pytest.importorskip("diffusers")
    (tmp_path / "model_index.json").write_text('{"_class_name": "WanDMDPipeline"}')
    Wan22Model(model_path_hf=str(tmp_path), sampler=SAMPLER_DMD)._check_sampler_fits_checkpoint()
    with pytest.raises(ValueError, match="sampler: dmd"):
        Wan22Model(model_path_hf=str(tmp_path))._check_sampler_fits_checkpoint()
