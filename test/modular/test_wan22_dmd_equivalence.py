"""FastWan2.2-TI2V-5B on M*'s DMD sampler vs FastVideo's DMD loop, on the GPU with real weights.

The reference is FastVideo's ``DmdDenoisingStage`` (fastvideo/pipelines/basic/wan/stages/dmd.py)
run over diffusers' ``WanTransformer3DModel`` from the same checkpoint, with FastVideo's CPU-generator
noise; ``test_wan22_dmd.py`` pins its step math on the CPU. Both sides share one UMT5 embedding and run
SDPA, so with the time embedding at the reference's row count (one row per sample, as FastVideo passes
the timestep) every step must match exactly.

Skips without CUDA, diffusers or the checkpoint in the HF cache. Run with the numerics pins of
test_wan22_reference_equivalence.py:

    CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        pytest test/modular/test_wan22_dmd_equivalence.py -v -s
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.dmd import dmd_schedule
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.config import SAMPLER_DMD
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, DMD_NOISE
from mstar.model.wan22.wan22_model import Wan22Model

MODEL_REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
PROMPT = "A cat walks on the grass, realistic style."
SEED = 42
# a short clip keeps the two 5B DiTs' forwards cheap; the sampler does not depend on the size
HEIGHT, WIDTH, FRAMES = 704, 1280, 21
# one batch-1 forward per step on both sides, time embedding at the same row count: nothing left to round apart
STEP_MAX_ABS = 0.0

DEVICE = torch.device("cuda")


def _cached() -> bool:
    root = os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache" / "huggingface" / "hub")
    return (Path(root) / f"models--{MODEL_REPO.replace('/', '--')}").exists()


pytest.importorskip("diffusers", reason="diffusers not installed")
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not _cached(), reason=f"{MODEL_REPO} not in the local HF cache"),
]


def _fastvideo_dmd(transformer, prompt_embeds, latent_tchw, timesteps, seed) -> list[torch.Tensor]:
    """FastVideo's loop: latents ``[1, T, C, H, W]`` after each step (its sigma lookups via dmd_schedule,
    whose table test_wan22_dmd.py holds to the reference's)."""
    sigmas = dmd_schedule(timesteps).sigmas.to(DEVICE)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, *latent_tchw), generator=generator, dtype=torch.float32).to(DEVICE)
    out = []
    for i, t in enumerate(timesteps):
        noise_latents = latents.clone()
        t_expand = torch.tensor([t], dtype=torch.long, device=DEVICE)
        pred_noise = transformer(
            hidden_states=latents.to(torch.bfloat16).permute(0, 2, 1, 3, 4), timestep=t_expand,
            encoder_hidden_states=prompt_embeds, return_dict=False,
        )[0].permute(0, 2, 1, 3, 4)
        pred_video = (noise_latents.double() - sigmas[i].double() * pred_noise.double()).to(pred_noise.dtype)
        if i < len(timesteps) - 1:
            noise = torch.randn(latents.shape, dtype=pred_video.dtype, generator=generator).to(DEVICE)
            sigma = sigmas[i + 1].reshape(1, 1, 1, 1, 1)
            latents = ((1 - sigma) * pred_video + sigma * noise).type_as(noise)
        else:
            latents = pred_video
        out.append(latents)
    return out


@contextmanager
def _per_sample_rows(transformer, frames: int):
    """Run M*'s time embedding on one row per sample, as the reference does, and hand the blocks
    one row per frame as usual."""
    embedder = transformer.condition_embedder
    per_frame_rows = embedder.forward

    def per_sample_rows(timestep, encoder_hidden_states):
        temb, proj, text = per_frame_rows(timestep[:, :1], encoder_hidden_states)
        return temb.expand(-1, frames, *temb.shape[2:]), proj.expand(-1, frames, *proj.shape[2:]), text

    embedder.forward = per_sample_rows
    try:
        yield
    finally:
        del embedder.forward


@pytest.fixture(scope="module")
def prompt_embeds() -> torch.Tensor:
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(MODEL_REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(DEVICE)
    with torch.no_grad():
        embeds = pipe._get_t5_prompt_embeds(PROMPT, max_sequence_length=512, device=DEVICE)
    del pipe
    torch.cuda.empty_cache()
    return embeds


def test_dmd_trajectory_matches_fastvideo(prompt_embeds):
    from diffusers import WanTransformer3DModel

    model = Wan22Model(model_path_hf=MODEL_REPO, sampler=SAMPLER_DMD)
    model.config.compile_dit = False
    dit = model.get_submodule("dit", device=str(DEVICE)).to(DEVICE)
    timesteps = model.config.dmd_timesteps
    lat_t, lat_h, lat_w = (FRAMES - 1) // 4 + 1, HEIGHT // 16, WIDTH // 16

    reference = WanTransformer3DModel.from_pretrained(
        MODEL_REPO, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    with torch.no_grad():
        expected = _fastvideo_dmd(reference, prompt_embeds, (lat_t, 48, lat_h, lat_w), timesteps, SEED)
    del reference
    torch.cuda.empty_cache()

    info = CurrentForwardPassInfo(
        request_id="dmd", graph_walk=Wan22Model.VIDEO_GEN_WALK, fwd_index=0, random_seed=SEED, max_tokens=0,
        step_metadata={
            "num_inference_steps": len(timesteps), "guidance_scale": 1.0,
            "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES,
        },
    )
    engine_inputs = ModelInputsFromEngine(request_ids=["dmd"], per_request_info={"dmd": info})
    text = {"text_embeds_pos": [prompt_embeds[0:1]], "text_embeds_neg": [prompt_embeds.new_zeros(1)]}
    carried: dict[str, list[torch.Tensor]] = {}
    diffs = []
    with torch.no_grad(), _per_sample_rows(dit.transformer, lat_t):
        for step in range(len(timesteps)):
            info.dynamic_loop_iter_counts[DENOISE_LOOP_NAME] = step
            node_inputs = dit.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, info, {**text, **carried})
            stacked = dit.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, [node_inputs])
            out = dit.forward(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **stacked)
            carried = {name: out[name] for name in ("latents", DMD_NOISE)}
            ours = out["latents"][0].float()
            diffs.append((ours - expected[step].transpose(1, 2).float()).abs().max().item())
    print(f"DMD per-step max|diff| vs FastVideo's loop: {diffs}")
    assert max(diffs) <= STEP_MAX_ABS, diffs
