"""Wan2.2-T2V-A14B with LightX2V's 4-step distill LoRAs on M*'s step-distill sampler vs LightX2V's loop, on the GPU.

The reference is LightX2V's ``WanStepDistillScheduler`` loop (test_wan22_step_distill.py pins its math on the
CPU) over diffusers' two experts from the same checkpoint, each with its LoRA folded in by the same static merge
M* serves with, and M*'s noise. Both sides run SDPA with one time-embedding row per sample, so every step must
match exactly.

Skips without CUDA, diffusers, the checkpoint or the LoRAs in the HF cache. Numerics pins as in
test_wan22_reference_equivalence.py:

    CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        pytest test/modular/test_wan22_step_distill_equivalence.py -v -s
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.lora.checkpoint import LoraFile
from mstar.model.components.diffusion.dmd import step_distill_schedule, step_distill_step
from mstar.model.components.diffusion.lora import apply_loras
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.config import SAMPLER_STEP_DISTILL, WAN22_VARIANT_T2V_A14B
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.model.wan22.weight_loader import convert_wan_lora_keys

MODEL_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LORA_REPO = "lightx2v/Wan2.2-Distill-Loras"
LORAS = (
    "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
    "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
)
PROMPT = "A cat walks on the grass, realistic style."
SEED = 42
HEIGHT, WIDTH, FRAMES = 480, 832, 21
STEP_MAX_ABS = 0.0
DEVICE = torch.device("cuda")


def _snapshot(repo: str) -> Path | None:
    root = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    snapshots = sorted((root / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    return snapshots[-1] if snapshots else None


pytest.importorskip("diffusers", reason="diffusers not installed")
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        _snapshot(MODEL_REPO) is None or _snapshot(LORA_REPO) is None,
        reason=f"{MODEL_REPO} or {LORA_REPO} not in the local HF cache",
    ),
]


def test_step_distill_trajectory_matches_lightx2v():
    from diffusers import WanPipeline, WanTransformer3DModel

    # the snapshot directory, so nothing asks the Hub for a revision offline
    repo = str(_snapshot(MODEL_REPO))
    loras = [LoraFile(str(_snapshot(LORA_REPO) / name)) for name in LORAS]
    pipe = WanPipeline.from_pretrained(
        repo, transformer=None, transformer_2=None, vae=None, torch_dtype=torch.bfloat16,
    )
    pipe.text_encoder.to(DEVICE)
    with torch.no_grad():
        embeds = pipe._get_t5_prompt_embeds(PROMPT, max_sequence_length=512, device=DEVICE)
    del pipe

    # LightX2V's loop, one expert resident at a time: high noise for steps 0-1 (boundary_step_index 2)
    timesteps = (1000, 750, 500, 250)
    schedule = step_distill_schedule(timesteps, 5.0)
    shape = (1, 16, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)
    latents = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(SEED), dtype=torch.float32).to(DEVICE)
    expected = []
    for subfolder, lora, steps in (("transformer", loras[0], (0, 1)), ("transformer_2", loras[1], (2, 3))):
        expert = WanTransformer3DModel.from_pretrained(
            repo, subfolder=subfolder, torch_dtype=torch.bfloat16,
        ).to(DEVICE).eval()
        apply_loras(expert, [lora], lambda key: key, convert_keys=convert_wan_lora_keys)
        with torch.no_grad():
            for k in steps:
                flow = expert(
                    hidden_states=latents.to(torch.bfloat16), timestep=schedule.timesteps[k:k + 1].to(DEVICE),
                    encoder_hidden_states=embeds, return_dict=False,
                )[0]
                last = k == len(timesteps) - 1
                latents = step_distill_step(
                    latents, flow, float(schedule.sigmas[k]), None if last else float(schedule.sigmas[k + 1]),
                )
                expected.append(latents)
        del expert
        gc.collect()
        torch.cuda.empty_cache()

    model = Wan22Model(
        model_path_hf=repo, variant=WAN22_VARIANT_T2V_A14B, sampler=SAMPLER_STEP_DISTILL,
        lora=[loras[0].path], lora_2=[loras[1].path],
    )
    model.config.compile_dit = False
    dit = model.get_submodule("dit", device=str(DEVICE)).to(DEVICE)
    info = CurrentForwardPassInfo(
        request_id="sd", graph_walk=Wan22Model.VIDEO_GEN_WALK, fwd_index=0, random_seed=SEED, max_tokens=0,
        step_metadata={
            "num_inference_steps": 4, "guidance_scale": 1.0, "guidance_scale_2": 1.0,
            "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES,
        },
    )
    engine_inputs = ModelInputsFromEngine(request_ids=["sd"], per_request_info={"sd": info})
    text = {"text_embeds_pos": [embeds], "text_embeds_neg": [embeds.new_zeros(1)]}
    carried: dict[str, list[torch.Tensor]] = {}
    diffs = []
    with torch.no_grad():
        for step in range(4):
            info.dynamic_loop_iter_counts[DENOISE_LOOP_NAME] = step
            node_inputs = dit.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, info, {**text, **carried})
            stacked = dit.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, [node_inputs])
            out = dit.forward(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **stacked)
            carried = {"latents": out["latents"]}
            diffs.append((out["latents"][0] - expected[step]).abs().max().item())
    print(f"step-distill per-step max|diff| vs LightX2V's loop: {diffs}")
    assert max(diffs) <= STEP_MAX_ABS, diffs
