"""Wan2.2-T2V-A14B on LightX2V's step-distill recipe (``sampler: step_distill``) and the original-Wan-layout
LoRA mapping its 4-step distill LoRAs need, on the CPU.

The oracle is LightX2V's ``WanStepDistillScheduler`` (lightx2v/models/schedulers/wan/step_distill/scheduler.py:
``set_denoising_timesteps`` and ``step_post``, copied below) and its expert switch at ``boundary_step_index``
(models/runners/wan/wan_runner.py). Stand-ins replace the two experts.
"""

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components.diffusion.dmd import step_distill_schedule, step_distill_step
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import SAMPLER_STEP_DISTILL, WAN22_VARIANT_T2V_A14B, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22StepDistillDitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.model.wan22.weight_loader import apply_dit_loras, convert_wan_lora_keys, remap_checkpoint_key

LIGHTX2V_STEPS, LIGHTX2V_SHIFT, BOUNDARY_STEP_INDEX = [1000, 750, 500, 250], 5.0, 2
STEP_METADATA = {
    "num_inference_steps": 4, "guidance_scale": 1.0, "guidance_scale_2": 1.0, "height": 64, "width": 64,
    "num_frames": 5,
}
LATENT = (1, 16, 2, 8, 8)


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


def _lightx2v_tables(steps, shift):
    """``WanStepDistillScheduler.set_denoising_timesteps``, verbatim (sigma_min 0, sigma_max 1)."""
    sigmas = torch.linspace(1.0, 0.0, 1000 + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * 1000
    index = [1000 - x for x in steps]
    return sigmas[index], timesteps[index]


def _lightx2v_step_post(latents, noise_pred, sigmas, step_index, infer_steps):
    flow_pred = noise_pred.to(torch.float32)
    sigma = sigmas[step_index].item()
    noisy = latents.to(torch.float32) - sigma * flow_pred
    if step_index < infer_steps - 1:
        noisy = noisy + flow_pred * sigmas[step_index + 1].item()
    return noisy.to(latents.dtype)


def test_schedule_is_lightx2vs():
    sigmas, timesteps = _lightx2v_tables(LIGHTX2V_STEPS, LIGHTX2V_SHIFT)
    schedule = step_distill_schedule(LIGHTX2V_STEPS, LIGHTX2V_SHIFT)
    assert torch.equal(schedule.sigmas, torch.cat([sigmas, torch.zeros(1)]))
    # the model is conditioned on the shifted sigma, not on the step list
    assert torch.equal(schedule.timesteps, timesteps)
    assert schedule.timesteps[1].item() == pytest.approx(937.5)


def test_step_matches_lightx2v_bitwise():
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(LATENT, generator=generator)
    flow = torch.randn(LATENT, generator=generator).to(torch.bfloat16)
    sigmas, _ = _lightx2v_tables(LIGHTX2V_STEPS, LIGHTX2V_SHIFT)
    for k in range(4):
        expected = _lightx2v_step_post(latents, flow, sigmas, k, 4)
        nxt = None if k == 3 else sigmas[k + 1].item()
        assert torch.equal(step_distill_step(latents, flow, sigmas[k].item(), nxt), expected), k


class _StandIn(torch.nn.Module):
    dtype = torch.bfloat16

    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        per_row = timestep.float().reshape(hidden_states.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1, 1)
        return (hidden_states.float() * 0.3 + self.offset + per_row / 1000).to(torch.bfloat16)


def _dit():
    config = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, sampler=SAMPLER_STEP_DISTILL, compile_dit=False)
    dit = Wan22StepDistillDitSubmodule(_StandIn(0.1), config, transformer_2=_StandIn(-0.2))
    dit.get_device = lambda: torch.device("cpu")
    return dit


def test_four_steps_match_lightx2v_with_its_expert_switch():
    dit = _dit()
    info = CurrentForwardPassInfo(
        request_id="r", graph_walk=Wan22Model.VIDEO_GEN_WALK, fwd_index=0, random_seed=5, max_tokens=0,
        step_metadata=dict(STEP_METADATA), dynamic_loop_iter_counts={DENOISE_LOOP_NAME: 0},
    )
    engine_inputs = ModelInputsFromEngine(request_ids=["r"], per_request_info={"r": info})
    text = {"text_embeds_pos": [torch.zeros(1, 8, 16)], "text_embeds_neg": [torch.zeros(1)]}

    sigmas, timesteps = _lightx2v_tables(LIGHTX2V_STEPS, LIGHTX2V_SHIFT)
    high, low = _StandIn(0.1), _StandIn(-0.2)
    latents = torch.randn(LATENT, generator=torch.Generator().manual_seed(5), dtype=torch.float32)
    carried = {}
    for k in range(4):
        info.dynamic_loop_iter_counts[DENOISE_LOOP_NAME] = k
        model = high if k < BOUNDARY_STEP_INDEX else low
        pred = model(latents.to(torch.bfloat16), timesteps[k:k + 1].view(1, 1), None)
        latents = _lightx2v_step_post(latents, pred, sigmas, k, 4)

        node_inputs = dit.prepare_inputs(Wan22Model.VIDEO_GEN_WALK, info, {**text, **carried})
        stacked = dit.preprocess(Wan22Model.VIDEO_GEN_WALK, engine_inputs, [node_inputs])
        out = dit.forward(Wan22Model.VIDEO_GEN_WALK, engine_inputs, **stacked)
        carried = {"latents": out["latents"]}
        assert torch.equal(out["latents"][0], latents), f"step {k}"


# --- the model seam ---

def _model(**kwargs):
    return Wan22Model(
        model_path_hf="test/a14b", variant=WAN22_VARIANT_T2V_A14B, skip_weight_loading=True,
        sampler=SAMPLER_STEP_DISTILL, **kwargs,
    )


def test_recipe_fixes_steps_and_turns_cfg_off():
    model = _model()
    assert model.get_loop_caps() == {DENOISE_LOOP_NAME: 4}
    (_, video) = model.walk_plan("default", ["text"], ["video"], {"text_inputs": [object()]}, {})
    assert video.step_metadata["guidance_scale"] == 1.0 and video.step_metadata["guidance_scale_2"] == 1.0
    loop = model.get_graph_walk_graphs()[Wan22Model.VIDEO_GEN_WALK].sections[0]
    assert {edge.name for edge in loop.section.outputs} == {"latents"}


@pytest.mark.parametrize("kwargs", [{"num_inference_steps": 40}, {"guidance_scale_2": 3.0}, {"guidance_scale": 4.0}])
def test_recipe_refuses_other_steps_and_cfg(kwargs):
    with pytest.raises(ValueError):
        _model().walk_plan("default", ["text"], ["video"], {"text_inputs": [object()]}, kwargs)


def test_lora_2_needs_two_experts():
    with pytest.raises(ValueError, match="lora_2"):
        Wan22Model(model_path_hf="test/wan22", skip_weight_loading=True, lora_2=["/x.safetensors"])


# --- the LoRA key mapping ---

_ORIGINAL_MODULES = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2",
)


def test_original_wan_lora_keys_land_on_every_block_linear():
    config = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B)
    with torch.device("meta"):
        params = dict(Wan22DiT(config).named_parameters())
    keys = {
        f"blocks.{n}.{module}.{kind}.weight": torch.zeros(1)
        for n in range(config.num_layers) for module in _ORIGINAL_MODULES for kind in ("lora_A", "lora_B")
    }
    converted = convert_wan_lora_keys(keys)
    modules = {key.rsplit(".lora_", 1)[0] for key in converted}
    targets = {remap_checkpoint_key(module + ".weight") for module in modules}
    assert len(targets) == 400 and targets <= params.keys()
    # every attention and FFN linear of every block, nothing else
    linears = {name for name in params if name.startswith("blocks.") and name.endswith(".weight")
               and any(f".{p}." in name for p in ("q_proj", "k_proj", "v_proj", "o_proj", "linear_in", "linear_out"))}
    assert targets == linears
    # diffusers-named keys pass through
    assert convert_wan_lora_keys({"blocks.0.attn1.to_q.lora_A.weight": 1}) == {"blocks.0.attn1.to_q.lora_A.weight": 1}


def test_lightx2v_lora_merges_at_scale_one(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    config = Wan22Config.for_variant(
        WAN22_VARIANT_T2V_A14B, num_attention_heads=2, attention_head_dim=8, num_layers=1, ffn_dim=32, text_dim=16,
        freq_dim=16,
    )
    dit = Wan22DiT(config)
    generator = torch.Generator().manual_seed(0)
    for p in dit.parameters():
        p.data = torch.randn(p.shape, generator=generator).to(p.dtype)
    before = {name: p.detach().clone() for name, p in dit.named_parameters()}
    down = torch.randn(4, 16, generator=generator).to(torch.bfloat16)
    up = torch.randn(16, 4, generator=generator).to(torch.bfloat16)
    path = tmp_path / "lora.safetensors"
    # LightX2V's layout: prefix, Kohya down/up, no alpha (its loader then scales by 1.0)
    safetensors.save_file({
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": down,
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": up,
    }, str(path))
    from mstar.engine.resources.lora.checkpoint import LoraFile

    apply_dit_loras(dit, [LoraFile(str(path), 0.5)])
    after = dict(dit.named_parameters())
    target = "blocks.0.self_attn.q_proj.weight"
    expected = (before[target].float() + 0.5 * (up.float() @ down.float())).to(before[target].dtype)
    assert torch.equal(after[target], expected)
    assert all(torch.equal(after[name], value) for name, value in before.items() if name != target)


def test_a14b_distill_yaml_builds_its_graphs():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "wan22_a14b_distill.yaml"
    kwargs = yaml.safe_load(path.read_text())["model_kwargs"]
    model = Wan22Model(skip_weight_loading=True, **kwargs)
    assert model.config.sampler == SAMPLER_STEP_DISTILL and len(model.loras) == len(model.loras_2) == 1
    graphs = model.get_worker_graphs(str(path))
    assert {walk for graph in graphs for walk in graph.graph_walks} == {
        Wan22Model.ENCODE_TEXT_WALK, Wan22Model.VIDEO_GEN_WALK,
    }
