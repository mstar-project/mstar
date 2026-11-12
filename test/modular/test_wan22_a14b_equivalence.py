"""Wan2.2-T2V-A14B vs diffusers: each expert's DiT, and a short generation end to end.

1. Per expert (``transformer``, ``transformer_2``): the native ``Wan22DiT`` loaded from
   the expert's subfolder against diffusers' ``WanTransformer3DModel``, both in the
   serving dtypes (bf16 + fp32 islands). Weights bit-identical through the loader's
   remap, and the forward on one batch-2 input at a scalar timestep per sample (the
   reference's ``t.expand(batch)``; the native module takes it ``[batch, 1]``, the same
   rows).
2. A short T2V generation (832x480, 17 frames, ``STEPS`` UniPC steps crossing the
   0.875 boundary, guidance 4.0 / 3.0, seed 42) through the wan22 text encoder, dit
   and VAE decoder submodules, driven as the engine drives them, against
   ``diffusers.WanPipeline`` with the same seed: every step's latents, the final
   latents, and the decoded frames. The served dit batches the two CFG branches into
   one forward (the reference runs two), which is the only deviation priced; with the
   branches run the reference's way the latents must match exactly.

Run with the reference's numerics pinned, as test_wan22_reference_equivalence.py does:

    CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        pytest test/modular/test_wan22_a14b_equivalence.py -v -s

Skipped without CUDA or the checkpoint in the local HF cache. Needs one ~80 GB GPU:
one expert pair (reference + native, ~57 GB) at a time in part 1, the pipeline then
the two native experts in part 2.
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
from mstar.model.components.diffusion.unipc import make_unipc_tables
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.config import WAN22_VARIANT_T2V_A14B, Wan22Config
from mstar.model.wan22.submodules import (
    DENOISE_LOOP_NAME,
    Wan22DitSubmodule,
    Wan22TextEncoderSubmodule,
    Wan22VaeDecoderSubmodule,
)
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.model.wan22.weight_loader import build_wan22_dit, remap_checkpoint_key

MODEL_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CUDA_AVAILABLE = torch.cuda.is_available()

PROMPT = "A red fox trotting through fresh snow in a pine forest, morning light, cinematic."
SEED = 42
HEIGHT, WIDTH, FRAMES = 480, 832, 17
STEPS = 6
GUIDANCE, GUIDANCE_2 = 4.0, 3.0

# ---------------------------------------------------------------------------
# Thresholds (H200, torch 2.11, cuDNN 9.19, diffusers 0.42.1). NEVER loosen one to pass.
# ---------------------------------------------------------------------------
# The native forward on the reference's own per-sample rows: the same ops (0.0 observed
# for both experts).
DIT_FORWARD_MAX_ABS = 0.0
# The served path's batch-2 CFG forward against the reference's two batch-1 ones: bf16
# kernel noise (1.4e-2 at step 0), amplified by the combine and compounded over the
# six large steps to 3.09 at the last. ~1.3x that; a wrong branch or expert is off by
# O(1) from step 0.
BATCHED_CFG_LATENTS_MAX_ABS = 4.0
# With the branches run the reference's way nothing is left to round differently
# (0.0 observed at every step, both experts).
SEQ_CFG_LATENTS_MAX_ABS = 0.0
# Decoded frames against the reference's (uint8 against the reference's floats scaled to
# 0..255). Bit-exact latents: only the decode dtype differs (the node decodes bf16 on
# this cuDNN, the reference fp32), 50.9 dB observed. Batched-CFG latents: 23.3 dB.
SEQ_CFG_FRAMES_MIN_PSNR_DB = 48.0
BATCHED_CFG_FRAMES_MIN_PSNR_DB = 21.0
# The shared Wan VAE decoder (channels-last, compiled) against stock vae.decode on
# the same latents: 57.4 dB observed.
FAST_DECODE_MIN_PSNR_DB = 52.0


def _hf_cache_has_checkpoint() -> bool:
    dirname = f"models--{MODEL_REPO.replace('/', '--')}"
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        root = os.environ.get(env)
        if not root:
            continue
        base = Path(root) if env == "HF_HUB_CACHE" else Path(root) / "hub"
        if (base / dirname).exists():
            return True
    return (Path.home() / ".cache" / "huggingface" / "hub" / dirname).exists()


pytest.importorskip("diffusers", reason="diffusers not installed")

pytestmark = [
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA"),
    pytest.mark.skipif(not _hf_cache_has_checkpoint(), reason=f"{MODEL_REPO} not in the local HF cache"),
]

DEVICE = torch.device("cuda")


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _psnr(ours: torch.Tensor, ref: torch.Tensor) -> float:
    mse = (ours.double() - ref.double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0**2 / mse)).item()


# ---------------------------------------------------------------------------
# 1. Each expert's DiT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subfolder", ["transformer", "transformer_2"])
def test_expert_dit_matches_diffusers(subfolder):
    from diffusers import WanTransformer3DModel
    from huggingface_hub import snapshot_download

    config = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B)
    # from the snapshot directory: a hub id makes diffusers' sharded load ask the Hub
    # for the revision even offline
    snapshot = snapshot_download(MODEL_REPO, allow_patterns=[f"{subfolder}/*"])
    wrapped = WanTransformer3DModel.from_pretrained(
        snapshot, subfolder=subfolder, torch_dtype=torch.bfloat16,
    ).eval().to(DEVICE)
    native = build_wan22_dit(config, MODEL_REPO, device=DEVICE, subfolder=subfolder)
    try:
        native_params = dict(native.named_parameters())
        wrapped_params = dict(wrapped.named_parameters())
        # 15 + 27 per block
        assert len(native_params) == len(wrapped_params) == 1095
        assert {remap_checkpoint_key(n) for n in wrapped_params} == set(native_params)
        bad = [
            n for n, p in wrapped_params.items()
            if native_params[remap_checkpoint_key(n)].dtype != p.dtype
            or not torch.equal(native_params[remap_checkpoint_key(n)], p)
        ]
        assert not bad, f"{subfolder}: {len(bad)} parameters differ, first {bad[:5]}"
        # time_embedder (4) + 41 scale_shift_tables + 40x norm2 weight+bias
        assert sum(p.dtype == torch.float32 for p in native_params.values()) == 125

        gen = torch.Generator(device="cpu").manual_seed(1234)
        hidden = torch.randn((2, 16, 5, HEIGHT // 8, WIDTH // 8), generator=gen).to(DEVICE, torch.bfloat16)
        text = torch.randn((2, 512, 4096), generator=gen).to(DEVICE, torch.bfloat16)
        t = torch.tensor([937, 937], dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            ref = wrapped(hidden_states=hidden, timestep=t, encoder_hidden_states=text, return_dict=False)[0]
            ours = native(hidden_states=hidden, timestep=t[:, None], encoder_hidden_states=text)
        diff = (ours.float() - ref.float()).abs().max().item()
        print(f"\n{subfolder} forward: max_abs={diff:.3e} (output abs max {ref.float().abs().max().item():.2f})")
        assert diff <= DIT_FORWARD_MAX_ABS, f"{subfolder} forward diverges: {diff:.3e}"
    finally:
        del wrapped, native
        _free()


# ---------------------------------------------------------------------------
# 2. A short generation, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference() -> dict:
    """``WanPipeline`` at the module's settings: every step's latents, the final
    latents and the decoded frames ``[F, H, W, 3]`` in 0..255; plus our text
    embeddings, from the pipeline's own UMT5 instance."""
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(MODEL_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_REPO, vae=vae, torch_dtype=torch.bfloat16).to(DEVICE)
    per_step: list[torch.Tensor] = []

    def keep(_pipe, _i, _t, kwargs):
        per_step.append(kwargs["latents"].detach().float().cpu())
        return {}

    with torch.no_grad():
        frames = pipe(
            prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=STEPS,
            guidance_scale=GUIDANCE, guidance_scale_2=GUIDANCE_2,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            output_type="pt", callback_on_step_end=keep,
        ).frames[0]  # [F, 3, H, W] in [0, 1]
    result = {
        "per_step": per_step,
        "final": per_step[-1],
        "frames": (frames.float() * 255).permute(0, 2, 3, 1).cpu(),
        "scheduler_config": dict(pipe.scheduler.config),
        "timesteps": pipe.scheduler.timesteps.cpu(),
    }

    model = Wan22Model(model_path_hf=MODEL_REPO, variant=WAN22_VARIANT_T2V_A14B)
    tokens = model.process_prompt(prompt=PROMPT, input_modalities=["text"], output_modalities=["video"])
    encoder = Wan22TextEncoderSubmodule(pipe.text_encoder, model.config)
    info = _fwd_info("encode_text")
    with torch.no_grad():
        out = _drive(encoder, "encode_text", info, {"text_inputs": tokens["text_inputs"]})
        ref_pos = pipe._get_t5_prompt_embeds(PROMPT, max_sequence_length=512, device=DEVICE)
    assert torch.equal(out["text_embeds_pos"][0], ref_pos)
    result["text"] = {"text_embeds_pos": out["text_embeds_pos"], "text_embeds_neg": out["text_embeds_neg"]}
    del pipe, encoder, vae
    _free()
    return result


def _step_metadata() -> dict:
    return {
        "num_inference_steps": STEPS, "guidance_scale": GUIDANCE, "guidance_scale_2": GUIDANCE_2,
        "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES,
    }


def _fwd_info(walk: str) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="a14b-r0", graph_walk=walk, fwd_index=0, random_seed=SEED, max_tokens=0,
        step_metadata=_step_metadata(),
    )


def _drive(submodule, walk: str, info: CurrentForwardPassInfo, inputs: dict) -> dict:
    """prepare_inputs -> preprocess -> forward, as the engine calls them."""
    node_inputs = submodule.prepare_inputs(graph_walk=walk, fwd_info=info, inputs=inputs)
    engine_inputs = ModelInputsFromEngine(request_ids=[info.request_id], per_request_info={info.request_id: info})
    stacked = submodule.preprocess(graph_walk=walk, engine_inputs=engine_inputs, inputs=[node_inputs])
    return submodule.forward(graph_walk=walk, engine_inputs=engine_inputs, **stacked)


@pytest.fixture(scope="module")
def dit(reference) -> Wan22DitSubmodule:
    model = Wan22Model(model_path_hf=MODEL_REPO, variant=WAN22_VARIANT_T2V_A14B)
    model.config.compile_dit = False
    submodule = model.get_submodule("dit", device=str(DEVICE))
    assert submodule.transformer_2 is not None
    return submodule


@pytest.fixture(scope="module")
def decoder() -> Wan22VaeDecoderSubmodule:
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(MODEL_REPO, subfolder="vae").eval()
    config = Wan22Config.for_variant(WAN22_VARIANT_T2V_A14B, vae_decode_fast_sizes=[f"{WIDTH}x{HEIGHT}"])
    return Wan22VaeDecoderSubmodule(vae, config).to(DEVICE)


def _sequential_noise_prediction(
    self, model_input, temp_ts, text_embeds_pos, text_embeds_neg, guidance_scale, transformer=None, combine_scale=None,
):
    """The reference's CFG: two batch-1 forwards, then the combine."""
    transformer = transformer or self.transformer
    timestep = temp_ts.reshape(1, -1)
    cond, uncond = (
        transformer(hidden_states=model_input, timestep=timestep, encoder_hidden_states=text.to(transformer.dtype))
        for text in (text_embeds_pos, text_embeds_neg)
    )
    scale = guidance_scale if combine_scale is None else combine_scale
    return uncond + scale * (cond - uncond)


def _generate(dit, reference, sequential_cfg: bool) -> tuple[list[float], torch.Tensor, list[bool]]:
    """Run the denoise loop through the dit; per-step max-abs against the reference,
    the final latents, and which steps ran the high-noise expert."""
    info = _fwd_info("video_gen")
    carried: dict = {}
    diffs, high = [], []
    original = Wan22DitSubmodule._noise_prediction
    if sequential_cfg:
        Wan22DitSubmodule._noise_prediction = _sequential_noise_prediction
    calls = []
    hook = dit.transformer.register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            for step in range(STEPS):
                info.dynamic_loop_iter_counts[DENOISE_LOOP_NAME] = step
                before = len(calls)
                outputs = _drive(dit, "video_gen", info, {**reference["text"], **carried})
                high.append(len(calls) > before)
                carried = {n: outputs[n] for n in ("latents", "unipc_model_outputs", "unipc_last_sample")}
                diffs.append((outputs["latents"][0].float().cpu() - reference["per_step"][step]).abs().max().item())
    finally:
        Wan22DitSubmodule._noise_prediction = original
        hook.remove()
    return diffs, carried["latents"][0], high


def _decode(decoder, latents: torch.Tensor) -> torch.Tensor:
    info = _fwd_info("video_gen")
    with torch.no_grad():
        return _drive(decoder, "video_gen", info, {"latents": [latents]})["video_output"][0].float().cpu()


def test_schedule_and_boundary_match_the_pipeline(reference):
    sigmas, timesteps = make_unipc_tables(STEPS, reference["scheduler_config"]["flow_shift"])
    assert torch.equal(timesteps, reference["timesteps"])
    high = [bool(t >= 0.875 * 1000) for t in timesteps]
    assert any(high) and not all(high), f"{STEPS} steps must cross the boundary: {timesteps.tolist()}"


@pytest.mark.parametrize("sequential_cfg", [True, False], ids=["reference_cfg", "batched_cfg"])
def test_generation_matches_the_pipeline(reference, dit, decoder, sequential_cfg, monkeypatch):
    monkeypatch.setenv("WAN22_VAE_DECODE_TILING", "untiled")
    diffs, latents, high = _generate(dit, reference, sequential_cfg)
    expected_high = [bool(t >= 875) for t in reference["timesteps"]]
    assert high == expected_high
    label = "reference CFG" if sequential_cfg else "batched CFG (served)"
    print(f"\n{label}: per-step latents max_abs {[f'{d:.3e}' for d in diffs]} (high-noise steps {high})")
    # stock vae.decode, the reference's, at an unlisted size
    decoder._fast_sizes = frozenset()
    frames = _decode(decoder, latents)
    psnr = _psnr(frames, reference["frames"])
    final = (latents.float().cpu() - reference["final"]).abs().max().item()
    print(f"{label}: final latents max_abs {final:.3e}, frames PSNR {psnr:.2f} dB")
    if sequential_cfg:
        assert max(diffs) <= SEQ_CFG_LATENTS_MAX_ABS
        assert psnr >= SEQ_CFG_FRAMES_MIN_PSNR_DB
    else:
        assert max(diffs) <= BATCHED_CFG_LATENTS_MAX_ABS
        assert psnr >= BATCHED_CFG_FRAMES_MIN_PSNR_DB


def test_fast_decode_matches_stock_decode(reference, decoder, monkeypatch):
    """The shared Wan VAE decoder takes the Wan2.1 VAE as it takes Wan2.2's."""
    monkeypatch.setenv("WAN22_VAE_DECODE_TILING", "untiled")
    latents = reference["final"].to(DEVICE)
    decoder._fast_sizes = frozenset()
    stock = _decode(decoder, latents)
    decoder._fast_sizes = frozenset({(WIDTH, HEIGHT)})
    fast = _decode(decoder, latents)
    psnr = _psnr(fast, stock)
    print(f"\nfast vs stock decode: PSNR {psnr:.2f} dB, max_abs {(fast - stock).abs().max().item():.0f} levels")
    assert psnr >= FAST_DECODE_MIN_PSNR_DB
