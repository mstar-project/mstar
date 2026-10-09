#!/usr/bin/env python3
"""Component parity against the recorded diffusers oracle (see ``record_oracle.py``).

    LTX25_ORACLE_DIR=<dir>/t2av NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        python test/ltx2_5/parity_harness.py {dit,text,loop}

Prints per-tensor error statistics; ``test_reference_equivalence.py`` asserts on the
same comparisons. Kept as a script because each stage loads tens of GB and a
human reading the numbers is the point while porting.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.components.diffusion.attention import sdpa_attention  # noqa: E402
from mstar.model.ltx2_5.components.transformer import LTX2Attends, build_rope  # noqa: E402
from mstar.model.ltx2_5.config import LTX25_REPO, SNAPSHOT_PATTERNS, LTX25Config  # noqa: E402
from mstar.utils.hf_snapshot import resolve_snapshot_dir  # noqa: E402

ORACLE = Path(os.environ.get("LTX25_ORACLE_DIR", "integration_testing/ltx25/oracle/t2av"))
SDPA = LTX2Attends(*([sdpa_attention] * 6))


def load(name: str):
    return torch.load(ORACLE / f"{name}.pt", weights_only=False)


def report(name: str, got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.float().cpu(), want.float().cpu()
    err = (got - want).abs()
    rel = err.norm() / want.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
    print(f"{name:28s} max_abs={err.max():.4e} mean_abs={err.mean():.4e} rel_l2={rel:.4e} cos={cos:.6f}")
    return float(rel)


def snapshot_and_config():
    snapshot = resolve_snapshot_dir(LTX25_REPO, allow_patterns=SNAPSHOT_PATTERNS)
    return snapshot, LTX25Config.from_snapshot(snapshot)


@torch.inference_mode()
def check_dit(device="cuda:0"):
    from mstar.model.ltx2_5.weight_loader import build_transformer

    snapshot, config = snapshot_and_config()
    dit = build_transformer(config, snapshot, device)
    inp = load("dit_in_000")
    rope = build_rope(config.transformer, inp["num_frames"], inp["height"], inp["width"], inp["fps"],
                      inp["audio_num_frames"], device)
    blocks = {}
    hooks = [
        dit.transformer_blocks[i].register_forward_hook(
            lambda m, a, out, _i=i: blocks.__setitem__(_i, out))
        for i in (0, 1, 2, 12, 24, 47)
    ]
    video, audio = dit(
        inp["hidden_states"].to(device), inp["audio_hidden_states"].to(device),
        inp["encoder_hidden_states"].to(device), inp["audio_encoder_hidden_states"].to(device),
        inp["timestep"].to(device), rope, SDPA,
    )
    for h in hooks:
        h.remove()
    for i, (x, a) in sorted(blocks.items()):
        ref_x, ref_a = load(f"block_{i:02d}")
        report(f"block {i} video", x, ref_x)
        report(f"block {i} audio", a, ref_a)
    ref_v, ref_a = load("dit_out_000")
    report("velocity video", video, ref_v)
    report("velocity audio", audio, ref_a)


@torch.inference_mode()
def check_text(device="cuda:1"):
    from mstar.model.ltx2_5.weight_loader import build_connectors, build_text_encoder

    snapshot, config = snapshot_and_config()
    ids, mask = load("input_ids"), load("attention_mask").bool()
    real = ids[mask].to(device)
    n = real.shape[0]
    positions = torch.arange(config.text_max_seq_len - n, config.text_max_seq_len, device=device)
    encoder = build_text_encoder(config, snapshot, device)
    hidden = encoder(real[None], positions[None])[0]
    ref_hidden = load("gemma_hidden")
    for layer in (0, 1, 6, 24, 47, 48):
        report(f"gemma hidden {layer}", hidden[..., layer], ref_hidden[..., layer])
    del encoder
    torch.cuda.empty_cache()
    connectors = build_connectors(config, snapshot, device)
    # feed the oracle's hidden states, so the connectors are checked in isolation
    video, audio = connectors(ref_hidden[None].to(device), [n], config.text_max_seq_len)
    report("connector video", video, load("text_video"))
    report("connector audio", audio, load("text_audio"))
    video2, audio2 = connectors(hidden[None], [n], config.text_max_seq_len)
    report("text path video (e2e)", video2, load("text_video"))
    report("text path audio (e2e)", audio2, load("text_audio"))


def distilled_schedule():
    import numpy as np

    from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
    from mstar.model.ltx2_5.config import DISTILLED_SIGMA_VALUES

    sigmas = torch.from_numpy(np.array(DISTILLED_SIGMA_VALUES).astype(np.float32))
    return FlowMatchSchedule(
        sigmas=torch.cat([sigmas, torch.zeros(1)]), timesteps=sigmas * 1000, mu=None,
    )


def ltx_step(x: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    """The reference's unguided step: velocity -> x0 -> velocity (fp32), then Euler."""
    from mstar.model.components.diffusion.flow_match import euler_step

    velocity = velocity.float()
    x0 = x - velocity * sigma
    return euler_step(x, (x - x0) / sigma, sigma, sigma_next)


@torch.inference_mode()
def check_loop(device="cuda:0"):
    from mstar.model.ltx2_5.weight_loader import build_transformer

    snapshot, config = snapshot_and_config()
    dit = build_transformer(config, snapshot, device)
    rope = build_rope(config.transformer, 16, 17, 30, 24.0, 126, device)
    sched = distilled_schedule()
    x, a = load("latents_init").to(device), load("audio_init").to(device)
    text, audio_text = load("text_video").to(device), load("text_audio").to(device)
    for k in range(sched.num_steps):
        sigma, sigma_next = sched.sigmas[k].to(device), sched.sigmas[k + 1].to(device)
        if k > 0:
            ref_in = load(f"dit_in_{k:03d}")
            report(f"step {k} video in", x.to(torch.bfloat16), ref_in["hidden_states"])
            report(f"step {k} audio in", a.to(torch.bfloat16), ref_in["audio_hidden_states"])
        v, va = dit(x.to(torch.bfloat16), a.to(torch.bfloat16), text, audio_text,
                    sched.timesteps[k:k + 1].to(device), rope, SDPA)
        x, a = ltx_step(x, v, sigma, sigma_next), ltx_step(a, va, sigma, sigma_next)
    report("final video latents", x, load("latents_007"))
    torch.save((x.cpu(), a.cpu()), ORACLE / "native_final_latents.pt")


@torch.inference_mode()
def check_decode(device="cuda:1"):
    """The serving environment's diffusers decoders (0.39) on the oracle's final latents
    against the oracle's own decode (0.41)."""
    from mstar.model.ltx2_5.ltx2_5_model import LTX25Model
    from mstar.model.ltx2_5.submodules import shape_from_metadata

    model = LTX25Model()
    meta = {"height": 544, "width": 960, "num_frames": 121, "fps": 24.0}
    shape = shape_from_metadata(model.config, meta)
    video_dec = model.get_submodule("vae_decoder", device)
    audio_dec = model.get_submodule("audio_decoder", device)
    frames = video_dec._run(latents=load("latents_007").to(device), shape=shape)["video_output"][0]
    ref = load("video").permute(3, 0, 1, 2)  # [F, H, W, 3] -> [3, F, H, W]
    diff = (frames.cpu().float() - ref.float())
    mse = diff.pow(2).mean()
    print(f"video frames  max_abs={diff.abs().max():.0f} psnr={10 * torch.log10(255 ** 2 / mse):.2f} dB "
          f"shape={tuple(frames.shape)}")
    report("vae decode input", _video_decode_input(video_dec, shape, device), load("vae_decode_in"))
    wave = audio_dec._run(audio_latents=load("audio_final").to(device))["audio_output"][0]
    report("audio waveform", wave, load("audio_wave"))


def _video_decode_input(video_dec, shape, device):
    from mstar.model.ltx2_5.submodules import unpack_video

    vae = video_dec.vae
    x = unpack_video(load("latents_007").to(device=device, dtype=vae.dtype), shape)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, vae.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, vae.dtype)
    return x * std / vae.config.scaling_factor + mean




@torch.inference_mode()
def check_forced(device="cuda:0", attention="flashinfer", compile_transformer=True):
    """Teacher-forced per-step velocity error of the served configuration: each step fed
    the oracle's exact input. Compare with the reference's own flash-vs-efficient spread
    (``reference_noise_floor.py``: video ~1.8-2.2%, audio ~0.7-1.0%)."""
    from mstar.model.components.diffusion.compile_utils import compile_transformer_forward
    from mstar.model.ltx2_5.weight_loader import build_transformer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark" / "ltx2_5"))
    from profile_step import ragged_attends

    from mstar.model.ltx2_5.submodules import shape_from_metadata

    snapshot, config = snapshot_and_config()
    dit = build_transformer(config, snapshot, device)
    if compile_transformer:
        compile_transformer_forward(dit, eager_rounding=True)
    shape = shape_from_metadata(config, {"height": 544, "width": 960, "num_frames": 121, "fps": 24.0})
    rope = build_rope(config.transformer, shape.frames, shape.height, shape.width, shape.fps, shape.audio_frames,
                      device)
    attends = SDPA if attention == "sdpa" else ragged_attends(config, shape, 1, device)
    for k in range(8):
        inp = load(f"dit_in_{k:03d}")
        v, va = dit(inp["hidden_states"].to(device), inp["audio_hidden_states"].to(device),
                    inp["encoder_hidden_states"].to(device), inp["audio_encoder_hidden_states"].to(device),
                    inp["timestep"].to(device), rope, attends)
        rv, ra = load(f"dit_out_{k:03d}")
        report(f"forced step {k} video", v, rv)
        report(f"forced step {k} audio", va, ra)





@torch.inference_mode()
def check_two_stage_seed(device="cuda:0"):
    """The two-stage recipe's hand-offs against the two-stage oracle: the latent
    upsampler, and stage 2's starting latents (re-normalize, pack, generator replay,
    sigma-0 noise blend)."""
    from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.ltx2_5.submodules import (
        REFINE_AUDIO,
        REFINE_LATENTS,
        LTXDenoiseSubmodule,
        pack_audio,
        pack_video,
        shape_from_metadata,
    )

    two = Path(str(ORACLE).replace("t2av", "two_stage"))
    snapshot, config = snapshot_and_config()
    up = LTX2LatentUpsamplerModel.from_pretrained(str(snapshot / "latent_upsampler"),
                                                  torch_dtype=torch.bfloat16).to(device)
    s1_video = torch.load(two / "s1_out_video.pt").to(device)
    upsampled = up(s1_video.to(torch.bfloat16))
    report("latent upsampler", upsampled, torch.load(two / "upsampled.pt"))

    vae = AutoencoderKLLTX2Video.from_pretrained(str(snapshot / "vae"))
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(str(snapshot / "audio_vae"))
    ref_up = torch.load(two / "upsampled.pt").to(device)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, ref_up.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, ref_up.dtype)
    refine_video = pack_video((ref_up - mean) * vae.config.scaling_factor / std)[0]
    s1_audio = torch.load(two / "s1_out_audio.pt").to(device)          # denormalized [1, 8, L, 16]
    a_mean, a_std = audio_vae.latents_mean.to(device), audio_vae.latents_std.to(device)
    refine_audio = ((pack_audio(s1_audio) - a_mean) / a_std)[0]

    sub = LTXDenoiseSubmodule(torch.nn.Linear(1, 1).to(device), config, loop_name="denoise_loop",
                              use_ragged_attention=False, refine_walks=frozenset({"refine_av"}))
    meta = {"height": 1088, "width": 1920, "num_frames": 121, "fps": 24.0}
    info = CurrentForwardPassInfo(request_id="r", graph_walk="refine_av", fwd_index=0, random_seed=0, max_tokens=0,
                                  step_metadata=meta)
    seeds = sub.initial_loop_back(info, {REFINE_LATENTS: [refine_video], REFINE_AUDIO: [refine_audio]},
                                  shape_from_metadata(config, meta), torch.Generator().manual_seed(0))
    report("stage-2 starting video", seeds["latents"], torch.load(two / "s2_latents_init.pt")[0])
    report("stage-2 starting audio", seeds["audio_latents"], torch.load(two / "s2_audio_init.pt")[0])


if __name__ == "__main__":
    {"dit": check_dit, "text": check_text, "loop": check_loop, "decode": check_decode,
     "forced": check_forced, "two_stage_seed": check_two_stage_seed}[sys.argv[1]]()
