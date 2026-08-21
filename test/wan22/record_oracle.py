#!/usr/bin/env python3
"""Record the diffusers golden-reference oracle for Wan2.2-TI2V-5B (T2V).

Records the oracle that ``test/modular/test_wan22_reference_equivalence.py``
compares against. Without it that suite cannot be run on a new machine, so it
ships with the model.

It runs the **stock diffusers ``WanPipeline``** — never any mstar code — on
``Wan-AI/Wan2.2-TI2V-5B-Diffusers``, saves the latents after every UniPC step via
``callback_on_step_end``, and writes every resolved parameter to metadata.json:

    <out-dir>/latents/step_000.pt ... step_049.pt   latents AFTER each step
    <out-dir>/final_latents.pt                      == the last step's tensor
    <out-dir>/wan22_ti2v5b_t2v_seed42.mp4           the reference video (PSNR target)
    <out-dir>/metadata.json                         resolved params + versions

Why the numerics flag below is load-bearing
-------------------------------------------
An oracle is only a reference if it is recorded under the **same numerics as the
serving process**. mstar's engine sets ``torch.set_float32_matmul_precision("high")``
process-wide (``mstar/engine/__init__.py``), which changes the fp32 matmul kernels.
Record the oracle without that flag and the *reference itself* moves — by 1.3e-3 at
step 0, growing to 4.2 by step 49. A server running under "high" then cannot match
an oracle recorded under the default, and the difference belongs to the reference,
not to the implementation. So this recorder sets the flag before any matmul runs,
and records it in metadata.json so the artifact states its own numerics.

The same argument applies to the torch build: **record the oracle with the same
torch as the server you will test.** Do not compare an oracle across torch builds.

Everything else below is a fact of the reference run. Change one and the
equivalence suite's thresholds stop describing anything:

    prompt / negative prompt  the constants below, recorded verbatim
    seed                      42 (--seed), via torch.Generator(device="cpu")
    size / frames             480 x 832 x 33 (--height/--width/--num-frames)
    steps / guidance          defaults unless --num-inference-steps / --guidance is
                              passed; pipeline defaults (50 / 5.0) resolved and
                              recorded afterwards. --guidance 1.0 records a no-CFG
                              oracle (one batch-1 forward per step).
    dtype                     bfloat16
    scheduler                 UniPCMultistepScheduler (checkpoint default)

Run it with the isolated oracle venv, never the mstar venv — the point is that
nothing mstar-side can influence the reference:

    CUDA_VISIBLE_DEVICES=0 /path/to/wan-oracle/bin/python test/wan22/record_oracle.py \
        --out-dir /path/to/oracle

Then point the suite at it (there is no default — unset, the suite skips):

    WAN22_ORACLE_DIR=/path/to/oracle \
        CUDA_VISIBLE_DEVICES=0 pytest test/modular/test_wan22_reference_equivalence.py -v
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# The serving process's numerics flag (mstar/engine/__init__.py). MUST be set
# before any matmul so the oracle is recorded under the serving numerics — see
# the module docstring for what happens when it is not.
torch.set_float32_matmul_precision("high")

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Fixed prompt for the golden run (recorded verbatim in metadata.json).
PROMPT = (
    "A fluffy orange tabby cat walks slowly across a sunlit wooden floor, "
    "its tail held high, while dust motes drift through a beam of warm "
    "afternoon light from a nearby window."
)

# The Wan pipeline's canonical negative prompt, shipped verbatim inside
# diffusers' pipeline_wan.py EXAMPLE_DOC_STRING (the signature default is None,
# which encode_prompt resolves to ""; the docstring example below is what every
# Wan reference run uses).
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards"
)

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 33
SEED = 42
FPS = 24  # Wan2.2-TI2V-5B model-card frame rate


@dataclass
class RunConfig:
    """Resolved recording knobs. ``guidance`` / ``num_inference_steps`` are None
    when the pipeline defaults are used (recorded afterwards from the run)."""

    height: int = HEIGHT
    width: int = WIDTH
    num_frames: int = NUM_FRAMES
    seed: int = SEED
    guidance: float | None = None
    num_inference_steps: int | None = None


def build_pipeline(offload: bool):
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def run(pipe, step_records: list, latents_dir: Path, run_cfg: "RunConfig"):
    generator = torch.Generator(device="cpu").manual_seed(run_cfg.seed)

    def save_step_latents(p, i, t, callback_kwargs):
        latents = callback_kwargs["latents"]
        torch.save(latents.detach().cpu(), latents_dir / f"step_{i:03d}.pt")
        step_records.append({
            "step_index": i,
            "timestep": float(t),
            "latents_shape": list(latents.shape),
            "latents_dtype": str(latents.dtype),
        })
        return {}

    call_kwargs = dict(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=run_cfg.height,
        width=run_cfg.width,
        num_frames=run_cfg.num_frames,
        generator=generator,
        callback_on_step_end=save_step_latents,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    # steps + guidance default to the pipeline's own values (recorded afterwards)
    # unless explicitly overridden. --guidance 1.0 records a no-CFG oracle (the
    # pipeline skips classifier-free guidance at guidance_scale <= 1), pricing the
    # dit submodule's batch-1 branch.
    if run_cfg.guidance is not None:
        call_kwargs["guidance_scale"] = run_cfg.guidance
    if run_cfg.num_inference_steps is not None:
        call_kwargs["num_inference_steps"] = run_cfg.num_inference_steps
    return pipe(**call_kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="where to write latents/, final_latents.pt, the mp4 and metadata.json "
             "(point WAN22_ORACLE_DIR at the same path when running the suite)",
    )
    ap.add_argument(
        "--expect-gpu", default="",
        help="substring the GPU name must contain (e.g. '5090', 'H100'). A recorded "
             "oracle is only valid for the GPU + torch build it was made on, so this "
             "guards against silently re-recording on the wrong device.",
    )
    # Recording knobs. The defaults reproduce the canonical CFG oracle exactly;
    # override them to record a variant (e.g. --guidance 1.0 for the no-CFG oracle).
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--guidance", type=float, default=None,
        help="guidance_scale; unset uses the pipeline default (CFG on). Pass 1.0 to "
             "record a no-CFG oracle (the pipeline runs one batch-1 forward per step).",
    )
    ap.add_argument(
        "--num-inference-steps", type=int, default=None,
        help="denoise steps; unset uses the pipeline default (50).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import diffusers
    import transformers
    from diffusers.utils import export_to_video

    out_dir = Path(args.out_dir)
    latents_dir = out_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to record the oracle.")
    gpu_name = torch.cuda.get_device_name(0)
    if args.expect_gpu and args.expect_gpu not in gpu_name:
        raise SystemExit(f"expected a GPU matching {args.expect_gpu!r}, got {gpu_name!r}")

    run_cfg = RunConfig(
        height=args.height, width=args.width, num_frames=args.num_frames,
        seed=args.seed, guidance=args.guidance,
        num_inference_steps=args.num_inference_steps,
    )

    step_records: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    # Try full-CUDA, fall back to CPU offload on OOM. The 5B pipeline does not
    # fit every card, and offload is not a compromise: it changes where modules
    # live, not what they compute. The suite's sequential-CFG trajectory test
    # confirms that end to end — an offloaded oracle is met bit-exactly by a
    # server holding everything on device. Which mode ran is recorded either way.
    mode = "full_cuda"
    pipe = build_pipeline(offload=False)
    try:
        result = run(pipe, step_records, latents_dir, run_cfg)
    except torch.cuda.OutOfMemoryError:
        print("OOM in full-CUDA mode; retrying with enable_model_cpu_offload()")
        mode = "cpu_offload"
        del pipe
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        step_records.clear()
        t_start = time.perf_counter()
        pipe = build_pipeline(offload=True)
        result = run(pipe, step_records, latents_dir, run_cfg)

    wall_time = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated()

    # Final (post-loop, pre-VAE-denormalization) latents = the last step
    # callback's tensor; re-saved under the canonical name the suite loads.
    final_step = max(r["step_index"] for r in step_records)
    final_latents = torch.load(latents_dir / f"step_{final_step:03d}.pt")
    torch.save(final_latents, out_dir / "final_latents.pt")

    video_path = out_dir / f"wan22_ti2v5b_t2v_seed{run_cfg.seed}.mp4"
    export_to_video(result.frames[0], str(video_path), fps=FPS)

    tf_cfg = dict(pipe.transformer.config)
    vae_cfg = dict(pipe.vae.config)
    metadata = {
        "model_id": MODEL_ID,
        "task": "t2v",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "negative_prompt_note": (
            "WanPipeline's signature default is None (resolved to '' by encode_prompt); "
            "this run uses the canonical Wan negative prompt from pipeline_wan.py's "
            "EXAMPLE_DOC_STRING, recorded verbatim above."
        ),
        "height": run_cfg.height,
        "width": run_cfg.width,
        "num_frames": run_cfg.num_frames,
        "seed": run_cfg.seed,
        "generator": f"torch.Generator(device='cpu').manual_seed({run_cfg.seed})",
        "fps_export": FPS,
        "torch_dtype": "bfloat16",
        "mode": mode,
        # The flag that makes this oracle comparable to the server (see docstring).
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        # Resolved from the pipeline call (the defaults were not overridden).
        "num_inference_steps": len(step_records),
        "guidance_scale": float(pipe.guidance_scale),
        "guidance_scale_2": (
            float(pipe._guidance_scale_2) if pipe._guidance_scale_2 is not None else None
        ),
        "max_sequence_length": 512,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "pipeline_config": {
            "expand_timesteps": pipe.config.expand_timesteps,
            # None for TI2V-5B: boundary_ratio is an A14B (MoE dual-DiT) concept.
            "boundary_ratio": pipe.config.get("boundary_ratio"),
            "has_transformer_2": pipe.transformer_2 is not None,
        },
        "transformer_config": {
            k: tf_cfg.get(k)
            for k in [
                "patch_size", "num_attention_heads", "attention_head_dim",
                "in_channels", "out_channels", "text_dim", "freq_dim",
                "ffn_dim", "num_layers", "cross_attn_norm", "qk_norm", "eps",
                "image_dim", "added_kv_proj_dim", "rope_max_seq_len",
                "pos_embed_seq_len",
            ]
        },
        "vae_config": {
            k: vae_cfg.get(k)
            for k in ["z_dim", "scale_factor_temporal", "scale_factor_spatial", "patch_size"]
        },
        "text_encoder_class": pipe.text_encoder.__class__.__name__,
        "tokenizer_class": pipe.tokenizer.__class__.__name__,
        "per_step": step_records,
        "final_latents_shape": list(final_latents.shape),
        "wall_time_s": wall_time,
        "peak_vram_bytes": int(peak_vram),
        "peak_vram_gib": round(peak_vram / 2**30, 2),
        "gpu": gpu_name,
        "python": platform.python_version(),
        # The equivalence suite is only meaningful against an oracle recorded on
        # the SAME torch build as the server under test — record it, loudly.
        "versions": {
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"DONE mode={mode} wall={wall_time:.1f}s peak_vram={peak_vram / 2**30:.2f}GiB")
    print(f"steps={len(step_records)} guidance={metadata['guidance_scale']} "
          f"matmul_precision={metadata['float32_matmul_precision']}")
    print(f"oracle -> {out_dir}")


if __name__ == "__main__":
    main()
