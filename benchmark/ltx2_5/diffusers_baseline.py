#!/usr/bin/env python3
"""Diffusers baseline for LTX-2.5 (run in the oracle venv, which has diffusers 0.41).

Sequential calls of ``LTX2Pipeline`` with the model card's distilled recipe, all
components resident on the GPUs (``device_map="balanced"``), after warmup calls.
diffusers is not a server, so its throughput at any concurrency is 1 / latency.

    .venv-ltx-oracle/bin/python benchmark/ltx2_5/diffusers_baseline.py --out <file.json>
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prompts import PROMPTS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=1, help="passes over the prompt set")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--recipe", choices=("single", "two_stage"), default="single",
                   help="two_stage: the card's half-res pass, x2 latent upsampler, 3-sigma full-res tail")
    args = p.parse_args()

    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )

    pipe = LTX2Pipeline.from_pretrained(args.snapshot, prompt_enhancer=None, torch_dtype=torch.bfloat16,
                                        device_map="balanced")
    if args.recipe == "two_stage":
        from diffusers import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        pipe.vae.enable_tiling()
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            args.snapshot, subfolder="latent_upsampler", torch_dtype=torch.bfloat16).to("cuda:0")
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)

    def run(prompt, seed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        shared = dict(
            prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, frame_rate=24.0, guidance_scale=1.0,
            audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0, modality_scale=1.0,
            audio_modality_scale=1.0, generator=torch.Generator("cpu").manual_seed(seed), return_dict=False,
        )
        if args.recipe == "single":
            pipe(height=args.height, width=args.width, num_frames=args.num_frames, sigmas=DISTILLED_SIGMA_VALUES,
                 output_type="np", **shared)
        else:
            video, audio = pipe(height=args.height // 2, width=args.width // 2, num_frames=args.num_frames,
                                sigmas=DISTILLED_SIGMA_VALUES, output_type="latent", **shared)
            up = upsample_pipe(latents=video, output_type="latent", return_dict=False)[0]
            pipe(num_frames=args.num_frames, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, latents=up, audio_latents=audio,
                 noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], output_type="np", **shared)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for i in range(args.warmup):
        run(PROMPTS[i % len(PROMPTS)], 1000 + i)
    latencies = [run(prompt, seed) for _ in range(args.repeats) for seed, prompt in enumerate(PROMPTS)]
    result = {
        "system": "diffusers", "recipe": args.recipe, "height": args.height, "width": args.width,
        "num_frames": args.num_frames,
        "latencies_s": latencies, "mean_s": statistics.mean(latencies), "median_s": statistics.median(latencies),
        "throughput_videos_per_min": 60.0 / statistics.mean(latencies),
        "peak_mem_gb": [round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(torch.cuda.device_count())],
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "latencies_s"}, indent=2))


if __name__ == "__main__":
    main()
