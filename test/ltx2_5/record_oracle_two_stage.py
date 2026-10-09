#!/usr/bin/env python3
"""Record the diffusers oracle for LTX-2.5's two-stage distilled recipe (model card):
stage 1 at half resolution with the 8 distilled sigmas, the x2 latent upsampler, then
the 3-sigma stage 2 at full resolution, one CPU generator across all calls, tiled
VAE decode. Oracle venv; see ``record_oracle.py`` for the numerics flags.

Writes to ``<out-dir>/<case>/``: ``s1_*`` / ``s2_*`` DiT inputs and outputs per step,
``s1_latents_init.pt`` / ``s1_audio_init.pt``, the stage-1 outputs (``s1_out_video.pt``
denormalized ``[1, 128, F, H, W]``, ``s1_out_audio.pt`` ``[1, 8, L, 16]``), the upsampler
output ``upsampled.pt``, stage 2's noised starting latents ``s2_latents_init.pt`` /
``s2_audio_init.pt``, and ``video.pt`` / ``audio_wave.pt`` / ``out.mp4``.

    .venv-ltx-oracle/bin/python test/ltx2_5/record_oracle_two_stage.py --out-dir <dir> --snapshot <snap>
"""
import argparse
import json
from pathlib import Path

import torch
from record_oracle import PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--case", default="two_stage")
    p.add_argument("--height", type=int, default=1088, help="final (stage 2) height")
    p.add_argument("--width", type=int, default=1920, help="final (stage 2) width")
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sdpa-backend", choices=("default", "efficient"), default="default",
                   help="force SDPA's memory-efficient kernel: a second, equally valid reference run")
    args = p.parse_args()
    if args.sdpa_backend == "efficient":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return run(args)
    return run(args)


def run(args):

    from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
    from diffusers.utils import encode_video

    out = Path(args.out_dir) / args.case
    out.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        torch.save(value, out / f"{name}.pt")

    pipe = LTX2Pipeline.from_pretrained(args.snapshot, prompt_enhancer=None, torch_dtype=torch.bfloat16,
                                        device_map="balanced")
    pipe.vae.enable_tiling()
    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        args.snapshot, subfolder="latent_upsampler", torch_dtype=torch.bfloat16).to("cuda:0")
    upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)

    stage = {"name": "s1", "i": 0}
    dit_forward = pipe.transformer.forward

    def dit(*a, **kw):
        tag = f"{stage['name']}_dit_{stage['i']:03d}"
        save(f"{tag}_in", {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kw.items()
                           if k != "attention_kwargs"})
        result = dit_forward(*a, **kw)
        save(f"{tag}_out", (result[0].float().cpu(), result[1].float().cpu()))
        stage["i"] += 1
        return result

    pipe.transformer.forward = dit
    for fn_name, what in (("prepare_latents", "latents_init"), ("prepare_audio_latents", "audio_init")):
        original = getattr(pipe, fn_name)

        def wrapped(*a, _original=original, _what=what, **kw):
            result = _original(*a, **kw)
            save(f"{stage['name']}_{_what}", result.cpu())
            return result

        setattr(pipe, fn_name, wrapped)

    generator = torch.Generator("cpu").manual_seed(args.seed)
    shared = dict(
        prompt=PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT, frame_rate=args.fps,
        guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0,
        modality_scale=1.0, audio_modality_scale=1.0, generator=generator, return_dict=False,
    )
    s1_video, s1_audio = pipe(
        height=args.height // 2, width=args.width // 2, num_frames=args.num_frames,
        sigmas=DISTILLED_SIGMA_VALUES, output_type="latent", **shared,
    )
    save("s1_out_video", s1_video.cpu())
    save("s1_out_audio", s1_audio.cpu())
    upsampled = upsample_pipe(latents=s1_video, output_type="latent", return_dict=False)[0]
    save("upsampled", upsampled.cpu())
    stage.update(name="s2", i=0)
    video, audio = pipe(
        num_frames=args.num_frames, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, latents=upsampled,
        audio_latents=s1_audio, noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0], output_type="np", **shared,
    )
    frames = torch.from_numpy((video[0] * 255).round().clip(0, 255).astype("uint8"))
    save("video", frames)
    save("audio_wave", audio[0].float().cpu())
    encode_video(video[0], fps=args.fps, audio=audio[0].float().cpu(),
                 audio_sample_rate=pipe.vocoder.config.output_sampling_rate, output_path=str(out / "out.mp4"))
    meta = {"prompt": PROMPT, "height": args.height, "width": args.width, "num_frames": args.num_frames,
            "fps": args.fps, "seed": args.seed, "frames": list(frames.shape)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
