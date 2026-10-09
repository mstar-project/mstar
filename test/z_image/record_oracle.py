#!/usr/bin/env python3
"""Record the diffusers golden-reference oracle for Z-Image-Turbo text-to-image, for
``test/modular/test_z_image_reference_equivalence.py``.

Runs the stock diffusers ``ZImagePipeline`` — never any mstar code — and writes the caption
embeddings, the seeded initial fp32 latents, the fp32 latents after every Euler step, the
final PNG and a metadata.json. Same numerics rules as the klein recorder: same environment
and GPU as the suite, ``NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`` for both,
noise from a CPU generator.

    CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        python test/z_image/record_oracle.py --out-dir /path/to/oracle
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

PROMPT = "A cozy bookshop on a rainy evening, warm lamplight in the window, a cat asleep on a stack of books"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=PROMPT, help="caption to record (e.g. a long one beyond the captured buckets)")
    args = ap.parse_args()

    import diffusers
    import transformers
    from diffusers import ZImagePipeline

    torch.set_float32_matmul_precision("high")
    out = Path(args.out_dir) / "t2i"
    out.mkdir(parents=True, exist_ok=True)
    pipe = ZImagePipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16).to("cuda")

    latents = pipe.prepare_latents(
        1, pipe.transformer.in_channels, args.height, args.width, torch.float32, torch.device("cuda"),
        torch.Generator(device="cpu").manual_seed(args.seed),
    )
    torch.save(latents.cpu(), out / "latents_init.pt")
    captured: dict[str, torch.Tensor] = {}

    def on_step_end(pipeline, i, t, kwargs):
        captured[f"latents_step_{i:03d}"] = kwargs["latents"].detach().clone().cpu()
        if "prompt_embeds" not in captured:
            captured["prompt_embeds"] = kwargs["prompt_embeds"][0].detach().clone().cpu()
        return {}

    result = pipe(
        prompt=args.prompt, height=args.height, width=args.width, num_inference_steps=args.steps, guidance_scale=0.0,
        generator=torch.Generator(device="cpu").manual_seed(args.seed), callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )
    for name, tensor in captured.items():
        torch.save(tensor, out / f"{name}.pt")
    result.images[0].save(out / "image.png")
    meta = {
        "repo": args.repo, "prompt": args.prompt, "height": args.height, "width": args.width, "steps": args.steps,
        "seed": args.seed, "generator_device": "cpu", "dtype": "bfloat16", "guidance_scale": 0.0,
        "torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    with open(Path(args.out_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"oracle written to {args.out_dir}")


if __name__ == "__main__":
    main()
