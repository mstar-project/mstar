#!/usr/bin/env python3
"""Record the diffusers golden-reference oracle for FLUX.2 [klein] (text-to-image and one
single-reference edit), for ``test/modular/test_flux2_klein_reference_equivalence.py``.

Runs the **stock diffusers ``Flux2KleinPipeline``** — never any mstar code — on the
checkpoint, and writes:

    <out-dir>/t2i/prompt_embeds.pt              text encoder output [1, 512, joint_dim] (bf16)
    <out-dir>/t2i/latents_init.pt               the seeded initial packed latents (bf16)
    <out-dir>/t2i/latents_step_NNN.pt           packed latents AFTER each Euler step
    <out-dir>/t2i/image.png                     the reference image
    <out-dir>/edit/...                          same for the edit run (ref image = the t2i image)
    <out-dir>/metadata.json                     every resolved parameter + versions

Why the numerics flags matter: an oracle is a reference only under the numerics of the
serving process. mstar's engine sets ``torch.set_float32_matmul_precision("high")``
process-wide, and cuBLAS GEMM algorithm selection is workspace- and history-dependent,
so record with the SAME python environment, torch build and GPU as the suite, and set
``NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`` for both. The initial noise is
drawn from a CPU generator (``torch.Generator("cpu").manual_seed(seed)``) so it is
device independent and equals what the mstar dit seeds for the same request seed.

    CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        python test/flux2_klein/record_oracle.py --out-dir /path/to/oracle
    FLUX2_KLEIN_ORACLE_DIR=/path/to/oracle CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0 \\
        CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest test/modular/test_flux2_klein_reference_equivalence.py -v
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

PROMPT = "A cat holding a sign that says hello world, studio lighting, detailed fur"
EDIT_PROMPT = "Make the sign say goodbye and turn the scene into a watercolor painting"
# extra references for --refs N (generated text-to-image, one per entry; the second is non-square on purpose)
REF_PROMPTS = (
    ("A cozy cabin in a snowy forest at night, warm light in the windows", 768, 1024),
    ("A red bicycle leaning against a yellow wall in Lisbon, afternoon sun", 1024, 1024),
    ("Macro photograph of a dew-covered spider web at dawn", 1024, 768),
)
MULTI_EDIT_PROMPT = "Put the cat with its sign from the first image in front of the scene from the second image"


def _record_run(pipe, out_dir: Path, *, prompt: str, image, height: int, width: int, steps: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    captured: dict[str, torch.Tensor] = {}

    def on_step_end(pipeline, i, t, kwargs):
        captured[f"latents_step_{i:03d}"] = kwargs["latents"].detach().clone().cpu()
        if i == 0 and "prompt_embeds" not in captured:
            captured["prompt_embeds"] = kwargs["prompt_embeds"].detach().clone().cpu()
        return {}

    result = pipe(
        prompt=prompt, image=image, height=height, width=width, num_inference_steps=steps, guidance_scale=1.0,
        generator=generator, callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
    )
    for name, tensor in captured.items():
        torch.save(tensor, out_dir / f"{name}.pt")
    result.images[0].save(out_dir / "image.png")
    return result.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edit-seed", type=int, default=7)
    ap.add_argument("--skip-edit", action="store_true")
    ap.add_argument("--refs", type=int, default=1,
                    help="references for an additional multi-reference edit (edit_multi/): the t2i image plus refs-1 "
                         "generated ones (REF_PROMPTS), edit seed = --edit-seed + 1")
    args = ap.parse_args()
    if not 1 <= args.refs <= 1 + len(REF_PROMPTS):
        raise SystemExit(f"--refs must be in [1, {1 + len(REF_PROMPTS)}]")

    import diffusers
    import transformers
    from diffusers import Flux2KleinPipeline

    # The same matmul precision the serving engine sets (mstar/engine/__init__.py).
    torch.set_float32_matmul_precision("high")
    out = Path(args.out_dir)
    pipe = Flux2KleinPipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16).to("cuda")

    # the initial noise, drawn exactly as the pipeline does, for a direct check
    latents, _ = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels // 4, args.height, args.width, torch.bfloat16, torch.device("cuda"),
        torch.Generator(device="cpu").manual_seed(args.seed),
    )
    (out / "t2i").mkdir(parents=True, exist_ok=True)
    torch.save(latents.cpu(), out / "t2i" / "latents_init.pt")
    image = _record_run(pipe, out / "t2i", prompt=PROMPT, image=None, height=args.height, width=args.width,
                        steps=args.steps, seed=args.seed)
    meta = {
        "repo": args.repo, "prompt": PROMPT, "height": args.height, "width": args.width, "steps": args.steps,
        "seed": args.seed, "generator_device": "cpu", "dtype": "bfloat16",
        "torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if not args.skip_edit:
        (out / "edit").mkdir(parents=True, exist_ok=True)
        _record_run(pipe, out / "edit", prompt=EDIT_PROMPT, image=image, height=args.height, width=args.width,
                    steps=args.steps, seed=args.edit_seed)
        meta.update({"edit_prompt": EDIT_PROMPT, "edit_seed": args.edit_seed, "edit_reference": "t2i/image.png"})
    if not args.skip_edit and args.refs > 1:
        references, paths = [image], ["t2i/image.png"]
        for i, (ref_prompt, h, w) in enumerate(REF_PROMPTS[: args.refs - 1], start=1):
            ref = _record_run(pipe, out / "refs" / f"ref_{i}", prompt=ref_prompt, image=None, height=h, width=w,
                              steps=args.steps, seed=args.seed + 100 + i)
            references.append(ref)
            paths.append(f"refs/ref_{i}/image.png")
        _record_run(pipe, out / "edit_multi", prompt=MULTI_EDIT_PROMPT, image=references, height=args.height,
                    width=args.width, steps=args.steps, seed=args.edit_seed + 1)
        meta.update({"edit_multi_prompt": MULTI_EDIT_PROMPT, "edit_multi_seed": args.edit_seed + 1,
                     "edit_multi_references": paths})
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"oracle written to {out}")


if __name__ == "__main__":
    main()
