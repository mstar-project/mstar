#!/usr/bin/env python3
"""Run FLUX.2 [klein] end to end through mstar's native modules, without the server.

A development and profiling harness: loads the four node submodules the way
``Flux2KleinModel.get_submodule`` does, runs prompt encoding -> N Euler steps -> VAE
decode for one request, reports per-stage GPU timings (CUDA events, after warmup)
and writes the PNG. Optional ``--oracle-dir`` compares the per-step latents and the
image against a recorded diffusers run (``test/flux2_klein/record_oracle.py``).

    python benchmark/flux2_klein/direct_pipeline.py --prompt "a cat holding a sign" --seed 42 \\
        --attention flashinfer --compile --out cat.png
    python benchmark/flux2_klein/direct_pipeline.py --oracle-dir /path/to/oracle --attention sdpa
    python benchmark/flux2_klein/direct_pipeline.py --prompts-file prompts_100.txt --count 100 \\
        --out-dir sdpa_ref   # one process, seed = prompt index, sdpa_NNN.png (for psnr.py --dirs)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.flux2_klein.config import DENOISE_LOOP, Flux2KleinConfig, resolve_snapshot_dir
from mstar.model.flux2_klein.flux2_klein_model import IMAGE_GEN_WALK, Flux2KleinModel
from mstar.model.flux2_klein.submodules import LATENTS, TEXT_EMBEDS
from mstar.model.submodule_base import ModelInputsFromEngine


class _Timer:
    """CUDA-event stage timer (ms)."""

    def __init__(self):
        self.stages: dict[str, list[float]] = {}

    def time(self, name: str, fn, *args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        end.synchronize()
        self.stages.setdefault(name, []).append(start.elapsed_time(end))
        return out


def _fwd_info(height, width, steps, seed, k):
    return CurrentForwardPassInfo(
        request_id="direct", graph_walk=IMAGE_GEN_WALK, fwd_index=k, random_seed=seed, max_tokens=0,
        step_metadata={"height": height, "width": width, "num_inference_steps": steps, "ref_grids": []},
        dynamic_loop_iter_counts={DENOISE_LOOP: k},
    )


def generate(model, subs, prompt, height, width, steps, seed, timer: _Timer, oracle: Path | None = None):
    text, dit, decoder = subs["text_encoder"], subs["dit"], subs["vae_decoder"]
    engine_inputs = ModelInputsFromEngine(request_ids=["direct"], per_request_info={})
    ids, mask = model.tokenize(prompt)
    with torch.inference_mode():
        embeds = timer.time("text_encoder", text.forward, IMAGE_GEN_WALK, engine_inputs,
                            text_inputs=ids[None], text_mask=mask[None])[TEXT_EMBEDS][0]
        inputs = {TEXT_EMBEDS: [embeds]}
        latents = None
        for k in range(steps):
            node_inputs = dit.prepare_inputs(IMAGE_GEN_WALK, _fwd_info(height, width, steps, seed, k),
                                             {**inputs, **({LATENTS: [latents]} if latents is not None else {})})
            kwargs = dit.preprocess(IMAGE_GEN_WALK, engine_inputs, [node_inputs])
            latents = timer.time(f"dit_step_{k}", dit.forward, IMAGE_GEN_WALK, engine_inputs, **kwargs)[LATENTS][0]
            if oracle is not None:
                expected = torch.load(oracle / "t2i" / f"latents_step_{k:03d}.pt")[0].to(latents.device)
                print(f"  step {k}: max_abs vs oracle {(latents.float() - expected.float()).abs().max().item():.3e}")
        image = timer.time("vae_decoder", decoder.forward, IMAGE_GEN_WALK, engine_inputs, latents=latents[None],
                           grid=model.config.latent_grid(height, width))["image_output"][0]
    dit.cleanup_request("direct")
    return image[0]


def generate_many(model, subs, args) -> None:
    """Prompts ``[seed_start, seed_start + count)`` of ``--prompts-file`` with seed = index, as the
    benchmark client requests them (``--seed 0``), written as ``<out-dir>/<prefix>_NNN.png``."""
    from mstar.model.components.diffusion.image_io import uint8_to_png

    prompts = Path(args.prompts_file).read_text().splitlines()
    indices = range(args.seed_start, args.seed_start + args.count)
    if indices.stop > len(prompts):
        raise SystemExit(f"{args.prompts_file} has {len(prompts)} prompts, need {indices.stop}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generate(model, subs, prompts[indices.start], args.height, args.width, args.steps, indices.start, _Timer())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in indices:
        image = generate(model, subs, prompts[i], args.height, args.width, args.steps, i, _Timer())
        (out_dir / f"{args.out_prefix}_{i:03d}.png").write_bytes(uint8_to_png(image.cpu()))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"wrote {len(indices)} images to {out_dir} in {wall:.1f}s ({wall / len(indices) * 1000:.0f} ms/image)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--prompt", default="A cat holding a sign that says hello world, studio lighting, detailed fur")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attention", choices=["sdpa", "flashinfer"], default="sdpa",
                    help="only sdpa here: flashinfer needs the engine's ragged resource (server path)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--oracle-dir", default="")
    ap.add_argument("--out", default="direct.png")
    ap.add_argument("--prompts-file", default="",
                    help="one prompt per line: generate prompts [seed-start, seed-start + count) with seed = "
                         "index into --out-dir in this one process (the reference side of a PSNR distribution)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out-dir", default="direct_out")
    ap.add_argument("--out-prefix", default="sdpa", help="file names <prefix>_NNN.png")
    args = ap.parse_args()
    if args.attention != "sdpa":
        raise SystemExit("direct_pipeline runs the modules without the engine, so only --attention sdpa is available")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = Flux2KleinModel(model_path_hf=args.repo, attention_backend="sdpa", compile=args.compile, cuda_graph=False)
    model.set_config(Flux2KleinConfig.from_snapshot(resolve_snapshot_dir(args.repo)))
    t0 = time.perf_counter()
    subs = {name: model.get_submodule(name, device=device) for name in ("text_encoder", "dit", "vae_decoder")}
    torch.cuda.synchronize()
    print(f"loaded in {time.perf_counter() - t0:.1f}s; peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    oracle = Path(args.oracle_dir) if args.oracle_dir else None
    if oracle is not None:
        meta = json.load(open(oracle / "metadata.json"))
        args.prompt, args.height, args.width, args.steps, args.seed = (
            meta["prompt"], meta["height"], meta["width"], meta["steps"], meta["seed"])

    if args.prompts_file:
        generate_many(model, subs, args)
        return

    for _ in range(args.warmup):
        generate(model, subs, args.prompt, args.height, args.width, args.steps, args.seed, _Timer())
    timer = _Timer()
    walls = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = generate(model, subs, args.prompt, args.height, args.width, args.steps, args.seed, timer, oracle)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    print(f"end-to-end wall: median {statistics.median(walls) * 1000:.1f} ms over {args.repeats} runs "
          f"(peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)")
    for name, samples in timer.stages.items():
        print(f"  {name:14s} median {statistics.median(samples):7.2f} ms")
    from mstar.model.components.diffusion.image_io import uint8_to_png

    Path(args.out).write_bytes(uint8_to_png(image.cpu()))
    print(f"wrote {args.out}")
    if oracle is not None:
        import numpy as np
        from PIL import Image

        ref = torch.from_numpy(np.array(Image.open(oracle / "t2i" / "image.png").convert("RGB"))).permute(2, 0, 1)
        mse = (image.cpu().float() - ref.float()).pow(2).mean().item()
        psnr = float("inf") if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)
        print(f"PSNR vs oracle image: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
