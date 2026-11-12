"""Stage-by-stage parity of the MiniMax-H3-Image port against ``oracle.py``'s recordings.

The port's modules, built from the checkpoint without a server and run on SDPA (the
reference's kernel), against each recorded stage: the prompt's layout (tags, positions),
the seeded noise, the references' VAE latents, Qwen3-VL's layer 50, every DiT step
teacher-forced, the free-running loop, and the decode. Each case is one recorded run
(``<tag>_seed<seed>_run<i>.pt`` next to its job's ``.json``); the models load once.
Exits nonzero when a stage misses its bound; with ``--exact``, also when a compared
tensor is not bit-identical.

    PYTHONPATH=<tree> python test/minimax_h3_image/parity.py --snapshot <MiniMax-H3-Image> \
        --case runs/t2i_feast_2048_seed42_run0.pt --case ... --device cuda:0 --out parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def stats(ours: torch.Tensor, ref: torch.Tensor) -> dict:
    ours, ref = ours.detach().float().cpu(), ref.detach().float().cpu()
    if ours.shape != ref.shape:
        return {"shape_mismatch": [list(ours.shape), list(ref.shape)]}
    diff = (ours - ref).abs()
    return {
        "max_abs": diff.max().item(), "rel_l2": (diff.norm() / ref.norm().clamp_min(1e-30)).item(),
        "equal_frac": (ours == ref).float().mean().item(), "bitexact": bool(torch.equal(ours, ref)),
    }


def psnr_u8(ours: torch.Tensor, ref: torch.Tensor) -> float:
    mse = (ours.float() - ref.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--case", action="append", required=True, type=Path, help="an oracle .pt (its job .json beside)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path)
    p.add_argument("--samples", type=Path, help="write the port's free-running PNGs here")
    p.add_argument("--stages", default="layout,noise,vae_encode,text,dit,loop,vae")
    p.add_argument("--max-step-rel", type=float, default=2e-2, help="bound on a teacher-forced step's rel L2")
    p.add_argument("--max-loop-rel", type=float, default=5e-2, help="bound on the free-running loop's rel L2")
    p.add_argument("--min-psnr", type=float, default=40.0, help="dB, the decode of the reference's latents")
    p.add_argument("--exact", action="store_true", help="also fail on a compared tensor that is not bit-identical")
    p.add_argument("--seed", type=int, help="seed the port with this instead of the job's (a negative control)")
    return p.parse_args()


def main():
    args = parse_args()
    report, failures = {}, []
    try:
        run(args, report, failures)
    except Exception:  # noqa: BLE001 -- keep the stages that ran
        import traceback

        report["crash"] = traceback.format_exc()
        failures.append("crash")
    report["failures"] = failures
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if args.out:
        args.out.write_text(text)
    sys.exit(1 if failures else 0)


def _inexact(entry, key: str = "") -> list[str]:
    if isinstance(entry, list):
        return [k for i, item in enumerate(entry) for k in _inexact(item, f"{key}[{i}]")]
    if not isinstance(entry, dict):
        return []
    own = [key] if entry.get("bitexact") is False or "shape_mismatch" in entry else []
    return own + [k for name, item in entry.items() for k in _inexact(item, f"{key}.{name}" if key else name)]


def run(args, report: dict, failures: list[str]) -> None:
    from PIL import Image

    import mstar
    from mstar.model.minimax_h3.components.layout import patchify_video
    from mstar.model.minimax_h3.components.transformer import H3Inputs
    from mstar.model.minimax_h3.submodules import TEXT_INPUTS, VISION_PIXELS, H3TextEncoderSubmodule
    from mstar.model.minimax_h3_image import weight_loader
    from mstar.model.minimax_h3_image.config import REFERENCE_NOISE_AUG
    from mstar.model.minimax_h3_image.layout import modality_tags, position_ids
    from mstar.model.minimax_h3_image.minimax_h3_image_model import H3ImageModel
    from mstar.model.minimax_h3_image.submodules import (
        IMAGE_OUTPUT,
        REFERENCE_PIXELS,
        H3ImageDecoderSubmodule,
        H3ImageDenoiseSubmodule,
        H3ImageEncoderSubmodule,
        h3_image_schedule,
        h3_image_step,
        layout_from_metadata,
    )

    report["mstar"] = mstar.__file__
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    stages = set(args.stages.split(","))
    dev = torch.device(args.device)
    model = H3ImageModel(model_path_hf=str(args.snapshot))
    config = model.config
    t0 = time.time()
    modules = types.SimpleNamespace()
    if "text" in stages or "dit" in stages or "loop" in stages:
        modules.encoder = weight_loader.build_text_encoder(config, args.snapshot, dev)
        modules.conditioner = weight_loader.build_text_conditioner(config, args.snapshot, dev)
        modules.text = H3TextEncoderSubmodule(
            modules.encoder, modules.conditioner, vision=weight_loader.build_vision_tower(args.snapshot, dev),
            config=config,
        ).to(dev)
    if "dit" in stages or "loop" in stages:
        modules.dit = weight_loader.build_transformer(config, args.snapshot, dev)
        # no engine binds its state: the attention runs SDPA, the reference's kernel
        modules.denoise = H3ImageDenoiseSubmodule(modules.dit, config, loop_name="denoise_loop")
    if "vae_encode" in stages:
        modules.vae_encoder = H3ImageEncoderSubmodule(weight_loader.build_image_encoder(config, args.snapshot, dev))
    if "vae" in stages or "loop" in stages:
        modules.vae = H3ImageDecoderSubmodule(weight_loader.build_image_decoder(config, args.snapshot, dev),
                                              config).to(dev)
    report["load_seconds"] = time.time() - t0
    def run_case(case: Path, entry: dict, bad: list[str]) -> None:
        oracle = torch.load(case, weights_only=False)
        job_path = case.with_name(case.stem.rsplit("_run", 1)[0] + ".json")
        job = json.loads(job_path.read_text())["job"]
        width, height = (int(v) for v in job["size"].lower().split("x"))
        seed = int(job["seed"]) if args.seed is None else args.seed
        kwargs = {"height": height, "width": width, "min_generation_area": int(job["min_area"])}
        prompt = Path(job["prompt_file"]).read_text().strip()
        images = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).permute(2, 0, 1)
                  for p in job["images"]]
        modalities = ["text", "image"] if images else ["text"]
        prompted = model.process_prompt(prompt, modalities, ["image"], tensors={"image_inputs": images}, **kwargs)
        ids = prompted[TEXT_INPUTS][0]
        signals = {name: [types.SimpleNamespace(dims=list(t.shape)) for t in prompted.get(name, [])]
                   for name in (TEXT_INPUTS, VISION_PIXELS, REFERENCE_PIXELS)}
        request = model.walk_plan("", modalities, ["image"], signals, kwargs)[1].step_metadata
        layout = layout_from_metadata(request)
        entry["layout"] = {"rows": layout.total_rows, "text": layout.text_len, "refs": list(layout.references),
                           "canvas": [request["canvas_width"], request["canvas_height"]]}
        n = layout.num_condition_rows

        if "layout" in stages:
            entry["text_len_equal"] = int(ids.shape[0]) == int(oracle["text_token_tags"].shape[0])
            entry["positions_equal"] = bool(torch.equal(position_ids(layout), oracle["position_ids"].double()))
            entry["tags_equal"] = bool(torch.equal(modality_tags(layout), oracle["token_tags"]))
            entry["reference_rows_equal"] = n == int(oracle["num_reference_rows"] or 0)
            sched = h3_image_schedule(config.sigmas)
            entry["timesteps_equal"] = bool(torch.equal(sched.timesteps, oracle["timesteps"].float()))
            bad += [key for key in ("text_len_equal", "positions_equal", "tags_equal", "reference_rows_equal",
                                    "timesteps_equal") if not entry[key]]

        reference_rows = None
        if n:
            reference_rows = torch.cat([patchify_video(r[0][:, None], layout.patch)
                                        for r in oracle["reference_latents"]])
        if "noise" in stages:
            gen = torch.Generator("cpu").manual_seed(seed)
            rows = H3ImageDenoiseSubmodule.seed_loop_back(modules_or_stub(), layout, gen)["latents"]
            if n:
                t = torch.tensor(REFERENCE_NOISE_AUG, dtype=torch.float32)
                rows = torch.cat([t * reference_rows + (1.0 - t) * rows[:n], rows[n:]])
            entry["noise"] = stats(rows, oracle["step_in"][0])
            if not entry["noise"].get("bitexact"):
                bad.append("noise")

        if "vae_encode" in stages and n:
            with torch.no_grad():
                ours = modules.vae_encoder.forward(None, None, prompted[REFERENCE_PIXELS], seed)
            entry["reference_latents"] = stats(ours["cond_latents"][0], reference_rows)

        text_rows = None
        if "text" in stages or "dit" in stages or "loop" in stages:
            with torch.no_grad():
                if images:
                    grids = [tuple(g) for g in request["keyframe_grids"]]
                    pixels = torch.cat([p.flatten(0, -2) for p in prompted[VISION_PIXELS]]).to(dev)
                    hidden = modules.text._encode_with_images(ids.to(dev), pixels, grids)
                else:
                    hidden = modules.encoder(ids[None].to(dev), torch.ones_like(ids[None]).to(dev), padded=False)
                entry["qwen_layer50"] = stats(hidden, oracle["prompt_embeds"])
                # the DiT's text rows from the reference's own layer 50, so the steps compare alone
                text_rows = modules.conditioner(oracle["prompt_embeds"].to(dev, hidden.dtype))
                entry["conditioner_own_vs_reference_input"] = stats(modules.conditioner(hidden), text_rows)

        sched = h3_image_schedule(config.sigmas)

        def velocity(k: int, rows: torch.Tensor) -> torch.Tensor:
            step_layout = modules.denoise.layout(layout, dev)
            t = sched.timesteps[k:k + 1]
            timesteps = torch.stack([t, sched.reference_timesteps[k:k + 1]], dim=-1) if n else t.view(1, 1)
            with torch.no_grad():
                v, _ = modules.dit(H3Inputs(video=rows[None].to(dev), audio=None, text=text_rows,
                                            timesteps=timesteps.to(dev)),
                                   step_layout, head=torch.tensor([k], device=dev))
            return v[0]

        if "dit" in stages:
            per_step = []
            for k in range(sched.num_steps):
                row = stats(velocity(k, oracle["step_in"][k]), oracle["velocity"][k][0])
                per_step.append(row)
                print(f"{case.stem} step {k}: rel {row.get('rel_l2', float('nan')):.3e} "
                      f"bitexact {row.get('bitexact')}", flush=True)
            entry["dit_teacher_forced"] = per_step
            worst = max(r.get("rel_l2", float("inf")) for r in per_step)
            entry["dit_worst_rel"] = worst
            if worst > args.max_step_rel:
                bad.append("dit")

        def decode(rows: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                out = modules.vae.run_batch(rows[None].to(dev), layout)[IMAGE_OUTPUT][0]
            png = model.postprocess(out.cpu(), "image", kwargs)
            import io

            return torch.from_numpy(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).copy())

        oracle_png = torch.from_numpy(np.asarray(Image.open(case.with_suffix(".png")).convert("RGB")).copy())
        if "vae" in stages:
            # the reference's own final rows through the port's decoder
            final_rows = oracle["step_out"][-1]
            ours = decode(final_rows)
            entry["decode_reference_latents"] = {"psnr_db": psnr_u8(ours, oracle_png),
                                                 "bitexact": bool(torch.equal(ours, oracle_png))}
            if not entry["decode_reference_latents"]["psnr_db"] >= args.min_psnr:
                bad.append("vae")

        if "loop" in stages:
            gen = torch.Generator("cpu").manual_seed(seed)
            rows = H3ImageDenoiseSubmodule.seed_loop_back(modules.denoise, layout, gen)["latents"].to(dev)
            if n:
                with torch.no_grad():
                    cond = modules.vae_encoder.forward(None, None, prompted[REFERENCE_PIXELS], seed)
                t = torch.tensor(REFERENCE_NOISE_AUG, dtype=torch.float32, device=dev)
                rows = torch.cat([t * cond["cond_latents"][0].to(dev) + (1.0 - t) * rows[:n], rows[n:]])
            # the port's own text rows from here on
            with torch.no_grad():
                text_rows = modules.conditioner(hidden)
            for k in range(sched.num_steps):
                v = velocity(k, rows)
                sigma, sigma_next = sched.sigmas[k].to(dev), sched.sigmas[k + 1].to(dev)
                rows = torch.cat([rows[:n], h3_image_step(rows[n:], v[n:], sigma, sigma_next)])
            entry["loop_final_rows"] = stats(rows, oracle["step_out"][-1])
            ours = decode(rows)
            entry["loop_image"] = {"psnr_db": psnr_u8(ours, oracle_png),
                                   "bitexact": bool(torch.equal(ours, oracle_png))}
            if args.samples:
                args.samples.mkdir(parents=True, exist_ok=True)
                Image.fromarray(ours.numpy()).save(args.samples / f"{case.stem}_mstar.png")
            if entry["loop_final_rows"].get("rel_l2", 1.0) > args.max_loop_rel:
                bad.append("loop")

    def modules_or_stub():
        stub = H3ImageDenoiseSubmodule.__new__(H3ImageDenoiseSubmodule)
        stub.config = config
        return stub

    for case in args.case:
        entry = report.setdefault(case.stem, {})
        bad: list[str] = []
        try:
            run_case(case, entry, bad)
        except Exception:  # noqa: BLE001
            import traceback

            entry["crash"] = traceback.format_exc()
            bad.append("crash")
        if args.exact:
            bad += [f"not_bitexact:{key}" for key in _inexact(entry)]
        failures += [f"{case.stem}:{name}" for name in bad]


if __name__ == "__main__":
    main()
