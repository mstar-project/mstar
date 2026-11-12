"""Stage-by-stage parity of the MiniMax-H3 port against ``oracle.py``'s recording.

The port's modules, built from the checkpoint without a server and run on SDPA (the
reference's kernel), against each recorded stage: token ids, the packed layout, the
seeded noise, Qwen3-VL's layer 50, the text conditioner, every DiT step teacher-forced,
the free-running loop and the decodes. An FL2VA recording adds the keyframes' pixels,
latents and step-0 noising. Exits nonzero when a stage misses its bound (``check_bounds``);
with ``--exact``, also when any compared tensor is not bit-identical (the docs' claim).

    PYTHONPATH=<tree> python test/minimax_h3/parity.py --snapshot <MiniMax-H3> \
        --oracle /tmp/h3_oracle/oracle.pt --device cuda:1 --out /tmp/h3_parity.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def _hide_broken_torchaudio() -> None:
    """An ABI-mismatched torchaudio (built for another torch) fails to load its library;
    transformers would import it on the way to Qwen3-VL. Nothing here uses it."""
    import sys

    try:
        import torchaudio  # noqa: F401
    except ImportError:
        pass  # not installed
    except OSError:
        sys.modules["torchaudio"] = None


def stats(ours: torch.Tensor, ref: torch.Tensor) -> dict:
    ours, ref = ours.detach().float().cpu(), ref.detach().float().cpu()
    if ours.shape != ref.shape:
        return {"shape_mismatch": [list(ours.shape), list(ref.shape)]}
    diff = (ours - ref).abs()
    rel = (diff.norm() / ref.norm().clamp_min(1e-30)).item()
    return {
        "max_abs": diff.max().item(), "rel_l2": rel, "equal_frac": (ours == ref).float().mean().item(),
        "bitexact": bool(torch.equal(ours, ref)),
    }


def psnr(ours: torch.Tensor, ref: torch.Tensor, peak: float = 1.0) -> float:
    mse = (ours.float().cpu() - ref.float().cpu()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)


def main():
    """Runs the stages; the report is written (and the exit code set) whatever stage fails."""
    _hide_broken_torchaudio()
    report: dict = {}
    failures: list[str] = []
    args = parse_args()
    try:
        run(args, report, failures)
    except Exception:  # noqa: BLE001 -- keep the stages that ran
        import traceback

        report["crash"] = traceback.format_exc()
        failures.append("crash")
    failures += check_bounds(report, args)
    report["failures"] = failures
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if args.out:
        args.out.write_text(text)
    sys.exit(1 if failures else 0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--oracle", required=True, type=Path)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", type=Path)
    p.add_argument("--stages", default="layout,noise,vae_encode,text,dit,loop,vae,audio")
    p.add_argument("--max-step-rel", type=float, default=2e-2, help="bound on a teacher-forced step's rel L2")
    p.add_argument("--max-latent-rel", type=float, default=1e-2, help="bound on the keyframe latents' rel L2")
    p.add_argument("--max-loop-rel", type=float, default=5e-2,
                   help="bound on the free-running loop's final latents' rel L2, each mode")
    p.add_argument("--min-psnr", type=float, default=40.0, help="dB, the video decode of the reference's latents")
    p.add_argument("--min-snr", type=float, default=40.0, help="dB, the audio decode of the reference's latents")
    p.add_argument("--exact", action="store_true", help="also fail on any compared tensor that is not bit-identical")
    return p.parse_args()


def check_bounds(report: dict, args) -> list[str]:
    """The stages ``run`` records without judging them: the keyframe latents, the
    free-running loop, and the decodes of the reference's own latents (the decodes of the
    port's latents follow the loop's). A stage that didn't run is not checked; one that
    crashed already failed."""
    failures = []

    def rel(entry):
        return entry.get("rel_l2", float("inf")) if isinstance(entry, dict) else float("inf")

    if "keyframe_latents" in report and rel(report["keyframe_latents"]) > args.max_latent_rel:
        failures.append("vae_encode")
    for mode in ("serving", "reference_shapes"):
        keys = [f"loop_{kind}_latents_{mode}" for kind in ("video", "audio")]
        if any(key in report for key in keys) and max(rel(report.get(key)) for key in keys) > args.max_loop_rel:
            failures.append(f"loop_{mode}")
    video = report.get("video_decode", {}).get("oracle_latents")
    if video is not None and not video["psnr_db"] >= args.min_psnr:
        failures.append("video_decode")
    audio = report.get("audio_decode", {}).get("oracle_latents")
    if audio is not None and not audio["snr_db"] >= args.min_snr:
        failures.append("audio_decode")
    if args.exact:
        failures += [f"not_bitexact:{key}" for key in _inexact(report)]
    return failures


def _inexact(entry, key: str = "") -> list[str]:
    """Every compared tensor in the report that isn't bit-identical, by its path."""
    if isinstance(entry, list):
        return [k for i, item in enumerate(entry) for k in _inexact(item, f"{key}[{i}]")]
    if not isinstance(entry, dict):
        return []
    own = []
    if entry.get("bitexact") is False or "shape_mismatch" in entry or entry.get("equal_frac", 1.0) < 1.0:
        own = [key]
    return own + [k for name, item in entry.items() for k in _inexact(item, f"{key}.{name}" if key else name)]


def run(args, report: dict, failures: list[str]) -> None:

    import mstar
    from mstar.engine.resources.sequence_shard import SequenceShard
    from mstar.engine.resources.step import StepContext
    from mstar.engine.state.adapters import node_handles, state_setup
    from mstar.model.components.diffusion.denoise_loop import StepKey
    from mstar.model.minimax_h3 import weight_loader
    from mstar.model.minimax_h3.components.layout import (
        modality_tags,
        patchify_video,
        position_ids,
        unpatchify_video,
    )
    from mstar.model.minimax_h3.components.transformer import H3Inputs
    from mstar.model.minimax_h3.config import (
        DIT_ATTN,
        DIT_NODE,
        DIT_SHARD,
        KEYFRAME_NOISE_AUG,
        REFINER_ATTN,
        TEXT_ENCODER_NODE,
    )
    from mstar.model.minimax_h3.keyframes import ANCHORS, vision_spans
    from mstar.model.minimax_h3.minimax_h3_model import H3Model
    from mstar.model.minimax_h3.schedule import h3_schedule, h3_step
    from mstar.model.minimax_h3.submodules import (
        KEYFRAME_PIXELS,
        TEXT_INPUTS,
        VISION_PIXELS,
        H3DenoiseSubmodule,
        H3TextEncoderSubmodule,
        H3VaeEncoderSubmodule,
        H3VideoDecoderSubmodule,
    )
    from mstar.model.submodule_base import NodeInputs, engine_step

    print("mstar from", mstar.__file__, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    oracle = torch.load(args.oracle, weights_only=False)
    a = oracle["args"]
    stages = set(args.stages.split(","))
    model = H3Model(model_path_hf=str(args.snapshot))
    config = model.config
    # every attention through its declared read, bound as `backend: sdpa` on its resource binds it
    on_sdpa, problems = state_setup({"resources": {key: {"backend": "sdpa"} for key in (DIT_ATTN, REFINER_ATTN)}},
                                    model.declare_state())
    assert not problems, problems
    kwargs = {"num_frames": int(a["num_frames"]), "num_inference_steps": int(a["steps"])}
    if a.get("height") not in (None, "None"):
        kwargs.update(height=int(a["height"]), width=int(a["width"]))
    report["mstar"] = mstar.__file__
    images, anchors = [], []
    for name, anchor in (("image", "first"), ("last_image", "last")):
        if a.get(name) not in (None, "None"):
            from PIL import Image

            image = Image.open(a[name]).convert("RGB")
            images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
            anchors.append(anchor)
    vision_pixels = keyframe_pixels = None
    if images:
        prompted = model.process_prompt(a["prompt"], ["text", "image"], ["video", "audio"],
                                        tensors={"image_inputs": images}, keyframe_anchors=anchors, **kwargs)
        ids, vision_pixels = prompted[TEXT_INPUTS][0], prompted[VISION_PIXELS][0]
        keyframe_pixels = prompted[KEYFRAME_PIXELS][0]
        kwargs.update(height=int(keyframe_pixels.shape[-2]), width=int(keyframe_pixels.shape[-1]))
        ref_inputs = oracle["conditioner_inputs"]
        report["token_ids_equal"] = bool(torch.equal(ids, ref_inputs["input_ids"][0]))
        # the conditioner call records the pixels after its cast to the tower's dtype
        report["vision_pixels"] = stats(vision_pixels.to(ref_inputs["pixel_values"].dtype), ref_inputs["pixel_values"])
    else:
        ids = model.tokenize(a["prompt"])
        report["token_ids_equal"] = bool(torch.equal(ids, oracle["token_ids"]))
    request = model.resolve_request(kwargs)
    report["request"] = request
    layout = model.request_layout(request, int(ids.shape[0]))
    if images:
        packed = tuple(x for x in ANCHORS if x in anchors)
        tokens = model.keyframe_tokens(request["height"], request["width"])
        layout = dataclasses.replace(layout, anchors=packed, text_video_spans=tuple(
            vision_spans(model._ensure_tokenizer(), [tokens] * len(packed))))
        grids = [(1, request["height"] // 16, request["width"] // 16)] * len(packed)
    report["layout"] = {"rows": layout.total_rows, "video_rows": layout.num_video_rows,
                        "audio_rows": layout.num_audio_rows, "text": layout.text_len}
    step0 = oracle["steps"][0]
    dev = torch.device(args.device)

    if "layout" in stages:
        pos = position_ids(layout)
        report["positions_equal"] = bool(torch.equal(pos, step0["position_ids"].double().cpu()))
        tags = modality_tags(layout)
        report["tags_equal"] = bool(torch.equal(tags, step0["token_tags"]))
        report["video_indices_equal"] = bool(torch.equal(torch.cat([
            torch.arange(layout.condition_start, layout.audio_start),
            torch.arange(layout.video_start, layout.total_rows),
        ]), step0["video_indices"]))
        report["audio_indices_equal"] = bool(torch.equal(
            torch.arange(layout.audio_start, layout.video_start), step0["audio_indices"]))
        sched = h3_schedule(request["num_inference_steps"], request["video_shift"], request["audio_shift"])
        ts = [s["timestep"] for s in oracle["steps"]]
        report["schedule_steps"] = [sched.num_steps, len(ts)]
        def distinct(k):
            values = [sched.timesteps[k], sched.audio_timesteps[k]]
            return torch.unique(torch.stack(values + ([sched.cond_timesteps[k]] if layout.num_condition_rows else [])))

        ok = all(torch.equal(distinct(k), ts[k]) for k in range(len(ts)))
        report["timesteps_equal"] = ok
        for key in ("token_ids_equal", "positions_equal", "tags_equal", "video_indices_equal",
                    "audio_indices_equal", "timesteps_equal"):
            if key in report and not report[key]:
                failures.append(key)

    if "noise" in stages:
        stub = H3DenoiseSubmodule.__new__(H3DenoiseSubmodule)
        stub.config = config
        gen = torch.Generator("cpu").manual_seed(int(a["seed"]))
        seeds = H3DenoiseSubmodule.seed_loop_back(stub, layout, gen)
        n = layout.num_condition_rows
        rows = seeds["latents"]
        if n:
            # the keyframe rows noised as the dit does at step 0, from the reference's own latents
            cond = torch.cat([patchify_video(c[0], layout.patch) for c in oracle["condition_latents"]])
            t = torch.tensor(KEYFRAME_NOISE_AUG, dtype=torch.float32)
            rows = torch.cat([t * cond + (1.0 - t) * rows[:n], rows[n:]])
        report["noise_video"] = stats(rows, step0["hidden_states"][0])
        report["noise_audio"] = stats(seeds["audio_latents"], step0["audio_hidden_states"][0])
        if not (report["noise_video"].get("bitexact") and report["noise_audio"].get("bitexact")):
            failures.append("noise")

    if "vae_encode" in stages and images:
        encoder = H3VaeEncoderSubmodule(weight_loader.build_keyframe_encoder(config, args.snapshot, dev)).to(dev)
        with torch.no_grad():
            ours = encoder.forward(None, None, keyframe_pixels, layout)["cond_latents"][0]
        want = torch.cat([patchify_video(c[0], layout.patch) for c in oracle["condition_latents"]])
        report["keyframe_latents"] = stats(ours, want)
        del encoder
        torch.cuda.empty_cache()

    text_rows = oracle["refined_text"]
    if "text" in stages:
        t0 = time.time()
        encoder = weight_loader.build_text_encoder(config, args.snapshot, dev)
        conditioner = weight_loader.build_text_conditioner(config, args.snapshot, dev)
        prompt_handles = node_handles(model, TEXT_ENCODER_NODE, {}, setup=on_sdpa)
        for module in conditioner.modules():
            if hasattr(module, "bind_state"):
                module.bind_state(prompt_handles)
        with torch.no_grad():
            if images:
                text_node = H3TextEncoderSubmodule(
                    encoder, conditioner, vision=weight_loader.build_vision_tower(args.snapshot, dev),
                ).to(dev)
                hidden = text_node._encode_with_images(ids.to(dev), vision_pixels.to(dev), grids)
            else:
                hidden = encoder(ids[None].to(dev), torch.ones_like(ids[None]).to(dev))
            report["qwen_layer50"] = stats(hidden, oracle["prompt_embeds"])
            refined_from_ref = conditioner(oracle["prompt_embeds"].to(dev))
            report["conditioner_teacher_forced"] = stats(refined_from_ref, text_rows)
            refined = conditioner(hidden)
            report["conditioner"] = stats(refined, text_rows)
        report["text_seconds"] = time.time() - t0
        del encoder, conditioner
        torch.cuda.empty_cache()
        if report["conditioner"].get("rel_l2", 1) > 1e-2:
            failures.append("text")

    loops = {}
    if stages & {"dit", "loop"}:
        dit = weight_loader.build_transformer(config, args.snapshot, dev)
        sub = H3DenoiseSubmodule(dit, config, loop_name="denoise_loop")
        one_rank = SequenceShard(("main",), 0, None, 0, types.SimpleNamespace(world_size=1))
        sub.bind_node_state(node_handles(model, DIT_NODE, {DIT_SHARD: one_rank}, setup=on_sdpa))
        step_layout = sub.build_layout(StepKey(layout), dev)
        sched = h3_schedule(request["num_inference_steps"], request["video_shift"], request["audio_shift"])
        text = text_rows.to(dev)
        plans = {entry["k"]: (entry["timestep"], entry["timestep_indices"]) for entry in oracle["steps"]}

        def forward(k, video, audio, reference_shapes=False):
            """``reference_shapes``: the step's timesteps laid out as the reference lays them
            (its distinct values, sorted, and each row's index into them), so the time MLP and
            the AdaLN GEMMs run at the reference's shapes. Serving runs a fixed (video, audio,
            keyframe) triple, or one timestep at step 0 without keyframes
            (``H3DenoiseSubmodule.request_inputs``)."""
            if reference_shapes:
                unique, indices = plans[k]
                t, row_class = unique[None].to(dev), indices.to(dev)
            elif k == 0 and not layout.num_condition_rows:
                t, row_class = sched.timesteps[:1][None].to(dev), None
            else:
                t = torch.stack([sched.timesteps[k], sched.audio_timesteps[k], sched.cond_timesteps[k]])[None].to(dev)
                row_class = None
            step = engine_step(sub, "gen", [0], [NodeInputs(resource_step_info=StepKey(layout))])
            step.set_ctx(StepContext(request_ids=[0], graph_walk="gen", slot=0, capture=False))
            step.begin_calls()
            with torch.no_grad():
                out = dit(H3Inputs(video=video, audio=audio, text=text, timesteps=t), step_layout, row_class=row_class)
            step.end_calls()
            return out

        if "dit" in stages:
            for mode, exact in (("serving", False), ("reference_shapes", True)):
                per_step = []
                for entry in oracle["steps"]:
                    if "noise_pred" not in entry:
                        continue
                    k = entry["k"]
                    v, au = forward(k, entry["hidden_states"].to(dev), entry["audio_hidden_states"].to(dev), exact)
                    row = {"k": k, "video": stats(v, entry["noise_pred"]),
                           "audio": stats(au, entry["audio_noise_pred"])}
                    per_step.append(row)
                    print(f"{mode} step {k}: video rel {row['video']['rel_l2']:.3e} audio rel "
                          f"{row['audio']['rel_l2']:.3e}", flush=True)
                report[f"dit_teacher_forced_{mode}"] = per_step
                worst = max(max(r["video"]["rel_l2"], r["audio"]["rel_l2"]) for r in per_step)
                report[f"dit_worst_rel_{mode}"] = worst
                if worst > args.max_step_rel:
                    failures.append(f"dit_{mode}")

        if "loop" in stages:
            for mode, exact in (("serving", False), ("reference_shapes", True)):
                video = step0["hidden_states"].to(dev).float()
                audio = step0["audio_hidden_states"].to(dev).float()
                n = layout.num_condition_rows
                for k in range(sched.num_steps):
                    v, au = forward(k, video, audio, exact)

                    def col(x, kk=k):
                        return x[kk:kk + 1].to(dev).view(-1, 1, 1)

                    stepped = h3_step(video[:, n:], v[:, n:], col(sched.timesteps), col(sched.sigmas),
                                      col(sched.sigmas, k + 1))
                    video = torch.cat([video[:, :n], stepped], dim=1)
                    audio = h3_step(audio, au, col(sched.audio_timesteps), col(sched.audio_sigmas),
                                    col(sched.audio_sigmas, k + 1))
                loops[mode] = (video, audio)
                final_video = unpatchify_video(video[:, layout.num_condition_rows:], layout,
                                               config.video_vae.latent_channels)
                final_audio = audio[0].reshape(2, -1, audio.shape[-1]).permute(0, 2, 1)
                report[f"loop_video_latents_{mode}"] = stats(final_video, oracle["latents"])
                report[f"loop_audio_latents_{mode}"] = stats(final_audio, oracle["audio_latents"])
        del dit, sub
        torch.cuda.empty_cache()

    if "vae" in stages and "videos" in oracle:
        vae = weight_loader.build_video_decoder(config, args.snapshot, dev)
        dec = H3VideoDecoderSubmodule(vae, config).to(dev)
        ref = oracle["videos"]
        ref = ref[0] if ref.ndim == 5 else ref  # [F, 3, H, W] in [0, 1]
        ref_u8 = (ref.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8)
        # the decoder drops the keyframe rows that lead the video rows; the reference's latents have none
        sources = {"oracle_latents": torch.cat([
            torch.zeros(1, layout.num_condition_rows, config.transformer.patch_dim),
            patchify_video(oracle["latents"][0], layout.patch)[None],
        ], dim=1)}
        sources.update({f"port_{mode}": video for mode, (video, _) in loops.items()})
        media = {}
        for name, rows in sources.items():
            with torch.no_grad():
                frames = dec.run_batch(rows.to(dev), layout)["video_output"][0].cpu()  # [F, H, W, 3] uint8
            media[name] = {"psnr_db": psnr(frames.float() / 255, ref_u8.float() / 255),
                           "equal_frac": (frames == ref_u8).float().mean().item(), "shape": list(frames.shape)}
            if args.out:
                torch.save(frames, args.out.with_suffix(f".{name}.frames.pt"))
        report["video_decode"] = media
        del vae, dec
        torch.cuda.empty_cache()

    if "audio" in stages and "audio" in oracle:
        from mstar.model.minimax_h3.submodules import H3AudioDecoderSubmodule

        avae = weight_loader.build_audio_vae(config, args.snapshot, dev)
        dec = H3AudioDecoderSubmodule(avae, config).to(dev)
        ref = oracle["audio"][0]
        ref_rows = oracle["audio_latents"].permute(0, 2, 1).reshape(-1, oracle["audio_latents"].shape[1])[None]
        sources = {"oracle_latents": ref_rows}
        sources.update({f"port_{mode}": audio for mode, (_, audio) in loops.items()})
        media = {}
        for name, rows in sources.items():
            with torch.no_grad():
                wave = dec.run_batch(rows.to(dev))["audio_output"][0].cpu()
            noise = (wave - ref).pow(2).mean().item()
            media[name] = {**stats(wave, ref),
                           "snr_db": float("inf") if noise == 0 else 10 * math.log10(ref.pow(2).mean().item() / noise)}
            if args.out:
                torch.save(wave, args.out.with_suffix(f".{name}.audio.pt"))
        report["audio_decode"] = media


if __name__ == "__main__":
    main()
