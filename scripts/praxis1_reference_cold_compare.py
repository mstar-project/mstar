"""Compare one saved M* cold DROID action reply with the pinned Praxis pipeline.

Run after the diagnostic M* server has stopped so both models never occupy
the GPU together. The same saved or generated observation, prompt and seed are used.
This is a numerical diagnostic; the WAN VAE implementations differ slightly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mstar-actions", type=Path, help="saved synthetic openpi actions")
    source.add_argument("--sim-capture", type=Path, help="NPZ from praxis1_sim_first_call_capture.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-pth", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--native-vae-root", type=Path,
                        help="also run the reference policy using M*'s Diffusers VAE encode")
    parser.add_argument("--trace-output", type=Path,
                        help="save reference input tensors and per-step sampler trace for stage isolation")
    parser.add_argument("--mstar-stage-dir", type=Path,
                        help="ablate M*'s saved text and VAE stage tensors into the reference cold call")
    parser.add_argument("--mstar-revision", required=True)
    parser.add_argument("--skip-last-video", action="store_true",
                        help="use the reference serving optimization; M* E5 currently decodes both steps")
    parser.add_argument("--reference-simple", action="store_true",
                        help="disable reference context caching, expert overlap and paired guidance")
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "comparison needs CUDA"
    for path in (args.mstar_actions or args.sim_capture, args.vae_pth,
                 args.qwen / "config.json", args.checkpoint / "config.json"):
        assert path.is_file(), f"missing input: {path}"
    if args.sim_capture:
        with np.load(args.sim_capture, allow_pickle=False) as capture:
            actual = capture["actions"]
            observation = {
                "image.exterior": capture["exterior"], "image.wrist": capture["wrist"],
                "robot_state.ee_pose": capture["ee_pose"], "grip": capture["grip"],
                "descriptor": str(capture["descriptor"]),
            }
            prompt = str(capture["prompt"])
            seed = 0
    else:
        actual = np.load(args.mstar_actions, allow_pickle=False)
        random = np.random.default_rng(71)
        exterior = random.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        wrist = random.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        observation = {
            "image.exterior": exterior,
            "image.wrist": wrist,
            "robot_state.ee_pose": np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "grip": np.array([0.0], dtype=np.float32),
        }
        prompt = "pick up the marker"
        seed = 17
    assert actual.shape == (21, 10) and actual.dtype == np.float32
    settings = SamplerSettings(num_steps=2, t_shift=12.5, cfg_scale=1.0, cfg_video_scale=4.0)
    started = time.perf_counter()
    pipeline = PraxisPipeline(
        args.checkpoint, args.vae_pth, args.qwen, device="cuda", sampler=settings,
        encoder_attention_backend="triton", decoder_attention_backend="triton",
        action_cuda_graphs=False, nvfp4_moe_cuda_graphs=False,
    )

    def session():
        current = pipeline.session(
            prompt, seed=seed, max_context_chunks=0,
            control_mode="none", exec_horizon=None, sampler=settings,
            skip_last_video=args.skip_last_video,
        )
        if args.reference_simple:
            current.ctx_cache = False
            current.overlap_expert = False
            current.pair_cfg = False
        return current

    def report(label: str, expected: np.ndarray) -> None:
        assert expected.shape == actual.shape, (expected.shape, actual.shape)
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        diff = np.abs(actual - expected)
        close = np.isclose(actual, expected, rtol=args.rtol, atol=args.atol)
        print(json.dumps({
            "phase": label, "mstar_revision": args.mstar_revision,
            "skip_last_video": args.skip_last_video,
            "reference_simple": args.reference_simple,
            "reference_revision": "68f4d7a254b4bf41951d2e97f44b07f6cac2bf69",
            "checkpoint": str(args.checkpoint), "shape": list(actual.shape),
            "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12)),
            "fraction_close": float(close.mean()), "all_close": bool(close.all()),
            "per_dim_max_abs": [float(x) for x in diff.max(axis=0)],
            "reference_seconds_including_load": round(time.perf_counter() - started, 2),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        }), flush=True)

    baseline_rel = None
    if args.trace_output is not None:
        with torch.inference_mode():
            baseline = session().step(observation)
        baseline_rel = np.asarray(baseline.action_rel, dtype=np.float32).copy()
        del baseline
        report("reference_pth_untraced", baseline_rel)
    current = session()
    events: list[tuple[str, torch.Tensor]] = []
    if args.trace_output is not None:
        original_sample = pipeline.policy.sample_window_joint

        def traced_sample(**kwargs):
            return original_sample(
                **kwargs,
                trace=lambda name, value: events.append((name, value.detach().cpu().contiguous())),
            )

        pipeline.policy.sample_window_joint = traced_sample
    with torch.inference_mode():
        reference = current.step(observation)
    torch.cuda.synchronize()
    report("reference_pth", np.asarray(reference.action_rel, dtype=np.float32))
    if args.trace_output is not None:
        traced_rel = np.asarray(reference.action_rel, dtype=np.float32)
        print(json.dumps({"phase": "trace_perturbation", "max_abs":
                          float(np.abs(traced_rel - baseline_rel).max()),
                          "exact": bool(np.array_equal(traced_rel, baseline_rel))}), flush=True)
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        def on_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().contiguous()
            if isinstance(value, (tuple, list)):
                return [on_cpu(item) for item in value]
            return value

        torch.save({
            "inputs": {key: on_cpu(value) for key, value in reference.inputs.items()},
            "pred_action_norm": reference.pred_action_norm.detach().cpu().contiguous(),
            "action_rel": np.asarray(reference.action_rel, dtype=np.float32),
            "events": events,
        }, args.trace_output)
        pipeline.policy.sample_window_joint = original_sample
        print(json.dumps({"phase": "reference_stage_trace", "path": str(args.trace_output),
                          "events": [name for name, _ in events]}), flush=True)
    del reference

    if args.native_vae_root is not None:
        from diffusers import AutoencoderKLWan

        from mstar.model.praxis1.config import Praxis1Config
        from mstar.model.praxis1.submodules import Praxis1VaeEncoderSubmodule

        assert (args.native_vae_root / "vae" / "config.json").is_file()
        native_vae = AutoencoderKLWan.from_pretrained(
            str(args.native_vae_root), subfolder="vae", torch_dtype=torch.bfloat16,
        ).to("cuda").eval().requires_grad_(False)
        native = Praxis1VaeEncoderSubmodule(Praxis1Config.from_checkpoint(args.checkpoint), native_vae)
        matched = session()
        matched.state.encode = native.vae_encode
        with torch.inference_mode():
            result = matched.step(observation)
        torch.cuda.synchronize()
        report("reference_with_native_vae", np.asarray(result.action_rel, dtype=np.float32))

    if args.mstar_stage_dir is not None:
        def one_stage(name: str) -> dict:
            paths = list(args.mstar_stage_dir.glob(f"*-{name}.pt"))
            if len(paths) != 1:
                raise ValueError(f"expected one saved {name} stage, found {len(paths)}")
            return torch.load(paths[0], map_location="cpu", weights_only=False)

        text_stage = one_stage("Praxis1TextEncoderSubmodule")
        vae_stage = one_stage("Praxis1VaeEncoderSubmodule")
        task_ids = text_stage["inputs"]["text_ids"]
        blank_ids = text_stage["inputs"]["blank_ids"]
        processor = pipeline.text.processor
        for label, value, prompt_value in (
            ("task", task_ids, prompt),
            ("blank", blank_ids, pipeline.variant.blank_caption),
        ):
            ref_ids = processor(text=[prompt_value], **pipeline.text.call_kwargs)["input_ids"][0]
            print(json.dumps({"phase": "token_check", "stream": label,
                              "same_ids": bool(torch.equal(value.cpu(), ref_ids.cpu())),
                              "mstar_len": int(value.numel()), "reference_len": int(ref_ids.numel())}), flush=True)

        task_embed = text_stage["outputs"]["text_embeds"][0].to("cuda")
        blank_embed = text_stage["outputs"]["text_embeds_blank"][0].to("cuda")
        anchor = vae_stage["outputs"]["obs_latents"][0].to("cuda")
        original_encode_text = pipeline.encode_text
        for use_text, use_vae in ((True, False), (False, True), (True, True)):
            current = session()
            if use_vae:
                calls = 0

                def encode_once(clip):
                    nonlocal calls
                    calls += 1
                    if calls != 1:
                        raise ValueError("cold ablation expected one VAE encode")
                    return anchor

                current.state.encode = encode_once
            if use_text:
                def encode_saved(value):
                    if value == prompt:
                        return task_embed
                    if value == pipeline.variant.blank_caption:
                        return blank_embed
                    raise ValueError(f"unexpected text in ablation: {value!r}")

                pipeline.encode_text = encode_saved
            try:
                with torch.inference_mode():
                    ablated = current.step(observation)
                torch.cuda.synchronize()
                report(f"mstar_{'text' if use_text else 'reference_text'}_"
                       f"{'vae' if use_vae else 'reference_vae'}",
                       np.asarray(ablated.action_rel, dtype=np.float32))
            finally:
                pipeline.encode_text = original_encode_text


if __name__ == "__main__":
    main()
