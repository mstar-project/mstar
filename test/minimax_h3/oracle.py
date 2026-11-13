"""Record MiniMax-H3's reference outputs: diffusers' ``minimax_h3`` modular pipeline, end to end.

``ModularPipeline`` with ``MiniMaxH3Blocks``, from the files the port loads: the
transformer converted by diffusers' own ``convert_minimax_h3_transformer_checkpoint_to_
diffusers`` (checked against the official diffusers shards), the text encoder cut to 51
layers (``hidden_states[50]`` of a causal stack can't see later ones), and the official
VAEs. Hooks record every DiT step's inputs and outputs and the token refiner's output::

    python test/minimax_h3/oracle.py --snapshot /path/to/MiniMax-H3 --out /tmp/h3_oracle \
        --height 256 --width 448 --num-frames 124 --steps 6 --seed 0

FL2VA: ``--image`` (first frame) and / or ``--last-image``; the canvas then follows the
first keyframe unless ``--height`` / ``--width`` are given. The Qwen3-VL call's inputs
(the presentation's token ids, the processor's pixels and grids) and the encoded
keyframe latents are recorded too.

``--distilled <FastH3 release>``: its diffusers-layout transformer instead of FL2VA's, sampled
as FastVideo samples it (``MiniMaxH3DenoisingStage._set_shifted_schedule``): the release's
rungs over 1000 and a terminal 0, shifted in fp32 by its video / audio shifts, through the
same scheduler step; ``--steps`` is then the rungs plus one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


_hide_broken_torchaudio()

DEFAULT_PROMPT = (
    "A golden retriever runs along a sunlit beach at the water's edge, kicking up spray, while gentle waves "
    "roll in. The camera tracks alongside at a low angle. The sound of crashing surf, the dog's panting and "
    "distant gulls."
)


def build_transformer(snapshot: Path, device: str):
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.loaders.single_file_utils import convert_minimax_h3_transformer_checkpoint_to_diffusers
    from safetensors import safe_open

    src = snapshot / "FL2VA" / "transformer"
    cfg = json.loads((snapshot / "transformer" / "config.json").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    state = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    converted = convert_minimax_h3_transformer_checkpoint_to_diffusers(state, cfg, qkv_layout="interleaved")
    del state
    # against the official diffusers shards present on disk, key for key
    checked = 0
    for shard in sorted((snapshot / "transformer").glob("diffusion_pytorch_model-*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                ref = f.get_tensor(key)
                if key not in converted or converted[key].dtype != ref.dtype or not torch.equal(converted[key], ref):
                    raise RuntimeError(f"converted FL2VA tensor {key} differs from the official diffusers one")
                checked += 1
    print(f"conversion check: {checked} tensors equal to the official diffusers shards", flush=True)
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**cfg)
    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    if unexpected or [m for m in missing if not m.endswith("rope.inv_freq")]:
        raise RuntimeError(f"transformer load: missing {missing[:4]}, unexpected {unexpected[:4]}")
    # the non-persistent rope buffer, recomputed as at init (CPU, fp32)
    n = model.config.rope_freq_dim
    model.rope.inv_freq = 1.0 / (model.config.rope_theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32) / (2 * n)))
    return model.to(device).eval()


def build_distilled_transformer(release: Path, device: str):
    """A distilled release's transformer (FastH3: diffusers' layout, every tensor bf16) as diffusers'
    ``from_pretrained(torch_dtype=bfloat16)`` holds it: ``_keep_in_fp32_modules`` upcast. VSA's
    compression gates are dropped: dense attention reads none."""
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open

    src = release / "transformer"
    cfg = {k: v for k, v in json.loads((src / "config.json").read_text()).items() if not k.startswith("_")}
    fp32 = tuple(f"{name}." for name in MiniMaxH3Transformer3DModel._keep_in_fp32_modules)
    state = {}
    for shard in sorted(src.glob("diffusion_pytorch_model-*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".attn.to_gate_compress." not in key:
                    tensor = f.get_tensor(key)
                    state[key] = tensor.float() if key.startswith(fp32) else tensor
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**cfg)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected or [m for m in missing if not m.endswith("rope.inv_freq")]:
        raise RuntimeError(f"transformer load: missing {missing[:4]}, unexpected {unexpected[:4]}")
    n = model.config.rope_freq_dim
    model.rope.inv_freq = 1.0 / (model.config.rope_theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32) / (2 * n)))
    return model.to(device).eval()


def rung_schedulers(release: Path):
    """The video and audio schedulers on the release's trained rungs, FastVideo's grid."""
    from diffusers import MiniMaxH3Scheduler

    contract = json.loads((release / "fastvideo_inference.json").read_text())
    rungs = contract["dmd_denoising_steps"]

    class RungScheduler(MiniMaxH3Scheduler):
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
            # the pipeline's step count and per-request shift are the recipe's; the grid is the rungs'
            base = torch.tensor([r / 1000.0 for r in rungs] + [0.0], dtype=torch.float32)
            shift = float(self.config.shift)
            super().set_timesteps(device=device, sigmas=shift * base / (1 + (shift - 1) * base))

    return tuple(RungScheduler(shift=float(contract[f"{m}_scheduler_shift"]), shift_provided_sigmas=False)
                 for m in ("video", "audio"))


def build_text_encoder(snapshot: Path, device: str, layers: int = 51):
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    path = snapshot / "FL2VA" / "text_encoder"
    config = AutoConfig.from_pretrained(path)
    config.text_config.num_hidden_layers = layers
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, config=config, torch_dtype=torch.bfloat16)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--height", type=int, default=None, help="default 256, or with keyframes their canvas")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--image", type=Path, default=None, help="FL2VA first keyframe")
    p.add_argument("--last-image", type=Path, default=None, help="FL2VA last keyframe")
    p.add_argument("--num-frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attention", default="native", help="diffusers attention backend (native / flash / ...)")
    p.add_argument("--record-every", type=int, default=1, help="record the transformer's I/O every k steps")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--distilled", type=Path, default=None, help="a distilled release (FastH3) whose DiT to sample")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    keyframes = {}
    if args.image or args.last_image:
        from PIL import Image

        for name, path in (("image", args.image), ("last_image", args.last_image)):
            if path is not None:
                keyframes[name] = Image.open(path).convert("RGB")  # diffusers.utils.load_image
    elif args.height is None:
        args.height, args.width = 256, 448

    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler, ModularPipeline
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    t0 = time.time()
    if args.distilled is None:
        transformer = build_transformer(args.snapshot, args.device)
        schedulers = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    else:
        transformer = build_distilled_transformer(args.distilled, args.device)
        schedulers = rung_schedulers(args.distilled)
    if args.attention != "native":
        transformer.set_attention_backend(args.attention)
    text_encoder = build_text_encoder(args.snapshot, args.device)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.snapshot / "vae", torch_dtype=torch.bfloat16).to(args.device)
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        args.snapshot / "audio_vae", torch_dtype=torch.bfloat16,
    ).to(args.device)
    dtype_report = {
        "vae": sorted({f"{n.split('.')[0]}.{n.split('.')[-1]}:{p.dtype}" for n, p in vae.named_parameters()}),
        "vae_sample": {n: str(p.dtype) for n, p in vae.named_parameters()
                       if n in ("post_quant_conv.weight", "decoder.proj_in.weight", "decoder.register_tokens",
                                "decoder.transformer_blocks.0.norm1.weight", "decoder.transformer_blocks.0.scale1",
                                "decoder.transformer_blocks.0.attn.to_q.weight", "decoder.norm_out.weight",
                                "decoder.proj_out.weight")},
        "audio_vae": sorted({str(p.dtype) for p in audio_vae.parameters()}),
        "transformer": {n: str(p.dtype) for n, p in transformer.named_parameters()
                        if not n.startswith("transformer_blocks.") or n.startswith("transformer_blocks.0.")},
    }
    (args.out / "dtypes.json").write_text(json.dumps(dtype_report, indent=1))
    print(f"components built in {time.time() - t0:.0f} s", flush=True)

    pipe = ModularPipeline.from_pretrained(str(args.snapshot))
    pipe.update_components(
        transformer=transformer, text_encoder=text_encoder, vae=vae, audio_vae=audio_vae,
        tokenizer=Qwen2TokenizerFast.from_pretrained(args.snapshot / "FL2VA" / "tokenizer"),
        processor=Qwen3VLProcessor.from_pretrained(args.snapshot / "FL2VA" / "processor"),
        scheduler=schedulers[0], audio_scheduler=schedulers[1],
    )

    steps = []

    def record(module, inputs, kwargs, output):
        k = len(steps)
        entry = {"k": k, "timestep": kwargs["timestep"].detach().cpu(),
                 "timestep_indices": kwargs["timestep_indices"].detach().cpu()}
        if k % args.record_every == 0:
            entry.update(
                hidden_states=kwargs["hidden_states"].detach().cpu(),
                audio_hidden_states=kwargs["audio_hidden_states"].detach().cpu(),
                noise_pred=output[0].detach().float().cpu(), audio_noise_pred=output[1].detach().float().cpu(),
            )
        if k == 0:
            entry.update(
                encoder_hidden_states=kwargs["encoder_hidden_states"].detach().cpu(),
                token_tags=kwargs["token_tags"].detach().cpu(), position_ids=kwargs["position_ids"].detach().cpu(),
                video_indices=kwargs["video_indices"].detach().cpu(),
                audio_indices=kwargs["audio_indices"].detach().cpu(),
                text_indices=kwargs["text_indices"].detach().cpu(),
            )
        steps.append(entry)

    refined = {}
    transformer.register_forward_hook(record, with_kwargs=True)


    def record_refined(module, inputs, output):
        # returns None: a forward hook that returns a value replaces the module's output
        refined.setdefault("text", output.detach().cpu())

    transformer.token_refiner.register_forward_hook(record_refined)

    conditioner_inputs = {}

    def record_conditioner(module, inputs, kwargs):
        if not conditioner_inputs:
            conditioner_inputs.update({k: v.detach().cpu() for k, v in kwargs.items()
                                       if k in ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")})

    text_encoder.model.register_forward_pre_hook(record_conditioner, with_kwargs=True)

    tokens = pipe.tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    generator = torch.Generator("cpu").manual_seed(args.seed)
    torch.cuda.synchronize()
    t1 = time.time()
    names = ["prompt_embeds", "latents", "audio_latents"] + ([] if args.skip_decode else ["videos", "audio"])
    if keyframes:
        names += ["condition_latents", "text_token_tags"]
    size = {} if args.height is None else {"height": args.height, "width": args.width}
    out = pipe(
        prompt=args.prompt, num_frames=args.num_frames, num_inference_steps=args.steps, generator=generator,
        output_type="pt", output=names, **size, **keyframes,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t1
    print(f"pipeline: {elapsed:.1f} s for {len(steps)} transformer calls", flush=True)

    record_out = {
        "args": {k: str(v) for k, v in vars(args).items()}, "token_ids": torch.tensor(tokens),
        "prompt_embeds": out["prompt_embeds"].detach().cpu(), "refined_text": refined.get("text"),
        "latents": out["latents"].detach().float().cpu(), "audio_latents": out["audio_latents"].detach().float().cpu(),
        "steps": steps, "seconds": elapsed, "conditioner_inputs": conditioner_inputs,
    }
    if keyframes:
        record_out["condition_latents"] = [c.detach().float().cpu() for c in out["condition_latents"]]
        record_out["text_token_tags"] = out["text_token_tags"].detach().cpu()
    if not args.skip_decode:
        record_out["videos"] = out["videos"].detach().float().cpu() if torch.is_tensor(out["videos"]) else out["videos"]
        record_out["audio"] = out["audio"].detach().float().cpu()
    torch.save(record_out, args.out / "oracle.pt")
    print(f"saved {args.out / 'oracle.pt'}", flush=True)


if __name__ == "__main__":
    main()
