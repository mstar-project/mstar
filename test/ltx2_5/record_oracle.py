#!/usr/bin/env python3
"""Record the diffusers golden-reference oracle for LTX-2.5 text-to-audio+video, for
``test/ltx2_5/test_reference_equivalence.py``.

Runs the **stock diffusers ``LTX2Pipeline``** — never any mstar code — on the
``Lightricks/LTX-2.5-Diffusers`` checkpoint with the model card's distilled recipe
(explicit ``DISTILLED_SIGMA_VALUES``, every guidance off), and writes to
``<out-dir>/<case>/``:

    input_ids.pt, attention_mask.pt      the left-padded prompt (1024 tokens)
    gemma_hidden.pt                      the 49 stacked Gemma hidden states, real tokens only
                                         [n, 3840, 49] bf16
    text_video.pt, text_audio.pt         connector outputs [1, 1024, 4096|2048] bf16
    latents_init.pt, audio_init.pt       seeded initial packed latents [1, L, 128] fp32
    dit_in_NNN.pt                        the DiT's inputs at step NNN (dict)
    dit_out_NNN.pt                       its (video, audio) velocities, fp32
    latents_NNN.pt                       packed video latents AFTER step NNN
    audio_final.pt                       packed audio latents after the last step, fp32
    vae_decode_in.pt, audio_vae_decode_in.pt, vocoder_in.pt   each decoder's input
    block_NN.pt                          (video, audio) hidden states after block NN, step 0
    video.pt                             decoded frames [F, H, W, 3] uint8
    audio_wave.pt                        vocoder output [2, samples] fp32
    out.mp4                              muxed, for a human to watch
    metadata.json                        every resolved parameter + versions

diffusers 0.41 and transformers >= 5.5 (Gemma-4) are newer than the serving
environment, so this runs in a separate venv with the same torch build; set the
same numerics flags for recording and for the suite:

    NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
        .venv-ltx-oracle/bin/python test/ltx2_5/record_oracle.py --out-dir /path/to/oracle

The noise comes from a CPU generator, so it is device independent and equals what the
mstar dit seeds for the same request seed. Component placement (``--device-map``) only
moves weights; each component still runs on one GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

PROMPT = (
    "A golden retriever runs along a beach at sunset, waves crashing behind it. "
    "The dog barks twice, and seagulls cry overhead. Warm cinematic lighting, "
    "handheld camera following the dog."
)
REPO = "Lightricks/LTX-2.5-Diffusers"
# blocks whose step-0 outputs are recorded, for localizing a parity failure
RECORDED_BLOCKS = (0, 1, 2, 12, 24, 47)


def record(pipe, out: Path, *, prompt: str, height: int, width: int, num_frames: int, fps: float, seed: int):
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
    from diffusers.utils import encode_video

    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, object] = {}

    def save(name: str, value):
        saved[name] = value
        torch.save(value, out / f"{name}.pt")

    # -- text encoder: the token ids in, the stacked hidden states out
    te_forward = pipe.text_encoder.forward

    def text_encoder(*args, **kwargs):
        result = te_forward(*args, **kwargs)
        mask = kwargs["attention_mask"][0].bool()
        save("input_ids", kwargs["input_ids"][0].cpu())
        save("attention_mask", kwargs["attention_mask"][0].cpu())
        stacked = torch.stack(result.hidden_states, dim=-1)[0]   # [1024, 3840, 49]
        save("gemma_hidden", stacked[mask].to(torch.bfloat16).cpu())
        return result

    pipe.text_encoder.forward = text_encoder

    conn_forward = pipe.connectors.forward

    def connectors(*args, **kwargs):
        video, audio, mask = conn_forward(*args, **kwargs)
        save("text_video", video.cpu())
        save("text_audio", audio.cpu())
        return video, audio, mask

    pipe.connectors.forward = connectors

    for name, fn_name in (("latents_init", "prepare_latents"), ("audio_init", "prepare_audio_latents")):
        original = getattr(pipe, fn_name)

        def wrapped(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            save(_name, result.cpu())
            return result

        setattr(pipe, fn_name, wrapped)

    step = {"i": 0}
    dit_forward = pipe.transformer.forward

    def dit(*args, **kwargs):
        i = step["i"]
        keep = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kwargs.items() if k != "attention_kwargs"}
        save(f"dit_in_{i:03d}", keep)
        result = dit_forward(*args, **kwargs)
        save(f"dit_out_{i:03d}", (result[0].float().cpu(), result[1].float().cpu()))
        step["i"] += 1
        return result

    pipe.transformer.forward = dit

    hooks = []
    for idx in RECORDED_BLOCKS:
        def hook(module, inputs, output, _idx=idx):
            if step["i"] == 0:
                save(f"block_{_idx:02d}", (output[0].cpu(), output[1].cpu()))
        hooks.append(pipe.transformer.transformer_blocks[idx].register_forward_hook(hook))

    # the final packed audio latents, and each decoder's input
    denorm_audio = pipe._denormalize_audio_latents

    def denormalize_audio(latents, *args, **kwargs):
        save("audio_final", latents.cpu())
        return denorm_audio(latents, *args, **kwargs)

    pipe._denormalize_audio_latents = denormalize_audio
    for module, attr, name in ((pipe.vae, "decode", "vae_decode_in"), (pipe.audio_vae, "decode", "audio_vae_decode_in"),
                               (pipe.vocoder, "forward", "vocoder_in")):
        original = getattr(module, attr)

        def wrapped(x, *args, _original=original, _name=name, **kwargs):
            save(_name, x.cpu())
            return _original(x, *args, **kwargs)

        setattr(module, attr, wrapped)

    def on_step_end(pipeline, i, t, kwargs):
        save(f"latents_{i:03d}", kwargs["latents"].detach().cpu())
        return {}

    video, audio = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=height, width=width, num_frames=num_frames, frame_rate=fps,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0,
        modality_scale=1.0, audio_modality_scale=1.0,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np", return_dict=False,
        callback_on_step_end=on_step_end, callback_on_step_end_tensor_inputs=["latents"],
    )
    for h in hooks:
        h.remove()
    frames = torch.from_numpy((video[0] * 255).round().clip(0, 255).astype("uint8"))
    save("video", frames)
    save("audio_wave", audio[0].float().cpu())
    encode_video(
        video[0], fps=fps, audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate, output_path=str(out / "out.mp4"),
    )
    return {
        "prompt": prompt, "height": height, "width": width, "num_frames": num_frames, "fps": fps,
        "seed": seed, "sigmas": list(DISTILLED_SIGMA_VALUES), "steps_recorded": step["i"],
        "audio_sample_rate": pipe.vocoder.config.output_sampling_rate,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--height", type=int, default=544)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case", default="t2av")
    p.add_argument("--device-map", default="balanced")
    p.add_argument("--sdpa-backend", choices=("default", "efficient"), default="default",
                   help="force SDPA's memory-efficient kernel: a second, equally valid reference run")
    args = p.parse_args()

    import diffusers
    import transformers
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained(
        REPO, prompt_enhancer=None, torch_dtype=torch.bfloat16, device_map=args.device_map,
    )
    from contextlib import nullcontext

    from torch.nn.attention import SDPBackend, sdpa_kernel

    ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]) if args.sdpa_backend == "efficient" else nullcontext()
    with ctx:
        meta = record(
            pipe, Path(args.out_dir) / args.case, prompt=PROMPT, height=args.height, width=args.width,
        num_frames=args.num_frames, fps=args.fps, seed=args.seed,
    )
    meta.update(
        torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
        device_map=getattr(pipe, "hf_device_map", None),
    )
    with open(Path(args.out_dir) / args.case / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
