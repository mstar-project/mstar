#!/usr/bin/env python3
"""Record the reference oracle for LingBot-World-V2 1.3B causal-fast.

Runs the official ``wan.WanI2VCausal`` from a checkout of
https://github.com/Robbyant/lingbot-world-v2 (never any mstar code) on one
example and saves what its causal-fast loop computes, through forward hooks
and wrappers that observe without changing it:

    <out>/inputs.pt     prompt, seed, sizes, tokenizer ids, T5 context, y
                        (mask + VAE latent), Plucker rays per chunk
    <out>/calls.pt      every DiT forward: x in, timestep, flow out, current_start
    <out>/latents.pt    the stacked x0 the reference decodes
    <out>/video.pt      the decoded video, [3, F, H, W] fp32 in [-1, 1]
    <out>/metadata.json resolved parameters and versions

Numerics: mstar's engine sets ``torch.set_float32_matmul_precision("high")``
process-wide, so the oracle is recorded under the same flag (see
``test/wan22/record_oracle.py`` for why that is load-bearing).

    PYTHONPATH=/path/to/lingbot-world-v2 python test/lingbot_world/record_oracle.py \
        --ckpt-dir <1.3b snapshot> --assets-dir <14b snapshot> --example <repo>/examples/03 \
        --out-dir /tmp/lingbot/oracle
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")

# run_fast.sh's prompt for examples/03.
PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
    "snow-capped mountains under a bright blue sky with drifting white clouds \u2014 gentle ripples "
    "reflect the tree and sky, creating a tranquil, meditative atmosphere."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-num", type=int, default=93)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-attn-size", type=int, default=18)
    ap.add_argument("--sink-size", type=int, default=6)
    args = ap.parse_args()

    import wan
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = WAN_CONFIGS["i2v-1.3B"]
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size,
        infer_mode="causal_fast", assets_dir=args.assets_dir,
    )

    calls: list[dict] = []
    plucker_chunks: list[torch.Tensor] = []

    def dit_hook(module, hook_args, kwargs, output):
        cond = kwargs.get("dit_cond_dict") or {}
        calls.append({
            "x": kwargs["x"][0].detach().cpu().clone(),
            "t": kwargs["t"].detach().cpu().clone(),
            "current_start": int(kwargs["current_start"]),
            "flow": output[0].detach().cpu().clone(),
        })
        if "c2ws_plucker_emb" in cond and len(plucker_chunks) * 5 < len(calls):
            plucker_chunks.append(cond["c2ws_plucker_emb"][0].detach().cpu().clone())

    pipe.model.register_forward_hook(dit_hook, with_kwargs=True)

    captured: dict = {}
    vae_encode, vae_decode = pipe.vae.encode, pipe.vae.decode

    def encode(videos):
        res = vae_encode(videos)
        captured["vae_input"] = videos[0].detach().cpu().clone()
        captured["vae_latent"] = res[0].detach().cpu().clone()
        return res

    def decode(zs):
        captured["latents"] = zs[0].detach().cpu().clone()
        return vae_decode(zs)

    pipe.vae.encode, pipe.vae.decode = encode, decode

    img = Image.open(Path(args.example) / "image.jpg").convert("RGB")
    t0 = time.perf_counter()
    video = pipe.generate(
        PROMPT, img, action_path=args.example, chunk_size=4, max_area=MAX_AREA_CONFIGS["480*832"],
        frame_num=args.frame_num, shift=cfg.sample_shift, seed=args.seed, offload_model=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ids, mask = pipe.text_encoder.tokenizer([PROMPT], return_mask=True, add_special_tokens=True)
    (context,) = next(iter(pipe._t5_cache.values()))
    import numpy as np

    torch.save({
        "prompt": PROMPT,
        "image_rgb": torch.from_numpy(np.asarray(img).copy()),
        "token_ids": ids[0][: int(mask[0].sum())].clone(),
        "context": context.detach().cpu().clone(),
        "vae_input": captured["vae_input"],
        "vae_latent": captured["vae_latent"],
        "plucker_chunks": plucker_chunks,
    }, out / "inputs.pt")
    torch.save(calls, out / "calls.pt")
    torch.save(captured["latents"], out / "latents.pt")
    torch.save(video.detach().cpu(), out / "video.pt")

    import diffusers
    import flash_attn
    meta = {
        "reference": "Robbyant/lingbot-world-v2 causal_fast (i2v-1.3B)",
        "prompt": PROMPT,
        "seed": args.seed,
        "frame_num_requested": args.frame_num,
        "video_shape": list(video.shape),
        "latents_shape": list(captured["latents"].shape),
        "num_dit_calls": len(calls),
        "chunk_size": 4,
        "local_attn_size": args.local_attn_size,
        "sink_size": args.sink_size,
        "shift": cfg.sample_shift,
        "timesteps_index": [0, 250, 500, 750],
        "max_area": MAX_AREA_CONFIGS["480*832"],
        "generate_seconds": elapsed,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": diffusers.__version__,
        "flash_attn": flash_attn.__version__,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
