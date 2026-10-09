#!/usr/bin/env python3
"""How far apart two equally valid runs of the *reference* DiT land: the same step-0
inputs under the flash and the memory-efficient SDPA backends (oracle venv). This is
the yardstick for the native port's per-step velocity error.

    .venv-ltx-oracle/bin/python test/ltx2_5/reference_noise_floor.py --oracle-dir <dir>/t2av --snapshot <snap>
"""
import argparse
from pathlib import Path

import torch
from diffusers import LTX2VideoTransformer3DModel
from torch.nn.attention import SDPBackend, sdpa_kernel


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return float((a - b).norm() / b.norm())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle-dir", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--steps", default="0,7")
    args = p.parse_args()
    model = LTX2VideoTransformer3DModel.from_pretrained(
        args.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda:0")
    for k in (int(s) for s in args.steps.split(",")):
        inp = torch.load(Path(args.oracle_dir) / f"dit_in_{k:03d}.pt", weights_only=False)
        kw = {n: (v.to("cuda:0") if torch.is_tensor(v) else v) for n, v in inp.items()}
        outs = {}
        for name, backends in (("flash", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
                               ("efficient", [SDPBackend.EFFICIENT_ATTENTION])):
            with torch.inference_mode(), sdpa_kernel(backends):
                outs[name] = model(**kw)
        recorded = torch.load(Path(args.oracle_dir) / f"dit_out_{k:03d}.pt", weights_only=False)
        print(f"step {k}: flash vs efficient   video {rel(outs['flash'][0], outs['efficient'][0]):.4e}"
              f"  audio {rel(outs['flash'][1], outs['efficient'][1]):.4e}")
        print(f"step {k}: recorded vs efficient video {rel(recorded[0], outs['efficient'][0]):.4e}"
              f"  audio {rel(recorded[1], outs['efficient'][1]):.4e}")
        print(f"step {k}: recorded vs flash     video {rel(recorded[0], outs['flash'][0]):.4e}"
              f"  audio {rel(recorded[1], outs['flash'][1]):.4e}")


if __name__ == "__main__":
    main()
