"""Compare one DROID canvas through the reference Wan .pth and M*'s diffusers VAE path.

This is a diagnostic gate for the auxiliary VAE choice. It uses one synthetic
camera canvas and does not establish full serving or simulation parity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from praxis.vae import Autoencoder

from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.submodules import Praxis1VaeEncoderSubmodule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="DROID checkpoint directory with config.json")
    parser.add_argument("--reference-vae", type=Path, required=True, help="Wan2.2_VAE.pth")
    parser.add_argument("--diffusers-root", type=Path, required=True, help="directory with vae/config.json and weights")
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "the VAE parity probe needs CUDA"
    assert (args.diffusers_root / "vae" / "config.json").is_file(), "missing diffusers vae/config.json"
    config = Praxis1Config.from_checkpoint(args.checkpoint)
    started = time.perf_counter()
    reference = Autoencoder.from_pretrained(
        args.reference_vae, device="cuda", decoder_impl="reference",
    )
    native_vae = AutoencoderKLWan.from_pretrained(
        str(args.diffusers_root), subfolder="vae", torch_dtype=torch.bfloat16,
    ).to("cuda").eval().requires_grad_(False)
    native = Praxis1VaeEncoderSubmodule(config, native_vae).eval()
    generator = torch.Generator(device="cuda").manual_seed(613)
    canvas = torch.randint(
        0, 256, (1, 3, 1, config.canvas_height, config.canvas_width),
        device="cuda", dtype=torch.uint8, generator=generator,
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        expected = reference.encode(canvas).float()
        actual = native.vae_encode(canvas).float()
    torch.cuda.synchronize()
    assert expected.shape == actual.shape, (expected.shape, actual.shape)
    assert expected.isfinite().all() and actual.isfinite().all(), "non-finite VAE output"
    diff = (actual - expected).abs()
    metrics = {
        "shape": list(actual.shape), "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "relative_l2": float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        "rtol": args.rtol, "atol": args.atol,
    }
    print(json.dumps(metrics), flush=True)
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    main()
