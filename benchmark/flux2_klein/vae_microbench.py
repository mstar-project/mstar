#!/usr/bin/env python3
"""FLUX.2 VAE decode micro-benchmark: eager vs channels_last vs cuDNN autotuning vs torch.compile.

    python benchmark/flux2_klein/vae_microbench.py --batch 1 4 8

Decodes a real klein latent grid (1024x1024 -> [B, 32, 128, 128] patched latents) with the
native ``Flux2VAE`` and reports the median wall time per variant, plus the max-abs deviation
of each variant from the eager result (channels_last and compile may reorder reductions).
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time

import torch

from mstar.model.flux2_klein.config import Flux2KleinConfig, resolve_snapshot_dir
from mstar.model.flux2_klein.weight_loader import build_vae


def _time(fn, repeats: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda")
    snapshot = resolve_snapshot_dir(args.repo)
    config = Flux2KleinConfig.from_snapshot(snapshot)
    vae = build_vae(config.vae, snapshot, device).eval()
    h, w = (s // config.vae.spatial_compression for s in args.size)
    print(f"{args.repo}: decode of [B, {config.vae.latent_channels}, {h}, {w}] latents "
          f"-> {args.size[0]}x{args.size[1]}")
    for batch in args.batch:
        latents = torch.randn(batch, config.vae.latent_channels, h, w, device=device, dtype=vae.dtype)
        with torch.no_grad():
            reference = vae.decode(latents)
            variants = {"eager": functools.partial(vae.decode, latents)}
            cl_latents = latents.to(memory_format=torch.channels_last)
            variants["channels_last input"] = functools.partial(vae.decode, cl_latents)
            torch.backends.cudnn.benchmark = True
            variants["eager + cudnn.benchmark"] = functools.partial(vae.decode, latents)
            compiled = torch.compile(vae.decode, fullgraph=False, dynamic=False)
            variants["compiled + cudnn.benchmark"] = functools.partial(compiled, latents)
            print(f"== batch {batch}")
            for name, fn in variants.items():
                ms = _time(fn, args.repeats)
                diff = (fn().float() - reference.float()).abs().max().item()
                print(f"  {name:28s} {ms:8.2f} ms   max_abs vs eager {diff:.3e}")
            torch.backends.cudnn.benchmark = False
        print(f"  peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
