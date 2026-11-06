"""Static KV scales from observed amax.

1. Serve the unquantized deployment with ``MSTAR_KV_OBSERVE=<dir>`` (eagerly, so
   no padding rows are observed) and send it calibration prompts.
2. Stop it, then ``python -m mstar.engine.resources.kv.calibrate <dir> --resource kv --out scales.safetensors``.
3. Set ``resources.kv.quant: {format: fp8_e4m3, scales: static, scales_path: scales.safetensors}``.
"""
from __future__ import annotations

import argparse
import glob
import os

import torch

from mstar.engine.resources.kv.quant import KV_FORMATS, MIN_SCALE, local_heads


def merge_observed(obs_dir: str, resource_key: str) -> torch.Tensor:
    """Every rank's amax, as fp32 [2 (K, V), layers, total kv heads]."""
    paths = sorted(glob.glob(os.path.join(obs_dir, f"{resource_key}.rank*.pt")))
    if not paths:
        raise FileNotFoundError(f"no observations for {resource_key!r} in {obs_dir}")
    records = [torch.load(path) for path in paths]
    world_size = {r["world_size"] for r in records}
    total = {r["total_heads"] for r in records}
    if len(world_size) != 1 or len(total) != 1:
        raise ValueError(f"observations in {obs_dir} come from different deployments")
    world_size, total = world_size.pop(), total.pop()
    ranks = sorted(r["rank"] for r in records)
    if ranks != list(range(world_size)):
        raise ValueError(f"observations cover ranks {ranks} of {world_size}")
    _, layers, _ = records[0]["amax"].shape
    out = torch.zeros(2, layers, total)
    for r in records:
        heads = local_heads(total, r["rank"], world_size)
        # a replicated head is observed on several ranks
        out[:, :, heads] = torch.maximum(out[:, :, heads], r["amax"])
    return out


def scales_from_amax(amax: torch.Tensor, fmt: str, headroom: float = 2.0) -> dict[str, torch.Tensor]:
    """The ``amax`` rule: ``headroom`` times the largest observed magnitude maps to the format's largest value.

    Headroom is nearly free in a float format, while a value past the calibrated range saturates.
    """
    qmax = torch.finfo(KV_FORMATS[fmt]).max
    if (amax == 0).any():
        raise ValueError("some layer or head was never written: calibrate on more prompts")
    if headroom < 1:
        raise ValueError(f"headroom {headroom} would clip the calibration data itself")
    scales = (headroom * amax / qmax).clamp_min(MIN_SCALE)
    return {"k_scale": scales[0].contiguous(), "v_scale": scales[1].contiguous()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("obs_dir")
    parser.add_argument("--resource", default="kv", help="the KV resource key")
    parser.add_argument("--format", default="fp8_e4m3", choices=sorted(KV_FORMATS))
    parser.add_argument("--headroom", type=float, default=2.0,
                        help="the observed amax times this maps to the format's max (default 2)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from safetensors.torch import save_file

    amax = merge_observed(args.obs_dir, args.resource)
    save_file(scales_from_amax(amax, args.format, args.headroom), args.out,
              metadata={"format": args.format, "headroom": str(args.headroom)})
    print(f"wrote {args.out}: {amax.shape[1]} layers x {amax.shape[2]} KV heads, "
          f"K amax max {amax[0].max():.3g}, V amax max {amax[1].max():.3g}")


if __name__ == "__main__":
    main()
