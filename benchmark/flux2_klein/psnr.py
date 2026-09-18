#!/usr/bin/env python3
"""PSNR between images (PNG / JPEG / WebP): ``psnr.py ref.png other.png [more.png ...]``.

Prices a serving-path deviation against a bit-exact reference: e.g. the FlashInfer + CUDA-graph
path's output against the SDPA path's (or diffusers') image for the same prompt, seed and size.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("others", nargs="+")
    args = ap.parse_args()
    ref = np.array(Image.open(args.reference).convert("RGB"))
    for other in args.others:
        value = psnr(ref, np.array(Image.open(other).convert("RGB")))
        print(f"{Path(other).name:40s} PSNR vs {Path(args.reference).name}: {value:.2f} dB")


if __name__ == "__main__":
    main()
