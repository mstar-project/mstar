"""The reference against itself: two ``oracle.py`` recordings of one job (a repeat, or the same
request on another SDPA kernel), per step and for the final image, the yardstick the port's
parity is held to. CPU only.

    python test/minimax_h3_image/compare_oracles.py runs/x_seed42_run0.pt runs/x_seed42_run1.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def compare(a_path: Path, b_path: Path) -> dict:
    a, b = torch.load(a_path, weights_only=False), torch.load(b_path, weights_only=False)
    out = {
        "a": a_path.stem, "b": b_path.stem,
        "prompt_embeds_bitexact": bool(torch.equal(a["prompt_embeds"], b["prompt_embeds"])),
        "velocity_rel": [rel(x, y) for x, y in zip(a["velocity"], b["velocity"], strict=True)],
        "final_rows_rel": rel(a["step_out"][-1], b["step_out"][-1]),
        "final_bitexact": bool(torch.equal(a["step_out"][-1], b["step_out"][-1])),
    }
    ia = np.asarray(Image.open(a_path.with_suffix(".png")).convert("RGB"))
    ib = np.asarray(Image.open(b_path.with_suffix(".png")).convert("RGB"))
    out["image_psnr_db"] = psnr_u8(ia, ib)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = p.parse_args()
    print(json.dumps(compare(args.a, args.b), indent=1))


if __name__ == "__main__":
    main()
