"""Compare two ``oracle.py`` recordings (two reference runs), for the parity yardstick.

The reference against itself under another attention kernel (``--attention
_native_efficient``) is the spread a correct port is held to; the reference against
itself with another seed is the negative control a parity gate must fail.

    python test/minimax_h3/compare_oracles.py A/oracle.pt B/oracle.pt [--out x.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def compare(a: dict, b: dict) -> dict:
    out = {"args_a": a["args"], "args_b": b["args"]}
    s0a, s0b = a["steps"][0], b["steps"][0]
    out["same_initial_noise"] = bool(torch.equal(s0a["hidden_states"], s0b["hidden_states"]))
    out["prompt_embeds_rel"] = rel(a["prompt_embeds"], b["prompt_embeds"])
    steps = []
    for ea, eb in zip(a["steps"], b["steps"], strict=False):
        if "noise_pred" in ea and "noise_pred" in eb:
            steps.append({"k": ea["k"], "video_rel": rel(ea["noise_pred"], eb["noise_pred"]),
                          "audio_rel": rel(ea["audio_noise_pred"], eb["audio_noise_pred"])})
    out["steps"] = steps
    out["latents_rel"] = rel(a["latents"], b["latents"])
    out["audio_latents_rel"] = rel(a["audio_latents"], b["audio_latents"])
    if "videos" in a and "videos" in b:
        out["video_psnr_db"] = psnr(a["videos"], b["videos"])
    if "audio" in a and "audio" in b:
        wa, wb = a["audio"].float(), b["audio"].float()
        noise = (wa - wb).pow(2).mean().item()
        out["audio_snr_db"] = float("inf") if noise == 0 else 10 * math.log10(wb.pow(2).mean().item() / noise)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    result = compare(torch.load(args.a, weights_only=False), torch.load(args.b, weights_only=False))
    text = json.dumps(result, indent=1, default=str)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
