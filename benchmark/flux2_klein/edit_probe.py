#!/usr/bin/env python3
"""Served image edits under concurrency, with several references and at other output sizes (FLUX.2 [klein]).

    python benchmark/flux2_klein/edit_probe.py --url http://localhost:8000 --out-dir results/c4_edits --concurrency 4

Through the SDK: (1) generate two references (1024^2 and 768x1024 text-to-image); (2) N single-reference edits with
different prompts / seeds, each alone and then all N concurrently, twice — a request's image inside a concurrent batch
must match its solo image (the edit walk batches requests at the same reference grid); (3) a two-reference edit alone
and in the batch; (4) edits at non-1024^2 output sizes (768x1024, 1024x768, 512x512) with the same request repeated
(determinism) — every image is saved, latencies and PSNRs go to the JSON.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from mstar import MStarClient

EDIT_PROMPTS = (
    "make the sign say goodbye, watercolor style",
    "turn the scene into a night scene with neon lights",
    "replace the cat with a corgi wearing sunglasses",
    "make it an oil painting in the style of Van Gogh",
    "add heavy snow falling and a red scarf on the cat",
    "turn it into a pencil sketch on yellowed paper",
    "make the sign glow and the background a beach at sunset",
    "give the scene a cyberpunk look with rain",
)
MULTI_PROMPT = "put the cat with its sign from the first image in front of the scene from the second image"
SIZES = ((768, 1024), (1024, 768), (512, 512))  # (width, height)


def psnr(a: bytes, b: bytes) -> float:
    x = np.array(Image.open(io.BytesIO(a)).convert("RGB")).astype(np.float64)
    y = np.array(Image.open(io.BytesIO(b)).convert("RGB")).astype(np.float64)
    if x.shape != y.shape:
        return float("nan")
    mse = float(np.mean((x - y) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def _fmt(v: float) -> str:
    return "inf" if math.isinf(v) else ("nan" if math.isnan(v) else f"{v:.2f}")


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=40.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = MStarClient(args.url)
    report: dict = {"solo": {}, "batched": {}, "multi": {}, "sizes": {}}
    worst = math.inf

    ref_a, t = timed(client.generate_image, "a cat holding a sign that says hello world, studio lighting", width=1024,
                     height=1024, seed=0, num_inference_steps=args.steps)
    (out / "ref_a.png").write_bytes(ref_a)
    ref_b, t2 = timed(client.generate_image, "a cozy cabin in a snowy forest at night, warm light in the windows",
                      width=768, height=1024, seed=101, num_inference_steps=args.steps)
    (out / "ref_b.png").write_bytes(ref_b)
    print(f"references: 1024x1024 in {t * 1000:.0f} ms, 768x1024 in {t2 * 1000:.0f} ms")

    prompts = EDIT_PROMPTS[: args.concurrency]
    solo: dict[int, bytes] = {}
    for i, prompt in enumerate(prompts):
        png, wall = timed(client.edit_image, prompt, [("ref_a.png", ref_a)], seed=10 + i,
                          num_inference_steps=args.steps)
        (out / f"edit_solo_{i}.png").write_bytes(png)
        solo[i] = png
        report["solo"][str(i)] = {"wall_s": wall}
        print(f"solo edit {i}: {wall * 1000:.0f} ms")
    for r in range(args.rounds):
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = {i: pool.submit(timed, client.edit_image, p, [("ref_a.png", ref_a)], seed=10 + i,
                                      num_inference_steps=args.steps) for i, p in enumerate(prompts)}
        for i, fut in futures.items():
            png, wall = fut.result()
            (out / f"edit_batch{r}_{i}.png").write_bytes(png)
            value = psnr(solo[i], png)
            worst = min(worst, value)
            report["batched"][f"round{r}/{i}"] = {"wall_s": wall,
                                                  "psnr_vs_solo_db": None if math.isinf(value) else value}
            print(f"round {r} edit {i} in a batch of {len(prompts)}: {wall * 1000:.0f} ms, "
                  f"PSNR vs solo {_fmt(value)} dB")

    multi_solo, wall = timed(client.edit_image, MULTI_PROMPT, [("ref_a.png", ref_a), ("ref_b.png", ref_b)], seed=77,
                             num_inference_steps=args.steps)
    (out / "multi_solo.png").write_bytes(multi_solo)
    print(f"two-reference edit alone: {wall * 1000:.0f} ms")
    with ThreadPoolExecutor(max_workers=len(prompts) + 1) as pool:
        futs = [pool.submit(timed, client.edit_image, p, [("ref_a.png", ref_a)], seed=10 + i,
                            num_inference_steps=args.steps) for i, p in enumerate(prompts)]
        multi_fut = pool.submit(timed, client.edit_image, MULTI_PROMPT, [("ref_a.png", ref_a), ("ref_b.png", ref_b)],
                                seed=77, num_inference_steps=args.steps)
    multi_batched, wall_b = multi_fut.result()
    (out / "multi_batched.png").write_bytes(multi_batched)
    value = psnr(multi_solo, multi_batched)
    worst = min(worst, value)
    report["multi"] = {"wall_solo_s": wall, "wall_mixed_s": wall_b,
                       "psnr_vs_solo_db": None if math.isinf(value) else value,
                       "singles_in_mix": [_fmt(psnr(solo[i], f.result()[0])) for i, f in enumerate(futs)]}
    print(f"two-reference edit among {len(prompts)} single-reference edits: {wall_b * 1000:.0f} ms, PSNR vs alone "
          f"{_fmt(value)} dB; the singles vs their solo images: {report['multi']['singles_in_mix']}")

    for w, h in SIZES:
        a, wall1 = timed(client.edit_image, prompts[0], [("ref_a.png", ref_a)], seed=5, width=w, height=h,
                         num_inference_steps=args.steps)
        b, wall2 = timed(client.edit_image, prompts[0], [("ref_a.png", ref_a)], seed=5, width=w, height=h,
                         num_inference_steps=args.steps)
        (out / f"size_{w}x{h}.png").write_bytes(a)
        size = Image.open(io.BytesIO(a)).size
        value = psnr(a, b)
        report["sizes"][f"{w}x{h}"] = {"output": list(size), "wall_s": [wall1, wall2],
                                     "repeat_psnr_db": None if math.isinf(value) else value}
        print(f"edit at {w}x{h}: output {size[0]}x{size[1]}, {wall1 * 1000:.0f} / {wall2 * 1000:.0f} ms, "
              f"repeat PSNR {_fmt(value)} dB")
        if size != (w, h):
            print(f"  WARNING: requested {w}x{h}, got {size}")
        worst = min(worst, value)
    verdict = "PASS" if worst >= args.threshold else "FAIL"
    print(f"{verdict}: worst PSNR {_fmt(worst)} dB (threshold {args.threshold:g})")
    if args.json:
        report["worst_psnr_db"] = None if math.isinf(worst) else worst
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
