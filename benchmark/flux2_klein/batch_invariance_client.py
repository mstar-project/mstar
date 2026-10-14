#!/usr/bin/env python3
"""Served batch invariance: the same seeded request alone vs inside a concurrent batch.

    python benchmark/flux2_klein/batch_invariance_client.py --port 8000 --model flux2_klein \\
        --prompts prompts_100.txt --index 0 --concurrency 8 --steps 4 --rounds 2 --out-dir results/bi

Sends prompt ``index`` (seed = index) alone, then ``rounds`` times concurrently with the next
``concurrency - 1`` prompts (seeds = their indices), so the scheduler batches them at the same
shape. Saves every image and prints, per request, the PSNR of its in-batch image against its solo
image (``--all``; default: the probe request only). A row-mixing bug or a padding leak shows up as a
finite PSNR; the CUDA-graph + compiled path was measured bit-exact across batch sizes on the eager
reference, so anything below ``--threshold`` is a finding.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def _request(url: str, body: dict, timeout: float) -> tuple[bytes, float]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return base64.b64decode(payload["data"][0]["b64_json"]), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--index", type=int, default=0, help="probe prompt (its seed too)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="also compare every companion with its own solo image")
    ap.add_argument("--threshold", type=float, default=40.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    prompts = Path(args.prompts).read_text().splitlines()
    indices = list(range(args.index, args.index + args.concurrency))
    if indices[-1] >= len(prompts):
        raise SystemExit(f"need prompts up to {indices[-1]}, file has {len(prompts)}")
    url = f"http://{args.host}:{args.port}/v1/images/generations"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def body(i: int) -> dict:
        return {"model": args.model, "prompt": prompts[i], "size": args.size, "n": 1, "response_format": "b64_json",
                "num_inference_steps": args.steps, "seed": i}

    compared = indices if args.all else [args.index]
    solo: dict[int, np.ndarray] = {}
    for i in compared:
        png, wall = _request(url, body(i), args.timeout)
        (out / f"solo_{i:03d}.png").write_bytes(png)
        solo[i] = np.array(Image.open(out / f"solo_{i:03d}.png").convert("RGB"))
        print(f"solo   {i:03d}: {wall * 1000:.0f} ms")

    results: dict[str, dict[str, float | None]] = {}
    worst = math.inf
    for r in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {i: pool.submit(_request, url, body(i), args.timeout) for i in indices}
        for i in indices:
            png, wall = futures[i].result()
            path = out / f"batch{r}_{i:03d}.png"
            path.write_bytes(png)
            if i in solo:
                value = psnr(solo[i], np.array(Image.open(path).convert("RGB")))
                worst = min(worst, value)
                results[f"round{r}/{i:03d}"] = {"psnr_db": None if math.isinf(value) else value, "wall_s": wall}
                print(f"round {r} {i:03d} in a batch of {args.concurrency}: {wall * 1000:.0f} ms, "
                      f"PSNR vs solo {'inf' if math.isinf(value) else f'{value:.2f}'} dB")
    verdict = "PASS" if worst >= args.threshold else "FAIL"
    print(f"{verdict}: worst in-batch vs solo PSNR {'inf' if math.isinf(worst) else f'{worst:.2f}'} dB "
          f"(threshold {args.threshold:g}) over {len(results)} comparisons")
    if args.json:
        summary = {"model": args.model, "index": args.index, "concurrency": args.concurrency, "rounds": args.rounds,
                   "worst_psnr_db": None if math.isinf(worst) else worst, "results": results}
        Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
