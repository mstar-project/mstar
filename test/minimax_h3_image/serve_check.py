"""Check a running MiniMax-H3-Image server end to end against ``oracle.py``'s recordings.

Each case is a recorded job (``<tag>_seed<seed>.json`` beside its PNGs): text-to-image through
``/v1/images/generations``, a one-reference edit through ``/v1/images/edits`` and a
several-reference one through ``POST /generate``. Checks: the image's size; PSNR against the
recording's PNG (``--min-psnr``; the served attention is not the reference's kernel, so the
bar is the reference against itself on another SDPA kernel, see ``compare_oracles.py``); two
identical requests return the same bytes; a malformed request is a 400 and the server serves
after it. Exits nonzero on a failed check.

    PYTHONPATH=<tree> python test/minimax_h3_image/serve_check.py --url http://127.0.0.1:8100 \
        --case runs/t2i_lighthouse_2048_seed42.json --case runs/edit1_podcast_2048_seed42.json --out /tmp/s
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def request(url: str, job: dict) -> tuple[bytes, float]:
    """One job's request on the route a client would use for it; the PNG and the seconds."""
    prompt = Path(job["prompt_file"]).read_text().strip()
    width, height = (int(v) for v in job["size"].lower().split("x"))
    fields = {"model": "minimax_h3_image", "prompt": prompt, "size": f"{width}x{height}", "seed": job["seed"],
              "response_format": "b64_json", "min_generation_area": job["min_area"]}
    images = job.get("images") or []
    t0 = time.perf_counter()
    if not images:
        res = requests.post(f"{url}/v1/images/generations", json=fields, timeout=3600)
        res.raise_for_status()
        png = base64.b64decode(res.json()["data"][0]["b64_json"])
    elif len(images) == 1:
        with open(images[0], "rb") as f:
            res = requests.post(f"{url}/v1/images/edits", data={k: str(v) for k, v in fields.items()},
                                files={"image": (Path(images[0]).name, f.read(), "image/png")}, timeout=3600)
        res.raise_for_status()
        png = base64.b64decode(res.json()["data"][0]["b64_json"])
    else:
        from mstar.client import MStarClient

        res = MStarClient(url, timeout=3600).generate(
            text=prompt, images=[str(p) for p in images], input_modalities=("image", "text"),
            output_modalities=("image",), height=height, width=width, seed=job["seed"],
            min_generation_area=job["min_area"],
        )
        png = next(c["bytes"] for c in res.raw if c["modality"] == "image")
    return png, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--case", action="append", required=True, type=Path, help="an oracle job .json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-psnr", type=float, default=24.0)
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report, failures = {}, []
    for case in args.case:
        job = json.loads(case.read_text())["job"]
        width, height = (int(v) for v in job["size"].lower().split("x"))
        ref = np.asarray(Image.open(case.with_name(f"{case.stem}_run0.png")).convert("RGB"))
        entry = report.setdefault(case.stem, {"seconds": []})
        pngs = []
        for i in range(args.repeat):
            png, seconds = request(args.url, job)
            (args.out / f"{case.stem}_{i}.png").write_bytes(png)
            entry["seconds"].append(round(seconds, 3))
            pngs.append(png)
        ours = np.asarray(Image.open(io.BytesIO(pngs[0])).convert("RGB"))
        entry["size_ok"] = ours.shape == (height, width, 3)
        entry["psnr_db"] = psnr(ours, ref) if entry["size_ok"] else None
        entry["repeat_identical"] = all(png == pngs[0] for png in pngs[1:])
        print(f"{case.stem}: {entry}", flush=True)
        failures += [f"{case.stem}:{k}" for k in ("size_ok", "repeat_identical") if not entry[k]]
        if entry["psnr_db"] is None or entry["psnr_db"] < args.min_psnr:
            failures.append(f"{case.stem}:psnr")
    bad = {"model": "minimax_h3_image", "prompt": "a cat", "size": "1024x1024", "min_generation_area": 10}
    res = requests.post(f"{args.url}/v1/images/generations", json=bad, timeout=600)
    report["malformed_status"] = res.status_code
    if res.status_code != 400:
        failures.append("malformed_not_400")
    job = json.loads(args.case[0].read_text())["job"]
    try:
        request(args.url, job)
        report["serves_after_malformed"] = True
    except requests.HTTPError:
        report["serves_after_malformed"] = False
        failures.append("serves_after_malformed")
    report["failures"] = failures
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
