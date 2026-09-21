#!/usr/bin/env python3
"""Soak an image server: keep N requests in flight for a while over mixed sizes and prompts, watch for drift.

    python benchmark/flux2_klein/soak.py --port 8000 --model flux2_klein --prompts prompts_100.txt \\
        --minutes 60 --concurrency 8 --sizes 1024x1024 768x1024 512x512 --json results/soak.json

Every minute: images/s, request latency p50 / p95 of that minute per size, error count, and the GPU's memory.used
(nvidia-smi) — so a leak, a latency drift or a recompile storm shows up as a trend. The final JSON has every minute.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def gpu_memory_mib(gpu: str) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sizes", nargs="+", default=["1024x1024", "768x1024", "512x512"])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    prompts = Path(args.prompts).read_text().splitlines()
    url = f"http://{args.host}:{args.port}/v1/images/generations"
    deadline = time.time() + args.minutes * 60
    lock = threading.Lock()
    minute: dict[str, list[float]] = defaultdict(list)
    errors: list[str] = []
    counter = [0]
    done = [0]

    def worker() -> None:
        while time.time() < deadline:
            with lock:
                i = counter[0]
                counter[0] += 1
            size = args.sizes[i % len(args.sizes)]
            body = {"model": args.model, "prompt": prompts[i % len(prompts)], "size": size, "n": 1,
                    "response_format": "b64_json", "num_inference_steps": args.steps, "seed": i}
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    payload = json.load(resp)
                ok = bool(payload.get("data"))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                ok = False
                with lock:
                    errors.append(f"{size} #{i}: {type(e).__name__}: {str(e)[:80]}")
            wall = time.perf_counter() - t0
            with lock:
                if ok:
                    minute[size].append(wall)
                    done[0] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    history = []
    start = time.time()
    last_done, last_errors = 0, 0
    while time.time() < deadline:
        time.sleep(60)
        with lock:
            snapshot = {s: list(v) for s, v in minute.items()}
            minute.clear()
            n_done, n_err = done[0], len(errors)
        elapsed = time.time() - start
        rate = (n_done - last_done) / 60.0
        entry = {"minute": round(elapsed / 60), "images_per_s": rate, "errors_this_minute": n_err - last_errors,
                 "memory_used_mib": gpu_memory_mib(args.gpu), "latency": {}}
        for size, walls in sorted(snapshot.items()):
            if walls:
                walls.sort()
                entry["latency"][size] = {"n": len(walls), "p50_s": statistics.median(walls),
                                          "p95_s": walls[min(len(walls) - 1, int(0.95 * (len(walls) - 1)))]}
        history.append(entry)
        lat = " ".join(f"{s}: p50 {v['p50_s']:.2f} p95 {v['p95_s']:.2f} (n={v['n']})"
                       for s, v in entry["latency"].items())
        print(f"min {entry['minute']:3d}: {rate:.2f} images/s, errors {entry['errors_this_minute']}, "
              f"memory {entry['memory_used_mib']} MiB | {lat}", flush=True)
        last_done, last_errors = n_done, n_err
    for t in threads:
        t.join(timeout=args.timeout)
    total_s = time.time() - start
    summary = {"model": args.model, "minutes": args.minutes, "concurrency": args.concurrency, "sizes": args.sizes,
               "images": done[0], "images_per_s": done[0] / total_s, "errors": errors, "history": history}
    rates = [h["images_per_s"] for h in history if h["images_per_s"] > 0]
    mems = [h["memory_used_mib"] for h in history if h["memory_used_mib"]]
    print(f"SOAK {'PASS' if not errors else 'FAIL'}: {done[0]} images in {total_s / 60:.1f} min "
          f"({done[0] / total_s:.2f} images/s; first/last minute {rates[0] if rates else 0:.2f}/"
          f"{rates[-1] if rates else 0:.2f}), "
          f"{len(errors)} errors, memory first/last {mems[0] if mems else '?'}/{mems[-1] if mems else '?'} MiB")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
