"""Closed-loop throughput of MiniMax-H3-Image text-to-image on M*, SGLang or vllm-omni.

``c`` clients each send their next request as soon as the last one returns, for each
concurrency in ``--concurrency``; one untimed wave of ``c`` requests warms each level. Every
request carries a prompt no other request in the run shares (one of ``PROMPTS``, which differ
in subject and length, then a run tag and the request's number), so no engine's prompt cache
serves it. ``--sizes`` lists the sizes the requests cycle through: one size, or several for a
mix. Request ``k`` of a level always gets the same prompt, size and seed, so two runs with the
same ``--tag`` can be compared image by image (``--save``).

Reported per level: images per minute over the window from the first measured request's
submit to the last one's PNG, p50 / p90 / p99 latency (submit to the last byte; per size too, for a mix), and, when
``nvidia-smi`` runs on this host, the GPU's busy % (``utilization.gpu``, the share of each
sample period in which a kernel ran) averaged over the window.

    python -m benchmark.minimax_h3_image.bench_concurrency --engine mstar --url http://127.0.0.1:8100 \\
        --out runs/m --sizes 1024x1024 --concurrency 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.minimax_h3_image.bench_image import generate

PROMPTS = (
    "A red fox curled up asleep in fresh snow at dawn, soft pink light on the fur.",
    "Macro photograph of a dewdrop on a blade of grass reflecting a tiny sunrise.",
    "An old fisherman mending a net on a wooden pier, overcast sky, documentary style.",
    "A neon-lit ramen stall on a rainy Tokyo side street at night, steam rising from the bowls, reflections "
    "of signs in the puddles, a lone customer in a yellow raincoat on a stool.",
    "Isometric illustration of a tiny floating island with a lighthouse, waterfalls spilling off its edges.",
    "A golden retriever puppy chewing a sneaker on a living room rug, afternoon sun through the blinds.",
    "Studio product shot of a matte black wristwatch on a slab of grey slate, single softbox from the left.",
    "Aerial view of terraced rice fields in Bali after rain, the water in each terrace mirroring the clouds, "
    "a farmer in a conical hat walking along a narrow earthen ridge, lush green everywhere.",
    "A watercolor painting of a Venetian canal with gondolas and laundry lines between the houses.",
    "A cutaway diagram of a steam locomotive, labeled parts, vintage engineering textbook style.",
    "Portrait of an elderly woman with silver braids laughing, wrinkles around her eyes, warm window light.",
    "A bowl of pho on a marble counter with basil, lime wedges and chili slices, top-down view.",
    "An astronaut sitting on a swing hanging from a crescent moon, stars all around, children's book art.",
    "A mountain cabin in a snowstorm at night with warm yellow windows, smoke from the chimney, pine trees "
    "bending under heavy snow, a sled leaning against the porch railing and a lantern hanging by the door.",
    "Close-up of a hummingbird feeding from a red trumpet flower, wings frozen mid-beat.",
    "A busy Moroccan spice market, pyramids of saffron, paprika and cumin in woven baskets, brass lamps.",
    "Minimalist poster of a single paper boat on a calm blue sea, lots of negative space.",
    "A cyberpunk street vendor repairing a robot arm under a flickering holographic sign.",
    "A stack of pancakes with blueberries and maple syrup dripping down the sides, shallow depth of field.",
    "A medieval stone bridge over a misty river in autumn, orange and red leaves floating on the water, a "
    "horse-drawn cart crossing slowly while crows circle above the bare branches of an old oak tree.",
    "A child in rain boots jumping into a puddle, splash frozen in the air, overcast city park.",
    "Blueprint-style technical drawing of a modern sailing yacht, top and side views.",
    "A sleepy tabby cat on a stack of old books in a cozy library, dust motes in the light.",
    "Underwater photograph of a sea turtle gliding over a coral reef, sunbeams cutting through the water.",
)


def prompt_for(tag: str, k: int) -> str:
    return f"{PROMPTS[k % len(PROMPTS)]} [{tag}-{k}]"


class GpuBusy:
    """``utilization.gpu`` of GPU 0 every ``period_ms``, from ``nvidia-smi -lms`` in the background."""

    def __init__(self, period_ms: int = 200):
        self.samples: list[tuple[float, float]] = []
        self.proc = None
        if shutil.which("nvidia-smi") is None:
            return
        self.proc = subprocess.Popen(
            ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             f"-lms={period_ms}"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def mean(self, start: float, end: float) -> float | None:
        window = [u for t, u in self.samples if start <= t <= end]
        return statistics.fmean(window) if window else None

    def close(self):
        if self.proc is not None:
            self.proc.terminate()


def run_level(args, c: int, n: int, gpu: GpuBusy) -> dict:
    sizes = args.sizes.split(",")

    def one(tag: str, k: int, save: bool) -> tuple[float, float, str]:
        size = sizes[k % len(sizes)]
        start = time.perf_counter()
        png = generate(args.url, args.engine, prompt_for(tag, k), size, args.seed + k, args.min_generation_area)
        end = time.perf_counter()
        if save:
            (args.out / f"c{c}").mkdir(exist_ok=True)
            (args.out / f"c{c}" / f"{k:03d}_{size}.png").write_bytes(png)
        return start, end, size

    with ThreadPoolExecutor(c) as pool:
        # warm the level: one wave at its concurrency, prompts of their own
        list(pool.map(lambda k: one(f"{args.tag}-w{c}", k, False), range(c)))
        tag = args.tag if args.same_prompts else f"{args.tag}-c{c}"
        spans = list(pool.map(lambda k: one(tag, k, args.save), range(n)))
    start, end = min(s for s, _, _ in spans), max(e for _, e, _ in spans)
    lat = sorted(e - s for s, e, _ in spans)

    def pct(p: float, values: list[float] = lat) -> float:
        return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]

    busy = gpu.mean(start, end)
    row = {"concurrency": c, "requests": n, "window_s": end - start, "images_per_min": 60 * n / (end - start),
           "p50_s": pct(50), "p90_s": pct(90), "p99_s": pct(99), "mean_s": statistics.fmean(lat),
           "gpu_busy_pct": busy}
    if len(sizes) > 1:
        for size in sizes:
            mine = sorted(e - s for s, e, z in spans if z == size)
            row[f"p50_s_{size}"], row[f"p99_s_{size}"] = pct(50, mine), pct(99, mine)
    print(f"{args.engine} {args.sizes} c={c}: {row['images_per_min']:.1f} img/min, p50 {row['p50_s']:.2f} s, "
          f"p99 {row['p99_s']:.2f} s, GPU busy {'n/a' if busy is None else f'{busy:.0f}%'} ({n} requests, "
          f"{row['window_s']:.1f} s)", flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=("mstar", "sglang", "vllm-omni"), required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", default="1024x1024", help="comma-separated; the requests cycle through them")
    p.add_argument("--concurrency", default="1,2,4,8,16")
    p.add_argument("--requests", type=int, default=None,
                   help="timed requests per level (default: max(12, 5 x concurrency))")
    p.add_argument("--min-generation-area", type=int, default=None,
                   help="sent when given; the model's default renders a smaller size on a 2048x2048-area canvas")
    p.add_argument("--seed", type=int, default=42, help="request k is seeded seed + k")
    p.add_argument("--tag", default=str(int(time.time())), help="part of every prompt, so runs share none")
    p.add_argument("--same-prompts", action="store_true",
                   help="every level sends the same timed prompts (for comparing images across levels)")
    p.add_argument("--save", action="store_true", help="write each timed PNG to <out>/c<c>/")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gpu = GpuBusy()
    rows = []
    try:
        for c in (int(x) for x in args.concurrency.split(",")):
            rows.append(run_level(args, c, args.requests or max(12, 5 * c), gpu))
            (args.out / "results.json").write_text(json.dumps(
                {"engine": args.engine, "args": vars(args) | {"out": str(args.out)}, "levels": rows}, indent=1))
    finally:
        gpu.close()


if __name__ == "__main__":
    main()
