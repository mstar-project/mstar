"""Single-request latency of MiniMax-H3-Image text-to-image on M*, SGLang or vllm-omni, same
workload.

One request at a time (concurrency 1), after ``--warmup`` untimed requests per size; the
client times submit to the last byte of the PNG. Every engine gets the OpenAI
``/v1/images/generations`` body the model card's ``examples/request.t2i.json`` sends
(``size``, ``seed``, a CPU generator, ``b64_json``, PNG), plus ``min_generation_area``
when asked. Each output is saved, so the arms' images can be compared side by side.
``--vary-prompt`` appends a request counter to each prompt, so no engine's prompt cache
(SGLang's conditioning cache, M*'s ``text_cache_entries``) serves a timed request.

    python -m benchmark.minimax_h3_image.bench_image --engine mstar --url http://127.0.0.1:8100 --out runs/m
    python -m benchmark.minimax_h3_image.bench_image --engine sglang --url http://127.0.0.1:30000 --out runs/s
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path

import requests

# The model card's README prompt for its t2i workflow (examples/revised_prompt.t2i.txt, a farm-to-table feast
# as H3-Image-Context-IR wrote it), shortened to its first sentences.
PROMPT = (
    "The image is a photorealistic overhead food photograph capturing a rustic farm-to-table feast spread across "
    "a heavily weathered, dark brown wooden table with deep visible grain and scratches. The composition is a "
    "top-down view looking directly at the table surface. In the center sits a large woven jute placemat "
    "supporting a freshly baked rustic sourdough loaf with a deeply scored, golden-brown crust and a dusting of "
    "flour. To the upper left, a hammered copper bowl overflows with a colorful mix of fresh seasonal vegetables, "
    "including purple kale, bright orange carrots, red beets, and green snap peas."
)
MODEL = {"mstar": "minimax_h3_image", "sglang": "MiniMaxAI/MiniMax-H3-Image", "vllm-omni": "MiniMaxAI/MiniMax-H3-Image"}


def generate(url: str, engine: str, prompt: str, size: str, seed: int, min_area: int | None) -> bytes:
    body = {"model": MODEL[engine], "prompt": prompt, "size": size, "seed": seed, "generator_device": "cpu",
            "response_format": "b64_json", "output_format": "png", "n": 1}
    if min_area is not None:
        # vllm-omni reads the model's own knobs from extra_args
        if engine == "vllm-omni":
            body["extra_args"] = {"min_generation_area": min_area}
        else:
            body["min_generation_area"] = min_area
    res = requests.post(f"{url}/v1/images/generations", json=body, timeout=3600)
    res.raise_for_status()
    return base64.b64decode(res.json()["data"][0]["b64_json"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=tuple(MODEL), required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--sizes", default="1024x1024,2048x2048,4096x4096")
    p.add_argument("--min-generation-area", type=int, default=None,
                   help="sent when given; the model's default renders a smaller size on a 2048x2048-area canvas")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--vary-prompt", action="store_true",
                   help="a distinct prompt per request (the prompt, then its request number), so none repeats")
    args = p.parse_args()
    count = iter(range(1 << 30))

    def prompt() -> str:
        return f"{args.prompt} {next(count)}" if args.vary_prompt else args.prompt

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for size in args.sizes.split(","):
        for _ in range(args.warmup):
            generate(args.url, args.engine, prompt(), size, args.seed, args.min_generation_area)
        seconds = []
        for i in range(args.repeat):
            start = time.perf_counter()
            png = generate(args.url, args.engine, prompt(), size, args.seed, args.min_generation_area)
            seconds.append(time.perf_counter() - start)
            (args.out / f"{size}_seed{args.seed}_{i}.png").write_bytes(png)
        results[size] = {"seconds": seconds, "median_s": statistics.median(seconds)}
        print(f"{args.engine} {size}: median {results[size]['median_s']:.2f} s of {seconds}", flush=True)
    (args.out / "results.json").write_text(json.dumps({"engine": args.engine, "args": vars(args) | {"out": str(
        args.out)}, "results": results}, indent=1))


if __name__ == "__main__":
    main()
