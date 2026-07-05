"""Apples-to-apples t2i latency client — hits the OpenAI /v1/images/generations
endpoint that BOTH our mstar server and vLLM-Omni (`vllm serve --omni`) expose, with
an identical payload, and reports client-side wall latency (warmup + median of N).

Same scope on both engines (client-side end-to-end incl. HTTP + b64 PNG), same config
(tiers, steps, guidance, seed, prompt). Run once per server (different --port/--model).

  python bench_t2i_oai.py --port 8000 --model nvidia/Cosmos3-Nano --tag vllm
  python bench_t2i_oai.py --port 8100 --model cosmos3_nano --tag ours
"""
import argparse
import base64
import json
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Nano")
ap.add_argument("--sizes", default="320x192,832x480,1280x720")  # 256p/480p/720p tiers
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--gs", type=float, default=6.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--tag", default="run")
ap.add_argument("--save", default="")  # optional PNG path prefix
args = ap.parse_args()

PROMPT = "A red cube resting on a polished wooden table, soft daylight."
NEG = "blurry, distorted, low quality"
URL = f"http://localhost:{args.port}/v1/images/generations"


def one(size):
    body = json.dumps({
        "model": args.model, "prompt": PROMPT, "negative_prompt": NEG,
        "size": size, "n": 1, "response_format": "b64_json",
        "num_inference_steps": args.steps, "guidance_scale": args.gs, "seed": args.seed,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        payload = json.load(r)
    dt = time.perf_counter() - t0
    b64 = payload["data"][0]["b64_json"]
    return dt, b64


print(f"=== {args.tag}  port={args.port} model={args.model}  steps={args.steps} gs={args.gs} seed={args.seed} ===", flush=True)
for size in args.sizes.split(","):
    try:
        for _ in range(args.warmup):
            one(size)
        ts = []
        last_b64 = None
        for _ in range(args.rounds):
            dt, last_b64 = one(size)
            ts.append(dt)
        ts.sort()
        med = statistics.median(ts)
        print(f"  {size:9s}  median {med:.3f}s  min {ts[0]:.3f}  max {ts[-1]:.3f}  (n={args.rounds})", flush=True)
        if args.save and last_b64:
            with open(f"{args.save}_{size}.png", "wb") as f:
                f.write(base64.b64decode(last_b64))
    except Exception as e:  # noqa: BLE001
        print(f"  {size:9s}  ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
print("DONE", flush=True)
