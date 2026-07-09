"""t2v/i2v latency — engine-aware (the video APIs differ, unlike t2i).

ours  : POST /v1/videos/generations (JSON), response data[0].b64_json = mp4.
vllm  : POST /v1/videos/sync (multipart form, via curl to match the recipe), raw mp4.

Same config on both (tiers, frames, steps, gs, seed, fps); client-side wall, median.
Video gen is slow + fairly deterministic, so few rounds. Reports MP4 byte size as a
sanity check (a real clip is large; a flat/empty one is tiny).

  python video_bench.py --engine ours --port 8100
  python video_bench.py --engine vllm --port 8000
"""
import argparse
import base64
import json
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["ours", "vllm"], required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Nano")
ap.add_argument("--tiers", default="320x192,832x480,1280x720")
ap.add_argument("--frames", type=int, default=189)
ap.add_argument("--steps", type=int, default=35)
ap.add_argument("--gs", type=float, default=6.0)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--flow-shift", type=float, default=10.0)
ap.add_argument("--image", default="")  # i2v: path to the conditioning frame (else t2v)
args = ap.parse_args()

PROMPT = "A robot arm is cleaning a plate in the kitchen, smooth natural motion."
NEG = "blurry, distorted, low quality, jittery, deformed"

# i2v conditioning frame: ours takes a base64 data-url in the JSON body; vLLM takes
# the raw file via multipart input_reference (curl reads args.image directly).
IMG_DATA_URI = None
if args.image:
    with open(args.image, "rb") as _f:
        IMG_DATA_URI = "data:image/jpeg;base64," + base64.b64encode(_f.read()).decode()


def gen_ours(size):
    payload = {
        "prompt": PROMPT, "negative_prompt": NEG, "size": size, "seed": args.seed,
        "guidance_scale": args.gs, "num_inference_steps": args.steps,
        "num_frames": args.frames, "fps": args.fps,
    }
    if IMG_DATA_URI:
        payload["image"] = IMG_DATA_URI
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/videos/generations",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    return dt, len(base64.b64decode(out["data"][0]["b64_json"]))


def gen_vllm(size):
    extra = json.dumps({"use_resolution_template": False, "use_duration_template": False})
    out_mp4 = "/tmp/vbench_vllm.mp4"
    cmd = [
        "curl", "-sS", "-X", "POST", f"http://127.0.0.1:{args.port}/v1/videos/sync",
        "-H", "Accept: video/mp4",
        "-F", f"model={args.model}", "-F", f"prompt={PROMPT}", "-F", f"negative_prompt={NEG}",
        "-F", f"size={size}", "-F", f"num_frames={args.frames}", "-F", f"fps={args.fps}",
        "-F", f"num_inference_steps={args.steps}", "-F", f"guidance_scale={args.gs}",
        "-F", "max_sequence_length=4096", "-F", f"flow_shift={args.flow_shift}",
        "-F", f"extra_params={extra}", "-F", f"seed={args.seed}",
    ]
    if args.image:
        cmd += ["-F", f"input_reference=@{args.image};type=image/jpeg"]
    cmd += ["-o", out_mp4, "-w", "%{http_code}"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    dt = time.perf_counter() - t0
    code = res.stdout.strip()[-3:]
    import os
    sz = os.path.getsize(out_mp4) if os.path.exists(out_mp4) else 0
    if code != "200":
        raise RuntimeError(f"http {code}, {sz}B")
    return dt, sz


gen = gen_ours if args.engine == "ours" else gen_vllm
print(f"=== {args.engine}  port={args.port}  frames={args.frames} steps={args.steps} "
      f"gs={args.gs} seed={args.seed} ===", flush=True)
for size in args.tiers.split(","):
    try:
        for _ in range(args.warmup):
            gen(size)
        ts, sz = [], 0
        for _ in range(args.rounds):
            dt, sz = gen(size)
            ts.append(dt)
        ts.sort()
        med = ts[len(ts) // 2]
        print(f"  {size:9s}  median {med:.2f}s  min {ts[0]:.2f}  max {ts[-1]:.2f}  "
              f"mp4={sz // 1024}KB  (n={args.rounds})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {size:9s}  ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
print("DONE", flush=True)
