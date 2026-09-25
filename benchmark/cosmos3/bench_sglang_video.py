"""SGLang-Diffusion (sglang 0.5.19, multimodal_gen) baseline client for Cosmos3-Edge video and image generation.
Async job API: POST /v1/videos (multipart; input_reference = conditioning image for i2v) -> poll GET /v1/videos/{id}
until status == completed -> GET /v1/videos/{id}/content (mp4). Images: POST /v1/images/generations (JSON, b64).
Same knobs as video_bench.py so the numbers line up (size, frames, steps, guidance, fps, seed, flow shift).

  python benchmark/cosmos3/bench_sglang_video.py --port 8400 --size 832x480 --frames 121 --steps 20 --gs 6.0 \
      --rounds 3 --warmup 1 --image <jpg>
  python benchmark/cosmos3/bench_sglang_video.py --port 8400 --t2i --size 640x640 --steps 20 --rounds 3
"""

import argparse
import base64
import json
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Edge")
ap.add_argument("--size", default="832x480")
ap.add_argument("--frames", type=int, default=121)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--gs", type=float, default=6.0)
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--flow-shift", type=float, default=None)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--image", default="", help="i2v conditioning frame; t2v when empty")
ap.add_argument("--prompt", default="A robot arm is cleaning a plate in the kitchen, smooth natural motion.")
ap.add_argument("--negative", default="")
ap.add_argument("--t2i", action="store_true")
ap.add_argument("--save", default="")
ap.add_argument("--poll", type=float, default=0.5)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"


def _multipart(fields, files):
    boundary = "----sgl" + str(int(time.time() * 1e6))
    body = b""
    for k, v in fields.items():
        if v is None:
            continue
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for name, filename, data, ctype in files:
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _json(method, path, payload=None, timeout=1800):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_video():
    fields = {
        "model": a.model,
        "prompt": a.prompt,
        "negative_prompt": a.negative or None,
        "size": a.size,
        "num_frames": str(a.frames),
        "fps": str(a.fps),
        "seed": str(a.seed),
        "num_inference_steps": str(a.steps),
        "guidance_scale": str(a.gs),
        "flow_shift": str(a.flow_shift) if a.flow_shift is not None else None,
    }
    files = []
    if a.image:
        files.append(("input_reference", a.image.rsplit("/", 1)[-1], open(a.image, "rb").read(), "image/jpeg"))
    body, ctype = _multipart(fields, files)
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/videos", data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=1800) as r:
        job = json.load(r)
    vid = job["id"]
    while True:
        st = _json("GET", f"/v1/videos/{vid}")
        if st.get("status") in ("completed", "failed", "cancelled", "error"):
            break
        time.sleep(a.poll)
    if st.get("status") != "completed":
        raise RuntimeError(f"job {vid}: {st.get('status')} {st.get('error')}")
    with urllib.request.urlopen(BASE + f"/v1/videos/{vid}/content", timeout=600) as r:
        mp4 = r.read()
    wall = time.perf_counter() - t0
    return wall, mp4, st.get("inference_time_s"), st.get("peak_memory_mb")


def run_image():
    w, h = a.size.split("x")
    payload = {
        "model": a.model,
        "prompt": a.prompt,
        "size": a.size,
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": a.steps,
        "guidance_scale": a.gs,
        "seed": a.seed,
    }
    if a.negative:
        payload["negative_prompt"] = a.negative
    if a.flow_shift is not None:
        payload["flow_shift"] = a.flow_shift
    t0 = time.perf_counter()
    out = _json("POST", "/v1/images/generations", payload)
    wall = time.perf_counter() - t0
    return wall, base64.b64decode(out["data"][0]["b64_json"]), None, None


run = run_image if a.t2i else run_video
print(
    f"=== sglang port={a.port} model={a.model} {'t2i' if a.t2i else ('i2v' if a.image else 't2v')} "
    f"size={a.size} frames={a.frames} steps={a.steps} gs={a.gs} seed={a.seed} ===",
    flush=True,
)
for i in range(a.warmup):
    w, _, _, _ = run()
    print(f"  warmup {i}: {w:.2f}s", flush=True)
walls, infer = [], []
for i in range(a.rounds):
    w, blob, it, mem = run()
    walls.append(w)
    if it:
        infer.append(float(it))
    if a.save:
        open(f"{a.save}_{i}.{'png' if a.t2i else 'mp4'}", "wb").write(blob)
    print(f"  round {i}: {w:.2f}s  bytes={len(blob)}  server_inference_s={it}  peak_mb={mem}", flush=True)
print(
    f"  {a.size}    median {statistics.median(walls):.2f}s  min {min(walls):.2f}  max {max(walls):.2f}"
    + (f"  server-side median {statistics.median(infer):.2f}s" if infer else "")
    + f"  (n={a.rounds})",
    flush=True,
)
