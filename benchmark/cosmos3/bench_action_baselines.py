"""Action-policy baseline clients. vLLM-Omni: POST /v1/videos (multipart, extra_params.action_mode=policy) -> poll
GET /v1/videos/{id} -> top-level `action` {data, shape, dtype, raw_action_dim}. Saves the actions as .npy (for
notes/served_action_parity.py --ref) and reports per-call latency and actions/s over N rounds.
SGLang: GET /v1/actions/metadata then one best-effort POST /v1/actions/generations (Cosmos3 may not enable it).

  python benchmark/cosmos3/bench_action_baselines.py vllm-omni --port 8200 --image <jpg> --rounds 5 \
      --out results/<date>/action_vllm_omni.npy
  python benchmark/cosmos3/bench_action_baselines.py sglang --port 8400 --image <jpg>
"""

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.request

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("engine", choices=["vllm-omni", "sglang"])
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Edge")
ap.add_argument("--image", required=True)
ap.add_argument("--prompt", default="Pick up the red cup and place it in the sink.")
ap.add_argument("--domain", default="droid_lerobot")
ap.add_argument("--action-dim", type=int, default=10)
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--size", default="832x480")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--gs", type=float, default=3.0)
ap.add_argument(
    "--flow-shift",
    type=float,
    default=5.0,
    help="the DROID policy recipe (vLLM-Omni ROBOLAB defaults: 4 steps, gs 3.0, shift 5.0)",
)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--poll", type=float, default=0.05)
ap.add_argument("--out", default="")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"


def data_url(path):
    ext = path.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
    return f"data:image/{ext};base64," + base64.b64encode(open(path, "rb").read()).decode()


def _multipart(fields, files=()):
    """files: (field, filename, bytes, content_type) tuples."""
    boundary = "----act" + str(int(time.time() * 1e6))
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, fn, data, ctype in files:
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def _body_of(e):
    try:
        return e.read().decode(errors="replace")[:600]
    except Exception:
        return ""


def _get(path, timeout=600):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def vllm_omni_once():
    extra = {
        "action_mode": "policy",
        "domain_name": a.domain,
        "raw_action_dim": a.action_dim,
        "action_chunk_size": a.chunk,
    }
    fields = {
        "model": a.model,
        "prompt": a.prompt,
        "size": a.size,
        "num_frames": str(a.chunk + 1),
        "fps": "15",
        "num_inference_steps": str(a.steps),
        "guidance_scale": str(a.gs),
        "flow_shift": str(a.flow_shift),
        "seed": str(a.seed),
        "extra_params": json.dumps(extra),
    }
    # The recipe ships the conditioning frame as the input_reference upload (video_bench.py does the same for i2v);
    # a data-URL image_reference is rejected with 400 "did not decode to an image".
    body, ctype = _multipart(
        fields, files=[("input_reference", a.image.rsplit("/", 1)[-1], open(a.image, "rb").read(), "image/jpeg")]
    )
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + "/v1/videos", data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=1800) as r:
        job = json.load(r)
    while job.get("status") not in ("completed", "failed"):
        time.sleep(a.poll)
        job = _get(f"/v1/videos/{job['id']}")
    wall = time.perf_counter() - t0
    if job.get("status") != "completed":
        raise RuntimeError(f"vllm-omni job failed: {job.get('error')}")
    act = job.get("action")
    if not act:
        raise RuntimeError(f"no action in job response; keys={list(job.keys())}")
    arr = np.array(act["data"], dtype=np.float32).reshape(act["shape"])
    while arr.ndim > 2:
        arr = arr[0]
    return arr, wall, act


def sglang_once():
    meta = _get("/v1/actions/metadata")
    print(
        json.dumps({"sglang_action_metadata": {k: meta.get(k) for k in ("policy_family", "input", "output")}}),
        flush=True,
    )
    keys = (meta.get("input") or {}).get("image_keys") or ["image"]
    # sglang.multimodal_gen.runtime.entrypoints.action.protocol: the JSON body is
    # {"input": {"task", "observation"}, "parameters"};
    # image values are {"b64_json": ...} dicts (data URLs are not decoded).
    # Cosmos3 policy parameters ride in "parameters".
    b64 = base64.b64encode(open(a.image, "rb").read()).decode()
    payload = {
        "input": {"task": a.prompt, "observation": {"images": {keys[0]: {"b64_json": b64}}}},
        "parameters": {
            "seed": a.seed,
            "num_inference_steps": a.steps,
            "guidance_scale": a.gs,
            "flow_shift": a.flow_shift,
            "action_mode": "policy",
            "domain_name": a.domain,
            "raw_action_dim": a.action_dim,
            "action_chunk_size": a.chunk,
            "num_frames": a.chunk + 1,
            "width": int(a.size.split("x")[0]),
            "height": int(a.size.split("x")[1]),
        },
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        BASE + "/v1/actions/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"sglang {e.code}: {_body_of(e)}") from None
    wall = time.perf_counter() - t0
    # action.protocol.action_generation_response:
    # {"data": [{"action": {"shape": [H, D], "values": [[...]], "raw_action_dim"}}], "usage"}
    act = out["data"][0]["action"]
    arr = np.array(act["values"], dtype=np.float32).reshape(act["shape"])
    raw = int(act.get("raw_action_dim") or a.action_dim)
    return arr[:, :raw], wall, {"shape": act["shape"], "raw_action_dim": raw, "usage": out.get("usage")}


once = vllm_omni_once if a.engine == "vllm-omni" else sglang_once
print(
    f"=== {a.engine} action policy port={a.port} domain={a.domain} chunk={a.chunk} "
    f"steps={a.steps} gs={a.gs} seed={a.seed} ===",
    flush=True,
)
for i in range(a.warmup):
    try:
        _, w, _ = once()
        print(f"  warmup {i}: {w:.3f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  warmup {i} failed: {exc!r}", flush=True)
        sys.exit(2)
walls, arrs = [], []
for i in range(a.rounds):
    arr, w, act = once()
    walls.append(w)
    arrs.append(arr)
    print(
        f"  round {i}: {w:.3f}s  actions {arr.shape} first={np.round(arr[0, : min(3, arr.shape[1])], 4).tolist()} "
        f"dtype={act.get('dtype')} raw_dim={act.get('raw_action_dim')}",
        flush=True,
    )
med = statistics.median(walls)
rep = float(max(np.abs(x - arrs[0]).max() for x in arrs[1:])) if len(arrs) > 1 else 0.0
print(
    f"  chunk latency median {med:.3f}s  p95 {sorted(walls)[int(0.95 * (len(walls) - 1))]:.3f}s"
    f"  -> {a.chunk / med:.1f} actions/s (sequential calls); repeat max-abs-diff {rep:.3e}",
    flush=True,
)
if a.out:
    np.save(a.out, arrs[0])
    print("  saved", a.out)
