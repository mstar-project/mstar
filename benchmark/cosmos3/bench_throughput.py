"""Throughput under load — same-machine concurrency sweep, M* vs vLLM-Omni.

Both engines expose OpenAI /v1/images/generations; we fire a closed-loop of `bs`
concurrent requests (ThreadPoolExecutor, exactly bs in flight) for bs*rounds total
and report sustained req/s + p50/p95/mean latency. This measures how each engine
handles concurrency: M* batches concurrent requests across its worker, while
vLLM-Omni runs one request at a time at default settings, so its req/s is flat in bs.

  python bench_throughput.py --port 8100 --model cosmos3_nano       --tag ours
  python bench_throughput.py --port 8000 --model nvidia/Cosmos3-Nano --tag vllm
"""
import argparse
import base64
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Nano")
ap.add_argument("--sizes", default="320x192,832x480")  # 256p, 480p (720p too slow for a sweep)
ap.add_argument("--bs", default="1,4,8")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--gs", type=float, default=6.0)
ap.add_argument("--rounds", type=int, default=5)   # measured requests per worker
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default="")
args = ap.parse_args()

PROMPT = "A red cube resting on a polished wooden table, soft daylight."
NEG = "blurry, distorted, low quality"
URL = f"http://127.0.0.1:{args.port}/v1/images/generations"


def one(size, seed):
    body = json.dumps({
        "model": args.model, "prompt": PROMPT, "negative_prompt": NEG,
        "size": size, "n": 1, "response_format": "b64_json",
        "num_inference_steps": args.steps, "guidance_scale": args.gs, "seed": seed,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            payload = json.load(r)
        dt = time.perf_counter() - t0
        nbytes = len(base64.b64decode(payload["data"][0]["b64_json"]))
        return dt, True, nbytes, ""
    except Exception as e:  # noqa: BLE001
        return time.perf_counter() - t0, False, 0, f"{type(e).__name__}:{str(e)[:90]}"


def pct(lats, q):
    if not lats:
        return float("nan")
    s = sorted(lats)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_cell(size, bs):
    # warm the server / graph at this size+concurrency (results discarded)
    with ThreadPoolExecutor(max_workers=bs) as ex:
        list(ex.map(lambda i: one(size, 900000 + i), range(max(args.warmup, bs))))
    n = bs * args.rounds
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=bs) as ex:
        res = list(ex.map(lambda i: one(size, i), range(n)))
    makespan = time.perf_counter() - t0
    oks = [r for r in res if r[1]]
    lats = [r[0] for r in oks]
    err = next((r[3] for r in res if not r[1]), "")
    return {
        "size": size, "bs": bs, "n": n, "ok": len(oks), "makespan": makespan,
        "thrpt": len(oks) / makespan if makespan > 0 else float("nan"),
        "p50": pct(lats, 50), "p95": pct(lats, 95),
        "mean": statistics.fmean(lats) if lats else float("nan"), "err": err,
    }


print(f"=== {args.tag}  port={args.port} model={args.model}  steps={args.steps} gs={args.gs} ===", flush=True)
cells = []
for size in args.sizes.split(","):
    base_thrpt = None
    for bs in [int(x) for x in args.bs.split(",")]:
        c = run_cell(size, bs)
        cells.append(c)
        if bs == 1:
            base_thrpt = c["thrpt"]
        if c["ok"] == 0:
            print(f"  {size:9s} bs={bs}: ALL {c['n']} FAILED ({c['err']})", flush=True)
            continue
        scale = c["thrpt"] / base_thrpt if base_thrpt else float("nan")
        tag = "" if c["ok"] == c["n"] else f" ({c['ok']}/{c['n']} ok)"
        print(f"  {size:9s} bs={bs}: thrpt {c['thrpt']:6.3f} req/s ({scale:4.2f}x bs1)  "
              f"p50 {c['p50']:6.2f}s  p95 {c['p95']:6.2f}s  mean {c['mean']:6.2f}s{tag}", flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(cells, f, indent=2)
    print(f"wrote {args.out}", flush=True)
print("DONE", flush=True)
