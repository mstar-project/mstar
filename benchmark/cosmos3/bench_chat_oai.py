"""Reasoner (understanding tower) latency client for the OpenAI chat endpoint
both M* (``mstar serve cosmos3_edge``) and vLLM (``vllm serve nvidia/Cosmos3-Edge``)
expose: time-to-first-token and decode tokens/s, streamed, at a chosen
concurrency, on the model card's reasoning prompt (image + text) or text only.

Same payload on both engines (greedy, fixed max_tokens, thinking off unless
asked), client-side timing, warmup excluded, median and p95 reported.

  python bench_chat_oai.py --port 8000 --model nvidia/Cosmos3-Edge --tag vllm --image assets/example_reasoning_input.png
  python bench_chat_oai.py --port 8100 --model cosmos3_edge --tag ours --image assets/example_reasoning_input.png
"""
import argparse
import base64
import concurrent.futures as cf
import json
import mimetypes
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="nvidia/Cosmos3-Edge")
ap.add_argument("--image", default="")  # optional conditioning image path (else text-only)
ap.add_argument("--prompt", default="The task is to put flower into the red bottle. Generate a plan consisting of "
                                    "subtasks for accomplish the task.")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--concurrency", default="1,8,32")
ap.add_argument("--requests", type=int, default=16)  # per concurrency level (>= concurrency)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--thinking", action="store_true")
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default="")  # optional JSON results path
args = ap.parse_args()

URL = f"http://localhost:{args.port}/v1/chat/completions"
content = []
if args.image:
    mime = mimetypes.guess_type(args.image)[0] or "image/png"
    with open(args.image, "rb") as f:
        data_url = f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    content.append({"type": "image_url", "image_url": {"url": data_url}})
content.append({"type": "text", "text": args.prompt})
BODY = {
    "model": args.model,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": args.max_tokens,
    "temperature": 0.0,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": bool(args.thinking)},
}


def one() -> dict:
    req = urllib.request.Request(URL, data=json.dumps(BODY).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n_chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                n_chunks += 1
                text.append(piece)
    end = time.perf_counter()
    ttft = (first or end) - t0
    total = end - t0
    decode = max(end - (first or end), 1e-9)
    return {"ttft": ttft, "total": total, "chunks": n_chunks, "tok_s": (n_chunks - 1) / decode if n_chunks > 1 else 0.0,
            "text": "".join(text)}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


results = {}
print(f"=== {args.tag} port={args.port} model={args.model} max_tokens={args.max_tokens} "
      f"image={'yes' if args.image else 'no'} thinking={args.thinking} ===", flush=True)
for _ in range(args.warmup):
    one()
for conc in [int(c) for c in args.concurrency.split(",")]:
    n = max(args.requests, conc)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        outs = list(ex.map(lambda _: one(), range(n)))
    wall = time.perf_counter() - t0
    ttfts = [o["ttft"] for o in outs]
    toks = [o["tok_s"] for o in outs]
    total_tokens = sum(o["chunks"] for o in outs)
    rec = {
        "concurrency": conc, "requests": n, "ttft_p50": statistics.median(ttfts), "ttft_p95": pct(ttfts, 0.95),
        "decode_tok_s_per_req_p50": statistics.median(toks), "aggregate_tok_s": total_tokens / wall,
        "wall_s": wall, "sample": outs[0]["text"][:120],
    }
    results[conc] = rec
    print(f"  conc={conc:3d}  TTFT p50 {rec['ttft_p50'] * 1000:7.1f} ms  p95 {rec['ttft_p95'] * 1000:7.1f} ms  "
          f"decode {rec['decode_tok_s_per_req_p50']:6.1f} tok/s/req  aggregate {rec['aggregate_tok_s']:7.1f} tok/s  "
          f"(n={n}, wall {wall:.1f}s)", flush=True)
print("  sample:", repr(results[min(results)]["sample"]))
if args.out:
    with open(args.out, "w") as f:
        json.dump({"tag": args.tag, "model": args.model, "max_tokens": args.max_tokens, "image": bool(args.image),
                   "results": results}, f, indent=2)
print("DONE", flush=True)
