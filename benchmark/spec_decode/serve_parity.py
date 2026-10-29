"""Greedy outputs of a running server on T2's prompt set, and the comparison of two such runs.

    python -m benchmark.spec_decode.serve_parity save --url http://127.0.0.1:8100 --out spec.json \\
        --tokenizer Qwen/Qwen3.5-9B [--concurrency 1]
    python -m benchmark.spec_decode.serve_parity compare base.json spec.json

The set is the one T2's acceptance harness used: the first 32 prompts of the sweep's chat set (256 new tokens)
and the 16 ~1k-token prompts (128 new), through the chat template with thinking off, greedy, streamed, so a
speculative server's multi-token chunks go through the same SSE path a client reads. ``save`` also records each
request's decode rate (generated tokens after the first over the time from the first chunk to the last), which
at concurrency 1 is the per-token latency a client sees.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
CHAT = REPO / "benchmark/assets/simple_text_queries.txt"
LONG = REPO / "benchmark/assets/long_text_queries_1k.txt"


def prompt_set(n_chat: int = 32, n_long: int = 16, chat_new: int = 256, long_new: int = 128):
    chat = [line.strip() for line in CHAT.read_text().splitlines() if line.strip()][:n_chat]
    long = [line.strip() for line in LONG.read_text().splitlines() if line.strip()][:n_long]
    return [(f"chat{i}", p, chat_new) for i, p in enumerate(chat)] + [
        (f"long{i}", p, long_new) for i, p in enumerate(long)
    ]


def one(url: str, model: str, pid: str, prompt: str, max_new: int, timeout: float) -> dict:
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_new,
        "temperature": 0.0, "stream": True, "enable_thinking": False,
    }
    t0 = time.monotonic()
    arrivals, parts = [], []
    with requests.post(f"{url}/v1/chat/completions", json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                break
            doc = json.loads(data)
            if "error" in doc:
                raise RuntimeError(doc["error"])
            for choice in doc.get("choices", []):
                content = (choice.get("delta") or {}).get("content")
                if content:
                    parts.append(content)
                    arrivals.append(time.monotonic())
    return {"id": pid, "text": "".join(parts), "chunks": len(parts), "e2e_s": time.monotonic() - t0,
            "ttft_s": arrivals[0] - t0 if arrivals else None,
            "decode_s": arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0}


def save(args) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    items = prompt_set()
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(one, args.url, args.tokenizer, pid, p, n, args.timeout) for pid, p, n in items]
        results = [f.result() for f in futures]
    wall = time.monotonic() - t0
    for r in results:
        r["tokens"] = len(tok.encode(r["text"], add_special_tokens=False))
        r["ms_per_token"] = 1000 * r["decode_s"] / (r["tokens"] - 1) if r["tokens"] > 1 else None
    rates = sorted(r["ms_per_token"] for r in results if r["ms_per_token"])
    out = {"url": args.url, "concurrency": args.concurrency, "wall_s": wall, "results": results,
           "ms_per_token_p50": rates[len(rates) // 2] if rates else None,
           "tokens_per_chunk": sum(r["tokens"] for r in results) / max(sum(r["chunks"] for r in results), 1)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(results)} prompts, wall {wall:.1f} s, decode p50 {out['ms_per_token_p50']} ms/token, "
          f"{out['tokens_per_chunk']:.2f} tokens/chunk")
    return 0


def compare(args) -> int:
    from transformers import AutoTokenizer

    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    tok = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else None
    ra = {r["id"]: r for r in a["results"]}
    same, diffs = 0, []
    for r in b["results"]:
        x, y = ra[r["id"]]["text"], r["text"]
        if x == y:
            same += 1
            continue
        div = next((i for i, (c1, c2) in enumerate(zip(x, y, strict=False)) if c1 != c2), min(len(x), len(y)))
        entry = {"id": r["id"], "char": div, "a": x[max(div - 40, 0):div + 40], "b": y[max(div - 40, 0):div + 40]}
        if tok is not None:
            ia, ib = tok.encode(x, add_special_tokens=False), tok.encode(y, add_special_tokens=False)
            pairs = enumerate(zip(ia, ib, strict=False))
            entry["token"] = next((i for i, (t1, t2) in pairs if t1 != t2), min(len(ia), len(ib)))
            entry["len_a"], entry["len_b"] = len(ia), len(ib)
        diffs.append(entry)
    print(f"identical {same}/{len(b['results'])}")
    for d in diffs:
        print(json.dumps(d))
    return 0 if not diffs else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--url", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tokenizer", required=True, help="the served checkpoint's HF id")
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--timeout", type=float, default=600)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tokenizer", default=None)
    args = ap.parse_args(argv)
    return save(args) if args.cmd == "save" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
