"""A closed-loop chat load whose requests name LoRA adapters, round-robin.

``benchmark.runner`` sends the base model's name in every request, so it cannot put
rows on adapters; this keeps ``--concurrency`` requests in flight, each naming the next
entry of ``--models`` (an adapter's name, or the base model's), with a fixed output
length (``ignore_eos``) so every request is the same work and the worker sits at one
batch size.

    python -m benchmark.lora.load --url http://127.0.0.1:8100 --concurrency 32 \
        --tokenizer Qwen/Qwen3.5-0.8B --models qwen3_5_0.8b,proofreader,router,pinocchio \
        --requests 256

Tokens are counted from the text, re-tokenized once the load is over: a non-streaming
chat's ``usage`` is not filled in. A reply under 90% of ``max_tokens`` counts as failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import threading
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash map handles collisions, with an example.",
    "Describe the water cycle to a ten-year-old.",
    "List the steps to make bread from scratch, with timings.",
]


def _chat(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> str:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True,
        "enable_thinking": False,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"] or ""


def closed_loop(
    url: str, tokenizer: str, models: list[str], concurrency: int, requests: int, max_tokens: int,
    warmup: int = 0, timeout: float = 600.0,
) -> dict:
    """``warmup`` requests first (not counted), then ``requests`` with ``concurrency``
    in flight. Returns throughput and latency; ``tokenizer`` is the model's HF repo."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)  # before the load: a bad name loses no run
    order = itertools.cycle(models)
    lock = threading.Lock()

    def one(i: int) -> tuple[float, str, str | None]:
        with lock:
            model = next(order)
        t0 = time.monotonic()
        try:
            text = _chat(url, model, PROMPTS[i % len(PROMPTS)], max_tokens, timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return time.monotonic() - t0, "", f"{model}: {exc}"
        return time.monotonic() - t0, text, None

    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, range(warmup)))
        t0 = time.monotonic()
        results = list(pool.map(one, range(requests)))
        wall = time.monotonic() - t0
    counts = [len(tok(r[1], add_special_tokens=False).input_ids) for r in results]
    results = [
        (latency, text, error or (f"short output: {n} tokens of {max_tokens}" if n < 0.9 * max_tokens else None))
        for (latency, text, error), n in zip(results, counts, strict=True)
    ]
    latencies = sorted(r[0] for r in results if r[2] is None)
    errors = [r[2] for r in results if r[2] is not None]
    tokens = sum(n for r, n in zip(results, counts, strict=True) if r[2] is None)

    def pct(p: float) -> float | None:
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))] if latencies else None

    return {
        "concurrency": concurrency, "requests": requests, "models": models,
        "max_tokens": max_tokens, "wall_s": wall, "output_tokens": tokens,
        "output_tok_per_s": tokens / wall if wall else 0.0,
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "latency_p50_s": pct(0.5), "latency_p99_s": pct(0.99),
        "failed": len(errors), "errors": errors[:5],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--tokenizer", required=True, help="the model's HF repo, to count the replies' tokens")
    p.add_argument("--models", required=True, help="comma-separated; requests take them in turn")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--requests", type=int, default=256)
    p.add_argument("--warmup", type=int, default=None, help="default: one wave (= concurrency)")
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    stats = closed_loop(
        args.url, args.tokenizer, args.models.split(","), args.concurrency, args.requests, args.max_tokens,
        warmup=args.concurrency if args.warmup is None else args.warmup,
    )
    print(json.dumps(stats, indent=1))
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
