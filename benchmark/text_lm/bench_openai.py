"""Closed-loop throughput and latency of a text LLM behind an OpenAI-compatible chat route.

The same requests go to any server (M*, vLLM, SGLang): streaming ``/v1/chat/completions``,
greedy, ``ignore_eos`` with a fixed ``max_tokens``, so every request does the same work on
every engine. ``--concurrency`` requests stay in flight; each level runs one warmup wave,
then at least ``--waves`` waves. Prompts are coding tasks built from this repo's own Python
source (``coding_prompts``), so the set is fixed without a download.

    python -m benchmark.text_lm.bench_openai --url http://127.0.0.1:8000 --model qwen3_coder_30b_a3b \
        --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --concurrency 1 8 32 --output-len 256 --out run.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TASKS = (
    "Write pytest unit tests for the following code. Cover the edge cases.",
    "Explain what the following code does, then point out one bug or fragile assumption.",
    "Refactor the following code for readability without changing its behaviour. Show the result.",
    "Add precise type hints and docstrings to the following code.",
)


def coding_prompts(root: Path, n: int, lines: tuple[int, int] = (60, 160)) -> list[list[dict]]:
    """``n`` chats, each a task over a slice of one of the repo's Python files, deterministic."""
    files = sorted(p for p in (root / "mstar").rglob("*.py") if p.stat().st_size > 4000)
    chats = []
    for i in range(n):
        path = files[(i * 7919) % len(files)]
        text = path.read_text(errors="replace").splitlines()
        span = lines[0] + (i * 37) % (lines[1] - lines[0])
        start = (i * 13) % max(1, len(text) - span)
        code = "\n".join(text[start:start + span])
        task = TASKS[i % len(TASKS)]
        chats.append([{"role": "user", "content": f"{task}\n\n```python\n# {path.relative_to(root)}\n{code}\n```"}])
    return chats


def one(url: str, model: str, messages: list[dict], max_tokens: int, extra: dict) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "ignore_eos": True, **extra}
    started = time.perf_counter()
    first, chunks, text = None, 0, []
    with requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, stream=True, timeout=3600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
                    text.append(piece)
    done = time.perf_counter()
    return {"ttft": (first or done) - started, "total": done - started, "chunks": chunks, "text": "".join(text)}


def level(url, model, chats, concurrency, waves, max_tokens, extra, tokenizer) -> dict:
    n = max(waves * concurrency, 16)
    work = [chats[i % len(chats)] for i in range(n)]
    lock, results = threading.Lock(), []

    def run(messages):
        res = one(url, model, messages, max_tokens, extra)
        with lock:
            results.append(res)

    # one warmup wave on other prompts, then the timed requests
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(lambda m: one(url, model, m, max_tokens, extra), chats[-concurrency:]))
    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(run, work))
    wall = time.perf_counter() - started
    out_tokens = [len(tokenizer.encode(r["text"])) for r in results] if tokenizer else [max_tokens] * len(results)
    ttft = sorted(r["ttft"] for r in results)
    tpot = sorted((r["total"] - r["ttft"]) / max(1, t - 1) for r, t in zip(results, out_tokens, strict=True))
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {
        "concurrency": concurrency, "requests": n, "wall_s": wall,
        "output_tokens": sum(out_tokens), "output_tok_s": sum(out_tokens) / wall,
        "short_replies": sum(t < max_tokens for t in out_tokens),
        "ttft_ms_p50": 1e3 * statistics.median(ttft), "ttft_ms_p90": 1e3 * q(ttft, 0.9),
        "tpot_ms_p50": 1e3 * statistics.median(tpot), "tpot_ms_p90": 1e3 * q(tpot, 0.9),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", help="HF tokenizer to count output tokens (default: trust max_tokens)")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--waves", type=int, default=5)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    chats = coding_prompts(Path(args.root), args.prompts)
    report = {"args": vars(args), "levels": []}
    if tokenizer is not None:
        lens = sorted(len(tokenizer.apply_chat_template(c, add_generation_prompt=True)) for c in chats)
        report["prompt_tokens"] = {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]}
        print("prompt tokens", report["prompt_tokens"], flush=True)
    for c in args.concurrency:
        row = level(args.url, args.model, chats, c, args.waves, args.output_len, json.loads(args.extra), tokenizer)
        report["levels"].append(row)
        print(json.dumps(row), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
