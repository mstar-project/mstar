"""Time to first token for long text prompts, and their greedy replies, against an M* server.

Built for the KV store tier (`resources.<kv>.store`): send a set of prompts to one server, restart or
switch servers, send them again, and compare. Each prompt is distinct filler text of about ``--tokens``
tokens, the same for a given ``--seed``, so two runs (or two servers) send byte-identical prompts.

    python -m benchmark.kv_store.ttft --url http://127.0.0.1:8101 --tokens 2048 8192 32768 \\
        --prompts 3 --label first --out results.jsonl
    python -m benchmark.kv_store.ttft --url http://127.0.0.1:8101 --filler 1500   # run decode steps

Each request writes one JSON line: label, tokens asked, prompt index, prompt tokens the server counted,
time to first token, total time, and the reply text.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx

_WORDS = (
    "river stone garden window paper cloud orange signal market harbor lantern copper meadow engine "
    "valley thread winter silver canvas planet bridge forest marble violet anchor shadow candle "
    "harvest ribbon island compass feather summit cobalt quarry saddle timber velvet beacon"
).split()


def prompt_text(tokens: int, index: int, seed: int) -> str:
    """Distinct filler of roughly ``tokens`` tokens with the chat template (about a token a word for these
    words), fixed by its arguments."""
    rng = random.Random(f"{seed}:{tokens}:{index}")
    words = [rng.choice(_WORDS) for _ in range(max(1, tokens - 32))]
    return f"Document {seed}-{tokens}-{index}. " + " ".join(words) + "\n\nSummarize the document in one sentence."


def ask(client: httpx.Client, url: str, model: str, text: str, max_tokens: int) -> dict:
    """One streamed greedy chat completion: time to the first content token, total time, reply, usage."""
    body = {
        "model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first, parts, usage = None, [], None
    with client.stream("POST", f"{url}/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or ():
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if first is None:
                        first = time.perf_counter()
                    parts.append(delta)
    end = time.perf_counter()
    return {
        "ttft_s": None if first is None else first - start, "total_s": end - start, "text": "".join(parts),
        "prompt_tokens": None if usage is None else usage.get("prompt_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default=None, help="the served model name; the first of /v1/models if unset")
    parser.add_argument("--tokens", type=int, nargs="*", default=[2048, 8192, 32768])
    parser.add_argument("--prompts", type=int, default=3, help="prompts per length")
    parser.add_argument("--first-index", type=int, default=0, help="index of the first prompt of each length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", type=Path, default=None, help="JSON lines file to append to")
    parser.add_argument("--filler", type=int, default=0,
                        help="instead, send one short prompt that generates this many tokens (runs decode steps)")
    args = parser.parse_args()

    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        model = args.model or client.get(f"{args.url}/v1/models").json()["data"][0]["id"]
        if args.filler:
            body = {"model": model, "messages": [{"role": "user", "content": "Count upward from one, forever."}],
                    "max_tokens": args.filler, "temperature": 0.0, "ignore_eos": True}
            start = time.perf_counter()
            client.post(f"{args.url}/v1/chat/completions", json=body).raise_for_status()
            print(f"filler: {args.filler} tokens in {time.perf_counter() - start:.1f} s", flush=True)
            return
        for tokens in args.tokens:
            for index in range(args.first_index, args.first_index + args.prompts):
                result = ask(client, args.url, model, prompt_text(tokens, index, args.seed), args.max_tokens)
                record = {"label": args.label, "tokens": tokens, "index": index, **result}
                ttft = "none" if result["ttft_s"] is None else f"{result['ttft_s']:.3f} s"
                print(f"{args.label} {tokens:>6} #{index}: ttft {ttft}, prompt "
                      f"{result['prompt_tokens']} tokens, reply {result['text'][:40]!r}", flush=True)
                if args.out is not None:
                    with args.out.open("a") as f:
                        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
