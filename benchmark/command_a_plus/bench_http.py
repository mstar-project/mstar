"""Throughput and latency benchmark for a running Command A+ server.

Sweeps concurrency and reports aggregate decode throughput, and per request
the time to first token (TTFT) and time per output token after it (TPOT).
Every request streams with ``ignore_eos`` and a fixed ``max_output_tokens``,
so each concurrency level does the same work and the levels are comparable.

    python -m benchmark.command_a_plus.bench_http --url http://127.0.0.1:8000 \
      --concurrency 1 8 32 64 --output bench.json
"""

import argparse
import base64
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPT = (
    "Write a detailed technical explanation of how a modern GPU executes a "
    "matrix multiplication, covering memory hierarchy, tiling, and warp "
    "scheduling. Be thorough and precise."
)


def generate(url, prompt, max_tokens, ignore_eos=True):
    """One streaming request. Returns (ttft_s, total_s, tokens_emitted)."""
    model_kwargs = {
        "max_output_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
    }
    started = time.perf_counter()
    first = None
    tokens = 0
    with requests.post(url.rstrip("/") + "/generate", data={
        "text": prompt, "streaming": "true", "output_modalities": "text",
        "model_kwargs": json.dumps(model_kwargs),
    }, stream=True, timeout=1800) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk["modality"] == "error":
                raise RuntimeError(base64.b64decode(chunk["data"]).decode())
            if chunk["modality"] == "text":
                tokens += 1
                if first is None:
                    first = time.perf_counter()
    total = time.perf_counter() - started
    return (first or started) - started, total, tokens


def measure_concurrency(url, concurrency, max_tokens):
    """Throughput and latency with ``concurrency`` requests in flight at once."""
    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(
            lambda _: generate(url, PROMPT, max_tokens), range(concurrency)
        ))
    wall = time.perf_counter() - started
    tokens = sum(count for _, _, count in results)
    ttft = [first for first, _, _ in results]
    tpot = [(total - first) / (count - 1) for first, total, count in results if count > 1]
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "tokens": tokens,
        "aggregate_tok_s": tokens / wall,
        "ttft_ms": statistics.median(ttft) * 1000,
        "tpot_ms": statistics.median(tpot) * 1000 if tpot else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    generate(args.url, "Warm up.", 8, ignore_eos=False)

    print(f"{'conc':>5} {'tokens':>8} {'wall s':>8} {'agg tok/s':>11} {'TTFT ms':>9} {'TPOT ms':>9}")
    rows = []
    for concurrency in args.concurrency:
        row = measure_concurrency(args.url, concurrency, args.max_tokens)
        rows.append(row)
        tpot = f"{row['tpot_ms']:>9.1f}" if row["tpot_ms"] is not None else f"{'-':>9}"
        print(
            f"{row['concurrency']:>5} {row['tokens']:>8} {row['wall_s']:>8.1f} "
            f"{row['aggregate_tok_s']:>11.1f} {row['ttft_ms']:>9.0f} {tpot}",
            flush=True,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"label": args.label, "max_tokens": args.max_tokens, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
