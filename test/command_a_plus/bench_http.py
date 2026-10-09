"""Throughput and latency benchmark for a running Command A+ server.

Sweeps concurrency and reports aggregate decode throughput, per-request
throughput, and time-to-first-token. Every request runs with ``ignore_eos`` and a
fixed ``max_output_tokens`` so each concurrency level does exactly the same
amount of work and the levels are comparable.

TTFT is measured separately, as the latency of a ``max_output_tokens=1`` request
(prefill plus one decode step) — the server's non-streaming ``/generate`` has no
earlier observable point.

    python -m test.command_a_plus.bench_http --url http://127.0.0.1:8000 \
      --concurrency 1 8 32 64 --output bench.json
"""

import argparse
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
    """One non-streaming request. Returns (seconds, tokens_emitted)."""
    model_kwargs = {
        "max_output_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": ignore_eos,
    }
    started = time.perf_counter()
    response = requests.post(url.rstrip("/") + "/generate", data={
        "text": prompt, "streaming": "false", "output_modalities": "text",
        "model_kwargs": json.dumps(model_kwargs),
    }, timeout=1800)
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    chunks = response.json()["outputs"].get("text", [])
    return elapsed, len(chunks)


def measure_ttft(url, samples):
    latencies = []
    for _ in range(samples):
        seconds, _ = generate(url, PROMPT, max_tokens=1, ignore_eos=False)
        latencies.append(seconds * 1000)
    return statistics.median(latencies)


def measure_concurrency(url, concurrency, max_tokens):
    """Aggregate throughput with ``concurrency`` requests in flight at once."""
    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(
            lambda _: generate(url, PROMPT, max_tokens), range(concurrency)
        ))
    wall = time.perf_counter() - started
    tokens = sum(count for _, count in results)
    per_request = statistics.median(
        count / seconds for seconds, count in results if seconds > 0
    )
    return {
        "concurrency": concurrency,
        "wall_s": wall,
        "tokens": tokens,
        "aggregate_tok_s": tokens / wall,
        "per_request_tok_s": per_request,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ttft-samples", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    generate(args.url, "Warm up.", 8, ignore_eos=False)

    ttft_ms = measure_ttft(args.url, args.ttft_samples)
    print(f"TTFT (median of {args.ttft_samples}, prefill + 1 token): {ttft_ms:.0f} ms\n", flush=True)

    print(f"{'conc':>5} {'tokens':>8} {'wall s':>8} {'agg tok/s':>11} {'per-req tok/s':>14}")
    rows = []
    for concurrency in args.concurrency:
        row = measure_concurrency(args.url, concurrency, args.max_tokens)
        rows.append(row)
        print(
            f"{row['concurrency']:>5} {row['tokens']:>8} {row['wall_s']:>8.1f} "
            f"{row['aggregate_tok_s']:>11.1f} {row['per_request_tok_s']:>14.1f}",
            flush=True,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"label": args.label, "ttft_ms": ttft_ms,
                 "max_tokens": args.max_tokens, "rows": rows},
                f, indent=2,
            )


if __name__ == "__main__":
    main()
