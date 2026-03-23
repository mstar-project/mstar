#!/usr/bin/env python3
"""Benchmarking / load-testing script for the BAGEL serving endpoint.

Requires: aiohttp  (pip install aiohttp)
"""

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import aiohttp

DEFAULT_PROMPTS = {
    "text_to_text": "What is the 7th value after the decimal point in pi?",
    "text_to_image": "A cat in a suit and tie",
    "image_to_text": "Please describe how this food is made",
}

REQUEST_TYPES = ["text_to_text", "text_to_image", "image_to_text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the BAGEL serving endpoint")
    parser.add_argument("--url", default="http://0.0.0.0:8000/generate", help="Server URL")
    parser.add_argument("--num-requests", type=int, default=10, help="Total number of requests to send")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second")
    parser.add_argument(
        "--request-type",
        default="text_to_text",
        choices=["text_to_text", "text_to_image", "image_to_text", "mixture"],
        help="Type of requests to send",
    )
    parser.add_argument("--image-path", default="test/bagel/bagel.png", help="Image file for image-input requests")
    parser.add_argument("--prompt", default=None, help="Text prompt override")
    parser.add_argument("--think-mode", action="store_true", help="Enable think_mode in model_kwargs")
    parser.add_argument("--output", default=None, help="Optional path to save JSON results")
    return parser.parse_args()


def get_request_types(request_type: str, num_requests: int) -> list[str]:
    """Return list of request types for each request index."""
    if request_type == "mixture":
        return [REQUEST_TYPES[i % len(REQUEST_TYPES)] for i in range(num_requests)]
    return [request_type] * num_requests


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    req_type: str,
    prompt: str | None,
    image_path: str,
    think_mode: bool,
) -> dict:
    """Send a single request and measure latency."""
    text = prompt or DEFAULT_PROMPTS[req_type]
    model_kwargs = json.dumps({"think_mode": think_mode})

    t_start = time.monotonic()
    t_first_token: float | None = None
    status = "success"
    error_msg = ""

    try:
        if req_type == "image_to_text":
            # Multipart form with file upload
            path = Path(image_path)
            file_bytes = path.read_bytes()

            form = aiohttp.FormData()
            form.add_field("text", text)
            form.add_field("model_kwargs", model_kwargs)
            form.add_field("files", file_bytes, filename=path.name, content_type="application/octet-stream")

            async with session.post(url, data=form) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("data"):
                        if t_first_token is None:
                            t_first_token = time.monotonic()
        else:
            # Regular form data
            data: dict[str, str] = {"text": text, "model_kwargs": model_kwargs}
            if req_type == "text_to_image":
                data["output_modalities"] = "image"

            async with session.post(url, data=data) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data_b64 = msg.get("data", "")
                    if data_b64:
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        # For image gen, check if we got the image
                        if req_type == "text_to_image" and msg.get("modality") == "image":
                            decoded = base64.b64decode(data_b64)
                            if decoded:
                                pass  # Image received successfully

    except Exception as e:
        status = "failed"
        error_msg = str(e)

    t_end = time.monotonic()

    ttft = (t_first_token - t_start) if t_first_token is not None else None
    e2e = t_end - t_start

    return {
        "request_id": request_id,
        "type": req_type,
        "ttft": ttft,
        "e2e_latency": e2e,
        "status": status,
        "error": error_msg,
    }


def percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def print_results(results: list[dict], rate: float, wall_time: float) -> None:
    """Print a summary table of benchmark results."""
    total = len(results)
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    # Type breakdown
    type_counts: dict[str, int] = {}
    for r in results:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))

    print(f"\nBenchmark Results ({total} requests, rate={rate} req/s)")
    print("\u2500" * 50)
    print(f"Request type breakdown: {breakdown}")
    print(f"Successful: {len(successful)}/{total}, Failed: {len(failed)}/{total}")

    # TTFT stats (successful requests with ttft)
    ttft_values = [r["ttft"] for r in successful if r["ttft"] is not None]
    if ttft_values:
        print(
            f"TTFT  (s):  mean={statistics.mean(ttft_values):.2f}  "
            f"p50={percentile(ttft_values, 50):.2f}  "
            f"p95={percentile(ttft_values, 95):.2f}  "
            f"p99={percentile(ttft_values, 99):.2f}"
        )
    else:
        print("TTFT  (s):  no data")

    # E2E stats (successful requests)
    e2e_values = [r["e2e_latency"] for r in successful]
    if e2e_values:
        print(
            f"E2E   (s):  mean={statistics.mean(e2e_values):.2f}  "
            f"p50={percentile(e2e_values, 50):.2f}  "
            f"p95={percentile(e2e_values, 95):.2f}  "
            f"p99={percentile(e2e_values, 99):.2f}"
        )
    else:
        print("E2E   (s):  no data")

    # Throughput
    if wall_time > 0 and successful:
        throughput = len(successful) / wall_time
        print(f"Throughput: {throughput:.2f} req/s (successful only)")
    else:
        print("Throughput: N/A")

    print(f"Total wall time: {wall_time:.2f}s")

    # Print errors if any
    if failed:
        print("\nFailed requests:")
        for r in failed:
            print(f"  #{r['request_id']} ({r['type']}): {r['error']}")


async def run_benchmark(args: argparse.Namespace) -> list[dict]:
    """Run the benchmark with controlled request rate."""
    request_types = get_request_types(args.request_type, args.num_requests)
    interval = 1.0 / args.rate if args.rate > 0 else 0.0

    results: list[dict] = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks: list[asyncio.Task] = []

        wall_start = time.monotonic()

        for i, req_type in enumerate(request_types):
            task = asyncio.create_task(
                send_request(
                    session=session,
                    url=args.url,
                    request_id=i,
                    req_type=req_type,
                    prompt=args.prompt,
                    image_path=args.image_path,
                    think_mode=args.think_mode,
                )
            )
            tasks.append(task)

            # Rate limiting: sleep between request launches (except after last)
            if i < len(request_types) - 1 and interval > 0:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks)
        wall_end = time.monotonic()

    wall_time = wall_end - wall_start
    results = list(results)

    print_results(results, args.rate, wall_time)

    return results


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_benchmark(args))

    if args.output:
        output_path = Path(args.output)
        # Convert to JSON-serializable format
        output_data = []
        for r in results:
            entry = dict(r)
            output_data.append(entry)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
