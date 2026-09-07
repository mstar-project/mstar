#!/usr/bin/env python3
"""Deterministic fixed-frame benchmark for the Qwen3-TTS HTTP endpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import statistics
import time
from typing import Any

import requests

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
DEFAULT_TEXT = (
    "Profiling Qwen three TTS performance with a fixed codec frame workload."
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _request(
    url: str,
    payload: dict[str, str],
    timeout: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    audio_bytes = 0
    with requests.post(
        url,
        data=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            message = json.loads(line)
            if message.get("modality") != "audio":
                continue
            encoded = message.get("data", "")
            if encoded:
                audio_bytes += len(base64.b64decode(encoded))
    return time.perf_counter() - start, audio_bytes


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_kwargs = {
        "voice": args.voice,
        "language": args.language,
        "seed": args.seed,
        "ignore_eos": True,
        "max_output_tokens": args.frames,
    }
    payload = {
        "text": args.text,
        "output_modalities": "audio",
        "model_kwargs": json.dumps(model_kwargs),
    }

    for _ in range(args.warmup):
        _request(args.url, payload, args.timeout)

    trial_measurements = []
    trial_wall_times = []
    for _ in range(args.trials):
        start = time.perf_counter()
        if args.concurrency == 1:
            measurements = [
                _request(args.url, payload, args.timeout)
                for _ in range(args.requests)
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                futures = [
                    pool.submit(_request, args.url, payload, args.timeout)
                    for _ in range(args.requests)
                ]
                measurements = [future.result() for future in futures]
        trial_wall_times.append(time.perf_counter() - start)
        trial_measurements.append(measurements)

    measurements = [
        measurement
        for trial in trial_measurements
        for measurement in trial
    ]
    latencies = [latency for latency, _ in measurements]
    byte_counts = [size for _, size in measurements]
    if not byte_counts or min(byte_counts) <= 0:
        raise RuntimeError("Qwen3-TTS benchmark received an empty audio response")
    actual_audio_durations = [
        size / (SAMPLE_RATE * SAMPLE_WIDTH) for size in byte_counts
    ]
    # The 12 Hz decoder emits 1,920 samples (80 ms) per valid codec frame.
    # ``ignore_eos`` fixes Talker compute, but Codec intentionally filters any
    # EOS values sampled inside that fixed sequence, so actual PCM length can
    # be a few frames shorter. Use nominal duration for comparable compute RTF
    # and report the observed output range separately.
    nominal_audio_duration = args.frames * 1_920 / SAMPLE_RATE
    trial_throughputs = [
        args.requests / wall_time for wall_time in trial_wall_times
    ]
    total_requests = args.requests * args.trials
    total_wall_time = sum(trial_wall_times)
    return {
        "label": args.label,
        "frames": args.frames,
        "requests_per_trial": args.requests,
        "total_requests": total_requests,
        "trials": args.trials,
        "warmup": args.warmup,
        "concurrency": args.concurrency,
        "audio_bytes_min": min(byte_counts),
        "audio_bytes_max": max(byte_counts),
        "audio_duration_mean_s": statistics.mean(actual_audio_durations),
        "nominal_audio_duration_s": nominal_audio_duration,
        "wall_time_total_s": total_wall_time,
        "throughput_requests_per_s": total_requests / total_wall_time,
        "throughput_trial_median_requests_per_s": statistics.median(
            trial_throughputs
        ),
        "throughput_trial_min_requests_per_s": min(trial_throughputs),
        "throughput_trial_max_requests_per_s": max(trial_throughputs),
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": _percentile(latencies, 0.95) * 1000,
        "latency_min_ms": min(latencies) * 1000,
        "latency_max_ms": max(latencies) * 1000,
        "rtf_nominal_mean": statistics.mean(latencies) / nominal_audio_duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--label", default="qwen3_tts")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--voice", default="Vivian")
    parser.add_argument("--language", default="English")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    if args.frames <= 0 or args.requests <= 0 or args.trials <= 0:
        parser.error("frames, requests, and trials must be positive")
    if args.warmup < 0:
        parser.error("warmup cannot be negative")
    if not 1 <= args.concurrency <= args.requests:
        parser.error("concurrency must be between 1 and requests")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
