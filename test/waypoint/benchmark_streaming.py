#!/usr/bin/env python3
"""Measure Waypoint's typed-frame streaming behavior without setting a gate.

This starts the normal mstar server, performs a short warmup, then records a
baseline SDK stream and an otherwise identical stream whose consumer pauses
between chunks. The JSON artifact contains latency, pacing, stalls,
backpressure observations, and process-group memory for both runs.

The benchmark deliberately has no release threshold. A nonzero exit means the
server, typed frame protocol, telemetry, or deterministic replay failed, not
that a performance number was judged too slow.

Example:

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 test/waypoint/benchmark_streaming.py \
        --variant 360p --physical-gpu 2 --steps 16 \
        --artifact /tmp/waypoint-streaming-360p.json \
        --save-videos /tmp/waypoint-streaming-360p-videos
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import serve_rollout as rollout  # noqa: E402

from mstar.client import MStarClient, VideoFrameChunk  # noqa: E402
from mstar.utils import profiler  # noqa: E402

# One 4-frame chunk at 60 fps is the realtime delivery budget for a single
# stream (matches the chunk geometry _actions/_check assume elsewhere).
REALTIME_CHUNK_BUDGET_MS = 4 / 60 * 1000

# Worker DEBUG lines are formatted by logging.basicConfig(format="%(asctime)s
# %(levelname)s [worker_id] %(name)s: %(message)s") (mstar/conductor/conductor.py);
# %(asctime)s defaults to "2026-09-17 21:38:40,605".
_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


@dataclass(frozen=True)
class ChunkObservation:
    arrival_seconds: float
    byte_count: int
    frame_index: int
    frame_count: int
    fps: float


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linearly interpolated percentile, or None when there are no samples."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1]; got {quantile}")
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stream_metrics(
    observations: Sequence[ChunkObservation],
    *,
    request_wall_seconds: float,
    stall_threshold_seconds: float,
    consumer_pause_seconds: float,
    consumer_pause_count: int,
    payload_sha256: str,
) -> dict:
    """Summarize SDK-observed arrivals using media time from frame metadata."""
    arrivals = [observation.arrival_seconds for observation in observations]
    gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:], strict=False)]
    total_frames = sum(observation.frame_count for observation in observations)
    total_bytes = sum(observation.byte_count for observation in observations)
    media_seconds = sum(
        observation.frame_count / observation.fps for observation in observations
    )

    # Startup is excluded from the sustained ratio. The numerator likewise
    # excludes the first chunk, which was already available at the start of the
    # measured delivery interval.
    if len(observations) > 1 and arrivals[-1] > arrivals[0]:
        delivered_after_first = sum(
            observation.frame_count / observation.fps
            for observation in observations[1:]
        )
        sustained_ratio = delivered_after_first / (arrivals[-1] - arrivals[0])
    else:
        sustained_ratio = None

    stalls = [gap for gap in gaps if gap > stall_threshold_seconds]
    realtime_budget_seconds = REALTIME_CHUNK_BUDGET_MS / 1000.0
    chunks_within_budget = sum(1 for gap in gaps if gap <= realtime_budget_seconds)
    return {
        "chunk_count": len(observations),
        "frame_count": total_frames,
        "payload_bytes": total_bytes,
        "payload_sha256": payload_sha256,
        "time_to_first_frame_seconds": arrivals[0] if arrivals else None,
        "request_wall_seconds": request_wall_seconds,
        "generated_media_seconds": media_seconds,
        "overall_media_to_wall_ratio": (
            media_seconds / request_wall_seconds if request_wall_seconds > 0 else None
        ),
        "sustained_media_to_wall_ratio": sustained_ratio,
        "inter_chunk_gap_seconds": {
            "sample_count": len(gaps),
            "p50": _percentile(gaps, 0.50),
            "p95": _percentile(gaps, 0.95),
            "mean": statistics.fmean(gaps) if gaps else None,
            "jitter_population_stddev": statistics.pstdev(gaps) if gaps else None,
            "maximum": max(gaps) if gaps else None,
        },
        "stalls": {
            "threshold_seconds": stall_threshold_seconds,
            "count": len(stalls),
            "longest_seconds": max(stalls) if stalls else None,
            "total_excess_seconds": sum(
                gap - stall_threshold_seconds for gap in stalls
            ),
        },
        "on_time_chunk_fraction": (
            chunks_within_budget / len(gaps) if gaps else None
        ),
        "consumer": {
            "pause_seconds": consumer_pause_seconds,
            "pause_count": consumer_pause_count,
            "injected_pause_seconds": consumer_pause_seconds * consumer_pause_count,
        },
    }


def _validate_chunk(
    chunk: VideoFrameChunk,
    chunk_index: int,
    variant: rollout.Variant,
) -> list[str]:
    expected_bytes = 4 * variant.height * variant.width * 3
    expected = {
        "width": variant.width,
        "height": variant.height,
        "fps": 60.0,
        "pixel_format": "rgb24",
        "frame_index": chunk_index * 4,
        "frame_count": 4,
    }
    failures = []
    if len(chunk.data) != expected_bytes:
        failures.append(
            f"chunk {chunk_index} has {len(chunk.data)} bytes; expected {expected_bytes}"
        )
    mismatches = {
        key: (chunk.metadata.get(key), wanted)
        for key, wanted in expected.items()
        if chunk.metadata.get(key) != wanted
    }
    if mismatches:
        failures.append(f"chunk {chunk_index} metadata mismatches: {mismatches}")
    return failures


def _startup_latency_metrics(samples: Sequence[float]) -> dict | None:
    """Time-to-first-frame across repeated world-slot reuse, or None if unmeasured."""
    if not samples:
        return None
    return {
        "sample_count": len(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "mean": statistics.fmean(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _encode_mp4(chunks: Sequence[VideoFrameChunk], out: Path, crf: int = 18) -> tuple[int, float]:
    """Encode already-consumed chunks to an H.264 mp4 with PyAV. Called after a
    stream's timed loop has finished, never inside it, so encoding cost is never
    counted as stream latency."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "--save-videos needs PyAV to encode mp4s; install with `uv pip install av`"
        ) from exc
    import numpy as np

    width, height, fps = chunks[0].width, chunks[0].height, chunks[0].fps
    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium"}
    frame_bytes = width * height * 3
    n = 0
    for chunk in chunks:
        buf = np.frombuffer(chunk.data, dtype=np.uint8)
        assert buf.size % frame_bytes == 0
        for frame in buf.reshape(-1, height, width, 3):
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
            n += 1
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return n, fps


def _measure_stream(
    client: MStarClient,
    seed_image: Path,
    variant: rollout.Variant,
    *,
    num_steps: int,
    request_id: str,
    rng_seed: int,
    consumer_pause_seconds: float,
    stall_threshold_seconds: float,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    enable_nvtx: bool = False,
    start_barrier: threading.Barrier | None = None,
    chunks_out: list[VideoFrameChunk] | None = None,
) -> tuple[dict, list[str]]:
    """Consume one stream while retaining only timings and an incremental hash.

    When ``chunks_out`` is given, each chunk's reference (not a copy) is also
    appended to it for later encoding; callers must do that encoding outside
    this function so it never counts toward the timings above.
    """
    stream = client.stream(
        images=seed_image,
        input_modalities=("image",),
        output_modalities=("video_frame",),
        request_id=request_id,
        num_steps=num_steps,
        actions=rollout._actions(num_steps),
        seed=rng_seed,
    )
    iterator = iter(stream)
    # Lazy: crossing here means every stream in the wave has built its request
    # body before any of them open the HTTP request, so the clock below starts
    # from a synchronized release rather than staggered request construction
    # (mirrors serve_rollout._rollout's start_barrier).
    if start_barrier is not None:
        start_barrier.wait(timeout=30)
    if enable_nvtx:
        profiler.range_push(f"benchmark.stream[{request_id}]")
    started = clock()
    observations: list[ChunkObservation] = []
    failures: list[str] = []
    digest = hashlib.sha256()
    pause_count = 0

    while True:
        try:
            # The span the sustained ratio is built from: SDK decode plus the
            # blocking socket read, ending at the instant `arrived` is stamped.
            if enable_nvtx:
                profiler.range_push(f"benchmark.await_chunk[{len(observations)}]")
            event = next(iterator)
            if enable_nvtx:
                profiler.range_pop()
        except StopIteration:
            if enable_nvtx:
                profiler.range_pop()  # the await range
                profiler.range_pop()  # benchmark.stream
            completed = clock()
            break
        arrived = clock()
        if enable_nvtx:
            profiler.mark(f"benchmark.chunk_arrival[{len(observations)}]")
        if not isinstance(event, VideoFrameChunk):
            failures.append(
                f"stream event {len(observations)} was {type(event).__name__}, not VideoFrameChunk"
            )
            continue

        chunk_index = len(observations)
        failures.extend(_validate_chunk(event, chunk_index, variant))
        digest.update(event.data)
        if chunks_out is not None:
            chunks_out.append(event)
        observations.append(
            ChunkObservation(
                arrival_seconds=arrived - started,
                byte_count=len(event.data),
                frame_index=event.frame_index,
                frame_count=event.frame_count,
                fps=event.fps,
            )
        )
        # Pause only between expected chunks. Sleeping after the final chunk
        # would measure delayed EOF discovery rather than stream backpressure.
        if consumer_pause_seconds and len(observations) < num_steps:
            sleep(consumer_pause_seconds)
            pause_count += 1

    if len(observations) != num_steps:
        failures.append(f"expected {num_steps} chunks, got {len(observations)}")
    expected_frames = 4 * num_steps
    actual_frames = sum(observation.frame_count for observation in observations)
    if actual_frames != expected_frames:
        failures.append(f"expected {expected_frames} generated frames, got {actual_frames}")

    metrics = _stream_metrics(
        observations,
        request_wall_seconds=completed - started,
        stall_threshold_seconds=stall_threshold_seconds,
        consumer_pause_seconds=consumer_pause_seconds,
        consumer_pause_count=pause_count,
        payload_sha256=digest.hexdigest(),
    )
    metrics["request_id"] = request_id
    return metrics, failures


def _wait_for_phase_sample(
    sampler: rollout.MemorySampler,
    proc: subprocess.Popen,
    phase: str,
    timeout: float = 30.0,
) -> None:
    before = sum(sample.phase == phase for sample in sampler.snapshot()[0])
    sampler.set_phase(phase)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples, error = sampler.snapshot()
        if sum(sample.phase == phase for sample in samples) > before:
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode} before {phase} telemetry"
            )
        if error:
            last_error = error
        time.sleep(0.05)
    detail = f": {last_error}" if "last_error" in locals() else ""
    raise RuntimeError(f"memory telemetry captured no {phase!r} sample{detail}")


def _memory_metrics(
    sampler: rollout.MemorySampler,
    phase: str,
    quiet: tuple[float, float],
) -> dict:
    samples, error = sampler.snapshot()
    if error:
        raise RuntimeError(f"memory telemetry failed: {error}")
    summary = rollout._summarize_wave_memory(samples, phase, quiet)
    result = asdict(summary)
    result["sample_count"] = sum(sample.phase == phase for sample in samples)
    return result


def _backpressure_metrics(
    baseline: dict,
    slow: dict,
    baseline_memory: dict,
    slow_memory: dict,
) -> dict:
    injected = slow["consumer"]["injected_pause_seconds"]
    wall_increase = slow["request_wall_seconds"] - baseline["request_wall_seconds"]
    return {
        "configured_consumer_pause_seconds": slow["consumer"]["pause_seconds"],
        "injected_pause_seconds": injected,
        "observed_request_wall_increase_seconds": wall_increase,
        "wall_increase_beyond_injected_pause_seconds": wall_increase - injected,
        "time_to_first_frame_change_seconds": (
            slow["time_to_first_frame_seconds"] - baseline["time_to_first_frame_seconds"]
        ),
        "sustained_media_to_wall_ratio_change": (
            slow["sustained_media_to_wall_ratio"]
            - baseline["sustained_media_to_wall_ratio"]
            if slow["sustained_media_to_wall_ratio"] is not None
            and baseline["sustained_media_to_wall_ratio"] is not None
            else None
        ),
        "peak_host_pss_change_mib": (
            slow_memory["peak_host_pss_mib"] - baseline_memory["peak_host_pss_mib"]
        ),
        "peak_gpu_memory_change_mib": (
            slow_memory["peak_gpu_mib"] - baseline_memory["peak_gpu_mib"]
        ),
        "payloads_match": slow["payload_sha256"] == baseline["payload_sha256"],
    }


def _run_concurrent_wave(
    client_factory: Callable[[], MStarClient],
    seed_image: Path,
    variant: rollout.Variant,
    *,
    num_steps: int,
    request_ids: Sequence[str],
    seeds: Sequence[int],
    stall_threshold_seconds: float,
    enable_nvtx: bool,
    chunk_lists: Sequence[list[VideoFrameChunk]] | None = None,
) -> tuple[list[tuple[dict, list[str]]], float, float]:
    """Run ``len(request_ids)`` streams together, each opening only once every
    thread has built its request body (mirrors serve_rollout._concurrent_rollouts).

    Returns per-stream (metrics, failures) in ``request_ids`` order, plus wave
    start/end on the driver's own wall clock (for cross-stream aggregate timing).
    ``chunk_lists``, if given, is one list per request id that each stream's
    chunks are appended to (see ``_measure_stream``'s ``chunks_out``).
    """
    barrier = threading.Barrier(len(request_ids) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(request_ids)) as executor:
        futures = [
            executor.submit(
                _measure_stream,
                client_factory(),
                seed_image,
                variant,
                num_steps=num_steps,
                request_id=request_id,
                rng_seed=seed,
                consumer_pause_seconds=0.0,
                stall_threshold_seconds=stall_threshold_seconds,
                enable_nvtx=enable_nvtx,
                start_barrier=barrier,
                chunks_out=chunk_lists[index] if chunk_lists is not None else None,
            )
            for index, (request_id, seed) in enumerate(zip(request_ids, seeds, strict=True))
        ]
        barrier.wait(timeout=30)
        wave_start = time.perf_counter()
        results = [future.result() for future in futures]
        wave_end = time.perf_counter()
    return results, wave_start, wave_end


def _stream_is_realtime(metrics: dict) -> bool:
    """A stream stayed realtime iff it sustained >=1x, its p95 inter-chunk gap
    fit the one-chunk (4 frames @ 60 fps) delivery budget, and it never stalled."""
    sustained = metrics["sustained_media_to_wall_ratio"]
    gap_p95 = metrics["inter_chunk_gap_seconds"]["p95"]
    gap_p95_ms = gap_p95 * 1000.0 if gap_p95 is not None else None
    return (
        sustained is not None
        and sustained >= 1.0
        and gap_p95_ms is not None
        and gap_p95_ms <= REALTIME_CHUNK_BUDGET_MS
        and metrics["stalls"]["count"] == 0
    )


def _concurrent_aggregate(per_stream: Sequence[dict], *, aggregate_fps: float | None, server: dict) -> dict:
    """Cross-stream realtime summary from each stream's _measure_stream metrics
    plus the already-parsed server-side rollout cadence."""
    ttff_ms = [
        metrics["time_to_first_frame_seconds"] * 1000.0
        for metrics in per_stream
        if metrics["time_to_first_frame_seconds"] is not None
    ]
    gap_p50_ms = [
        metrics["inter_chunk_gap_seconds"]["p50"] * 1000.0
        for metrics in per_stream
        if metrics["inter_chunk_gap_seconds"]["p50"] is not None
    ]
    gap_p95_ms = [
        metrics["inter_chunk_gap_seconds"]["p95"] * 1000.0
        for metrics in per_stream
        if metrics["inter_chunk_gap_seconds"]["p95"] is not None
    ]
    gap_max_ms = [
        metrics["inter_chunk_gap_seconds"]["maximum"] * 1000.0
        for metrics in per_stream
        if metrics["inter_chunk_gap_seconds"]["maximum"] is not None
    ]
    sustained = [
        metrics["sustained_media_to_wall_ratio"]
        for metrics in per_stream
        if metrics["sustained_media_to_wall_ratio"] is not None
    ]
    realtime_per_stream = [_stream_is_realtime(metrics) for metrics in per_stream]
    realtime_count = sum(realtime_per_stream)
    chunk_fractions = [
        metrics["on_time_chunk_fraction"]
        for metrics in per_stream
        if metrics["on_time_chunk_fraction"] is not None
    ]
    gap_p50_median = statistics.median(gap_p50_ms) if gap_p50_ms else None
    server_step_p50 = server["step_spacing_ms"]["p50"]

    return {
        "aggregate_fps": aggregate_fps,
        "ttff_ms": {"p50": _percentile(ttff_ms, 0.50), "p95": _percentile(ttff_ms, 0.95)},
        "gap_ms": {
            "p50_worst": max(gap_p50_ms) if gap_p50_ms else None,
            "p95_worst": max(gap_p95_ms) if gap_p95_ms else None,
            "max_worst": max(gap_max_ms) if gap_max_ms else None,
            "p50_median": gap_p50_median,
        },
        "sustained_min": min(sustained) if sustained else None,
        "stall_count_total": sum(metrics["stalls"]["count"] for metrics in per_stream),
        "realtime_per_stream": realtime_per_stream,
        "realtime_count": realtime_count,
        "realtime_fraction": (
            realtime_count / len(realtime_per_stream) if realtime_per_stream else None
        ),
        "all_realtime": all(realtime_per_stream) if realtime_per_stream else False,
        "on_time_chunk_fraction_min": min(chunk_fractions) if chunk_fractions else None,
        "on_time_chunk_fraction_mean": (
            statistics.fmean(chunk_fractions) if chunk_fractions else None
        ),
        "server": server,
        "delivery_bound": (
            gap_p50_median is not None
            and server_step_p50 is not None
            and gap_p50_median > 1.2 * server_step_p50
        ),
    }


def _dit_step_timestamps(log_text: str, request_ids: set[str]) -> list[float]:
    """Wall-clock seconds (from each line's %(asctime)s prefix) of every rollout
    DiT step whose batch includes a request_id, in log order. Same marker as
    serve_rollout._dit_schedule, extended with the timestamp it discards."""
    marker = "Executing: dit graph_walk=rollout "
    timestamps = []
    for line in log_text.splitlines():
        if marker not in line:
            continue
        try:
            batch = ast.literal_eval(line.split(marker, 1)[1].strip())
        except (SyntaxError, ValueError):
            continue
        if not isinstance(batch, (list, tuple)) or not any(rid in request_ids for rid in batch):
            continue
        match = _LOG_TIMESTAMP_RE.match(line)
        if match is None:
            continue
        timestamps.append(datetime.strptime(match.group(1), _LOG_TIMESTAMP_FORMAT).timestamp())
    return timestamps


def _concurrent_server_metrics(log_text: str, request_ids: set[str], startup_seconds: float) -> dict:
    """Rows-per-step histogram and step-to-step spacing for the measured
    wave's DiT rollout executions, parsed from worker DEBUG log lines."""
    schedule = rollout._dit_schedule(log_text, request_ids)
    histogram: dict[int, int] = {}
    for batch in schedule:
        histogram[len(batch)] = histogram.get(len(batch), 0) + 1

    timestamps = sorted(_dit_step_timestamps(log_text, request_ids))
    spacing_ms = [(later - earlier) * 1000.0 for earlier, later in zip(timestamps, timestamps[1:], strict=False)]
    return {
        "rows_per_step_histogram": histogram,
        "step_spacing_ms": {"p50": _percentile(spacing_ms, 0.50), "p95": _percentile(spacing_ms, 0.95)},
        "startup_seconds": startup_seconds,
    }


def _run_concurrent_phase(
    client_factory: Callable[[], MStarClient],
    seed_image: Path,
    variant: rollout.Variant,
    *,
    streams: int,
    worlds: int,
    batch: int,
    num_steps: int,
    warmup_steps: int,
    request_id_prefix: str,
    rng_seed: int,
    stall_threshold_seconds: float,
    enable_nvtx: bool,
    log_path: Path,
    proc: subprocess.Popen,
    request_timeout: float,
    sampler: rollout.MemorySampler,
    startup_seconds: float,
    save_videos_dir: Path | None = None,
) -> tuple[dict, list[str]]:
    """N-stream concurrent phase: a discarded warmup wave (compiles the
    batch-``streams`` CUDA graph bucket), then a measured wave whose per-stream
    metrics and server-side cadence decide whether every stream stayed realtime.

    ``save_videos_dir``, if given, saves only the measured wave's streams (the
    warmup wave is throwaway CUDA graph compilation)."""
    failures: list[str] = []

    warmup_ids = [f"{request_id_prefix}-concurrent-warmup-{i}" for i in range(streams)]
    warmup_seeds = [rng_seed + i for i in range(streams)]
    print(f"concurrent warmup: {streams} streams, {warmup_steps} step(s) each")
    _wait_for_phase_sample(sampler, proc, "concurrent-warmup")
    warmup_results, _, _ = _run_concurrent_wave(
        client_factory,
        seed_image,
        variant,
        num_steps=warmup_steps,
        request_ids=warmup_ids,
        seeds=warmup_seeds,
        stall_threshold_seconds=stall_threshold_seconds,
        enable_nvtx=enable_nvtx,
    )
    for request_id, (_, stream_failures) in zip(warmup_ids, warmup_results, strict=True):
        failures.extend(f"concurrent-warmup {request_id}: {failure}" for failure in stream_failures)
    rollout._wait_for_cleanup(log_path, tuple(warmup_ids), proc, request_timeout)

    measured_ids = [f"{request_id_prefix}-concurrent-measured-{i}" for i in range(streams)]
    measured_seeds = [rng_seed + i for i in range(streams)]
    print(f"concurrent measured: {streams} streams, {num_steps} step(s) each")
    _wait_for_phase_sample(sampler, proc, "concurrent-measured")
    log_offset = log_path.stat().st_size
    chunk_lists = [[] for _ in measured_ids] if save_videos_dir is not None else None
    measured_results, wave_start, wave_end = _run_concurrent_wave(
        client_factory,
        seed_image,
        variant,
        num_steps=num_steps,
        request_ids=measured_ids,
        seeds=measured_seeds,
        stall_threshold_seconds=stall_threshold_seconds,
        enable_nvtx=enable_nvtx,
        chunk_lists=chunk_lists,
    )
    per_stream = []
    for request_id, (metrics, stream_failures) in zip(measured_ids, measured_results, strict=True):
        failures.extend(f"concurrent-measured {request_id}: {failure}" for failure in stream_failures)
        per_stream.append(metrics)
    rollout._wait_for_cleanup(log_path, tuple(measured_ids), proc, request_timeout, offset=log_offset)

    if save_videos_dir is not None:
        for request_id, chunks in zip(measured_ids, chunk_lists, strict=True):
            out = save_videos_dir / f"{request_id}.mp4"
            n, _fps = _encode_mp4(chunks, out)
            print(f"saved video: {out} ({n} frames)")

    total_frames = sum(metrics["frame_count"] for metrics in per_stream)
    aggregate_fps = total_frames / (wave_end - wave_start) if wave_end > wave_start else None

    wave_log = rollout._read_log_since(log_path, log_offset)
    server = _concurrent_server_metrics(wave_log, set(measured_ids), startup_seconds)

    samples, _ = sampler.snapshot()
    measured_gpu_mib = [sample.gpu_mib for sample in samples if sample.phase == "concurrent-measured"]

    result = {
        "streams": streams,
        "worlds": worlds,
        "batch": batch,
        "per_stream": per_stream,
        "gpu_peak_mib": max(measured_gpu_mib) if measured_gpu_mib else None,
        **_concurrent_aggregate(per_stream, aggregate_fps=aggregate_fps, server=server),
    }
    return result, failures


def _format_number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _human_summary(result: dict, artifact: Path) -> str:
    lines = [
        (
            f"Waypoint streaming viability: {result['variant']} "
            f"({result['geometry']['width']}x{result['geometry']['height']})"
        )
    ]
    for name in ("baseline", "slow_consumer"):
        run = result["runs"][name]
        gaps = run["inter_chunk_gap_seconds"]
        memory = run["memory"]
        lines.append(
            f"  {name}: TTFF={_format_number(run['time_to_first_frame_seconds'])}s "
            f"sustained={_format_number(run['sustained_media_to_wall_ratio'])}x "
            f"gap p50/p95={_format_number(gaps['p50'])}/{_format_number(gaps['p95'])}s "
            f"jitter={_format_number(gaps['jitter_population_stddev'])}s "
            f"stalls={run['stalls']['count']}"
        )
        lines.append(
            f"    memory peak/quiet: host={memory['peak_host_pss_mib']:.1f}/"
            f"{memory['quiet_host_pss_mib']:.1f} MiB, GPU={memory['peak_gpu_mib']:.1f}/"
            f"{memory['quiet_gpu_mib']:.1f} MiB"
        )
    startup = result["startup_latency_seconds"]
    if startup is not None:
        lines.append(
            f"  startup TTFF over {startup['sample_count']} reused slots: "
            f"p50/p95={_format_number(startup['p50'])}/"
            f"{_format_number(startup['p95'])}s "
            f"mean={_format_number(startup['mean'])}s"
        )
    concurrent = result.get("concurrent")
    if concurrent is not None:
        lines.append(
            f"  concurrent: {concurrent['streams']} streams "
            f"(worlds={concurrent['worlds']} batch={concurrent['batch']})"
        )
        for idx, (stream, realtime) in enumerate(
            zip(concurrent["per_stream"], concurrent["realtime_per_stream"], strict=True)
        ):
            gaps = stream["inter_chunk_gap_seconds"]
            lines.append(
                f"    stream {idx}: TTFF={_format_number(stream['time_to_first_frame_seconds'])}s "
                f"gap p50/p95/max={_format_number(gaps['p50'])}/{_format_number(gaps['p95'])}/"
                f"{_format_number(gaps['maximum'])}s "
                f"sustained={_format_number(stream['sustained_media_to_wall_ratio'])}x "
                f"stalls={stream['stalls']['count']} "
                f"realtime={'yes' if realtime else 'no'}"
            )
        lines.append(
            f"    aggregate: fps={_format_number(concurrent['aggregate_fps'])} "
            f"ttff p50/p95={_format_number(concurrent['ttff_ms']['p50'])}/"
            f"{_format_number(concurrent['ttff_ms']['p95'])}ms "
            f"gap p50_median/p95_worst={_format_number(concurrent['gap_ms']['p50_median'])}/"
            f"{_format_number(concurrent['gap_ms']['p95_worst'])}ms "
            f"sustained_min={_format_number(concurrent['sustained_min'])}x "
            f"stalls_total={concurrent['stall_count_total']} "
            f"all_realtime={concurrent['all_realtime']} "
            f"delivery_bound={concurrent['delivery_bound']} "
            f"gpu_peak={_format_number(concurrent['gpu_peak_mib'], 1)}MiB"
        )

    backpressure = result["backpressure"]
    lines.extend(
        [
            (
                "  slow-consumer effect: "
                f"wall +{backpressure['observed_request_wall_increase_seconds']:.3f}s "
                f"for {backpressure['injected_pause_seconds']:.3f}s injected; "
                f"host peak delta={backpressure['peak_host_pss_change_mib']:.1f} MiB, "
                f"GPU peak delta={backpressure['peak_gpu_memory_change_mib']:.1f} MiB"
            ),
            "  release threshold: not defined (measurement baseline only)",
            f"  correctness: {'PASS' if result['correctness']['passed'] else 'FAIL'}",
            f"  artifact: {artifact}",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=rollout.DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=sorted(rollout.VARIANTS), required=True)
    parser.add_argument("--source", choices=("local", "hub"), default="local")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--ae-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--seed-image", type=Path, default=rollout.DEFAULT_ROOT / "seed/default.jpg"
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--streams",
        type=int,
        default=1,
        help="concurrent streams; >1 runs the concurrent batching phase",
    )
    parser.add_argument(
        "--worlds", type=int, help="server world slots (kv.num_sessions); defaults to --streams"
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="rows per rollout step (model_kwargs.step_batch_size); defaults to --streams",
    )
    parser.add_argument(
        "--startup-repeats",
        type=int,
        default=0,
        help="short streams measured before the baseline, for startup p50/p95",
    )
    parser.add_argument("--startup-steps", type=int, default=1)
    parser.add_argument("--slow-consumer-delay", type=float, default=0.25)
    parser.add_argument(
        "--stall-threshold",
        type=float,
        help="gap classified as a stall; default=max(stall-floor, stall-multiplier*4/60)",
    )
    parser.add_argument("--stall-multiplier", type=float, default=4.0)
    parser.add_argument("--stall-floor", type=float, default=0.25)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--memory-sample-interval", type=float, default=0.10)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--request-id", default="waypoint-streaming-benchmark")
    parser.add_argument("--seed", type=int, default=112464007)
    parser.add_argument(
        "--artifact", type=Path, default=Path("/tmp/waypoint_streaming_benchmark.json")
    )
    parser.add_argument(
        "--save-videos",
        type=Path,
        help=(
            "write every measured stream (baseline, slow-consumer, and each "
            "concurrent stream) as DIR/<request_id>.mp4, plus a copy of --artifact"
        ),
    )
    parser.add_argument(
        "--log", type=Path, default=Path("/tmp/waypoint_streaming_benchmark_server.log")
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=("binary", "ndjson"),
        default="binary",
        help="streaming wire format; 'ndjson' forces the base64 path for A/B runs",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.startup_repeats < 0:
        parser.error("--startup-repeats cannot be negative")
    if args.startup_steps <= 0:
        parser.error("--startup-steps must be positive")
    if args.streams <= 0:
        parser.error("--streams must be positive")
    if args.worlds is None:
        args.worlds = args.streams
    if args.batch is None:
        args.batch = args.streams
    if args.worlds <= 0:
        parser.error("--worlds must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.batch > args.worlds:
        parser.error("--batch must be <= --worlds")
    if args.slow_consumer_delay < 0:
        parser.error("--slow-consumer-delay cannot be negative")
    if args.stall_threshold is not None and args.stall_threshold <= 0:
        parser.error("--stall-threshold must be positive")
    if args.stall_multiplier <= 0:
        parser.error("--stall-multiplier must be positive")
    if args.stall_floor < 0:
        parser.error("--stall-floor cannot be negative")
    if args.physical_gpu < 0:
        parser.error("--physical-gpu cannot be negative")
    if args.memory_sample_interval <= 0:
        parser.error("--memory-sample-interval must be positive")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    if args.source == "hub" and (args.checkpoint_dir is not None or args.ae_path is not None):
        parser.error("--source hub cannot be combined with --checkpoint-dir or --ae-path")
    return args


def _resolve_stall_threshold(args: argparse.Namespace) -> float:
    if args.stall_threshold is not None:
        return args.stall_threshold
    return max(args.stall_floor, args.stall_multiplier * 4.0 / 60.0)


def _run_benchmark(args: argparse.Namespace) -> dict:
    variant = rollout.VARIANTS[args.variant]
    if args.source == "local":
        checkpoint_dir = args.checkpoint_dir or variant.checkpoint_dir
        ae_path = args.ae_path or rollout.DEFAULT_ROOT / "taehv1_5"
        weight_source = str(checkpoint_dir)
    else:
        checkpoint_dir = None
        ae_path = None
        weight_source = f"registry Hub mapping for {variant.model_variant}"

    stall_threshold = _resolve_stall_threshold(args)
    if args.save_videos is not None:
        args.save_videos.mkdir(parents=True, exist_ok=True)

    port = args.port or rollout._free_port()
    url = f"http://127.0.0.1:{port}"
    workdir = Path(tempfile.mkdtemp(prefix="waypoint-stream-benchmark-"))
    config = rollout._run_config(
        args.config,
        variant,
        checkpoint_dir,
        ae_path,
        workdir / "run.yaml",
        worlds=args.worlds,
        batch=args.batch,
    )
    seed_image = rollout._seed_png(args.seed_image, variant, workdir / "seed.png")
    # serve_rollout forces DEBUG for its own concurrent mode so the worker's
    # per-step "Executing: dit graph_walk=..." lines are emitted; the
    # concurrent phase below needs the same to parse rows/step and spacing.
    server_log_level = "DEBUG" if args.streams > 1 else args.log_level
    server_command = rollout._server_command(
        config,
        port,
        workdir,
        server_log_level,
        args.request_timeout,
        args.cache_dir,
        args.enable_nvtx,
    )
    print(
        f"starting {args.variant} Waypoint server on {url}\n"
        f"weights: {weight_source}\nserver log: {args.log}"
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("wb") as log:
        proc = subprocess.Popen(
            server_command,
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)},
        )

    sampler: rollout.MemorySampler | None = None
    failures: list[str] = []
    runs: dict[str, dict] = {}
    startup_ttffs: list[float] = []
    concurrent_result: dict | None = None
    startup_started = time.perf_counter()
    try:
        client = MStarClient(
            url,
            timeout=args.request_timeout,
            enable_nvtx=args.enable_nvtx,
            prefer_binary=args.protocol == "binary",
        )
        rollout._wait_for_health(client, proc, args.startup_timeout)
        startup_seconds = time.perf_counter() - startup_started
        print(f"server ready after {startup_seconds:.1f}s")
        sampler = rollout.MemorySampler(
            proc.pid,
            args.physical_gpu,
            interval=args.memory_sample_interval,
        )
        sampler.start()
        rollout._wait_for_first_memory_sample(sampler, proc)

        if args.warmup_steps:
            phase = "warmup"
            print(f"warmup: {args.warmup_steps} step(s)")
            _wait_for_phase_sample(sampler, proc, phase)
            warmup_id = f"{args.request_id}-warmup"
            _, warmup_failures = _measure_stream(
                client,
                seed_image,
                variant,
                num_steps=args.warmup_steps,
                request_id=warmup_id,
                rng_seed=args.seed,
                consumer_pause_seconds=0.0,
                stall_threshold_seconds=stall_threshold,
                enable_nvtx=args.enable_nvtx,
            )
            failures.extend(f"warmup: {failure}" for failure in warmup_failures)
            rollout._wait_for_cleanup(
                args.log, (warmup_id,), proc, args.request_timeout
            )
            rollout._wait_for_quiescent_memory(sampler, "warmup-quiet")

        # Startup samples: short streams, each followed by a full cleanup, so
        # every TTFF is measured against a reused world slot rather than a
        # freshly warmed one. This is the population the prime-capture A/B
        # compares.
        for index in range(args.startup_repeats):
            phase = f"startup-{index:02d}"
            _wait_for_phase_sample(sampler, proc, phase)
            request_id = f"{args.request_id}-{phase}"
            metrics, startup_failures = _measure_stream(
                client,
                seed_image,
                variant,
                num_steps=args.startup_steps,
                request_id=request_id,
                rng_seed=args.seed,
                consumer_pause_seconds=0.0,
                stall_threshold_seconds=stall_threshold,
                enable_nvtx=args.enable_nvtx,
            )
            failures.extend(f"{phase}: {failure}" for failure in startup_failures)
            rollout._wait_for_cleanup(
                args.log, (request_id,), proc, args.request_timeout
            )
            ttff = metrics["time_to_first_frame_seconds"]
            if ttff is not None:
                startup_ttffs.append(ttff)
        if args.startup_repeats:
            print(
                f"startup: {len(startup_ttffs)}/{args.startup_repeats} samples, "
                f"p50={_format_number(_percentile(startup_ttffs, 0.50))}s"
            )

        for name, pause in (
            ("baseline", 0.0),
            ("slow_consumer", args.slow_consumer_delay),
        ):
            phase = name.replace("_", "-")
            print(
                f"{phase}: {args.steps} steps, "
                f"consumer pause {pause:.3f}s between chunks"
            )
            _wait_for_phase_sample(sampler, proc, phase)
            request_id = f"{args.request_id}-{phase}"
            saved_chunks: list[VideoFrameChunk] | None = (
                [] if args.save_videos is not None else None
            )
            metrics, stream_failures = _measure_stream(
                client,
                seed_image,
                variant,
                num_steps=args.steps,
                request_id=request_id,
                rng_seed=args.seed,
                consumer_pause_seconds=pause,
                stall_threshold_seconds=stall_threshold,
                enable_nvtx=args.enable_nvtx,
                chunks_out=saved_chunks,
            )
            failures.extend(f"{name}: {failure}" for failure in stream_failures)
            rollout._wait_for_cleanup(
                args.log, (request_id,), proc, args.request_timeout
            )
            quiet = rollout._wait_for_quiescent_memory(sampler, f"{phase}-quiet")
            metrics["memory"] = _memory_metrics(sampler, phase, quiet)
            runs[name] = metrics
            print(
                f"  received {metrics['chunk_count']} chunks in "
                f"{metrics['request_wall_seconds']:.3f}s"
            )
            if saved_chunks:
                out = args.save_videos / f"{request_id}.mp4"
                n, _fps = _encode_mp4(saved_chunks, out)
                print(f"saved video: {out} ({n} frames)")

        if runs["baseline"]["payload_sha256"] != runs["slow_consumer"]["payload_sha256"]:
            failures.append(
                "slow-consumer payload differs from the identical-seed baseline"
            )

        if args.streams > 1:
            print(
                f"concurrent: {args.streams} streams, {args.steps} steps each "
                f"(worlds={args.worlds} batch={args.batch})"
            )
            concurrent_result, concurrent_failures = _run_concurrent_phase(
                client_factory=lambda: MStarClient(
                    url, timeout=args.request_timeout, prefer_binary=args.protocol == "binary"
                ),
                seed_image=seed_image,
                variant=variant,
                streams=args.streams,
                worlds=args.worlds,
                batch=args.batch,
                num_steps=args.steps,
                warmup_steps=args.warmup_steps,
                request_id_prefix=args.request_id,
                rng_seed=args.seed,
                stall_threshold_seconds=stall_threshold,
                enable_nvtx=args.enable_nvtx,
                log_path=args.log,
                proc=proc,
                request_timeout=args.request_timeout,
                sampler=sampler,
                startup_seconds=startup_seconds,
                save_videos_dir=args.save_videos,
            )
            failures.extend(concurrent_failures)
    finally:
        try:
            if sampler is not None:
                sampler.stop()
        finally:
            rollout._shutdown(proc)

    result = {
        "schema_version": 1,
        "benchmark": "waypoint_streaming_viability",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "release_threshold": None,
        "variant": args.variant,
        "model_variant": variant.model_variant,
        "geometry": {"width": variant.width, "height": variant.height, "fps": 60.0},
        "configuration": {
            "weight_source": weight_source,
            "stream_protocol": args.protocol,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "rng_seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "stall_threshold_seconds": stall_threshold,
            "slow_consumer_delay_seconds": args.slow_consumer_delay,
            "memory_sample_interval_seconds": args.memory_sample_interval,
            "server_command": server_command,
            "server_log": str(args.log),
        },
        "server": {"startup_seconds": startup_seconds},
        "startup_latency_seconds": _startup_latency_metrics(startup_ttffs),
        "runs": runs,
        "backpressure": _backpressure_metrics(
            runs["baseline"],
            runs["slow_consumer"],
            runs["baseline"]["memory"],
            runs["slow_consumer"]["memory"],
        ),
        "correctness": {"passed": not failures, "failures": failures},
        "metric_definitions": {
            "time_to_first_frame_seconds": (
                "request iterator start to first fully decoded SDK VideoFrameChunk"
            ),
            "startup_latency_seconds": (
                "time_to_first_frame_seconds over --startup-repeats short streams, "
                "each after a full request cleanup, so every sample reuses a world slot"
            ),
            "sustained_media_to_wall_ratio": (
                "media seconds in chunks after the first divided by first-to-last chunk arrival time"
            ),
            "jitter_population_stddev": "population standard deviation of inter-chunk gaps",
            "stall": "inter-chunk gap strictly greater than stall_threshold_seconds",
            "on_time_chunk_fraction": (
                f"per stream: fraction of chunks delivered on time (inter-chunk gap <= "
                f"{REALTIME_CHUNK_BUDGET_MS:.1f}ms, one 4-frame/60fps chunk of playback); "
                "the pacing pillar of streaming viability"
            ),
            "memory": "PSS and nvidia-smi GPU process memory summed over the server process group only",
            "backpressure": (
                "delta between an unpaused stream and an identical stream paused between SDK reads"
            ),
        },
    }

    if concurrent_result is not None:
        result["concurrent"] = concurrent_result
        result["metric_definitions"].update(
            {
                "concurrent.aggregate_fps": (
                    "total frames delivered across streams / (last chunk arrival - earliest "
                    "stream start), wall-clock, over the measured concurrent wave"
                ),
                "concurrent.gap_ms.p50_median": "median across streams of each stream's own inter-chunk gap p50",
                "concurrent.gap_ms.*_worst": (
                    "largest across streams of each stream's own inter-chunk gap p50/p95/maximum"
                ),
                "concurrent.realtime_per_stream": (
                    f"per stream: sustained ratio >= 1.0 and gap p95 <= {REALTIME_CHUNK_BUDGET_MS:.1f}ms "
                    "(one 4-frame/60fps chunk) and zero stalls"
                ),
                "concurrent.realtime_count": (
                    "how many of the batch's streams stayed realtime (sum of realtime_per_stream); "
                    "realtime_fraction is that over the stream count"
                ),
                "concurrent.on_time_chunk_fraction_min": (
                    "smallest per-stream on_time_chunk_fraction across the batch (mean is the average); "
                    "the worst stream's on-time chunk rate under batch-B scheduling"
                ),
                "concurrent.delivery_bound": (
                    "client gap_ms.p50_median > 1.2x server.step_spacing_ms.p50: clients are "
                    "slower than the GPU produces steps, so the limit is delivery rather than compute"
                ),
                "concurrent.server.rows_per_step_histogram": (
                    "count of measured-wave DiT rollout steps by batch row count"
                ),
                "concurrent.server.step_spacing_ms": (
                    "gap between consecutive measured-wave DiT rollout step log timestamps"
                ),
            }
        )
    return result


def _write_artifact(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = _run_benchmark(args)
    except (Exception, SystemExit) as exc:  # lifecycle helpers use SystemExit
        result = {
            "schema_version": 1,
            "benchmark": "waypoint_streaming_viability",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "release_threshold": None,
            "variant": args.variant,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_artifact(args.artifact, result)
        print(f"ERROR {result['error']}", file=sys.stderr)
        print(f"artifact: {args.artifact}", file=sys.stderr)
        return 2

    _write_artifact(args.artifact, result)
    if args.save_videos is not None:
        shutil.copy2(args.artifact, args.save_videos / args.artifact.name)
    print(_human_summary(result, args.artifact))
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
