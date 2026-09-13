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
        --artifact /tmp/waypoint-streaming-360p.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
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
) -> tuple[dict, list[str]]:
    """Consume one stream while retaining only timings and an incremental hash."""
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
        "--log", type=Path, default=Path("/tmp/waypoint_streaming_benchmark_server.log")
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--enable-nvtx", action="store_true")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
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

    port = args.port or rollout._free_port()
    url = f"http://127.0.0.1:{port}"
    workdir = Path(tempfile.mkdtemp(prefix="waypoint-stream-benchmark-"))
    config = rollout._run_config(
        args.config,
        variant,
        checkpoint_dir,
        ae_path,
        workdir / "run.yaml",
        worlds=1,
    )
    seed_image = rollout._seed_png(args.seed_image, variant, workdir / "seed.png")
    server_command = rollout._server_command(
        config,
        port,
        workdir,
        args.log_level,
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
    startup_started = time.perf_counter()
    try:
        client = MStarClient(url, timeout=args.request_timeout)
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

        if runs["baseline"]["payload_sha256"] != runs["slow_consumer"]["payload_sha256"]:
            failures.append(
                "slow-consumer payload differs from the identical-seed baseline"
            )
    finally:
        try:
            if sampler is not None:
                sampler.stop()
        finally:
            rollout._shutdown(proc)

    return {
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
            "sustained_media_to_wall_ratio": (
                "media seconds in chunks after the first divided by first-to-last chunk arrival time"
            ),
            "jitter_population_stddev": "population standard deviation of inter-chunk gaps",
            "stall": "inter-chunk gap strictly greater than stall_threshold_seconds",
            "memory": "PSS and nvidia-smi GPU process memory summed over the server process group only",
            "backpressure": (
                "delta between an unpaused stream and an identical stream paused between SDK reads"
            ),
        },
    }


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
    print(_human_summary(result, args.artifact))
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
