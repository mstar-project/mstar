#!/usr/bin/env python3
"""Send measured Waypoint streaming requests to an already-running mstar server.

Unlike ``benchmark_streaming.py``, this script never launches a server, samples
its memory, or measures its startup time: it only exercises the client side, so
the server can be started (and its lifecycle managed) independently -- under an
external profiler, with custom flags, or on a machine this script never touches.
It reuses ``benchmark_streaming``'s stream measurement and ``serve_rollout``'s
request-building helpers rather than re-implementing them.

Start a server independently, then point this script at its port:

    python -m mstar.api_server.entrypoint --config configs/waypoint.yaml --port 8123

    python3 benchmark/waypoint/benchmark_live_server.py \
        --port 8123 --variant 720p --steps 16 \
        --seed-image /path/to/seed.jpg \
        --artifact /tmp/waypoint-live-720p.json
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
# serve_rollout (request building, seed resize) stays with the end-to-end tests
# in test/waypoint; benchmark_streaming (stream measurement, metrics, video
# encoding) is this script's sibling module.
ROLLOUT_DIR = REPO / "test" / "waypoint"
for path in (ROLLOUT_DIR, REPO, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_streaming as streaming  # noqa: E402
import serve_rollout as rollout  # noqa: E402

from mstar.client import MStarClient  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="port of the running server")
    parser.add_argument("--variant", choices=sorted(rollout.VARIANTS), required=True)
    parser.add_argument(
        "--seed-image", type=Path, required=True, help="16:9 seed image, resized to the variant's geometry"
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--streams", type=int, default=1, help="concurrent streams sent together"
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=112464007)
    parser.add_argument("--request-id", default="waypoint-live-server-benchmark")
    parser.add_argument(
        "--artifact", type=Path, default=Path("/tmp/waypoint_live_server_benchmark.json")
    )
    parser.add_argument(
        "--save-videos",
        type=Path,
        help="write every measured stream as DIR/<request_id>.mp4, plus a copy of --artifact",
    )
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.streams <= 0:
        parser.error("--streams must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def _run_live_benchmark(args: argparse.Namespace) -> dict:
    """Warm up, then send ``--streams`` concurrent measured streams to the
    server at ``--host``/``--port``. No server launch, memory sampling, or
    startup measurement -- those all need the launched server's process handle
    or log file, which this script never has."""
    variant = rollout.VARIANTS[args.variant]
    # Same formula benchmark_streaming derives from --stall-multiplier/--stall-floor;
    # this script doesn't expose those knobs, so it reuses their defaults directly.
    stall_threshold = streaming._resolve_stall_threshold(
        SimpleNamespace(stall_threshold=None, stall_multiplier=4.0, stall_floor=0.25)
    )
    workdir = Path(tempfile.mkdtemp(prefix="waypoint-live-bench-"))
    seed_image = rollout._seed_png(args.seed_image, variant, workdir / "seed.png")
    url = f"http://{args.host}:{args.port}"

    def client_factory() -> MStarClient:
        return MStarClient(url, timeout=args.request_timeout, prefer_binary=True)

    if args.save_videos is not None:
        args.save_videos.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    if args.warmup_steps:
        warmup_ids = [f"{args.request_id}-warmup-{i}" for i in range(args.streams)]
        warmup_seeds = [args.seed + i for i in range(args.streams)]
        print(f"warmup: {args.streams} stream(s), {args.warmup_steps} step(s) each")
        warmup_results, _, _ = streaming._run_concurrent_wave(
            client_factory,
            seed_image,
            variant,
            num_steps=args.warmup_steps,
            request_ids=warmup_ids,
            seeds=warmup_seeds,
            stall_threshold_seconds=stall_threshold,
            enable_nvtx=False,
        )
        for request_id, (_, stream_failures) in zip(warmup_ids, warmup_results, strict=True):
            failures.extend(f"warmup {request_id}: {failure}" for failure in stream_failures)

    measured_ids = [f"{args.request_id}-{i}" for i in range(args.streams)]
    measured_seeds = [args.seed + i for i in range(args.streams)]
    chunk_lists = [[] for _ in measured_ids] if args.save_videos is not None else None
    print(f"measured: {args.streams} stream(s), {args.steps} step(s) each")
    measured_results, wave_start, wave_end = streaming._run_concurrent_wave(
        client_factory,
        seed_image,
        variant,
        num_steps=args.steps,
        request_ids=measured_ids,
        seeds=measured_seeds,
        stall_threshold_seconds=stall_threshold,
        enable_nvtx=False,
        chunk_lists=chunk_lists,
    )
    per_stream = []
    for request_id, (metrics, stream_failures) in zip(measured_ids, measured_results, strict=True):
        failures.extend(f"{request_id}: {failure}" for failure in stream_failures)
        per_stream.append(metrics)

    if chunk_lists is not None:
        for request_id, chunks in zip(measured_ids, chunk_lists, strict=True):
            out = args.save_videos / f"{request_id}.mp4"
            n, _fps = streaming._encode_mp4(chunks, out)
            print(f"saved video: {out} ({n} frames)")

    total_frames = sum(metrics["frame_count"] for metrics in per_stream)
    aggregate_fps = total_frames / (wave_end - wave_start) if wave_end > wave_start else None

    # No launched-server log to parse rollout step spacing from, so the
    # server-side half of _concurrent_aggregate is unmeasured, not "not
    # delivery-bound" -- overwrite the flag rather than report a false negative.
    aggregate = streaming._concurrent_aggregate(
        per_stream,
        aggregate_fps=aggregate_fps,
        server={"step_spacing_ms": {"p50": None, "p95": None}},
    )
    aggregate["delivery_bound"] = None

    return {
        "schema_version": 1,
        "benchmark": "waypoint_live_server",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "variant": args.variant,
        "model_variant": variant.model_variant,
        "geometry": {"width": variant.width, "height": variant.height, "fps": 60.0},
        "server_url": url,
        "configuration": {
            "steps": args.steps,
            "streams": args.streams,
            "warmup_steps": args.warmup_steps,
            "rng_seed": args.seed,
            "stall_threshold_seconds": stall_threshold,
        },
        "per_stream": per_stream,
        **aggregate,
        "correctness": {"passed": not failures, "failures": failures},
    }


def _human_summary(result: dict, artifact: Path) -> str:
    fmt = streaming._format_number
    lines = [
        f"Waypoint live-server benchmark: {result['variant']} against {result['server_url']}"
    ]
    for idx, (stream, realtime) in enumerate(
        zip(result["per_stream"], result["realtime_per_stream"], strict=True)
    ):
        gaps = stream["inter_chunk_gap_seconds"]
        lines.append(
            f"  stream {idx}: TTFF={fmt(stream['time_to_first_frame_seconds'])}s "
            f"gap p50/p95/max={fmt(gaps['p50'])}/{fmt(gaps['p95'])}/{fmt(gaps['maximum'])}s "
            f"sustained={fmt(stream['sustained_media_to_wall_ratio'])}x "
            f"stalls={stream['stalls']['count']} "
            f"realtime={'yes' if realtime else 'no'}"
        )
    lines.append(
        f"  aggregate: fps={fmt(result['aggregate_fps'])} "
        f"ttff p50/p95={fmt(result['ttff_ms']['p50'])}/{fmt(result['ttff_ms']['p95'])}ms "
        f"gap p50_median/p95_worst={fmt(result['gap_ms']['p50_median'])}/"
        f"{fmt(result['gap_ms']['p95_worst'])}ms "
        f"sustained_min={fmt(result['sustained_min'])}x "
        f"stalls_total={result['stall_count_total']} "
        f"all_realtime={result['all_realtime']}"
    )
    lines.append(f"  correctness: {'PASS' if result['correctness']['passed'] else 'FAIL'}")
    lines.append(f"  artifact: {artifact}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level)
    result = _run_live_benchmark(args)
    streaming._write_artifact(args.artifact, result)
    if args.save_videos is not None:
        shutil.copy2(args.artifact, args.save_videos / args.artifact.name)
    print(_human_summary(result, args.artifact))
    return 0 if result["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
