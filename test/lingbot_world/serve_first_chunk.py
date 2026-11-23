#!/usr/bin/env python3
"""Measure served LingBot-World chunk timing, with optional complete rollouts.

Starts the API server on configs/lingbot_world_1_3b.yaml (as test/lingbot_world/serve_rollout.py
does), warms it with one 93-frame rollout, then for each case streams a request, records every
chunk's arrival for the first --chunks chunks, and closes the stream (an abort). With
--full-rollout, wait for the whole response and measure whether frame delivery keeps
up with the metadata's playback FPS. Chunk arrival measures include client/network
overhead; they are not model-only kernel timings.

Trajectories longer than the example's 269 poses are the example's poses played forward and
back (0..268, 267..0, 1..), so the path stays continuous.

    python test/lingbot_world/serve_first_chunk.py --example <repo>/examples/03 --out /tmp/lingbot/first \
        --chunks 16 --cases 3001:4x4 4077:4x4 4077:3x4

    python test/lingbot_world/serve_first_chunk.py --example <repo>/examples/03 --out /tmp/lingbot/full \
        --full-rollout --cases 3001:4x4

A case is ``frames:layout``, the layout ``4x4`` or ``3x4`` (poses without the constant bottom row).
"""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
    "snow-capped mountains under a bright blue sky with drifting white clouds \u2014 gentle ripples "
    "reflect the tree and sky, creating a tranquil, meditative atmosphere."
)


def summarize_chunks(arrivals_s: list[float], frame_counts: list[int], fps_values: list[float]) -> dict:
    """Summarize delivered frame cadence, starting playback at the first chunk.

    The first chunk's frames are excluded from generated_fps because their generation
    time is represented by first_chunk_s. A playback stall occurs when the next chunk
    arrives after all previously delivered frames would have played.
    """
    if not arrivals_s or len(arrivals_s) != len(frame_counts) or len(arrivals_s) != len(fps_values):
        raise ValueError("chunk arrivals, frame counts and FPS must have equal nonzero lengths")
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in arrivals_s):
        raise ValueError("chunk arrivals must be finite and nonnegative")
    if any(b <= a for a, b in zip(arrivals_s, arrivals_s[1:], strict=False)):
        raise ValueError("chunk arrivals must increase")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in frame_counts):
        raise ValueError("frame counts must be positive integers")
    if any(
        isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
        or fps != fps_values[0]
        for fps in fps_values
    ):
        raise ValueError("chunk playback FPS must be finite, positive and constant")

    gaps = [b - a for a, b in zip(arrivals_s, arrivals_s[1:], strict=False)]
    playback_until = arrivals_s[0] + frame_counts[0] / fps_values[0]
    stall_s = 0.0
    for arrival, count in zip(arrivals_s[1:], frame_counts[1:], strict=True):
        stall_s += max(0.0, arrival - playback_until)
        playback_until = max(playback_until, arrival) + count / fps_values[0]
    generated_fps = sum(frame_counts[1:]) / sum(gaps) if gaps else None
    return {
        "first_chunk_s": arrivals_s[0],
        "chunk_gap_s": gaps,
        "chunk_gap_s_p50": statistics.median(gaps) if gaps else None,
        "generated_fps": generated_fps,
        "playback_fps": fps_values[0],
        "playback_ratio": generated_fps / fps_values[0] if generated_fps is not None else None,
        "playback_stall_s": stall_s,
        "generated_frames": sum(frame_counts),
    }


def pingpong(poses: np.ndarray, n: int) -> np.ndarray:
    if len(poses) < 1:
        raise ValueError("poses must be non-empty")
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"n must be an int; got {n!r}")
    if n < 0:
        raise ValueError("n must be non-negative")
    period = np.concatenate([np.arange(len(poses)), np.arange(len(poses) - 2, 0, -1)])
    return poses[np.resize(period, n)]


def expected_chunk_frame_counts(num_frames: int) -> list[int]:
    """The served model floors requests to whole latent chunks before decoding.

    This mirrors LingBotWorldModel.num_latent_frames and the causal VAE decoder:
    four latent frames emit 13 video frames first, then 16 per later chunk.
    """
    from mstar.model.lingbot_world.config import lingbot_world_1_3b

    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise ValueError(f"num_frames must be an int; got {num_frames!r}")
    cfg = lingbot_world_1_3b()
    stride = cfg.vae_stride[0]
    requested_latents = (num_frames - 1) // stride + 1
    chunks = requested_latents // cfg.chunk_size
    if chunks < 1 or chunks > cfg.max_chunks:
        raise ValueError(f"num_frames must produce 1..{cfg.max_chunks} chunks; got {num_frames}")
    return [1 + stride * (cfg.chunk_size - 1)] + [stride * cfg.chunk_size] * (chunks - 1)


def run_case(client, chunk_type, poses: np.ndarray, intr: list[float], example: Path,
             num_frames: int, layout: str, max_chunks: int | None, rid: str) -> dict:
    if layout not in {"4x4", "3x4"}:
        raise ValueError(f"layout must be 4x4 or 3x4; got {layout!r}")
    expected_counts = expected_chunk_frame_counts(num_frames)
    traj = pingpong(poses, num_frames)
    if layout == "3x4":
        traj = traj[:, :3]
    payload = len(json.dumps({"camera_poses": traj.tolist(), "intrinsics": intr, "num_frames": num_frames}))
    res = {"num_frames": num_frames, "layout": layout, "model_kwargs_bytes": payload,
           "expected_frames": sum(expected_counts), "expected_chunks": len(expected_counts),
           "arrivals_s": [], "frame_counts": [], "frame_indices": [], "fps_values": [],
           "sample_complete": False}
    t0 = time.perf_counter()
    naturally_finished = False
    try:
        stream = client.stream(
            text=PROMPT, images=str(example / "image.jpg"), input_modalities=("text", "image"),
            output_modalities=("video_frame",), request_id=rid, seed=42, num_frames=num_frames,
            camera_poses=traj.tolist(), intrinsics=intr,
        )
        try:
            for event in stream:
                if not isinstance(event, chunk_type):
                    raise RuntimeError(f"unexpected stream event {type(event).__name__}")
                chunk_num = len(res["frame_counts"])
                next_index = sum(res["frame_counts"])
                if event.frame_index != next_index:
                    raise ValueError(f"chunk {chunk_num} frame_index {event.frame_index}; expected {next_index}")
                if chunk_num >= len(expected_counts) or event.frame_count != expected_counts[chunk_num]:
                    expected = expected_counts[chunk_num] if chunk_num < len(expected_counts) else None
                    raise ValueError(f"chunk {chunk_num} frame_count {event.frame_count}; expected {expected}")
                res["arrivals_s"].append(time.perf_counter() - t0)
                res["frame_counts"].append(event.frame_count)
                res["frame_indices"].append(event.frame_index)
                res["fps_values"].append(event.fps)
                if max_chunks is not None and len(res["arrivals_s"]) >= max_chunks:
                    break
            else:
                naturally_finished = True
        finally:
            if hasattr(stream, "close"):
                stream.close()
        res["sample_complete"] = naturally_finished and sum(res["frame_counts"]) == res["expected_frames"]
        if naturally_finished and not res["sample_complete"]:
            res["error"] = f"expected {res['expected_frames']} frames; got {sum(res['frame_counts'])}"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:400]
    res["wall_s"] = time.perf_counter() - t0
    if res["arrivals_s"]:
        try:
            res.update(summarize_chunks(res["arrivals_s"], res["frame_counts"], res["fps_values"]))
        except ValueError as e:
            res.setdefault("error", f"ValueError: {e}"[:400])
            res["sample_complete"] = False
        # Retain the old name/definition for consumers of earlier first-chunk reports.
        gaps = res.get("chunk_gap_s", [])
        res["chunk_gap_s_median"] = statistics.median(gaps[1:]) if len(gaps) > 1 else None
    else:
        res["first_chunk_s"] = None
    print(json.dumps({k: v for k, v in res.items() if k != "arrivals_s"}), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--config", type=Path, default=Path("configs/lingbot_world_1_3b.yaml"),
                    help="server config YAML (relative to --repo unless absolute)")
    ap.add_argument("--example", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=["3001:4x4"])
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--full-rollout", action="store_true", help="measure every chunk through natural completion")
    ap.add_argument("--port", type=int, default=18451)
    ap.add_argument("--timeout", type=float, default=900.0, help="startup and per-request timeout in seconds")
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))
    if args.chunks < 1:
        ap.error("--chunks must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("--timeout must be finite and positive")
    if not (1 <= args.port <= 65535):
        ap.error("--port must be between 1 and 65535")
    config_path = args.config if args.config.is_absolute() else (
        args.repo / args.config if (args.repo / args.config).is_file() or not args.config.is_file()
        else args.config
    )
    config_path = config_path.resolve()
    if not config_path.is_file():
        ap.error(f"config file not found: {config_path}")
    cases = []
    for case in args.cases:
        try:
            count_text, layout = case.split(":")
            count = int(count_text)
            if layout not in {"4x4", "3x4"}:
                raise ValueError("layout must be 4x4 or 3x4")
            expected_chunk_frame_counts(count)
        except ValueError as e:
            ap.error(f"invalid case {case!r}: {e}")
        cases.append((count, layout))
    if not (args.example / "image.jpg").is_file():
        ap.error(f"example image not found: {args.example / 'image.jpg'}")
    try:
        poses = np.load(args.example / "poses.npy")
        intr_arr = np.load(args.example / "intrinsics.npy")
    except (OSError, ValueError, EOFError) as e:
        ap.error(f"failed to load example trajectory: {e}")
    if not isinstance(poses, np.ndarray) or not isinstance(intr_arr, np.ndarray):
        for obj in (poses, intr_arr):
            if not isinstance(obj, np.ndarray) and hasattr(obj, "close"):
                obj.close()
        ap.error("poses.npy and intrinsics.npy must be .npy arrays")
    if (
        not (np.issubdtype(poses.dtype, np.floating) or np.issubdtype(poses.dtype, np.integer))
        or poses.ndim != 3
        or len(poses) < 1
        or poses.shape[1:] != (4, 4)
        or not np.isfinite(poses).all()
    ):
        ap.error(f"poses.npy must be finite real numeric with shape (N, 4, 4) and N >= 1; got {poses.shape}")
    if (
        not (np.issubdtype(intr_arr.dtype, np.floating) or np.issubdtype(intr_arr.dtype, np.integer))
        or intr_arr.ndim != 2
        or len(intr_arr) < 1
        or intr_arr.shape[1] != 4
        or not np.isfinite(intr_arr[0]).all()
    ):
        ap.error(f"intrinsics.npy must have finite real first row with shape (M, 4) and M >= 1; got {intr_arr.shape}")
    intr = intr_arr[0].tolist()
    from mstar.client import MStarClient, VideoFrameChunk

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = None
    log = None
    proc = None
    report = {"cases": [], "full_rollout": args.full_rollout}

    try:
        workdir = Path(tempfile.mkdtemp(prefix="lingbot-ff-"))
        log = open(args.out / "server.log", "w")
        proc = subprocess.Popen(
            [sys.executable, str(args.repo / "mstar/api_server/entrypoint.py"),
             "--config", str(config_path),
             "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(workdir / "sock"),
             "--upload-dir", str(workdir / "uploads"), "--tensor-comm-protocol", "SHM", "--log-level", "INFO",
             "--timeout", str(args.timeout)],
            stdout=log, stderr=subprocess.STDOUT, cwd=args.repo,
        )
        client = MStarClient(f"http://127.0.0.1:{args.port}", timeout=args.timeout)
        deadline = time.time() + args.timeout
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()})")
            time.sleep(2)
        report["warmup"] = run_case(client, VideoFrameChunk, poses, intr, args.example, 93, "4x4", None, "warm")
        report["warm_93"] = run_case(client, VideoFrameChunk, poses, intr, args.example, 93, "4x4", None, "warm2")
        for i, (n, layout) in enumerate(cases):
            report["cases"].append(run_case(client, VideoFrameChunk, poses, intr, args.example, n, layout,
                                            None if args.full_rollout else args.chunks, f"case{i}"))
            time.sleep(5)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if log is not None:
            log.close()
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
    (args.out / "first_frame.json").write_text(json.dumps(report, indent=2))
    runs = [report["warmup"], report["warm_93"], *report["cases"]]
    return int(any(r.get("error") for r in runs) or not all(r["sample_complete"] for r in runs[:2])
               or (args.full_rollout and not all(r["sample_complete"] for r in report["cases"])))


if __name__ == "__main__":
    raise SystemExit(main())
