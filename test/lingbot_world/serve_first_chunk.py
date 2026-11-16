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
    if any(not math.isfinite(t) or t < 0 for t in arrivals_s):
        raise ValueError("chunk arrivals must be finite and nonnegative")
    if any(b <= a for a, b in zip(arrivals_s, arrivals_s[1:], strict=False)):
        raise ValueError("chunk arrivals must increase")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in frame_counts):
        raise ValueError("frame counts must be positive integers")
    if any(not math.isfinite(fps) or fps <= 0 or fps != fps_values[0] for fps in fps_values):
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
    period = np.concatenate([np.arange(len(poses)), np.arange(len(poses) - 2, 0, -1)])
    return poses[np.resize(period, n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--example", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=["3001:4x4"])
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--full-rollout", action="store_true", help="measure every chunk through natural completion")
    ap.add_argument("--port", type=int, default=18451)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    from mstar.client import MStarClient, VideoFrameChunk

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="lingbot-ff-"))
    log = open(args.out / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(args.repo / "mstar/api_server/entrypoint.py"),
         "--config", str(args.repo / "configs/lingbot_world_1_3b.yaml"),
         "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(workdir / "sock"),
         "--upload-dir", str(workdir / "uploads"), "--tensor-comm-protocol", "SHM", "--log-level", "INFO",
         "--timeout", "900"],
        stdout=log, stderr=subprocess.STDOUT, cwd=args.repo,
    )
    client = MStarClient(f"http://127.0.0.1:{args.port}", timeout=900)
    poses = np.load(args.example / "poses.npy")
    intr = np.load(args.example / "intrinsics.npy")[0].tolist()
    if args.chunks < 1:
        ap.error("--chunks must be positive")
    report = {"cases": [], "full_rollout": args.full_rollout}

    def run(num_frames: int, layout: str, max_chunks: int | None, rid: str) -> dict:
        traj = pingpong(poses, num_frames)
        if layout == "3x4":
            traj = traj[:, :3]
        payload = len(json.dumps({"camera_poses": traj.tolist(), "intrinsics": intr, "num_frames": num_frames}))
        res = {"num_frames": num_frames, "layout": layout, "model_kwargs_bytes": payload,
               "arrivals_s": [], "frame_counts": [], "fps_values": [], "sample_complete": max_chunks is None}
        t0 = time.perf_counter()
        try:
            stream = client.stream(
                text=PROMPT, images=str(args.example / "image.jpg"), input_modalities=("text", "image"),
                output_modalities=("video_frame",), request_id=rid, seed=42, num_frames=num_frames,
                camera_poses=traj.tolist(), intrinsics=intr,
            )
            for event in stream:
                if not isinstance(event, VideoFrameChunk):
                    raise RuntimeError(f"unexpected stream event {type(event).__name__}")
                res["arrivals_s"].append(time.perf_counter() - t0)
                res["frame_counts"].append(event.frame_count)
                res["fps_values"].append(event.fps)
                if max_chunks is not None and len(res["arrivals_s"]) >= max_chunks:
                    break
            if hasattr(stream, "close"):
                stream.close()
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"[:400]
            res["sample_complete"] = False
        res["wall_s"] = time.perf_counter() - t0
        if res["arrivals_s"]:
            res.update(summarize_chunks(res["arrivals_s"], res["frame_counts"], res["fps_values"]))
            # Retain the old name/definition for consumers of earlier first-chunk reports.
            gaps = res["chunk_gap_s"]
            res["chunk_gap_s_median"] = statistics.median(gaps[1:]) if len(gaps) > 1 else None
        else:
            res["first_chunk_s"] = None
        print(json.dumps({k: v for k, v in res.items() if k != "arrivals_s"}), flush=True)
        return res

    try:
        deadline = time.time() + 900
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()})")
            time.sleep(2)
        report["warmup"] = run(93, "4x4", None, "warm")
        report["warm_93"] = run(93, "4x4", None, "warm2")
        for i, case in enumerate(args.cases):
            n, layout = case.split(":")
            report["cases"].append(run(int(n), layout, None if args.full_rollout else args.chunks, f"case{i}"))
            time.sleep(5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    (args.out / "first_frame.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
