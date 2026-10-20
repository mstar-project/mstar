#!/usr/bin/env python3
"""Time to the first chunk for long LingBot-World rollouts through the served path.

Starts the API server on configs/lingbot_world_1_3b.yaml (as test/lingbot_world/serve_rollout.py
does), warms it with one 93-frame rollout, then for each case streams a request, records every
chunk's arrival for the first --chunks chunks, and closes the stream (an abort).

Trajectories longer than the example's 269 poses are the example's poses played forward and
back (0..268, 267..0, 1..), so the path stays continuous.

    python test/lingbot_world/serve_first_chunk.py --example <repo>/examples/03 --out /tmp/lingbot/first \
        --chunks 16 --cases 3001:4x4 4077:4x4 4077:3x4

A case is ``frames:layout``, the layout ``4x4`` or ``3x4`` (poses without the constant bottom row).
"""

import argparse
import json
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
    report = {"cases": []}

    def run(num_frames: int, layout: str, max_chunks: int, rid: str) -> dict:
        traj = pingpong(poses, num_frames)
        if layout == "3x4":
            traj = traj[:, :3]
        payload = len(json.dumps({"camera_poses": traj.tolist(), "intrinsics": intr, "num_frames": num_frames}))
        res = {"num_frames": num_frames, "layout": layout, "model_kwargs_bytes": payload, "arrivals_s": []}
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
                if len(res["arrivals_s"]) >= max_chunks:
                    break
            if hasattr(stream, "close"):
                stream.close()
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"[:400]
        res["wall_s"] = time.perf_counter() - t0
        gaps = np.diff(res["arrivals_s"]).tolist()
        res["first_chunk_s"] = res["arrivals_s"][0] if res["arrivals_s"] else None
        res["chunk_gap_s_median"] = float(np.median(gaps[1:])) if len(gaps) > 1 else None
        print(json.dumps({k: v for k, v in res.items() if k != "arrivals_s"}), flush=True)
        return res

    try:
        deadline = time.time() + 900
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()})")
            time.sleep(2)
        report["warmup"] = run(93, "4x4", 99, "warm")
        report["warm_93"] = run(93, "4x4", 99, "warm2")
        for i, case in enumerate(args.cases):
            n, layout = case.split(":")
            report["cases"].append(run(int(n), layout, args.chunks, f"case{i}"))
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
