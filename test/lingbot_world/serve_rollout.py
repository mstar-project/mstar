#!/usr/bin/env python3
"""Serve LingBot-World-V2 1.3B through mstar and stream one rollout, end to end.

Launches ``mstar/api_server/entrypoint.py`` on ``configs/lingbot_world_1_3b.yaml``
(API server, conductor and GPU worker), sends the example's image, prompt and
camera trajectory through ``MStarClient``, and collects the ``video_frame``
chunks: one per rollout chunk, 13 frames then 16 each. The request runs twice
under different ids with the same seed; identical bytes the second time show
the first rollout left no state behind. With ``--oracle-dir`` the frames are
also compared against the reference video (quantized the way the dit node
quantizes); the server decodes the JPEG with torchvision rather than PIL, so
the conditioning pixels, and with them the rollout, may differ slightly.

    python test/lingbot_world/serve_rollout.py --example <repo>/examples/03 \
        --oracle-dir /tmp/lingbot/oracle --out /tmp/lingbot/serve
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mstar.client import MStarClient, VideoFrameChunk  # noqa: E402

PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
    "snow-capped mountains under a bright blue sky with drifting white clouds \u2014 gentle ripples "
    "reflect the tree and sky, creating a tranquil, meditative atmosphere."
)


def rollout(
    client: MStarClient, example: Path, request_id: str, num_frames: int,
) -> tuple[list[bytes], list[dict], list[float]]:
    chunks, metas, arrivals = [], [], []
    t0 = time.perf_counter()
    for event in client.stream(
        text=PROMPT, images=str(example / "image.jpg"), input_modalities=("text", "image"),
        output_modalities=("video_frame",), request_id=request_id, seed=42, num_frames=num_frames,
        camera_poses=np.load(example / "poses.npy").tolist(),
        intrinsics=np.load(example / "intrinsics.npy")[0].tolist(),
    ):
        if not isinstance(event, VideoFrameChunk):
            raise RuntimeError(f"unexpected stream event {type(event).__name__}")
        chunks.append(event.data)
        metas.append(event.metadata)
        arrivals.append(time.perf_counter() - t0)
        print(f"  {request_id} chunk {len(chunks) - 1}  bytes={len(event.data)}  "
              f"t={arrivals[-1]:.2f}s  {event.metadata}")
    return chunks, metas, arrivals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oracle-dir", type=Path)
    ap.add_argument("--num-frames", type=int, default=93)
    ap.add_argument("--port", type=int, default=18431)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="lingbot-serve-"))
    log = open(args.out / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "mstar/api_server/entrypoint.py"),
         "--config", str(REPO / "configs/lingbot_world_1_3b.yaml"),
         "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(workdir / "sock"),
         "--upload-dir", str(workdir / "uploads"), "--tensor-comm-protocol", "SHM", "--log-level", "INFO",
         "--timeout", "600"],
        stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
    )
    client = MStarClient(f"http://127.0.0.1:{args.port}")
    report: dict = {}
    try:
        deadline = time.time() + 900
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()}); see {args.out / 'server.log'}")
            time.sleep(2)
        first, metas, arrivals = rollout(client, args.example, "lingbot-a", args.num_frames)
        second, _, arrivals_b = rollout(client, args.example, "lingbot-b", args.num_frames)
        report["chunks"] = len(first)
        report["chunk_metadata"] = metas
        report["arrival_s"] = {"a": arrivals, "b": arrivals_b}
        report["repeat_identical"] = first == second
        h, w = metas[0].get("height"), metas[0].get("width")
        video = np.concatenate([np.frombuffer(c, dtype=np.uint8).reshape(-1, h, w, 3) for c in first])
        report["frames"] = int(video.shape[0])
        np.save(args.out / "served_frames.npy", video)
        if args.oracle_dir is not None:
            import torch

            ref = torch.load(args.oracle_dir / "video.pt")
            ref_u8 = (ref / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
            mse = None
            if video.shape == ref_u8.shape:
                mse = float(np.mean((video.astype(np.float64) - ref_u8.astype(np.float64)) ** 2))
            report["vs_oracle"] = {
                "shape_equal": video.shape == ref_u8.shape,
                "bit_exact": bool(video.shape == ref_u8.shape and np.array_equal(video, ref_u8)),
                "psnr_db": None if mse is None else (math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)),
            }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    (args.out / "serve.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "chunk_metadata"}, indent=2, default=str))
    if not report.get("repeat_identical") or report.get("vs_oracle", {}).get("bit_exact") is False:
        print("served rollout FAILED: the two runs differ or the frames differ from the oracle")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
